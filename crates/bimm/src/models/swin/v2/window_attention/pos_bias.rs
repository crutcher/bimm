use crate::layers::activation::{Activation, ActivationConfig};
use crate::models::swin::v2::window_attention::{
    window_attention_relative_position_index, window_log1p_relative_offset_grid,
};
use bimm_contracts::assert_shape_contract_periodically;
use burn::config::Config;
use burn::module::Module;
use burn::nn;
use burn::prelude::{Backend, Int, Tensor};
use burn::tensor::activation::sigmoid;

/// Common introspection interface for relative position bias modules.
pub trait RelativePositionBiasMeta {
    /// Returns the number of attention heads.
    fn num_heads(&self) -> usize;

    /// Returns the shape of the window.
    fn window_shape(&self) -> [usize; 2];

    /// Returns the height of the window.
    fn window_height(&self) -> usize {
        self.window_shape()[0]
    }

    /// Returns the width of the window.
    fn window_width(&self) -> usize {
        self.window_shape()[1]
    }

    /// Returns the base value for the relative position bias.
    fn base(&self) -> f64;
}

/// Configuration for the relative position bias module.
#[derive(Config, Debug)]
pub struct RelativePositionBiasConfig {
    /// The number of attention heads.
    pub num_heads: usize,

    /// The shape of the window ``[height, width]``.
    pub window_shape: [usize; 2],

    /// The base value for the relative position bias.
    #[config(default = 8.0)]
    pub base: f64,

    /// The hidden dimension of the MLP.
    #[config(default = 512)]
    pub mlp_hidden_dim: usize,

    /// The activation layer configuration.
    #[config(default = "ActivationConfig::Relu")]
    pub mlp_activation: ActivationConfig,
}

impl RelativePositionBiasMeta for RelativePositionBiasConfig {
    fn num_heads(&self) -> usize {
        self.num_heads
    }

    fn window_shape(&self) -> [usize; 2] {
        self.window_shape
    }
    fn base(&self) -> f64 {
        self.base
    }
}

impl RelativePositionBiasConfig {
    /// Initializes an `OffsetGridRelativePositionBias` module.
    ///
    /// # Arguments
    ///
    /// * `device`: The device on which the module will be created.
    ///
    /// # Returns
    ///
    /// An `OffsetGridRelativePositionBias` module.
    #[inline(always)]
    #[must_use]
    pub fn init_offset_grid_rpb<B: Backend>(
        &self,
        device: &B::Device,
    ) -> OffsetGridRelativePositionBias<B> {
        OffsetGridRelativePositionBias {
            base: self.base,
            num_heads: self.num_heads,
            window_shape: self.window_shape,

            rel_coords_table: window_log1p_relative_offset_grid(
                self.window_shape,
                self.base,
                device,
            ),

            rel_index: window_attention_relative_position_index::<B>(self.window_shape, device),

            cbp: ContinuousPositionBiasMlpConfig::new(self.num_heads)
                .with_d_hidden(self.mlp_hidden_dim)
                .with_activation(self.mlp_activation.clone())
                .init(device),
        }
    }
}

/// An offset grid relative position bias module.
///
/// Published/Used in SWIN-Transformer v2.
#[derive(Module, Debug)]
pub struct OffsetGridRelativePositionBias<B: Backend> {
    /// The base value for the relative position bias.
    pub base: f64,

    /// The number of attention heads.
    pub num_heads: usize,

    /// The shape of the window.
    pub window_shape: [usize; 2],

    /// The relative coordinates table for the window.
    pub rel_coords_table: Tensor<B, 3>,

    /// The relative position index for the window.
    pub rel_index: Tensor<B, 2, Int>,

    /// The continuous position bias MLP.
    pub cbp: ContinuousPositionBiasMlp<B>,
}

impl<B: Backend> RelativePositionBiasMeta for OffsetGridRelativePositionBias<B> {
    fn base(&self) -> f64 {
        self.base
    }
    fn num_heads(&self) -> usize {
        self.num_heads
    }

    fn window_shape(&self) -> [usize; 2] {
        self.window_shape
    }
}

impl<B: Backend> OffsetGridRelativePositionBias<B> {
    /// Returns the learned relative position bias.
    ///
    /// This is hashed such that all pairs of locations in the window with the same
    /// inter-pair relative offset will have the same bias across all heads.
    ///
    /// # Returns
    ///
    /// A 3D tensor of shape (`num_heads`, height * width, height * width)
    /// containing the relative position bias for each head and position
    /// pair.
    #[must_use]
    pub fn forward(&self) -> Tensor<B, 3> {
        let [h, w] = self.window_shape;
        let hw = h * w;

        let lookup_table = self.cbp.forward(self.rel_coords_table.clone());
        assert_shape_contract_periodically!(
            [
                "height" = "two" * "win_height" - "clip",
                "width" = "two" * "win_width" - "clip",
                "heads"
            ],
            &lookup_table.dims(),
            &[
                ("two", 2),
                ("clip", 1),
                ("win_height", self.window_height()),
                ("win_width", self.window_width()),
                ("heads", self.num_heads()),
            ],
        );

        let rpb_table = lookup_table.reshape([-1, self.num_heads as i32]);
        // 2*((h-1) * (w-1)), heads

        let idx = self.rel_index.clone().reshape([-1]);
        // (h*w)^2

        let rpb = rpb_table.select(0, idx);
        // (h*w)^2, heads

        let rpb = rpb.reshape([hw, hw, self.num_heads]);
        let rpb = rpb.permute([2, 0, 1]);
        // heads, h*w, h*w

        let x = sigmoid(rpb).mul_scalar(2.0 * self.base);
        assert_shape_contract_periodically!(
            ["heads", "flat" = "height" * "width", "height" * "width"],
            &x.dims(),
            &[
                ("heads", self.num_heads()),
                ("height", self.window_height()),
                ("width", self.window_width()),
            ],
        );

        x
    }
}

/// Common introspection interface for continuous position bias MLPs.
pub trait ContinuousPositionBiasMlpMeta {
    /// Returns the hidden dimension of the MLP.
    fn d_hidden(&self) -> usize;

    /// Returns the number of heads.
    fn num_heads(&self) -> usize;
}

/// Configuration for `ContinuousPositionBiasMlp`.
#[derive(Config, Debug)]
pub struct ContinuousPositionBiasMlpConfig {
    /// The number of heads.
    pub num_heads: usize,

    /// The hidden dimension of the MLP.
    #[config(default = 512)]
    pub d_hidden: usize,

    /// The activation layer configuration.
    #[config(default = "ActivationConfig::Relu")]
    pub activation: ActivationConfig,
}

impl ContinuousPositionBiasMlpMeta for ContinuousPositionBiasMlpConfig {
    fn d_hidden(&self) -> usize {
        self.d_hidden
    }

    fn num_heads(&self) -> usize {
        self.num_heads
    }
}

/// A multi-layer perceptron (MLP) for continuous position bias.
#[derive(Module, Debug)]
pub struct ContinuousPositionBiasMlp<B: Backend> {
    l1: nn::Linear<B>,
    act: Activation<B>,
    l2: nn::Linear<B>,
}

impl<B: Backend> ContinuousPositionBiasMlpMeta for ContinuousPositionBiasMlp<B> {
    fn d_hidden(&self) -> usize {
        self.l1.weight.dims()[1]
    }

    fn num_heads(&self) -> usize {
        self.l2.weight.dims()[1]
    }
}

impl ContinuousPositionBiasMlpConfig {
    /// Initializes the MLP with the given device.
    fn init<B: Backend>(
        &self,
        device: &B::Device,
    ) -> ContinuousPositionBiasMlp<B> {
        ContinuousPositionBiasMlp {
            l1: nn::LinearConfig::new(2, self.d_hidden).init(device),

            act: self.activation.init(device),

            l2: nn::LinearConfig::new(self.d_hidden, self.num_heads)
                .with_bias(false)
                .init(device),
        }
    }
}

impl<B: Backend> ContinuousPositionBiasMlp<B> {
    /// Applies the MLP to the input tensor.
    ///
    /// # Arguments
    ///
    /// * `x`: A tensor of ``[..., 2]`` of the relative log-offset coordinates table.
    ///
    /// # Returns
    ///
    /// A tensor of ``[..., num_heads]`` of the learned bias table.
    #[must_use]
    pub fn forward<const D: usize>(
        &self,
        x: Tensor<B, D>,
    ) -> Tensor<B, D> {
        let x = self.l1.forward(x);
        let x = self.act.forward(x);
        self.l2.forward(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::swin::v2::window_attention::{
        window_attention_relative_position_index, window_log1p_relative_offset_grid,
    };
    use bimm_contracts::assert_shape_contract;
    use burn::backend::NdArray;

    #[test]
    fn test_rpb_meta() {
        let config = RelativePositionBiasConfig::new(12, [3, 2]);

        assert_eq!(config.base(), 8.0);
        assert_eq!(config.num_heads(), 12);
        assert_eq!(config.window_shape(), [3, 2]);
        assert_eq!(config.window_height(), 3);
        assert_eq!(config.window_width(), 2);

        let device = Default::default();
        let rpb = config.init_offset_grid_rpb::<NdArray>(&device);

        assert_eq!(rpb.base(), 8.0);
        assert_eq!(rpb.num_heads(), 12);
        assert_eq!(rpb.window_shape(), [3, 2]);
        assert_eq!(rpb.window_height(), 3);
        assert_eq!(rpb.window_width(), 2);
    }

    #[test]
    fn test_og_rpb() {
        let device = Default::default();

        let window_shape = [3, 2];
        let num_heads = 8;

        let config = RelativePositionBiasConfig::new(num_heads, window_shape);
        let rpb = config.init_offset_grid_rpb::<NdArray>(&device);

        assert_eq!(rpb.base(), 8.0);
        assert_eq!(rpb.num_heads(), num_heads);
        assert_eq!(rpb.window_shape(), window_shape);
        assert_eq!(rpb.window_height(), window_shape[0]);
        assert_eq!(rpb.window_width(), window_shape[1]);

        rpb.rel_coords_table.to_data().assert_eq(
            &window_log1p_relative_offset_grid::<NdArray>(window_shape, 8.0, &device).to_data(),
            true,
        );

        rpb.rel_index.to_data().assert_eq(
            &window_attention_relative_position_index::<NdArray>(window_shape, &device).to_data(),
            true,
        );

        let table = rpb.forward();
        // heads, h*w, h*w

        assert_shape_contract!(
            ["heads", "flat" = "h" * "w", "flat" = "h" * "w"],
            &table,
            &[
                ("heads", num_heads),
                ("h", window_shape[0]),
                ("w", window_shape[1]),
            ],
        );
    }

    #[test]
    fn test_cpb_mlp_meta() {
        let config = ContinuousPositionBiasMlpConfig::new(8).with_d_hidden(512);

        assert_eq!(config.d_hidden(), 512);
        assert_eq!(config.num_heads(), 8);

        let device = Default::default();
        let mlp = config.init::<NdArray>(&device);

        assert_eq!(mlp.d_hidden(), 512);
        assert_eq!(mlp.num_heads(), 8);
    }
}
