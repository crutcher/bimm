//! # Operational Block for Swin Transformer v2.
use crate::layers::drop::drop_path::{DropPath, DropPathConfig};
use crate::models::swin::v2::window_attention::{
    WindowAttention, WindowAttentionConfig, WindowAttentionMeta, sw_attn_mask,
};
use crate::models::swin::v2::windowing::{window_partition, window_reverse};
use bimm_contracts::{assert_shape_contract_periodically, define_shape_contract};
use burn::config::Config;
use burn::module::Module;
use burn::nn::activation::{Activation, ActivationConfig};
use burn::nn::{Dropout, DropoutConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::prelude::{Backend, Tensor};
use burn::tensor::BasicOps;

/// Common meta-interface for `BlockMlp` config.
pub trait BlockMlpMeta {
    /// Get the input dimension size.
    fn d_input(&self) -> usize;

    /// Get the hidden dimension size.
    fn d_hidden(&self) -> usize;

    /// Get the output dimension size.
    fn d_output(&self) -> usize;

    /// Get the dropout rate.
    fn drop(&self) -> f64;
}

/// Configuration for `BlockMlp`.
#[derive(Config, Debug)]
pub struct BlockMlpConfig {
    d_input: usize,

    #[config(default = "None")]
    d_hidden: Option<usize>,

    #[config(default = "None")]
    d_output: Option<usize>,

    #[config(default = 0.)]
    drop: f64,

    /// The activation layer configuration.
    #[config(default = "ActivationConfig::Relu")]
    pub activation: ActivationConfig,
}

impl BlockMlpMeta for BlockMlpConfig {
    fn d_input(&self) -> usize {
        self.d_input
    }

    fn d_hidden(&self) -> usize {
        self.d_hidden.unwrap_or(self.d_input)
    }

    fn d_output(&self) -> usize {
        self.d_output.unwrap_or(self.d_input)
    }

    fn drop(&self) -> f64 {
        self.drop
    }
}

impl BlockMlpConfig {
    /// Creates a new `BlockMlp`.
    ///
    /// # Arguments
    ///
    /// - `device`: The device on which the MLP will be initialized.
    ///
    /// # Returns
    ///
    /// A new `BlockMlp` instance.
    pub fn init<B: Backend>(
        &self,
        device: &B::Device,
    ) -> BlockMlp<B> {
        let d_input = self.d_input();
        let d_hidden = self.d_hidden();
        let d_output = self.d_output();

        BlockMlp {
            fc1: LinearConfig::new(d_input, d_hidden).init(device),
            fc2: LinearConfig::new(d_hidden, d_output).init(device),
            act: self.activation.init(device),
            drop: DropoutConfig { prob: self.drop }.init(),
        }
    }
}

/// Swin MLP Module
#[derive(Module, Debug)]
pub struct BlockMlp<B: Backend> {
    /// First linear layer.
    fc1: Linear<B>,

    /// Second linear layer.
    fc2: Linear<B>,

    /// Activation function.
    act: Activation<B>,

    /// Dropout layer.
    drop: Dropout,
}

impl<B: Backend> BlockMlpMeta for BlockMlp<B> {
    fn d_input(&self) -> usize {
        self.fc1.weight.dims()[0]
    }

    fn d_hidden(&self) -> usize {
        self.fc1.weight.dims()[1]
    }

    fn d_output(&self) -> usize {
        self.fc2.weight.dims()[1]
    }

    fn drop(&self) -> f64 {
        self.drop.prob
    }
}

impl<B: Backend> BlockMlp<B> {
    /// Apply the MLP to the input tensor.
    ///
    /// # Arguments
    ///
    /// - `x`: a tensor of ``[batch = ..., in]``.
    ///
    /// # Returns
    ///
    /// A tensor of ``[batch = ... out]``
    #[must_use]
    pub fn forward<const D: usize>(
        &self,
        x: Tensor<B, D>,
    ) -> Tensor<B, D> {
        assert_shape_contract_periodically!(
            [..., "in"],
            &x,
            &[("in", self.d_input())]
        );

        let x = self.fc1.forward(x);
        assert_shape_contract_periodically!(
            [..., "h"],
            &x,
            &[("h", self.d_hidden())]
        );

        let x = self.act.forward(x);

        let x = self.drop.forward(x);

        let x = self.fc2.forward(x);
        assert_shape_contract_periodically!(
            [..., "out"],
            &x,
            &[("out", self.d_output())]
        );

        self.drop.forward(x)
    }
}

/// Applies an inner function under conditional cyclic shift.
///
/// This is used for shifted window attention. When `swa_enabled` is true,
/// it cyclically shifts the input tensor by `shift_size` in the last two dimensions,
/// applies the function `f`, and then reverses the cyclic shift.
///
/// When `swa_enabled` is false, it simply applies the function `f` without any shift.
///
/// # Arguments
///
/// * `x` - Input tensor of ``[batch, height, width, channels]``.
/// * `f` - Function to apply on the shifted tensor.
///
/// # Returns
///
/// A new tensor of the same shape as `x`, with the function `f` applied after cyclic shifting.
#[must_use]
#[inline(always)]
fn with_shift<B: Backend, F, K>(
    x: Tensor<B, 4, K>,
    shift: isize,
    f: F,
) -> Tensor<B, 4, K>
where
    K: BasicOps<B>,
    F: FnOnce(Tensor<B, 4, K>) -> Tensor<B, 4, K>,
{
    let dims = [1, 2];

    // Cyclic shift for shifted window attention.
    let x = if shift != 0 {
        x.roll(&dims, &[-shift, -shift])
    } else {
        x
    };

    let x = f(x);

    // Reverse cyclic shift.
    if shift != 0 {
        x.roll(&dims, &[shift, shift])
    } else {
        x
    }
}

/// Common introspection interface for `TransformerBlock`.
pub trait ShiftedWindowTransformerBlockMeta {
    /// Get the input dimension size.
    fn d_input(&self) -> usize;

    /// Get the input resolution.
    fn input_resolution(&self) -> [usize; 2];

    /// Get the input height.
    fn input_height(&self) -> usize {
        self.input_resolution()[0]
    }

    /// Get the input width.
    fn input_width(&self) -> usize {
        self.input_resolution()[1]
    }

    /// Get the output dimension size.
    fn d_output(&self) -> usize {
        self.d_input()
    }

    /// Get the output resolution.
    fn output_resolution(&self) -> [usize; 2] {
        self.input_resolution()
    }

    /// Get the output height.
    fn output_height(&self) -> usize {
        self.input_height()
    }

    /// Get the output width.
    fn output_width(&self) -> usize {
        self.input_width()
    }

    /// Get the number of attention heads.
    fn num_heads(&self) -> usize;

    /// Window size for window attention.
    fn window_size(&self) -> usize;

    /// Shift size for shifted window attention; 0 means no shift.
    fn shift_size(&self) -> usize;

    /// Is shifted window attention enabled?
    fn swa_enabled(&self) -> bool {
        self.shift_size() > 0
    }

    /// Whether to enable QKV bias.
    fn enable_qkv_bias(&self) -> bool;

    /// Dropout rate for MLP.
    fn drop_rate(&self) -> f64;

    /// Dropout rate for attention.
    fn attn_drop_rate(&self) -> f64;

    /// Ratio of hidden dimension to input dimension in MLP.
    fn mlp_ratio(&self) -> f64;

    /// Drop path rate for stochastic depth.
    fn drop_path_rate(&self) -> f64;
}

/// Configuration for `TransformerBlock`.
#[derive(Config, Debug)]
pub struct ShiftedWindowTransformerBlockConfig {
    /// Input dimension size.
    pub d_input: usize,

    /// Input resolution as ``[height, width]``.
    pub input_resolution: [usize; 2],

    /// Number of attention heads.
    pub num_heads: usize,

    /// Window size for window attention.
    #[config(default = 7)]
    pub window_size: usize,

    /// Shift size for shifted window attention; 0 means no shift.
    #[config(default = 0)]
    pub shift_size: usize,

    /// Ratio of hidden dimension to input dimension in MLP.
    #[config(default = 4.0)]
    pub mlp_ratio: f64,

    /// Whether to enable QKV bias.
    #[config(default = true)]
    pub enable_qkv_bias: bool,

    /// Dropout rate for MLP.
    #[config(default = 0.0)]
    pub drop_rate: f64,

    /// Dropout rate for attention.
    #[config(default = 0.0)]
    pub attn_drop_rate: f64,

    /// The hidden dimension of the MLP.
    #[config(default = 512)]
    pub attn_rpb_mlp_hidden_dim: usize,

    /// The activation layer configuration.
    #[config(default = "ActivationConfig::Relu")]
    pub attn_rpb_mlp_activation: ActivationConfig,

    /// Drop path rate for stochastic depth.
    #[config(default = 0.0)]
    pub drop_path_rate: f64,
    // TODO/Check: act_layer, norm_layer
}

impl ShiftedWindowTransformerBlockMeta for ShiftedWindowTransformerBlockConfig {
    fn d_input(&self) -> usize {
        self.d_input
    }

    fn input_resolution(&self) -> [usize; 2] {
        self.input_resolution
    }

    fn num_heads(&self) -> usize {
        self.num_heads
    }

    fn window_size(&self) -> usize {
        self.window_size
    }

    fn shift_size(&self) -> usize {
        self.shift_size
    }

    fn enable_qkv_bias(&self) -> bool {
        self.enable_qkv_bias
    }

    fn drop_rate(&self) -> f64 {
        self.drop_rate
    }

    fn attn_drop_rate(&self) -> f64 {
        self.attn_drop_rate
    }

    fn mlp_ratio(&self) -> f64 {
        self.mlp_ratio
    }

    fn drop_path_rate(&self) -> f64 {
        self.drop_path_rate
    }
}

impl ShiftedWindowTransformerBlockConfig {
    #[inline(always)]
    fn check(&self) {
        assert!(
            self.d_input > 0,
            "d_input must be greater than zero: {self:#?}"
        );
        assert!(
            self.num_heads > 0,
            "num_heads must be greater than zero: {self:#?}"
        );
        assert!(
            self.window_size > 0,
            "window_size must be greater than zero: {self:#?}"
        );
        let [h, w] = self.input_resolution;
        assert!(
            h > 0 && w > 0,
            "input_resolution must be greater than zero: {self:#?}"
        );
        assert!(
            h % self.window_size == 0 && w % self.window_size == 0,
            "input_resolution must be divisible by window size: {self:#?}",
        );
    }

    /// Initializes a new `SwinTransformerBlock`.
    ///
    /// # Arguments
    ///
    /// * `device` - The device on which the block will be created.
    ///
    /// # Returns
    ///
    /// A new `SwinTransformerBlock` configured with the specified parameters.
    #[must_use]
    pub fn init<B: Backend>(
        &self,
        device: &B::Device,
    ) -> ShiftedWindowTransformerBlock<B> {
        self.check();

        let hidden_dim = (self.d_input as f64 * self.mlp_ratio) as usize;
        let block_mlp = BlockMlpConfig::new(self.d_input)
            .with_d_hidden(Some(hidden_dim))
            .with_drop(self.drop_rate)
            .init(device);

        let win_attn = WindowAttentionConfig::new(
            self.d_input,
            [self.window_size, self.window_size],
            self.num_heads,
        )
        .with_enable_qkv_bias(self.enable_qkv_bias)
        .with_attn_drop(self.attn_drop_rate)
        .with_proj_drop(self.drop_rate)
        .with_rpb_mlp_hidden_dim(self.attn_rpb_mlp_hidden_dim)
        .with_rpb_mlp_activation(self.attn_rpb_mlp_activation.clone())
        .init(device);

        let shift_mask = if self.shift_size == 0 {
            None
        } else {
            Some(
                sw_attn_mask(
                    self.input_resolution,
                    self.window_size,
                    self.shift_size,
                    device,
                )
                .float()
                .mul_scalar(-100.0),
            )
        };

        ShiftedWindowTransformerBlock {
            input_resolution: self.input_resolution,
            window_size: self.window_size,
            shift_size: self.shift_size,
            shift_mask,
            drop_path: DropPathConfig::new()
                .with_drop_prob(self.drop_path_rate)
                .init(),
            norm1: LayerNormConfig::new(self.d_input).init(device),
            norm2: LayerNormConfig::new(self.d_input).init(device),
            win_attn,
            block_mlp,
        }
    }
}

/// Basic Swin Transformer Block.
///
/// Equivalent to the ``SwinTransformerBlock`` in the python source.
///
/// Applies one layer of Swin Transformer block with window attention and MLP.
#[derive(Module, Debug)]
pub struct ShiftedWindowTransformerBlock<B: Backend> {
    /// Input resolution of the block, as ``[H, W]``.
    pub input_resolution: [usize; 2],

    /// Window size for window attention.
    pub window_size: usize,

    /// Shift size for shifted window attention; 0 means no shift.
    pub shift_size: usize,

    /// Shift mask for shifted window attention.
    pub shift_mask: Option<Tensor<B, 3>>,

    /// Drop path for stochastic depth.
    pub drop_path: DropPath,

    /// Layer normalization 1.
    pub norm1: LayerNorm<B>,

    /// Layer normalization 2.
    pub norm2: LayerNorm<B>,

    /// Window attention block.
    pub win_attn: WindowAttention<B>,

    /// MLP block.
    pub block_mlp: BlockMlp<B>,
}

impl<B: Backend> ShiftedWindowTransformerBlockMeta for ShiftedWindowTransformerBlock<B> {
    fn d_input(&self) -> usize {
        self.win_attn.d_input()
    }

    fn input_resolution(&self) -> [usize; 2] {
        self.input_resolution
    }

    fn num_heads(&self) -> usize {
        self.win_attn.num_heads()
    }

    fn window_size(&self) -> usize {
        self.window_size
    }

    fn shift_size(&self) -> usize {
        self.shift_size
    }

    fn enable_qkv_bias(&self) -> bool {
        self.win_attn.enable_qkv_bias()
    }

    fn drop_rate(&self) -> f64 {
        self.block_mlp.drop()
    }

    fn attn_drop_rate(&self) -> f64 {
        self.win_attn.attn_drop()
    }

    fn mlp_ratio(&self) -> f64 {
        self.block_mlp.d_hidden() as f64 / self.d_input() as f64
    }

    fn drop_path_rate(&self) -> f64 {
        self.drop_path.drop_prob
    }
}

impl<B: Backend> ShiftedWindowTransformerBlock<B> {
    /// Applies the forward pass on the input tensor.
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor of ``[batch, height * width, channels]``.
    ///
    /// # Returns
    ///
    /// A new tensor of ``[batch, height * width, channels]``.
    ///
    /// # Panics
    ///
    /// On shape contract failure.
    #[must_use]
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        let [h, w] = self.input_resolution;
        let env = [("height", h), ("width", w)];

        define_shape_contract!(CONTRACT, ["batch", "height" * "width", "channels"]);
        let [b, c] = CONTRACT.unpack_shape(&x, &["batch", "channels"], &env);

        let x = self.with_skip(x, |x| {
            let x = x.reshape([b, h, w, c]);

            let x = with_shift(x, self.shift_size as isize, |x| self.apply_window(x, c));
            // b, h, w, c

            let x = x.reshape([b, h * w, c]);
            self.norm1.forward(x)
        });
        // b, h * w, c

        assert_shape_contract_periodically!(CONTRACT, &x, &env);

        let x = self.with_skip(x, |x| self.norm2.forward(self.block_mlp.forward(x)));

        assert_shape_contract_periodically!(CONTRACT, &x, &env);

        x
    }

    /// Applies an inner function under conditional stochastic residual/depth-skip connection.
    #[must_use]
    #[inline(always)]
    fn with_skip<const D: usize, F>(
        &self,
        x: Tensor<B, D>,
        f: F,
    ) -> Tensor<B, D>
    where
        F: FnOnce(Tensor<B, D>) -> Tensor<B, D>,
    {
        self.drop_path.with_skip(x, f)
    }

    /// Applies window attention to the input tensor.
    ///
    /// This function partitions the input tensor into windows, applies window attention,
    /// and then merges the windows back to the original shape.
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor of ``[batch, height, width, channels]``.
    /// * `c` - Number of channels in the input tensor.
    ///
    /// # Returns
    ///
    /// A new tensor of ``[batch, height, width, channels]`` with window attention applied.
    #[must_use]
    #[inline(always)]
    fn apply_window(
        &self,
        x: Tensor<B, 4>,
        c: usize,
    ) -> Tensor<B, 4> {
        let [h, w] = self.input_resolution;
        let ws = self.window_size as i32;
        let c = c as i32;

        // Partition into windows.
        let x_windows = window_partition(x, self.window_size);
        // b*nW, ws, ws, c
        let x_windows = x_windows.reshape([-1, ws * ws, c]);
        // b*nW, ws*ws, c

        let attn_windows = self.win_attn.forward(x_windows, self.shift_mask.clone());
        // b*nW, ws*ws, c

        // Merge windows back to the original shape.
        let attn_windows = attn_windows.reshape([-1, ws, ws, c]);
        window_reverse(attn_windows, self.window_size, h, w)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::tensor::Distribution;

    #[test]
    fn test_block_mlp_meta() {
        {
            let d_input = 4;
            let config = BlockMlpConfig::new(d_input);

            assert_eq!(config.d_input(), d_input);
            assert_eq!(config.d_hidden(), d_input);
            assert_eq!(config.d_output(), d_input);
            assert_eq!(config.drop(), 0.);
        }

        {
            let d_input = 4;
            let d_hidden = 8;
            let d_output = 6;
            let drop = 0.1;

            let config = BlockMlpConfig::new(d_input)
                .with_d_hidden(Some(d_hidden))
                .with_d_output(Some(d_output))
                .with_drop(drop);

            assert_eq!(config.d_input(), d_input);
            assert_eq!(config.d_hidden(), d_hidden);
            assert_eq!(config.d_output(), d_output);
            assert_eq!(config.drop(), drop);
        }
    }

    #[test]
    fn test_mlp() {
        impl_test_mlp::<NdArray>();
    }

    fn impl_test_mlp<B: Backend>() {
        let device: B::Device = Default::default();

        let a = 2;
        let b = 3;
        let d_input = 4;
        let d_hidden = 8;
        let d_output = 6;
        let drop = 0.1;

        let config = BlockMlpConfig::new(d_input)
            .with_d_hidden(Some(d_hidden))
            .with_d_output(Some(d_output))
            .with_drop(drop);

        let mlp: BlockMlp<B> = config.init(&device);

        assert_eq!(mlp.d_input(), config.d_input());
        assert_eq!(mlp.d_hidden(), config.d_hidden());
        assert_eq!(mlp.d_output(), config.d_output());
        assert_eq!(mlp.drop(), config.drop());

        let distribution = Distribution::Normal(0., 1.);
        let x = Tensor::random([a, b, d_input], distribution, &device);

        let y = mlp.forward(x);

        assert_eq!(y.dims(), [a, b, d_output]);
    }

    #[test]
    fn test_with_shift() {
        let device = Default::default();
        let b = 1;
        let h = 4;
        let w = 4;
        let c = 3;

        let distribution = burn::tensor::Distribution::Uniform(0.0, 1.0);
        let input = Tensor::<NdArray, 4>::random([b, h, w, c], distribution, &device);

        let idx: Tensor<NdArray, 4> =
            Tensor::arange(0..input.shape().num_elements() as i64, &device)
                .reshape([b, h, w, c])
                .float();

        // No-op shift:
        with_shift(input.clone(), 0, |x| x + idx.clone())
            .to_data()
            .assert_eq(&(input.clone() + idx.clone()).to_data(), true);

        with_shift(input.clone(), 1, |x| x + idx.clone())
            .to_data()
            .assert_eq(
                &({
                    let x = input.clone();
                    let x = x.roll(&[1, 2], &[-1, -1]);
                    let x = x + idx.clone();
                    x.roll(&[1, 2], &[1, 1])
                })
                .to_data(),
                true,
            );
    }

    #[test]
    fn test_shifted_window_transformer_block_meta() {
        let d_input = 128;
        let num_heads = 4;
        let input_resolution = [14, 14];

        let config = ShiftedWindowTransformerBlockConfig::new(d_input, input_resolution, num_heads);

        assert_eq!(config.d_input(), d_input);
        assert_eq!(config.input_resolution(), input_resolution);
        assert_eq!(config.input_height(), 14);
        assert_eq!(config.input_width(), 14);
        assert_eq!(config.d_output(), d_input);
        assert_eq!(config.output_resolution(), input_resolution);
        assert_eq!(config.output_height(), 14);
        assert_eq!(config.output_width(), 14);
        assert_eq!(config.num_heads(), num_heads);
        assert_eq!(config.window_size(), 7);
        assert_eq!(config.shift_size(), 0);
        assert!(!config.swa_enabled());
        assert!(config.enable_qkv_bias());
        assert_eq!(config.drop_rate(), 0.0);
        assert_eq!(config.attn_drop_rate(), 0.0);
        assert_eq!(config.mlp_ratio(), 4.0);
        assert_eq!(config.drop_path_rate(), 0.0);

        let device = Default::default();
        let block = config.init::<NdArray>(&device);

        assert_eq!(block.d_input(), d_input);
        assert_eq!(block.input_resolution(), input_resolution);
        assert_eq!(block.input_height(), 14);
        assert_eq!(block.input_width(), 14);
        assert_eq!(block.d_output(), d_input);
        assert_eq!(block.output_resolution(), input_resolution);
        assert_eq!(block.output_height(), 14);
        assert_eq!(block.output_width(), 14);
        assert_eq!(block.num_heads(), num_heads);
        assert_eq!(block.window_size(), 7);
        assert_eq!(block.shift_size(), 0);
        assert!(!block.swa_enabled());
        assert!(block.enable_qkv_bias());
        assert_eq!(block.drop_rate(), 0.0);
        assert_eq!(block.attn_drop_rate(), 0.0);
        assert_eq!(block.mlp_ratio(), 4.0);
        assert_eq!(block.drop_path_rate(), 0.0);
    }

    #[should_panic(expected = "input_resolution must be greater than zero")]
    #[test]
    fn test_shifted_window_transformer_block_config_zero_resolution() {
        let d_input = 128;
        let num_heads = 4;
        let input_resolution = [0, 14];

        let config = ShiftedWindowTransformerBlockConfig::new(d_input, input_resolution, num_heads);

        let _d = config.init::<NdArray>(&Default::default());
    }

    #[should_panic(expected = "input_resolution must be divisible by window size")]
    #[test]
    fn test_shifted_window_transformer_block_config_invalid_resolution() {
        let d_input = 128;
        let num_heads = 4;
        let input_resolution = [15, 14]; // Not divisible by default window size of 7

        let config = ShiftedWindowTransformerBlockConfig::new(d_input, input_resolution, num_heads);

        let _d = config.init::<NdArray>(&Default::default());
    }

    #[should_panic(expected = "d_input must be greater than zero")]
    #[test]
    fn test_shifted_window_transformer_block_config_zero_d_input() {
        let d_input = 0; // Invalid d_input
        let num_heads = 4;
        let input_resolution = [14, 14];

        let config = ShiftedWindowTransformerBlockConfig::new(d_input, input_resolution, num_heads);

        let _d = config.init::<NdArray>(&Default::default());
    }

    #[should_panic(expected = "num_heads must be greater than zero")]
    #[test]
    fn test_shifted_window_transformer_block_config_zero_num_heads() {
        let d_input = 128;
        let num_heads = 0; // Invalid num_heads
        let input_resolution = [14, 14];

        let config = ShiftedWindowTransformerBlockConfig::new(d_input, input_resolution, num_heads);

        let _d = config.init::<NdArray>(&Default::default());
    }

    #[should_panic(expected = "window_size must be greater than zero")]
    #[test]
    fn test_shifted_window_transformer_block_config_zero_window_size() {
        let d_input = 128;
        let num_heads = 4;
        let input_resolution = [14, 14];
        let window_size = 0; // Invalid window size

        let config = ShiftedWindowTransformerBlockConfig::new(d_input, input_resolution, num_heads)
            .with_window_size(window_size);

        let _d = config.init::<NdArray>(&Default::default());
    }

    #[test]
    fn test_block() {
        let b = 1;
        let num_heads = 4;
        let channels_per_head = 3;
        let d_input = num_heads * channels_per_head;
        let window_size = 4;

        let h = 2 * window_size;
        let w = 3 * window_size;
        let input_resolution = [h, w];

        let config = ShiftedWindowTransformerBlockConfig::new(d_input, input_resolution, num_heads)
            .with_window_size(window_size);

        let device = Default::default();
        let block = config.init::<NdArray>(&device);

        let distribution = burn::tensor::Distribution::Uniform(0.0, 1.0);
        let input = Tensor::<NdArray, 3>::random([b, h * w, d_input], distribution, &device);

        let _output = block.forward(input.clone());

        assert_eq!(input.dims(), _output.dims());
    }
}
