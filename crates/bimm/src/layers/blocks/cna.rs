//! # `CNA2d` - conv/norm/activation block.
//!
//! A [`CNA2d`] module is:
//! * a [`Conv2d`] layer,
//! * a [`Normalization`] layer,
//! * a [`Activation`] layer.
//!
//! With support for hooking the forward method,
//! to run code between the norm and application layers.

use crate::compat::activation_wrapper::{Activation, ActivationConfig};
use crate::compat::normalization_wrapper::{Normalization, NormalizationConfig};
use bimm_contracts::{assert_shape_contract_periodically, unpack_shape_contract};
use burn::config::Config;
use burn::module::Module;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::prelude::{Backend, Tensor};

/// Abstract policy for [`CNA2d`] Config.
///
/// Defines a [`NormalizationConfig`] and [`ActivationConfig`],
/// and can be lifted to a [`CNA2dConfig`] to match a [`Conv2dConfig`].
///
/// The abstract [`NormalizationConfig`] will be feature matched
/// with the target [`Conv2dConfig`].
#[derive(Config, Debug)]
pub struct AbstractCNA2dConfig {
    /// The [`Normalization`] config.
    pub norm: NormalizationConfig,

    /// Activation Config.
    #[config(default = "ActivationConfig::Relu")]
    pub act: ActivationConfig,
}

impl AbstractCNA2dConfig {
    /// Merge with a [`Conv2dConfig`] to construct a [`CNA2dConfig`].
    ///
    /// The abstract [`NormalizationConfig`] will be feature matched
    /// with the target [`Conv2dConfig`], resulting in a normalization
    /// layer sized appropriately for the input convolution.
    pub fn build_config(
        &self,
        conv: Conv2dConfig,
    ) -> CNA2dConfig {
        CNA2dConfig {
            conv,
            norm: self.norm.clone(),
            act: self.act.clone(),
        }
        .match_norm_features()
    }
}

/// [`CNA2d`] Meta.
pub trait CNA2dMeta {
    /// Number of input channels.
    fn in_channels(&self) -> usize;

    /// Number of groups.
    fn groups(&self) -> usize;

    /// Number of output channels.
    fn out_channels(&self) -> usize;

    /// Get the stride.
    fn stride(&self) -> [usize; 2];
}

/// [`CNA2d`] Config.
///
/// Implements [`CNA2dMeta`].
#[derive(Config, Debug)]
pub struct CNA2dConfig {
    /// The [`Conv2d`] config.
    pub conv: Conv2dConfig,

    /// The [`Normalization`] config.
    pub norm: NormalizationConfig,

    /// The [`Activation`] config.
    #[config(default = "ActivationConfig::Relu")]
    pub act: ActivationConfig,
}

impl CNA2dMeta for CNA2dConfig {
    fn in_channels(&self) -> usize {
        self.conv.channels[0]
    }

    fn groups(&self) -> usize {
        self.conv.groups
    }

    fn out_channels(&self) -> usize {
        self.conv.channels[1]
    }

    fn stride(&self) -> [usize; 2] {
        self.conv.stride.clone()
    }
}

impl CNA2dConfig {
    /// Initialize a [`CNA2d`].
    ///
    /// Auto-matches the norm layer input channels
    /// to the conv layer's output channels.
    pub fn init<B: Backend>(
        self,
        device: &B::Device,
    ) -> CNA2d<B> {
        let cfg = self.match_norm_features();
        CNA2d {
            conv: cfg.conv.init(device),
            norm: cfg.norm.init(device),
            act: cfg.act.init(device),
        }
    }

    /// Adjust the norm features to match the conv output size.
    ///
    /// ['`CNA2dConfig::init`'] does this automatically.
    pub fn match_norm_features(self) -> Self {
        let features = self.out_channels();
        let norm = self.norm.with_num_features(features);
        Self { norm, ..self }
    }
}

/// Sequenced conv/norm/activation block.
///
/// A [`CNA2d`] module is:
/// * a [`Conv2d`] layer,
/// * a [`Normalization`] layer,
/// * a [`Activation`] layer.
///
/// With support for hooking the forward method,
/// to run code between the norm and application layers.
///
/// Implements [`CNA2dMeta`].
#[derive(Module, Debug)]
pub struct CNA2d<B: Backend> {
    /// Internal Conv2d layer.
    pub conv: Conv2d<B>,

    /// Internal Norm Layer.
    pub norm: Normalization<B>,

    /// Activation layer.
    pub act: Activation<B>,
}

impl<B: Backend> CNA2dMeta for CNA2d<B> {
    fn in_channels(&self) -> usize {
        self.conv.weight.shape().dims[1] * self.groups()
    }

    fn groups(&self) -> usize {
        self.conv.groups
    }

    fn out_channels(&self) -> usize {
        self.conv.weight.shape().dims[0]
    }

    fn stride(&self) -> [usize; 2] {
        self.conv.stride.clone()
    }
}

impl<B: Backend> CNA2d<B> {
    /// Forward Pass.
    ///
    /// Applies the conv/norm/act layers in sequence:
    ///
    /// ```rust,ignore
    /// let x = self.conv.forward(input);
    /// let x = self.norm.forward(x);
    /// let x = self.act.forward(x);
    /// return x
    /// ```
    ///
    /// # Arguments
    ///
    /// - `input`: \
    ///   ``[batch, in_channels, in_height=out_height*stride, in_width=out_width*stride]``.
    ///
    /// # Returns
    ///
    /// ``[batch, out_channels, out_height, out_width]``
    pub fn forward(
        &self,
        input: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        self.hook_forward(input, |x| x)
    }

    /// Hooked Forward Pass.
    ///
    /// Applies the hook after normalization but before activation.
    ///
    /// ```rust,ignore
    /// let x = self.conv.forward(input);
    /// let x = self.norm.forward(x);
    /// let x = hook(x);
    /// let x = self.act.forward(x);
    /// return x
    /// ```
    ///
    /// # Arguments
    ///
    /// - `input`: \
    ///   ``[batch, in_channels, in_height=out_height*stride, in_width=out_width*stride]``.
    ///
    /// # Returns
    ///
    /// ``[batch, out_channels, out_height, out_width]``
    pub fn hook_forward<F>(
        &self,
        input: Tensor<B, 4>,
        hook: F,
    ) -> Tensor<B, 4>
    where
        F: FnOnce(Tensor<B, 4>) -> Tensor<B, 4>,
    {
        let [batch, out_height, out_width] = unpack_shape_contract!(
            [
                "batch",
                "in_channels",
                "in_height" = "out_height" * "height_stride",
                "in_width" = "out_width" * "width_stride"
            ],
            &input,
            &["batch", "out_height", "out_width"],
            &[
                ("in_channels", self.in_channels()),
                ("height_stride", self.stride()[0]),
                ("width_stride", self.stride()[1]),
            ]
        );
        let x = self.conv.forward(input);

        assert_shape_contract_periodically!(
            ["batch", "out_channels", "out_height", "out_width"],
            &x,
            &[
                ("batch", batch),
                ("out_channels", self.out_channels()),
                ("out_height", out_height),
                ("out_width", out_width)
            ]
        );

        let x = self.norm.forward(x);

        let x = hook(x);

        let x = self.act.forward(x);

        assert_shape_contract_periodically!(
            ["batch", "out_channels", "out_height", "out_width"],
            &x,
            &[
                ("batch", batch),
                ("out_channels", self.out_channels()),
                ("out_height", out_height),
                ("out_width", out_width)
            ]
        );

        x
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::{Autodiff, NdArray};
    use burn::nn::{BatchNormConfig, PaddingConfig2d};
    use burn::tensor::Distribution;

    #[test]
    fn test_conv_norm_config() {
        let abstract_config =
            AbstractCNA2dConfig::new(NormalizationConfig::Batch(BatchNormConfig::new(0)));

        let conv_config = Conv2dConfig::new([2, 4], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_bias(false);

        let config: CNA2dConfig = abstract_config.build_config(conv_config.clone());

        assert_eq!(config.in_channels(), 2);
        assert_eq!(config.out_channels(), 4);
        assert_eq!(config.groups(), 1);
        assert_eq!(config.stride(), [2, 2]);
    }

    #[test]
    fn test_cna() {
        type B = Autodiff<NdArray<f32>>;
        let device = Default::default();

        let config = CNA2dConfig::new(
            Conv2dConfig::new([2, 4], [3, 3])
                .with_stride([2, 2])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .with_bias(false),
            NormalizationConfig::Batch(BatchNormConfig::new(0)),
        )
        .with_act(ActivationConfig::Relu);

        let layer: CNA2d<B> = config.init(&device);
        assert_eq!(layer.in_channels(), 2);
        assert_eq!(layer.out_channels(), 4);
        assert_eq!(layer.groups(), 1);
        assert_eq!(layer.stride(), [2, 2]);

        let batch_size = 2;
        let height = 10;
        let width = 10;
        let channels = 2;

        let input = Tensor::random(
            [batch_size, channels, height, width],
            Distribution::Default,
            &device,
        );

        {
            let output = layer.forward(input.clone());
            let expected = {
                let x = layer.conv.forward(input.clone());
                let x = layer.norm.forward(x);
                let x = layer.act.forward(x);
                x
            };
            output.to_data().assert_eq(&expected.to_data(), true);
        }

        {
            let hook = |x| x * 2.0;

            let output = layer.hook_forward(input.clone(), hook);
            let expected = {
                let x = layer.conv.forward(input.clone());
                let x = layer.norm.forward(x);
                let x = hook(x);
                let x = layer.act.forward(x);
                x
            };
            output.to_data().assert_eq(&expected.to_data(), true);
        }
    }
}
