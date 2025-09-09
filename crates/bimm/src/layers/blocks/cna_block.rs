//! # `CNA2d` Blocks
//!
//! A [`CNA2d`] module is:
//! * a [`Conv2d`] layer,
//! * a [`BatchNorm`] layer,
//! * a [`Application`] layer.
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

/// [`CNA2d`] Meta.
pub trait CNA2dMeta {
    /// Number of input channels.
    fn in_channels(&self) -> usize;

    /// Number of groups.
    fn groups(&self) -> usize;

    /// Number of output channels.
    fn out_channels(&self) -> usize;

    /// Get the stride.
    fn stride(&self) -> &[usize; 2];
}

/// Abstract policy for [`CNA2d`] Config.
///
/// Defines a norm and act layer,
/// and can be lifted to a [`CNA2dConfig`]
/// to match a [`Conv2dConfig`].
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

/// [`CNA2d`] Config.
#[derive(Config, Debug)]
pub struct CNA2dConfig {
    /// The [`Conv2d`] config.
    pub conv: Conv2dConfig,

    /// The [`Normalization`] config.
    pub norm: NormalizationConfig,

    /// Activation Config.
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

    fn stride(&self) -> &[usize; 2] {
        &self.conv.stride
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

    /// Adjust the norm features to match the conv output.
    pub fn match_norm_features(self) -> Self {
        let features = self.out_channels();
        let norm = self.norm.with_num_features(features);
        Self { norm, ..self }
    }
}

/// Grouped [`Conv2d`] and [`BatchNorm`] layer.
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

    fn stride(&self) -> &[usize; 2] {
        &self.conv.stride
    }
}

impl<B: Backend> CNA2d<B> {
    /// Forward Pass.
    pub fn forward(
        &self,
        input: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        self.hook_forward(input, |x| x)
    }

    /// Hooked Forward Pass.
    ///
    /// Applies the hook after normalization, but before activation.
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
    use burn::nn::{BatchNormConfig, PaddingConfig2d};

    #[test]
    fn test_conv_norm_config() {
        let inner_config = Conv2dConfig::new([2, 4], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_bias(false);

        let config: CNA2dConfig = CNA2dConfig::new(
            inner_config.clone(),
            NormalizationConfig::Batch(BatchNormConfig::new(0)),
        );

        assert_eq!(&config.conv.channels, &inner_config.channels);
        assert_eq!(&config.conv.kernel_size, &inner_config.kernel_size);
        assert_eq!(&config.conv.stride, &inner_config.stride);
    }
}
