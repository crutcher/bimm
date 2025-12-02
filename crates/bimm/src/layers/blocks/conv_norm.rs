//! # `ConvNorm` Module
//!
//! A [`ConvNorm2d`] module is a [`Conv2d`] layer followed by a [`BatchNorm`] layer.

use bimm_contracts::{assert_shape_contract_periodically, unpack_shape_contract};
use burn::config::Config;
use burn::module::Module;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::{BatchNorm, BatchNormConfig, Initializer};
use burn::prelude::{Backend, Tensor};

/// [`ConvNorm2d`] Meta.
pub trait ConvNorm2dMeta {
    /// Number of input channels.
    fn in_channels(&self) -> usize;

    /// Number of groups.
    fn groups(&self) -> usize;

    /// Number of output channels.
    fn out_channels(&self) -> usize;

    /// Get the stride.
    fn stride(&self) -> &[usize; 2];
}

/// [`ConvNorm2d`] Config.
#[derive(Config, Debug)]
pub struct ConvNorm2dConfig {
    /// The [`Conv2d`] config.
    pub conv: Conv2dConfig,
}

impl ConvNorm2dMeta for ConvNorm2dConfig {
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

impl From<Conv2dConfig> for ConvNorm2dConfig {
    fn from(conv: Conv2dConfig) -> Self {
        Self { conv }
    }
}

impl ConvNorm2dConfig {
    /// Initialize a [`ConvNorm2d`].
    pub fn init<B: Backend>(
        self,
        device: &B::Device,
    ) -> ConvNorm2d<B> {
        ConvNorm2d {
            conv: self.conv.init(device),

            norm: BatchNormConfig::new(self.conv.channels[1]).init(device),
        }
    }

    /// Set the initializer.
    pub fn with_initializer(
        self,
        initializer: Initializer,
    ) -> Self {
        Self {
            conv: self.conv.with_initializer(initializer),
        }
    }
}

/// Grouped [`Conv2d`] and [`BatchNorm`] layer.
#[derive(Module, Debug)]
pub struct ConvNorm2d<B: Backend> {
    /// Internal Conv2d layer.
    pub conv: Conv2d<B>,

    /// Internal Norm Layer.
    pub norm: BatchNorm<B>,
}

impl<B: Backend> ConvNorm2dMeta for ConvNorm2d<B> {
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

impl<B: Backend> ConvNorm2d<B> {
    /// Zero initialize the norm layer's weights.
    ///
    /// This is used by / referenced in upstream `ResNet` init.
    ///
    /// TODO: Track down the paper that recommends this.
    pub fn zero_init_norm(&mut self) {
        self.norm.gamma = self.norm.gamma.clone().map(|p| p.slice_fill([..], 0.0));
    }
    /// Forward Pass.
    pub fn forward(
        &self,
        input: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        let [batch, out_height, out_width] = unpack_shape_contract!(
            [
                "batch",
                "in_channels",
                "in_height" = "out_height" * "height_stride",
                "in_width" = "out_width" * "width_stride"
            ],
            &input.dims(),
            &["batch", "out_height", "out_width"],
            &[
                ("in_channels", self.in_channels()),
                ("height_stride", self.stride()[0]),
                ("width_stride", self.stride()[1]),
            ]
        );
        let x = self.conv.forward(input);

        let x = self.norm.forward(x);

        assert_shape_contract_periodically!(
            ["batch", "out_channels", "out_height", "out_width"],
            &x.dims(),
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
    use burn::nn::PaddingConfig2d;

    #[test]
    fn test_conv_norm_config() {
        let inner_config = Conv2dConfig::new([2, 4], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_bias(false);

        let config: ConvNorm2dConfig = inner_config.clone().into();

        assert_eq!(&config.conv.channels, &inner_config.channels);
        assert_eq!(&config.conv.kernel_size, &inner_config.kernel_size);
        assert_eq!(&config.conv.stride, &inner_config.stride);
    }
}
