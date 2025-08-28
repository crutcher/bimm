use burn::config::Config;
use burn::module::Module;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::{BatchNorm, BatchNormConfig};
use burn::prelude::{Backend, Tensor};

/// [`ConvNorm`] Config.
#[derive(Config, Debug)]
pub struct ConvNormConfig {
    /// The [`Conv2D`] config.
    pub conv: Conv2dConfig,
}

impl From<Conv2dConfig> for ConvNormConfig {
    fn from(conv: Conv2dConfig) -> Self {
        Self { conv }
    }
}

impl ConvNormConfig {
    /// Initialize a [`ConvNorm`].
    pub fn init<B: Backend>(
        self,
        device: &B::Device,
    ) -> ConvNorm<B> {
        ConvNorm {
            conv: self.conv.init(device),

            norm: BatchNormConfig::new(self.conv.channels[1]).init(device),
        }
    }
}

/// Grouped [`Conv2d`] and [`BatchNorm`] layer.
#[derive(Module, Debug)]
pub struct ConvNorm<B: Backend> {
    /// Internal Conv2d layer.
    pub conv: Conv2d<B>,

    /// Internal Norm Layer.
    pub norm: BatchNorm<B, 2>,
}

impl<B: Backend> ConvNorm<B> {
    /// Forward Pass.
    pub fn forward(
        &self,
        input: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        let x = self.conv.forward(input);

        self.norm.forward(x)
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

        let config: ConvNormConfig = inner_config.clone().into();

        assert_eq!(&config.conv.channels, &inner_config.channels);
        assert_eq!(&config.conv.kernel_size, &inner_config.kernel_size);
        assert_eq!(&config.conv.stride, &inner_config.stride);
    }
}
