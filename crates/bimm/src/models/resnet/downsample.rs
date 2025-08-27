//! # The `ResNet` Downsample Implementation.

use crate::models::resnet::util::CONV_INTO_RELU_INITIALIZER;
use crate::models::resnet::util::stride_div_output_resolution;
use bimm_contracts::{assert_shape_contract_periodically, unpack_shape_contract};
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::{BatchNorm, BatchNormConfig, Initializer, PaddingConfig2d};
use burn::prelude::{Backend, Config, Module, Tensor};

/// [`ConvDownsample`] Meta trait.
pub trait ConvDownsampleMeta {
    /// The size of the in channels dimension.
    fn in_channels(&self) -> usize;

    /// The size of the out channels dimension.
    fn out_channels(&self) -> usize;

    /// The stride of the downsample layer.
    fn stride(&self) -> usize;

    /// Get the output resolution for a given input resolution.
    ///
    /// The input must be a multiple of the stride.
    ///
    /// # Arguments
    ///
    /// - `input_resolution`: ``[in_height=out_height*stride, in_width=out_width*stride]``.
    ///
    /// # Returns
    ///
    /// ``[out_height, out_width]``
    ///
    /// # Panics
    ///
    /// If the input resolution is not a multiple of the stride.
    fn output_resolution(
        &self,
        input_resolution: [usize; 2],
    ) -> [usize; 2] {
        stride_div_output_resolution(input_resolution, self.stride())
    }
}

/// [`ConvDownsample`] configuration.
#[derive(Config, Debug)]
pub struct ConvDownsampleConfig {
    /// The size of the in channels dimension.
    in_channels: usize,

    /// The size of the out channels dimension.
    out_channels: usize,

    /// The stride of the downsample layer.
    #[config(default = 1)]
    stride: usize,

    /// The [`Conv2D`] initializer.
    #[config(default = "CONV_INTO_RELU_INITIALIZER.clone()")]
    pub initializer: Initializer,
}

impl ConvDownsampleMeta for ConvDownsampleConfig {
    #[inline]
    fn in_channels(&self) -> usize {
        self.in_channels
    }

    #[inline]
    fn out_channels(&self) -> usize {
        self.out_channels
    }

    #[inline]
    fn stride(&self) -> usize {
        self.stride
    }
}

impl ConvDownsampleConfig {
    /// Initialize a [`ConvDownsample`] `Module`.
    pub fn init<B: Backend>(
        &self,
        device: &B::Device,
    ) -> ConvDownsample<B> {
        ConvDownsample {
            conv: Conv2dConfig::new([self.in_channels, self.out_channels], [1, 1])
                .with_stride([self.stride, self.stride])
                .with_padding(PaddingConfig2d::Explicit(0, 0))
                .with_initializer(self.initializer.clone())
                .with_bias(false)
                .init(device),

            bn: BatchNormConfig::new(self.out_channels).init(device),
        }
    }
}

/// Downsample layer applies a 1x1 conv to reduce the resolution (H, W) and adjust the number of channels.
///
/// Maps ``[batch_size, in_channels, in_height, in_width]`` to
/// ``[batch_size, out_channels, out_height, out_width]`` tensors.
#[derive(Module, Debug)]
pub struct ConvDownsample<B: Backend> {
    conv: Conv2d<B>,
    bn: BatchNorm<B, 2>,
}

impl<B: Backend> ConvDownsampleMeta for ConvDownsample<B> {
    #[inline]
    fn in_channels(&self) -> usize {
        self.conv.weight.shape().dims[1]
    }

    #[inline]
    fn out_channels(&self) -> usize {
        self.conv.weight.shape().dims[0]
    }

    #[inline]
    fn stride(&self) -> usize {
        self.conv.stride[0]
    }
}

impl<B: Backend> ConvDownsample<B> {
    /// Forward pass.
    ///
    /// # Arguments
    ///
    /// - `input`: a ``[batch, in_channels, in_height=out_height*stride, in_width=out_width*stride]`` tensor.
    ///
    /// # Returns
    ///
    /// A ``[batch_size, out_channels, h_out, w_out]`` tensor.
    pub fn forward(
        &self,
        input: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        let [batch, out_height, out_width] = unpack_shape_contract!(
            [
                "batch",
                "in_channels",
                "in_height" = "out_height" * "stride",
                "in_width" = "out_width" * "stride"
            ],
            &input,
            &["batch", "out_height", "out_width"],
            &[
                ("in_channels", self.in_channels()),
                ("stride", self.stride())
            ]
        );

        let out = self.conv.forward(input);
        let out = self.bn.forward(out);

        assert_shape_contract_periodically!(
            ["batch", "out_channels", "out_height", "out_width"],
            &out,
            &[
                ("batch", batch),
                ("out_channels", self.out_channels()),
                ("out_height", out_height),
                ("out_width", out_width)
            ]
        );

        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bimm_contracts::assert_shape_contract;
    use burn::backend::NdArray;

    #[test]
    fn test_downsample_config() {
        let config = ConvDownsampleConfig::new(2, 4);
        assert_eq!(config.in_channels(), 2);
        assert_eq!(config.out_channels(), 4);
        assert_eq!(config.stride(), 1);
        assert_eq!(config.output_resolution([8, 8]), [8, 8]);

        let config = config.with_stride(2);
        assert_eq!(config.stride(), 2);
        assert_eq!(config.output_resolution([8, 8]), [4, 4]);
    }

    #[test]
    #[should_panic(expected = "7 !~ in_height=(out_height*stride)")]
    fn test_downsample_config_panic() {
        let config = ConvDownsampleConfig::new(2, 4).with_stride(2);
        config.output_resolution([7, 7]);
    }

    #[test]
    fn test_downsample() {
        type B = NdArray<f32>;
        let device = Default::default();

        let batch_size = 2;
        let in_channels = 2;
        let out_channels = 4;
        let in_height = 8;
        let in_width = 8;

        let downsample: ConvDownsample<B> = ConvDownsampleConfig::new(in_channels, out_channels)
            .with_stride(2)
            .init(&device);

        let tensor = Tensor::ones([batch_size, in_channels, in_height, in_width], &device);
        let out = downsample.forward(tensor);

        assert_shape_contract!(
            ["batch", "out_channels", "out_height", "out_width"],
            &out,
            &[
                ("batch", batch_size),
                ("out_channels", out_channels),
                ("out_height", in_height / 2),
                ("out_width", in_width / 2)
            ]
        );
    }
}
