//! # The `ResNet` Downsample Implementation.

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

    /// Predict the output resolution.
    ///
    /// The input must be a multiple of the stride.
    ///
    /// # Arguments
    ///
    /// - `input_resolution`: ``[height_in, width_in]``.
    ///
    /// # Returns
    ///
    /// Output `[[height_out, width_out]]`.
    ///
    /// # Panics
    ///
    /// If the input resolution is not a multiple of the stride.
    fn output_resolution(
        &self,
        input_resolution: [usize; 2],
    ) -> [usize; 2] {
        let [h, w] = input_resolution;
        let stride = self.stride();
        assert!(
            h % self.stride() == 0 && w % self.stride() == 0,
            "input resolution {:?} is not a multiple of the stride {:?}",
            input_resolution,
            stride,
        );
        [h / stride, w / stride]
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

    /// The type of function used to initialize neural network parameters
    /// Default is recommended for use with Relu.
    #[config(
        default = "Initializer::KaimingNormal{gain:std::f64::consts::SQRT_2, fan_out_only:true}"
    )]
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
/// Maps ``[batch_size, channels_in, height_in, width_in]`` to
/// ``[batch_size, channels_out, height_out, width_out]`` tensors.
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
    /// - `input`: a ``[batch, channels_in, height_in=height_out*stride, width_in=width_out*stride]`` tensor.
    ///
    /// # Returns
    ///
    /// A ``[batch_size, channels_out, h_out, w_out]`` tensor.
    pub fn forward(
        &self,
        input: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        let [batch, height_out, width_out] = unpack_shape_contract!(
            [
                "batch",
                "channels_in",
                "height_in" = "height_out" * "stride",
                "width_in" = "width_out" * "stride"
            ],
            &input,
            &["batch", "height_out", "width_out"],
            &[("channels_in", self.in_channels()), ("stride", self.stride())]
        );

        let out = self.conv.forward(input);
        let out = self.bn.forward(out);

        assert_shape_contract_periodically!(
            ["batch", "channels_out", "height_out", "width_out"],
            &out,
            &[
                ("batch", batch),
                ("channels_out", self.out_channels()),
                ("height_out", height_out),
                ("width_out", width_out)
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
    #[should_panic(expected = "input resolution [7, 7] is not a multiple of the stride 2")]
    fn test_downsample_config_panic() {
        let config = ConvDownsampleConfig::new(2, 4)
            .with_stride(2);
        config.output_resolution([7, 7]);
    }

    #[test]
    fn test_downsample() {
        type B = NdArray<f32>;
        let device = Default::default();

        let batch_size = 2;
        let in_channels = 2;
        let out_channels = 4;
        let height_in = 8;
        let width_in = 8;

        let downsample: ConvDownsample<B> = ConvDownsampleConfig::new(in_channels, out_channels)
            .with_stride(2)
            .init(&device);

        let tensor = Tensor::ones([batch_size, in_channels, height_in, width_in], &device);
        let out = downsample.forward(tensor);

        assert_shape_contract!(
            ["batch", "channels_out", "height_out", "width_out"],
            &out,
            &[
                ("batch", batch_size),
                ("channels_out", out_channels),
                ("height_out", height_in / 2),
                ("width_out", width_in / 2)
            ]
        );
    }
}
