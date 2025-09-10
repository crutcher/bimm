//! # The `ResNet` Downsample Implementation.

use crate::compat::normalization_wrapper::{Normalization, NormalizationConfig};
use crate::layers::blocks::conv_norm::{ConvNorm2d, ConvNorm2dConfig, ConvNorm2dMeta};
use crate::layers::pool::{AvgPool2dSame, AvgPool2dSameConfig};
use crate::models::resnet::util::{build_square_conv2d_padding_config, CONV_INTO_RELU_INITIALIZER};
use crate::models::resnet::util::{scalar_to_array, stride_div_output_resolution};
use bimm_contracts::{assert_shape_contract_periodically, unpack_shape_contract};
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::pool::{AvgPool2d, AvgPool2dConfig};
use burn::nn::{BatchNormConfig, Initializer, PaddingConfig2d};
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

    /// The [`ConvNorm2d`] initializer.
    #[config(default = "CONV_INTO_RELU_INITIALIZER.clone()")]
    pub initializer: Initializer,
}

impl ConvDownsampleMeta for ConvDownsampleConfig {
    fn in_channels(&self) -> usize {
        self.in_channels
    }

    fn out_channels(&self) -> usize {
        self.out_channels
    }

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
        let config: ConvNorm2dConfig =
            Conv2dConfig::new([self.in_channels, self.out_channels], [1, 1])
                .with_stride([self.stride, self.stride])
                .with_padding(PaddingConfig2d::Explicit(0, 0))
                .with_initializer(self.initializer.clone())
                .with_bias(false)
                .into();

        ConvDownsample {
            conv_norm: config.init(device),
        }
    }
}

/// Downsample layer applies a 1x1 conv to reduce the resolution (H, W) and adjust the number of channels.
///
/// Maps ``[batch_size, in_channels, in_height, in_width]`` to
/// ``[batch_size, out_channels, out_height, out_width]`` tensors.
#[derive(Module, Debug)]
pub struct ConvDownsample<B: Backend> {
    /// Embedded conv/norm.
    pub conv_norm: ConvNorm2d<B>,
}

impl<B: Backend> ConvDownsampleMeta for ConvDownsample<B> {
    fn in_channels(&self) -> usize {
        self.conv_norm.in_channels()
    }

    fn out_channels(&self) -> usize {
        self.conv_norm.out_channels()
    }

    fn stride(&self) -> usize {
        self.conv_norm.stride()[0]
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

        let out = self.conv_norm.forward(input);

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

/// [`AvgPool`] Config.
#[derive(Config, Debug)]
pub enum AvgPoolConfig {
    /// [`AvgPool2d`] Config.
    Avg(AvgPool2dConfig),

    /// [`AvgPool2dSame`] Config.
    AvgSame(AvgPool2dSameConfig),
}

impl From<AvgPool2dConfig> for AvgPoolConfig {
    fn from(config: AvgPool2dConfig) -> Self {
        AvgPoolConfig::Avg(config)
    }
}

impl From<AvgPool2dSameConfig> for AvgPoolConfig {
    fn from(config: AvgPool2dSameConfig) -> Self {
        AvgPoolConfig::AvgSame(config)
    }
}

impl AvgPoolConfig {
    /// Initialize a [`AvgPool`].
    pub fn init(self) -> AvgPool {
        match self {
            AvgPoolConfig::Avg(config) => config.init().into(),
            AvgPoolConfig::AvgSame(config) => config.init().into(),
        }
    }
}

/// AvgPool Wrapper.
#[derive(Module, Clone, Debug)]
pub enum AvgPool {
    /// [`AvgPool2d`] Layer.
    Avg(AvgPool2d),

    /// [`AvgPool2dSame`] Layer.
    AvgSame(AvgPool2dSame),
}

impl From<AvgPool2d> for AvgPool {
    fn from(pool: AvgPool2d) -> Self {
        AvgPool::Avg(pool)
    }
}

impl From<AvgPool2dSame> for AvgPool {
    fn from(pool: AvgPool2dSame) -> Self {
        AvgPool::AvgSame(pool)
    }
}

/// [`Downsample`] Meta trait.
pub trait DownsampleMeta {
    /// The size of the in channels dimension.
    fn in_channels(&self) -> usize;

    /// The size of the out channels dimension.
    fn out_channels(&self) -> usize;

    /// The kernel size of conv.
    fn kernel_size(&self) -> usize;

    /// The dilation of the conv.
    fn dilation(&self) -> usize;

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

/// [`Downsample`] configuration.
///
/// Implements [`DownsampleMeta`].
#[derive(Config, Debug)]
pub struct DownsampleConfig {
    /// The size of the in channels dimension.
    in_channels: usize,

    /// The size of the out channels dimension.
    out_channels: usize,

    /// The kernel size of conv.
    kernel_size: usize,

    /// The stride of the conv.
    #[config(default = 1)]
    stride: usize,

    /// The dilation of the conv.
    #[config(default = 1)]
    dilation: usize,

    /// The [`Normalization`] config.
    ///
    /// The feature size will be auto-matched.
    #[config(default = "NormalizationConfig::Batch(BatchNormConfig::new(0))")]
    norm: NormalizationConfig,
}

impl DownsampleMeta for DownsampleConfig {
    fn in_channels(&self) -> usize {
        self.in_channels
    }

    fn out_channels(&self) -> usize {
        self.out_channels
    }

    fn kernel_size(&self) -> usize {
        self.kernel_size
    }

    fn dilation(&self) -> usize {
        self.dilation
    }

    fn stride(&self) -> usize {
        self.stride
    }
}

impl DownsampleConfig {
    /// Initialize a [`Downsample`] `Module`.
    pub fn init<B: Backend>(
        &self,
        device: &B::Device,
    ) -> Downsample<B> {
        Downsample {
            pool: None,
            conv: Conv2dConfig::new(
                scalar_to_array(self.in_channels),
                scalar_to_array(self.kernel_size),
            )
            .with_stride(scalar_to_array(self.stride))
            .with_padding(build_square_conv2d_padding_config(
                self.kernel_size,
                self.stride,
                self.dilation,
            ))
            .with_dilation(scalar_to_array(self.dilation))
            .init(device),

            norm: self.norm.init(device),
        }
    }
}

/// `ResNet` downsample layer.
///
/// Implements [`DownsampleMeta`].
#[derive(Module, Debug)]
pub struct Downsample<B: Backend> {
    pool: Option<AvgPool>,
    conv: Conv2d<B>,
    norm: Normalization<B>,
}

impl<B: Backend> DownsampleMeta for Downsample<B> {
    fn in_channels(&self) -> usize {
        self.conv.weight.shape().dims[1]
    }

    fn out_channels(&self) -> usize {
        self.conv.weight.shape().dims[0]
    }

    fn kernel_size(&self) -> usize {
        self.conv.kernel_size[0]
    }

    fn dilation(&self) -> usize {
        self.conv.dilation[0]
    }

    fn stride(&self) -> usize {
        self.conv.stride[0]
    }
}

impl<B: Backend> Downsample<B> {
    /// Forward pass.
    ///
    /// # Arguments
    ///
    /// - `input`: \
    ///   ``[batch, in_channels, in_height=out_height*stride, in_width=out_width*stride]``
    ///
    /// # Returns
    ///
    /// ``[batch_size, out_channels, h_out, w_out]``
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
        let out = self.norm.forward(out);

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
    fn test_conv_downsample_config() {
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
    fn test_conv_downsample_config_panic() {
        let config = ConvDownsampleConfig::new(2, 4).with_stride(2);
        config.output_resolution([7, 7]);
    }

    #[test]
    fn test_conv_downsample() {
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

    #[test]
    fn test_downsample_config() {
        let config = DownsampleConfig::new(2, 4, 3);
        assert_eq!(config.in_channels(), 2);
        assert_eq!(config.out_channels(), 4);
        assert_eq!(config.kernel_size(), 3);
        assert_eq!(config.stride(), 1);
        assert_eq!(config.dilation(), 1);
        assert_eq!(config.output_resolution([8, 8]), [8, 8]);

        let config = config.with_stride(2);
        assert_eq!(config.stride(), 2);
        assert_eq!(config.output_resolution([8, 8]), [4, 4]);
    }
}
