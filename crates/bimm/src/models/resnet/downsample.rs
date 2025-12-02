//! # The `ResNet` Downsample Implementation.

use crate::compat::conv_shape::expect_conv_output_shape;
use crate::models::resnet::util::scalar_to_array;
use crate::models::resnet::util::{build_square_conv2d_padding_config, get_square_conv2d_padding};
use bimm_contracts::{assert_shape_contract_periodically, unpack_shape_contract};
use burn::nn::BatchNormConfig;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::norm::{Normalization, NormalizationConfig};
use burn::prelude::{Backend, Config, Module, Tensor};

/// [`ResNetDownsample`] Meta trait.
///
/// Implemented by:
/// * [`ResNetDownsampleConfig`]
/// * [`ResNetDownsample`]
///
/// # Missing Features
///
/// - *avg*: support for average pooling is blocked on support for ``ceil_mode`` in [`burn`].
pub trait ResNetDownsampleMeta {
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
        expect_conv_output_shape(
            input_resolution,
            scalar_to_array(self.kernel_size()),
            scalar_to_array(self.stride()),
            scalar_to_array(get_square_conv2d_padding(
                self.kernel_size(),
                self.stride(),
                self.dilation(),
            )),
            scalar_to_array(self.dilation()),
        )
    }
}

/// [`ResNetDownsample`] configuration.
///
/// Implements [`ResNetDownsampleMeta`].
///
/// # Missing Features
///
/// - *avg*: support for average pooling is blocked on support for ``ceil_mode`` in [`burn`].
#[derive(Config, Debug)]
pub struct ResNetDownsampleConfig {
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

impl ResNetDownsampleMeta for ResNetDownsampleConfig {
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

impl ResNetDownsampleConfig {
    /// Initialize a [`ResNetDownsample`] `Module`.
    pub fn init<B: Backend>(
        self,
        device: &B::Device,
    ) -> ResNetDownsample<B> {
        let kernel_size = if self.stride == 1 && self.dilation == 1 {
            1
        } else {
            self.kernel_size
        };
        let dilation = if kernel_size > 1 { self.dilation } else { 1 };
        let padding = build_square_conv2d_padding_config(kernel_size, self.stride, dilation);

        let conv = Conv2dConfig::new(
            [self.in_channels, self.out_channels],
            scalar_to_array(kernel_size),
        )
        .with_stride(scalar_to_array(self.stride))
        .with_padding(padding)
        .with_dilation(scalar_to_array(dilation))
        .with_bias(false);

        ResNetDownsample {
            conv: conv.init(device),

            norm: self.norm.with_num_features(self.out_channels).init(device),
        }
    }
}

/// `ResNet` Downsample Layer.
///
/// Implements [`ResNetDownsampleMeta`].
///
/// # Missing Features
///
/// - *avg*: support for average pooling is blocked on support for ``ceil_mode`` in [`burn`].
#[derive(Module, Debug)]
pub struct ResNetDownsample<B: Backend> {
    /// Conv layer.
    pub conv: Conv2d<B>,

    /// Norm layer.
    pub norm: Normalization<B>,
}

impl<B: Backend> ResNetDownsampleMeta for ResNetDownsample<B> {
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

impl<B: Backend> ResNetDownsample<B> {
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
        let [batch, in_height, in_width] = unpack_shape_contract!(
            ["batch", "in_channels", "in_height", "in_width",],
            &input.dims(),
            &["batch", "in_height", "in_width"],
            &[("in_channels", self.in_channels()),]
        );

        let out = self.conv.forward(input);
        let out = self.norm.forward(out);

        let [out_height, out_width] = self.output_resolution([in_height, in_width]);

        assert_shape_contract_periodically!(
            ["batch", "out_channels", "out_height", "out_width"],
            &out.dims(),
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
        let config = ResNetDownsampleConfig::new(2, 4, 3);
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

    #[test]
    fn test_downsample() {
        type B = NdArray<f32>;
        let device = Default::default();

        let batch_size = 2;
        let in_channels = 2;
        let out_channels = 4;
        let in_height = 8;
        let in_width = 8;

        let downsample: ResNetDownsample<B> =
            ResNetDownsampleConfig::new(in_channels, out_channels, 1)
                .with_stride(2)
                .init(&device);

        let tensor = Tensor::ones([batch_size, in_channels, in_height, in_width], &device);
        let out = downsample.forward(tensor);

        assert_shape_contract!(
            ["batch", "out_channels", "out_height", "out_width"],
            &out.dims(),
            &[
                ("batch", batch_size),
                ("out_channels", out_channels),
                ("out_height", in_height / 2),
                ("out_width", in_width / 2)
            ]
        );
    }
}
