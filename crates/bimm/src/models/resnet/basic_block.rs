//! # Basic Block for `ResNet`

use crate::layers::activation::{ActivationLayer, ActivationLayerConfig};
use crate::layers::drop::drop_block::{DropBlock2d, DropBlock2dConfig, DropBlockOptions};
use crate::models::resnet::downsample::ConvDownsample;
use crate::models::resnet::util;
use bimm_contracts::{
    assert_shape_contract_periodically, define_shape_contract, unpack_shape_contract,
};
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::{BatchNorm, BatchNormConfig, Initializer, PaddingConfig2d};
use burn::prelude::{Backend, Config, Module, Tensor};

/// [`BasicBlock`] Meta trait.
pub trait BasicBlockMeta {
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
    /// - `input_resolution`: ``[height_in=height_out*stride, width_in=width_out*stride]``.
    ///
    /// # Returns
    ///
    /// ``[height_out, width_out]``
    ///
    /// # Panics
    ///
    /// If the input resolution is not a multiple of the stride.
    fn output_resolution(
        &self,
        input_resolution: [usize; 2],
    ) -> [usize; 2] {
        util::stride_div_output_resolution(input_resolution, self.stride())
    }
}

/// [`BasicBlock`] Config.
#[derive(Config, Debug)]
pub struct BasicBlockConfig {
    /// The size of the in channels dimension.
    pub in_channels: usize,

    /// The size of the out channels dimension.
    pub out_channels: usize,

    /// The stride of the downsample layer.
    #[config(default = 1)]
    pub stride: usize,

    /// The drop block config.
    #[config(default = "None")]
    pub drop_block: Option<DropBlockOptions>,

    /// The activation layer config.
    #[config(default = "ActivationLayerConfig::Relu")]
    pub activation: ActivationLayerConfig,

    /// The [`Conv2D`] initializer.
    /// Default is recommended for use with Relu.
    #[config(
        default = "Initializer::KaimingNormal{gain:std::f64::consts::SQRT_2, fan_out_only:true}"
    )]
    pub initializer: Initializer,
}

impl BasicBlockMeta for BasicBlockConfig {
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

impl BasicBlockConfig {
    /// Initialize a [`BasicBlock`].
    pub fn init<B: Backend>(
        self,
        device: &B::Device,
    ) -> BasicBlock<B> {
        if self.in_channels != self.out_channels {
            panic!("in_channels != out_channels requires downsample layer")
        }

        BasicBlock {
            conv1: Conv2dConfig::new([self.in_channels, self.out_channels], [3, 3])
                .with_stride([self.stride, self.stride])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .with_bias(false)
                .with_initializer(self.initializer.clone())
                .init(device),

            bn1: BatchNormConfig::new(self.out_channels).init(device),

            drop_block: self.drop_block.as_ref().map(|options| {
                DropBlock2dConfig::new()
                    .with_options(options.clone())
                    .init()
            }),

            act1: self.activation.init(device),

            conv2: Conv2dConfig::new([self.out_channels, self.out_channels], [3, 3])
                .with_stride([1, 1])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .with_bias(false)
                .with_initializer(self.initializer)
                .init(device),

            bn2: BatchNormConfig::new(self.out_channels).init(device),

            act2: self.activation.init(device),

            downsample: None,
        }
    }
}

/// Basic Block for `ResNet`.
#[derive(Module, Debug)]
pub struct BasicBlock<B: Backend> {
    conv1: Conv2d<B>,
    bn1: BatchNorm<B, 2>,
    drop_block: Option<DropBlock2d>,
    act1: ActivationLayer<B>,

    // TODO: aa: anti-aliasing layer
    conv2: Conv2d<B>,
    bn2: BatchNorm<B, 2>,
    act2: ActivationLayer<B>,

    // TODO: se: attention layer
    // TODO: drop_path: drop path layer
    downsample: Option<ConvDownsample<B>>,
}

impl<B: Backend> BasicBlockMeta for BasicBlock<B> {
    fn in_channels(&self) -> usize {
        self.conv1.weight.shape().dims[1]
    }

    fn out_channels(&self) -> usize {
        self.conv2.weight.shape().dims[0]
    }

    fn stride(&self) -> usize {
        self.conv1.stride[0]
    }
}

impl<B: Backend> BasicBlock<B> {
    /// Forward Pass.
    ///
    /// # Arguments
    ///
    /// - `input`: ``[batch, in_channels, in_height=out_height*stride, in_width=out_width*stride]``.
    ///
    /// # Returns
    ///
    /// A ``[batch, out_channels, out_height, out_width]`` tensor.
    pub fn forward(
        &self,
        input: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        let shortcut = input.clone();

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
            ],
        );
        define_shape_contract!(
            OUT_CONTRACT,
            ["batch", "out_channels", "out_height", "out_width"]
        );
        let bindings = [
            ("batch", batch),
            ("out_channels", self.out_channels()),
            ("out_height", out_height),
            ("out_width", out_width),
        ];

        let x = self.conv1.forward(input);
        assert_shape_contract_periodically!(OUT_CONTRACT, &x, &bindings);
        let x = self.bn1.forward(x);
        let x = match &self.drop_block {
            Some(drop_block) => drop_block.forward(x),
            None => x,
        };
        let x = self.act1.forward(x);

        // aa? - anti-aliasing?

        let x = self.conv2.forward(x);
        let x = self.bn2.forward(x);

        // se? - attention?
        // drop_path?

        let shortcut = match &self.downsample {
            Some(downsample) => downsample.forward(shortcut),
            None => shortcut,
        };
        assert_shape_contract_periodically!(OUT_CONTRACT, &shortcut, &bindings);

        let x = x + shortcut;
        let x = self.act2.forward(x);

        assert_shape_contract_periodically!(OUT_CONTRACT, &x, &bindings);
        x
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bimm_contracts::assert_shape_contract;
    use burn::backend::{Autodiff, NdArray};

    #[test]
    fn test_basic_block_config() {
        let in_channels = 16;
        let out_channels = 32;
        let config = BasicBlockConfig::new(in_channels, out_channels);
        assert_eq!(config.in_channels(), in_channels);
        assert_eq!(config.out_channels(), out_channels);
        assert_eq!(config.stride(), 1);
        assert_eq!(config.output_resolution([16, 16]), [16, 16]);
        assert!(matches!(config.activation, ActivationLayerConfig::Relu));

        let config = config
            .with_stride(2)
            .with_activation(ActivationLayerConfig::Sigmoid);
        assert_eq!(config.stride(), 2);
        assert_eq!(config.output_resolution([16, 16]), [8, 8]);
        assert!(matches!(config.activation, ActivationLayerConfig::Sigmoid));
    }

    #[test]
    #[should_panic(expected = "7 !~ height_in=(height_out*stride)")]
    fn test_downsample_config_panic() {
        let config = BasicBlockConfig::new(16, 32).with_stride(2);
        assert_eq!(config.stride(), 2);
        config.output_resolution([7, 7]);
    }

    #[test]
    fn test_basic_block_meta() {
        type B = NdArray<f32>;
        let device = Default::default();

        let in_channels = 2;
        let out_channels = in_channels;

        let block: BasicBlock<B> = BasicBlockConfig::new(in_channels, out_channels).init(&device);

        assert_eq!(block.in_channels(), in_channels);
        assert_eq!(block.out_channels(), out_channels);
        assert_eq!(block.stride(), 1);
        assert_eq!(block.output_resolution([16, 16]), [16, 16]);
    }

    #[test]
    fn test_basic_block_forward_same_channels_no_downsample_autodiff() {
        type B = Autodiff<NdArray<f32>>;
        let device = Default::default();

        let batch_size = 2;
        let in_channels = 2;
        let out_channels = in_channels;
        let height_in = 8;
        let width_in = 8;

        let block: BasicBlock<B> = BasicBlockConfig::new(in_channels, out_channels)
            .with_drop_block(Some(DropBlockOptions::default()))
            .init(&device);

        let input = Tensor::ones([batch_size, in_channels, height_in, width_in], &device);
        let output = block.forward(input);

        assert_shape_contract!(
            ["batch", "out_channels", "out_height", "out_width"],
            &output,
            &[
                ("batch", batch_size),
                ("out_channels", out_channels),
                ("out_height", height_in),
                ("out_width", width_in)
            ],
        );
    }
}
