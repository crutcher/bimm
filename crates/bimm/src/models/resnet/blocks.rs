//! # Basic Block for `ResNet`

use crate::layers::activation::{ActivationLayer, ActivationLayerConfig};
use crate::layers::drop::drop_block::{DropBlock2d, DropBlock2dConfig, DropBlockOptions};
use crate::layers::drop::drop_path::{DropPath, DropPathConfig};
use crate::models::resnet::conv_norm::{ConvNorm, ConvNormConfig, ConvNormMeta};
use crate::models::resnet::downsample::{ConvDownsample, ConvDownsampleConfig};
use crate::models::resnet::util::{CONV_INTO_RELU_INITIALIZER, stride_div_output_resolution};
use crate::utility::probability::expect_probability;
use bimm_contracts::{
    assert_shape_contract_periodically, define_shape_contract, unpack_shape_contract,
};
use burn::nn::conv::Conv2dConfig;
use burn::nn::{Initializer, PaddingConfig2d};
use burn::prelude::{Backend, Config, Module, Tensor};

/// [`BasicBlock`] Meta trait.
pub trait BasicBlockMeta {
    /// The size of the in channels dimension.
    fn in_planes(&self) -> usize;

    /// Configures the size of `first_planes` and `out_planes`.
    fn planes(&self) -> usize;

    /// Control factor for `out_planes()`
    fn out_planes_expansion_factor(&self) -> usize;

    /// The size of the out channels dimension.
    ///
    /// ``out_planes = planes * out_planes_expansion_factor``
    fn out_planes(&self) -> usize {
        self.planes() * self.out_planes_expansion_factor()
    }

    /// Control factor for `first_planes()`
    fn first_planes_reduction_factor(&self) -> usize;

    /// First conv/norm layer output channels.
    ///
    /// ``first_planes = planes // first_planes_reduction_factor``
    fn first_planes(&self) -> usize {
        self.planes() / self.first_planes_reduction_factor()
    }

    /// The stride of the downsample layer.
    fn stride(&self) -> usize;

    /// Dilation rate for conv layers.
    fn dilation(&self) -> usize;

    /// Optional dilation rate for the first conv.
    fn first_dilation(&self) -> Option<usize>;

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

/// [`BasicBlock`] Config.
#[derive(Config, Debug)]
pub struct BasicBlockConfig {
    /// The size of the in channels dimension.
    pub in_planes: usize,

    /// Configures the `out_planes` as a function of `expansion_factor`.
    pub planes: usize,

    /// Control factor for `out_planes()`
    #[config(default = 1)]
    pub out_planes_expansion_factor: usize,

    /// Control factor for `first_planes()`
    #[config(default = 1)]
    pub first_planes_reduction_factor: usize,

    /// The stride of the downsample layer.
    #[config(default = 1)]
    pub stride: usize,

    /// Dilation rate for conv layers.
    #[config(default = 1)]
    pub dilation: usize,

    /// Optional dilation rate for the first conv.
    #[config(default = "None")]
    pub first_dilation: Option<usize>,

    /// Drop path probability.
    #[config(default = "0.0")]
    pub drop_path_prob: f64,

    /// The drop block config.
    #[config(default = "None")]
    pub drop_block: Option<DropBlockOptions>,

    /// The activation layer config.
    #[config(default = "ActivationLayerConfig::Relu")]
    pub activation: ActivationLayerConfig,

    /// The [`Conv2D`] initializer.
    #[config(default = "CONV_INTO_RELU_INITIALIZER.clone()")]
    pub initializer: Initializer,
}

impl BasicBlockMeta for BasicBlockConfig {
    fn in_planes(&self) -> usize {
        self.in_planes
    }

    fn planes(&self) -> usize {
        self.planes
    }

    fn out_planes_expansion_factor(&self) -> usize {
        self.out_planes_expansion_factor
    }

    fn first_planes_reduction_factor(&self) -> usize {
        self.first_planes_reduction_factor
    }

    fn stride(&self) -> usize {
        self.stride
    }

    fn dilation(&self) -> usize {
        self.dilation
    }

    fn first_dilation(&self) -> Option<usize> {
        self.first_dilation
    }
}

impl BasicBlockConfig {
    /// Initialize a [`BasicBlock`].
    pub fn init<B: Backend>(
        self,
        device: &B::Device,
    ) -> BasicBlock<B> {
        let drop_path_prob = expect_probability(self.drop_path_prob);

        let in_planes = self.in_planes();
        let first_planes = self.first_planes();
        let first_dilation = self.first_dilation().unwrap_or(self.dilation());
        let stride = self.stride();

        // TODO: conditional stride logic for anti-aliasing.
        let first_stride = stride;

        let conv_norm1_cfg = ConvNormConfig::from(
            Conv2dConfig::new([in_planes, first_planes], [3, 3])
                .with_stride([first_stride, first_stride])
                .with_dilation([first_dilation, first_dilation])
                .with_padding(PaddingConfig2d::Explicit(first_dilation, first_dilation))
                .with_bias(false)
                .with_initializer(self.initializer.clone()),
        );

        let drop_block_cfg = match &self.drop_block {
            Some(options) => DropBlock2dConfig::from(options.clone()).into(),
            None => None,
        };

        let out_planes = self.out_planes();
        let dilation = self.dilation();

        let conv_norm2_cfg = ConvNormConfig::from(
            Conv2dConfig::new([first_planes, out_planes], [3, 3])
                .with_stride([1, 1])
                .with_dilation([dilation, dilation])
                .with_padding(PaddingConfig2d::Explicit(dilation, dilation))
                .with_bias(false)
                .with_initializer(self.initializer.clone()),
        );

        let drop_path_cfg = if drop_path_prob == 0.0 {
            None
        } else {
            DropPathConfig::new().with_drop_prob(drop_path_prob).into()
        };

        let downsample_required = self.stride() != 1 || self.in_planes() != self.out_planes();
        let residual_downsample_cfg = if downsample_required {
            // If present, downsample is used to adapt the skip connection.
            ConvDownsampleConfig::new(self.in_planes(), self.out_planes())
                .with_stride(self.stride())
                .with_initializer(self.initializer)
                .into()
        } else {
            None
        };

        BasicBlock {
            out_planes_expansion_factor: self.out_planes_expansion_factor,
            first_planes_reduction_factor: self.first_planes_reduction_factor,

            conv_norm1: conv_norm1_cfg.init(device),

            drop_block: drop_block_cfg.map(|cfg| cfg.init()),

            act1: self.activation.init(device),

            conv_norm2: conv_norm2_cfg.init(device),

            act2: self.activation.init(device),

            drop_path: drop_path_cfg.map(|cfg| cfg.init()),

            residual_downsample: residual_downsample_cfg.map(|cfg| cfg.init(device)),
        }
    }
}

/// Basic Block for `ResNet`.
#[derive(Module, Debug)]
pub struct BasicBlock<B: Backend> {
    out_planes_expansion_factor: usize,
    first_planes_reduction_factor: usize,

    /// First conv/norm layer.
    pub conv_norm1: ConvNorm<B>,

    /// Optional `DropBlock` layer.
    pub drop_block: Option<DropBlock2d>,

    /// First conv/norm layer.
    pub act1: ActivationLayer<B>,

    // TODO: aa: anti-aliasing layer
    /// Second conv/norm layer.
    pub conv_norm2: ConvNorm<B>,

    /// Second activation layer.
    pub act2: ActivationLayer<B>,

    // TODO: se: attention layer
    // TODO: drop_path: drop path layer
    /// Optional `DropPath` layer.
    pub drop_path: Option<DropPath>,

    /// Optional `DownSample` layer; for the residual connection.
    pub residual_downsample: Option<ConvDownsample<B>>,
}

impl<B: Backend> BasicBlockMeta for BasicBlock<B> {
    fn in_planes(&self) -> usize {
        self.conv_norm1.in_channels()
    }

    fn planes(&self) -> usize {
        self.conv_norm1.out_channels() / self.out_planes_expansion_factor()
    }

    fn out_planes_expansion_factor(&self) -> usize {
        self.out_planes_expansion_factor
    }

    fn out_planes(&self) -> usize {
        self.conv_norm2.out_channels()
    }

    fn first_planes_reduction_factor(&self) -> usize {
        self.first_planes_reduction_factor
    }

    fn first_planes(&self) -> usize {
        self.conv_norm1.out_channels()
    }

    fn stride(&self) -> usize {
        self.conv_norm1.stride()[0]
    }

    fn dilation(&self) -> usize {
        self.conv_norm2.conv.dilation[0]
    }

    fn first_dilation(&self) -> Option<usize> {
        let d1 = self.conv_norm1.conv.dilation[0];
        let d2 = self.conv_norm2.conv.dilation[0];
        if d1 == d2 { None } else { Some(d1) }
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
        let [batch, out_height, out_width] = unpack_shape_contract!(
            [
                "batch",
                "in_channels",
                "in_height" = "out_height" * "stride",
                "in_width" = "out_width" * "stride"
            ],
            &input,
            &["batch", "out_height", "out_width"],
            &[("in_channels", self.in_planes()), ("stride", self.stride())],
        );
        define_shape_contract!(
            OUT_CONTRACT,
            ["batch", "out_channels", "out_height", "out_width"]
        );
        let bindings = [
            ("batch", batch),
            ("out_channels", self.out_planes()),
            ("out_height", out_height),
            ("out_width", out_width),
        ];

        let shortcut = input.clone();
        let shortcut = match &self.residual_downsample {
            // If present, downsample is used to adapt the skip connection.
            Some(downsample) => downsample.forward(shortcut),
            None => shortcut,
        };
        assert_shape_contract_periodically!(OUT_CONTRACT, &shortcut, &bindings);

        let x = self.conv_norm1.forward(input);
        // This is the only main operation that changes the shape of the input.
        assert_shape_contract_periodically!(OUT_CONTRACT, &x, &bindings);

        let x = match &self.drop_block {
            Some(drop_block) => drop_block.forward(x),
            None => x,
        };
        let x = self.act1.forward(x);

        // aa? - anti-aliasing?

        let x = self.conv_norm2.forward(x);

        // se? - attention?

        let x = match &self.drop_path {
            Some(drop_path) => drop_path.forward(x),
            None => x,
        };

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
        assert_eq!(config.in_planes(), in_channels);
        assert_eq!(config.out_planes(), out_channels);
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
    #[should_panic(expected = "7 !~ in_height=(out_height*stride)")]
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

        assert_eq!(block.in_planes(), in_channels);
        assert_eq!(block.out_planes(), out_channels);
        assert_eq!(block.stride(), 1);
        assert_eq!(block.output_resolution([16, 16]), [16, 16]);
    }

    #[test]
    fn test_basic_block_forward_same_channels_no_downsample_autodiff() {
        type B = Autodiff<NdArray<f32>>;
        let device = Default::default();

        let batch_size = 2;
        let in_planes = 2;
        let planes = 8;
        let in_height = 8;
        let in_width = 8;

        let block: BasicBlock<B> = BasicBlockConfig::new(in_planes, planes).init(&device);
        let out_planes = block.out_planes();

        let input = Tensor::ones([batch_size, in_planes, in_height, in_width], &device);
        let output = block.forward(input);

        assert_shape_contract!(
            ["batch", "out_channels", "out_height", "out_width"],
            &output,
            &[
                ("batch", batch_size),
                ("out_channels", out_planes),
                ("out_height", in_height),
                ("out_width", in_width)
            ],
        );
    }

    #[test]
    fn test_basic_block_forward_downsample_drop_block_drop_path_autodiff() {
        type B = Autodiff<NdArray<f32>>;
        let device = Default::default();

        let batch_size = 2;
        let in_planes = 2;
        let planes = 4;
        let in_height = 8;
        let in_width = 8;

        let block: BasicBlock<B> = BasicBlockConfig::new(in_planes, planes)
            .with_drop_path_prob(0.1)
            .with_drop_block(Some(DropBlockOptions::default()))
            .with_stride(2)
            .init(&device);

        let out_planes = block.out_planes();

        let [out_height, out_width] = block.output_resolution([in_height, in_width]);
        assert_eq!(out_height, 4);
        assert_eq!(out_width, 4);

        let input = Tensor::ones([batch_size, in_planes, in_height, in_width], &device);
        let output = block.forward(input);

        assert_shape_contract!(
            ["batch", "out_channels", "out_height", "out_width"],
            &output,
            &[
                ("batch", batch_size),
                ("out_channels", out_planes),
                ("out_height", out_height),
                ("out_width", out_width)
            ],
        );
    }
}
