//! # Basic Block for `ResNet`
//!
//! [`BasicBlock`] is the core `ResNet` convolution unit.
//!
//! [`BasicBlockMeta`] defines a common meta API for [`BasicBlock`]
//! and [`BasicBlockConfig`].
//!
//! [`BasicBlockConfig`] implements [`Config`], and provides
//! [`BasicBlockConfig::init`] to initialize a [`BasicBlock`].
//!
//! [`BasicBlock`] implements [`Module`], and provides
//! [`BasicBlock::forward`].

use crate::layers::blocks::cna::{AbstractCNA2dConfig, CNA2d, CNA2dConfig, CNA2dMeta};
use crate::layers::drop::drop_block::{DropBlock2d, DropBlock2dConfig, DropBlockOptions};
use crate::layers::drop::drop_path::{DropPath, DropPathConfig};
use crate::models::resnet::downsample::{ResNetDownsample, ResNetDownsampleConfig};
use crate::models::resnet::util::{scalar_to_array, stride_div_output_resolution};
use crate::utility::probability::expect_probability;
use burn::nn::BatchNormConfig;
use burn::nn::PaddingConfig2d;
use burn::nn::activation::ActivationConfig;
use burn::nn::conv::Conv2dConfig;
use burn::nn::norm::NormalizationConfig;
use burn::prelude::{Backend, Config, Module, Tensor};

/// [`BasicBlock`] Meta trait.
pub trait BasicBlockMeta {
    /// The size of the in channels dimension.
    fn in_planes(&self) -> usize;

    /// Dilation rate for conv layers.
    fn dilation(&self) -> usize;

    /// Optional dilation rate for the first conv.
    fn first_dilation(&self) -> Option<usize>;

    /// Configures the size of `first_planes` and `out_planes`.
    fn planes(&self) -> usize;

    /// Control factor for `out_planes()`
    fn expansion_factor(&self) -> usize;

    /// Control factor for `first_planes()`
    fn reduction_factor(&self) -> usize;

    /// First conv/norm layer output channels.
    ///
    /// ``first_planes = planes // reduction_factor``
    fn first_planes(&self) -> usize {
        self.planes() / self.reduction_factor()
    }

    /// The size of the out channels dimension.
    ///
    /// ``out_planes = planes * expansion_factor``
    fn out_planes(&self) -> usize {
        self.planes() * self.expansion_factor()
    }

    /// The stride of convolution.
    ///
    /// Affects downsample behavior.
    fn stride(&self) -> usize;

    /// Get the output resolution for a given input resolution.
    ///
    /// The input must be a multiple of the stride.
    ///
    /// # Arguments
    ///
    /// - `input_resolution`: \
    ///   ``[in_height=out_height*stride, in_width=out_width*stride]``.
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

    /// Effective first dilation.
    ///
    /// Resolves `first_dilation()` vrs `dilation()`.
    fn effective_first_dilation(&self) -> usize {
        self.first_dilation().unwrap_or(self.dilation())
    }
}

/// [`BasicBlock`] Config.
///
/// Implements [`BasicBlockMeta`].
#[derive(Config, Debug)]
pub struct BasicBlockConfig {
    /// The size of the in channels dimension.
    pub in_planes: usize,

    /// Configures the `out_planes` as a function of `expansion_factor`.
    pub planes: usize,

    /// Control factor for `out_planes()`
    #[config(default = 1)]
    pub expansion_factor: usize,

    /// Control factor for `first_planes()`
    #[config(default = 1)]
    pub reduction_factor: usize,

    /// The stride of the downsample layer.
    #[config(default = 1)]
    pub stride: usize,

    /// Dilation rate for conv layers.
    #[config(default = 1)]
    pub dilation: usize,

    /// Optional dilation rate for the first conv.
    #[config(default = "None")]
    pub first_dilation: Option<usize>,

    /// Size of the kernel for the downsample layer.
    #[config(default = "1")]
    pub down_kernel_size: usize,

    /// Drop path probability.
    #[config(default = "0.0")]
    pub drop_path_prob: f64,

    /// The drop block config.
    #[config(default = "None")]
    pub drop_block: Option<DropBlockOptions>,

    /// [`crate::compat::normalization_wrapper::Normalization`] config.
    ///
    /// The feature size of this config will be replaced
    /// with the appropriate feature size for the input layer.
    #[config(default = "NormalizationConfig::Batch(BatchNormConfig::new(0))")]
    pub normalization: NormalizationConfig,

    /// [`crate::compat::activation_wrapper::Activation`] config.
    #[config(default = "ActivationConfig::Relu")]
    pub activation: ActivationConfig,
}

impl BasicBlockMeta for BasicBlockConfig {
    fn in_planes(&self) -> usize {
        self.in_planes
    }

    fn dilation(&self) -> usize {
        self.dilation
    }

    fn first_dilation(&self) -> Option<usize> {
        self.first_dilation
    }

    fn planes(&self) -> usize {
        self.planes
    }

    fn expansion_factor(&self) -> usize {
        self.expansion_factor
    }

    fn reduction_factor(&self) -> usize {
        self.reduction_factor
    }

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
        let drop_path_prob = expect_probability(self.drop_path_prob);

        let in_planes = self.in_planes();
        let first_planes = self.first_planes();
        let out_planes = self.out_planes();

        let first_dilation = self.effective_first_dilation();
        let dilation = self.dilation();

        let stride = self.stride();

        // TODO: conditional stride logic for anti-aliasing.
        // use_aa = aa_layer is not None and (stride == 2 or first_dilation != dilation)
        // stride = 1 if use_aa else stride

        let downsample = if stride != 1 || in_planes != out_planes {
            ResNetDownsampleConfig::new(self.in_planes(), self.out_planes(), self.down_kernel_size)
                .with_stride(self.stride())
                .with_dilation(first_dilation)
                .with_norm(self.normalization.clone())
                .into()
        } else {
            None
        };

        let cna_builder = AbstractCNA2dConfig {
            norm: self.normalization.clone(),
            act: self.activation.clone(),
        };

        let cna1: CNA2dConfig = cna_builder.build_config(
            Conv2dConfig::new([in_planes, first_planes], scalar_to_array(3))
                .with_stride(scalar_to_array(stride))
                .with_dilation(scalar_to_array(first_dilation))
                .with_padding(PaddingConfig2d::Explicit(first_dilation, first_dilation))
                .with_bias(false),
        );

        let cna2: CNA2dConfig = cna_builder.build_config(
            Conv2dConfig::new([first_planes, out_planes], scalar_to_array(3))
                .with_dilation(scalar_to_array(dilation))
                .with_padding(PaddingConfig2d::Explicit(dilation, dilation))
                .with_bias(false),
        );

        BasicBlock {
            expansion_factor: self.expansion_factor,
            reduction_factor: self.reduction_factor,

            downsample: downsample.as_ref().map(|cfg| cfg.clone().init(device)),

            // Group 1
            cna1: cna1.init(device),
            cna2: cna2.init(device),

            drop_block: self
                .drop_block
                .as_ref()
                .map(|options| DropBlock2dConfig::from(options.clone()).init()),
            drop_path: if drop_path_prob != 0.0 {
                DropPathConfig::new()
                    .with_drop_prob(drop_path_prob)
                    .init()
                    .into()
            } else {
                None
            },
        }
    }
}

/// Basic Block for `ResNet`.
///
/// Implements [`BasicBlockMeta`].
#[derive(Module, Debug)]
pub struct BasicBlock<B: Backend> {
    /// Expansion factor.
    pub expansion_factor: usize,

    /// Reduction factor.
    pub reduction_factor: usize,

    /// Optional `DownSample` layer; for the residual connection.
    pub downsample: Option<ResNetDownsample<B>>,

    /// First Conv/Norm/Act Block.
    pub cna1: CNA2d<B>,
    /// Second Conv/Norm/Act Block.
    pub cna2: CNA2d<B>,

    /// Optional `DropBlock` layer.
    pub drop_block: Option<DropBlock2d>,

    /// Optional `DropPath` layer.
    pub drop_path: Option<DropPath>,
}

impl<B: Backend> BasicBlockMeta for BasicBlock<B> {
    fn in_planes(&self) -> usize {
        self.cna1.in_channels()
    }

    fn dilation(&self) -> usize {
        self.cna1.conv.dilation[0]
    }

    fn first_dilation(&self) -> Option<usize> {
        let d1 = self.cna1.conv.dilation[0];
        let d2 = self.cna2.conv.dilation[0];
        if d1 == d2 { None } else { Some(d1) }
    }

    fn planes(&self) -> usize {
        self.cna1.out_channels() / self.expansion_factor()
    }

    fn expansion_factor(&self) -> usize {
        self.expansion_factor
    }

    fn reduction_factor(&self) -> usize {
        self.reduction_factor
    }

    fn first_planes(&self) -> usize {
        self.cna1.out_channels()
    }

    fn out_planes(&self) -> usize {
        self.cna2.out_channels()
    }

    fn stride(&self) -> usize {
        self.cna1.stride()[0]
    }
}

impl<B: Backend> BasicBlock<B> {
    /// Forward Pass.
    ///
    /// # Arguments
    ///
    /// - `input`: ``[batch, in_planes, in_height=out_height*stride, in_width=out_width*stride]``.
    ///
    /// # Returns
    ///
    /// A ``[batch, out_planes=planes*expansion_factor, out_height, out_width]`` tensor.
    pub fn forward(
        &self,
        input: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        #[cfg(debug_assertions)]
        let [batch, out_height, out_width] = bimm_contracts::unpack_shape_contract!(
            [
                "batch",
                "in_planes",
                "in_height" = "out_height" * "stride",
                "in_width" = "out_width" * "stride"
            ],
            &input,
            &["batch", "out_height", "out_width"],
            &[("in_planes", self.in_planes()), ("stride", self.stride())],
        );

        let identity = match &self.downsample {
            Some(downsample) => downsample.forward(input.clone()),
            None => input.clone(),
        };

        #[cfg(debug_assertions)]
        bimm_contracts::define_shape_contract!(
            OUT_CONTRACT,
            ["batch", "out_planes", "out_height", "out_width"],
        );
        #[cfg(debug_assertions)]
        let out_bindings = [
            ("batch", batch),
            ("out_planes", self.out_planes()),
            ("out_height", out_height),
            ("out_width", out_width),
        ];
        // #[cfg(debug_assertions)]
        //  bimm_contracts::assert_shape_contract_periodically!(OUT_CONTRACT, &identity, &out_bindings);

        let x = self.cna1.hook_forward(input, |x| match &self.drop_block {
            Some(drop_block) => drop_block.forward(x),
            None => x,
        });

        #[cfg(debug_assertions)]
        bimm_contracts::assert_shape_contract_periodically!(
            ["batch", "first_planes", "out_height", "out_width"],
            &x,
            &[
                ("batch", batch),
                ("first_planes", self.first_planes()),
                ("out_height", out_height),
                ("out_width", out_width),
            ]
        );

        // TODO: anti-aliasing

        let x = self.cna2.hook_forward(x, |x| {
            // TODO: attention

            let x = match &self.drop_path {
                Some(drop_path) => drop_path.forward(x),
                None => x,
            };

            x + identity
        });

        #[cfg(debug_assertions)]
        bimm_contracts::assert_shape_contract_periodically!(OUT_CONTRACT, &x, &out_bindings);

        x
    }

    /// Set the drop path probability.
    pub fn with_drop_path_prob(
        self,
        drop_path_prob: f64,
    ) -> Self {
        let drop_path_prob = expect_probability(drop_path_prob);
        Self {
            drop_path: if drop_path_prob == 0.0 {
                None
            } else {
                DropPathConfig::new()
                    .with_drop_prob(drop_path_prob)
                    .init()
                    .into()
            },
            ..self
        }
    }

    /// Set the drop block behavior.
    pub fn with_drop_block(
        self,
        drop_block: Option<DropBlockOptions>,
    ) -> Self {
        Self {
            drop_block: drop_block.map(|options| DropBlock2dConfig::from(options).init()),
            ..self
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bimm_contracts::assert_shape_contract;
    use burn::backend::{Autodiff, NdArray};
    use burn::nn::activation::ActivationConfig;

    #[test]
    fn test_basic_block_config() {
        let in_channels = 16;
        let out_channels = 32;
        let config = BasicBlockConfig::new(in_channels, out_channels);
        assert_eq!(config.in_planes(), in_channels);
        assert_eq!(config.out_planes(), out_channels);
        assert_eq!(config.stride(), 1);
        assert_eq!(config.output_resolution([16, 16]), [16, 16]);
        assert!(matches!(config.activation, ActivationConfig::Relu));

        let config = config
            .with_stride(2)
            .with_activation(ActivationConfig::Sigmoid);
        assert_eq!(config.stride(), 2);
        assert_eq!(config.output_resolution([16, 16]), [8, 8]);
        assert!(matches!(config.activation, ActivationConfig::Sigmoid));
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
