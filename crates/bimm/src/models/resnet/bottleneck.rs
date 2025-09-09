//! # [`BottleneckBlock`] Block for `ResNet`
//!
//! [`BottleneckBlock`] is the bottleneck form of the core `ResNet` convolution unit.
//!
//! [`BottleneckBlockMeta`] defines a common meta API for [`BottleneckBlock`]
//! and [`BottleneckBlockConfig`].
//!
//! [`BottleneckBlockConfig`] implements [`Config`], and provides
//! [`BottleneckBlockConfig::init`] to initialize a [`BottleneckBlock`].
//!
//! [`BottleneckBlock`] implements [`Module`], and provides
//! [`BottleneckBlock::forward`].

use crate::compat::activation_wrapper::ActivationConfig;
use crate::compat::normalization_wrapper::NormalizationConfig;
use crate::layers::blocks::cna::{AbstractCNA2dConfig, CNA2d, CNA2dConfig, CNA2dMeta};
use crate::layers::drop::drop_block::{DropBlock2d, DropBlock2dConfig, DropBlockOptions};
use crate::layers::drop::drop_path::{DropPath, DropPathConfig};
use crate::models::resnet::downsample::{ConvDownsample, ConvDownsampleConfig};
use crate::models::resnet::util::stride_div_output_resolution;
use crate::utility::probability::expect_probability;
use burn::nn::conv::Conv2dConfig;
use burn::nn::{BatchNormConfig, PaddingConfig2d};
use burn::prelude::{Backend, Config, Module, Tensor};

/// [`BottleneckBlock`] Meta trait.
pub trait BottleneckBlockMeta {
    /// The number of input feature planes.
    fn in_planes(&self) -> usize;

    /// Dilation rate for conv layers.
    fn dilation(&self) -> usize;

    /// Configures the size of `first_planes` and `out_planes`.
    fn planes(&self) -> usize;

    /// Groups of the conv filters.
    fn cardinality(&self) -> usize;

    /// Control factor for `width()`.
    fn base_width(&self) -> usize;

    /// Control factor for `first_planes()`
    fn reduction_factor(&self) -> usize;

    /// Control factor for `out_planes()`
    fn expansion_factor(&self) -> usize;

    /// First conv/norm layer output channels.
    ///
    /// ``first_planes = planes // reduction_factor``
    fn first_planes(&self) -> usize {
        self.width() / self.reduction_factor()
    }

    /// Get Width Plane.
    ///
    /// ``planes * (base_width / 64) * cardinality``
    fn width(&self) -> usize {
        self.planes() * (self.base_width() / 64) * self.cardinality()
    }

    /// The number of output feature planes.
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

/// [`BottleneckBlock`] Config.
///
/// Implements [`BottleneckBlockMeta`].
#[derive(Config, Debug)]
pub struct BottleneckBlockConfig {
    /// The size of the in channels dimension.
    pub in_planes: usize,

    /// Configures the size of `first_planes` and `out_planes`.
    pub planes: usize,

    /// Groups of the conv filters.
    #[config(default = "4")]
    pub cardinality: usize,

    /// Base width used to determine the number of output channels.
    #[config(default = "64")]
    pub base_width: usize,

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

impl BottleneckBlockMeta for BottleneckBlockConfig {
    fn in_planes(&self) -> usize {
        self.in_planes
    }

    fn dilation(&self) -> usize {
        self.dilation
    }

    fn planes(&self) -> usize {
        self.planes
    }

    fn cardinality(&self) -> usize {
        self.cardinality
    }

    fn base_width(&self) -> usize {
        self.base_width
    }

    fn reduction_factor(&self) -> usize {
        self.reduction_factor
    }

    fn expansion_factor(&self) -> usize {
        self.expansion_factor
    }

    fn stride(&self) -> usize {
        self.stride
    }
}

impl BottleneckBlockConfig {
    /// Initialize a [`BottleneckBlock`].
    pub fn init<B: Backend>(
        self,
        device: &B::Device,
    ) -> BottleneckBlock<B> {
        let drop_path_prob = expect_probability(self.drop_path_prob);

        let in_planes = self.in_planes();
        let width = self.width();
        let first_planes = width / self.reduction_factor;
        let out_planes = self.out_planes();

        let dilation = self.dilation();

        let stride = self.stride();

        // TODO: conditional stride logic for anti-aliasing.
        // use_aa = aa_layer is not None and stride == 2
        // stride = 1 if use_aa else stride

        let downsample = if stride != 1 || in_planes != out_planes {
            // TODO: mechanism to select different pool operations.
            ConvDownsampleConfig::new(in_planes, out_planes)
                .with_stride(stride)
                .into()
        } else {
            None
        };

        let cna_builder = AbstractCNA2dConfig {
            norm: self.normalization.clone(),
            act: self.activation.clone(),
        };

        let cna1: CNA2dConfig = cna_builder
            .build_config(Conv2dConfig::new([in_planes, first_planes], [1, 1]).with_bias(false));

        let cna2: CNA2dConfig = cna_builder.build_config(
            Conv2dConfig::new([first_planes, width], [3, 3])
                .with_stride([stride, stride])
                .with_dilation([dilation, dilation])
                .with_padding(PaddingConfig2d::Explicit(dilation, dilation))
                .with_groups(self.cardinality())
                .with_bias(false),
        );

        let cna3: CNA2dConfig = cna_builder
            .build_config(Conv2dConfig::new([width, out_planes], [1, 1]).with_bias(false));

        BottleneckBlock {
            base_width: self.base_width,
            expansion_factor: self.expansion_factor,
            reduction_factor: self.reduction_factor,

            downsample: downsample.as_ref().map(|c| c.init(device)),

            cna1: cna1.init(device),
            cna2: cna2.init(device),
            cna3: cna3.init(device),

            drop_block: self
                .drop_block
                .as_ref()
                .map(|options| DropBlock2dConfig::from(options.clone()).init()),
            ae: None,

            se: None,
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

/// Bottleneck Block for `ResNet`.
///
/// Implements [`BottleneckBlockMeta`].
#[derive(Module, Debug)]
pub struct BottleneckBlock<B: Backend> {
    /// Base width.
    pub base_width: usize,

    /// Expansion factor.
    pub expansion_factor: usize,

    /// Reduction factor.
    pub reduction_factor: usize,

    /// Optional `DownSample` layer; for the residual connection.
    pub downsample: Option<ConvDownsample<B>>,

    /// First conv/norm/act layer.
    pub cna1: CNA2d<B>,
    /// Second conv/norm/act layer.
    pub cna2: CNA2d<B>,
    /// Third conv/norm/act layer.
    pub cna3: CNA2d<B>,

    /// Optional `DropBlock` layer.
    pub drop_block: Option<DropBlock2d>,

    /// Optional anti-aliasing layer.
    // TODO: aa: anti-aliasing layer
    pub ae: Option<usize>,

    /// Optional attention layer.
    // TODO: se: attention layer
    pub se: Option<usize>,

    /// Optional `DropPath` layer.
    pub drop_path: Option<DropPath>,
}

impl<B: Backend> BottleneckBlockMeta for BottleneckBlock<B> {
    fn in_planes(&self) -> usize {
        self.cna1.in_channels()
    }

    fn dilation(&self) -> usize {
        self.cna3.conv.dilation[0]
    }

    fn planes(&self) -> usize {
        self.out_planes() / self.expansion_factor()
    }

    fn cardinality(&self) -> usize {
        self.cna2.groups()
    }

    fn base_width(&self) -> usize {
        self.base_width
    }

    fn reduction_factor(&self) -> usize {
        self.reduction_factor
    }

    fn expansion_factor(&self) -> usize {
        self.expansion_factor
    }

    fn first_planes(&self) -> usize {
        self.cna1.out_channels()
    }

    fn width(&self) -> usize {
        self.cna3.in_channels()
    }

    fn out_planes(&self) -> usize {
        self.cna3.out_channels()
    }

    fn stride(&self) -> usize {
        self.cna2.stride()[0]
    }
}

impl<B: Backend> BottleneckBlock<B> {
    /// Forward Pass.
    ///
    /// # Arguments
    ///
    /// - `input`: ``[batch, in_planes, in_height=out_height*stride, in_width=out_width*stride]``.
    ///
    /// # Returns
    ///
    /// A ``[batch, out_planes=planes*expansion_factor, out_height, out_width]`` tensor;
    pub fn forward(
        &self,
        input: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        #[cfg(debug_assertions)]
        let [batch, in_height, out_height, in_width, out_width] = bimm_contracts::unpack_shape_contract!(
            [
                "batch",
                "in_planes",
                "in_height" = "out_height" * "stride",
                "in_width" = "out_width" * "stride"
            ],
            &input,
            &["batch", "in_height", "out_height", "in_width", "out_width"],
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
        #[cfg(debug_assertions)]
        bimm_contracts::assert_shape_contract_periodically!(OUT_CONTRACT, &identity, &out_bindings);

        let x = self.cna1.forward(input);

        #[cfg(debug_assertions)]
        bimm_contracts::assert_shape_contract_periodically!(
            ["batch", "first_planes", "in_height", "in_width"],
            &x,
            &[
                ("batch", batch),
                ("first_planes", self.first_planes()),
                ("in_height", in_height),
                ("in_width", in_width),
            ],
        );

        let x = self.cna2.hook_forward(x, |x| match &self.drop_block {
            Some(drop_block) => drop_block.forward(x),
            None => x,
        });

        let x = match &self.ae {
            Some(_) => unimplemented!("anti-aliasing is not implemented"),
            None => x,
        };

        self.cna3.hook_forward(x, |x| {
            #[cfg(debug_assertions)]
            bimm_contracts::assert_shape_contract_periodically!(OUT_CONTRACT, &x, &out_bindings);

            let x = match &self.se {
                Some(_) => unimplemented!("attention is not implemented"),
                None => x,
            };
            let x = match &self.drop_path {
                Some(drop_path) => drop_path.forward(x),
                None => x,
            };

            x + identity
        })
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
    use burn::backend::NdArray;

    #[test]
    fn test_basic_block_config() {
        let in_channels = 16;
        let out_channels = 32;
        let config = BottleneckBlockConfig::new(in_channels, out_channels);
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
        let config = BottleneckBlockConfig::new(16, 32).with_stride(2);
        assert_eq!(config.stride(), 2);
        config.output_resolution([7, 7]);
    }

    #[test]
    fn test_basic_block_meta() {
        type B = NdArray<f32>;
        let device = Default::default();

        let in_channels = 2;
        let out_channels = in_channels;

        let block: BottleneckBlock<B> =
            BottleneckBlockConfig::new(in_channels, out_channels).init(&device);

        assert_eq!(block.in_planes(), in_channels);
        assert_eq!(block.out_planes(), out_channels);
        assert_eq!(block.stride(), 1);
        assert_eq!(block.output_resolution([16, 16]), [16, 16]);
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn test_conv2d_example_metal() {
        // FIXME: Conv2d with groups is broken in 0.18.0; but fixed in 0.19.0
        type B = burn::backend::Wgpu;
        let device = Default::default();

        let input: Tensor<B, 4> = Tensor::ones([2, 32, 64, 64], &device);

        let layer = Conv2dConfig::new([32, 32], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_groups(4)
            .with_bias(false)
            .init(&device);

        let result = layer.forward(input);
        assert_eq!(&result.shape().dims, &[2, 32, 64, 64]);
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn test_basic_block_forward_same_channels_no_downsample_autodiff() {
        // FIXME: Conv2d with groups is broken in 0.18.0; but fixed in 0.19.0
        use bimm_contracts::assert_shape_contract;
        use burn::backend::{Autodiff, Wgpu};
        type B = Autodiff<Wgpu>;

        let device = Default::default();

        let batch_size = 2;
        let in_planes = 2;
        let planes = 8;
        let in_height = 8;
        let in_width = 8;

        let block: BottleneckBlock<B> = BottleneckBlockConfig::new(in_planes, planes).init(&device);
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

    #[cfg(feature = "wgpu")]
    #[test]
    fn test_basic_block_forward_downsample_drop_block_drop_path_autodiff() {
        // FIXME: Conv2d with groups is broken in 0.18.0; but fixed in 0.19.0
        use bimm_contracts::assert_shape_contract;
        use burn::backend::{Autodiff, Wgpu};
        type B = Autodiff<Wgpu>;

        let device = Default::default();

        let batch_size = 2;
        let in_planes = 2;
        let planes = 4;
        let in_height = 8;
        let in_width = 8;

        let block: BottleneckBlock<B> = BottleneckBlockConfig::new(in_planes, planes)
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
