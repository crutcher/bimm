//! # Residual Block Wrapper
//!
//! [`CNAResidualBlock`] is a abstraction wrapper around either:
//! * [`CNABasicBlock`] - the basic `ResNet` conv block, or
//! * [`CNABottleneckBlock`] - the bottleneck variant `ResNet` conv block.
//!
//! [`CNAResidualBlockMeta`] defines a common meta api shared by:
//! * [`CNAResidualBlock`], and
//! * [`CNAResidualBlockConfig`]
//!
//! [`CNAResidualBlockConfig`] implements [`Config`], and provides an
//! [`CNAResidualBlockConfig::init`] constructor pathway to [`CNAResidualBlock`].
//!
//! [`CNAResidualBlock`] implements [`Module`],
//! and provides [`CNAResidualBlock::forward`].
//!
//! [`CNAResidualBlock`] can also be constructed via:
//! * [`From<CNABasicBlock<B>>`](`CNABasicBlock`),
//! * [`From<CNABottleneckBlock<B>>`](`CNABottleneckBlock`).
use crate::layers::drop::drop_block::DropBlockOptions;
use crate::models::resnet::cna_basic_block::{
    CNABasicBlock, CNABasicBlockConfig, CNABasicBlockMeta,
};
use crate::models::resnet::cna_bottleneck::{
    CNABottleneckBlock, CNABottleneckBlockConfig, CNABottleneckBlockMeta,
};
use crate::models::resnet::util::stride_div_output_resolution;
use crate::utility::probability::expect_probability;
use burn::prelude::{Backend, Config, Module, Tensor};

/// [`CNAResidualBlock`] Meta API.
///
/// Defines a shared API for [`CNAResidualBlock`] and [`CNAResidualBlockConfig`].
pub trait CNAResidualBlockMeta {
    /// The number of input feature planes.
    fn in_planes(&self) -> usize;

    /// The number of outpu feature planes.
    fn out_planes(&self) -> usize;

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
}

/// [`CNAResidualBlock`] Config.
///
/// Implements [`CNAResidualBlockMeta`].
#[derive(Config, Debug)]
pub enum CNAResidualBlockConfig {
    /// A `ResNet` [`CNABasicBlock`].
    Basic(CNABasicBlockConfig),

    /// A `ResNet` [`CNABottleneckBlock`].
    Bottleneck(CNABottleneckBlockConfig),
}

impl CNAResidualBlockMeta for CNAResidualBlockConfig {
    fn in_planes(&self) -> usize {
        match self {
            Self::Basic(config) => config.in_planes(),
            Self::Bottleneck(config) => config.in_planes(),
        }
    }

    fn out_planes(&self) -> usize {
        match self {
            Self::Basic(config) => config.out_planes(),
            Self::Bottleneck(config) => config.out_planes(),
        }
    }

    fn stride(&self) -> usize {
        match self {
            Self::Basic(config) => config.stride(),
            Self::Bottleneck(config) => config.stride(),
        }
    }

    fn output_resolution(
        &self,
        input_resolution: [usize; 2],
    ) -> [usize; 2] {
        match self {
            Self::Basic(config) => config.output_resolution(input_resolution),
            Self::Bottleneck(config) => config.output_resolution(input_resolution),
        }
    }
}

impl From<CNABasicBlockConfig> for CNAResidualBlockConfig {
    fn from(config: CNABasicBlockConfig) -> Self {
        Self::Basic(config)
    }
}

impl From<CNABottleneckBlockConfig> for CNAResidualBlockConfig {
    fn from(config: CNABottleneckBlockConfig) -> Self {
        Self::Bottleneck(config)
    }
}

impl CNAResidualBlockConfig {
    /// Initialize a [`CNAResidualBlock`].
    pub fn init<B: Backend>(
        self,
        device: &B::Device,
    ) -> CNAResidualBlock<B> {
        match self {
            Self::Basic(config) => config.init(device).into(),
            Self::Bottleneck(config) => config.init(device).into(),
        }
    }

    /// Legacy constructor.
    pub fn new(
        in_planes: usize,
        out_planes: usize,
        stride: usize,
        bottleneck: bool,
    ) -> Self {
        if bottleneck {
            CNABottleneckBlockConfig::new(in_planes, out_planes)
                .with_stride(stride)
                .into()
        } else {
            CNABasicBlockConfig::new(in_planes, out_planes)
                .with_stride(stride)
                .into()
        }
    }

    /// Set drop block options.
    pub fn with_drop_block(
        self,
        options: Option<DropBlockOptions>,
    ) -> Self {
        match self {
            Self::Basic(config) => config.with_drop_block(options).into(),
            Self::Bottleneck(config) => config.with_drop_block(options).into(),
        }
    }

    /// Set the drop path probability.
    pub fn with_drop_path_prob(
        self,
        drop_path_prob: f64,
    ) -> Self {
        let drop_path_prob = expect_probability(drop_path_prob);
        match self {
            Self::Basic(config) => config.with_drop_path_prob(drop_path_prob).into(),
            Self::Bottleneck(config) => config.with_drop_path_prob(drop_path_prob).into(),
        }
    }
}

/// A `ResNet` [`CNABasicBlock`] or [`CNABottleneckBlock`] wrapper.
#[derive(Module, Debug)]
#[allow(clippy::large_enum_variant)]
pub enum CNAResidualBlock<B: Backend> {
    /// A `ResNet` [`CNABasicBlock`].
    Basic(CNABasicBlock<B>),

    /// A `ResNet` [`CNABottleneckBlock`].
    Bottleneck(CNABottleneckBlock<B>),
}

impl<B: Backend> From<CNABasicBlock<B>> for CNAResidualBlock<B> {
    fn from(block: CNABasicBlock<B>) -> Self {
        Self::Basic(block)
    }
}

impl<B: Backend> From<CNABottleneckBlock<B>> for CNAResidualBlock<B> {
    fn from(block: CNABottleneckBlock<B>) -> Self {
        Self::Bottleneck(block)
    }
}

impl<B: Backend> CNAResidualBlockMeta for CNAResidualBlock<B> {
    fn in_planes(&self) -> usize {
        match self {
            Self::Basic(block) => block.in_planes(),
            Self::Bottleneck(block) => block.in_planes(),
        }
    }

    fn out_planes(&self) -> usize {
        match self {
            Self::Basic(block) => block.out_planes(),
            Self::Bottleneck(block) => block.out_planes(),
        }
    }

    fn stride(&self) -> usize {
        match self {
            Self::Basic(block) => block.stride(),
            Self::Bottleneck(block) => block.stride(),
        }
    }
}

impl<B: Backend> CNAResidualBlock<B> {
    /// Apply the wrapped block to the input.
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
        match self {
            Self::Basic(block) => block.forward(input),
            Self::Bottleneck(block) => block.forward(input),
        }
    }

    /// Set the drop path probability.
    pub fn with_drop_path_prob(
        self,
        drop_path_prob: f64,
    ) -> Self {
        let drop_path_prob = expect_probability(drop_path_prob);
        match self {
            Self::Basic(block) => block.with_drop_path_prob(drop_path_prob).into(),
            Self::Bottleneck(block) => block.with_drop_path_prob(drop_path_prob).into(),
        }
    }

    /// Set drop block options.
    pub fn with_drop_block(
        self,
        options: Option<DropBlockOptions>,
    ) -> Self {
        match self {
            Self::Basic(config) => config.with_drop_block(options).into(),
            Self::Bottleneck(config) => config.with_drop_block(options).into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bimm_contracts::assert_shape_contract;
    use burn::backend::NdArray;

    #[test]
    fn test_residual_block_config() {
        let in_planes = 16;
        let planes = 32;

        {
            let inner_cfg = CNABasicBlockConfig::new(in_planes, planes).with_stride(2);
            let cfg: CNAResidualBlockConfig = inner_cfg.clone().into();
            assert!(matches!(cfg, CNAResidualBlockConfig::Basic(_)));
            assert_eq!(cfg.in_planes(), in_planes);
            assert_eq!(cfg.out_planes(), planes);
            assert_eq!(cfg.stride(), 2);
            assert_eq!(cfg.output_resolution([20, 20]), [10, 10]);
        }

        {
            let inner_cfg = CNABottleneckBlockConfig::new(in_planes, planes).with_stride(2);
            let cfg: CNAResidualBlockConfig = inner_cfg.clone().into();
            assert!(matches!(cfg, CNAResidualBlockConfig::Bottleneck(_)));
            assert_eq!(cfg.in_planes(), in_planes);
            assert_eq!(cfg.out_planes(), planes);
            assert_eq!(cfg.stride(), 2);
            assert_eq!(cfg.output_resolution([20, 20]), [10, 10]);
        }
    }

    #[test]
    fn test_residual_block_basic_block() {
        type B = NdArray;
        let device = Default::default();

        let batch_size = 2;
        let in_planes = 16;
        let planes = 32;
        let in_height = 8;
        let in_width = 8;
        let out_height = 4;
        let out_width = 4;

        let cfg: CNAResidualBlockConfig = CNABasicBlockConfig::new(in_planes, planes)
            .with_stride(2)
            .into();

        let block: CNAResidualBlock<B> = cfg.init(&device);
        assert!(matches!(block, CNAResidualBlock::Basic(_)));
        assert_eq!(block.in_planes(), in_planes);
        assert_eq!(block.out_planes(), planes);
        assert_eq!(block.stride(), 2);
        assert_eq!(block.output_resolution([20, 20]), [10, 10]);

        let input = Tensor::ones([batch_size, in_planes, in_height, in_width], &device);
        let output = block.forward(input);

        assert_shape_contract!(
            ["batch", "out_channels", "out_height", "out_width"],
            &output,
            &[
                ("batch", batch_size),
                ("out_planes", planes),
                ("out_height", out_height),
                ("out_width", out_width)
            ],
        );
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn test_residual_block_bottleneck_block() {
        // FIXME: Conv2d with groups is broken in 0.18.0; but fixed in 0.19.0
        use burn::backend::Wgpu;
        type B = Wgpu;
        let device = Default::default();

        let batch_size = 2;
        let in_planes = 16;
        let planes = 32;
        let in_height = 8;
        let in_width = 8;
        let out_height = 4;
        let out_width = 4;

        let cfg: CNAResidualBlockConfig = CNABottleneckBlockConfig::new(in_planes, planes)
            .with_stride(2)
            .into();

        let block: CNAResidualBlock<B> = cfg.init(&device);
        assert!(matches!(block, CNAResidualBlock::Bottleneck(_)));
        assert_eq!(block.in_planes(), in_planes);
        assert_eq!(block.out_planes(), planes);
        assert_eq!(block.stride(), 2);
        assert_eq!(block.output_resolution([20, 20]), [10, 10]);

        let input = Tensor::ones([batch_size, in_planes, in_height, in_width], &device);
        let output = block.forward(input);

        assert_shape_contract!(
            ["batch", "out_channels", "out_height", "out_width"],
            &output,
            &[
                ("batch", batch_size),
                ("out_planes", planes),
                ("out_height", out_height),
                ("out_width", out_width)
            ],
        );
    }
}
