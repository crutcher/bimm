//! # Residual Block Wrapper

use crate::layers::drop::drop_block::DropBlockOptions;
use crate::models::resnet::basic_block::{BasicBlock, BasicBlockConfig, BasicBlockMeta};
use crate::models::resnet::bottleneck::{
    BottleneckBlock, BottleneckBlockConfig, BottleneckBlockMeta,
};
use crate::models::resnet::util::stride_div_output_resolution;
use crate::utility::probability::expect_probability;
use burn::config::Config;
use burn::prelude::{Backend, Module, Tensor};

/// [`ResidualBlock`] Meta API.
pub trait ResidualBlockMeta {
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

/// [`ResidualBlock`] Config.
#[derive(Config, Debug)]
pub enum ResidualBlockConfig {
    /// A `ResNet` [`BasicBlock`].
    Basic(BasicBlockConfig),

    /// A `ResNet` [`BottleneckBlock`].
    Bottleneck(BottleneckBlockConfig),
}

impl ResidualBlockMeta for ResidualBlockConfig {
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

impl From<BasicBlockConfig> for ResidualBlockConfig {
    fn from(config: BasicBlockConfig) -> Self {
        Self::Basic(config)
    }
}

impl From<BottleneckBlockConfig> for ResidualBlockConfig {
    fn from(config: BottleneckBlockConfig) -> Self {
        Self::Bottleneck(config)
    }
}

impl ResidualBlockConfig {
    /// Initialize a [`ResidualBlock`].
    pub fn init<B: Backend>(
        &self,
        device: &B::Device,
    ) -> ResidualBlock<B> {
        match self {
            Self::Basic(config) => ResidualBlock::Basic(config.clone().init(device)),
            Self::Bottleneck(config) => ResidualBlock::Bottleneck(config.clone().init(device)),
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
            BottleneckBlockConfig::new(in_planes, out_planes)
                .with_stride(stride)
                .into()
        } else {
            BasicBlockConfig::new(in_planes, out_planes)
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

/// A `ResNet` [`BasicBlock`] or [`BottleneckBlock`] wrapper.
#[derive(Module, Debug)]
#[allow(clippy::large_enum_variant)]
pub enum ResidualBlock<B: Backend> {
    /// A `ResNet` [`BasicBlock`].
    Basic(BasicBlock<B>),

    /// A `ResNet` [`BottleneckBlock`].
    Bottleneck(BottleneckBlock<B>),
}

impl<B: Backend> From<BasicBlock<B>> for ResidualBlock<B> {
    fn from(block: BasicBlock<B>) -> Self {
        Self::Basic(block)
    }
}

impl<B: Backend> From<BottleneckBlock<B>> for ResidualBlock<B> {
    fn from(block: BottleneckBlock<B>) -> Self {
        Self::Bottleneck(block)
    }
}

impl<B: Backend> ResidualBlockMeta for ResidualBlock<B> {
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

impl<B: Backend> ResidualBlock<B> {
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

    /// Create a config from this module.
    pub fn to_config(&self) -> ResidualBlockConfig {
        match self {
            Self::Basic(block) => block.to_config().into(),
            Self::Bottleneck(block) => block.to_config().into(),
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
            let inner_cfg = BasicBlockConfig::new(in_planes, planes).with_stride(2);
            let cfg: ResidualBlockConfig = inner_cfg.clone().into();
            assert!(matches!(cfg, ResidualBlockConfig::Basic(_)));
            assert_eq!(cfg.in_planes(), in_planes);
            assert_eq!(cfg.out_planes(), planes);
            assert_eq!(cfg.stride(), 2);
            assert_eq!(cfg.output_resolution([20, 20]), [10, 10]);
        }

        {
            let inner_cfg = BottleneckBlockConfig::new(in_planes, planes).with_stride(2);
            let cfg: ResidualBlockConfig = inner_cfg.clone().into();
            assert!(matches!(cfg, ResidualBlockConfig::Bottleneck(_)));
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

        let cfg: ResidualBlockConfig = BasicBlockConfig::new(in_planes, planes)
            .with_stride(2)
            .into();

        let block: ResidualBlock<B> = cfg.init(&device);
        assert!(matches!(block, ResidualBlock::Basic(_)));
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

        let cfg: ResidualBlockConfig = BottleneckBlockConfig::new(in_planes, planes)
            .with_stride(2)
            .into();

        let block: ResidualBlock<B> = cfg.init(&device);
        assert!(matches!(block, ResidualBlock::Bottleneck(_)));
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
