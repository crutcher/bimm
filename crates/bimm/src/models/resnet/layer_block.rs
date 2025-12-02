//! # `ResNet` Layer Block
//!
//! A [`LayerBlock`] is a sequence of [`ResidualBlock`]s.
//!
//! [`LayerBlockMeta`] defines a common introspection API for [`LayerBlock`]
//! and [`LayerBlockStructureConfig`].
//!
//! [`LayerBlockStructureConfig`] implements [`Config`], and provides
//! [`LayerBlockStructureConfig::init`] to initialize a [`LayerBlock`].
//!
//! [`LayerBlock`] implements [`Module`], and provides
//! [`LayerBlock::forward`].

use super::bottleneck_block::BottleneckPolicyConfig;
use super::residual_block::{
    ResidualBlock, ResidualBlockContractConfig, ResidualBlockMeta, ResidualBlockStructureConfig,
};
use crate::layers::drop::drop_block::DropBlockOptions;
use crate::models::resnet::util::stride_div_output_resolution;
use crate::utility::probability::expect_probability;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use bimm_contracts::{assert_shape_contract_periodically, unpack_shape_contract};
use burn::config::Config;
use burn::nn::BatchNormConfig;
use burn::nn::activation::ActivationConfig;
use burn::nn::norm::NormalizationConfig;
use burn::prelude::{Backend, Module, Tensor};

/// Abstract [`LayerBlock`] Config.
#[derive(Config, Debug)]
pub struct LayerBlockContractConfig {
    /// The number of internal blocks.
    pub num_blocks: usize,

    /// The number of input feature planes.
    pub in_planes: usize,

    /// The number of output feature planes.
    pub out_planes: usize,

    /// Dilation rate for conv layers.
    #[config(default = 1)]
    pub dilation: usize,

    /// If set, override the first dilation rate.
    #[config(default = "None")]
    pub first_dilation: Option<usize>,

    /// Downsample the input by 2x?
    #[config(default = "false")]
    pub downsample_input: bool,

    /// Select between [`super::basic_block::BasicBlock`] and [`super::bottleneck_block::BottleneckBlock`].
    #[config(default = "None")]
    pub bottleneck_policy: Option<BottleneckPolicyConfig>,

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

impl LayerBlockContractConfig {
    /// Build the [`ResidualBlockContractConfig`]s for this layer block.
    pub fn to_block_contracts(self) -> Vec<ResidualBlockContractConfig> {
        let mut first_dilation = self.first_dilation;

        let mut blocks = Vec::with_capacity(self.num_blocks);

        for b in 0..self.num_blocks {
            let downsample_input = b == 0 && self.downsample_input;
            let in_planes = if b == 0 {
                self.in_planes
            } else {
                self.out_planes
            };

            blocks.push(
                ResidualBlockContractConfig::new(in_planes, self.out_planes)
                    .with_downsample_input(downsample_input)
                    .with_first_dilation(first_dilation)
                    .with_dilation(self.dilation)
                    .with_bottleneck_policy(self.bottleneck_policy.clone())
                    .with_normalization(self.normalization.clone())
                    .with_activation(self.activation.clone()),
            );

            first_dilation = Some(self.dilation);
        }

        blocks
    }

    /// Convert to [`LayerBlockStructureConfig`].
    pub fn to_structure(self) -> LayerBlockStructureConfig {
        LayerBlockStructureConfig {
            blocks: self
                .to_block_contracts()
                .into_iter()
                .map(|cfg| cfg.into())
                .collect(),
        }
    }
}

impl From<LayerBlockContractConfig> for LayerBlockStructureConfig {
    fn from(config: LayerBlockContractConfig) -> Self {
        config.to_structure()
    }
}

/// [`LayerBlock`] Meta API.
pub trait LayerBlockMeta {
    /// The number of blocks.
    fn len(&self) -> usize;

    /// Check if the layer block is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The number of input feature planes.
    fn in_planes(&self) -> usize;

    /// The number of output feature planes.
    fn out_planes(&self) -> usize;

    /// Get the effective stride of the layers.
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

/// [`LayerBlock`] Configuration.
///
/// Implements [`LayerBlockMeta`].
#[derive(Config, Debug)]
pub struct LayerBlockStructureConfig {
    /// The component blocks.
    pub blocks: Vec<ResidualBlockStructureConfig>,
}

impl From<Vec<ResidualBlockStructureConfig>> for LayerBlockStructureConfig {
    fn from(blocks: Vec<ResidualBlockStructureConfig>) -> Self {
        Self { blocks }
    }
}

impl LayerBlockMeta for LayerBlockStructureConfig {
    fn len(&self) -> usize {
        self.blocks.len()
    }

    fn in_planes(&self) -> usize {
        self.blocks[0].in_planes()
    }

    fn out_planes(&self) -> usize {
        self.blocks[self.blocks.len() - 1].out_planes()
    }

    fn stride(&self) -> usize {
        self.blocks
            .iter()
            .fold(1, |acc, block| acc * block.stride())
    }
}

impl LayerBlockStructureConfig {
    /// Check if the config is valid.
    ///
    /// # Returns
    ///
    /// A `Result<(), String>`
    pub fn try_validate(&self) -> Result<(), String> {
        if self.is_empty() {
            return Err("blocks is empty".to_string());
        }

        for idx in 1..self.blocks.len() {
            let prev = &self.blocks[idx - 1];
            let curr = &self.blocks[idx];
            if prev.out_planes() != curr.in_planes() {
                return Err(format!(
                    "block[{}].out_planes({}) != block[{}].in_planes({})\n{:#?}",
                    idx - 1,
                    prev.out_planes(),
                    idx,
                    curr.in_planes(),
                    self,
                ));
            }
        }
        Ok(())
    }

    /// Panic if `try_validate` returns an error.
    pub fn expect_valid(&self) {
        match self.try_validate() {
            Ok(_) => (),
            Err(err) => panic!("{}", err),
        }
    }

    /// Initialize a new [`LayerBlock`].
    pub fn init<B: Backend>(
        self,
        device: &B::Device,
    ) -> LayerBlock<B> {
        self.expect_valid();

        LayerBlock {
            blocks: self
                .blocks
                .into_iter()
                .map(|block| block.init(device))
                .collect(),
        }
    }

    /// Apply a mapping over the blocks.
    pub fn map_blocks<F>(
        self,
        f: &mut F,
    ) -> Self
    where
        F: FnMut(usize, ResidualBlockStructureConfig) -> ResidualBlockStructureConfig,
    {
        Self {
            blocks: self
                .blocks
                .into_iter()
                .enumerate()
                .map(|(idx, block)| f(idx, block))
                .collect(),
        }
    }

    /// Update the drop block options.
    pub fn with_drop_block<O>(
        self,
        options: O,
    ) -> Self
    where
        O: Into<Option<DropBlockOptions>>,
    {
        let options = options.into();
        self.map_blocks(&mut |idx, block| {
            if idx == 0 {
                block.with_drop_block(None)
            } else {
                block.with_drop_block(options.clone())
            }
        })
    }
}

/// Layer block; stack of [`ResidualBlock`]s.
///
/// Implements [`LayerBlockMeta`].
#[derive(Module, Debug)]
pub struct LayerBlock<B: Backend> {
    /// Internal blocks.
    pub blocks: Vec<ResidualBlock<B>>,
}

impl<B: Backend> LayerBlockMeta for LayerBlock<B> {
    fn len(&self) -> usize {
        self.blocks.len()
    }

    fn in_planes(&self) -> usize {
        self.blocks[0].in_planes()
    }

    fn out_planes(&self) -> usize {
        self.blocks[self.blocks.len() - 1].out_planes()
    }

    fn stride(&self) -> usize {
        self.blocks
            .iter()
            .fold(1, |acc, block| acc * block.stride())
    }
}

impl<B: Backend> LayerBlock<B> {
    /// Debug print.
    pub fn debug_print(&self) {
        println!("## LayerBlock: len={}", self.len());
        for (idx, block) in self.blocks.iter().enumerate() {
            println!("### block[{}]:", idx);
            block.debug_print();
            println!();
        }
    }

    /// Apply the layer block.
    pub fn forward(
        &self,
        input: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        let [batch, out_height, out_width] = unpack_shape_contract!(
            [
                "batch",
                "in_planes",
                "in_height" = "out_height" * "stride",
                "in_width" = "out_width" * "stride"
            ],
            &input.dims(),
            &["batch", "out_height", "out_width"],
            &[("in_planes", self.in_planes()), ("stride", self.stride())],
        );

        let mut x = input;
        for block in self.blocks.iter() {
            x = block.forward(x);
        }

        assert_shape_contract_periodically!(
            ["batch", "out_planes", "out_height", "out_width"],
            &x.dims(),
            &[
                ("batch", batch),
                ("out_planes", self.out_planes()),
                ("out_height", out_height),
                ("out_width", out_width)
            ],
        );

        x
    }

    /// Apply a mapping over the blocks.
    pub fn map_blocks<F>(
        self,
        f: &mut F,
    ) -> Self
    where
        F: FnMut(usize, ResidualBlock<B>) -> ResidualBlock<B>,
    {
        Self {
            blocks: self
                .blocks
                .into_iter()
                .enumerate()
                .map(|(idx, block)| f(idx, block))
                .collect(),
        }
    }

    /// Update the drop path probability.
    pub fn with_drop_path_prob(
        self,
        prob: f64,
    ) -> Self {
        let prob = expect_probability(prob);
        self.map_blocks(&mut |_, block| block.with_drop_path_prob(prob))
    }

    /// Update the drop block options.
    pub fn with_drop_block<O>(
        self,
        options: O,
    ) -> Self
    where
        O: Into<Option<DropBlockOptions>>,
    {
        let options = options.into();
        self.map_blocks(&mut |_, block| block.with_drop_block(options.clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::resnet::basic_block::BasicBlockConfig;
    use alloc::vec;
    use bimm_contracts::assert_shape_contract;
    use burn::backend::NdArray;

    #[test]
    fn test_layer_block_config_build() {
        let num_blocks = 2;
        let in_planes = 16;
        let planes = 32;
        let config: LayerBlockStructureConfig =
            LayerBlockContractConfig::new(num_blocks, in_planes, planes)
                .with_downsample_input(true)
                .into();
        config.expect_valid();
        assert_eq!(config.len(), 2);
        assert_eq!(config.in_planes(), in_planes);
        assert_eq!(config.out_planes(), planes);
        assert_eq!(config.stride(), 2);
        assert_eq!(config.output_resolution([12, 24]), [6, 12]);

        let block1 = &config.blocks[0];
        assert_eq!(block1.in_planes(), in_planes);
        assert_eq!(block1.out_planes(), planes);
        assert_eq!(block1.stride(), 2);
        assert_eq!(block1.output_resolution([12, 24]), [6, 12]);

        let block2 = &config.blocks[1];
        assert_eq!(block2.in_planes(), planes);
        assert_eq!(block2.out_planes(), planes);
        assert_eq!(block2.stride(), 1);
        assert_eq!(block2.output_resolution([12, 24]), [12, 24]);
    }

    #[test]
    pub fn test_layer_block() {
        type B = NdArray;
        let device = Default::default();

        let a_planes = 16;
        let b_planes = 32;
        let c_planes = 64;

        let config = LayerBlockStructureConfig::from(vec![
            BasicBlockConfig::new(a_planes, b_planes)
                .with_stride(2)
                .into(),
            BasicBlockConfig::new(b_planes, c_planes)
                .with_stride(1)
                .into(),
            BasicBlockConfig::new(c_planes, c_planes)
                .with_stride(1)
                .into(),
        ]);

        config.expect_valid();

        assert_eq!(config.in_planes(), a_planes);
        assert_eq!(config.out_planes(), c_planes);
        assert_eq!(config.stride(), 2);
        assert_eq!(config.output_resolution([20, 16]), [10, 8]);

        let block: LayerBlock<B> = config.init(&device);

        assert_eq!(block.in_planes(), a_planes);
        assert_eq!(block.out_planes(), c_planes);
        assert_eq!(block.stride(), 2);
        assert_eq!(block.output_resolution([20, 16]), [10, 8]);

        let batch_size = 2;
        let input = Tensor::ones([batch_size, a_planes, 20, 16], &device);

        let output = block.forward(input.clone());
        assert_shape_contract!(
            ["batch", "out_planes", "out_height", "out_width"],
            &output.dims(),
            &[
                ("batch", batch_size),
                ("out_planes", c_planes),
                ("out_height", 10),
                ("out_width", 8)
            ],
        );

        let mut expected = input;
        for block in block.blocks.iter() {
            expected = block.forward(expected);
        }
        output.to_data().assert_eq(&expected.to_data(), true);
    }
}
