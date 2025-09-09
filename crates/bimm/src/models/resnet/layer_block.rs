//! # `ResNet` Layer Block
//!
//! A [`LayerBlock`] is a sequence of [`ResidualBlock`]s.
//!
//! [`LayerBlockMeta`] defines a common meta API for [`LayerBlock`]
//! and [`LayerBlockConfig`].
//!
//! [`LayerBlockConfig`] implements [`Config`], and provides
//! [`LayerBlockConfig::init`] to initialize a [`LayerBlock`].
//!
//! [`LayerBlock`] implements [`Module`], and provides
//! [`LayerBlock::forward`].

use crate::layers::drop::drop_block::DropBlockOptions;
use crate::models::resnet::residual_block::{
    ResidualBlock, ResidualBlockConfig, ResidualBlockMeta,
};
use crate::models::resnet::util::stride_div_output_resolution;
use crate::utility::probability::expect_probability;
use bimm_contracts::{assert_shape_contract_periodically, unpack_shape_contract};
use burn::config::Config;
use burn::prelude::{Backend, Module, Tensor};

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
#[derive(Config, Debug)]
pub struct LayerBlockConfig {
    /// The component blocks.
    pub blocks: Vec<ResidualBlockConfig>,
}

impl From<Vec<ResidualBlockConfig>> for LayerBlockConfig {
    fn from(blocks: Vec<ResidualBlockConfig>) -> Self {
        Self { blocks }
    }
}

impl LayerBlockMeta for LayerBlockConfig {
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

impl LayerBlockConfig {
    /// Build a config.
    pub fn build(
        num_blocks: usize,
        in_planes: usize,
        out_planes: usize,
        stride: usize,
        bottleneck: bool,
    ) -> Self {
        let blocks = (0..num_blocks)
            .map(|b| {
                if b == 0 {
                    ResidualBlockConfig::build(in_planes, out_planes, stride, bottleneck)
                } else {
                    ResidualBlockConfig::build(out_planes, out_planes, 1, bottleneck)
                }
            })
            .collect();

        Self { blocks }
    }

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
        F: FnMut(usize, ResidualBlockConfig) -> ResidualBlockConfig,
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
        self.map_blocks(&mut |_, block| block.with_drop_block(options.clone()))
    }

    /// Update the drop path probability.
    pub fn with_drop_path_prob(
        self,
        prob: f64,
    ) -> Self {
        let prob = expect_probability(prob);
        self.map_blocks(&mut |_, block| block.with_drop_path_prob(prob))
    }
}

/// Layer block.
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
            &input,
            &["batch", "out_height", "out_width"],
            &[("in_planes", self.in_planes()), ("stride", self.stride())],
        );

        let x = self.blocks.iter().fold(input, |x, block| block.forward(x));

        assert_shape_contract_periodically!(
            ["batch", "out_planes", "out_height", "out_width"],
            &x,
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

    /// Extend the layer block.
    ///
    /// Duplicates the config of the last layer `size` times, interleaved into the layers
    /// after the first.
    ///
    /// # Arguments
    ///
    /// - `size`: additional layers to add.
    pub fn extend(
        self,
        size: usize,
    ) -> Self {
        let device = &self.devices()[0];
        let mut blocks = self.blocks;
        let source_cfg = blocks.last().unwrap().to_config();
        let mut idx = 1;
        for _ in 0..size {
            blocks.insert(idx, source_cfg.clone().init(device));
            idx += 2;
            if idx >= blocks.len() {
                idx = 1;
            }
        }
        Self { blocks }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::resnet::basic_block::BasicBlockConfig;
    use bimm_contracts::assert_shape_contract;
    use burn::backend::NdArray;

    #[test]
    fn test_layer_block_config_build() {
        let config = LayerBlockConfig::build(2, 16, 32, 2, false);
        config.expect_valid();
        assert_eq!(config.len(), 2);
        assert_eq!(config.in_planes(), 16);
        assert_eq!(config.out_planes(), 32);
        assert_eq!(config.stride(), 2);
        assert_eq!(config.output_resolution([12, 24]), [6, 12]);

        let block1 = &config.blocks[0];
        assert_eq!(block1.in_planes(), 16);
        assert_eq!(block1.out_planes(), 32);
        assert_eq!(block1.stride(), 2);
        assert_eq!(block1.output_resolution([12, 24]), [6, 12]);

        let block2 = &config.blocks[1];
        assert_eq!(block2.in_planes(), 32);
        assert_eq!(block2.out_planes(), 32);
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

        let config = LayerBlockConfig::from(vec![
            BasicBlockConfig::new(a_planes, b_planes)
                .with_stride(2)
                .into(),
            BasicBlockConfig::new(b_planes, c_planes)
                .with_stride(3)
                .into(),
        ]);

        config.expect_valid();

        assert_eq!(config.len(), 2);
        assert_eq!(config.in_planes(), a_planes);
        assert_eq!(config.out_planes(), c_planes);
        assert_eq!(config.stride(), 2 * 3);
        assert_eq!(config.output_resolution([12, 24]), [2, 4]);

        let block: LayerBlock<B> = config.init(&device);

        assert_eq!(block.len(), 2);
        assert_eq!(block.in_planes(), a_planes);
        assert_eq!(block.out_planes(), c_planes);
        assert_eq!(block.stride(), 2 * 3);
        assert_eq!(block.output_resolution([12, 24]), [2, 4]);

        let batch_size = 2;
        let input = Tensor::ones([batch_size, a_planes, 12, 24], &device);

        let output = block.forward(input.clone());
        assert_shape_contract!(
            ["batch", "out_planes", "out_height", "out_width"],
            &output,
            &[
                ("batch", batch_size),
                ("out_planes", c_planes),
                ("out_height", 2),
                ("out_width", 4)
            ],
        );

        let mut expected = input;
        for block in block.blocks.iter() {
            expected = block.forward(expected);
        }
        output.to_data().assert_eq(&expected.to_data(), true);
    }
}
