//! # `ResNet` Layer Block
//!
//! Collects a series of [`ResidualBlock`]s into a single module.

use crate::models::resnet::residual_block::{
    ResidualBlock, ResidualBlockConfig, ResidualBlockMeta,
};
use crate::models::resnet::util::stride_div_output_resolution;
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
                ResidualBlockConfig::new(
                    in_planes,
                    out_planes,
                    if b == 0 { stride } else { 1 },
                    bottleneck,
                )
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
        if self.blocks.is_empty() {
            return Err("blocks is empty".to_string());
        }

        for idx in 1..self.blocks.len() {
            let prev = &self.blocks[idx - 1];
            let curr = &self.blocks[idx];
            if prev.out_planes() != curr.in_planes() {
                return Err(format!(
                    "block[{}].out_planes({}) != block[{}].in_planes({})",
                    idx - 1,
                    prev.out_planes(),
                    idx,
                    curr.in_planes(),
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::resnet::basic_block::BasicBlockConfig;
    use bimm_contracts::assert_shape_contract;
    use burn::backend::NdArray;

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
