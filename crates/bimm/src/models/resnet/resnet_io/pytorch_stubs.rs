#![allow(missing_docs, dead_code)]
//! # ResNet-18 Stubs.
//!
//! These are stub modules to smooth over loading issues with the current
//! version of `burn-import`. There is insufficient information in loaded
//! weights to derive information about stateless modules (such as
//! ``Activation::Relu``).

use alloc::vec::Vec;
use std::path::PathBuf;

use anyhow::Context;
use bunsen::blocks::images::conv::{
    cna::CNA2d,
    conv_norm::ConvNorm2d,
};
use burn::{
    module::Module,
    nn::{
        BatchNorm,
        Linear,
        conv::Conv2d,
        norm::Normalization,
    },
    prelude::Backend,
};

use crate::models::resnet::{
    basic_block::BasicBlock,
    bottleneck_block::BottleneckBlock,
    downsample::ResNetDownsample,
    layer_block::LayerBlock,
    residual_block::ResidualBlock,
    resnet_model::ResNet,
};

/// Load a [`ResNetStubRecord`] from ``torch`` weights path.
#[allow(unused)]
pub fn load_resnet_stub<B: Backend>(
    path: PathBuf,
    stub: &mut ResNetStub<B>,
) -> anyhow::Result<()> {
    use burn::store::{
        ModuleSnapshot,
        PytorchStore,
    };

    let mut store = PytorchStore::from_file(path.clone())
        .with_key_remapping(r"downsample\.0", "downsample.conv")
        .with_key_remapping(r"downsample\.1", "downsample.bn")
        .with_key_remapping("layer([1-4])", "layers.$1.blocks");

    stub.load_from(&mut store)
        .with_context(|| format!("Failed to load ResNet stub from {:?}", path))?;

    Ok(())
}

#[derive(Module, Debug)]
pub struct ResNetStub<B: Backend> {
    pub conv1: Conv2d<B>,
    pub bn1: BatchNorm<B>,
    pub layers: Vec<LayerBlockStub<B>>,
    pub fc: Linear<B>,
}

impl<B: Backend> ResNetStub<B> {
    pub fn copy_stub_weights(
        self,
        target: ResNet<B>,
    ) -> ResNet<B> {
        ResNet {
            input_conv_norm: ConvNorm2d {
                conv: self.conv1.clone(),
                norm: self.bn1.clone(),
            },
            layers: self
                .layers
                .into_iter()
                .zip(target.layers)
                .map(|(lbs, t)| lbs.copy_weights(t))
                .collect(),
            output_fc: target.output_fc.clone(),
            ..target
        }
    }
}

#[derive(Module, Debug)]
pub struct LayerBlockStub<B: Backend> {
    pub blocks: Vec<ResidualBlockStub<B>>,
}

impl<B: Backend> LayerBlockStub<B> {
    pub fn copy_weights(
        &self,
        target: LayerBlock<B>,
    ) -> LayerBlock<B> {
        LayerBlock {
            blocks: self
                .blocks
                .iter()
                .zip(target.blocks)
                .map(|(b, tb)| b.copy_weights(tb))
                .collect::<Vec<_>>(),
        }
    }
}

#[derive(Module, Debug)]
#[allow(clippy::large_enum_variant)]
pub enum ResidualBlockStub<B: Backend> {
    Bottleneck(BottleneckStub<B>),
    Basic(BasicBlockStub<B>),
}

impl<B: Backend> ResidualBlockStub<B> {
    pub fn copy_weights(
        &self,
        target: ResidualBlock<B>,
    ) -> ResidualBlock<B> {
        use ResidualBlock as T;
        use ResidualBlockStub as S;
        match (self, target) {
            (S::Basic(stub), T::Basic(block)) => stub.copy_weights(block).into(),
            (S::Bottleneck(stub), T::Bottleneck(block)) => stub.copy_weights(block).into(),
            (S::Basic(_), T::Bottleneck(_)) => {
                panic!("Cannot apply basic block stub to bottleneck block")
            }
            (S::Bottleneck(_), T::Basic(_)) => {
                panic!("Cannot apply bottleneck block stub to basic block")
            }
        }
    }
}

pub fn copy_downsample_mod_weights<B: Backend>(
    downsample: &Option<DownsampleStub<B>>,
    target: Option<ResNetDownsample<B>>,
) -> Option<ResNetDownsample<B>> {
    match (downsample, target) {
        (Some(stub), Some(target)) => Some(stub.copy_weights(target)),
        (None, None) => None,
        (None, Some(_)) => panic!("None stub cannot be applied to Some<Downsample>"),
        (Some(_), None) => panic!("Some<Downsample> stub cannot be applied to None"),
    }
}

#[derive(Module, Debug)]
pub struct DownsampleStub<B: Backend> {
    pub conv: Conv2d<B>,
    pub bn: BatchNorm<B>,
}

impl<B: Backend> DownsampleStub<B> {
    pub fn copy_weights(
        &self,
        mut target: ResNetDownsample<B>,
    ) -> ResNetDownsample<B> {
        match &mut target.norm {
            Normalization::Batch(norm) => {
                norm.gamma = self.bn.gamma.clone();
                norm.beta = self.bn.beta.clone();
            }
            _ => panic!("Stub cannot be applied to {:?}", target.norm),
        }
        target.conv = copy_conv2d_mod_weights(&self.conv, target.conv);
        target
    }
}

pub fn copy_batchnorm_mod_weights<B: Backend>(
    source: &BatchNorm<B>,
    target: BatchNorm<B>,
) -> BatchNorm<B> {
    BatchNorm {
        beta: source.beta.clone(),
        gamma: source.gamma.clone(),
        ..target
    }
}

pub fn copy_conv2d_mod_weights<B: Backend>(
    source: &Conv2d<B>,
    target: Conv2d<B>,
) -> Conv2d<B> {
    let device = source.weight.device();
    let dtype = source.weight.dtype();

    Conv2d {
        weight: source
            .weight
            .clone()
            .map(|w| w.cast(dtype).to_device(&device)),
        bias: source
            .bias
            .as_ref()
            .map(|p| p.clone().map(|w| w.cast(dtype).to_device(&device))),
        ..target
    }
}

pub fn copy_cna_mod_weights<B: Backend>(
    conv: &Conv2d<B>,
    bn: &BatchNorm<B>,
    mut target: CNA2d<B>,
) -> CNA2d<B> {
    match &mut target.norm {
        Normalization::Batch(norm) => {
            norm.gamma = bn.gamma.clone();
            norm.beta = bn.beta.clone();
        }
        _ => panic!("Stub cannot be applied to {:?}", target.norm),
    }
    target.conv = conv.clone();
    target
}

#[derive(Module, Debug)]
pub struct BasicBlockStub<B: Backend> {
    pub conv1: Conv2d<B>,
    pub bn1: BatchNorm<B>,
    pub conv2: Conv2d<B>,
    pub bn2: BatchNorm<B>,
    pub downsample: Option<DownsampleStub<B>>,
}

impl<B: Backend> BasicBlockStub<B> {
    pub fn copy_weights(
        &self,
        target: BasicBlock<B>,
    ) -> BasicBlock<B> {
        BasicBlock {
            cna1: copy_cna_mod_weights(&self.conv1, &self.bn1, target.cna1),
            cna2: copy_cna_mod_weights(&self.conv2, &self.bn2, target.cna2),
            downsample: copy_downsample_mod_weights(&self.downsample, target.downsample),
            ..target
        }
    }
}

#[derive(Module, Debug)]
pub struct BottleneckStub<B: Backend> {
    pub conv1: Conv2d<B>,
    pub bn1: BatchNorm<B>,
    pub conv2: Conv2d<B>,
    pub bn2: BatchNorm<B>,
    pub conv3: Conv2d<B>,
    pub bn3: BatchNorm<B>,
    pub downsample: Option<DownsampleStub<B>>,
}

impl<B: Backend> BottleneckStub<B> {
    pub fn copy_weights(
        &self,
        target: BottleneckBlock<B>,
    ) -> BottleneckBlock<B> {
        BottleneckBlock {
            cna1: copy_cna_mod_weights(&self.conv1, &self.bn1, target.cna1),
            cna2: copy_cna_mod_weights(&self.conv2, &self.bn2, target.cna2),
            cna3: copy_cna_mod_weights(&self.conv3, &self.bn3, target.cna3),
            downsample: copy_downsample_mod_weights(&self.downsample, target.downsample),
            ..target
        }
    }
}
