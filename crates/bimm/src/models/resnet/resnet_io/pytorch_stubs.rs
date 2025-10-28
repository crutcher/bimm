#![allow(missing_docs, dead_code)]
//! # ResNet-18 Stubs.
//!
//! These are stub modules to smooth over loading issues with the current version
//! of `burn-import`. There is insufficient information in loaded weights to
//! derive information about stateless modules (such as ``Activation::Relu``).
use crate::layers::blocks::cna::CNA2d;
use crate::layers::blocks::conv_norm::ConvNorm2d;
use crate::models::resnet::basic_block::BasicBlock;
use crate::models::resnet::bottleneck::BottleneckBlock;
use crate::models::resnet::downsample::ResNetDownsample;
use crate::models::resnet::layer_block::LayerBlock;
use crate::models::resnet::residual_block::ResidualBlock;
use crate::models::resnet::resnet_model::ResNet;
use burn::module::Module;
use burn::nn::conv::{Conv2d, Conv2dRecord};
use burn::nn::norm::Normalization;
use burn::nn::{BatchNorm, BatchNormRecord, Linear};
use burn::prelude::Backend;
use burn::record::{FullPrecisionSettings, Recorder};
use burn_import::pytorch::PyTorchFileRecorder;
use std::path::PathBuf;

/// Load a [`ResNetStubRecord`] from ``torch`` weights path.
pub fn load_resnet_stub_record<B: Backend>(
    path: PathBuf,
    device: &B::Device,
) -> anyhow::Result<ResNetStubRecord<B>> {
    let load_args = burn_import::pytorch::LoadArgs::new(path)
        .with_key_remap(r"downsample\.0", "downsample.conv")
        .with_key_remap(r"downsample\.1", "downsample.bn")
        .with_key_remap("layer([1-4])", "layers.$1.blocks");

    let record = PyTorchFileRecorder::<FullPrecisionSettings>::new().load(load_args, device)?;

    Ok(record)
}

#[derive(Module, Debug)]
pub struct ResNetStub<B: Backend> {
    pub conv1: Conv2d<B>,
    pub bn1: BatchNorm<B>,
    pub layers: Vec<LayerBlockStub<B>>,
    pub fc: Linear<B>,
}

impl<B: Backend> ResNetStubRecord<B> {
    pub fn cna_copy_weights(
        self,
        target: ResNet<B>,
    ) -> ResNet<B> {
        ResNet {
            input_conv_norm: copy_conv_norm_weights(self.conv1, self.bn1, target.input_conv_norm),
            layers: self
                .layers
                .into_iter()
                .zip(target.layers)
                .map(|(s, t)| s.cna_copy_weights(t))
                .collect(),
            output_fc: target.output_fc.load_record(self.fc),
            ..target
        }
    }
}

#[derive(Module, Debug)]
pub struct LayerBlockStub<B: Backend> {
    pub blocks: Vec<ResidualBlockStub<B>>,
}

impl<B: Backend> LayerBlockStubRecord<B> {
    pub fn cna_copy_weights(
        self,
        target: LayerBlock<B>,
    ) -> LayerBlock<B> {
        LayerBlock {
            blocks: self
                .blocks
                .into_iter()
                .zip(target.blocks)
                .map(|(s, t)| s.cna_copy_weights(t))
                .collect(),
        }
    }
}

#[derive(Module, Debug)]
#[allow(clippy::large_enum_variant)]
pub enum ResidualBlockStub<B: Backend> {
    Bottleneck(BottleneckStub<B>),
    Basic(BasicBlockStub<B>),
}

impl<B: Backend> ResidualBlockStubRecord<B> {
    pub fn cna_copy_weights(
        self,
        target: ResidualBlock<B>,
    ) -> ResidualBlock<B> {
        use ResidualBlock as T;
        use ResidualBlockStubRecord as S;
        match (self, target) {
            (S::Basic(stub), T::Basic(block)) => stub.cna_copy_weights(block).into(),
            (S::Bottleneck(stub), T::Bottleneck(block)) => stub.cna_copy_weights(block).into(),
            (S::Basic(_), T::Bottleneck(_)) => {
                panic!("Cannot apply basic block stub to bottleneck block")
            }
            (S::Bottleneck(_), T::Basic(_)) => {
                panic!("Cannot apply bottleneck block stub to basic block")
            }
        }
    }
}

pub fn copy_downsample_weights<B: Backend>(
    downsample: Option<DownsampleStubRecord<B>>,
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

impl<B: Backend> DownsampleStubRecord<B> {
    pub fn copy_weights(
        self,
        target: ResNetDownsample<B>,
    ) -> ResNetDownsample<B> {
        match target.norm {
            Normalization::Batch(norm) => ResNetDownsample {
                conv: target.conv.load_record(self.conv),
                norm: norm.load_record(self.bn).into(),
            },
            _ => panic!("Stub cannot be applied to {:?}", target.norm),
        }
    }
}

pub fn copy_cna_weights<B: Backend>(
    conv: Conv2dRecord<B>,
    bn: BatchNormRecord<B>,
    target: CNA2d<B>,
) -> CNA2d<B> {
    match target.norm {
        Normalization::Batch(norm) => CNA2d {
            conv: target.conv.load_record(conv),
            norm: norm.load_record(bn).into(),
            ..target
        },
        _ => panic!("Stub cannot be applied to {:?}", target.norm),
    }
}

pub fn copy_conv_norm_weights<B: Backend>(
    conv: Conv2dRecord<B>,
    bn: BatchNormRecord<B>,
    target: ConvNorm2d<B>,
) -> ConvNorm2d<B> {
    ConvNorm2d {
        conv: target.conv.load_record(conv),
        norm: target.norm.load_record(bn),
    }
}

#[derive(Module, Debug)]
pub struct BasicBlockStub<B: Backend> {
    pub conv1: Conv2d<B>,
    pub bn1: BatchNorm<B>,
    pub conv2: Conv2d<B>,
    pub bn2: BatchNorm<B>,
    pub downsample: Option<DownsampleStub<B>>,
}

impl<B: Backend> BasicBlockStubRecord<B> {
    pub fn cna_copy_weights(
        self,
        target: BasicBlock<B>,
    ) -> BasicBlock<B> {
        BasicBlock {
            downsample: copy_downsample_weights(self.downsample, target.downsample),
            cna1: copy_cna_weights(self.conv1, self.bn1, target.cna1),
            cna2: copy_cna_weights(self.conv2, self.bn2, target.cna2),
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

impl<B: Backend> BottleneckStubRecord<B> {
    pub fn cna_copy_weights(
        self,
        target: BottleneckBlock<B>,
    ) -> BottleneckBlock<B> {
        BottleneckBlock {
            cna1: copy_cna_weights(self.conv1, self.bn1, target.cna1),
            cna2: copy_cna_weights(self.conv2, self.bn2, target.cna2),
            cna3: copy_cna_weights(self.conv3, self.bn3, target.cna3),
            downsample: copy_downsample_weights(self.downsample, target.downsample),
            ..target
        }
    }
}
