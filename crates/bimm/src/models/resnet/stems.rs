//! # Input Stems
use crate::layers::activation::{Activation, ActivationConfig};
use crate::layers::blocks::conv_norm::{ConvNorm2d, ConvNorm2dConfig, ConvNorm2dMeta};
use crate::models::resnet::util::CONV_INTO_RELU_INITIALIZER;
use burn::config::Config;
use burn::nn::Initializer;
use burn::prelude::{Backend, Module};
use burn::tensor::Tensor;

/// [`ShallowStem`] Meta API.
pub trait StemMeta {
    /// The number of input channels.
    fn in_planes(&self) -> usize;

    /// The number of output channels.
    fn out_planes(&self) -> usize;

    /// The stride of the first convolution.
    fn stride(&self) -> [usize; 2];
}

/// [`ResNetStem`] configuration.
#[derive(Debug, Clone, Default)]
pub enum StemAbstractConfig {
    /// Default; single 7x7 convolution with stride 2.
    #[default]
    Default,

    /// Three 3x3 convolutions:
    /// 1. ``stem_width, stride=2``
    /// 2. ``stem_width, stride=1``
    /// 3. ``stem_width * 2, stride=1``
    Deep {
        /// The width of the stem convolutions.
        stem_width: usize,
    },

    /// Three 3x3 convolutions:
    DeepTiered {
        /// The width of the stem convolutions.
        /// 1. ``3 * (stem_width//4), stride=2``
        /// 2. ``stem_width, stride=1``
        /// 3. ``stem_width * 2, stride=1``
        stem_width: usize,
    },
}

/// [`StemStage`] configuration.
#[derive(Config, Debug)]
pub struct StemStageConfig {
    /// Convolution + Normalization layer.
    pub conv_norm: ConvNorm2dConfig,

    /// Activation function.
    #[config(default = "ActivationConfig::Relu")]
    pub activation: ActivationConfig,

    /// Initializer for the convolutional layers.
    #[config(default = "CONV_INTO_RELU_INITIALIZER.clone()")]
    pub intializer: Initializer,
}

impl StemMeta for StemStageConfig {
    fn in_planes(&self) -> usize {
        self.conv_norm.in_channels()
    }

    fn out_planes(&self) -> usize {
        self.conv_norm.out_channels()
    }

    fn stride(&self) -> [usize; 2] {
        *self.conv_norm.stride()
    }
}

impl StemStageConfig {
    /// Initialize the [`StemStage`].
    pub fn init<B: Backend>(
        self,
        device: &B::Device,
    ) -> StemStage<B> {
        StemStage {
            conv_norm: self
                .conv_norm
                .with_initializer(self.intializer)
                .init(device),

            activation: self.activation.init(device),
        }
    }
}

/// `ResNet` [`StemStage`].
#[derive(Module, Debug)]
pub struct StemStage<B: Backend> {
    /// Convolution + Normalization layer.
    pub conv_norm: ConvNorm2d<B>,

    /// Activation function.
    pub activation: Activation<B>,
}

impl<B: Backend> StemStage<B> {
    /// Forward pass.
    pub fn forward(
        &self,
        input: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        let x = self.conv_norm.forward(input);
        self.activation.forward(x)
    }
}

/// [`Stem]` configuration.
#[derive(Config, Debug)]
pub struct StemConfig {
    /// Stem stages.
    pub stages: Vec<StemStageConfig>,
}

impl StemConfig {
    /// Initialize a [`Stem`].
    pub fn init<B: Backend>(
        self,
        device: &B::Device,
    ) -> Stem<B> {
        // TODO: check that the stages have valid input/output sizes.
        Stem {
            stages: self
                .stages
                .into_iter()
                .map(|stage| stage.init(device))
                .collect(),
        }
    }
}

/// `ResNet` Input [`Stem`] Module.
#[derive(Module, Debug)]
pub struct Stem<B: Backend> {
    stages: Vec<StemStage<B>>,
}

impl<B: Backend> Stem<B> {
    /// Forward pass.
    pub fn forward(
        &self,
        input: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        self.stages.iter().fold(input, |x, stage| stage.forward(x))
    }
}
