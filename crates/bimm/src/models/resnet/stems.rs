//! # Input Stems
//!
//! This is incompletely implemented.
//!
//! The target surface is this pile from ``class ResNet`` in ``timm``:
//! ```python,ignore
//! # Stem
//! deep_stem = 'deep' in stem_type
//! inplanes = stem_width * 2 if deep_stem else 64
//! if deep_stem:
//!     stem_chs = (stem_width, stem_width)
//!     if 'tiered' in stem_type:
//!         stem_chs = (3 * (stem_width // 4), stem_width)
//!     self.conv1 = nn.Sequential(*[
//!         nn.Conv2d(in_chans, stem_chs[0], 3, stride=2, padding=1, bias=False),
//!         norm_layer(stem_chs[0]),
//!         act_layer(inplace=True),
//!         nn.Conv2d(stem_chs[0], stem_chs[1], 3, stride=1, padding=1, bias=False),
//!         norm_layer(stem_chs[1]),
//!         act_layer(inplace=True),
//!         nn.Conv2d(stem_chs[1], inplanes, 3, stride=1, padding=1, bias=False)
//!     ])
//! else:
//!    self.conv1 = nn.Conv2d(in_chans, inplanes, kernel_size=7, stride=2, padding=3, bias=False)
//!
//! self.bn1 = norm_layer(inplanes)
//! self.act1 = act_layer(inplace=True)
//! self.feature_info = [dict(num_chs=inplanes, reduction=2, module='act1')]
//!
//! # Stem pooling. The name 'maxpool' remains for weight compatibility.
//! if replace_stem_pool:
//!     self.maxpool = nn.Sequential(*filter(None, [
//!         nn.Conv2d(inplanes, inplanes, 3, stride=1 if aa_layer else 2, padding=1, bias=False),
//!         create_aa(aa_layer, channels=inplanes, stride=2) if aa_layer is not None else None,
//!         norm_layer(inplanes),
//!         act_layer(inplace=True),
//!     ]))
//! else:
//!     if aa_layer is not None:
//!         if issubclass(aa_layer, nn.AvgPool2d):
//!             self.maxpool = aa_layer(2)
//!     else:
//!         self.maxpool = nn.Sequential(*[
//!             nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
//!             aa_layer(channels=inplanes, stride=2)
//!         ])
//!     else:
//!         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
//! ```
//!
//! `ConvNormAct`:
//!   conv
//!   norm
//!   act
//!
//! Stem:
//!   head: vec<(`ConvNormAct`)>
//!   tail: Union[
//!     Conv/[AA]?/Norm/Act |
//!     [Max|AvgPool]? [AA]?
//!   ]
//!
use crate::layers::activation::{Activation, ActivationConfig};
use crate::layers::blocks::conv_norm::{ConvNorm2d, ConvNorm2dConfig, ConvNorm2dMeta};
use crate::models::resnet::util::CONV_INTO_RELU_INITIALIZER;
use burn::config::Config;
use burn::nn::Initializer;
use burn::prelude::{Backend, Module};
use burn::tensor::Tensor;

/// [`Stem`] Meta API.
pub trait StemMeta {
    /// The number of input channels.
    fn in_planes(&self) -> usize;

    /// The number of output channels.
    fn out_planes(&self) -> usize;

    /// The stride of the first convolution.
    fn stride(&self) -> [usize; 2];
}

/// [`Stem`] configuration.
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
