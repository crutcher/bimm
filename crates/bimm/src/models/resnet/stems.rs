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
//!         else:
//!             self.maxpool = nn.Sequential(*[
//!                 nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
//!                 aa_layer(channels=inplanes, stride=2)
//!             ])
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
//! ```text,ignore
//!   head: vec<(`ConvNormAct`)>
//!   tail: Union[
//!     Conv/[AA]?/Norm/Act |
//!     [Max|AvgPool]? [AA]?
//!   ]
//! ```
//!

use crate::compat::activation_wrapper::ActivationConfig;
use crate::compat::normalization_wrapper::NormalizationConfig;
use crate::layers::blocks::cna::{CNA2d, CNA2dConfig};
use burn::module::Module;
use burn::nn::PaddingConfig2d;
use burn::nn::conv::Conv2dConfig;
use burn::nn::pool::{MaxPool2d, MaxPool2dConfig};
use burn::prelude::{Backend, Tensor};

/// stem contract configuration.
#[derive(Debug, Clone, Default)]
pub enum ResNetStemContractConfig {
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

impl ResNetStemContractConfig {
    /// Convert to a [`ResNetStemStructureConfig`].
    pub fn to_structure(
        &self,
        in_channels: usize,
        normalization: NormalizationConfig,
        activation: ActivationConfig,
    ) -> ResNetStemStructureConfig {
        match self {
            ResNetStemContractConfig::Default => (),
            _ => unimplemented!("{:?}", self),
        }

        let cna1 = CNA2dConfig {
            conv: Conv2dConfig::new([in_channels, 64], [7, 7])
                .with_stride([2, 2])
                .with_padding(PaddingConfig2d::Explicit(3, 3))
                .with_bias(false),
            norm: normalization.clone(),
            act: activation.clone(),
        };

        let pool = Some(
            MaxPool2dConfig::new([3, 3])
                .with_strides([2, 2])
                .with_padding(PaddingConfig2d::Explicit(1, 1)),
        );

        ResNetStemStructureConfig {
            cna1,
            cna2: None,
            cna3: None,
            pool,
        }
    }
}

/// stem contract configuration.
#[derive(Debug, Clone)]
pub struct ResNetStemStructureConfig {
    /// The first convolution.
    pub cna1: CNA2dConfig,

    /// The second convolution.
    pub cna2: Option<CNA2dConfig>,

    /// The third convolution.
    pub cna3: Option<CNA2dConfig>,

    /// The pooling layer.
    pub pool: Option<MaxPool2dConfig>,
}

/// stem impl.
#[derive(Module, Debug)]
pub struct ResNetStem<B: Backend> {
    /// The first convolution.
    pub cna1: CNA2d<B>,
    /// The second convolution.
    pub cna2: Option<CNA2d<B>>,
    /// The third convolution.
    pub cna3: Option<CNA2d<B>>,
    /// The pooling.
    pub pool: Option<MaxPool2d>,
}

impl<B: Backend> ResNetStem<B> {
    /// forward pass.
    pub fn forward(
        &self,
        input: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        let mut x = input;
        x = self.cna1.forward(x);
        if let Some(cna2) = &self.cna2 {
            x = cna2.forward(x);
        }
        if let Some(cna3) = &self.cna3 {
            x = cna3.forward(x);
        }
        if let Some(pool) = &self.pool {
            x = pool.forward(x);
        }
        x
    }
}
