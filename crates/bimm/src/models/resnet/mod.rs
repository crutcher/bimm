//! # `ResNet`
//!
//! Implementation of the *`ResNet`* family of models for image recognition.
//! See: [arXiv:1512.03385v1 [cs.CV]](<https://arxiv.org/abs/1512.03385>)
//!
//! ## Configuration
//!
//! This module uses 2-layer configuration.
//!
//! * [`ResNetContractConfig`]
//! * [`ResNetStructureConfig`]
//!
//! The high-level abstract config describes `ResNet` modules semantically,
//! while the low-level config describes the structural config.
//!
//! As the compatibility matrix grows; we may differentiate the abstract configs
//! for different variants of `ResNet`; but the goal is that for any given
//! family, there should be a simple to configure abstract config.
//!
//! The structural config is focused on the actual structure of the model,
//! and the intended user base is people working on variants of `ResNet`.
//!
//! It should continue to get cheaper to build `ResNet` variants.
//!
//! ## Compatibility
//!
//! `ResNet` has evolved into a large family of models, and this crate aims
//! to evolve towards equity with the ``timm`` library of models.
//!
//! Unfortunately, the equivalence matrix itself represents a large
//! amount of work and is not yet complete.
//!
//! An incomplete list of missing features includes:
//! * all the fancy-stem options
//! * injectable norm layers
//! * injectable activation layers
//! * conv / avg downsample switching
//! * anti-aliasing
//! * block attention

pub mod basic_block;
pub mod bottleneck;
pub mod downsample;
pub mod layer_block;
pub mod pretrained;
pub mod residual_block;
pub mod resnet_io;
pub mod resnet_model;
pub mod stems;
pub mod util;

pub use pretrained::*;
pub use resnet_model::*;
