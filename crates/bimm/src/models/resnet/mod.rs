//! # `ResNet`
//!
//! Implementation of the *`ResNet`* family of models for image recognition.
//! See: [arXiv:1512.03385v1 [cs.CV]](<https://arxiv.org/abs/1512.03385>)
//!
//! ## Example
//!
//! Example of building a pretrained ResNet-18 module:
//! ```rust,no_run
//! use bimm::cache::fetch_model_weights;
//! use bimm::models::resnet::{ResNet, ResNetContractConfig};
//! use burn::backend::NdArray;
//!
//! let device = Default::default();
//!
//! let source =
//!     "https://download.pytorch.org/models/resnet18-f37072fd.pth";
//! let source_classes = 1000;
//! let weights_path= fetch_model_weights(source).unwrap();
//!
//! let my_classes = 10;
//!
//! let model: ResNet<NdArray> = ResNetContractConfig::resnet18(source_classes)
//!     .to_structure()
//!     .init(&device)
//!     .load_pytorch_weights(weights_path)
//!     .expect("Model should be loaded successfully")
//!     .with_classes(my_classes)
//!     // Enable (drop_block_prob) stochastic block drops for training:
//!     .with_stochastic_drop_block(0.2)
//!     // Enable (drop_path_prob) stochastic depth for training:
//!     .with_stochastic_path_depth(0.1);
//! ```
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
pub mod residual_block;
pub mod resnet_io;
pub mod resnet_model;
pub mod stems;
pub mod util;

pub use resnet_model::*;
