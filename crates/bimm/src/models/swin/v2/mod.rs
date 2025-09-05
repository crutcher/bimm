//! # Implementation of the Swin Transformer V2 model.
//! See: [SWIN-V2](https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer_v2.py)
//!
//! ## Example
//!
//! ```rust,no_run
//! use bimm::models::swin::v2::swin_model::{SwinTransformerV2, SwinTransformerV2Config};
//! use bimm::models::swin::v2::swin_model::LayerConfig;
//! use burn::backend::NdArray;
//!
//! let image_dimensions = [224, 224];
//! let patch_size = 4;
//! let image_channels = 3;
//! let num_classes = 10;
//! let embed_dim = 96;
//! let window_size = 8;
//!
//! let device = Default::default();
//!
//! let swin_model: SwinTransformerV2<NdArray> = SwinTransformerV2Config::new(
//!     image_dimensions,
//!     patch_size,
//!     image_channels,
//!     num_classes,
//!     embed_dim,
//!     vec![LayerConfig::new(8, 6), LayerConfig::new(8, 12)],
//! )
//! .with_window_size(window_size)
//! .with_attn_drop_rate(0.2)
//! .with_drop_rate(0.2)
//! .init(&device);
//! ```

pub mod block_sequence;
pub mod patch_merge;
pub mod swin_block;
pub mod swin_model;
pub mod window_attention;
pub mod windowing;
