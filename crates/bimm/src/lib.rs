#![warn(missing_docs)]
//!# bimm - Burn Image Models
//!
//! ## Notable Components
//!
//! * [`cache`] - weight loading cache.
//! * [`compat`] - compat code, ported or planned for an upcoming release of ``burn``.
//!   * [`compat::activation_wrapper::Activation`] - activation layer abstraction wrapper.
//!   * [`compat::normalization_wrapper::Normalization`] - norm layer abstraction wrapper.
//! * [`layers`] - reusable neural network modules.
//!   * [`layers::blocks`] - miscellaneous blocks.
//!     * [`layers::blocks::conv_norm`] - ``Conv2d + BatchNorm2d`` block.
//!   * [`layers::drop`] - dropout layers.
//!     * [`layers::drop::drop_block`] - 2d drop block / spatial dropout.
//!     * [`layers::drop::drop_path`] - drop path / stochastic depth.
//!   * [`layers::patching`] - patching layers.
//!     * [`layers::patching::patch_embed`] - 2d patch embedding layer.
//! * [`models`] - complete model families.
//!   * [`models::resnet`] - `ResNet`
//!   * [`models::swin`] - The SWIN Family.
//!     * [`models::swin::v2`] - The SWIN-V2 Model.

extern crate core;
/// Test-only macro import.
#[cfg(test)]
#[allow(unused_imports)]
#[macro_use]
extern crate hamcrest;

#[allow(dead_code)]
pub mod compat;

#[cfg(test)]
#[allow(dead_code)]
pub(crate) mod testing;

pub mod layers;

pub mod cache;
pub mod models;
pub mod utility;
