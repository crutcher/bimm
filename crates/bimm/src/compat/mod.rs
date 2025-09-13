//! Compat code to bridge `burn_support` updates.
//!
//! This module is crate-private; and exists only to permit this
//! module to link against multiple versions of `burn_support`.

pub mod activation_wrapper;
pub mod conv_shape;
pub mod normalization_wrapper;
pub mod ops;
