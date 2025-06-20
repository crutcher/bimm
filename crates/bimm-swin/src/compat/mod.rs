//! Compat code to bridge burn updates.
//!
//! This module is crate-private; and exists only to permit this
//! module to link against multiple versions of burn.

/// The grid module.
pub mod grid;

/// Dimension / Indexing utilities.
pub mod indexing;

/// The linear algebra module.
pub mod linalg;

/// Tensor ops.
pub mod ops;
