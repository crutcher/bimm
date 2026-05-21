//!# bimm - Burn Image Models
//!
//! ## Notable Components
//!
//! * [`cache`] - weight loading cache.
//! * [`compat`] - compat code, ported or planned for an upcoming release of
//!   ``burn``.
//! * [`models`] - complete model families.
//!   * [`models::resnet`] - `ResNet`
//!   * [`models::swin`] - The SWIN Family.
//!     * [`models::swin::v2`] - The SWIN-V2 Model.
#![warn(missing_docs)]

extern crate alloc;

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

pub mod cache;
pub mod models;
