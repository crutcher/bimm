#![warn(missing_docs)]
//!# bimm - Burn Image Models
//!
//! ## Models
//!
//! * [`models::resnet`] - preview of ResNet-18
//! * [`models::swin`] - preview of SWIN-V2

extern crate core;
/// Test-only macro import.
#[cfg(test)]
#[allow(unused_imports)]
#[macro_use]
extern crate hamcrest;

#[allow(dead_code)]
pub(crate) mod compat;

#[cfg(test)]
#[allow(dead_code)]
pub(crate) mod testing;

pub mod layers;

pub mod cache;
pub mod models;
pub mod utility;
