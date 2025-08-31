#![warn(missing_docs)]
//!# bimm - Burn Image Models

/// Test-only macro import.
#[cfg(test)]
#[allow(unused_imports)]
#[macro_use]
extern crate hamcrest;
extern crate core;

#[allow(dead_code)]
pub(crate) mod compat;

#[cfg(test)]
#[allow(dead_code)]
pub(crate) mod testing;

pub mod layers;

pub mod models;
pub mod utility;
