//! [`Tensor`] ops.

use alloc::vec;
use alloc::vec::Vec;
use burn::prelude::{Backend, Tensor};
use std::f64;

/// Create a vector with evenly spaced floating point values.
///
/// This function generates a vector starting from `start`, ending at `end`, and incrementing by `step`.
///
/// # Parameters
///
/// - `start`: The starting value of the range.
/// - `end`: The end value of the range (exclusive).
/// - `step`: An optional step value. If not provided, defaults to `1.0` if `start < end`, or `-1.0` if `start > end`.
///
/// # Returns
///
/// A vector containing the generated floating point values.
#[must_use]
pub fn float_vec_arange(
    start: f64,
    end: f64,
    step: Option<f64>,
) -> Vec<f64> {
    assert_ne!(start, end);
    let step = if start < end {
        let step = step.unwrap_or(1.0);
        if step <= 0.0 {
            panic!("Step must be positive when start < end");
        }
        step
    } else {
        let step = step.unwrap_or(-1.0);
        if step >= 0.0 {
            panic!("Step must be negative when start > end");
        }
        step
    };

    let mut values: Vec<f64> = Vec::new();
    loop {
        let acc = start + values.len() as f64 * step;
        if (step > 0.0 && acc > end) || (step < 0.0 && acc < end) {
            break;
        }
        values.push(acc);
    }

    values
}

/// Create a vector with evenly spaced floating point values.
///
/// This function generates a vector with `num` values starting from `start`, ending at `end`, and evenly spaced.
///
/// # Parameters
///
/// - `start`: The starting value of the range.
/// - `end`: The end value of the range (inclusive).
/// - `num`: The number of points to generate in the range.
///
/// # Returns
///
/// A vector containing the generated floating point values.
#[must_use]
pub fn float_vec_linspace(
    start: f64,
    end: f64,
    num: usize,
) -> Vec<f64> {
    assert!(num > 0, "Number of points must be positive");

    if num == 1 {
        return vec![start];
    }

    let step = (end - start) / (num as f64 - 1.0);

    let end = if step > 0.0 {
        end + f64::EPSILON // Avoid floating point precision issues
    } else {
        end - f64::EPSILON // Avoid floating point precision issues
    };

    float_vec_arange(start, end, Some(step))
}
/// Create a 1D tensor with evenly spaced floating point values.
///
/// This function generates a tensor with values starting from `start`, ending at `end`, and incrementing by `step`.
/// If `step` is not provided, it defaults to `1.0` if `start < end`, or `-1.0` if `start > end`.
///
/// # Parameters
///
/// - `start`: The starting value of the range.
/// - `end`: The end value of the range (exclusive).
/// - `step`: An optional step value. If not provided, defaults to `1.0` or `-1.0` based on the order of `start` and `end`.
///
/// # Returns
///
/// A 1D tensor containing the generated floating point values.
#[must_use]
pub fn float_arange<B: Backend>(
    start: f64,
    end: f64,
    step: Option<f64>,
    device: &B::Device,
) -> Tensor<B, 1> {
    let values = float_vec_arange(start, end, step);
    Tensor::from_data(values.as_slice(), device)
}

/// Create a 1D tensor with evenly spaced floating point values.
///
/// This function generates a tensor with `num` values starting from `start`, ending at `end`, and evenly spaced.
///
/// # Parameters
///
/// - `start`: The starting value of the range.
/// - `end`: The end value of the range (inclusive).
/// - `num`: The number of points to generate in the range.
///
/// # Returns
///
/// A 1D tensor containing the generated floating point values.
#[must_use]
pub fn float_linspace<B: Backend>(
    start: f64,
    end: f64,
    num: usize,
    device: &B::Device,
) -> Tensor<B, 1> {
    let values = float_vec_linspace(start, end, num);
    Tensor::from_data(values.as_slice(), device)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::prelude::TensorData;

    #[test]
    fn test_float_arange() {
        let device = Default::default();
        let start: f64 = 3.0;
        let end: f64 = -1.0 - f64::EPSILON;

        let actual = float_arange::<NdArray>(start, end, None, &device);

        actual
            .to_data()
            .assert_eq(&TensorData::from([3.0, 2.0, 1.0, 0.0, -1.0]), false);
    }

    #[should_panic(expected = "Step must be negative when start > end")]
    #[test]
    fn test_float_arange_panic_step_negative() {
        let device = Default::default();
        // This should panic because the step is not negative
        let _ = float_arange::<NdArray>(3.0, -1.0, Some(1.0), &device);
    }

    #[should_panic(expected = "Step must be positive when start < end")]
    #[test]
    fn test_float_arange_panic_step_positive() {
        let device = Default::default();
        // This should panic because the step is not positive
        let _ = float_arange::<NdArray>(-1.0, 3.0, Some(-1.0), &device);
    }

    #[test]
    fn test_float_vec_linspace_int_step() {
        let device = Default::default();
        let start: f64 = 0.0;
        let end: f64 = 1.0;
        let num: usize = 5;

        let actual = float_linspace::<NdArray>(start, end, num, &device);

        actual
            .to_data()
            .assert_eq(&TensorData::from([0.0, 0.25, 0.5, 0.75, 1.0]), false);
    }

    #[test]
    fn test_float_vec_linspace_neg_float_step() {
        let device = Default::default();
        let start: f64 = 1.0;
        let end: f64 = -0.2;
        let num: usize = 5;

        let actual = float_linspace::<NdArray>(start, end, num, &device);

        actual
            .to_data()
            .assert_eq(&TensorData::from([1.0, 0.7, 0.4, 0.1, -0.2]), false);
    }

    #[test]
    fn test_float_vec_linspace_n1() {
        let device = Default::default();
        let start: f64 = 0.0;
        let end: f64 = 1.0;
        let num: usize = 1;

        let actual = float_linspace::<NdArray>(start, end, num, &device);
        // println!("{actual:?}");

        actual.to_data().assert_eq(&TensorData::from([0.0]), false);
    }
}
