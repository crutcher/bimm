//! # Support for Convolution Kernels

use crate::utility::zspace::expect_point_bounds_check;
use burn::prelude::{Backend, Tensor};
use burn::tensor::Numeric;

/// Build a filter of kernel midpoints.
///
/// Filter is `1.0` at the mid-points of kernels; `0.0` everywhere else.
///
/// This predicts the kernel midpoints that ``conv2d`` (and related kernel functions)
/// would place a kernel.
///
/// The *midpoint* of a kernel is computed as ``size / 2``:
/// * the midpoint of odd kernels is the middle: `mid(3) == 1`
/// * the midpoint of even kernels is the first point in the second half: `mid(4) == 2`
///
/// # Argument
///
/// - `shape`: the mask shape.
/// - `kernel`: the shape of the kernel.
/// - `device`: the device to construct the mask on.
#[inline]
pub fn conv2d_kernel_midpoint_filter<B: Backend, K>(
    shape: [usize; 2],
    kernel: [usize; 2],
    device: &B::Device,
) -> Tensor<B, 2, K>
where
    K: Numeric<B>,
{
    // TODO: kernel <= shape
    // This is wrong:
    // expect_point_bounds_check(&kernel, &[0; 2], &shape);

    let region = [
        (kernel[0] / 2)..shape[0] - ((kernel[0] - 1) / 2),
        (kernel[1] / 2)..shape[1] - ((kernel[1] - 1) / 2),
    ];
    Tensor::zeros(shape, device).slice_fill(region, 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::prelude::{Device, TensorData};

    #[test]
    fn test_conv2d_kernel_midpoint_filter() {
        type B = NdArray;
        let device = Default::default();

        let shape = [7, 9];
        let kernel_shape = [2, 3];

        let mask: Tensor<B, 2> = conv2d_kernel_midpoint_filter(shape, kernel_shape, &device);
        mask.to_data().assert_eq(
            &TensorData::from([
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
            ]),
            false,
        );
    }
}
