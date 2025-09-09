//! # `ResNet` Utilities
use bimm_contracts::unpack_shape_contract;
use burn::nn::{Initializer, PaddingConfig2d};

/// Convert a `T` to a `[T; D]`.
pub fn scalar_to_array<const D: usize, T>(v: T) -> [T; D]
where
    T: Copy,
{
    [v; D]
}

/// Get the output resolution for a given input resolution.
///
/// The input must be a multiple of the stride.
///
/// # Arguments
///
/// - `input_resolution`: ``[in_height=out_height*stride, in_width=out_width*stride]``.
///
/// # Returns
///
/// ``[out_height, out_width]``
///
/// # Panics
///
/// If the input resolution is not a multiple of the stride.
pub fn stride_div_output_resolution(
    input_resolution: [usize; 2],
    stride: usize,
) -> [usize; 2] {
    unpack_shape_contract!(
        [
            "in_height" = "out_height" * "stride",
            "in_width" = "out_width" * "stride"
        ],
        &input_resolution,
        &["out_height", "out_width"],
        &[("stride", stride)]
    )
}

/// Recommended initializer for conv layers feeding into a relu.
pub static CONV_INTO_RELU_INITIALIZER: Initializer = Initializer::KaimingNormal {
    gain: std::f64::consts::SQRT_2,
    fan_out_only: true,
};

/// Compute the necessary [`burn::nn::Conv2d`] padding for the given square parameters.
///
/// All parameters are assumed square (the same in height and width).
///
/// # Arguments
///
/// - `kernel`: The size of the kernel; must be odd.
/// - `stride`: The stride of the convolution; must be >= 1.
/// - `dilation`: The dilation of the convolution; must be >= 1.
///
/// # Returns
///
/// Computed padding.
pub fn get_square_conv2d_padding(
    kernel: usize,
    stride: usize,
    dilation: usize,
) -> usize {
    assert_eq!(kernel % 2, 1, "Kernel size must be odd");
    assert!(stride >= 1, "Stride must be >= 1");
    assert!(dilation >= 1, "Dilation must be >= 1");
    ((stride - 1) + dilation * (kernel - 1)) / 2
}

/// Compute the necessary [`burn::nn::Conv2d`] padding for the given square parameters.
///
/// All parameters are assumed square (the same in height and width).
///
/// # Arguments
///
/// - `kernel`: The size of the kernel; must be odd.
/// - `stride`: The stride of the convolution; must be >= 1.
/// - `dilation`: The dilation of the convolution; must be >= 1.
///
/// # Returns
///
/// Computed [`PaddingConfig2d`].
pub fn build_square_conv2d_padding_config(
    kernel: usize,
    stride: usize,
    dilation: usize,
) -> PaddingConfig2d {
    let padding = get_square_conv2d_padding(kernel, stride, dilation);
    PaddingConfig2d::Explicit(padding, padding)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_to_narray() {
        assert_eq!(scalar_to_array::<4, usize>(1), [1, 1, 1, 1]);
    }

    #[test]
    fn test_get_padding() {
        assert_eq!(get_square_conv2d_padding(1, 1, 1), 0);
        assert_eq!(get_square_conv2d_padding(3, 1, 1), 1);
        assert_eq!(get_square_conv2d_padding(5, 1, 1), 2);

        assert_eq!(get_square_conv2d_padding(1, 2, 1), 0);
        assert_eq!(get_square_conv2d_padding(3, 2, 1), 1);
        assert_eq!(get_square_conv2d_padding(5, 2, 1), 2);

        assert_eq!(get_square_conv2d_padding(1, 1, 2), 0);
        assert_eq!(get_square_conv2d_padding(3, 1, 2), 2);
        assert_eq!(get_square_conv2d_padding(5, 1, 2), 4);

        assert_eq!(get_square_conv2d_padding(1, 2, 2), 0);
        assert_eq!(get_square_conv2d_padding(3, 2, 2), 2);
        assert_eq!(get_square_conv2d_padding(5, 2, 2), 4);
    }

    #[test]
    #[should_panic(expected = "Kernel size must be odd")]
    fn test_get_padding_panic() {
        get_square_conv2d_padding(2, 1, 1);
    }

    #[test]
    #[should_panic(expected = "Stride must be >= 1")]
    fn test_get_padding_panic_stride() {
        get_square_conv2d_padding(1, 0, 1);
    }

    #[test]
    #[should_panic(expected = "Dilation must be >= 1")]
    fn test_get_padding_panic_dilation() {
        get_square_conv2d_padding(1, 1, 0);
    }

    #[test]
    fn test_build_square_conv2d_padding_config() {
        assert_eq!(
            build_square_conv2d_padding_config(1, 1, 1),
            PaddingConfig2d::Explicit(0, 0)
        );

        assert_eq!(
            build_square_conv2d_padding_config(3, 2, 2),
            PaddingConfig2d::Explicit(2, 2)
        );
    }
}
