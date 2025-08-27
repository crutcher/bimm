//! # `ResNet` Utilities
use bimm_contracts::unpack_shape_contract;
use burn::nn::Initializer;

/// Get the output resolution for a given input resolution.
///
/// The input must be a multiple of the stride.
///
/// # Arguments
///
/// - `input_resolution`: ``[height_in=height_out*stride, width_in=width_out*stride]``.
///
/// # Returns
///
/// ``[height_out, width_out]``
///
/// # Panics
///
/// If the input resolution is not a multiple of the stride.
#[inline(always)]
pub fn stride_div_output_resolution(
    input_resolution: [usize; 2],
    stride: usize,
) -> [usize; 2] {
    unpack_shape_contract!(
        [
            "height_in" = "height_out" * "stride",
            "width_in" = "width_out" * "stride"
        ],
        &input_resolution,
        &["height_out", "width_out"],
        &[("stride", stride)]
    )
}

/// Recommended initializer for conv layers feeding into a relu.
pub static CONV_INTO_RELU_INITIALIZER: Initializer = Initializer::KaimingNormal {
    gain: std::f64::consts::SQRT_2,
    fan_out_only: true,
};
