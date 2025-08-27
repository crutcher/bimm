//! # `ResNet` Utilities
use bimm_contracts::unpack_shape_contract;
use burn::nn::Initializer;

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
#[inline(always)]
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
