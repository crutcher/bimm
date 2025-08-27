//! # `ResNet` Utilities
use bimm_contracts::unpack_shape_contract;

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
