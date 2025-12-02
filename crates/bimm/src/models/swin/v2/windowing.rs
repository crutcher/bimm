//! Windowing operations for Swin Transformer v2

use bimm_contracts::unpack_shape_contract;
use burn::prelude::{Backend, Tensor};
use burn::tensor::BasicOps;

/// Window Partition
///
/// # Arguments
///
/// - `tensor`: Input tensor of ``[batch, height, width, channels]``.
/// - `window_size`: Window size.
///
/// # Returns
///
/// Output tensor of ``[batch * h_windows * w_windows, window_size, window_size, channels]``.
///
/// # Panics
///
/// On shape contract failure.
#[inline]
#[must_use]
pub fn window_partition<B: Backend, K>(
    tensor: Tensor<B, 4, K>,
    window_size: usize,
) -> Tensor<B, 4, K>
where
    K: BasicOps<B>,
{
    let [b, h_wins, w_wins, c] = unpack_shape_contract!(
        [
            "batch",
            "h_wins" * "window_size",
            "w_wins" * "window_size",
            "channels"
        ],
        &tensor.dims(),
        &["batch", "h_wins", "w_wins", "channels"],
        &[("window_size", window_size)]
    );

    tensor
        .reshape([b, h_wins, window_size, w_wins, window_size, c])
        .swap_dims(2, 3)
        .reshape([b * h_wins * w_wins, window_size, window_size, c])
}

/// Window Reverse
///
/// # Arguments
///
/// - `windows`: Input tensor of ``[batch * h_windows * w_windows, window_size, window_size, channels]``.
/// - `window_size`: Window size.
/// - `height`: Height of the original image.
/// - `width`: Width of the original image.
///
/// # Returns
///
/// Output tensor of shape ``[batch, height, width, channels]``.
///
/// # Panics
///
/// On shape contract failure.
#[inline]
#[must_use]
pub fn window_reverse<B: Backend, K>(
    windows: Tensor<B, 4, K>,
    window_size: usize,
    height: usize,
    width: usize,
) -> Tensor<B, 4, K>
where
    K: BasicOps<B>,
{
    let h_wins = height / window_size;
    let w_wins = width / window_size;

    let [b, c] = unpack_shape_contract!(
        [
            "batch" * "h_wins" * "w_wins",
            "window_size",
            "window_size",
            "channels"
        ],
        &windows.dims(),
        &["batch", "channels"],
        &[
            ("h_wins", h_wins),
            ("w_wins", w_wins),
            ("window_size", window_size),
        ],
    );

    windows
        .reshape([b, h_wins, w_wins, window_size, window_size, c])
        .swap_dims(2, 3)
        .reshape([b, height, width, c])
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::prelude::Tensor;
    use burn::tensor::{Distribution, Tolerance};

    #[test]
    fn test_window_partition() {
        let device = Default::default();

        let b = 3;
        let window_size = 4;
        let channels = 3;

        let h_wins = 2;
        let w_wins = 3;
        let h = h_wins * window_size;
        let w = w_wins * window_size;

        let distribution = Distribution::Uniform(0.0, 1.0);
        let input = Tensor::<NdArray, 4>::random([b, h, w, channels], distribution, &device);

        let windows = window_partition(input.clone(), window_size);

        assert_eq!(
            &windows.dims(),
            &[b * h_wins * w_wins, window_size, window_size, channels]
        );

        let reverse = window_reverse(windows, window_size, h, w);
        assert_eq!(&reverse.dims(), &[b, h, w, channels]);

        reverse
            .to_data()
            .assert_approx_eq(&input.to_data(), Tolerance::<f64>::default());
    }
}
