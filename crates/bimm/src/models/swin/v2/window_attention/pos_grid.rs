use burn::prelude::{Backend, Int, Tensor};
use burn::tensor::grid::{IndexPos, meshgrid_stack};

/// Creates a grid of 2D offset indices for a given window shape.
///
/// # Arguments
///
/// * `window_shape`: A (height, width) tuple describing the window shape.
/// * `device`: The device on which the tensor will be created.
///
/// # Returns
///
/// A 3D tensor of shape (2*height-1, 2*width-1, 2) containing the offset indices.
///
/// The first dimension represents the height offsets, the second dimension represents the width offsets,
/// and the third dimension contains the respective indices.
#[inline(always)]
#[must_use]
pub fn window_index_offset_grid<B: Backend>(
    window_shape: [usize; 2],
    device: &B::Device,
) -> Tensor<B, 3, Int> {
    let [h, w] = window_shape;
    assert_ne!(h, 0, "Height must be non-zero");
    assert_ne!(w, 0, "Width must be non-zero");

    let h = h as i64;
    let w = w as i64;

    meshgrid_stack(
        &[
            Tensor::arange_step(-(h - 1)..h, 1, device),
            Tensor::arange_step(-(w - 1)..w, 1, device),
        ],
        IndexPos::Last,
    )
}

/// Creates a grid of 2D relative offsets for a given window shape.
///
/// Converts `window_index_offset_grid` to ``.float()`` and scales by the maximum offset value.
///
/// # Arguments
///
/// * `window_shape`: A (height, width) tuple describing the window shape.
/// * `device`: The device on which the tensor will be created.
///
/// # Returns
///
/// A 3D tensor of shape (2*height-1, 2*width-1, 2) containing the relative offsets.
#[inline(always)]
#[must_use]
pub fn window_relative_offset_grid<B: Backend>(
    window_shape: [usize; 2],
    device: &B::Device,
) -> Tensor<B, 3> {
    let x: Tensor<B, 3> = window_index_offset_grid(window_shape, device).float();

    let d = Tensor::<B, 1, Int>::from_data(window_shape, device) - 1;

    x.div(d.unsqueeze().float())
}

/// Creates a grid of 2D log-scaled relative offsets for a given window shape.
///
/// This is a variant of `window_relative_offset_grid` that applies a logarithmic transformation
/// to the relative offsets.
///
/// # Arguments
///
/// * `window_shape`: A (height, width) tuple describing the window shape.
/// * `base`: The base for the logarithm.
/// * `device`: The device on which the tensor will be created.
///
/// # Returns
///
/// A 3D tensor of shape (2*height-1, 2*width-1, 2) containing the log-scaled relative offsets.
#[inline(always)]
#[must_use]
pub fn window_log1p_relative_offset_grid<B: Backend>(
    window_shape: [usize; 2],
    base: f64,
    device: &B::Device,
) -> Tensor<B, 3> {
    let x = window_relative_offset_grid(window_shape, device);

    let x = x * base;
    let sign = x.clone().sign();

    sign * x.abs().log1p() / base.ln()
}

/// Create a 2D attention bias relative position index for a given window shape.
///
/// # Arguments
///
/// * `window_shape`: A (height, width) tuple describing the window shape.
/// * `device`: The device on which the tensor will be created.
///
/// # Returns
///
/// A 2D tensor of shape (height * width, height * width) containing the relative position indices.
///
/// This has a translation-invariant property, meaning that the relative position indices are
/// the same regardless of the position of the window in the input tensor.
///
/// ```math
/// \forall i, j \in [0, h * w), \text{rel}[i, j] = \text{rel}[i + \Delta_i, j + \Delta_j]
/// ```
///
/// # Example
///
/// ```rust.notest
/// window_attention_relative_position_index([2, 3], device);
/// [
///   [7, 6, 5, 2, 1, 0],
///   [8, 7, 6, 3, 2, 1],
///   [9, 8, 7, 4, 3, 2],
///   [12, 11, 10, 7, 6, 5],
///   [13, 12, 11, 8, 7, 6],
///   [14, 13, 12, 9, 8, 7],
/// ]
/// ```
#[inline(always)]
#[must_use]
pub fn window_attention_relative_position_index<B: Backend>(
    window_shape: [usize; 2],
    device: &B::Device,
) -> Tensor<B, 2, Int> {
    let [h, w] = window_shape;
    let h = h as i64;
    let w = w as i64;
    let hw = h * w;

    // 2, h, w
    let positions: Tensor<B, 3, Int> = meshgrid_stack(
        &[
            Tensor::<B, 1, Int>::arange(0..h, device),
            Tensor::<B, 1, Int>::arange(0..w, device),
        ],
        IndexPos::First,
    );

    // 2, h*w
    let coords: Tensor<B, 2, Int> = positions.flatten(1, 2);

    // 2, h*w, h*w
    let a: Tensor<B, 3, Int> = coords
        .clone()
        .reshape([2, hw as usize, 1])
        .expand([2, hw, hw]);

    // 2, h*w, h*w; rotated
    let b = a.clone().permute([0, 2, 1]);

    // 2, h*w, h*w - relative offset.
    let rel = a - b;

    let rel: Tensor<B, 3, Int> = rel.permute([1, 2, 0]);

    let d = Tensor::<B, 1, Int>::from_data(window_shape, device) - 1;

    let rel: Tensor<B, 3, Int> = rel + d.unsqueeze();

    let s = Tensor::<B, 1, Int>::from_data([2 * window_shape[1] - 1, 1], device);
    let rel = rel.mul(s.unsqueeze());

    let rel: Tensor<B, 2, Int> = rel.sum_dim(2).squeeze_dim::<2>(2);

    rel
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::prelude::TensorData;
    use burn::tensor::Tolerance;

    #[test]
    fn test_window_index_offset_grid() {
        let device = Default::default();

        window_index_offset_grid::<NdArray>([3, 2], &device)
            .to_data()
            .assert_eq(
                &TensorData::from([
                    [[-2, -1], [-2, 0], [-2, 1]],
                    [[-1, -1], [-1, 0], [-1, 1]],
                    [[0, -1], [0, 0], [0, 1]],
                    [[1, -1], [1, 0], [1, 1]],
                    [[2, -1], [2, 0], [2, 1]],
                ]),
                false,
            );
    }

    #[test]
    fn test_window_relative_offset_grid() {
        let device = Default::default();

        window_relative_offset_grid::<NdArray>([3, 2], &device)
            .clone()
            .to_data()
            .assert_eq(
                &TensorData::from([
                    [[-1., -1.], [-1., 0.], [-1., 1.]],
                    [[-0.5, -1.], [-0.5, 0.], [-0.5, 1.]],
                    [[0., -1.], [0., 0.], [0., 1.]],
                    [[0.5, -1.], [0.5, 0.], [0.5, 1.]],
                    [[1., -1.], [1., 0.], [1., 1.]],
                ]),
                false,
            );
    }

    #[test]
    fn test_window_log_offset_grid() {
        let device = Default::default();
        let base = 8.0;

        let actual = window_log1p_relative_offset_grid::<NdArray>([3, 2], base, &device);

        actual.to_data().assert_approx_eq(
            &TensorData::from([
                [
                    [-1.0566417, -1.0566417],
                    [-1.0566417, 0.0],
                    [-1.0566417, 1.0566417],
                ],
                [
                    [-0.773976, -1.0566417],
                    [-0.773976, 0.0],
                    [-0.773976, 1.0566417],
                ],
                [[0.0, -1.0566417], [0.0, 0.0], [0.0, 1.0566417]],
                [
                    [0.773976, -1.0566417],
                    [0.773976, 0.0],
                    [0.773976, 1.0566417],
                ],
                [
                    [1.0566417, -1.0566417],
                    [1.0566417, 0.0],
                    [1.0566417, 1.0566417],
                ],
            ]),
            Tolerance::<f64>::absolute(1e-5),
        );
    }

    #[test]
    fn test_relative_position_index() {
        let window_shape = [2, 3];

        let device = Default::default();
        let rel = window_attention_relative_position_index::<NdArray>(window_shape, &device);
        rel.clone().to_data().assert_eq(
            &TensorData::from([
                [7, 6, 5, 2, 1, 0],
                [8, 7, 6, 3, 2, 1],
                [9, 8, 7, 4, 3, 2],
                [12, 11, 10, 7, 6, 5],
                [13, 12, 11, 8, 7, 6],
                [14, 13, 12, 9, 8, 7],
            ]),
            false,
        );
    }
}
