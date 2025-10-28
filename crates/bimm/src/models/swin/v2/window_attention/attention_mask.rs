use crate::models::swin::v2::windowing::window_partition;
use burn::prelude::{Backend, Bool, Int, Tensor};

/// Apply an attention mask.
///
/// # Arguments
///
/// - `b_nw`: Batch size times number of windows.
/// - `n`: Number of elements in the input tensor, Wh*Ww.
/// - `num_heads`: Number of attention heads.
/// - `attn`: Attention logits tensor of shape (`b_nw`, `num_heads`, Wh*Ww, Wh*Ww).
/// - `mask`: Mask tensor of shape (`num_windows`, Wh*Ww, Wh*Ww).
///
/// # Returns
///
/// - Output tensor of shape (`b_nw`, `num_heads`, Wh*Ww, Wh*Ww).
#[inline(always)]
#[must_use]
pub fn apply_attention_mask<B: Backend>(
    b_nw: usize,
    n: usize,
    num_heads: usize,
    attn: Tensor<B, 4>,
    mask: Tensor<B, 3>,
) -> Tensor<B, 4> {
    // Attention mask
    let num_windows = mask.dims()[0];
    let b = b_nw / num_windows;

    let attn = attn.reshape([b, num_windows, num_heads, n, n]);

    let mask = mask.unsqueeze_dim::<4>(1).unsqueeze::<5>();
    // 1, num_windows, 1, Wh*Ww, Wh*Ww

    let attn: Tensor<B, 5> = attn + mask;
    // b, num_windows, num_heads, Wh*Ww, Wh*Ww

    attn.reshape([-1, num_heads as i32, n as i32, n as i32])
    // b_nw, num_heads, Wh*Ww, Wh*Ww
}

/// Internal function to create a shifted window image mask.
///
/// This function generates a mask for the shifted window attention mechanism.
///
/// # Arguments
///
/// - `input_shape`: The shape of the input tensor; must be divisible by the window size.
/// - `window_size`: The size of the window.
/// - `shift_size`: The size of the shift.
/// - `device`: The device on which the tensor will be created.
///
/// # Returns
///
/// A tensor representing the shifted window image mask.
#[must_use]
fn sw_img_mask<B: Backend>(
    input_shape: [usize; 2],
    window_size: usize,
    shift_size: usize,
    device: &B::Device,
) -> Tensor<B, 2, Int> {
    let [h, w] = input_shape;

    let mut img_mask = Tensor::<B, 2, Int>::zeros([h, w], device);

    let h = h as i32;
    let w = w as i32;

    let window_size = window_size as i32;
    let shift_size = shift_size as i32;

    assert_eq!(
        h % window_size,
        0,
        "Height {h} is not divisible by window size {window_size}"
    );
    assert_eq!(
        w % window_size,
        0,
        "Width {w} is not divisible by window size {window_size}"
    );

    let h_slices = [
        0..(h - window_size) as usize,
        (h - window_size) as usize..(h - shift_size) as usize,
        (h - shift_size) as usize..h as usize,
    ];
    let w_slices = [
        0..(w - window_size) as usize,
        (w - window_size) as usize..(w - shift_size) as usize,
        (w - shift_size) as usize..w as usize,
    ];

    let mut cnt = 0;
    for h in h_slices.iter() {
        for w in w_slices.iter() {
            let slice_shape = img_mask.clone().slice([h.clone(), w.clone()]).dims();
            let val: Tensor<B, 1, Int> = Tensor::from_data([cnt], device);
            let val = val.unsqueeze::<2>().expand(slice_shape);

            img_mask = img_mask.slice_assign([h.clone(), w.clone()], val);
            cnt += 1;
        }
    }

    img_mask
}

/// Create a shifted window attention mask.
///
/// This function generates a mask for the shifted window attention mechanism.
///
/// # Arguments
///
/// - `input_shape`: The shape of the input tensor.
/// - `window_size`: The size of the window.
/// - `shift_size`: The size of the shift.
/// - `device`: The device on which the tensor will be created.
///
/// # Returns
///
/// A tensor representing the shifted window attention mask.
#[must_use]
pub fn sw_attn_mask<B: Backend>(
    input_shape: [usize; 2],
    window_size: usize,
    shift_size: usize,
    device: &B::Device,
) -> Tensor<B, 3, Bool> {
    let img_mask = sw_img_mask(input_shape, window_size, shift_size, device);
    // ws, ws
    let img_mask = img_mask.unsqueeze_dim::<3>(2).unsqueeze::<4>();
    // b_nw=1, ws, ws, 1

    let mask_windows = window_partition(img_mask, window_size);
    // b_nw=1, ws, ws, 1

    let mask_windows = mask_windows.reshape([-1, (window_size * window_size) as i32]);
    // b_nw=nW, ws * ws

    let mask =
        mask_windows.clone().unsqueeze_dim::<3>(1) - mask_windows.clone().unsqueeze_dim::<3>(2);

    mask.not_equal_elem(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::prelude::{Tensor, TensorData, s};

    #[should_panic(expected = "Height 5 is not divisible by window size 2")]
    #[test]
    fn test_sw_img_mask_height_not_divisible() {
        let device = Default::default();
        let _d = sw_img_mask::<NdArray>([5, 4], 2, 1, &device);
    }

    #[should_panic(expected = "Width 5 is not divisible by window size 2")]
    #[test]
    fn test_sw_img_mask_width_not_divisible() {
        let device = Default::default();
        let _d = sw_img_mask::<NdArray>([4, 5], 2, 1, &device);
    }

    #[test]
    fn test_apply_attention_mask() {
        let b = 2;
        let nw = 2;
        let b_nw = b * nw;
        let ws = 2;
        let n = ws * ws;
        let num_heads = 5;

        let device = Default::default();
        let attn = Tensor::<NdArray, 4>::zeros([b_nw, num_heads, n, n], &device);
        // (b*nw, num_heads, ws*ws, ws*ws)

        // (nw, ws*ws, ws*ws)
        let mask = Tensor::<NdArray, 3>::from_data(
            [
                [
                    [0.0, 0.25, 0.5, 0.75],
                    [1.0, 1.25, 1.5, 1.75],
                    [2.0, 2.25, 2.5, 2.75],
                    [3.0, 3.25, 3.5, 3.75],
                ],
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            ],
            &device,
        );

        let res = apply_attention_mask(b_nw, n, num_heads, attn, mask.clone());
        assert_eq!(res.dims(), [b_nw, num_heads, n, n]);

        let res = res.reshape([b, nw, num_heads, n, n]);

        for bi in 0..b {
            for wi in 0..nw {
                let window = res
                    .clone()
                    .slice(s![bi, wi, .., ..])
                    .squeeze_dim::<4>(0)
                    .squeeze_dim::<3>(0);

                let wmask: Tensor<NdArray, 2> =
                    mask.clone().slice(s![wi, .., ..]).squeeze_dim::<2>(0);

                for hi in 0..num_heads {
                    let h_attn = window.clone().slice(s![hi, .., ..]).squeeze_dim::<2>(0);

                    h_attn.to_data().assert_eq(&wmask.to_data(), true);
                }
            }
        }
    }

    #[test]
    fn test_attn_mask() {
        // let b_nw = 1;
        let device = Default::default();

        sw_attn_mask::<NdArray>([4, 4], 2, 1, &device)
            .to_data()
            .assert_eq(
                &TensorData::from([
                    [
                        [false, false, false, false],
                        [false, false, false, false],
                        [false, false, false, false],
                        [false, false, false, false],
                    ],
                    [
                        [false, true, false, true],
                        [true, false, true, false],
                        [false, true, false, true],
                        [true, false, true, false],
                    ],
                    [
                        [false, false, true, true],
                        [false, false, true, true],
                        [true, true, false, false],
                        [true, true, false, false],
                    ],
                    [
                        [false, true, true, true],
                        [true, false, true, true],
                        [true, true, false, true],
                        [true, true, true, false],
                    ],
                ]),
                true,
            );

        sw_attn_mask::<NdArray>([6, 6], 3, 1, &device)
            .to_data()
            .assert_eq(
                &TensorData::from([
                    [
                        [
                            false, false, false, false, false, false, false, false, false,
                        ],
                        [
                            false, false, false, false, false, false, false, false, false,
                        ],
                        [
                            false, false, false, false, false, false, false, false, false,
                        ],
                        [
                            false, false, false, false, false, false, false, false, false,
                        ],
                        [
                            false, false, false, false, false, false, false, false, false,
                        ],
                        [
                            false, false, false, false, false, false, false, false, false,
                        ],
                        [
                            false, false, false, false, false, false, false, false, false,
                        ],
                        [
                            false, false, false, false, false, false, false, false, false,
                        ],
                        [
                            false, false, false, false, false, false, false, false, false,
                        ],
                    ],
                    [
                        [false, false, true, false, false, true, false, false, true],
                        [false, false, true, false, false, true, false, false, true],
                        [true, true, false, true, true, false, true, true, false],
                        [false, false, true, false, false, true, false, false, true],
                        [false, false, true, false, false, true, false, false, true],
                        [true, true, false, true, true, false, true, true, false],
                        [false, false, true, false, false, true, false, false, true],
                        [false, false, true, false, false, true, false, false, true],
                        [true, true, false, true, true, false, true, true, false],
                    ],
                    [
                        [false, false, false, false, false, false, true, true, true],
                        [false, false, false, false, false, false, true, true, true],
                        [false, false, false, false, false, false, true, true, true],
                        [false, false, false, false, false, false, true, true, true],
                        [false, false, false, false, false, false, true, true, true],
                        [false, false, false, false, false, false, true, true, true],
                        [true, true, true, true, true, true, false, false, false],
                        [true, true, true, true, true, true, false, false, false],
                        [true, true, true, true, true, true, false, false, false],
                    ],
                    [
                        [false, false, true, false, false, true, true, true, true],
                        [false, false, true, false, false, true, true, true, true],
                        [true, true, false, true, true, false, true, true, true],
                        [false, false, true, false, false, true, true, true, true],
                        [false, false, true, false, false, true, true, true, true],
                        [true, true, false, true, true, false, true, true, true],
                        [true, true, true, true, true, true, false, false, true],
                        [true, true, true, true, true, true, false, false, true],
                        [true, true, true, true, true, true, true, true, false],
                    ],
                ]),
                true,
            );

        sw_attn_mask::<NdArray>([6, 6], 3, 2, &device)
            .to_data()
            .assert_eq(
                &TensorData::from([
                    [
                        [
                            false, false, false, false, false, false, false, false, false,
                        ],
                        [
                            false, false, false, false, false, false, false, false, false,
                        ],
                        [
                            false, false, false, false, false, false, false, false, false,
                        ],
                        [
                            false, false, false, false, false, false, false, false, false,
                        ],
                        [
                            false, false, false, false, false, false, false, false, false,
                        ],
                        [
                            false, false, false, false, false, false, false, false, false,
                        ],
                        [
                            false, false, false, false, false, false, false, false, false,
                        ],
                        [
                            false, false, false, false, false, false, false, false, false,
                        ],
                        [
                            false, false, false, false, false, false, false, false, false,
                        ],
                    ],
                    [
                        [false, true, true, false, true, true, false, true, true],
                        [true, false, false, true, false, false, true, false, false],
                        [true, false, false, true, false, false, true, false, false],
                        [false, true, true, false, true, true, false, true, true],
                        [true, false, false, true, false, false, true, false, false],
                        [true, false, false, true, false, false, true, false, false],
                        [false, true, true, false, true, true, false, true, true],
                        [true, false, false, true, false, false, true, false, false],
                        [true, false, false, true, false, false, true, false, false],
                    ],
                    [
                        [false, false, false, true, true, true, true, true, true],
                        [false, false, false, true, true, true, true, true, true],
                        [false, false, false, true, true, true, true, true, true],
                        [true, true, true, false, false, false, false, false, false],
                        [true, true, true, false, false, false, false, false, false],
                        [true, true, true, false, false, false, false, false, false],
                        [true, true, true, false, false, false, false, false, false],
                        [true, true, true, false, false, false, false, false, false],
                        [true, true, true, false, false, false, false, false, false],
                    ],
                    [
                        [false, true, true, true, true, true, true, true, true],
                        [true, false, false, true, true, true, true, true, true],
                        [true, false, false, true, true, true, true, true, true],
                        [true, true, true, false, true, true, false, true, true],
                        [true, true, true, true, false, false, true, false, false],
                        [true, true, true, true, false, false, true, false, false],
                        [true, true, true, false, true, true, false, true, true],
                        [true, true, true, true, false, false, true, false, false],
                        [true, true, true, true, false, false, true, false, false],
                    ],
                ]),
                true,
            );
    }
}
