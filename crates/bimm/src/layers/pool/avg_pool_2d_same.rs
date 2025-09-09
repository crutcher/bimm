use burn::module::Module;
use burn::nn::pool::AvgPool2d;
use burn::prelude::{Backend, Tensor};
use std::cmp::max;

/// `AvgPool2dSame`
///
/// # Reference Python
///
/// ```python,ignore
/// # Calculate asymmetric TensorFlow-like 'SAME' padding for a convolution
/// def get_same_padding(x: int, kernel_size: int, stride: int, dilation: int):
///     if isinstance(x, torch.Tensor):
///         return torch.clamp(((x / stride).ceil() - 1) * stride + (kernel_size - 1) * dilation + 1 - x, min=0)
///     else:
///         return max((math.ceil(x / stride) - 1) * stride + (kernel_size - 1) * dilation + 1 - x, 0)
///
/// # Dynamically pad input x with 'SAME' padding for conv with specified args
/// def pad_same(
///         x,
///         kernel_size: List[int],
///         stride: List[int],
///         dilation: List[int] = (1, 1),
///         value: float = 0,
///     ):
///     ih, iw = x.size()[-2:]
///     pad_h = get_same_padding(ih, kernel_size[0], stride[0], dilation[0])
///     pad_w = get_same_padding(iw, kernel_size[1], stride[1], dilation[1])
///     x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
///     return x
///
///
/// def avg_pool2d_same(
///         x: torch.Tensor,
///         kernel_size: List[int],
///         stride: List[int],
///         padding: List[int] = (0, 0),
///         ceil_mode: bool = False,
///         count_include_pad: bool = True,
///     ):
///     # FIXME how to deal with count_include_pad vs not for external padding?
///     x = pad_same(x, kernel_size, stride)
///     return F.avg_pool2d(x, kernel_size, stride, (0, 0), ceil_mode, count_include_pad)
///
/// class AvgPool2d(nn.Module):
///     def __init__(
///         self,
///         kernel_size: _size_2_t,
///         stride: Optional[_size_2_t] = None,
///         padding: _size_2_t = 0,
///         ceil_mode: bool = False,
///         count_include_pad: bool = True,
///         divisor_override: Optional[int] = None,
///     ) -> None:
///         super().__init__()
///         self.kernel_size = kernel_size
///         self.stride = stride if (stride is not None) else kernel_size
///         self.padding = padding
///         self.ceil_mode = ceil_mode
///         self.count_include_pad = count_include_pad
///         self.divisor_override = divisor_override
///
///     def forward(self, input: Tensor) -> Tensor:
///         return F.avg_pool2d(
///             input,
///             self.kernel_size,
///             self.stride,
///             self.padding,
///             self.ceil_mode,
///             self.count_include_pad,
///             self.divisor_override,
///         )
///
/// class AvgPool2dSame(nn.AvgPool2d):
///     """Tensorflow like 'SAME' wrapper for 2D average pooling."""
///     def __init__(
///             self,
///             kernel_size: _size_2_t,
///             stride: Optional[_size_2_t] = None,
///             padding: _size_2_t = 0,
///             ceil_mode=False,
///             count_include_pad=True,
///     ):
///         super(AvgPool2dSame, self).__init__(
///             kernel_size=kernel_size,
///             stride=stride,
///             padding=(0, 0), # padding is dropped, is this a bug?
///             ceil_mode=ceil_mode,
///             count_include_pad=count_include_pad,
///         )
///
///     def forward(self, x):
///         x = pad_same(x, self.kernel_size, self.stride)
///         return F.avg_pool2d(
///             x,
///             self.kernel_size,
///             self.stride,
///             self.padding,
///             self.ceil_mode,
///             self.count_include_pad,
///         )
///
/// ```
#[derive(Module, Clone, Debug)]
pub struct AvgPool2dSame {
    pool: AvgPool2d,
}

impl AvgPool2dSame {
    /// Forward Pass.
    pub fn forward<B: Backend>(
        &self,
        input: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        let x = pad_same(input, self.pool.kernel_size, self.pool.stride, [1, 1], 0.0);
        self.pool.forward(x)
    }
}

/// Calculate asymmetric TensorFlow-like 'SAME' padding for a convolution.
pub fn get_same_padding(
    size: usize,
    kernel_size: usize,
    stride: usize,
    dilation: usize,
) -> usize {
    max(
        (((size + (stride / 2)) / stride) - 1) * stride + (kernel_size - 1) * dilation + 1 - size,
        0,
    )
}

/// Dynamically pad input x with 'SAME' padding for conv with specified args.
pub fn pad_same<B: Backend>(
    input: Tensor<B, 4>,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    dilation: [usize; 2],
    value: f32,
) -> Tensor<B, 4> {
    let ih = input.shape().dims[2];
    let iw = input.shape().dims[3];
    let pad_h = get_same_padding(ih, kernel_size[0], stride[0], dilation[0]);
    let pad_w = get_same_padding(iw, kernel_size[1], stride[1], dilation[1]);
    input.pad(
        (pad_w / 2, pad_w - pad_w / 2, pad_h / 2, pad_h - pad_h / 2),
        value,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_same_padding() {
        assert_eq!(get_same_padding(10, 1, 1, 1), 0);

        assert_eq!(get_same_padding(10, 3, 2, 1), 1);

        assert_eq!(get_same_padding(10, 3, 2, 2), 3);
    }
}
