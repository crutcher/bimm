//! # `DropBlock` Layers
//!
//! Based upon [DropBlock (Ghiasi et al., 2018)](https://arxiv.org/pdf/1810.12890.pdf);
//! inspired also by the `python-image-models` implementation.

use crate::layers::drop::size_config::SizeConfig;
use crate::utility::burn::kernels;
use crate::utility::burn::noise::NoiseConfig;
use crate::utility::probability::expect_probability;
use alloc::format;
use bimm_contracts::unpack_shape_contract;
use burn::config::Config;
use burn::module::{Content, Module, ModuleDisplay, ModuleDisplayDefault};
use burn::prelude::{Backend, Float, Tensor};
use burn::tensor::module::max_pool2d;
use burn::tensor::{DType, Distribution};
use core::ops::Range;
use serde::{Deserialize, Serialize};

/// Configuration for `DropBlock`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DropBlockOptions {
    /// The drop probability.
    pub drop_prob: f64,

    /// The block size.
    pub kernel: [SizeConfig; 2],

    /// The gamma scale.
    pub gamma_scale: f64,

    /// Whether to compute batchwise blocks selection noise.
    /// The alternative is to compute per-image/channel noise.
    /// This results in a speedup proportional to the batchsize.
    pub batchwise: bool,

    /// Should color drop be coupled, or independent?
    pub couple_channels: bool,

    /// Permit partial blocks at the edges.
    /// This results in a significant speedup when using noise.
    pub partial_edge_blocks: bool,

    /// The noise configuration.
    pub noise: Option<NoiseConfig>,
}

impl ModuleDisplay for DropBlockOptions {}
impl ModuleDisplayDefault for DropBlockOptions {
    fn content(
        &self,
        content: Content,
    ) -> Option<Content> {
        Some(
            content
                .add("drop_prob", &self.drop_prob)
                .add("kernel", format!("{:?}", &self.kernel).as_str())
                .add("gamma_scale", &self.gamma_scale)
                .add("batchwise", &self.batchwise)
                .add("couple_channels", &self.couple_channels)
                .add("partial_edge_blocks", &self.partial_edge_blocks)
                .add("noise_cfg", &self.noise),
        )
    }
}

impl Default for DropBlockOptions {
    fn default() -> Self {
        Self {
            drop_prob: 0.1,
            kernel: [7.into(); 2],
            gamma_scale: 1.0,
            noise: None,
            batchwise: true,
            couple_channels: false,
            partial_edge_blocks: false,
        }
    }
}

impl DropBlockOptions {
    /// Extend the options with the given probability.
    ///
    /// # Arguments
    ///
    /// - `drop_prob` - the probability.
    ///
    /// # Panics
    ///
    /// If the `drop_prob` is not in ``[0.0, 1.0]``.
    pub fn with_drop_prob(
        self,
        drop_prob: f64,
    ) -> Self {
        Self {
            drop_prob: expect_probability(drop_prob),
            ..self
        }
    }

    /// Set the symmetric kernel block size.
    ///
    /// This is equivalent to `.with_kernel([block_size; 2])`
    ///
    /// # Arguments
    ///
    /// - `block_size` - the symmetric kernel size.
    pub fn with_block_size<S>(
        self,
        block_size: S,
    ) -> Self
    where
        S: Into<SizeConfig>,
    {
        Self {
            kernel: [block_size.into(); 2],
            ..self
        }
    }

    /// Sets the kernel size.
    ///
    /// # Arguments
    ///
    /// - `kernel` - the kernel size.
    pub fn with_kernel<S>(
        self,
        kernel: [S; 2],
    ) -> Self
    where
        S: Into<SizeConfig>,
    {
        let [x, y] = kernel;
        let kernel = [x.into(), y.into()];
        Self { kernel, ..self }
    }

    /// Set the gamma scale.
    pub fn with_gamma_scale(
        self,
        gamma_scale: f64,
    ) -> Self {
        Self {
            gamma_scale,
            ..self
        }
    }

    /// Set whether to use noise.
    pub fn with_noise<N>(
        self,
        noise_cfg: N,
    ) -> Self
    where
        N: Into<Option<NoiseConfig>>,
    {
        Self {
            noise: noise_cfg.into(),
            ..self
        }
    }

    /// Set if batchwise noise should be used.
    ///
    /// When `batchwise` is enabled, the entire batch shares the same noise;
    /// otherwise, each image gets its own noise.
    ///
    /// Depending on the batchsize, this could be a significant speedup.
    ///
    /// # Arguments
    ///
    /// - `batchwise` - the mode.
    pub fn with_batchwise(
        self,
        batchwise: bool,
    ) -> Self {
        Self { batchwise, ..self }
    }

    /// Set if channels should be coupled.
    pub fn with_couple_channels(
        self,
        couple_channels: bool,
    ) -> Self {
        Self {
            couple_channels,
            ..self
        }
    }

    /// Set if partial edge blocks should be used.
    ///
    /// Partial edge blocks are blocks which overlap the edge;
    /// and are smaller than the full kernel; but not performing
    /// the edge-check saves compute time.
    ///
    /// # Arguments
    ///
    /// - `partial_edge_blocks`: the mode.
    pub fn with_partial_edge_blocks(
        self,
        partial_edge_blocks: bool,
    ) -> Self {
        Self {
            partial_edge_blocks,
            ..self
        }
    }

    /// Clip the kernel to fit within the shape.
    ///
    /// # Arguments
    ///
    /// - `shape`: the shape of the target image.
    ///
    /// # Returns
    ///
    /// The dimension-wise clipped kernel shape.
    #[inline]
    pub fn clipped_kernel(
        &self,
        shape: [usize; 2],
    ) -> [usize; 2] {
        let [h, w] = shape;
        let [kh, kw] = self.kernel;
        let kh = kh.resolve(h);
        let kw = kw.resolve(w);
        [core::cmp::min(h, kh), core::cmp::min(w, kw)]
    }

    /// Compute the adjusted gamma rate.
    ///
    /// Gamma is the adjusted probability that any given point is the midpoint
    /// of a dropped block; given the desired `drop_rate`, the block size, and the input size.
    ///
    /// # Arguments
    ///
    /// - `shape`: the shape of the target tensor.
    #[inline]
    pub fn gamma(
        &self,
        shape: [usize; 2],
    ) -> f64 {
        let [h, w] = shape;
        let [kh, kw] = self.clipped_kernel(shape);

        (self.gamma_scale * self.drop_prob * ((h * w) as f64))
            / ((kh * kw) as f64)
            / (((h - kh + 1) * (w - kw + 1)) as f64)
    }

    /// Compute the gamma noise for the given noise batch shape.
    ///
    /// # Args
    ///
    /// - `shape`: the target noise batch shape.
    ///
    /// # Returns
    ///
    /// Gamma noise, sampled at ``self.gamma([h, w])`` rate.
    pub fn gamma_noise<B: Backend>(
        &self,
        noise_shape: [usize; 4],
        device: &B::Device,
    ) -> Tensor<B, 4> {
        let [_, _, h, w] = noise_shape;
        let gamma = self.gamma([h, w]);
        Tensor::random(noise_shape, Distribution::Bernoulli(gamma), device)
    }
}

/// Convert drop block gamma noise to a selection filter.
///
/// This is a deterministic internal component of `drop_block`.
///
/// # Arguments
///
/// * `selected_blocks` - Input selection noise;
///   `1.0` at the midpoints of selected blocks to drop,
///   `0.0` everywhere else. Expected to be gamma noise.
/// * `kernel_shape` - the shape of the kernel.
/// * `partial_edge_blocks` - permit partial blocks at the edges, faster.
fn drop_block_2d_drop_filter_<B: Backend>(
    selected_blocks: Tensor<B, 4>,
    kernel_shape: [usize; 2],
    partial_edge_blocks: bool,
) -> Tensor<B, 4> {
    let [_, _, h, w] = unpack_shape_contract!(["b", "c", "h", "w"], &selected_blocks);
    let [kh, kw] = kernel_shape;

    assert!(
        kh <= h && kw <= w,
        "Kernel size ({kh}, {kw}) is larger than input size ({h}, {w})",
    );

    let dtype = selected_blocks.dtype();
    let device = &selected_blocks.device();

    let mut selection = selected_blocks;

    if !partial_edge_blocks {
        selection = selection
            * kernels::conv2d_kernel_midpoint_filter::<B, Float>([h, w], kernel_shape, device)
                .unsqueeze_dims::<4>(&[0, 1])
                .cast(dtype);
    }

    selection = max_pool2d(selection, kernel_shape, [1, 1], [kh / 2, kw / 2], [1, 1]);

    // Clip even-kernel padding artifacts.
    if (kh % 2) == 0 || (kw % 2) == 0 {
        let mut ranges: [Range<usize>; 4] = selection.shape().into_ranges().try_into().unwrap();
        ranges[2].start = ((kh % 2) == 0) as usize;
        ranges[3].start = ((kw % 2) == 0) as usize;

        selection = selection.slice(ranges);
    }

    selection
}

/// `drop_block_2d`
///
/// Drops block kernels from a tensor; many options.
///
/// Dropped values can be resampled from a noise distribution,
/// kept values can be re-normalized.
/// The drop probability, block size, and several performance/quality tradeoffs
/// can be configured.
///
/// Based upon [DropBlock (Ghiasi, et al., 2018)](https://arxiv.org/pdf/1810.12890.pdf);
/// inspired also by the `python-image-models` implementation.
///
/// # Arguments
///
/// * `tensor` - the tensor to operate on, ``[batch, channels, height, width]``.
/// * `options` - the algorithm options.
///
/// # Returns
///
/// A ``[batch, channels, height, width]`` tensor.
pub fn drop_block_2d<B: Backend>(
    tensor: Tensor<B, 4>,
    options: &DropBlockOptions,
) -> Tensor<B, 4> {
    if options.drop_prob == 0.0 {
        // This is a no-op.
        return tensor;
    }

    let [b, c, h, w] = tensor.shape().dims();
    let kernel = options.clipped_kernel([h, w]);

    let t_shape = tensor.shape();
    let device = &tensor.device();
    let dtype = tensor.dtype();

    let noise_shape = [
        if options.batchwise { 1 } else { b },
        if options.couple_channels { 1 } else { c },
        h,
        w,
    ];

    let gamma_noise = options.gamma_noise(noise_shape, device);

    let drop_filter: Tensor<B, 4> =
        drop_block_2d_drop_filter_(gamma_noise, kernel, options.partial_edge_blocks).cast(dtype);
    let keep_filter: Tensor<B, 4> = 1.0 - drop_filter.clone();

    if let Some(noise_cfg) = &options.noise {
        // Fill in the dropped regions with sampled noise.
        let noise: Tensor<B, 4> = noise_cfg.noise(noise_shape, device).cast(dtype);
        let noise = noise * drop_filter;

        tensor * keep_filter.expand(t_shape.clone()) + noise.expand(t_shape)
    } else {
        // Rescale to normalize to 1.0.
        let count = keep_filter.shape().num_elements() as f32;
        let total = keep_filter.clone().cast(DType::F32).sum();
        let norm_scale = count / total.add_scalar(1e-7);

        tensor * keep_filter.expand(t_shape.clone()) * norm_scale.cast(dtype).expand(t_shape)
    }
}

/// Config for [`DropBlock2dConfig`] modules.
#[derive(Config, Debug)]
pub struct DropBlock2dConfig {
    /// The options for the drop block algorithm.
    #[config(default = "DropBlockOptions::default()")]
    pub options: DropBlockOptions,
}

impl From<DropBlockOptions> for DropBlock2dConfig {
    fn from(options: DropBlockOptions) -> Self {
        Self { options }
    }
}

impl DropBlock2dConfig {
    /// Initialize a [`DropBlock2d`] module.
    pub fn init(&self) -> DropBlock2d {
        DropBlock2d {
            options: self.options.clone(),
        }
    }
}

/// `DropBlock2d`
///
/// A module that applies drop block (when training).
///
/// Based upon [DropBlock (Ghiasi et al., 2018)](https://arxiv.org/pdf/1810.12890.pdf);
/// inspired also by the `python-image-models` implementation.
#[derive(Module, Clone, Debug)]
pub struct DropBlock2d {
    /// The options for the drop block algorithm.
    pub options: DropBlockOptions,
}

impl DropBlock2d {
    /// When training, applies `drop_block_2d` to the tensor;
    /// otherwise, is a no-op pass-through.
    ///
    /// # Arguments
    ///
    /// - `tensor` - the input.
    ///
    /// # Returns
    ///
    /// A tensor of the same shape, type, and device.
    pub fn forward<B: Backend>(
        &self,
        tensor: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        if B::ad_enabled() {
            drop_block_2d(tensor.clone(), &self.options)
        } else {
            tensor
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::utility::burn::noise::NoiseConfig;
    use burn::backend::{Autodiff, NdArray};
    use burn::module::DisplaySettings;
    use burn::prelude::TensorData;

    #[test]
    fn test_drop_block_options() {
        let options = DropBlockOptions::default();
        assert_eq!(options.drop_prob, 0.1);
        assert_eq!(options.kernel, [SizeConfig::Fixed(7); 2]);
        assert_eq!(options.gamma_scale, 1.0);
        assert_eq!(options.couple_channels, false);
        assert!(options.noise.is_none());
        assert_eq!(options.batchwise, true);

        let options = options.with_couple_channels(true);
        assert_eq!(options.couple_channels, true);

        let options = options.with_drop_prob(0.2);
        assert_eq!(options.drop_prob, 0.2);

        let options = options.with_block_size(0.25);
        assert_eq!(options.kernel, [SizeConfig::Ratio(0.25); 2]);

        let options = options.with_gamma_scale(0.5);
        assert_eq!(options.gamma_scale, 0.5);

        let options = options.with_batchwise(false);
        assert_eq!(options.batchwise, false);
    }

    #[test]
    fn test_drop_block_options_display() {
        let options = DropBlockOptions::default().with_noise(NoiseConfig::default());
        let settings = DisplaySettings::default();

        assert_eq!(
            options.format(settings),
            indoc::indoc! {r#"
                DropBlockOptions {
                  drop_prob: 0.1
                  kernel: [Fixed(7), Fixed(7)]
                  gamma_scale: 1
                  batchwise: true
                  couple_channels: false
                  partial_edge_blocks: false
                  noise_cfg: NoiseConfig {
                      distribution: Distribution::Default
                      clamp: None
                    }
                }"#
            }
        )
    }

    #[test]
    fn test_gamma() {
        let options = DropBlockOptions::default()
            .with_drop_prob(0.1)
            .with_gamma_scale(1.2)
            .with_kernel([2, 3]);

        let shape = [7, 9];
        let [h, w] = shape;
        let [kh, kw] = options.clipped_kernel(shape);

        let total_size = (h * w) as f64;

        let gamma = options.gamma([h, w]);

        let expected = (options.gamma_scale * options.drop_prob * total_size)
            / ((kh * kw) as f64)
            / (((h - kh + 1) * (w - kw + 1)) as f64);

        assert_eq!(gamma, expected);
    }

    #[test]
    fn test_drop_block_2d_drop_filter() {
        type B = NdArray;
        let device = Default::default();

        let selected_blocks: Tensor<B, 4> = Tensor::<B, 2>::from_data(
            [
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            &device,
        )
        .unsqueeze_dims::<4>(&[0, 1]);

        // No partial edge blocks.
        let drop_filter = drop_block_2d_drop_filter_(selected_blocks.clone(), [2, 3], false);
        drop_filter.squeeze_dims::<2>(&[0, 1]).to_data().assert_eq(
            &TensorData::from([
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]),
            false,
        );

        // Partial edge blocks.
        let drop_filter = drop_block_2d_drop_filter_(selected_blocks.clone(), [2, 3], true);
        drop_filter.squeeze_dims::<2>(&[0, 1]).to_data().assert_eq(
            &TensorData::from([
                [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]),
            false,
        );
    }

    #[test]
    fn test_drop_block_2d_no_op() {
        type B = NdArray;
        let device = Default::default();

        let shape = [2, 3, 7, 9];
        let tensor: Tensor<B, 4> = Tensor::ones(shape, &device);

        let drop_prob = 0.0;

        let drop = drop_block_2d(
            tensor.clone(),
            &DropBlockOptions::default()
                .with_drop_prob(drop_prob)
                .with_kernel([2, 3]),
        );

        drop.to_data().assert_eq(&tensor.to_data(), false);
    }

    #[test]
    fn test_drop_block_2d_with_norm() {
        type B = NdArray;
        let device = Default::default();

        let shape = [2, 3, 100, 100];
        let tensor: Tensor<B, 4> = Tensor::ones(shape, &device);

        let drop_prob = 0.1;

        let drop = drop_block_2d(
            tensor,
            &DropBlockOptions::default()
                .with_partial_edge_blocks(false)
                .with_drop_prob(drop_prob)
                .with_kernel([2, 3]),
        );

        // Count all 1.0; which are the non-dropped values.
        let numel = drop.shape().num_elements();

        // They've all been rescaled upwards.
        let keep_count = drop.clone().greater_elem(1.0).int().sum().into_scalar() as usize;
        let drop_count = numel - keep_count;
        let drop_ratio = drop_count as f64 / numel as f64;
        assert!((drop_ratio - drop_prob).abs() < 0.15);

        let total = drop.sum().into_scalar() as f64;
        let norm = total / numel as f64;
        assert!((norm - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_drop_block_2d_with_noise() {
        type B = NdArray;
        let device = Default::default();

        let shape = [2, 3, 100, 100];
        let tensor: Tensor<B, 4> = Tensor::ones(shape, &device);

        let drop_prob = 0.1;

        let drop = drop_block_2d(
            tensor,
            &DropBlockOptions::default()
                .with_noise(NoiseConfig::default())
                .with_partial_edge_blocks(false)
                .with_drop_prob(drop_prob)
                .with_kernel([2, 3]),
        );

        // Count all 1.0; which are the non-dropped values.
        let numel = drop.shape().num_elements();

        // This should be an exact match; the Distribution::Default is [0.0, 1.0);
        // and will never generate a 1.0.
        let keep_count = drop.equal_elem(1.0).int().sum().into_scalar() as usize;

        let drop_count = numel - keep_count;

        let drop_ratio = drop_count as f64 / numel as f64;

        assert!((drop_ratio - drop_prob).abs() < 0.15);
    }

    #[test]
    fn test_module_inference() {
        type B = NdArray;
        let device = Default::default();

        let config = DropBlock2dConfig::new();

        let module = config.init();

        let batch_size = 2;
        let channels = 3;
        let height = 100;
        let width = height;
        let shape = [batch_size, channels, height, width];

        let tensor: Tensor<B, 4> = Tensor::ones(shape, &device);

        assert_eq!(B::ad_enabled(), false);
        let result = module.forward(tensor.clone());

        // Not under training; so a no-op.
        result.to_data().assert_eq(&tensor.to_data(), false);
    }

    #[test]
    fn test_module_training() {
        type B = Autodiff<NdArray>;
        let device = Default::default();

        let drop_prob = 0.1;

        let config = DropBlock2dConfig::new().with_options(
            DropBlockOptions::default()
                .with_drop_prob(drop_prob)
                .with_kernel([2, 3]),
        );

        let module = config.init();

        let batch_size = 2;
        let channels = 3;
        let height = 100;
        let width = height;
        let shape = [batch_size, channels, height, width];

        let tensor: Tensor<B, 4> = Tensor::ones(shape, &device);

        assert_eq!(B::ad_enabled(), true);
        let drop = module.forward(tensor.clone());

        // Count all 1.0; which are the non-dropped values.
        let numel = drop.shape().num_elements();

        // They've all been rescaled upwards.
        let keep_count = drop.clone().greater_elem(1.0).int().sum().into_scalar() as usize;
        let drop_count = numel - keep_count;
        let drop_ratio = drop_count as f64 / numel as f64;
        assert!((drop_ratio - drop_prob).abs() < 0.15);

        let total = drop.sum().into_scalar() as f64;
        let norm = total / numel as f64;
        assert!((norm - 1.0).abs() < 0.01);
    }
}
