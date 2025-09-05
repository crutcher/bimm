//! # Patch Merging
use crate::models::swin::v2::windowing::{window_partition, window_reverse};
use bimm_contracts::{assert_shape_contract_periodically, unpack_shape_contract};
use burn::config::Config;
use burn::module::Module;
use burn::nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::prelude::{Backend, Tensor};
use burn::tensor::BasicOps;

/// Metadata for `PatchMerging`.
pub trait PatchMergingMeta {
    /// Input feature dimension size.
    fn d_input(&self) -> usize;

    /// Output feature dimension size.
    fn d_output(&self) -> usize {
        2 * self.d_input()
    }

    /// Input resolution (height, width).
    fn input_resolution(&self) -> [usize; 2];

    /// Input height.
    fn input_height(&self) -> usize {
        self.input_resolution()[0]
    }

    /// Input width.
    fn input_width(&self) -> usize {
        self.input_resolution()[1]
    }

    /// Output resolution (height, width).
    fn output_resolution(&self) -> [usize; 2] {
        [self.output_height(), self.output_width()]
    }

    /// Output height.
    fn output_height(&self) -> usize {
        self.input_height() / 2
    }

    /// Output width.
    fn output_width(&self) -> usize {
        self.input_width() / 2
    }
}

/// Configuration for `PatchMerging`.
#[derive(Config, Debug)]
pub struct PatchMergingConfig {
    /// Input resolution (height, width).
    input_resolution: [usize; 2],

    /// Input feature dimension size.
    d_input: usize,
}

impl PatchMergingMeta for PatchMergingConfig {
    fn d_input(&self) -> usize {
        self.d_input
    }

    fn input_resolution(&self) -> [usize; 2] {
        self.input_resolution
    }
}

impl PatchMergingConfig {
    /// Create a new `PatchMerging` configuration.
    ///
    /// # Arguments
    ///
    /// - `device`: The backend device to initialize the module on.
    ///
    /// # Returns
    ///
    /// A new `PatchMerging` module initialized with the given configuration.
    ///
    /// # Panics
    ///
    /// If the config is invalid.
    #[must_use]
    pub fn init<B: Backend>(
        &self,
        device: &B::Device,
    ) -> PatchMerging<B> {
        let [h, w] = self.input_resolution;
        assert!(
            h % 2 == 0 && w % 2 == 0,
            "Input resolution must be divisible by 2: {:?}",
            self.input_resolution
        );

        PatchMerging {
            input_resolution: self.input_resolution,
            reduction: LinearConfig::new(2 * self.d_output(), self.d_output())
                .with_bias(false)
                .init(device),
            norm: LayerNormConfig::new(self.d_output()).init(device),
        }
    }
}

/// SWIN-Transformer v2 `PatchMerging` module.
///
/// This module accepts ``[batch, height * width, channels]`` inputs, and then:
/// - Collates interleaved patches of size ``[height/2, width/2]`` into
///   ``[batch, height/2 * width/2, 4 * channels]``.
/// - Applies a linear layer to reduce the feature dimension to ``2 * channels``.
/// - Applies layer normalization.
/// - Yields output of shape ``[batch, height/2 * width/2, 2 * channels]``.
///
/// See: <https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer_v2.py>
#[derive(Module, Debug)]
pub struct PatchMerging<B: Backend> {
    /// Input resolution (height, width).
    input_resolution: [usize; 2],

    /// Linear layer for reducing the feature dimension.
    reduction: Linear<B>,

    /// Layer norm layer.
    norm: LayerNorm<B>,
}

impl<B: Backend> PatchMergingMeta for PatchMerging<B> {
    fn d_input(&self) -> usize {
        self.reduction.weight.dims()[0] / 4
    }

    fn input_resolution(&self) -> [usize; 2] {
        self.input_resolution
    }
}

impl<B: Backend> PatchMerging<B> {
    /// Forward pass of the `PatchMerging` module.
    ///
    /// # Arguments
    ///
    /// - `x` - Input tensor of shape ``[batch, height * width, channels]``.
    ///
    /// # Returns
    ///
    /// A tensor of shape ``[batch, height/2 * width/2, 2 * channels]``.
    ///
    /// # Panics
    ///
    /// On shape contract failure.
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        let [b, h, w] = unpack_shape_contract!(
            ["batch", "flat" = "height" * "width", "d_in"],
            &x,
            &["batch", "height", "width"],
            &[
                ("height", self.input_height()),
                ("width", self.input_width()),
                ("d_in", self.d_input()),
            ]
        );

        let x = collate_patches(x, h, w);

        let x = self.reduction.forward(x);

        let x = self.norm.forward(x);
        assert_shape_contract_periodically!(
            ["batch", "flat" = "half_height" * "half_width", "d_out"],
            &x,
            &[
                ("batch", b),
                ("half_height", self.output_height()),
                ("half_width", self.output_width()),
                ("d_out", self.d_output()),
            ]
        );

        x
    }
}

/// Collate patches from a tensor.
///
/// This splits the input into 4 interleaved patches, each ``[height/2, width/2]``;
/// and then concatenates them along the last dimension.
///
/// # Arguments
///
/// * `x` - Input tensor of shape ``[batch, height * width, channels]``.
/// * `h` - Height of the input tensor.
/// * `w` - Width of the input tensor.
///
/// # Returns
///
/// * A tensor of shape ``[batch, height/2 * width/2, 4 * channels]``.
///
/// # Panics
///
/// On shape contract failure.
pub fn collate_patches<B: Backend, K>(
    x: Tensor<B, 3, K>,
    h: usize,
    w: usize,
) -> Tensor<B, 3, K>
where
    K: BasicOps<B>,
{
    let [b, h, w, c] = unpack_shape_contract!(
        ["batch", "flat" = "height" * "width", "channels"],
        &x,
        &["batch", "height", "width", "channels"],
        &[("height", h), ("width", w)]
    );

    let h2 = h / 2;
    let w2 = w / 2;
    let h2w2 = h2 * w2;

    // TODO(crutcher): re-using `window_partition` requires us to double-reshape.
    // Is it worth writing a trait to permit windowing on 3D and 4D tensors?
    // In SWIN Source; window_partition is *always* immediately resized.
    let x = x.reshape([b, h, w, c]);
    let x = window_partition(x, 2);

    x.reshape([b, h2w2, 4 * c])
}

/// Decollate patches from a tensor.
///
/// This splits the input into 4 patches, each ``[height/2, width/2]``;
/// and interleaves them along the height and width dimensions.
///
/// The inverse operation of `collate_patches`.
///
/// # Arguments
///
/// * `x` - Input tensor of shape ``[batch, height/2 * width/2, 4 * channels]``.
/// * `height` - Height of the input tensor.
/// * `width` - Width of the input tensor.
///
/// # Returns
///
/// * A tensor of shape ``[batch, height * width, channels]``.
///
/// # Panics
///
/// On shape contract failure.
pub fn decollate_patches<B: Backend, K>(
    x: Tensor<B, 3, K>,
    height: usize,
    width: usize,
) -> Tensor<B, 3, K>
where
    K: BasicOps<B>,
{
    let h2 = height / 2;
    let w2 = width / 2;

    let [b, c] = unpack_shape_contract!(
        ["batch", "half_height" * "half_width", "channels"],
        &x,
        &["batch", "channels"],
        &[("half_height", h2), ("half_width", w2)]
    );

    let c = c / 4;

    let x = x.reshape([b * h2 * w2, 2, 2, c]);
    let x = window_reverse(x, 2, height, width);

    x.reshape([b, height * width, c])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::patching::patch_embed::{PatchEmbedConfig, PatchEmbedMeta};
    use burn::backend::NdArray;
    use burn::prelude::Backend;
    use burn::tensor::Distribution;

    #[test]
    fn test_collate_patches() {
        let b = 2;
        let h = 4;
        let w = 6;
        let c = 5;

        let device = Default::default();

        let distribution = Distribution::Normal(0., 1.);
        let x = Tensor::<NdArray, 3>::random([b, h * w, c], distribution, &device);

        let y = collate_patches(x.clone(), h, w);
        assert_eq!(&y.dims(), &[b, (h / 2) * (w / 2), 4 * c]);

        decollate_patches(y.clone(), h, w)
            .into_data()
            .assert_eq(&x.into_data(), true);
    }

    #[test]
    fn test_patch_merging_meta() {
        let config = PatchMergingConfig {
            input_resolution: [12, 8],
            d_input: 3,
        };

        assert_eq!(config.input_resolution(), [12, 8]);
        assert_eq!(config.d_input(), 3);
        assert_eq!(config.d_output(), 6);
        assert_eq!(config.output_resolution(), [6, 4]);
        assert_eq!(config.output_height(), 6);
        assert_eq!(config.output_width(), 4);

        let patch_merging = config.init::<NdArray>(&Default::default());

        assert_eq!(patch_merging.input_resolution(), [12, 8]);
        assert_eq!(patch_merging.d_input(), 3);
        assert_eq!(patch_merging.d_output(), 6);
        assert_eq!(patch_merging.output_resolution(), [6, 4]);
        assert_eq!(patch_merging.output_height(), 6);
        assert_eq!(patch_merging.output_width(), 4);
    }

    #[should_panic(expected = "Input resolution must be divisible by 2")]
    #[test]
    fn test_patch_merging_invalid_resolution() {
        let config = PatchMergingConfig {
            input_resolution: [13, 8], // Invalid height
            d_input: 3,
        };
        let device = Default::default();
        let _d = config.init::<NdArray>(&device);
    }

    #[test]
    fn test_patch_merging() {
        impl_test_patch_merging::<NdArray>();
    }

    fn impl_test_patch_merging<B: Backend>() {
        let device: B::Device = Default::default();

        let b = 2;
        let h = 12;
        let w = 8;
        let c = 3;

        let config = PatchMergingConfig {
            input_resolution: [h, w],
            d_input: c,
        };
        let patch_merging = config.init::<B>(&device);

        let distribution = Distribution::Normal(0., 1.);
        let x = Tensor::random([b, h * w, c], distribution, &device);

        let y = patch_merging.forward(x.clone());
        assert_eq!(&y.dims(), &[b, h / 2 * w / 2, 2 * c]);
    }

    #[test]
    fn test_patch_embed_meta() {
        let config = PatchEmbedConfig::new([12, 8], 4, 3, 6).with_enable_patch_norm(false);

        assert_eq!(config.input_resolution(), [12, 8]);
        assert_eq!(config.patch_size(), 4);
        assert_eq!(config.d_input(), 3);
        assert_eq!(config.d_output(), 6);
        assert!(!config.enable_patch_norm());
        assert_eq!(config.patches_resolution(), [3, 2]);
        assert_eq!(config.patches_height(), 3);
        assert_eq!(config.patches_width(), 2);

        let patch_embed = config.init::<NdArray>(&Default::default());

        assert_eq!(patch_embed.input_resolution(), [12, 8]);
        assert_eq!(patch_embed.patch_size(), 4);
        assert_eq!(patch_embed.d_input(), 3);
        assert_eq!(patch_embed.d_output(), 6);
        assert!(!patch_embed.enable_patch_norm());
        assert_eq!(patch_embed.patches_resolution(), [3, 2]);
        assert_eq!(patch_embed.patches_height(), 3);
        assert_eq!(patch_embed.patches_width(), 2);
    }

    #[test]
    fn test_patch_embed() {
        let device = Default::default();

        let b = 2;
        let h = 12;
        let w = 8;

        let patch_size = 4;
        let d_input = 3;
        let d_output = d_input * 2;

        let distribution = Distribution::Normal(0., 1.);
        let x = Tensor::random([b, d_input, h, w], distribution, &device);

        // W/O Norm.
        {
            let config = PatchEmbedConfig::new([h, w], patch_size, d_input, d_output)
                .with_enable_patch_norm(false);
            let patch_embed = config.init::<NdArray>(&device);

            let y = patch_embed.forward(x.clone());
            assert_eq!(&y.dims(), &[b, (h / 4) * (w / 4), d_output]);

            let z = patch_embed.projection.forward(x.clone());
            let z: Tensor<NdArray, 3> = z.flatten(2, 3);
            let z = z.swap_dims(1, 2);

            y.into_data().assert_eq(&z.into_data(), true);
        }

        // With Norm.
        {
            let config = PatchEmbedConfig::new([h, w], patch_size, d_input, d_output);
            let patch_embed = config.init::<NdArray>(&device);

            let y = patch_embed.forward(x.clone());
            assert_eq!(&y.dims(), &[b, (h / 4) * (w / 4), d_output]);

            let z = patch_embed.projection.forward(x.clone());
            let z: Tensor<NdArray, 3> = z.flatten(2, 3);
            let z = z.swap_dims(1, 2);
            let z = patch_embed.norm.as_ref().unwrap().forward(z);

            y.into_data().assert_eq(&z.into_data(), true);
        }
    }
}
