//! # Patch Embedding layers and operations.
use bimm_contracts::assert_shape_contract_periodically;
use burn::config::Config;
use burn::module::Module;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::{LayerNorm, LayerNormConfig};
use burn::prelude::{Backend, Tensor};

/// Common introspection interface for `PatchEmbed` modules.
pub trait PatchEmbedMeta {
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

    /// Input feature dimension size.
    fn d_input(&self) -> usize;

    /// The size of each patch.
    fn patch_size(&self) -> usize;

    /// Image resolution, measured in patches.
    fn patches_resolution(&self) -> [usize; 2] {
        [self.patches_height(), self.patches_width()]
    }

    /// Height of the image, measured in patches.
    fn patches_height(&self) -> usize {
        self.input_height() / self.patch_size()
    }

    /// Width of the image, measured in patches.
    fn patches_width(&self) -> usize {
        self.input_width() / self.patch_size()
    }

    /// Total number of patches.
    fn num_patches(&self) -> usize {
        self.patches_height() * self.patches_width()
    }

    /// Output feature dimension size.
    fn d_output(&self) -> usize;

    /// Enable patch normalization.
    fn enable_patch_norm(&self) -> bool;
}

/// Configuration for `PatchEmbed`.
#[derive(Config, Debug, Copy)]
pub struct PatchEmbedConfig {
    /// Input resolution (height, width).
    input_resolution: [usize; 2],

    /// Patch size.
    patch_size: usize,

    /// Input feature dimension size.
    d_input: usize,

    /// Output feature dimension size.
    d_output: usize,

    /// Enable patch normalization.
    #[config(default = true)]
    enable_patch_norm: bool,
}

impl PatchEmbedMeta for PatchEmbedConfig {
    fn input_resolution(&self) -> [usize; 2] {
        self.input_resolution
    }

    fn patch_size(&self) -> usize {
        self.patch_size
    }

    fn d_input(&self) -> usize {
        self.d_input
    }

    fn d_output(&self) -> usize {
        self.d_output
    }

    fn enable_patch_norm(&self) -> bool {
        self.enable_patch_norm
    }
}

impl PatchEmbedConfig {
    /// Initialize a `PatchEmbed` module with the given configuration.
    ///
    /// # Arguments
    ///
    /// * `device` - The device on which the module will be initialized.
    ///
    /// # Returns
    ///
    /// * A `PatchEmbed` module configured with the specified parameters.
    #[must_use]
    pub fn init<B: Backend>(
        &self,
        device: &B::Device,
    ) -> PatchEmbed<B> {
        let [h, w] = self.input_resolution;
        assert!(
            h % self.patch_size == 0 && w % self.patch_size == 0,
            "Input resolution must be divisible by patch size: {:?}",
            self.input_resolution
        );

        let stride = [self.patch_size, self.patch_size];

        PatchEmbed {
            input_resolution: self.input_resolution,
            patch_size: self.patch_size,
            projection: Conv2dConfig::new([self.d_input, self.d_output], stride)
                .with_stride(stride)
                .init(device),
            norm: match self.enable_patch_norm {
                true => Some(LayerNormConfig::new(self.d_output()).init(device)),
                false => None,
            },
        }
    }
}

/// SWIN-Transformer v2 `PatchEmbed` module.
#[derive(Module, Debug)]
pub struct PatchEmbed<B: Backend> {
    /// Input resolution (height, width).
    pub input_resolution: [usize; 2],

    /// Size of each patch.
    pub patch_size: usize,

    /// Convolutional layer for patch projection.
    pub projection: Conv2d<B>,

    /// Patch normalization layer, if enabled.
    pub norm: Option<LayerNorm<B>>,
}

impl<B: Backend> PatchEmbedMeta for PatchEmbed<B> {
    fn input_resolution(&self) -> [usize; 2] {
        self.input_resolution
    }

    fn patch_size(&self) -> usize {
        self.patch_size
    }

    fn d_input(&self) -> usize {
        self.projection.weight.dims()[1]
    }

    fn d_output(&self) -> usize {
        self.projection.weight.dims()[0]
    }

    fn enable_patch_norm(&self) -> bool {
        self.norm.is_some()
    }
}

impl<B: Backend> PatchEmbed<B> {
    /// Apply the `PatchEmbed` module to an input tensor.
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor of shape ``[B, C, H, W]``.
    ///
    /// # Returns
    ///
    /// * Output tensor of shape ``[B, H/patch_size * W/patch_size, d_output]``.
    #[must_use]
    pub fn forward(
        &self,
        x: Tensor<B, 4>,
    ) -> Tensor<B, 3> {
        assert_shape_contract_periodically!(
            ["batch", "d_input", "height", "width"],
            &x.dims(),
            &[
                ("d_input", self.d_input()),
                ("height", self.input_height()),
                ("width", self.input_width()),
            ]
        );

        let batch = x.dims()[0];

        let x = self.projection.forward(x);
        assert_shape_contract_periodically!(
            ["batch", "d_output", "patches_height", "patches_width"],
            &x.dims(),
            &[
                ("batch", batch),
                ("d_output", self.d_output()),
                ("patches_height", self.patches_height()),
                ("patches_width", self.patches_width()),
            ],
        );

        let x = x.flatten(2, 3);
        let x = x.swap_dims(1, 2);
        assert_shape_contract_periodically!(
            ["batch", "num_patches", "d_output"],
            &x.dims(),
            &[
                ("batch", batch),
                ("num_patches", self.num_patches()),
                ("d_output", self.d_output()),
            ],
        );

        let x = match self.norm {
            None => x,
            Some(ref norm) => norm.forward(x),
        };
        assert_shape_contract_periodically!(
            ["batch", "num_patches", "d_output"],
            &x.dims(),
            &[
                ("batch", batch),
                ("num_patches", self.num_patches()),
                ("d_output", self.d_output()),
            ],
        );

        x
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::tensor::TensorData;

    #[test]
    fn test_patch_embed_meta() {
        let config = PatchEmbedConfig {
            input_resolution: [224, 224],
            patch_size: 16,
            d_input: 3,
            d_output: 768,
            enable_patch_norm: true,
        };

        assert_eq!(config.input_resolution(), [224, 224]);
        assert_eq!(config.patch_size(), 16);
        assert_eq!(config.d_input(), 3);
        assert_eq!(config.d_output(), 768);
        assert!(config.enable_patch_norm());
        assert_eq!(config.patches_resolution(), [14, 14]);
        assert_eq!(config.patches_height(), 14);
        assert_eq!(config.patches_width(), 14);
        assert_eq!(config.num_patches(), 196);
        assert_eq!(config.d_output(), 768);
        assert!(config.enable_patch_norm());

        let device = Default::default();
        let patch_embed = config.init::<NdArray>(&device);

        assert_eq!(patch_embed.input_resolution(), [224, 224]);
        assert_eq!(patch_embed.patch_size(), 16);
        assert_eq!(patch_embed.d_input(), 3);
        assert_eq!(patch_embed.d_output(), 768);
        assert!(patch_embed.enable_patch_norm());
        assert_eq!(patch_embed.patches_resolution(), [14, 14]);
        assert_eq!(patch_embed.patches_height(), 14);
        assert_eq!(patch_embed.patches_width(), 14);
        assert_eq!(patch_embed.num_patches(), 196);
        assert_eq!(patch_embed.d_output(), 768);
        assert!(patch_embed.enable_patch_norm());
    }

    #[should_panic(expected = "Input resolution must be divisible by patch size")]
    #[test]
    fn test_patch_embed_invalid_resolution() {
        let config = PatchEmbedConfig {
            input_resolution: [224, 223], // Invalid resolution
            patch_size: 16,
            d_input: 3,
            d_output: 768,
            enable_patch_norm: true,
        };
        let device = Default::default();
        let _d = config.init::<NdArray>(&device);
    }

    #[test]
    fn test_patch_embed_forward() {
        let config = PatchEmbedConfig {
            input_resolution: [224, 224],
            patch_size: 16,
            d_input: 3,
            d_output: 768,
            enable_patch_norm: true,
        };
        let device = Default::default();
        let patch_embed = config.init::<NdArray>(&device);

        let input = Tensor::<NdArray, 4>::from_data(
            TensorData::new(vec![1.0; 3 * 224 * 224], [1, 3, 224, 224]),
            &device,
        );

        let output = patch_embed.forward(input);
        assert_eq!(output.dims(), [1, 196, 768]);
    }

    #[test]
    fn test_patch_embed_without_norm() {
        let config = PatchEmbedConfig {
            input_resolution: [224, 224],
            patch_size: 16,
            d_input: 3,
            d_output: 768,
            enable_patch_norm: false,
        };
        let device = Default::default();
        let patch_embed = config.init::<NdArray>(&device);

        let input = Tensor::<NdArray, 4>::from_data(
            TensorData::new(vec![1.0; 3 * 224 * 224], [1, 3, 224, 224]),
            &device,
        );

        let output = patch_embed.forward(input);
        assert_eq!(output.dims(), [1, 196, 768]);
    }
}
