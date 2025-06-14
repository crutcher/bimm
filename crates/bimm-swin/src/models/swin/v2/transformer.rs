use crate::models::swin::v2::block_sequence::{
    StochasticDepthTransformerBlockSequence, StochasticDepthTransformerBlockSequenceConfig,
    StochasticDepthTransformerBlockSequenceMeta,
};
use crate::models::swin::v2::drop_path_rate_table::DropPathRateDepthTable;
use crate::models::swin::v2::patch_embed::{PatchEmbed, PatchEmbedConfig, PatchEmbedMeta};
use crate::models::swin::v2::patch_merge::{PatchMerging, PatchMergingConfig};
use burn::config::Config;
use burn::module::{Module, Param};
use burn::nn::pool::{AdaptiveAvgPool1d, AdaptiveAvgPool1dConfig};
use burn::nn::{
    Dropout, DropoutConfig, Initializer, LayerNorm, LayerNormConfig, Linear, LinearConfig,
};
use burn::prelude::{Backend, Tensor};

#[derive(Config, Debug)]
pub struct LayerConfig {
    pub depth: usize,
    pub num_heads: usize,
}

pub trait SwinTransformerV2Meta {
    /// The input image resolution as [height, width].
    fn input_resolution(&self) -> [usize; 2];

    /// The input height of the image.
    fn input_height(&self) -> usize {
        self.input_resolution()[0]
    }

    /// The input width of the image.
    fn input_width(&self) -> usize {
        self.input_resolution()[1]
    }

    /// The number of input channels.
    fn d_input(&self) -> usize;

    /// The patch size of the input image.
    fn patch_size(&self) -> usize;

    /// Number of classes.
    fn num_classes(&self) -> usize;

    /// The size of the embedding dimension.
    fn d_embed(&self) -> usize;

    /// The window size for window attention.
    fn window_size(&self) -> usize;

    /// Depth of each layer.
    fn layer_configs(&self) -> Vec<LayerConfig>;

    /// Ratio of hidden dimension to input dimension in MLP.
    fn mlp_ratio(&self) -> f64;

    /// Whether to enable QKV bias.
    fn enable_qkv_bias(&self) -> bool;

    /// Dropout rate for MLP.
    fn drop_rate(&self) -> f64;

    /// Dropout rate for attention.
    fn attn_drop_rate(&self) -> f64;

    /// Drop path rate for stochastic depth.
    fn drop_path_rate(&self) -> f64;

    /// Enable APE (Absolute Positional Encoding).
    fn enable_ape(&self) -> bool;

    /// Enable patch normalization?
    fn enable_patch_norm(&self) -> bool;
}

#[derive(Config, Debug)]
pub struct SwinTransformerV2Config {
    /// The input image resolution as [height, width].
    pub input_resolution: [usize; 2],

    /// The patch size of the input image.
    pub patch_size: usize,

    /// The number of input channels.
    pub d_input: usize,

    /// Number of classes.
    pub num_classes: usize,

    /// The size of the embedding dimension.
    pub d_embed: usize,

    /// Depth of each layer.
    pub layer_configs: Vec<LayerConfig>,

    /// The window size for window attention.
    #[config(default = 7)]
    pub window_size: usize,

    /// Ratio of hidden dimension to input dimension in MLP.
    #[config(default = 4.0)]
    pub mlp_ratio: f64,

    /// Whether to enable QKV bias.
    #[config(default = true)]
    pub enable_qkv_bias: bool,

    /// Dropout rate for MLP.
    #[config(default = 0.0)]
    pub drop_rate: f64,

    /// Dropout rate for attention.
    #[config(default = 0.0)]
    pub attn_drop_rate: f64,

    /// Drop path rate for stochastic depth.
    #[config(default = 0.1)]
    pub drop_path_rate: f64,

    /// Enable APE (Absolute Positional Encoding).
    #[config(default = true)]
    pub enable_ape: bool,

    /// Enable patch normalization?
    #[config(default = true)]
    pub enable_patch_norm: bool,
}

impl SwinTransformerV2Meta for SwinTransformerV2Config {
    fn input_resolution(&self) -> [usize; 2] {
        self.input_resolution
    }

    fn patch_size(&self) -> usize {
        self.patch_size
    }

    fn d_input(&self) -> usize {
        self.d_input
    }

    fn num_classes(&self) -> usize {
        self.num_classes
    }

    fn d_embed(&self) -> usize {
        self.d_embed
    }

    fn window_size(&self) -> usize {
        self.window_size
    }

    fn layer_configs(&self) -> Vec<LayerConfig> {
        self.layer_configs.clone()
    }

    fn mlp_ratio(&self) -> f64 {
        self.mlp_ratio
    }

    fn enable_qkv_bias(&self) -> bool {
        self.enable_qkv_bias
    }

    fn drop_rate(&self) -> f64 {
        self.drop_rate
    }

    fn attn_drop_rate(&self) -> f64 {
        self.attn_drop_rate
    }

    fn drop_path_rate(&self) -> f64 {
        self.drop_path_rate
    }

    fn enable_ape(&self) -> bool {
        self.enable_ape
    }

    fn enable_patch_norm(&self) -> bool {
        self.enable_patch_norm
    }
}

#[derive(Debug)]
pub struct SwinTransformerV2Plan {
    pub patch_config: PatchEmbedConfig,

    pub layer_resolutions: Vec<[usize; 2]>,
    pub layer_dims: Vec<usize>,

    pub block_configs: Vec<StochasticDepthTransformerBlockSequenceConfig>,
}

impl SwinTransformerV2Config {
    /// Check config validity and return a plan for the Swin Transformer V2 model.
    ///
    /// Performs model constraint validation tests without initializing a model.
    ///
    /// # Returns
    ///
    /// A [SwinTransformerV2Plan] containing the patch embedding configuration.
    pub fn validate(&self) -> Result<SwinTransformerV2Plan, String> {
        let patch_config = PatchEmbedConfig::new(
            self.input_resolution,
            self.patch_size,
            self.d_input,
            self.d_embed,
        )
        .with_enable_patch_norm(self.enable_patch_norm);

        if self.layer_configs.is_empty() {
            return Err("At least one layer configuration is required".to_string());
        }

        let mut layer_resolutions: Vec<[usize; 2]> = Vec::with_capacity(self.layer_configs.len());
        let mut layer_dims: Vec<usize> = Vec::with_capacity(self.layer_configs.len());

        for layer_i in 0..self.layer_configs.len() {
            let layer_p = 2_usize.pow(layer_i as u32); // Power of 2 for each layer

            layer_resolutions.push([
                patch_config.patches_height() / layer_p,
                patch_config.patches_width() / layer_p,
            ]);
            layer_dims.push(patch_config.d_output() * layer_p);
        }

        let output_resolution = *layer_resolutions.last().unwrap();
        let [last_h, last_w] = output_resolution;
        assert!(
            last_h > 0 && last_w > 0,
            "Output resolution must be non-zero: {:?}",
            output_resolution
        );
        assert!(
            last_h % self.window_size == 0 && last_w % self.window_size == 0,
            "Output resolution must be divisible by window size: {:?} / {:?}",
            output_resolution,
            self.window_size
        );
        let expansion_scale = 2_usize.pow((self.layer_configs.len() - 1) as u32) * self.patch_size;
        assert_eq!(
            patch_config.input_resolution(),
            [last_h * expansion_scale, last_w * expansion_scale],
            "Input resolution must match [<c> * <window_size:{:?}> * 2^(<layers:{:?}>-1) * <patch_size:{:?}, ...]:\n{:?}",
            self.window_size,
            self.layer_configs.len(),
            self.patch_size,
            patch_config.input_resolution(),
        );

        // Stochastic depth delay rule
        let dpr_layer_rates = DropPathRateDepthTable::dpr_layer_rates(
            self.drop_path_rate,
            &self
                .layer_configs
                .iter()
                .map(|c| c.depth)
                .collect::<Vec<usize>>(),
        );

        let block_configs: Vec<StochasticDepthTransformerBlockSequenceConfig> =
            (0..self.layer_configs.len())
                .map(|layer_i| {
                    let cfg = &self.layer_configs[layer_i];

                    let layer_resolution = layer_resolutions[layer_i];
                    let layer_dim = layer_dims[layer_i];

                    StochasticDepthTransformerBlockSequenceConfig::new(
                        // Double the embedding size for each layer
                        layer_dim,
                        layer_resolution,
                        cfg.depth,
                        cfg.num_heads,
                        self.window_size,
                    )
                    .with_mlp_ratio(self.mlp_ratio)
                    .with_enable_qkv_bias(self.enable_qkv_bias)
                    .with_drop_path_rates(Some(dpr_layer_rates[layer_i].clone()))
                })
                .collect();

        Ok(SwinTransformerV2Plan {
            patch_config,
            layer_resolutions,
            layer_dims,
            block_configs,
        })
    }

    /// Initialize a new [SwinTransformerV2](SwinTransformerV2) model.
    #[must_use]
    pub fn init<B: Backend>(
        self,
        device: &B::Device,
    ) -> SwinTransformerV2<B> {
        let plan = self.validate().unwrap();
        // println!("plan: {:#?}", plan);

        let patch_embed: PatchEmbed<B> = plan.patch_config.init(device);

        // ape: trunc_normal: ([1, num_patches, d_embed], std=0.02)
        // defaults: (mean=0.0, a=-2.0, b=2.0)
        let patch_ape: Option<Param<Tensor<B, 3>>> = if self.enable_ape {
            Some(
                Initializer::Normal {
                    mean: 0.0,
                    std: 0.02,
                }
                .init([1_usize, patch_embed.num_patches(), self.d_embed], device),
            )
        } else {
            None
        };

        let grid_transformer_block_sequences: Vec<StochasticDepthTransformerBlockSequence<B>> =
            plan.block_configs
                .iter()
                .map(|config| config.init::<B>(device))
                .collect();

        let grid_merge_layers: Vec<PatchMerging<B>> = (0..grid_transformer_block_sequences.len()
            - 1)
            .map(|layer_i| {
                let block = &grid_transformer_block_sequences[layer_i];
                PatchMergingConfig::new(block.input_resolution(), block.d_input()).init(device)
            })
            .collect();

        let grid_output_features = *plan.layer_dims.last().unwrap();

        SwinTransformerV2 {
            patch_embed,
            patch_ape,
            grid_input_dropout: DropoutConfig::new(self.drop_rate).init(),
            grid_transformer_block_sequences,
            grid_merge_layers,
            grid_output_norm: LayerNormConfig::new(grid_output_features).init(device),
            grid_output_features,
            head_avgpool: AdaptiveAvgPool1dConfig::new(1).init(),
            head: LinearConfig::new(grid_output_features, self.num_classes).init(device),
        }
    }
}

#[derive(Module, Debug)]
pub struct SwinTransformerV2<B: Backend> {
    pub patch_embed: PatchEmbed<B>,
    pub patch_ape: Option<Param<Tensor<B, 3>>>,
    pub grid_input_dropout: Dropout,
    pub grid_transformer_block_sequences: Vec<StochasticDepthTransformerBlockSequence<B>>,
    pub grid_merge_layers: Vec<PatchMerging<B>>,
    pub grid_output_norm: LayerNorm<B>,
    pub grid_output_features: usize,
    pub head_avgpool: AdaptiveAvgPool1d,
    pub head: Linear<B>,
}

impl<B: Backend> SwinTransformerV2<B> {
    /// Apply patch embedding and absolute positional encoding (APE) to the input image tensor.
    #[inline(always)]
    #[must_use]
    fn apply_patching(
        &self,
        input: Tensor<B, 4>,
    ) -> Tensor<B, 3> {
        let x = self.patch_embed.forward(input);

        match self.patch_ape {
            Some(ref ape) => x + ape.val(),
            None => x,
        }
    }

    /// Applies the layer stack to a patch tensor.
    #[inline(always)]
    #[must_use]
    fn apply_stack(
        &self,
        input: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        let mut x = self.grid_input_dropout.forward(input);

        for layer_i in 0..self.grid_transformer_block_sequences.len() {
            if layer_i > 0 {
                x = self.grid_merge_layers[layer_i - 1].forward(x);
            }

            x = self.grid_transformer_block_sequences[layer_i].forward(x);
        }
        // B L C

        self.grid_output_norm.forward(x)
        // B L C
    }

    /// Aggregates the grid into a single vector per batch.
    #[inline(always)]
    #[must_use]
    fn aggregate_grid(
        &self,
        input: Tensor<B, 3>,
    ) -> Tensor<B, 2> {
        // input: B L C
        let x = input.swap_dims(1, 2);
        let x = self.head_avgpool.forward(x);
        // B C 1
        x.squeeze(2)
        // B C
    }

    /// Applies the final classification head to the transformed output.
    #[inline(always)]
    #[must_use]
    fn apply_head(
        &self,
        input: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        self.head.forward(input)
    }

    /// Applies the model to the input image tensor and returns the classification logits.
    ///
    /// # Arguments
    ///
    /// * `input`: A 4D tensor representing the input image with shape `[B, C, H, W]`,
    ///
    /// # Returns
    ///
    /// A 2D tensor of shape `[B, num_classes]` representing the classification logits.
    #[must_use]
    pub fn forward(
        &self,
        input: Tensor<B, 4>,
    ) -> Tensor<B, 2> {
        let x = self.apply_patching(input);
        let x = self.apply_stack(x);
        let x = self.aggregate_grid(x);
        self.apply_head(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::tensor::Distribution;

    #[test]
    fn test_swin_transformer_v2_config() {
        let config = SwinTransformerV2Config {
            input_resolution: [224, 224],
            patch_size: 4,
            d_input: 3,
            num_classes: 1000,
            d_embed: 96,
            layer_configs: vec![
                LayerConfig {
                    depth: 2,
                    num_heads: 3,
                },
                LayerConfig {
                    depth: 2,
                    num_heads: 6,
                },
                LayerConfig {
                    depth: 18,
                    num_heads: 12,
                },
            ],
            window_size: 7,
            mlp_ratio: 4.0,
            enable_qkv_bias: true,
            drop_rate: 0.0,
            attn_drop_rate: 0.0,
            drop_path_rate: 0.1,
            enable_ape: true,
            enable_patch_norm: true,
        };

        assert_eq!(config.input_resolution, [224, 224]);
        assert_eq!(config.patch_size, 4);
        assert_eq!(config.d_input, 3);
        assert_eq!(config.num_classes, 1000);
        assert_eq!(config.d_embed, 96);
        assert_eq!(config.window_size, 7);
        assert_eq!(config.mlp_ratio, 4.0);
        assert!(config.enable_qkv_bias);
        assert_eq!(config.drop_rate, 0.0);
        assert_eq!(config.attn_drop_rate, 0.0);
        assert_eq!(config.drop_path_rate, 0.1);
        assert!(config.enable_ape);
        assert!(config.enable_patch_norm);
    }

    #[test]
    fn test_smoke_test() {
        smoke_test_impl::<NdArray>();
    }
    fn smoke_test_impl<B: Backend>() {
        let device = Default::default();

        let b = 2;
        let d_input = 3;

        let layer_configs = vec![
            LayerConfig {
                depth: 2,
                num_heads: 3,
            },
            LayerConfig {
                depth: 2,
                num_heads: 6,
            },
        ];

        let patch_size = 4;
        let window_size = 3;

        let last_wh = 2;
        let last_ww = 2;
        let last_h = last_wh * window_size;
        let last_w = last_ww * window_size;

        let merge_steps = (layer_configs.len() - 1) as u32;
        let expansion_scale = 2_usize.pow(merge_steps);
        let h = last_h * expansion_scale * patch_size;
        let w = last_w * expansion_scale * patch_size;

        let num_classes = 12;

        let d_embed = (d_input * patch_size * patch_size) / 2;

        let model = SwinTransformerV2Config::new(
            [h, w],
            patch_size,
            d_input,
            num_classes,
            d_embed,
            layer_configs,
        )
        .with_window_size(window_size)
        .init(&device);

        let distribution = Distribution::Normal(0.0, 0.02);
        let input = Tensor::<B, 4>::random([b, d_input, h, w], distribution, &device);

        let output = model.forward(input.clone());
        assert_eq!(output.dims(), [b, num_classes]);

        output.to_data().assert_eq(
            &{
                let patched = model.apply_patching(input.clone());
                assert_eq!(
                    patched.dims(),
                    [b, h * w / (patch_size * patch_size), d_embed]
                );

                let stacked = model.apply_stack(patched);
                assert_eq!(
                    stacked.dims(),
                    [b, last_h * last_w, model.grid_output_features]
                );

                let aggregated = model.aggregate_grid(stacked);
                assert_eq!(aggregated.dims(), [b, model.grid_output_features]);

                let classed = model.apply_head(aggregated);
                assert_eq!(classed.dims(), [b, num_classes]);

                classed
            }
            .to_data(),
            true,
        );
    }
}
