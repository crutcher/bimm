//! # `ResNet` Core Model
//!
//! [`ResNet`] is the core `ResNet` module.
//!
//! [`ResNetContractConfig`] implements [`Config`], and provides
//! a high-level configuration interface.
//! It provides [`ResNetContractConfig::to_structure`] to convert
//! to a [`ResNetStructureConfig`].
//!
//! [`ResNetStructureConfig`] implements [`Config`], and provides
//! [`ResNetStructureConfig::init`] to initialize a [`ResNet`].
//!
//! [`ResNet`] implements [`Module`], and provides
//! [`ResNet::forward`].

use crate::compat::activation_wrapper::{Activation, ActivationConfig};
use crate::compat::normalization_wrapper::NormalizationConfig;
use crate::layers::blocks::conv_norm::{ConvNorm2d, ConvNorm2dConfig};
use crate::layers::drop::drop_block::DropBlockOptions;
use crate::models::resnet::layer_block::{
    LayerBlock, LayerBlockContractConfig, LayerBlockMeta, LayerBlockStructureConfig,
};
use crate::models::resnet::residual_block::{ResidualBlock, ResidualBlockStructureConfig};
use crate::models::resnet::resnet_io::pytorch_stubs::load_resnet_stub_record;
use crate::models::resnet::util::CONV_INTO_RELU_INITIALIZER;
use crate::utility::probability::expect_probability;
use burn::module::Module;
use burn::nn::BatchNormConfig;
use burn::nn::conv::Conv2dConfig;
use burn::nn::pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig, MaxPool2d, MaxPool2dConfig};
use burn::nn::{Initializer, Linear, LinearConfig, PaddingConfig2d};
use burn::prelude::{Backend, Config, Tensor};
use std::path::PathBuf;

/// ResNet-18 block depths.
pub const RESNET18_BLOCKS: [usize; 4] = [2, 2, 2, 2];
/// ResNet-34 block depths.
pub const RESNET34_BLOCKS: [usize; 4] = [3, 4, 6, 3];
/// ResNet-50 block depths.
pub const RESNET50_BLOCKS: [usize; 4] = [3, 4, 6, 3];
/// ResNet-101 block depths.
pub const RESNET101_BLOCKS: [usize; 4] = [3, 4, 23, 3];
/// ResNet-152 block depths.
pub const RESNET152_BLOCKS: [usize; 4] = [3, 8, 36, 3];

/// High-level [`ResNet`] model configuration.
#[derive(Config, Debug)]
pub struct ResNetContractConfig {
    /// Layer block depths.
    pub layers: [usize; 4],

    /// Number of classification classes.
    pub num_classes: usize,

    /// Number of channels in stem convolutions.
    /// TODO: Replace with a ``ResNetStem`` module.
    #[config(default = "64")]
    pub stem_width: usize,

    /// Model feature expansion rate.
    #[config(default = "1")]
    pub expansion: usize,

    /// Use bottleneck blocks.
    #[config(default = "false")]
    pub bottleneck: bool,

    /// [`crate::compat::normalization_wrapper::Normalization`] config.
    ///
    /// The feature size of this config will be replaced
    /// with the appropriate feature size for the input layer.
    #[config(default = "NormalizationConfig::Batch(BatchNormConfig::new(0))")]
    pub normalization: NormalizationConfig,

    /// [`crate::compat::activation_wrapper::Activation`] config.
    #[config(default = "ActivationConfig::Relu")]
    pub activation: ActivationConfig,
}

impl From<ResNetContractConfig> for ResNetStructureConfig {
    fn from(config: ResNetContractConfig) -> Self {
        assert!(
            config.expansion == 1 || config.expansion == 4,
            "ResNet module only supports expansion values [1, 4] for residual blocks"
        );
        let expansion = config.expansion;

        let make_block = |idx: usize, in_factor: usize, out_factor: usize, down: bool| {
            LayerBlockContractConfig::new(config.layers[idx], 64 * in_factor, 64 * out_factor)
                .with_downsample(down)
                .with_bottleneck(config.bottleneck)
                .with_normalization(config.normalization.clone())
                .with_activation(config.activation.clone())
                .into()
        };

        ResNetStructureConfig::new(
            ConvNorm2dConfig::from(
                Conv2dConfig::new([3, config.stem_width], [7, 7])
                    .with_stride([2, 2])
                    .with_padding(PaddingConfig2d::Explicit(3, 3))
                    .with_bias(false),
            )
            .with_initializer(CONV_INTO_RELU_INITIALIZER.clone()),
            vec![
                make_block(0, 1, expansion, false),
                make_block(1, expansion, 2 * expansion, true),
                make_block(2, 2 * expansion, 4 * expansion, true),
                make_block(3, 4 * expansion, 8 * expansion, true),
            ],
            config.num_classes,
        )
    }
}

impl ResNetContractConfig {
    /// Convert to a [`ResNetStructureConfig`].
    pub fn to_structure(self) -> ResNetStructureConfig {
        self.into()
    }

    /// Create a ResNet-18 model.
    pub fn resnet18(num_classes: usize) -> Self {
        Self::new(RESNET18_BLOCKS, num_classes) // .with_bottleneck(true)
    }
}

/// [`ResNet`] Structure Config.
///
/// This config defines the structure of a converted [`ResNet`] model.
/// It is not a semantic configuration and does not check the validity
/// of the internal sizes before or during construction.
#[derive(Config, Debug)]
pub struct ResNetStructureConfig {
    /// The input Conv/Norm block configuration.
    pub input_conv_norm: ConvNorm2dConfig,

    /// Optional override for the input Conv2d initializer.
    #[config(default = "CONV_INTO_RELU_INITIALIZER.clone().into()")]
    pub input_conv_norm_initializer: Option<Initializer>,

    /// The input activation configuration.
    #[config(default = "ActivationConfig::Relu")]
    pub input_act: ActivationConfig,

    /// The inner layers configuration.
    pub layers: Vec<LayerBlockStructureConfig>,

    /// The number of classes.
    pub num_classes: usize,
}

impl ResNetStructureConfig {
    /// Initialize a [`ResNet`] model.
    pub fn init<B: Backend>(
        self,
        device: &B::Device,
    ) -> ResNet<B> {
        let mut input_conv_norm = self.input_conv_norm.clone();
        if self.input_conv_norm_initializer.is_some() {
            input_conv_norm.conv = input_conv_norm
                .conv
                .with_initializer(self.input_conv_norm_initializer.unwrap());
        }

        let head_planes = self.layers.last().unwrap().out_planes();

        ResNet {
            input_conv_norm: input_conv_norm.init(device),
            input_act: self.input_act.init(device),
            input_pool: MaxPool2dConfig::new([3, 3])
                .with_strides([2, 2])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .init(),

            layers: self
                .layers
                .into_iter()
                .map(|c| c.init(device))
                .collect::<Vec<_>>(),

            output_pool: AdaptiveAvgPool2dConfig::new([1, 1]).init(),
            output_fc: LinearConfig::new(head_planes, self.num_classes).init(device),
        }
    }

    /// Apply the given standard drop block probability scheme.
    pub fn with_standard_drop_block_prob(
        self,
        drop_prob: f64,
    ) -> Self {
        let drop_prob = expect_probability(drop_prob);
        let k = self.layers.len();
        let mut blocks = vec![None; k];
        if drop_prob > 0.0 {
            blocks[k - 2] = DropBlockOptions::default()
                .with_drop_prob(drop_prob)
                .with_block_size(5)
                .with_gamma_scale(0.25)
                .into();
            blocks[k - 1] = DropBlockOptions::default()
                .with_drop_prob(drop_prob)
                .with_block_size(3)
                .with_gamma_scale(1.0)
                .into();
        }
        self.with_drop_block_options(blocks)
    }

    /// Update the config with stochastic depth.
    pub fn with_stochastic_depth_drop_path_rate(
        self,
        drop_path_rate: f64,
    ) -> Self {
        let drop_path_rate = expect_probability(drop_path_rate);

        let net_num_blocks = self.layers.iter().map(|b| b.len()).sum::<usize>();
        let mut net_block_idx = 0;
        let mut update_drop_path = |_idx: usize, block: ResidualBlockStructureConfig| {
            // stochastic depth linear decay rule
            let block_dpr = drop_path_rate * (net_block_idx as f64) / ((net_num_blocks - 1) as f64);
            net_block_idx += 1;
            if block_dpr > 0.0 {
                block.with_drop_path_prob(block_dpr)
            } else {
                block
            }
        };

        Self {
            layers: self
                .layers
                .into_iter()
                .map(|b| b.map_blocks(&mut update_drop_path))
                .collect(),
            ..self
        }
    }

    /// Update the config with the given drop block options.
    ///
    /// # Arguments
    ///
    /// - `options`: a vector of options, one for each layer.
    pub fn with_drop_block_options(
        self,
        options: Vec<Option<DropBlockOptions>>,
    ) -> Self {
        assert_eq!(options.len(), self.layers.len());
        Self {
            layers: self
                .layers
                .into_iter()
                .zip(options)
                .map(|(b, o)| b.with_drop_block(o))
                .collect(),
            ..self
        }
    }
}

/// `ResNet` model.
#[derive(Module, Debug)]
pub struct ResNet<B: Backend> {
    /// Input conv/norm.
    pub input_conv_norm: ConvNorm2d<B>,
    /// Input activation.
    pub input_act: Activation<B>,
    /// Input pool.
    pub input_pool: MaxPool2d,

    /// Layers.
    pub layers: Vec<LayerBlock<B>>,

    /// Head pooling.
    pub output_pool: AdaptiveAvgPool2d,
    /// Head classifier.
    pub output_fc: Linear<B>,
}

impl<B: Backend> ResNet<B> {
    /// Forward pass.
    pub fn forward(
        &self,
        input: Tensor<B, 4>,
    ) -> Tensor<B, 2> {
        // Prep block
        let x = self.input_conv_norm.forward(input);
        let x = self.input_act.forward(x);
        let x = self.input_pool.forward(x);

        // Residual blocks
        let x = self.layers.iter().fold(x, |x, layer| layer.forward(x));

        // Head
        let x = self.output_pool.forward(x);
        // Reshape [B, C, 1, 1] -> [B, C]
        let x = x.flatten(1, 3);
        self.output_fc.forward(x)
    }

    /// Load weights from a `PyTorch` weights path.
    pub fn load_pytorch_weights(
        self,
        path: PathBuf,
    ) -> anyhow::Result<Self> {
        let device = &self.devices()[0];
        let record = load_resnet_stub_record::<B>(path, device)?;
        let resnet = self.with_classes(record.fc.weight.dims()[0]);
        Ok(record.cna_copy_weights(resnet))
    }

    /// Re-initialize the last layer with the specified number of output classes.
    pub fn with_classes(
        mut self,
        num_classes: usize,
    ) -> Self {
        let [d_input, _d_output] = self.output_fc.weight.dims();
        self.output_fc =
            LinearConfig::new(d_input, num_classes).init(&self.output_fc.weight.device());
        self
    }

    /// Update the config with stochastic depth.
    pub fn with_stochastic_path_depth(
        self,
        drop_path_rate: f64,
    ) -> Self {
        let drop_path_rate = expect_probability(drop_path_rate);

        let net_num_blocks = self.layers.iter().map(|b| b.len()).sum::<usize>();
        let mut net_block_idx = 0;
        let mut update_drop_path = |_idx: usize, block: ResidualBlock<B>| {
            // stochastic depth linear decay rule
            let block_dpr = drop_path_rate * (net_block_idx as f64) / ((net_num_blocks - 1) as f64);
            net_block_idx += 1;
            if block_dpr > 0.0 {
                block.with_drop_path_prob(block_dpr)
            } else {
                block
            }
        };

        Self {
            layers: self
                .layers
                .into_iter()
                .map(|b| b.map_blocks(&mut update_drop_path))
                .collect(),
            ..self
        }
    }

    /// Update the config with the given drop block options.
    ///
    /// # Arguments
    ///
    /// - `options`: a vector of options, one for each layer.
    pub fn with_drop_block_options(
        self,
        options: Vec<Option<DropBlockOptions>>,
    ) -> Self {
        assert_eq!(options.len(), self.layers.len());
        Self {
            layers: self
                .layers
                .into_iter()
                .zip(options)
                .map(|(b, o)| b.with_drop_block(o))
                .collect(),
            ..self
        }
    }

    /// Apply the given standard drop block probability scheme.
    pub fn with_stochastic_drop_block(
        self,
        drop_prob: f64,
    ) -> Self {
        let drop_prob = expect_probability(drop_prob);
        let k = self.layers.len();
        let mut blocks = vec![None; k];
        if drop_prob > 0.0 {
            blocks[k - 2] = DropBlockOptions::default()
                .with_drop_prob(drop_prob)
                .with_block_size(5)
                .with_gamma_scale(0.25)
                .into();
            blocks[k - 1] = DropBlockOptions::default()
                .with_drop_prob(drop_prob)
                .with_block_size(3)
                .with_gamma_scale(1.0)
                .into();
        }
        self.with_drop_block_options(blocks)
    }

    /// Apply a mapping over layers.
    pub fn map_layers<F>(
        self,
        f: F,
    ) -> Self
    where
        F: Fn(Vec<LayerBlock<B>>) -> Vec<LayerBlock<B>>,
    {
        Self {
            layers: f(self.layers),
            ..self
        }
    }
}
