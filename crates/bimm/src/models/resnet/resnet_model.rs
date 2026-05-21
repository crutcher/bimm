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

use alloc::{
    vec,
    vec::Vec,
};
use std::path::PathBuf;

use bunsen::{
    blocks::images::{
        conv::conv_norm::{
            ConvNorm2d,
            ConvNorm2dConfig,
        },
        drop::drop_block::DropBlockOptions,
    },
    support::validators::expect_probability,
};
use burn::{
    module::Module,
    nn::{
        BatchNorm,
        BatchNormConfig,
        Initializer,
        Linear,
        LinearConfig,
        PaddingConfig2d,
        activation::{
            Activation,
            ActivationConfig,
        },
        conv::Conv2dConfig,
        norm::{
            Normalization,
            NormalizationConfig,
        },
        pool::{
            AdaptiveAvgPool2d,
            AdaptiveAvgPool2dConfig,
            MaxPool2d,
            MaxPool2dConfig,
        },
    },
    prelude::{
        Backend,
        Config,
        Tensor,
    },
};

use super::{
    bottleneck_block::BottleneckPolicyConfig,
    layer_block::{
        LayerBlock,
        LayerBlockContractConfig,
        LayerBlockMeta,
        LayerBlockStructureConfig,
    },
    residual_block::{
        ResidualBlock,
        ResidualBlockStructureConfig,
    },
    util::CONV_INTO_RELU_INITIALIZER,
};
use crate::models::resnet::resnet_io::pytorch_stubs::{
    BasicBlockStub,
    BottleneckStub,
    DownsampleStub,
    LayerBlockStub,
    ResNetStub,
    ResidualBlockStub,
    load_resnet_stub,
};

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
    /// Must have the same length as `channels`.
    pub layers: Vec<usize>,

    /// Number of classification classes.
    pub num_classes: usize,

    /// Number of channels in stem convolutions.
    /// TODO: Replace with a ``ResNetStem`` module.
    #[config(default = "64")]
    pub stem_width: usize,

    /// Output stride.
    #[config(default = "32")]
    pub output_stride: usize,

    /// Select between [`BasicBlock`] and [`BottleneckBlock`].
    #[config(default = "None")]
    pub bottleneck_policy: Option<BottleneckPolicyConfig>,

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

impl ResNetContractConfig {
    /// Enable default bottleneck policy.
    pub fn with_bottleneck(
        self,
        enable: bool,
    ) -> Self {
        let policy = if enable {
            Some(Default::default())
        } else {
            None
        };
        self.with_bottleneck_policy(policy)
    }

    /// Build the [`LayerBlockContractConfig`] stack.
    #[allow(unused)]
    pub fn to_layer_contracts(&self) -> Vec<LayerBlockContractConfig> {
        let mut net_stride = 4;
        let mut dilation = 1;
        let mut prev_dilation = 1;
        let mut layers: Vec<LayerBlockContractConfig> = Default::default();
        let mut in_planes = self.stem_width;
        for (stage_idx, &num_blocks) in self.layers.iter().enumerate() {
            let downsample_input = {
                let mut stride = if stage_idx == 0 { 1 } else { 2 };
                if net_stride >= self.output_stride {
                    dilation *= stride;
                    stride = 1;
                } else {
                    net_stride *= stride;
                }
                stride != 1
            };

            let first_dilation = prev_dilation;

            let out_planes = if stage_idx == 0 {
                match &self.bottleneck_policy {
                    Some(policy) => in_planes * policy.pinch_factor,
                    None => in_planes,
                }
            } else {
                2 * in_planes
            };

            layers.push(
                LayerBlockContractConfig::new(num_blocks, in_planes, out_planes)
                    .with_downsample_input(downsample_input)
                    .with_first_dilation(Some(first_dilation))
                    .with_dilation(dilation)
                    .with_bottleneck_policy(self.bottleneck_policy.clone())
                    .with_normalization(self.normalization.clone())
                    .with_activation(self.activation.clone()),
            );

            in_planes = out_planes;
            prev_dilation = dilation;
        }

        layers
    }

    /// Convert to a [`ResNetStructureConfig`].
    pub fn to_structure(self) -> ResNetStructureConfig {
        ResNetStructureConfig::new(
            ConvNorm2dConfig::from(
                Conv2dConfig::new([3, self.stem_width], [7, 7])
                    .with_stride([2, 2])
                    .with_padding({
                        let d = 3;
                        PaddingConfig2d::Explicit(d, d, d, d)
                    })
                    .with_bias(false),
            )
            .with_initializer(CONV_INTO_RELU_INITIALIZER.clone()),
            self.to_layer_contracts()
                .into_iter()
                .map(|c| c.into())
                .collect::<Vec<_>>(),
            self.num_classes,
        )
    }

    /// Create a ResNet-18 model.
    pub fn resnet18(num_classes: usize) -> Self {
        Self::new(RESNET18_BLOCKS.to_vec(), num_classes) // .with_bottleneck(true)
    }
}

impl From<ResNetContractConfig> for ResNetStructureConfig {
    #[allow(unused)]
    fn from(config: ResNetContractConfig) -> Self {
        config.to_structure()
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
        if let Some(initializer) = &self.input_conv_norm_initializer {
            input_conv_norm.conv = input_conv_norm.conv.with_initializer(initializer.clone());
        }

        let head_planes = self.layers.last().unwrap().out_planes();

        ResNet {
            input_conv_norm: input_conv_norm.init(device),
            input_act: self.input_act.init(device),
            input_pool: MaxPool2dConfig::new([3, 3])
                .with_strides([2, 2])
                .with_padding({
                    let d = 1;
                    PaddingConfig2d::Explicit(d, d, d, d)
                })
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

        let net_num_blocks = self.layers.iter().map(|b| b.len()).sum::<usize>() - self.layers.len();
        let mut net_block_idx = 0;
        let mut update_drop_path = |idx: usize, block: ResidualBlockStructureConfig| {
            // stochastic depth linear decay rule
            let block_dpr = drop_path_rate * (net_block_idx as f64) / ((net_num_blocks - 1) as f64);
            net_block_idx += 1;
            if idx != 0 && block_dpr > 0.0 {
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
    /// Debug Printout.
    pub fn debug_print(&self) {
        for (idx, layer) in self.layers.iter().enumerate() {
            println!(
                "# Stage[{idx}]/{}:: {} :> {}",
                layer.len(),
                layer.in_planes(),
                layer.out_planes()
            );
            layer.debug_print();
            println!();
        }
    }

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
        let mut x = x;
        for layer in self.layers.iter() {
            x = layer.forward(x);
        }

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
        fn expect_batch_norm<B: Backend>(norm: &Normalization<B>) -> anyhow::Result<&BatchNorm<B>> {
            match norm {
                Normalization::Batch(bn) => Ok(bn),
                _ => Err(anyhow::anyhow!("Expected BatchNorm")),
            }
        }

        let mut stub: ResNetStub<B> = ResNetStub {
            conv1: self.input_conv_norm.conv.clone(),
            bn1: self.input_conv_norm.norm.clone(),
            layers: self
                .layers
                .iter()
                .map(|layer| LayerBlockStub {
                    blocks: layer
                        .blocks
                        .iter()
                        .map(|block| {
                            use ResidualBlock as T;
                            use ResidualBlockStub as S;

                            match block {
                                T::Basic(block) => S::Basic(BasicBlockStub {
                                    conv1: block.cna1.conv.clone(),
                                    bn1: expect_batch_norm(&block.cna1.norm).unwrap().clone(),
                                    conv2: block.cna2.conv.clone(),
                                    bn2: expect_batch_norm(&block.cna2.norm).unwrap().clone(),
                                    downsample: block.downsample.as_ref().map(|d| DownsampleStub {
                                        conv: d.conv.clone(),
                                        bn: expect_batch_norm(&d.norm).unwrap().clone(),
                                    }),
                                }),
                                T::Bottleneck(block) => S::Bottleneck(BottleneckStub {
                                    conv1: block.cna1.conv.clone(),
                                    bn1: expect_batch_norm(&block.cna1.norm).unwrap().clone(),
                                    conv2: block.cna2.conv.clone(),
                                    bn2: expect_batch_norm(&block.cna2.norm).unwrap().clone(),
                                    conv3: block.cna3.conv.clone(),
                                    bn3: expect_batch_norm(&block.cna3.norm).unwrap().clone(),
                                    downsample: block.downsample.as_ref().map(|d| DownsampleStub {
                                        conv: d.conv.clone(),
                                        bn: expect_batch_norm(&d.norm).unwrap().clone(),
                                    }),
                                }),
                            }
                        })
                        .collect::<Vec<_>>(),
                })
                .collect::<Vec<_>>(),
            fc: self.output_fc.clone(),
        };

        load_resnet_stub(path, &mut stub)?;
        let adapted_target = self.with_classes(stub.fc.weight.dims()[0]);
        Ok(stub.copy_stub_weights(adapted_target))
    }

    /// Re-initialize the last layer with the specified number of output
    /// classes.
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

    /// Freeze the layers.
    pub fn freeze_layers(self) -> Self {
        self.map_layers(|layers| layers.into_iter().map(|layer| layer.no_grad()).collect())
    }
}

#[cfg(test)]
mod tests {
    use bunsen::support::testing::PerfTestBackend;

    use super::*;

    #[test]
    fn test_to_layers_34_basic() {
        let cfg = ResNetContractConfig::new(RESNET34_BLOCKS.to_vec(), 1000);

        let layers = cfg.to_layer_contracts();

        println!("{:#?}", layers);

        // assert!(false);
    }

    #[test]
    fn test_to_layers_50_bottleneck() {
        type B = PerfTestBackend;
        let device = Default::default();

        let cfg = ResNetContractConfig::new(RESNET50_BLOCKS.to_vec(), 1000).with_bottleneck(true);
        let layers = cfg.to_layer_contracts();

        let first_stage = layers[0].clone();
        println!("block[0] cfg:\n{:#?}", first_stage);
        println!();

        let blocks = first_stage
            .to_block_contracts()
            .into_iter()
            .map(|b| b.to_structure())
            .collect::<Vec<_>>();
        println!("blocks ...");
        println!("{:#?}", blocks);
        println!();

        let model: ResNet<B> = cfg.to_structure().init(&device);

        model.debug_print();

        // assert!(false);
    }
}
