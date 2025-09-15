//! # Pretrained `ResNet` Models and Configs

use crate::cache::prefabs::{PreFabConfig, StaticPreFabConfig, StaticPreFabMap};
use crate::cache::weights::{StaticPretrainedWeightsDescriptor, StaticPretrainedWeightsMap};
use crate::models::resnet::{ResNetContractConfig, ResNetStructureConfig};
use std::sync::Arc;

impl PreFabConfig<ResNetContractConfig> {
    /// Convert to a prefab for [`ResNetStructureConfig`].
    pub fn to_structure_prefab(&self) -> PreFabConfig<ResNetStructureConfig> {
        let builder = self.builder.clone();
        PreFabConfig {
            name: self.name.clone(),
            description: self.description.clone(),
            builder: Arc::new(move || builder().to_structure()),
            weights: self.weights.clone(),
        }
    }
}

impl From<&StaticPreFabConfig<ResNetContractConfig>> for PreFabConfig<ResNetStructureConfig> {
    fn from(config: &StaticPreFabConfig<ResNetContractConfig>) -> Self {
        config.to_prefab().to_structure_prefab()
    }
}

impl From<&PreFabConfig<ResNetContractConfig>> for PreFabConfig<ResNetStructureConfig> {
    fn from(config: &PreFabConfig<ResNetContractConfig>) -> Self {
        config.to_structure_prefab()
    }
}
/// Pretrained [`super::ResNet`] configs and weights.
pub static PREFAB_RESNET_MAP: StaticPreFabMap<ResNetContractConfig> = StaticPreFabMap {
    name: "resnet",
    description: "Well-Know ResNet configs",

    items: &[
        &StaticPreFabConfig {
            name: "resnet18",
            description: "ResNet-18 [2, 2, 2, 2] BasicBlocks",
            builder: || ResNetContractConfig::new([2, 2, 2, 2], 1000),

            weights: Some(&StaticPretrainedWeightsMap {
                items: &[
                    &StaticPretrainedWeightsDescriptor {
                        name: "tv_in1k",
                        description: "ResNet18 pretrained on ImageNet",
                        license: Some("bsd-3-clause"),
                        origin: Some("https://github.com/pytorch/vision"),
                        urls: &["https://download.pytorch.org/models/resnet18-f37072fd.pth"],
                    },
                    /*
                    DeserializeError("Candle Tensor error: invalid Zip archive: Could not find central directory end")
                    &StaticPretrainedWeightsDescriptor {
                        name: "a1_in1k",
                        description: "ResNet18 pretrained on ImageNet",
                        license: None,
                        origin: Some("https://github.com/huggingface/pytorch-image-models"),
                        urls: &[
                            "https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet18_a1_0-d63eafa0.pth",
                        ],
                    },
                    */
                ],
            }),
        },
        &StaticPreFabConfig {
            name: "resnet34",
            description: "ResNet-34 [3, 4, 6, 3] BasicBlocks",
            builder: || ResNetContractConfig::new([3, 4, 6, 3], 1000),

            weights: Some(&StaticPretrainedWeightsMap {
                items: &[&StaticPretrainedWeightsDescriptor {
                    name: "tv_in1k",
                    description: "ResNet-34 pretrained on ImageNet",
                    license: Some("bsd-3-clause"),
                    origin: Some("https://github.com/pytorch/vision"),
                    urls: &["https://download.pytorch.org/models/resnet34-b627a593.pth"],
                }],
            }),
        },
        /*
        FIXME: The loaded weights have a downsample that the config does not have.
        DeserializeError("Candle Tensor error: invalid Zip archive: Could not find central directory end")
        &StaticPreFabConfig {
            name: "resnet50",
            description: "ResNet-50 [3, 4, 6, 3] Bottleneck",
            builder: || ResNetContractConfig::new([3, 4, 6, 3], 1000).with_bottleneck(true),

            weights: Some(&StaticPretrainedWeightsMap {
                items: &[
                    &StaticPretrainedWeightsDescriptor {
                        name: "tv_in1k",
                        description: "ResNet-50 pretrained on ImageNet",
                        license: Some("bsd-3-clause"),
                        origin: Some("https://github.com/pytorch/vision"),
                        urls: &["https://download.pytorch.org/models/resnet50-0676ba61.pth"],
                    },
                    &StaticPretrainedWeightsDescriptor {
                        name: "tv_in2k",
                        description: "ResNet-50 pretrained on ImageNet",
                        license: Some("bsd-3-clause"),
                        origin: Some("https://github.com/pytorch/vision"),
                        urls: &["https://download.pytorch.org/models/resnet50-11ad3fa6.pth"],
                    },
                ],
            }),
        },
         */
    ],
};
