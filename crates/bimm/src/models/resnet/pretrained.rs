//! # Pretrained `ResNet` Models and Configs

use crate::cache::prefabs::{PreFabConfig, StaticPreFabConfig, StaticPreFabMap};
use crate::cache::weights::{StaticPretrainedWeightsDescriptor, StaticPretrainedWeightsMap};
use crate::models::resnet::{ResNetContractConfig, ResNetStructureConfig};
use alloc::sync::Arc;
use alloc::vec;

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
            builder: || ResNetContractConfig::new(vec![2, 2, 2, 2], 1000),

            weights: Some(&StaticPretrainedWeightsMap {
                items: &[
                    &StaticPretrainedWeightsDescriptor {
                        name: "tv_in1k",
                        description: "TorchVision ResNet-18",
                        license: Some("bsd-3-clause"),
                        origin: Some("https://github.com/pytorch/vision"),
                        urls: &["https://download.pytorch.org/models/resnet18-f37072fd.pth"],
                    },
                    &StaticPretrainedWeightsDescriptor {
                        name: "a1_in1k",
                        description: "RSB Paper ResNet-18 a1",
                        license: None,
                        origin: Some("https://github.com/huggingface/pytorch-image-models"),
                        urls: &[
                            "https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet18_a1_0-d63eafa0.pth",
                        ],
                    },
                    &StaticPretrainedWeightsDescriptor {
                        name: "a2_in1k",
                        description: "RSB Paper ResNet-18 a2",
                        license: None,
                        origin: Some("https://github.com/huggingface/pytorch-image-models"),
                        urls: &[
                            "https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet18_a2_0-b61bd467.pth",
                        ],
                    },
                    &StaticPretrainedWeightsDescriptor {
                        name: "a3_in1k",
                        description: "RSB Paper ResNet-18 a3",
                        license: None,
                        origin: Some("https://github.com/huggingface/pytorch-image-models"),
                        urls: &[
                            "https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet18_a3_0-40c531c8.pth",
                        ],
                    },
                ],
            }),
        },
        &StaticPreFabConfig {
            name: "resnet26",
            description: "ResNet-26 [2, 2, 2, 2] Bottleneck",
            builder: || ResNetContractConfig::new(vec![2, 2, 2, 2], 1000).with_bottleneck(true),

            weights: Some(&StaticPretrainedWeightsMap {
                items: &[&StaticPretrainedWeightsDescriptor {
                    name: "bt_in1k",
                    description: "ResNet-26 pretrained on ImageNet",
                    license: None,
                    origin: None,
                    urls: &[
                        "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet26-9aa10e23.pth",
                    ],
                }],
            }),
        },
        &StaticPreFabConfig {
            name: "resnet34",
            description: "ResNet-34 [3, 4, 6, 3] BasicBlocks",
            builder: || ResNetContractConfig::new(vec![3, 4, 6, 3], 1000),

            weights: Some(&StaticPretrainedWeightsMap {
                items: &[
                    &StaticPretrainedWeightsDescriptor {
                        name: "tv_in1k",
                        description: "TorchVision ResNet-34",
                        license: Some("bsd-3-clause"),
                        origin: Some("https://github.com/pytorch/vision"),
                        urls: &["https://download.pytorch.org/models/resnet34-b627a593.pth"],
                    },
                    &StaticPretrainedWeightsDescriptor {
                        name: "a1_in1k",
                        description: "RSB Paper ResNet-32 a1",
                        license: None,
                        origin: Some(
                            "https://github.com/huggingface/pytorch-image-models/releases",
                        ),
                        urls: &[
                            "https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet34_a1_0-46f8f793.pth",
                        ],
                    },
                    &StaticPretrainedWeightsDescriptor {
                        name: "a2_in1k",
                        description: "RSB Paper ResNet-32 a2",
                        license: None,
                        origin: Some(
                            "https://github.com/huggingface/pytorch-image-models/releases",
                        ),
                        urls: &[
                            "https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet34_a2_0-82d47d71.pth",
                        ],
                    },
                    &StaticPretrainedWeightsDescriptor {
                        name: "a3_in1k",
                        description: "RSB Paper ResNet-32 a3",
                        license: None,
                        origin: Some(
                            "https://github.com/huggingface/pytorch-image-models/releases",
                        ),
                        urls: &[
                            "https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet34_a3_0-a20cabb6.pth",
                        ],
                    },
                    &StaticPretrainedWeightsDescriptor {
                        name: "bt_in1k",
                        description: "ResNet-34 pretrained on ImageNet",
                        license: None,
                        origin: Some(
                            "https://github.com/huggingface/pytorch-image-models/releases",
                        ),
                        urls: &[
                            "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet34-43635321.pth",
                        ],
                    },
                ],
            }),
        },
        &StaticPreFabConfig {
            name: "resnet50",
            description: "ResNet-50 [3, 4, 6, 3] Bottleneck",
            builder: || ResNetContractConfig::new(vec![3, 4, 6, 3], 1000).with_bottleneck(true),

            weights: Some(&StaticPretrainedWeightsMap {
                items: &[&StaticPretrainedWeightsDescriptor {
                    name: "tv_in1k",
                    description: "TorchVision ResNet-50",
                    license: Some("bsd-3-clause"),
                    origin: Some("https://github.com/pytorch/vision"),
                    urls: &["https://download.pytorch.org/models/resnet50-0676ba61.pth"],
                }],
            }),
        },
        /*
        &StaticPreFabConfig {
            name: "resnet50_gn",
            description: "ResNet-50 [3, 4, 6, 3] Bottleneck with GroupNorm",
            builder: || {
                ResNetContractConfig::new(vec![3, 4, 6, 3], 1000)
                    .with_normalization(NormalizationConfig::Group(GroupNormConfig::new(32, 0)))
                    .with_bottleneck(true)
            },

            weights: Some(&StaticPretrainedWeightsMap {
                items: &[&StaticPretrainedWeightsDescriptor {
                    name: "a1h_in1k",
                    description: "ResNet-50 with GroupNorm pretrained on ImageNet",
                    license: None,
                    origin: None,
                    urls: &[
                        "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet50_gn_a1h2-8fe6c4d0.pth",
                    ],
                }],
            }),
        },
         */
        &StaticPreFabConfig {
            name: "resnet101",
            description: "ResNet-101 [3, 4, 23, 3] Bottleneck",
            builder: || ResNetContractConfig::new(vec![3, 4, 23, 3], 1000).with_bottleneck(true),
            weights: Some(&StaticPretrainedWeightsMap {
                items: &[
                    &StaticPretrainedWeightsDescriptor {
                        name: "tv_in1k",
                        description: "TorchVision ResNet-101",
                        license: Some("bsd-3-clause"),
                        origin: Some("https://github.com/pytorch/vision"),
                        urls: &["https://download.pytorch.org/models/resnet101-63fe2227.pth"],
                    },
                    &StaticPretrainedWeightsDescriptor {
                        name: "a1_in1k",
                        description: "ResNet-101 pretrained on ImageNet",
                        license: None,
                        origin: Some(
                            "https://github.com/huggingface/pytorch-image-models/releases",
                        ),
                        urls: &[
                            "https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet101_a1_0-cdcb52a9.pth",
                        ],
                    },
                ],
            }),
        },
        &StaticPreFabConfig {
            name: "resnet152",
            description: "ResNet-152 [3, 8, 36, 3] Bottleneck",
            builder: || ResNetContractConfig::new(vec![3, 8, 36, 3], 1000).with_bottleneck(true),
            weights: Some(&StaticPretrainedWeightsMap {
                items: &[&StaticPretrainedWeightsDescriptor {
                    name: "tv_in1k",
                    description: "TorchVision ResNet-152",
                    license: Some("bsd-3-clause"),
                    origin: Some("https://github.com/pytorch/vision"),
                    urls: &["https://download.pytorch.org/models/resnet152-394f9c45.pth"],
                }],
            }),
        },
    ],
};
