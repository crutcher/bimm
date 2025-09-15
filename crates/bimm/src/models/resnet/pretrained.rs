//! # Pretrained `ResNet` Models and Configs

use crate::cache::prefabs::{PreFabConfig, PreFabMap, StaticPreFabConfig, StaticPreFabMap};
use crate::cache::weights::{StaticPretrainedWeightsDescriptor, StaticPretrainedWeightsMap};
use crate::models::resnet::{RESNET18_BLOCKS, ResNetContractConfig, ResNetStructureConfig};
use std::sync::Arc;

/// Static builder for [`ResNetPreFabContractConfig`].
pub type StaticResNetPreFabContractConfig = StaticPreFabConfig<ResNetContractConfig>;

/// A [`ResNetContractConfig`] Well-Known Pre-Fab.
pub type ResNetPreFabContractConfig = PreFabConfig<ResNetContractConfig>;

/// Static builder for [`ResNetPreFabContractConfig`].
pub type StaticResNetPreFabStructureConfig = StaticPreFabConfig<ResNetStructureConfig>;

/// A [`ResNetStructureConfig`] Well-Known Pre-Fab.
pub type ResNetPreFabStructureConfig = PreFabConfig<ResNetStructureConfig>;

impl From<&StaticResNetPreFabContractConfig> for ResNetPreFabStructureConfig {
    fn from(config: &StaticResNetPreFabContractConfig) -> Self {
        config.to_prefab().to_structure_prefab()
    }
}

impl ResNetPreFabContractConfig {
    /// Convert to a [`ResNetPreFabStructureConfig`].
    pub fn to_structure_prefab(&self) -> ResNetPreFabStructureConfig {
        let builder = self.builder.clone();
        ResNetPreFabStructureConfig {
            name: self.name.clone(),
            description: self.description.clone(),
            builder: Arc::new(move || builder().to_structure()),
            weights: self.weights.clone(),
        }
    }
}

impl From<&ResNetPreFabContractConfig> for ResNetPreFabStructureConfig {
    fn from(config: &ResNetPreFabContractConfig) -> Self {
        config.to_structure_prefab()
    }
}

/// Static builder for [`ResNetPreFabContractMap`].
pub type StaticResNetPreFabContractMap<'a> = StaticPreFabMap<'a, ResNetContractConfig>;

/// A map of [`ResNetContractConfig`]s.
pub type ResNetPreFabContractMap = PreFabMap<ResNetContractConfig>;

/// Static builder for [`ResNetPreFabStructureMap`].
pub type StaticResNetPreFabStructureMap<'a> = StaticPreFabMap<'a, ResNetStructureConfig>;

/// A map of [`ResNetStructureConfig`]s.
pub type ResNetPreFabStructureMap = PreFabMap<ResNetStructureConfig>;

/// `ResNet18` pretrained on `ImageNet`.
pub static PRETRAINED_RESNETS: StaticResNetPreFabContractMap = StaticResNetPreFabContractMap {
    name: "resnet",
    description: "Well-Know ResNet configs",

    items: &[&StaticResNetPreFabContractConfig {
        name: "resnet-18",
        description: "ResNet-18 [2, 2, 2, 2] BasicBlocks",
        builder: || ResNetContractConfig::new(RESNET18_BLOCKS, 1000),

        weights: Some(&StaticPretrainedWeightsMap {
            items: &[&StaticPretrainedWeightsDescriptor {
                name: "tv_in1k",
                description: "ResNet18 pretrained on ImageNet",
                license: Some("bsd-3-clause"),
                origin: Some("https://github.com/pytorch/vision"),
                urls: &["https://download.pytorch.org/models/resnet18-f37072fd.pth"],
            }],
        }),
    }],
};
