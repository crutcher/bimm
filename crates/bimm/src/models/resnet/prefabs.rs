//! # `ResNet` Model Loaders

// Loader goals:
//
// * named pretrained models:
//   `&str` -> `Loader`.
// * For all: build a structure config
//   Loader -> `ResNetStructureConfig`.
// * For some: build a contract config
//   Loader -> `ResNetContractConfig`.
// * For either config: an opportunity to edit the config before loading.
//
// For weight sources:
// There is a `1:N` relationship between model configs and weight sources.
//
// The symbolic id of a source, and the load path to pull that weight set,
// may be distinct. Weights are generally published as URLs; but in a pre-built
// model, they may be loaded from another location.
//
// PreFab - a well known model config.
// PreTrained - a pretrained model.
//
// name 1:1 PreFab
//   (every PreFab has unique name)
//
// name 1:1 PreTrained
//   (every PreTrained has unique name)
//
// PreFab 1:N PreTrained
//   (every PreFab may have multiple PreTraineds, every PreTrained has a unique PreFab)

use crate::cache::prefabs::{PreFabConfig, StaticPreFabConfig};
use crate::cache::weights::StaticPretrainedWeightsDescriptor;
use crate::models::resnet::{RESNET18_BLOCKS, ResNetContractConfig, ResNetStructureConfig};
use std::sync::Arc;

/// `ResNet18` pretrained on `ImageNet`.
pub static RESNET18_TORCHVISION: StaticPretrainedWeightsDescriptor =
    StaticPretrainedWeightsDescriptor {
        name: "resnet18",
        description: "ResNet18 pretrained on ImageNet",
        urls: &["https://download.pytorch.org/models/resnet18-f37072fd.pth"],
    };

/// `ResNet18` pretrained on `ImageNet`.
pub static RESNET18_PREFAB: StaticResNetPreFabContractConfig = StaticResNetPreFabContractConfig {
    name: "resnet18",
    description: "ResNet18 pretrained on ImageNet",
    builder: || ResNetContractConfig::new(RESNET18_BLOCKS, 1000),
};

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
        }
    }
}

impl From<&ResNetPreFabContractConfig> for ResNetPreFabStructureConfig {
    fn from(config: &ResNetPreFabContractConfig) -> Self {
        config.to_structure_prefab()
    }
}
