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
pub static RESNET18_PREFAB: StaticResNetPreFabContract = StaticResNetPreFabContract {
    name: "resnet18",
    description: "ResNet18 pretrained on ImageNet",
    builder: || ResNetContractConfig::new(RESNET18_BLOCKS, 1000),
};

/// Static builder for [`ResNetPreFabContract`].
pub type StaticResNetPreFabContract = StaticPreFabConfig<ResNetContractConfig>;

/// A [`ResNetContractConfig`] Well-Known Pre-Fab.
pub type ResNetPreFabContract = PreFabConfig<ResNetContractConfig>;

/// Static builder for [`ResNetPreFabContract`].
pub type StaticResNetPreFabStructure = StaticPreFabConfig<ResNetStructureConfig>;

/// A [`ResNetStructureConfig`] Well-Known Pre-Fab.
pub type ResNetPreFabStructure = PreFabConfig<ResNetStructureConfig>;

impl From<&StaticResNetPreFabContract> for ResNetPreFabStructure {
    fn from(config: &StaticResNetPreFabContract) -> Self {
        config.to_prefab().into()
    }
}

impl ResNetPreFabContract {
    /// Convert to a [`ResNetPreFabStructure`].
    pub fn to_structure_prefab(self) -> ResNetPreFabStructure {
        let builder = self.builder;
        ResNetPreFabStructure {
            name: self.name,
            description: self.description,
            builder: Arc::new(move || builder().to_structure()),
        }
    }
}

impl From<ResNetPreFabContract> for ResNetPreFabStructure {
    fn from(config: ResNetPreFabContract) -> Self {
        config.to_structure_prefab()
    }
}
