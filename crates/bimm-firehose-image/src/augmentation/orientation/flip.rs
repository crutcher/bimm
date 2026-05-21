//! Image flip stages.
use std::sync::Arc;

use image::DynamicImage;
use serde::{
    Deserialize,
    Serialize,
};
use serde_json::Value;

use crate::{
    augmentation::{
        AugmentationStage,
        AugmentationStageConfig,
        ImageAugContext,
        PluginBuilder,
        WithAugmentationStageBuilder,
    },
    define_image_aug_plugin,
};

define_image_aug_plugin!(HORIZONTAL_FLIP, HorizontalFlipStage::build_stage);

/// An `AugmentationStage` that flips the image horizontally.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HorizontalFlipStage;

impl Default for HorizontalFlipStage {
    fn default() -> Self {
        Self::new()
    }
}

impl HorizontalFlipStage {
    /// Creates a new `HorizontalFlipStage`.
    pub fn new() -> Self {
        Self {}
    }
}

impl WithAugmentationStageBuilder for HorizontalFlipStage {
    fn build_stage(
        config: &AugmentationStageConfig,
        _builder: &dyn PluginBuilder,
    ) -> anyhow::Result<Arc<dyn AugmentationStage>> {
        Ok(Arc::new(serde_json::from_value::<HorizontalFlipStage>(
            config.body.clone(),
        )?))
    }
}

impl AugmentationStage for HorizontalFlipStage {
    fn name(&self) -> &str {
        HORIZONTAL_FLIP
    }

    fn as_config_body(&self) -> Value {
        Value::Null
    }

    fn augment_image(
        &self,
        image: DynamicImage,
        _ctx: &mut ImageAugContext,
    ) -> anyhow::Result<DynamicImage> {
        Ok(image.fliph())
    }
}

define_image_aug_plugin!(VERTICAL_FLIP, VerticalFlipStage::build_stage);

/// An `AugmentationStage` that flips the image vertically.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerticalFlipStage;

impl Default for VerticalFlipStage {
    fn default() -> Self {
        Self::new()
    }
}

impl VerticalFlipStage {
    /// Creates a new `VerticalFlipStage`.
    pub fn new() -> Self {
        Self {}
    }
}

impl WithAugmentationStageBuilder for VerticalFlipStage {
    fn build_stage(
        config: &AugmentationStageConfig,
        _builder: &dyn PluginBuilder,
    ) -> anyhow::Result<Arc<dyn AugmentationStage>> {
        Ok(Arc::new(serde_json::from_value::<VerticalFlipStage>(
            config.body.clone(),
        )?))
    }
}

impl AugmentationStage for VerticalFlipStage {
    fn name(&self) -> &str {
        VERTICAL_FLIP
    }

    fn as_config_body(&self) -> Value {
        Value::Null
    }

    fn augment_image(
        &self,
        image: DynamicImage,
        _ctx: &mut ImageAugContext,
    ) -> anyhow::Result<DynamicImage> {
        Ok(image.flipv())
    }
}
