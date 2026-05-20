//! Stage that runs a sequence of stages.
use std::sync::Arc;

use image::DynamicImage;
use serde::{
    Deserialize,
    Serialize,
};

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

define_image_aug_plugin!(STAGE_SEQUENCE, StageSequence::build_stage);

/// Serialized configuration for `StageSequence`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageSequenceConfig {
    /// The sequence of augmentation stages.
    pub stages: Vec<AugmentationStageConfig>,
}

/// A stage which applies a sequence of augmentation stages.
#[derive(Debug, Clone)]
pub struct StageSequence {
    /// The sequence of augmentation stages.
    stages: Vec<Arc<dyn AugmentationStage>>,
}

impl WithAugmentationStageBuilder for StageSequence {
    /// Builder for `SequencePlugin`.
    fn build_stage(
        config: &AugmentationStageConfig,
        builder: &dyn PluginBuilder,
    ) -> anyhow::Result<Arc<dyn AugmentationStage>> {
        let config: StageSequenceConfig = serde_json::from_value(config.body.clone())?;
        let stages = builder.build_stage_vector(&config.stages)?;
        Ok(Arc::new(StageSequence { stages }))
    }
}

impl AugmentationStage for StageSequence {
    fn name(&self) -> &str {
        STAGE_SEQUENCE
    }

    fn as_config_body(&self) -> serde_json::Value {
        serde_json::to_value(StageSequenceConfig {
            stages: self
                .stages
                .iter()
                .map(|s| s.as_config())
                .collect::<Vec<_>>(),
        })
        .unwrap()
    }

    fn augment_image(
        &self,
        image: DynamicImage,
        ctx: &mut ImageAugContext,
    ) -> anyhow::Result<DynamicImage> {
        let mut image = image;
        for stage in &self.stages {
            image = stage.augment_image(image, ctx)?;
        }
        Ok(image)
    }
}
