//! Image blur stages.
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

define_image_aug_plugin!(BLUR, BlurStage::build_stage);

/// A stage of blurring.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum BlurStage {
    /// A Gaussian blur.
    Gaussian {
        /// The standard deviation for the Gaussian blur.
        sigma: f32,
    },
}

impl Default for BlurStage {
    fn default() -> Self {
        Self::Gaussian { sigma: 1.0 }
    }
}

impl WithAugmentationStageBuilder for BlurStage {
    fn build_stage(
        config: &AugmentationStageConfig,
        _builder: &dyn PluginBuilder,
    ) -> anyhow::Result<Arc<dyn AugmentationStage>> {
        Ok(Arc::new(serde_json::from_value::<BlurStage>(
            config.body.clone(),
        )?))
    }
}

impl AugmentationStage for BlurStage {
    fn name(&self) -> &str {
        BLUR
    }

    fn as_config_body(&self) -> Value {
        serde_json::to_value(self).unwrap()
    }

    fn augment_image(
        &self,
        image: DynamicImage,
        _ctx: &mut ImageAugContext,
    ) -> anyhow::Result<DynamicImage> {
        Ok(match self {
            BlurStage::Gaussian { sigma } => image.blur(*sigma),
        })
    }
}
