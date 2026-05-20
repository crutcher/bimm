//! Image speckle noise stages.
use std::sync::Arc;

use image::{
    DynamicImage,
    GenericImage,
    GenericImageView,
};
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

define_image_aug_plugin!(SPECKLE, SpeckleStage::build_stage);

/// A stage of speckle noise augmentation.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum SpeckleStage {
    /// Uniform speckle noise.
    Uniform {
        /// The pixel density of the speckles.
        density: f32,
    },
}

impl Default for SpeckleStage {
    fn default() -> Self {
        Self::Uniform {
            density: 1.0 / 100.0,
        }
    }
}

impl WithAugmentationStageBuilder for SpeckleStage {
    fn build_stage(
        config: &AugmentationStageConfig,
        _builder: &dyn PluginBuilder,
    ) -> anyhow::Result<Arc<dyn AugmentationStage>> {
        Ok(Arc::new(serde_json::from_value::<SpeckleStage>(
            config.body.clone(),
        )?))
    }
}

impl AugmentationStage for SpeckleStage {
    fn name(&self) -> &str {
        SPECKLE
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
            SpeckleStage::Uniform { density } => {
                let mut image = image.clone();

                let (w, h) = image.dimensions();
                let speckle_count = (((h * w) as f32) * *density) as usize;

                for _ in 0..speckle_count {
                    let a = (w as f32 * rand::random::<f32>()) as u32;
                    let b = (h as f32 * rand::random::<f32>()) as u32;
                    let x = (w as f32 * rand::random::<f32>()) as u32;
                    let y = (h as f32 * rand::random::<f32>()) as u32;
                    image.put_pixel(x, y, image.get_pixel(a, b));
                }

                image
            }
        })
    }
}
