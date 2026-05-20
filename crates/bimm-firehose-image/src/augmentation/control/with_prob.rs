//! Stage that randomly runs, or skips, a child.
use std::sync::Arc;

use bimm_firehose::utility::probability::try_probability;
use image::DynamicImage;
use rand::Rng;
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

define_image_aug_plugin!(WITH_PROB, WithProbStage::build_stage);

/// Serialized configuration for the `WithProbStage`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WithProbStageConfig {
    /// The probability of applying the inner stage.
    pub prob: f64,

    /// The inner stage.
    pub inner: AugmentationStageConfig,
}

/// A plugin which stochastically applies an inner stage.
#[derive(Debug, Clone)]
pub struct WithProbStage {
    /// The probability of applying the inner stage.
    prob: f64,

    /// The wrapped stage.
    inner: Arc<dyn AugmentationStage>,
}

impl WithProbStage {
    /// Construct a new `WithProbStage`.
    pub fn new(
        prob: f64,
        inner: Arc<dyn AugmentationStage>,
    ) -> Self {
        Self { prob, inner }
    }
}

impl WithAugmentationStageBuilder for WithProbStage {
    /// Builder for `WithProbStage`.
    fn build_stage(
        config: &AugmentationStageConfig,
        builder: &dyn PluginBuilder,
    ) -> anyhow::Result<Arc<dyn AugmentationStage>> {
        let config: WithProbStageConfig = serde_json::from_value(config.body.clone())?;

        Ok(Arc::new(Self {
            prob: try_probability(config.prob)?,
            inner: builder.build_stage(&config.inner)?,
        }))
    }
}

impl AugmentationStage for WithProbStage {
    fn name(&self) -> &str {
        WITH_PROB
    }

    fn as_config_body(&self) -> serde_json::Value {
        serde_json::to_value(WithProbStageConfig {
            prob: self.prob,
            inner: self.inner.as_config(),
        })
        .unwrap()
    }

    fn augment_image(
        &self,
        image: DynamicImage,
        ctx: &mut ImageAugContext,
    ) -> anyhow::Result<DynamicImage> {
        let mut image = image;
        if ctx.rng_mut().random::<f64>() < self.prob {
            image = self.inner.augment_image(image, ctx)?;
        }
        Ok(image)
    }
}
