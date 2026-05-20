//! Stage that randomly selects one of its children.
use std::sync::Arc;

use anyhow::bail;
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

define_image_aug_plugin!(CHOOSE_ONE, ChooseOneStage::build_stage);

/// Serialized configuration for `ChooseOneStage`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChoiceItemConfig {
    /// The weight of the stage; default, treated as `1.0`.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    pub weight: Option<f32>,

    /// The augmentation stage.
    pub stage: AugmentationStageConfig,
}

/// Serialized configuration for `StageChoice`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChooseOneStageConfig {
    /// The sequence of augmentation stages.
    pub choices: Vec<ChoiceItemConfig>,
}

/// A stage which applies a sequence of augmentation stages.
#[derive(Debug, Clone)]
pub struct ChooseOneStage {
    /// The weights of the stages.
    weights: Vec<Option<f32>>,

    /// The sequence of augmentation stages.
    stages: Vec<Arc<dyn AugmentationStage>>,
}

impl Default for ChooseOneStage {
    fn default() -> Self {
        Self::new()
    }
}

impl ChooseOneStage {
    /// Create a new, empty stage.
    pub fn new() -> Self {
        Self {
            weights: Vec::new(),
            stages: Vec::new(),
        }
    }

    /// Get the total weight of the stage.
    pub fn total_weight(&self) -> f32 {
        self.weights.iter().map(|w| w.unwrap_or(1.0)).sum()
    }

    /// Append a choice to the stage.
    pub fn with_choice<W>(
        self,
        weight: W,
        stage: Arc<dyn AugmentationStage>,
    ) -> Self
    where
        W: Into<Option<f32>>,
    {
        let mut weights = self.weights;
        let mut stages = self.stages;
        weights.push(weight.into());
        stages.push(stage);
        Self { weights, stages }
    }

    /// Append a noop stage to the stage.
    pub fn with_noop_weight<W>(
        self,
        weight: W,
    ) -> Self
    where
        W: Into<Option<f32>>,
    {
        self.with_choice(
            weight,
            Arc::new(crate::augmentation::control::noop::NoOpStage {}),
        )
    }
}

impl WithAugmentationStageBuilder for ChooseOneStage {
    /// Builder for `ChoicePlugin`.
    fn build_stage(
        config: &AugmentationStageConfig,
        builder: &dyn PluginBuilder,
    ) -> anyhow::Result<Arc<dyn AugmentationStage>> {
        let config: ChooseOneStageConfig = serde_json::from_value(config.body.clone())?;

        let mut stages = Vec::with_capacity(config.choices.len());
        let mut weights = Vec::with_capacity(config.choices.len());
        for (idx, choice) in config.choices.iter().enumerate() {
            weights.push(choice.weight);
            let weight = choice.weight.unwrap_or(1.0);
            if weight < 0.0 {
                bail!(
                    "Invalid weight ({}) at index ({idx}):\n{}",
                    weight,
                    serde_json::to_string_pretty(&config)?
                );
            }
            stages.push(builder.build_stage(&choice.stage)?);
        }

        Ok(Arc::new(Self { weights, stages }))
    }
}

impl AugmentationStage for ChooseOneStage {
    fn name(&self) -> &str {
        CHOOSE_ONE
    }

    fn as_config_body(&self) -> serde_json::Value {
        serde_json::to_value(ChooseOneStageConfig {
            choices: self
                .stages
                .iter()
                .enumerate()
                .map(|(idx, stage)| ChoiceItemConfig {
                    weight: self.weights[idx],
                    stage: stage.as_config(),
                })
                .collect::<Vec<_>>(),
        })
        .unwrap()
    }

    fn augment_image(
        &self,
        image: DynamicImage,
        ctx: &mut ImageAugContext,
    ) -> anyhow::Result<DynamicImage> {
        let mut r = self.total_weight() * ctx.rng_mut().random::<f32>();
        let mut idx = 0;
        for weight in &self.weights {
            let weight = weight.unwrap_or(1.0);
            if r < weight {
                break;
            }
            idx += 1;
            r -= weight;
        }
        let stage = &self.stages[idx];
        stage.augment_image(image, ctx)
    }
}
