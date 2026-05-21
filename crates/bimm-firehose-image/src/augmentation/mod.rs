//! # Image augmentation operators.
use std::{
    fmt::Debug,
    sync::Arc,
};

use anyhow::Context;
use bimm_firehose::{
    core::{
        FirehoseRowReader,
        FirehoseRowTransaction,
        FirehoseRowWriter,
        operations::{
            factory::{
                FirehoseOperatorFactory,
                FirehoseOperatorInitContext,
            },
            operator::FirehoseOperator,
            planner::OperationPlan,
            signature::{
                FirehoseOperatorSignature,
                ParameterSpec,
            },
        },
    },
    define_firehose_operator,
};
pub use image::{
    ColorType,
    DynamicImage,
    imageops::FilterType,
};
use rand::SeedableRng;
use serde::{
    Deserialize,
    Serialize,
};

pub mod control;

pub mod noise;
pub mod orientation;

/// Defines an ID and registers a [`AugmentationStage`].
#[macro_export]
macro_rules! define_image_aug_plugin {
    ($name:ident, $builder:expr) => {
        $crate::define_image_aug_plugin_id!($name);
        $crate::register_image_aug_plugin!($name, $builder);
    };
}

/// Registers a [`AugmentationStage`].
///
/// # Arguments
///
/// - `name`: the local reference name of the plugin ID; used by
///   `define_image_aug_plugin!()`.
/// - `builder`: the plugin builder operation.
#[macro_export]
macro_rules! register_image_aug_plugin {
    ($name:ident, $builder:expr) => {
        inventory::submit! {
            $crate::augmentation::AugmentationStageGlobalRegistration {
                name: $name,
                build_stage: |cfg, builder| ($builder)(cfg, builder),
            }
        }
    };
}

/// Defines a self-referential plugin ID for [`AugmentationStage`].
#[macro_export]
macro_rules! define_image_aug_plugin_id {
    ($name:ident) => {
        bimm_firehose::define_self_referential_id!("fh:iaug", $name);
    };
}

inventory::collect!(AugmentationStageGlobalRegistration);

/// Registration record for [`AugmentationStage`] builders.
pub struct AugmentationStageGlobalRegistration {
    /// The plugin ID.
    pub name: &'static str,

    /// The plugin builder.
    pub build_stage: fn(
        config: &AugmentationStageConfig,
        builder: &dyn PluginBuilder,
    ) -> anyhow::Result<Arc<dyn AugmentationStage>>,
}

/// Builder for the `PluginConfig` to `ImageAugPlugin` path.
pub trait PluginBuilder {
    /// Build an `AugmentationStage` from config.
    fn build_stage(
        &self,
        config: &AugmentationStageConfig,
    ) -> anyhow::Result<Arc<dyn AugmentationStage>>;

    /// Build a vector of
    fn build_stage_vector(
        &self,
        configs: &Vec<AugmentationStageConfig>,
    ) -> anyhow::Result<Vec<Arc<dyn AugmentationStage>>> {
        let mut plugins = Vec::with_capacity(configs.len());
        for config in configs {
            plugins.push(self.build_stage(config)?);
        }
        Ok(plugins)
    }
}

/// Trait defining an associated builder for a plugin.
pub trait WithAugmentationStageBuilder {
    /// Build a plugin.
    fn build_stage(
        config: &AugmentationStageConfig,
        builder: &dyn PluginBuilder,
    ) -> anyhow::Result<Arc<dyn AugmentationStage>>;
}

/// Global plugin registry builder.
pub struct GlobalRegistryBuilder;

impl PluginBuilder for GlobalRegistryBuilder {
    fn build_stage(
        &self,
        config: &AugmentationStageConfig,
    ) -> anyhow::Result<Arc<dyn AugmentationStage>> {
        let name = config.name.as_str();

        let reg = inventory::iter::<AugmentationStageGlobalRegistration>
            .into_iter()
            .find(|reg| reg.name == name)
            .ok_or_else(|| anyhow::anyhow!("No plugin named {}", name))?;

        (reg.build_stage)(config, self)
    }
}

/// Augmentation context, used by `AugmentationStage::augment_image`.
#[derive(Debug, Clone)]
pub struct ImageAugContext {
    /// The context random number generator.
    rng: rand::rngs::StdRng,
}

impl ImageAugContext {
    /// Construct a new context.
    pub fn new(rng: rand::rngs::StdRng) -> Self {
        Self { rng }
    }

    /// Get a mutable reference to the context's random number generator.
    pub fn rng_mut(&mut self) -> &mut rand::rngs::StdRng {
        &mut self.rng
    }
}

/// A trait defining a plugin for image augmentation.
pub trait AugmentationStage: Debug + Send + Sync {
    /// Get the stage name.
    fn name(&self) -> &str;

    /// Construct the body of a config for this stage.
    fn as_config_body(&self) -> serde_json::Value;

    /// Construct a config for this.
    fn as_config(&self) -> AugmentationStageConfig {
        AugmentationStageConfig {
            name: self.name().to_string(),
            body: self.as_config_body(),
        }
    }

    /// Apply the stage to the image.
    ///
    /// # Arguments
    ///
    /// - `image`: the image to augment.
    /// - `ctx`: the `AugmentationContex` being operated in.
    ///
    /// # Returns
    ///
    /// A (modified?) image.
    fn augment_image(
        &self,
        image: DynamicImage,
        ctx: &mut ImageAugContext,
    ) -> anyhow::Result<DynamicImage>;
}

/// Serializable config for name + body for `AugmentationStage`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AugmentationStageConfig {
    /// The plugin name.
    pub name: String,

    /// The body of the plugin.
    #[serde(default, skip_serializing_if = "serde_json::Value::is_null")]
    pub body: serde_json::Value,
}

/// Image augmentation operator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AugmentImageConfig {
    /// The stages to apply.
    pub stages: Vec<AugmentationStageConfig>,
}

impl AugmentImageConfig {
    /// Converts into an `OperationPlanner`
    ///
    /// # Arguments
    ///
    /// * `seed_column`: The name of the input column containing the
    ///   augmentation seed.
    /// * `source_column`: The name of the input image column.
    /// * `result_column`: The name of the output image column.
    pub fn to_plan(
        &self,
        seed_column: &str,
        source_column: &str,
        result_column: &str,
    ) -> OperationPlan {
        OperationPlan::for_operation_id(AUGMENT_IMAGE)
            .with_input("seed", seed_column)
            .with_input("source", source_column)
            .with_output("result", result_column)
            .with_config(self.clone())
    }
}

define_firehose_operator!(AUGMENT_IMAGE, AugmentImageOperatorFactory::new());

/// Image augmentation operator factory.
#[derive(Debug)]
pub struct AugmentImageOperatorFactory {
    /// The operator signature.
    signature: FirehoseOperatorSignature,
}

impl Default for AugmentImageOperatorFactory {
    fn default() -> Self {
        Self::new()
    }
}

impl AugmentImageOperatorFactory {
    /// Construct a new factory.
    pub fn new() -> Self {
        Self {
            signature: FirehoseOperatorSignature::from_operator_id(AUGMENT_IMAGE)
                .with_description("Loads an image from disk.")
                .with_input(
                    ParameterSpec::new::<u64>("seed").with_description("Augmentation seed."),
                )
                .with_input(
                    ParameterSpec::new::<DynamicImage>("source").with_description("Source image."),
                )
                .with_output(
                    ParameterSpec::new::<DynamicImage>("result").with_description("Result image."),
                ),
        }
    }
}

impl FirehoseOperatorFactory for AugmentImageOperatorFactory {
    fn signature(&self) -> &FirehoseOperatorSignature {
        &self.signature
    }

    fn init(
        &self,
        context: &dyn FirehoseOperatorInitContext,
    ) -> anyhow::Result<Box<dyn FirehoseOperator>> {
        let config = &context.build_plan().config;
        let cfg: AugmentImageConfig =
            serde_json::from_value(config.clone()).with_context(|| {
                format!(
                    "Failed to deserialize operator config for {}: {}",
                    self.signature.operator_id.as_deref().unwrap_or("unknown"),
                    serde_json::to_string_pretty(config).unwrap()
                )
            })?;

        let builder = GlobalRegistryBuilder {};
        let stages = builder.build_stage_vector(&cfg.stages)?;

        Ok(Box::new(AugmentImageOperation { stages }))
    }
}

/// Image augmentation operator.
#[derive(Debug, Clone)]
pub struct AugmentImageOperation {
    /// The stages to apply.
    stages: Vec<Arc<dyn AugmentationStage>>,
}

impl AugmentImageOperation {
    /// Construct a new operation.
    pub fn new(stages: Vec<Arc<dyn AugmentationStage>>) -> Self {
        Self { stages }
    }

    /// Convert into a config.
    pub fn to_config(&self) -> AugmentImageConfig {
        AugmentImageConfig {
            stages: self.stages.iter().map(|s| s.as_config()).collect(),
        }
    }

    /// Converts into an `OperationPlanner`
    ///
    /// # Arguments
    ///
    /// * `seed_column`: The name of the input column containing the
    ///   augmentation seed.
    /// * `source_column`: The name of the input image column.
    /// * `result_column`: The name of the output image column.
    pub fn to_plan(
        &self,
        seed_column: &str,
        source_column: &str,
        result_column: &str,
    ) -> OperationPlan {
        OperationPlan::for_operation_id(AUGMENT_IMAGE)
            .with_input("seed", seed_column)
            .with_input("source", source_column)
            .with_output("result", result_column)
            .with_config(self.to_config())
    }
}

impl FirehoseOperator for AugmentImageOperation {
    fn apply_to_row(
        &self,
        txn: &mut FirehoseRowTransaction,
    ) -> anyhow::Result<()> {
        let mut image = txn.expect_get_ref::<DynamicImage>("source").clone();
        let mut ctx = ImageAugContext::new(rand::rngs::StdRng::seed_from_u64(
            txn.expect_get_parsed("seed"),
        ));

        for stage in &self.stages {
            image = stage.augment_image(image, &mut ctx)?;
        }

        txn.expect_set_from_box("result", Box::new(image));

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::augmentation::control::{
        noop::NOOP_STAGE,
        sequence::STAGE_SEQUENCE,
    };

    #[test]
    fn test_plugin_builder() -> anyhow::Result<()> {
        let cfg_json = json! {
            {
                "name": STAGE_SEQUENCE,
                "body": {
                  "stages": [
                    {
                      "name": NOOP_STAGE,
                    }
                  ]
                }
            }
        };
        let pretty = serde_json::to_string_pretty(&cfg_json)?;
        println!("{pretty}");

        let cfg = serde_json::from_value::<AugmentationStageConfig>(cfg_json.clone())?;

        let builder = GlobalRegistryBuilder {};

        let plugin = builder.build_stage(&cfg)?;

        let new_cfg = plugin.as_config();
        let new_pretty = serde_json::to_string_pretty(&new_cfg)?;
        println!("{new_pretty}");

        // assert!(false);

        Ok(())
    }
}
