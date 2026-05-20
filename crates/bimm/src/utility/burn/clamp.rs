//! # Tensor Clamping Support

use burn::{
    module::{
        Content,
        ModuleDisplay,
        ModuleDisplayDefault,
    },
    prelude::{
        Backend,
        Tensor,
    },
};
use serde::{
    Deserialize,
    Serialize,
};

/// Configuration for clamping.
#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClampConfig {
    /// The minimum value.
    min: Option<f64>,

    /// The maximum value.
    max: Option<f64>,
}

impl ModuleDisplay for ClampConfig {}
impl ModuleDisplayDefault for ClampConfig {
    fn content(
        &self,
        content: Content,
    ) -> Option<Content> {
        Some(content.add("min", &self.min).add("max", &self.max))
    }
}

impl ClampConfig {
    /// Create a new clamp with both minimum and maximum values.
    pub fn min_max(
        min: f64,
        max: f64,
    ) -> Self {
        Self {
            min: Some(min),
            max: Some(max),
        }
    }

    /// Extend the clamp with a minimum value.
    pub fn with_min(
        self,
        min: f64,
    ) -> Self {
        Self {
            min: Some(min),
            ..self
        }
    }

    /// Extend the clamp with a maximum value.
    pub fn with_max(
        self,
        max: f64,
    ) -> Self {
        Self {
            max: Some(max),
            ..self
        }
    }

    /// Apply the clamp.
    pub fn clamp<B: Backend, const D: usize>(
        &self,
        tensor: Tensor<B, D>,
    ) -> Tensor<B, D> {
        match (self.min, self.max) {
            (Some(min), Some(max)) => tensor.clamp(min, max),
            (Some(min), None) => tensor.clamp_min(min),
            (None, Some(max)) => tensor.clamp_max(max),
            (None, None) => tensor,
        }
    }
}

#[cfg(test)]
mod tests {
    use bunsen::support::testing::SetupTestBackend;
    use burn::{
        module::{
            DisplaySettings,
            ModuleDisplay,
        },
        tensor::TensorData,
    };
    use num_traits::clamp;

    use super::*;
    use crate::layers::drop::drop_block::DropBlockOptions;

    #[test]
    fn test_clamp_config_display() {
        let config = ClampConfig::default().with_min(0.5);
        let settings = DisplaySettings::default();

        assert_eq!(
            config.format(settings),
            indoc::indoc! {
                r#"
                ClampConfig {
                  min: 0.5
                  max: None
                }"#
            }
        )
    }

    #[test]
    fn test_config_min_max() {
        let cfg = ClampConfig::min_max(-1.0, 1.0);
        assert_eq!(
            cfg,
            ClampConfig {
                min: Some(-1.0),
                max: Some(1.0),
            }
        );
    }

    #[test]
    fn test_config() {
        type B = SetupTestBackend;
        let device = Default::default();

        let cfg = ClampConfig::default();
        assert_eq!(
            cfg,
            ClampConfig {
                min: None,
                max: None,
            }
        );
        let tensor = Tensor::<B, 1>::from_data([-1.0, 0.0, 1.0], &device);
        let tensor = cfg.clamp(tensor);
        tensor
            .to_data()
            .assert_eq(&TensorData::from([-1.0, 0.0, 1.0]), false);

        let cfg = ClampConfig::default().with_min(-0.5).with_max(0.5);
        assert_eq!(
            cfg,
            ClampConfig {
                min: Some(-0.5),
                max: Some(0.5),
            }
        );
        let tensor = Tensor::<B, 1>::from_data([-1.0, 0.0, 1.0], &device);
        let tensor = cfg.clamp(tensor);
        tensor
            .to_data()
            .assert_eq(&TensorData::from([-0.5, 0.0, 0.5]), false);
    }
}
