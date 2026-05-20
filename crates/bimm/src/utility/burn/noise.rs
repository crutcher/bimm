//! # Tensor Noise Generation Utilities.

use burn::{
    module::{
        Content,
        ModuleDisplay,
        ModuleDisplayDefault,
    },
    prelude::{
        Backend,
        Shape,
        Tensor,
    },
    tensor::Distribution,
};
use serde::{
    Deserialize,
    Serialize,
};

use crate::utility::burn::{
    clamp::ClampConfig,
    distribution::DistributionDisplayAdapter,
};

/// Noise Configuration.
///
/// Carries a [`Distribution`] and an optional [`ClampConfig`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NoiseConfig {
    /// The noise distribution.
    pub distribution: Distribution,

    /// The noise clip range.
    pub clamp: Option<ClampConfig>,
}

impl ModuleDisplay for NoiseConfig {}
impl ModuleDisplayDefault for NoiseConfig {
    fn content(
        &self,
        content: Content,
    ) -> Option<Content> {
        Some(
            content
                .add(
                    "distribution",
                    &DistributionDisplayAdapter::new(self.distribution),
                )
                .add("clamp", &self.clamp),
        )
    }
}

impl Default for NoiseConfig {
    fn default() -> Self {
        Self {
            distribution: Distribution::Default,
            clamp: None,
        }
    }
}

impl NoiseConfig {
    /// Extend the config with the given [`Distribution`].
    pub fn with_distribution(
        self,
        distribution: Distribution,
    ) -> Self {
        Self {
            distribution,
            ..self
        }
    }

    /// Extend the config with the given [`ClampConfig`].
    pub fn with_clamp<C>(
        self,
        clamp: C,
    ) -> Self
    where
        C: Into<Option<ClampConfig>>,
    {
        Self {
            clamp: clamp.into(),
            ..self
        }
    }

    /// Generate noise.
    ///
    /// Noise is drawn from the distribution; and optionally clamped.
    ///
    /// # Arguments
    ///
    /// - `shape` - the shape of the noise tensor to generate.
    /// - `device` - the device to build the tensor on.
    ///
    /// # Returns
    ///
    /// A new tensor with the given shape and device, filled with noise.
    pub fn noise<B: Backend, S, const D: usize>(
        &self,
        shape: S,
        device: &B::Device,
    ) -> Tensor<B, D>
    where
        S: Into<Shape>,
    {
        let noise = Tensor::random(shape.into(), self.distribution, device);
        match &self.clamp {
            None => noise,
            Some(clamp_cfg) => clamp_cfg.clamp(noise),
        }
    }

    /// Generates noise like a reference tensor.
    ///
    /// # Arguments
    ///
    /// - `tensor`: A reference tensor to match the shape and device.
    ///
    /// # Returns
    ///
    /// A new tensor with the same shape and device as the reference.
    pub fn noise_like<B: Backend, const D: usize>(
        &self,
        tensor: &Tensor<B, D>,
    ) -> Tensor<B, D> {
        self.noise(tensor.shape(), &tensor.device())
    }
}

#[cfg(test)]
mod tests {
    use bunsen::support::testing::PerfTestBackend;
    use burn::{
        backend::NdArray,
        module::DisplaySettings,
    };
    use num_traits::real::Real;

    use super::*;
    #[test]
    fn test_noise_config_display() {
        let config = NoiseConfig::default().with_clamp(ClampConfig::min_max(0.5, 1.0));
        let settings = DisplaySettings::default();

        assert_eq!(
            config.format(settings),
            indoc::indoc! {r#"
                NoiseConfig {
                  distribution: Distribution::Default
                  clamp: ClampConfig {
                      min: 0.5
                      max: 1
                    }
                }"#
            }
        )
    }

    #[test]
    fn test_noise_default() {
        let cfg = NoiseConfig::default();
        assert_eq!(
            cfg,
            NoiseConfig {
                distribution: Distribution::Default,
                clamp: None
            }
        );

        let cfg = NoiseConfig::default()
            .with_distribution(Distribution::Bernoulli(0.3))
            .with_clamp(ClampConfig::default());
        assert_eq!(
            cfg,
            NoiseConfig {
                distribution: Distribution::Bernoulli(0.3),
                clamp: Some(ClampConfig::default())
            }
        );

        let cfg = NoiseConfig::default().with_clamp(Some(ClampConfig::default()));
        assert_eq!(
            cfg,
            NoiseConfig {
                distribution: Distribution::Default,
                clamp: Some(ClampConfig::default())
            }
        );

        let cfg = NoiseConfig::default()
            .with_clamp(ClampConfig::default())
            .with_clamp(None);
        assert_eq!(
            cfg,
            NoiseConfig {
                distribution: Distribution::Default,
                clamp: None,
            }
        );
    }

    #[test]
    fn test_noise_like_default_clamp() {
        type B = PerfTestBackend;
        let device = Default::default();

        let reference: Tensor<B, 2> = Tensor::ones([20, 20], &device);
        let numel = reference.shape().num_elements() as f64;

        let p = 0.1;

        let noise = NoiseConfig::default()
            .with_clamp(ClampConfig::default().with_min(0.5))
            .noise_like(&reference);

        assert_eq!(noise.shape(), reference.shape());
        assert_eq!(noise.device(), reference.device());

        // * Half of values should be exactly 0.5
        // * All values should be in [0.5, 1.0)

        // count 0.5
        let count_05 = noise.clone().equal_elem(0.5).int().sum().into_scalar() as f64;
        assert!((0.5 - (count_05 / numel)).abs() < 0.15);

        let count_ge_1 = noise
            .clone()
            .greater_equal_elem(1.0)
            .int()
            .sum()
            .into_scalar();
        assert_eq!(count_ge_1, 0);
    }

    #[test]
    fn test_noise_like_bernoulli() {
        type B = PerfTestBackend;
        let device = Default::default();

        let reference: Tensor<B, 2> = Tensor::ones([20, 20], &device);

        let p = 0.1;

        let noise = NoiseConfig::default()
            .with_distribution(Distribution::Bernoulli(p))
            .noise_like(&reference);

        assert_eq!(noise.shape(), reference.shape());
        assert_eq!(noise.device(), reference.device());

        let ratio =
            (noise.clone().sum().into_scalar() as f64) / (noise.shape().num_elements() as f64);
        assert!((ratio - p).abs() < 0.05);
    }
}
