//! # Normalization Wrapper
//!
//! Provides support for built-in ``burn::nn::norm`` norm layers:
//! * [`Batch`] - [`BatchNorm`]
//! * [`Group`] - [`GroupNorm`]
//! * [`Instance`] - [`InstanceNorm`]
//! * [`Layer`] - [`LayerNorm`]
//!
//! ## ['`RmsNorm`'] is not supported.
//!
//! * In v0.18.0, this fails to implement Debug.
//! * In v0.19.0-dev, this does.
//!
//! The enum is non-exhaustive, to prepare for future additions.

use burn::nn::{
    BatchNorm, BatchNormConfig, GroupNorm, GroupNormConfig, InstanceNorm, InstanceNormConfig,
    LayerNorm, LayerNormConfig,
};
use burn::prelude::{Backend, Config, Module, Tensor};

/// ['Normalization'] Configuration.
///
/// ## ['`RmsNorm`'] is not supported.
///
/// * In v0.18.0, this fails to implement Debug.
/// * In v0.19.0-dev, this does.
///
/// The enum is non-exhaustive to prepare for future additions.
#[derive(Config, Debug)]
#[non_exhaustive]
pub enum NormalizationConfig {
    /// ['`BatchNorm`'] Configuration.
    Batch(BatchNormConfig),

    /// ['`GroupNorm`'] Configuration.
    Group(GroupNormConfig),

    /// ['`InstanceNorm`'] Configuration.
    Instance(InstanceNormConfig),

    /// ['`LayerNorm`'] Configuration.
    Layer(LayerNormConfig),
}

impl From<BatchNormConfig> for NormalizationConfig {
    fn from(config: BatchNormConfig) -> Self {
        Self::Batch(config)
    }
}

impl From<GroupNormConfig> for NormalizationConfig {
    fn from(config: GroupNormConfig) -> Self {
        Self::Group(config)
    }
}

impl From<InstanceNormConfig> for NormalizationConfig {
    fn from(config: InstanceNormConfig) -> Self {
        Self::Instance(config)
    }
}

impl From<LayerNormConfig> for NormalizationConfig {
    fn from(config: LayerNormConfig) -> Self {
        Self::Layer(config)
    }
}

impl NormalizationConfig {
    /// Initialize a ['Norm'] layer.
    pub fn init<B: Backend>(
        &self,
        device: &B::Device,
    ) -> Normalization<B> {
        match self {
            NormalizationConfig::Batch(config) => config.init(device).into(),
            NormalizationConfig::Group(config) => config.init(device).into(),
            NormalizationConfig::Instance(config) => config.init(device).into(),
            NormalizationConfig::Layer(config) => config.init(device).into(),
            // NormalizationConfig::Rms(config) => config.init(device).into(),
        }
    }

    /// Adjust a norm config to the feature size.
    pub fn with_num_features(
        self,
        num_features: usize,
    ) -> Self {
        match self {
            NormalizationConfig::Batch(config) => BatchNormConfig {
                num_features,
                ..config
            }
            .into(),
            NormalizationConfig::Group(config) => GroupNormConfig {
                num_channels: num_features,
                ..config
            }
            .into(),
            NormalizationConfig::Instance(config) => InstanceNormConfig {
                num_channels: num_features,
                ..config
            }
            .into(),
            NormalizationConfig::Layer(config) => LayerNormConfig {
                d_model: num_features,
                ..config
            }
            .into(),
        }
    }

    /// Get the number of features.
    pub fn num_features(&self) -> usize {
        match self {
            NormalizationConfig::Batch(config) => config.num_features,
            NormalizationConfig::Group(config) => config.num_channels,
            NormalizationConfig::Instance(config) => config.num_channels,
            NormalizationConfig::Layer(config) => config.d_model,
        }
    }
}

/// Normalization Layer Wrapper
///
/// Provides support for built-in ``burn::nn::norm`` norm layers:
/// * [`Batch`] - [`BatchNorm`]
/// * [`Group`] - [`GroupNorm`]
/// * [`Instance`] - [`InstanceNorm`]
/// * [`Layer`] - [`LayerNorm`]
/// * ['`RmsNorm`'] - Not Supported.
///   * In v0.18.0, this fails to implement Debug.
///   * In v0.19.0-dev, this does.
///
/// The enum is non-exhaustive, to prepare for future additions.
#[derive(Module, Debug)]
#[non_exhaustive]
pub enum Normalization<B: Backend> {
    /// [`BatchNorm`] layer; restricted to `BatchNorm`<2>.
    Batch(BatchNorm<B, 2>),

    /// [`GroupNorm`] layer.
    Group(GroupNorm<B>),

    /// ['`InstanceNorm`'] layer.
    Instance(InstanceNorm<B>),

    /// [`LayerNorm`] layer.
    Layer(LayerNorm<B>),
}

impl<B: Backend> From<BatchNorm<B, 2>> for Normalization<B> {
    fn from(layer: BatchNorm<B, 2>) -> Self {
        Self::Batch(layer)
    }
}

impl<B: Backend> From<GroupNorm<B>> for Normalization<B> {
    fn from(layer: GroupNorm<B>) -> Self {
        Self::Group(layer)
    }
}

impl<B: Backend> From<InstanceNorm<B>> for Normalization<B> {
    fn from(layer: InstanceNorm<B>) -> Self {
        Self::Instance(layer)
    }
}

impl<B: Backend> From<LayerNorm<B>> for Normalization<B> {
    fn from(layer: LayerNorm<B>) -> Self {
        Self::Layer(layer)
    }
}

impl<B: Backend> Normalization<B> {
    /// Applies normalization to a tensor.
    ///
    /// The normalization contract depends upon the wrapped norm layer;
    /// but all norm layers assume an input of at least rank 2;
    /// and produce an output of the same rank and shape.
    pub fn forward<const D: usize>(
        &self,
        input: Tensor<B, D>,
    ) -> Tensor<B, D> {
        match self {
            Normalization::Batch(norm) => norm.forward(input),
            Normalization::Group(norm) => norm.forward(input),
            Normalization::Instance(norm) => norm.forward(input),
            Normalization::Layer(norm) => norm.forward(input),
            // Normalization::Rms(norm) => norm.forward(input),
        }
    }

    /// Get the number of features.
    pub fn num_features(&self) -> usize {
        match self {
            Normalization::Batch(norm) => norm.gamma.shape().dims[0],
            Normalization::Group(norm) => norm.num_channels,
            Normalization::Instance(norm) => norm.num_channels,
            Normalization::Layer(norm) => norm.gamma.shape().dims[0],
        }
    }
}

#[cfg(feature = "std")]
#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::{Autodiff, NdArray};

    #[test]
    fn test_batch_norm() {
        type B = Autodiff<NdArray<f32>>;
        let device = Default::default();

        let num_features = 12;
        let input: Tensor<B, 4> = Tensor::ones([2, num_features, 3, 4], &device);

        let config: NormalizationConfig = BatchNormConfig::new(12).into();

        let layer: Normalization<B> = config.init(&device);

        let expected = match &layer {
            Normalization::Batch(inner) => inner.forward(input.clone()),
            _ => panic!("Unexpected layer type"),
        };

        let output = layer.forward(input);

        output.to_data().assert_eq(&expected.to_data(), true);
    }

    #[test]
    fn test_group_norm() {
        type B = Autodiff<NdArray<f32>>;
        let device = Default::default();

        let num_features = 12;
        let input: Tensor<B, 4> = Tensor::ones([2, num_features, 3, 4], &device);

        let config: NormalizationConfig = GroupNormConfig::new(3, num_features).into();

        let layer: Normalization<B> = config.init(&device);

        let expected = match &layer {
            Normalization::Group(inner) => inner.forward(input.clone()),
            _ => panic!("Unexpected layer type"),
        };

        let output = layer.forward(input);

        output.to_data().assert_eq(&expected.to_data(), true);
    }

    #[test]
    fn test_instance_norm() {
        type B = Autodiff<NdArray<f32>>;
        let device = Default::default();

        let num_features = 12;
        let input: Tensor<B, 4> = Tensor::ones([2, num_features, 3, 4], &device);

        let config: NormalizationConfig = InstanceNormConfig::new(num_features).into();

        let layer: Normalization<B> = config.init(&device);

        let expected = match &layer {
            Normalization::Instance(inner) => inner.forward(input.clone()),
            _ => panic!("Unexpected layer type"),
        };

        let output = layer.forward(input);

        output.to_data().assert_eq(&expected.to_data(), true);
    }

    #[test]
    fn test_layer_norm() {
        type B = Autodiff<NdArray<f32>>;
        let device = Default::default();

        let num_features = 12;
        let input: Tensor<B, 4> = Tensor::ones([2, 3, 4, num_features], &device);

        let config: NormalizationConfig = LayerNormConfig::new(num_features).into();

        let layer: Normalization<B> = config.init(&device);

        let expected = match &layer {
            Normalization::Layer(inner) => inner.forward(input.clone()),
            _ => panic!("Unexpected layer type"),
        };

        let output = layer.forward(input);

        output.to_data().assert_eq(&expected.to_data(), true);
    }
}
