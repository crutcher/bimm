//! Burn implementation of the `DropPath` (Stochastic Depth) regularization
//! layer.
//!
//! Papers:
//! `DropBlock`: A regularization method for convolutional networks (<https://arxiv.org/abs/1810.12890>)
//!
//! Deep Networks with Stochastic Depth (<https://arxiv.org/abs/1603.09382>)
//!
//! Inspired by the python implementation from the timm library:
//! <https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/drop.py>

use burn::{
    config::Config,
    module::Module,
    prelude::{
        Backend,
        Tensor,
    },
    tensor::Distribution,
};

use crate::utility::probability;

/// `DropPath` (stochastic depth) regularization.
///
/// # Arguments
///
/// * `x`: Input tensor.
/// * `drop_prob`: Probability of dropping a path.
/// * `training`: Whether the model is in training mode.
/// * `scale_by_keep`: Whether to scale the output by `1 / (1 - drop_prob)`
///
/// # Returns
///
/// * Output tensor with the same shape as the input tensor.
#[must_use]
pub fn drop_path<B: Backend, const D: usize>(
    x: Tensor<B, D>,
    drop_prob: f64,
    training: bool,
    scale_by_keep: bool,
) -> Tensor<B, D> {
    _drop_path_sample(
        x,
        drop_prob,
        training,
        scale_by_keep,
        |shape, keep_prob, device| {
            Tensor::<B, D>::random(shape, Distribution::Bernoulli(keep_prob), device)
        },
    )
}

/// Internal implementation of `DropPath`.
///
/// Deferred to a separate function to allow for testing sampling.
///
/// # Arguments
///
/// * `x`: Input tensor.
/// * `drop_prob`: Probability of dropping a path.
/// * `training`: Whether the model is in training mode.
/// * `scale_by_keep`: Whether to scale the output by `1 / (1 - drop_prob)`
/// * `sample`: Sampling function to generate the random tensor.
///
/// # Returns
///
/// * Output tensor with the same shape as the input tensor.
#[inline(always)]
#[must_use]
fn _drop_path_sample<B: Backend, const D: usize>(
    x: Tensor<B, D>,
    drop_prob: f64,
    training: bool,
    scale_by_keep: bool,
    sample: fn([usize; D], f64, &B::Device) -> Tensor<B, D>,
) -> Tensor<B, D> {
    probability::expect_probability(drop_prob);

    if !training || drop_prob == 0.0 {
        return x;
    }

    let keep_prob = 1.0 - drop_prob;

    let mut shape = [1; D];
    shape[0] = x.dims()[0];

    let random_tensor = sample(shape, keep_prob, &x.device());

    let random_tensor = if keep_prob > 0.0 && scale_by_keep {
        random_tensor.div_scalar(keep_prob)
    } else {
        random_tensor
    };

    x * random_tensor
}

/// Common introspection interface for `DropPath` modules.
pub trait DropPathMeta {
    /// Returns the drop probability.
    fn drop_prob(&self) -> f64;

    /// Returns the keep probability, which is `1.0 - drop_prob`.
    fn keep_prob(&self) -> f64 {
        1.0 - self.drop_prob()
    }

    /// Returns whether the output is scaled by `1 / (1 - drop_prob)`.
    fn scale_by_keep(&self) -> bool;
}

/// Configuration for the `DropPath` module.
#[derive(Config, Debug)]
pub struct DropPathConfig {
    /// Probability of dropping a path.
    #[config(default = 0.0)]
    pub drop_prob: f64,

    /// Whether to scale the output by `1 / (1 - drop_prob)`.
    #[config(default = true)]
    pub scale_by_keep: bool,
}

impl DropPathMeta for DropPathConfig {
    fn drop_prob(&self) -> f64 {
        self.drop_prob
    }

    fn scale_by_keep(&self) -> bool {
        self.scale_by_keep
    }
}

impl DropPathConfig {
    /// Initializes a new `DropPath` module.
    #[inline(always)]
    #[must_use]
    pub fn init(&self) -> DropPath {
        DropPath {
            drop_prob: probability::expect_probability(self.drop_prob),
            scale_by_keep: self.scale_by_keep,
        }
    }
}

/// The `DropPath` module.
///
/// Burn Module that implements the `DropPath` (Stochastic Depth)
/// regularization.
#[derive(Module, Clone, Debug)]
pub struct DropPath {
    /// Probability of dropping a path.
    pub drop_prob: f64,

    /// Whether to scale the output by `1 / (1 - drop_prob)`.
    pub scale_by_keep: bool,
}

impl DropPathMeta for DropPath {
    fn drop_prob(&self) -> f64 {
        self.drop_prob
    }

    fn scale_by_keep(&self) -> bool {
        self.scale_by_keep
    }
}

impl DropPath {
    /// Applies `drop_path` pass on the input tensor.
    #[must_use]
    pub fn forward<B: Backend, const D: usize>(
        &self,
        input: Tensor<B, D>,
    ) -> Tensor<B, D> {
        let training = B::ad_enabled(&input.device());
        drop_path(input, self.drop_prob, training, self.scale_by_keep)
    }

    /// Applies an inner function under conditional stochastic
    /// residual/depth-skip connection.
    ///
    /// This is used for stochastic depth in the transformer block.
    ///
    /// # Parameters
    ///
    /// * `x` - Input tensor of shape (B, D).
    /// * `f` - Function to apply on the input tensor.
    ///
    /// # Returns
    ///
    /// The result of the function application, with a stochastic skip
    /// connection applied.
    #[inline]
    #[must_use]
    pub fn with_skip<B: Backend, const D: usize, F>(
        &self,
        x: Tensor<B, D>,
        f: F,
    ) -> Tensor<B, D>
    where
        F: FnOnce(Tensor<B, D>) -> Tensor<B, D>,
    {
        x.clone() + self.forward(f(x))
    }
}

#[cfg(test)]
mod tests {
    use bunsen::support::testing::PerfTestBackend;
    use burn::{
        prelude::Tensor,
        tensor::Distribution,
    };

    use super::*;

    #[test]
    fn test_drop_path() {
        type B = PerfTestBackend;
        let device = Default::default();
        let drop_prob = 0.5;
        let scale_by_keep = true;

        let config = DropPathConfig {
            drop_prob,
            scale_by_keep,
        };

        let module = config.init();

        let input = Tensor::<B, 4>::random([2, 3, 4, 5], Distribution::Uniform(0.0, 1.0), &device);
        let output = module.forward(input.clone());

        assert_eq!(input.dims(), output.dims());
    }

    #[test]
    fn test_drop_path_wrapper() {
        type B = PerfTestBackend;
        let device = Default::default();

        let n = 3;
        let shape = [n, 2, 4];

        let x = Tensor::<B, 3>::random(shape, Distribution::Uniform(0.0, 1.0), &device);

        // No-op case: not training and drop_prob = 0.0
        let training = false;
        let drop_prob = 0.0;
        let scale_by_keep = false;
        let res = drop_path(x.clone(), drop_prob, training, scale_by_keep);
        assert_eq!(res.dims(), x.dims());
    }

    #[test]
    fn test_drop_path_sample() {
        type B = PerfTestBackend;
        let device = Default::default();

        let n = 3;
        let shape = [n, 2, 4];

        let x = Tensor::<B, 3>::random(shape, Distribution::Uniform(0.0, 1.0), &device);

        // No-op case: not training and drop_prob = 0.0
        let training = false;
        let drop_prob = 0.0;
        let scale_by_keep = false;
        let res = _drop_path_sample(
            x.clone(),
            drop_prob,
            training,
            scale_by_keep,
            |shape, keep_prob, device| {
                assert_eq!(shape, [3, 1, 1]);
                assert_eq!(keep_prob, 1.0);
                Tensor::<B, 3>::from_data([[[1.0]], [[0.0]], [[1.0]]], device)
            },
        );
        res.to_data().assert_eq(&x.clone().to_data(), true);

        // No-op case: training, but drop_prob = 0.0
        let training = true;
        let drop_prob = 0.0;
        let scale_by_keep = false;
        let res = _drop_path_sample(
            x.clone(),
            drop_prob,
            training,
            scale_by_keep,
            |shape, keep_prob, device| {
                assert_eq!(shape, [3, 1, 1]);
                assert_eq!(keep_prob, 1.0);
                Tensor::<B, 3>::from_data([[[1.0]], [[0.0]], [[1.0]]], device)
            },
        );
        res.to_data().assert_eq(&x.clone().to_data(), true);

        // Training, but no scaling
        let training = true;
        let drop_prob = 0.5;
        let scale_by_keep = false;
        let res = _drop_path_sample(
            x.clone(),
            drop_prob,
            training,
            scale_by_keep,
            |shape, keep_prob, device| {
                assert_eq!(shape, [3, 1, 1]);
                assert_eq!(keep_prob, 0.5);
                Tensor::<B, 3>::from_data([[[1.0]], [[0.0]], [[1.0]]], device)
            },
        );
        res.to_data().assert_eq(
            &(x.clone() * Tensor::<B, 3>::from_data([[[1.0]], [[0.0]], [[1.0]]], &device))
                .to_data(),
            true,
        );

        // Training, with scaling
        let training = true;
        let drop_prob = 0.5;
        let keep_prob = 1.0 - drop_prob;
        let scale_by_keep = true;
        let res = _drop_path_sample(
            x.clone(),
            drop_prob,
            training,
            scale_by_keep,
            |shape, keep_prob, device| {
                assert_eq!(shape, [3, 1, 1]);
                assert_eq!(keep_prob, 0.5);
                Tensor::<B, 3>::from_data([[[1.0]], [[0.0]], [[1.0]]], device)
            },
        );
        res.to_data().assert_eq(
            &(x.clone() * Tensor::<B, 3>::from_data([[[1.0]], [[0.0]], [[1.0]]], &device))
                .div_scalar(keep_prob)
                .to_data(),
            true,
        );
    }

    #[test]
    fn test_droppath_module() {
        type B = PerfTestBackend;

        let drop_prob = 0.2;
        let config = DropPathConfig::new().with_drop_prob(drop_prob);

        assert_eq!(config.drop_prob(), 0.2);
        assert_eq!(config.keep_prob(), 1.0 - drop_prob);
        assert!(config.scale_by_keep());

        let module = config.init();
        assert_eq!(module.drop_prob(), 0.2);
        assert_eq!(module.keep_prob(), 1.0 - drop_prob);
        assert!(module.scale_by_keep());

        let device = Default::default();
        let shape = [2, 3, 4];
        let x = Tensor::<B, 3>::random(shape, Distribution::Uniform(0.0, 1.0), &device);

        // TODO(crutcher): work out how to enable/disable training mode in tests.
        let output = module.forward(x.clone());
        assert_eq!(x.dims(), output.dims());
    }
}
