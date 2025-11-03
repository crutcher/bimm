//! # Probability Utilities.

use anyhow::bail;
use core::fmt::Debug;
use num_traits::Float;

/// Validate a probability in the range ``[0.0, 1.0]``.
///
/// # Arguments
///
/// - `prob`: the prob to check.
///
/// # Returns
///
/// An `anyhow::Result<prob>`
#[inline]
pub fn try_probability<F: Float + Debug>(prob: F) -> anyhow::Result<F> {
    if prob < F::zero() || prob > F::one() {
        bail!("probability must be in [0.0, 1.0]: {prob:?}");
    }
    Ok(prob)
}

/// Expect a probability to be in range ``[0.0, 1.0]``, or panic.
///
/// # Arguments
///
/// - `prob`: the prob to check.
///
/// # Returns
///
/// `prob`.
///
/// # Panics
///
/// On range error.
#[inline]
pub fn expect_probability<F: Float + Debug>(prob: F) -> F {
    match try_probability(prob) {
        Ok(prob) => prob,
        Err(e) => panic!("{}", e),
    }
}

#[cfg(test)]
mod tests {
    use crate::utility::probability::{expect_probability, try_probability};

    #[test]
    fn test_probability() {
        assert_eq!(expect_probability(0f32), 0f32);
        assert_eq!(expect_probability(1f32), 1f32);
        assert_eq!(expect_probability(0.5f32), 0.5f32);

        assert_eq!(expect_probability(0f64), 0f64);
        assert_eq!(expect_probability(1f64), 1f64);
        assert_eq!(expect_probability(0.5f64), 0.5f64);

        assert!(try_probability(-1.0f32).is_err());
        assert!(try_probability(2.0f32).is_err());

        assert!(try_probability(-1.0f64).is_err());
        assert!(try_probability(2.0f64).is_err());
    }

    #[should_panic(expected = "probability must be in [0.0, 1.0]: -1.0")]
    #[test]
    fn test_probability_panic() {
        expect_probability(-1.0);
    }
}
