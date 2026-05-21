//! Private module for internal use in the `bimm` crate.
use std::fmt::Debug;

use num_traits::float::Float;

pub fn assert_close_to_vec<T>(
    actual: &[T],
    expected: &[T],
    tolerance: T,
) where
    T: Float + std::ops::Sub<Output = T> + std::ops::Add<Output = T> + Copy + Debug,
{
    let mut pass = actual.len() == expected.len();
    for (&a, &e) in actual.iter().zip(expected.iter()) {
        if !pass {
            break;
        }
        if (a - e).abs() > tolerance {
            pass = false;
            break;
        }
    }
    if !pass {
        panic!("Expected (+/- {tolerance:?}):\n{expected:?}\nActual:\n{actual:?}");
    }
}

#[cfg(test)]
mod tests {
    use crate::testing::assert_close_to_vec;

    #[test]
    fn test_assert_close_to_vec() {
        let actual = vec![1.0, 2.0, 3.0];
        let expected = vec![1.0, 2.0, 3.0];
        assert_close_to_vec(&actual, &expected, 0.01);

        let actual = vec![1.0, 2.0, 3.1];
        let expected = vec![1.0, 2.0, 3.0];
        assert_close_to_vec(&actual, &expected, 0.2);
    }

    #[test]
    #[should_panic]
    fn test_assert_close_to_vec_bad_values() {
        let actual = vec![1.0, 2.0, 3.0];
        let expected = vec![1.0, 2.0, 3.5];
        assert_close_to_vec(&actual, &expected, 0.01);
    }

    #[test]
    #[should_panic]
    fn test_assert_close_to_vec_different_lengths() {
        let actual = vec![1.0, 2.0];
        let expected = vec![1.0, 2.0, 3.0];
        assert_close_to_vec(&actual, &expected, 0.01);
    }
}
