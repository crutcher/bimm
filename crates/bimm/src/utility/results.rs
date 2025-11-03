//! # Result Utilities
//!
//! Methods for [`std::result::Result`] manipulation.

use core::fmt::Debug;

/// Unwraps Result, or Panics.
///
/// Unlike the `.unwrap()` method, this does not add a prefix about
/// `.unwrap()`
///
/// Useful for building ``expect_<func>(...) -> T`` variants of
/// ``try_<func>(...) -> Result<T, E>`` methods.
pub fn expect_unwrap<T, E>(result: Result<T, E>) -> T
where
    E: Debug,
{
    match result {
        Ok(t) => t,
        Err(e) => panic!("{:?}", e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::bail;

    fn try_example(
        value: i32,
        throw: bool,
    ) -> anyhow::Result<i32> {
        if throw { bail!("throwing") } else { Ok(value) }
    }

    #[test]
    fn test_expect_unwrap() {
        assert_eq!(expect_unwrap(try_example(42, false)), 42);
    }

    #[should_panic(expected = "throwing")]
    #[test]
    fn test_expect_unwrap_panic() {
        expect_unwrap(try_example(42, true));
    }
}
