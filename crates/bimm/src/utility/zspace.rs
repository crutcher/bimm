//! # Z-Space utilities.
//!
//! Z-Space is frequently used to define the semantics of n-dimensional
//! integer coordinate systems.
//!
//! Z-Space refers to n-dimensional spaces indexed by integer tuples.
//! It is Manhattan / Taxi-Cab Space, with the addition of a partial ordering.
//!
//! Z-Space has a limited notion of regions; limited to axis-aligned
//! orthogonal regions. The partial ordering is chosen to simplify
//! the description and containment testing of these regions.
use crate::utility::results::expect_unwrap;
use anyhow::bail;
use core::cmp::Ordering;
use core::fmt::Debug;

/// Z-space `PartialOrd`
///
/// Compares the partial ordering of two slices (of equal length)
/// by z-space tuple dominance.
///
/// For example, the following orderings would hold:
/// * ``cmp([1, 2], [1, 2]) == Some(Ordering::Equal)``
/// * ``cmp([0, 0], [0, 1]) == Some(Ordering::Less)``
/// * ``cmp([1, 0], [0, 0]) == Some(Ordering::Greater)``
/// * ``cmp([0, 0], [1, 1]) == Some(Ordering::Less)``
/// * ``cmp([1, 0], [0, 1]) == None``
///
/// # Arguments
///
/// - `a`: the first slice to compare.
/// - `b`: the second slice to compare.
///
/// # Returns
///
/// An `Option<Ordering>`, where `None` represents incomparable.
///
/// # Panics
///
/// If called on slices of unequal lengths.
pub fn zspace_partial_cmp<T: PartialOrd>(
    a: &[T],
    b: &[T],
) -> Option<Ordering> {
    assert_eq!(
        a.len(),
        b.len(),
        "length mismatch: {} != {}",
        a.len(),
        b.len()
    );
    let mut ord = Ordering::Equal;
    for (ai, bi) in a.iter().zip(b.iter()) {
        match ai.partial_cmp(bi) {
            None => return None,
            Some(Ordering::Equal) => (),
            Some(Ordering::Less) => {
                if ord == Ordering::Greater {
                    return None;
                }
                ord = Ordering::Less;
            }
            Some(Ordering::Greater) => {
                if ord == Ordering::Less {
                    return None;
                }
                ord = Ordering::Greater;
            }
        }
    }
    Some(ord)
}

/// Check if a `point` is in the half-open range ``[start, end)``
///
/// # Returns
///
/// An `anyhow::Result<()>` that is `Ok(())` if the point is in the range,
/// and a formatted bounds error otherwise.
pub fn try_point_bounds_check<T>(
    point: &[T],
    start: &[T],
    end: &[T],
) -> anyhow::Result<()>
where
    T: PartialOrd + Debug,
{
    if !matches!(
        zspace_partial_cmp(start, point),
        Some(Ordering::Less) | Some(Ordering::Equal)
    ) || zspace_partial_cmp(point, end) != Some(Ordering::Less)
    {
        bail!("{point:?} is not in [ {start:?}, {end:?} )");
    }
    Ok(())
}

/// Expects that a `point` is in the half-open range ``[start, end)``
#[allow(dead_code)]
pub fn expect_point_bounds_check<T>(
    point: &[T],
    start: &[T],
    end: &[T],
) where
    T: PartialOrd + Debug,
{
    expect_unwrap(try_point_bounds_check(point, start, end))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[should_panic(expected = "length mismatch: 2 != 3")]
    #[test]
    fn test_zspace_partial_cmp_panic() {
        zspace_partial_cmp(&[2, 3], &[2, 3, 4]);
    }

    #[test]
    fn test_zspace_partial_cmp() {
        assert_eq!(zspace_partial_cmp(&[2, 3], &[2, 3]), Some(Ordering::Equal));
        assert_eq!(zspace_partial_cmp(&[2, 3], &[2, 4]), Some(Ordering::Less));
        assert_eq!(
            zspace_partial_cmp(&[2, 3], &[2, 2]),
            Some(Ordering::Greater)
        );
        assert_eq!(zspace_partial_cmp(&[4, 3], &[2, 4]), None);

        assert_eq!(
            zspace_partial_cmp(&[2.0, 3.0], &[2.0, 3.0]),
            Some(Ordering::Equal)
        );
        assert_eq!(
            zspace_partial_cmp(&[2.0, 3.0], &[2.0, 4.0]),
            Some(Ordering::Less)
        );
        assert_eq!(
            zspace_partial_cmp(&[2.0, 3.0], &[2.0, 2.0]),
            Some(Ordering::Greater)
        );
        assert_eq!(zspace_partial_cmp(&[4.0, 3.0], &[2.0, 4.0]), None);
    }

    #[test]
    fn test_zspace_bounds_check() {
        assert!(try_point_bounds_check(&[0, 0], &[0, 0], &[2, 3]).is_ok());
        assert!(try_point_bounds_check(&[0, 1], &[0, 0], &[2, 3]).is_ok());
        assert!(try_point_bounds_check(&[1, 0], &[0, 0], &[2, 3]).is_ok());
        assert!(try_point_bounds_check(&[1, 2], &[0, 0], &[2, 3]).is_ok());

        assert!(
            try_point_bounds_check(&[-1, 2], &[0, 0], &[2, 3])
                .unwrap_err()
                .to_string()
                .contains("[-1, 2] is not in [ [0, 0], [2, 3] )")
        );
    }
}
