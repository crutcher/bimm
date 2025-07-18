use crate::bindings::{MutableStackEnvironment, MutableStackMap, StackEnvironment, StackMap};
use crate::expressions::{DimExpr, TryMatchResult};
use crate::shape_argument::ShapeArgument;
use std::fmt::{Display, Formatter};

/// A term in a shape pattern.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DimMatcher<'a> {
    /// Matches any dimension size.
    Any {
        /// An optional label for the matcher.
        label: Option<&'a str>,
    },

    /// Matches a variable number of dimensions (ellipsis).
    Ellipsis {
        /// An optional label for the matcher.
        label: Option<&'a str>,
    },

    /// A dimension size expression that must match a specific value.
    Expr {
        /// An optional label for the matcher.
        label: Option<&'a str>,

        /// The dimension expression that must match a specific value.
        expr: DimExpr<'a>,
    },
}

impl<'a> DimMatcher<'a> {
    /// Create a new `DimMatcher` that matches any dimension size.
    pub const fn any() -> Self {
        DimMatcher::Any { label: None }
    }

    /// Create a new `DimMatcher` that matches a variable number of dimensions (ellipsis).
    pub const fn ellipsis() -> Self {
        DimMatcher::Ellipsis { label: None }
    }

    /// Create a new `DimMatcher` from a dimension expression.
    ///
    /// ## Arguments
    ///
    /// - `expr`: a dimension expression that must match a specific value.
    ///
    /// ## Returns
    ///
    /// A new `DimMatcher` that matches the given expression.
    pub const fn expr(expr: DimExpr<'a>) -> Self {
        DimMatcher::Expr { label: None, expr }
    }

    /// Get the label of the matcher, if any.
    pub const fn label(&self) -> Option<&'a str> {
        match self {
            DimMatcher::Any { label } => *label,
            DimMatcher::Ellipsis { label } => *label,
            DimMatcher::Expr { label, .. } => *label,
        }
    }

    /// Attach a label to the matcher.
    ///
    /// ## Arguments
    ///
    /// - `label`: an optional label to attach to the matcher.
    ///
    /// ## Returns
    ///
    /// A new `DimMatcher` with the label attached.
    pub const fn with_label(
        self,
        label: Option<&'a str>,
    ) -> Self {
        match self {
            DimMatcher::Any { .. } => DimMatcher::Any { label },
            DimMatcher::Ellipsis { .. } => DimMatcher::Ellipsis { label },
            DimMatcher::Expr { expr, .. } => DimMatcher::Expr { label, expr },
        }
    }
}

impl Display for DimMatcher<'_> {
    fn fmt(
        &self,
        f: &mut Formatter<'_>,
    ) -> std::fmt::Result {
        if let Some(label) = self.label() {
            write!(f, "{label}: ")?;
        }
        match self {
            DimMatcher::Any { label: _ } => write!(f, "_"),
            DimMatcher::Ellipsis { label: _ } => write!(f, "..."),
            DimMatcher::Expr { label: _, expr } => write!(f, "{expr}"),
        }
    }
}

/// A shape pattern, which is a sequence of terms that can match a shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShapeContract<'a> {
    /// The terms in the pattern.
    pub terms: &'a [DimMatcher<'a>],

    /// The position of the ellipsis in the pattern, if any.
    pub ellipsis_pos: Option<usize>,
}

impl Display for ShapeContract<'_> {
    fn fmt(
        &self,
        f: &mut Formatter<'_>,
    ) -> std::fmt::Result {
        write!(f, "[")?;
        for (idx, expr) in self.terms.iter().enumerate() {
            if idx > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{expr}")?;
        }
        write!(f, "]")
    }
}

impl<'a> ShapeContract<'a> {
    /// Create a new shape pattern from a slice of terms.
    ///
    /// ## Arguments
    ///
    /// - `terms`: a slice of `ShapePatternTerm` that defines the pattern.
    ///
    /// ## Returns
    ///
    /// A new `ShapePattern` instance.
    pub const fn new(terms: &'a [DimMatcher<'a>]) -> Self {
        let mut i = 0;
        let mut ellipsis_pos: Option<usize> = None;

        while i < terms.len() {
            if matches!(terms[i], DimMatcher::Ellipsis { label: _ }) {
                match ellipsis_pos {
                    Some(_) => panic!("Multiple ellipses in pattern"),
                    None => ellipsis_pos = Some(i),
                }
            }
            i += 1;
        }

        ShapeContract {
            terms,
            ellipsis_pos,
        }
    }

    /// Assert that the shape matches the pattern.
    ///
    /// ## Arguments
    ///
    /// - `shape`: the shape to match.
    /// - `env`: the params which are already bound.
    ///
    /// ## Panics
    ///
    /// If the shape does not match the pattern, or if there is a conflict in the bindings.
    #[inline(always)]
    pub fn assert_shape<S>(
        &'a self,
        shape: S,
        env: StackEnvironment<'a>,
    ) where
        S: ShapeArgument,
    {
        let result = self.maybe_assert_shape(shape, env);
        if result.is_err() {
            panic!("{}", result.unwrap_err());
        }
    }

    /// Assert that the shape matches the pattern.
    ///
    /// ## Arguments
    ///
    /// - `shape`: the shape to match.
    /// - `env`: the params which are already bound.
    ///
    /// ## Returns
    ///
    /// - `Ok(())`: if the shape matches the pattern.
    /// - `Err(String)`: if the shape does not match the pattern, with an error message.
    #[inline(always)]
    pub fn maybe_assert_shape<S>(
        &'a self,
        shape: S,
        env: StackEnvironment<'a>,
    ) -> Result<(), String>
    where
        S: ShapeArgument,
    {
        let mut mut_env = MutableStackEnvironment::new(env);

        self.resolve_match(shape, &mut mut_env)
    }

    /// Match and unpack a shape pattern.
    ///
    /// Wraps `maybe_unpack_shape` and panics if the shape does not match.
    ///
    /// ## Arguments
    ///
    /// - `shape`: the shape to match.
    /// - `keys`: the bound keys to export.
    /// - `env`: the params which are already bound.
    ///
    /// ## Returns
    ///
    /// The list of key values.
    ///
    /// ## Panics
    ///
    /// If the shape does not match the pattern, or if there is a conflict in the bindings.
    #[must_use]
    #[inline(always)]
    pub fn unpack_shape<S, const K: usize>(
        &'a self,
        shape: S,
        keys: &[&'a str; K],
        env: StackEnvironment<'a>,
    ) -> [usize; K]
    where
        S: ShapeArgument,
    {
        match self.maybe_unpack_shape(shape, keys, env) {
            Ok(values) => values,
            Err(msg) => panic!("{msg}"),
        }
    }

    /// Match and unpack a shape pattern.
    ///
    /// ## Arguments
    ///
    /// - `shape`: the shape to match.
    /// - `keys`: the bound keys to export.
    /// - `env`: the params which are already bound.
    ///
    /// ## Returns
    ///
    /// Either the list of key values; or an error.
    #[must_use]
    #[inline(always)]
    pub fn maybe_unpack_shape<S, const K: usize>(
        &'a self,
        shape: S,
        keys: &[&'a str; K],
        env: StackEnvironment<'a>,
    ) -> Result<[usize; K], String>
    where
        S: ShapeArgument,
    {
        let mut mut_env = MutableStackEnvironment::new(env);

        self.resolve_match(shape, &mut mut_env)?;

        Ok(mut_env.export_key_values(keys))
    }

    /// Resolve the match for the shape against the pattern.
    ///
    /// ## Arguments
    ///
    /// - `shape`: the shape to match.
    /// - `env`: the mutable environment to bind parameters.
    ///
    /// ## Returns
    ///
    /// - `Ok(())`: if the shape matches the pattern.
    /// - `Err(String)`: if the shape does not match the pattern, with an error message.
    #[must_use]
    #[inline(always)]
    fn resolve_match<S>(
        &'a self,
        shape: S,
        env: &mut MutableStackEnvironment<'a>,
    ) -> Result<(), String>
    where
        S: ShapeArgument,
    {
        let shape = &shape.get_shape().dims;

        let fail = |msg| -> String {
            format!(
                "Shape Error:: {msg}\n shape:\n  {shape:?}\n expected:\n  {self}\n  {{{}}}",
                env.backing
                    .iter()
                    .map(|(k, v)| format!("\"{k}\": {v}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        };
        let fail_at = |shape_idx, term_idx, msg| -> String {
            fail(format!(
                "{} !~ {} :: {msg}",
                shape[shape_idx], self.terms[term_idx]
            ))
        };

        let rank = shape.len();

        let (e_start, e_size) = match self.check_ellipsis_split(rank) {
            Ok((e_start, e_size)) => (e_start, e_size),
            Err(msg) => return Err(fail(msg)),
        };

        for (shape_idx, &dim_size) in shape.iter().enumerate() {
            let term_idx = if shape_idx < e_start {
                shape_idx
            } else if shape_idx < (e_start + e_size) {
                continue;
            } else {
                shape_idx + 1 - e_size
            };

            let expr = match &self.terms[term_idx] {
                DimMatcher::Any { label: _ } => continue,
                DimMatcher::Ellipsis { label: _ } => {
                    unreachable!("Ellipsis should have been handled before");
                }
                DimMatcher::Expr { label: _, expr } => expr,
            };

            match expr.try_match(dim_size as isize, env) {
                Ok(TryMatchResult::Match) => continue,
                Ok(TryMatchResult::Conflict) => {
                    return Err(fail_at(shape_idx, term_idx, "Value MissMatch".to_string()));
                }
                Ok(TryMatchResult::ParamConstraint(param_name, value)) => {
                    env.bind(param_name, value as usize);
                }
                Err(msg) => return Err(fail_at(shape_idx, term_idx, msg)),
            }
        }

        Ok(())
    }

    /// Check if the pattern has an ellipsis.
    ///
    /// ## Arguments
    ///
    /// - `rank`: the number of dims of the shape to match.
    ///
    /// ## Returns
    ///
    /// - `Ok((usize, usize))`: the position of the ellipsis and the number of dimensions it matches.
    /// - `Err(String)`: an error message if the pattern does not match the expected size.
    #[inline(always)]
    #[must_use]
    fn check_ellipsis_split(
        &self,
        rank: usize,
    ) -> Result<(usize, usize), String> {
        let k = self.terms.len();
        match self.ellipsis_pos {
            None => {
                if rank != k {
                    Err(format!("Shape rank {rank} != pattern dim count {k}",))
                } else {
                    Ok((k, 0))
                }
            }
            Some(pos) => {
                let non_ellipsis_terms = k - 1;
                if rank < non_ellipsis_terms {
                    return Err(format!(
                        "Shape rank {rank} < non-ellipsis pattern term count {non_ellipsis_terms}",
                    ));
                }
                Ok((pos, rank - non_ellipsis_terms))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contracts::{DimMatcher, ShapeContract};

    #[should_panic(expected = "Multiple ellipses in pattern")]
    #[test]
    fn test_bad_new() {
        // Multiple ellipses in pattern should panic.
        let _ = ShapeContract::new(&[
            DimMatcher::any(),
            DimMatcher::ellipsis(),
            DimMatcher::ellipsis(),
        ]);
    }
    #[test]
    fn test_check_ellipsis_split() {
        {
            // With ellipsis.
            static PATTERN: ShapeContract = ShapeContract::new(&[
                DimMatcher::any(),
                DimMatcher::ellipsis(),
                DimMatcher::expr(DimExpr::Param("b")),
            ]);

            assert_eq!(PATTERN.check_ellipsis_split(2), Ok((1, 0)));
            assert_eq!(PATTERN.check_ellipsis_split(3), Ok((1, 1)));
            assert_eq!(PATTERN.check_ellipsis_split(4), Ok((1, 2)));

            assert_eq!(
                PATTERN.check_ellipsis_split(1),
                Err("Shape rank 1 < non-ellipsis pattern term count 2".to_string())
            );
        }
        {
            // Without ellipsis.
            static PATTERN: ShapeContract =
                ShapeContract::new(&[DimMatcher::any(), DimMatcher::expr(DimExpr::Param("b"))]);

            assert_eq!(PATTERN.check_ellipsis_split(2), Ok((2, 0)));

            assert_eq!(
                PATTERN.check_ellipsis_split(1),
                Err("Shape rank 1 != pattern dim count 2".to_string())
            );
        }
    }

    #[test]
    fn test_format_pattern() {
        static PATTERN: ShapeContract = ShapeContract::new(&[
            DimMatcher::any(),
            DimMatcher::ellipsis(),
            DimMatcher::expr(DimExpr::Param("b")),
            DimMatcher::expr(DimExpr::Prod(&[
                DimExpr::Param("h"),
                DimExpr::Sum(&[DimExpr::Param("a"), DimExpr::Negate(&DimExpr::Param("b"))]),
            ])),
            DimMatcher::expr(DimExpr::Pow(&DimExpr::Param("h"), 2)),
        ]);

        assert_eq!(PATTERN.to_string(), "[_, ..., b, (h*(a+(-b))), (h)^2]");
    }

    #[test]
    fn test_panic_msg() {
        static CONTRACT: ShapeContract = ShapeContract::new(&[
            DimMatcher::any(),
            DimMatcher::expr(DimExpr::Param("b")),
            DimMatcher::ellipsis(),
            DimMatcher::expr(DimExpr::Prod(&[DimExpr::Param("h"), DimExpr::Param("p")]))
                .with_label(Some("height")),
            DimMatcher::expr(DimExpr::Prod(&[DimExpr::Param("w"), DimExpr::Param("p")]))
                .with_label(Some("width")),
            DimMatcher::expr(DimExpr::Pow(&DimExpr::Param("z"), 3)),
            DimMatcher::expr(DimExpr::Param("c")),
        ]);

        let b = 2;
        let h = 3;
        let w = 2;
        let p = 4;
        let c = 5;
        let z = 4;

        let shape = [12, b, 1, 2, 3, h * p, w * p, 1 + z * z * z, c];

        let result =
            CONTRACT.maybe_unpack_shape(&shape, &["b", "h", "w", "z"], &[("p", p), ("c", c)]);
        assert!(result.is_err());
        let err_msg = result.unwrap_err();
        assert_eq!(
            err_msg,
            "\
Shape Error:: 65 !~ (z)^3 :: No integer solution.
 shape:
  [12, 2, 1, 2, 3, 12, 8, 65, 5]
 expected:
  [_, b, ..., height: (h*p), width: (w*p), (z)^3, c]
  {\"p\": 4, \"c\": 5}"
        );
    }

    #[test]
    fn test_unpack_shape() {
        static CONTRACT: ShapeContract = ShapeContract::new(&[
            DimMatcher::any(),
            DimMatcher::expr(DimExpr::Param("b")),
            DimMatcher::ellipsis(),
            DimMatcher::expr(DimExpr::Prod(&[DimExpr::Param("h"), DimExpr::Param("p")])),
            DimMatcher::expr(DimExpr::Prod(&[DimExpr::Param("w"), DimExpr::Param("p")])),
            DimMatcher::expr(DimExpr::Pow(&DimExpr::Param("z"), 3)),
            DimMatcher::expr(DimExpr::Param("c")),
        ]);

        let b = 2;
        let h = 3;
        let w = 2;
        let p = 4;
        let c = 5;
        let z = 4;

        let shape = [12, b, 1, 2, 3, h * p, w * p, z * z * z, c];
        let env = [("p", p), ("c", c)];

        CONTRACT.assert_shape(&shape, &env);

        let [u_b, u_h, u_w, u_z] = CONTRACT.unpack_shape(&shape, &["b", "h", "w", "z"], &env);

        assert_eq!(u_b, b);
        assert_eq!(u_h, h);
        assert_eq!(u_w, w);
        assert_eq!(u_z, z);
    }

    #[should_panic(expected = "Shape rank 3 != pattern dim count 1")]
    #[test]
    fn test_shape_mismatch_no_ellipsis() {
        // This should panic because the shape does not match the pattern.
        static PATTERN: ShapeContract =
            ShapeContract::new(&[DimMatcher::expr(DimExpr::Param("a"))]);
        let shape = [1, 2, 3];
        PATTERN.assert_shape(&shape, &[]);
    }

    #[should_panic(expected = "Shape rank 3 < non-ellipsis pattern term count 4")]
    #[test]
    fn test_shape_mismatch_with_ellipsis() {
        // This should panic because the shape does not match the pattern.
        static PATTERN: ShapeContract = ShapeContract::new(&[
            DimMatcher::any(),
            DimMatcher::any(),
            DimMatcher::ellipsis(),
            DimMatcher::expr(DimExpr::Param("b")),
            DimMatcher::expr(DimExpr::Param("c")),
        ]);
        let shape = [1, 2, 3];
        PATTERN.assert_shape(&shape, &[]);
    }

    #[should_panic(expected = "Value MissMatch")]
    #[test]
    fn test_shape_mismatch_value() {
        // This should panic because the value does not match the constraint.
        static PATTERN: ShapeContract = ShapeContract::new(&[
            DimMatcher::expr(DimExpr::Param("a")),
            DimMatcher::expr(DimExpr::Param("b")),
        ]);
        let shape = [2, 3];
        PATTERN.assert_shape(&shape, &[("a", 2), ("b", 4)]);
    }
}
