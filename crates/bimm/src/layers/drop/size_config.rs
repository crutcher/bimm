use serde::{Deserialize, Serialize};

/// Helper option to describe the size of a wrapper, relative to a wrapped object.
///
/// TODO: point this at `burn::...::SizeConfig` in "0.19.0"
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub enum SizeConfig {
    /// Use the size of the source dataset.
    #[default]
    Default,

    /// Use the size as a ratio of the source dataset size.
    ///
    /// Must be >= 0.
    Ratio(f64),

    /// Use a fixed size.
    Fixed(usize),
}

impl SizeConfig {
    /// Construct a source which will have the same size as the source dataset.
    pub fn source() -> Self {
        Self::Default
    }

    /// Resolve the effective size.
    ///
    /// ## Arguments
    ///
    /// - `source_size`: the size of the source dataset.
    ///
    /// ## Returns
    ///
    /// The resolved size of the wrapper dataset.
    pub fn resolve(
        self,
        source_size: usize,
    ) -> usize {
        match self {
            SizeConfig::Default => source_size,
            SizeConfig::Ratio(ratio) => {
                assert!(ratio >= 0.0, "Ratio must be positive: {ratio}");
                ((source_size as f64) * ratio) as usize
            }
            SizeConfig::Fixed(size) => size,
        }
    }
}

impl From<usize> for SizeConfig {
    fn from(size: usize) -> Self {
        Self::Fixed(size)
    }
}

impl From<f64> for SizeConfig {
    fn from(ratio: f64) -> Self {
        Self::Ratio(ratio)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_size_config() {
        assert_eq!(SizeConfig::default(), SizeConfig::Default);

        assert_eq!(SizeConfig::from(42), SizeConfig::Fixed(42));

        assert_eq!(SizeConfig::from(1.5), SizeConfig::Ratio(1.5));

        assert_eq!(SizeConfig::source(), SizeConfig::Default);
        assert_eq!(SizeConfig::source().resolve(50), 50);
    }
}
