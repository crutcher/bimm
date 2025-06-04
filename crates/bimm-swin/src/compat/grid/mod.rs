#[cfg(feature = "burn_0_18_0")]
pub use burn::grid::*;

#[cfg(not(feature = "burn_0_18_0"))]
mod meshgrid;

#[cfg(not(feature = "burn_0_18_0"))]
mod compat {
    pub use super::meshgrid::*;

    /// Enum to specify index cardinal layout.
    #[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
    pub enum GridIndexing {
        /// Dimensions are in the same order as the cardinality of the inputs.
        /// Equivalent to "ij" indexing in NumPy and PyTorch.
        #[default]
        Matrix,

        /// The same as Matrix, but the first two dimensions are swapped.
        /// Equivalent to "xy" indexing in NumPy and PyTorch.
        Cartesian,
    }

    /// Enum to specify grid sparsity mode.
    #[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
    #[allow(dead_code)]
    pub enum GridSparsity {
        /// The grid is fully expanded to the full cartesian product shape.
        #[default]
        Dense,

        /// The grid is sparse, expanded only at the cardinal dimensions.
        Sparse,
    }

    /// Grid policy options.
    #[derive(Default, Debug, Copy, Clone)]
    pub struct GridOptions {
        /// Indexing mode.
        pub indexing: GridIndexing,

        /// Sparsity mode.
        pub sparsity: GridSparsity,
    }

    impl From<GridIndexing> for GridOptions {
        fn from(value: GridIndexing) -> Self {
            Self {
                indexing: value,
                ..Default::default()
            }
        }
    }
    impl From<GridSparsity> for GridOptions {
        fn from(value: GridSparsity) -> Self {
            Self {
                sparsity: value,
                ..Default::default()
            }
        }
    }
}

#[cfg(not(feature = "burn_0_18_0"))]
pub use compat::*;
