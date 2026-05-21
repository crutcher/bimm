use std::fmt::Debug;

use crate::pipeline::DataLoadMetaDataItem;

/// Support super-trait for loadable data types.
///
/// # Trait Requirements
///
/// - `Debug`: the data item must be debuggable.
/// - `Clone`: the data item must be cloneable.
/// - `Send` and `Sync`: the data item must be thread-safe.
pub trait DataLoadDataItem: Debug + Clone + Send + Sync {}

/// Blanket implementation of `DataLoadDataItem` for any type that meets the
/// requirements.
impl<T> DataLoadDataItem for T where T: Debug + Clone + Send + Sync {}

/// Trait for data load operators.
pub trait DataLoadOperator<M, T>: Send + Sync + Debug
where
    M: DataLoadMetaDataItem,
    T: DataLoadDataItem,
{
    /// Loads data based on the provided metadata.
    ///
    /// # Arguments
    ///
    /// * `meta` - Metadata item that contains information needed to load the
    ///   data.
    ///
    /// # Returns
    ///
    /// A `Result` containing the loaded data item of type `T` on success,
    /// or a `DataLoadError` on failure.
    fn load(
        &self,
        meta: &M,
    ) -> anyhow::Result<T>;
}

/// A function-based operator for loading data.
#[derive(Clone)]
pub struct FnOperator<M, T, F>
where
    M: DataLoadMetaDataItem,
    T: DataLoadDataItem,
    F: Fn(&M) -> anyhow::Result<T> + Send + Sync,
{
    /// The function that defines how to load the data.
    func: F,

    /// Phantom data to associate the operator with specific metadata and data
    /// types.
    phantom: std::marker::PhantomData<(M, T)>,
}

impl<M, T, F> Debug for FnOperator<M, T, F>
where
    M: DataLoadMetaDataItem,
    T: DataLoadDataItem,
    F: Fn(&M) -> anyhow::Result<T> + Send + Sync,
{
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        f.debug_struct("FnOperator")
            .field("func", &"Function")
            .finish()
    }
}

impl<M, T, F> FnOperator<M, T, F>
where
    M: DataLoadMetaDataItem,
    T: DataLoadDataItem,
    F: Fn(&M) -> anyhow::Result<T> + Send + Sync,
{
    /// Creates a new `FnOperator` with the provided function.
    pub fn new(func: F) -> Self {
        // TODO(crutcher): maybe From::from(func) would be better?
        Self {
            func,
            phantom: std::marker::PhantomData,
        }
    }
}

impl<M, T, F> DataLoadOperator<M, T> for FnOperator<M, T, F>
where
    M: DataLoadMetaDataItem,
    T: DataLoadDataItem,
    F: Fn(&M) -> anyhow::Result<T> + Send + Sync,
{
    fn load(
        &self,
        meta: &M,
    ) -> anyhow::Result<T> {
        (self.func)(meta)
    }
}
