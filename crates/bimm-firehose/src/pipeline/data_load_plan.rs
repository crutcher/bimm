use std::sync::Arc;

use crate::pipeline::{
    DataLoadDataItem,
    DataLoadMetaDataItem,
    DataLoadOperator,
    DataLoadSchedule,
};

/// Represents a plan for loading data, including a schedule and an optional
/// operator.
#[derive(Debug, Clone)]
pub struct DataLoadPlan<M, T>
where
    M: DataLoadMetaDataItem,
    T: DataLoadDataItem,
{
    /// The schedule that defines when and how to load the data.
    pub schedule: DataLoadSchedule<M>,

    /// An optional operator that defines how to load the data.
    pub op: Option<Arc<dyn DataLoadOperator<M, T>>>,
}

impl<M, T> DataLoadPlan<M, T>
where
    M: DataLoadMetaDataItem,
    T: DataLoadDataItem,
{
    /// Creates a new `DataLoadPlan` with an empty schedule and no operator.
    pub fn new() -> Self {
        Self {
            schedule: DataLoadSchedule::new(),
            op: None,
        }
    }

    /// Initializes a `DataLoadPlan` with a given schedule and an optional
    /// operator.
    pub fn init(
        schedule: DataLoadSchedule<M>,
        op: Option<Arc<dyn DataLoadOperator<M, T>>>,
    ) -> Self {
        Self { schedule, op }
    }
}

impl<M, T> Default for DataLoadPlan<M, T>
where
    M: DataLoadMetaDataItem,
    T: DataLoadDataItem,
{
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::pipeline::FnOperator;

    #[test]
    fn test_index_to_str_plan() {
        let schedule: DataLoadSchedule<usize> = vec![3, 1, 2].into();

        let load = |idx: &usize| -> anyhow::Result<String> { Ok(format!("i:{idx}")) };

        let _plan = DataLoadPlan::init(schedule, Some(Arc::new(FnOperator::new(load))));
    }
}
