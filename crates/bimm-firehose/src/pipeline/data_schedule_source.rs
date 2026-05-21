use std::fmt::Debug;

use crate::pipeline::{
    DataLoadMetaDataItem,
    DataLoadSchedule,
};

/// Data-pipeline trait for schedule sources.
pub trait DataScheduleSource<M>: Debug
where
    M: DataLoadMetaDataItem,
{
    /// Builds a schedule from the source.
    fn build_schedule(&self) -> anyhow::Result<DataLoadSchedule<M>>;
}

/// Boxed extension trait for `DataScheduleSource`.
pub trait DataScheduleSourceExt<M>: DataScheduleSource<M> + Sized
where
    M: DataLoadMetaDataItem,
{
    /// Convert to a boxed version.
    fn boxed(self) -> Box<Self> {
        Box::new(self)
    }
}

/// Blanket implementation of `DataScheduleSourceExt` for any type that meets
/// the requirements.
impl<M, S> DataScheduleSourceExt<M> for S
where
    M: DataLoadMetaDataItem,
    S: DataScheduleSource<M> + Sized + Clone,
{
}

/// A pre-built schedule source that provides a fixed schedule.
#[derive(Debug, Clone)]
pub struct FixedScheduleSource<M>
where
    M: DataLoadMetaDataItem,
{
    /// The fixed schedule to be provided by this source.
    schedule: DataLoadSchedule<M>,
}

impl<M> From<DataLoadSchedule<M>> for FixedScheduleSource<M>
where
    M: DataLoadMetaDataItem,
{
    fn from(schedule: DataLoadSchedule<M>) -> Self {
        Self { schedule }
    }
}

impl<M> From<Vec<M>> for FixedScheduleSource<M>
where
    M: DataLoadMetaDataItem,
{
    fn from(schedule: Vec<M>) -> Self {
        Self::from(DataLoadSchedule::from(schedule))
    }
}

impl<M> DataScheduleSource<M> for FixedScheduleSource<M>
where
    M: DataLoadMetaDataItem,
{
    fn build_schedule(&self) -> anyhow::Result<DataLoadSchedule<M>> {
        Ok(self.schedule.clone())
    }
}

/// Demo of a simple filter source that filters items based on a predicate
/// function.
pub struct SimpleFilterSource<M>
where
    M: DataLoadMetaDataItem,
{
    /// The inner source that provides the initial schedule.
    inner: Box<dyn DataScheduleSource<M>>,

    /// The predicate function used to filter items.
    predicate: Box<dyn Fn(&M) -> bool>,
}

impl<M> Debug for SimpleFilterSource<M>
where
    M: DataLoadMetaDataItem,
{
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        f.debug_struct("SimpleFilterSource")
            .field("inner", &self.inner)
            .finish()
    }
}

impl<M> SimpleFilterSource<M>
where
    M: DataLoadMetaDataItem,
{
    /// Creates a new `SimpleFilterSource` with the provided inner source and
    /// predicate.
    ///
    /// # Arguments
    ///
    /// - `inner`: A boxed source that implements `DataScheduleSource<M>`.
    /// - `predicate`: A predicate function.
    pub fn new(
        inner: Box<dyn DataScheduleSource<M>>,
        predicate: Box<dyn Fn(&M) -> bool>,
    ) -> Self {
        Self { inner, predicate }
    }
}

impl<M> DataScheduleSource<M> for SimpleFilterSource<M>
where
    M: DataLoadMetaDataItem,
{
    fn build_schedule(&self) -> anyhow::Result<DataLoadSchedule<M>> {
        self.inner
            .build_schedule()
            .map(|schedule| schedule.filter(|item| (self.predicate)(item)))
    }
}

/// A wrapper that maps the output of a schedule source to a different type
/// using a mapping function.
pub struct ScheduleSourceMappingWrapper<A, B, F>
where
    A: DataLoadMetaDataItem,
    B: DataLoadMetaDataItem,
    F: Fn(&DataLoadSchedule<A>) -> anyhow::Result<DataLoadSchedule<B>> + Send + Sync + 'static,
{
    /// The inner source that provides the initial schedule.
    inner: Box<dyn DataScheduleSource<A>>,

    /// The mapping function that transforms the schedule.
    map_func: F,
}

impl<A, B, F> Debug for ScheduleSourceMappingWrapper<A, B, F>
where
    A: DataLoadMetaDataItem,
    B: DataLoadMetaDataItem,
    F: Fn(&DataLoadSchedule<A>) -> anyhow::Result<DataLoadSchedule<B>> + Send + Sync + 'static,
{
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        f.debug_struct("ScheduleSourceMappingWrapper")
            .field("inner", &self.inner)
            .finish()
    }
}

impl<A, B, F> ScheduleSourceMappingWrapper<A, B, F>
where
    A: DataLoadMetaDataItem,
    B: DataLoadMetaDataItem,
    F: Fn(&DataLoadSchedule<A>) -> anyhow::Result<DataLoadSchedule<B>> + Send + Sync + 'static,
{
    /// Creates a new `ScheduleSourceMappingWrapper` with the provided inner
    /// source and mapping function.
    ///
    /// # Arguments
    ///
    /// - `inner`: A boxed source that implements `DataScheduleSource<A>`.
    /// - `map`: A mapping function that transforms the schedule.
    pub fn new(
        inner: Box<dyn DataScheduleSource<A>>,
        map_func: F,
    ) -> Self {
        Self { inner, map_func }
    }
}

impl<A, B, F> DataScheduleSource<B> for ScheduleSourceMappingWrapper<A, B, F>
where
    A: DataLoadMetaDataItem,
    B: DataLoadMetaDataItem,
    F: Fn(&DataLoadSchedule<A>) -> anyhow::Result<DataLoadSchedule<B>> + Send + Sync + 'static,
{
    fn build_schedule(&self) -> anyhow::Result<DataLoadSchedule<B>> {
        self.inner
            .build_schedule()
            .and_then(|schedule| (self.map_func)(&schedule))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::DataLoadSchedule;

    #[test]
    fn test_fixed_schedule_source() -> anyhow::Result<()> {
        let items = vec![2, 3, 5];
        let source = FixedScheduleSource::from(items.clone());

        assert_eq!(
            source.build_schedule()?,
            DataLoadSchedule::from(items.clone())
        );

        Ok(())
    }

    #[test]
    fn test_simple_filter_source() -> anyhow::Result<()> {
        let items = vec![1, 2, 3, 4, 5];
        let source = FixedScheduleSource::from(items.clone());
        let filter_source = SimpleFilterSource::new(source.boxed(), Box::new(|&x| x % 2 == 0));

        let expected = vec![2, 4];
        assert_eq!(
            filter_source.build_schedule()?,
            DataLoadSchedule::from(expected)
        );

        Ok(())
    }
}
