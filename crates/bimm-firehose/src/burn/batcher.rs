use std::sync::Arc;

use anyhow::Context;
use burn::{
    data::dataloader::batcher::Batcher,
    prelude::Backend,
};

use crate::core::{
    FirehoseRowBatch,
    operations::executor::FirehoseBatchExecutor,
};

/// Input Adapter for `HackyBatcher`.
pub trait BatcherInputAdapter<I>: Send + Sync
where
    I: Send + Sync + Clone + std::fmt::Debug + 'static,
{
    /// Converts a vector of inputs of type `I` to a `FirehoseRowBatch`.
    fn apply(
        &self,
        inputs: Vec<I>,
    ) -> anyhow::Result<FirehoseRowBatch>;
}

/// Output Adapter for `HackyBatcher`.
pub trait BatcherOutputAdapter<B, O>: Send + Sync
where
    B: Backend,
    O: Send + Clone + std::fmt::Debug + 'static,
{
    /// Converts a `FirehoseRowBatch` to an output of type `O`.
    fn apply(
        &self,
        batch: &FirehoseRowBatch,
        device: &B::Device,
    ) -> anyhow::Result<O>;
}

/// Firehose Row Burn Batcher.
pub struct FirehoseExecutorBatcher<B, I, O>
where
    B: Backend,
    I: Send + Sync + Clone + std::fmt::Debug + 'static,
    O: Send + Clone + std::fmt::Debug + 'static,
{
    /// The executor used to run the batch.
    executor: Arc<dyn FirehoseBatchExecutor>,

    /// Map `Vec<I>` input to a `FirehoseRowBatch`.
    input_adapter: Arc<dyn BatcherInputAdapter<I>>,

    /// Map a `FirehoseRowBatch` to an output of type `O`.
    output_adapter: Arc<dyn BatcherOutputAdapter<B, O>>,
}

impl<B, I, O> FirehoseExecutorBatcher<B, I, O>
where
    B: Backend,
    I: Send + Sync + Clone + std::fmt::Debug + 'static,
    O: Send + Clone + std::fmt::Debug + 'static,
{
    /// Creates a new `HackyBatcher` with the given executor, input adapter, and
    /// output adapter.
    pub fn new(
        executor: Arc<dyn FirehoseBatchExecutor>,
        input_adapter: Arc<dyn BatcherInputAdapter<I>>,
        output_adapter: Arc<dyn BatcherOutputAdapter<B, O>>,
    ) -> Self {
        Self {
            executor,
            input_adapter,
            output_adapter,
        }
    }

    /// Executes a batch of items and returns the result.
    ///
    /// # Arguments
    ///
    /// * `items` - A vector of items to be processed.
    /// * `device` - The device on which the output will be processed.
    ///
    /// # Returns
    ///
    /// An `anyhow::Result` containing the output of type `O`.
    fn batch_result(
        &self,
        items: Vec<I>,
        device: &B::Device,
    ) -> anyhow::Result<O> {
        let mut batch = self.input_adapter.apply(items)?;

        self.executor
            .execute_batch(&mut batch)
            .with_context(|| "Failed to execute batch".to_string())?;

        self.output_adapter.apply(&batch, device)
    }
}

impl<B, I, O> Batcher<B, I, O> for FirehoseExecutorBatcher<B, I, O>
where
    B: Backend,
    I: Send + Sync + Clone + std::fmt::Debug + 'static,
    O: Send + Clone + std::fmt::Debug + 'static,
{
    fn batch(
        &self,
        items: Vec<I>,
        device: &B::Device,
    ) -> O {
        self.batch_result(items, device)
            .expect("Failed to execute batch")
    }
}
