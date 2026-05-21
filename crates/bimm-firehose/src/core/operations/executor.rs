use std::{
    fmt::Debug,
    sync::Arc,
};

use crate::core::{
    operations::{
        environment::FirehoseOperatorEnvironment,
        operator::OperationRunner,
    },
    rows::FirehoseRowBatch,
    schema::FirehoseTableSchema,
};

/// Trait for executing a batch of operations on a `RowBatch`.
pub trait FirehoseBatchExecutor: Debug + Send + Sync {
    /// Returns the schema used by this executor.
    fn schema(&self) -> &Arc<FirehoseTableSchema>;

    /// Returns the operator environment used by this executor.
    fn environment(&self) -> &Arc<dyn FirehoseOperatorEnvironment>;

    /// Runs the butch under the policy of the executor.
    fn execute_batch(
        &self,
        batch: &mut FirehoseRowBatch,
    ) -> anyhow::Result<()>;
}

/// A sequential batch executor.
///
/// Runs every `BuildPlan` in the batch schema;
/// executes serially with no threading.
#[derive(Debug)]
pub struct SequentialBatchExecutor {
    /// The schema of the batch to execute.
    schema: Arc<FirehoseTableSchema>,

    /// The operator environment to use for executing the batch.
    environment: Arc<dyn FirehoseOperatorEnvironment>,

    /// The operation runners for each plan in the schema.
    op_runners: Vec<Arc<OperationRunner>>,
}

impl SequentialBatchExecutor {
    /// Creates a new `DefaultBatchExecutor` with the given operator
    /// environment.
    pub fn new(
        schema: Arc<FirehoseTableSchema>,
        environment: Arc<dyn FirehoseOperatorEnvironment>,
    ) -> anyhow::Result<Self> {
        let mut op_runners = Vec::new();
        let (_base, build_order) = schema.build_order()?;
        for plan in &build_order {
            let plan = Arc::new(plan.clone());
            op_runners.push(Arc::new(OperationRunner::new_for_plan(
                schema.clone(),
                plan,
                environment.as_ref(),
            )?));
        }

        Ok(SequentialBatchExecutor {
            schema,
            environment,
            op_runners,
        })
    }
}

impl FirehoseBatchExecutor for SequentialBatchExecutor {
    fn schema(&self) -> &Arc<FirehoseTableSchema> {
        &self.schema
    }

    fn environment(&self) -> &Arc<dyn FirehoseOperatorEnvironment> {
        &self.environment
    }

    fn execute_batch(
        &self,
        batch: &mut FirehoseRowBatch,
    ) -> anyhow::Result<()> {
        for runner in &self.op_runners {
            runner.apply_to_batch(batch)?;
        }
        Ok(())
    }
}

/* Disabled because of the Send + Sync requirement on the
   FirehoseOperatorEnvironment trait, which is not satisfied by the
   current implementation of the environment.

/// A threaded batch executor.
#[derive(Debug)]
pub struct ThreadedBatchExecutor {
    /// The number of worker threads to use for executing the batch.
    num_workers: usize,

    /// The thread pool used for executing operations in parallel.
    pool: threadpool::ThreadPool,

    /// The schema of the batch to execute.
    schema: Arc<FirehoseTableSchema>,

    /// The operator environment to use for executing the batch.
    environment: Arc<dyn FirehoseOperatorEnvironment>,

    /// The operation runners for each plan in the schema.
    op_runners: Vec<Arc<OperationRunner>>,

    /// The sender for sending processed chunks back to the main thread.
    tx: std::sync::mpsc::Sender<(usize, FirehoseRowBatch)>,

    /// The receiver for receiving processed chunks from worker threads.
    rx: std::sync::mpsc::Receiver<(usize, FirehoseRowBatch)>,
}

impl ThreadedBatchExecutor {
    /// Creates a new `ThreadedBatchExecutor` with the given number of workers and operator environment.
    pub fn new(
        num_workers: usize,
        schema: Arc<FirehoseTableSchema>,
        environment: Arc<dyn FirehoseOperatorEnvironment>,
    ) -> anyhow::Result<Self> {
        let mut op_runners = Vec::new();
        let (_base, build_order) = schema.build_order()?;
        for plan in &build_order {
            let plan = Arc::new(plan.clone());
            op_runners.push(Arc::new(OperationRunner::new_for_plan(
                schema.clone(),
                plan,
                environment.as_ref(),
            )?));
        }

        let pool = threadpool::ThreadPool::new(num_workers);

        let (tx, rx) = std::sync::mpsc::channel();

        Ok(ThreadedBatchExecutor {
            schema,
            environment,
            num_workers,
            pool,
            op_runners,
            tx,
            rx,
        })
    }
}

impl FirehoseBatchExecutor for ThreadedBatchExecutor {
    fn schema(&self) -> &Arc<FirehoseTableSchema> {
        &self.schema
    }

    fn environment(&self) -> &Arc<dyn FirehoseOperatorEnvironment> {
        &self.environment
    }

    fn execute_batch(
        &self,
        batch: &mut FirehoseRowBatch,
    ) -> anyhow::Result<()> {
        let chunk_size = batch.len() / self.num_workers;
        for idx in 0..self.num_workers {
            let mut chunk = batch.empty_like();
            let k: usize = std::cmp::min(chunk_size, batch.len());
            batch.drain_rows(0..k).for_each(|r| chunk.add_row(r));

            let tx = self.tx.clone();
            let op_runners = self.op_runners.clone();
            self.pool.execute(move || {
                let mut chunk = chunk;
                for runner in &op_runners {
                    runner
                        .apply_to_batch(&mut chunk)
                        .expect("Failed to apply operation");
                }
                tx.send((idx, chunk))
                    .expect("Failed to send processed chunk");
            });
        }

        let mut chunks = Vec::with_capacity(self.num_workers);
        for _ in 0..self.num_workers {
            let recieved = self.rx.recv().expect("Failed to receive chunk");
            chunks.push(recieved);
        }
        chunks.sort_by_key(|(idx, _)| *idx);

        chunks
            .into_iter()
            .for_each(|(_, chunk)| batch.append_batch(chunk));

        Ok(())
    }
}
 */

#[cfg(test)]
mod tests {
    use super::*;

    const SE_IS_SEND: fn() = || {
        fn assert_send<T: Send>() {}
        assert_send::<SequentialBatchExecutor>();
    };
    #[test]
    fn test_sequential_batch_executor_is_send() {
        SE_IS_SEND();
    }
}
