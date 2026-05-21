/// Defines legal identifiers for firehose tables.
pub mod identifiers;
/// Defines the operator environment for firehose tables.
pub mod operations;
/// Defines rows and row batches for firehose tables.
pub mod rows;
/// Defines the symbolic schema for firehose tables.
pub mod schema;

/// Defines `ValueBox`, a sum type for Json Values and boxed values.
pub mod values;

// TODO: Work out what the `$crate::core::*` re-exports should be.
pub use rows::{
    FirehoseBatchTransaction,
    FirehoseRow,
    FirehoseRowBatch,
    FirehoseRowReader,
    FirehoseRowTransaction,
    FirehoseRowWriter,
};
pub use schema::FirehoseTableSchema;
pub use values::FirehoseValue;
