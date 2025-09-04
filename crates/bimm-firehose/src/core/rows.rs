use crate::core::FirehoseValue;
use crate::core::operations::signature::FirehoseOperatorSignature;
use crate::core::schema::{BuildPlan, FirehoseTableSchema};
use anyhow::Context;
use serde::Serialize;
use serde::de::DeserializeOwned;
use std::any::Any;
use std::collections::HashMap;
use std::fmt::Debug;
use std::ops::{Index, IndexMut, RangeBounds};
use std::sync::Arc;
use std::vec::Drain;

/// Represents a row in a Firehose table, containing values for each column.
pub struct FirehoseRow {
    /// The schema of the row, defining the columns and their data types.
    schema: Arc<FirehoseTableSchema>,

    /// The values in the row, where each value is an `Option<ValueBox>`.
    slots: Vec<Option<FirehoseValue>>,
}

impl Debug for FirehoseRow {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        write!(f, "ValueRow ")?;
        if f.alternate() {
            self.inner_fmt_vert(f)
        } else {
            self.inner_fmt_horiz(f)
        }
    }
}

impl FirehoseRow {
    /// Creates a new `ValueRow` with the given schema and initializes all slots to `None`.
    pub fn new(schema: Arc<FirehoseTableSchema>) -> Self {
        let mut slots = Vec::with_capacity(schema.columns.len());
        slots.resize_with(schema.columns.len(), || None);
        FirehoseRow { schema, slots }
    }

    /// Vertical inner formatting function for `ValueRow`.
    pub(crate) fn inner_fmt_vert(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        writeln!(f, "{{")?;
        for (idx, col) in self.schema.iter().enumerate() {
            if idx > 0 {
                writeln!(f)?;
            }

            let name = col.name.as_str();
            let type_name = col.data_type.type_name.as_str();

            writeln!(f, "  # {type_name}")?;
            write!(f, "  {name}: ")?;

            let value = &self.slots[idx];
            match value {
                Some(v) => writeln!(f, "{v:?},")?,
                None => writeln!(f, "None,")?,
            }
        }
        write!(f, "}}")
    }

    /// Horizontal inner formatting function for `ValueRow`.
    pub(crate) fn inner_fmt_horiz(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        write!(f, "{{ ")?;
        for (idx, (name, value)) in self.iter().enumerate() {
            if idx > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{name}: ")?;
            match value {
                Some(v) => write!(f, "{v:?}")?,
                None => write!(f, "None")?,
            }
        }
        write!(f, " }}")
    }
}

/// A trait for reading rows from a Firehose table.
pub trait FirehoseRowReader {
    /// Returns the schema of the row.
    fn schema(&self) -> &Arc<FirehoseTableSchema>;

    /// Returns an iterator over the column names and their corresponding values.
    fn iter(&self) -> impl Iterator<Item = (&str, Option<&FirehoseValue>)>;

    /// Returns true if the row has a value for the specified column name.
    ///
    /// # Arguments
    ///
    /// - `column_name`: The name of the column to check for a value.
    ///
    /// # Panics
    ///
    /// Panics if the column name does not exist in the schema.
    ///
    /// # Returns
    ///
    /// A boolean indicating whether the row has a value for the specified column.
    fn has_column_value(
        &self,
        column_name: &str,
    ) -> bool;

    /// Gets a reference to the value of the specified column.
    ///
    /// # Arguments
    ///
    /// * `column_name`: The name of the column to retrieve the value from.
    ///
    /// # Returns
    ///
    /// An `Option<&ValueBox>` containing a reference to the value of the specified column,
    /// or `None` if the column does not exist or has no value.
    fn maybe_get(
        &self,
        column_name: &str,
    ) -> Option<&FirehoseValue>;

    /// Gets the column.
    ///
    /// # Arguments
    ///
    /// - `column_name`: the name of the column.
    ///
    /// # Returns
    ///
    /// An `anyhow::Result<&FirehoseValue>` reference to the column value; or an error.
    fn try_get(
        &self,
        column_name: &str,
    ) -> anyhow::Result<&FirehoseValue> {
        self.maybe_get(column_name)
            .with_context(|| format!("Column not found: {}", column_name))
    }

    /// Gets the column.
    ///
    /// # Arguments
    ///
    /// - `column_name`: the name of the column.
    ///
    /// # Returns
    ///
    /// A reference to the column.
    ///
    /// # Panics
    ///
    /// If the column is absent.
    fn expect_get(
        &self,
        column_name: &str,
    ) -> &FirehoseValue {
        self.try_get(column_name).unwrap()
    }

    /// Gets the column, parsing it as a T.
    ///
    /// # Generic Parameters
    ///
    /// - `T`: the type to parse the column as.
    ///
    /// # Arguments
    ///
    /// - `column_name`: the name of the column.
    ///
    /// # Panics
    ///
    /// If the column isn't found or if parsing as `T` fails.
    fn expect_get_parsed<T>(
        &self,
        column_name: &str,
    ) -> T
    where
        T: DeserializeOwned + 'static,
    {
        self.expect_get(column_name).expect_parse_as::<T>()
    }

    /// Gets the column, dereferencing it as a &T.
    ///
    /// # Generic Parameters
    ///
    /// - `T`: the type to parse the column as.
    ///
    /// # Arguments
    ///
    /// - `column_name`: the name of the column.
    ///
    /// # Panics
    ///
    /// If the column isn't a boxed value, or if dereferencing as `T` fails.
    fn expect_get_ref<T>(
        &self,
        column_name: &str,
    ) -> &T
    where
        T: 'static,
    {
        self.expect_get(column_name).expect_as_ref::<T>()
    }
}

/// Helper trait for (key, `ValueBox`) source pairs.
pub trait IntoColumns {
    /// The iterator type returned by `into_columns`.
    type Iter: Iterator<Item = (Self::Key, FirehoseValue)>;

    /// The key type returned by `into_columns`.
    type Key: AsRef<str>;

    /// Convert into a column (key, value) iterator.
    fn into_columns(self) -> Self::Iter;
}

impl<K> IntoColumns for Vec<(K, FirehoseValue)>
where
    K: AsRef<str>,
{
    type Iter = std::vec::IntoIter<(K, FirehoseValue)>;
    type Key = K;

    fn into_columns(self) -> Self::Iter {
        self.into_iter()
    }
}

impl<K, const N: usize> IntoColumns for [(K, FirehoseValue); N]
where
    K: AsRef<str>,
{
    type Iter = std::array::IntoIter<(K, FirehoseValue), N>;
    type Key = K;

    fn into_columns(self) -> Self::Iter {
        self.into_iter()
    }
}

impl<K> IntoColumns for HashMap<K, FirehoseValue>
where
    K: AsRef<str>,
{
    type Iter = std::collections::hash_map::IntoIter<K, FirehoseValue>;
    type Key = K;

    fn into_columns(self) -> Self::Iter {
        self.into_iter()
    }
}

/// A trait for writing rows to a Firehose table.
pub trait FirehoseRowWriter {
    /// Sets the value of the specified column.
    ///
    /// # Arguments
    ///
    /// * `column_name`: The name of the column to set the value for.
    /// * `value`: The value to set for the specified column, wrapped in a `ValueBox`.
    ///
    /// # Panics
    ///
    /// Panics if the column name does not exist in the schema.
    fn expect_set(
        &mut self,
        column_name: &str,
        value: FirehoseValue,
    );

    /// Sets a column, boxing the object.
    fn expect_set_boxing<T>(
        &mut self,
        column_name: &str,
        obj: T,
    ) where
        T: Any + 'static + Send,
    {
        self.expect_set(column_name, FirehoseValue::boxing(obj));
    }

    /// Sets a column, using the boxed value.
    fn expect_set_from_box<T>(
        &mut self,
        column_name: &str,
        obj: Box<T>,
    ) where
        T: Any + 'static + Send,
    {
        self.expect_set(column_name, FirehoseValue::from_box(obj));
    }

    /// Sets a column, serializing the object.
    fn expect_set_serialized<T>(
        &mut self,
        column_name: &str,
        obj: T,
    ) where
        T: Serialize + 'static,
    {
        self.expect_set(column_name, FirehoseValue::serialized(obj).unwrap());
    }

    /// Sets the columns provided.
    ///
    /// # Arguments
    ///
    /// - `columns`: the columns to set, provided as an `IntoColumns` type.
    fn expect_set_columns<S>(
        &mut self,
        columns: S,
    ) where
        S: IntoColumns,
    {
        for (name, value) in columns.into_columns() {
            self.expect_set(name.as_ref(), value);
        }
    }

    /// Take the value of the column, setting it to `None`, and returning it as an `Option<ValueBox>`.
    fn take_column(
        &mut self,
        column_name: &str,
    ) -> Option<FirehoseValue>;
}

impl FirehoseRowReader for FirehoseRow {
    fn schema(&self) -> &Arc<FirehoseTableSchema> {
        &self.schema
    }

    fn iter(&self) -> impl Iterator<Item = (&str, Option<&FirehoseValue>)> {
        self.schema
            .names_iter()
            .zip(self.slots.iter().map(|v| v.as_ref()))
    }

    fn has_column_value(
        &self,
        column_name: &str,
    ) -> bool {
        let index = self.schema.check_column_index(column_name).unwrap();
        self.slots[index].is_some()
    }

    fn maybe_get(
        &self,
        column_name: &str,
    ) -> Option<&FirehoseValue> {
        let index = self.schema.column_index(column_name)?;
        self.slots.get(index)?.as_ref()
    }
}

impl FirehoseRowWriter for FirehoseRow {
    fn expect_set(
        &mut self,
        column_name: &str,
        value: FirehoseValue,
    ) {
        let index = self.schema.check_column_index(column_name).unwrap();
        self.slots[index] = Some(value);
    }

    fn take_column(
        &mut self,
        column_name: &str,
    ) -> Option<FirehoseValue> {
        let index = self.schema.check_column_index(column_name).unwrap();
        self.slots[index].take()
    }
}

/// Represents a batch of `ValueRow`, with `FirehoseTableSchema` as its schema.
pub struct FirehoseRowBatch {
    /// The schema of the row batch.
    schema: Arc<FirehoseTableSchema>,

    /// The rows in the batch.
    pub(crate) rows: Vec<FirehoseRow>,
}

impl Debug for FirehoseRowBatch {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        writeln!(f, "ValueRowBatch {{")?;
        for (idx, row) in self.iter().enumerate() {
            write!(f, "  {idx}: ")?;
            row.inner_fmt_horiz(f)?;
            writeln!(f, ",")?;
        }
        write!(f, "}}")
    }
}

impl FirehoseRowBatch {
    /// Creates a new `ValueRowBatch` with the given schema and initializes an empty vector of rows.
    pub fn new(schema: Arc<FirehoseTableSchema>) -> Self {
        Self::new_with_size(schema, 0)
    }

    /// Creates an empty `ValueRowBatch` with the same schema as this batch.
    pub fn empty_like(&self) -> Self {
        FirehoseRowBatch::new_with_size(self.schema.clone(), 0)
    }

    /// Creates a new `ValueRowBatch` with the given schema and row count.
    pub fn new_with_size(
        schema: Arc<FirehoseTableSchema>,
        count: usize,
    ) -> Self {
        let mut rows = Vec::with_capacity(count);
        rows.resize_with(count, || FirehoseRow::new(schema.clone()));
        FirehoseRowBatch { schema, rows }
    }

    /// The number of rows in the batch.
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    /// Checks if the batch is empty.
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Returns an iterator over the rows in the batch.
    pub fn iter(&self) -> impl Iterator<Item = &FirehoseRow> {
        self.rows.iter()
    }

    /// Returns a mutable iterator over the rows in the batch.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut FirehoseRow> {
        self.rows.iter_mut()
    }

    /// Returns a reference to the schema of the row batch.
    pub fn schema(&self) -> &Arc<FirehoseTableSchema> {
        &self.schema
    }

    /// Adds a new row to the batch.
    ///
    /// # Arguments
    ///
    /// * `row`: The `ValueRow` to add to the batch.
    ///
    /// # Panics
    ///
    /// Panics if the row's schema does not match the batch's schema.
    pub fn add_row(
        &mut self,
        row: FirehoseRow,
    ) {
        if row.schema != self.schema {
            panic!("Cannot add row with different schema");
        }
        self.rows.push(row);
    }

    /// Creates a new row in the batch with the same schema as the batch.
    ///
    /// This initializes the row with all slots set to `None`.
    ///
    /// # Returns
    ///
    /// A mutable reference to the newly created `ValueRow`.
    pub fn new_row(&mut self) -> &mut FirehoseRow {
        let row = FirehoseRow::new(self.schema.clone());
        self.add_row(row);
        self.rows.last_mut().unwrap()
    }

    /// Removes the subslice indicated by the given range from the vector,
    /// returning a double-ended iterator over the removed subslice.
    ///
    /// If the iterator is dropped before being fully consumed,
    /// it drops the remaining removed elements.
    ///
    /// The returned iterator keeps a mutable borrow on the vector to optimize
    /// its implementation.
    ///
    /// # Panics
    ///
    /// Panics if the starting point is greater than the end point or if
    /// the end point is greater than the length of the vector.
    pub fn drain_rows<R>(
        &mut self,
        range: R,
    ) -> Drain<'_, FirehoseRow>
    where
        R: RangeBounds<usize>,
    {
        self.rows.drain(range)
    }

    /// Consume and append the rows in a batch to this batch.
    pub fn append_batch(
        &mut self,
        other: FirehoseRowBatch,
    ) {
        for row in other.rows {
            self.add_row(row);
        }
    }
}

impl Index<usize> for FirehoseRowBatch {
    type Output = FirehoseRow;

    /// Returns a reference to the row at the specified index.
    ///
    /// # Arguments
    ///
    /// * `index`: The index of the row to retrieve.
    fn index(
        &self,
        index: usize,
    ) -> &Self::Output {
        &self.rows[index]
    }
}

impl IndexMut<usize> for FirehoseRowBatch {
    /// Returns a mutable reference to the row at the specified index.
    ///
    /// # Arguments
    ///
    /// * `index`: The index of the row to retrieve.
    fn index_mut(
        &mut self,
        index: usize,
    ) -> &mut Self::Output {
        &mut self.rows[index]
    }
}

/// A map of formal parameters to their corresponding input and output columns in a build plan.
struct ParameterMapper {
    /// The build plan that describes the operator and its inputs/outputs.
    build_plan: Arc<BuildPlan>,
}

impl ParameterMapper {
    /// Creates a new `ParameterMapper` for the given build plan.
    fn new(build_plan: Arc<BuildPlan>) -> Self {
        ParameterMapper { build_plan }
    }

    /// Maps an input parameter name to its corresponding column name in the build plan.
    ///
    /// # Arguments
    ///
    /// * `parameter_name`: The name of the input parameter to map.
    ///
    /// # Returns
    ///
    /// An `Option<&str>` containing the column name if the parameter is an input parameter,
    fn try_map_input_name(
        &self,
        parameter_name: &str,
    ) -> Option<&str> {
        self.build_plan
            .inputs
            .get(parameter_name)
            .map(|s| s.as_str())
    }

    /// Returns the build plan associated with this parameter mapper.
    ///
    /// # Panics
    ///
    /// Panics if this parameter is not an input parameter.
    fn assert_map_input_name(
        &self,
        parameter_name: &str,
    ) -> &str {
        match self.try_map_input_name(parameter_name) {
            Some(name) => name,
            None => panic!("Parameter '{parameter_name}' is not an input parameter"),
        }
    }

    /// Maps an output parameter name to its corresponding column name in the build plan.
    ///
    /// # Arguments
    ///
    /// * `parameter_name`: The name of the output parameter to map.
    ///
    /// # Returns
    ///
    /// An `Option<&str>` containing the column name if the parameter is an output parameter,
    fn try_map_output_name(
        &self,
        parameter_name: &str,
    ) -> Option<&str> {
        self.build_plan
            .outputs
            .get(parameter_name)
            .map(|s| s.as_str())
    }

    /// Returns the build plan associated with this parameter mapper.
    ///
    /// # Panics
    ///
    /// Panics if this parameter is not an output parameter.
    fn assert_map_output_name(
        &self,
        parameter_name: &str,
    ) -> &str {
        match self.try_map_output_name(parameter_name) {
            Some(name) => name,
            None => panic!("Parameter '{parameter_name}' is not an output parameter"),
        }
    }
}

/// A transaction for a batch of rows processed by an operator.
pub struct FirehoseBatchTransaction<'a> {
    /// A mapper for translating between operator parameters and column names.
    parameter_mapper: ParameterMapper,

    /// The original row batch being processed.
    original: &'a mut FirehoseRowBatch,

    /// A batch of updates to be applied to the original row batch.
    updates: FirehoseRowBatch,

    /// The build plan that describes the operator and its inputs/outputs.
    build_plan: Arc<BuildPlan>,

    /// Signature of the operator being executed.
    signature: Arc<FirehoseOperatorSignature>,
}

impl<'a> FirehoseBatchTransaction<'a> {
    /// Creates a new `OperatorBatchTransaction` for the given row batch.
    pub fn new(
        original: &'a mut FirehoseRowBatch,
        build_plan: Arc<BuildPlan>,
        signature: Arc<FirehoseOperatorSignature>,
    ) -> FirehoseBatchTransaction<'a> {
        let updates = FirehoseRowBatch::new_with_size(original.schema().clone(), original.len());

        FirehoseBatchTransaction {
            parameter_mapper: ParameterMapper::new(build_plan.clone()),
            original,
            updates,
            build_plan,
            signature,
        }
    }

    /// Commit the updates made in this transaction to the original row batch.
    ///
    /// This applies all changes made to the rows in the `updates` batch
    /// back to the corresponding rows in the `original` batch.
    ///
    /// # Returns
    ///
    /// An `anyhow::Result<()>` indicating success or failure.
    pub fn commit(mut self) -> anyhow::Result<()> {
        for (original, update) in self.original.iter_mut().zip(self.updates.iter_mut()) {
            for target_column in self.build_plan.outputs.values() {
                // Transfer the ownership of the value from the update row to the original row.
                if let Some(value) = update.take_column(target_column) {
                    original.expect_set(target_column, value);
                }
            }
        }
        Ok(())
    }

    /// The length of the row batch.
    pub fn len(&self) -> usize {
        self.original.len()
    }

    /// Checks if the row batch is empty.
    pub fn is_empty(&self) -> bool {
        self.original.is_empty()
    }

    /// Returns a view of the operation signature.
    pub fn signature(&self) -> &Arc<FirehoseOperatorSignature> {
        &self.signature
    }

    /// Construct a mutable row-transaction for the row at the given index.
    pub fn mut_row_transaction(
        &mut self,
        index: usize,
    ) -> FirehoseRowTransaction<'_> {
        FirehoseRowTransaction {
            parameter_mapper: &self.parameter_mapper,
            original: &mut self.original.rows[index],
            updates: &mut self.updates.rows[index],
        }
    }
}

/// A row-transaction for a single row in a batch, allowing both reading and writing of values.
///
/// This is a view-class of a backing `FirehoseBatchTransaction`.
pub struct FirehoseRowTransaction<'a> {
    /// A mapper for translating between operator parameters and column names.
    parameter_mapper: &'a ParameterMapper,

    /// The original row being processed.
    original: &'a mut FirehoseRow,

    /// A row of updates to be applied to the original row.
    updates: &'a mut FirehoseRow,
}

impl FirehoseRowReader for FirehoseRowTransaction<'_> {
    fn schema(&self) -> &Arc<FirehoseTableSchema> {
        self.original.schema()
    }

    fn iter(&self) -> impl Iterator<Item = (&str, Option<&FirehoseValue>)> {
        self.parameter_mapper
            .build_plan
            .inputs
            .iter()
            .map(|(pname, cname)| (pname.as_str(), self.original.maybe_get(cname)))
    }

    fn has_column_value(
        &self,
        column_name: &str,
    ) -> bool {
        match self.parameter_mapper.try_map_input_name(column_name) {
            None => false,
            Some(name) => self.original.has_column_value(name),
        }
    }

    fn maybe_get(
        &self,
        column_name: &str,
    ) -> Option<&FirehoseValue> {
        let column_name = self.parameter_mapper.assert_map_input_name(column_name);
        self.original.maybe_get(column_name)
    }
}

impl FirehoseRowWriter for FirehoseRowTransaction<'_> {
    fn expect_set(
        &mut self,
        column_name: &str,
        value: FirehoseValue,
    ) {
        let column_name = self.parameter_mapper.assert_map_output_name(column_name);
        self.updates.expect_set(column_name, value);
    }

    fn take_column(
        &mut self,
        _column_name: &str,
    ) -> Option<FirehoseValue> {
        unimplemented!("take() is not supported in FirehoseRowTransaction");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::operations::signature::ParameterSpec;
    use crate::core::schema::{ColumnSchema, DataTypeDescription, FirehoseTableSchema};
    use std::sync::Arc;

    /// Ensures that `ValueRow` is `Send`, allowing it to be safely shared across threads.
    const VALUE_ROW_IS_SEND: fn() = || {
        fn assert_send<T: Send>() {}
        assert_send::<FirehoseRow>();
    };
    #[test]
    fn test_value_row_is_send() {
        VALUE_ROW_IS_SEND();
    }

    #[test]
    fn test_row_creation() {
        let schema = Arc::new(FirehoseTableSchema::from_columns(&[
            ColumnSchema::new::<i32>("foo"),
            ColumnSchema::new::<String>("bar"),
        ]));

        let row = FirehoseRow::new(schema.clone());
        assert_eq!(row.slots.len(), 2);
        assert!(!row.has_column_value("foo"));
        assert!(!row.has_column_value("bar"));

        assert_eq!(row.schema(), &schema);
    }

    #[test]
    fn test_batch_creation() {
        let schema = Arc::new(FirehoseTableSchema::from_columns(&[
            ColumnSchema::new::<i32>("foo"),
            ColumnSchema::new::<String>("bar"),
        ]));

        let mut batch = FirehoseRowBatch::new(schema.clone());
        assert_eq!(batch.schema(), &schema);
        assert_eq!(batch.len(), 0);
        assert!(batch.is_empty());

        batch.new_row();
        assert_eq!(batch.len(), 1);
        assert!(!batch.is_empty());

        let empty = batch.empty_like();
        assert_eq!(empty.len(), 0);
        assert!(empty.is_empty());
        assert_eq!(empty.schema(), &schema);
    }

    #[test]
    fn test_batch_index() {
        let schema = Arc::new(FirehoseTableSchema::from_columns(&[
            ColumnSchema::new::<i32>("foo"),
            ColumnSchema::new::<String>("bar"),
        ]));

        let mut batch = FirehoseRowBatch::new(schema.clone());
        batch.new_row();
        // IndexMut<usize>
        batch[0].expect_set_serialized("foo", 42);

        let row2 = batch.new_row();
        row2.expect_set_serialized("bar", "Hello");

        // Index<usize>
        assert_eq!(batch[0].expect_get_parsed::<i32>("foo"), 42);
        assert_eq!(batch[1].expect_get_parsed::<String>("bar"), "Hello");
    }

    #[should_panic(expected = "Cannot add row with different schema")]
    #[test]
    fn test_add_row_with_different_schema() {
        let schema1 = Arc::new(FirehoseTableSchema::from_columns(&[
            ColumnSchema::new::<i32>("foo"),
        ]));

        let schema2 = Arc::new(FirehoseTableSchema::from_columns(&[ColumnSchema::new::<
            String,
        >("bar")]));

        let mut batch = FirehoseRowBatch::new(schema1);
        let row = FirehoseRow::new(schema2);
        batch.add_row(row); // This should panic
    }

    #[test]
    fn test_drain_rows() {
        let schema = Arc::new(FirehoseTableSchema::from_columns(&[
            ColumnSchema::new::<i32>("foo"),
            ColumnSchema::new::<String>("bar"),
        ]));

        let mut batch = FirehoseRowBatch::new(schema.clone());
        for i in 0..5 {
            let row = batch.new_row();
            row.expect_set_serialized("foo", i);
            row.expect_set_serialized("bar", format!("Row {i}"));
        }

        assert_eq!(batch.len(), 5);
        let drained: Vec<_> = batch.drain_rows(1..3).collect();
        assert_eq!(drained.len(), 2);
        assert_eq!(batch.len(), 3);
    }

    #[test]
    fn test_append_batch() {
        let schema = Arc::new(FirehoseTableSchema::from_columns(&[
            ColumnSchema::new::<i32>("foo"),
            ColumnSchema::new::<String>("bar"),
        ]));

        let mut batch1 = FirehoseRowBatch::new(schema.clone());
        for i in 0..3 {
            let row = batch1.new_row();
            row.expect_set_serialized("foo", i);
            row.expect_set_serialized("bar", format!("Row {i}"));
        }

        let mut batch2 = FirehoseRowBatch::new(schema.clone());
        for i in 3..5 {
            let row = batch2.new_row();
            row.expect_set_serialized("foo", i);
            row.expect_set_serialized("bar", format!("Row {i}"));
        }

        assert_eq!(batch1.len(), 3);
        assert_eq!(batch2.len(), 2);

        batch1.append_batch(batch2);
        assert_eq!(batch1.len(), 5);
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct MyStruct {
        pub value: i32,
    }

    #[test]
    fn test_row_mutation() -> anyhow::Result<()> {
        let schema = Arc::new(FirehoseTableSchema::from_columns(&[
            ColumnSchema::new::<i32>("foo"),
            ColumnSchema::new::<MyStruct>("bar"),
            ColumnSchema::new::<String>("qux"),
        ]));

        let mut row = FirehoseRow::new(schema.clone());
        assert!(!row.has_column_value("foo"));
        assert!(!row.has_column_value("bar"));

        row.expect_set_serialized("foo", 42);
        assert!(row.has_column_value("foo"));
        assert_eq!(row.expect_get_parsed::<f32>("foo"), 42_f32);

        let my_struct = MyStruct { value: 100 };
        row.expect_set_boxing("bar", my_struct.clone());
        assert!(row.has_column_value("bar"));
        assert_eq!(row.expect_get_ref::<MyStruct>("bar").value, my_struct.value);

        let my_struct = Box::new(MyStruct { value: 100 });
        row.expect_set_from_box("bar", my_struct.clone());
        assert!(row.has_column_value("bar"));
        assert_eq!(row.expect_get_ref::<MyStruct>("bar").value, my_struct.value);

        assert_eq!(
            format!("{row:?}"),
            "ValueRow { foo: {42}, bar: [Any { .. }], qux: None }"
        );
        assert_eq!(
            format!("{row:#?}"),
            indoc::indoc! {r#"
              ValueRow {
                # i32
                foo: {42},

                # bimm_firehose::core::rows::tests::MyStruct
                bar: [Any { .. }],

                # alloc::string::String
                qux: None,
              }"#},
        );

        let val = row.take_column("foo");
        assert!(!row.has_column_value("foo"));
        assert_eq!(val.unwrap().parse_as::<i32>()?, 42);

        assert!(row.take_column("foo").is_none());

        Ok(())
    }

    #[test]
    fn test_set_columns() -> anyhow::Result<()> {
        let schema = Arc::new(FirehoseTableSchema::from_columns(&[
            ColumnSchema::new::<i32>("foo"),
            ColumnSchema::new::<String>("bar"),
        ]));

        {
            // Array
            let mut row = FirehoseRow::new(schema.clone());
            row.expect_set_columns([
                ("foo", FirehoseValue::serialized(42)?),
                ("bar", FirehoseValue::serialized("Hello")?),
            ]);
            assert_eq!(row.expect_get_parsed::<i32>("foo"), 42);
            assert_eq!(row.expect_get_parsed::<String>("bar"), "Hello");
        }
        {
            // Vec
            let mut row = FirehoseRow::new(schema.clone());
            row.expect_set_columns(vec![
                ("foo", FirehoseValue::serialized(42)?),
                ("bar", FirehoseValue::serialized("Hello")?),
            ]);
            assert_eq!(row.expect_get_parsed::<i32>("foo"), 42);
            assert_eq!(row.expect_get_parsed::<String>("bar"), "Hello");
        }
        {
            // HashMap
            let mut row = FirehoseRow::new(schema.clone());
            row.expect_set_columns(HashMap::from([
                ("foo", FirehoseValue::serialized(42)?),
                ("bar", FirehoseValue::serialized("Hello")?),
            ]));
            assert_eq!(row.expect_get_parsed::<i32>("foo"), 42);
            assert_eq!(row.expect_get_parsed::<String>("bar"), "Hello");
        }

        Ok(())
    }

    #[test]
    fn test_value_row_batch_debug() {
        let schema = Arc::new(FirehoseTableSchema::from_columns(&[
            ColumnSchema::new::<i32>("foo"),
            ColumnSchema::new::<String>("bar"),
        ]));

        let mut batch = FirehoseRowBatch::new(schema.clone());
        let row = batch.new_row();
        row.expect_set_serialized("foo", 42);

        let row = batch.new_row();
        row.expect_set_serialized("bar", "Hello");

        assert_eq!(
            format!("{batch:#?}"),
            indoc::indoc! {r#"
                ValueRowBatch {
                  0: { foo: {42}, bar: None },
                  1: { foo: None, bar: {"Hello"} },
                }"#}
        );
    }

    #[test]
    fn test_batch_transaction() -> anyhow::Result<()> {
        let mut schema = FirehoseTableSchema::from_columns(&[
            ColumnSchema::new::<i32>("foo"),
            ColumnSchema::new::<String>("bar"),
        ]);

        let build_plan = BuildPlan::for_operator("foo/bar")
            .with_inputs(&[("source", "foo")])
            .with_outputs(&[("result", "xyz")]);
        schema.add_build_plan_and_outputs(
            build_plan.clone(),
            &[(
                "result".to_string(),
                DataTypeDescription::new::<String>(),
                None,
            )],
        )?;
        let schema = Arc::new(schema);

        let mut batch = FirehoseRowBatch::new(schema.clone());
        let row = batch.new_row();
        row.expect_set_serialized("foo", 42);
        row.expect_set_serialized("bar", "Hello");

        let signature = Arc::new(
            FirehoseOperatorSignature::new()
                .with_input(
                    ParameterSpec::new::<String>("source")
                        .with_description("Source parameter for the operator"),
                )
                .with_output(
                    ParameterSpec::new::<String>("result")
                        .with_description("Result parameter for the operator"),
                ),
        );

        let mut txn = FirehoseBatchTransaction::new(
            &mut batch,
            Arc::new(build_plan.clone()),
            signature.clone(),
        );

        assert_eq!(txn.len(), 1);
        assert!(!txn.is_empty());

        assert_eq!(txn.signature(), &signature);

        let row_txn = txn.mut_row_transaction(0);
        assert_eq!(row_txn.schema(), &schema);
        assert_eq!(row_txn.expect_get_parsed::<i32>("source"), 42);

        assert!(row_txn.has_column_value("source"));
        assert!(!row_txn.has_column_value("foo"));

        let vals = row_txn.iter().collect::<Vec<_>>();
        assert_eq!(vals.len(), 1);
        assert_eq!(vals[0].0, "source");
        assert_eq!(vals[0].1.unwrap().parse_as::<i32>()?, 42);

        Ok(())
    }

    #[should_panic(expected = "take() is not supported in FirehoseRowTransaction")]
    #[test]
    fn test_row_transaction_take() {
        let mut schema = FirehoseTableSchema::from_columns(&[
            ColumnSchema::new::<i32>("foo"),
            ColumnSchema::new::<String>("bar"),
        ]);

        let build_plan = BuildPlan::for_operator("foo/bar")
            .with_inputs(&[("source", "foo")])
            .with_outputs(&[("result", "xyz")]);
        schema
            .add_build_plan_and_outputs(
                build_plan.clone(),
                &[(
                    "result".to_string(),
                    DataTypeDescription::new::<String>(),
                    None,
                )],
            )
            .unwrap();
        let schema = Arc::new(schema);

        let mut batch = FirehoseRowBatch::new(schema.clone());
        let row = batch.new_row();
        row.expect_set_serialized("foo", 42);
        row.expect_set_serialized("bar", "Hello");

        let signature = Arc::new(
            FirehoseOperatorSignature::new()
                .with_input(
                    ParameterSpec::new::<String>("source")
                        .with_description("Source parameter for the operator"),
                )
                .with_output(
                    ParameterSpec::new::<String>("result")
                        .with_description("Result parameter for the operator"),
                ),
        );

        let mut txn = FirehoseBatchTransaction::new(
            &mut batch,
            Arc::new(build_plan.clone()),
            signature.clone(),
        );

        let mut row_txn = txn.mut_row_transaction(0);
        row_txn.take_column("foo");
    }
}
