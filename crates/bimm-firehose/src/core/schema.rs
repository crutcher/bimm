use std::{
    collections::{
        BTreeMap,
        HashSet,
    },
    ops::{
        Index,
        IndexMut,
    },
};

use anyhow::{
    anyhow,
    bail,
};
use serde::{
    Deserialize,
    Serialize,
};

use crate::core::identifiers;

/// A serializable description of a data type.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DataTypeDescription {
    /// The name of the data type.
    pub type_name: String,
}

impl DataTypeDescription {
    /// Creates a `DataTypeDescription` for a specific type `T`.
    ///
    /// This function uses `std::any::type_name` to get the type name of `T`.
    pub fn new<T>() -> Self {
        Self::from_type_name(std::any::type_name::<T>())
    }

    /// Creates a `DataTypeDescription` from a type name.
    pub fn from_type_name(type_name: &str) -> Self {
        DataTypeDescription {
            type_name: type_name.to_string(),
        }
    }
}

/// A build plan for columns in a table schema.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BuildPlan {
    /// The ID of the operator.
    pub operator_id: String,

    /// The description of the operator.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    pub description: Option<String>,

    /// Additional configuration for the operation, serialized as JSON.
    #[serde(skip_serializing_if = "serde_json::Value::is_null")]
    #[serde(default)]
    pub config: serde_json::Value,

    /// The input column bindings ``{parameter_name: column_name}``.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    #[serde(default)]
    pub inputs: BTreeMap<String, String>,

    /// The output column bindings ``{parameter_name: column_name}``.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    #[serde(default)]
    pub outputs: BTreeMap<String, String>,
}

impl BuildPlan {
    /// Creates a new `ColumnBuildPlan` with the given operator spec.
    pub fn for_operator<S>(id: S) -> Self
    where
        S: AsRef<str>,
    {
        BuildPlan {
            operator_id: id.as_ref().to_string(),
            description: None,
            config: serde_json::Value::Null,
            inputs: BTreeMap::new(),
            outputs: BTreeMap::new(),
        }
    }

    /// Extends the build plan with a description.
    ///
    /// # Arguments
    ///
    /// - `description`: The description to attach to the build plan.
    ///
    /// # Returns
    ///
    /// A new `ColumnBuildPlan` with the description attached.
    pub fn with_description(
        self,
        description: &str,
    ) -> Self {
        BuildPlan {
            description: Some(description.to_string()),
            ..self
        }
    }

    /// Extends the build plan with a configuration.
    ///
    /// # Arguments
    ///
    /// - `config`: The configuration to attach to the build plan, serialized as
    ///   JSON.
    ///
    /// # Returns
    ///
    /// A new `ColumnBuildPlan` with the configuration attached.
    pub fn with_config<T>(
        self,
        config: T,
    ) -> Self
    where
        T: Serialize,
    {
        BuildPlan {
            config: serde_json::to_value(config).expect("Failed to serialize config"),
            ..self
        }
    }

    /// Builds a name map from a slice of tuples.
    ///
    /// # Arguments
    ///
    /// - `assoc`: A slice of tuples where each tuple contains a parameter name
    ///   and a column name.
    ///
    /// # Returns
    ///
    /// A `BTreeMap<String, String>` mapping parameter names to column names.
    fn build_name_map(assoc: &[(&str, &str)]) -> BTreeMap<String, String> {
        let mut btree = BTreeMap::new();
        for (param, column) in assoc {
            identifiers::check_ident(param).expect("Invalid parameter name");
            identifiers::check_ident(column).expect("Invalid column name");
            btree.insert(param.to_string(), column.to_string());
        }
        btree
    }

    /// Extends the build plan with input columns.
    ///
    /// # Arguments
    ///
    /// - `inputs`: A slice of tuples where each tuple contains a parameter name
    ///   and a column name.
    ///
    /// # Returns
    ///
    /// A new `ColumnBuildPlan` with the input columns attached.
    pub fn with_inputs(
        self,
        inputs: &[(&str, &str)],
    ) -> Self {
        BuildPlan {
            inputs: Self::build_name_map(inputs),
            ..self
        }
    }

    /// Extends the build plan with output columns.
    ///
    /// # Arguments
    ///
    /// - `outputs`: A slice of tuples where each tuple contains a parameter
    ///   name and a column name.
    ///
    /// # Returns
    ///
    /// A new `ColumnBuildPlan` with the output columns attached.
    pub fn with_outputs(
        self,
        outputs: &[(&str, &str)],
    ) -> Self {
        BuildPlan {
            outputs: Self::build_name_map(outputs),
            ..self
        }
    }

    /// Translation helper for parameter names to column names.
    #[inline(always)]
    fn translate_parameter_name<'a>(
        &self,
        parameter_type: &str,
        parameter_name: &str,
        parameter_map: &'a BTreeMap<String, String>,
    ) -> anyhow::Result<&'a str> {
        parameter_map
            .get(parameter_name)
            .map(|column_name| column_name.as_str())
            .ok_or_else(|| {
                anyhow!(
                    "'{parameter_name}' is not a {parameter_type} parameter\n:{:?}",
                    self
                )
            })
    }

    /// Translates an input parameter name to its corresponding column name.
    ///
    /// # Arguments
    ///
    /// - `parameter_name`: The name of the input parameter to translate.
    ///
    /// # Returns
    ///
    /// An `anyhow::Result<&str>` containing the column name corresponding to
    /// the input parameter.
    pub fn translate_input_name(
        &self,
        parameter_name: &str,
    ) -> anyhow::Result<&str> {
        self.translate_parameter_name("input", parameter_name, &self.inputs)
    }

    /// Translates an output parameter name to its corresponding column name.
    ///
    /// # Arguments
    ///
    /// - `parameter_name`: The name of the output parameter to translate.
    ///
    /// # Returns
    ///
    /// An `anyhow::Result<&str>` containing the column name corresponding to
    /// the output parameter.
    pub fn translate_output_name(
        &self,
        parameter_name: &str,
    ) -> anyhow::Result<&str> {
        self.translate_parameter_name("output", parameter_name, &self.outputs)
    }
}

/// A description of a column in a data table.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ColumnSchema {
    /// The name of the column.
    pub name: String,

    /// An optional description of the column.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    pub description: Option<String>,

    /// The type of the column.
    pub data_type: DataTypeDescription,
}

impl ColumnSchema {
    /// Creates a new `DataColumnDescription` with the given name and type.
    pub fn new<T>(name: &str) -> Self {
        Self::new_with_type(name, DataTypeDescription::new::<T>())
    }

    /// Creates a new `DataColumnDescription` with the given name and data type.
    pub fn new_with_type(
        name: &str,
        data_type: DataTypeDescription,
    ) -> Self {
        identifiers::check_ident(name).unwrap();
        ColumnSchema {
            name: name.to_string(),
            description: None,
            data_type,
        }
    }

    /// Extends the schema with a description.
    ///
    /// # Arguments
    ///
    /// - `description`: The description to attach to the column.
    ///
    /// # Returns
    ///
    /// A new `BimmColumnSchema` with the description attached.
    pub fn with_description(
        self,
        description: &str,
    ) -> Self {
        ColumnSchema {
            description: Some(description.to_string()),
            ..self
        }
    }
}

/// Bimm Table Schema.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FirehoseTableSchema {
    /// The columns in the table.
    pub columns: Vec<ColumnSchema>,

    /// Build plans for the table.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    #[serde(default)]
    pub build_plans: Vec<BuildPlan>,
}

impl Index<usize> for FirehoseTableSchema {
    type Output = ColumnSchema;

    fn index(
        &self,
        index: usize,
    ) -> &Self::Output {
        &self.columns[index]
    }
}

impl IndexMut<usize> for FirehoseTableSchema {
    fn index_mut(
        &mut self,
        index: usize,
    ) -> &mut Self::Output {
        &mut self.columns[index]
    }
}

impl Index<&str> for FirehoseTableSchema {
    type Output = ColumnSchema;

    fn index(
        &self,
        index: &str,
    ) -> &Self::Output {
        let name = index;
        match self
            .column_index(name)
            .and_then(|idx| self.columns.get(idx))
        {
            Some(column) => column,
            None => panic!(
                "Column {name:?} not found in schema:\n{}",
                serde_json::to_string_pretty(&self).unwrap()
            ),
        }
    }
}

impl IndexMut<&str> for FirehoseTableSchema {
    fn index_mut(
        &mut self,
        index: &str,
    ) -> &mut Self::Output {
        let name = index;
        match self.column_index(name) {
            Some(idx) => &mut self.columns[idx],
            None => panic!(
                "Column {name:?} not found in schema:\n{}",
                serde_json::to_string_pretty(&self).unwrap()
            ),
        }
    }
}

impl FirehoseTableSchema {
    /// Returns an iterator over the columns in the schema.
    pub fn iter(&self) -> impl Iterator<Item = &ColumnSchema> {
        self.columns.iter()
    }

    /// Returns an iterator over the names of the columns in the schema.
    pub fn names_iter(&self) -> impl Iterator<Item = &str> {
        self.iter().map(|c| c.name.as_str())
    }

    /// Gets a reference to a column by its name.
    pub fn get_column(
        &self,
        name: &str,
    ) -> Option<&ColumnSchema> {
        self.column_index(name)
            .and_then(|idx| self.columns.get(idx))
    }

    /// Gets a mutable reference to a column by its name.
    pub fn get_column_mut(
        &mut self,
        name: &str,
    ) -> Option<&mut ColumnSchema> {
        self.column_index(name)
            .and_then(|idx| self.columns.get_mut(idx))
    }

    /// Checks the build graph for the table schema.
    ///
    /// This function verifies the dependencies between build plans
    /// and ensures that all columns can be built in a valid order.
    ///
    /// # Arguments
    ///
    /// - `columns`: A slice of `ColumnSchema` representing the columns in the
    ///   table.
    /// - `plans`: A slice of `BuildPlan` representing the build plans for the
    ///   table.
    ///
    /// # Returns
    ///
    /// An `anyhow::Result<(Vec<String>, Vec<BuildPlan>)>` containing the base
    /// columns and the ordered build plans.
    fn check_graph(
        columns: &[ColumnSchema],
        plans: &[BuildPlan],
    ) -> anyhow::Result<(Vec<String>, Vec<BuildPlan>)> {
        let column_names: Vec<String> = columns.iter().map(|c| c.name.clone()).collect();

        let mut base_columns: Vec<String> = column_names.clone();
        for plan in plans {
            for (_, cname) in plan.outputs.iter() {
                base_columns.retain(|c| c != cname);
            }
        }

        let base_columns = base_columns;

        let mut scheduled_columns = base_columns.clone();
        let mut plan_order = Vec::new();

        // TODO: build reified topo dep-graph;
        // support printing, querying, cycle detection, etc.

        loop {
            let mut progress = false;

            for (idx, plan) in plans.iter().enumerate() {
                if plan_order.contains(&idx) {
                    // This plan is already scheduled.
                    continue;
                }
                if plan
                    .inputs
                    .values()
                    .all(|cname| scheduled_columns.contains(cname))
                {
                    // All inputs are scheduled, we can schedule this plan.
                    for (_, cname) in plan.outputs.iter() {
                        if !scheduled_columns.contains(cname) {
                            scheduled_columns.push(cname.clone());
                        }
                    }
                    progress = true;
                    plan_order.push(idx);
                } else {
                    // Not all inputs are scheduled, skip this plan.
                    continue;
                }
            }

            if !progress {
                // No progress was made, we are done.
                break;
            }
        }

        if scheduled_columns.len() != column_names.len() {
            bail!(
                "Not all columns are scheduled: expected {}, got {}",
                column_names.len(),
                scheduled_columns.len()
            );
        }

        let order: Vec<BuildPlan> = plan_order.iter().map(|&idx| plans[idx].clone()).collect();

        Ok((base_columns, order))
    }

    /// Compute the build order for the table schema.
    ///
    /// This function checks the build plans and their dependencies to determine
    /// the order in which they should be executed.
    ///
    /// # Returns
    ///
    /// An `anyhow::Result<(Vec<String>, Vec<BuildPlan>)>` containing the base
    /// columns and the ordered build plans.
    pub fn build_order(&self) -> anyhow::Result<(Vec<String>, Vec<BuildPlan>)> {
        Self::check_graph(&self.columns, &self.build_plans)
    }

    /// Creates a new `DataTableDescription` with the given columns.
    #[must_use]
    pub fn from_columns(columns: &[ColumnSchema]) -> Self {
        let mut schema = Self {
            columns: Vec::new(),
            build_plans: Vec::new(),
        };

        for column in columns {
            schema.add_column(column.clone());
        }

        schema
    }

    /// Checks if a column name is invalid, or conflicts with existing columns.
    ///
    /// # Arguments
    ///
    /// - `name`: The name of the column to check.
    fn check_name(
        &self,
        name: &str,
    ) -> anyhow::Result<()> {
        identifiers::check_ident(name)?;

        if self.columns.iter().any(|c| c.name == name) {
            Err(anyhow!("Duplicate column name '{name}'"))
        } else {
            Ok(())
        }
    }

    /// Adds a column to the table description.
    pub fn add_column(
        &mut self,
        column: ColumnSchema,
    ) {
        self.check_name(&column.name).unwrap();

        self.columns.push(column);
    }

    /// Adds a build plan to the table description.
    pub fn add_build_plan(
        &mut self,
        plan: BuildPlan,
    ) -> anyhow::Result<()> {
        // Check that all the input and output columns exist.
        for cname in plan.inputs.values() {
            if self.column_index(cname).is_none() {
                bail!("Input column '{cname}' does not exist in the schema");
            }
        }
        for cname in plan.outputs.values() {
            if self.column_index(cname).is_none() {
                bail!("Output column '{cname}' does not exist in the schema");
            }

            for alt_plan in &self.build_plans {
                if alt_plan.outputs.values().any(|v| v == cname) {
                    bail!("Output column '{cname}' already exists in another build plan");
                }
            }
        }

        {
            let mut plans = self.build_plans.clone();
            plans.push(plan.clone());
            Self::check_graph(&self.columns, plans.as_slice())?;
        }

        self.build_plans.push(plan);
        Ok(())
    }

    /// Adds a build plan and its output columns to the table description.
    ///
    /// # Arguments
    ///
    /// - `plan`: The build plan to add.
    /// - `output_info`: A slice of tuples where each tuple contains the output
    ///   column name, its data type, and a description.
    ///
    /// # Returns
    ///
    /// An `anyhow::Result<()>` indicating success or containing an error if the
    /// operation fails.
    pub fn add_build_plan_and_outputs(
        &mut self,
        plan: BuildPlan,
        output_info: &[(String, DataTypeDescription, Option<String>)],
    ) -> anyhow::Result<()> {
        // Now add the output columns.
        for (pname, data_type, description) in output_info {
            let cname = plan.outputs.get(pname).expect("Output column not found");
            let column = ColumnSchema {
                name: cname.to_string(),
                description: description.clone(),
                data_type: data_type.clone(),
            };
            self.add_column(column);
        }

        self.add_build_plan(plan)
    }

    /// Extends the table schema with a build plan and its output columns.
    ///
    /// # Arguments
    ///
    /// - `plan`: The build plan to extend the schema with.
    /// - `output_columns`: A map of output column names to their schemas.
    ///
    /// # Returns
    ///
    /// An `anyhow::Result<()>` indicating success or containing an error if the
    /// operation fails.
    pub fn extend_via_plan(
        &mut self,
        plan: BuildPlan,
        output_columns: &BTreeMap<String, ColumnSchema>,
    ) -> anyhow::Result<()> {
        let all_plan_output_cnames = plan.outputs.values().collect::<HashSet<_>>();
        let all_output_cnames = output_columns.keys().collect::<HashSet<_>>();

        if all_plan_output_cnames != all_output_cnames {
            bail!(
                "Output columns in plan do not match provided output columns: {all_plan_output_cnames:?} != {all_output_cnames:?}"
            );
        }

        for column in output_columns.values() {
            self.add_column(column.clone());
        }

        self.add_build_plan(plan)
    }

    /// Renames a column; and all references.
    ///
    /// Updates build plans to reflect the new column name.
    pub fn rename_column(
        &mut self,
        old_name: &str,
        new_name: &str,
    ) -> anyhow::Result<()> {
        let index = match self.column_index(old_name) {
            Some(idx) => idx,
            None => {
                bail!("Column '{old_name}' not found");
            }
        };

        if old_name == new_name {
            // No change needed
            return Ok(());
        }

        if self.column_index(new_name).is_some() {
            bail!("Column name '{new_name}' already exists");
        }

        self.check_name(new_name)?;

        // Update the column name in the schema.
        self.columns[index].name = new_name.to_string();

        // Update all references to the old column name in build plans.
        for plan in &mut self.build_plans {
            for cname in plan.inputs.values_mut() {
                if cname == old_name {
                    *cname = new_name.to_string();
                }
            }
            for cname in plan.outputs.values_mut() {
                if cname == old_name {
                    *cname = new_name.to_string();
                }
            }
        }

        Ok(())
    }

    /// Returns the index of a column by its name.
    ///
    /// # Arguments
    ///
    /// - `name`: The name of the column to find.
    ///
    /// # Returns
    ///
    /// An `Option<usize>` containing the index of the column if found, or
    /// `None` if not found.
    pub fn column_index(
        &self,
        name: &str,
    ) -> Option<usize> {
        for (idx, column) in self.columns.iter().enumerate() {
            if column.name == name {
                return Some(idx);
            }
        }
        None
    }

    /// Checks if a column with the given name exists and returns its index.
    ///
    /// # Arguments
    ///
    /// - `name`: The name of the column to check.
    ///
    /// # Returns
    ///
    /// An `anyhow::Result<usize>` containing the index of the column if it
    /// exists.
    pub fn check_column_index(
        &self,
        name: &str,
    ) -> anyhow::Result<usize> {
        match self.column_index(name) {
            Some(idx) => Ok(idx),
            None => bail!("Column '{name}' not found in schema"),
        }
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;

    use super::*;

    /// Ensures that `FirehoseTableSchema` is `Send`.
    const FIREHOSE_TABLE_SCHEMA_IS_SEND: fn() = || {
        fn assert_send<T: Send>() {}
        assert_send::<FirehoseTableSchema>();
    };
    #[test]
    fn test_firehose_table_schema_is_send() {
        FIREHOSE_TABLE_SCHEMA_IS_SEND();
    }

    #[test]
    fn test_data_type_description() {
        let schema = DataTypeDescription::new::<Option<i32>>();
        assert_eq!(schema.type_name, std::any::type_name::<Option<i32>>(),);
    }

    #[test]
    fn test_column_description() {
        let column = ColumnSchema::new::<Option<i32>>("abc");

        assert_eq!(column.name, "abc");
        assert_eq!(column.data_type, DataTypeDescription::new::<Option<i32>>());
    }

    #[test]
    fn test_schema() -> anyhow::Result<()> {
        let mut schema = FirehoseTableSchema::from_columns(&[ColumnSchema::new::<i32>("foo")]);
        schema.add_column(ColumnSchema::new::<String>("bar"));

        // Index<usize>
        assert_eq!(schema[0].name, "foo");
        // IndexMut<usize>
        schema[0].description = Some("Overwritten".to_string());

        // Index<&str>
        assert_eq!(schema["foo"].name, "foo");
        // IndexMut<usize>
        schema["foo"].description = Some("An integer column".to_string());
        // Index<usize>
        assert_eq!(
            schema["foo"].description,
            Some("An integer column".to_string())
        );
        // .get_column
        assert_eq!(
            schema.get_column("foo").unwrap().description,
            Some("An integer column".to_string())
        );
        // .get_column_mut
        schema.get_column_mut("bar").unwrap().description = Some("A string column".to_string());

        assert_eq!(schema.columns.len(), 2);
        assert_eq!(schema.columns[0].name, "foo");
        assert_eq!(schema.columns[1].name, "bar");

        assert_eq!(schema.column_index("foo"), Some(0));
        assert_eq!(schema.column_index("bar"), Some(1));
        assert_eq!(schema.column_index("qux"), None);

        assert_eq!(
            serde_json::to_string_pretty(&schema).unwrap(),
            indoc! {r#"
                {
                  "columns": [
                    {
                      "name": "foo",
                      "description": "An integer column",
                      "data_type": {
                        "type_name": "i32"
                      }
                    },
                    {
                      "name": "bar",
                      "description": "A string column",
                      "data_type": {
                        "type_name": "alloc::string::String"
                      }
                    }
                  ]
                }"#
            }
        );

        // No-op.
        schema.rename_column("foo", "foo").unwrap();

        schema.rename_column("foo", "xxx").unwrap();
        assert_eq!(
            serde_json::to_string_pretty(&schema).unwrap(),
            indoc! {r#"
                {
                  "columns": [
                    {
                      "name": "xxx",
                      "description": "An integer column",
                      "data_type": {
                        "type_name": "i32"
                      }
                    },
                    {
                      "name": "bar",
                      "description": "A string column",
                      "data_type": {
                        "type_name": "alloc::string::String"
                      }
                    }
                  ]
                }"#
            }
        );

        Ok(())
    }

    #[should_panic(expected = "Column \"nonexistent\" not found in schema")]
    #[test]
    fn test_lookup_column_nonexistent() {
        let schema = FirehoseTableSchema::from_columns(&[ColumnSchema::new::<i32>("foo")]);
        let _ = schema["nonexistent"];
    }

    #[should_panic(expected = "Column \"nonexistent\" not found in schema")]
    #[test]
    fn test_lookup_mut_column_nonexistent() {
        let mut schema = FirehoseTableSchema::from_columns(&[ColumnSchema::new::<i32>("foo")]);
        schema["nonexistent"].description = Some("A string column".to_string());
    }

    #[test]
    #[should_panic(expected = "Duplicate column name 'foo'")]
    fn conflicting_column_names_on_validate() {
        let _schema = FirehoseTableSchema::from_columns(&[
            ColumnSchema::new::<i32>("foo"),
            ColumnSchema::new::<String>("foo"),
        ]);
    }

    #[test]
    #[should_panic(expected = "Duplicate column name 'foo'")]
    fn conflicting_column_names_on_add() {
        let mut schema = FirehoseTableSchema::from_columns(&[ColumnSchema::new::<i32>("foo")]);
        schema.add_column(ColumnSchema::new::<String>("foo"));
    }

    #[test]
    fn test_build_plan_translation() {
        let plan = BuildPlan::for_operator("test_operator")
            .with_description("A test operator")
            .with_config(serde_json::json!({"key": "value"}))
            .with_inputs(&[("input1", "column1"), ("input2", "column2")])
            .with_outputs(&[("output1", "column3")]);

        assert_eq!(plan.operator_id, "test_operator");
        assert_eq!(plan.description, Some("A test operator".to_string()));
        assert_eq!(plan.config, serde_json::json!({"key": "value"}));
        assert_eq!(plan.inputs.get("input1"), Some(&"column1".to_string()));
        assert_eq!(plan.outputs.get("output1"), Some(&"column3".to_string()));

        let translated_input = plan.translate_input_name("input1").unwrap();
        assert_eq!(translated_input, "column1");

        let translated_output = plan.translate_output_name("output1").unwrap();
        assert_eq!(translated_output, "column3");
    }
}
