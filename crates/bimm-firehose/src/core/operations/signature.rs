use std::collections::BTreeMap;

use anyhow::{
    Context,
    bail,
};
use serde::{
    Deserialize,
    Serialize,
};

use crate::core::schema::{
    BuildPlan,
    ColumnSchema,
    DataTypeDescription,
};

/// Defines a single parameter specification
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParameterSpec {
    /// The name of the parameter.
    pub name: String,

    /// An optional description of the parameter.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    pub description: Option<String>,

    /// The data type of the parameter.
    pub data_type: DataTypeDescription,
}

impl ParameterSpec {
    /// Creates a new `ParameterSpec` with the given name and data type.
    ///
    /// The type `T` is used to infer the data type of the parameter.
    ///
    /// # Parameters
    ///
    /// - `name`: The name of the parameter.
    /// - `T`: The type of the parameter, which is used to determine the data
    ///   type description.
    ///
    /// # Returns
    ///
    /// A new `ParameterSpec` instance with the specified name, data type, and
    /// required arity.
    pub fn new<T>(name: &str) -> Self {
        Self {
            name: name.to_string(),
            description: None,
            data_type: DataTypeDescription::new::<T>(),
        }
    }

    /// Extends the parameter specification with a description.
    pub fn with_description(
        self,
        description: impl Into<String>,
    ) -> Self {
        Self {
            name: self.name,
            description: Some(description.into()),
            data_type: self.data_type,
        }
    }
}

/// Defines the call signature of a firehose operator.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FirehoseOperatorSignature {
    /// The identifier for the operator.
    pub operator_id: Option<String>,

    /// Optional description.
    pub description: Option<String>,

    /// A list of input parameters for the operator.
    pub inputs: Vec<ParameterSpec>,

    /// A list of output parameters for the operator.
    pub outputs: Vec<ParameterSpec>,
}

impl Default for FirehoseOperatorSignature {
    fn default() -> Self {
        Self::new()
    }
}

impl FirehoseOperatorSignature {
    /// Returns a reference to the input parameter named `name`, if it exists.
    pub fn get_input(
        &self,
        name: &str,
    ) -> Option<&ParameterSpec> {
        self.inputs.iter().find(|spec| spec.name == name)
    }

    /// Returns a reference to the output parameter named `name`, if it exists.
    pub fn get_output(
        &self,
        name: &str,
    ) -> Option<&ParameterSpec> {
        self.outputs.iter().find(|spec| spec.name == name)
    }

    /// Creates a new `OperatorSpec` with no inputs or outputs.
    pub const fn new() -> Self {
        Self {
            operator_id: None,
            description: None,
            inputs: Vec::new(),
            outputs: Vec::new(),
        }
    }

    /// Creates a new `FirehoseOperatorSignature` with the specified operator
    /// ID.
    pub fn from_operator_id(operator_id: &str) -> Self {
        Self {
            operator_id: Some(operator_id.into()),
            description: None,
            inputs: Vec::new(),
            outputs: Vec::new(),
        }
    }

    /// Extends the operator specification with an operator ID.
    pub fn with_operator_id(
        self,
        operator_id: &str,
    ) -> Self {
        Self {
            operator_id: Some(operator_id.into()),
            description: self.description,
            inputs: self.inputs,
            outputs: self.outputs,
        }
    }

    /// Extends the operator specification with a description.
    pub fn with_description(
        self,
        description: &str,
    ) -> Self {
        Self {
            operator_id: self.operator_id,
            description: Some(description.to_string()),
            inputs: self.inputs,
            outputs: self.outputs,
        }
    }

    /// Internal helper to add a parameter specification to the list of inputs
    /// or outputs.
    ///
    /// # Parameters
    ///
    /// * `spec`: The parameter specification to add.
    /// * `ptype`: A string indicating the type of parameter ("input" or
    ///   "output").
    /// * `specs`: The current list of parameter specifications (either inputs
    ///   or outputs).
    ///
    /// # Returns
    ///
    /// An `anyhow::Result<Vec<ParameterSpec>>` containing a new vector of
    /// parameter specifications with the added parameter.
    fn with_parameter(
        spec: ParameterSpec,
        ptype: &str,
        specs: &[ParameterSpec],
    ) -> anyhow::Result<Vec<ParameterSpec>> {
        if let Some(that) = specs.iter().find(|prev| prev.name == spec.name) {
            bail!(
                "Duplicate {ptype} parameter '{}':\na. {:?}\nb. {:?}",
                spec.name,
                that,
                spec
            )
        } else {
            let mut new_specs = specs.to_vec();
            new_specs.push(spec);
            Ok(new_specs)
        }
    }

    /// Extends the operator specification with an input parameter.
    ///
    /// # Parameters
    ///
    /// * `spec`: The input parameter specification to add.
    ///
    /// # Returns
    ///
    /// An `Result<Self, String>` where:
    /// * `Ok(Self)`: A new `FirehoseOperatorSignature` with the input parameter
    ///   added.
    /// * `Err(String)`: An error message if the input parameter name already
    ///   exists in the signature.
    pub fn with_input_result(
        self,
        spec: ParameterSpec,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            operator_id: self.operator_id,
            description: self.description,
            inputs: Self::with_parameter(spec, "input", &self.inputs)?,
            outputs: self.outputs.clone(),
        })
    }

    /// Extends the operator specification with an input parameter.
    ///
    /// # Parameters
    ///
    /// * `spec`: The input parameter specification to add.
    ///
    /// # Returns
    ///
    /// A new `FirehoseOperatorSignature` with the input parameter added.
    ///
    /// # Panics
    ///
    /// If the input parameter name already exists in the signature,
    pub fn with_input(
        self,
        spec: ParameterSpec,
    ) -> Self {
        match self.with_input_result(spec) {
            Ok(signature) => signature,
            Err(e) => panic!("{e}"),
        }
    }

    /// Extends the operator specification with an output parameter.
    ///
    /// # Parameters
    ///
    /// * `spec`: The output parameter specification to add.
    ///
    /// # Returns
    ///
    /// An `Result<Self, String>` where:
    /// * `Ok(Self)`: A new `FirehoseOperatorSignature` with the output
    ///   parameter added.
    /// * `Err(String)`: An error message if the output parameter name already
    ///   exists in the signature.
    pub fn with_output_result(
        self,
        spec: ParameterSpec,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            operator_id: self.operator_id,
            description: self.description,
            inputs: self.inputs,
            outputs: Self::with_parameter(spec, "output", &self.outputs)?,
        })
    }

    /// Extends the operator specification with an output parameter.
    ///
    /// # Parameters
    ///
    /// * `spec`: The output parameter specification to add.
    ///
    /// # Returns
    ///
    /// A new `FirehoseOperatorSignature` with the output parameter added.
    ///
    /// # Panics
    ///
    /// If the output parameter name already exists in the signature,
    /// it will panic with an error message.
    pub fn with_output(
        self,
        spec: ParameterSpec,
    ) -> Self {
        match self.with_output_result(spec) {
            Ok(signature) => signature,
            Err(e) => panic!("{e}"),
        }
    }

    /// Generates a map of output column schemas for the given build plan.
    pub fn output_column_schemas_for_plan(
        &self,
        build_plan: &BuildPlan,
    ) -> anyhow::Result<BTreeMap<String, ColumnSchema>> {
        let mut result = BTreeMap::new();

        for output_param in &self.outputs {
            let param_name = &output_param.name;

            let column_name = build_plan.outputs.get(param_name).with_context(|| {
                format!("Output parameter '{param_name}' not found in build plan")
            })?;

            let data_type = output_param.data_type.clone();

            let mut col_schema = ColumnSchema::new_with_type(column_name, data_type);
            if output_param.description.is_some() {
                col_schema.description = output_param.description.clone();
            }

            result.insert(column_name.clone(), col_schema);
        }

        Ok(result)
    }

    /// Validates both inputs and outputs
    pub fn validate(
        &self,
        input_types: &BTreeMap<String, DataTypeDescription>,
        output_types: &BTreeMap<String, DataTypeDescription>,
    ) -> anyhow::Result<()> {
        self.validate_parameters("input", &self.inputs, input_types)?;
        self.validate_parameters("output", &self.outputs, output_types)?;
        Ok(())
    }

    /// Utility function to validate parameters against their specifications.
    ///
    /// # Arguments
    ///
    /// * `param_type`: A string indicating the type of parameters being
    ///   validated ("input" or "output").
    /// * `specs`: A slice of `ParameterSpec` that defines the expected
    ///   parameters.
    /// * `provided`: A map of provided parameters, where keys are parameter
    ///   names and values are their data types.
    ///
    /// # Returns
    ///
    /// An `anyhow::Result<()>` indicating success or failure.
    fn validate_parameters(
        &self,
        param_type: &str,
        specs: &[ParameterSpec],
        provided: &BTreeMap<String, DataTypeDescription>,
    ) -> anyhow::Result<()> {
        // Check for required parameters
        let required_params = specs;

        for spec in required_params {
            if !provided.contains_key(&spec.name) {
                bail!(
                    "Missing required {} parameter '{}' of type {:?}",
                    param_type,
                    spec.name,
                    spec.data_type
                );
            }

            if provided[&spec.name].type_name != spec.data_type.type_name {
                bail!(
                    "{} parameter '{}' expected type {:?}, but got {:?}",
                    param_type,
                    spec.name,
                    spec.data_type,
                    provided[&spec.name]
                );
            }
        }

        // Check for unknown parameters
        let expected_names: BTreeMap<String, &DataTypeDescription> = specs
            .iter()
            .map(|spec| (spec.name.clone(), &spec.data_type))
            .collect();

        for (name, data_type) in provided {
            match expected_names.get(name) {
                Some(expected_type) => {
                    if data_type.type_name != expected_type.type_name {
                        bail!(
                            "{param_type} parameter '{name}' expected type {expected_type:?}, but got {data_type:?}"
                        );
                    }
                }
                None => {
                    bail!(
                        "Unexpected {} parameter '{}'. Expected parameters: [{}]",
                        param_type,
                        name,
                        specs
                            .iter()
                            .map(|s| s.name.as_str())
                            .collect::<Vec<_>>()
                            .join(", ")
                    );
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parameter_spec_new() {
        let spec: ParameterSpec = ParameterSpec::new::<i32>("count");
        assert_eq!(spec.name, "count");
        assert_eq!(spec.data_type.type_name, "i32");
    }

    #[test]
    fn test_parameter_spec_with_description() {
        let spec = ParameterSpec::new::<String>("name").with_description("The name of the user");
        assert_eq!(spec.description, Some("The name of the user".to_string()));
    }

    #[test]
    fn test_firehose_operator_signature_new() {
        let signature = FirehoseOperatorSignature::default()
            .with_operator_id("foo::bar::baz")
            .with_description("This is a test operator")
            .with_input(ParameterSpec::new::<i32>("count"))
            .with_output(ParameterSpec::new::<String>("result"));

        assert_eq!(
            signature.get_input("count"),
            Some(&ParameterSpec::new::<i32>("count"))
        );
        assert_eq!(signature.get_input("none"), None);
        assert_eq!(
            signature.get_output("result"),
            Some(&ParameterSpec::new::<String>("result"))
        );
        assert_eq!(signature.get_output("none"), None);

        assert_eq!(signature.operator_id, Some("foo::bar::baz".to_string()));
        assert_eq!(
            signature.description,
            Some("This is a test operator".to_string())
        );
        assert_eq!(signature.inputs, vec![ParameterSpec::new::<i32>("count")]);
        assert_eq!(
            signature.outputs,
            vec![ParameterSpec::new::<String>("result")]
        );
    }

    #[should_panic(expected = "Duplicate input parameter 'count':")]
    #[test]
    fn test_duplicate_input_parameter() {
        FirehoseOperatorSignature::default()
            .with_operator_id("foo::bar::baz")
            .with_input(ParameterSpec::new::<i32>("count"))
            .with_input(ParameterSpec::new::<String>("count")); // Duplicate name
    }

    #[should_panic(expected = "Duplicate output parameter 'count':")]
    #[test]
    fn test_duplicate_output_parameter() {
        FirehoseOperatorSignature::default()
            .with_operator_id("foo::bar::baz")
            .with_output(ParameterSpec::new::<i32>("count"))
            .with_output(ParameterSpec::new::<String>("count")); // Duplicate nam
    }
}
