/// Operator lookup environments.
pub mod environment;
/// Operator runner for executing operations on rows.
pub mod executor;
/// Defines operator factories and their registration.
pub mod factory;
/// Module defining the runtime operator implementation interface.
pub mod operator;
/// Call planners for symbolically defining operator calls in schemas.
pub mod planner;
/// Global registration module for firehose operators.
pub mod registration;
/// Operator signature and parameter specification.
pub mod signature;

/// Combined macro to define and register a firehose operator.
///
/// # Arguments
///
/// * `$name`: The name of the operator ID to define;
///   will create a self-referential static string constant.
/// * `$constructor`: A closure that returns an `Arc<dyn FirehoseOperatorFactory>`.
///
/// This macro combines the functionality of `define_firehose_operator_id`
/// and `register_firehose_operator_factory`.
#[macro_export]
macro_rules! define_firehose_operator {
    ($name:ident, $constructor:expr) => {
        $crate::define_firehose_operator_id!($name);
        $crate::register_firehose_operator_factory!($name, $constructor);
    };
}

/// Define a self-referential operator ID.
///
/// The id will be defined as a static string constant that refers to its own namespace path.
///
/// # Arguments
///
/// * `$name`: The name of the operator ID to define.
///
#[macro_export]
macro_rules! define_firehose_operator_id {
    ($name:ident) => {
        $crate::define_self_referential_id!("fh:op", $name);
    };
}

/// Macro to register a default operator factory.
///
/// Builders which do not require runtime configuration can be registered
/// using this macro; and collected globally using `list_default_operator_builders`.
///
/// You can also collect a default environment with all registered builders
/// using `new_default_operator_environment`.
#[macro_export]
macro_rules! register_firehose_operator_factory {
    ($name:ident, $constructor:expr) => {
        inventory::submit! {
            $crate::core::operations::registration::FirehoseOperatorFactoryRegistration {
                operator_id: $name,
                supplier: || {
                    let v = ($constructor);
                    std::sync::Arc::from(v) as
                    std::sync::Arc<dyn $crate::core::operations::factory::FirehoseOperatorFactory>
                },
            }
        }
    };
}

#[cfg(test)]
mod tests {
    use crate::core::operations::environment::{FirehoseOperatorEnvironment, MapOpEnvironment};
    use crate::core::operations::factory::SimpleConfigOperatorFactory;
    use crate::core::operations::operator::OperationRunner;
    use crate::core::operations::operator::{FirehoseOperator, OperatorSchedulingMetadata};
    use crate::core::operations::signature::{FirehoseOperatorSignature, ParameterSpec};

    use crate::core::FirehoseValue;
    use crate::core::rows::{
        FirehoseRowBatch, FirehoseRowReader, FirehoseRowTransaction, FirehoseRowWriter,
    };
    use crate::core::schema::{BuildPlan, ColumnSchema, DataTypeDescription, FirehoseTableSchema};
    // use crate::define_firehose_operator_id;
    use indoc::indoc;
    use serde::{Deserialize, Serialize};
    use std::collections::BTreeMap;
    use std::fmt::Debug;
    use std::sync::Arc;

    define_firehose_operator_id!(ADD);

    #[derive(Debug, Serialize, Deserialize)]
    struct AddOperator {
        bias: i32,
    }

    fn add_operator_op_binding() -> Arc<SimpleConfigOperatorFactory<AddOperator>> {
        Arc::new(SimpleConfigOperatorFactory::new(
            FirehoseOperatorSignature::new()
                .with_operator_id(ADD)
                .with_description("Adds inputs with a bias")
                .with_input(ParameterSpec::new::<i32>("x").with_description("First input"))
                .with_input(ParameterSpec::new::<i32>("y").with_description("Second input"))
                .with_output(
                    ParameterSpec::new::<i32>("result")
                        .with_description("Result of addition with bias"),
                ),
        ))
    }

    impl FirehoseOperator for AddOperator {
        fn apply_to_row(
            &self,
            txn: &mut FirehoseRowTransaction,
        ) -> anyhow::Result<()> {
            let x = txn.maybe_get("x").unwrap().parse_as::<i32>()?;
            let y = txn.maybe_get("y").unwrap().parse_as::<i32>()?;

            let result: i32 = x + y + self.bias;

            txn.expect_set("result", FirehoseValue::serialized(result)?);

            Ok(())
        }
    }

    #[test]
    #[should_panic(expected = "'x' expected type")]
    fn test_bad_input_data_type() {
        let mut schema = FirehoseTableSchema::from_columns(&[
            ColumnSchema::new::<String>("a").with_description("First input"),
            ColumnSchema::new::<i32>("b").with_description("Second input"),
            ColumnSchema::new::<i32>("c").with_description("Output"),
        ]);

        schema
            .add_build_plan(
                BuildPlan::for_operator(ADD)
                    .with_config(AddOperator { bias: 10 })
                    .with_inputs(&[("x", "a"), ("y", "b")])
                    .with_outputs(&[("result", "c")]),
            )
            .unwrap();

        let env =
            Arc::new(MapOpEnvironment::from_operators(vec![add_operator_op_binding()]).unwrap())
                as Arc<dyn FirehoseOperatorEnvironment>;

        let _builder = OperationRunner::new_for_plan(
            Arc::new(schema.clone()),
            Arc::new(schema.build_plans[0].clone()),
            env.as_ref(),
        )
        .unwrap();
    }

    #[test]
    #[should_panic(expected = "'result' expected type")]
    fn test_bad_output_data_type() {
        let mut schema = FirehoseTableSchema::from_columns(&[
            ColumnSchema::new::<i32>("a").with_description("First input"),
            ColumnSchema::new::<i32>("b").with_description("Second input"),
            ColumnSchema::new::<String>("c").with_description("Output"),
        ]);

        schema
            .add_build_plan(
                BuildPlan::for_operator(ADD)
                    .with_config(AddOperator { bias: 10 })
                    .with_inputs(&[("x", "a"), ("y", "b")])
                    .with_outputs(&[("result", "c")]),
            )
            .unwrap();

        let env = MapOpEnvironment::from_operators(vec![add_operator_op_binding()]).unwrap();

        let _builder = OperationRunner::new_for_plan(
            Arc::new(schema.clone()),
            Arc::new(schema.build_plans[0].clone()),
            &env,
        )
        .unwrap();
    }

    #[test]
    fn test_simple_op() -> anyhow::Result<()> {
        let mut schema = FirehoseTableSchema::from_columns(&[
            ColumnSchema::new::<i32>("a").with_description("First input"),
            ColumnSchema::new::<i32>("b").with_description("Second input"),
            ColumnSchema::new::<i32>("c").with_description("Output"),
        ]);

        schema
            .add_build_plan(
                BuildPlan::for_operator(ADD)
                    .with_description("Adds inputs with a bias")
                    .with_config(AddOperator { bias: 10 })
                    .with_inputs(&[("x", "a"), ("y", "b")])
                    .with_outputs(&[("result", "c")]),
            )
            .unwrap();

        let env = MapOpEnvironment::from_operators(vec![add_operator_op_binding()]).unwrap();

        let runner = OperationRunner::new_for_plan(
            Arc::new(schema.clone()),
            Arc::new(schema.build_plans[0].clone()),
            &env,
        )
        .unwrap();

        assert_eq!(
            format!("{runner:#?}"),
            indoc! {r#"
               ColumnBuilder {
                   build_plan: BuildPlan {
                       operator_id: "fh:op://bimm_firehose::core::operations::tests::ADD",
                       description: Some(
                           "Adds inputs with a bias",
                       ),
                       config: Object {
                           "bias": Number(10),
                       },
                       inputs: {
                           "x": "a",
                           "y": "b",
                       },
                       outputs: {
                           "result": "c",
                       },
                   },
               }"#,
            }
        );

        assert_eq!(
            runner.scheduling_metadata(),
            OperatorSchedulingMetadata {
                effective_batch_size: 1,
            }
        );

        assert_eq!(runner.build_plan.operator_id, ADD);

        let mut batch = FirehoseRowBatch::new_with_size(Arc::new(schema.clone()), 2);
        batch[0].expect_set("a", FirehoseValue::serialized(10)?);
        batch[0].expect_set("b", FirehoseValue::serialized(20)?);
        batch[1].expect_set("a", FirehoseValue::serialized(-5)?);
        batch[1].expect_set("b", FirehoseValue::serialized(2)?);

        runner.apply_to_batch(&mut batch).unwrap();

        assert_eq!(batch[0].maybe_get("c").unwrap().parse_as::<i32>()?, 40);
        assert_eq!(batch[1].maybe_get("c").unwrap().parse_as::<i32>()?, 7);

        Ok(())
    }

    #[test]
    fn test_operator_spec_validation() {
        let spec = FirehoseOperatorSignature::new()
            .with_input(ParameterSpec::new::<i32>("input1"))
            .with_input(ParameterSpec::new::<String>("input2"))
            .with_output(ParameterSpec::new::<f64>("output"));

        let mut input_types = BTreeMap::new();
        input_types.insert("input1".to_string(), DataTypeDescription::new::<i32>());
        input_types.insert("input2".to_string(), DataTypeDescription::new::<String>());

        let mut output_types = BTreeMap::new();
        output_types.insert("output".to_string(), DataTypeDescription::new::<f64>());

        assert!(spec.validate(&input_types, &output_types).is_ok());
    }

    #[test]
    fn test_path_ident() {
        define_firehose_operator_id!(FOO);

        assert_eq!(FOO, concat!("fh:op://", module_path!(), "::FOO"));
    }

    #[test]
    fn test_map_op_environment() {
        let _env = MapOpEnvironment::new();
    }
}
