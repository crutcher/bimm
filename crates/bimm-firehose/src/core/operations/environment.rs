use std::{
    collections::BTreeMap,
    fmt::Debug,
    sync::Arc,
};

use anyhow::{
    Context,
    bail,
};

use crate::core::{
    operations::{
        factory::{
            FirehoseOperatorFactory,
            FirehoseOperatorInitContext,
        },
        operator::FirehoseOperator,
        planner::OperationPlan,
        signature::FirehoseOperatorSignature,
    },
    schema::{
        BuildPlan,
        DataTypeDescription,
        FirehoseTableSchema,
    },
};

/// `OpEnvironment` is a trait that provides access to a collection of operator
/// bindings.
pub trait FirehoseOperatorEnvironment: Debug + Send + Sync {
    /// Returns a reference to the map of operator bindings.
    // TODO: This should be an iterator.
    fn operators(&self) -> &BTreeMap<String, Arc<dyn FirehoseOperatorFactory>>;

    /// Validates a build plan against the input and output types.
    ///
    /// # Arguments
    ///
    /// * `build_plan` - The build plan for the operator.
    /// * `input_types` - A map of input parameter names to their data types.
    /// * `output_types` - A map of output parameter names to their data types.
    ///
    /// # Returns
    ///
    /// An `anyhow::Result<Arc<dyn FirehoseOperatorFactory>>` containing the
    /// operator factory.
    fn lookup_operator_factory(
        &self,
        operator_id: &str,
    ) -> anyhow::Result<Arc<dyn FirehoseOperatorFactory>> {
        Ok(self
            .operators()
            .get(operator_id)
            .with_context(|| format!("Operator '{operator_id}' not found in environment."))?
            .clone())
    }

    /// Validates the operator's context against the environment.
    ///
    /// By default, this method calls `init_operator` to perform the validation;
    /// and maps successful results to `Ok(())`.
    ///
    /// # Arguments
    ///
    /// * `context` - The context containing the build plan and input/output
    ///   types.
    ///
    /// # Returns
    ///
    /// An `anyhow::Result<()>` indicating successful validation.
    fn validate_context(
        &self,
        plan_context: BuildPlanContext,
    ) -> anyhow::Result<()> {
        self.init_operator(plan_context).map(|_| ())
    }

    /// Initializes an operator based on the provided context.
    ///
    /// # Arguments
    ///
    /// * `plan_context` - The context containing the build plan and
    ///   input/output types.
    ///
    /// # Returns
    ///
    /// An `anyhow::Result<Box<dyn FirehoseOperator>>` containing the
    /// initialized operator.
    fn init_operator(
        &self,
        plan_context: BuildPlanContext,
    ) -> anyhow::Result<Box<dyn FirehoseOperator>> {
        let factory = self.lookup_operator_factory(plan_context.operator_id())?;

        let context = plan_context.bind_signature(factory.signature())?;

        factory.init(&context)
    }

    /// Extends a schema with a new operation.
    ///
    /// Plans new output columns and adds a build plan for the operation.
    ///
    /// Validates the inputs, outputs, and configs against the environment.
    ///
    /// # Arguments
    ///
    /// * `schema` - A mutable reference to the `TableSchema` to be extended.
    /// * `planner` - An `OperationPlanner` that contains the details of the
    ///   operation to be planned.
    ///
    /// # Returns
    ///
    /// An `anyhow::Result<BuildPlan>` containing the build plan for the
    /// operation.
    fn apply_plan_to_schema(
        &self,
        schema: &mut FirehoseTableSchema,
        planner: OperationPlan,
    ) -> anyhow::Result<BuildPlan> {
        let operator_id = &planner.operator_id;

        let factory = self.lookup_operator_factory(operator_id)?;
        let signature = factory.signature();

        let (plan, output_cols) = planner.plan_for_signature(signature)?;

        {
            let mut tmp_schema = schema.clone();
            tmp_schema.extend_via_plan(plan.clone(), &output_cols)?;

            let builder = BuildPlanContext::new(Arc::new(tmp_schema), Arc::new(plan.clone()));
            self.validate_context(builder)?;
        }

        schema.extend_via_plan(plan.clone(), &output_cols)?;

        Ok(plan)
    }
}

/// `MapOpEnvironment` is a simple implementation of `OpEnvironment` that uses a
/// `BTreeMap` to store operators.
#[derive(Debug)]
pub struct MapOpEnvironment {
    /// A map of operator IDs to their corresponding operator factories.
    operators: BTreeMap<String, Arc<dyn FirehoseOperatorFactory>>,
}

impl Default for MapOpEnvironment {
    fn default() -> Self {
        Self::new()
    }
}

impl MapOpEnvironment {
    /// Creates a new `MapOpEnvironment` with an empty operator map.
    pub fn new() -> Self {
        Self {
            operators: BTreeMap::new(),
        }
    }

    /// Creates a `MapOpEnvironment` from a list of operator bindings.
    ///
    /// # Arguments
    ///
    /// * `factories` - factories to add.
    ///
    /// # Returns
    ///
    /// An `anyhow::Result<Self>` containing the initialized environment.
    pub fn from_operators(
        factories: Vec<Arc<dyn FirehoseOperatorFactory>>
    ) -> anyhow::Result<Self> {
        let mut this = Self::new();
        this.add_all_operators(factories)?;
        Ok(this)
    }

    /// Adds a new binding to the environment.
    ///
    /// # Arguments
    ///
    /// * `factory` - factory to add.
    ///
    /// # Returns
    ///
    /// An `anyhow::Result<()>` indicating success or containing an error if the
    /// binding already exists.
    pub fn add_operator(
        &mut self,
        factory: Arc<dyn FirehoseOperatorFactory>,
    ) -> anyhow::Result<()> {
        let id = factory.operator_id();
        if self.operators.contains_key(id) {
            bail!("Operator with ID '{id}' already exists in MapOpEnvironment.");
        }
        self.operators.insert(id.clone(), factory);
        Ok(())
    }

    /// Adds multiple bindings to the environment.
    ///
    /// # Arguments
    ///
    /// * `factories` - factories to add.
    ///
    /// # Returns
    ///
    /// An `anyhow::Result<()>` indicating success or containing an error if any
    /// binding fails to be added.
    pub fn add_all_operators(
        &mut self,
        factories: Vec<Arc<dyn FirehoseOperatorFactory>>,
    ) -> anyhow::Result<()> {
        for binding in factories.into_iter() {
            self.add_operator(binding.clone())?;
        }
        Ok(())
    }
}

impl FirehoseOperatorEnvironment for MapOpEnvironment {
    fn operators(&self) -> &BTreeMap<String, Arc<dyn FirehoseOperatorFactory>> {
        // TODO: This should be an iterator.
        &self.operators
    }
}

/// A partially bound context for initializing an operator build plan.
///
/// This context is bound to a schema and build plan,
/// but not yet to an operator signature.
#[derive(Debug, Clone)]
pub struct BuildPlanContext {
    /// The table schema that this context is bound to.
    table_schema: Arc<FirehoseTableSchema>,

    /// The build plan that this context is bound to.
    build_plan: Arc<BuildPlan>,
}

impl BuildPlanContext {
    /// Creates a new `OperationInitPlanContext` with the given table schema and
    /// build plan.
    pub fn new(
        table_schema: Arc<FirehoseTableSchema>,
        build_plan: Arc<BuildPlan>,
    ) -> Self {
        Self {
            table_schema,
            build_plan,
        }
    }

    /// Returns a reference to the table schema.
    pub fn operator_id(&self) -> &str {
        &self.build_plan().operator_id
    }

    /// Returns the table schema that this context is bound to.
    pub fn table_schema(&self) -> &FirehoseTableSchema {
        &self.table_schema
    }

    /// Returns the build plan that this context is bound to.
    pub fn build_plan(&self) -> &BuildPlan {
        &self.build_plan
    }

    /// Binds the context to a specific operator signature.
    ///
    /// # Returns
    ///
    /// An `anyhow::Result<OperationInitializationContext>` containing the bound
    /// context.
    pub fn bind_signature(
        self,
        signature: &FirehoseOperatorSignature,
    ) -> anyhow::Result<OperationInitializationContext> {
        OperationInitializationContext::init(self, signature.clone())
    }

    /// Computes the input types for the operator based on the build plan and
    /// table schema.
    pub fn input_types(&self) -> BTreeMap<String, DataTypeDescription> {
        self.build_plan
            .inputs
            .iter()
            .map(|(pname, cname)| {
                (
                    pname.clone(),
                    self.table_schema[cname.as_ref()].data_type.clone(),
                )
            })
            .collect()
    }

    /// Computes the output types for the operator based on the build plan and
    /// table schema.
    pub fn output_types(&self) -> BTreeMap<String, DataTypeDescription> {
        self.build_plan
            .outputs
            .iter()
            .map(|(pname, cname)| {
                (
                    pname.clone(),
                    self.table_schema[cname.as_ref()].data_type.clone(),
                )
            })
            .collect()
    }
}

/// An operator factory which deserializes the operator from a JSON value.
/// Context for validating and initializing a column build operation.
#[derive(Debug, Clone)]
pub struct OperationInitializationContext {
    /// The build plan context that this signature context wraps.
    plan_context: BuildPlanContext,

    /// The table schema that this operator is bound to.
    signature: FirehoseOperatorSignature,
}

impl FirehoseOperatorInitContext for OperationInitializationContext {
    fn operator_id(&self) -> &str {
        &self.build_plan().operator_id
    }

    fn table_schema(&self) -> &FirehoseTableSchema {
        &self.plan_context.table_schema
    }

    fn build_plan(&self) -> &BuildPlan {
        self.plan_context().build_plan()
    }

    fn signature(&self) -> &FirehoseOperatorSignature {
        &self.signature
    }
}

impl OperationInitializationContext {
    /// Creates a new `OperationInitSignatureContext` with the given plan
    /// context and signature.
    ///
    /// # Returns
    ///
    /// An `anyhow::Result<Self>` containing the initialized context.
    pub fn init(
        plan_context: BuildPlanContext,
        signature: FirehoseOperatorSignature,
    ) -> anyhow::Result<Self> {
        signature.validate(&plan_context.input_types(), &plan_context.output_types())?;

        Ok(Self {
            plan_context,
            signature,
        })
    }

    /// Returns the plan context.
    pub fn plan_context(&self) -> &BuildPlanContext {
        &self.plan_context
    }

    /// Returns a reference to the table schema.
    pub fn table_schema(&self) -> &FirehoseTableSchema {
        self.plan_context().table_schema()
    }

    /// Returns the signature of the operator.
    pub fn signature(&self) -> &FirehoseOperatorSignature {
        &self.signature
    }
}
