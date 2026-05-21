//! Type mapper for changing the data type of tensors in a module.
use burn::{
    Tensor,
    module::{
        ModuleMapper,
        Param,
    },
    prelude::Backend,
    tensor::DType,
};

/// Type mapper for changing the data type of tensors in a module.
pub struct DTypeMapper<B: Backend> {
    /// Target data type for tensor conversion.
    pub dt: DType,

    /// Phantom data to ensure that `B` is a valid backend.
    phantom: std::marker::PhantomData<B>,
}

impl<B: Backend> DTypeMapper<B> {
    /// Create a new type mapper with the specified target data type.
    pub fn new(dt: DType) -> Self {
        Self {
            dt,
            phantom: std::marker::PhantomData,
        }
    }
}

impl<B: Backend> ModuleMapper<B> for DTypeMapper<B> {
    fn map_float<const D: usize>(
        &mut self,
        param: Param<Tensor<B, D>>,
    ) -> Param<Tensor<B, D>> {
        let (id, tensor, mapper) = param.consume();
        Param::from_mapped_value(id, tensor.cast(self.dt), mapper)
    }
}
