use burn::nn::{
    Gelu, HardSigmoid, HardSigmoidConfig, LeakyRelu, LeakyReluConfig, PRelu, PReluConfig, Relu,
    Sigmoid, SwiGlu, SwiGluConfig, Tanh,
};
use burn::prelude::{Backend, Config, Module, Tensor};

/// [`ActivationLayer`] Configuration.
#[derive(Config, Debug)]
#[non_exhaustive]
pub enum ActivationLayerConfig {
    /// [`Gelu`] activation layer.
    GeLu,

    /// [`PRelu`] activation layer.
    PRelu(PReluConfig),

    /// [`Relu`] activation layer.
    Relu,

    /// [`LeakyRelu`] activation layer.
    LeakyRelu(LeakyReluConfig),

    /// [`SwiGlu`] activation layer.
    SwiGlu(SwiGluConfig),

    /// [`Sigmoid`] activation layer.
    Sigmoid,

    /// [`Tanh`] activation layer.
    Tanh,

    /// [`HardSigmoid`] activation layer.
    HardSigmoid(HardSigmoidConfig),
}

impl Default for ActivationLayerConfig {
    fn default() -> Self {
        Self::Relu
    }
}

impl ActivationLayerConfig {
    /// Initialize a wrapped activation layer.
    pub fn init<B: Backend>(
        &self,
        device: &B::Device,
    ) -> ActivationLayer<B> {
        match self {
            ActivationLayerConfig::Relu => ActivationLayer::Relu(Relu),
            ActivationLayerConfig::LeakyRelu(conf) => ActivationLayer::LeakyRelu(conf.init()),
            ActivationLayerConfig::GeLu => ActivationLayer::Gelu(Gelu),
            ActivationLayerConfig::PRelu(conf) => ActivationLayer::PRelu(conf.init(device)),
            ActivationLayerConfig::SwiGlu(conf) => ActivationLayer::SwiGlu(conf.init(device)),
            ActivationLayerConfig::HardSigmoid(conf) => ActivationLayer::HardSigmoid(conf.init()),
            ActivationLayerConfig::Sigmoid => ActivationLayer::Sigmoid(Sigmoid),
            ActivationLayerConfig::Tanh => ActivationLayer::Tanh(Tanh),
        }
    }
}

/// Activation Layer Wrapper.
///
/// Provides support for many in-built `burn::nn` activations.
#[derive(Module, Debug)]
#[non_exhaustive]
pub enum ActivationLayer<B: Backend> {
    /// [`Gelu`] activation layer.
    Gelu(Gelu),

    /// [`PRelu`] activation layer.
    PRelu(PRelu<B>),

    /// [`Relu`] activation layer.
    Relu(Relu),

    /// [`LeakyRelu`] activation layer.
    LeakyRelu(LeakyRelu),

    /// [`SwiGlu`] activation layer.
    SwiGlu(SwiGlu<B>),

    /// [`Sigmoid`] activation layer.
    Sigmoid(Sigmoid),

    /// [`Tanh`] activation layer.
    Tanh(Tanh),

    /// [`HardSigmoid`] activation layer.
    HardSigmoid(HardSigmoid),
}

impl<B: Backend> ActivationLayer<B> {
    /// Forward pass.
    pub fn forward<const D: usize>(
        &self,
        input: Tensor<B, D>,
    ) -> Tensor<B, D> {
        match self {
            ActivationLayer::Relu(layer) => layer.forward(input),
            ActivationLayer::LeakyRelu(layer) => layer.forward(input),
            ActivationLayer::Gelu(layer) => layer.forward(input),
            ActivationLayer::PRelu(layer) => layer.forward(input),
            ActivationLayer::SwiGlu(layer) => layer.forward(input),
            ActivationLayer::HardSigmoid(layer) => layer.forward(input),
            ActivationLayer::Sigmoid(layer) => layer.forward(input),
            ActivationLayer::Tanh(layer) => layer.forward(input),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::nn::{HardSigmoidConfig, LeakyReluConfig, PReluConfig, SwiGlu, SwiGluConfig};

    type TestBackend = NdArray<f32>;

    fn make_input<B: Backend>(device: &B::Device) -> Tensor<B, 2> {
        Tensor::from_data([[-1.0, -0.5, 0.0], [1.0, 0.5, 0.0]], device)
    }

    fn expect_tensor<B: Backend, const D: usize>(
        actual: Tensor<B, D>,
        expected: Tensor<B, D>,
    ) {
        actual.to_data().assert_eq(&expected.to_data(), true);
    }

    fn check_stateless_config_output<B: Backend, const D: usize>(
        config: ActivationLayerConfig,
        input: Tensor<B, D>,
        expected: Tensor<B, D>,
        device: &B::Device,
    ) {
        let act = config.init(device);
        let output = act.forward(input);
        expect_tensor(output, expected);
    }

    #[test]
    fn test_gelu() {
        let device = Default::default();
        let input = make_input::<TestBackend>(&device);

        let expected = Gelu::default().forward(input.clone());

        check_stateless_config_output(ActivationLayerConfig::GeLu, input, expected, &device)
    }

    #[test]
    fn test_prelu() {
        let device = Default::default();
        let input = make_input::<TestBackend>(&device);

        let inner_config = PReluConfig::new();
        let expected = inner_config.init(&device).forward(input.clone());

        check_stateless_config_output(
            ActivationLayerConfig::PRelu(inner_config),
            input,
            expected,
            &device,
        )
    }

    #[test]
    fn test_relu() {
        let device = Default::default();
        let input = make_input::<TestBackend>(&device);

        let expected = Relu::default().forward(input.clone());

        check_stateless_config_output(ActivationLayerConfig::Relu, input, expected, &device)
    }

    #[test]
    fn test_leaky_relu() {
        let device = Default::default();
        let input = make_input::<TestBackend>(&device);

        let inner_config = LeakyReluConfig::new();
        let expected = inner_config.init().forward(input.clone());

        check_stateless_config_output(
            ActivationLayerConfig::LeakyRelu(inner_config),
            input,
            expected,
            &device,
        )
    }

    #[test]
    fn test_swi_glu() {
        let device = Default::default();
        let input = make_input::<TestBackend>(&device);

        let d_input = input.shape().dims[1];
        let d_output = 2 * d_input;

        let inner_config = SwiGluConfig::new(d_input, d_output);
        let mut reference: SwiGlu<TestBackend> = inner_config.init(&device);

        let config = ActivationLayerConfig::SwiGlu(inner_config);
        let layer = config.init(&device);

        match &layer {
            ActivationLayer::SwiGlu(inner) => {
                // Clone the initialized weights.
                let state = inner.clone().into_record();
                reference = reference.load_record(state);
            }
            _ => unreachable!(),
        };

        expect_tensor(
            layer.forward(input.clone()),
            reference.forward(input.clone()),
        )
    }

    #[test]
    fn test_sigmoid() {
        let device = Default::default();
        let input = make_input::<TestBackend>(&device);

        let expected = Sigmoid::default().forward(input.clone());

        check_stateless_config_output(ActivationLayerConfig::Sigmoid, input, expected, &device)
    }

    #[test]
    fn test_tanh() {
        let device = Default::default();
        let input = make_input::<TestBackend>(&device);

        let expected = Tanh::default().forward(input.clone());

        check_stateless_config_output(ActivationLayerConfig::Tanh, input, expected, &device)
    }

    #[test]
    fn test_hard_sigmoid() {
        let device = Default::default();
        let input = make_input::<TestBackend>(&device);

        let inner_config = HardSigmoidConfig::new();
        let expected = inner_config.init().forward(input.clone());

        check_stateless_config_output(
            ActivationLayerConfig::HardSigmoid(inner_config),
            input,
            expected,
            &device,
        )
    }
}
