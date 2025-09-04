//! # Activation Layer Wrapper
use burn::nn::{
    Gelu, HardSigmoid, HardSigmoidConfig, LeakyRelu, LeakyReluConfig, PRelu, PReluConfig, Relu,
    Sigmoid, SwiGlu, SwiGluConfig, Tanh,
};
use burn::prelude::{Backend, Config, Module, Tensor};

/// [`Activation`] Configuration.
#[derive(Config, Debug)]
#[non_exhaustive]
pub enum ActivationConfig {
    /// [`Gelu`] activation layer.
    Gelu,

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

impl From<LeakyReluConfig> for ActivationConfig {
    fn from(config: LeakyReluConfig) -> Self {
        Self::LeakyRelu(config)
    }
}

impl From<PReluConfig> for ActivationConfig {
    fn from(config: PReluConfig) -> Self {
        Self::PRelu(config)
    }
}

impl From<SwiGluConfig> for ActivationConfig {
    fn from(config: SwiGluConfig) -> Self {
        Self::SwiGlu(config)
    }
}

impl From<HardSigmoidConfig> for ActivationConfig {
    fn from(config: HardSigmoidConfig) -> Self {
        Self::HardSigmoid(config)
    }
}

impl Default for ActivationConfig {
    fn default() -> Self {
        Self::Relu
    }
}

impl ActivationConfig {
    /// Initialize a wrapped activation layer.
    pub fn init<B: Backend>(
        &self,
        device: &B::Device,
    ) -> Activation<B> {
        match self {
            ActivationConfig::Relu => Activation::Relu(Relu),
            ActivationConfig::LeakyRelu(conf) => Activation::LeakyRelu(conf.init()),
            ActivationConfig::Gelu => Activation::Gelu(Gelu),
            ActivationConfig::PRelu(conf) => Activation::PRelu(conf.init(device)),
            ActivationConfig::SwiGlu(conf) => Activation::SwiGlu(conf.init(device)),
            ActivationConfig::HardSigmoid(conf) => Activation::HardSigmoid(conf.init()),
            ActivationConfig::Sigmoid => Activation::Sigmoid(Sigmoid),
            ActivationConfig::Tanh => Activation::Tanh(Tanh),
        }
    }
}

/// Activation Layer Wrapper.
///
/// Provides support for many in-built `burn::nn` activations.
#[derive(Module, Debug)]
#[non_exhaustive]
pub enum Activation<B: Backend> {
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

impl<B: Backend> Activation<B> {
    /// Forward pass.
    #[tracing::instrument]
    pub fn forward<const D: usize>(
        &self,
        input: Tensor<B, D>,
    ) -> Tensor<B, D> {
        match self {
            Activation::Relu(layer) => layer.forward(input),
            Activation::LeakyRelu(layer) => layer.forward(input),
            Activation::Gelu(layer) => layer.forward(input),
            Activation::PRelu(layer) => layer.forward(input),
            Activation::SwiGlu(layer) => layer.forward(input),
            Activation::HardSigmoid(layer) => layer.forward(input),
            Activation::Sigmoid(layer) => layer.forward(input),
            Activation::Tanh(layer) => layer.forward(input),
        }
    }

    /// Build a [`ActivationConfig`] for this module.
    pub fn to_config(&self) -> ActivationConfig {
        match self {
            Activation::Relu(_) => ActivationConfig::Relu,
            Activation::LeakyRelu(layer) => LeakyReluConfig::new()
                .with_negative_slope(layer.negative_slope)
                .into(),
            Activation::Gelu(_) => ActivationConfig::Gelu,
            Activation::PRelu(layer) => PReluConfig::new()
                .with_alpha(layer.alpha_value)
                .with_num_parameters(layer.num_params())
                .into(),
            Activation::SwiGlu(layer) => {
                let [d_output, d_input] = layer.linear_inner.weight.shape().dims();
                SwiGluConfig::new(d_input, d_output)
                    .with_bias(layer.linear_inner.bias.is_some())
                    .into()
            }
            Activation::HardSigmoid(layer) => HardSigmoidConfig::new()
                .with_alpha(layer.alpha)
                .with_beta(layer.beta)
                .into(),
            Activation::Sigmoid(_) => ActivationConfig::Sigmoid,
            Activation::Tanh(_) => ActivationConfig::Tanh,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::nn::{
        HardSigmoidConfig, LeakyReluConfig, Linear, LinearConfig, PReluConfig, SwiGlu, SwiGluConfig,
    };

    type TestBackend = NdArray<f32>;

    #[derive(Config, Debug)]
    pub struct TestConfig {
        fc: LinearConfig,
        act: ActivationConfig,
    }

    impl TestConfig {
        pub fn init<B: Backend>(
            self,
            device: &B::Device,
        ) -> TestModule<B> {
            let fc = self.fc.init(device);
            let act = self.act.init(device);
            TestModule { fc, act }
        }
    }

    #[derive(Module, Debug)]
    pub struct TestModule<B: Backend> {
        fc: Linear<B>,
        act: Activation<B>,
    }

    impl<B: Backend> TestModule<B> {
        pub fn forward(
            &self,
            input: Tensor<B, 2>,
        ) -> Tensor<B, 2> {
            let output = self.fc.forward(input);
            self.act.forward(output)
        }
    }

    #[test]
    fn test_embedded_roundtrip() {
        let device = Default::default();
        let config = TestConfig {
            fc: LinearConfig::new(2, 2),
            act: ActivationConfig::Gelu,
        };

        let source_module: TestModule<TestBackend> = config.clone().init(&device);

        let input = Tensor::from_data([[1.0, 2.0], [3.0, 4.0]], &device);
        let output1 = source_module.forward(input.clone());

        let record = source_module.into_record();

        let reload_module: TestModule<TestBackend> = config.init(&device).load_record(record);
        let output2 = reload_module.forward(input.clone());

        output1.to_data().assert_eq(&output2.to_data(), true);
    }

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
        config: ActivationConfig,
        input: Tensor<B, D>,
        expected: Tensor<B, D>,
        device: &B::Device,
    ) {
        let act1 = config.init(device);
        let record: ActivationRecord<B> = act1.into_record();

        let act = config.init(device).load_record(record);

        let output = act.forward(input);
        expect_tensor(output, expected);
    }

    #[test]
    fn test_gelu() {
        let device = Default::default();
        let input = make_input::<TestBackend>(&device);

        let expected = Gelu::default().forward(input.clone());

        check_stateless_config_output(ActivationConfig::Gelu, input, expected, &device)
    }

    #[test]
    fn test_prelu() {
        let device = Default::default();
        let input = make_input::<TestBackend>(&device);

        let inner_config = PReluConfig::new();
        let expected = inner_config.init(&device).forward(input.clone());

        check_stateless_config_output(
            ActivationConfig::PRelu(inner_config),
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

        check_stateless_config_output(ActivationConfig::Relu, input, expected, &device)
    }

    #[test]
    fn test_leaky_relu() {
        let device = Default::default();
        let input = make_input::<TestBackend>(&device);

        let inner_config = LeakyReluConfig::new();
        let expected = inner_config.init().forward(input.clone());

        check_stateless_config_output(
            ActivationConfig::LeakyRelu(inner_config),
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

        let config = ActivationConfig::SwiGlu(inner_config);
        let layer = config.init(&device);

        match &layer {
            Activation::SwiGlu(inner) => {
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

        check_stateless_config_output(ActivationConfig::Sigmoid, input, expected, &device)
    }

    #[test]
    fn test_tanh() {
        let device = Default::default();
        let input = make_input::<TestBackend>(&device);

        let expected = Tanh::default().forward(input.clone());

        check_stateless_config_output(ActivationConfig::Tanh, input, expected, &device)
    }

    #[test]
    fn test_hard_sigmoid() {
        let device = Default::default();
        let input = make_input::<TestBackend>(&device);

        let inner_config = HardSigmoidConfig::new();
        let expected = inner_config.init().forward(input.clone());

        check_stateless_config_output(
            ActivationConfig::HardSigmoid(inner_config),
            input,
            expected,
            &device,
        )
    }
}
