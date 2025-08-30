//! # `ResNet` Core Model
use crate::layers::activation::{Activation, ActivationConfig};
use crate::layers::blocks::conv_norm::{ConvNorm2d, ConvNorm2dConfig};
use crate::models::resnet::layer_block::{LayerBlock, LayerBlockConfig, LayerBlockMeta};
use crate::models::resnet::util::CONV_INTO_RELU_INITIALIZER;
use burn::module::Module;
use burn::nn::conv::Conv2dConfig;
use burn::nn::pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig, MaxPool2d, MaxPool2dConfig};
use burn::nn::{Initializer, Linear, LinearConfig, PaddingConfig2d};
use burn::prelude::{Backend, Config, Tensor};

/// [`ResNet`] Structure Config.
///
/// This config defines the structure of a converted `ResNet` model.
/// It is not a semantic configuration and does not check the validity
/// of the internal sizes before or during construction.
#[derive(Config, Debug)]
pub struct ResnetStructureConfig {
    /// The input Conv/Norm block configuration.
    pub input_conv_norm: ConvNorm2dConfig,

    /// Optional override for the input Conv2d initializer.
    #[config(default = "CONV_INTO_RELU_INITIALIZER.clone().into()")]
    pub input_conv_norm_initializer: Option<Initializer>,

    /// The input activation configuration.
    #[config(default = "ActivationConfig::Relu")]
    pub input_act: ActivationConfig,

    /// The inner layers configuration.
    pub layers: Vec<LayerBlockConfig>,

    /// The number of classes.
    pub num_classes: usize,
}

impl ResnetStructureConfig {
    /// Initialize a [`ResNet`] model.
    pub fn init<B: Backend>(
        self,
        device: &B::Device,
    ) -> ResNet<B> {
        let mut input_conv_norm = self.input_conv_norm.clone();
        if self.input_conv_norm_initializer.is_some() {
            input_conv_norm.conv = input_conv_norm
                .conv
                .with_initializer(self.input_conv_norm_initializer.unwrap());
        }

        let head_planes = self.layers.last().unwrap().out_planes();

        ResNet {
            input_conv_norm: input_conv_norm.init(device),
            input_act: self.input_act.init(device),
            input_pool: MaxPool2dConfig::new([3, 3])
                .with_strides([2, 2])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .init(),

            layers: self
                .layers
                .into_iter()
                .map(|c| c.init(device))
                .collect::<Vec<_>>(),

            output_pool: AdaptiveAvgPool2dConfig::new([1, 1]).init(),
            output_fc: LinearConfig::new(head_planes, self.num_classes).init(device),
        }
    }
}

/// `ResNet` model.
#[derive(Module, Debug)]
pub struct ResNet<B: Backend> {
    input_conv_norm: ConvNorm2d<B>,
    input_act: Activation<B>,
    input_pool: MaxPool2d,

    layers: Vec<LayerBlock<B>>,

    output_pool: AdaptiveAvgPool2d,
    output_fc: Linear<B>,
}

impl<B: Backend> ResNet<B> {
    /// Create a new instance of `ResNet`.
    pub fn legacy_new(
        blocks: [usize; 4],
        num_classes: usize,
        expansion: usize,
        device: &B::Device,
    ) -> Self {
        // `new()` is private but still check just in case...
        assert!(
            expansion == 1 || expansion == 4,
            "ResNet module only supports expansion values [1, 4] for residual blocks"
        );

        // Residual blocks
        let bottleneck = expansion > 1;

        let make_block = |idx: usize, in_factor: usize, out_factor: usize, stride: usize| {
            LayerBlockConfig::build(
                blocks[idx],
                64 * in_factor,
                64 * out_factor,
                stride,
                bottleneck,
            )
        };

        ResnetStructureConfig::new(
            ConvNorm2dConfig::from(
                Conv2dConfig::new([3, 64], [7, 7])
                    .with_stride([2, 2])
                    .with_padding(PaddingConfig2d::Explicit(3, 3))
                    .with_bias(false),
            ),
            vec![
                make_block(blocks[0], 1, expansion, 1),
                make_block(blocks[1], expansion, 2 * expansion, 2),
                make_block(blocks[2], 2 * expansion, 4 * expansion, 2),
                make_block(blocks[3], 4 * expansion, 8 * expansion, 2),
            ],
            num_classes,
        )
        .init(device)
    }

    /// `ResNet` forward pass.
    pub fn forward(
        &self,
        input: Tensor<B, 4>,
    ) -> Tensor<B, 2> {
        // Prep block
        let x = self.input_conv_norm.forward(input);
        let x = self.input_act.forward(x);
        let x = self.input_pool.forward(x);

        // Residual blocks
        let x = self.layers.iter().fold(x, |x, layer| layer.forward(x));

        // Head
        let x = self.output_pool.forward(x);
        // Reshape [B, C, 1, 1] -> [B, C]
        let x = x.flatten(1, 3);
        self.output_fc.forward(x)
    }
}
