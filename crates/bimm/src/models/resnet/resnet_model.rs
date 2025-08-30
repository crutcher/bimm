//! # `ResNet` Core Model
use crate::layers::activation::{Activation, ActivationConfig};
use crate::layers::blocks::conv_norm::{Conv2dNormBlock, Conv2dNormBlockConfig};
use crate::models::resnet::layer_block::{LayerBlock, LayerBlockConfig};
use burn::module::Module;
use burn::nn::conv::Conv2dConfig;
use burn::nn::pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig, MaxPool2d, MaxPool2dConfig};
use burn::nn::{Linear, LinearConfig, PaddingConfig2d};
use burn::prelude::{Backend, Tensor};

/// `ResNet` model.
#[derive(Module, Debug)]
pub struct ResNet<B: Backend> {
    cn: Conv2dNormBlock<B>,
    act: Activation<B>,
    maxpool: MaxPool2d,

    layer1: LayerBlock<B>,
    layer2: LayerBlock<B>,
    layer3: LayerBlock<B>,
    layer4: LayerBlock<B>,

    avgpool: AdaptiveAvgPool2d,
    fc: Linear<B>,
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

        // 7x7 conv, 64, /2
        let conv1: Conv2dNormBlockConfig = Conv2dConfig::new([3, 64], [7, 7])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(3, 3))
            .with_bias(false)
            .into();

        // 3x3 maxpool, /2
        let maxpool = MaxPool2dConfig::new([3, 3])
            .with_strides([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1));

        // Residual blocks
        let bottleneck = expansion > 1;
        let layer1 = LayerBlockConfig::build(blocks[0], 64, 64 * expansion, 1, bottleneck);
        let layer2 =
            LayerBlockConfig::build(blocks[1], 64 * expansion, 128 * expansion, 2, bottleneck);
        let layer3 =
            LayerBlockConfig::build(blocks[2], 128 * expansion, 256 * expansion, 2, bottleneck);
        let layer4 =
            LayerBlockConfig::build(blocks[3], 256 * expansion, 512 * expansion, 2, bottleneck);

        // Average pooling [B, 512 * expansion, H, W] -> [B, 512 * expansion, 1, 1]
        let avgpool = AdaptiveAvgPool2dConfig::new([1, 1]);

        // Output layer
        let fc = LinearConfig::new(512 * expansion, num_classes);

        Self {
            cn: conv1.init(device),
            act: ActivationConfig::Relu.init(device),
            maxpool: maxpool.init(),

            layer1: layer1.init(device),
            layer2: layer2.init(device),
            layer3: layer3.init(device),
            layer4: layer4.init(device),

            avgpool: avgpool.init(),
            fc: fc.init(device),
        }
    }

    /// `ResNet` forward pass.
    pub fn forward(
        &self,
        input: Tensor<B, 4>,
    ) -> Tensor<B, 2> {
        // Prep block
        let x = self.cn.forward(input);
        let x = self.act.forward(x);
        let x = self.maxpool.forward(x);

        // Residual blocks
        let x = self.layer1.forward(x);
        let x = self.layer2.forward(x);
        let x = self.layer3.forward(x);
        let x = self.layer4.forward(x);

        // Head
        let x = self.avgpool.forward(x);
        // Reshape [B, C, 1, 1] -> [B, C]
        let x = x.flatten(1, 3);

        self.fc.forward(x)
    }
}
