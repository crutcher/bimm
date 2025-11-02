# resnet-finetune example

The dataload on this example is base on the tracel-ai models repository:
https://github.com/tracel-ai/models/blob/main/resnet-burn/examples/finetune/examples/finetune.rs

## Running the Example

This will download both model weights and a fine-tune dataset:

```bash
cargo run --release -p resnet_finetune
```

## List Available Models

This will list all available pretrained models:

```terminaloutput
$ cargo run --release -p resnet_finetune -- --pretrained list
Available pretrained models:
* "resnet18"
ResNetContractConfig { layers: [2, 2, 2, 2], num_classes: 1000, stem_width: 64, output_stride: 32, bottleneck_policy: None, normalization: Batch(BatchNormConfig { num_features: 0, epsilon: 1e-5, momentum: 0.1 }), activation: Relu }
  - "resnet18.tv_in1k": TorchVision ResNet-18
  - "resnet18.a1_in1k": RSB Paper ResNet-18 a1
  - "resnet18.a2_in1k": RSB Paper ResNet-18 a2
  - "resnet18.a3_in1k": RSB Paper ResNet-18 a3
* "resnet34"
ResNetContractConfig { layers: [3, 4, 6, 3], num_classes: 1000, stem_width: 64, output_stride: 32, bottleneck_policy: None, normalization: Batch(BatchNormConfig { num_features: 0, epsilon: 1e-5, momentum: 0.1 }), activation: Relu }
  - "resnet34.tv_in1k": TorchVision ResNet-34
  - "resnet34.a1_in1k": RSB Paper ResNet-32 a1
  - "resnet34.a2_in1k": RSB Paper ResNet-32 a2
  - "resnet34.a3_in1k": RSB Paper ResNet-32 a3
  - "resnet34.bt_in1k": ResNet-34 pretrained on ImageNet
* "resnet50"
ResNetContractConfig { layers: [3, 4, 6, 3], num_classes: 1000, stem_width: 64, output_stride: 32, bottleneck_policy: Some(BottleneckPolicyConfig { pinch_factor: 4 }), normalization: Batch(BatchNormConfig { num_features: 0, epsilon: 1e-5, momentum: 0.1 }), activation: Relu }
  - "resnet50.tv_in1k": TorchVision ResNet-50
* "resnet101"
ResNetContractConfig { layers: [3, 4, 23, 3], num_classes: 1000, stem_width: 64, output_stride: 32, bottleneck_policy: Some(BottleneckPolicyConfig { pinch_factor: 4 }), normalization: Batch(BatchNormConfig { num_features: 0, epsilon: 1e-5, momentum: 0.1 }), activation: Relu }
  - "resnet101.tv_in1k": TorchVision ResNet-101
  - "resnet101.a1_in1k": ResNet-101 pretrained on ImageNet
```
