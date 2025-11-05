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
Available pretrained models:
* "resnet18"
ResNetContractConfig { layers: [2, 2, 2, 2], num_classes: 1000, stem_width: 64, output_stride: 32, bottleneck_policy: None, normalization: Batch(BatchNormConfig { num_features: 0, epsilon: 1e-5, momentum: 0.1 }), activation: Relu }
  - "resnet18.tv_in1k": TorchVision ResNet-18
  - "resnet18.a1_in1k": RSB Paper ResNet-18 a1
  - "resnet18.a2_in1k": RSB Paper ResNet-18 a2
  - "resnet18.a3_in1k": RSB Paper ResNet-18 a3
* "resnet26"
ResNetContractConfig { layers: [2, 2, 2, 2], num_classes: 1000, stem_width: 64, output_stride: 32, bottleneck_policy: Some(BottleneckPolicyConfig { pinch_factor: 4 }), normalization: Batch(BatchNormConfig { num_features: 0, epsilon: 1e-5, momentum: 0.1 }), activation: Relu }
  - "resnet26.bt_in1k": ResNet-26 pretrained on ImageNet
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
* "resnet152"
ResNetContractConfig { layers: [3, 8, 36, 3], num_classes: 1000, stem_width: 64, output_stride: 32, bottleneck_policy: Some(BottleneckPolicyConfig { pinch_factor: 4 }), normalization: Batch(BatchNormConfig { num_features: 0, epsilon: 1e-5, momentum: 0.1 }), activation: Relu }
  - "resnet152.tv_in1k": TorchVision ResNet-152
```

## Various Options to Rewrite Models

```terminaloutput

$ cargo run --release -p resnet_finetune --features cuda -- --cautious-weight-decay
| Split | Metric                         | Min.     | Epoch    | Max.     | Epoch    |
|-------|--------------------------------|----------|----------|----------|----------|
| Train | CPU Memory                     | 16.340   | 1        | 19.275   | 59       |
| Train | CPU Usage                      | 1.752    | 26       | 3.208    | 59       |
| Train | Hamming Score @ Threshold(0.5) | 82.597   | 1        | 97.160   | 48       |
| Train | Learning Rate                  | 1.284e-8 | 58       | 4.999e-5 | 1        |
| Train | Loss                           | 0.147    | 50       | 0.642    | 1        |
| Valid | CPU Memory                     | 16.268   | 1        | 19.412   | 59       |
| Valid | CPU Usage                      | 1.682    | 7        | 3.643    | 58       |
| Valid | Hamming Score @ Threshold(0.5) | 84.843   | 1        | 95.235   | 38       |
| Valid | Loss                           | 0.146    | 52       | 0.624    | 1        |
Args {
    seed: 0,
    train_percentage: 70,
    artifact_dir: "/tmp/resnet_finetune",
    batch_size: 24,
    grads_accumulation: 8,
    smoothing: Some(
        0.1,
    ),
    num_workers: 4,
    num_epochs: 60,
    patience: 20,
    pretrained: "resnet50.tv_in1k",
    replace_activation: None,
    freeze_layers: false,
    drop_block_prob: 0.2,
    stochastic_depth_prob: 0.05,
    learning_rate: 5e-5,
    cautious_weight_decay: true,
    weight_decay: 0.02,
}

$ cargo run --release -p resnet_finetune --features cuda --
| Split | Metric                         | Min.     | Epoch    | Max.     | Epoch    |
|-------|--------------------------------|----------|----------|----------|----------|
| Train | CPU Memory                     | 16.192   | 1        | 18.592   | 59       |
| Train | CPU Usage                      | 1.704    | 57       | 2.672    | 47       |
| Train | Hamming Score @ Threshold(0.5) | 82.580   | 1        | 97.126   | 49       |
| Train | Learning Rate                  | 1.284e-8 | 58       | 4.999e-5 | 1        |
| Train | Loss                           | 0.147    | 50       | 0.643    | 1        |
| Valid | CPU Memory                     | 16.235   | 1        | 18.607   | 59       |
| Valid | CPU Usage                      | 1.704    | 1        | 3.640    | 41       |
| Valid | Hamming Score @ Threshold(0.5) | 84.902   | 1        | 95.196   | 53       |
| Valid | Loss                           | 0.148    | 55       | 0.621    | 1        |
Args {
    seed: 0,
    train_percentage: 70,
    artifact_dir: "/tmp/resnet_finetune",
    batch_size: 24,
    grads_accumulation: 8,
    smoothing: Some(
        0.1,
    ),
    num_workers: 4,
    num_epochs: 60,
    patience: 20,
    pretrained: "resnet50.tv_in1k",
    replace_activation: None,
    freeze_layers: false,
    drop_block_prob: 0.2,
    stochastic_depth_prob: 0.05,
    learning_rate: 5e-5,
    cautious_weight_decay: false,
    weight_decay: 0.02,
}

$ cargo run --release -p resnet_finetune --features cuda -- --replace-activation leaky-relu --cautious-weight-decay
| Split | Metric                         | Min.     | Epoch    | Max.     | Epoch    |
|-------|--------------------------------|----------|----------|----------|----------|
| Train | CPU Memory                     | 16.486   | 1        | 18.592   | 60       |
| Train | CPU Usage                      | 1.492    | 31       | 2.761    | 3        |
| Train | Hamming Score @ Threshold(0.5) | 82.647   | 1        | 97.168   | 55       |
| Train | Learning Rate                  | 1.284e-8 | 58       | 4.999e-5 | 1        |
| Train | Loss                           | 0.146    | 50       | 0.642    | 1        |
| Valid | CPU Memory                     | 16.585   | 1        | 18.600   | 60       |
| Valid | CPU Usage                      | 1.452    | 32       | 5.302    | 3        |
| Valid | Hamming Score @ Threshold(0.5) | 84.941   | 1        | 95.294   | 39       |
| Valid | Loss                           | 0.147    | 55       | 0.620    | 1        |

Training completed in 12m53s
Args {
    seed: 0,
    train_percentage: 70,
    artifact_dir: "/tmp/resnet_finetune",
    batch_size: 24,
    grads_accumulation: 8,
    smoothing: Some(
        0.1,
    ),
    num_workers: 4,
    num_epochs: 60,
    patience: 20,
    pretrained: "resnet50.tv_in1k",
    replace_activation: Some(
        LeakyRelu,
    ),
    freeze_layers: false,
    drop_block_prob: 0.2,
    stochastic_depth_prob: 0.05,
    learning_rate: 5e-5,
    cautious_weight_decay: true,
    weight_decay: 0.02,
}

$ cargo run --release -p resnet_finetune --features cuda -- --replace-activation gelu --cautious-weight-decay
| Split | Metric                         | Min.     | Epoch    | Max.     | Epoch    |
|-------|--------------------------------|----------|----------|----------|----------|
| Train | CPU Memory                     | 16.276   | 1        | 18.669   | 60       |
| Train | CPU Usage                      | 1.491    | 46       | 2.209    | 2        |
| Train | Hamming Score @ Threshold(0.5) | 83.244   | 1        | 95.790   | 58       |
| Train | Learning Rate                  | 1.284e-8 | 58       | 4.999e-5 | 1        |
| Train | Loss                           | 0.171    | 50       | 0.654    | 1        |
| Valid | CPU Memory                     | 16.409   | 1        | 18.673   | 60       |
| Valid | CPU Usage                      | 1.494    | 1        | 2.902    | 15       |
| Valid | Hamming Score @ Threshold(0.5) | 83.373   | 1        | 94.706   | 57       |
| Valid | Loss                           | 0.159    | 55       | 0.637    | 1        |

Training completed in 13m12s
Args {
    seed: 0,
    train_percentage: 70,
    artifact_dir: "/tmp/resnet_finetune",
    batch_size: 24,
    grads_accumulation: 8,
    smoothing: Some(
        0.1,
    ),
    num_workers: 4,
    num_epochs: 60,
    patience: 20,
    pretrained: "resnet50.tv_in1k",
    replace_activation: Some(
        Gelu,
    ),
    freeze_layers: false,
    drop_block_prob: 0.2,
    stochastic_depth_prob: 0.05,
    learning_rate: 5e-5,
    cautious_weight_decay: true,
    weight_decay: 0.02,
}
```