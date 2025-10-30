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
- "resnet18.tv_in1k": ResNet18 pretrained on ImageNet
- "resnet18.a1_in1k": ResNet18 pretrained on ImageNet
- "resnet34.tv_in1k": ResNet-34 pretrained on ImageNet
```
