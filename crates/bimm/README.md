# bimm - Burn Image Models

[![Crates.io Version](https://img.shields.io/crates/v/bimm)](https://crates.io/crates/bimm)
[![docs.rs](https://img.shields.io/docsrs/bimm)](https://docs.rs/bimm/latest/)

This is a Rust crate for image models, inspired by the Python `timm` package.

Examples of loading pre-trained ResNet-18 model:

```rust,no_run
use bimm::cache::fetch_model_weights;
use bimm::models::resnet::{ResNet, ResNetAbstractConfig};
use burn::backend::Wgpu;

type B = Wgpu;
let device = Default::default();

let source =
    "https://download.pytorch.org/models/resnet18-f37072fd.pth";
let source_classes = 1000;
let weights_path= fetch_model_weights(source).unwrap();

let my_classes = 10;

let model: ResNet<B> = ResNetAbstractConfig::resnet18(source_classes)
    .to_structure()
    .init(&device)
    .load_pytorch_weights(weights_path)
    .expect("Model should be loaded successfully")
    .with_classes(my_classes)
    // Enable (drop_block_prob) stochastic block drops for training:
    .with_stochastic_drop_block(0.2)
    // Enable (drop_path_prob) stochastic depth for training:
    .with_stochastic_path_depth(0.1);
```

* [bimm::cache](https://docs.rs/bimm/latest/bimm/cache) - weight loading cache.
* [bimm::layers](https://docs.rs/bimm/latest/bimm/layers) - reusable neural network modules.
    * [bimm::layers::activation](https://docs.rs/bimm/latest/bimm/layers/activation) - activation
      layers.
        * [bimm::layers::activation::Activation](https://docs.rs/bimm/latest/bimm/layers/activation/enum.Activation.html)
            - activation layer abstraction wrapper.
    * [bimm::layers::blocks](https://docs.rs/bimm/latest/bimm/layers/blocks) - miscellaneous
      blocks.
        * [bimm::layers::blocks::conv_norm](https://docs.rs/bimm/latest/bimm/layers/blocks/conv_norm) -
          ``Conv2d + BatchNorm2d`` block.
    * [bimm::layers::drop](https://docs.rs/bimm/latest/bimm/layers/drop) - dropout layers.
        * [bimm::layers::drop::drop_block](https://docs.rs/bimm/latest/bimm/layers/drop/drop_block) -
          2d drop
          block / spatial dropout.
        * [bimm::layers::drop::drop_path](https://docs.rs/bimm/latest/bimm/layers/drop/drop_path) -
          drop
          path /
          stochastic depth.
    * [bimm::layers::patching](https://docs.rs/bimm/latest/bimm/layers/patching) - patching layers.
        * [bimm::layers::patching::patch_embed](https://docs.rs/bimm/latest/bimm/layers/patching/patch_embed) -
          2d patch embedding layer.
* [bimm::models](https://docs.rs/bimm/latest/bimm/models) - complete model families.
    * [bimm::models::resnet](https://docs.rs/bimm/latest/bimm/models/resnet) - `ResNet`
    * [bimm::models::swin](https://docs.rs/bimm/latest/bimm/models/swin) - The SWIN Family.
        * [bimm::models::swin::v2](https://docs.rs/bimm/latest/bimm/models/swin/v2) - The
          SWIN-V2 Model.

#### Recent Changes

* **0.3.3**
    * Preview of ResNet-18 support.
* **0.3.2**
    * Fixed visibility for `DropBlock3d` / `drop_block_3d` support.
* **0.3.1**
    * added `DropBlock2d` / `drop_block_2d` support.
* **0.2.0**
    * bumped `burn` dependency to `0.18.0`.
