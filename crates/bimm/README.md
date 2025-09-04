# Burn Image Models

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

#### Recent Changes

* **0.3.3**
    * Preview of ResNet-18 support.
* **0.3.2**
    * Fixed visibility for `DropBlock3d` / `drop_block_3d` support.
* **0.3.1**
    * added `DropBlock2d` / `drop_block_2d` support.
* **0.2.0**
    * bumped `burn` dependency to `0.18.0`.
