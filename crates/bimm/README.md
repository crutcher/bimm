# Burn Image Models

[![Crates.io Version](https://img.shields.io/crates/v/bimm)](https://crates.io/crates/bimm)
[![docs.rs](https://img.shields.io/docsrs/bimm)](https://docs.rs/bimm/latest/)

This is a Rust crate for image models, inspired by the Python `timm` package.

Examples of loading pretrained model:

```rust,no_run
use burn::backend::Wgpu;
use bimm::cache::disk::DiskCacheConfig;
use bimm::models::resnet::{PREFAB_RESNET_MAP, ResNet};

let device = Default::default();

let prefab = PREFAB_RESNET_MAP.expect_lookup_prefab("resnet18");

let weights = prefab
    .expect_lookup_pretrained_weights("tv_in1k")
    .fetch_weights(&DiskCacheConfig::default())
    .expect("Failed to fetch weights");

let model: ResNet<Wgpu> = prefab
    .to_config()
    .to_structure()
    .init(&device)
    .load_pytorch_weights(weights)
    .expect("Failed to load weights")
    // re-head the model to 10 classes:
    .with_classes(10)
    // Enable (drop_block_prob) stochastic block drops for training:
    .with_stochastic_drop_block(0.2)
    // Enable (drop_path_prob) stochastic depth for training:
    .with_stochastic_path_depth(0.1);
```

* [bimm::cache](https://docs.rs/bimm/latest/bimm/cache) - weight loading cache.
* [bimm::compat](https://docs.rs/bimm/latest/bimm/compat) - future-porting burn mechanisms.
    * [bimm::compat::activation_wrapper::Activation](https://docs.rs/bimm/latest/bimm/compat/activation_wrapper/enum.Activation.html) -
      activation layer abstraction wrapper.
    * [bimm::compat::normalization_wrapper::Normalization](https://docs.rs/bimm/latest/bimm/compat/normalization_wrapper/enum.Normalization.html) -
      normalization layer abstraction wrapper.
* [bimm::layers](https://docs.rs/bimm/latest/bimm/layers) - reusable neural network modules.
    * [bimm::layers::blocks](https://docs.rs/bimm/latest/bimm/layers/blocks) - miscellaneous
      blocks.
        * [bimm::layers::blocks::conv_norm::ConvNorm2d](https://docs.rs/bimm/latest/bimm/layers/blocks/conv_norm) -
          ``Conv2d + BatchNorm2d`` block.
        * [bimm::layers::blocks::cna::CNA2d](https://docs.rs/bimm/latest/bimm/layers/blocks/conv_norm) -
          ``Conv2d + Normalization + Activation`` block.
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

* **0.19.0**
    * Bumped to track `burn` 0.19.0.
* **0.3.3**
    * Preview of ResNet-18 support.
* **0.3.2**
    * Fixed visibility for `DropBlock3d` / `drop_block_3d` support.
* **0.3.1**
    * added `DropBlock2d` / `drop_block_2d` support.
* **0.2.0**
    * bumped `burn` dependency to `0.18.0`.
