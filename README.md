# Burn Image Models

[![Coverage Status](https://coveralls.io/repos/github/crutcher/bimm/badge.svg?branch=main)](https://coveralls.io/github/crutcher/bimm?branch=main)

## Overview

This is a repository for burn image models; inspired by the Python `timm` package.

The future feature list is:

* `timm` => `bimm`

## Contributing

See the [CONTRIBUTING](CONTRIBUTING.md) guide for build and contribution instructions.

## Crates

### [bimm](crates/bimm) - the main crate for image models.

[![Crates.io Version](https://img.shields.io/crates/v/bimm)](https://crates.io/crates/bimm)
[![docs.rs](https://img.shields.io/docsrs/bimm)](https://docs.rs/bimm/latest/bimm/)

* [bimm::cache](crates/bimm/src/cache) - weight loading cache.
* [bimm::layers] - reusable neural network modules.
    * [bimm::layers::activation](crates/bimm/src/layers/activation) - activation layers.
        * [bimm::layers::activation::Activation](crates/bimm/src/layers/activation/activation_wrapper.rs)
            - activation layer abstraction wrapper.
    * [bimm::layers::blocks](crates/bimm/src/layers/blocks) - miscellaneous blocks.
        * [bimm::layers::blocks::conv_norm](crates/bimm/src/layers/blocks/conv_norm.rs) -
          ``Conv2d + BatchNorm2d`` block.
    * [bimm::layers::drop](crates/bimm/src/layers/drop) - dropout layers.
        * [bimm::layers::drop::drop_block](crates/bimm/src/layers/drop/drop_block.rs) - 2d drop
          block / spatial dropout.
        * [bimm::layers::drop::drop_path](crates/bimm/src/layers/drop/drop_path.rs) - drop
          path /
          stochastic depth.
    * [bimm::layers::patching](crates/bimm/src/layers/patching) - patching layers.
        * [bimm::layers::patching::patch_embed](crates/bimm/src/layers/patching/patch_embed.rs) -
          2d patch embedding layer.
* [bimm::models](crates/bimm/src/models) - complete model families.
    * [bimm::models::resnet](crates/bimm/src/models/resnet/mod.rs) - `ResNet`
    * [bimm::models::swin](crates/bimm/src/models/swin/mod.rs) - The SWIN Family.
        * [bimm::models::swin::v2](crates/bimm/src/models/swin/v2/mod.rs) - The SWIN-V2 Model.

#### Example

Example of building a pretrained ResNet-18 module:

```rust,no_run
use bimm::cache::fetch_model_weights;
use bimm::models::resnet::{ResNet, ResNetAbstractConfig};
use burn::backend::NdArray;

let device = Default::default();

let source =
    "https://download.pytorch.org/models/resnet18-f37072fd.pth";
let source_classes = 1000;
let weights_path= fetch_model_weights(source).unwrap();

let my_classes = 10;

let model: ResNet<NdArray> = ResNetAbstractConfig::resnet18(source_classes)
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

### [bimm-contracts](https://github.com/crutcher/bimm-contracts) - a crate for static shape contracts for tensors.

[![Crates.io Version](https://img.shields.io/crates/v/bimm-contracts)](https://crates.io/crates/bimm-contracts)
[![docs.rs](https://img.shields.io/docsrs/bimm-contracts)](https://docs.rs/bimm-contracts/latest/bimm-contracts/)

This crate is now hosted in its own repository:
[bimm-contracts](https://github.com/crutcher/bimm-contracts)

This crate provides a stand-alone library for defining and enforcing tensor shape contracts
in-line with the Burn framework modules and methods.

```rust
use bimm_contracts::{unpack_shape_contract, shape_contract, run_periodically};

pub fn window_partition<B: Backend, K>(
    tensor: Tensor<B, 4, K>,
    window_size: usize,
) -> Tensor<B, 4, K>
where
    K: BasicOps<B>,
{
    let [b, h_wins, w_wins, c] = unpack_shape_contract!(
        [
            "batch",
            "height" = "h_wins" * "window_size",
            "width" = "w_wins" * "window_size",
            "channels"
        ],
        &tensor,
        &["batch", "h_wins", "w_wins", "channels"],
        &[("window_size", window_size)],
    );

    let tensor = tensor
        .reshape([b, h_wins, window_size, w_wins, window_size, c])
        .swap_dims(2, 3)
        .reshape([b * h_wins * w_wins, window_size, window_size, c]);

    // Run an amortized check on the output shape.
    //
    // `run_periodically!{}` runs the first 10 times,
    // then on an incrementally lengthening schedule,
    // until it reaches its default period of 1000.
    //
    // Due to amortization, in release builds, this averages ~4ns:
    assert_shape_contract_periodically!(
        [
            "batch" * "h_wins" * "w_wins",
            "window_size",
            "window_size",
            "channels"
        ],
        &tensor,
        &[
            ("batch", b),
            ("h_wins", h_wins),
            ("w_wins", w_wins),
            ("window_size", window_size),
            ("channels", c),
        ]
    );

    tensor
}
```

### [bimm-firehose](crates/bimm-firehose) - a data loading and augmentation framework.

[![Crates.io Version](https://img.shields.io/crates/v/bimm-firehose)](https://crates.io/crates/bimm-firehose)
[![docs.rs](https://img.shields.io/docsrs/bimm-firehose)](https://docs.rs/bimm/latest/bimm-firehose/)

This crate provides a SQL-inspired table + operations framework for modular data pipeline
construction.

It's still very much a work in progress, and any issues/design bugs reported
are very appreciated.

This crate provides a set of image-specific operations for `bimm-firehose`.

Add-on crates:

* [bimm-firehose-image](crates/bimm-firehose-image)

