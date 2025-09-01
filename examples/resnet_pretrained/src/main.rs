#![recursion_limit = "256"]

extern crate core;

use bimm::cache;
use bimm::models::resnet::resnet_model::{ResNet, ResNetAbstractConfig};
use burn::backend::Cuda;
use clap::{Parser, arg};
use core::clone::Clone;
use core::option::Option;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// Url of the pretrained weights.
    #[arg(
        long,
        default_value = "https://download.pytorch.org/models/resnet18-f37072fd.pth"
    )]
    weights: String,

    #[arg(long, default_value = "1000")]
    num_classes: usize,
}

#[allow(dead_code, unused)]
fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    type B = Cuda;
    let device = Default::default();

    let weights = cache::fetch_model_weights(&args.weights)?;

    let model: ResNet<B> = ResNetAbstractConfig::resnet18(args.num_classes)
        .to_structure()
        .init(&device)
        .load_pytorch_weights(weights)?;

    Ok(())
}
