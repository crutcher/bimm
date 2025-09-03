#![allow(dead_code, unused)]
#![recursion_limit = "256"]

extern crate core;
mod data;
mod dataset;
mod training;

use crate::dataset::download;
use crate::training::train;
use burn::backend::Autodiff;
use burn::tensor::backend::Backend;
use clap::{arg, Parser};
use core::clone::Clone;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// Url of the pretrained weights.
    #[arg(
        long,
        default_value = "https://download.pytorch.org/models/resnet18-f37072fd.pth"
    )]
    pretrained_weights: String,

    #[arg(long, default_value = "1000")]
    pretrained_classes: usize,

    /// Directory to save the artifacts.
    #[arg(long, default_value = "/tmp/resnet_tiny")]
    artifact_dir: String,

    /// Batch size for processing
    #[arg(short, long, default_value_t = 24)]
    batch_size: usize,

    /// Number of workers for data loading.
    #[arg(long, default_value = "4")]
    num_workers: Option<usize>,

    /// Number of epochs to train the model.
    #[arg(long, default_value = "100")]
    num_epochs: usize,

    /// Drop Block Prob
    #[arg(long, default_value = "0.25")]
    drop_block_prob: f64,

    /// Drop Path Prob
    #[arg(long, default_value = "0.15")]
    drop_path_prob: f64,

    /// Early stopping patience
    #[arg(long, default_value = "6")]
    patience: usize,
}

#[allow(dead_code)]
const ARTIFACT_DIR: &str = "/tmp/resnet-finetune";

fn main() {
    let args = Args::parse();

    wgpu::run(&args);
}

#[allow(dead_code)]
fn run<B: Backend>(
    args: &Args,
    device: &B::Device,
) {
    let _source_tree = download();
    println!("{:?}", _source_tree);

    train::<Autodiff<B>>(&args, device);
    // infer::<B>(ARTIFACT_DIR, device, 0.5);
}

mod wgpu {
    use super::*;
    use burn::backend::wgpu::{Wgpu, WgpuDevice};

    pub fn run(args: &Args) {
        super::run::<Wgpu>(args, &WgpuDevice::default());
    }
}
