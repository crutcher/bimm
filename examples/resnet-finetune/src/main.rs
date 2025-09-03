#![allow(dead_code, unused)]
#![recursion_limit = "256"]

extern crate core;
mod data;
mod dataset;
mod training;

use crate::dataset::download;
use crate::training::train;
use burn::backend::{Autodiff, Cuda};
use burn::tensor::backend::Backend;
use clap::{arg, Parser};
use core::clone::Clone;

/*
tracel-ai/models reference:
| Split | Metric                         | Min.     | Epoch    | Max.     | Epoch    |
|-------|--------------------------------|----------|----------|----------|----------|
| Train | Hamming Score @ Threshold(0.5) | 91.311   | 1        | 95.277   | 5        |
| Train | Loss                           | 0.122    | 5        | 0.250    | 1        |
| Valid | Hamming Score @ Threshold(0.5) | 88.490   | 1        | 93.843   | 3        |
| Valid | Loss                           | 0.168    | 3        | 0.512    | 1        |

$ --drop-path-prob 0.1 --drop-block-prob 0.2 --num-epochs 20 --batch-size=32
| Split | Metric                         | Min.     | Epoch    | Max.     | Epoch    |
|-------|--------------------------------|----------|----------|----------|----------|
| Train | Hamming Score @ Threshold(0.5) | 88.437   | 1        | 96.176   | 20       |
| Train | Loss                           | 0.099    | 20       | 0.312    | 1        |
| Valid | Hamming Score @ Threshold(0.5) | 84.725   | 1        | 94.824   | 15       |
| Valid | Loss                           | 0.146    | 7        | 0.432    | 1        |

$ --drop-path-prob=0.1 --drop-block-prob=0.1 --batch-size=24
| Split | Metric                         | Min.     | Epoch    | Max.     | Epoch    |
|-------|--------------------------------|----------|----------|----------|----------|
| Train | Hamming Score @ Threshold(0.5) | 89.345   | 1        | 92.513   | 4        |
| Train | Loss                           | 0.207    | 4        | 0.304    | 1        |
| Valid | Hamming Score @ Threshold(0.5) | 88.902   | 3        | 93.784   | 5        |
| Valid | Loss                           | 0.180    | 5        | 0.486    | 1        |

$ --drop-path-prob=0.0 --drop-block-prob=0.0 --batch-size=24
| Split | Metric                         | Min.     | Epoch    | Max.     | Epoch    |
|-------|--------------------------------|----------|----------|----------|----------|
| Train | Hamming Score @ Threshold(0.5) | 91.437   | 1        | 94.454   | 4        |
| Train | Loss                           | 0.144    | 5        | 0.238    | 1        |
| Valid | Hamming Score @ Threshold(0.5) | 79.569   | 1        | 93.353   | 5        |
| Valid | Loss                           | 0.181    | 3        | 1.060    | 1        |

$ --drop-block-prob=0.25 --drop-path-prob=0.15 --batch-size=24
| Split | Metric                         | Min.     | Epoch    | Max.     | Epoch    |
|-------|--------------------------------|----------|----------|----------|----------|
| Train | Hamming Score @ Threshold(0.5) | 89.941   | 1        | 92.597   | 5        |
| Train | Loss                           | 0.205    | 5        | 0.307    | 1        |
| Valid | Hamming Score @ Threshold(0.5) | 88.843   | 2        | 93.314   | 5        |
| Valid | Loss                           | 0.176    | 5        | 0.482    | 2        |

$ --drop-path-prob=0.1 --drop-block-prob=0.1 --num-epochs=10 --batch-size=24
| Split | Metric                         | Min.     | Epoch    | Max.     | Epoch    |
|-------|--------------------------------|----------|----------|----------|----------|
| Train | Hamming Score @ Threshold(0.5) | 88.613   | 1        | 93.790   | 10       |
| Train | Loss                           | 0.161    | 10       | 0.325    | 1        |
| Valid | Hamming Score @ Threshold(0.5) | 87.098   | 1        | 93.667   | 10       |
| Valid | Loss                           | 0.174    | 8        | 0.783    | 1        |

$ --drop-path-prob=0.1 --drop-block-prob=0.1 --num-epochs=15 --batch-size=32
| Split | Metric                         | Min.     | Epoch    | Max.     | Epoch    |
|-------|--------------------------------|----------|----------|----------|----------|
| Train | Hamming Score @ Threshold(0.5) | 89.118   | 1        | 95.261   | 15       |
| Train | Loss                           | 0.122    | 15       | 0.307    | 1        |
| Valid | Hamming Score @ Threshold(0.5) | 83.902   | 1        | 94.471   | 7        |
| Valid | Loss                           | 0.156    | 11       | 0.859    | 1        |
 */

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
    #[arg(long, default_value = "5")]
    num_epochs: usize,

    /// Drop Block Prob
    #[arg(long, default_value = "0.25")]
    drop_block_prob: f64,

    /// Drop Path Prob
    #[arg(long, default_value = "0.15")]
    drop_path_prob: f64,

    /// Learning rate
    #[arg(long, default_value = "1e-3")]
    pub learning_rate: f64,
}

#[allow(dead_code)]
const ARTIFACT_DIR: &str = "/tmp/resnet-finetune";

fn main() {
    let args = Args::parse();

    let _source_tree = download();

    let device = Default::default();
    train::<Autodiff<Cuda>>(&args, &device);
}

#[allow(dead_code)]
fn run<B: Backend>(
    args: &Args,
    device: &B::Device,
) {
    train::<Autodiff<B>>(args, device);
    // infer::<B>(ARTIFACT_DIR, device, 0.5);
}
