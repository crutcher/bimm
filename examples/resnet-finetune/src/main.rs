#![allow(dead_code, unused)]
#![recursion_limit = "256"]

extern crate core;
mod data;
mod dataset;

use crate::data::{ClassificationBatch, ClassificationBatcher};
use crate::dataset::{CLASSES, PlanetLoader, download};
use bimm::cache::disk::DiskCacheConfig;
use bimm::models::resnet::ResNet;
use bimm::models::resnet::pretrained::PRETRAINED_RESNETS;
use burn::backend::{Autodiff, Cuda};
use burn::config::Config;
use burn::data::dataloader::DataLoaderBuilder;
use burn::data::dataset::transform::ShuffledDataset;
use burn::data::dataset::vision::ImageFolderDataset;
use burn::module::Module;
use burn::nn::loss::BinaryCrossEntropyLossConfig;
use burn::optim::AdamConfig;
use burn::optim::decay::WeightDecayConfig;
use burn::prelude::{Int, Tensor};
use burn::record::CompactRecorder;
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::train::metric::{HammingScore, LossMetric};
use burn::train::{
    LearnerBuilder, MultiLabelClassificationOutput, TrainOutput, TrainStep, ValidStep,
};
use clap::{Parser, arg};
use core::clone::Clone;
use std::time::Instant;
/*
tracel-ai/models reference:
| Split | Metric                         | Min.     | Epoch    | Max.     | Epoch    |
|-------|--------------------------------|----------|----------|----------|----------|
| Train | Hamming Score @ Threshold(0.5) | 91.311   | 1        | 95.277   | 5        |
| Train | Loss                           | 0.122    | 5        | 0.250    | 1        |
| Valid | Hamming Score @ Threshold(0.5) | 88.490   | 1        | 93.843   | 3        |
| Valid | Loss                           | 0.168    | 3        | 0.512    | 1        |


$ --drop-path-prob=0.1 --drop-block-prob=0.2 --num-epochs=30 --batch-size=32 --learning-rate=1e-4
| Split | Metric                         | Min.     | Epoch    | Max.     | Epoch    |
|-------|--------------------------------|----------|----------|----------|----------|
| Train | Hamming Score @ Threshold(0.5) | 82.958   | 1        | 97.916   | 28       |
| Train | Loss                           | 0.072    | 28       | 0.515    | 1        |
| Valid | Hamming Score @ Threshold(0.5) | 91.765   | 1        | 95.706   | 12       |
| Valid | Loss                           | 0.123    | 17       | 0.411    | 1        |

$ --drop-path-prob=0.1 --drop-block-prob=0.2 --num-epochs=20 --batch-size=32
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
    // /// Number of classes in the pretrained weights.
    // #[arg(long, default_value = "1000")]
    // pretrained_classes: usize,
    /// Directory to save the artifacts.
    #[arg(long, default_value = "/tmp/resnet-finetune")]
    artifact_dir: String,

    /// Batch size for processing
    #[arg(short, long, default_value_t = 32)]
    batch_size: usize,

    /// Number of workers for data loading.
    #[arg(long, default_value = "0")]
    num_workers: Option<usize>,

    /// Number of epochs to train the model.
    #[arg(long, default_value = "20")]
    num_epochs: usize,

    /// Resnet Model Config
    #[arg(long, default_value = "resnet-18")]
    resnet_prefab: String,

    /// Resnet Pretrained
    #[arg(long, default_value = "tv_in1k")]
    resnet_pretrained: String,

    /// Drop Block Prob
    #[arg(long, default_value = "0.2")]
    drop_block_prob: f64,

    /// Drop Path Prob
    #[arg(long, default_value = "0.0")]
    drop_path_prob: f64,

    /// Learning rate
    #[arg(long, default_value = "1e-4")]
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

pub trait MultiLabelClassification<B: Backend> {
    fn forward_classification(
        &self,
        images: Tensor<B, 4>,
        targets: Tensor<B, 2, Int>,
    ) -> MultiLabelClassificationOutput<B>;
}

impl<B: Backend> MultiLabelClassification<B> for ResNet<B> {
    fn forward_classification(
        &self,
        images: Tensor<B, 4>,
        targets: Tensor<B, 2, Int>,
    ) -> MultiLabelClassificationOutput<B> {
        let output = self.forward(images);
        let loss = BinaryCrossEntropyLossConfig::new()
            .with_logits(true)
            .init(&output.device())
            .forward(output.clone(), targets.clone());

        MultiLabelClassificationOutput::new(loss, output, targets)
    }
}

impl<B: AutodiffBackend> TrainStep<ClassificationBatch<B>, MultiLabelClassificationOutput<B>>
    for ResNet<B>
{
    fn step(
        &self,
        batch: ClassificationBatch<B>,
    ) -> TrainOutput<MultiLabelClassificationOutput<B>> {
        let item = self.forward_classification(batch.images, batch.targets);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<ClassificationBatch<B>, MultiLabelClassificationOutput<B>>
    for ResNet<B>
{
    fn step(
        &self,
        batch: ClassificationBatch<B>,
    ) -> MultiLabelClassificationOutput<B> {
        self.forward_classification(batch.images, batch.targets)
    }
}

#[derive(Config)]
pub struct TrainingConfig {
    #[config(default = 5)]
    pub num_epochs: usize,

    #[config(default = 24)]
    pub batch_size: usize,

    #[config(default = 4)]
    pub num_workers: usize,

    #[config(default = 42)]
    pub seed: u64,

    #[config(default = 1e-3)]
    pub learning_rate: f64,

    #[config(default = 5e-5)]
    pub weight_decay: f32,

    #[config(default = 70)]
    pub train_percentage: u8,

    pub num_classes: usize,
}

fn create_artifact_dir(artifact_dir: &str) {
    // Remove existing artifacts before to get an accurate learner summary
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

pub fn train<B: AutodiffBackend>(
    args: &Args,
    device: &B::Device,
) -> anyhow::Result<()> {
    let artifact_dir = args.artifact_dir.as_ref();
    create_artifact_dir(artifact_dir);

    let disk_cache = DiskCacheConfig::default();

    let prefab = PRETRAINED_RESNETS.expect_lookup_prefab(&args.resnet_prefab);

    let weights = prefab
        .expect_lookup_weights(&args.resnet_pretrained)
        .fetch_weights_to_disk_cache(&disk_cache)?;

    let resnet_config = prefab
        .new_config()
        // .with_activation(PReluConfig::new().into())
        .to_structure();

    let model: ResNet<B> = resnet_config
        .init(device)
        .load_pytorch_weights(weights)?
        .with_classes(CLASSES.len())
        .with_stochastic_drop_block(args.drop_block_prob)
        .with_stochastic_path_depth(args.drop_path_prob);

    // Config
    let training_config = TrainingConfig::new(CLASSES.len())
        .with_learning_rate(args.learning_rate)
        .with_num_epochs(args.num_epochs)
        .with_batch_size(args.batch_size);

    let optimizer = AdamConfig::new()
        .with_weight_decay(Some(WeightDecayConfig::new(training_config.weight_decay)))
        .init();

    training_config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Config should be saved successfully");

    B::seed(training_config.seed);

    // Dataloaders
    let batcher_train = ClassificationBatcher::<B>::new(device.clone());
    let batcher_valid = ClassificationBatcher::<B::InnerBackend>::new(device.clone());

    let (train, valid) = ImageFolderDataset::planet_train_val_split(
        training_config.train_percentage,
        training_config.seed,
    )
    .unwrap();

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(training_config.batch_size)
        .shuffle(training_config.seed)
        .num_workers(training_config.num_workers)
        .build(ShuffledDataset::with_seed(train, training_config.seed));

    let dataloader_test = DataLoaderBuilder::new(batcher_valid)
        .batch_size(training_config.batch_size)
        .num_workers(training_config.num_workers)
        .build(valid);

    // Learner config
    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(HammingScore::new())
        .metric_valid_numeric(HammingScore::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device.clone()])
        .num_epochs(training_config.num_epochs)
        .summary()
        .build(model, optimizer, training_config.learning_rate);

    // Training
    let now = Instant::now();
    let model_trained = learner.fit(dataloader_train, dataloader_test);
    let elapsed = now.elapsed().as_secs();
    println!("Training completed in {}m{}s", (elapsed / 60), elapsed % 60);

    model_trained
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .expect("Trained model should be saved successfully");

    Ok(())
}
