#![allow(dead_code, unused)]
#![recursion_limit = "256"]

extern crate core;
mod data;
mod dataset;

use crate::data::{ClassificationBatch, ClassificationBatcher};
use crate::dataset::{CLASSES, PlanetLoader, download};
use bimm::cache::disk::DiskCacheConfig;
use bimm::compat::activation_wrapper::ActivationConfig;
use bimm::models::resnet::{PREFAB_RESNET_MAP, ResNet};
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
use burn::train::metric::store::{Aggregate, Direction, Split};
use burn::train::metric::{HammingScore, LossMetric};
use burn::train::{
    LearnerBuilder, MetricEarlyStoppingStrategy, MultiLabelClassificationOutput, StoppingCondition,
    TrainOutput, TrainStep, ValidStep,
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

resnet18
$ --drop-path-prob=0.1 --drop-block-prob=0.2 --learning-rate=1e-4
| Split | Metric                         | Min.     | Epoch    | Max.     | Epoch    |
|-------|--------------------------------|----------|----------|----------|----------|
| Train | Hamming Score @ Threshold(0.5) | 82.958   | 1        | 97.916   | 28       |
| Train | Loss                           | 0.072    | 28       | 0.515    | 1        |
| Valid | Hamming Score @ Threshold(0.5) | 91.765   | 1        | 95.706   | 12       |
| Valid | Loss                           | 0.123    | 17       | 0.411    | 1        |

resnet34
$ --drop-path-prob=0.15 --drop-block-prob=0.25 --learning-rate=1e-5
| Split | Metric                         | Min.     | Epoch    | Max.     | Epoch    |
|-------|--------------------------------|----------|----------|----------|----------|
| Train | Hamming Score @ Threshold(0.5) | 72.588   | 1        | 96.050   | 58       |
| Train | Loss                           | 0.132    | 58       | 0.737    | 1        |
| Valid | Hamming Score @ Threshold(0.5) | 78.137   | 1        | 95.255   | 43       |
| Valid | Loss                           | 0.130    | 60       | 0.691    | 1        |
 */

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// Directory to save the artifacts.
    #[arg(long, default_value = "/tmp/resnet-finetune")]
    artifact_dir: String,

    /// Batch size for processing
    #[arg(short, long, default_value_t = 32)]
    batch_size: usize,

    /// Number of workers for data loading.
    #[arg(long, default_value = "2")]
    num_workers: Option<usize>,

    /// Number of epochs to train the model.
    #[arg(long, default_value = "60")]
    num_epochs: usize,

    /// Resnet Model Config
    #[arg(long, default_value = "resnet34")]
    resnet_prefab: String,

    /// Resnet Pretrained
    #[arg(long, default_value = "tv_in1k")]
    resnet_pretrained: String,

    /// Drop Block Prob
    #[arg(long, default_value = "0.2")]
    drop_block_prob: f64,

    /// Drop Path Prob
    #[arg(long, default_value = "0.1")]
    drop_path_prob: f64,

    /// Learning rate
    #[arg(long, default_value = "1e-5")]
    pub learning_rate: f64,

    /// Early stopping patience
    #[arg(long, default_value = "10")]
    patience: usize,
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

    let prefab = PREFAB_RESNET_MAP.expect_lookup_prefab(&args.resnet_prefab);

    let weights = prefab
        .expect_lookup_pretrained_weights(&args.resnet_pretrained)
        .fetch_weights(&disk_cache)
        .expect("Failed to fetch pretrained weights");

    let resnet_config = prefab
        .to_config()
        .with_activation(ActivationConfig::Gelu)
        .to_structure();

    let model: ResNet<B> = resnet_config
        .init(device)
        .load_pytorch_weights(weights)
        .expect("Failed to load pretrained weights")
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
        .early_stopping(MetricEarlyStoppingStrategy::new(
            &LossMetric::<B>::new(),
            Aggregate::Mean,
            Direction::Lowest,
            Split::Valid,
            StoppingCondition::NoImprovementSince {
                n_epochs: args.patience,
            },
        ))
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
