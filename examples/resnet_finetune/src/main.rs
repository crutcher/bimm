#![allow(dead_code)]
#![recursion_limit = "256"]

extern crate core;
mod data;
mod dataset;

use crate::data::{ClassificationBatch, ClassificationBatcher};
use crate::dataset::{CLASSES, PlanetLoader, download};
use bimm::cache::disk::DiskCacheConfig;
use bimm::models::resnet::{PREFAB_RESNET_MAP, ResNet};
use burn::backend::Autodiff;
use burn::config::Config;
use burn::data::dataloader::DataLoaderBuilder;
use burn::data::dataset::transform::ShuffledDataset;
use burn::data::dataset::vision::ImageFolderDataset;
use burn::module::Module;
use burn::nn::activation::ActivationConfig;
use burn::nn::loss::BinaryCrossEntropyLossConfig;
use burn::optim::AdamWConfig;
use burn::prelude::{Int, Tensor};
use burn::record::CompactRecorder;
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::train::metric::store::{Aggregate, Direction, Split};
use burn::train::metric::{HammingScore, LossMetric};
use burn::train::renderer::{
    EvaluationName, EvaluationProgress, MetricState, MetricsRenderer, MetricsRendererEvaluation,
    MetricsRendererTraining, TrainingProgress,
};
use burn::train::{
    LearnerBuilder, LearningStrategy, MetricEarlyStoppingStrategy, MultiLabelClassificationOutput,
    StoppingCondition, TrainOutput, TrainStep, ValidStep,
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
 */

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// Random seed for reproducibility.
    #[arg(short, long, default_value = "0")]
    seed: u64,

    /// Train percentage.
    #[arg(long, default_value = "70")]
    pub train_percentage: u8,

    /// Directory to save the artifacts.
    #[arg(long, default_value = "/tmp/resnet-finetune")]
    artifact_dir: String,

    /// Batch size for processing
    #[arg(short, long, default_value_t = 24)]
    batch_size: usize,

    /// Grads accumulation size for processing
    #[arg(short, long, default_value_t = 8)]
    grads_accumulation: usize,

    /// Category smoothing factor for training.
    #[arg(long, default_value = "0.1")]
    smoothing: Option<f32>,

    /// Number of workers for data loading.
    #[arg(long, default_value = "4")]
    num_workers: usize,

    /// Number of epochs to train the model.
    #[arg(long, default_value = "60")]
    num_epochs: usize,

    /// Pretrained Resnet Model.
    /// Use "list" to list all available pretrained models.
    #[arg(long, default_value = "resnet34.tv_in1k")]
    pretrained: String,

    /// Freeze the body layers during training.
    #[arg(long, default_value = "false")]
    freeze_layers: bool,

    /// Drop Block Prob
    #[arg(long, default_value = "0.2")]
    drop_block_prob: f64,

    /// Drop Path Prob
    #[arg(long, default_value = "0.1")]
    drop_path_prob: f64,

    /// Learning rate
    #[arg(long, default_value_t = 5e-5)]
    pub learning_rate: f64,

    /// Early stopping patience
    #[arg(long, default_value_t = 10)]
    patience: usize,

    /// Enable cautious weight decay.
    #[arg(long, default_value = "false")]
    pub cautious_weight_decay: bool,

    /// Optimizer Weight decay.
    #[arg(long, default_value_t = 5e-4)]
    pub weight_decay: f32,
}

#[allow(clippy::too_many_arguments)]
mod local {
    use bimm::models::resnet::ResNetContractConfig;
    use burn::config::Config;

    /// Log config.
    ///
    /// Only exists for logging.
    #[derive(Config, Debug)]
    pub struct LogConfig {
        pub seed: u64,
        pub train_percentage: u8,
        pub batch_size: usize,
        pub num_epochs: usize,
        pub resnet_prefab: String,
        pub resnet_pretrained: String,
        pub drop_block_prob: f64,
        pub drop_path_prob: f64,
        pub learning_rate: f64,
        pub patience: usize,
        pub weight_decay: f32,
        pub resnet: ResNetContractConfig,
    }
}
use local::*;

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let _source_tree = download();

    #[cfg(feature = "wgpu")]
    return train::<Autodiff<burn::backend::Wgpu>>(&args);

    #[cfg(feature = "cuda")]
    return train::<Autodiff<burn::backend::Cuda>>(&args);

    #[cfg(feature = "metal")]
    return train::<Autodiff<burn::backend::Metal>>(&args);
}

fn ensure_artifact_dir(artifact_dir: &str) -> anyhow::Result<()> {
    let _ignored = std::fs::remove_dir_all(artifact_dir);
    std::fs::create_dir_all(artifact_dir)?;
    Ok(())
}

#[must_use]
pub fn train<B: AutodiffBackend>(args: &Args) -> anyhow::Result<()> {
    let device: B::Device = Default::default();

    // TODO: lift to clap parser.
    if args.pretrained == "list" {
        println!("Available pretrained models:");
        for prefab in PREFAB_RESNET_MAP.items {
            if let Some(weights) = prefab.weights {
                if weights.items.is_empty() {
                    continue;
                }

                let cfg = (prefab.builder)();
                println!("* \"{}\"", prefab.name);
                println!("{cfg:?}");

                for item in weights.items {
                    println!(
                        "  - \"{}.{}\": {}",
                        prefab.name, item.name, item.description
                    );
                }
            }
        }
        return Ok(());
    }
    let [resnet_prefab, resnet_pretrained] = args
        .pretrained
        .splitn(2, ".")
        .map(|s| s.to_string())
        .collect::<Vec<String>>()
        .try_into()
        .unwrap();

    // Remove existing artifacts before to get an accurate learner summary
    let artifact_dir: &str = args.artifact_dir.as_ref();
    ensure_artifact_dir(artifact_dir)?;

    B::seed(&device, args.seed);

    let disk_cache = DiskCacheConfig::default();

    let prefab = PREFAB_RESNET_MAP.expect_lookup_prefab(&resnet_prefab);

    let weights = prefab
        .expect_lookup_pretrained_weights(&resnet_pretrained)
        .fetch_weights(&disk_cache)
        .expect("Failed to fetch pretrained weights");

    let resnet_config = prefab.to_config().with_activation(ActivationConfig::Gelu);

    let mut model: ResNet<B> = resnet_config
        .clone()
        .to_structure()
        .init(&device)
        .load_pytorch_weights(weights)
        .expect("Failed to load pretrained weights")
        .with_classes(CLASSES.len())
        .with_stochastic_drop_block(args.drop_block_prob)
        .with_stochastic_path_depth(args.drop_path_prob);

    if args.freeze_layers {
        model = model.freeze_layers();
    }

    let host: Host<B> = Host {
        smoothing: args.smoothing,
        resnet: model,
    };

    let optimizer = AdamWConfig::new()
        .with_cautious_weight_decay(args.cautious_weight_decay)
        .with_weight_decay(args.weight_decay)
        .init();

    LogConfig {
        seed: args.seed,
        train_percentage: args.train_percentage,
        batch_size: args.batch_size,
        num_epochs: args.num_epochs,
        resnet_prefab: resnet_prefab.clone(),
        resnet_pretrained: resnet_pretrained.clone(),
        drop_block_prob: args.drop_block_prob,
        drop_path_prob: args.drop_path_prob,
        learning_rate: args.learning_rate,
        patience: args.patience,
        weight_decay: args.weight_decay,
        resnet: resnet_config,
    }
    .save(format!("{artifact_dir}/config.json"))
    .expect("Config should be saved successfully");

    // Dataloaders
    let batcher_train = ClassificationBatcher::<B>::new(device.clone());
    let batcher_valid = ClassificationBatcher::<B::InnerBackend>::new(device.clone());

    let (train, valid) =
        ImageFolderDataset::planet_train_val_split(args.train_percentage, args.seed).unwrap();

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(args.batch_size)
        .shuffle(args.seed)
        .num_workers(args.num_workers)
        .build(ShuffledDataset::new(train, args.seed));

    let dataloader_test = DataLoaderBuilder::new(batcher_valid)
        .batch_size(args.batch_size)
        .num_workers(args.num_workers)
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
        .learning_strategy(LearningStrategy::SingleDevice(device.clone()))
        .grads_accumulation(args.grads_accumulation)
        .num_epochs(args.num_epochs)
        .summary()
        /*
        .renderer(CustomRenderer {})
        .with_application_logger(None)
         */
        .build(host, optimizer, args.learning_rate);

    // Training
    let now = Instant::now();
    let model_trained = learner.fit(dataloader_train, dataloader_test);
    let elapsed = now.elapsed().as_secs();
    println!("Training completed in {}m{}s", (elapsed / 60), elapsed % 60);

    model_trained
        .model
        .resnet
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .expect("Trained model should be saved successfully");

    Ok(())
}

struct CustomRenderer {}

impl MetricsRendererTraining for CustomRenderer {
    fn update_train(
        &mut self,
        _state: MetricState,
    ) {
    }

    fn update_valid(
        &mut self,
        _state: MetricState,
    ) {
    }

    fn render_train(
        &mut self,
        item: TrainingProgress,
    ) {
        dbg!(item);
    }

    fn render_valid(
        &mut self,
        item: TrainingProgress,
    ) {
        dbg!(item);
    }
}

impl MetricsRenderer for CustomRenderer {
    fn manual_close(&mut self) {
        // Nothing to do.
    }
}

impl MetricsRendererEvaluation for CustomRenderer {
    fn update_test(
        &mut self,
        _name: EvaluationName,
        _state: MetricState,
    ) {
    }

    fn render_test(
        &mut self,
        item: EvaluationProgress,
    ) {
        dbg!(item);
    }
}

#[derive(Module, Debug)]
pub struct Host<B: Backend> {
    pub smoothing: Option<f32>,

    pub resnet: ResNet<B>,
}

pub trait MultiLabelClassification<B: Backend> {
    fn forward_classification(
        &self,
        images: Tensor<B, 4>,
        targets: Tensor<B, 2, Int>,
    ) -> MultiLabelClassificationOutput<B>;
}

impl<B: Backend> MultiLabelClassification<B> for Host<B> {
    fn forward_classification(
        &self,
        images: Tensor<B, 4>,
        targets: Tensor<B, 2, Int>,
    ) -> MultiLabelClassificationOutput<B> {
        let output = self.resnet.forward(images);

        let mut loss_cfg = BinaryCrossEntropyLossConfig::new().with_logits(true);

        if B::ad_enabled() {
            loss_cfg = loss_cfg.with_smoothing(self.smoothing);
        }

        let loss = loss_cfg
            .init(&output.device())
            .forward(output.clone(), targets.clone());

        MultiLabelClassificationOutput::new(loss, output, targets)
    }
}

impl<B: AutodiffBackend> TrainStep<ClassificationBatch<B>, MultiLabelClassificationOutput<B>>
    for Host<B>
{
    fn step(
        &self,
        batch: ClassificationBatch<B>,
    ) -> TrainOutput<MultiLabelClassificationOutput<B>> {
        let item = self.forward_classification(batch.images, batch.targets);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<ClassificationBatch<B>, MultiLabelClassificationOutput<B>> for Host<B> {
    fn step(
        &self,
        batch: ClassificationBatch<B>,
    ) -> MultiLabelClassificationOutput<B> {
        self.forward_classification(batch.images, batch.targets)
    }
}
