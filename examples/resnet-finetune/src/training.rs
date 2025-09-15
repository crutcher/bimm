use crate::Args;
use crate::data::{ClassificationBatch, ClassificationBatcher};
use crate::dataset::{CLASSES, PlanetLoader};
use bimm::cache::weights;
use bimm::models::resnet::{RESNET34_BLOCKS, ResNet, ResNetContractConfig};
use burn::data::dataloader::DataLoaderBuilder;
use burn::data::dataset::transform::ShuffledDataset;
use burn::data::dataset::vision::ImageFolderDataset;
use burn::nn::loss::BinaryCrossEntropyLossConfig;
use burn::optim::AdamConfig;
use burn::optim::decay::WeightDecayConfig;
use burn::prelude::{Backend, Config, Int, Module, Tensor};
use burn::record::CompactRecorder;
use burn::tensor::backend::AutodiffBackend;
use burn::train::metric::{HammingScore, LossMetric};
use burn::train::{
    LearnerBuilder, MultiLabelClassificationOutput, TrainOutput, TrainStep, ValidStep,
};
use std::time::Instant;

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

    // Config
    let config = TrainingConfig::new(CLASSES.len())
        .with_learning_rate(args.learning_rate)
        .with_num_epochs(args.num_epochs)
        .with_batch_size(args.batch_size);

    let optimizer = AdamConfig::new()
        .with_weight_decay(Some(WeightDecayConfig::new(config.weight_decay)))
        .init();

    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Config should be saved successfully");

    B::seed(config.seed);

    // Dataloaders
    let batcher_train = ClassificationBatcher::<B>::new(device.clone());
    let batcher_valid = ClassificationBatcher::<B::InnerBackend>::new(device.clone());

    let (train, valid) =
        ImageFolderDataset::planet_train_val_split(config.train_percentage, config.seed).unwrap();

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(ShuffledDataset::with_seed(train, config.seed));

    let dataloader_test = DataLoaderBuilder::new(batcher_valid)
        .batch_size(config.batch_size)
        .num_workers(config.num_workers)
        .build(valid);

    // let pretrained_weights = &args.pretrained_weights;
    //let pretrained_weights = "https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet34_a1_0-46f8f793.pth";
    let pretrained_weights = "https://download.pytorch.org/models/resnet34-b627a593.pth";

    let weights_path = weights::fetch_model_weights(pretrained_weights)?;

    let model: ResNet<B> = ResNetContractConfig::new(RESNET34_BLOCKS, 1000)
        // .with_activation(PReluConfig::new().into())
        .to_structure()
        .init(device)
        .load_pytorch_weights(weights_path)
        .expect("Model should be loaded successfully")
        .with_classes(CLASSES.len())
        .with_stochastic_drop_block(args.drop_block_prob)
        .with_stochastic_path_depth(args.drop_path_prob);

    // Learner config
    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(HammingScore::new())
        .metric_valid_numeric(HammingScore::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device.clone()])
        .num_epochs(config.num_epochs)
        .summary()
        .build(model, optimizer, config.learning_rate);

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
