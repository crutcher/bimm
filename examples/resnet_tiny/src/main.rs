#![recursion_limit = "256"]
extern crate core;

use bimm::cache::disk::DiskCacheConfig;
use bimm::models::resnet::{PREFAB_RESNET_MAP, ResNet};
use bimm_firehose::burn::batcher::{
    BatcherInputAdapter, BatcherOutputAdapter, FirehoseExecutorBatcher,
};
use bimm_firehose::burn::path_scanning;
use bimm_firehose::core::operations::executor::SequentialBatchExecutor;
use bimm_firehose::core::schema::ColumnSchema;
use bimm_firehose::core::{
    FirehoseRowBatch, FirehoseRowReader, FirehoseRowWriter, FirehoseTableSchema,
};
use bimm_firehose::ops::init_default_operator_environment;
use bimm_firehose_image::augmentation::AugmentImageOperation;
use bimm_firehose_image::augmentation::control::with_prob::WithProbStage;
use bimm_firehose_image::augmentation::orientation::flip::HorizontalFlipStage;
use bimm_firehose_image::burn_support::{ImageToTensorData, stack_tensor_data_column};
use bimm_firehose_image::loader::{ImageLoader, ResizeSpec};
use bimm_firehose_image::{ColorType, ImageShape};
use burn::backend::{Autodiff, Cuda};
use burn::data::dataloader::{DataLoaderBuilder, Dataset};
use burn::data::dataset::transform::ShuffledDataset;
use burn::lr_scheduler::cosine::CosineAnnealingLrSchedulerConfig;
use burn::nn::PReluConfig;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::optim::AdamConfig;
use burn::optim::decay::WeightDecayConfig;
use burn::prelude::{Backend, Int, Module, Tensor};
use burn::record::CompactRecorder;
use burn::tensor::backend::AutodiffBackend;
use burn::train::metric::store::{Aggregate, Direction, Split};
use burn::train::metric::{
    AccuracyMetric, CpuMemory, CpuUse, CudaMetric, LearningRateMetric, LossMetric,
    TopKAccuracyMetric,
};
use burn::train::{
    ClassificationOutput, LearnerBuilder, MetricEarlyStoppingStrategy, StoppingCondition,
};
use burn::train::{TrainOutput, TrainStep, ValidStep};
use clap::{Parser, arg};
use core::clone::Clone;
use core::default::Default;
use core::iter::Iterator;
use core::option::Option;
use rand::{Rng, rng};
use std::sync::Arc;

const PATH_COLUMN: &str = "path";
const SEED_COLUMN: &str = "seed";
const CLASS_COLUMN: &str = "class";
const IMAGE_COLUMN: &str = "image";
const AUG_COLUMN: &str = "aug";
const DATA_COLUMN: &str = "data";

// $ --drop-path-prob=0.1 --drop-block-prob=0.2 --num-epochs=30 --batch-size=32 --learning-rate=1e-4
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// Random seed for reproducibility.
    #[arg(short, long, default_value = "0")]
    seed: u64,

    /// Batch size for processing
    #[arg(short, long, default_value_t = 128)]
    batch_size: usize,

    /// Number of workers for data loading.
    #[arg(long, default_value = "4")]
    num_workers: Option<usize>,

    /// Number of epochs to train the model.
    #[arg(long, default_value = "100")]
    num_epochs: usize,

    /// Embedding ratio: ``ratio * channels * patch_size * patch_size``
    #[arg(long, default_value = "1.25")]
    embed_ratio: f64,

    /// Ratio of oversampling the training dataset.
    #[arg(long, default_value = "2.5")]
    oversample_ratio: f64,

    /// Drop Block Rate
    #[arg(long, default_value = "0.15")]
    drop_block_rate: f64,

    /// Learning rate for the optimizer.
    #[arg(long, default_value = "1.0e-4")]
    learning_rate: f64,

    /// Learning rate decay gamma.
    #[arg(long, default_value = "0.9997")]
    lr_gamma: f64,

    /// Directory to save the artifacts.
    #[arg(long, default_value = "/tmp/resnet_tiny")]
    artifact_dir: Option<String>,

    /// Root directory of the training dataset.
    #[arg(long)]
    training_root: String,

    /// Root directory of the validation dataset.
    #[arg(long)]
    validation_root: String,

    /// Resnet Model Config
    #[arg(long, default_value = "resnet101")]
    resnet_prefab: String,

    /// Resnet Pretrained
    #[arg(long, default_value = None)]
    resnet_pretrained: Option<String>,

    /// Drop Block Prob
    #[arg(long, default_value = "0.25")]
    drop_block_prob: f64,

    /// Drop Path Prob
    #[arg(long, default_value = "0.2")]
    drop_path_prob: f64,

    /// Early stopping patience
    #[arg(long, default_value = "6")]
    patience: usize,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    type B = Autodiff<Cuda>;

    let devices = vec![Default::default()];
    backend_main::<B>(&args, devices)
}

/// Create the artifact directory for saving training artifacts.
fn create_artifact_dir(artifact_dir: &str) {
    // Remove existing artifacts before to get an accurate learner summary
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

/// Train the model with the given configuration and devices.
pub fn backend_main<B: AutodiffBackend>(
    args: &Args,
    devices: Vec<B::Device>,
) -> anyhow::Result<()> {
    let image_shape = ImageShape {
        height: 32,
        width: 32,
    };
    let num_classes = 10;

    B::seed(args.seed);

    let device = &devices[0];

    let prefab = PREFAB_RESNET_MAP.expect_lookup_prefab(&args.resnet_prefab);

    let resnet_config = prefab
        .to_config()
        .with_activation(PReluConfig::new().into())
        .to_structure();

    let resnet: ResNet<B> = resnet_config.init(device);

    let resnet: ResNet<B> = match &args.resnet_pretrained {
        None => resnet,
        Some(pretrained) => {
            let weights = prefab
                .expect_lookup_pretrained_weights(pretrained)
                .fetch_weights(&DiskCacheConfig::default())?;
            resnet.load_pytorch_weights(weights)?
        }
    }
    .with_classes(num_classes)
    .with_stochastic_drop_block(args.drop_block_prob)
    .with_stochastic_path_depth(args.drop_path_prob);

    let model: Model<B> = Model { resnet };

    let optim_config = AdamConfig::new().with_weight_decay(WeightDecayConfig::new(5e-4).into());
    // .with_grad_clipping(Some(GradientClippingConfig::Norm(3.0)));

    let artifact_dir = args.artifact_dir.as_ref().unwrap().as_ref();
    create_artifact_dir(artifact_dir);

    // training_config
    //     .save(format!("{artifact_dir}/config.json"))
    //     .expect("Config should be saved successfully");

    let firehose_env = Arc::new(init_default_operator_environment());

    let common_schema = {
        let mut schema = FirehoseTableSchema::from_columns(&[
            ColumnSchema::new::<String>(PATH_COLUMN).with_description("path to the image"),
            ColumnSchema::new::<i32>(CLASS_COLUMN).with_description("image class"),
            ColumnSchema::new::<u64>(SEED_COLUMN).with_description("instance rng seed"),
        ]);

        // Load the image from the path, resize it to 32x32 pixels, and convert it to RGB8.
        ImageLoader::default()
            .with_resize(ResizeSpec::new(image_shape))
            .with_recolor(ColorType::Rgb8)
            .to_plan(PATH_COLUMN, IMAGE_COLUMN)
            .apply_to_schema(&mut schema, firehose_env.as_ref())?;

        schema
    };

    let train_size: usize;
    let train_dataloader = {
        let ds = path_scanning::image_dataset_for_folder(args.training_root.clone())?;

        let ds = ShuffledDataset::with_seed(ds, args.seed);
        // let num_samples = (args.oversample_ratio * (ds.len() as f64)).ceil() as usize;
        // let ds = SamplerDataset::with_replacement(ds, num_samples);
        train_size = ds.len();

        let schema = Arc::new({
            let mut schema = common_schema.clone();

            AugmentImageOperation::new(vec![Arc::new(WithProbStage::new(
                0.5,
                Arc::new(HorizontalFlipStage::new()),
            ))])
            .to_plan(SEED_COLUMN, IMAGE_COLUMN, AUG_COLUMN)
            .apply_to_schema(&mut schema, firehose_env.as_ref())?;

            // Convert the image to a tensor of shape (3, 32, 32) with float32 dtype.
            ImageToTensorData::new()
                .to_plan(AUG_COLUMN, DATA_COLUMN)
                .apply_to_schema(&mut schema, firehose_env.as_ref())?;

            schema
        });

        let batcher = FirehoseExecutorBatcher::new(
            Arc::new(SequentialBatchExecutor::new(
                schema.clone(),
                firehose_env.clone(),
            )?),
            Arc::new(InputAdapter::new(schema.clone())),
            Arc::new(OutputAdapter::<B>::default()),
        );

        let mut builder = DataLoaderBuilder::new(batcher)
            .shuffle(args.seed)
            .batch_size(args.batch_size);
        if let Some(num_workers) = args.num_workers {
            builder = builder.num_workers(num_workers);
        }
        builder.build(ds)
    };

    let validation_dataloader = {
        let ds = path_scanning::image_dataset_for_folder(args.validation_root.clone())?;
        let schema = Arc::new({
            let mut schema = common_schema.clone();

            // Convert the image to a tensor of shape (3, 32, 32) with float32 dtype.
            ImageToTensorData::new()
                .to_plan(IMAGE_COLUMN, DATA_COLUMN)
                .apply_to_schema(&mut schema, firehose_env.as_ref())?;

            schema
        });

        let batcher = FirehoseExecutorBatcher::new(
            Arc::new(SequentialBatchExecutor::new(
                schema.clone(),
                firehose_env.clone(),
            )?),
            Arc::new(InputAdapter::new(schema.clone())),
            // Use the InnerBackend for validation.
            Arc::new(OutputAdapter::<B::InnerBackend>::default()),
        );

        let mut builder = DataLoaderBuilder::new(batcher).batch_size(args.batch_size);
        if let Some(num_workers) = args.num_workers {
            builder = builder.num_workers(num_workers);
        }
        builder.build(ds)
    };

    /*
    let lr_scheduler = ExponentialLrSchedulerConfig::new(args.learning_rate, args.lr_gamma)
        .init()
        .map_err(|e| anyhow::anyhow!("Failed to initialize learning rate scheduler: {}", e))?;
     */

    let batches_per_epoch = train_size / args.batch_size;
    let epochs_per_restart = 10;
    let iters_per_restart = batches_per_epoch * epochs_per_restart;
    let lr_scheduler = CosineAnnealingLrSchedulerConfig::new(args.learning_rate, iters_per_restart)
        .init()
        .map_err(|e| anyhow::anyhow!("Failed to initialize learning rate scheduler: {}", e))?;

    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .metric_train_numeric(TopKAccuracyMetric::new(2))
        .metric_valid_numeric(TopKAccuracyMetric::new(2))
        .metric_train(CudaMetric::new())
        .metric_valid(CudaMetric::new())
        .metric_train_numeric(CpuUse::new())
        .metric_valid_numeric(CpuUse::new())
        .metric_train_numeric(CpuMemory::new())
        .metric_valid_numeric(CpuMemory::new())
        .metric_train_numeric(LearningRateMetric::new())
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
        .devices(devices.clone())
        .num_epochs(args.num_epochs)
        .summary()
        .build(model, optim_config.init(), lr_scheduler);

    let model_trained = learner.fit(train_dataloader, validation_dataloader);

    model_trained
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .expect("Trained model should be saved successfully");

    Ok(())
}

/*
#[derive(Config, Debug)]
pub struct ModelConfig {
    pub drop_block: DropBlock2dConfig,
    pub swin: SwinTransformerV2Config,
}

impl ModelConfig {
    pub fn init<B: Backend>(
        self,
        device: &B::Device,
    ) -> Model<B> {
        Model {
            drop_block: self.drop_block.init(),
            swin: self.swin.init::<B>(device),
        }
    }
}
 */

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    pub resnet: ResNet<B>,
}

impl<B: Backend> Model<B> {
    pub fn forward_classification(
        &self,
        images: Tensor<B, 4>,
        targets: Tensor<B, 1, Int>,
    ) -> ClassificationOutput<B> {
        let output = self.resnet.forward(images);

        let loss = CrossEntropyLossConfig::new()
            // .with_logits(true)
            // .with_smoothing(Some(0.1))
            .init(&output.device())
            .forward(output.clone(), targets.clone());

        ClassificationOutput::new(loss, output, targets)
    }
}

impl<B: AutodiffBackend> TrainStep<(Tensor<B, 4>, Tensor<B, 1, Int>), ClassificationOutput<B>>
    for Model<B>
{
    fn step(
        &self,
        batch: (Tensor<B, 4>, Tensor<B, 1, Int>),
    ) -> TrainOutput<ClassificationOutput<B>> {
        let (images, targets) = batch;
        let item = self.forward_classification(images, targets);
        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<(Tensor<B, 4>, Tensor<B, 1, Int>), ClassificationOutput<B>>
    for Model<B>
{
    fn step(
        &self,
        batch: (Tensor<B, 4>, Tensor<B, 1, Int>),
    ) -> ClassificationOutput<B> {
        let (images, targets) = batch;
        self.forward_classification(images, targets)
    }
}

fn init_batch_from_dataset_items(
    inputs: &Vec<(String, usize)>,
    batch: &mut FirehoseRowBatch,
) -> anyhow::Result<()> {
    let mut local_rng = rng();
    for item in inputs {
        let (path, class) = item;
        let row = batch.new_row();
        row.expect_set_serialized(PATH_COLUMN, path.clone());
        row.expect_set_serialized(CLASS_COLUMN, *class as i32);
        row.expect_set_serialized(SEED_COLUMN, local_rng.random::<u64>());
    }

    Ok(())
}

struct InputAdapter {
    schema: Arc<FirehoseTableSchema>,
}
impl InputAdapter {
    pub fn new(schema: Arc<FirehoseTableSchema>) -> Self {
        Self { schema }
    }
}
impl BatcherInputAdapter<(String, usize)> for InputAdapter {
    fn apply(
        &self,
        inputs: Vec<(String, usize)>,
    ) -> anyhow::Result<FirehoseRowBatch> {
        let mut batch = FirehoseRowBatch::new(self.schema.clone());
        init_batch_from_dataset_items(&inputs, &mut batch)?;
        Ok(batch)
    }
}

struct OutputAdapter<B: Backend> {
    phantom: std::marker::PhantomData<B>,
}
impl<B> Default for OutputAdapter<B>
where
    B: Backend,
{
    fn default() -> Self {
        Self {
            phantom: std::marker::PhantomData,
        }
    }
}
impl<B: Backend> BatcherOutputAdapter<B, (Tensor<B, 4>, Tensor<B, 1, Int>)> for OutputAdapter<B> {
    fn apply(
        &self,
        batch: &FirehoseRowBatch,
        device: &B::Device,
    ) -> anyhow::Result<(Tensor<B, 4>, Tensor<B, 1, Int>)> {
        let image_batch = Tensor::<B, 4>::from_data(
            stack_tensor_data_column(batch, DATA_COLUMN)
                .expect("Failed to stack tensor data column"),
            device,
        )
        // Change from [B, H, W, C] to [B, C, H, W]
        .permute([0, 3, 1, 2])
        // Fixed normalization for Cinic-10 dataset
        .sub_scalar(0.4)
        // Fixed normalization for Cinic-10 dataset
        .div_scalar(0.2);

        let target_batch = Tensor::from_data(
            batch
                .iter()
                .map(|row| row.expect_get_parsed::<u32>(CLASS_COLUMN))
                .collect::<Vec<_>>()
                .as_slice(),
            device,
        );

        Ok((image_batch, target_batch))
    }
}
