//! # Image/Tensor conversion utilities.
use bimm_firehose::core::operations::factory::SimpleConfigOperatorFactory;
use bimm_firehose::core::operations::signature::{FirehoseOperatorSignature, ParameterSpec};
use bimm_firehose::core::{FirehoseRowBatch, FirehoseRowReader};
use bimm_firehose::define_firehose_operator;
use burn::prelude::{Backend, Tensor};
use burn::tensor::TensorData;
use image::DynamicImage;

pub mod image_to_tensor_data;

pub mod pixeldepth_support;

pub use image_to_tensor_data::*;
use pixeldepth_support::image_to_pixeldepth_vec;

define_firehose_operator!(
    IMAGE_TO_TENSOR_DATA,
    SimpleConfigOperatorFactory::<ImageToTensorData>::new(
        FirehoseOperatorSignature::new()
            .with_operator_id(IMAGE_TO_TENSOR_DATA)
            .with_description("Converts an image to TensorData.")
            .with_input(
                ParameterSpec::new::<DynamicImage>("image")
                    .with_description("Image to convert to a tensor."),
            )
            .with_output(
                ParameterSpec::new::<TensorData>("data")
                    .with_description("TensorData representation of the image."),
            ),
    )
);

/// Stacks the tensor data from a batch of rows into a single `TensorData`.
///
/// # Arguments
///
/// * `batch` - The batch of rows containing the tensor data.
/// * `column_name` - The name of the column containing the tensor data.
///
/// # Returns
///
/// An `anyhow::Result<TensorData`.
pub fn stack_tensor_data_column(
    batch: &FirehoseRowBatch,
    column_name: &str,
) -> anyhow::Result<TensorData> {
    assert!(!batch.is_empty());

    let item_shape = batch[0]
        .expect_get_ref::<TensorData>(column_name)
        .shape
        .clone();
    let stack_shape = [batch.len(), item_shape[0], item_shape[1], item_shape[2]];

    let data_vec = batch
        .iter()
        .map(|row| {
            row.expect_get_ref::<TensorData>(column_name)
                .as_slice::<f32>()
                .map_err(|_| "Failed to get slice from tensor data")
                .unwrap()
        })
        .collect::<Vec<_>>();

    let total_len = data_vec.iter().map(|&d| d.len()).sum::<usize>();
    let mut stack_data = Vec::with_capacity(total_len);
    data_vec.iter().for_each(|d| {
        stack_data.extend_from_slice(d);
    });

    Ok(TensorData::new(stack_data, stack_shape))
}

/// Converts an image to a tensor `[h, w, c]` Float tensor of type `f32`.
///
/// # Arguments
///
/// * `image` - The image to convert.
/// * `device` - The device to create the tensor on.
///
/// # Returns
///
/// A tensor representation of the image with shape `[height, width, channels]`.
pub fn image_to_f32_tensor<B: Backend>(
    image: &DynamicImage,
    device: &B::Device,
) -> Tensor<B, 3> {
    let height = image.height() as usize;
    let width = image.width() as usize;
    let colors = image.color().channel_count() as usize;
    let shape = vec![height, width, colors];

    let pixvec = image_to_pixeldepth_vec(image);
    let data: Vec<f32> = pixvec
        .iter()
        .map(|p| pixeldepth_support::pixel_depth_to_f32(*p))
        .collect();

    Tensor::from_data_dtype(
        TensorData::new(data, shape),
        device,
        burn::tensor::DType::F32,
    )
}
