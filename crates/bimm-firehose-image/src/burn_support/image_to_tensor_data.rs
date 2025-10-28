//! Image to tensor data conversion functions.
use crate::burn_support::{IMAGE_TO_TENSOR_DATA, pixeldepth_support};
use bimm_firehose::core::operations::operator::FirehoseOperator;
use bimm_firehose::core::operations::planner::OperationPlan;
use bimm_firehose::core::rows::FirehoseRowTransaction;
use bimm_firehose::core::{FirehoseRowReader, FirehoseRowWriter};
use burn::prelude::TensorData;
use image::DynamicImage;
use serde::{Deserialize, Serialize};

/// The `ImageToTensorData` operator.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ImageToTensorData {}

impl Default for ImageToTensorData {
    fn default() -> Self {
        Self::new()
    }
}

impl ImageToTensorData {
    /// Creates a new `ImgToTensorConfig`.
    pub fn new() -> Self {
        ImageToTensorData {}
    }

    /// Converts this configuration to an `OperationPlanner` for the `ImgToTensor` operator.
    ///
    /// # Arguments
    ///
    /// * `image_column` - The name of the input column containing the image.
    /// * `data_column` - The name of the output column for the tensor data.
    pub fn to_plan(
        self,
        image_column: &str,
        data_column: &str,
    ) -> OperationPlan {
        OperationPlan::for_operation_id(IMAGE_TO_TENSOR_DATA)
            .with_input("image", image_column)
            .with_output("data", data_column)
            .with_config(self)
    }
}

impl FirehoseOperator for ImageToTensorData {
    fn apply_to_row(
        &self,
        txn: &mut FirehoseRowTransaction,
    ) -> anyhow::Result<()> {
        let image: &DynamicImage = txn.expect_get_ref("image");

        let height = image.height() as usize;
        let width = image.width() as usize;
        let colors = image.color().channel_count() as usize;
        let shape = vec![height, width, colors];

        let pixvec = crate::burn_support::image_to_pixeldepth_vec(image);
        let data: Vec<f32> = pixvec
            .iter()
            .map(|p| pixeldepth_support::pixel_depth_to_f32(*p))
            .collect();

        let data = TensorData::new(data, shape);

        txn.expect_set_boxing("data", data);

        Ok(())
    }
}
