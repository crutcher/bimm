use anyhow::Context;
use bimm_firehose::{
    core::{
        FirehoseRowReader,
        FirehoseRowWriter,
        FirehoseValue,
        operations::{
            factory::SimpleConfigOperatorFactory,
            operator::FirehoseOperator,
            planner::OperationPlan,
            signature::{
                FirehoseOperatorSignature,
                ParameterSpec,
            },
        },
        rows::FirehoseRowTransaction,
    },
    define_firehose_operator,
};
pub use image::{
    ColorType,
    DynamicImage,
    imageops::FilterType,
};
use serde::{
    Deserialize,
    Serialize,
};

/// # # [`Image`] loader operators.
use crate::{
    ImageShape,
    colortype_support,
};

define_firehose_operator!(
    LOAD_IMAGE,
    SimpleConfigOperatorFactory::<ImageLoader>::new(
        FirehoseOperatorSignature::from_operator_id(LOAD_IMAGE)
            .with_description("Loads an image from disk.")
            .with_input(
                ParameterSpec::new::<String>("path").with_description("Path to the image file."),
            )
            .with_output(
                ParameterSpec::new::<DynamicImage>("image")
                    .with_description("Image loaded from disk."),
            ),
    )
);

/// Represents the resize specification for an image.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResizeSpec {
    /// The target shape of the image after resizing.
    pub shape: ImageShape,

    /// The filter type to use for resizing.
    pub filter: FilterType,
}

impl ResizeSpec {
    /// Creates a new `ResizeSpec` with the specified shape and filter.
    pub fn new(shape: ImageShape) -> Self {
        ResizeSpec {
            shape,
            filter: FilterType::CatmullRom,
        }
    }

    /// Extends the `ResizeSpec` with a new shape, keeping the existing filter
    /// type.
    pub fn with_filter(
        self,
        filter: FilterType,
    ) -> Self {
        ResizeSpec {
            shape: self.shape,
            filter,
        }
    }
}

/// An operator that loads an image from disk and optionally resizes it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageLoader {
    /// If enabled, the loader will resize images to the specified shape.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    pub resize: Option<ResizeSpec>,

    /// If enabled, the loader will recolor images to the specified color type.
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Upstream `image::ColorType` is not serializable in the latest release;
    /// but *is* in the base repo, and this will change.
    #[serde(
        serialize_with = "colortype_support::serialization::option_colortype_serializer",
        deserialize_with = "colortype_support::serialization::option_colortype_deserializer"
    )]
    #[serde(default)]
    pub recolor: Option<ColorType>,
}

impl Default for ImageLoader {
    fn default() -> Self {
        Self::new()
    }
}

impl ImageLoader {
    /// Creates a new `ImageLoader` with optional resize and recolor
    /// specifications.
    pub fn new() -> Self {
        ImageLoader {
            resize: None,
            recolor: None,
        }
    }

    /// Extends the `ImageLoader` with a resize specification.
    ///
    /// # Arguments
    ///
    /// * `resize`: The resize specification to apply to the image.
    ///
    /// # Returns
    ///
    /// A new `ImageLoader` instance with the specified resize applied.
    pub fn with_resize(
        self,
        resize: ResizeSpec,
    ) -> Self {
        ImageLoader {
            resize: Some(resize),
            recolor: self.recolor,
        }
    }

    /// Extends the `ImageLoader` with a recolor specification.
    ///
    /// # Arguments
    ///
    /// * `recolor`: The color type to convert the image to.
    ///
    /// # Returns
    ///
    /// A new `ImageLoader` instance with the specified recolor applied.
    pub fn with_recolor(
        self,
        recolor: ColorType,
    ) -> Self {
        ImageLoader {
            resize: self.resize,
            recolor: Some(recolor),
        }
    }

    /// Converts this `ImageLoader` configuration into an `OperationPlanner`
    ///
    /// # Arguments
    ///
    /// * `path_column`: The name of the input column containing the image file
    ///   paths.
    /// * `image_column`: The name of the output column where the loaded images
    ///   will be stored.
    pub fn to_plan(
        &self,
        path_column: &str,
        image_column: &str,
    ) -> OperationPlan {
        OperationPlan::for_operation_id(LOAD_IMAGE)
            .with_input("path", path_column)
            .with_output("image", image_column)
            .with_config(self.clone())
    }
}

impl FirehoseOperator for ImageLoader {
    fn apply_to_row(
        &self,
        txn: &mut FirehoseRowTransaction,
    ) -> anyhow::Result<()> {
        let path = txn.maybe_get("path").unwrap().parse_as::<String>()?;

        let mut image = image::open(path.clone())
            .with_context(|| format!("Failed to load image from path: {path}"))?;

        if let Some(spec) = &self.resize
            && (image.width() != spec.shape.width || image.height() != spec.shape.height)
        {
            image = image.resize_exact(spec.shape.width, spec.shape.height, spec.filter);
        }

        if let Some(color) = &self.recolor {
            let color: ColorType = *color;
            image = colortype_support::convert_to_colortype(image, color);
        }

        txn.expect_set("image", FirehoseValue::boxing(image));

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use anyhow::Context;
    use bimm_firehose::{
        core::{
            FirehoseRowBatch,
            FirehoseRowReader,
            FirehoseRowWriter,
            FirehoseValue,
            operations::executor::{
                FirehoseBatchExecutor,
                SequentialBatchExecutor,
            },
            schema::{
                ColumnSchema,
                FirehoseTableSchema,
            },
        },
        ops::init_default_operator_environment,
    };
    use image::DynamicImage;

    use super::*;
    use crate::test_util::{
        assert_image_close,
        generate_gradient_pattern,
    };

    #[test]
    fn test_image_loader() -> anyhow::Result<()> {
        let temp_dir =
            tempfile::tempdir().with_context(|| "Failed to create temporary directory")?;

        let image_path = temp_dir
            .path()
            .join("gradient.png")
            .to_string_lossy()
            .to_string();

        let source_image: DynamicImage = generate_gradient_pattern(ImageShape {
            width: 32,
            height: 32,
        })
        .into();

        source_image
            .save(&image_path)
            .expect("Failed to save test image");

        let env = Arc::new(init_default_operator_environment());

        let schema = {
            let mut schema =
                FirehoseTableSchema::from_columns(&[ColumnSchema::new::<String>("path")]);

            ImageLoader::default()
                .to_plan("path", "raw_image")
                .apply_to_schema(&mut schema, env.as_ref())?;

            ImageLoader::default()
                .with_resize(
                    ResizeSpec::new(ImageShape {
                        width: 16,
                        height: 16,
                    })
                    .with_filter(FilterType::Nearest),
                )
                .with_recolor(ColorType::L16)
                .to_plan("path", "resized_gray")
                .apply_to_schema(&mut schema, env.as_ref())?;

            Arc::new(schema)
        };

        let executor = SequentialBatchExecutor::new(schema.clone(), env.clone())?;

        let mut batch = FirehoseRowBatch::new_with_size(schema.clone(), 1);
        batch[0].expect_set("path", FirehoseValue::serialized(image_path)?);

        executor.execute_batch(&mut batch)?;

        let loaded_image = batch[0]
            .maybe_get("raw_image")
            .unwrap()
            .as_ref::<DynamicImage>()?;
        assert_eq!(loaded_image.width(), 32);
        assert_eq!(loaded_image.height(), 32);
        assert_eq!(loaded_image.color(), ColorType::Rgb8);
        assert_image_close(loaded_image, &source_image, None);

        let gray_image = batch[0]
            .maybe_get("resized_gray")
            .unwrap()
            .as_ref::<DynamicImage>()?;
        assert_eq!(gray_image.width(), 16);
        assert_eq!(gray_image.height(), 16);
        assert_eq!(gray_image.color(), ColorType::L16);
        assert_image_close(
            gray_image,
            &source_image
                .resize_exact(16, 16, FilterType::Nearest)
                .to_luma8()
                .into(),
            None,
        );

        Ok(())
    }
}
