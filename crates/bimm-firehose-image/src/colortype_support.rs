//! # [`ColorType`] Utilities
use image::{
    ColorType,
    DynamicImage,
};

/// Convert an `image::DynamicImage` to a specific `ColorType`.
///
/// # Parameters
///
/// - `img`: The input image to convert.
/// - `target_type`: The target color type to convert the image to.
///
/// # Returns
///
/// A new `DynamicImage` with the specified color type.
pub fn convert_to_colortype(
    img: DynamicImage,
    target_type: ColorType,
) -> DynamicImage {
    if img.color() == target_type {
        return img;
    }
    match target_type {
        ColorType::L8 => img.to_luma8().into(),
        ColorType::La8 => img.to_luma_alpha8().into(),
        ColorType::Rgb8 => img.to_rgb8().into(),
        ColorType::Rgba8 => img.to_rgba8().into(),
        ColorType::L16 => img.to_luma16().into(),
        ColorType::La16 => img.to_luma_alpha16().into(),
        ColorType::Rgb16 => img.to_rgb16().into(),
        ColorType::Rgba16 => img.to_rgba16().into(),
        ColorType::Rgb32F => img.to_rgb32f().into(),
        ColorType::Rgba32F => img.to_rgba32f().into(),
        _ => panic!("Unsupported ColorType: {target_type:?}"),
    }
}

pub mod serialization {
    //! Serde serialization support for `image::ColorType`.
    //!
    //! Upstream `image::ColorType` has Serde support for Serialize/Deserialize
    //! at head; but not in the current released crate. These mechanisms
    //! provide a Serialization / Deserialization workaround.

    use image::ColorType;
    use serde::{
        Deserialize,
        Serialize,
    };

    /// Temporary color type (with support for serialization and
    /// deserialization). To be replaced with `image::ColorType` on the next
    /// release.
    #[derive(Copy, PartialEq, Eq, Debug, Clone, Hash, Serialize, Deserialize)]
    pub enum TmpColorType {
        /// Pixel is 8-bit luminance
        L8,
        /// Pixel is 8-bit luminance with an alpha channel
        La8,
        /// Pixel contains 8-bit R, G and B channels
        Rgb8,
        /// Pixel is 8-bit RGB with an alpha channel
        Rgba8,

        /// Pixel is 16-bit luminance
        L16,
        /// Pixel is 16-bit luminance with an alpha channel
        La16,
        /// Pixel is 16-bit RGB
        Rgb16,
        /// Pixel is 16-bit RGBA
        Rgba16,

        /// Pixel is 32-bit float RGB
        Rgb32F,
        /// Pixel is 32-bit float RGBA
        Rgba32F,
    }

    impl From<ColorType> for TmpColorType {
        fn from(color: ColorType) -> Self {
            match color {
                ColorType::L8 => TmpColorType::L8,
                ColorType::La8 => TmpColorType::La8,
                ColorType::Rgb8 => TmpColorType::Rgb8,
                ColorType::Rgba8 => TmpColorType::Rgba8,
                ColorType::L16 => TmpColorType::L16,
                ColorType::La16 => TmpColorType::La16,
                ColorType::Rgb16 => TmpColorType::Rgb16,
                ColorType::Rgba16 => TmpColorType::Rgba16,
                ColorType::Rgb32F => TmpColorType::Rgb32F,
                ColorType::Rgba32F => TmpColorType::Rgba32F,
                _ => panic!("Unsupported ColorType: {color:?}"),
            }
        }
    }

    impl From<TmpColorType> for ColorType {
        fn from(color: TmpColorType) -> Self {
            match color {
                TmpColorType::L8 => ColorType::L8,
                TmpColorType::La8 => ColorType::La8,
                TmpColorType::Rgb8 => ColorType::Rgb8,
                TmpColorType::Rgba8 => ColorType::Rgba8,
                TmpColorType::L16 => ColorType::L16,
                TmpColorType::La16 => ColorType::La16,
                TmpColorType::Rgb16 => ColorType::Rgb16,
                TmpColorType::Rgba16 => ColorType::Rgba16,
                TmpColorType::Rgb32F => ColorType::Rgb32F,
                TmpColorType::Rgba32F => ColorType::Rgba32F,
            }
        }
    }

    /// Adapter Serializer for `Option<ColorType>`.
    ///
    /// This exists to allow serde to serialize `Option<ColorType>`,
    /// owing to a lagging bug in the upstream
    /// `ColorType` serialization implementation.
    ///
    /// # Example
    ///
    /// ```rust.norun
    /// use serde::{Serialize, Deserialize};
    ///
    /// #[derive(Serialize, Deserialize)]
    /// pub struct Example {
    ///     #[serde(skip_serializing_if = "Option::is_none")]
    ///     #[serde(serialize_with = "option_colortype_serializer")]
    ///     #[serde(deserialize_with = "option_colortype_deserializer")]
    ///     #[serde(default)]
    ///     color: Option<image::ColorType>,
    /// }
    /// ```
    pub fn option_colortype_serializer<S>(
        color: &Option<ColorType>,
        serializer: S,
    ) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        if color.is_none() {
            serializer.serialize_none()
        } else {
            let color = color.as_ref().unwrap();
            let color: Option<TmpColorType> = Some((*color).into());
            color.serialize(serializer)
        }
    }

    /// Adapter Deserializer for `Option<ColorType>`.
    ///
    /// This exists to allow serde to deserialize `Option<ColorType>`,
    /// owing to a lagging bug in the upstream
    /// `ColorType` serialization implementation.
    ///
    /// # Example
    ///
    /// ```rust.norun
    /// use serde::{Serialize, Deserialize};
    ///
    /// #[derive(Serialize, Deserialize)]
    /// pub struct Example {
    ///     #[serde(skip_serializing_if = "Option::is_none")]
    ///     #[serde(serialize_with = "option_colortype_serializer")]
    ///     #[serde(deserialize_with = "option_colortype_deserializer")]
    ///     #[serde(default)]
    ///     color: Option<image::ColorType>,
    /// }
    /// ```
    pub fn option_colortype_deserializer<'de, D>(
        deserializer: D
    ) -> Result<Option<ColorType>, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let color: Option<TmpColorType> = Option::deserialize(deserializer)?;
        Ok(color.map(|c| c.into()))
    }
}

#[cfg(test)]
mod tests {
    use image::{
        ColorType,
        DynamicImage,
    };
    use serde::{
        Deserialize,
        Serialize,
    };

    use super::{
        serialization,
        *,
    };
    use crate::{
        ImageShape,
        test_util::generate_gradient_pattern,
    };

    #[derive(Debug, Serialize, Deserialize)]
    pub struct ExampleStruct {
        #[serde(
            serialize_with = "serialization::option_colortype_serializer",
            deserialize_with = "serialization::option_colortype_deserializer"
        )]
        color: Option<ColorType>,
    }

    #[test]
    fn test_example_struct_serialization() {
        let example = ExampleStruct {
            color: Some(ColorType::Rgba8),
        };
        let serialized = serde_json::to_string(&example).unwrap();
        assert!(serialized.contains("\"color\":\"Rgba8\""));

        let deserialized: ExampleStruct = serde_json::from_str(&serialized).unwrap();
        assert_eq!(deserialized.color, Some(ColorType::Rgba8));

        let example_none = ExampleStruct { color: None };
        let serialized_none = serde_json::to_string(&example_none).unwrap();
        assert!(serialized_none.contains("\"color\":null"));

        let deserialized_none: ExampleStruct = serde_json::from_str(&serialized_none).unwrap();
        assert_eq!(deserialized_none.color, None);
    }

    #[test]
    fn test_to_from_color_type() {
        let color_types = [
            ColorType::L8,
            ColorType::La8,
            ColorType::Rgb8,
            ColorType::Rgba8,
            ColorType::L16,
            ColorType::La16,
            ColorType::Rgb16,
            ColorType::Rgba16,
            ColorType::Rgb32F,
            ColorType::Rgba32F,
        ];

        for color in color_types {
            let tmp_color: serialization::TmpColorType = color.into();
            let back_to_color: ColorType = tmp_color.into();
            assert_eq!(color, back_to_color);
        }
    }

    #[test]
    fn test_convert_to_colortype() {
        let shape = ImageShape {
            width: 32,
            height: 32,
        };
        let source = DynamicImage::from(generate_gradient_pattern(shape));

        assert_eq!(
            convert_to_colortype(source.clone(), ColorType::L8),
            DynamicImage::from(source.clone().to_luma8())
        );
        assert_eq!(
            convert_to_colortype(source.clone(), ColorType::La8),
            DynamicImage::from(source.clone().to_luma_alpha8())
        );
        assert_eq!(
            convert_to_colortype(source.clone(), ColorType::Rgb8),
            DynamicImage::from(source.clone().to_rgb8())
        );
        assert_eq!(
            convert_to_colortype(source.clone(), ColorType::Rgba8),
            DynamicImage::from(source.clone().to_rgba8())
        );
        assert_eq!(
            convert_to_colortype(source.clone(), ColorType::L16),
            DynamicImage::from(source.clone().to_luma16())
        );
        assert_eq!(
            convert_to_colortype(source.clone(), ColorType::La16),
            DynamicImage::from(source.clone().to_luma_alpha16())
        );
        assert_eq!(
            convert_to_colortype(source.clone(), ColorType::Rgb16),
            DynamicImage::from(source.clone().to_rgb16())
        );
        assert_eq!(
            convert_to_colortype(source.clone(), ColorType::Rgba16),
            DynamicImage::from(source.clone().to_rgba16())
        );
        assert_eq!(
            convert_to_colortype(source.clone(), ColorType::Rgb32F),
            DynamicImage::from(source.clone().to_rgb32f())
        );
        assert_eq!(
            convert_to_colortype(source.clone(), ColorType::Rgba32F),
            DynamicImage::from(source.clone().to_rgba32f())
        );
    }
}
