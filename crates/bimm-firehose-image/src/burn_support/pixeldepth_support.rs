//! Pixel depth conversion functions.
use burn::data::dataset::vision::PixelDepth;
use image::{
    ColorType,
    DynamicImage,
};

/// Converts a `PixelDepth` to a `u8`.
pub fn pixel_depth_to_u8(p: PixelDepth) -> u8 {
    match p {
        PixelDepth::U8(v) => v,
        // Convert U16 to U8 by taking the high byte
        PixelDepth::U16(v) => (v >> 8) as u8,
        // Scale F32 to U8
        PixelDepth::F32(v) => (v * 255.0) as u8,
    }
}

/// Converts a `PixelDepth` to a `u16`.
pub fn pixel_depth_to_u16(p: PixelDepth) -> u16 {
    match p {
        // Convert U8 to U16 by shifting left
        PixelDepth::U8(v) => (v as u16) << 8,
        PixelDepth::U16(v) => v,
        // Scale F32 to U16
        PixelDepth::F32(v) => (v * 65535.0) as u16,
    }
}

/// Converts a `PixelDepth` to a `f32`.
pub fn pixel_depth_to_f32(p: PixelDepth) -> f32 {
    match p {
        // Scale U8 to F32
        PixelDepth::U8(v) => v as f32 / 255.0,
        // Scale U16 to F32
        PixelDepth::U16(v) => v as f32 / 65535.0,
        PixelDepth::F32(v) => v,
    }
}

/// Converts an image to a vector of pixel depths.
pub fn image_to_pixeldepth_vec(image: &DynamicImage) -> Vec<PixelDepth> {
    let image = image.clone();
    // Image as Vec<PixelDepth>
    match image.color() {
        ColorType::L8 => image
            .into_luma8()
            .iter()
            .map(|&x| PixelDepth::U8(x))
            .collect(),
        ColorType::La8 => image
            .into_luma_alpha8()
            .iter()
            .map(|&x| PixelDepth::U8(x))
            .collect(),
        ColorType::L16 => image
            .into_luma16()
            .iter()
            .map(|&x| PixelDepth::U16(x))
            .collect(),
        ColorType::La16 => image
            .into_luma_alpha16()
            .iter()
            .map(|&x| PixelDepth::U16(x))
            .collect(),
        ColorType::Rgb8 => image
            .into_rgb8()
            .iter()
            .map(|&x| PixelDepth::U8(x))
            .collect(),
        ColorType::Rgba8 => image
            .into_rgba8()
            .iter()
            .map(|&x| PixelDepth::U8(x))
            .collect(),
        ColorType::Rgb16 => image
            .into_rgb16()
            .iter()
            .map(|&x| PixelDepth::U16(x))
            .collect(),
        ColorType::Rgba16 => image
            .into_rgba16()
            .iter()
            .map(|&x| PixelDepth::U16(x))
            .collect(),
        ColorType::Rgb32F => image
            .into_rgb32f()
            .iter()
            .map(|&x| PixelDepth::F32(x))
            .collect(),
        ColorType::Rgba32F => image
            .into_rgba32f()
            .iter()
            .map(|&x| PixelDepth::F32(x))
            .collect(),
        _ => panic!("Unrecognized image color type"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pixel_depth_to_u8() {
        assert_eq!(pixel_depth_to_u8(PixelDepth::U8(128)), 128);

        assert_eq!(pixel_depth_to_u8(PixelDepth::U16((12 << 8) + 231)), 12);

        assert_eq!(pixel_depth_to_u8(PixelDepth::F32(0.5)), 127);
        assert_eq!(pixel_depth_to_u8(PixelDepth::F32(1.0)), 255);
    }

    #[test]
    fn test_pixel_depth_to_u16() {
        assert_eq!(pixel_depth_to_u16(PixelDepth::U8(128)), 128 << 8);

        assert_eq!(
            pixel_depth_to_u16(PixelDepth::U16((12 << 8) + 231)),
            (12 << 8) + 231
        );

        assert_eq!(pixel_depth_to_u16(PixelDepth::F32(0.5)), (128 << 8) - 1);
    }

    #[test]
    fn test_pixel_depth_to_f32() {
        assert_eq!(pixel_depth_to_f32(PixelDepth::U8(127)), 127.0 / 255.0);

        assert_eq!(
            pixel_depth_to_f32(PixelDepth::U16((128 << 8) + 231)),
            ((128 << 8) + 231) as f32 / 65535.0,
        );

        assert_eq!(pixel_depth_to_f32(PixelDepth::F32(0.5)), 0.5);
    }
}
