//! # Image test utilities.
use image::{
    DynamicImage,
    ImageBuffer,
    Rgb,
    RgbImage,
};
use image_compare::BlendInput;

use crate::ImageShape;

/// Generates a simple gradient pattern image.
pub fn generate_gradient_pattern(shape: ImageShape) -> RgbImage {
    let r_scale = shape.width as f32 - 1.0;
    let g_scale = shape.height as f32 - 1.0;
    let b_scale = (shape.width + shape.height - 2) as f32;

    ImageBuffer::from_fn(shape.width, shape.height, |x, y| {
        let a = x as f32 * 255.0;
        let b = y as f32 * 255.0;

        let r = (a / r_scale) as u8;
        let g = (b / g_scale) as u8;
        let b = ((a + b) / b_scale) as u8;

        Rgb([r, g, b])
    })
}

/// Asserts that two images are similar within a given tolerance.
///
/// This function uses the `image_compare` crate to compare two images and
/// checks if their similarity score is above a specified tolerance.
///
/// # Arguments
///
/// * `actual` - The actual image to compare.
/// * `expected` - The expected image to compare against.
/// * `tolerance` - An optional tolerance value for the similarity score. If not
///   provided, defaults to 0.01.
pub fn assert_image_close_rgba<'a, A, B>(
    actual: A,
    expected: B,
    tolerance: Option<f64>,
) where
    A: Into<BlendInput<'a>>,
    B: Into<BlendInput<'a>>,
{
    let tolerance = tolerance.unwrap_or(0.01);
    let actual = actual.into();
    let expected = expected.into();

    let white = Rgb([255, 255, 255]);
    match image_compare::rgba_blended_hybrid_compare(actual, expected, white) {
        Ok(similarity) => {
            let target_score = 1.0 - tolerance;
            assert!(
                similarity.score >= target_score,
                "Image similarity {} < target {target_score}",
                similarity.score
            );
        }
        Err(e) => panic!("Image comparison failed: {e:?}"),
    }
}

/// Asserts that two `DynamicImage` instances are similar within a given
/// tolerance.
///
/// This function converts both images to RGBA format before comparing them.
///
/// # Arguments
///
/// * `actual` - The actual image to compare.
/// * `expected` - The expected image to compare against.
/// * `tolerance` - An optional tolerance value for the similarity score. If not
///   provided, defaults to 0.01.
///
/// # Panics
///
/// This function will panic if the images are not similar enough according to
/// the specified tolerance.
pub fn assert_image_close(
    actual: &DynamicImage,
    expected: &DynamicImage,
    tolerance: Option<f64>,
) {
    let actual = actual.to_rgba8();
    let expected = expected.to_rgba8();
    assert_image_close_rgba(&actual, &expected, tolerance);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_gradient_pattern() {
        let shape = ImageShape {
            width: 32,
            height: 32,
        };

        let image = generate_gradient_pattern(shape);
        assert_image_close_rgba(&image, &image, None);

        let dyn_image = DynamicImage::from(image.clone());
        assert_image_close(&dyn_image, &dyn_image, None);

        assert_eq!(image.width(), 32);
        assert_eq!(image.height(), 32);

        // Check some pixel values
        assert_eq!(image.get_pixel(0, 0), &Rgb([0, 0, 0]));
        assert_eq!(image.get_pixel(16, 16), &Rgb([131, 131, 131]));
        assert_eq!(image.get_pixel(16, 31), &Rgb([131, 255, 193]));
        assert_eq!(image.get_pixel(31, 31), &Rgb([255, 255, 255]));
    }
}
