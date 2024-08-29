use std::num::NonZeroU32;

#[cfg(feature = "tract")]
pub use crate::tract_backend::ArcFaceTract as ArcFace;

#[cfg(feature = "ort")]
pub use crate::ort_backend::ArcFaceOrt as ArcFace;

pub const ARCFACE_X: usize = 112;
pub const ARCFACE_Y: usize = 112;

pub fn arcface_dims() -> (NonZeroU32, NonZeroU32, NonZeroU32, NonZeroU32) {
    (
        NonZeroU32::new(1).unwrap(),
        NonZeroU32::new(3).unwrap(),
        NonZeroU32::new(ARCFACE_Y as u32).unwrap(),
        NonZeroU32::new(ARCFACE_X as u32).unwrap(),
    )
}

// #[cfg(test)]
// mod tests {
//     use super::ArcFace;
//     use crate::image_utils::{crop_and_resize, resize};
//     use crate::similarity;
//     use crate::EmbeddingGenerator;
//     use crate::{face_detector::*, FaceDetector};
//     use fast_image_resize as fr;
//     use image::EncodableLayout;
//     use std::num::NonZeroU32;

//     const ARCFACE_PATH: &str = "../models/arcface.onnx";
//     const ULTRAFACE_PATH: &str = "../models/ultraface-RFB-320.onnx";

//     #[test]
//     fn similarity_sameperson_gthalf() {
//         const A_PATH: &str = "../test_data/me.jpg";
//         const B_PATH: &str = "../test_data/me2.jpg";
//         let similarity = calculate_similarity(A_PATH, B_PATH).unwrap();
//         println!("Similarity (same) {}", similarity);
//         assert!(similarity > 0.5);
//     }

//     #[test]
//     fn similarity_difperson_lthalf() {
//         const A_PATH: &str = "../test_data/me.jpg";
//         const B_PATH: &str = "../test_data/son.jpg";
//         let similarity = calculate_similarity(A_PATH, B_PATH).unwrap();
//         println!("Similarity (dif) {}", similarity);
//         assert!(similarity < 0.5);
//     }

//     fn calculate_similarity(a_path: &str, b_path: &str) -> anyhow::Result<f32> {
//         let arcface = ArcFace::new(ARCFACE_PATH)?;
//         let det = UltrafaceDetector::new(
//             UltrafaceDetectorConfig {
//                 top_k: NonZeroU32::new(1).unwrap(),
//                 ..Default::default()
//             },
//             ULTRAFACE_PATH,
//         )
//         .unwrap();
//         let a = get_embedding(a_path, &arcface, &det)?;
//         let b = get_embedding(b_path, &arcface, &det)?;
//         Ok(similarity(a.as_slice(), b.as_slice()))
//     }

//     fn get_embedding(
//         path: &str,
//         arcface: &impl EmbeddingGenerator,
//         det: &impl FaceDetector,
//     ) -> anyhow::Result<Vec<f32>> {
//         let embedding_dims = arcface.dims();
//         let det_dims = det.dims();
//         let a_in = image::open(path)?;
//         let rgb = a_in.as_rgb8().expect("Expected rgb8 image");
//         let x = NonZeroU32::new(rgb.width()).expect("Zero width");
//         let y = NonZeroU32::new(rgb.height()).expect("Zero height");
//         let img = resize(
//             rgb.as_bytes(),
//             x,
//             y,
//             det_dims.3,
//             det_dims.2,
//             fr::ResizeAlg::Nearest,
//         )
//         .expect("Resize img");
//         let face = det.detect(img.buffer())?[0];
//         let crop = face.bounding_box.to_crop_box(x.get(), y.get());
//         let mut vec = rgb.clone().into_vec();
//         let slice = vec.as_mut_slice();
//         let region = fr::Image::from_slice_u8(x, y, slice, fr::PixelType::U8x3)?;
//         let roi = crop_and_resize(
//             &mut (region.view()),
//             embedding_dims.3,
//             embedding_dims.2,
//             crop,
//             fr::ResizeAlg::Nearest,
//         )?
//         .into_vec();
//         arcface.generate_embedding(roi.as_bytes())
//     }
// }
