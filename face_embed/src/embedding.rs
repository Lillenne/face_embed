use std::num::NonZeroU32;

use crate::face_detector::ModelDims;

/// Defines an embedding generating algorithm
pub trait EmbeddingGenerator: ModelDims {
    /// Generates embeddings from an image
    fn generate_embedding(&self, face: &[u8]) -> anyhow::Result<Vec<f32>>;
}

pub fn similarity<T: num_traits::Float>(embedding: &[T], nearest_embed: &[T]) -> T {
    embedding
        .iter()
        .zip(nearest_embed.iter())
        .map(|(a, b)| *a * *b)
        .reduce(|a, b| a + b)
        .unwrap()
}

pub struct ArcFace {
    model: ort::Session,
}

impl ModelDims for ArcFace {
    fn dims(&self) -> (NonZeroU32, NonZeroU32, NonZeroU32, NonZeroU32) {
        (
            NonZeroU32::new(1).unwrap(),
            NonZeroU32::new(3).unwrap(),
            NonZeroU32::new(ArcFace::ARCFACE_Y as u32).unwrap(),
            NonZeroU32::new(ArcFace::ARCFACE_X as u32).unwrap(),
        )
    }
}

impl ArcFace {
    const ARCFACE_X: usize = 112;
    const ARCFACE_Y: usize = 112;

    pub fn new(path: &str) -> anyhow::Result<ArcFace> {
        let session = ort::Session::builder()?
            .with_optimization_level(ort::GraphOptimizationLevel::Level1)?
            .with_parallel_execution(false)?
            .with_intra_threads(2)?
            .commit_from_file(path)?;

        Ok(ArcFace { model: session })
    }

    fn preprocess(
        &self,
        face: &[u8],
    ) -> anyhow::Result<ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<[usize; 4]>>> {
        if face.len() != ArcFace::ARCFACE_X * ArcFace::ARCFACE_Y * 3 {
            return Err(anyhow::anyhow!("Incorrect image dimensions"));
        }

        let input = ndarray::Array4::from_shape_fn(
            (1, 3, ArcFace::ARCFACE_Y, ArcFace::ARCFACE_X),
            |(_, c, y, x)| {
                let idx = (y * ArcFace::ARCFACE_X as usize + x) * 3 + c;
                face[idx] as f32
            },
        );

        Ok(input)
    }
}

impl EmbeddingGenerator for ArcFace {
    fn generate_embedding(&self, face: &[u8]) -> anyhow::Result<Vec<f32>> {
        let input = self.preprocess(face)?;
        let res = self.model.run(ort::inputs!("data" => input)?)?;
        let mut output = res["fc1"].try_extract_raw_tensor::<f32>()?.1.to_vec();
        // let mut output = res["fc1"].extract_tensor::<f32>()?.view().to_owned().into_raw_vec();
        // normalize to unit length
        let magnitude = output
            .iter()
            .zip(output.iter())
            .map(|(a, b)| a * b)
            .reduce(|a, b| a + b)
            .unwrap()
            .sqrt();
        for v in output.iter_mut() {
            *v = *v / magnitude;
        }
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::similarity;
    use crate::embedding::{ArcFace, EmbeddingGenerator};
    use crate::face_detector::*;
    use crate::image_utils::{crop_and_resize, resize};
    use fast_image_resize as fr;
    use image::EncodableLayout;
    use std::num::NonZeroU32;

    const ARCFACE_PATH: &str = "../models/arcface.onnx";
    const ULTRAFACE_PATH: &str = "../models/ultraface-RFB-320.onnx";

    #[test]
    fn similarity_sameperson_gthalf() {
        const A_PATH: &str = "../test_data/me.jpg";
        const B_PATH: &str = "../test_data/me2.jpg";
        let similarity = calculate_similarity(A_PATH, B_PATH).unwrap();
        println!("Similarity (same) {}", similarity);
        assert!(similarity > 0.5);
    }

    #[test]
    fn similarity_difperson_lthalf() {
        const A_PATH: &str = "../test_data/me.jpg";
        const B_PATH: &str = "../test_data/son.jpg";
        let similarity = calculate_similarity(A_PATH, B_PATH).unwrap();
        println!("Similarity (dif) {}", similarity);
        assert!(similarity < 0.5);
    }

    fn calculate_similarity(a_path: &str, b_path: &str) -> anyhow::Result<f32> {
        let arcface = ArcFace::new(ARCFACE_PATH)?;
        let det = UltrafaceDetector::new(
            UltrafaceDetectorConfig {
                top_k: NonZeroU32::new(1).unwrap(),
                ..Default::default()
            },
            ULTRAFACE_PATH,
        )
        .unwrap();
        let a = get_embedding(a_path, &arcface, &det)?;
        let b = get_embedding(b_path, &arcface, &det)?;
        Ok(similarity(a.as_slice(), b.as_slice()))
    }

    fn get_embedding(
        path: &str,
        arcface: &impl EmbeddingGenerator,
        det: &impl FaceDetector,
    ) -> anyhow::Result<Vec<f32>> {
        let embedding_dims = arcface.dims();
        let det_dims = det.dims();
        let a_in = image::open(path)?;
        let rgb = a_in.as_rgb8().expect("Expected rgb8 image");
        let x = NonZeroU32::new(rgb.width()).expect("Zero width");
        let y = NonZeroU32::new(rgb.height()).expect("Zero height");
        let img = resize(
            rgb.as_bytes(),
            x,
            y,
            det_dims.3,
            det_dims.2,
            fr::ResizeAlg::Nearest,
        )
        .expect("Resize img");
        let face = det.detect(img.buffer())?[0];
        let crop = face.bounding_box.to_crop_box(x.get(), y.get());
        let mut vec = rgb.clone().into_vec();
        let slice = vec.as_mut_slice();
        let region = fr::Image::from_slice_u8(x, y, slice, fr::PixelType::U8x3)?;
        let roi = crop_and_resize(
            &mut (region.view()),
            embedding_dims.3,
            embedding_dims.2,
            crop,
            fr::ResizeAlg::Nearest,
        )?
        .into_vec();
        Ok(arcface.generate_embedding(roi.as_bytes())?)
    }
}
