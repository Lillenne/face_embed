use std::num::NonZeroU32;

use tract_onnx::prelude::*;

use crate::face_detector::ModelDims;

type ArcFaceModel = SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

/// Defines an embedding generating algorithm
pub trait EmbeddingGenerator: ModelDims {

    /// Generates embeddings from an image
    fn generate_embedding(&self, face: &[u8]) -> anyhow::Result<Vec<f32>> ;
}

pub struct ArcFace {
    model: ArcFaceModel
}

impl ArcFace {
    const ARCFACE_X: usize = 112;
    const ARCFACE_Y: usize = 112;
    pub fn new(path: &str) -> anyhow::Result<ArcFace> {
        let model = tract_onnx::onnx()
        .model_for_path(path)?
        .into_optimized()?
        .into_runnable()?;
        Ok(ArcFace { model })
    }
}

impl ModelDims for ArcFace {
    fn dims(&self) -> (NonZeroU32, NonZeroU32, NonZeroU32, NonZeroU32) {
        (NonZeroU32::new(1).unwrap(),
         NonZeroU32::new(3).unwrap(),
         NonZeroU32::new(ArcFace::ARCFACE_Y as u32).unwrap(),
         NonZeroU32::new(ArcFace::ARCFACE_X as u32).unwrap())
    }
}

impl EmbeddingGenerator for ArcFace {
    fn generate_embedding(&self, face: &[u8]) -> anyhow::Result<Vec<f32>> {
        //anyhow::Result<tract_ndarray::ArrayBase<OwnedRepr<f32>, tract_ndarray::Dim<[usize; 2]>>> {
        if face.len() != ArcFace::ARCFACE_X * ArcFace::ARCFACE_Y * 3 {
            return Err(anyhow::anyhow!("Incorrect image dimensions"))
        }

        let input = Tensor::from_shape(&[1,3,ArcFace::ARCFACE_X, ArcFace::ARCFACE_Y], face)?;
        let cast = input.cast_to_dt(DatumType::F32)?.cast_to_dt(DatumType::F32)?.into_owned();
        let result = self.model.run(tvec!(cast.into()))?;
        Ok(result[0].to_array_view::<f32>()?.into_owned().into_raw_vec())
    }
}
