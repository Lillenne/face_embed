use std::num::NonZeroU32;

use tract_onnx::prelude::*;

use crate::{
    embedding::{arcface_dims, ARCFACE_X, ARCFACE_Y},
    EmbeddingGenerator, ModelDims,
};

pub struct ArcFaceTract {
    model: SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>,
}

impl ModelDims for ArcFaceTract {
    fn dims(&self) -> (NonZeroU32, NonZeroU32, NonZeroU32, NonZeroU32) {
        arcface_dims()
    }
}

impl ArcFaceTract {
    pub fn new(path: &str) -> anyhow::Result<Self> {
        let model = tract_onnx::onnx()
            .model_for_path(path)?
            .into_optimized()?
            .into_runnable()?;
        Ok(Self { model })
    }

    fn preprocess(&self, face: &[u8]) -> anyhow::Result<Tensor> {
        if face.len() != ARCFACE_X * ARCFACE_Y * 3 {
            return Err(anyhow::anyhow!("Incorrect image dimensions"));
        }

        let input =
            tract_ndarray::Array4::from_shape_fn((1, 3, ARCFACE_Y, ARCFACE_X), |(_, c, y, x)| {
                let idx = (y * ARCFACE_X + x) * 3 + c;
                face[idx] as f32
            });

        Ok(input.into())
    }
}

impl EmbeddingGenerator for ArcFaceTract {
    fn generate_embedding(&self, face: &[u8]) -> anyhow::Result<Vec<f32>> {
        let input = self.preprocess(face)?;
        let result = self.model.run(tvec!(input.into()))?;

        let mut result = result[0]
            .to_array_view::<f32>()?
            .into_owned()
            .into_shape(512)?;
        // normalize to unit length
        let magnitude = result.dot(&result).sqrt();
        for v in result.iter_mut() {
            *v = *v / magnitude;
        }
        Ok(result.into_raw_vec())
    }
}

use crate::*;
use std::num::NonZeroU32;
use tract_ndarray::{prelude::*, Array4, ArrayView2, Dim, ViewRepr};
use tract_onnx::{prelude::*, tract_core::tract_data::itertools::Itertools};

type UltrafaceModel = SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

pub struct UltrafaceDetectorTract {
    model: UltrafaceModel,
    cfg: UltrafaceDetectorConfig,
}

impl UltrafaceDetectorTract {
    const ULTRAFACE_MEAN: f32 = 127.0;
    const ULTRAFACE_DIV: f32 = 128.0;
    const ULTRAFACE_N_BOXES: usize = 4420;
    const ULTRAFACE_FG_IDX: usize = 1;

    pub fn new(cfg: UltrafaceDetectorConfig, path: &str) -> anyhow::Result<UltrafaceDetector> {
        if !std::path::Path::new(path).exists() {
            return Err(anyhow::anyhow!("Model path does not exist"))
        }
        let model = tract_onnx::onnx()
            .model_for_path(path)?
            .into_optimized()?
            .into_runnable()?;
        Ok(UltrafaceDetector { cfg, model })
    }

    /// Preprocess an RGB image
    fn preprocess(&self, data: &[u8]) -> anyhow::Result<Tensor> {
        if self.n_elements() != data.len() {
            return Err(anyhow::anyhow!("Incorrect data shape"))
        }
        let input: Tensor = Array4::from_shape_fn(
            (
                1,
                3,
                self.cfg.model_height.get() as usize,
                self.cfg.model_width.get() as usize,
            ),
            |(_, c, y, x)| {
                let idx = (y * self.cfg.model_width.get() as usize + x) * 3 + c;
                (data[idx] as f32 - UltrafaceDetector::ULTRAFACE_MEAN)
                    / UltrafaceDetector::ULTRAFACE_DIV
            },
        )
        .into();
        Ok(input)
    }

    /// Convert Ultraface bounding box outputs (x1, y1, x2, y2) to Rect
    fn get_rect(&self, view: ArrayBase<ViewRepr<&f32>, Dim<[usize; 1]>>) -> Rect {
        if view.len() != 4 {
            panic!("View len != 4")
        }
        let left = view[0].clamp(0.0, 1.0);
        let top = view[1].clamp(0.0, 1.0);
        let width = (view[2] - left).clamp(0.0, 1.0 - left);
        let height = (view[3] - top).clamp(0.0, 1.0 - top);
        Rect { left, top, width, height }
    }

    fn n_elements(&self) -> usize {
        self.cfg.model_width.get() as usize * self.cfg.model_height.get() as usize * 3
    }

    fn nms(&self, probs: ArrayView2<f32>, boxes: ArrayView2<f32>) -> Vec<DetectedObject> {
        probs.column(UltrafaceDetector::ULTRAFACE_FG_IDX)
        .iter()
        .enumerate()
        .filter(move |v| {
            if *v.1 < self.cfg.prob_threshold.value {
                // probability too low
                return false
            }

            // perform nms
            let this_box = self.get_rect(boxes.row(v.0));

            for (i, bbox) in boxes.axis_iter(Axis(0)).enumerate() {
                let rect = self.get_rect(bbox);
                let iou = this_box.iou(&rect);
                if iou < self.cfg.iou_threshold.value {
                    // insufficient intersection to suppress
                    continue
                }
                if *v.1 < probs[[i, UltrafaceDetector::ULTRAFACE_FG_IDX]] {
                    // overlapping box with higher probability exists
                    return false
                }
            }
            true
        })
        .sorted_by(|a, b| b.1.partial_cmp(&a.1).unwrap())
        .take(self.cfg.top_k.get() as usize)
        .map(|i| DetectedObject {
            class: ULTRAFACE_FG_IDX,
            confidence: *i.1,
            bounding_box: self.get_rect(boxes.row(i.0)),
        })
        .collect_vec()
    }
}

impl ModelDims for UltrafaceDetectorTract {
    fn dims(&self) -> (NonZeroU32, NonZeroU32, NonZeroU32, NonZeroU32) {
        (NonZeroU32::new(1).unwrap(), NonZeroU32::new(3).unwrap(), self.cfg.model_height, self.cfg.model_width)
    }
}

impl FaceDetector for UltrafaceDetectorTract {
    fn detect(&self, frame: &[u8]) -> anyhow::Result<Vec<DetectedObject>> {
        let data = self.preprocess(frame)?;
        let result = self.model.run(tvec!(data.into()))?;
        let probs = result[0]
            .to_array_view::<f32>()?
            .into_shape((UltrafaceDetector::ULTRAFACE_N_BOXES, 2))?;
        let boxes = result[1]
            .to_array_view::<f32>()?
            .into_shape((UltrafaceDetector::ULTRAFACE_N_BOXES, 4))?;
        Ok(self.nms(probs, boxes))
    }
}
