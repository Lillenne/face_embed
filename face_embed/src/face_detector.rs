use crate::*;
use std::num::NonZeroU32;
use tract_ndarray::{prelude::*, Array4, ArrayView2, Dim, ViewRepr};
use tract_onnx::{prelude::*, tract_core::tract_data::itertools::Itertools};

type UltrafaceModel = SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

pub trait ModelDims {
    /// Returns the model dimensions (b,c,h,w)
    fn dims(&self) -> (NonZeroU32, NonZeroU32, NonZeroU32, NonZeroU32);
}

/// Defines a face detection algorithm
pub trait FaceDetector: ModelDims {
    /// Detects faces in an RGB image.
    fn detect(&self, frame: &[u8]) -> anyhow::Result<Vec<Rect>>;
}

#[derive(Clone, Copy, Debug)]
pub struct UltrafaceDetectorConfig {
    pub iou_threshold: ZeroToOneF32,
    pub prob_threshold: ZeroToOneF32,
    pub top_k: NonZeroU32,
    pub model_width: NonZeroU32,
    pub model_height: NonZeroU32,
}

impl Default for UltrafaceDetectorConfig {
    fn default() -> Self {
        UltrafaceDetectorConfig {
            iou_threshold: ZeroToOneF32::new(0.2).unwrap(),
            prob_threshold: ZeroToOneF32::new(0.75).unwrap(),
            top_k: NonZeroU32::new(1).unwrap(),
            model_width: NonZeroU32::new(320).unwrap(),
            model_height: NonZeroU32::new(240).unwrap(),
        }
    }
}

pub struct UltrafaceDetector {
    model: UltrafaceModel,
    cfg: UltrafaceDetectorConfig,
}

impl UltrafaceDetector {
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
        Rect { left: view[0], top: view[1], width: view[2] - view[0], height: view[3] - view[1] }
    }

    fn n_elements(&self) -> usize {
        self.cfg.model_width.get() as usize * self.cfg.model_height.get() as usize * 3
    }

    fn nms(&self, probs: ArrayView2<f32>, boxes: ArrayView2<f32>) -> Vec<Rect> {
            probs.column(UltrafaceDetector::ULTRAFACE_FG_IDX)
            .iter()
            .enumerate()
            .filter(|v| {
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
            .sorted_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|i| self.get_rect(boxes.row(i.0)))
            .take(self.cfg.top_k.get() as usize)
            .collect_vec()
    }
}

impl ModelDims for UltrafaceDetector {
    fn dims(&self) -> (NonZeroU32, NonZeroU32, NonZeroU32, NonZeroU32) {
        (NonZeroU32::new(1).unwrap(), NonZeroU32::new(3).unwrap(), self.cfg.model_height, self.cfg.model_width)
    }
}

impl FaceDetector for UltrafaceDetector {
    fn detect(&self, frame: &[u8]) -> anyhow::Result<Vec<Rect>> {
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
