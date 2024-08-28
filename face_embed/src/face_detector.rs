use crate::*;
use std::num::NonZeroU32;

use ndarray::{prelude::*, Array4, ArrayView2, Dim, ViewRepr};
use ort::Session;

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
            prob_threshold: ZeroToOneF32::new(0.85).unwrap(),
            top_k: NonZeroU32::new(20).unwrap(),
            model_width: NonZeroU32::new(320).unwrap(),
            model_height: NonZeroU32::new(240).unwrap(),
        }
    }
}

pub struct UltrafaceDetector {
    model: ort::Session,
    cfg: UltrafaceDetectorConfig,
}

impl UltrafaceDetector {
    const ULTRAFACE_MEAN: f32 = 127.0;
    const ULTRAFACE_DIV: f32 = 128.0;
    const ULTRAFACE_N_BOXES: usize = 4420;
    const ULTRAFACE_FG_IDX: usize = 1;

    pub fn new(cfg: UltrafaceDetectorConfig, path: &str) -> anyhow::Result<UltrafaceDetector> {
        if !std::path::Path::new(path).exists() {
            return Err(anyhow::anyhow!("Model path does not exist"));
        }

        let model = Session::builder()?.commit_from_file(path)?;
        Ok(UltrafaceDetector { cfg, model })
    }

    fn preprocess(
        &self,
        data: &[u8],
    ) -> anyhow::Result<ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<[usize; 4]>>> {
        if self.n_elements() != data.len() {
            return Err(anyhow::anyhow!("Incorrect data shape"));
        }
        let input = Array4::from_shape_fn(
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
        );
        Ok(input)
    }

    /// Convert Ultraface bounding box outputs (x1, y1, x2, y2) to Rect
    fn get_rect(&self, view: ArrayBase<ViewRepr<&f32>, Dim<[usize; 1]>>) -> Rect {
        let left = view[0].clamp(0.0, 1.0);
        let top = view[1].clamp(0.0, 1.0);
        let width = (view[2] - left).clamp(0.0, 1.0 - left);
        let height = (view[3] - top).clamp(0.0, 1.0 - top);
        Rect {
            left,
            top,
            width,
            height,
        }
    }

    fn n_elements(&self) -> usize {
        self.cfg.model_width.get() as usize * self.cfg.model_height.get() as usize * 3
    }

    fn nms(&self, probs: &ArrayView2<f32>, boxes: &ArrayView2<f32>) -> Vec<DetectedObject> {
        let iter = probs
            .column(UltrafaceDetector::ULTRAFACE_FG_IDX)
            .into_iter()
            .enumerate()
            .filter(move |v| {
                if *v.1 < self.cfg.prob_threshold.value {
                    // probability too low
                    return false;
                }

                // perform nms
                let this_box = self.get_rect(boxes.row(v.0));

                for (i, bbox) in boxes.axis_iter(Axis(0)).enumerate() {
                    let rect = self.get_rect(bbox);
                    let iou = this_box.iou(&rect);
                    if iou < self.cfg.iou_threshold.value {
                        // insufficient intersection to suppress
                        continue;
                    }
                    if *v.1 < probs[[i, UltrafaceDetector::ULTRAFACE_FG_IDX]] {
                        // overlapping box with higher probability exists
                        return false;
                    }
                }
                true
            });

        let mut vec = iter.collect::<Vec<_>>();
        vec.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
        let mut output: Vec<DetectedObject> = vec![];
        for prob in vec.into_iter().take(self.cfg.top_k.get() as _) {
            let bbox = DetectedObject {
                class: UltrafaceDetector::ULTRAFACE_FG_IDX,
                confidence: *prob.1,
                bounding_box: self.get_rect(boxes.row(prob.0)),
            };
            output.push(bbox);
        }
        output
    }
}

impl ModelDims for UltrafaceDetector {
    fn dims(&self) -> (NonZeroU32, NonZeroU32, NonZeroU32, NonZeroU32) {
        (
            NonZeroU32::new(1).unwrap(),
            NonZeroU32::new(3).unwrap(),
            self.cfg.model_height,
            self.cfg.model_width,
        )
    }
}

impl FaceDetector for UltrafaceDetector {
    fn detect(&self, frame: &[u8]) -> anyhow::Result<Vec<DetectedObject>> {
        let data = self.preprocess(frame)?;
        let result = self.model.run(ort::inputs!("input" => data)?)?;

        let btensor = result["boxes"].try_extract_tensor::<f32>()?;
        let bview = btensor.view();
        let boxes = bview.to_shape((UltrafaceDetector::ULTRAFACE_N_BOXES, 4))?;

        let probsa = result["scores"].try_extract_tensor::<f32>()?;
        let probsb = probsa.view();
        let probs = probsb.to_shape((UltrafaceDetector::ULTRAFACE_N_BOXES, 2))?;

        Ok(self.nms(&probs.view(), &boxes.view()))
    }
}
