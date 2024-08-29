use std::num::NonZeroU32;

use ndarray::{Array4, ArrayBase, ArrayView2, Axis, Dim, ViewRepr};

use crate::{
    embedding::{arcface_dims, ARCFACE_X, ARCFACE_Y},
    face_detector::UltrafaceDetectorConfig,
    DetectedObject, EmbeddingGenerator, FaceDetector, ModelDims, Rect,
};

pub struct ArcFaceOrt {
    model: ort::Session,
}

impl ArcFaceOrt {
    pub fn new(path: &str) -> anyhow::Result<Self> {
        let session = ort::Session::builder()?
            .with_optimization_level(ort::GraphOptimizationLevel::Level1)?
            .with_parallel_execution(false)?
            .with_intra_threads(2)?
            .commit_from_file(path)?;

        Ok(Self { model: session })
    }

    fn preprocess(
        &self,
        face: &[u8],
    ) -> anyhow::Result<ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<[usize; 4]>>> {
        if face.len() != ARCFACE_X * ARCFACE_Y * 3 {
            return Err(anyhow::anyhow!("Incorrect image dimensions"));
        }

        let input = ndarray::Array4::from_shape_fn((1, 3, ARCFACE_Y, ARCFACE_X), |(_, c, y, x)| {
            let idx = (y * ARCFACE_X + x) * 3 + c;
            face[idx] as f32
        });

        Ok(input)
    }
}

impl ModelDims for ArcFaceOrt {
    fn dims(
        &self,
    ) -> (
        std::num::NonZeroU32,
        std::num::NonZeroU32,
        std::num::NonZeroU32,
        std::num::NonZeroU32,
    ) {
        arcface_dims()
    }
}

impl EmbeddingGenerator for ArcFaceOrt {
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
            *v /= magnitude;
        }
        Ok(output)
    }
}

pub struct UltrafaceDetectorOrt {
    model: ort::Session,
    cfg: UltrafaceDetectorConfig,
}

impl UltrafaceDetectorOrt {
    const ULTRAFACE_MEAN: f32 = 127.0;
    const ULTRAFACE_DIV: f32 = 128.0;
    const ULTRAFACE_N_BOXES: usize = 4420;
    const ULTRAFACE_FG_IDX: usize = 1;

    pub fn new(cfg: UltrafaceDetectorConfig, path: &str) -> anyhow::Result<Self> {
        if !std::path::Path::new(path).exists() {
            return Err(anyhow::anyhow!("Model path does not exist"));
        }

        let model = ort::Session::builder()?.commit_from_file(path)?;
        Ok(Self { cfg, model })
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
                (data[idx] as f32 - Self::ULTRAFACE_MEAN) / Self::ULTRAFACE_DIV
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
            .column(Self::ULTRAFACE_FG_IDX)
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
                    if *v.1 < probs[[i, Self::ULTRAFACE_FG_IDX]] {
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
                class: Self::ULTRAFACE_FG_IDX,
                confidence: *prob.1,
                bounding_box: self.get_rect(boxes.row(prob.0)),
            };
            output.push(bbox);
        }
        output
    }
}

impl ModelDims for UltrafaceDetectorOrt {
    fn dims(&self) -> (NonZeroU32, NonZeroU32, NonZeroU32, NonZeroU32) {
        (
            NonZeroU32::new(1).unwrap(),
            NonZeroU32::new(3).unwrap(),
            self.cfg.model_height,
            self.cfg.model_width,
        )
    }
}

impl FaceDetector for UltrafaceDetectorOrt {
    fn detect(&self, frame: &[u8]) -> anyhow::Result<Vec<DetectedObject>> {
        let data = self.preprocess(frame)?;
        let result = self.model.run(ort::inputs!("input" => data)?)?;

        let btensor = result["boxes"].try_extract_tensor::<f32>()?;
        let bview = btensor.view();
        let boxes = bview.to_shape((Self::ULTRAFACE_N_BOXES, 4))?;

        let probsa = result["scores"].try_extract_tensor::<f32>()?;
        let probsb = probsa.view();
        let probs = probsb.to_shape((Self::ULTRAFACE_N_BOXES, 2))?;

        Ok(self.nms(&probs.view(), &boxes.view()))
    }
}
