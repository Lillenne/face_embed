use crate::*;
use std::num::NonZeroU32;

#[cfg(feature = "tract")]
pub use crate::tract_backend::UltrafaceDetectorTract as UltrafaceDetector;

#[cfg(feature = "ort")]
pub use crate::ort_backend::UltrafaceDetectorOrt as UltrafaceDetector;

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

pub const ULTRAFACE_MEAN: f32 = 127.0;
pub const ULTRAFACE_DIV: f32 = 128.0;
pub const ULTRAFACE_N_BOXES: usize = 4420;
pub const ULTRAFACE_FG_IDX: usize = 1;
