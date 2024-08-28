pub mod cache;
pub mod db;
#[cfg(feature = "ort")]
pub mod embedding;
#[cfg(feature = "ort")]
pub mod face_detector;
pub mod image_utils;
pub mod messaging;
pub mod path_utils;
pub mod pipeline;
pub mod storage;
use std::num::NonZeroU32;

use fast_image_resize as fr;

#[cfg(feature = "ort")]
pub use ort;

pub trait ModelDims {
    /// Returns the model dimensions (b,c,h,w)
    fn dims(&self) -> (NonZeroU32, NonZeroU32, NonZeroU32, NonZeroU32);
}

/// Defines a face detection algorithm
pub trait FaceDetector: ModelDims {
    /// Detects faces in an RGB image.
    fn detect(&self, frame: &[u8]) -> anyhow::Result<Vec<DetectedObject>>;
}

/// Defines an embedding generating algorithm
pub trait EmbeddingGenerator: ModelDims {
    /// Generates embeddings from an image
    fn generate_embedding(&self, face: &[u8]) -> anyhow::Result<Vec<f32>>;
}

#[derive(Debug, Clone, Copy)]
pub struct DetectedObject {
    /// The index of the output class with the highest probability
    pub class: usize,
    /// The predicted likelihood
    pub confidence: f32,
    pub bounding_box: Rect,
}

pub fn similarity<T: num_traits::Float>(embedding: &[T], nearest_embed: &[T]) -> T {
    embedding
        .iter()
        .zip(nearest_embed.iter())
        .map(|(a, b)| *a * *b)
        .reduce(|a, b| a + b)
        .unwrap()
}

#[derive(Clone, Copy, Debug)]
pub struct ZeroToOneF32 {
    value: f32,
}

impl ZeroToOneF32 {
    pub fn new(value: f32) -> Option<ZeroToOneF32> {
        if (0.0..=1.0).contains(&value) {
            Some(ZeroToOneF32 { value })
        } else {
            None
        }
    }
}

impl From<ZeroToOneF32> for f32 {
    fn from(source: ZeroToOneF32) -> Self {
        source.value
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Rect {
    pub left: f32,
    pub top: f32,
    pub width: f32,
    pub height: f32,
}

impl Rect {
    pub fn right(&self) -> f32 {
        self.left + self.width
    }

    pub fn bottom(&self) -> f32 {
        self.top + self.height
    }

    pub fn area(&self) -> f32 {
        self.width * self.height
    }

    pub fn iou(&self, other: &Self) -> f32 {
        let overlap = self.intersection(other);
        if overlap == 0.0 {
            overlap
        } else {
            overlap / self.union(other)
        }
    }

    pub fn union(&self, other: &Self) -> f32 {
        self.area() + other.area() - self.intersection(other)
    }

    pub fn intersection(&self, other: &Self) -> f32 {
        if other.left >= self.right()
            || self.left >= other.right()
            || self.bottom() <= other.top
            || other.bottom() <= self.top
        {
            return 0.0;
        }
        let intersection_width = self.right().min(other.right()) - self.left.max(other.left);
        let intersection_height = self.bottom().min(other.bottom()) - self.top.max(other.top);
        intersection_width * intersection_height
    }

    pub fn to_crop_box(&self, w: u32, h: u32) -> fr::CropBox {
        let left = (self.left * w as f32) as f64;
        let top = (self.top * h as f32) as f64;
        let width = (self.width * w as f32).clamp(1.0, (w as f64 - left) as f32) as f64;
        let height = (self.height * h as f32).clamp(1.0, (h as f64 - top) as f32) as f64;
        fr::CropBox {
            left,
            top,
            width,
            height,
        }
    }
}
