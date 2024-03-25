pub mod embedding;
pub mod face_detector;
pub mod cache;
pub mod db;
pub mod messaging;
pub mod path_utils;
pub mod image_utils;
pub mod storage;
use fast_image_resize as fr;

pub use ort;
pub use rmp_serde;

#[derive(Clone, Copy, Debug)]
pub struct ZeroToOneF32 {
    value: f32,
}

impl ZeroToOneF32 {
    pub fn new(value: f32) -> Option<ZeroToOneF32> {
        if value >= 0.0 && value <= 1.0 {
            Some(ZeroToOneF32 { value })
        }
        else {
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
        }
        else {
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
            return 0.0
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
        fr::CropBox {left, top, width, height}
    }
}
