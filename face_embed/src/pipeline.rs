use std::num::NonZeroU32;

use crate::{embedding::EmbeddingGenerator, face_detector::FaceDetector};
use fast_image_resize as fr;
use fr::{CropBox, DynamicImageView};
use imgref::*;

pub struct Pipeline<T: FaceDetector, U: EmbeddingGenerator> {
    detector: T,
    embedder: U,
}

impl<T: FaceDetector, U: EmbeddingGenerator> Pipeline<T, U> {
    pub fn new(detector: T, embedder: U) -> Self {
        Self { detector, embedder }
    }

    pub fn process(&self, data: ImgRef<[u8; 3]>) -> anyhow::Result<Vec<Vec<f32>>> {
        let det_input = self.resize_for_detection(data)?;
        let output = self.detector.detect(det_input)?;
        let mut embeds: Vec<Vec<f32>> = Vec::with_capacity(output.len());
        for object in output {
            let cbox = object
                .bounding_box
                .to_crop_box(data.width() as _, data.height() as _);
            let embed_input = self.resize_for_embedding(data, cbox)?;
            let embedding = self.embedder.generate_embedding(embed_input.as_slice())?;
            embeds.push(embedding);
        }
        Ok(embeds)
    }

    fn to_bytes(data: ImgRef<[u8; 3]>) -> &[u8] {
        bytemuck::cast_slice::<[u8; 3], u8>(data.buf())
    }

    fn resize_for_detection(&self, data: ImgRef<[u8; 3]>) -> anyhow::Result<&[u8]> {
        let (_, _, dh, dw) = self.detector.dims();
        if data.width() == data.stride()
            && dw.get() == data.width() as u32
            && dh.get() == data.height() as u32
        {
            // Already the right size
            Ok(Pipeline::<T, U>::to_bytes(data))
        } else {
            let cbox = CropBox {
                left: 0.0,
                top: 0.0,
                width: data.width() as _,
                height: data.height() as _,
            };
            let (_, _, dh, dw) = self.detector.dims();
            let rsz = Pipeline::<T, U>::crop_and_resize(data, cbox, dw, dh)?;
            Ok(rsz.as_slice())
        }
    }

    fn crop_and_resize(
        data: ImgRef<[u8; 3]>,
        crop: CropBox,
        dw: NonZeroU32,
        dh: NonZeroU32,
    ) -> anyhow::Result<Vec<u8>> {
        let bytes = Pipeline::<T, U>::to_bytes(data);
        let mut src_view = fr::ImageView::from_buffer(
            NonZeroU32::new(data.stride() as _).unwrap(),
            NonZeroU32::new(data.height_padded() as _).unwrap(),
            bytes,
        )?;
        src_view.set_crop_box(crop);
        let src_view = DynamicImageView::U8x3(src_view);

        let mut dst_image = fr::Image::new(dw, dh, fr::PixelType::U8x3);
        let mut resizer = fr::Resizer::new(fr::ResizeAlg::Convolution(fr::FilterType::Bilinear));
        resizer.resize(&src_view, &mut dst_image.view_mut())?;
        Ok(dst_image.into_vec())
    }

    fn resize_for_embedding(
        &self,
        data: ImgRef<[u8; 3]>,
        crop: CropBox,
    ) -> anyhow::Result<Vec<u8>> {
        let (_, _, dh, dw) = self.embedder.dims();
        let bytes = Pipeline::<T, U>::to_bytes(data);
        let mut src_view = fr::ImageView::from_buffer(
            NonZeroU32::new(data.stride() as _).unwrap(),
            NonZeroU32::new(data.height_padded() as _).unwrap(),
            bytes,
        )?;
        src_view.set_crop_box(crop);
        let src_view = DynamicImageView::U8x3(src_view);

        let mut dst_image = fr::Image::new(dw, dh, fr::PixelType::U8x3);
        let mut resizer = fr::Resizer::new(fr::ResizeAlg::Convolution(fr::FilterType::Bilinear));
        resizer.resize(&src_view, &mut dst_image.view_mut())?;
        Ok(dst_image.into_vec())
    }
}
