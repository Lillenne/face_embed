use std::{io::Cursor, num::NonZeroU32, time::Instant};

use crate::{
    cache::SlidingVectorCache,
    db::{Database, EmbeddingTime},
    embedding::EmbeddingGenerator,
    face_detector::FaceDetector,
    messaging::{Event, Messenger},
};
use chrono::{DateTime, Utc};
use fast_image_resize as fr;
use fr::{CropBox, DynamicImageView};
use image::{ImageBuffer, Rgb};
use imgref::*;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use s3::Bucket;
use tokio::io::AsyncRead;
use uuid::Uuid;

pub struct Detector<T: FaceDetector> {
    detector: T,
    embedder_width: NonZeroU32,
    embedder_height: NonZeroU32,
}

impl<T: FaceDetector> Detector<T> {
    pub fn new(detector: T, embedder_width: NonZeroU32, embedder_height: NonZeroU32) -> Self {
        Self {
            detector,
            embedder_width,
            embedder_height,
        }
    }

    pub fn process(&mut self, data: ImgRef<[u8; 3]>) -> anyhow::Result<Vec<ImgVec<[u8; 3]>>> {
        let (_, _, dh, dw) = self.detector.dims();
        let output = if data.width() == data.stride()
            && dw.get() == data.width() as u32
            && dh.get() == data.height() as u32
        {
            // Already the right size
            let input = to_bytes(data);
            self.detector.detect(input)
        } else {
            // need to resize
            let cbox = CropBox {
                left: 0.0,
                top: 0.0,
                width: data.width() as _,
                height: data.height() as _,
            };
            let rsz = crop_and_resize(data, cbox, dw, dh)?;
            self.detector.detect(rsz.as_slice())
        }?;
        let mut collect = vec![];
        for object in output {
            let cbox = object
                .bounding_box
                .to_crop_box(data.width() as _, data.height() as _);
            let embed_input = self.resize_for_embedding(data, cbox)?;
            collect.push(embed_input);
        }
        Ok(collect)
    }

    fn resize_for_embedding(
        &self,
        data: ImgRef<[u8; 3]>,
        crop: CropBox,
    ) -> anyhow::Result<ImgVec<[u8; 3]>> {
        let bytes = to_bytes(data);
        let mut src_view = fr::ImageView::from_buffer(
            NonZeroU32::new(data.stride() as _).unwrap(),
            NonZeroU32::new(data.height_padded() as _).unwrap(),
            bytes,
        )?;
        src_view.set_crop_box(crop)?;
        let src_view = DynamicImageView::U8x3(src_view);

        let mut dst_image = fr::Image::new(
            self.embedder_width,
            self.embedder_height,
            fr::PixelType::U8x3,
        );
        let mut resizer = fr::Resizer::new(fr::ResizeAlg::Convolution(fr::FilterType::Bilinear));
        resizer.resize(&src_view, &mut dst_image.view_mut())?;
        let vec = dst_image.into_vec();
        let vec = bytemuck::cast_vec::<u8, [u8; 3]>(vec);
        let img = ImgVec::new(
            vec,
            self.embedder_width.get() as _,
            self.embedder_height.get() as _,
        );
        Ok(img)
    }
}

pub struct Publisher<T: EmbeddingGenerator + Sync> {
    generator: T,
    messenger: Messenger,
    database: Database,
    bucket: Bucket,
    cache: Option<SlidingVectorCache<f32, EmbeddingTime>>,
    similarity_threshold: f32,
}

impl<T: EmbeddingGenerator + Sync> Publisher<T> {
    pub fn new(
        generator: T,
        messenger: Messenger,
        database: Database,
        bucket: Bucket,
        cache: Option<SlidingVectorCache<f32, EmbeddingTime>>,
        similarity_threshold: f32,
    ) -> Self {
        Self {
            generator,
            messenger,
            database,
            bucket,
            cache,
            similarity_threshold,
        }
    }

    pub async fn process<'a>(
        &mut self,
        data: ImgRef<'a, [u8; 3]>,
        time: DateTime<Utc>,
    ) -> anyhow::Result<()> {
        let embedding = self.embed(data, time)?;
        self.publish(data, embedding).await?;
        Ok(())
    }

    pub async fn process_many<'a>(
        &mut self,
        data: Vec<(ImgRef<'a, [u8; 3]>, DateTime<Utc>)>,
    ) -> anyhow::Result<()> {
        let embeds = self.embed_many(&data)?;
        let mut imgs = vec![];
        for (img, _) in data {
            imgs.push(img);
        }
        self.publish_many(imgs.as_slice(), embeds).await?;
        Ok(())
    }

    pub fn embed(
        &mut self,
        data: ImgRef<[u8; 3]>,
        time: DateTime<Utc>,
    ) -> anyhow::Result<EmbeddingTime> {
        let item = self.create_embedding(data, time)?;
        if let Some(ref mut cache) = self.cache {
            if cache.push(Instant::now(), &item) {
                Ok(item)
            } else {
                Err(anyhow::anyhow!("Embedding not unique!"))
            }
        } else {
            Ok(item)
        }
    }

    pub fn embed_many(
        &mut self,
        data: &Vec<(ImgRef<'_, [u8; 3]>, DateTime<Utc>)>,
    ) -> anyhow::Result<Vec<EmbeddingTime>> {
        let embeds: Vec<_> = data
            .par_iter()
            .flat_map(|(buf, time)| self.create_embedding(*buf, *time))
            .collect();
        // intermediate collect since cache needs mutable access
        let filtered = embeds
            .into_iter()
            .flat_map(|item| {
                if let Some(ref mut cache) = self.cache {
                    if cache.push(Instant::now(), &item) {
                        Some(item)
                    } else {
                        None
                    }
                } else {
                    Some(item)
                }
            })
            .collect::<Vec<EmbeddingTime>>();
        Ok(filtered)
    }

    pub async fn publish<'a>(
        &self,
        data: ImgRef<'a, [u8; 3]>,
        embedding: EmbeddingTime,
    ) -> anyhow::Result<()> {
        let path = self.push_jpg(data, &embedding.time).await?;
        // TODO single query
        let id = self
            .database
            .save_captured_embedding_to_db(&embedding)
            .await?;
        let user = self
            .database
            .get_label(id, self.similarity_threshold)
            .await?;
        let event = Event {
            id,
            time: embedding.time,
            path,
            user,
        };
        let payload = rmp_serde::to_vec_named(&event)?;
        self.messenger.publish(payload.as_slice()).await?;
        Ok(())
    }

    pub async fn publish_many<'a>(
        &self,
        imgs: &[ImgRef<'a, [u8; 3]>],
        data: Vec<EmbeddingTime>,
    ) -> anyhow::Result<()> {
        let paths = self
            .push_jpgs(imgs.iter().zip(data.iter().map(|et| &et.time)))
            .await?;
        // TODO single query
        let ids = self.database.save_captured_embeddings_to_db(data).await?;
        for ((id, time), path) in ids.into_iter().zip(paths) {
            let user = self
                .database
                .get_label(id, self.similarity_threshold)
                .await?;
            let event = Event {
                id,
                time,
                path,
                user,
            };
            let payload = rmp_serde::to_vec_named(&event)?;
            self.messenger.publish(payload.as_slice()).await?;
        }
        Ok(())
    }

    fn create_embedding(
        &self,
        buf: ImgRef<[u8; 3]>,
        time: DateTime<Utc>,
    ) -> anyhow::Result<EmbeddingTime> {
        // Validate dimensions
        let (_, _, dh, dw) = self.generator.dims();
        if buf.width() != buf.stride()
            || dw.get() != buf.width() as u32
            || dh.get() != buf.height() as u32
        {
            return Err(anyhow::anyhow!("Incorrect input dimensions!"));
        }
        let embedding = self.generator.generate_embedding(to_bytes(buf))?;
        let embedding = pgvector::Vector::from(embedding);
        Ok(EmbeddingTime { embedding, time })
    }

    /// Encodes images as JPG and pushes them to storage with new guids & timestamp tags
    async fn push_jpg<'a>(
        &self,
        buf: ImgRef<'a, [u8; 3]>,
        time: &DateTime<Utc>,
    ) -> anyhow::Result<String> {
        let bytes = Self::encode_jpg(Vec::<u8>::new(), buf)?;
        self.push_object(&mut bytes.as_slice(), time).await
    }

    async fn push_jpgs<'a, V: Iterator<Item = (&'a ImgRef<'a, [u8; 3]>, &'a DateTime<Utc>)>>(
        &self,
        data: V,
    ) -> anyhow::Result<Vec<String>> {
        let mut buffer: Vec<u8> = vec![];
        let mut collect: Vec<String> = vec![];
        for item in data {
            buffer = Self::encode_jpg(buffer, *item.0)?; // encode into buffer and return same buf
            collect.push(self.push_object(&mut buffer.as_slice(), item.1).await?);
        }
        Ok(collect)
    }

    fn encode_jpg(mut bytes: Vec<u8>, buf: ImgRef<[u8; 3]>) -> anyhow::Result<Vec<u8>> {
        let slice = buf.to_contiguous_buf(); // Copy shouldn't happen
        let b = slice.0.into_owned();
        let img = ImageBuffer::<Rgb<u8>, &[u8]>::from_raw(
            slice.1 as _,
            slice.2 as _,
            bytemuck::cast_slice::<[u8; 3], u8>(b.as_slice()),
        )
        .unwrap();
        let mut cursor = Cursor::new(&mut bytes);
        img.write_to(&mut cursor, image::ImageFormat::Jpeg)?;
        Ok(bytes)
    }

    async fn push_object<R: AsyncRead + Unpin>(
        &self,
        bytes: &mut R,
        time: &DateTime<Utc>,
    ) -> anyhow::Result<String> {
        let mut guid = String::new();
        guid.push('/');
        guid.push_str(&Uuid::now_v7().to_string());
        self.bucket
            .put_object_stream_with_content_type(bytes, &guid, "image/jpeg")
            .await?;
        self.bucket
            .put_object_tagging(&guid, &[("time", &time.to_string())])
            .await?;
        Ok(guid)
    }
}

pub struct Pipeline<T, U>
where
    T: FaceDetector,
    U: EmbeddingGenerator + Sync,
{
    detector: Detector<T>,
    publisher: Publisher<U>,
}

impl<T, U> Pipeline<T, U>
where
    T: FaceDetector,
    U: EmbeddingGenerator + Sync,
{
    pub fn new(detector: Detector<T>, publisher: Publisher<U>) -> Self {
        Self {
            detector,
            publisher,
        }
    }

    pub async fn process(
        &mut self,
        image: ImgRef<'_, [u8; 3]>,
        time: DateTime<Utc>,
    ) -> anyhow::Result<()> {
        let data = self.detector.process(image)?;
        for img in data {
            self.publisher.process(img.as_ref(), time).await?
        }
        Ok(())
    }
}

fn crop_and_resize(
    data: ImgRef<[u8; 3]>,
    crop: CropBox,
    dw: NonZeroU32,
    dh: NonZeroU32,
) -> anyhow::Result<Vec<u8>> {
    let bytes = to_bytes(data);
    let mut src_view = fr::ImageView::from_buffer(
        NonZeroU32::new(data.stride() as _).unwrap(),
        NonZeroU32::new(data.height_padded() as _).unwrap(),
        bytes,
    )?;
    src_view.set_crop_box(crop)?;
    let src_view = DynamicImageView::U8x3(src_view);

    let mut dst_image = fr::Image::new(dw, dh, fr::PixelType::U8x3);
    let mut resizer = fr::Resizer::new(fr::ResizeAlg::Convolution(fr::FilterType::Bilinear));
    resizer.resize(&src_view, &mut dst_image.view_mut())?;
    Ok(dst_image.into_vec())
}

fn to_bytes(data: ImgRef<[u8; 3]>) -> &[u8] {
    bytemuck::cast_slice::<[u8; 3], u8>(data.buf())
}
