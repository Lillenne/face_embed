use std::{io::Cursor, num::NonZeroU32, time::Instant};

use crate::{
    cache::SlidingVectorCache,
    db::{Database, EmbeddingTime, User},
    messaging::{DetectionEvent, LabelEvent, Messenger},
    EmbeddingGenerator, FaceDetector,
};
use anyhow::bail;
use chrono::{DateTime, Utc};
use fast_image_resize as fr;
use fr::{CropBox, DynamicImageView};
use image::{ImageBuffer, Rgb};
use imgref::*;
use pgvector::Vector;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use s3::Bucket;
use tokio::io::AsyncRead;
use uuid::Uuid;

#[derive(Debug)]
pub enum PublisherError {
    IncorrectInputDimensions,
    EmbeddingNotUnique,
    FaceDetectorError(anyhow::Error),
    EmbedderError(anyhow::Error),
    DatabaseError(anyhow::Error),
    MessageBusError(anyhow::Error),
    TODO(anyhow::Error),
}

pub struct LivePublisher {
    generator: Box<dyn EmbeddingGenerator + Send + Sync>,
    generator_tag: String,
    messenger: Messenger,
    database: Database,
    bucket: Bucket,
    cache: SlidingVectorCache<f32, EmbeddingTime>,
    similarity_threshold: f32,
}

impl LivePublisher {
    pub fn new(
        generator: Box<dyn EmbeddingGenerator + Send + Sync>,
        generator_tag: String,
        messenger: Messenger,
        database: Database,
        bucket: Bucket,
        cache: SlidingVectorCache<f32, EmbeddingTime>,
        similarity_threshold: f32,
    ) -> Self {
        Self {
            generator,
            generator_tag,
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
    ) -> Result<(), PublisherError> {
        let embedding = self.embed(data, time)?;

        match self.publish(data, embedding).await {
            Ok(_) => Ok(()),
            Err(e) => Err(PublisherError::TODO(e)),
        }
    }

    pub async fn process_many<'a>(
        &mut self,
        data: &[(ImgRef<'a, [u8; 3]>, DateTime<Utc>)],
    ) -> anyhow::Result<()> {
        let embeds = embed_many(data, &self.generator, &mut self.cache)?;
        let mut filtered_embeds = vec![];
        let mut imgs = vec![];
        for ((img, _), et) in data.iter().zip(embeds.into_iter()) {
            if let Some(et) = et {
                imgs.push(*img);
                filtered_embeds.push(et)
            }
        }
        self.publish_many(imgs.as_slice(), filtered_embeds).await?;
        Ok(())
    }

    pub fn embed(
        &mut self,
        data: ImgRef<[u8; 3]>,
        time: DateTime<Utc>,
    ) -> Result<EmbeddingTime, PublisherError> {
        let item = create_embedding(&self.generator, data, time)?;
        if self.cache.push(Instant::now(), &item) {
            Ok(item)
        } else {
            Err(PublisherError::EmbeddingNotUnique)
        }
    }

    pub fn embed_many(
        &mut self,
        data: &[(ImgRef<'_, [u8; 3]>, DateTime<Utc>)],
    ) -> anyhow::Result<Vec<Option<EmbeddingTime>>> {
        let embeds: Vec<_> = data
            .par_iter()
            .flat_map(|(buf, time)| create_embedding(&self.generator, *buf, *time))
            .collect();
        // intermediate collect since cache needs mutable access
        let filtered = embeds
            .into_iter()
            .map(|item| {
                if self.cache.push(Instant::now(), &item) {
                    Some(item)
                } else {
                    None
                }
            })
            .collect::<Vec<Option<EmbeddingTime>>>();
        Ok(filtered)
    }

    pub async fn publish<'a>(
        &self,
        data: ImgRef<'a, [u8; 3]>,
        embedding: EmbeddingTime,
    ) -> anyhow::Result<()> {
        let path = self.push_jpg(data, &embedding.time).await?;
        let id = self
            .database
            .save_captured_embedding_to_db(&embedding)
            .await?;
        let event = if let Some((user, similarity)) = self
            .database
            .get_label(id, self.similarity_threshold)
            .await?
        {
            DetectionEvent {
                id,
                time: embedding.time,
                path,
                user: Some(user),
                similarity,
            }
        } else {
            DetectionEvent {
                id,
                time: embedding.time,
                path,
                user: None,
                similarity: 0.0,
            }
        };
        let payload = serde_json::to_vec_pretty(&event)?;
        self.messenger.publish(payload.as_slice()).await?;
        Ok(())
    }

    async fn publish_many(
        &self,
        imgs: &[ImgRef<'_, [u8; 3]>],
        data: Vec<EmbeddingTime>,
    ) -> anyhow::Result<()> {
        let paths = self
            .push_jpgs(imgs.iter().zip(data.iter().map(|et| &et.time)))
            .await?;
        let ids = self.database.save_captured_embeddings_to_db(data).await?;
        for ((id, time), path) in ids.into_iter().zip(paths) {
            let user = self
                .database
                .get_label(id, self.similarity_threshold)
                .await?;
            let (user, similarity) = if let Some((user, similarity)) = user {
                (Some(user), similarity)
            } else {
                (None, 0.0)
            };
            let event = DetectionEvent {
                id,
                time,
                path,
                user,
                similarity,
            };
            let payload = serde_json::to_vec_pretty(&event)?;
            self.messenger.publish(payload.as_slice()).await?;
        }
        Ok(())
    }

    async fn push_object<R: AsyncRead + Unpin>(
        &self,
        bytes: &mut R,
        time: &DateTime<Utc>,
    ) -> anyhow::Result<String> {
        let mut guid = String::new();
        guid.push('/');
        guid.push_str(&Uuid::now_v7().to_string());
        let status = &self
            .bucket
            .put_object_stream_with_content_type(bytes, &guid, "image/jpeg")
            .await?;
        if let 200 | 201 = *status {
            bail!("Failed to put object")
        }
        let status = &self
            .bucket
            .put_object_tagging(
                &guid,
                &[("time", &time.to_string()), ("embed", &self.generator_tag)],
            )
            .await?;
        match status.status_code() {
            200 | 201 => Ok(guid),
            _ => bail!("Failed to put metadata"),
        }
    }

    /// Encodes images as JPG and pushes them to storage with new guids & timestamp tags
    async fn push_jpg<'a>(
        &self,
        buf: ImgRef<'a, [u8; 3]>,
        time: &DateTime<Utc>,
    ) -> anyhow::Result<String> {
        let bytes = encode_jpg(Vec::<u8>::new(), buf)?;
        self.push_object(&mut bytes.as_slice(), time).await
    }

    async fn push_jpgs<'a, V: Iterator<Item = (&'a ImgRef<'a, [u8; 3]>, &'a DateTime<Utc>)>>(
        &self,
        data: V,
    ) -> anyhow::Result<Vec<String>> {
        let mut buffer: Vec<u8> = vec![];
        let mut collect: Vec<String> = vec![];
        for item in data {
            buffer = encode_jpg(buffer, *item.0)?; // encode into buffer and return same buf
            collect.push(self.push_object(&mut buffer.as_slice(), item.1).await?);
        }
        Ok(collect)
    }
}

pub struct LabelPublisher {
    detector: Box<dyn FaceDetector + Send + Sync>,
    bucket: Bucket,
    db: Database,
    messenger: Messenger,
    generator: Box<dyn EmbeddingGenerator + Send + Sync>,
    generator_tag: String,
}

impl LabelPublisher {
    pub fn new(
        detector: Box<dyn FaceDetector + Send + Sync>,
        generator: Box<dyn EmbeddingGenerator + Send + Sync>,
        generator_tag: String,
        db: Database,
        bucket: Bucket,
        messenger: Messenger,
    ) -> Self {
        LabelPublisher {
            detector,
            bucket,
            db,
            messenger,
            generator,
            generator_tag,
        }
    }

    pub async fn publish_from_files(&self, data: User, imgs: Vec<Vec<u8>>) -> anyhow::Result<()> {
        let mut collect: Vec<Img<Vec<[u8; 3]>>> = vec![];
        for img in imgs {
            let cursor = Cursor::new(img.as_slice());
            let decoded = image::ImageReader::new(cursor)
                .with_guessed_format()?
                .decode()?
                .into_rgb8();
            let (w, h) = (decoded.width(), decoded.height());
            let buffer = decoded.into_raw();
            let imgref = bytes_as_imgref(buffer.as_slice(), w as _, h as _)?;
            let (_, _, dest_h, dest_w) = self.generator.dims();
            let mut faces = get_embedder_inputs(&self.detector, imgref, dest_w, dest_h)?;
            collect.append(&mut faces);
        }
        let refs: Vec<ImgRef<[u8; 3]>> = collect.iter().map(|item| item.as_ref()).collect();
        self.publish(data, refs.as_slice()).await?;
        Ok(())
    }

    pub async fn publish<'a>(
        &self,
        data: User,
        imgs: &[ImgRef<'a, [u8; 3]>],
        // signatures: Vec<Vector>
    ) -> anyhow::Result<()> {
        let mut buf: Vec<u8> = vec![];
        let mut paths: Vec<String> = vec![];
        for img in imgs {
            let mut guid = String::new();
            guid.push('/');
            guid.push_str(&Uuid::now_v7().to_string());
            buf = encode_jpg(buf, *img)?;
            self.bucket
                .put_object_stream_with_content_type(&mut buf.as_slice(), &guid, "image/jpeg")
                .await?;
            paths.push(guid);
        }

        let mut sigs: Vec<Vector> = vec![];
        for img in imgs {
            let sig = self.generator.generate_embedding(to_bytes(*img))?;
            sigs.push(sig.into());
        }
        let id = self
            .db
            .insert_label(
                data.name.clone(),
                data.email.clone(),
                sigs.as_slice(),
                paths.as_slice(),
                &self.generator_tag,
            )
            .await?;

        let mut signatures: Vec<Vec<f32>> = vec![];
        for s in sigs {
            signatures.push(s.to_vec());
        }
        let event = LabelEvent {
            event_id: id,
            user: User { id, ..data },
            signatures,
        };
        let msg = serde_json::to_vec_pretty(&event)?;
        self.messenger.publish(msg.as_slice()).await?;
        Ok(())
    }
}

pub fn bytes_as_imgref(bytes: &[u8], w: usize, h: usize) -> anyhow::Result<ImgRef<'_, [u8; 3]>> {
    let slice = bytemuck::cast_slice::<u8, [u8; 3]>(bytes);
    Ok(ImgRef::new(slice, w, h))
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

fn create_embedding(
    generator: &Box<dyn EmbeddingGenerator + Sync + Send>,
    buf: ImgRef<[u8; 3]>,
    time: DateTime<Utc>,
) -> Result<EmbeddingTime, PublisherError> {
    // Validate dimensions
    let (_, _, dh, dw) = generator.dims();
    if buf.width() != buf.stride()
        || dw.get() != buf.width() as u32
        || dh.get() != buf.height() as u32
    {
        return Err(PublisherError::IncorrectInputDimensions);
    }
    match generator.generate_embedding(to_bytes(buf)) {
        Ok(embedding) => {
            let embedding = pgvector::Vector::from(embedding);
            Ok(EmbeddingTime { embedding, time })
        }
        Err(e) => Err(PublisherError::EmbedderError(e)),
    }
}

fn embed_many(
    data: &[(ImgRef<'_, [u8; 3]>, DateTime<Utc>)],
    embedder: &Box<dyn EmbeddingGenerator + Sync + Send>,
    cache: &mut SlidingVectorCache<f32, EmbeddingTime>,
) -> anyhow::Result<Vec<Option<EmbeddingTime>>> {
    let embeds: Vec<_> = data
        .par_iter()
        .flat_map(|(buf, time)| create_embedding(embedder, *buf, *time))
        .collect();
    // intermediate collect since cache needs mutable access
    let filtered = embeds
        .into_iter()
        .map(|item| {
            if cache.push(Instant::now(), &item) {
                Some(item)
            } else {
                None
            }
        })
        .collect::<Vec<Option<EmbeddingTime>>>();
    Ok(filtered)
}

pub fn get_embedder_inputs(
    detector: &Box<dyn FaceDetector + Send + Sync>,
    data: ImgRef<[u8; 3]>,
    dest_w: NonZeroU32,
    dest_h: NonZeroU32,
) -> anyhow::Result<Vec<ImgVec<[u8; 3]>>> {
    let (db, _, dh, dw) = detector.dims();
    if db.get() != 1 {
        anyhow::bail!(
            "Face detector implementation requires batch size {}",
            db.get()
        )
    }
    let output = if data.width() == data.stride()
        && dw.get() == data.width() as u32
        && dh.get() == data.height() as u32
    {
        // Already the right size
        let input = to_bytes(data);
        detector.detect(input)
    } else {
        // need to resize
        let cbox = CropBox {
            left: 0.0,
            top: 0.0,
            width: data.width() as _,
            height: data.height() as _,
        };
        let rsz = crop_and_resize(data, cbox, dw, dh)?;
        detector.detect(rsz.as_slice())
    }?;
    let mut collect = vec![];
    for object in output {
        let cbox = object
            .bounding_box
            .to_crop_box(data.width() as _, data.height() as _);
        let embed_input = resize_for_embedding(data, cbox, dest_w, dest_h)?;
        collect.push(embed_input);
    }
    Ok(collect)
}

fn resize_for_embedding(
    data: ImgRef<[u8; 3]>,
    crop: CropBox,
    w: NonZeroU32,
    h: NonZeroU32,
) -> anyhow::Result<ImgVec<[u8; 3]>> {
    let bytes = to_bytes(data);
    let mut src_view = fr::ImageView::from_buffer(
        NonZeroU32::new(data.stride() as _).unwrap(),
        NonZeroU32::new(data.height_padded() as _).unwrap(),
        bytes,
    )?;
    src_view.set_crop_box(crop)?;
    let src_view = DynamicImageView::U8x3(src_view);

    let mut dst_image = fr::Image::new(w, h, fr::PixelType::U8x3);
    let mut resizer = fr::Resizer::new(fr::ResizeAlg::Convolution(fr::FilterType::Bilinear));
    resizer.resize(&src_view, &mut dst_image.view_mut())?;
    let vec = dst_image.into_vec();
    let vec = bytemuck::cast_vec::<u8, [u8; 3]>(vec);
    let img = ImgVec::new(vec, w.get() as _, h.get() as _);
    Ok(img)
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
