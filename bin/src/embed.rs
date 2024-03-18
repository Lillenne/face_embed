pub(crate) use std::num::NonZeroU32;
use std::time::Duration;

use rayon::prelude::*;
use nokhwa::{Camera, pixel_format::RgbFormat, utils::{RequestedFormat, CameraIndex, CameraFormat, Resolution}};
use sqlx::{QueryBuilder, types::chrono::{Utc, DateTime}};
use tokio::sync::mpsc::Sender;
use crate::cache::SlidingVectorCache;

use crate::*;

type FaceTimeStamp = (Vec<u8>, chrono::DateTime<chrono::Utc>);

pub(crate) async fn detect(args: DetectArgs, v: bool) -> anyhow::Result<()> {
    todo!();
}

pub(crate) async fn embed(args: EmbedArgs, v: bool) -> anyhow::Result<()> {
    let (s_handle, s_task) = setup_signal_handlers()?;
    let pool_task = setup_sqlx(args.database.conn_str.as_str(), args.database.table_name.as_str());

    if v { println!("Generating embedding model..."); }
    let arcface = Box::new(ArcFace::new(args.arcface_path.as_str())?);
    let embedding_dims = arcface.dims();
    if v { println!("Embedding model generated."); }

    let det = create_face_detector(&args.ultraface_path)?;

    let (tx, rx) = mpsc::channel::<>(10000);
    let pool = pool_task.await?;
    let embed_task: JoinHandle<anyhow::Result<()>> = tokio::spawn(async move {
        spawn_embedder_thread(rx, arcface, pool, args.database.table_name, std::time::Duration::from_secs(60 * 60 * 2 /* 2 hr */), args.similarity_threshold, v).await?;
        Ok(())
    });

    if let Some(glob) = args.source.glob {
        embed_files(glob, tx, det, embedding_dims.3, embedding_dims.2, v).await?;
    } else {
        process_feed(args.source.x_res, args.source.y_res, args.source.fps, tx, det, embedding_dims.3, embedding_dims.2, v).await?;
    }
    embed_task.await??;
    s_handle.close();
    s_task.await?;
    Ok(())
}

async fn process_feed(src_x: u32, src_y: u32, src_fps: u32, tx: Sender<FaceTimeStamp>, det: impl FaceDetector, emb_x: NonZeroU32, emb_y: NonZeroU32, v: bool) -> anyhow::Result<()> {
    let (mut cam, x, y) = get_cam(src_x, src_y, src_fps, v)?;
    let mut buffer: Vec<u8> = vec![0; (y.get() * x.get() * 3) as usize];

    if v { println!("Beginning stream..."); }
    while !SHUTDOWN_REQUESTED.load(atomic::Ordering::Acquire) {
        let slice = buffer.as_mut_slice();
        cam.write_frame_to_buffer::<RgbFormat>(slice)?;
        embed_frame(
            &tx,
        slice,
            &det,
            x,
            y,
            emb_x,
            emb_y,
            v).await?;
    }
    Ok(())
}

async fn embed_files(glob: String, tx: Sender<FaceTimeStamp>, det: impl FaceDetector, emb_x: NonZeroU32, emb_y: NonZeroU32, v: bool) -> anyhow::Result<()> {
    for path in glob::glob(glob.as_str())? {
        if SHUTDOWN_REQUESTED.load(atomic::Ordering::Acquire) {
            break;
        }
        match path {
            Ok(file) => {
                if v { println!("Processing file: {}", file.as_os_str().to_string_lossy()) }
                let bind = image::open(file)?.into_rgb8();
                let x = NonZeroU32::new(bind.width()).unwrap();
                let y = NonZeroU32::new(bind.height()).unwrap();
                let mut vec = bind.into_vec();
                embed_frame(
                    &tx,
                vec.as_mut_slice(),
                    &det,
                    x,
                    y,
                    emb_x,
                    emb_y,
                    v).await?;
            },
            Err(e) => {
                if v { println!("Failed to process file: {:?}", e) }
                continue;
            }
        }
    }
    Ok(())
}

async fn embed_frame(
    tx: &tokio::sync::mpsc::Sender<FaceTimeStamp>,
    buffer: &mut [u8],
    det: &impl FaceDetector,
    x: NonZeroU32,
    y: NonZeroU32,
    emb_x: NonZeroU32,
    emb_y: NonZeroU32,
    v: bool) -> anyhow::Result<()> {
    let time = chrono::Utc::now();
    let det_dims = det.dims();
    // resize image for model
    let img = resize(
        &buffer,
        x,
        y,
        det_dims.3,
        det_dims.2,
        fr::ResizeAlg::Convolution(fr::FilterType::Bilinear),
    )?;

    let faces = det.detect(img.buffer())?;
    if faces.len() == 0 {
        return Ok(());
    }
    for face in faces {
        let crop = face.bounding_box.to_crop_box(x.get(), y.get());
        let region =
            fr::Image::from_slice_u8(x, y, buffer, fr::PixelType::U8x3).unwrap();
        let roi = face_embed::crop_and_resize(
            &mut (region.view()),
            emb_x,
            emb_y,
            crop,
            fr::ResizeAlg::Convolution(fr::FilterType::Mitchell),
        )?
        .into_vec();
        if let Err(_) = tx.send((roi, time)).await {
            // Likely shutdown requested or queue full. Return to beginning of loop for shutdown code & drop this frame's outputs
            break;
        }
    }
    Ok(())
}

/// Spin up a thread to generate embeddings from face crops
async fn spawn_embedder_thread(
    mut rx: tokio::sync::mpsc::Receiver<FaceTimeStamp>,
    model: Box<dyn EmbeddingGenerator + Send + Sync>,
    pool: Pool<Postgres>,
    table_name: String,
    cache_duration: Duration,
    similarity_threshold: f32,
    v: bool
) -> anyhow::Result<()> {
    let mut buf: Vec<FaceTimeStamp> = vec!();
    let mut cache = SlidingVectorCache::new(cache_duration, similarity_threshold);
    loop {
        let n_rec = rx.recv_many(&mut buf, 300).await;
        if n_rec == 0 {
            break
        }
        if SHUTDOWN_REQUESTED.load(atomic::Ordering::Relaxed) {
            rx.close();
        }

        let embeddings: Vec<_> = buf.par_iter()
            .map(|(face, time)| {
                let embedding = model.generate_embedding(face.as_slice())?;
                let embedding_vec = pgvector::Vector::from(embedding);
                let item = EmbeddingTime {
                    embedding: embedding_vec,
                    time: time.clone(),
                };
                Ok::<EmbeddingTime, anyhow::Error>(item)
            })
            .flatten()
            .collect();

        let new = embeddings
            .into_iter()
            .filter(|et| cache.push(std::time::Instant::now(), et))
            .collect::<Vec<_>>();
        if new.len() > 0 {
            if v { println!("Detected {} new face(s)", new.len()); }
            // Push
            let query_init = format!("INSERT INTO {} (embedding, time)", table_name);
            let mut qb: QueryBuilder<Postgres> = QueryBuilder::new(query_init);
            qb.push_values(new, |mut b, et| {
                b.push_bind(et.embedding).push_bind(et.time);
            });
            qb.build()
                .execute(&pool)
                .await?;
        }

        buf.clear();
    }
    if v { println!("Embedding calculations complete."); }
    Ok(())
}

fn get_cam(x: u32, y: u32, fps: u32, v: bool) -> anyhow::Result<(Camera, NonZeroU32, NonZeroU32)> {
    if v { println!("Initializing camera..."); }
    let i = CameraIndex::Index(0);
    let req = RequestedFormat::new::<RgbFormat>(nokhwa::utils::RequestedFormatType::Closest(
        CameraFormat::new(
            Resolution {
                width_x: x,
                height_y: y,
            },
            nokhwa::utils::FrameFormat::MJPEG,
            fps,
        ),
    ));
    let mut cam = Camera::new(i, req)?;
    let Resolution {
        width_x: x,
        height_y: y,
    } = cam.resolution();
    let x = NonZeroU32::new(x).ok_or(anyhow::anyhow!("Unable to get proper camera x res"))?;
    let y = NonZeroU32::new(y).ok_or(anyhow::anyhow!("Unable to get proper camera y res"))?;
    cam.open_stream()?;
    if v { println!("Setup camera with resolution ({}, {}) at {} fps.", x.get(), y.get(), cam.frame_rate()); }
    Ok((cam, x, y))
}

#[derive(Debug, Clone)]
struct EmbeddingTime {
    pub embedding: Vector,
    pub time: chrono::DateTime<chrono::Utc>,
}

impl EmbeddingRef<f32> for EmbeddingTime {
    fn embedding_ref(&self) -> &[f32] {
        self.embedding.as_slice()
    }
}
