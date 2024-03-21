pub(crate) use std::num::NonZeroU32;
use std::time::Duration;

use face_embed::{path_utils::path_parser, db::{setup_sqlx, EmbeddingTime, insert_captures}, messaging::create_queue, image_utils::{resize, crop_and_resize}};
use lapin::{options::BasicPublishOptions, Channel, BasicProperties};
use rayon::prelude::*;
use nokhwa::{Camera, pixel_format::RgbFormat, utils::{RequestedFormat, CameraIndex, CameraFormat, Resolution}};
use sqlx::types::chrono::{DateTime, Utc};
use tokio::sync::mpsc::Sender;
use fast_image_resize as fr;
use crate::cache::SlidingVectorCache;

use crate::*;

type FaceTimeStamp = (Vec<u8>, chrono::DateTime<chrono::Utc>);

#[derive(Args, Debug)]
pub(crate) struct EmbedArgs {
    /// The path to the ultraface detection model.
    #[arg(short, long, default_value = ULTRAFACE_PATH, value_parser = path_parser)]
    ultraface_path: String,

    /// The path to the arface embedding model.
    #[arg(short, long, default_value = ARCFACE_PATH, value_parser = path_parser)]
    arcface_path: String,

    #[arg(short, long, default_value_t = 0.5)]
    similarity_threshold: f32,

    #[command(flatten)]
    source: Source,

    #[command(flatten)]
    database: DatabaseArgs,

    #[command(flatten)]
    message_bus: MessageBusArgs,

    #[arg(long, default_value_t = 60 * 60 * 2 /* 2 hr */)]
    cache_duration_seconds: u64,

    #[arg(long, default_value_t = 1000)]
    channel_bound: usize,

    #[arg(short, long)]
    verbose: bool
}

pub(crate) async fn embed(args: EmbedArgs) -> anyhow::Result<()> {
    let v = args.verbose;
    let (s_handle, s_task) = setup_signal_handlers()?;
    let pool_task = setup_sqlx(args.database.conn_str.as_str(), 5, args.database.table_name.as_str());
    let ch_task = create_queue(&args.message_bus.address, &args.message_bus.queue_name);

    if v { println!("Generating embedding model..."); }
    let arcface = Box::new(ArcFace::new(args.arcface_path.as_str())?);
    let embedding_dims = arcface.dims();
    if v { println!("Embedding model generated."); }

    println!("Generating face detection model...");
    let uf_cfg = UltrafaceDetectorConfig {
        top_k: NonZeroU32::new(3).unwrap(),
        ..Default::default()
    };
    let det = UltrafaceDetector::new(uf_cfg, &args.ultraface_path)?;
    println!("Face detection model generated.");

    let (tx, rx) = mpsc::channel(args.channel_bound);
    let pool = pool_task.await?;
    let ch = ch_task.await?;
    let embed_task: JoinHandle<anyhow::Result<()>> = tokio::spawn(async move {
        spawn_embedder_thread(rx, arcface, pool, args.database.table_name, std::time::Duration::from_secs(args.cache_duration_seconds), args.similarity_threshold, ch, &args.message_bus.queue_name, v).await?;
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
            emb_y).await?;
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
                    emb_y).await?;
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
    emb_y: NonZeroU32) -> anyhow::Result<()> {
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
        let roi = crop_and_resize(
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
    channel: Channel,
    route: &str,
    v: bool
) -> anyhow::Result<()> {
    let mut buf: Vec<FaceTimeStamp> = vec!();
    let mut cache = SlidingVectorCache::new(cache_duration, similarity_threshold);
    loop {
        let n_rec = rx.recv_many(&mut buf, 40).await;
        if n_rec == 0 || SHUTDOWN_REQUESTED.load(atomic::Ordering::Relaxed) {
            break
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
        buf.clear();

        let new = embeddings
            .into_iter()
            .filter(|et| cache.push(std::time::Instant::now(), et))
            .collect::<Vec<_>>();
        if new.len() == 0 { continue }
        if v { println!("Detected {} new face(s)", new.len()); }

        // TODO outbox pattern or similar
        let times: Vec<DateTime<Utc>> = new.iter().map(|et| et.time.clone()).collect();
        let ids = insert_captures(new, &pool, &table_name).await?;

        for (time, id) in times.into_iter().zip(ids) {
            let payload = format!("{{ id: {}, time: {} }}", id, time);
            _ = channel
                .basic_publish(
                    "",
                    route,
                    BasicPublishOptions::default(),
                    payload.as_bytes(),
                    BasicProperties::default(),
                )
                .await?
                .await?;
        }
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
