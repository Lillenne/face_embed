pub(crate) use std::num::NonZeroU32;

use nokhwa::{Camera, pixel_format::RgbFormat, utils::{RequestedFormat, CameraIndex, CameraFormat, Resolution}};
use tokio::sync::mpsc::Sender;

use crate::*;

pub(crate) async fn detect(args: DetectArgs, v: bool) -> anyhow::Result<()> {
    Ok(())
}

pub(crate) async fn embed(args: EmbedArgs, v: bool) -> anyhow::Result<()> {
    let (s_handle, s_task) = setup_signal_handlers()?;
    let pool_task = setup_sqlx(args.database.conn_str.as_str(), args.database.table_name.as_str());

    if v { println!("Generating embedding model..."); }
    let arcface = Box::new(ArcFace::new(args.arcface_path.as_str())?);
    let embedding_dims = arcface.dims();
    if v { println!("Embedding model generated."); }

    let det = create_face_detector(&args.ultraface_path)?;

    let (tx, rx) = mpsc::channel::<Vec<u8>>(10000);
    let pool = pool_task.await?;
    let embed_task: JoinHandle<anyhow::Result<()>> = tokio::spawn(async move {
        spawn_embedder_thread(rx, arcface, pool, v).await?;
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

async fn process_feed(src_x: u32, src_y: u32, src_fps: u32, tx: Sender<Vec<u8>>, det: impl FaceDetector, emb_x: NonZeroU32, emb_y: NonZeroU32, v: bool) -> anyhow::Result<()> {
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

async fn embed_files(glob: String, tx: Sender<Vec<u8>>, det: impl FaceDetector, emb_x: NonZeroU32, emb_y: NonZeroU32, v: bool) -> anyhow::Result<()> {
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
    tx: &tokio::sync::mpsc::Sender<Vec<u8>>,
    buffer: &mut [u8],
    det: &impl FaceDetector,
    x: NonZeroU32,
    y: NonZeroU32,
    emb_x: NonZeroU32,
    emb_y: NonZeroU32,
    v: bool) -> anyhow::Result<()> {
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
    if v { println!("Detected {} face(s)", faces.len())}
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
        if let Err(_) = tx.send(roi).await {
            // Likely shutdown requested or queue full. Return to beginning of loop for shutdown code & drop this frame's outputs
            break;
        }
    }
    Ok(())
}

/// Spin up a thread to generate embeddings from face crops
async fn spawn_embedder_thread(
    mut rx: tokio::sync::mpsc::Receiver<Vec<u8>>,
    model: Box<dyn EmbeddingGenerator + Send>,
    pool: Pool<Postgres>,
    v: bool
) -> anyhow::Result<()> {
    while let Some(face) = rx.recv().await {
        if SHUTDOWN_REQUESTED.load(atomic::Ordering::Relaxed) {
            break;
        }
        let t = std::time::Instant::now();
        let embedding = model.generate_embedding(face.as_slice())?;
        let embedding_vec = pgvector::Vector::from(embedding.clone()); // copy to send to postgres
        if v { println!("Calculated embedding in {} ms", t.elapsed().as_millis()); }

        // Get nearest neighbor from DB and calculate similarity
        let nearest = sqlx::query("SELECT * FROM items ORDER BY embedding <-> $1 LIMIT 1")
            .bind(embedding_vec)
            .fetch_all(&pool)
            .await?;
        if let Some(n) = nearest.get(0) {
            let nearest_embed: Vector = n.try_get("embedding")?;
            let similarity = similarity(embedding.as_slice(), nearest_embed.as_slice());
            if similarity > 0.3 {
                // TODO publish event, get id
                println!("Similarity: {}", similarity);
            }
        }

        let embedding = pgvector::Vector::from(embedding);
        sqlx::query("INSERT INTO items (embedding) VALUES ($1)")
            .bind(embedding)
            .execute(&pool)
            .await?;
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
