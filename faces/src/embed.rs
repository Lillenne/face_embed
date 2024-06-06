pub(crate) use std::num::NonZeroU32;

use crate::cache::SlidingVectorCache;
use face_embed::{
    db::Database,
    messaging::Messenger,
    pipeline::{bytes_as_imgref, Detector, LivePublisher},
    storage::get_or_create_bucket,
};
use imgref::{ImgRef, ImgVec};
use opencv::videoio::VideoCapture;
use opencv::{
    core::Vector,
    prelude::*,
    videoio::{VideoCaptureProperties::*, CAP_ANY},
};
use tokio::sync::mpsc::Sender;

use crate::*;

type FaceTimeStamp = (ImgVec<[u8; 3]>, chrono::DateTime<chrono::Utc>);

pub(crate) async fn embed(args: EmbedArgs, token: CancellationToken) -> anyhow::Result<()> {
    let arcface = Box::new(ArcFace::new(args.embedder_path.as_str())?);
    let (_, _, h, w) = arcface.dims();
    let uf_cfg = UltrafaceDetectorConfig {
        top_k: NonZeroU32::new(3).unwrap(),
        ..Default::default()
    };
    let det = Detector::new(
        Box::new(UltrafaceDetector::new(uf_cfg, &args.detector_path)?),
        w,
        h,
    );

    let (tx, mut rx) = mpsc::channel(args.channel_bound);
    let token_clone = token.clone();
    let embed_task: JoinHandle<anyhow::Result<()>> = tokio::spawn(async move {
        let db = Database::new(&args.database.db_conn_str, 50).await?;
        let messenger = Messenger::new(
            &args.message_bus.bus_address,
            args.message_bus.live_queue_name,
        )
        .await?;
        let bucket = get_or_create_bucket(
            &args.storage.s3_live_bucket,
            args.storage.s3_url,
            &args.storage.s3_access_key,
            &args.storage.s3_secret_key,
        )
        .await?;
        let cache_duration = std::time::Duration::from_secs(args.cache_duration_seconds);
        let cache = SlidingVectorCache::new(cache_duration, args.similarity_threshold);
        let mut publisher = LivePublisher::new(
            arcface,
            messenger,
            db,
            bucket,
            cache,
            args.similarity_threshold,
        );

        let mut buf: Vec<FaceTimeStamp> = vec![];
        loop {
            let n_rec = rx.recv_many(&mut buf, 40).await;
            if n_rec == 0 || token_clone.is_cancelled() {
                break;
            }

            // TODO remove intermediate allocation
            let mut refs = vec![];
            for owned in buf.iter() {
                refs.push((owned.0.as_ref(), owned.1))
            }

            tokio::select!(
                val = publisher.process_many(refs.as_slice()) => {
                    if val.is_err() {
                        break;
                    }
                }
                _ = token_clone.cancelled() => {
                    break;
                }
            );

            buf.clear();
        }
        Ok(())
    });

    process_feed(
        args.source.x_res,
        args.source.y_res,
        args.source.fps,
        tx,
        det,
        token,
        args.verbose,
    )
    .await?;
    embed_task.await??;
    Ok(())
}

async fn process_feed(
    src_x: u32,
    src_y: u32,
    src_fps: u32,
    tx: Sender<FaceTimeStamp>,
    mut det: Detector,
    token: CancellationToken,
    v: bool,
) -> anyhow::Result<()> {
    let mut cam = VideoCapture::new(0, CAP_ANY)?;
    let mut vec: Vector<i32> = Vector::new();
    vec.push(CAP_PROP_FPS.into());
    vec.push(src_fps as _);
    vec.push(CAP_PROP_FRAME_WIDTH.into());
    vec.push(src_x as _);
    vec.push(CAP_PROP_FRAME_HEIGHT.into());
    vec.push(src_y as _);
    cam.open_with_params(0, CAP_ANY, &vec)?;
    if !opencv::videoio::VideoCapture::is_opened(&cam)? {
        anyhow::bail!(
            "Failed to open camera with specified parameters w: {}, h: {}, fps: {}!",
            src_x,
            src_y,
            src_fps
        );
    }
    while !token.is_cancelled() {
        let mut frame = Mat::default();
        cam.read(&mut frame)?;
        let imgref = mat_to_imgref(&frame);
        embed_frame(&tx, imgref, &mut det).await?;
    }
    Ok(())
}

async fn embed_frame<'a>(
    tx: &tokio::sync::mpsc::Sender<FaceTimeStamp>,
    buffer: ImgRef<'a, [u8; 3]>,
    det: &mut Detector,
) -> anyhow::Result<()> {
    let time = chrono::Utc::now();
    let faces = det.process(buffer)?;
    if faces.is_empty() {
        return Ok(());
    }

    for face in faces {
        if (tx.send((face, time)).await).is_err() {
            println!("Dropped message...");
            break;
        }
    }
    Ok(())
}

fn mat_to_imgref(mat: &Mat) -> ImgRef<'_, [u8; 3]> {
    let slice = mat.data_bytes().expect("Expected contigous mat");
    bytes_as_imgref(slice, mat.cols() as _, mat.rows() as _)
        .expect("Failed to create view into buffer.")
}
