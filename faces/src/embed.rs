pub(crate) use std::num::NonZeroU32;

use crate::cache::SlidingVectorCache;
use face_embed::{
    db::Database,
    messaging::Messenger,
    pipeline::{bytes_as_imgref, Detector, LivePublisher},
    storage::get_or_create_bucket,
};
use imgref::{ImgRef, ImgVec};
use nokhwa::{
    pixel_format::RgbFormat,
    utils::{CameraFormat, CameraIndex, RequestedFormat, Resolution},
    Camera,
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
    let (mut cam, x, y) = get_cam(src_x, src_y, src_fps, v)?;
    let mut buffer: Vec<u8> = vec![0; (y.get() * x.get() * 3) as usize];
    while !token.is_cancelled() {
        let slice = buffer.as_mut_slice();
        if cam.write_frame_to_buffer::<RgbFormat>(slice).is_err() {
            break;
        }
        let imgref = bytes_as_imgref(slice, x.get() as _, y.get() as _)?;
        embed_frame(&tx, imgref, &mut det).await?;
    }
    if cam.is_stream_open() {
        cam.stop_stream()?;
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

fn get_cam(x: u32, y: u32, fps: u32, v: bool) -> anyhow::Result<(Camera, NonZeroU32, NonZeroU32)> {
    if v {
        println!("Initializing camera...");
    }
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
    if let Err(e) = cam.open_stream() {
        println!("Failed to open camera stream with error {}", e);
        return Err(e.into());
    }
    if v {
        println!(
            "Setup camera with resolution ({}, {}) at {} fps.",
            x.get(),
            y.get(),
            cam.frame_rate()
        );
    }
    Ok((cam, x, y))
}
