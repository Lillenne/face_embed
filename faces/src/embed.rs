use anyhow::bail;
use face_embed::{
    messaging::{self, Messenger},
    storage::get_or_create_bucket,
    DetectedObject, Rect,
};
use messaging::CaptureEvent;
use opencv::{
    core::{Size, Vector},
    imgcodecs::imencode,
    imgproc,
    objdetect::{self, CascadeClassifier},
    prelude::*,
    videoio::{VideoCapture, VideoCaptureProperties::*, CAP_ANY},
};
use tokio::sync::mpsc::Sender;
use tracing::{error, info, trace, warn};
use uuid::Uuid;

use crate::*;

type FaceTimeStamp = (chrono::DateTime<chrono::Utc>, Mat, Vec<DetectedObject>);

pub(crate) async fn embed(args: EmbedArgs, token: CancellationToken) -> anyhow::Result<()> {
    let (tx, mut rx) = mpsc::channel::<FaceTimeStamp>(args.channel_bound);
    let token_clone = token.clone();
    let embed_task: JoinHandle<anyhow::Result<()>> = tokio::spawn(async move {
        info!("Initializing message bus connection...");
        let messenger = Messenger::new_connection(
            &args.message_bus.bus_address,
            args.message_bus.capture_queue_name,
        )
        .await;
        if messenger.is_err() {
            let reason = "Failed to initialize message bus connection!";
            error!(reason);
            token_clone.cancel();
            bail!(reason);
        }
        let messenger = messenger.unwrap();
        messenger.queue_declare().await?;
        info!("Message bus connection initialized.");

        info!("Initializing remote storage...");
        let bucket = get_or_create_bucket(
            &args.storage.s3_capture_bucket,
            args.storage.s3_url,
            &args.storage.s3_access_key,
            &args.storage.s3_secret_key,
        )
        .await;
        if bucket.is_err() {
            let reason = "Failed to initialize remote storage!";
            error!(reason);
            token_clone.cancel();
            bail!(reason);
        }
        let bucket = bucket.unwrap();
        info!("Remote storage initialized...");

        info!("Beginning processing loop");
        while let Some((time, mat, _)) = tokio::select!(
            v = rx.recv() => v,
            _ = token_clone.cancelled() => {
                info!("Aborted publishing loop due to cancellation request.");
                None
            }
        ) {
            let mut jpg = Vector::<u8>::new();
            let flags = Vector::<i32>::new();
            imencode(".jpg", &mat, &mut jpg, &flags)?;
            let id = Uuid::now_v7();
            let mut path = String::new();
            path.push('/');
            path.push_str(&id.to_string());
            bucket
                .put_object_stream_with_content_type(&mut jpg.as_slice(), &path, "image/jpeg")
                .await?;
            bucket
                .put_object_tagging(&path, &[("time", &time.to_string()), ("model", "haar")])
                .await?;
            let event = CaptureEvent { id, time, path };
            let payload = serde_json::to_vec_pretty(&event)?;
            messenger.publish(&payload).await?;
            info!(
                "Published event: id={}, time={}, path={}",
                &event.id, &event.time, &event.path
            );
        }
        Ok(())
    });

    process_feed(
        args.source.x_res,
        args.source.y_res,
        args.source.fps,
        tx,
        HaarDetector::new(&args.haar_path, 4)?,
        token,
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
    mut det: HaarDetector,
    token: CancellationToken,
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
    info!(
        "Opened camera with parameters w: {}, h: {}, fps: {}!",
        src_x, src_y, src_fps
    );

    while !token.is_cancelled() {
        let mut frame = Mat::default();
        cam.read(&mut frame)?;
        let time = chrono::Utc::now();
        if frame.empty() {
            bail!("Read an empty frame.")
        }
        if frame.typ() != 16 {
            bail!("Video stream must be 8-bit color.")
        }

        let faces = det.detect(&frame)?;
        if faces.is_empty() {
            continue;
        }
        trace!("Detected {} faces", faces.len());

        if tx.send((time, frame, faces)).await.is_err() {
            warn!("Dropped message...");
        };
    }
    Ok(())
}

struct HaarDetector {
    classifer: CascadeClassifier,
    scale_factor: u32,
}

impl HaarDetector {
    pub fn new(path: &str, scale_factor: u32) -> anyhow::Result<Self> {
        let s = if scale_factor == 0 { 1 } else { scale_factor };
        println!("path: {}", path);
        Ok(HaarDetector {
            classifer: opencv::objdetect::CascadeClassifier::new(path)?,
            scale_factor: s,
        })
    }

    fn detect(&mut self, frame: &Mat) -> anyhow::Result<Vec<DetectedObject>> {
        let mut gray = Mat::default();
        imgproc::cvt_color_def(&frame, &mut gray, imgproc::COLOR_BGR2GRAY)?;
        let mut reduced = Mat::default();
        let factor = 1.0 / self.scale_factor as f64;
        imgproc::resize(
            &gray,
            &mut reduced,
            Size::new(0, 0),
            factor,
            factor,
            imgproc::INTER_LINEAR,
        )?;
        let mut faces = Vector::new();
        self.classifer.detect_multi_scale(
            &reduced,
            &mut faces,
            1.1,
            2,
            objdetect::CASCADE_SCALE_IMAGE,
            Size::new(30, 30),
            Size::new(0, 0),
        )?;
        let size = reduced.size()?;
        Ok(faces
            .iter()
            .map(|face| DetectedObject {
                confidence: 1.0,
                class: 1,
                bounding_box: Rect {
                    left: face.x as f32 / size.width as f32,
                    top: face.y as f32 / size.height as f32,
                    width: face.width as f32 / size.width as f32,
                    height: face.height as f32 / size.height as f32,
                },
            })
            // opencv::core::Rect::new(face.x * self.scale_factor,
            //                  face.y * self.scale_factor,
            //                  face.width * self.scale_factor,
            //                  face.height * self.scale_factor)))
            .collect::<Vec<_>>())
    }
}
