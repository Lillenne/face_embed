use clap::{arg, Args, Parser};
use face_embed::cache::SlidingVectorCache;
use face_embed::embedding::ArcFace;
use face_embed::face_detector::{UltrafaceDetector, UltrafaceDetectorConfig};
use face_embed::messaging::CaptureEvent;
use face_embed::pipeline::{bytes_as_imgref, get_embedder_inputs, LivePublisher, PublisherError};
use face_embed::storage::get_or_create_bucket;
use face_embed::{FaceDetector, ModelDims};
use futures_lite::stream::StreamExt;
use image::codecs::jpeg::JpegDecoder;
use image::{EncodableLayout, ImageDecoder, ImageReader};
use lapin::options::{BasicConsumeOptions, QueueDeclareOptions};
use lapin::protocol::connection;
use lapin::Channel;
use lapin::{options::BasicAckOptions, ConnectionProperties};
use lapin::{types::FieldTable, Connection};
use lettre::transport::smtp::authentication::Credentials;
use lettre::{
    message::{header::ContentType, Mailbox},
    Transport,
};
use lettre::{Message, SmtpTransport};
use mini_moka::sync::Cache;
use std::io::{BufRead, Cursor};
use std::num::NonZeroU32;
use std::{env::var, time::Duration};
use tracing::{error, info, trace, warn, Instrument};

use face_embed::{
    db::Database,
    messaging::{DetectionEvent, LabelEvent, Messenger},
    path_utils::path_parser,
};

const ARCFACE_PATH: &str = "../models/arcface-int8.onnx";
const ULTRAFACE_PATH: &str = "../models/ultraface-int8.onnx";

#[derive(Parser, Debug)]
pub struct EmbedArgs {
    /// The path to the ultraface detection model.
    #[arg(short, long, default_value = ULTRAFACE_PATH, value_parser = path_parser, env)]
    detector_path: String,

    /// The path to the arface embedding model.
    #[arg(short, long, default_value = ARCFACE_PATH, value_parser = path_parser, env)]
    embedder_path: String,

    #[arg(short, long, default_value_t = 0.5, env)]
    similarity_threshold: f32,

    /// The Postgresql database connection string.
    #[arg(long, env)]
    db_conn_str: String,

    #[arg(long, default_value_t = 60 * 60 * 2 /* 2 hr */, env)]
    cache_duration_seconds: u64,

    #[command(flatten)]
    storage: S3Args,
}

#[derive(Args, Debug)]
pub(crate) struct S3Args {
    /// The url of the S3 compatible image storage service.
    #[arg(long, env)]
    s3_url: String,

    /// The access credentials for the S3 compatible image storage service.
    #[arg(long, env)]
    s3_access_key: String,

    /// The secret credentials for the S3 compatible image storage service.
    #[arg(long, env)]
    s3_secret_key: String,

    /// The name of the bucket to use for the S3 compatible image storage service.
    #[arg(long, env)]
    s3_capture_bucket: String,
}

#[tokio::main(flavor = "multi_thread", worker_threads = 2)]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt().init();
    let args = EmbedArgs::parse();
    let addr = var("BUS_ADDRESS")?;
    let capture_queue_name = var("CAPTURE_QUEUE_NAME")?;
    let detection_queue_name = var("DETECTION_QUEUE_NAME")?;

    info!("Initializing sign-up message bus connection...");
    let conn = Connection::connect(&addr, ConnectionProperties::default()).await?;

    let sign_up = Messenger::new_channel(&conn, var("SIGN_UP_QUEUE_NAME")?).await?;
    sign_up.queue_declare().await?;
    info!("Connection initialized.");
    let sign = tokio::spawn(async move {
        let _ = consume_sign_up_events(sign_up, args.similarity_threshold).await;
    });

    info!("Initializing capture processing message bus connection...");
    let capture_channel = Messenger::new_channel(&conn, capture_queue_name).await?;
    capture_channel.queue_declare().await?;
    info!("Connection initialized.");

    info!("Initializing detection message bus connection...");
    let detection_channel = Messenger::new_channel(&conn, detection_queue_name).await?;
    detection_channel.queue_declare().await?;
    info!("Connection initialized.");
    let capture = tokio::spawn(async move {
        let _ = consume_capture_events(args, capture_channel, detection_channel).await;
    });
    // sign.await?;
    capture.await?;
    Ok(())
}

async fn consume_capture_events(
    args: EmbedArgs,
    detection_messenger: Messenger,
    messenger: Messenger,
) -> anyhow::Result<()> {
    info!(
        "Initializing ArcFace model from file {}...",
        args.embedder_path
    );
    let arcface = Box::new(ArcFace::new(args.embedder_path.as_str())?);
    info!("Model initialized.");
    let (_, _, dest_h, dest_w) = arcface.dims();
    info!("Initializing UltraFace model...");
    let uf_cfg = UltrafaceDetectorConfig {
        top_k: NonZeroU32::new(3).unwrap(),
        ..Default::default()
    };
    let uf: Box<dyn FaceDetector + Send + Sync> =
        Box::new(UltrafaceDetector::new(uf_cfg, &args.detector_path)?);
    info!("Model initialized.");
    info!("Initializing DB connection");
    let db = Database::new(&args.db_conn_str, 50).await?;
    info!("DB connection initialized.");
    let cache_duration = std::time::Duration::from_secs(args.cache_duration_seconds);
    let cache = SlidingVectorCache::new(cache_duration, args.similarity_threshold);
    let bucket = get_or_create_bucket(
        &args.storage.s3_capture_bucket,
        args.storage.s3_url,
        &args.storage.s3_access_key,
        &args.storage.s3_secret_key,
    )
    .await?;
    let mut consumer = detection_messenger.get_consumer("embedder").await?;
    let mut publisher = LivePublisher::new(
        arcface,
        "Arcface-8".into(),
        messenger,
        db,
        bucket.clone(),
        cache,
        args.similarity_threshold,
    );

    info!("Begin processing messages...");
    while let Some(Ok(delivery)) = consumer.next().await {
        trace!("Processing message...");
        match serde_json::from_slice::<CaptureEvent>(delivery.data.as_slice()) {
            Ok(data) => {
                let response = bucket.get_object(&data.path).await?;
                if response.status_code() != 200 {
                    warn!("Failed to pull detection event!");
                    continue;
                }
                info!("Pulled message id={}", data.id);

                let image = ImageReader::new(Cursor::new(response.bytes()))
                    .with_guessed_format()?
                    .decode()?
                    .into_rgb8();

                if let Ok(slice) =
                    bytes_as_imgref(image.as_bytes(), image.width() as _, image.height() as _)
                {
                    let inputs = get_embedder_inputs(&uf, slice, dest_w, dest_h)?;
                    trace!("Detected faces for id={}", data.id);
                    for input in inputs {
                        if let Err(e) = publisher.process(input.as_ref(), data.time).await {
                            match e {
                                PublisherError::EmbeddingNotUnique => {
                                    trace!("Non-unique capture event: {}", data.id);
                                }
                                _ => {
                                    warn!("Rejected capture event: {}", data.id);
                                    continue;
                                }
                            }
                        }
                    }
                    delivery.ack(BasicAckOptions::default()).await?;
                    info!("Successfully processed event id={}", data.id);
                } else {
                    continue;
                }
            }
            Err(e) => {
                error!("Failed to deserialize event: {}!", e);
                if let Ok(s) = String::from_utf8(delivery.data) {
                    info!(s)
                }
                continue;
            }
        }
    }
    info!("End processing messages.");
    Ok(())
}

async fn consume_detection_events(messenger: Messenger) -> anyhow::Result<()> {
    let cache = Cache::builder()
        .time_to_live(Duration::from_secs(60 * 60 /* 1 hr */))
        .build();
    let from_email: Mailbox = var("EMAIL_SENDER")?.parse()?;
    let subject = var("EMAIL_SUBJECT")?;
    let creds = Credentials::new(var("EMAIL_USERNAME")?, var("EMAIL_PASSWORD")?);
    let mut consumer = messenger.get_consumer("live-consumer").await?;
    while let Some(Ok(delivery)) = consumer.next().await {
        let data: DetectionEvent = serde_json::from_slice(delivery.data.as_slice())?;
        info!("Received detection event: {:?}", &data);
        let Some(user) = data.user else {
            info!("Unidentified user.");
            delivery.ack(BasicAckOptions::default()).await?;
            continue;
        };
        if cache.get(&user.id).is_some() {
            info!(
                "Received duplicate event in time frame for user {}. Skipping.",
                user.name
            );
            delivery.ack(BasicAckOptions::default()).await?;
            continue;
        }
        let Ok(addr) = user.email.parse() else {
            error!("Failed to parse address: {}.", user.email);
            continue;
        };
        let to = Mailbox::new(Some(user.name.clone()), addr);
        let Ok(email) = Message::builder()
            .from(from_email.clone())
            .to(to)
            .subject(subject.clone())
            .header(ContentType::TEXT_PLAIN)
            .body(format!("Hi {}, you were here today!", user.name))
        else {
            error!("Failed to make email");
            continue;
        };

        // Open a remote connection to gmail
        let mailer = SmtpTransport::relay("smtp.gmail.com")?
            .credentials(creds.clone())
            .build();

        // Send the email
        match mailer.send(&email) {
            Ok(_) => {
                cache.insert(user.id, 0_u8);
                delivery.ack(BasicAckOptions::default()).await?;
            }
            Err(e) => error!("Error: {}", e),
        }
    }
    Ok(())
}

async fn consume_sign_up_events(messenger: Messenger, threshold: f32) -> anyhow::Result<()> {
    let db = Database::new(var("DB_CONN_STR")?.as_str(), 5).await?;
    let mut consumer = messenger.get_consumer("sign-up-consumer").await?;

    while let Some(Ok(delivery)) = consumer.next().await {
        let data: LabelEvent = serde_json::from_slice(delivery.data.as_slice())?;
        info!(
            "Received label event: id={}, name={}",
            data.event_id, data.user.name
        );
        let events = db.update_past_captures(data.user, threshold).await?;
        if !events.is_empty() {
            info!(
                "Event {} matched {} existing detections",
                data.event_id,
                events.len()
            );
            for event in events {
                let ser = serde_json::to_vec_pretty(&event)?;
                messenger.publish(ser.as_slice()).await?;
            }
        }
        delivery.ack(BasicAckOptions::default()).await?;
    }
    Ok(())
}
