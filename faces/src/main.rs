use std::sync::atomic;

use clap::{arg, Args, Parser};
use face_embed::path_utils::path_parser;
use face_embed::{embedding::*, face_detector::*, *};
use signal_hook::consts::*;
use signal_hook_tokio::{Handle, Signals};

use futures_util::stream::StreamExt;
use sqlx::types::chrono;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

mod embed;

// Defaults
const ARCFACE_PATH: &str = "./models/arcface-int8.onnx";
const ULTRAFACE_PATH: &str = "./models/ultraface-int8.onnx";
const POSTGRES_CONN_STR: &str = "postgres://postgres:postgres@localhost:5432";
const TABLE_NAME: &str = "items";
const RABBITMQ_DEFAULT_ADDR: &str = "amqp://127.0.0.1:5672/%2f";
const RABBITMQ_DEFAULT_QUEUE: &str = "embed-events";
const OBJECT_STORAGE_DEFAULT_URL: &str = "http://127.0.0.1:9000";
const OBJECT_STORAGE_DEFAULT_ACCESS_KEY: &str = "minioadmin";
const OBJECT_STORAGE_DEFAULT_SECRET_KEY: &str = "minioadmin";
const OBJECT_STORAGE_DEFAULT_BUCKET: &str = "feed";

// Global shutdown
// pub(crate) static SHUTDOWN_REQUESTED: atomic::AtomicBool = atomic::AtomicBool::new(false);

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let token = CancellationToken::new();
    let (s_handle, s_task) = setup_signal_handlers(token.clone())?;
    let args = EmbedArgs::parse();
    if args.rayon_pool_num_threads > 0 && (args.rayon_pool_num_threads as usize) < num_cpus::get() {
        rayon::ThreadPoolBuilder::new()
            .num_threads(args.rayon_pool_num_threads as usize)
            .build_global()?;
    }
    embed::embed(args, token).await?;
    s_handle.close();
    s_task.await?;
    Ok(())
}

#[derive(Parser, Debug)]
pub(crate) struct EmbedArgs {
    /// The path to the ultraface detection model.
    #[arg(short, long, default_value = ULTRAFACE_PATH, value_parser = path_parser, env)]
    detector_path: String,

    /// The path to the arface embedding model.
    #[arg(short, long, default_value = ARCFACE_PATH, value_parser = path_parser, env)]
    embedder_path: String,

    #[arg(short, long, default_value_t = 0.5, env)]
    similarity_threshold: f32,

    #[command(flatten)]
    source: Source,

    #[command(flatten)]
    database: DatabaseArgs,

    #[command(flatten)]
    message_bus: MessageBusArgs,

    #[command(flatten)]
    storage: S3Args,

    #[arg(long, default_value_t = 60 * 60 * 2 /* 2 hr */, env)]
    cache_duration_seconds: u64,

    #[arg(long, default_value_t = 1000, env)]
    channel_bound: usize,

    #[arg(long, default_value_t = -1, env)]
    rayon_pool_num_threads: i32,

    #[arg(short, long, env)]
    verbose: bool,
}

#[derive(Args, Debug)]
pub(crate) struct MessageBusArgs {
    #[arg(short, long, default_value = RABBITMQ_DEFAULT_ADDR, env)]
    address: String,
    #[arg(short, long, default_value = RABBITMQ_DEFAULT_QUEUE, env)]
    queue_name: String,
}

#[derive(Args, Debug)]
pub(crate) struct DatabaseArgs {
    /// The Postgresql database connection string.
    #[arg(short, long, default_value = POSTGRES_CONN_STR, env)]
    conn_str: String,

    #[arg(short, long, default_value = TABLE_NAME, env)]
    /// The table name in the postgresql database.
    table_name: String,
}

#[derive(Args, Debug)]
pub(crate) struct S3Args {
    #[arg(long, default_value = OBJECT_STORAGE_DEFAULT_URL, env)]
    url: String,

    #[arg(long, default_value = OBJECT_STORAGE_DEFAULT_ACCESS_KEY, env)]
    access_key: String,

    #[arg(long, default_value = OBJECT_STORAGE_DEFAULT_SECRET_KEY, env)]
    secret_key: String,

    #[arg(long, default_value = OBJECT_STORAGE_DEFAULT_BUCKET, env)]
    bucket: String,
}

#[derive(Args, Debug)]
pub(crate) struct Source {
    /// The desired X resolution of the camera. This should be a resolution the camera is capable of capturing at.
    #[arg(short, long, default_value_t = 640, env)]
    x_res: u32,

    /// The desired Y resolution of the camera. This should be a resolution the camera is capable of capturing at.
    #[arg(short, long, default_value_t = 480, env)]
    y_res: u32,

    /// The desired FPS of the camera. This should be a FPS value the camera is capable of capturing at.
    #[arg(short, long, default_value_t = 30, env)]
    fps: u32,
}

fn setup_signal_handlers(token: CancellationToken) -> anyhow::Result<(Handle, JoinHandle<()>)> {
    let mut signals = Signals::new(&[SIGTERM, SIGINT, SIGQUIT])?;
    let handle = signals.handle();
    let signals_task = tokio::spawn(async move {
        while let Some(signal) = signals.next().await {
            match signal {
                SIGTERM | SIGINT | SIGQUIT => {
                    if !token.is_cancelled() {
                        println!("Received termination request. Finishing active operations and shutting down...");
                        token.cancel();
                    }
                }
                _ => unreachable!(),
            }
        }
    });
    Ok((handle, signals_task))
}
