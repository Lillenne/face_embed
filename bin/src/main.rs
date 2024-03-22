use std::io::{Cursor, Write};
use std::sync::atomic;

use clap::{arg, Parser, Subcommand, Args};
use embed::EmbedArgs;
use detect::DetectArgs;
use face_embed::{embedding::*, face_detector::*, *};
use signal_hook::consts::*;
use signal_hook_tokio::{Signals, Handle};

use futures_util::stream::StreamExt;
use sqlx::types::chrono;
use sqlx::{Pool, Postgres};
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use visualize::VisualizeArgs;

mod embed;
mod visualize;
mod detect;

// Defaults
const ARCFACE_PATH: &str = "./models/arcface.onnx";
const ULTRAFACE_PATH: &str = "./models/ultraface-RFB-320.onnx";
const POSTGRES_CONN_STR: &str = "postgres://postgres:postgres@localhost:5432";
const TABLE_NAME: &str = "items";
const RABBITMQ_DEFAULT_ADDR: &str = "amqp://127.0.0.1:5672/%2f";
const RABBITMQ_DEFAULT_QUEUE: &str = "embed-events";
const OBJECT_STORAGE_DEFAULT_URL: &str = "http://127.0.0.1:9000";
const OBJECT_STORAGE_DEFAULT_ACCESS_KEY: &str = "minioadmin";
const OBJECT_STORAGE_DEFAULT_SECRET_KEY: &str = "minioadmin";
const OBJECT_STORAGE_DEFAULT_BUCKET: &str = "feed";

// Global shutdown
pub(crate) static SHUTDOWN_REQUESTED: atomic::AtomicBool = atomic::AtomicBool::new(false);


#[tokio::main]
async fn main() -> anyhow::Result<()> {
    setup_signal_handlers()?;
    rayon::ThreadPoolBuilder::new().num_threads(4).build_global()?;
    let cli = Cli::parse();
    match cli.command {
        Command::Detect(args) => { detect::detect(args).await? },
        Command::Embed(args) => { embed::embed(args).await? },
        Command::Visualize(args) => { visualize::visualize(args).await? },
    };
    Ok(())
}


#[derive(Parser, Debug)]
#[command(name = "faces", version, about)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Detect faces
    Detect(DetectArgs),

    /// Detect faces and generate face signatures
    Embed(EmbedArgs),

    /// Visualize face signatures in a dimensionality-reduced plot
    Visualize(VisualizeArgs)
}

#[derive(Args, Debug)]
pub(crate) struct MessageBusArgs {
    #[arg(short, long, default_value = RABBITMQ_DEFAULT_ADDR)]
    address: String,
    #[arg(short, long, default_value = RABBITMQ_DEFAULT_QUEUE)]
    queue_name: String
}

#[derive(Args, Debug)]
pub(crate) struct DatabaseArgs {
    /// The Postgresql database connection string.
    #[arg(short, long, default_value = POSTGRES_CONN_STR)]
    conn_str: String,

    #[arg(short, long, default_value = TABLE_NAME)]
    /// The table name in the postgresql database.
    table_name: String,
}

#[derive(Args, Debug)]
pub(crate) struct S3Args {
    #[arg(long, default_value = OBJECT_STORAGE_DEFAULT_URL)]
    url: String,

    #[arg(long, default_value = OBJECT_STORAGE_DEFAULT_ACCESS_KEY)]
    access_key: String,

    #[arg(long, default_value = OBJECT_STORAGE_DEFAULT_SECRET_KEY)]
    secret_key: String,

    #[arg(long, default_value = OBJECT_STORAGE_DEFAULT_BUCKET)]
    bucket: String
}

#[derive(Args, Debug)]
pub(crate) struct Source {
    /// The desired X resolution of the camera. This should be a resolution the camera is capable of capturing at.
    #[arg(short, long, default_value_t = 640)]
    x_res: u32,

    /// The desired Y resolution of the camera. This should be a resolution the camera is capable of capturing at.
    #[arg(short, long, default_value_t = 480)]
    y_res: u32,

    /// The desired FPS of the camera. This should be a FPS value the camera is capable of capturing at.
    #[arg(short, long, default_value_t = 30)]
    fps: u32,

    /// Process images on disk using the specified glob pattern instead of a live feed from a camera.
    #[arg(short, long)]
    glob: Option<String>,
}

fn setup_signal_handlers() -> anyhow::Result<(Handle, JoinHandle<()>)> {
    let mut signals = Signals::new(&[
            SIGTERM,
            SIGINT,
            SIGQUIT,
        ])?;
        let handle = signals.handle();
        let signals_task = tokio::spawn(async move {
            while let Some(signal) = signals.next().await {
                match signal {
                    SIGTERM | SIGINT | SIGQUIT => {
                        if !SHUTDOWN_REQUESTED.load(atomic::Ordering::Acquire) {
                            println!("Received termination request. Finishing active operations and shutting down...");
                            SHUTDOWN_REQUESTED.store(true, atomic::Ordering::Release)
                       }
                    },
                    _ => unreachable!(),
                }
            }
        });
    Ok((handle, signals_task))
}
