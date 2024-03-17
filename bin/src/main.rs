use std::io::{Cursor, Write};
use std::num::NonZeroU32;
use std::path::Path;
use std::sync::atomic;

use clap::{arg, Parser, Subcommand, Args};
use face_embed::{embedding::*, face_detector::*, *};
use pgvector::*;
use signal_hook::consts::*;
use signal_hook_tokio::{Signals, Handle};

use futures_util::stream::StreamExt;
use sqlx::postgres::PgPoolOptions;
use sqlx::types::chrono;
use sqlx::{Pool, Postgres, Row, FromRow};
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

use crate::embed::detect;

mod embed;
mod visualize;

// Defaults
const ARCFACE_PATH: &str = "./models/arcface.onnx";
const ULTRAFACE_PATH: &str = "./models/ultraface-RFB-320.onnx";
const POSTGRES_CONN_STR: &str = "postgres://postgres:postgres@localhost:5432";
const TABLE_NAME: &str = "items";

// Global state
pub(crate) static SHUTDOWN_REQUESTED: atomic::AtomicBool = atomic::AtomicBool::new(false);


#[tokio::main]
async fn main() -> anyhow::Result<()> {
    setup_signal_handlers()?;
    let cli = Cli::parse();
    println!("Executing with args: {:?}", cli);
    match cli.command {
        Command::Detect(args) => { detect(args, cli.verbose).await? },
        Command::Embed(args) => { embed::embed(args, cli.verbose).await? },
        Command::Visualize(args) => { visualize::visualize(args, cli.verbose).await? },
    };
    Ok(())
}


#[derive(Parser, Debug)]
#[command(name = "faces", version, about, next_line_help = true)]
struct Cli {
    #[command(subcommand)]
    command: Command,
    #[arg(short, long)]
    verbose: bool
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
pub(crate) struct DetectArgs {
    /// The directory to store crops of the detected faces. A single face will be saved per image,
    /// with timestamped names (live) or mirroring the input file name (glob).
    #[arg(short, long, value_parser = crate::path_parser)]
    output_dir: String,

    /// The path to the ultraface detection model.
    #[arg(short, long, default_value = ULTRAFACE_PATH, value_parser = crate::path_parser)]
    ultraface_path: String,

    #[command(flatten)]
    source: Source,
}

#[derive(Args, Debug)]
pub(crate) struct EmbedArgs {
    /// The path to the ultraface detection model.
    #[arg(short, long, default_value = ULTRAFACE_PATH, value_parser = crate::path_parser)]
    ultraface_path: String,

    /// The path to the arface embedding model.
    #[arg(short, long, default_value = ARCFACE_PATH, value_parser = crate::path_parser)]
    arcface_path: String,

    #[command(flatten)]
    source: Source,

    #[command(flatten)]
    database: DatabaseArgs
}

#[derive(Args, Debug)]
pub(crate) struct VisualizeArgs {

    /// The output path for the output PNG.
    #[arg(short, long, value_parser = expand_path)]
    output_path: String,

    #[arg(short, long, default_value_t = 2, value_parser = clap::value_parser!(u8).range(2..=3))]
    dimensions: u8,

    #[arg(short, long, default_value_t = 5000)]
    limit: usize,

    #[command(flatten)]
    alg: ReductionAlg,

    #[command(flatten)]
    database: DatabaseArgs
}

#[derive(Args, Debug)]
#[group(required = true, multiple = false)]
struct ReductionAlg {
    #[arg(long, group = "dim_reduc")]
    pca: bool,

    #[arg(long, group = "dim_reduc")]
    tsne: bool,
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

#[derive(FromRow, Debug)]
pub(crate) struct EmbeddingData {
    pub id: i64,
    pub embedding: Vector,
    pub time: chrono::DateTime<chrono::Utc>,
    pub class_id: Option<i64>
}

fn expand_path(path: &str) -> anyhow::Result<String> {
    let path = Path::new(path);
    if path.is_relative() {
        let cwd = std::env::current_dir()?;
        let mut bp = std::path::PathBuf::new();
        bp.push(cwd);
        bp.push(path.as_os_str());
        if let Ok(path) = std::fs::canonicalize(bp) {
            if let Some(str) = path.as_os_str().to_str() {
                Ok(str.into())
            } else {
                Err(anyhow::anyhow!("Failed to get relative path: {:?}", path))
            }
        }
        else {
            Err(anyhow::anyhow!("Failed to get relative path"))
        }
    } else {
        if let Some(str) = path.to_str() {
            Ok(str.to_string())
        } else {
            Err(anyhow::anyhow!("Path contains invalid characters"))
        }
    }
}

fn path_parser(path: &str) -> anyhow::Result<String> {
    let path = expand_path(path)?;
    if Path::new(path.as_str()).exists() {
        Ok(path)
    } else {
        Err(anyhow::anyhow!("Path does not exist!"))
    }
}

async fn setup_sqlx(conn: &str, table: &str) -> anyhow::Result<Pool<Postgres>> {
    let pool = PgPoolOptions::new()
        .max_connections(5)
        .connect(conn)
        .await?;
    sqlx::query("CREATE EXTENSION IF NOT EXISTS vector")
        .execute(&pool)
        .await?;
    let query = "CREATE TABLE IF NOT EXISTS classes (id bigserial PRIMARY KEY, name varchar(40))";
    sqlx::query(query).execute(&pool).await?;
    let query = format!("CREATE TABLE IF NOT EXISTS {} (id BIGSERIAL PRIMARY KEY, embedding VECTOR(512) NOT NULL, time TIMESTAMPTZ NOT NULL, class_id BIGINT REFERENCES classes(id) NULL)", table);
    sqlx::query(query.as_str())
        .execute(&pool)
        .await?;
    Ok(pool)
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

fn create_face_detector(path: &str) -> anyhow::Result<impl FaceDetector> {
    println!("Generating face detection model...");
    let uf_cfg = UltrafaceDetectorConfig {
        top_k: NonZeroU32::new(3).unwrap(),
        ..Default::default()
    };
    let uf = UltrafaceDetector::new(uf_cfg, path)?;
    println!("Face detection model generated.");
    Ok(uf)
}

fn save_jpg(data: &[u8], path: &str, w: u32, h: u32) -> anyhow::Result<()> {
    let i = image::ImageBuffer::<image::Rgb<u8>, &[u8]>::from_raw(w, h, data).unwrap();
    let mut bytes: Vec<u8> = Vec::new();
    let mut file = std::fs::File::create(path)?;
    i.write_to(&mut Cursor::new(&mut bytes), image::ImageFormat::Jpeg)?;
    file.write_all(bytes.as_slice())?;
    Ok(())
}
