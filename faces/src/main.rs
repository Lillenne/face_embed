use std::iter::SkipWhile;

use clap::{arg, Args, Parser};
use face_embed::path_utils::path_parser;
use signal_hook::consts::*;
use signal_hook_tokio::{Handle, Signals};

use futures_util::stream::StreamExt;
use sqlx::types::chrono;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use tracing::{info, Instrument};

mod embed;

const HAAR_PATH: &str = "../models/haarcascade_frontalface_default.xml";

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt().init();
    let token = CancellationToken::new();
    let (s_handle, s_task) = setup_signal_handlers(token.clone())?;
    let args = EmbedArgs::parse();
    embed::embed(args, token).await?;
    s_handle.close();
    s_task.await?;
    Ok(())
}

#[derive(Parser, Debug)]
pub(crate) struct EmbedArgs {
    #[command(flatten)]
    source: Source,

    #[command(flatten)]
    message_bus: MessageBusArgs,

    #[command(flatten)]
    storage: S3Args,

    /// The path to the haar cascade weights.
    #[arg(long, default_value = HAAR_PATH, value_parser = path_parser, env)]
    haar_path: String,

    /// The maximum number of images that may be processing at a time.
    #[arg(long, default_value_t = 1000, env)]
    channel_bound: usize,
}

#[derive(Args, Debug)]
pub(crate) struct MessageBusArgs {
    /// The publishing address of the amqp message bus.
    #[arg(short, long, env)]
    bus_address: String,

    /// The name of the message bus queue to publish to.
    #[arg(short, long, env)]
    capture_queue_name: String,
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
    let mut signals = Signals::new([SIGTERM, SIGINT, SIGQUIT])?;
    let handle = signals.handle();
    let signals_task = tokio::spawn(async move {
        while let Some(signal) = signals.next().await {
            match signal {
                SIGTERM | SIGINT | SIGQUIT => {
                    if !token.is_cancelled() {
                        info!("Received termination request. Finishing active operations and shutting down...");
                        token.cancel();
                    }
                }
                _ => unreachable!(),
            }
        }
    });
    Ok((handle, signals_task))
}
