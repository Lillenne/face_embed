use std::io::{Write, Cursor};

use clap::Args;
use face_embed::path_utils::path_parser;

use crate::Source;

#[derive(Args, Debug)]
pub(crate) struct DetectArgs {
    /// The directory to store crops of the detected faces. A single face will be saved per image,
    /// with timestamped names (live) or mirroring the input file name (glob).
    #[arg(short, long, value_parser = path_parser)]
    output_dir: String,

    /// The path to the ultraface detection model.
    #[arg(short, long, default_value = crate::ULTRAFACE_PATH, value_parser = path_parser)]
    ultraface_path: String,

    #[command(flatten)]
    source: Source,

    #[arg(short, long)]
    verbose: bool
}

pub(crate) async fn detect(args: DetectArgs) -> anyhow::Result<()> {
    todo!();
}

fn save_jpg(data: &[u8], path: &str, w: u32, h: u32) -> anyhow::Result<()> {
    if let Some(i) = image::ImageBuffer::<image::Rgb<u8>, &[u8]>::from_raw(w, h, data) {
        let mut bytes: Vec<u8> = Vec::new();
        let mut file = std::fs::File::create(path)?;
        i.write_to(&mut Cursor::new(&mut bytes), image::ImageFormat::Jpeg)?;
        file.write_all(bytes.as_slice())?;
        Ok(())
    } else { Err(anyhow::anyhow!("Failed to create imagebuffer from raw data")) }
}
