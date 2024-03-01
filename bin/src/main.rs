use std::sync::mpsc::Receiver;
use std::thread::JoinHandle;
use std::{num::NonZeroU32, sync::mpsc, thread};
use std::io::{Cursor, Write};

use face_embed::{*, embedding::*, face_detector::*};
use nokhwa::utils::Resolution;
use nokhwa::{
    pixel_format::RgbFormat,
    utils::{CameraFormat, CameraIndex, RequestedFormat},
    Camera,
};

const X_RES: u32 = 640;
const Y_RES: u32 = 360;
const FPS: u32 = 30;

fn main() -> anyhow::Result<()> {
    // setup channels
    let (tx, rx) = mpsc::channel::<Vec<u8>>();

    println!("Generating embedding model...");
    //const ARCFACE_PATH: &str = "../models/arcface.onnx";
    const ARCFACE_PATH: &str = "/home/aus/Projects/Rust/face_embed/models/arcface.onnx";
    let arcface = ArcFace::new(ARCFACE_PATH)?;
    println!("Embedding model generated.");
    let embedding_dims = arcface.dims();

    // Spin up embedding calculator thread
    let _ = spawn_embedder_thread(rx, Box::new(arcface));

    // Get face detector
    let det = create_face_detector()?;
    let det_dims = det.dims();

    // Setup cam
    println!("Initializing camera...");
    let mut cam = get_cam(X_RES, Y_RES, FPS)?;
    cam.open_stream()?;
    let Resolution { width_x: x , height_y: y} = cam.resolution();
    let x = NonZeroU32::new(x).unwrap();
    let y = NonZeroU32::new(y).unwrap();
    let mut buffer: Vec<u8> = vec![0; (Y_RES * X_RES * 3) as usize];

    println!("Beginning stream...");
    loop {
        cam.write_frame_to_buffer::<RgbFormat>(buffer.as_mut_slice())?;

        // resize image for model
        let img = resize(&buffer, x, y, det_dims.3, det_dims.2, fr::ResizeAlg::Nearest)?;

        let now = std::time::Instant::now();
        let faces = det.detect(img.buffer())?;
        let elapsed = now.elapsed().as_millis();
        if faces.len() == 0 {
            continue
        }
        println!("Detected {} face(s) in {} ms", faces.len(), elapsed);
        for face in faces {
            let crop = face.to_crop_box(X_RES, Y_RES);
            let region = fr::Image::from_slice_u8(x, y, buffer.as_mut_slice(), fr::PixelType::U8x3).unwrap();
            let roi = face_embed::crop_and_resize(
                &mut (region.view()),
                embedding_dims.3,
                embedding_dims.2,
                crop,
                fr::ResizeAlg::Nearest)?.into_vec();
            tx.send(roi)?;
        }
    }

    Ok(())
}

fn create_face_detector() -> anyhow::Result<impl FaceDetector> {
    println!("Generating face detection model...");
    //const ULTRAFACE_PATH: &str = "../models/ultraface-RFB-320.onnx";
     const ULTRAFACE_PATH: &str = "/home/aus/Projects/Rust/face_embed/models/ultraface-RFB-320.onnx";
    let uf_cfg = UltrafaceDetectorConfig {
        top_k: NonZeroU32::new(3).unwrap(),
        ..Default::default()
    };
    let uf = UltrafaceDetector::new(uf_cfg, ULTRAFACE_PATH)?;
    println!("Face detection model generated.");
    Ok(uf)
}

fn get_cam(x: u32, y: u32, fps: u32) -> Result<Camera, nokhwa::NokhwaError> {
    let i = CameraIndex::Index(0);
    let req = RequestedFormat::new::<RgbFormat>(nokhwa::utils::RequestedFormatType::Closest(
        CameraFormat::new(
            nokhwa::utils::Resolution {
                width_x: x,
                height_y: y,
            },
            nokhwa::utils::FrameFormat::MJPEG,
            fps,
        ),
    ));
    Camera::new(i, req)
}

fn save(data: &[u8], path: &str, w: u32, h: u32) -> anyhow::Result<()> {
    let i = image::ImageBuffer::<image::Rgb<u8>, &[u8]>::from_raw(w, h, data).unwrap();
    let mut bytes: Vec<u8> = Vec::new();
    let mut s = String::new();
    s.push_str("/home/aus/Projects/Rust/face_embed/");
    s.push_str(path);
    s.push_str(".png");
    println!("Saving file to {}", s);
    let mut file = std::fs::File::create(s)?;
    i.write_to(&mut Cursor::new(&mut bytes), image::ImageOutputFormat::Png)?;
    file.write_all(bytes.as_slice())?;
    Ok(())
}

/// Spin up a thread to generate embeddings from face crops
fn spawn_embedder_thread(rx: Receiver<Vec<u8>>, model: Box<dyn EmbeddingGenerator + Send>) -> JoinHandle<()> {
    thread::spawn(move || {
        for face in rx.iter() {
            let t = std::time::Instant::now();
            let embedding = model.generate_embedding(face.as_slice()).unwrap();
            println!("Calculated embedding in {} ms", t.elapsed().as_millis());
        }
    })
}
