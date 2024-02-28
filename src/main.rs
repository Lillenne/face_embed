use std::{num::NonZeroU32, time::Instant};
use fast_image_resize as fr;
use fr::CropBox;
use nokhwa::{Camera, utils::{CameraIndex, RequestedFormat, CameraFormat}, pixel_format::RgbFormat};
use tract_onnx::prelude::*;
use tract_ndarray::prelude::*;

const X_RES: u32 = 640;
const Y_RES: u32 = 360;
const FPS: u32 = 30;
const ARCFACE_X: u32 = 112;
const ARCFACE_Y: u32 = 112;
const ULTRAFACE_X: u32 = 320;
const ULTRAFACE_Y: u32 = 240;
const ULTRAFACE_MEAN: f32 = 127.0;
const ULTRAFACE_DIV: f32 = 128.0;
const ULTRAFACE_N_BOXES: usize = 4420;

fn main() -> Result<(), anyhow::Error> {
    // Setup cam
    println!("Initializing camera...");
    let mut cam = get_cam(X_RES, Y_RES, FPS)?;

    // Get ultraface model
    println!("Generating ultraface model...");
    const ULTRAFACE_PATH: &str = "models/ultraface-RFB-320.onnx";
    let ultraface = tract_onnx::onnx()
        .model_for_path(ULTRAFACE_PATH)?
        .into_optimized()?
        .into_runnable()?;
    println!("Ultraface model generated.");

    // Spin up embedding calculator thread
    // Get arcface embedding model
    println!("Generating arcface model...");
    const ARCFACE_PATH: &str = "models/arcface.onnx";
    let arcface = tract_onnx::onnx()
        .model_for_path(ARCFACE_PATH)?
        .into_optimized()?
        .into_runnable()?;
    println!("Arcface model generated.");

    let mut buffer: Vec<u8> = vec![0; (Y_RES * X_RES * 3) as usize];
    cam.open_stream()?;

    println!("Beginning stream...");
    loop {
        cam.write_frame_to_buffer::<RgbFormat>(buffer.as_mut_slice())?;
        let loop_time = Instant::now();

        println!("Resizing image...");
        // Resize to ultraface tensor size
        let src_image = fr::Image::from_slice_u8(
            NonZeroU32::new(X_RES).unwrap(),
           NonZeroU32::new(Y_RES).unwrap(),
        buffer.as_mut_slice(),
        fr::PixelType::U8x3,
        ).unwrap();

        let mut dst_image = fr::Image::new(
            NonZeroU32::new(ULTRAFACE_X).unwrap(),
            NonZeroU32::new(ULTRAFACE_Y).unwrap(),
            src_image.pixel_type(),
        );

        let mut resizer = fr::Resizer::new(fr::ResizeAlg::Nearest);
        resizer.resize(&src_image.view(), &mut dst_image.view_mut())?;

        let data = dst_image.buffer();

        println!("Creating tensor...");
        // Convert buffer to rust Image. Apply preprocessing (subtract mean (127) & divide by 128)
        let input: Tensor = tract_ndarray::Array4::from_shape_fn((1,3,ULTRAFACE_Y as usize, ULTRAFACE_X as usize), |(_,c,y,x)| {
            let idx = (y * ULTRAFACE_X as usize + x) * 3 + c;
            (data[idx] as f32 - ULTRAFACE_MEAN) / ULTRAFACE_DIV
        }).into();

        println!("Running face detection model...");
        let now = Instant::now();
        let res = ultraface.run(tvec!(input.into()))?;
        let elapsed = now.elapsed();
        println!("Detected face in {:?} ms", elapsed.as_millis());

        let view = res[0]
            .to_array_view::<f32>()?
            .into_shape((ULTRAFACE_N_BOXES, 2))?;
        let binding = view
            .slice(s![..,1]);
        // TODO NMS
        let (idx, score) = binding
            .iter()
            .enumerate()
            .max_by(|a,b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();

        if *score < 0.75 { continue; }
        println!("Found face with probability {score}");

        // Extract bounding box from ultraface
        let view = &res[1].to_array_view::<f32>()?.into_shape((ULTRAFACE_N_BOXES, 4))?;
        let bbox = view.slice(s![idx, ..]);

        // Bounding box is represented as x1, y1, x2, y2
        let w: f32 = ULTRAFACE_X as f32;
        let h: f32 = ULTRAFACE_Y as f32;
        let left = (w * bbox[0] as f32) as f64;
        let top = (h * bbox[1] as f32) as f64;
        let width = (w * bbox[2] as f32) as f64 - left;
        let height = (h * bbox[3] as f32) as f64 - top;

        println!("Extracting face...");
        // crop and resize original image for arcface embedding
        let mut dst_image = fr::Image::new(
            NonZeroU32::new(ARCFACE_X).unwrap(),
            NonZeroU32::new(ARCFACE_Y).unwrap(),
            src_image.pixel_type(),
        );
        let crop = CropBox { height, left, top, width};
        let mut src_view = src_image.view();
        src_view.set_crop_box(crop)?;
        resizer.resize(&src_view, &mut dst_image.view_mut())?;

        println!("Generating facial embedding...");
        let now = Instant::now();
        let input = Tensor::from_shape(&[1,3,ARCFACE_X as usize, ARCFACE_Y as usize], dst_image.buffer())?;
        let cast = input.cast_to_dt(DatumType::F32)?.cast_to_dt(DatumType::F32)?.into_owned();
        let result = arcface.run(tvec!(cast.into()))?;
        let elapsed = now.elapsed();
        println!("Generated embedding in {:?} ms", elapsed.as_millis());
        // TODO normalize to unit len
        println!("Result: {:?}", result);

        let elapsed = loop_time.elapsed();
        println!("Completed loop in {:?} ms", elapsed.as_millis());

        break;
    }

    Ok(())
}

fn get_cam(x: u32, y: u32, fps: u32) -> Result<Camera, nokhwa::NokhwaError> {
    let i = CameraIndex::Index(0);
    let req = RequestedFormat::new::<RgbFormat>(
nokhwa::utils::RequestedFormatType::Closest(
            CameraFormat::new(nokhwa::utils::Resolution { width_x: x, height_y: y },
    nokhwa::utils::FrameFormat::MJPEG,
fps)));
    Camera::new(i, req)
}
