use std::num::NonZeroU32;
use fast_image_resize as fr;
use fr::DynamicImageView;

pub fn resize(buffer: &[u8], src_w: NonZeroU32, src_h: NonZeroU32, w: NonZeroU32, h: NonZeroU32, alg: fr::ResizeAlg) -> anyhow::Result<fr::Image> {
    let src_view = DynamicImageView::U8x3(fr::ImageView::from_buffer(src_w, src_h, buffer)?);
    let mut dst_image = fr::Image::new(w, h, fr::PixelType::U8x3);
    let mut resizer = fr::Resizer::new(alg);
    resizer.resize(&src_view, &mut dst_image.view_mut())?;
    Ok(dst_image)
}

pub fn crop_and_resize<'a>(
    image: &mut fr::DynamicImageView,
    w: NonZeroU32,
    h: NonZeroU32,
    crop: fr::CropBox,
    alg: fr::ResizeAlg,
) -> anyhow::Result<fr::Image<'a>> {
    if crop.width > image.width().get() as f64 || crop.height > image.height().get() as f64 {
        return Err(anyhow::anyhow!("Resize cropbox bigger than image"))
    }
    image.set_crop_box(crop)?;
    let mut dst_image = fr::Image::new(w, h, image.pixel_type());
    let mut dst_view = dst_image.view_mut();
    let mut resizer = fr::Resizer::new(alg);
    resizer.resize(image, &mut dst_view)?;
    Ok(dst_image)
}
