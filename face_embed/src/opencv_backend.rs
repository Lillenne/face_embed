use opencv::{
    core::{Mat, MatTraitConst, MatTraitConstManual, Ptr, Size},
    imgproc,
    objdetect::{
        FaceDetectorYN, FaceRecognizerSF, FaceRecognizerSFTrait, FaceRecognizerSFTraitConst,
    },
    prelude::FaceDetectorYNTrait,
};

use crate::{DetectedObject, Rect};

pub struct FaceDetectAndEmbedder {
    fd: Ptr<FaceDetectorYN>,
    emb: Ptr<FaceRecognizerSF>,
}

impl FaceDetectAndEmbedder {
    pub fn new(
        fd_path: &str,
        emb_path: &str,
        score_threshold: f32,
        nms_threshold: f32,
        top_k: u32,
        img_w: u32,
        img_h: u32,
    ) -> anyhow::Result<FaceDetectAndEmbedder> {
        let detector = FaceDetectorYN::create(
            fd_path,
            "",
            Size::new(img_w as _, img_h as _),
            score_threshold,
            nms_threshold,
            top_k as _,
            0,
            0,
        )?;
        let face_recognizer = FaceRecognizerSF::create_def(emb_path, "").unwrap();
        Ok(Self {
            fd: detector,
            emb: face_recognizer,
        })
    }

    pub fn process(&mut self, mat: &Mat) -> anyhow::Result<()> {
        if mat.size()? != self.fd.get_input_size()? {
            let mut rsz = Mat::default();
            imgproc::resize(
                &mat,
                &mut rsz,
                self.fd.get_input_size()?,
                0.,
                0.,
                imgproc::INTER_LINEAR,
            )?;
            self.do_process(&rsz)
        } else {
            self.do_process(mat)
        };
        Ok(())
    }

    fn do_process(&mut self, input: &Mat) -> anyhow::Result<Vec<(DetectedObject, Vec<f32>)>> {
        let mut faces = Mat::default();
        // 0,1 xy, 2,3 wh, 14 conf
        let _ = self.fd.detect(&input, &mut faces)?;
        let inp_w = input.size()?.width;
        let inp_h = input.size()?.height;

        let mut feature = Mat::default();
        let mut result = vec![];
        let end = i32::min(faces.rows(), self.fd.get_top_k()?);
        let mut aligned_face1 = Mat::default();
        for i in 0..end {
            let row = faces.row(i)?;
            let conf = row.at::<f32>(14)?;
            let x = row.at::<f32>(0)?;
            let y = row.at::<f32>(0)?;
            let w = row.at::<f32>(2)?;
            let h = row.at::<f32>(3)?;
            let obj = DetectedObject {
                class: 1,
                confidence: *conf,
                bounding_box: Rect {
                    left: x / inp_w as f32,
                    top: y / inp_h as f32,
                    width: w / inp_w as f32,
                    height: h / inp_w as f32,
                },
            };
            self.emb.align_crop(&input, &row, &mut aligned_face1)?;
            self.emb.feature(&aligned_face1, &mut feature);
            let feature_vec = feature.data_typed::<f32>()?.to_vec();
            result.push((obj, feature_vec));
        }
        Ok(result)
    }
}
