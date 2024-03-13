use std::path::PathBuf;

use crate::*;


fn get_paths(glob: &str) -> anyhow::Result<impl Iterator<Item = PathBuf>> {
    let paths = glob::glob(glob)?;
    Ok(paths .filter(|p| p.is_ok()) .map(|p| p.unwrap()))
}

fn labels_from_dirs(glob: &str) -> Vec<String> {
    glob::glob(glob).unwrap()
    .map(|p| {
        let n = p.unwrap() ;
        n.as_path()
            .parent().unwrap()
        .to_string_lossy().rsplit_terminator('/').next().unwrap().to_owned()
    })
    .collect::<Vec<_>>()
}

async fn generate_embeddings_from_file(args: &EmbedArgs) -> anyhow::Result<()> {
    let pool = setup_sqlx(args.database.conn_str.as_str(), args.database.table_name.as_str()).await?;
    let arcface = Box::new(ArcFace::new(args.arcface_path.as_str())?);
    let embedding_dims = arcface.dims();
    println!("Embedding model generated.");

    let det = create_face_detector(&args.ultraface_path)?;
    let det_dims = det.dims();
    println!("Detection model generated.");

    let aw = det_dims.3.get();
    let ah = det_dims.2.get();
    for entry in glob::glob("/home/aus/Projects/Rust/lfw/**/*.jpg").unwrap() {
        let path = entry.unwrap();
        println!("Generating embedding for {:?}", path);
        let name = path.as_path().file_name().unwrap();
        let name = name.to_string_lossy().into_owned();
        // let class = classes.iter().enumerate().filter(|(i,c)| name.contains(c.as_str())).next().unwrap();
        let bind = image::open(path.to_str().unwrap()).unwrap();
        let img = bind.as_rgb8().unwrap();
        let rsz = image::imageops::resize(img, aw, ah, image::imageops::Triangle);
        let n = path.file_stem().unwrap().to_str().unwrap();
        let rect = det.detect(rsz.as_bytes())?[0].bounding_box.to_crop_box(img.width(), img.height());
        println!("{rect:?}");
        let mut vec = bind.as_rgb8().unwrap().to_vec();
        let region =
            fr::Image::from_slice_u8(NonZeroU32::new(img.width()).unwrap(), NonZeroU32::new(img.height()).unwrap(),
                                     vec.as_mut_slice(), fr::PixelType::U8x3).unwrap();
        let roi = face_embed::crop_and_resize(
            &mut (region.view()),
            embedding_dims.3,
            embedding_dims.2,
            rect,
            fr::ResizeAlg::Convolution(fr::FilterType::CatmullRom)
        )?;
        let embedding = arcface.generate_embedding(roi.buffer())?;
        let embedding = Vector::from(embedding);
        // sqlx::query("INSERT INTO lfwa (embedding, class_idx, class_name) VALUES ($1, $2, $3)")
        //     .bind(embedding)
        //     .bind(class.0 as i64)
        //     .bind(class.1.as_str())
        //     .execute(&pool)
        //     .await?;
    }
    // visualize_lfw_embeddings().await?;
    Ok(())
}


// async fn generate_embeddings_from_prealigned_images() -> anyhow::Result<()> {
//     let classes = glob::glob("/home/aus/Projects/Rust/lfw/aligned/**/*.jpg").unwrap()
//         .map(|p| {
//             let n = p.unwrap() ;
//             n.as_path()
//              .parent().unwrap()
//             .to_string_lossy().rsplit_terminator('/').next().unwrap().to_owned()
//         })
//         .collect::<Vec<_>>();
//     println!("{classes:?}");

//     let pool = setup_sqlx_lfw().await?;
//     let arcface = Box::new(ArcFace::new(ARCFACE_PATH)?);
//     let embedding_dims = arcface.dims();
//     println!("Embedding model generated.");

//     for entry in glob::glob("/home/aus/Projects/Rust/lfw/aligned/**/*.jpg").unwrap() {
//         let path = entry.unwrap();
//         println!("Generating embedding for {:?}", path);
//         let name = path.as_path().file_name().unwrap();
//         let name = name.to_string_lossy().into_owned();
//         let class = classes.iter().enumerate().filter(|(i,c)| name.contains(c.as_str())).next().unwrap();
//         let bind = image::open(path.to_str().unwrap()).unwrap();
//         let img = bind.as_rgb8().unwrap();
//         let n = path.file_stem().unwrap().to_str().unwrap();
//         let v = img.to_vec();
//         let embedding = arcface.generate_embedding(v.as_slice())?;
//         let embedding = Vector::from(embedding);
//         sqlx::query("INSERT INTO lfwa (embedding, class_idx, class_name) VALUES ($1, $2, $3)")
//             .bind(embedding)
//             .bind(class.0 as i64)
//             .bind(class.1.as_str())
//             .execute(&pool)
//             .await?;
//     }
//     // visualize_lfw_embeddings().await?;
//     Ok(())
// }
