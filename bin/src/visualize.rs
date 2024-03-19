use std::str::FromStr;

use crate::DatabaseArgs;
use clap::Args;
use face_embed::{db::create_pool_if_table_exists, path_utils::expand_path};
use linfa::{Dataset, traits::{Transformer, Fit}, DatasetBase};
use linfa_preprocessing::linear_scaling::LinearScaler;
use linfa_reduction::Pca;
use linfa_tsne::TSneParams;
use ndarray::{Axis, OwnedRepr, ArrayBase, Dim};
use pgvector::Vector;
use plotters::prelude::*;
use sqlx::Row;

type DS = DatasetBase<ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>, ArrayBase<OwnedRepr<f64>, Dim<[usize; 1]>>>;

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
    database: DatabaseArgs,

    #[arg(short, long)]
    verbose: bool
}

#[derive(Args, Debug)]
#[group(required = true, multiple = false)]
struct ReductionAlg {
    #[arg(long, group = "dim_reduc")]
    pca: bool,

    #[arg(long, group = "dim_reduc")]
    tsne: bool,
}

pub(crate) async fn visualize(args: VisualizeArgs) -> anyhow::Result<()> {
    let v = args.verbose;
    if v { println!("Generating dataset...") }
    let ds = create_dataset(&args).await?;
    if v { println!("Clustering dataset...") }
    let clustered = reduce_dims(&args, ds)?;
    if v { println!("Normalizing dataset...") }
    let clustered = LinearScaler::min_max() .fit(&clustered)?.transform(clustered);
    if v { println!("Plotting...") }
    plot(args, &clustered)?;
    if v { println!("Complete.") }
    Ok(())
}

fn plot(args: VisualizeArgs, clustered: &DS) -> anyhow::Result<()> {
    let path = if args.output_path.ends_with("png") {
        args.output_path
    } else {
        let mut s = String::from_str(&args.output_path)?;
        s.push_str(".png");
        s
    };
    let root = BitMapBackend::new(&path, (1280, 960)).into_drawing_area();
    root.fill(&WHITE)?;

    if args.dimensions == 2 {
        let mut chart = ChartBuilder::on(&root)
            .margin(20)
            .caption("Embeddings", ("sans-serif", 40))
            .build_cartesian_2d(0.0..1.0, 0.0..1.0)?;
        chart.configure_mesh().draw()?;


        chart.draw_series(
            clustered.records.axis_iter(Axis(0)).enumerate().map(|(idx,row)| {
                let pt = (row[0], row[1]);
                let class = clustered.targets[idx];
                let color = Palette99::pick(class as _);
                TriangleMarker::new(pt, 5, color)
            }))?;
    } else {
        let mut chart = ChartBuilder::on(&root)
            .margin(20)
            .caption("Embeddings", ("sans-serif", 40))
            .build_cartesian_3d(0.0..1.0, 0.0..1.0, 0.0..1.0)?;
        chart.configure_axes().draw()?;

        chart.draw_series(
            clustered.records.axis_iter(Axis(0)).enumerate().map(|(idx,row)| {
                let pt = (row[0], row[1], row[2]);
                let class = clustered.targets[idx];
                let color = Palette99::pick(class as _);
                TriangleMarker::new(pt, 5, color)
            }))?;
    }
    root.present()?;
    Ok(())
}

async fn create_dataset(args: &VisualizeArgs) -> anyhow::Result<DS> {
    let pool = create_pool_if_table_exists(&args.database.conn_str, 5, &args.database.table_name).await?;
    let query = format!("
    SELECT class_id, embedding
    FROM {}
    LIMIT {}", args.database.table_name, args.limit);
    let nearest = sqlx::query(&query)
        .fetch_all(&pool)
        .await?;
    let targets = nearest
        .iter()
        .map(|r| {
            let i: Result<i64, sqlx::Error> = r.try_get("class_id");
            if let Ok(i) = i {i as f64} else { -1.0 }
        })
        .collect::<Vec<f64>>();
    let embeddings = nearest
        .into_iter()
        .flat_map(|r|{
            let v: Vector = r.try_get("embedding").unwrap();
            let v: Vec<f32> = v.into();
            v
        }).map(|f| f as f64)
        .collect::<Vec<_>>();

    let data = ndarray::Array2::from_shape_vec((embeddings.len() / 512, 512), embeddings)?;
    let targets = ndarray::Array1::from_shape_vec(targets.len(), targets)?;
   Ok(Dataset::new(data, targets))
}

fn reduce_dims(args: &VisualizeArgs, ds: DS) -> anyhow::Result<DS> {
    if args.alg.pca {
        let pca = Pca::params(args.dimensions as _).whiten(true).fit(&ds)?;
        Ok(pca.transform(ds))
    } else if args.alg.tsne {
        Ok(TSneParams::embedding_size(args.dimensions as _)
                .perplexity(10.0)
                .approx_threshold(0.0)
                .transform(ds)?)
    } else {
        // shouldn't happen via clap
        Err(anyhow::anyhow!("No dimensionality reduction algorithm chosen"))
    }
}
