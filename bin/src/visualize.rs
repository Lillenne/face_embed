use std::str::FromStr;

use crate::{VisualizeArgs, DatabaseArgs};
use linfa::{Dataset, traits::{Transformer, Fit}, DatasetBase};
use linfa_reduction::Pca;
use linfa_tsne::TSneParams;
use ndarray::{Axis, OwnedRepr, ArrayBase, Dim};
use pgvector::Vector;
use plotters::prelude::*;
use sqlx::{Postgres, Pool, postgres::PgPoolOptions, Row};

type DS = DatasetBase<ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>, ArrayBase<OwnedRepr<f64>, Dim<[usize; 1]>>>;

pub(crate) async fn visualize(args: VisualizeArgs, v: bool) -> anyhow::Result<()> {
    if v { println!("Generating dataset...") }
    let ds = create_dataset(&args).await?;
    if v { println!("Clustering dataset...") }
    let mut clustered = reduce_dims(&args, ds)?;
    normalize(&mut clustered);
    if v { println!("Plotting...") }
    plot(args, &mut clustered)?;
    if v { println!("Complete.") }
    Ok(())
}

fn plot(args: VisualizeArgs, clustered: &mut DS) -> anyhow::Result<()> {
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

async fn query_sqlx(args: &DatabaseArgs) -> anyhow::Result<Pool<Postgres>> {
    let pool = PgPoolOptions::new()
        .max_connections(5)
        .connect(&args.conn_str)
        .await?;
    let exists_query = format!("SELECT EXISTS (
    SELECT FROM
        pg_tables
    WHERE
        schemaname = 'public' AND
        tablename  = '{}'
    );", args.table_name);
    if let Some(_) = sqlx::query(&exists_query)
        .fetch_optional(&pool)
        .await? {
            Ok(pool)
        } else {
            Err(anyhow::anyhow!("Table not created"))
        }
}

fn normalize(clustered: &mut DS) {
    let maxes = clustered.records.axis_iter(Axis(1)).map(|a| {
        let (mut min, mut max) = (f64::MAX, f64::MIN);
        for v in a {
            min = v.min(min);
            max = v.max(max);
        }
        (min, max)
    }).collect::<Vec<(f64, f64)>>();

    let mut idx: usize = 0;
    for ax in clustered.records.axis_iter_mut(Axis(1)) {
        for v in ax {
            *v = (*v - maxes[idx].0) / (maxes[idx].1 - maxes[idx].0);
        }
        idx += 1;
    }
}

async fn create_dataset(args: &VisualizeArgs) -> anyhow::Result<DS> {
    let pool = query_sqlx(&args.database).await?;
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
