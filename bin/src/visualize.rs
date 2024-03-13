use std::str::FromStr;

use crate::{VisualizeArgs, DatabaseArgs};
use linfa::{Dataset, traits::{Transformer, Fit}};
use linfa_reduction::Pca;
use linfa_tsne::TSneParams;
use ndarray::Axis;
use pgvector::Vector;
use plotters::prelude::*;
use sqlx::{Postgres, Pool, postgres::PgPoolOptions, Row};


pub(crate) async fn visualize(args: VisualizeArgs, v: bool) -> anyhow::Result<()> {
    let pool = query_sqlx(&args.database).await?;
    let query = format!("
SELECT {}.class_idx, embedding, n
FROM (SELECT class_idx, count(*) as n
FROM {}
GROUP BY class_idx) q
JOIN {} on {}.class_idx = q.class_idx
ORDER BY n DESC
LIMIT {};
", &args.database.table_name, &args.database.table_name, &args.database.table_name, &args.database.table_name, args.limit);
    let nearest = sqlx::query(&query)
        .fetch_all(&pool)
        .await? ;
    let embeddings = nearest
        .iter()
        .take(args.limit)
        .flat_map(|row| {
            let emb: Vector = row.try_get("embedding").unwrap();
            emb.as_slice().iter().map(|v| *v as f64).collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let targets = nearest
        .iter()
        // .take(1000)
        .map(|row| {
            // let id: i64 = row.try_get("id").unwrap();
            let id: i32 = row.try_get("class_idx").unwrap();
            id as f64
        })
        .collect::<Vec<f64>>();

    let data = ndarray::Array2::from_shape_vec((embeddings.len() / 512, 512), embeddings)?;
    let targets = ndarray::Array1::from_shape_vec(targets.len(), targets)?;
    let max_idx = targets.iter().map(|v| *v).reduce(f64::max).unwrap();
    let ds = Dataset::new(data, targets);

    let mut clustered = if args.pca {
        let pca = Pca::params(args.dimensions as _).whiten(true).fit(&ds)?;
        pca.transform(ds)
    } else if args.tsne {
    TSneParams::embedding_size(2)
            .perplexity(10.0)
            .approx_threshold(0.1)
            .transform(ds)?
    } else {
        // shouldn't happen via clap
        return Err(anyhow::anyhow!("No dimensionality reduction algorithm chosen"))
    };


    let maxes = clustered.records.axis_iter(Axis(1)).map(|a| {
        let (mut min, mut max) = (f64::MAX, f64::MIN);
        for v in a {
            min = v.min(min);
            max = v.max(max);
        }
        (min, max)
    }).collect::<Vec<(f64, f64)>>();

    // normalize to 0..=1
    let mut idx: usize = 0;
    for ax in clustered.records.axis_iter_mut(Axis(1)) {
        for v in ax {
            *v = (*v - maxes[idx].0) / (maxes[idx].1 - maxes[idx].0);
        }
        idx += 1;
    }

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
            clustered.records.axis_iter(Axis(0)).map(|row| {
                let pt = (row[0], row[1]);
                let color = Palette99::pick(idx as _);
                TriangleMarker::new(pt, 5, color)
            }))?;
    } else {
        let mut chart = ChartBuilder::on(&root)
            .margin(20)
            .caption("Embeddings", ("sans-serif", 40))
            .build_cartesian_3d(0.0..1.0, 0.0..1.0, 0.0..1.0)?;
        chart.configure_axes().draw()?;

        chart.draw_series(
            clustered.records.axis_iter(Axis(0)).map(|row| {
                let pt = (row[0], row[1], row[2]);
                let color = Palette99::pick(idx as _);
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
        tablename  = {}
    );", args.table_name);
    if let Some(_) = sqlx::query(&exists_query)
        .fetch_optional(&pool)
        .await? {
            Ok(pool)
        } else {
            Err(anyhow::anyhow!("Postgres table unavailable"))
        }
}
