use pgvector::Vector;
use serde::{Deserialize, Serialize};
use sqlx::{postgres::PgPoolOptions, Postgres, Pool, prelude::FromRow, types::chrono::{DateTime, Utc}, QueryBuilder, Row};

use crate::cache::EmbeddingRef;

#[derive(FromRow, Debug, Clone)]
pub struct EmbeddingData {
    pub id: i64,
    pub embedding: Vector,
    pub time: DateTime<Utc>,
    pub class_id: Option<i64>
}

#[derive(Debug, Clone)]
pub struct EmbeddingTime {
    pub embedding: Vector,
    pub time: DateTime<Utc>,
}

impl EmbeddingRef<f32> for EmbeddingTime {
    fn embedding_ref(&self) -> &[f32] {
        self.embedding.as_slice()
    }
}

pub async fn setup_sqlx(conn: &str, max_conn: u32, table: &str) -> anyhow::Result<Pool<Postgres>> {
    let pool = PgPoolOptions::new()
        .max_connections(max_conn)
        .connect(conn)
        .await?;
    sqlx::query("CREATE EXTENSION IF NOT EXISTS vector")
        .execute(&pool)
        .await?;
    sqlx::query!("CREATE TABLE IF NOT EXISTS users (id BIGSERIAL PRIMARY KEY, name VARCHAR(40), email VARCHAR(40))").execute(&pool).await?;
    sqlx::query!("CREATE TABLE IF NOT EXISTS classes (id BIGSERIAL PRIMARY KEY, signature VECTOR(512), user_id BIGINT REFERENCES users(id))").execute(&pool).await?; // labels table
    let query = format!("CREATE TABLE IF NOT EXISTS {} (id BIGSERIAL PRIMARY KEY, embedding VECTOR(512) NOT NULL, time TIMESTAMPTZ NOT NULL, class_id BIGINT REFERENCES classes(id) NULL)", table);
    sqlx::query(query.as_str())
        .execute(&pool)
        .await?;
    sqlx::query!("CREATE INDEX ON items USING hnsw (embedding vector_ip_ops)").execute(&pool).await?; // use hnsw index since requires no data
    Ok(pool)
}

pub async fn create_pool_if_table_exists(conn_str: &str, max_conn: u32, table_name: &str) -> anyhow::Result<Pool<Postgres>> {
    let pool = PgPoolOptions::new()
        .max_connections(max_conn)
        .connect(conn_str)
        .await?;
    let exists_query = format!("SELECT EXISTS (
    SELECT FROM
        pg_tables
    WHERE
        schemaname = 'public' AND
        tablename  = '{}'
    );", table_name);
    if let Some(_) = sqlx::query(&exists_query)
        .fetch_optional(&pool)
        .await? {
            Ok(pool)
        } else {
            Err(anyhow::anyhow!("Table not created"))
        }
}

pub async fn save_captured_embeddings_to_db<T: IntoIterator<Item = EmbeddingTime>>(values: T, pool: &Pool<Postgres>, table_name: &str) -> anyhow::Result<Vec<i64>> {
    let query_init = format!("INSERT INTO {} (embedding, time)", table_name);
    let mut qb: QueryBuilder<Postgres> = QueryBuilder::new(query_init);
    qb.push_values(values, |mut b, et| {
        b.push_bind(et.embedding).push_bind(et.time);
    });
    qb.push("RETURNING id");
    let ids: Vec<i64> = qb.build()
        .fetch_all(pool)
        .await?
        .iter()
        .map(|r| r.get("id"))
        .collect();
    Ok(ids)
}

pub async fn get_label(id: i64, table_name: &str, pool: &Pool<Postgres>, thresh: f32) -> anyhow::Result<Option<User>> {
    let query = format!("
SELECT * FROM (SELECT ((classes.signature <#> (SELECT embedding FROM {} WHERE id = {})) * -1) as similarity, users.*
FROM classes JOIN users on classes.user_id=users.id
WHERE classes.user_id IS NOT NULL
ORDER BY similarity)
WHERE similarity > {};
", table_name, id, thresh);
    let row = sqlx::query_as::<_, User>(&query)
        .fetch_optional(pool)
        .await?;
    Ok(row)
}

#[derive(FromRow, Serialize, Deserialize, Debug)]
pub struct User {
    pub id: i64,
    pub name: String,
    pub email: String,
    pub similarity: f64,
}
