use pgvector::Vector;
use serde::{Deserialize, Serialize};
use sqlx::{
    postgres::PgPoolOptions,
    prelude::FromRow,
    types::chrono::{DateTime, Utc},
    Pool, Postgres, QueryBuilder, Row,
};

use crate::cache::EmbeddingRef;

#[derive(FromRow, Serialize, Deserialize, Debug)]
pub struct User {
    pub id: i64,
    pub name: String,
    pub email: String,
    pub similarity: f64,
}

#[derive(FromRow, Debug, Clone)]
pub struct EmbeddingData {
    pub id: i64,
    pub embedding: Vector,
    pub time: DateTime<Utc>,
    pub class_id: Option<i64>,
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

pub struct Database {
    pool: Pool<Postgres>,
}

impl Database {
    pub async fn new(conn: &str, max_conn: u32) -> anyhow::Result<Self> {
        let pool = PgPoolOptions::new()
            .max_connections(max_conn)
            .connect(conn)
            .await?;
        sqlx::query!("CREATE EXTENSION IF NOT EXISTS vector")
            .execute(&pool)
            .await?;
        sqlx::query!("CREATE TABLE IF NOT EXISTS users (id BIGSERIAL PRIMARY KEY, name VARCHAR(40), email VARCHAR(40))").execute(&pool).await?;
        sqlx::query!("CREATE TABLE IF NOT EXISTS classes (id BIGSERIAL PRIMARY KEY, signature VECTOR(512), user_id BIGINT REFERENCES users(id))").execute(&pool).await?; // labels table
        sqlx::query!("CREATE TABLE IF NOT EXISTS items (id BIGSERIAL PRIMARY KEY, embedding VECTOR(512) NOT NULL, time TIMESTAMPTZ NOT NULL, class_id BIGINT REFERENCES classes(id) NULL)").execute(&pool).await?;
        sqlx::query!(
            "CREATE INDEX IF NOT EXISTS idx ON items USING hnsw (embedding vector_ip_ops)"
        )
        .execute(&pool)
        .await?; // use hnsw index since requires no data
        Ok(Self { pool })
    }

    pub async fn save_captured_embeddings_to_db<T: IntoIterator<Item = EmbeddingTime>>(
        &self,
        values: T,
    ) -> anyhow::Result<Vec<i64>> {
        let query_init = "INSERT INTO items (embedding, time)";
        let mut qb: QueryBuilder<Postgres> = QueryBuilder::new(query_init);
        qb.push_values(values, |mut b, et| {
            b.push_bind(et.embedding).push_bind(et.time);
        });
        qb.push("RETURNING id");
        let ids: Vec<i64> = qb
            .build()
            .fetch_all(&self.pool)
            .await?
            .iter()
            .map(|r| r.get("id"))
            .collect();
        Ok(ids)
    }

    pub async fn get_label(&self, id: i64, thresh: f32) -> anyhow::Result<Option<User>> {
        let query = format!("
    SELECT * FROM (SELECT ((classes.signature <#> (SELECT embedding FROM items WHERE id = {})) * -1) as similarity, users.*
    FROM classes JOIN users on classes.user_id=users.id
    WHERE classes.user_id IS NOT NULL
    ORDER BY similarity)
    WHERE similarity > {};
    ", id, thresh);
        let row = sqlx::query_as::<_, User>(&query)
            .fetch_optional(&self.pool)
            .await?;
        Ok(row)
    }

    pub async fn insert_label(
        &self,
        name: String,
        email: String,
        signature: Vector,
    ) -> anyhow::Result<i64> {
        let record = sqlx::query!(
            "
            WITH row AS 
                (INSERT INTO users (name, email) 
                VALUES ($1, $2) RETURNING id)
            INSERT INTO classes (signature, user_id)
            SELECT $3, id
            FROM row
            returning user_id
            ",
            name,
            email,
            signature as _
        )
        .fetch_one(&self.pool)
        .await?;
        if let Some(id) = record.user_id {
            Ok(id)
        } else {
            Err(anyhow::anyhow!("Failed to return new user id"))
        }
    }
}
