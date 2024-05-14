use pgvector::Vector;
use serde::{Deserialize, Serialize};
use sqlx::{
    postgres::PgPoolOptions,
    prelude::FromRow,
    types::chrono::{DateTime, Utc},
    Pool, Postgres,
};

use crate::cache::EmbeddingRef;

#[derive(FromRow, Serialize, Deserialize, Debug)]
pub struct User {
    pub id: i64,
    pub name: String,
    pub email: String,
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

    pub async fn save_captured_embedding_to_db(
        &self,
        value: &EmbeddingTime,
    ) -> anyhow::Result<i64> {
        Ok(sqlx::query!(
            "
INSERT INTO items (embedding, time) 
VALUES ($1, $2)
RETURNING id ",
            value.embedding as _,
            value.time
        )
        .fetch_one(&self.pool)
        .await?
        .id)
    }

    pub async fn save_captured_embeddings_to_db<T: IntoIterator<Item = EmbeddingTime>>(
        &self,
        values: T,
    ) -> anyhow::Result<Vec<(i64, DateTime<Utc>)>> {
        // TODO optimize...
        let mut embeds: Vec<Vector> = vec![];
        let mut times: Vec<DateTime<Utc>> = vec![];
        for item in values {
            embeds.push(item.embedding);
            times.push(item.time);
        }
        let ids = sqlx::query!(
            "
        INSERT INTO items (embedding, time)
        (SELECT * FROM UNNEST ($1::vector(512)[], $2::timestamptz[]))
        RETURNING id, time",
            &embeds as _,
            &times
        )
        .fetch_all(&self.pool)
        .await?
        .iter()
        .map(|r| (r.id, r.time))
        .collect();
        Ok(ids)
    }

    pub async fn get_label(&self, id: i64, thresh: f32) -> anyhow::Result<Option<(User, f32)>> {
        let row = sqlx::query!("
            SELECT * FROM (SELECT ((classes.signature <#> (SELECT embedding FROM items WHERE id = $1)) * -1) as similarity, users.*
            FROM classes JOIN users on classes.user_id=users.id
            WHERE classes.user_id IS NOT NULL
            ORDER BY similarity)
            WHERE similarity > $2;
        ", id, thresh as f64 // TODO should this be f32?
        ).fetch_optional(&self.pool).await?;
        if let Some(items) = row {
            let user = User {
                id: items.id,
                name: items.name.expect("Expected user name, recieved None."),
                email: items.email.expect("Expected user email, recieved None."),
            };
            Ok(Some((
                user,
                items
                    .similarity
                    .expect("Expected similarity, recieved None.") as f32,
            )))
        } else {
            Ok(None)
        }
    }

    pub async fn insert_label(
        &self,
        name: String,
        email: String,
        signatures: &[Vector],
    ) -> anyhow::Result<i64> {
        // TODO single query -- translate this to many
        // "
        // WITH row AS
        //     (INSERT INTO users (name, email)
        //     VALUES ($1, $2) RETURNING id)
        // INSERT INTO classes (signature, user_id)
        // SELECT $3, id
        // FROM row
        // returning user_id
        // ",
        let record = sqlx::query!(
            "INSERT INTO users (name, email) VALUES ($1, $2) RETURNING id",
            name,
            email
        )
        .fetch_one(&self.pool)
        .await?;
        for signature in signatures {
            _ = sqlx::query!(
                "INSERT INTO classes (signature, user_id) VALUES ($1, $2)",
                *signature as _,
                record.id
            )
            .execute(&self.pool)
            .await?;
        }
        // let record = sqlx::query!(
        //     "
        // WITH
        //     row AS (INSERT INTO users (name, email) VALUES ($1, $2) RETURNING id),
        //     sigs AS (SELECT UNNEST($3::vector(512)[]) as signature)
        // INSERT INTO classes (signature, user_id)
        // VALUES ((
        // SELECT signature
        // FROM sigs), (SELECT id from row))
        // returning user_id
        // ",
        //     name,
        //     email,
        //     signatures as _
        // );
        // INSERT INTO classes (signature, user_id)
        // if let Some(id) = record.user_id {
        //     Ok(id)
        // } else {
        //     Err(anyhow::anyhow!("Failed to return new user id"))
        // }
        Ok(record.id)
    }
}
