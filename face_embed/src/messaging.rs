use lapin::{
    options::{BasicConsumeOptions, BasicPublishOptions, QueueDeclareOptions},
    types::FieldTable,
    BasicProperties, Channel, Connection, ConnectionProperties, Consumer,
};
use serde::{Deserialize, Serialize};
use sqlx::types::chrono::{DateTime, Utc};
use uuid::Uuid;

use crate::db::User;

#[derive(Serialize, Deserialize, Debug)]
pub struct CaptureEvent {
    pub id: Uuid,
    pub time: DateTime<Utc>,
    pub path: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct DetectionEvent {
    pub id: i64,
    pub time: DateTime<Utc>,
    pub path: String,
    pub user: Option<User>,
    pub similarity: f32,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct LabelEvent {
    pub event_id: i64,
    pub user: User,
    pub signatures: Vec<Vec<f32>>,
}

#[derive(Clone)]
pub struct Messenger {
    channel: Channel,
    queue_name: String,
}

impl Messenger {
    pub async fn new_connection(address: &str, queue_name: String) -> anyhow::Result<Self> {
        let conn = Connection::connect(address, ConnectionProperties::default()).await?;
        let channel = conn.create_channel().await?;

        Ok(Messenger {
            channel,
            queue_name,
        })
    }

    pub async fn new_channel(conn: &Connection, queue_name: String) -> anyhow::Result<Self> {
        let channel = conn.create_channel().await?;
        Ok(Messenger {
            channel,
            queue_name,
        })
    }

    pub async fn from_channel(channel: Channel, queue_name: String) -> Self {
        Messenger {
            channel,
            queue_name,
        }
    }

    pub async fn queue_declare(&self) -> anyhow::Result<()> {
        let _queue = self
            .channel
            .queue_declare(
                &self.queue_name,
                QueueDeclareOptions::default(),
                FieldTable::default(),
            )
            .await?;
        Ok(())
    }

    pub async fn publish(&self, payload: &[u8]) -> anyhow::Result<()> {
        self.channel
            .basic_publish(
                "",
                &self.queue_name,
                BasicPublishOptions::default(),
                payload,
                BasicProperties::default(),
            )
            .await?
            .await?;
        Ok(())
    }

    pub async fn get_consumer(&self, consumer_tag: &str) -> Result<Consumer, lapin::Error> {
        self.channel
            .basic_consume(
                &self.queue_name,
                consumer_tag,
                BasicConsumeOptions::default(),
                FieldTable::default(),
            )
            .await
    }
}
