use lapin::{
    options::{BasicConsumeOptions, BasicPublishOptions, QueueDeclareOptions},
    types::FieldTable,
    BasicProperties, Channel, Connection, ConnectionProperties, Consumer,
};
use serde::{Deserialize, Serialize};
use sqlx::types::chrono::{DateTime, Utc};

use crate::db::User;

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

pub struct Messenger {
    queue_name: String,
    channel: Channel,
}

impl Messenger {
    pub async fn new(address: &str, queue_name: String) -> anyhow::Result<Self> {
        let conn = Connection::connect(address, ConnectionProperties::default()).await?;

        let channel = conn.create_channel().await?;
        _ = channel
            .queue_declare(
                &queue_name,
                QueueDeclareOptions::default(),
                FieldTable::default(),
            )
            .await?;

        Ok(Messenger {
            queue_name,
            channel,
        })
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

    pub async fn get_consumer(
        &self,
        consumer_tag: &str,
        consume_options: Option<BasicConsumeOptions>,
        field_table: Option<FieldTable>,
    ) -> Result<Consumer, lapin::Error> {
        self.channel
            .basic_consume(
                &self.queue_name,
                consumer_tag,
                consume_options.unwrap_or_default(),
                field_table.unwrap_or_default(),
            )
            .await
    }
}
