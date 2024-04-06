use futures_util::StreamExt;
use lapin::{
    options::{BasicAckOptions, BasicConsumeOptions, BasicPublishOptions, QueueDeclareOptions},
    types::FieldTable,
    BasicProperties, Channel, Connection, ConnectionProperties,
};
use serde::{Deserialize, Serialize};
use sqlx::types::chrono::{DateTime, Utc};
use tokio::task::JoinHandle;

type Handle = tokio::task::JoinHandle<anyhow::Result<()>>;

use crate::db::User;

#[derive(Serialize, Deserialize, Debug)]
pub struct Event {
    pub id: i64,
    pub time: DateTime<Utc>,
    pub path: String,
    pub user: Option<User>,
}

pub struct Messenger {
    queue_name: String,
    channel: Channel,
    consumers: Vec<Handle>,
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
            consumers: Vec::new(),
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

    pub async fn create_consumer(&mut self, consumer_name: &str) -> anyhow::Result<()> {
        let mut consumer = self
            .channel
            .basic_consume(
                &self.queue_name,
                consumer_name,
                BasicConsumeOptions::default(),
                FieldTable::default(),
            )
            .await?;
        let handle: JoinHandle<anyhow::Result<()>> = tokio::spawn(async move {
            while let Some(delivery) = consumer.next().await {
                let delivery = delivery.expect("error in consumer");
                let event: Event = rmp_serde::from_slice(&delivery.data)?;
                delivery.ack(BasicAckOptions::default()).await.expect("ack");
                println!("Consumed {:?}", &event);
            }
            Ok(())
        });
        self.consumers.push(handle);
        Ok(())
    }

    pub async fn wait_for_completion(&mut self) -> anyhow::Result<()> {
        let mut err: Option<tokio::task::JoinError> = None;
        while let Some(handle) = self.consumers.pop() {
            let res = handle.await;
            if let Err(e) = res {
                err = Some(e);
            }
        }
        if let Some(e) = err {
            Err(e.into())
        } else {
            Ok(())
        }
    }
}
