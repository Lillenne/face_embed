use futures_util::StreamExt;
use lapin::{Connection, ConnectionProperties, Channel, options::{QueueDeclareOptions, BasicConsumeOptions, BasicAckOptions}, types::FieldTable};
use serde::{Deserialize, Serialize};
use sqlx::types::chrono::{DateTime, Utc};
use tokio::task::JoinHandle;

use crate::db::User;

pub async fn create_queue(address: &str, queue_name: &str) -> anyhow::Result<Channel> {
    let conn = Connection::connect(
        address,
        ConnectionProperties::default(),
    )
    .await?;

    let channel = conn.create_channel().await?;
    _ = channel
                .queue_declare(
                    queue_name,
                    QueueDeclareOptions::default(),
                    FieldTable::default(),
                )
                .await?;

    let _ = create_consumer(&conn, queue_name, "consumer").await?;
    Ok(channel)
}

pub async fn create_consumer(conn: &Connection,
                             queue_name: &str,
                             consumer_name: &str
) -> anyhow::Result<JoinHandle<anyhow::Result<()>>> {
    let channel = conn.create_channel().await?;
    let mut consumer = channel.basic_consume(
        queue_name,
        consumer_name,
        BasicConsumeOptions::default(),
        FieldTable::default()).await?;
    let task: JoinHandle<anyhow::Result<()>> = tokio::spawn(async move {
        while let Some(delivery) = consumer.next().await {
            let delivery = delivery.expect("error in consumer");
            let event: Event = rmp_serde::from_slice(&delivery.data)?;
            delivery
                .ack(BasicAckOptions::default())
                .await
                .expect("ack");
            println!("Consumed {:?}", &event);
        }
        Ok(())
    });
    Ok(task)
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Event {
    pub id: i64,
    pub time: DateTime<Utc>,
    pub path: String,
    pub user: Option<User>,
}
