use futures_util::StreamExt;
use lapin::{Connection, ConnectionProperties, Channel, options::{QueueDeclareOptions, BasicConsumeOptions, BasicAckOptions}, types::FieldTable};
use sqlx::{Postgres, Pool};

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
    Ok(channel)
}

pub async fn create_consumer(conn: Connection,
                             queue_name: &str,
                             consumer_name: &str,
                             pool: Pool<Postgres>,
                             table_name: String,
) -> anyhow::Result<()> {
    let channel = conn.create_channel().await?;
    let mut consumer = channel.basic_consume(
        queue_name,
        consumer_name,
        BasicConsumeOptions::default(),
        FieldTable::default()).await?;
    tokio::spawn(async move {
        while let Some(delivery) = consumer.next().await {
            let delivery = delivery.expect("error in consumer");
            // let new =
            // let ids = save_captured_embeddings_to_db(new, &pool, &table_name).await?;
            println!("Consumed!");
            delivery
                .ack(BasicAckOptions::default())
                .await
                .expect("ack");
            println!("Acknowledged!");
        }
    });
    Ok(())
}
