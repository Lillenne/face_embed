use futures_lite::stream::StreamExt;
use lapin::options::BasicAckOptions;
use lettre::transport::smtp::authentication::Credentials;
use lettre::{
    message::{header::ContentType, Mailbox},
    Transport,
};
use lettre::{Message, SmtpTransport};
use mini_moka::sync::Cache;
use std::{env::var, time::Duration};
use tracing::{error, info};

use face_embed::{
    db::Database,
    messaging::{DetectionEvent, LabelEvent, Messenger},
};
type BoxResult<T> = Result<T, Box<dyn std::error::Error>>;

#[tokio::main(flavor = "multi_thread", worker_threads = 2)]
async fn main() -> BoxResult<()> {
    tracing_subscriber::fmt().init();
    let sign = tokio::spawn(async move {
        let _ = consume_sign_up_events().await;
    });
    let live = tokio::spawn(async move {
        let _ = consume_live_events().await;
    });
    sign.await?;
    live.await?;
    Ok(())
}

async fn consume_live_events() -> BoxResult<()> {
    let cache = Cache::builder()
        .time_to_live(Duration::from_secs(60 * 60 /* 1 hr */))
        .build();
    let from_email: Mailbox = var("EMAIL_SENDER")?.parse()?;
    let subject = var("EMAIL_SUBJECT")?;
    let creds = Credentials::new(var("EMAIL_USERNAME")?, var("EMAIL_PASSWORD")?);
    let messenger = Messenger::new(&var("BUS_ADDRESS")?, var("LIVE_QUEUE_NAME")?).await?;
    let mut consumer = messenger.get_consumer("live-consumer", None, None).await?;
    while let Some(Ok(delivery)) = consumer.next().await {
        let data: DetectionEvent = rmp_serde::from_slice(delivery.data.as_slice())?;
        info!("Received detection event: {:?}", &data);
        let Some(user) = data.user else {
            info!("Unidentified user.");
            delivery.ack(BasicAckOptions::default()).await?;
            continue;
        };
        if cache.get(&user.id).is_some() {
            info!(
                "Received duplicate event in time frame for user {}. Skipping.",
                user.name
            );
            delivery.ack(BasicAckOptions::default()).await?;
            continue;
        }
        let Ok(addr) = user.email.parse() else {
            error!("Failed to parse address: {}.", user.email);
            continue;
        };
        let to = Mailbox::new(Some(user.name.clone()), addr);
        let Ok(email) = Message::builder()
            .from(from_email.clone())
            .to(to)
            .subject(subject.clone())
            .header(ContentType::TEXT_PLAIN)
            .body(format!("Hi {}, you were here today!", user.name))
        else {
            error!("Failed to make email");
            continue;
        };

        // Open a remote connection to gmail
        let mailer = SmtpTransport::relay("smtp.gmail.com")?
            .credentials(creds.clone())
            .build();

        // Send the email
        match mailer.send(&email) {
            Ok(_) => {
                cache.insert(user.id, 0_u8);
                delivery.ack(BasicAckOptions::default()).await?;
            }
            Err(e) => error!("Error: {}", e),
        }
    }
    Ok(())
}

async fn consume_sign_up_events() -> BoxResult<()> {
    let db = Database::new(var("DB_CONN_STR")?.as_str(), 5).await?;
    let sign_up_messenger =
        Messenger::new(&var("BUS_ADDRESS")?, var("SIGN_UP_QUEUE_NAME")?).await?;
    let live_messenger = Messenger::new(&var("BUS_ADDRESS")?, var("LIVE_QUEUE_NAME")?).await?;
    let threshold: f32 = match var("SIMILARITY_THRESHOLD") {
        Ok(str_val) => str_val.parse().unwrap_or(0.3),
        Err(_) => 0.3,
    };
    let mut consumer = sign_up_messenger
        .get_consumer("sign-up-consumer", None, None)
        .await?;

    while let Some(Ok(delivery)) = consumer.next().await {
        let data: LabelEvent = rmp_serde::from_slice(delivery.data.as_slice())?;
        info!(
            "Received label event: id={}, name={}",
            data.event_id, data.user.name
        );
        let events = db.update_past_captures(data.user, threshold).await?;
        if !events.is_empty() {
            info!(
                "Event {} matched {} existing detections",
                data.event_id,
                events.len()
            );
            for event in events {
                let ser = rmp_serde::to_vec_named(&event)?;
                live_messenger.publish(ser.as_slice()).await?;
            }
        }
        delivery.ack(BasicAckOptions::default()).await?;
    }
    Ok(())
}
