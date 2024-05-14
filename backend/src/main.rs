use std::sync::Arc;

// sample https://github.com/tokio-rs/axum/blob/main/examples/sqlx-postgres/src/main.rs
use axum::{
    extract::{DefaultBodyLimit, Multipart, State},
    http::StatusCode,
    routing::{get, post},
    Router,
};
use face_embed::{
    db::{Database, User},
    embedding::ArcFace,
    face_detector::{ModelDims, UltrafaceDetector},
    messaging::Messenger,
    pipeline::{Detector, LabelPublisher},
    storage::get_or_create_bucket,
};
use tokio::net::TcpListener;
use tracing::{error, info, warn};

type BoxResult<T> = Result<T, Box<dyn std::error::Error>>;

#[tokio::main]
async fn main() -> BoxResult<()> {
    tracing_subscriber::fmt::init();

    let detector = UltrafaceDetector::new(Default::default(), "./models/ultraface-int8.onnx")?;
    let generator = ArcFace::new("./models/arcface-int8.onnx")?;
    let (_, _, dh, dw) = generator.dims();
    let detector = Detector::new(Box::new(detector), dw, dh);
    // TODO sync constants, extract to env var / cli
    const QUEUE: &str = "sign-ups";
    let messenger = Messenger::new("amqp://127.0.0.1:5672/%2f", QUEUE.into()).await?;
    let db = Database::new("postgres://postgres:postgres@localhost:5432", 5).await?;
    const OBJECT_STORAGE_DEFAULT_URL: &str = "http://127.0.0.1:9000";
    const OBJECT_STORAGE_DEFAULT_ACCESS_KEY: &str = "minioadmin";
    const OBJECT_STORAGE_DEFAULT_SECRET_KEY: &str = "minioadmin";
    const OBJECT_STORAGE_DEFAULT_BUCKET: &str = QUEUE;
    let bucket = get_or_create_bucket(
        OBJECT_STORAGE_DEFAULT_BUCKET,
        OBJECT_STORAGE_DEFAULT_URL.into(),
        OBJECT_STORAGE_DEFAULT_ACCESS_KEY,
        OBJECT_STORAGE_DEFAULT_SECRET_KEY,
    )
    .await?;
    let publisher = Arc::new(LabelPublisher::new(
        detector,
        Box::new(generator),
        db,
        messenger,
        bucket,
    ));

    let state = AppState { publisher };
    let app = Router::new()
        .route("/", get(get_home))
        .route("/upload", post(sign_up))
        .layer(DefaultBodyLimit::max(10 * 1024 * 1024 /* 10 mb */))
        .with_state(state);

    let addr = std::env::var("BACKEND_ADDRESS").unwrap_or("127.0.0.1:3000".to_owned());
    let listener = TcpListener::bind(addr).await?;
    tracing::debug!("Listening on {}", listener.local_addr()?);
    axum::serve(listener, app).await?;
    Ok(())
}

#[derive(Clone)]
struct AppState {
    publisher: Arc<LabelPublisher>,
}

async fn get_home() -> Result<String, (StatusCode, String)> {
    Ok("my home screen".into())
}

async fn sign_up(
    State(state): State<AppState>,
    mut multipart: Multipart,
) -> Result<String, (StatusCode, String)> {
    // TODO publish event, process elsewhere
    let mut imgs: Vec<Vec<u8>> = vec![];
    let mut name: Option<String> = None;
    let mut email: Option<String> = None;
    while let Ok(Some(field)) = multipart.next_field().await {
        if let Some(field_name) = field.name() {
            match field_name {
                "name" => {
                    name = if let Ok(nm) = field.text().await {
                        Some(nm)
                    } else {
                        None
                    }
                }
                "email" => {
                    email = if let Ok(em) = field.text().await {
                        Some(em)
                    } else {
                        None
                    }
                }
                "image-upload" => imgs.push(field.bytes().await.expect("Expected image").to_vec()),
                _ => warn!("Unknown field"),
            }
        }
    }

    if name.is_none() || email.is_none() || imgs.is_empty() {
        info!("Dropped invalid request");
        return Err((
            StatusCode::BAD_REQUEST,
            "Invalid or missing name, email, or images.".to_owned(),
        ));
    }
    let user = User {
        id: 0,
        name: name.unwrap(),
        email: email.unwrap(),
    };
    info!(
        "Received sign up request: name={}, email={}",
        user.name, user.email
    );
    if let Err(e) = state.publisher.publish_from_files(user, imgs).await {
        error!("Error {:?}", e);
        Err((StatusCode::BAD_REQUEST, "Sign-up failed!".into()))
    } else {
        Ok("Sign-up complete!".into())
    }
}
