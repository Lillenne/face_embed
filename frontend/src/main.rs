#![allow(non_snake_case)]

use std::num::NonZeroU32;

use dioxus::prelude::*;
use face_embed::{
    embedding::ArcFace,
    face_detector::{UltrafaceDetector, UltrafaceDetectorConfig},
    messaging::*,
};
use log::LevelFilter;

mod canvas;

#[derive(Clone, Routable, Debug, PartialEq)]
enum Route {
    #[route("/")]
    Form {},
    // TODO Visualize page with plotters
}

fn main() {
    dioxus_logger::init(LevelFilter::Debug).expect("failed to init logger");
    console_error_panic_hook::set_once();

    launch(App);
}

fn App() -> Element {
    rsx! {
        Router::<Route> {}
    }
}

#[component]
fn Form() -> Element {
    let mut err = use_signal_sync(String::new);
    let mut form_data = use_signal_sync(MyFormData::new);
    let uploaded = move |evt: FormEvent| async move {
        let mut data = form_data.write();
        if let Some(engine) = evt.files() {
            for name in engine.files() {
                if let Some(contents) = engine.read_file(&name).await {
                    data.files.push(ImageFile { name, contents });
                }
            }
        };
    };

    rsx! {
        if !err().is_empty() {
            p { "Error: {err}" }
        }

        h1 { "Sign-up" }
        form { class: "my-form",
            onsubmit: move |event| {
                async move {
                    let mut mt = form_data.write();
                    mt.name = event.values()["name"][0].clone();
                    mt.email = event.values()["email"][0].clone();
                    drop(mt);
                    if let Err(e) = submit(form_data()).await {
                        err.set(format!("{:?}", e))
                    };
                }
            },
            label { r#for: "name", "Name" }
            input { name: "name", placeholder: "Your name...", required: "true" },
            label { r#for: "email", "Email" }
            input { r#type: "email", name: "email", placeholder: "my.email@emailprovider.com",
                pattern: ".+@.+\\..+",
                required: "true" },
            span {
                label { r#for: "image-upload", style: "margin: 8px 10px 8px 0px; padding: 12px 0", "Images" }
                input {
                    id: "image-upload",
                    r#type: "file",
                    accept: ".jpg,.jpeg,.png",
                    multiple: true,
                    name: "image-upload",
                    required: "true",
                    onchange: uploaded
                }
            }
            input { r#type: "submit", value: "Confirm" },
        }

        div { style: "display: flex; flex-direction: row; flex-wrap: wrap",
            for upload in form_data().files {
                img { src: upload.data_url(), height: 300, width:300 }
            }
        }
    }
}

// #[server]
async fn submit(data: MyFormData) -> Result<String, ServerFnError> {
    let f: String = format!("{} {} {}", data.name, data.email, data.files.len());
    let cfg = UltrafaceDetectorConfig {
        top_k: NonZeroU32::new(1).unwrap(),
        ..Default::default()
    };
    if let Ok(uf) = UltrafaceDetector::new(cfg, "") {
        if let Ok(af) = ArcFace::new("") {
            let face = uf.
        } else {
            return Err(ServerFnError::new("Failed to create embedder"));
        }
    } else {
        return Err(ServerFnError::new("Failed to create face detector"));
    }
    Ok(f)
}
