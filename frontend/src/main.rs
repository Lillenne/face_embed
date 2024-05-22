#![allow(non_snake_case)]

use base64::prelude::*;
use std::fmt::Display;

use dioxus::prelude::*;
use tracing::Level;

mod canvas;

#[derive(Clone, Routable, Debug, PartialEq)]
enum Route {
    #[route("/")]
    Form {},
}

fn main() {
    dioxus_logger::init(Level::INFO).expect("Failed to init logger");
    // console_error_panic_hook::set_once();

    launch(App);
}

fn App() -> Element {
    rsx! {
        Router::<Route> {}
    }
}

#[component]
fn Form() -> Element {
    let mut form_data = use_signal(MyFormData::new);
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

    // let mut ep_sig = use_signal(String::new);
    // let mut endpt = std::env::var("BACKEND_ADDRESS").unwrap_or_default();
    let mut endpt = "http://127.0.0.1:3000".to_owned();
    endpt.push_str("/upload");
    // ep_sig.write().push_str(&endpt);

    rsx! {
        h1 { "Sign-up" }
        form { class: "my-form",
            action: endpt,
            method: "post",
            enctype: "multipart/form-data",
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

#[derive(Clone, Debug)]
pub struct MyFormData {
    pub name: String,
    pub email: String,
    pub files: Vec<ImageFile>,
}

impl MyFormData {
    pub fn new() -> Self {
        MyFormData {
            name: String::new(),
            email: String::new(),
            files: vec![],
        }
    }
}

impl Default for MyFormData {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct ImageFile {
    pub name: String,
    pub contents: Vec<u8>,
}

impl ImageFile {
    pub fn data_url(&self) -> Option<String> {
        let ext = self.name.split('.').last()?;
        let mut b64 = String::new();
        let lower = ext.to_lowercase();
        b64.push_str("data:image/");
        if lower.ends_with("jpg") || lower.ends_with("jpeg") {
            b64.push_str("jpeg");
        } else if lower.ends_with("png") {
            b64.push_str("png");
        } else {
            return None;
        }
        b64.push_str(";base64,");
        let encoded = BASE64_STANDARD.encode(&self.contents);
        b64.push_str(&encoded);
        Some(b64)
    }
}

impl Display for ImageFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.name)
    }
}
