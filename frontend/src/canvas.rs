// use std::error::Error;
//
// use dioxus::prelude::*;
// use plotters::{
//     chart::ChartBuilder,
//     drawing::IntoDrawingArea,
//     series::LineSeries,
//     style::{FontDesc, RGBColor, RED, WHITE},
// };
// use plotters_canvas::CanvasBackend;
//
// #[component]
// fn Plot(id: &'static str, width: i64, height: i64) -> Element {
//     let mut a = use_signal(|| 1);
//     use_effect(move || write_canvas(a(), id));
//     rsx! {
//         canvas { id: id, width, height,
//                  onclick: move |_| { write_canvas(a(), id); a += 1; },
//                  onmousemove: move |_| { a += 1}
//                  // onmouseenter wasn't working
//         }
//     }
// }
//
// #[derive(PartialEq, Props, Clone)]
// pub struct VisProps {
//     id: String,
//     #[props(default = "test".into())]
//     caption: String,
//     #[props(default = WHITE)]
//     fill: RGBColor,
//     #[props(default = 200)]
//     width: i64,
//     #[props(default = 200)]
//     height: i64,
// }
//
// pub fn EmbedPlot(props: VisProps) -> Element {
//     let effect = props.clone();
//     use_effect(move || {
//         _ = write_vis(&effect);
//     });
//     rsx! {
//         canvas { id: props.id, width: props.width, height: props.height,
//         }
//     }
// }
//
// pub fn write_vis(props: &VisProps) -> Result<(), Box<dyn Error>> {
//     let backend = CanvasBackend::new(&props.id);
//     if backend.is_none() {
//         return Err("Couldn't find canvas element".into());
//     }
//     let root = backend.unwrap().into_drawing_area();
//     root.fill(&props.fill)?;
//
//     let font: FontDesc = ("sans-serif", 20.0).into();
//     let mut chart = ChartBuilder::on(&root)
//         .margin(20u32)
//         .caption(&props.caption, font)
//         .x_label_area_size(30u32)
//         .y_label_area_size(30u32)
//         .build_cartesian_2d(-1f32..1f32, -1.2f32..1.2f32)?;
//     chart.configure_mesh().x_labels(3).y_labels(3).draw()?;
//
//     // TODO make series
//
//     root.present()?;
//     Ok(())
// }
//
// // TODO visualize graph here w/ props?
// pub fn write_canvas(power: i32, id: &str) {
//     // -> DrawResult<()> {
//     let backend = CanvasBackend::new(id);
//     if let None = backend {
//         return;
//     }
//     let root = backend.unwrap().into_drawing_area();
//     let font: FontDesc = ("sans-serif", 20.0).into();
//
//     root.fill(&WHITE).unwrap();
//
//     let mut chart = ChartBuilder::on(&root)
//         .margin(20u32)
//         .caption(format!("y=x^{}", power), font)
//         .x_label_area_size(30u32)
//         .y_label_area_size(30u32)
//         .build_cartesian_2d(-1f32..1f32, -1.2f32..1.2f32)
//         .unwrap();
//
//     chart
//         .configure_mesh()
//         .x_labels(3)
//         .y_labels(3)
//         .draw()
//         .unwrap();
//
//     chart
//         .draw_series(LineSeries::new(
//             (-50..=50)
//                 .map(|x| x as f32 / 50.0)
//                 .map(|x| (x, x.powf(power as f32))),
//             &RED,
//         ))
//         .unwrap();
//
//     root.present().unwrap();
//     // return Ok(chart.into_coord_trans());
// }
