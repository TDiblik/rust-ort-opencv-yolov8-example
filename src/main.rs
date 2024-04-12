use ndarray::{s, Array, ArrayBase, Axis, Dim, IxDynImpl, OwnedRepr};
use opencv::core::{Mat, MatTraitConst};
use opencv::imgcodecs::IMREAD_COLOR;
use opencv::imgproc::INTER_AREA;
use ort::{inputs, SessionOutputs};
use std::path::Path;

// Default for yolov(s) afaik
const MODEL_WH: i32 = 640;
#[rustfmt::skip]
const YOLOV8_CLASS_LABELS: [&str; 80] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
	"fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
	"bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
	"sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
	"wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
	"carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
	"tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
	"book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
];

const CONFIDENCE_TO_MATCH: f32 = 0.25; // 0 - 1
fn main() -> anyhow::Result<()> {
    // Running without ANY settings, tweak config based on your needs using the ort docs.
    ort::init().commit()?;
    let model = ort::Session::builder()?
        .commit_from_file(Path::new(env!("CARGO_MANIFEST_DIR")).join("yolov8s.onnx"))?; // replace with your own model

    // Load image in BGR
    let mut original_image = opencv::imgcodecs::imread(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("test.jpg")
            .into_os_string()
            .into_string()
            .unwrap()
            .as_str(),
        IMREAD_COLOR,
    )?;

    // scaling factors
    let fx = original_image.cols() as f64 / MODEL_WH as f64;
    let fy = original_image.rows() as f64 / MODEL_WH as f64;

    let mut resized_image: Mat = Mat::default();
    opencv::imgproc::resize(
        &original_image,
        &mut resized_image,
        opencv::core::Size::new(MODEL_WH, MODEL_WH),
        fx,
        fy,
        INTER_AREA, // This depends on how you trained your model. You can't go (that) wrong with INTER_AREA tho.
    )?;

    // Pixels are in B G R and we need to pass it as R G B, that's why we use [2], [1], [0]
    // alternatively you could convert to RGB beforehand and feed it in the correct [0], [1], [2]
    let mut model_input = Array::zeros((1, 3, MODEL_WH as usize, MODEL_WH as usize));
    for x in 0..MODEL_WH {
        for y in 0..MODEL_WH {
            let pixel = resized_image.at_2d::<opencv::core::Vec3b>(x, y)?;
            model_input[[0, 0, x as usize, y as usize]] = pixel[2] as f32 / 255.;
            model_input[[0, 1, x as usize, y as usize]] = pixel[1] as f32 / 255.;
            model_input[[0, 2, x as usize, y as usize]] = pixel[0] as f32 / 255.;
        }
    }

    // Run inference
    let outputs: SessionOutputs = model.run(inputs!["images" => model_input]?)?;
    let output = outputs["output0"]
        .try_extract_tensor::<f32>()?
        .t()
        .into_owned();

    // Parse results
    let bb_results = output_into_bb_results(
        output,
        original_image.cols() as f32,
        original_image.rows() as f32,
        MODEL_WH as f32,
        CONFIDENCE_TO_MATCH,
    );
    dbg!(&bb_results);

    // Draw bounding boxes
    let red_color = opencv::core::Scalar::new(0., 0., 255.0, 0.);
    for bb in bb_results {
        let w = bb.0.x2 - bb.0.x1;
        let h = bb.0.y2 - bb.0.y1;
        opencv::imgproc::rectangle(
            &mut original_image,
            opencv::core::Rect::new(
                bb.0.x1 as i32, // x center
                bb.0.y1 as i32, // y center
                w as i32,
                h as i32,
            ),
            red_color,
            2,
            opencv::imgproc::LINE_8,
            0,
        )?;
    }

    opencv::highgui::imshow("Rust ort + opencv + yolov8 example!", &original_image)?;
    opencv::highgui::wait_key(100_000)?;

    Ok(())
}

#[derive(Debug, Clone, Copy)]
pub struct BoundingBox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
}

#[inline(always)]
fn bb_intersection(box1: &BoundingBox, box2: &BoundingBox) -> f32 {
    (box1.x2.min(box2.x2) - box1.x1.max(box2.x1)) * (box1.y2.min(box2.y2) - box1.y1.max(box2.y1))
}

#[inline(always)]
fn bb_union(box1: &BoundingBox, box2: &BoundingBox) -> f32 {
    ((box1.x2 - box1.x1) * (box1.y2 - box1.y1)) + ((box2.x2 - box2.x1) * (box2.y2 - box2.y1))
        - bb_intersection(box1, box2)
}

pub fn output_into_bb_results<'a>(
    output: ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>>,
    original_img_width: f32,
    original_img_height: f32,
    img_wh: f32,
    accepted_prob: f32, // 0 - 1
) -> Vec<(BoundingBox, &'a str, f32)> {
    let fx = original_img_width / img_wh;
    let fy = original_img_height / img_wh;

    let mut boxes = Vec::new();
    let output = output.slice(s![.., .., 0]);
    for row in output.axis_iter(Axis(0)) {
        let row: Vec<f32> = row.iter().copied().collect();
        let (class_id, prob) = row
            .iter()
            .skip(4)
            .enumerate()
            .map(|(index, value)| (index, *value))
            .reduce(|accum, row| if row.1 > accum.1 { row } else { accum })
            .unwrap();

        if prob < accepted_prob {
            continue;
        }

        let label = YOLOV8_CLASS_LABELS[class_id];
        let xc = row[0] * fx;
        let yc = row[1] * fy;
        let w = row[2] * fx;
        let h = row[3] * fy;

        boxes.push((
            BoundingBox {
                x1: xc - w / 2.,
                y1: yc - h / 2.,
                x2: xc + w / 2.,
                y2: yc + h / 2.,
            },
            label,
            prob,
        ));
    }

    let mut bb_results = Vec::new();
    boxes.sort_by(|box1, box2| box2.2.total_cmp(&box1.2));
    while !boxes.is_empty() {
        let f = boxes[0];
        bb_results.push(f);
        boxes.retain(|box1| bb_intersection(&f.0, &box1.0) / bb_union(&f.0, &box1.0) < 0.7);
    }

    bb_results
}
