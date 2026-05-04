use ort::{session::Session, value::Tensor};
use std::error::Error;
use std::path::Path;

pub struct Prediction {
    pub predicted: i32,
    pub actual: Option<i32>,
}

pub fn predict_batch(
    model_path: &Path,
    features: &[Vec<f64>],
    labels: &[i32],
) -> Result<Vec<Prediction>, Box<dyn Error>> {
    if features.is_empty() {
        return Ok(vec![]);
    }
    if labels.len() != features.len() {
        return Err(format!(
            "预测特征和标签数量不匹配: features={}, labels={}",
            features.len(),
            labels.len()
        )
        .into());
    }

    let feature_count = features[0].len();
    if feature_count == 0 {
        return Err("预测特征维度为 0".into());
    }
    if features.iter().any(|row| row.len() != feature_count) {
        return Err("预测特征维度不一致".into());
    }

    let mut session = Session::builder()?.commit_from_file(model_path)?;
    let flat_features: Vec<f32> = features
        .iter()
        .flat_map(|row| row.iter().map(|v| *v as f32))
        .collect();
    let input = Tensor::from_array(([features.len(), feature_count], flat_features))?;
    let outputs = session.run(ort::inputs!["float_input" => input])?;
    let label_value = outputs
        .get("label")
        .or_else(|| outputs.get("output_label"))
        .ok_or("ONNX 模型缺少 label/output_label 输出")?;
    let labels_output = label_value.try_extract_array::<i64>()?;

    if labels_output.len() != features.len() {
        return Err(format!(
            "预测结果数量与特征数量不匹配: preds={}, features={}",
            labels_output.len(),
            features.len()
        )
        .into());
    }

    Ok(labels_output
        .iter()
        .enumerate()
        .map(|(i, predicted)| Prediction {
            predicted: *predicted as i32,
            actual: labels.get(i).copied(),
        })
        .collect())
}
