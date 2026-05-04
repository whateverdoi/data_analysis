use serde::Deserialize;
use std::error::Error;
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Debug, Clone, Deserialize)]
pub struct TrainingMetrics {
    pub train_accuracy: f64,
    pub train_time_ms: u64,
}

pub fn write_feature_csv(
    path: &Path,
    features: &[Vec<f64>],
    labels: &[i32],
) -> Result<(), Box<dyn Error>> {
    if features.len() != labels.len() {
        return Err(format!(
            "特征和标签数量不匹配: features={}, labels={}",
            features.len(),
            labels.len()
        )
        .into());
    }

    let Some(first_row) = features.first() else {
        return Err("特征数据为空，无法写入 CSV".into());
    };
    let feature_count = first_row.len();
    if features.iter().any(|row| row.len() != feature_count) {
        return Err("特征维度不一致，无法写入 CSV".into());
    }

    let mut writer = csv::Writer::from_path(path)?;
    let mut header: Vec<String> = (0..feature_count).map(|i| format!("f{}", i)).collect();
    header.push("label".to_string());
    writer.write_record(&header)?;

    for (row, label) in features.iter().zip(labels.iter()) {
        let mut record: Vec<String> = row.iter().map(|v| v.to_string()).collect();
        record.push(label.to_string());
        writer.write_record(&record)?;
    }

    writer.flush()?;
    Ok(())
}

pub fn train_python_random_forest(
    python_bin: &Path,
    train_csv: &Path,
    model_output: &Path,
    metrics_output: &Path,
    n_trees: u32,
) -> Result<TrainingMetrics, Box<dyn Error>> {
    let script_path = train_script_path();
    let output = Command::new(python_bin)
        .arg(&script_path)
        .arg("--train-csv")
        .arg(train_csv)
        .arg("--model-output")
        .arg(model_output)
        .arg("--metrics-output")
        .arg(metrics_output)
        .arg("--n-trees")
        .arg(n_trees.to_string())
        .output()
        .map_err(|err| {
            format!(
                "启动 Python 训练失败: python={}, script={}, error={}",
                python_bin.display(),
                script_path.display(),
                err
            )
        })?;

    if !output.status.success() {
        return Err(format!(
            "Python 训练失败: status={}\nstdout:\n{}\nstderr:\n{}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        )
        .into());
    }

    let metrics_text = std::fs::read_to_string(metrics_output)?;
    let metrics = serde_json::from_str::<TrainingMetrics>(&metrics_text)?;
    Ok(metrics)
}

fn train_script_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("scripts/train_model.py")
}
