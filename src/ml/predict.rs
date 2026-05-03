use smartcore::{
    ensemble::random_forest_classifier::RandomForestClassifier,
    linalg::basic::matrix::DenseMatrix,
};

pub struct Prediction {
    pub predicted: i32,
    pub actual: Option<i32>,
}

pub fn predict_batch(
    model: &RandomForestClassifier<f64, i32, DenseMatrix<f64>, Vec<i32>>,
    features: &[Vec<f64>],
    labels: &[i32],
) -> Vec<Prediction> {
    if features.is_empty() {
        return vec![];
    }
    let features_vec: Vec<Vec<f64>> = features.to_vec();
    let x = DenseMatrix::from_2d_vec(&features_vec).expect("构建特征矩阵失败");
    let preds = model.predict(&x).expect("预测失败");

    features
        .iter()
        .enumerate()
        .map(|(i, _)| {
            let actual = labels.get(i).copied();
            Prediction {
                predicted: preds[i],
                actual,
            }
        })
        .collect()
}
