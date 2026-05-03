use smartcore::{
    ensemble::random_forest_classifier::RandomForestClassifier,
    linalg::basic::matrix::DenseMatrix,
};
use std::time::Instant;

pub struct TrainedModel {
    pub model: RandomForestClassifier<f64, i32, DenseMatrix<f64>, Vec<i32>>,
    pub train_accuracy: f64,
    pub train_time_ms: u64,
}

pub fn train_random_forest(
    features: &[Vec<f64>],
    labels: &[i32],
    _n_trees: u32,
) -> TrainedModel {
    let start = Instant::now();

    let features_vec: Vec<Vec<f64>> = features.to_vec();
    let labels_vec: Vec<i32> = labels.to_vec();

    let x = DenseMatrix::from_2d_vec(&features_vec).expect("构建特征矩阵失败");

    let model = RandomForestClassifier::fit(
        &x,
        &labels_vec,
        Default::default(),
    )
    .expect("训练失败");

    let predictions = model.predict(&x).expect("训练集预测失败");
    let correct = predictions
        .iter()
        .zip(labels_vec.iter())
        .filter(|(p, t)| p == t)
        .count();
    let train_accuracy = correct as f64 / labels_vec.len() as f64;
    let train_time = start.elapsed().as_millis() as u64;

    TrainedModel {
        model,
        train_accuracy,
        train_time_ms: train_time,
    }
}
