use std::collections::VecDeque;

pub struct FeatureBuilder {
    window_size: usize,
    num_features_per_bar: usize,
    bar_buffer: VecDeque<Vec<f64>>,
    close_buffer: VecDeque<f64>,
    features: Vec<Vec<f64>>,
    labels: Vec<i32>,
}

impl FeatureBuilder {
    pub fn new(window_size: usize, num_features_per_bar: usize) -> Self {
        Self {
            window_size,
            num_features_per_bar,
            bar_buffer: VecDeque::with_capacity(window_size),
            close_buffer: VecDeque::with_capacity(window_size + 1),
            features: Vec::new(),
            labels: Vec::new(),
        }
    }

    pub fn push(&mut self, feat: Vec<f64>, close_current: f64, close_next: f64) {
        self.bar_buffer.push_back(feat);
        self.close_buffer.push_back(close_current);
        if self.bar_buffer.len() > self.window_size {
            self.bar_buffer.pop_front();
            self.close_buffer.pop_front();
        }
        if self.bar_buffer.len() == self.window_size {
            let label = if close_next > close_current { 1 } else { 0 };
            let mut feature_vec = Vec::with_capacity(self.window_size * self.num_features_per_bar);
            for idx in 0..self.window_size {
                feature_vec.extend_from_slice(&self.bar_buffer[idx]);
            }
            self.features.push(feature_vec);
            self.labels.push(label);
        }
    }

    pub fn feature_count(&self) -> usize {
        self.window_size * self.num_features_per_bar
    }

    pub fn take_features(&self) -> Vec<Vec<f64>> {
        debug_assert_eq!(
            self.features.len(),
            self.labels.len(),
            "特征({})和标签({})数量不一致",
            self.features.len(),
            self.labels.len()
        );
        self.features.clone()
    }

    pub fn take_labels(&self) -> Vec<i32> {
        debug_assert_eq!(
            self.features.len(),
            self.labels.len(),
            "特征({})和标签({})数量不一致",
            self.features.len(),
            self.labels.len()
        );
        self.labels.clone()
    }
}
