use crate::indicators::engine::IndicatorSpec;
use crate::MarketData;
use std::collections::BTreeMap;

pub type LabelFn = fn(&LabelContext<'_>) -> Option<i32>;
pub type PredictionToSignalFn = fn(i32) -> i32;

#[derive(Clone)]
pub struct ExperimentSpec {
    pub window_size: usize,
    pub indicators: Vec<IndicatorSpec>,
    pub label_fn: LabelFn,
    pub prediction_to_signal: PredictionToSignalFn,
}

pub struct LabelContext<'a> {
    pub index: usize,
    market_data: &'a MarketData,
}

impl<'a> LabelContext<'a> {
    pub fn new(index: usize, market_data: &'a MarketData) -> Self {
        Self { index, market_data }
    }

    #[allow(dead_code)]
    pub fn open(&self, index: usize) -> Option<f64> {
        self.market_data.opens.get(index).copied()
    }

    #[allow(dead_code)]
    pub fn high(&self, index: usize) -> Option<f64> {
        self.market_data.highs.get(index).copied()
    }

    #[allow(dead_code)]
    pub fn low(&self, index: usize) -> Option<f64> {
        self.market_data.lows.get(index).copied()
    }

    pub fn close(&self, index: usize) -> Option<f64> {
        self.market_data.closes.get(index).copied()
    }

    #[allow(dead_code)]
    pub fn volume(&self, index: usize) -> Option<f64> {
        self.market_data.volumes.get(index).copied()
    }

    #[allow(dead_code)]
    pub fn indicator(&self, index: usize, indicator_index: usize) -> Option<f64> {
        self.market_data
            .indicator_values
            .get(index)?
            .get(indicator_index)
            .copied()
            .flatten()
    }
}

pub struct FeatureDataset {
    pub features: Vec<Vec<f64>>,
    pub labels: Vec<i32>,
    pub feature_indices: Vec<usize>,
    pub feature_count: usize,
}

pub struct FeatureBuildReport {
    pub features_before_clean: usize,
    pub rows_with_nan: usize,
    pub skipped_by_label: usize,
    pub label_counts: BTreeMap<i32, usize>,
}

pub fn build_feature_dataset(
    market_data: &MarketData,
    spec: &ExperimentSpec,
) -> Option<(FeatureDataset, FeatureBuildReport)> {
    let feature_count = spec.window_size * spec.indicators.len();
    if spec.window_size == 0 || spec.indicators.is_empty() || market_data.len() < spec.window_size {
        return None;
    }

    let mut features = Vec::new();
    let mut labels = Vec::new();
    let mut feature_indices = Vec::new();
    let mut rows_with_nan = 0;
    let mut skipped_by_label = 0;
    let mut features_before_clean = 0;

    for idx in (spec.window_size - 1)..market_data.len() {
        let label_ctx = LabelContext::new(idx, market_data);
        let Some(label) = (spec.label_fn)(&label_ctx) else {
            skipped_by_label += 1;
            continue;
        };

        let mut feature_vec = Vec::with_capacity(feature_count);
        for bar_idx in (idx + 1 - spec.window_size)..=idx {
            let row = &market_data.indicator_values[bar_idx];
            feature_vec.extend(row.iter().map(|value| value.unwrap_or(0.0)));
        }

        features_before_clean += 1;
        if feature_vec.iter().any(|value| value.is_nan()) {
            rows_with_nan += 1;
            continue;
        }

        features.push(feature_vec);
        labels.push(label);
        feature_indices.push(idx);
    }

    if features.is_empty() {
        return None;
    }

    let mut label_counts = BTreeMap::new();
    for &label in &labels {
        *label_counts.entry(label).or_insert(0) += 1;
    }

    Some((
        FeatureDataset {
            features,
            labels,
            feature_indices,
            feature_count,
        },
        FeatureBuildReport {
            features_before_clean,
            rows_with_nan,
            skipped_by_label,
            label_counts,
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_market_data() -> MarketData {
        MarketData {
            opens: vec![1.0, 2.0, 3.0, 4.0],
            highs: vec![1.5, 2.5, 3.5, 4.5],
            lows: vec![0.5, 1.5, 2.5, 3.5],
            closes: vec![1.0, 2.0, 3.0, 4.0],
            volumes: vec![10.0, 20.0, 30.0, 40.0],
            timestamps: vec!["0".into(), "1".into(), "2".into(), "3".into()],
            indicator_values: vec![
                vec![Some(10.0), Some(20.0)],
                vec![Some(11.0), Some(21.0)],
                vec![Some(12.0), Some(22.0)],
                vec![Some(13.0), Some(23.0)],
            ],
        }
    }

    fn label_next_close(ctx: &LabelContext<'_>) -> Option<i32> {
        let current = ctx.close(ctx.index)?;
        let next = ctx.close(ctx.index + 1)?;
        Some(if next > current { 1 } else { 0 })
    }

    #[test]
    fn label_context_returns_none_out_of_range() {
        let market_data = fixture_market_data();
        let ctx = LabelContext::new(2, &market_data);

        assert_eq!(ctx.open(2), Some(3.0));
        assert_eq!(ctx.high(2), Some(3.5));
        assert_eq!(ctx.low(2), Some(2.5));
        assert_eq!(ctx.close(2), Some(3.0));
        assert_eq!(ctx.volume(2), Some(30.0));
        assert_eq!(ctx.close(99), None);
        assert_eq!(ctx.indicator(1, 0), Some(11.0));
        assert_eq!(ctx.indicator(1, 99), None);
    }

    #[test]
    fn builds_windowed_features_and_indices() {
        let market_data = fixture_market_data();
        let spec = ExperimentSpec {
            window_size: 2,
            indicators: vec![IndicatorSpec::sma("a", 2), IndicatorSpec::sma("b", 2)],
            label_fn: label_next_close,
            prediction_to_signal: default_prediction_to_signal,
        };

        let (dataset, report) = build_feature_dataset(&market_data, &spec).unwrap();

        assert_eq!(dataset.feature_count, 4);
        assert_eq!(dataset.features.len(), 2);
        assert_eq!(dataset.features[0], vec![10.0, 20.0, 11.0, 21.0]);
        assert_eq!(dataset.feature_indices, vec![1, 2]);
        assert_eq!(dataset.labels, vec![1, 1]);
        assert_eq!(report.skipped_by_label, 1);
    }

    #[test]
    fn skips_nan_rows() {
        let mut market_data = fixture_market_data();
        market_data.indicator_values[0][0] = Some(f64::NAN);
        let spec = ExperimentSpec {
            window_size: 2,
            indicators: vec![IndicatorSpec::sma("a", 2), IndicatorSpec::sma("b", 2)],
            label_fn: label_next_close,
            prediction_to_signal: default_prediction_to_signal,
        };

        let (dataset, report) = build_feature_dataset(&market_data, &spec).unwrap();

        assert_eq!(report.rows_with_nan, 1);
        assert_eq!(dataset.feature_indices, vec![2]);
    }

    fn default_prediction_to_signal(label: i32) -> i32 {
        if label == 1 {
            1
        } else {
            0
        }
    }
}
