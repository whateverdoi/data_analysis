use crate::data::market_data::MarketData;
use crate::indicators::specs::IndicatorSpec;
#[allow(unused_imports)]
use crate::indicators::specs::IndicatorKind;
use crate::labels::label_fns::{LabelContext, LabelFn, PredictionToSignalFn};
use std::collections::BTreeMap;

const NUM_AGG_STATS: usize = 9;

#[derive(Clone)]
pub struct ExperimentSpec {
    pub window_size: usize,
    pub indicators: Vec<IndicatorSpec>,
    pub label_fn: LabelFn,
    pub prediction_to_signal: PredictionToSignalFn,
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

fn agg_stats(values: &[f64]) -> Vec<f64> {
    let n = values.len() as f64;
    let sum: f64 = values.iter().sum();
    let mean = sum / n;
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    let std = variance.sqrt();

    let last = *values.last().unwrap_or(&0.0);
    let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = max - min;

    let slope = if n > 1.0 {
        let x_sum = (n - 1.0) * n / 2.0;
        let x2_sum = (n - 1.0) * n * (2.0 * n - 1.0) / 6.0;
        let xy_sum: f64 = values.iter().enumerate().map(|(i, &v)| i as f64 * v).sum();
        (n * xy_sum - x_sum * sum) / (n * x2_sum - x_sum * x_sum)
    } else {
        0.0
    };

    let mut sorted: Vec<f64> = values.to_vec();
    sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = if sorted.is_empty() {
        0.0
    } else {
        let mid = sorted.len() / 2;
        if sorted.len() % 2 == 0 {
            (sorted[mid - 1] + sorted[mid]) / 2.0
        } else {
            sorted[mid]
        }
    };

    let skewness = if std > 0.0 {
        let m3 = values.iter().map(|v| (v - mean).powi(3)).sum::<f64>() / n;
        m3 / (std * std * std)
    } else {
        0.0
    };

    vec![last, mean, std, slope, min, max, median, range, skewness]
}

pub fn build_feature_dataset(
    market_data: &MarketData,
    spec: &ExperimentSpec,
) -> Option<(FeatureDataset, FeatureBuildReport)> {
    let num_indicators = spec.indicators.len();
    let feature_count = num_indicators * NUM_AGG_STATS;
    if spec.window_size == 0 || num_indicators == 0 || market_data.len() < spec.window_size {
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

        for ind_idx in 0..num_indicators {
            let mut raw = Vec::with_capacity(spec.window_size);
            for bar_idx in (idx + 1 - spec.window_size)..=idx {
                raw.push(
                    market_data.indicator_values[bar_idx]
                        .get(ind_idx)
                        .copied()
                        .flatten()
                        .unwrap_or(0.0),
                );
            }
            let has_nan = raw.iter().any(|v| v.is_nan());
            if has_nan {
                feature_vec.clear();
                break;
            }
            feature_vec.extend(agg_stats(&raw));
        }

        if feature_vec.is_empty() {
            rows_with_nan += 1;
            continue;
        }

        features_before_clean += 1;
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

    fn agg_for(values: &[f64]) -> Vec<f64> {
        agg_stats(values)
    }

    #[test]
    fn builds_aggregated_features() {
        let market_data = fixture_market_data();
        let spec = ExperimentSpec {
            window_size: 2,
            indicators: vec![
                IndicatorSpec::new("a", IndicatorKind::Sma { period: 2 }),
                IndicatorSpec::new("b", IndicatorKind::Sma { period: 2 }),
            ],
            label_fn: label_next_close,
            prediction_to_signal: default_prediction_to_signal,
        };

        let (dataset, report) = build_feature_dataset(&market_data, &spec).unwrap();

        assert_eq!(dataset.feature_count, 18);
        assert_eq!(dataset.features.len(), 2);
        // indicator A at idx=1: [10.0, 11.0]
        let a_stats = agg_for(&[10.0, 11.0]);
        // indicator B at idx=1: [20.0, 21.0]
        let b_stats = agg_for(&[20.0, 21.0]);
        let expected: Vec<f64> = a_stats.into_iter().chain(b_stats).collect();
        assert_eq!(dataset.features[0], expected);
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
            indicators: vec![
                IndicatorSpec::new("a", IndicatorKind::Sma { period: 2 }),
                IndicatorSpec::new("b", IndicatorKind::Sma { period: 2 }),
            ],
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
