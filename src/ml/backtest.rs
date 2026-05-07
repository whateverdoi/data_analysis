use crate::ml::predict::Prediction;
use std::collections::BTreeMap;

#[derive(Debug, Clone)]
pub struct BacktestMetrics {
    pub total_predictions: usize,
    #[allow(dead_code)]
    pub correct_predictions: usize,
    pub accuracy: f64,
    pub label_metrics: BTreeMap<i32, LabelMetrics>,
    pub sharpe_ratio: f64,
    pub max_drawdown_pct: f64,
    pub equity_curve: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct LabelMetrics {
    pub total: usize,
    #[allow(dead_code)]
    pub correct: usize,
    pub accuracy: f64,
}

/// 评估模型预测的回测性能
/// prices 应该与 predictions 长度相同
pub fn evaluate(
    predictions: &[Prediction],
    prices: &[f64],
    initial_capital: f64,
    prediction_to_signal: fn(i32) -> i32,
) -> BacktestMetrics {
    let total = predictions.len();
    let correct = predictions
        .iter()
        .filter(|p| p.actual == Some(p.predicted))
        .count();

    let label_metrics = label_metrics(predictions);

    let accuracy = if total > 0 {
        correct as f64 / total as f64
    } else {
        0.0
    };

    let mut equity_curve = vec![initial_capital];
    let mut equity = initial_capital;

    // 验证数据对齐
    if predictions.len() != prices.len() {
        eprintln!(
            "警告: predictions数量({})与prices数量({})不匹配，这会导致回测结果不准确",
            predictions.len(),
            prices.len()
        );
    }

    for (i, pred) in predictions.iter().enumerate() {
        if prediction_to_signal(pred.predicted) == 1 && i + 1 < prices.len() {
            let price_return = (prices[i + 1] - prices[i]) / prices[i];
            equity *= 1.0 + price_return;
        }
        equity_curve.push(equity);
    }

    let returns: Vec<f64> = equity_curve
        .windows(2)
        .map(|w| (w[1] - w[0]) / w[0])
        .collect();

    let sharpe = if returns.len() > 1 {
        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance =
            returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (returns.len() - 1) as f64;
        let std_dev = variance.sqrt();
        if std_dev > 0.0 {
            mean / std_dev * (252.0_f64).sqrt()
        } else {
            0.0
        }
    } else {
        0.0
    };

    let mut peak = initial_capital;
    let mut max_dd = 0.0;
    for &e in &equity_curve {
        if e > peak {
            peak = e;
        }
        let dd = (peak - e) / peak;
        if dd > max_dd {
            max_dd = dd;
        }
    }

    BacktestMetrics {
        total_predictions: total,
        correct_predictions: correct,
        accuracy,
        label_metrics,
        sharpe_ratio: sharpe,
        max_drawdown_pct: max_dd * 100.0,
        equity_curve,
    }
}

fn label_metrics(predictions: &[Prediction]) -> BTreeMap<i32, LabelMetrics> {
    let mut counts: BTreeMap<i32, (usize, usize)> = BTreeMap::new();
    for prediction in predictions {
        let Some(actual) = prediction.actual else {
            continue;
        };
        let entry = counts.entry(actual).or_insert((0, 0));
        entry.0 += 1;
        if prediction.predicted == actual {
            entry.1 += 1;
        }
    }

    counts
        .into_iter()
        .map(|(label, (total, correct))| {
            let accuracy = if total > 0 {
                correct as f64 / total as f64
            } else {
                0.0
            };
            (
                label,
                LabelMetrics {
                    total,
                    correct,
                    accuracy,
                },
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn long_only(label: i32) -> i32 {
        if label == 2 {
            1
        } else {
            0
        }
    }

    #[test]
    fn reports_multiclass_label_metrics() {
        let predictions = vec![
            Prediction {
                predicted: 0,
                actual: Some(0),
            },
            Prediction {
                predicted: 2,
                actual: Some(1),
            },
            Prediction {
                predicted: 2,
                actual: Some(2),
            },
            Prediction {
                predicted: 1,
                actual: Some(2),
            },
        ];
        let metrics = evaluate(&predictions, &[1.0, 1.1, 1.2, 1.3], 100.0, long_only);

        assert_eq!(metrics.correct_predictions, 2);
        assert_eq!(metrics.label_metrics[&0].total, 1);
        assert_eq!(metrics.label_metrics[&1].accuracy, 0.0);
        assert_eq!(metrics.label_metrics[&2].total, 2);
        assert_eq!(metrics.label_metrics[&2].correct, 1);
    }
}
