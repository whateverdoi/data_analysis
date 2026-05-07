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
    pub total_return_pct: f64,
    pub total_trades: usize,
}

#[derive(Debug, Clone)]
pub struct LabelMetrics {
    pub total: usize,
    #[allow(dead_code)]
    pub correct: usize,
    pub recall: f64,       // 召回率：P(pred=L | true=L)
    pub precision: f64,    // 精确率：P(true=L | pred=L)
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

    let total_return_pct = (equity - initial_capital) / initial_capital * 100.0;

    BacktestMetrics {
        total_predictions: total,
        correct_predictions: correct,
        accuracy,
        label_metrics,
        sharpe_ratio: sharpe,
        max_drawdown_pct: max_dd * 100.0,
        equity_curve,
        total_return_pct,
        total_trades: 0,
    }
}

fn label_metrics(predictions: &[Prediction]) -> BTreeMap<i32, LabelMetrics> {
    // recall: grouped by actual label
    let mut recall_counts: BTreeMap<i32, (usize, usize)> = BTreeMap::new();
    // precision: grouped by predicted label
    let mut precision_counts: BTreeMap<i32, (usize, usize)> = BTreeMap::new();

    for prediction in predictions {
        let Some(actual) = prediction.actual else { continue; };

        let r = recall_counts.entry(actual).or_insert((0, 0));
        r.0 += 1;
        if prediction.predicted == actual { r.1 += 1; }

        let p = precision_counts.entry(prediction.predicted).or_insert((0, 0));
        p.0 += 1;
        if actual == prediction.predicted { p.1 += 1; }
    }

    let mut result = BTreeMap::new();
    for (label, (total, correct)) in recall_counts {
        let recall = if total > 0 { correct as f64 / total as f64 } else { 0.0 };
        let (pred_total, pred_correct) = precision_counts.get(&label).copied().unwrap_or((0, 0));
        let precision = if pred_total > 0 { pred_correct as f64 / pred_total as f64 } else { 0.0 };
        result.insert(
            label,
            LabelMetrics { total, correct, recall, precision },
        );
    }
    // also add labels that appear only in predictions (not in actual)
    for (label, (pred_total, pred_correct)) in precision_counts {
        if !result.contains_key(&label) {
            result.insert(
                label,
                LabelMetrics { total: 0, correct: 0, recall: 0.0, precision: if pred_total > 0 { pred_correct as f64 / pred_total as f64 } else { 0.0 } },
            );
        }
    }
    result
}

fn atr_sma(closes: &[f64], highs: &[f64], lows: &[f64], index: usize, period: usize) -> Option<f64> {
    if index < period { return None; }
    let mut sum = 0.0;
    for i in (index + 1 - period)..=index {
        let hi = highs.get(i)?;
        let lo = lows.get(i)?;
        let pc = closes.get(i - 1)?;
        sum += (hi - lo).max((hi - pc).abs()).max((lo - pc).abs());
    }
    Some(sum / period as f64)
}

pub fn evaluate_triple_barrier(
    predictions: &[Prediction],
    test_indices: &[usize],
    closes: &[f64],
    highs: &[f64],
    lows: &[f64],
    initial_capital: f64,
    prediction_to_signal: fn(i32) -> i32,
    atr_period: usize,
    upper_mult: f64,
    lower_mult: f64,
    max_hold: usize,
) -> BacktestMetrics {
    let total = predictions.len();
    let correct = predictions
        .iter()
        .filter(|p| p.actual == Some(p.predicted))
        .count();
    let label_metrics = label_metrics(predictions);
    let accuracy = if total > 0 { correct as f64 / total as f64 } else { 0.0 };

    let mut equity_curve = vec![initial_capital];
    let mut equity = initial_capital;
    let n_bars = closes.len();
    let mut occupied_until: Option<usize> = None;
    let mut total_trades: usize = 0;

    for (i, pred) in predictions.iter().enumerate() {
        let entry_idx = test_indices.get(i).copied().unwrap_or(0);

        if let Some(until) = occupied_until {
            if entry_idx <= until {
                equity_curve.push(equity);
                continue;
            }
        }

        if prediction_to_signal(pred.predicted) == 1 {
            let entry_price = closes[entry_idx];
            let atr = atr_sma(closes, highs, lows, entry_idx, atr_period);

            if let Some(atr) = atr {
                let upper = entry_price + upper_mult * atr;
                let lower = entry_price - lower_mult * atr;
                let end = n_bars - 1;
                let limit = (entry_idx + max_hold).min(end);

                let mut exit_return = 0.0;
                let mut exit_idx = limit;
                for j in (entry_idx + 1)..=limit {
                    let hi = highs[j];
                    let lo = lows[j];
                    if hi >= upper {
                        exit_return = upper / entry_price - 1.0;
                        exit_idx = j;
                        break;
                    }
                    if lo <= lower {
                        exit_return = lower / entry_price - 1.0;
                        exit_idx = j;
                        break;
                    }
                    if j == limit {
                        exit_return = closes[j] / entry_price - 1.0;
                    }
                }
                equity *= 1.0 + exit_return;
                occupied_until = Some(exit_idx);
                total_trades += 1;
            }
        }
        equity_curve.push(equity);
    }

    let total_return_pct = (equity - initial_capital) / initial_capital * 100.0;

    let returns: Vec<f64> = equity_curve
        .windows(2)
        .map(|w| (w[1] - w[0]) / w[0])
        .collect();

    let sharpe = if returns.len() > 1 {
        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (returns.len() - 1) as f64;
        let std_dev = variance.sqrt();
        if std_dev > 0.0 {
            mean / std_dev * (252.0_f64).sqrt()
        } else { 0.0 }
    } else { 0.0 };

    let mut peak = initial_capital;
    let mut max_dd = 0.0;
    for &e in &equity_curve {
        if e > peak { peak = e; }
        let dd = (peak - e) / peak;
        if dd > max_dd { max_dd = dd; }
    }

    BacktestMetrics {
        total_predictions: total,
        correct_predictions: correct,
        accuracy,
        label_metrics,
        sharpe_ratio: sharpe,
        max_drawdown_pct: max_dd * 100.0,
        equity_curve,
        total_return_pct,
        total_trades,
    }
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
        assert_eq!(metrics.label_metrics[&1].recall, 0.0);
        assert_eq!(metrics.label_metrics[&2].total, 2);
        assert_eq!(metrics.label_metrics[&2].correct, 1);
    }
}
