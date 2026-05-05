use crate::ml::predict::Prediction;

#[derive(Debug, Clone)]
pub struct BacktestMetrics {
    pub total_predictions: usize,
    #[allow(dead_code)]
    pub correct_predictions: usize,
    pub accuracy: f64,
    pub up_accuracy: f64,
    pub down_accuracy: f64,
    pub sharpe_ratio: f64,
    pub max_drawdown_pct: f64,
    pub equity_curve: Vec<f64>,
}

/// 评估模型预测的回测性能
/// prices 应该与 predictions 长度相同
pub fn evaluate(
    predictions: &[Prediction],
    prices: &[f64],
    initial_capital: f64,
) -> BacktestMetrics {
    let total = predictions.len();
    let correct = predictions
        .iter()
        .filter(|p| p.actual == Some(p.predicted))
        .count();

    let up_total = predictions.iter().filter(|p| p.actual == Some(1)).count();
    let up_correct = predictions
        .iter()
        .filter(|p| p.actual == Some(1) && p.predicted == 1)
        .count();

    let down_total = predictions.iter().filter(|p| p.actual == Some(0)).count();
    let down_correct = predictions
        .iter()
        .filter(|p| p.actual == Some(0) && p.predicted == 0)
        .count();

    let accuracy = if total > 0 {
        correct as f64 / total as f64
    } else {
        0.0
    };
    let up_accuracy = if up_total > 0 {
        up_correct as f64 / up_total as f64
    } else {
        0.0
    };
    let down_accuracy = if down_total > 0 {
        down_correct as f64 / down_total as f64
    } else {
        0.0
    };

    let mut equity_curve = vec![initial_capital];
    let mut equity = initial_capital;
    
    // 验证数据对齐
    if predictions.len() != prices.len() {
        eprintln!("警告: predictions数量({})与prices数量({})不匹配，这会导致回测结果不准确", 
            predictions.len(), prices.len());
    }
    
    for (i, pred) in predictions.iter().enumerate() {
        if pred.predicted == 1 && i + 1 < prices.len() {
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
        up_accuracy,
        down_accuracy,
        sharpe_ratio: sharpe,
        max_drawdown_pct: max_dd * 100.0,
        equity_curve,
    }
}
