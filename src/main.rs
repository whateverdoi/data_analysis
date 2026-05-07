mod config;
mod data;
mod error;
mod features;
mod indicators;
mod labels;
mod ml;
mod plotting;

use config::Config;
use data::market_data::MarketData;
use data::reader::read_csv;
use data::writer::{create_writer, write_header, write_row};
use features::builder::{
    build_feature_dataset as build_feature_dataset_from_spec, ExperimentSpec, FeatureDataset,
};
use indicators::engine::IndicatorEngine;
use labels::label_by_name;
use log::{info, warn};
use ml::backtest::{evaluate, evaluate_triple_barrier, BacktestMetrics};
use ml::predict::predict_batch;
use ml::train::{train_python_random_forest, write_feature_csv};
use plotting::{html::render_dashboard, png::render_ohlcv};
use std::io::Write;

fn main() {
    let config = Config::load();

    println!("=== 金融时间序列数据分析 ===");
    println!("CSV 输入: {}", config.csv_input.display());
    println!("训练比例: {:.0}%", config.train_split * 100.0);
    println!("随机森林树数: {}", config.n_trees);
    println!();
    config.check_output_dirs().expect("无法创建输出目录");

    // 获取实验配置
    let (label_fn, prediction_to_signal) = label_by_name(&config.label_fn_name)
        .expect("标签函数未找到");
    let experiment = ExperimentSpec {
        window_size: config.window_size,
        indicators: config.indicator_specs.clone(),
        label_fn,
        prediction_to_signal,
    };

    // 初始化指标引擎
    let mut engine =
        IndicatorEngine::new(&config.symbol, &experiment.indicators).expect("创建指标引擎失败");
    let num_indicators = engine.num_indicators();
    let indicator_names = engine.indicator_names().to_vec();

    // 打印启用的技术指标
    print_indicator_names(num_indicators, &indicator_names);

    // 读取CSV + 计算指标 + 写入增强CSV
    let market_data = build_enhanced_csv(&config, &mut engine, &indicator_names);
    
    // 构建特征数据集
    let Some(mut feature_dataset) = build_feature_dataset(&market_data, &experiment) else {
        return;
    };
    
    // 切分训练/测试集
    let (train_features, _train_labels, test_features, test_labels, train_size) =
        split_and_write_features(&config, &mut feature_dataset);

    // 训练模型
    train_model(&config, train_features.len());
    
    // 回测
    let (metrics, pred_signals) = run_backtest(
        &config,
        &market_data,
        &feature_dataset,
        &test_features,
        &test_labels,
        train_size,
        experiment.prediction_to_signal,
    );

    // 生成可视化
    render_outputs(&config, &market_data, &metrics, &pred_signals);
    
    // 输出汇总
    print_output_summary(&config);
}

fn print_indicator_names(num_indicators: usize, indicator_names: &[String]) {
    // 输出当前启用的技术指标名称，便于核对增强 CSV 的列。
    println!("启用了 {} 个技术指标:", num_indicators);
    for name in indicator_names {
        println!("  - {}", name);
    }
    println!();
}

fn build_enhanced_csv(
    config: &Config,
    engine: &mut IndicatorEngine,
    indicator_names: &[String],
) -> MarketData {
    let mut market_data = MarketData::new();

    // 创建增强后的 CSV 文件，并先写入表头。
    let mut wtr_out = create_writer(&config.csv_output).expect("创建输出 CSV writer 失败");
    write_header(&mut wtr_out, indicator_names).expect("写 CSV 头失败");

    // 逐行读取原始 CSV：解析时间、更新指标、写出增强行，并把数据缓存到内存。
    let mut bar_count: usize = 0;
    let mut read_total: usize = 0;
    let csv_iter = read_csv(config.csv_input.to_str().unwrap());

    for result in csv_iter {
        read_total += 1;
        match result {
            Ok(row) => {
                // 指标引擎需要纳秒时间戳；解析失败时用 0 兜底。
                let ts = row.timestamp.clone();
                let dt = row.parse_timestamp();
                let ts_nanos = dt
                    .map(|d| d.and_utc().timestamp_nanos_opt().unwrap_or(0))
                    .unwrap_or(0);

                // 用当前 K 线更新所有技术指标，返回这一行对应的指标值。
                let indicator_vals =
                    engine.update(ts_nanos, row.open, row.high, row.low, row.close, row.volume);

                // 把原始行情和新算出的指标写入增强 CSV。
                write_row(
                    &mut wtr_out,
                    &ts,
                    row.open,
                    row.high,
                    row.low,
                    row.close,
                    row.volume,
                    &indicator_vals,
                )
                .expect("写 CSV 行失败");

                // 把数据留在内存中，供后续训练、回测和图表生成使用。
                market_data.opens.push(row.open);
                market_data.highs.push(row.high);
                market_data.lows.push(row.low);
                market_data.closes.push(row.close);
                market_data.volumes.push(row.volume);
                market_data.timestamps.push(ts);
                market_data.indicator_values.push(indicator_vals);

                // 每处理一万行刷新一次进度输出，避免大文件运行时看起来无响应。
                bar_count += 1;
                if bar_count % 10000 == 0 {
                    print!("\r处理进度: {} 行", bar_count);
                    std::io::stdout().flush().ok();
                }
            }
            Err(e) => {
                warn!("读取 CSV 行失败, 跳过: {}", e);
            }
        }
    }

    // 确保增强 CSV 的缓冲区全部落盘，并输出读取统计。
    wtr_out.flush().expect("刷新 CSV writer 失败");
    println!(
        "\r处理完成: 读取 {} 行, 成功 {} 行, 跳过 {} 行",
        read_total,
        bar_count,
        read_total - bar_count
    );
    println!("增强后 CSV 已保存: {}", config.csv_output.display());

    market_data
}

fn build_feature_dataset(
    market_data: &MarketData,
    experiment: &ExperimentSpec,
) -> Option<FeatureDataset> {
    // 把连续窗口内的指标序列转换成机器学习特征，并用实验规格生成标签。
    info!("特征工程");
    let Some((dataset, report)) = build_feature_dataset_from_spec(market_data, experiment) else {
        warn!("特征数据不足，无法训练");
        return None;
    };

    println!("特征计算前: {} 行", report.features_before_clean);
    println!(
        "特征清理后: {} 行 (删除了 {} 行含 NaN 数据, 删除率 {:.2}%, 标签跳过 {} 行)",
        dataset.features.len(),
        report.rows_with_nan,
        if report.features_before_clean > 0 {
            report.rows_with_nan as f64 * 100.0 / report.features_before_clean as f64
        } else {
            0.0
        },
        report.skipped_by_label
    );

    println!("窗口大小: {}", experiment.window_size);
    println!("特征维度: {}", dataset.feature_count);
    println!("label 分布:");
    for (label, count) in report.label_counts {
        println!("  label {}: {}", label, count);
    }

    Some(dataset)
}

fn split_and_write_features(
    config: &Config,
    dataset: &mut FeatureDataset,
) -> (Vec<Vec<f64>>, Vec<i32>, Vec<Vec<f64>>, Vec<i32>, usize) {
    debug_assert!(dataset.feature_count > 0);

    // 按时间顺序切分训练集和测试集，避免未来数据泄漏到训练阶段。
    let train_size = (dataset.features.len() as f64 * config.train_split) as usize;
    let test_features = dataset.features.split_off(train_size);
    let test_labels = dataset.labels.split_off(train_size);
    let train_features = std::mem::take(&mut dataset.features);
    let train_labels = std::mem::take(&mut dataset.labels);

    write_feature_csv(
        &config.train_features_output,
        &train_features,
        &train_labels,
    )
    .expect("写训练特征 CSV 失败");
    write_feature_csv(&config.test_features_output, &test_features, &test_labels)
        .expect("写测试特征 CSV 失败");

    (
        train_features,
        train_labels,
        test_features,
        test_labels,
        train_size,
    )
}

fn train_model(config: &Config, train_feature_count: usize) {
    // 使用 Python 训练随机森林，并导出给 Rust 推理使用的 ONNX 模型。
    info!("模型训练");
    let trained = train_python_random_forest(
        &config.python_bin,
        &config.train_features_output,
        &config.model_output,
        &config.train_metrics_output,
        config.n_trees,
        config.max_depth,
        config.min_samples_leaf,
    )
    .expect("Python 模型训练失败");
    println!("训练集大小: {}", train_feature_count);
    println!("训练准确率: {:.2}%", trained.train_accuracy * 100.0);
    println!("训练耗时: {}ms", trained.train_time_ms);
}

fn align_test_indices(
    dataset: &FeatureDataset,
    train_size: usize,
) -> Vec<usize> {
    dataset
        .feature_indices
        .iter()
        .skip(train_size)
        .copied()
        .collect()
}

fn is_triple_barrier(config: &Config) -> bool {
    config.label_fn_name == "triple_barrier"
}

fn run_backtest(
    config: &Config,
    market_data: &MarketData,
    dataset: &FeatureDataset,
    test_features: &[Vec<f64>],
    test_labels: &[i32],
    train_size: usize,
    prediction_to_signal: fn(i32) -> i32,
) -> (BacktestMetrics, Vec<(usize, i32)>) {
    info!("回测预测");
    let predictions =
        predict_batch(&config.model_output, test_features, test_labels).expect("ONNX 推理失败");

    let test_indices = align_test_indices(dataset, train_size);

    let metrics = if is_triple_barrier(config) {
        evaluate_triple_barrier(
            &predictions,
            &test_indices,
            &market_data.closes,
            &market_data.highs,
            &market_data.lows,
            100_000.0,
            prediction_to_signal,
            14,
            2.0,
            2.0,
            20,
        )
    } else {
        let test_prices: Vec<f64> = test_indices
            .iter()
            .map(|&idx| market_data.closes[idx])
            .collect();
        evaluate(&predictions, &test_prices, 100_000.0, prediction_to_signal)
    };
    println!("测试集预测数: {}", metrics.total_predictions);
    println!("回测准确率: {:.2}%", metrics.accuracy * 100.0);
    println!("总收益率: {:.2}%", metrics.total_return_pct);
    println!("总交易数: {}", metrics.total_trades);
    println!("各 label 分析:");
    for (label, lm) in &metrics.label_metrics {
        println!(
            "  label {}: 样本 {} | 召回率 {:.2}% (真值={}时预测正确) | 精确率 {:.2}% (预测={}时实际正确)",
            label, lm.total, lm.recall * 100.0, label, lm.precision * 100.0, label
        );
    }
    println!("夏普比率: {:.2}", metrics.sharpe_ratio);
    println!("最大回撤: {:.2}%", metrics.max_drawdown_pct);

    let pred_signals: Vec<(usize, i32)> = predictions
        .iter()
        .enumerate()
        .map(|(i, p)| {
            (
                test_indices.get(i).copied().unwrap_or(0),
                prediction_to_signal(p.predicted),
            )
        })
        .collect();

    (metrics, pred_signals)
}

fn render_outputs(
    config: &Config,
    market_data: &MarketData,
    metrics: &BacktestMetrics,
    pred_signals: &[(usize, i32)],
) {
    info!("生成图表");
    let display_count = market_data.len().min(500);

    let _png = render_ohlcv(
        &config.png_output,
        &market_data.timestamps[..display_count],
        &market_data.opens[..display_count],
        &market_data.highs[..display_count],
        &market_data.lows[..display_count],
        &market_data.closes[..display_count],
        &market_data.volumes[..display_count],
        &market_data.indicator_values,
        &config.indicator_names(),
    );

    let dashboard_signals: Vec<(usize, i32)> = pred_signals
        .iter()
        .filter(|(idx, _)| *idx < display_count)
        .copied()
        .collect();
    let dashboard_equity_count = metrics.equity_curve.len().min(display_count);

    render_dashboard(
        &config.html_output,
        &market_data.timestamps[..display_count],
        &market_data.opens[..display_count],
        &market_data.highs[..display_count],
        &market_data.lows[..display_count],
        &market_data.closes[..display_count],
        &market_data.volumes[..display_count],
        &market_data.indicator_values,
        &config.indicator_names(),
        &metrics.equity_curve[..dashboard_equity_count],
        &dashboard_signals,
    );
}

fn print_output_summary(config: &Config) {
    // 汇总输出文件位置。
    println!("\n=== 全部完成 ===");
    println!("输出文件:");
    println!("  CSV:  {}", config.csv_output.display());
    println!("  SVG:  {}", config.png_output.display());
    println!("  HTML: {}", config.html_output.display());
    println!("  ONNX: {}", config.model_output.display());
}


