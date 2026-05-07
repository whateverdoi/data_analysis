mod config;
mod data;
mod features;
mod indicators;
mod ml;
mod plotting;

use config::Config;
use data::reader::read_csv;
use data::writer::{create_writer, write_header, write_row};
use features::builder::{
    build_feature_dataset as build_feature_dataset_from_spec, ExperimentSpec, FeatureDataset,
    LabelContext,
};
use indicators::engine::{IndicatorEngine, IndicatorSpec};
use ml::backtest::{evaluate, BacktestMetrics};
use ml::predict::predict_batch;
use ml::train::{train_python_random_forest, write_feature_csv};
use plotting::{html::render_dashboard, png::render_ohlcv};
use rust_decimal_macros::dec;
use std::io::Write;
use std::path::PathBuf;

pub(crate) struct MarketData {
    pub(crate) opens: Vec<f64>,
    pub(crate) highs: Vec<f64>,
    pub(crate) lows: Vec<f64>,
    pub(crate) closes: Vec<f64>,
    pub(crate) volumes: Vec<f64>,
    pub(crate) timestamps: Vec<String>,
    pub(crate) indicator_values: Vec<Vec<Option<f64>>>,
}

impl MarketData {
    fn new() -> Self {
        Self {
            opens: Vec::new(),
            highs: Vec::new(),
            lows: Vec::new(),
            closes: Vec::new(),
            volumes: Vec::new(),
            timestamps: Vec::new(),
            indicator_values: Vec::new(),
        }
    }

    fn len(&self) -> usize {
        self.closes.len()
    }
}

fn main() {
    let config = read_config();
    let experiment = default_experiment();

    let mut engine =
        IndicatorEngine::new(&config.symbol, &experiment.indicators).expect("创建指标引擎失败");
    let num_indicators = engine.num_indicators();
    let indicator_names = engine.indicator_names().to_vec();

    print_indicator_names(num_indicators, &indicator_names);

    let market_data = build_enhanced_csv(&config, &mut engine, &indicator_names);
    let Some(mut feature_dataset) = build_feature_dataset(&market_data, &experiment) else {
        return;
    };
    let (train_features, _train_labels, test_features, test_labels, train_size) =
        split_and_write_features(&config, &mut feature_dataset);

    train_model(&config, train_features.len());
    let (metrics, pred_signals) = run_backtest(
        &config,
        &market_data,
        &feature_dataset,
        &test_features,
        &test_labels,
        train_size,
        experiment.prediction_to_signal,
    );

    render_outputs(&config, &market_data, &metrics, &pred_signals);
    print_output_summary(&config);
}

fn default_experiment() -> ExperimentSpec {
    ExperimentSpec {
        window_size: 60,
        indicators: vec![
            IndicatorSpec::sma("sma20", 20),
            IndicatorSpec::sma("sma50", 50),
            IndicatorSpec::ema("ema12", 12),
            IndicatorSpec::ema("ema26", 26),
            IndicatorSpec::rsi("rsi14", 14),
            IndicatorSpec::macd("macd", 12, 26, 9),
            IndicatorSpec::stochastic_k("stoch_k", 14),
            IndicatorSpec::williams_r("willr14", 14),
            IndicatorSpec::cci("cci20", 20),
            IndicatorSpec::roc("roc10", 10),
            IndicatorSpec::bollinger_b("bb_sma", 20, dec!(2)),
            IndicatorSpec::atr("atr14", 14),
            IndicatorSpec::obv("obv"),
            IndicatorSpec::mfi("mfi14", 14),
        ],
        label_fn: label_next_close_direction,
        prediction_to_signal: default_prediction_to_signal,
    }
}

fn label_next_close_direction(ctx: &LabelContext<'_>) -> Option<i32> {
    let current = ctx.close(ctx.index)?;
    let next = ctx.close(ctx.index + 1)?;
    Some(if next > current { 1 } else { 0 })
}

#[allow(dead_code)]
fn label_future_return_5(ctx: &LabelContext<'_>) -> Option<i32> {
    let current = ctx.close(ctx.index)?;
    let future = ctx.close(ctx.index + 5)?;
    let ret = future / current - 1.0;

    Some(if ret > 0.003 {
        2
    } else if ret < -0.003 {
        0
    } else {
        1
    })
}

fn default_prediction_to_signal(label: i32) -> i32 {
    if label == 1 {
        1
    } else {
        0
    }
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
                eprintln!("读取 CSV 行失败, 跳过: {}", e);
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
    println!("\n=== 特征工程 ===");
    let Some((dataset, report)) = build_feature_dataset_from_spec(market_data, experiment) else {
        eprintln!("特征数据不足，无法训练");
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
    println!("\n=== 模型训练 ===");
    let trained = train_python_random_forest(
        &config.python_bin,
        &config.train_features_output,
        &config.model_output,
        &config.train_metrics_output,
        config.n_trees,
    )
    .expect("Python 模型训练失败");
    println!("训练集大小: {}", train_feature_count);
    println!("训练准确率: {:.2}%", trained.train_accuracy * 100.0);
    println!("训练耗时: {}ms", trained.train_time_ms);
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
    // Rust 加载 Python 导出的 ONNX 模型预测标签，并把标签映射为交易信号后回测。
    println!("\n=== 回测预测 ===");
    let predictions =
        predict_batch(&config.model_output, test_features, test_labels).expect("ONNX 推理失败");

    // 使用特征索引映射来确保回测的价格序列与预测结果精确对齐。
    let test_feature_start = train_size;
    let test_feature_indices: Vec<usize> = dataset
        .feature_indices
        .iter()
        .skip(test_feature_start)
        .copied()
        .collect();

    // 从原始价格序列中提取对应的价格
    let prices_for_backtest: Vec<f64> = test_feature_indices
        .iter()
        .map(|&idx| market_data.closes[idx])
        .collect();

    let metrics = evaluate(
        &predictions,
        &prices_for_backtest,
        100_000.0,
        prediction_to_signal,
    );
    println!("测试集预测数: {}", metrics.total_predictions);
    println!("回测准确率: {:.2}%", metrics.accuracy * 100.0);
    println!("各 label 准确率:");
    for (label, label_metrics) in &metrics.label_metrics {
        println!(
            "  label {}: 样本 {}, 准确率 {:.2}%",
            label,
            label_metrics.total,
            label_metrics.accuracy * 100.0
        );
    }
    println!("夏普比率: {:.2}", metrics.sharpe_ratio);
    println!("最大回撤: {:.2}%", metrics.max_drawdown_pct);

    // 把测试集预测映射回原始行情序列中的位置，方便在图上标出信号。
    let pred_signals: Vec<(usize, i32)> = predictions
        .iter()
        .enumerate()
        .map(|(i, p)| {
            (
                test_feature_indices.get(i).copied().unwrap_or(0),
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
    // 生成静态行情图和 HTML 仪表盘，便于可视化检查指标、价格和预测信号。
    println!("\n=== 生成图表 ===");
    let display_count = market_data.len().min(500);

    // 这里默认取前两个指标作为 SMA20 和 SMA50，用于叠加到 K 线图上。
    let sma20: Vec<Option<f64>> = market_data
        .indicator_values
        .iter()
        .map(|vals| vals.first().copied().flatten())
        .collect();
    let sma50: Vec<Option<f64>> = market_data
        .indicator_values
        .iter()
        .map(|vals| vals.get(1).copied().flatten())
        .collect();

    let _png = render_ohlcv(
        &config.png_output,
        &market_data.timestamps[..display_count],
        &market_data.opens[..display_count],
        &market_data.highs[..display_count],
        &market_data.lows[..display_count],
        &market_data.closes[..display_count],
        &market_data.volumes[..display_count],
        &sma20[..display_count],
        &sma50[..display_count],
    );

    let dashboard_signals: Vec<(usize, i32)> = pred_signals
        .iter()
        .filter(|(idx, _)| *idx < display_count)
        .copied()
        .collect();
    let dashboard_equity_count = metrics.equity_curve.len().min(display_count);

    // 渲染包含行情、均线、资金曲线和预测信号的 HTML 页面。
    render_dashboard(
        &config.html_output,
        &market_data.timestamps[..display_count],
        &market_data.opens[..display_count],
        &market_data.highs[..display_count],
        &market_data.lows[..display_count],
        &market_data.closes[..display_count],
        &market_data.volumes[..display_count],
        &sma20[..display_count],
        &sma50[..display_count],
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

fn read_config() -> Config {
    // 读取默认配置，并确保输出目录已经存在。
    let mut config = Config::default();
    config.csv_input =
        PathBuf::from("/home/lhh/Documents/lhhrustprojects/bars/data/test_dollar_run.csv");
    config.check_output_dirs().expect("无法创建输出目录");

    // 打印本次运行的核心参数，方便确认输入文件和模型配置。
    println!("=== 金融时间序列数据分析 ===");
    println!("CSV 输入: {}", config.csv_input.display());
    println!("训练比例: {:.0}%", config.train_split * 100.0);
    println!("随机森林树数: {}", config.n_trees);
    println!();
    config
}
