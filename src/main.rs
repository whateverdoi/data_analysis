mod config;
mod data;
mod features;
mod indicators;
mod ml;
mod plotting;

use config::Config;
use data::reader::read_csv;
use data::writer::{create_writer, write_header, write_row};
use features::builder::FeatureBuilder;
use indicators::engine::IndicatorEngine;
use ml::backtest::evaluate;
use ml::predict::predict_batch;
use ml::train::{train_python_random_forest, write_feature_csv};
use plotting::{html::render_dashboard, png::render_ohlcv};
use std::io::Write;
use std::path::PathBuf;

fn main() {
    // 读取默认配置，并确保输出目录已经存在。
    let mut config = Config::default();
    config.csv_input =
        PathBuf::from("/home/lhh/Documents/lhhrustprojects/bars/data/test_dollar_run.csv");
    config.check_output_dirs().expect("无法创建输出目录");

    // 打印本次运行的核心参数，方便确认输入文件和模型配置。
    println!("=== 金融时间序列数据分析 ===");
    println!("CSV 输入: {}", config.csv_input.display());
    println!("窗口大小: {}", config.window_size);
    println!("训练比例: {:.0}%", config.train_split * 100.0);
    println!("随机森林树数: {}", config.n_trees);
    println!();

    // 创建技术指标引擎，后面每读入一根 K 线都会用它更新指标值。
    let mut engine = IndicatorEngine::new(&config.symbol).expect("创建指标引擎失败");
    let num_indicators = engine.num_indicators();
    let indicator_names = engine.indicator_names().to_vec();

    // 输出当前启用的技术指标名称，便于核对增强 CSV 的列。
    println!("启用了 {} 个技术指标:", num_indicators);
    for name in &indicator_names {
        println!("  - {}", name);
    }
    println!();

    // 保存完整行情和指标序列，后续特征工程、回测和画图都会复用这些数据。
    let mut all_opens: Vec<f64> = Vec::new();
    let mut all_highs: Vec<f64> = Vec::new();
    let mut all_lows: Vec<f64> = Vec::new();
    let mut all_closes: Vec<f64> = Vec::new();
    let mut all_volumes: Vec<f64> = Vec::new();
    let mut all_timestamps: Vec<String> = Vec::new();
    let mut all_indicator_values: Vec<Vec<Option<f64>>> = Vec::new();

    // 创建增强后的 CSV 文件，并先写入表头。
    let mut wtr_out = create_writer(&config.csv_output).expect("创建输出 CSV writer 失败");
    write_header(&mut wtr_out, &indicator_names).expect("写 CSV 头失败");

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
                all_opens.push(row.open);
                all_highs.push(row.high);
                all_lows.push(row.low);
                all_closes.push(row.close);
                all_volumes.push(row.volume);
                all_timestamps.push(ts);
                all_indicator_values.push(indicator_vals);

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

    // 把连续窗口内的指标序列转换成机器学习特征，并生成涨跌标签。
    println!("\n=== 特征工程 ===");
    let mut feature_builder = FeatureBuilder::new(config.window_size, num_indicators);

    for i in 0..bar_count.saturating_sub(1) {
        // 尚未计算出来的指标值用 0.0 填充，避免模型输入中出现空值。
        let feat: Vec<f64> = all_indicator_values[i]
            .iter()
            .map(|v| v.unwrap_or(0.0))
            .collect();
        // 用下一根 K 线的收盘价和当前收盘价比较，得到上涨或下跌标签。
        feature_builder.push(feat, all_closes[i], all_closes[i + 1]);
    }

    // 取出构建好的全部特征和标签，并打印基本分布。
    let all_features = feature_builder.take_features();
    let all_labels = feature_builder.take_labels();
    println!("特征向量数: {}", all_features.len());
    println!("特征维度: {}", feature_builder.feature_count());
    println!(
        "label 分布: UP={}, DOWN={}",
        all_labels.iter().filter(|&&l| l == 1).count(),
        all_labels.iter().filter(|&&l| l == 0).count(),
    );

    if all_features.is_empty() {
        eprintln!("特征数据不足，无法训练");
        return;
    }

    // 按时间顺序切分训练集和测试集，避免未来数据泄漏到训练阶段。
    let train_size = (all_features.len() as f64 * config.train_split) as usize;
    let train_features: Vec<Vec<f64>> = all_features[..train_size].to_vec();
    let train_labels: Vec<i32> = all_labels[..train_size].to_vec();
    let test_features: Vec<Vec<f64>> = all_features[train_size..].to_vec();
    let test_labels: Vec<i32> = all_labels[train_size..].to_vec();

    write_feature_csv(
        &config.train_features_output,
        &train_features,
        &train_labels,
    )
    .expect("写训练特征 CSV 失败");
    write_feature_csv(&config.test_features_output, &test_features, &test_labels)
        .expect("写测试特征 CSV 失败");

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
    println!("训练集大小: {}", train_features.len());
    println!("训练准确率: {:.2}%", trained.train_accuracy * 100.0);
    println!("训练耗时: {}ms", trained.train_time_ms);

    // Rust 加载 Python 导出的 ONNX 模型预测涨跌，并把预测结果放入简单回测中评估收益表现。
    println!("\n=== 回测预测 ===");
    let predictions =
        predict_batch(&config.model_output, &test_features, &test_labels).expect("ONNX 推理失败");
    // 特征窗口会消耗前 window_size 条数据，所以回测价格需要做同样的偏移。
    let test_offset = train_size + config.window_size;
    let prices_for_backtest: Vec<f64> = all_closes[test_offset..].to_vec();

    let metrics = evaluate(&predictions, &prices_for_backtest, 100_000.0);
    println!("测试集预测数: {}", metrics.total_predictions);
    println!("回测准确率: {:.2}%", metrics.accuracy * 100.0);
    println!("上涨准确率: {:.2}%", metrics.up_accuracy * 100.0);
    println!("下跌准确率: {:.2}%", metrics.down_accuracy * 100.0);
    println!("夏普比率: {:.2}", metrics.sharpe_ratio);
    println!("最大回撤: {:.2}%", metrics.max_drawdown_pct);

    // 生成静态行情图和 HTML 仪表盘，便于可视化检查指标、价格和预测信号。
    println!("\n=== 生成图表 ===");
    let display_count = bar_count.min(500);

    // 这里默认取前两个指标作为 SMA20 和 SMA50，用于叠加到 K 线图上。
    let sma20: Vec<Option<f64>> = all_indicator_values
        .iter()
        .map(|vals| vals.first().copied().flatten())
        .collect();
    let sma50: Vec<Option<f64>> = all_indicator_values
        .iter()
        .map(|vals| vals.get(1).copied().flatten())
        .collect();

    let _png = render_ohlcv(
        &config.png_output,
        &all_timestamps[..display_count],
        &all_opens[..display_count],
        &all_highs[..display_count],
        &all_lows[..display_count],
        &all_closes[..display_count],
        &all_volumes[..display_count],
        &sma20[..display_count],
        &sma50[..display_count],
    );

    // 把测试集预测映射回原始行情序列中的位置，方便在图上标出信号。
    let pred_signals: Vec<(usize, i32)> = predictions
        .iter()
        .enumerate()
        .map(|(i, p)| (test_offset + i, p.predicted))
        .collect();

    // 渲染包含行情、均线、资金曲线和预测信号的 HTML 页面。
    render_dashboard(
        &config.html_output,
        &all_timestamps,
        &all_opens,
        &all_highs,
        &all_lows,
        &all_closes,
        &all_volumes,
        &sma20,
        &sma50,
        &metrics.equity_curve,
        &pred_signals,
    );

    // 汇总输出文件位置。
    println!("\n=== 全部完成 ===");
    println!("输出文件:");
    println!("  CSV:  {}", config.csv_output.display());
    println!("  SVG:  {}", config.png_output.display());
    println!("  HTML: {}", config.html_output.display());
    println!("  ONNX: {}", config.model_output.display());
}
