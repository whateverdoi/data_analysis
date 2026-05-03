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
use ml::train::train_random_forest;
use plotting::{html::render_dashboard, png::render_ohlcv};
use std::io::Write;

fn main() {
    let config = Config::default();
    config.check_output_dirs().expect("无法创建输出目录");

    println!("=== 金融时间序列数据分析 ===");
    println!("CSV 输入: {}", config.csv_input.display());
    println!("窗口大小: {}", config.window_size);
    println!("训练比例: {:.0}%", config.train_split * 100.0);
    println!("随机森林树数: {}", config.n_trees);
    println!();

    let mut engine =
        IndicatorEngine::new(&config.symbol).expect("创建指标引擎失败");
    let num_indicators = engine.num_indicators();
    let indicator_names = engine.indicator_names().to_vec();

    println!("启用了 {} 个技术指标:", num_indicators);
    for name in &indicator_names {
        println!("  - {}", name);
    }
    println!();

    let mut all_opens: Vec<f64> = Vec::new();
    let mut all_highs: Vec<f64> = Vec::new();
    let mut all_lows: Vec<f64> = Vec::new();
    let mut all_closes: Vec<f64> = Vec::new();
    let mut all_volumes: Vec<f64> = Vec::new();
    let mut all_timestamps: Vec<String> = Vec::new();
    let mut all_indicator_values: Vec<Vec<Option<f64>>> = Vec::new();

    let mut wtr_out =
        create_writer(&config.csv_output).expect("创建输出 CSV writer 失败");
    write_header(&mut wtr_out, &indicator_names).expect("写 CSV 头失败");

    let mut bar_count: usize = 0;
    let mut read_total: usize = 0;
    let csv_iter = read_csv(config.csv_input.to_str().unwrap());

    for result in csv_iter {
        read_total += 1;
        match result {
            Ok(row) => {
                let ts = row.timestamp.clone();
                let dt = row.parse_timestamp();
                let ts_nanos = dt
                    .map(|d| {
                        d.and_utc().timestamp_nanos_opt().unwrap_or(0)
                    })
                    .unwrap_or(0);

                let indicator_vals = engine.update(
                    ts_nanos,
                    row.open,
                    row.high,
                    row.low,
                    row.close,
                    row.volume,
                );

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

                all_opens.push(row.open);
                all_highs.push(row.high);
                all_lows.push(row.low);
                all_closes.push(row.close);
                all_volumes.push(row.volume);
                all_timestamps.push(ts);
                all_indicator_values.push(indicator_vals);

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

    wtr_out.flush().expect("刷新 CSV writer 失败");
    println!("\r处理完成: 读取 {} 行, 成功 {} 行, 跳过 {} 行",
        read_total, bar_count, read_total - bar_count);
    println!("增强后 CSV 已保存: {}", config.csv_output.display());

    println!("\n=== 特征工程 ===");
    let mut feature_builder =
        FeatureBuilder::new(config.window_size, num_indicators);

    for i in 0..bar_count.saturating_sub(1) {
        let feat: Vec<f64> = all_indicator_values[i]
            .iter()
            .map(|v| v.unwrap_or(0.0))
            .collect();
        feature_builder.push(feat, all_closes[i], all_closes[i + 1]);
    }

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

    let train_size =
        (all_features.len() as f64 * config.train_split) as usize;
    let train_features: Vec<Vec<f64>> = all_features[..train_size].to_vec();
    let train_labels: Vec<i32> = all_labels[..train_size].to_vec();
    let test_features: Vec<Vec<f64>> = all_features[train_size..].to_vec();
    let test_labels: Vec<i32> = all_labels[train_size..].to_vec();

    println!("\n=== 模型训练 ===");
    let trained = train_random_forest(
        &train_features,
        &train_labels,
        config.n_trees,
    );
    println!("训练集大小: {}", train_features.len());
    println!(
        "训练准确率: {:.2}%",
        trained.train_accuracy * 100.0
    );
    println!("训练耗时: {}ms", trained.train_time_ms);

    println!("\n=== 回测预测 ===");
    let predictions =
        predict_batch(&trained.model, &test_features, &test_labels);
    let test_offset = train_size + config.window_size;
    let prices_for_backtest: Vec<f64> =
        all_closes[test_offset..].to_vec();

    let metrics =
        evaluate(&predictions, &prices_for_backtest, 100_000.0);
    println!("测试集预测数: {}", metrics.total_predictions);
    println!(
        "回测准确率: {:.2}%",
        metrics.accuracy * 100.0
    );
    println!(
        "上涨准确率: {:.2}%",
        metrics.up_accuracy * 100.0
    );
    println!(
        "下跌准确率: {:.2}%",
        metrics.down_accuracy * 100.0
    );
    println!("夏普比率: {:.2}", metrics.sharpe_ratio);
    println!(
        "最大回撤: {:.2}%",
        metrics.max_drawdown_pct
    );

    println!("\n=== 生成图表 ===");
    let display_count = bar_count.min(500);

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

    let pred_signals: Vec<(usize, i32)> = predictions
        .iter()
        .enumerate()
        .map(|(i, p)| (test_offset + i, p.predicted))
        .collect();

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

    println!("\n=== 全部完成 ===");
    println!("输出文件:");
    println!("  CSV:  {}", config.csv_output.display());
    println!("  SVG:  {}", config.png_output.display());
    println!("  HTML: {}", config.html_output.display());
}
