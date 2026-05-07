use csv::Writer;
use std::path::Path;

pub fn create_writer(path: &Path) -> csv::Result<Writer<std::fs::File>> {
    Writer::from_path(path)
}

pub fn write_header(
    wtr: &mut Writer<std::fs::File>,
    indicator_names: &[String],
) -> csv::Result<()> {
    let mut header = vec!["timestamp", "open", "high", "low", "close", "volume"];
    header.extend(indicator_names.iter().map(String::as_str));
    wtr.write_record(&header)?;
    Ok(())
}

pub fn write_row(
    wtr: &mut Writer<std::fs::File>,
    ts: &str,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    volume: f64,
    indicator_values: &[Option<f64>],
) -> csv::Result<()> {
    let mut record: Vec<String> = vec![
        ts.to_string(),
        format!("{}", open),
        format!("{}", high),
        format!("{}", low),
        format!("{}", close),
        format!("{}", volume),
    ];
    for (idx, val) in indicator_values.iter().enumerate() {
        match val {
            Some(v) => {
                // 检查是否为 NaN 或 Inf
                if v.is_nan() || v.is_infinite() {
                    eprintln!("警告: 指标 {} 包含 NaN 或 Inf 值: {}", idx, v);
                    record.push(String::new());
                } else {
                    record.push(format!("{:.6}", v))
                }
            }
            None => record.push(String::new()), // 指标计算期间的空值，在增强 CSV 中以空字符串表示
        }
    }
    wtr.write_record(&record)?;
    Ok(())
}
