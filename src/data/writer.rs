use csv::Writer;
use std::path::Path;

pub fn create_writer(path: &Path) -> csv::Result<Writer<std::fs::File>> {
    Writer::from_path(path)
}

pub fn write_header(
    wtr: &mut Writer<std::fs::File>,
    indicator_names: &[&str],
) -> csv::Result<()> {
    let mut header = vec![
        "timestamp", "open", "high", "low", "close", "volume",
    ];
    header.extend(indicator_names);
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
    for val in indicator_values {
        match val {
            Some(v) => record.push(format!("{:.6}", v)),
            None => record.push(String::new()),
        }
    }
    wtr.write_record(&record)?;
    Ok(())
}
