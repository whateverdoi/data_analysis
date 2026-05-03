use chrono::{DateTime, NaiveDateTime};
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct CsvRow {
    pub timestamp: String,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

impl CsvRow {
    pub fn parse_timestamp(&self) -> Option<NaiveDateTime> {
        let s = self.timestamp.trim();

        if let Ok(v) = NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S%.3f") {
            return Some(v);
        }
        if let Ok(v) = NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S") {
            return Some(v);
        }
        if let Ok(v) = NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S") {
            return Some(v);
        }
        if let Ok(d) = chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d") {
            return d.and_hms_opt(0, 0, 0);
        }
        if let Ok(secs) = s.parse::<i64>() {
            return DateTime::from_timestamp(secs, 0)
                .map(|dt| dt.naive_utc());
        }
        if let Ok(millis) = s.parse::<i64>() {
            let secs = millis / 1000;
            let nsecs = ((millis % 1000) * 1_000_000) as u32;
            return DateTime::from_timestamp(secs, nsecs)
                .map(|dt| dt.naive_utc());
        }
        None
    }
}

pub fn read_csv(path: &str) -> impl Iterator<Item = csv::Result<CsvRow>> {
    csv::ReaderBuilder::new()
        .has_headers(true)
        .flexible(true)
        .from_path(path)
        .expect("无法打开 CSV 文件")
        .into_deserialize::<CsvRow>()
}
