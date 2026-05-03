use std::path::PathBuf;

#[derive(Clone, Debug)]
pub struct Config {
    pub csv_input: PathBuf,
    pub csv_output: PathBuf,
    pub png_output: PathBuf,
    pub html_output: PathBuf,
    pub window_size: usize,
    pub train_split: f64,
    pub n_trees: u32,
    pub symbol: String,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            csv_input: PathBuf::from(
                "/home/lhh/Documents/lhhrustprojects/bars/data/test_dollar.csv",
            ),
            csv_output: PathBuf::from("output/enhanced.csv"),
            png_output: PathBuf::from("output/chart.svg"),
            html_output: PathBuf::from("output/dashboard.html"),
            window_size: 60,
            train_split: 0.6,
            n_trees: 100,
            symbol: "SYM".into(),
        }
    }
}

impl Config {
    pub fn check_output_dirs(&self) -> std::io::Result<()> {
        if let Some(parent) = self.csv_output.parent() {
            std::fs::create_dir_all(parent)?;
        }
        if let Some(parent) = self.png_output.parent() {
            std::fs::create_dir_all(parent)?;
        }
        if let Some(parent) = self.html_output.parent() {
            std::fs::create_dir_all(parent)?;
        }
        Ok(())
    }
}
