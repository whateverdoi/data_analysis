use std::path::PathBuf;

use clap::Parser;
use serde::Deserialize;

use crate::indicators::specs::{IndicatorKind, IndicatorSpec};

#[derive(Debug, Clone, Deserialize)]
struct IndicatorConfig {
    r#type: String,
    name: String,
    period: Option<usize>,
    fast_period: Option<usize>,
    slow_period: Option<usize>,
    signal_period: Option<usize>,
    std_dev: Option<f64>,
    multiplier: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct ExperimentConfig {
    window_size: Option<usize>,
    label_fn: Option<String>,
    indicators: Option<Vec<IndicatorConfig>>,
}

#[derive(Debug, Clone, Deserialize)]
struct PathsConfig {
    csv_input: Option<PathBuf>,
    csv_output: Option<PathBuf>,
    png_output: Option<PathBuf>,
    html_output: Option<PathBuf>,
    train_features_output: Option<PathBuf>,
    test_features_output: Option<PathBuf>,
    model_output: Option<PathBuf>,
    train_metrics_output: Option<PathBuf>,
}

#[derive(Debug, Clone, Deserialize)]
struct MlConfig {
    python_bin: Option<PathBuf>,
    train_split: Option<f64>,
    n_trees: Option<u32>,
    max_depth: Option<u32>,
    min_samples_leaf: Option<u32>,
    symbol: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct TomlConfig {
    paths: Option<PathsConfig>,
    experiment: Option<ExperimentConfig>,
    ml: Option<MlConfig>,
}

#[derive(Debug, Clone)]
pub struct Config {
    pub csv_input: PathBuf,
    pub csv_output: PathBuf,
    pub png_output: PathBuf,
    pub html_output: PathBuf,
    pub train_features_output: PathBuf,
    pub test_features_output: PathBuf,
    pub model_output: PathBuf,
    pub train_metrics_output: PathBuf,
    pub python_bin: PathBuf,
    pub train_split: f64,
    pub n_trees: u32,
    pub max_depth: Option<u32>,
    pub min_samples_leaf: Option<u32>,
    pub symbol: String,
    pub window_size: usize,
    pub label_fn_name: String,
    pub indicator_specs: Vec<IndicatorSpec>,
}

impl Config {
    fn default_paths() -> PathsConfig {
        PathsConfig {
            csv_input: None,
            csv_output: Some(PathBuf::from("output/enhanced.csv")),
            png_output: Some(PathBuf::from("output/chart.svg")),
            html_output: Some(PathBuf::from("output/dashboard.html")),
            train_features_output: Some(PathBuf::from("output/train_features.csv")),
            test_features_output: Some(PathBuf::from("output/test_features.csv")),
            model_output: Some(PathBuf::from("output/model.onnx")),
            train_metrics_output: Some(PathBuf::from("output/train_metrics.json")),
        }
    }

    fn default_experiment() -> ExperimentConfig {
        ExperimentConfig {
            window_size: Some(60),
            label_fn: Some("next_close_direction".into()),
            indicators: Some(vec![
                IndicatorConfig { r#type: "sma".into(), name: "sma20".into(), period: Some(20), fast_period: None, slow_period: None, signal_period: None, std_dev: None, multiplier: None },
                IndicatorConfig { r#type: "sma".into(), name: "sma50".into(), period: Some(50), fast_period: None, slow_period: None, signal_period: None, std_dev: None, multiplier: None },
                IndicatorConfig { r#type: "ema".into(), name: "ema12".into(), period: Some(12), fast_period: None, slow_period: None, signal_period: None, std_dev: None, multiplier: None },
                IndicatorConfig { r#type: "ema".into(), name: "ema26".into(), period: Some(26), fast_period: None, slow_period: None, signal_period: None, std_dev: None, multiplier: None },
                IndicatorConfig { r#type: "rsi".into(), name: "rsi14".into(), period: Some(14), fast_period: None, slow_period: None, signal_period: None, std_dev: None, multiplier: None },
                IndicatorConfig { r#type: "macd".into(), name: "macd".into(), period: None, fast_period: Some(12), slow_period: Some(26), signal_period: Some(9), std_dev: None, multiplier: None },
                IndicatorConfig { r#type: "stoch_k".into(), name: "stoch_k".into(), period: Some(14), fast_period: None, slow_period: None, signal_period: None, std_dev: None, multiplier: None },
                IndicatorConfig { r#type: "willr".into(), name: "willr14".into(), period: Some(14), fast_period: None, slow_period: None, signal_period: None, std_dev: None, multiplier: None },
                IndicatorConfig { r#type: "cci".into(), name: "cci20".into(), period: Some(20), fast_period: None, slow_period: None, signal_period: None, std_dev: None, multiplier: None },
                IndicatorConfig { r#type: "roc".into(), name: "roc10".into(), period: Some(10), fast_period: None, slow_period: None, signal_period: None, std_dev: None, multiplier: None },
                IndicatorConfig { r#type: "bb".into(), name: "bb_sma".into(), period: Some(20), fast_period: None, slow_period: None, signal_period: None, std_dev: Some(2.0), multiplier: None },
                IndicatorConfig { r#type: "atr".into(), name: "atr14".into(), period: Some(14), fast_period: None, slow_period: None, signal_period: None, std_dev: None, multiplier: None },
                IndicatorConfig { r#type: "obv".into(), name: "obv".into(), period: None, fast_period: None, slow_period: None, signal_period: None, std_dev: None, multiplier: None },
                IndicatorConfig { r#type: "mfi".into(), name: "mfi14".into(), period: Some(14), fast_period: None, slow_period: None, signal_period: None, std_dev: None, multiplier: None },
                IndicatorConfig { r#type: "adx".into(), name: "adx14".into(), period: Some(14), fast_period: None, slow_period: None, signal_period: None, std_dev: None, multiplier: None },
                IndicatorConfig { r#type: "aroon".into(), name: "aroon14".into(), period: Some(14), fast_period: None, slow_period: None, signal_period: None, std_dev: None, multiplier: None },
                IndicatorConfig { r#type: "force".into(), name: "force13".into(), period: Some(13), fast_period: None, slow_period: None, signal_period: None, std_dev: None, multiplier: None },
                IndicatorConfig { r#type: "chaikin".into(), name: "chaikin".into(), period: None, fast_period: Some(3), slow_period: Some(10), signal_period: None, std_dev: None, multiplier: None },
                IndicatorConfig { r#type: "vwap".into(), name: "vwap".into(), period: None, fast_period: None, slow_period: None, signal_period: None, std_dev: None, multiplier: None },
                IndicatorConfig { r#type: "mom".into(), name: "mom10".into(), period: Some(10), fast_period: None, slow_period: None, signal_period: None, std_dev: None, multiplier: None },
                IndicatorConfig { r#type: "trix".into(), name: "trix15".into(), period: Some(15), fast_period: None, slow_period: None, signal_period: None, std_dev: None, multiplier: None },
                IndicatorConfig { r#type: "keltner".into(), name: "keltner20".into(), period: Some(20), fast_period: None, slow_period: None, signal_period: None, std_dev: None, multiplier: Some(2.0) },
                IndicatorConfig { r#type: "wad".into(), name: "wad".into(), period: None, fast_period: None, slow_period: None, signal_period: None, std_dev: None, multiplier: None },
                IndicatorConfig { r#type: "elder".into(), name: "elder13".into(), period: Some(13), fast_period: None, slow_period: None, signal_period: None, std_dev: None, multiplier: None },
            ]),
        }
    }

    fn as_indicator_spec(cfg: &IndicatorConfig) -> Option<IndicatorSpec> {
        let kind = match cfg.r#type.as_str() {
            "sma" => Some(IndicatorKind::Sma { period: cfg.period? }),
            "ema" => Some(IndicatorKind::Ema { period: cfg.period? }),
            "rsi" => Some(IndicatorKind::Rsi { period: cfg.period? }),
            "macd" => Some(IndicatorKind::Macd {
                fast_period: cfg.fast_period?,
                slow_period: cfg.slow_period?,
                signal_period: cfg.signal_period?,
            }),
            "stoch_k" => Some(IndicatorKind::StochasticK { period: cfg.period? }),
            "willr" => Some(IndicatorKind::WilliamsR { period: cfg.period? }),
            "cci" => Some(IndicatorKind::Cci { period: cfg.period? }),
            "roc" => Some(IndicatorKind::Roc { period: cfg.period? }),
            "bb" => Some(IndicatorKind::BollingerB {
                period: cfg.period?,
                std_dev: cfg.std_dev.unwrap_or(2.0),
            }),
            "atr" => Some(IndicatorKind::Atr { period: cfg.period? }),
            "obv" => Some(IndicatorKind::Obv),
            "mfi" => Some(IndicatorKind::Mfi { period: cfg.period? }),
            "adx" => Some(IndicatorKind::Adx { period: cfg.period? }),
            "aroon" => Some(IndicatorKind::Aroon { period: cfg.period? }),
            "force" => Some(IndicatorKind::ForceIndex { period: cfg.period? }),
            "chaikin" => Some(IndicatorKind::ChaikinOsc {
                fast_period: cfg.fast_period?,
                slow_period: cfg.slow_period?,
            }),
            "vwap" => Some(IndicatorKind::Vwap),
            "mom" => Some(IndicatorKind::Momentum { period: cfg.period? }),
            "trix" => Some(IndicatorKind::Trix { period: cfg.period? }),
            "keltner" => Some(IndicatorKind::KeltnerChannel {
                period: cfg.period?,
                multiplier: cfg.multiplier.unwrap_or(2.0),
            }),
            "wad" => Some(IndicatorKind::WilliamsAD),
            "elder" => Some(IndicatorKind::ElderRay { period: cfg.period? }),
            _ => None,
        };
        kind.map(|k| IndicatorSpec::new(&cfg.name, k))
    }

    pub fn load() -> Self {
        let cli = CliArgs::parse();

        if let Some(level) = cli.verbose {
            let log_level = match level {
                0 => "warn",
                1 => "info",
                _ => "debug",
            };
            std::env::set_var("RUST_LOG", log_level);
        }

        let config_path = cli
            .config
            .clone()
            .or_else(|| std::env::current_dir().ok().map(|d| d.join("config.toml")));

        let toml_config: Option<TomlConfig> = config_path
            .as_ref()
            .and_then(|p| std::fs::read_to_string(p).ok())
            .and_then(|content| toml::from_str(&content).ok());

        let paths = toml_config
            .as_ref()
            .and_then(|c| c.paths.clone())
            .unwrap_or_else(Self::default_paths);

        let experiment = toml_config
            .as_ref()
            .and_then(|c| c.experiment.clone())
            .unwrap_or_else(Self::default_experiment);

        let ml = toml_config.as_ref().and_then(|c| c.ml.clone());

        let default_python = PathBuf::from("python3");

        let indicator_specs: Vec<IndicatorSpec> = experiment
            .indicators
            .unwrap_or_default()
            .iter()
            .filter_map(|c| Self::as_indicator_spec(c))
            .collect();

        Self {
            csv_input: cli.csv_input.or(paths.csv_input).unwrap_or_else(|| PathBuf::from("data/input.csv")),
            csv_output: paths.csv_output.unwrap_or_else(|| PathBuf::from("output/enhanced.csv")),
            png_output: paths.png_output.unwrap_or_else(|| PathBuf::from("output/chart.svg")),
            html_output: paths.html_output.unwrap_or_else(|| PathBuf::from("output/dashboard.html")),
            train_features_output: paths.train_features_output.unwrap_or_else(|| PathBuf::from("output/train_features.csv")),
            test_features_output: paths.test_features_output.unwrap_or_else(|| PathBuf::from("output/test_features.csv")),
            model_output: paths.model_output.unwrap_or_else(|| PathBuf::from("output/model.onnx")),
            train_metrics_output: paths.train_metrics_output.unwrap_or_else(|| PathBuf::from("output/train_metrics.json")),
            python_bin: ml.as_ref().and_then(|m| m.python_bin.clone()).or(cli.python_bin).unwrap_or(default_python),
            train_split: ml.as_ref().and_then(|m| m.train_split).or(cli.train_split).unwrap_or(0.6),
            n_trees: ml.as_ref().and_then(|m| m.n_trees).or(cli.n_trees).unwrap_or(300),
            max_depth: ml.as_ref().and_then(|m| m.max_depth),
            min_samples_leaf: ml.as_ref().and_then(|m| m.min_samples_leaf),
            symbol: ml.as_ref().and_then(|m| m.symbol.clone()).unwrap_or_else(|| "SYM".into()),
            window_size: experiment.window_size.or(cli.window_size).unwrap_or(60),
            label_fn_name: experiment.label_fn.unwrap_or_else(|| "next_close_direction".into()),
            indicator_specs,
        }
    }

    pub fn indicator_names(&self) -> Vec<String> {
        self.indicator_specs.iter().map(|s| s.name().to_string()).collect()
    }

    pub fn check_output_dirs(&self) -> std::io::Result<()> {
        let dirs = [
            &self.csv_output,
            &self.png_output,
            &self.html_output,
            &self.train_features_output,
            &self.test_features_output,
            &self.model_output,
            &self.train_metrics_output,
        ];
        for dir in dirs {
            if let Some(parent) = dir.parent() {
                std::fs::create_dir_all(parent)?;
            }
        }
        Ok(())
    }
}

#[derive(Parser, Debug)]
#[command(name = "data_analysis", version, about = "金融时间序列数据分析与ML回测")]
struct CliArgs {
    #[arg(short, long)]
    config: Option<PathBuf>,

    #[arg(short = 'i', long)]
    csv_input: Option<PathBuf>,

    #[arg(long)]
    python_bin: Option<PathBuf>,

    #[arg(long)]
    window_size: Option<usize>,

    #[arg(long)]
    train_split: Option<f64>,

    #[arg(long)]
    n_trees: Option<u32>,

    #[arg(short, long, action = clap::ArgAction::Count)]
    verbose: Option<u8>,
}
