use fin_primitives::{
    ohlcv::OhlcvBar,
    signals::pipeline::SignalPipeline,
    signals::SignalValue,
    types::{NanoTimestamp, Price, Quantity, Symbol},
    FinError,
};
use rust_decimal::prelude::ToPrimitive;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;

use fin_primitives::signals::indicators::{
    Atr, BollingerB, Cci, Ema, Macd, Mfi, Obv, Roc, Rsi, Sma, StochasticK, WilliamsR,
};

pub struct IndicatorEngine {
    pipeline: SignalPipeline,
    names: Vec<String>,
    symbol: Symbol,
}

#[derive(Debug, Clone)]
pub enum IndicatorSpec {
    Sma {
        name: String,
        period: usize,
    },
    Ema {
        name: String,
        period: usize,
    },
    Rsi {
        name: String,
        period: usize,
    },
    Macd {
        name: String,
        fast_period: usize,
        slow_period: usize,
        signal_period: usize,
    },
    StochasticK {
        name: String,
        period: usize,
    },
    WilliamsR {
        name: String,
        period: usize,
    },
    Cci {
        name: String,
        period: usize,
    },
    Roc {
        name: String,
        period: usize,
    },
    BollingerB {
        name: String,
        period: usize,
        std_dev: Decimal,
    },
    Atr {
        name: String,
        period: usize,
    },
    Obv {
        name: String,
    },
    Mfi {
        name: String,
        period: usize,
    },
}

impl IndicatorSpec {
    pub fn sma(name: impl Into<String>, period: usize) -> Self {
        Self::Sma {
            name: name.into(),
            period,
        }
    }

    pub fn ema(name: impl Into<String>, period: usize) -> Self {
        Self::Ema {
            name: name.into(),
            period,
        }
    }

    pub fn rsi(name: impl Into<String>, period: usize) -> Self {
        Self::Rsi {
            name: name.into(),
            period,
        }
    }

    pub fn macd(
        name: impl Into<String>,
        fast_period: usize,
        slow_period: usize,
        signal_period: usize,
    ) -> Self {
        Self::Macd {
            name: name.into(),
            fast_period,
            slow_period,
            signal_period,
        }
    }

    pub fn stochastic_k(name: impl Into<String>, period: usize) -> Self {
        Self::StochasticK {
            name: name.into(),
            period,
        }
    }

    pub fn williams_r(name: impl Into<String>, period: usize) -> Self {
        Self::WilliamsR {
            name: name.into(),
            period,
        }
    }

    pub fn cci(name: impl Into<String>, period: usize) -> Self {
        Self::Cci {
            name: name.into(),
            period,
        }
    }

    pub fn roc(name: impl Into<String>, period: usize) -> Self {
        Self::Roc {
            name: name.into(),
            period,
        }
    }

    pub fn bollinger_b(name: impl Into<String>, period: usize, std_dev: Decimal) -> Self {
        Self::BollingerB {
            name: name.into(),
            period,
            std_dev,
        }
    }

    pub fn atr(name: impl Into<String>, period: usize) -> Self {
        Self::Atr {
            name: name.into(),
            period,
        }
    }

    pub fn obv(name: impl Into<String>) -> Self {
        Self::Obv { name: name.into() }
    }

    pub fn mfi(name: impl Into<String>, period: usize) -> Self {
        Self::Mfi {
            name: name.into(),
            period,
        }
    }

    pub fn name(&self) -> &str {
        match self {
            Self::Sma { name, .. }
            | Self::Ema { name, .. }
            | Self::Rsi { name, .. }
            | Self::Macd { name, .. }
            | Self::StochasticK { name, .. }
            | Self::WilliamsR { name, .. }
            | Self::Cci { name, .. }
            | Self::Roc { name, .. }
            | Self::BollingerB { name, .. }
            | Self::Atr { name, .. }
            | Self::Obv { name }
            | Self::Mfi { name, .. } => name,
        }
    }
}

impl IndicatorEngine {
    pub fn new(symbol_name: &str, specs: &[IndicatorSpec]) -> Result<Self, FinError> {
        let symbol = Symbol::new(symbol_name)?;
        let mut pipeline = SignalPipeline::new();
        for spec in specs {
            pipeline = match spec {
                IndicatorSpec::Sma { name, period } => pipeline.add(Sma::new(name, *period)?),
                IndicatorSpec::Ema { name, period } => pipeline.add(Ema::new(name, *period)?),
                IndicatorSpec::Rsi { name, period } => pipeline.add(Rsi::new(name, *period)?),
                IndicatorSpec::Macd {
                    name,
                    fast_period,
                    slow_period,
                    signal_period,
                } => pipeline.add(Macd::new(name, *fast_period, *slow_period, *signal_period)?),
                IndicatorSpec::StochasticK { name, period } => {
                    pipeline.add(StochasticK::new(name, *period)?)
                }
                IndicatorSpec::WilliamsR { name, period } => {
                    pipeline.add(WilliamsR::new(name, *period)?)
                }
                IndicatorSpec::Cci { name, period } => pipeline.add(Cci::new(name, *period)?),
                IndicatorSpec::Roc { name, period } => pipeline.add(Roc::new(name, *period)?),
                IndicatorSpec::BollingerB {
                    name,
                    period,
                    std_dev,
                } => pipeline.add(BollingerB::new(name, *period, *std_dev)?),
                IndicatorSpec::Atr { name, period } => pipeline.add(Atr::new(name, *period)?),
                IndicatorSpec::Obv { name } => pipeline.add(Obv::new(name)),
                IndicatorSpec::Mfi { name, period } => pipeline.add(Mfi::new(name, *period)?),
            };
        }

        let names = specs.iter().map(|spec| spec.name().to_string()).collect();

        Ok(Self {
            pipeline,
            names,
            symbol,
        })
    }

    pub fn indicator_names(&self) -> &[String] {
        &self.names
    }

    pub fn num_indicators(&self) -> usize {
        self.names.len()
    }

    pub fn update(
        &mut self,
        ts_nanos: i64,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
        volume: f64,
    ) -> Vec<Option<f64>> {
        let bar = OhlcvBar {
            symbol: self.symbol.clone(),
            open: Price::new(Decimal::from_f64_retain(open).unwrap_or(dec!(0))).unwrap(),
            high: Price::new(Decimal::from_f64_retain(high).unwrap_or(dec!(0))).unwrap(),
            low: Price::new(Decimal::from_f64_retain(low).unwrap_or(dec!(0))).unwrap(),
            close: Price::new(Decimal::from_f64_retain(close).unwrap_or(dec!(0))).unwrap(),
            volume: Quantity::new(Decimal::from_f64_retain(volume).unwrap_or(dec!(0))).unwrap(),
            ts_open: NanoTimestamp::new(ts_nanos),
            ts_close: NanoTimestamp::new(ts_nanos + 60_000_000_000),
            tick_count: 1,
        };

        let signal_map = self.pipeline.update(&bar);

        self.names
            .iter()
            .map(|name| match signal_map.get(name) {
                Some(SignalValue::Scalar(v)) => v.to_f64(),
                _ => None,
            })
            .collect()
    }
}
