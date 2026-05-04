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
    names: Vec<&'static str>,
    symbol: Symbol,
}

impl IndicatorEngine {
    pub fn new(symbol_name: &str) -> Result<Self, FinError> {
        let symbol = Symbol::new(symbol_name)?;
        let pipeline = SignalPipeline::new()
            .add(Sma::new("sma20", 20).unwrap())
            .add(Sma::new("sma50", 50).unwrap())
            .add(Ema::new("ema12", 12).unwrap())
            .add(Ema::new("ema26", 26).unwrap())
            .add(Rsi::new("rsi14", 14).unwrap())
            .add(Macd::new("macd", 12, 26, 9).unwrap())
            .add(StochasticK::new("stoch_k", 14).unwrap())
            .add(WilliamsR::new("willr14", 14).unwrap())
            .add(Cci::new("cci20", 20).unwrap())
            .add(Roc::new("roc10", 10).unwrap())
            .add(BollingerB::new("bb_sma", 20, dec!(2)).unwrap())
            .add(Atr::new("atr14", 14).unwrap())
            .add(Obv::new("obv"))
            .add(Mfi::new("mfi14", 14).unwrap());

        let names = vec![
            "sma20", "sma50", "ema12", "ema26", "rsi14", "macd", "stoch_k", "willr14", "cci20",
            "roc10", "bb_sma", "atr14", "obv", "mfi14",
        ];

        Ok(Self {
            pipeline,
            names,
            symbol,
        })
    }

    pub fn indicator_names(&self) -> &[&'static str] {
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
