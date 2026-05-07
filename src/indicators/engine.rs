use fin_primitives::{
    ohlcv::OhlcvBar,
    signals::SignalValue,
    types::{NanoTimestamp, Price, Quantity, Symbol},
    FinError,
};
use rust_decimal::prelude::{FromPrimitive, ToPrimitive};
use rust_decimal::Decimal;

use crate::indicators::specs::IndicatorSpec;

pub struct IndicatorEngine {
    pipeline: fin_primitives::signals::pipeline::SignalPipeline,
    names: Vec<String>,
    symbol: Symbol,
}

impl IndicatorEngine {
    pub fn new(symbol_name: &str, specs: &[IndicatorSpec]) -> Result<Self, FinError> {
        let symbol = Symbol::new(symbol_name)?;
        let mut pipeline = fin_primitives::signals::pipeline::SignalPipeline::new();
        for spec in specs {
            pipeline = spec.build(pipeline)?;
        }
        let names = specs.iter().map(|s| s.name().to_string()).collect();
        Ok(Self { pipeline, names, symbol })
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
            open: Price::new(Decimal::from_f64(open).unwrap_or_default()).unwrap(),
            high: Price::new(Decimal::from_f64(high).unwrap_or_default()).unwrap(),
            low: Price::new(Decimal::from_f64(low).unwrap_or_default()).unwrap(),
            close: Price::new(Decimal::from_f64(close).unwrap_or_default()).unwrap(),
            volume: Quantity::new(Decimal::from_f64(volume).unwrap_or_default()).unwrap(),
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
