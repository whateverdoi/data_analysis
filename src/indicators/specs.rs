use fin_primitives::error::FinError;
use fin_primitives::signals::indicators::{
    Adx, Aroon, Atr, BollingerB, Cci, ChaikinOsc, ElderRay, Ema, ForceIndex, KeltnerChannel,
    Macd, Mfi, Momentum, Obv, Roc, Rsi, Sma, StochasticK, Trix, Vwap, WilliamsAD, WilliamsR,
};
use fin_primitives::signals::pipeline::SignalPipeline;
use rust_decimal::prelude::FromPrimitive;
use rust_decimal::Decimal;

#[derive(Debug, Clone)]
pub enum IndicatorKind {
    Sma     { period: usize },
    Ema     { period: usize },
    Rsi     { period: usize },
    Macd    { fast_period: usize, slow_period: usize, signal_period: usize },
    StochasticK { period: usize },
    WilliamsR    { period: usize },
    Cci     { period: usize },
    Roc     { period: usize },
    BollingerB   { period: usize, std_dev: f64 },
    Atr     { period: usize },
    Obv,
    Mfi     { period: usize },
    Adx     { period: usize },
    Aroon   { period: usize },
    ForceIndex { period: usize },
    ChaikinOsc  { fast_period: usize, slow_period: usize },
    Vwap,
    Momentum    { period: usize },
    Trix    { period: usize },
    KeltnerChannel { period: usize, multiplier: f64 },
    WilliamsAD,
    ElderRay    { period: usize },
}

#[derive(Debug, Clone)]
pub struct IndicatorSpec {
    pub name: String,
    pub kind: IndicatorKind,
}

impl IndicatorSpec {
    pub fn new(name: impl Into<String>, kind: IndicatorKind) -> Self {
        Self { name: name.into(), kind }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub(crate) fn build(&self, pipeline: SignalPipeline) -> Result<SignalPipeline, FinError> {
        match &self.kind {
            IndicatorKind::Sma { period }
                => Ok(pipeline.add(Sma::new(&self.name, *period)?)),
            IndicatorKind::Ema { period }
                => Ok(pipeline.add(Ema::new(&self.name, *period)?)),
            IndicatorKind::Rsi { period }
                => Ok(pipeline.add(Rsi::new(&self.name, *period)?)),
            IndicatorKind::Macd { fast_period, slow_period, signal_period }
                => Ok(pipeline.add(Macd::new(&self.name, *fast_period, *slow_period, *signal_period)?)),
            IndicatorKind::StochasticK { period }
                => Ok(pipeline.add(StochasticK::new(&self.name, *period)?)),
            IndicatorKind::WilliamsR { period }
                => Ok(pipeline.add(WilliamsR::new(&self.name, *period)?)),
            IndicatorKind::Cci { period }
                => Ok(pipeline.add(Cci::new(&self.name, *period)?)),
            IndicatorKind::Roc { period }
                => Ok(pipeline.add(Roc::new(&self.name, *period)?)),
            IndicatorKind::BollingerB { period, std_dev }
                => Ok(pipeline.add(BollingerB::new(&self.name, *period, Decimal::from_f64(*std_dev).unwrap_or_default())?)),
            IndicatorKind::Atr { period }
                => Ok(pipeline.add(Atr::new(&self.name, *period)?)),
            IndicatorKind::Obv
                => Ok(pipeline.add(Obv::new(&self.name))),
            IndicatorKind::Mfi { period }
                => Ok(pipeline.add(Mfi::new(&self.name, *period)?)),
            IndicatorKind::Adx { period }
                => Ok(pipeline.add(Adx::new(&self.name, *period)?)),
            IndicatorKind::Aroon { period }
                => Ok(pipeline.add(Aroon::new(&self.name, *period)?)),
            IndicatorKind::ForceIndex { period }
                => Ok(pipeline.add(ForceIndex::new(&self.name, *period)?)),
            IndicatorKind::ChaikinOsc { fast_period, slow_period }
                => Ok(pipeline.add(ChaikinOsc::new(&self.name, *fast_period, *slow_period)?)),
            IndicatorKind::Vwap
                => Ok(pipeline.add(Vwap::new(&self.name))),
            IndicatorKind::Momentum { period }
                => Ok(pipeline.add(Momentum::new(&self.name, *period)?)),
            IndicatorKind::Trix { period }
                => Ok(pipeline.add(Trix::new(&self.name, *period)?)),
            IndicatorKind::KeltnerChannel { period, multiplier }
                => Ok(pipeline.add(KeltnerChannel::new(&self.name, *period, Decimal::from_f64(*multiplier).unwrap_or_default())?)),
            IndicatorKind::WilliamsAD
                => Ok(pipeline.add(WilliamsAD::new(&self.name))),
            IndicatorKind::ElderRay { period }
                => Ok(pipeline.add(ElderRay::new(&self.name, *period)?)),
        }
    }
}
