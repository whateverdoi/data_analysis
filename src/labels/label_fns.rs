use crate::data::market_data::MarketData;

pub type LabelFn = fn(&LabelContext<'_>) -> Option<i32>;
pub type PredictionToSignalFn = fn(i32) -> i32;

pub struct LabelContext<'a> {
    pub index: usize,
    market_data: &'a MarketData,
}

impl<'a> LabelContext<'a> {
    pub fn new(index: usize, market_data: &'a MarketData) -> Self {
        Self { index, market_data }
    }

    #[allow(dead_code)]
    pub fn open(&self, index: usize) -> Option<f64> {
        self.market_data.opens.get(index).copied()
    }

    #[allow(dead_code)]
    pub fn high(&self, index: usize) -> Option<f64> {
        self.market_data.highs.get(index).copied()
    }

    #[allow(dead_code)]
    pub fn low(&self, index: usize) -> Option<f64> {
        self.market_data.lows.get(index).copied()
    }

    pub fn close(&self, index: usize) -> Option<f64> {
        self.market_data.closes.get(index).copied()
    }

    #[allow(dead_code)]
    pub fn volume(&self, index: usize) -> Option<f64> {
        self.market_data.volumes.get(index).copied()
    }

    #[allow(dead_code)]
    pub fn indicator(&self, index: usize, indicator_index: usize) -> Option<f64> {
        self.market_data
            .indicator_values
            .get(index)?
            .get(indicator_index)
            .copied()
            .flatten()
    }
}

pub fn label_next_close_direction(ctx: &LabelContext<'_>) -> Option<i32> {
    let current = ctx.close(ctx.index)?;
    let next = ctx.close(ctx.index + 1)?;
    Some(if next > current { 1 } else { 0 })
}

#[allow(dead_code)]
pub fn label_future_return_5(ctx: &LabelContext<'_>) -> Option<i32> {
    let current = ctx.close(ctx.index)?;
    let future = ctx.close(ctx.index + 5)?;
    let ret = future / current - 1.0;

    Some(if ret > 0.003 {
        2
    } else if ret < -0.003 {
        0
    } else {
        1
    })
}

pub fn triple_barrier(ctx: &LabelContext<'_>) -> Option<i32> {
    let atr_period: usize = 14;
    let upper_mult: f64 = 2.0;
    let lower_mult: f64 = 2.0;
    let max_hold: usize = 20;

    if ctx.index < atr_period {
        return None;
    }

    let entry = ctx.close(ctx.index)?;
    let atr = atr_sma(ctx, ctx.index + 1 - atr_period, ctx.index)?;

    let upper = entry + upper_mult * atr;
    let lower = entry - lower_mult * atr;
    let end = ctx.index + max_hold;
    let last = ctx.market_data.closes.len() - 1;

    for i in (ctx.index + 1)..=end.min(last) {
        let hi = ctx.high(i)?;
        let lo = ctx.low(i)?;
        if hi >= upper {
            return Some(1);
        }
        if lo <= lower {
            return Some(0);
        }
    }

    Some(2)
}

fn atr_sma(ctx: &LabelContext<'_>, start: usize, end: usize) -> Option<f64> {
    let mut sum = 0.0;
    let n = end - start + 1;
    for i in start..=end {
        let hi = ctx.high(i)?;
        let lo = ctx.low(i)?;
        let pc = ctx.close(i - 1)?;
        sum += (hi - lo).max((hi - pc).abs()).max((lo - pc).abs());
    }
    Some(sum / n as f64)
}

pub fn default_prediction_to_signal(label: i32) -> i32 {
    if label == 1 {
        1
    } else {
        0
    }
}
