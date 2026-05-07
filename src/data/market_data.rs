pub(crate) struct MarketData {
    pub(crate) opens: Vec<f64>,
    pub(crate) highs: Vec<f64>,
    pub(crate) lows: Vec<f64>,
    pub(crate) closes: Vec<f64>,
    pub(crate) volumes: Vec<f64>,
    pub(crate) timestamps: Vec<String>,
    pub(crate) indicator_values: Vec<Vec<Option<f64>>>,
}

impl MarketData {
    pub(crate) fn new() -> Self {
        Self {
            opens: Vec::new(),
            highs: Vec::new(),
            lows: Vec::new(),
            closes: Vec::new(),
            volumes: Vec::new(),
            timestamps: Vec::new(),
            indicator_values: Vec::new(),
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.closes.len()
    }
}
