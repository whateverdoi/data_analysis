use plotly::{
    common::{Line, Mode, Title},
    layout::{Axis, Layout, Legend},
    Plot, Scatter,
};
use std::path::Path;

pub fn render_dashboard(
    path: &Path,
    _dates: &[String],
    _opens: &[f64],
    _highs: &[f64],
    _lows: &[f64],
    closes: &[f64],
    _volumes: &[f64],
    sma20: &[Option<f64>],
    sma50: &[Option<f64>],
    equity_curve: &[f64],
    predictions: &[(usize, i32)],
) {
    let mut plot = Plot::new();

    let idx: Vec<usize> = (0..closes.len()).collect();

    let close_trace = Scatter::new(idx.clone(), closes.to_vec())
        .mode(Mode::LinesMarkers)
        .name("Close");

    plot.add_trace(close_trace);

    if sma20.iter().any(|v| v.is_some()) {
        let sma: Vec<f64> = sma20.iter().map(|v| v.unwrap_or(f64::NAN)).collect();
        plot.add_trace(
            Scatter::new(idx.clone(), sma)
                .mode(Mode::Lines)
                .name("SMA20")
                .line(Line::new().color("orange").width(1.5)),
        );
    }

    if sma50.iter().any(|v| v.is_some()) {
        let sma: Vec<f64> = sma50.iter().map(|v| v.unwrap_or(f64::NAN)).collect();
        plot.add_trace(
            Scatter::new(idx.clone(), sma)
                .mode(Mode::Lines)
                .name("SMA50")
                .line(Line::new().color("blue").width(1.5)),
        );
    }

    let n = closes.len();
    let buy_x: Vec<usize> = predictions
        .iter()
        .filter(|(i, p)| *p == 1 && *i < n)
        .map(|(i, _)| *i)
        .collect();
    let buy_y: Vec<f64> = buy_x.iter().map(|&i| closes[i]).collect();
    if !buy_x.is_empty() {
        plot.add_trace(
            Scatter::new(buy_x, buy_y)
                .mode(Mode::Markers)
                .name("Predict UP")
                .marker(plotly::common::Marker::new().size(10).color("green")),
        );
    }

    let sell_x: Vec<usize> = predictions
        .iter()
        .filter(|(i, p)| *p == 0 && *i < n)
        .map(|(i, _)| *i)
        .collect();
    let sell_y: Vec<f64> = sell_x.iter().map(|&i| closes[i]).collect();
    if !sell_x.is_empty() {
        plot.add_trace(
            Scatter::new(sell_x, sell_y)
                .mode(Mode::Markers)
                .name("Predict DOWN")
                .marker(plotly::common::Marker::new().size(10).color("red")),
        );
    }

    let eq_x: Vec<usize> = (0..equity_curve.len()).collect();
    plot.add_trace(
        Scatter::new(eq_x, equity_curve.to_vec())
            .mode(Mode::Lines)
            .name("Equity")
            .y_axis("y2")
            .line(Line::new().color("purple").width(2.0)),
    );

    let layout = Layout::new()
        .title(Title::new().text("Trading Dashboard"))
        .x_axis(Axis::new().title(Title::new().text("Bar Index")))
        .y_axis(Axis::new().title(Title::new().text("Price")))
        .y_axis2(
            Axis::new()
                .overlaying("y")
                .side(plotly::common::AxisSide::Right)
                .title(Title::new().text("Equity")),
        )
        .legend(Legend::new().x(0.01).y(0.99));

    plot.set_layout(layout);
    plot.write_html(path.to_str().unwrap_or("dashboard.html"));
    println!("Dashboard saved to {}", path.display());
}
