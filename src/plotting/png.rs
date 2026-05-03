use plotters::prelude::*;
use std::path::Path;

pub fn render_ohlcv(
    path: &Path,
    _dates: &[String],
    opens: &[f64],
    highs: &[f64],
    lows: &[f64],
    closes: &[f64],
    volumes: &[f64],
    sma20: &[Option<f64>],
    sma50: &[Option<f64>],
) {
    let n = closes.len();
    if n == 0 {
        return;
    }

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        draw_chart_inner(path, opens, highs, lows, closes, volumes, sma20, sma50, n)
    }));

    match result {
        Ok(Ok(())) => println!("Chart saved to {}", path.display()),
        Ok(Err(e)) => eprintln!("Chart render error: {}", e),
        Err(_) => eprintln!("Chart render panicked (font not available) - skipping PNG output"),
    }
}

fn draw_chart_inner(
    path: &Path,
    opens: &[f64],
    highs: &[f64],
    lows: &[f64],
    closes: &[f64],
    volumes: &[f64],
    sma20: &[Option<f64>],
    sma50: &[Option<f64>],
    n: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let price_min = lows.iter().cloned().fold(f64::INFINITY, f64::min);
    let price_max = highs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let vol_max = volumes.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let root =
        SVGBackend::new(path.to_str().unwrap_or("chart.svg"), (1200, 800))
            .into_drawing_area();
    root.fill(&RGBColor(24, 24, 27))?;

    let (upper, lower) = root.split_vertically(560);

    let margin = (price_max - price_min) * 0.05;
    let mut price_chart = ChartBuilder::on(&upper)
        .margin(5)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(0f64..n as f64, price_min - margin..price_max + margin)?;

    price_chart.configure_mesh().draw()?;

    for i in 0..n {
        let color = if closes[i] >= opens[i] {
            RGBColor(34, 197, 94)
        } else {
            RGBColor(239, 68, 68)
        };
        let x = i as f64;
        let bar_w = 0.6;
        let hi = highs[i];
        let lo = lows[i];
        let body_top = closes[i].max(opens[i]);
        let body_bottom = closes[i].min(opens[i]);

        price_chart
            .draw_series(std::iter::once(Rectangle::new(
                [(x - bar_w / 2.0, body_bottom), (x + bar_w / 2.0, body_top)],
                color.filled(),
            )))
            .ok();

        price_chart
            .draw_series(LineSeries::new(
                vec![(x, lo), (x, hi)],
                color.stroke_width(1),
            ))
            .ok();
    }

    if sma20.iter().any(|v| v.is_some()) {
        let pts: Vec<(f64, f64)> = sma20
            .iter()
            .enumerate()
            .filter_map(|(i, v)| v.map(|val| (i as f64, val)))
            .collect();
        price_chart
            .draw_series(LineSeries::new(pts, &RGBColor(251, 191, 36)))
            .ok();
    }

    if sma50.iter().any(|v| v.is_some()) {
        let pts: Vec<(f64, f64)> = sma50
            .iter()
            .enumerate()
            .filter_map(|(i, v)| v.map(|val| (i as f64, val)))
            .collect();
        price_chart
            .draw_series(LineSeries::new(pts, &RGBColor(59, 130, 246)))
            .ok();
    }

    let mut vol_chart = ChartBuilder::on(&lower)
        .margin(5)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(0f64..n as f64, 0f64..vol_max)?;

    vol_chart.configure_mesh().draw()?;

    for i in 0..n {
        let color = if closes[i] >= opens[i] {
            RGBColor(34, 197, 94).mix(0.6)
        } else {
            RGBColor(239, 68, 68).mix(0.6)
        };
        let x = i as f64;
        vol_chart
            .draw_series(std::iter::once(Rectangle::new(
                [(x - 0.3, 0.0), (x + 0.3, volumes[i])],
                color.filled(),
            )))
            .ok();
    }

    root.present()?;
    Ok(())
}
