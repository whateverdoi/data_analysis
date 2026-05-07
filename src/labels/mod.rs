pub mod label_fns;

use label_fns::{
    default_prediction_to_signal, label_future_return_5, label_next_close_direction,
    triple_barrier, LabelFn, PredictionToSignalFn,
};

pub fn label_by_name(name: &str) -> Option<(LabelFn, PredictionToSignalFn)> {
    match name {
        "triple_barrier" => Some((triple_barrier, default_prediction_to_signal)),
        "next_close_direction" => Some((label_next_close_direction, default_prediction_to_signal)),
        "future_return_5" => Some((label_future_return_5, default_prediction_to_signal)),
        _ => None,
    }
}
