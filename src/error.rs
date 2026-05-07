#![allow(dead_code)]

use std::fmt;

#[derive(Debug)]
pub enum DataAnalysisError {
    Io(std::io::Error),
    Csv(csv::Error),
    Fin(fin_primitives::FinError),
    SerdeJson(serde_json::Error),
    Ort(String),
    Config(String),
    Other(String),
}

impl fmt::Display for DataAnalysisError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "IO error: {}", e),
            Self::Csv(e) => write!(f, "CSV error: {}", e),
            Self::Fin(e) => write!(f, "fin_primitives error: {}", e),
            Self::SerdeJson(e) => write!(f, "JSON error: {}", e),
            Self::Ort(e) => write!(f, "ONNX Runtime error: {}", e),
            Self::Config(e) => write!(f, "Configuration error: {}", e),
            Self::Other(e) => write!(f, "{}", e),
        }
    }
}

impl std::error::Error for DataAnalysisError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::Csv(e) => Some(e),
            Self::Fin(e) => Some(e),
            Self::SerdeJson(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for DataAnalysisError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<csv::Error> for DataAnalysisError {
    fn from(e: csv::Error) -> Self {
        Self::Csv(e)
    }
}

impl From<fin_primitives::FinError> for DataAnalysisError {
    fn from(e: fin_primitives::FinError) -> Self {
        Self::Fin(e)
    }
}

impl From<serde_json::Error> for DataAnalysisError {
    fn from(e: serde_json::Error) -> Self {
        Self::SerdeJson(e)
    }
}

impl From<String> for DataAnalysisError {
    fn from(e: String) -> Self {
        Self::Other(e)
    }
}

impl From<&str> for DataAnalysisError {
    fn from(e: &str) -> Self {
        Self::Other(e.to_string())
    }
}

pub type Result<T> = std::result::Result<T, DataAnalysisError>;
