//! Everything after sync: chunking, synthetic queries, datasets, training,
//! evaluation and export.
//!
//! Each step saves an [`artifact`] keyed by everything that went into it, so
//! repeating a step with the same inputs is free and an interrupted step can
//! be resumed. Training and evaluation run in the Python [`sidecar`].

pub mod advice;
pub mod artifact;
pub mod catalog;
pub mod chunk;
pub mod config;
pub mod dataset;
pub mod exec;
pub mod export;
pub mod fake_trainer;
pub mod fetch;
pub mod generator;
pub mod graph;
pub mod hub;
pub mod ids;
pub mod interrupt;
pub mod lang;
pub mod ops;
pub mod queries;
pub mod report;
pub mod report_html;
pub mod sidecar;
pub mod stats;
pub mod sync;
pub mod tokenizer;
pub mod train;

pub use config::PipelineConfig;
