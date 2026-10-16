//! Core types and traits shared by every synk crate.

mod connector;
mod document;
mod error;
mod source;

pub use connector::{Connector, FileState, KnownFiles, Skipped, SourceState, SyncBatch};
pub use document::Document;
pub use error::{Applicability, ErrorAction, ErrorClass, ErrorCode, Result, SynkError};
pub use source::{FOLDER, SourceSpec};
