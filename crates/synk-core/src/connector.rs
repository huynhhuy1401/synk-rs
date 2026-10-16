use std::collections::HashMap;

use serde::Serialize;

use crate::{Document, Result};

/// What the store already knows about a source: document ID → last seen state.
pub type KnownFiles = HashMap<String, FileState>;

/// What a connector gets before a scan: the documents already stored, and
/// the cursor it returned last time.
#[derive(Debug, Default, Clone)]
pub struct SourceState {
    pub known: KnownFiles,
    /// An opaque position an API connector saved after its last scan (a
    /// page token, a timestamp). Folders don't use it.
    pub cursor: Option<serde_json::Value>,
}

impl From<KnownFiles> for SourceState {
    fn from(known: KnownFiles) -> Self {
        Self {
            known,
            cursor: None,
        }
    }
}

/// Reads one source and reports what changed since the last sync.
pub trait Connector {
    /// Name of the source this connector reads.
    fn source(&self) -> &str;

    /// Scans the source and returns only the differences from what the
    /// store knows.
    fn scan(&self, state: &SourceState) -> Result<SyncBatch>;
}

/// Change-detection state for one document.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct FileState {
    pub size: u64,
    /// Modification time, in nanoseconds since the Unix epoch.
    pub modified: i64,
    /// blake3 hash of the raw file contents.
    pub hash: String,
}

#[derive(Debug, Default)]
pub struct SyncBatch {
    /// New or edited documents.
    pub upserts: Vec<Document>,
    /// Documents whose metadata changed but whose contents did not.
    pub touched: Vec<(String, FileState)>,
    /// IDs that are no longer in the source.
    pub deleted: Vec<String>,
    pub unchanged: usize,
    pub skipped: Vec<Skipped>,
    /// The cursor to hand back on the next scan, if the connector uses one.
    pub cursor: Option<serde_json::Value>,
}

/// A file the connector saw but could not read.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Skipped {
    pub path: String,
    pub reason: String,
}
