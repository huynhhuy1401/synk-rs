use serde::Serialize;

use crate::FileState;

/// A normalized text document produced by a connector.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Document {
    /// Stable ID: `<source>/<relative path>` for folder sources, or
    /// `<source>` for single-file sources.
    pub id: String,
    pub source: String,
    /// Path relative to the source root, with `/` separators.
    pub rel_path: String,
    pub title: String,
    pub text: String,
    #[serde(flatten)]
    pub state: FileState,
}
