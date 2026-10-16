use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// One `[[sources]]` entry: a file or a folder to read documents from.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceSpec {
    pub name: String,
    /// The connector that reads it. Only `folder` (files and folders) exists
    /// today.
    #[serde(rename = "type", default = "folder", skip_serializing_if = "is_folder")]
    pub kind: String,
    pub path: PathBuf,
    /// Glob patterns, relative to the source root, to leave out.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub exclude: Vec<String>,
}

/// The connector for local files and folders.
pub const FOLDER: &str = "folder";

fn folder() -> String {
    FOLDER.to_owned()
}

fn is_folder(kind: &str) -> bool {
    kind == FOLDER
}

impl SourceSpec {
    /// A file or folder source.
    pub fn folder(name: impl Into<String>, path: PathBuf, exclude: Vec<String>) -> Self {
        Self {
            name: name.into(),
            kind: folder(),
            path,
            exclude,
        }
    }
}
