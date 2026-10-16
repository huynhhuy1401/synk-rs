//! The Hugging Face Hub: the local cache, repository metadata, and which
//! files a download needs.
//!
//! Metadata requests send only a repository name (and `HF_TOKEN` when set,
//! for private or gated models). `HF_ENDPOINT` points at a mirror, as for
//! every other Hugging Face tool. Downloads themselves happen in the
//! trainer, with `huggingface_hub`, into the same shared cache.

use std::fs;
use std::path::PathBuf;
use std::time::Duration;

use serde_json::Value;
use synk_core::{ErrorCode, Result, SynkError};

/// `HF_ENDPOINT`, or the Hugging Face Hub.
pub fn endpoint() -> String {
    std::env::var("HF_ENDPOINT")
        .unwrap_or_else(|_| "https://huggingface.co".to_owned())
        .trim_end_matches('/')
        .to_owned()
}

/// The Hub cache directory (`HF_HUB_CACHE`, `HF_HOME/hub`, or
/// `~/.cache/huggingface/hub`).
pub fn cache_dir() -> Option<PathBuf> {
    std::env::var_os("HF_HUB_CACHE")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HF_HOME").map(|h| PathBuf::from(h).join("hub")))
        .or_else(|| dirs::home_dir().map(|h| h.join(".cache").join("huggingface").join("hub")))
}

/// `model` or `dataset`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RepoType {
    Model,
    Dataset,
}

impl RepoType {
    fn cache_prefix(self) -> &'static str {
        match self {
            Self::Model => "models",
            Self::Dataset => "datasets",
        }
    }

    fn api(self) -> &'static str {
        match self {
            Self::Model => "models",
            Self::Dataset => "datasets",
        }
    }
}

/// The cached snapshot of `repo` at `revision` (a commit or a branch), if
/// one is there with at least one file in it.
pub fn snapshot(kind: RepoType, repo: &str, revision: &str) -> Option<PathBuf> {
    let dir = cache_dir()?.join(format!(
        "{}--{}",
        kind.cache_prefix(),
        repo.replace('/', "--")
    ));
    let commit = fs::read_to_string(dir.join("refs").join(revision))
        .map(|c| c.trim().to_owned())
        .unwrap_or_else(|_| revision.to_owned());
    let path = dir.join("snapshots").join(commit);
    let has_files = fs::read_dir(&path).ok()?.next().is_some();
    has_files.then_some(path)
}

/// A repository at one commit, and the files a download takes.
#[derive(Debug, Clone)]
pub struct RepoInfo {
    pub commit: String,
    /// `(path, bytes)` of the files synk downloads.
    pub files: Vec<(String, u64)>,
}

impl RepoInfo {
    pub fn bytes(&self) -> u64 {
        self.files.iter().map(|(_, b)| b).sum()
    }
}

/// Files a sentence-transformers model doesn't need: other frameworks'
/// weights, exported copies, and `.bin` weights when safetensors exist.
pub fn wanted(path: &str, has_safetensors: bool) -> bool {
    let lower = path.to_ascii_lowercase();
    let skip_dirs = [
        "onnx/",
        "openvino/",
        "coreml/",
        "tflite/",
        "imgs/",
        "images/",
    ];
    if skip_dirs
        .iter()
        .any(|d| lower.starts_with(d) || lower.contains(&format!("/{d}")))
    {
        return false;
    }
    let skip_ext = [
        ".onnx", ".h5", ".msgpack", ".ot", ".gguf", ".tflite", ".mlmodel",
    ];
    if skip_ext.iter().any(|e| lower.ends_with(e)) {
        return false;
    }
    !(has_safetensors && lower.ends_with(".bin"))
}

/// Patterns for `snapshot_download(ignore_patterns=...)` matching
/// [`wanted`].
pub fn ignore_patterns(has_safetensors: bool) -> Vec<&'static str> {
    let mut out = vec![
        "onnx/*",
        "openvino/*",
        "coreml/*",
        "tflite/*",
        "*.onnx",
        "*.h5",
        "*.msgpack",
        "*.ot",
        "*.gguf",
        "*.tflite",
        "*.mlmodel",
    ];
    if has_safetensors {
        out.push("*.bin");
    }
    out
}

/// The commit `revision` points at and the files synk would download.
pub fn info(kind: RepoType, repo: &str, revision: &str) -> Result<RepoInfo> {
    let url = format!(
        "{}/api/{}/{repo}/revision/{revision}?blobs=true",
        endpoint(),
        kind.api()
    );
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_global(Some(Duration::from_secs(15)))
        .http_status_as_error(false)
        .build()
        .into();
    let mut request = agent.get(&url);
    if let Ok(token) = std::env::var("HF_TOKEN") {
        request = request.header("Authorization", &format!("Bearer {token}"));
    }
    let mut response = request.call().map_err(|e| {
        SynkError::new(
            ErrorCode::DownloadFailed,
            format!("could not reach {}: {e}", endpoint()),
        )
    })?;
    match response.status().as_u16() {
        200 => {}
        401 | 403 => {
            return Err(SynkError::new(
                ErrorCode::ModelAccessDenied,
                format!("{repo} is private or gated, and this account can't read it"),
            )
            .with_hint(format!(
                "accept its terms on {}/{repo}, then set HF_TOKEN to a token of that account",
                endpoint()
            )));
        }
        404 => {
            return Err(SynkError::new(
                ErrorCode::ModelNotFound,
                format!("{repo}@{revision} doesn't exist on {}", endpoint()),
            )
            .with_next("synk model list"));
        }
        code => {
            return Err(SynkError::new(
                ErrorCode::DownloadFailed,
                format!("{} answered HTTP {code} for {repo}", endpoint()),
            ));
        }
    }
    let body: Value = response
        .body_mut()
        .read_json()
        .map_err(|e| SynkError::new(ErrorCode::DownloadFailed, e.to_string()))?;
    let commit = body["sha"].as_str().unwrap_or(revision).to_owned();
    let all: Vec<(String, u64)> = body["siblings"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|s| {
            Some((
                s["rfilename"].as_str()?.to_owned(),
                s["size"]
                    .as_u64()
                    .or_else(|| s["lfs"]["size"].as_u64())
                    .unwrap_or(0),
            ))
        })
        .collect();
    let has_safetensors = all.iter().any(|(p, _)| p.ends_with(".safetensors"));
    let files = all
        .into_iter()
        .filter(|(p, _)| kind == RepoType::Dataset || wanted(p, has_safetensors))
        .collect();
    Ok(RepoInfo { commit, files })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_the_weights_sentence_transformers_loads() {
        assert!(wanted("model.safetensors", true));
        assert!(wanted("1_Pooling/config.json", true));
        assert!(!wanted("pytorch_model.bin", true));
        assert!(wanted("pytorch_model.bin", false));
        assert!(!wanted("onnx/model.onnx", true));
        assert!(!wanted("openvino/openvino_model.xml", false));
        assert!(!wanted("tf_model.h5", false));
    }
}
