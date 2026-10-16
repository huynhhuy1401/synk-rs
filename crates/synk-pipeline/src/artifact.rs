//! Saved outputs of pipeline steps.
//!
//! An artifact lives in `.synk/artifacts/<kind>/<key>/` with a
//! `manifest.json`. The key is a hash of the step's kind, its version, its
//! parameters and the keys of its inputs, so changing any of them produces a
//! new artifact instead of silently reusing a stale one. A step writes into
//! `<key>.tmp/` and renames it when done, so a crash never leaves something
//! that looks complete.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use synk_core::{Result, SynkError};

pub const MANIFEST_SCHEMA: u32 = 1;
const MANIFEST_FILE: &str = "manifest.json";
const LATEST_FILE: &str = "latest";

/// Key for an artifact. `params` is hashed as canonical JSON (serde_json
/// sorts object keys), so field order never changes a key.
pub fn key(kind: &str, kind_version: u32, params: &Value, inputs: &[String]) -> String {
    let canonical = json!({
        "kind": kind,
        "kind_version": kind_version,
        "params": params,
        "inputs": inputs,
    });
    blake3::hash(canonical.to_string().as_bytes()).to_hex()[..16].to_owned()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    pub schema: u32,
    pub kind: String,
    pub key: String,
    pub synk_version: String,
    /// Seconds since the Unix epoch.
    pub created_at: u64,
    pub params: Value,
    pub inputs: Vec<String>,
    /// What the step produced, for `synk status` and `--json` output.
    pub summary: Value,
}

impl Manifest {
    pub fn new(kind: &str, key: &str, params: Value, inputs: Vec<String>, summary: Value) -> Self {
        Self {
            schema: MANIFEST_SCHEMA,
            kind: kind.to_owned(),
            key: key.to_owned(),
            synk_version: env!("CARGO_PKG_VERSION").to_owned(),
            created_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |d| d.as_secs()),
            params,
            inputs,
            summary,
        }
    }
}

pub struct Artifacts {
    root: PathBuf,
}

impl Artifacts {
    /// `root` is `.synk/artifacts`.
    pub fn new(root: PathBuf) -> Self {
        Self { root }
    }

    /// `.synk/`, the folder above the artifacts.
    pub fn data_dir(&self) -> PathBuf {
        self.root
            .parent()
            .map_or_else(|| self.root.clone(), Path::to_path_buf)
    }

    pub fn dir(&self, kind: &str, key: &str) -> PathBuf {
        self.root.join(kind).join(key)
    }

    pub fn tmp_dir(&self, kind: &str, key: &str) -> PathBuf {
        self.root.join(kind).join(format!("{key}.tmp"))
    }

    /// The manifest of a finished artifact, if it exists.
    pub fn manifest(&self, kind: &str, key: &str) -> Result<Option<Manifest>> {
        let path = self.dir(kind, key).join(MANIFEST_FILE);
        if !path.is_file() {
            return Ok(None);
        }
        let text = fs::read_to_string(&path).map_err(|e| SynkError::io(path.display(), e))?;
        serde_json::from_str(&text)
            .map(Some)
            .map_err(|e| SynkError::io(path.display(), std::io::Error::other(e)))
    }

    /// Creates the work directory for an artifact. With `fresh`, anything an
    /// interrupted earlier attempt left there is removed first.
    pub fn begin(&self, kind: &str, key: &str, fresh: bool) -> Result<PathBuf> {
        let tmp = self.tmp_dir(kind, key);
        if fresh && tmp.exists() {
            remove_dir(&tmp)?;
        }
        fs::create_dir_all(&tmp).map_err(|e| SynkError::io(tmp.display(), e))?;
        Ok(tmp)
    }

    /// Writes the manifest and moves the work directory into place.
    pub fn commit(&self, manifest: &Manifest) -> Result<PathBuf> {
        let tmp = self.tmp_dir(&manifest.kind, &manifest.key);
        let path = tmp.join(MANIFEST_FILE);
        let text = serde_json::to_string_pretty(manifest).expect("manifest serializes");
        fs::write(&path, text).map_err(|e| SynkError::io(path.display(), e))?;
        let dir = self.dir(&manifest.kind, &manifest.key);
        if dir.exists() {
            remove_dir(&dir)?;
        }
        fs::rename(&tmp, &dir).map_err(|e| SynkError::io(dir.display(), e))?;
        Ok(dir)
    }

    /// Removes a finished artifact and any unfinished attempt at it.
    pub fn remove(&self, kind: &str, key: &str) -> Result<()> {
        for dir in [self.dir(kind, key), self.tmp_dir(kind, key)] {
            if dir.exists() {
                remove_dir(&dir)?;
            }
        }
        Ok(())
    }

    /// Every finished artifact of `kind`, newest first.
    pub fn list(&self, kind: &str) -> Result<Vec<Manifest>> {
        let dir = self.root.join(kind);
        let entries = match fs::read_dir(&dir) {
            Ok(entries) => entries,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => return Err(SynkError::io(dir.display(), e)),
        };
        let mut out = Vec::new();
        for entry in entries {
            let entry = entry.map_err(|e| SynkError::io(dir.display(), e))?;
            let name = entry.file_name().to_string_lossy().into_owned();
            if entry.path().is_dir()
                && !name.ends_with(".tmp")
                && let Some(m) = self.manifest(kind, &name)?
            {
                out.push(m);
            }
        }
        out.sort_by(|a, b| b.created_at.cmp(&a.created_at).then(a.key.cmp(&b.key)));
        Ok(out)
    }

    /// A finished artifact by its key or a unique start of it.
    pub fn find(&self, kind: &str, prefix: &str) -> Result<Option<Manifest>> {
        if let Some(m) = self.manifest(kind, prefix)? {
            return Ok(Some(m));
        }
        let mut found: Vec<Manifest> = self
            .list(kind)?
            .into_iter()
            .filter(|m| m.key.starts_with(prefix))
            .collect();
        match found.len() {
            0 | 1 => Ok(found.pop()),
            n => Err(SynkError::new(
                synk_core::ErrorCode::InvalidArgs,
                format!("`{prefix}` matches {n} {kind} artifacts; give more of the key"),
            )),
        }
    }

    pub fn set_latest(&self, kind: &str, key: &str) -> Result<()> {
        let path = self.root.join(kind).join(LATEST_FILE);
        fs::write(&path, key).map_err(|e| SynkError::io(path.display(), e))
    }

    /// The artifact of `kind` that the last command produced or reused.
    pub fn latest(&self, kind: &str) -> Result<Option<Manifest>> {
        let path = self.root.join(kind).join(LATEST_FILE);
        match fs::read_to_string(&path) {
            Ok(key) => self.manifest(kind, key.trim()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(SynkError::io(path.display(), e)),
        }
    }
}

fn remove_dir(path: &Path) -> Result<()> {
    fs::remove_dir_all(path).map_err(|e| SynkError::io(path.display(), e))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_ignores_field_order_but_not_values() {
        let a = key("dataset", 1, &json!({"a": 1, "b": 2}), &["x".into()]);
        let b = key("dataset", 1, &json!({"b": 2, "a": 1}), &["x".into()]);
        assert_eq!(a, b);
        assert_ne!(
            a,
            key("dataset", 2, &json!({"a": 1, "b": 2}), &["x".into()])
        );
        assert_ne!(
            a,
            key("dataset", 1, &json!({"a": 1, "b": 3}), &["x".into()])
        );
        assert_ne!(
            a,
            key("dataset", 1, &json!({"a": 1, "b": 2}), &["y".into()])
        );
    }

    #[test]
    fn unfinished_work_is_invisible_until_commit() {
        let dir = tempfile::tempdir().unwrap();
        let artifacts = Artifacts::new(dir.path().join("artifacts"));
        let tmp = artifacts.begin("dataset", "k1", true).unwrap();
        fs::write(tmp.join("data.jsonl"), "{}").unwrap();
        assert!(artifacts.manifest("dataset", "k1").unwrap().is_none());

        let manifest = Manifest::new("dataset", "k1", json!({}), vec![], json!({"n": 1}));
        let done = artifacts.commit(&manifest).unwrap();
        assert!(done.join("data.jsonl").is_file());
        assert!(!artifacts.tmp_dir("dataset", "k1").exists());
        assert_eq!(
            artifacts
                .manifest("dataset", "k1")
                .unwrap()
                .unwrap()
                .summary["n"],
            1
        );

        artifacts.set_latest("dataset", "k1").unwrap();
        assert_eq!(artifacts.latest("dataset").unwrap().unwrap().key, "k1");
        assert_eq!(artifacts.find("dataset", "k").unwrap().unwrap().key, "k1");
        assert!(artifacts.find("dataset", "x").unwrap().is_none());
        assert_eq!(artifacts.list("dataset").unwrap().len(), 1);
        assert!(artifacts.list("train").unwrap().is_empty());
    }
}
