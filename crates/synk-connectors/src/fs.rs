//! Local files and folders.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use globset::{Glob, GlobSet, GlobSetBuilder};
use synk_core::{
    Connector, Document, ErrorCode, FileState, KnownFiles, Result, Skipped, SourceSpec,
    SourceState, SyncBatch, SynkError,
};

use crate::readers::{self, is_supported};

const MAX_FILE_BYTES: u64 = 20 * 1024 * 1024;

/// Reads a single file, or every supported file under a folder.
///
/// Folders are walked recursively. Hidden files and anything matched by a
/// `.gitignore` are skipped.
pub struct FsConnector {
    name: String,
    root: PathBuf,
    exclude: GlobSet,
}

impl FsConnector {
    /// `spec.path` must already be resolved to an absolute path.
    pub fn new(spec: &SourceSpec) -> Result<Self> {
        let mut builder = GlobSetBuilder::new();
        for pattern in &spec.exclude {
            let glob = Glob::new(pattern).map_err(|e| {
                SynkError::new(
                    ErrorCode::ConfigInvalid,
                    format!(
                        "source `{}`: invalid exclude pattern `{pattern}`: {e}",
                        spec.name
                    ),
                )
            })?;
            builder.add(glob);
        }
        let exclude = builder
            .build()
            .map_err(|e| SynkError::new(ErrorCode::ConfigInvalid, e.to_string()))?;
        Ok(Self {
            name: spec.name.clone(),
            root: spec.path.clone(),
            exclude,
        })
    }

    fn scan_file(&self, known: &KnownFiles, batch: &mut SyncBatch, seen: &mut HashSet<String>) {
        if !is_supported(&self.root) {
            batch
                .skipped
                .push(skipped(&self.root, "unsupported file type"));
            return;
        }
        let id = self.name.clone();
        seen.insert(id.clone());
        self.visit(&self.root, id, file_name(&self.root), known, batch);
    }

    fn scan_folder(&self, known: &KnownFiles, batch: &mut SyncBatch, seen: &mut HashSet<String>) {
        let walker = ignore::WalkBuilder::new(&self.root)
            .require_git(false)
            .build();
        for entry in walker {
            let entry = match entry {
                Ok(entry) => entry,
                Err(e) => {
                    batch.skipped.push(skipped(&self.root, e.to_string()));
                    continue;
                }
            };
            if !entry.file_type().is_some_and(|t| t.is_file()) {
                continue;
            }
            let path = entry.path();
            let rel = rel_path(&self.root, path);
            if !is_supported(path) || self.exclude.is_match(&rel) {
                continue;
            }
            let id = format!("{}/{rel}", self.name);
            // Mark as seen before reading, so a file we fail to read is
            // skipped rather than treated as deleted.
            seen.insert(id.clone());
            self.visit(path, id, rel, known, batch);
        }
    }

    fn visit(
        &self,
        path: &Path,
        id: String,
        rel: String,
        known: &KnownFiles,
        batch: &mut SyncBatch,
    ) {
        let meta = match std::fs::metadata(path) {
            Ok(meta) => meta,
            Err(e) => return batch.skipped.push(skipped(path, e.to_string())),
        };
        let size = meta.len();
        if size > MAX_FILE_BYTES {
            return batch.skipped.push(skipped(path, "larger than 20 MB"));
        }
        let modified = meta
            .modified()
            .ok()
            .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
            .map_or(0, |d| d.as_nanos() as i64);

        // Cheap check first: same size and mtime means unchanged.
        let previous = known.get(&id);
        if previous.is_some_and(|p| p.size == size && p.modified == modified) {
            batch.unchanged += 1;
            return;
        }

        let bytes = match std::fs::read(path) {
            Ok(bytes) => bytes,
            Err(e) => return batch.skipped.push(skipped(path, e.to_string())),
        };
        let state = FileState {
            size,
            modified,
            hash: blake3::hash(&bytes).to_hex().to_string(),
        };
        if previous.is_some_and(|p| p.hash == state.hash) {
            batch.touched.push((id, state));
            return;
        }
        let (title, text) = match readers::read(path, &bytes) {
            Ok(read) => read,
            Err(reason) => return batch.skipped.push(skipped(path, reason)),
        };
        batch.upserts.push(Document {
            id,
            source: self.name.clone(),
            rel_path: rel,
            title,
            text,
            state,
        });
    }
}

impl Connector for FsConnector {
    fn source(&self) -> &str {
        &self.name
    }

    fn scan(&self, state: &SourceState) -> Result<SyncBatch> {
        let known = &state.known;
        let meta = std::fs::metadata(&self.root).map_err(|_| {
            SynkError::new(
                ErrorCode::SourceMissing,
                format!(
                    "source `{}`: path does not exist: {}",
                    self.name,
                    self.root.display()
                ),
            )
            .with_next(format!(
                "Restore the path, or run: synk source remove {}",
                self.name
            ))
        })?;

        let mut batch = SyncBatch::default();
        let mut seen = HashSet::new();
        if meta.is_file() {
            self.scan_file(known, &mut batch, &mut seen);
        } else {
            self.scan_folder(known, &mut batch, &mut seen);
        }

        batch.deleted = known
            .keys()
            .filter(|id| !seen.contains(*id))
            .cloned()
            .collect();
        batch.deleted.sort();
        Ok(batch)
    }
}

fn skipped(path: &Path, reason: impl Into<String>) -> Skipped {
    Skipped {
        path: path.display().to_string(),
        reason: reason.into(),
    }
}

fn file_name(path: &Path) -> String {
    path.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default()
}

fn rel_path(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/")
}

#[cfg(test)]
mod tests {
    use std::fs::{self, File};
    use std::time::{Duration, SystemTime};

    use super::*;

    fn connector(root: &Path, exclude: &[&str]) -> FsConnector {
        FsConnector::new(&SourceSpec::folder(
            "notes",
            root.to_path_buf(),
            exclude.iter().map(|s| s.to_string()).collect(),
        ))
        .unwrap()
    }

    fn known_after(batch: &SyncBatch) -> KnownFiles {
        batch
            .upserts
            .iter()
            .map(|d| (d.id.clone(), d.state.clone()))
            .collect()
    }

    fn ids(batch: &SyncBatch) -> Vec<&str> {
        let mut ids: Vec<_> = batch.upserts.iter().map(|d| d.id.as_str()).collect();
        ids.sort();
        ids
    }

    #[test]
    fn reads_supported_files_recursively() {
        let dir = tempfile::tempdir().unwrap();
        fs::create_dir(dir.path().join("sub")).unwrap();
        fs::write(dir.path().join("a.md"), "# Alpha\n\nbody").unwrap();
        fs::write(dir.path().join("sub/b.txt"), "plain").unwrap();
        fs::write(dir.path().join("image.png"), [0u8, 1, 2]).unwrap();

        let batch = connector(dir.path(), &[])
            .scan(&SourceState::default())
            .unwrap();

        assert_eq!(ids(&batch), ["notes/a.md", "notes/sub/b.txt"]);
        let a = batch.upserts.iter().find(|d| d.id == "notes/a.md").unwrap();
        assert_eq!(a.title, "Alpha");
        assert_eq!(a.rel_path, "a.md");
    }

    #[test]
    fn second_scan_reports_only_changes() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("keep.md"), "keep").unwrap();
        fs::write(dir.path().join("edit.md"), "before").unwrap();
        fs::write(dir.path().join("gone.md"), "gone").unwrap();
        let c = connector(dir.path(), &[]);
        let known = known_after(&c.scan(&SourceState::default()).unwrap());

        fs::write(dir.path().join("edit.md"), "after, and longer").unwrap();
        fs::remove_file(dir.path().join("gone.md")).unwrap();
        fs::write(dir.path().join("new.md"), "new").unwrap();
        let batch = c.scan(&known.clone().into()).unwrap();

        assert_eq!(ids(&batch), ["notes/edit.md", "notes/new.md"]);
        assert_eq!(batch.deleted, ["notes/gone.md"]);
        assert_eq!(batch.unchanged, 1);
    }

    #[test]
    fn touched_file_with_same_contents_is_not_reparsed() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a.md");
        fs::write(&path, "same").unwrap();
        let c = connector(dir.path(), &[]);
        let known = known_after(&c.scan(&SourceState::default()).unwrap());

        let later = SystemTime::now() + Duration::from_secs(60);
        File::options()
            .write(true)
            .open(&path)
            .unwrap()
            .set_modified(later)
            .unwrap();
        let batch = c.scan(&known.clone().into()).unwrap();

        assert!(batch.upserts.is_empty());
        assert_eq!(batch.touched.len(), 1);
    }

    #[test]
    fn respects_exclude_globs_gitignore_and_hidden_files() {
        let dir = tempfile::tempdir().unwrap();
        fs::create_dir_all(dir.path().join("archive")).unwrap();
        fs::create_dir_all(dir.path().join(".obsidian")).unwrap();
        fs::write(dir.path().join("archive/old.md"), "old").unwrap();
        fs::write(dir.path().join(".obsidian/cache.md"), "hidden").unwrap();
        fs::write(dir.path().join("draft.md"), "ignored").unwrap();
        fs::write(dir.path().join(".gitignore"), "draft.md\n").unwrap();
        fs::write(dir.path().join("kept.md"), "kept").unwrap();

        let batch = connector(dir.path(), &["archive/**"])
            .scan(&SourceState::default())
            .unwrap();

        assert_eq!(ids(&batch), ["notes/kept.md"]);
    }

    #[test]
    fn single_file_source_uses_source_name_as_id() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("handbook.txt");
        fs::write(&path, "rules").unwrap();

        let batch = connector(&path, &[]).scan(&SourceState::default()).unwrap();

        assert_eq!(ids(&batch), ["notes"]);
        assert_eq!(batch.upserts[0].title, "handbook");
    }

    #[test]
    fn missing_source_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let err = connector(&dir.path().join("nope"), &[])
            .scan(&SourceState::default())
            .unwrap_err();
        assert_eq!(err.code, ErrorCode::SourceMissing);
    }

    #[test]
    fn unreadable_files_are_skipped_not_deleted() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("scan.pdf"), b"%PDF-1.4 broken").unwrap();
        fs::write(dir.path().join("ok.md"), "fine").unwrap();
        let batch = connector(dir.path(), &[])
            .scan(&SourceState::default())
            .unwrap();
        assert_eq!(ids(&batch), ["notes/ok.md"]);
        assert_eq!(batch.skipped.len(), 1);
        assert_eq!(batch.skipped[0].reason, readers::reason::UNREADABLE);
        assert!(batch.deleted.is_empty());
    }
}
