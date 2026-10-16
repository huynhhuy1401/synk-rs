//! The `sync` step: reading sources into the store.
//!
//! [`preview`] scans without writing, so `synk train --dry-run` can say whether sync
//! has anything to do; [`run`] applies the changes.

use std::collections::HashSet;

use serde::Serialize;
use synk_connectors::FsConnector;
use synk_core::{Connector, ErrorCode, Result, Skipped, SourceSpec, SyncBatch, SynkError};
use synk_store::{Store, SyncStats};

#[derive(Debug, Clone, Serialize)]
pub struct SourceReport {
    pub name: String,
    pub stats: SyncStats,
    pub skipped: Vec<Skipped>,
}

#[derive(Debug, Default, Serialize)]
pub struct Outcome {
    pub sources: Vec<SourceReport>,
    pub totals: SyncStats,
    /// Sources whose path is gone; their documents are kept as they were.
    pub warnings: Vec<SynkError>,
    /// Sources no longer in synk.toml; their documents were marked deleted.
    pub removed_sources: Vec<String>,
    /// True when synk reads files differently than when they were synced
    /// (a new reader version), so every file was read again; their new text
    /// counts as `updated`.
    pub reread: bool,
}

impl Outcome {
    /// Whether anything in the store changes.
    pub fn changed(&self) -> bool {
        self.totals.added + self.totals.updated + self.totals.deleted > 0
            || !self.removed_sources.is_empty()
    }
}

/// Scans every source; a missing source is a warning, not an error.
/// Batches per source, and the sources that are missing.
type Scanned = (Vec<(String, SyncBatch)>, Vec<SynkError>);

/// The store key of the reader version that wrote the documents' text.
const READER_KEY: &str = "reader_version";

/// Whether the stored text came from an older reader. A store without the
/// key and without documents is new, not stale.
fn reader_changed(store: &Store) -> Result<bool> {
    let current = synk_connectors::readers::READER_VERSION.to_string();
    Ok(match store.meta(READER_KEY)? {
        Some(v) => v != current,
        None => !store.counts()?.is_empty(),
    })
}

fn scan(specs: &[SourceSpec], store: &Store) -> Result<Scanned> {
    let mut batches = Vec::new();
    let mut warnings = Vec::new();
    let reread = reader_changed(store)?;
    for spec in specs {
        let connector = FsConnector::new(spec)?;
        let mut known = store.known_files(connector.source())?;
        if reread {
            // Forget sizes and hashes, so every file is read again; the
            // store still knows the documents, so they count as updated.
            for state in known.values_mut() {
                state.size = u64::MAX;
                state.hash.clear();
            }
        }
        match connector.scan(&known.into()) {
            Ok(batch) => batches.push((spec.name.clone(), batch)),
            Err(e) if e.code == ErrorCode::SourceMissing => warnings.push(e),
            Err(e) => return Err(e),
        }
    }
    Ok((batches, warnings))
}

/// Sources in the store that synk.toml no longer lists.
fn removed(specs: &[SourceSpec], store: &Store) -> Result<Vec<String>> {
    let configured: HashSet<&str> = specs.iter().map(|s| s.name.as_str()).collect();
    Ok(store
        .counts()?
        .into_iter()
        .map(|(name, _)| name)
        .filter(|name| !configured.contains(name.as_str()))
        .collect())
}

/// What `run` would change, without changing anything.
pub fn preview(specs: &[SourceSpec], store: &Store) -> Result<Outcome> {
    let reread = reader_changed(store)?;
    let (batches, warnings) = scan(specs, store)?;
    let mut out = Outcome {
        warnings,
        removed_sources: removed(specs, store)?,
        reread,
        ..Outcome::default()
    };
    for (name, batch) in batches {
        let known = store.known_files(&name)?;
        let added = batch
            .upserts
            .iter()
            .filter(|d| !known.contains_key(&d.id))
            .count();
        let stats = SyncStats {
            added,
            updated: batch.upserts.len() - added,
            deleted: batch.deleted.len(),
            unchanged: batch.unchanged + batch.touched.len(),
            skipped: batch.skipped.len(),
        };
        out.totals.merge(stats);
        out.sources.push(SourceReport {
            name,
            stats,
            skipped: batch.skipped,
        });
    }
    Ok(out)
}

/// Applies every source's changes to the store.
pub fn run(specs: &[SourceSpec], store: &mut Store) -> Result<Outcome> {
    let reread = reader_changed(store)?;
    let (batches, warnings) = scan(specs, store)?;
    let mut out = Outcome {
        warnings,
        reread,
        ..Outcome::default()
    };
    for (name, batch) in batches {
        let stats = store.apply(&batch)?;
        out.totals.merge(stats);
        out.sources.push(SourceReport {
            name,
            stats,
            skipped: batch.skipped,
        });
    }
    for name in removed(specs, store)? {
        out.totals.deleted += store.tombstone_source(&name)?;
        out.removed_sources.push(name);
    }
    store.set_meta(
        READER_KEY,
        &synk_connectors::readers::READER_VERSION.to_string(),
    )?;
    Ok(out)
}
