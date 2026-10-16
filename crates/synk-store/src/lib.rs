//! Local SQLite storage for synced documents.
//!
//! Deleted documents are tombstoned rather than removed. The store keeps only
//! the latest text of each document; datasets snapshot the text they use, so
//! past training runs stay reproducible.

use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use rusqlite::{Connection, OptionalExtension, params};
use serde::Serialize;
use synk_core::{Document, ErrorCode, FileState, KnownFiles, Result, SyncBatch, SynkError};

/// Schema migrations, applied in order. `PRAGMA user_version` records how
/// many have run. Never edit a migration that has shipped; append a new one.
const MIGRATIONS: &[&str] = &[
    // 1: documents. `IF NOT EXISTS` because databases created before
    // migrations existed already have this table at user_version 0.
    "
CREATE TABLE IF NOT EXISTS documents (
    id           TEXT PRIMARY KEY,
    source       TEXT NOT NULL,
    rel_path     TEXT NOT NULL,
    title        TEXT NOT NULL,
    text         TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    size         INTEGER NOT NULL,
    modified     INTEGER NOT NULL,
    deleted      INTEGER NOT NULL DEFAULT 0,
    synced_at    INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS documents_source ON documents(source);
",
    // 2: synthetic queries, cached per chunk so an interrupted or repeated
    // dataset build never asks the LLM twice for the same thing.
    "
CREATE TABLE query_cache (
    chunk_hash    TEXT NOT NULL,
    generator_key TEXT NOT NULL,
    queries       TEXT NOT NULL,
    created_at    INTEGER NOT NULL,
    PRIMARY KEY (chunk_hash, generator_key)
);
",
    // 3: real queries a person imported. Each names the document that
    // answers it, and optionally the passage within it.
    "
CREATE TABLE real_queries (
    id          INTEGER PRIMARY KEY,
    query       TEXT NOT NULL,
    doc         TEXT NOT NULL,
    passage     TEXT,
    imported_at INTEGER NOT NULL,
    deleted     INTEGER NOT NULL DEFAULT 0
);
",
    // 4: generated queries one by one, so asking for more per chunk later
    // only generates the missing ones. `generator_key` leaves out the
    // count.
    "
CREATE TABLE query_slots (
    chunk_hash    TEXT NOT NULL,
    generator_key TEXT NOT NULL,
    slot          INTEGER NOT NULL,
    text          TEXT NOT NULL,
    lang          TEXT,
    cross_lang    INTEGER NOT NULL,
    created_at    INTEGER NOT NULL,
    PRIMARY KEY (chunk_hash, generator_key, slot)
);
-- What was already asked for, so an LLM that answers with fewer queries
-- than asked isn't asked again on every build.
CREATE TABLE query_asks (
    chunk_hash    TEXT NOT NULL,
    generator_key TEXT NOT NULL,
    asked         INTEGER NOT NULL,
    cross_langs   TEXT NOT NULL,
    PRIMARY KEY (chunk_hash, generator_key)
);
",
    // 5: the real queries shown to the LLM as style examples, chosen once
    // and kept, so importing more doesn't change every chunk's prompt.
    "
ALTER TABLE real_queries ADD COLUMN style INTEGER NOT NULL DEFAULT 0;
",
];

/// One cached generated query.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct QuerySlot {
    pub text: String,
    pub lang: Option<String>,
    /// Written in another language than its chunk.
    pub cross: bool,
}

/// A query a person wrote, with the document (and optionally the passage)
/// that answers it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RealQuery {
    pub id: i64,
    pub query: String,
    pub doc: String,
    pub passage: Option<String>,
    /// Shown to the LLM as an example of the user's style (and therefore
    /// left out of evaluation).
    pub style: bool,
}

/// What `import_real_queries` did.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ImportStats {
    pub added: usize,
    /// Already imported (same query, document and passage).
    pub unchanged: usize,
    /// Marked deleted by `replace`.
    pub removed: usize,
}

/// Counts of what one sync changed.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct SyncStats {
    pub added: usize,
    pub updated: usize,
    pub deleted: usize,
    pub unchanged: usize,
    pub skipped: usize,
}

impl SyncStats {
    pub fn merge(&mut self, other: SyncStats) {
        self.added += other.added;
        self.updated += other.updated;
        self.deleted += other.deleted;
        self.unchanged += other.unchanged;
        self.skipped += other.skipped;
    }
}

pub struct Store {
    conn: Connection,
}

impl Store {
    pub fn open(path: &Path) -> Result<Self> {
        Self::init(Connection::open(path).map_err(store_err)?)
    }

    pub fn open_in_memory() -> Result<Self> {
        Self::init(Connection::open_in_memory().map_err(store_err)?)
    }

    fn init(mut conn: Connection) -> Result<Self> {
        let applied: usize = conn
            .query_row("PRAGMA user_version", [], |r| r.get::<_, i64>(0))
            .map_err(store_err)? as usize;
        if applied > MIGRATIONS.len() {
            return Err(SynkError::new(
                ErrorCode::Store,
                format!(
                    "the store was created by a newer synk (schema {applied}, this synk knows {})",
                    MIGRATIONS.len()
                ),
            )
            .with_next("Upgrade synk"));
        }
        let tx = conn.transaction().map_err(store_err)?;
        for (i, sql) in MIGRATIONS.iter().enumerate().skip(applied) {
            tx.execute_batch(sql).map_err(store_err)?;
            tx.pragma_update(None, "user_version", (i + 1) as i64)
                .map_err(store_err)?;
        }
        tx.commit().map_err(store_err)?;
        Ok(Self { conn })
    }

    /// Every live document, ordered by ID.
    pub fn live_documents(&self) -> Result<Vec<Document>> {
        let mut stmt = self
            .conn
            .prepare(
                "SELECT id, source, rel_path, title, text, size, modified, content_hash
                 FROM documents WHERE deleted = 0 ORDER BY id",
            )
            .map_err(store_err)?;
        let rows = stmt.query_map([], document_row).map_err(store_err)?;
        rows.collect::<rusqlite::Result<_>>().map_err(store_err)
    }

    /// Cached generator output for a chunk, as stored by `put_queries`.
    pub fn cached_queries(&self, chunk_hash: &str, generator_key: &str) -> Result<Option<String>> {
        self.conn
            .query_row(
                "SELECT queries FROM query_cache WHERE chunk_hash = ?1 AND generator_key = ?2",
                [chunk_hash, generator_key],
                |r| r.get(0),
            )
            .optional()
            .map_err(store_err)
    }

    pub fn put_queries(&self, chunk_hash: &str, generator_key: &str, queries: &str) -> Result<()> {
        self.conn
            .execute(
                "INSERT OR REPLACE INTO query_cache (chunk_hash, generator_key, queries, created_at)
                 VALUES (?1, ?2, ?3, ?4)",
                params![chunk_hash, generator_key, queries, unix_now()],
            )
            .map(drop)
            .map_err(store_err)
    }

    /// Runs `f` and undoes every change it made to the store, for plans
    /// that need to see what the store would look like after a sync.
    pub fn dry_run<T>(&mut self, f: impl FnOnce(&mut Store) -> Result<T>) -> Result<T> {
        self.conn
            .execute_batch("SAVEPOINT synk_dry_run")
            .map_err(store_err)?;
        let result = f(self);
        self.conn
            .execute_batch("ROLLBACK TO synk_dry_run; RELEASE synk_dry_run")
            .map_err(store_err)?;
        result
    }

    /// The cached queries of a chunk, in the order they were generated.
    pub fn query_slots(&self, chunk_hash: &str, generator_key: &str) -> Result<Vec<QuerySlot>> {
        let mut stmt = self
            .conn
            .prepare_cached(
                "SELECT text, lang, cross_lang FROM query_slots
                 WHERE chunk_hash = ?1 AND generator_key = ?2 ORDER BY slot",
            )
            .map_err(store_err)?;
        let rows = stmt
            .query_map([chunk_hash, generator_key], |r| {
                Ok(QuerySlot {
                    text: r.get(0)?,
                    lang: r.get(1)?,
                    cross: r.get::<_, i64>(2)? != 0,
                })
            })
            .map_err(store_err)?;
        rows.collect::<std::result::Result<_, _>>()
            .map_err(store_err)
    }

    /// The most same-language queries ever asked for a chunk, and the cross
    /// languages asked for.
    pub fn query_asked(
        &self,
        chunk_hash: &str,
        generator_key: &str,
    ) -> Result<(usize, Vec<String>)> {
        let row: Option<(i64, String)> = self
            .conn
            .query_row(
                "SELECT asked, cross_langs FROM query_asks WHERE chunk_hash = ?1 AND generator_key = ?2",
                [chunk_hash, generator_key],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .optional()
            .map_err(store_err)?;
        Ok(row.map_or((0, Vec::new()), |(n, langs)| {
            (
                usize::try_from(n).unwrap_or(0),
                langs
                    .split(',')
                    .filter(|l| !l.is_empty())
                    .map(str::to_owned)
                    .collect(),
            )
        }))
    }

    pub fn set_query_asked(
        &self,
        chunk_hash: &str,
        generator_key: &str,
        asked: usize,
        cross_langs: &[String],
    ) -> Result<()> {
        self.conn
            .execute(
                "INSERT OR REPLACE INTO query_asks (chunk_hash, generator_key, asked, cross_langs)
                 VALUES (?1, ?2, ?3, ?4)",
                params![
                    chunk_hash,
                    generator_key,
                    asked as i64,
                    cross_langs.join(",")
                ],
            )
            .map(drop)
            .map_err(store_err)
    }

    /// Appends queries after the chunk's existing ones.
    pub fn add_query_slots(
        &self,
        chunk_hash: &str,
        generator_key: &str,
        queries: &[QuerySlot],
    ) -> Result<()> {
        let next: i64 = self
            .conn
            .query_row(
                "SELECT COALESCE(MAX(slot) + 1, 0) FROM query_slots
                 WHERE chunk_hash = ?1 AND generator_key = ?2",
                [chunk_hash, generator_key],
                |r| r.get(0),
            )
            .map_err(store_err)?;
        for (i, q) in queries.iter().enumerate() {
            self.conn
                .execute(
                    "INSERT INTO query_slots
                     (chunk_hash, generator_key, slot, text, lang, cross_lang, created_at)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                    params![
                        chunk_hash,
                        generator_key,
                        next + i as i64,
                        q.text,
                        q.lang,
                        i64::from(q.cross),
                        unix_now()
                    ],
                )
                .map_err(store_err)?;
        }
        Ok(())
    }

    /// Adds real queries. With `replace`, the ones not in `rows` are marked
    /// deleted first. Identical rows are not added twice.
    pub fn import_real_queries(
        &mut self,
        rows: &[(String, String, Option<String>)],
        replace: bool,
    ) -> Result<ImportStats> {
        let tx = self.conn.savepoint().map_err(store_err)?;
        let mut stats = ImportStats::default();
        if replace {
            let keep: std::collections::HashSet<_> = rows.iter().collect();
            let live = {
                let mut stmt = tx
                    .prepare("SELECT id, query, doc, passage FROM real_queries WHERE deleted = 0")
                    .map_err(store_err)?;
                stmt.query_map([], |r| {
                    Ok((r.get::<_, i64>(0)?, (r.get(1)?, r.get(2)?, r.get(3)?)))
                })
                .map_err(store_err)?
                .collect::<rusqlite::Result<Vec<(i64, (String, String, Option<String>))>>>()
                .map_err(store_err)?
            };
            for (id, row) in live {
                if !keep.contains(&row) {
                    tx.execute("UPDATE real_queries SET deleted = 1 WHERE id = ?1", [id])
                        .map_err(store_err)?;
                    stats.removed += 1;
                }
            }
        }
        for (query, doc, passage) in rows {
            let exists: bool = tx
                .query_row(
                    "SELECT EXISTS(SELECT 1 FROM real_queries WHERE deleted = 0
                     AND query = ?1 AND doc = ?2 AND passage IS ?3)",
                    params![query, doc, passage],
                    |r| r.get(0),
                )
                .map_err(store_err)?;
            if exists {
                stats.unchanged += 1;
                continue;
            }
            tx.execute(
                "INSERT INTO real_queries (query, doc, passage, imported_at) VALUES (?1, ?2, ?3, ?4)",
                params![query, doc, passage, unix_now()],
            )
            .map_err(store_err)?;
            stats.added += 1;
        }
        tx.commit().map_err(store_err)?;
        Ok(stats)
    }

    /// Live real queries, in import order.
    pub fn real_queries(&self) -> Result<Vec<RealQuery>> {
        let mut stmt = self
            .conn
            .prepare(
                "SELECT id, query, doc, passage, style FROM real_queries WHERE deleted = 0 ORDER BY id",
            )
            .map_err(store_err)?;
        let rows = stmt
            .query_map([], |r| {
                Ok(RealQuery {
                    id: r.get(0)?,
                    query: r.get(1)?,
                    doc: r.get(2)?,
                    passage: r.get(3)?,
                    style: r.get::<_, i64>(4)? == 1,
                })
            })
            .map_err(store_err)?;
        rows.collect::<rusqlite::Result<_>>().map_err(store_err)
    }

    /// Makes exactly `ids` the style examples (none clears the pool, so the
    /// next build chooses again).
    pub fn set_style_queries(&self, ids: &[i64]) -> Result<()> {
        // One statement, so the pool changes all at once. IDs are integers.
        let list = ids.iter().map(i64::to_string).collect::<Vec<_>>().join(",");
        self.conn
            .execute(
                &format!("UPDATE real_queries SET style = (id IN ({list}))"),
                [],
            )
            .map_err(store_err)?;
        Ok(())
    }

    /// Live documents of `source`, keyed by ID.
    pub fn known_files(&self, source: &str) -> Result<KnownFiles> {
        let mut stmt = self
            .conn
            .prepare(
                "SELECT id, size, modified, content_hash FROM documents
                 WHERE source = ?1 AND deleted = 0",
            )
            .map_err(store_err)?;
        let rows = stmt
            .query_map([source], |row| {
                let state = FileState {
                    size: row.get::<_, i64>(1)? as u64,
                    modified: row.get(2)?,
                    hash: row.get(3)?,
                };
                Ok((row.get::<_, String>(0)?, state))
            })
            .map_err(store_err)?;
        rows.collect::<rusqlite::Result<_>>().map_err(store_err)
    }

    /// Applies one connector batch in a single transaction.
    pub fn apply(&mut self, batch: &SyncBatch) -> Result<SyncStats> {
        let now = unix_now();
        let mut stats = SyncStats {
            unchanged: batch.unchanged + batch.touched.len(),
            skipped: batch.skipped.len(),
            ..SyncStats::default()
        };
        let tx = self.conn.savepoint().map_err(store_err)?;

        for doc in &batch.upserts {
            let live: bool = tx
                .query_row(
                    "SELECT deleted = 0 FROM documents WHERE id = ?1",
                    [&doc.id],
                    |r| r.get(0),
                )
                .optional()
                .map_err(store_err)?
                .unwrap_or(false);
            tx.execute(
                "INSERT INTO documents
                   (id, source, rel_path, title, text, content_hash, size, modified, deleted, synced_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, 0, ?9)
                 ON CONFLICT(id) DO UPDATE SET
                   source = excluded.source, rel_path = excluded.rel_path,
                   title = excluded.title, text = excluded.text,
                   content_hash = excluded.content_hash, size = excluded.size,
                   modified = excluded.modified, deleted = 0, synced_at = excluded.synced_at",
                params![
                    doc.id,
                    doc.source,
                    doc.rel_path,
                    doc.title,
                    doc.text,
                    doc.state.hash,
                    doc.state.size as i64,
                    doc.state.modified,
                    now
                ],
            )
            .map_err(store_err)?;
            if live {
                stats.updated += 1;
            } else {
                stats.added += 1;
            }
        }

        for (id, state) in &batch.touched {
            tx.execute(
                "UPDATE documents SET size = ?2, modified = ?3 WHERE id = ?1",
                params![id, state.size as i64, state.modified],
            )
            .map_err(store_err)?;
        }

        for id in &batch.deleted {
            stats.deleted += tx
                .execute(
                    "UPDATE documents SET deleted = 1, synced_at = ?2 WHERE id = ?1 AND deleted = 0",
                    params![id, now],
                )
                .map_err(store_err)?;
        }

        tx.commit().map_err(store_err)?;
        Ok(stats)
    }

    /// Tombstones every live document of `source`. Returns how many.
    pub fn tombstone_source(&self, source: &str) -> Result<usize> {
        self.conn
            .execute(
                "UPDATE documents SET deleted = 1, synced_at = ?2 WHERE source = ?1 AND deleted = 0",
                params![source, unix_now()],
            )
            .map_err(store_err)
    }

    /// Live document count per source, sorted by source name.
    pub fn counts(&self) -> Result<Vec<(String, usize)>> {
        let mut stmt = self
            .conn
            .prepare(
                "SELECT source, COUNT(*) FROM documents WHERE deleted = 0
                 GROUP BY source ORDER BY source",
            )
            .map_err(store_err)?;
        let rows = stmt
            .query_map([], |row| Ok((row.get(0)?, row.get::<_, i64>(1)? as usize)))
            .map_err(store_err)?;
        rows.collect::<rusqlite::Result<_>>().map_err(store_err)
    }

    /// A live document by ID.
    pub fn document(&self, id: &str) -> Result<Option<Document>> {
        self.conn
            .query_row(
                "SELECT id, source, rel_path, title, text, size, modified, content_hash
                 FROM documents WHERE id = ?1 AND deleted = 0",
                [id],
                document_row,
            )
            .optional()
            .map_err(store_err)
    }
}

fn document_row(row: &rusqlite::Row) -> rusqlite::Result<Document> {
    Ok(Document {
        id: row.get(0)?,
        source: row.get(1)?,
        rel_path: row.get(2)?,
        title: row.get(3)?,
        text: row.get(4)?,
        state: FileState {
            size: row.get::<_, i64>(5)? as u64,
            modified: row.get(6)?,
            hash: row.get(7)?,
        },
    })
}

fn store_err(e: rusqlite::Error) -> SynkError {
    SynkError::new(ErrorCode::Store, e.to_string())
}

fn unix_now() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs() as i64)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn doc(id: &str, hash: &str) -> Document {
        Document {
            id: id.into(),
            source: "notes".into(),
            rel_path: id.trim_start_matches("notes/").into(),
            title: "t".into(),
            text: "body".into(),
            state: FileState {
                size: 4,
                modified: 1,
                hash: hash.into(),
            },
        }
    }

    fn upserts(docs: Vec<Document>) -> SyncBatch {
        SyncBatch {
            upserts: docs,
            ..SyncBatch::default()
        }
    }

    #[test]
    fn apply_counts_added_then_updated() {
        let mut store = Store::open_in_memory().unwrap();
        let first = store
            .apply(&upserts(vec![
                doc("notes/a.md", "h1"),
                doc("notes/b.md", "h1"),
            ]))
            .unwrap();
        assert_eq!((first.added, first.updated), (2, 0));

        let second = store
            .apply(&upserts(vec![doc("notes/a.md", "h2")]))
            .unwrap();
        assert_eq!((second.added, second.updated), (0, 1));
        assert_eq!(store.known_files("notes").unwrap()["notes/a.md"].hash, "h2");
    }

    #[test]
    fn deleted_documents_are_tombstoned_and_can_return() {
        let mut store = Store::open_in_memory().unwrap();
        store
            .apply(&upserts(vec![doc("notes/a.md", "h1")]))
            .unwrap();

        let stats = store
            .apply(&SyncBatch {
                deleted: vec!["notes/a.md".into()],
                ..SyncBatch::default()
            })
            .unwrap();
        assert_eq!(stats.deleted, 1);
        assert!(store.known_files("notes").unwrap().is_empty());
        assert!(store.document("notes/a.md").unwrap().is_none());

        let back = store
            .apply(&upserts(vec![doc("notes/a.md", "h1")]))
            .unwrap();
        assert_eq!(back.added, 1);
    }

    #[test]
    fn touched_updates_metadata_only() {
        let mut store = Store::open_in_memory().unwrap();
        store
            .apply(&upserts(vec![doc("notes/a.md", "h1")]))
            .unwrap();
        let touched = FileState {
            size: 4,
            modified: 99,
            hash: "h1".into(),
        };
        let stats = store
            .apply(&SyncBatch {
                touched: vec![("notes/a.md".into(), touched)],
                ..SyncBatch::default()
            })
            .unwrap();
        assert_eq!(stats.unchanged, 1);
        assert_eq!(
            store.known_files("notes").unwrap()["notes/a.md"].modified,
            99
        );
    }

    #[test]
    fn migrations_upgrade_a_pre_migration_store() {
        let dir = std::env::temp_dir().join(format!("synk-migrate-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("old.db");
        let _ = std::fs::remove_file(&path);
        {
            // A store written before migrations existed: the documents table
            // at user_version 0.
            let conn = Connection::open(&path).unwrap();
            conn.execute_batch(MIGRATIONS[0]).unwrap();
        }
        let mut store = Store::open(&path).unwrap();
        store
            .apply(&upserts(vec![doc("notes/a.md", "h1")]))
            .unwrap();
        store.put_queries("c1", "g1", "[\"q\"]").unwrap();
        assert_eq!(
            store.cached_queries("c1", "g1").unwrap().as_deref(),
            Some("[\"q\"]")
        );
        drop(store);
        // Opening again is a no-op.
        assert_eq!(
            Store::open(&path).unwrap().live_documents().unwrap().len(),
            1
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn tombstone_source_and_counts() {
        let mut store = Store::open_in_memory().unwrap();
        store
            .apply(&upserts(vec![
                doc("notes/a.md", "h"),
                doc("notes/b.md", "h"),
            ]))
            .unwrap();
        assert_eq!(store.counts().unwrap(), [("notes".to_string(), 2)]);

        assert_eq!(store.tombstone_source("notes").unwrap(), 2);
        assert!(store.counts().unwrap().is_empty());
    }

    #[test]
    fn real_queries_import_dedupes_and_replace_tombstones() {
        let mut store = Store::open_in_memory().unwrap();
        let row = |q: &str| (q.to_owned(), "notes/a.md".to_owned(), None);
        let first = store
            .import_real_queries(&[row("q1"), row("q2")], false)
            .unwrap();
        assert_eq!((first.added, first.unchanged), (2, 0));
        let again = store
            .import_real_queries(&[row("q2"), row("q3")], true)
            .unwrap();
        assert_eq!((again.added, again.unchanged, again.removed), (1, 1, 1));
        let live: Vec<_> = store
            .real_queries()
            .unwrap()
            .into_iter()
            .map(|q| q.query)
            .collect();
        assert_eq!(live, ["q2", "q3"]);
    }
}
