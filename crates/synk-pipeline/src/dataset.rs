//! Building a dataset: chunks plus synthetic queries, split into train and
//! test by document.
//!
//! The dataset artifact holds its own copy of every chunk's text, because the
//! store keeps only the latest text of each document. That copy is what makes
//! a training run reproducible after the documents change.
//!
//! Files (format version 1, one JSON object per line):
//! - `corpus.jsonl`: `{v, id, doc, title, text, split, lang}`
//! - `queries.jsonl`: `{v, id, chunk, text, lang, cross, kind}`, plus
//!   `relevant_chunks` for real queries (`kind: "real"`) that a whole
//!   document answers. Real queries are only for evaluation.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use serde::Serialize;
use serde_json::{Value, json};
use synk_core::{Document, ErrorCode, Result, SynkError};
use synk_store::Store;

use crate::artifact::{self, Artifacts, Manifest};
use crate::chunk::{self, chunk_by};
use crate::config::{AutoOr, DatasetConfig};
use crate::generator::{GeneratedQuery, QueryGenerator};
use crate::lang;
use crate::tokenizer::TokenCounter;

pub const KIND: &str = "dataset";
/// Bump when chunking, splitting or the file format changes.
// 5: `test_percent = "auto"`, and `--sample` keeps every test document.
const KIND_VERSION: u32 = 5;
/// `test_percent = "auto"` holds out at least this many documents...
const AUTO_MIN_TEST_DOCUMENTS: usize = 10;
/// ...unless that would be more than this share of them.
const AUTO_MAX_TEST_PERCENT: usize = 50;
/// The share `auto` starts from, and the old fixed default.
const DEFAULT_TEST_PERCENT: u8 = 20;
pub const FORMAT_VERSION: u32 = 1;
const MIN_CHARS: usize = 100;
const MIN_TOKENS: usize = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Split {
    Train,
    Test,
}

#[derive(Serialize)]
struct CorpusRow<'a> {
    v: u32,
    id: &'a str,
    doc: &'a str,
    title: &'a str,
    text: &'a str,
    split: Split,
    lang: Option<&'static str>,
}

#[derive(Serialize)]
struct QueryRow<'a> {
    v: u32,
    id: String,
    chunk: &'a str,
    text: &'a str,
    lang: Option<&'a str>,
    cross: bool,
    kind: &'static str,
    /// Every chunk that answers the query, when more than `chunk` does.
    #[serde(skip_serializing_if = "Option::is_none")]
    relevant_chunks: Option<&'a [String]>,
}

/// A real query, resolved to the dataset's chunks.
struct RealRow {
    id: i64,
    text: String,
    lang: Option<&'static str>,
    /// The chunks that answer it: the one holding the given passage, or
    /// every chunk of the document.
    chunks: Vec<String>,
}

struct Chunk {
    id: String,
    doc: String,
    title: String,
    /// The headings above the chunk in its document, e.g. `Ops > Backups`.
    heading: String,
    text: String,
    /// Cache key for generated queries: everything the prompt sees.
    hash: String,
    split: Split,
    lang: Option<whatlang::Lang>,
    /// Longer than the model reads; `None` without a tokenizer.
    truncated: Option<bool>,
}

/// How documents are cut into chunks.
pub struct Chunking {
    max: usize,
    unit: &'static str,
    counter: Option<TokenCounter>,
    max_seq_length: usize,
}

impl Chunking {
    /// `counter` is the base model's tokenizer. It is required unless the
    /// deprecated `chunk_chars` is set; then it only counts truncated chunks.
    pub fn new(
        cfg: &DatasetConfig,
        max_seq_length: u32,
        counter: Option<TokenCounter>,
    ) -> Result<Self> {
        let max_seq_length = max_seq_length as usize;
        let (max, unit) = match (cfg.chunk_chars, &counter) {
            (Some(chars), _) => (chars.max(MIN_CHARS), "chars"),
            (None, Some(counter)) => {
                let tokens = match cfg.chunk_tokens {
                    AutoOr::Auto => max_seq_length.saturating_sub(counter.overhead()),
                    AutoOr::Value(n) => n,
                };
                (tokens.max(MIN_TOKENS), "tokens")
            }
            (None, None) => {
                return Err(SynkError::new(
                    ErrorCode::ModelNotFound,
                    "chunking by tokens needs the base model's tokenizer",
                ));
            }
        };
        Ok(Self {
            max,
            unit,
            counter,
            max_seq_length,
        })
    }

    fn split(&self, text: &str) -> Vec<String> {
        match (&self.counter, self.unit) {
            (Some(counter), "tokens") => chunk_by(text, self.max, &|s| counter.count(s)),
            _ => chunk_by(text, self.max, &chunk::chars),
        }
    }

    fn truncated(&self, text: &str) -> Option<bool> {
        self.counter
            .as_ref()
            .map(|c| c.overhead() + c.count(text) > self.max_seq_length)
    }

    /// Everything about chunking that changes the dataset.
    fn describe(&self) -> Value {
        json!({
            "unit": self.unit,
            "max": self.max,
            "max_seq_length": self.max_seq_length,
            "tokenizer": self.counter.as_ref().map(TokenCounter::id),
        })
    }
}

/// Chunks and the dataset key, before any query is generated.
pub struct Prepared {
    chunks: Vec<Chunk>,
    documents: usize,
    key: String,
    params: Value,
    inputs: Vec<String>,
    /// True when `max_chunks` left chunks out.
    sampled: bool,
    real: Vec<RealRow>,
    /// Real queries whose document is gone or not in this dataset.
    real_skipped: usize,
    /// Documents with the same contents as an earlier one, left out.
    duplicate_documents: Vec<String>,
    /// Chunks whose text (ignoring case and spacing) repeats another chunk.
    duplicate_chunks: usize,
    /// Of those, test chunks dropped because the same text is in training.
    leaked_chunks_removed: usize,
    /// Same-language queries per chunk, with `auto` resolved.
    queries_per_chunk: usize,
}

impl Prepared {
    pub fn key(&self) -> &str {
        &self.key
    }

    fn count(&self, split: Split) -> usize {
        self.chunks.iter().filter(|c| c.split == split).count()
    }

    /// Documents with at least one chunk in `split`. The verdict counts
    /// test documents, not test queries.
    fn documents_in(&self, split: Split) -> usize {
        self.chunks
            .iter()
            .filter(|c| c.split == split)
            .map(|c| c.doc.as_str())
            .collect::<HashSet<_>>()
            .len()
    }

    fn truncated(&self) -> Option<usize> {
        self.chunks
            .iter()
            .map(|c| c.truncated.map(usize::from))
            .sum()
    }
}

pub struct Outcome {
    pub manifest: Manifest,
    pub dir: PathBuf,
    /// True when an identical dataset already existed.
    pub reused: bool,
}

/// Progress of query generation: chunks done, chunks total. Returning
/// `false` stops the build with `CANCELLED`; cached queries are kept.
pub type Progress<'a> = &'a mut dyn FnMut(usize, usize) -> bool;

/// Chunks the live documents and works out the dataset's key.
/// Training queries `queries_per_chunk = "auto"` aims for. **[unverified]**
const AUTO_TARGET_QUERIES: usize = 1500;
const AUTO_MIN_PER_CHUNK: usize = 3;
const AUTO_MAX_PER_CHUNK: usize = 8;
/// At most this many real queries become style examples.
const MAX_STYLE_EXAMPLES: usize = 5;

/// Same-language queries per chunk for `setting` and this many training
/// chunks.
pub fn queries_per_chunk(setting: AutoOr<usize>, train_chunks: usize) -> usize {
    match setting {
        AutoOr::Value(n) => n.max(1),
        AutoOr::Auto => AUTO_TARGET_QUERIES
            .div_ceil(train_chunks.max(1))
            .clamp(AUTO_MIN_PER_CHUNK, AUTO_MAX_PER_CHUNK),
    }
}

/// Real queries shown to the generator as examples of the user's style, and
/// therefore left out of evaluation: one in four, at most five, picked by a
/// hash of their text so the choice is stable.
pub fn style_pool(store: &Store) -> Result<Vec<synk_store::RealQuery>> {
    let mut real = store.real_queries()?;
    let n = (real.len() / 4).min(MAX_STYLE_EXAMPLES);
    real.sort_by_key(|q| short_hash(&q.query));
    real.truncate(n);
    Ok(real)
}

/// The headings above `chunk` in `doc_text`, outermost first.
fn heading_path(doc_text: &str, chunk: &str) -> String {
    let probe: String = chunk.chars().take(60).collect();
    let Some(pos) = doc_text.find(probe.trim()) else {
        return String::new();
    };
    let mut stack: Vec<(usize, &str)> = Vec::new();
    for line in doc_text[..pos].lines() {
        let level = line.chars().take_while(|c| *c == '#').count();
        if (1..=6).contains(&level) && line[level..].starts_with(' ') {
            let title = line[level..].trim();
            while stack.last().is_some_and(|(l, _)| *l >= level) {
                stack.pop();
            }
            if !title.is_empty() {
                stack.push((level, title));
            }
        }
    }
    stack
        .iter()
        .map(|(_, t)| *t)
        .collect::<Vec<_>>()
        .join(" > ")
}

pub fn prepare(
    store: &Store,
    cfg: &DatasetConfig,
    queries_setting: AutoOr<usize>,
    chunking: &Chunking,
    generator: &dyn QueryGenerator,
) -> Result<Prepared> {
    // A document copied into two sources would be its own hard negative,
    // and could sit in both splits.
    let mut seen_contents = HashSet::new();
    let mut duplicate_documents = Vec::new();
    let docs: Vec<Document> = store
        .live_documents()?
        .into_iter()
        .filter(|d| {
            let first = seen_contents.insert(d.state.hash.clone());
            if !first {
                duplicate_documents.push(d.id.clone());
            }
            first
        })
        .collect();
    let splits = split_documents(&docs, cfg.test_percent)?;

    let mut candidates = Vec::new();
    for doc in &docs {
        for text in chunking.split(&doc.text) {
            let heading = heading_path(&doc.text, &text);
            candidates.push(Chunk {
                id: short_hash(&format!("{}\0{text}", doc.id)),
                doc: doc.id.clone(),
                title: doc.title.clone(),
                // Everything the prompt sees about the chunk.
                hash: short_hash(&format!("{}\0{heading}\0{text}", doc.title)),
                heading,
                split: splits[&doc.id],
                lang: lang::detect(&text),
                truncated: chunking.truncated(&text),
                text,
            });
        }
    }
    let (mut chunks, duplicate_chunks, leaked_chunks_removed) = dedup_chunks(candidates);
    chunks.sort_by(|a, b| a.id.cmp(&b.id));
    let total = chunks.len();
    if let Some(max) = cfg.max_chunks {
        chunks = sample(chunks, max);
    }

    let documents_digest = {
        let mut hasher = blake3::Hasher::new();
        for doc in &docs {
            hasher.update(doc.id.as_bytes());
            hasher.update(b"\0");
            hasher.update(doc.state.hash.as_bytes());
            hasher.update(b"\n");
        }
        hasher.finalize().to_hex()[..16].to_owned()
    };
    let train_chunks = chunks.iter().filter(|c| c.split == Split::Train).count();
    let per_chunk = queries_per_chunk(queries_setting, train_chunks);
    let params = json!({
        "chunking": chunking.describe(),
        "test_percent": cfg.test_percent,
        "max_chunks": cfg.max_chunks,
        "generator": generator.key(),
        "queries_per_chunk": per_chunk,
    });
    let mut inputs = vec![format!("documents:{documents_digest}")];
    let style: HashSet<i64> = style_pool(store)?.into_iter().map(|q| q.id).collect();
    let (real, real_skipped) = real_rows(store, &chunks, &style)?;
    if !real.is_empty() || real_skipped > 0 {
        let mut hasher = blake3::Hasher::new();
        for r in &real {
            hasher.update(format!("{}\0{}\0{}\n", r.id, r.text, r.chunks.join(",")).as_bytes());
        }
        inputs.push(format!(
            "real_queries:{}",
            &hasher.finalize().to_hex()[..16]
        ));
    }
    let key = artifact::key(KIND, KIND_VERSION, &params, &inputs);
    Ok(Prepared {
        real,
        real_skipped,
        duplicate_documents,
        duplicate_chunks,
        leaked_chunks_removed,
        sampled: chunks.len() < total,
        chunks,
        documents: docs.len(),
        queries_per_chunk: per_chunk,
        key,
        params,
        inputs,
    })
}

/// Keeps one chunk per text, compared ignoring case and spacing. Repeated
/// boilerplate would otherwise be one query with several right answers, and
/// a text in both splits would leak test data into training: the training
/// copy is kept. Returns the kept chunks, how many were dropped, and how
/// many of those were test chunks whose text is in training.
fn dedup_chunks(candidates: Vec<Chunk>) -> (Vec<Chunk>, usize, usize) {
    let normal = |t: &str| {
        t.split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
            .to_lowercase()
    };
    let mut first: HashMap<String, usize> = HashMap::new();
    let mut kept: Vec<Option<Chunk>> = Vec::new();
    let (mut dropped, mut leaked) = (0, 0);
    for chunk in candidates {
        let key = normal(&chunk.text);
        match first.get(&key) {
            None => {
                first.insert(key, kept.len());
                kept.push(Some(chunk));
            }
            Some(&i) => {
                dropped += 1;
                let existing = kept[i].as_ref().map(|c| c.split);
                if existing == Some(Split::Test) && chunk.split == Split::Train {
                    leaked += 1;
                    kept[i] = Some(chunk);
                } else if existing == Some(Split::Train) && chunk.split == Split::Test {
                    leaked += 1;
                }
            }
        }
    }
    (kept.into_iter().flatten().collect(), dropped, leaked)
}

/// Resolves the imported real queries to chunks of this dataset.
fn real_rows(
    store: &Store,
    chunks: &[Chunk],
    style: &HashSet<i64>,
) -> Result<(Vec<RealRow>, usize)> {
    let mut by_doc: BTreeMap<&str, Vec<&Chunk>> = BTreeMap::new();
    for c in chunks {
        by_doc.entry(&c.doc).or_default().push(c);
    }
    let (mut rows, mut skipped) = (Vec::new(), 0);
    for q in store.real_queries()? {
        // Style examples were shown to the generator, so they can't test.
        if style.contains(&q.id) {
            continue;
        }
        let Some(doc_chunks) = by_doc.get(q.doc.as_str()) else {
            skipped += 1;
            continue;
        };
        let holding: Option<&&Chunk> = q.passage.as_deref().and_then(|p| {
            let needle = crate::queries::normalize(p);
            doc_chunks
                .iter()
                .find(|c| crate::queries::normalize(&c.text).contains(&needle))
        });
        let chunks = match holding {
            Some(c) => vec![c.id.clone()],
            // No passage, or it spans chunks: the whole document answers.
            None => doc_chunks.iter().map(|c| c.id.clone()).collect(),
        };
        rows.push(RealRow {
            id: q.id,
            lang: lang::detect(&q.query).map(lang::code),
            text: q.query,
            chunks,
        });
    }
    Ok((rows, skipped))
}

/// What a chunk still needs: same-language queries, and the cross
/// languages without a query. `None` when it has everything.
fn missing(
    have: &[synk_store::QuerySlot],
    asked: &(usize, Vec<String>),
    chunk: &Chunk,
    n: usize,
    languages: &[whatlang::Lang],
) -> Option<(usize, Vec<whatlang::Lang>)> {
    let same = have.iter().filter(|q| !q.cross).count();
    let cross: Vec<whatlang::Lang> = languages
        .iter()
        .copied()
        .filter(|l| Some(*l) != chunk.lang)
        .filter(|l| !asked.1.iter().any(|a| a == lang::code(*l)))
        .filter(|l| {
            !have
                .iter()
                .any(|q| q.cross && q.lang.as_deref() == Some(lang::code(*l)))
        })
        .collect();
    // Asking again for what the LLM already declined to write is wasted.
    let need = if n > asked.0 {
        n.saturating_sub(same)
    } else {
        0
    };
    (need > 0 || !cross.is_empty()).then_some((need, cross))
}

/// The queries of one chunk: cached ones, plus whatever is missing asked
/// for and cached. Returns `None` when the chunk has none at all, and
/// whether the generator was called.
fn queries_for(
    store: &Store,
    generator: &dyn QueryGenerator,
    generator_key: &str,
    chunk: &Chunk,
    n: usize,
) -> Result<(Option<Vec<GeneratedQuery>>, bool)> {
    let mut have = store.query_slots(&chunk.hash, generator_key)?;
    let before = store.query_asked(&chunk.hash, generator_key)?;
    let mut asked = false;
    if let Some((need, cross)) = missing(&have, &before, chunk, n, generator.languages()) {
        asked = true;
        let avoid: Vec<String> = have.iter().map(|q| q.text.clone()).collect();
        let ask = crate::generator::Ask {
            title: &chunk.title,
            heading: &chunk.heading,
            chunk: &chunk.text,
            chunk_lang: chunk.lang,
            n: need,
            cross: &cross,
            avoid: &avoid,
        };
        if let Some(new) = generator.generate(&ask)? {
            let slots: Vec<synk_store::QuerySlot> = new
                .into_iter()
                .map(|q| synk_store::QuerySlot {
                    text: q.text,
                    lang: q.lang,
                    cross: q.cross,
                })
                .collect();
            store.add_query_slots(&chunk.hash, generator_key, &slots)?;
            have.extend(slots);
            // Only after an answer: a failed call is retried next time.
            let mut langs = before.1.clone();
            langs.extend(cross.iter().map(|l| lang::code(*l).to_owned()));
            store.set_query_asked(&chunk.hash, generator_key, n.max(before.0), &langs)?;
        }
    }
    if have.is_empty() {
        return Ok((None, asked));
    }
    // The first `n` same-language queries, and one per cross language.
    let mut out = Vec::new();
    let mut same = 0;
    let mut cross_seen = HashSet::new();
    for q in have {
        let keep = if q.cross {
            cross_seen.insert(q.lang.clone())
        } else {
            same += 1;
            same <= n
        };
        if keep {
            out.push(GeneratedQuery {
                text: q.text,
                lang: q.lang,
                cross: q.cross,
            });
        }
    }
    Ok((Some(out), asked))
}

/// Takes `max` chunks. The verdict's power comes from the number of test
/// documents, so every test document keeps at least one chunk (while there
/// is room), and the test's share of the sample is spread evenly over
/// them; training chunks fill the rest. Chunk IDs are hashes, so taking the
/// first chunks is a stable random sample.
fn sample(chunks: Vec<Chunk>, max: usize) -> Vec<Chunk> {
    if chunks.len() <= max {
        return chunks;
    }
    let (test, train): (Vec<Chunk>, Vec<Chunk>) =
        chunks.into_iter().partition(|c| c.split == Split::Test);
    let total = test.len() + train.len();
    let mut by_doc: BTreeMap<&str, Vec<&Chunk>> = BTreeMap::new();
    for c in &test {
        by_doc.entry(c.doc.as_str()).or_default().push(c);
    }
    // The test's proportional share, but at least one chunk per test
    // document, and never more than half the sample: training needs
    // examples too.
    let share = (max * test.len() + total / 2) / total;
    let budget = share
        .max(by_doc.len())
        .min(test.len())
        .min((max / 2).max(1));
    let mut keep: HashSet<&str> = HashSet::new();
    let mut taken = 0;
    for round in 0.. {
        let mut added = false;
        for chunks in by_doc.values() {
            if taken == budget {
                break;
            }
            if let Some(c) = chunks.get(round) {
                keep.insert(c.id.as_str());
                taken += 1;
                added = true;
            }
        }
        if taken == budget || !added {
            break;
        }
    }
    let keep: HashSet<String> = keep.into_iter().map(str::to_owned).collect();
    let train_n = (max - taken).min(train.len());
    let mut out: Vec<Chunk> = test.into_iter().filter(|c| keep.contains(&c.id)).collect();
    out.extend(train.into_iter().take(train_n));
    out.sort_by(|a, b| a.id.cmp(&b.id));
    out
}

/// What `build` would do, without writing the dataset. Up to `samples`
/// uncached chunks are sent to the generator to time it; their queries are
/// cached, so the work is not lost.
pub fn plan(
    store: &Store,
    artifacts: &Artifacts,
    prepared: &Prepared,
    generator: &dyn QueryGenerator,
    samples: usize,
) -> Result<Value> {
    let existing = artifacts.manifest(KIND, &prepared.key)?.is_some();
    let generator_key = generator.key();
    let n = prepared.queries_per_chunk;
    let mut uncached = Vec::new();
    for chunk in &prepared.chunks {
        let have = store.query_slots(&chunk.hash, &generator_key)?;
        let asked = store.query_asked(&chunk.hash, &generator_key)?;
        if missing(&have, &asked, chunk, n, generator.languages()).is_some() {
            uncached.push(chunk);
        }
    }
    let mut timings = Vec::new();
    if !existing {
        for chunk in uncached.iter().take(samples) {
            let started = Instant::now();
            queries_for(store, generator, &generator_key, chunk, n)?;
            timings.push(started.elapsed().as_secs_f64());
        }
    }
    let llm_calls = if existing {
        0
    } else {
        uncached.len() - timings.len()
    };
    let per_call =
        (!timings.is_empty()).then(|| timings.iter().sum::<f64>() / timings.len() as f64);
    let train_chunks = prepared.count(Split::Train);
    Ok(json!({
        "dataset": prepared.key,
        "existing": existing,
        "documents": prepared.documents,
        "chunks": prepared.chunks.len(),
        "train_chunks": train_chunks,
        "test_chunks": prepared.count(Split::Test),
        "train_documents": prepared.documents_in(Split::Train),
        "test_documents": prepared.documents_in(Split::Test),
        "truncated_chunks": prepared.truncated(),
        "cached_chunks": prepared.chunks.len() - uncached.len() + timings.len(),
        "llm_calls": llm_calls,
        "seconds_per_call": per_call.map(|s| (s * 100.0).round() / 100.0),
        "estimated_seconds": per_call.map(|s| (s * llm_calls as f64).round()),
        "train_queries": train_chunks * n,
        "queries_per_chunk": n,
        "real_queries": prepared.real.len(),
        "real_queries_skipped": prepared.real_skipped,
        "chunking": prepared.params["chunking"],
    }))
}

pub fn build(
    store: &Store,
    artifacts: &Artifacts,
    prepared: Prepared,
    generator: &dyn QueryGenerator,
    force: bool,
    progress: Progress,
) -> Result<Outcome> {
    let Prepared {
        chunks,
        documents,
        key,
        params,
        inputs,
        sampled,
        real,
        real_skipped,
        duplicate_documents,
        duplicate_chunks,
        leaked_chunks_removed,
        queries_per_chunk,
    } = prepared;
    if force {
        artifacts.remove(KIND, &key)?;
    } else if let Some(manifest) = artifacts.manifest(KIND, &key)? {
        artifacts.set_latest(KIND, &key)?;
        return Ok(Outcome {
            dir: artifacts.dir(KIND, &key),
            manifest,
            reused: true,
        });
    }

    // Queries are cached per chunk in the store, so an interrupted build
    // picks up where it stopped.
    let generator_key = generator.key();
    let mut generated: Vec<(usize, Vec<GeneratedQuery>)> = Vec::new();
    let (mut failed, mut cached) = (0, 0);
    for (i, chunk) in chunks.iter().enumerate() {
        if !progress(i, chunks.len()) {
            return Err(SynkError::new(ErrorCode::Cancelled, "cancelled"));
        }
        let (queries, asked) =
            queries_for(store, generator, &generator_key, chunk, queries_per_chunk)?;
        if !asked {
            cached += 1;
        }
        match queries {
            Some(qs) => generated.push((i, qs)),
            None => failed += 1,
        }
    }
    generator.release();
    progress(chunks.len(), chunks.len());

    // The same query for two chunks would have two right answers. Keep the
    // first; chunks are in a stable order.
    let mut seen = HashSet::new();
    let mut duplicates = 0;
    let mut rows: Vec<(usize, usize, GeneratedQuery)> = Vec::new();
    for (i, qs) in generated {
        for (n, q) in qs.into_iter().enumerate() {
            if seen.insert(q.text.to_lowercase()) {
                rows.push((i, n, q));
            } else {
                duplicates += 1;
            }
        }
    }

    let count = |split: Split| {
        rows.iter()
            .filter(|(i, ..)| chunks[*i].split == split)
            .count()
    };
    let (train_queries, test_queries) = (count(Split::Train), count(Split::Test));
    if train_queries < 2 || test_queries == 0 {
        let next = if sampled {
            "Raise --sample (or dataset.max_chunks), or remove it, then run synk train again"
        } else {
            "Add more documents with synk source add, then run synk train again"
        };
        return Err(SynkError::new(
            ErrorCode::DatasetTooSmall,
            format!("got {train_queries} training and {test_queries} test queries from {} chunks; training needs at least 2 and evaluation at least 1", chunks.len()),
        )
        .with_hint(next));
    }

    let tmp = artifacts.begin(KIND, &key, true)?;
    write_corpus(&tmp, &chunks)?;
    let by_lang = write_queries(&tmp, &chunks, &rows, &real)?;

    let truncated: Option<usize> = chunks.iter().map(|c| c.truncated.map(usize::from)).sum();
    let summary = json!({
        "documents": documents,
        "chunks": chunks.len(),
        "train_chunks": chunks.iter().filter(|c| c.split == Split::Train).count(),
        "test_chunks": chunks.iter().filter(|c| c.split == Split::Test).count(),
        "queries": train_queries + test_queries,
        "train_queries": train_queries,
        "test_queries": test_queries,
        "queries_by_lang": by_lang,
        "failed_chunks": failed,
        "cached_chunks": cached,
        "duplicate_queries": duplicates,
        "real_queries": real.len(),
        "real_queries_skipped": real_skipped,
        "queries_per_chunk": queries_per_chunk,
        "duplicate_documents": duplicate_documents,
        "duplicate_chunks": duplicate_chunks,
        "leaked_chunks_removed": leaked_chunks_removed,
        "truncated_chunks": truncated,
        "chunking": params["chunking"],
        "generator": generator.describe(),
    });
    let manifest = Manifest::new(KIND, &key, params, inputs, summary);
    let dir = artifacts.commit(&manifest)?;
    artifacts.set_latest(KIND, &key)?;
    Ok(Outcome {
        manifest,
        dir,
        reused: false,
    })
}

/// Assigns whole documents to train or test, so no test chunk's document
/// was seen in training. The split hashes the document's path within its
/// source rather than its ID, which embeds the source's name: renaming a
/// source must not reshuffle the split and leak test data into training.
fn split_documents(docs: &[Document], test_percent: AutoOr<u8>) -> Result<BTreeMap<String, Split>> {
    if docs.len() < 2 {
        return Err(SynkError::new(
            ErrorCode::DatasetTooSmall,
            format!("{} document(s) synced; a dataset needs at least 2, one of them held out for evaluation", docs.len()),
        )
        .with_hint("add documents: synk source add <path>, then synk train"));
    }
    let bucket = |doc: &Document| {
        let hash = blake3::hash(doc.rel_path.as_bytes());
        u64::from_le_bytes(hash.as_bytes()[..8].try_into().expect("8 bytes")) % 10_000
    };
    let percent = test_percent.value().unwrap_or(DEFAULT_TEST_PERCENT);
    let threshold = u64::from(percent.min(100)) * 100;
    let mut splits: BTreeMap<String, Split> = docs
        .iter()
        .map(|d| {
            let split = if bucket(d) < threshold {
                Split::Test
            } else {
                Split::Train
            };
            (d.id.clone(), split)
        })
        .collect();
    // `auto` on a small project: top up the test with the next documents in
    // the same hash order, so it reaches AUTO_MIN_TEST_DOCUMENTS (at most
    // half). With enough documents this is exactly the 20% split.
    if test_percent.is_auto() {
        let tests = splits.values().filter(|s| **s == Split::Test).count();
        let want = AUTO_MIN_TEST_DOCUMENTS
            .min(docs.len() * AUTO_MAX_TEST_PERCENT / 100)
            .max(tests);
        let mut order: Vec<&Document> = docs.iter().collect();
        order.sort_by(|a, b| bucket(a).cmp(&bucket(b)).then(a.rel_path.cmp(&b.rel_path)));
        for d in order.into_iter().take(want) {
            splits.insert(d.id.clone(), Split::Test);
        }
    }
    // Small projects can land entirely on one side; move the document
    // closest to the threshold so both splits exist.
    let tests = splits.values().filter(|s| **s == Split::Test).count();
    if tests == 0 || tests == docs.len() {
        let want = if tests == 0 {
            Split::Test
        } else {
            Split::Train
        };
        let doc = docs
            .iter()
            .min_by_key(|d| bucket(d).abs_diff(threshold))
            .expect("at least 2 documents");
        splits.insert(doc.id.clone(), want);
    }
    Ok(splits)
}

fn write_corpus(dir: &Path, chunks: &[Chunk]) -> Result<()> {
    write_lines(
        dir,
        "corpus.jsonl",
        chunks.iter().map(|c| {
            serde_json::to_string(&CorpusRow {
                v: FORMAT_VERSION,
                id: &c.id,
                doc: &c.doc,
                title: &c.title,
                text: &c.text,
                split: c.split,
                lang: c.lang.map(lang::code),
            })
        }),
    )
}

/// Writes the queries and returns how many there are per language.
fn write_queries(
    dir: &Path,
    chunks: &[Chunk],
    rows: &[(usize, usize, GeneratedQuery)],
    real: &[RealRow],
) -> Result<BTreeMap<String, usize>> {
    let mut by_lang = BTreeMap::new();
    let mut lines = Vec::new();
    for (i, n, q) in rows {
        let chunk = &chunks[*i];
        *by_lang
            .entry(q.lang.clone().unwrap_or_else(|| "unknown".to_owned()))
            .or_insert(0) += 1;
        lines.push(serde_json::to_string(&QueryRow {
            v: FORMAT_VERSION,
            id: format!("{}-{n}", chunk.id),
            chunk: &chunk.id,
            text: &q.text,
            lang: q.lang.as_deref(),
            cross: q.cross,
            kind: "synthetic",
            relevant_chunks: None,
        }));
    }
    for r in real {
        lines.push(serde_json::to_string(&QueryRow {
            v: FORMAT_VERSION,
            id: format!("real-{}", r.id),
            chunk: &r.chunks[0],
            text: &r.text,
            lang: r.lang,
            cross: false,
            kind: "real",
            relevant_chunks: (r.chunks.len() > 1).then_some(r.chunks.as_slice()),
        }));
    }
    write_lines(dir, "queries.jsonl", lines.into_iter())?;
    Ok(by_lang)
}

fn write_lines(
    dir: &Path,
    name: &str,
    lines: impl Iterator<Item = serde_json::Result<String>>,
) -> Result<()> {
    let path = dir.join(name);
    let io = |e| SynkError::io(path.display(), e);
    let mut out = BufWriter::new(fs::File::create(&path).map_err(io)?);
    for line in lines {
        let line = line.expect("rows serialize");
        out.write_all(line.as_bytes()).map_err(io)?;
        out.write_all(b"\n").map_err(io)?;
    }
    out.flush().map_err(io)
}

fn short_hash(s: &str) -> String {
    blake3::hash(s.as_bytes()).to_hex()[..16].to_owned()
}

/// The summary of a dataset, as shown by `synk status`.
pub fn summary(manifest: &Manifest) -> Value {
    json!({ "key": manifest.key, "created_at": manifest.created_at, "summary": manifest.summary })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    use synk_core::{FileState, SyncBatch};

    struct FakeGenerator {
        calls: Cell<usize>,
    }

    impl QueryGenerator for FakeGenerator {
        fn key(&self) -> String {
            "fake".into()
        }
        fn languages(&self) -> &[whatlang::Lang] {
            &[]
        }
        fn generate(&self, ask: &crate::generator::Ask<'_>) -> Result<Option<Vec<GeneratedQuery>>> {
            self.calls.set(self.calls.get() + 1);
            let first_word = ask.chunk.split_whitespace().next().unwrap_or("x");
            Ok(Some(vec![GeneratedQuery {
                text: format!("about {first_word}"),
                lang: Some("en".into()),
                cross: false,
            }]))
        }
        fn describe(&self) -> String {
            "fake".into()
        }
    }

    fn chunking() -> Chunking {
        let cfg = DatasetConfig::default();
        let counter =
            TokenCounter::from_bytes(crate::tokenizer::word_level_json().as_bytes(), "").unwrap();
        Chunking::new(&cfg, 256, Some(counter)).unwrap()
    }

    fn run(
        store: &Store,
        artifacts: &Artifacts,
        cfg: &DatasetConfig,
        generator: &dyn QueryGenerator,
        force: bool,
    ) -> Result<Outcome> {
        let prepared = prepare(store, cfg, AutoOr::Value(2), &chunking(), generator)?;
        build(store, artifacts, prepared, generator, force, &mut |_, _| {
            true
        })
    }

    fn store_with(n: usize) -> Store {
        let mut store = Store::open_in_memory().unwrap();
        let upserts = (0..n)
            .map(|i| Document {
                id: format!("notes/doc{i}.md"),
                source: "notes".into(),
                rel_path: format!("doc{i}.md"),
                title: format!("doc{i}"),
                text: format!("topic{i} is described here in enough words to be a chunk."),
                state: FileState {
                    size: 1,
                    modified: 1,
                    hash: format!("h{i}"),
                },
            })
            .collect();
        store
            .apply(&SyncBatch {
                upserts,
                ..SyncBatch::default()
            })
            .unwrap();
        store
    }

    #[test]
    fn builds_once_and_reuses_identical_inputs() {
        let store = store_with(6);
        let dir = tempfile::tempdir().unwrap();
        let artifacts = Artifacts::new(dir.path().join("artifacts"));
        let generator = FakeGenerator {
            calls: Cell::new(0),
        };
        let cfg = DatasetConfig::default();

        let first = run(&store, &artifacts, &cfg, &generator, false).unwrap();
        assert!(!first.reused);
        assert_eq!(first.manifest.summary["chunks"], 6);
        assert_eq!(first.manifest.summary["queries"], 6);
        assert!(first.manifest.summary["test_queries"].as_u64().unwrap() >= 1);
        let corpus = fs::read_to_string(first.dir.join("corpus.jsonl")).unwrap();
        assert!(
            corpus.contains("\"text\":\"topic0 is described"),
            "the dataset keeps its own copy of the text"
        );

        let second = run(&store, &artifacts, &cfg, &generator, false).unwrap();
        assert!(second.reused);
        assert_eq!(second.manifest.key, first.manifest.key);

        // --force rebuilds, but the per-chunk cache means no new LLM calls.
        let forced = run(&store, &artifacts, &cfg, &generator, true).unwrap();
        assert!(!forced.reused);
        assert_eq!(forced.manifest.summary["cached_chunks"], 6);
        assert_eq!(generator.calls.get(), 6);
    }

    #[test]
    fn needs_two_documents() {
        let store = store_with(1);
        let dir = tempfile::tempdir().unwrap();
        let generator = FakeGenerator {
            calls: Cell::new(0),
        };
        let err = run(
            &store,
            &Artifacts::new(dir.path().join("artifacts")),
            &DatasetConfig::default(),
            &generator,
            false,
        )
        .err()
        .unwrap();
        assert_eq!(err.code, ErrorCode::DatasetTooSmall);
        assert_eq!(generator.calls.get(), 0);
    }

    struct SameQuery;

    impl QueryGenerator for SameQuery {
        fn key(&self) -> String {
            "same".into()
        }
        fn languages(&self) -> &[whatlang::Lang] {
            &[]
        }
        fn generate(&self, ask: &crate::generator::Ask<'_>) -> Result<Option<Vec<GeneratedQuery>>> {
            let chunk = ask.chunk;
            let q = |text: &str| GeneratedQuery {
                text: text.into(),
                lang: None,
                cross: false,
            };
            let first = chunk.split_whitespace().next().unwrap_or("x");
            Ok(Some(vec![q("Shared Query"), q(&format!("about {first}"))]))
        }
        fn describe(&self) -> String {
            "same".into()
        }
    }

    #[test]
    fn duplicate_queries_across_chunks_are_dropped() {
        let store = store_with(6);
        let dir = tempfile::tempdir().unwrap();
        let artifacts = Artifacts::new(dir.path().join("artifacts"));
        let out = run(
            &store,
            &artifacts,
            &DatasetConfig::default(),
            &SameQuery,
            false,
        )
        .unwrap();
        assert_eq!(out.manifest.summary["queries"], 7);
        assert_eq!(out.manifest.summary["duplicate_queries"], 5);
        assert_eq!(out.manifest.summary["truncated_chunks"], 0);
        assert_eq!(out.manifest.summary["chunking"]["unit"], "tokens");
    }

    #[test]
    fn a_limit_keeps_both_splits_and_plan_counts_llm_calls() {
        let store = store_with(30);
        let dir = tempfile::tempdir().unwrap();
        let artifacts = Artifacts::new(dir.path().join("artifacts"));
        let generator = FakeGenerator {
            calls: Cell::new(0),
        };
        let cfg = DatasetConfig {
            max_chunks: Some(3),
            ..DatasetConfig::default()
        };
        let prepared = prepare(&store, &cfg, AutoOr::Value(1), &chunking(), &generator).unwrap();
        assert_eq!(prepared.count(Split::Train), 2);
        assert_eq!(prepared.count(Split::Test), 1);

        let plan = plan(&store, &artifacts, &prepared, &generator, 1).unwrap();
        assert_eq!(plan["chunks"], 3);
        assert_eq!(plan["llm_calls"], 2);
        assert_eq!(plan["cached_chunks"], 1);
        assert_eq!(plan["existing"], false);
        assert!(plan["seconds_per_call"].is_number());
        assert_eq!(generator.calls.get(), 1);

        // The timed sample was cached, so building calls the LLM twice more.
        let out = build(
            &store,
            &artifacts,
            prepared,
            &generator,
            false,
            &mut |_, _| true,
        )
        .unwrap();
        assert_eq!(out.manifest.summary["cached_chunks"], 1);
        assert_eq!(generator.calls.get(), 3);
    }

    #[test]
    fn chunks_by_tokens_and_counts_truncation_in_chars_mode() {
        let counter = || {
            TokenCounter::from_bytes(crate::tokenizer::word_level_json().as_bytes(), "").unwrap()
        };
        let auto = Chunking::new(&DatasetConfig::default(), 10, Some(counter())).unwrap();
        // 10 tokens minus [CLS] and [SEP], raised to the minimum.
        assert_eq!(auto.max, MIN_TOKENS);
        let chars = DatasetConfig {
            chunk_chars: Some(100),
            ..DatasetConfig::default()
        };
        let by_chars = Chunking::new(&chars, 10, Some(counter())).unwrap();
        let text = "word ".repeat(19);
        assert_eq!(by_chars.split(&text).len(), 1);
        assert_eq!(by_chars.truncated(&text), Some(true));
        assert!(
            Chunking::new(&chars, 10, None)
                .unwrap()
                .truncated(&text)
                .is_none()
        );
        assert_eq!(
            Chunking::new(&DatasetConfig::default(), 10, None)
                .err()
                .unwrap()
                .code,
            ErrorCode::ModelNotFound
        );
    }

    #[test]
    fn real_queries_are_written_for_evaluation() {
        let mut store = store_with(6);
        store
            .import_real_queries(
                &[
                    ("whole doc".into(), "notes/doc1.md".into(), None),
                    ("gone".into(), "notes/missing.md".into(), None),
                ],
                false,
            )
            .unwrap();
        let dir = tempfile::tempdir().unwrap();
        let artifacts = Artifacts::new(dir.path().join("artifacts"));
        let generator = FakeGenerator {
            calls: Cell::new(0),
        };
        let out = run(
            &store,
            &artifacts,
            &DatasetConfig::default(),
            &generator,
            false,
        )
        .unwrap();
        assert_eq!(out.manifest.summary["real_queries"], 1);
        assert_eq!(out.manifest.summary["real_queries_skipped"], 1);
        assert_eq!(
            out.manifest.summary["queries"], 6,
            "real queries aren't counted as training data"
        );
        let queries = fs::read_to_string(out.dir.join("queries.jsonl")).unwrap();
        let real: Value = serde_json::from_str(queries.lines().last().unwrap()).unwrap();
        assert_eq!(
            (&real["kind"], &real["text"]),
            (&json!("real"), &json!("whole doc"))
        );
        assert!(
            out.manifest
                .inputs
                .iter()
                .any(|i| i.starts_with("real_queries:"))
        );
    }

    #[test]
    fn split_always_has_both_sides_and_ignores_source_names() {
        let doc = |source: &str, rel: &str| Document {
            id: format!("{source}/{rel}"),
            source: source.into(),
            rel_path: rel.into(),
            title: String::new(),
            text: String::new(),
            state: FileState {
                size: 0,
                modified: 0,
                hash: String::new(),
            },
        };
        for percent in [0, 100] {
            let splits = split_documents(
                &[doc("a", "x.md"), doc("a", "y.md")],
                AutoOr::Value(percent),
            )
            .unwrap();
            assert!(splits.values().any(|s| *s == Split::Test));
            assert!(splits.values().any(|s| *s == Split::Train));
        }
        let docs_a: Vec<_> = (0..20).map(|i| doc("a", &format!("{i}.md"))).collect();
        let docs_b: Vec<_> = (0..20).map(|i| doc("b", &format!("{i}.md"))).collect();
        let a: Vec<_> = split_documents(&docs_a, AutoOr::Value(20))
            .unwrap()
            .into_values()
            .collect();
        let b: Vec<_> = split_documents(&docs_b, AutoOr::Value(20))
            .unwrap()
            .into_values()
            .collect();
        assert_eq!(a, b);
    }

    fn docs(n: usize) -> Vec<Document> {
        (0..n)
            .map(|i| Document {
                id: format!("s/{i}.md"),
                source: "s".into(),
                rel_path: format!("{i}.md"),
                title: String::new(),
                text: String::new(),
                state: FileState {
                    size: 0,
                    modified: 0,
                    hash: String::new(),
                },
            })
            .collect()
    }

    fn tests_in(splits: &BTreeMap<String, Split>) -> Vec<String> {
        splits
            .iter()
            .filter(|(_, s)| **s == Split::Test)
            .map(|(d, _)| d.clone())
            .collect()
    }

    #[test]
    fn auto_holds_out_at_least_ten_documents_but_at_most_half() {
        let auto = |n| tests_in(&split_documents(&docs(n), AutoOr::Auto).unwrap()).len();
        assert_eq!(auto(2), 1);
        assert_eq!(auto(11), 5);
        assert_eq!(auto(20), 10);
        assert_eq!(auto(30), 10);
        // With enough documents, auto is exactly the 20% split.
        for n in [80, 200] {
            let auto = split_documents(&docs(n), AutoOr::Auto).unwrap();
            let fixed = split_documents(&docs(n), AutoOr::Value(20)).unwrap();
            assert!(tests_in(&fixed).len() >= AUTO_MIN_TEST_DOCUMENTS);
            assert_eq!(auto, fixed);
        }
        // A number is respected.
        assert!(tests_in(&split_documents(&docs(11), AutoOr::Value(20)).unwrap()).len() < 5);
    }

    #[test]
    fn auto_tops_up_in_hash_order() {
        // Every document `auto` tests at 30 documents is still a test
        // document with 31, unless the extra one ranks ahead of it.
        let small = tests_in(&split_documents(&docs(30), AutoOr::Auto).unwrap());
        let big = tests_in(&split_documents(&docs(31), AutoOr::Auto).unwrap());
        let moved = small.iter().filter(|d| !big.contains(d)).count();
        assert!(moved <= 1, "{moved} documents moved");
    }

    #[test]
    fn sampling_keeps_every_test_document() {
        let chunk = |doc: usize, i: usize, split: Split| Chunk {
            id: format!("{doc:03}-{i:03}"),
            doc: format!("d{doc}"),
            title: String::new(),
            heading: String::new(),
            text: String::new(),
            hash: String::new(),
            split,
            lang: None,
            truncated: None,
        };
        let mut chunks = Vec::new();
        // 8 test documents with 10 chunks each, 40 training documents.
        for d in 0..8 {
            chunks.extend((0..10).map(|i| chunk(d, i, Split::Test)));
        }
        for d in 8..48 {
            chunks.extend((0..10).map(|i| chunk(d, i, Split::Train)));
        }
        let sampled = sample(chunks, 30);
        assert_eq!(sampled.len(), 30);
        let test_docs: HashSet<&str> = sampled
            .iter()
            .filter(|c| c.split == Split::Test)
            .map(|c| c.doc.as_str())
            .collect();
        assert_eq!(test_docs.len(), 8, "every test document keeps a chunk");
        assert!(sampled.iter().any(|c| c.split == Split::Train));
    }

    #[test]
    fn duplicate_chunks_keep_the_training_copy() {
        let chunk = |doc: &str, text: &str, split: Split| Chunk {
            id: doc.to_owned(),
            doc: doc.to_owned(),
            title: String::new(),
            heading: String::new(),
            text: text.to_owned(),
            hash: String::new(),
            split,
            lang: None,
            truncated: None,
        };
        let (kept, dropped, leaked) = dedup_chunks(vec![
            chunk("t1", "Backups  run nightly", Split::Test),
            chunk("a1", "backups run nightly", Split::Train),
            chunk("a2", "Footer: (c) ACME", Split::Train),
            chunk("a3", "footer: (c) acme", Split::Train),
            chunk("t2", "Only in test", Split::Test),
        ]);
        let ids: Vec<&str> = kept.iter().map(|c| c.id.as_str()).collect();
        assert_eq!(ids, ["a1", "a2", "t2"]);
        assert_eq!((dropped, leaked), (2, 1));
    }

    #[test]
    fn headings_and_auto_query_counts() {
        let doc = "# Ops

intro

## Backups

Run pg_dump nightly.

## Nginx

Raise the timeout.";
        assert_eq!(heading_path(doc, "Run pg_dump nightly."), "Ops > Backups");
        assert_eq!(heading_path(doc, "Raise the timeout."), "Ops > Nginx");
        assert_eq!(heading_path(doc, "intro"), "Ops");
        assert_eq!(queries_per_chunk(AutoOr::Auto, 1000), 3);
        assert_eq!(queries_per_chunk(AutoOr::Auto, 250), 6);
        assert_eq!(queries_per_chunk(AutoOr::Auto, 10), 8);
        assert_eq!(queries_per_chunk(AutoOr::Value(5), 10), 5);
    }

    #[test]
    fn asking_for_more_queries_only_generates_the_missing_ones() {
        let store = store_with(4);
        let artifacts_dir = tempfile::tempdir().unwrap();
        let artifacts = Artifacts::new(artifacts_dir.path().to_path_buf());
        let cfg = DatasetConfig::default();
        struct Counting(Cell<usize>, Cell<usize>);
        impl QueryGenerator for Counting {
            fn key(&self) -> String {
                "counting".into()
            }
            fn languages(&self) -> &[whatlang::Lang] {
                &[]
            }
            fn generate(
                &self,
                ask: &crate::generator::Ask<'_>,
            ) -> Result<Option<Vec<GeneratedQuery>>> {
                self.0.set(self.0.get() + 1);
                self.1.set(self.1.get() + ask.n);
                Ok(Some(
                    (0..ask.n)
                        .map(|i| GeneratedQuery {
                            text: format!(
                                "{} q{}",
                                ask.chunk.split_whitespace().next().unwrap_or("x"),
                                ask.avoid.len() + i
                            ),
                            lang: None,
                            cross: false,
                        })
                        .collect(),
                ))
            }
            fn describe(&self) -> String {
                "counting".into()
            }
        }
        let generator = Counting(Cell::new(0), Cell::new(0));
        let prepared = prepare(&store, &cfg, AutoOr::Value(2), &chunking(), &generator).unwrap();
        let chunks = prepared.chunks.len();
        build(
            &store,
            &artifacts,
            prepared,
            &generator,
            false,
            &mut |_, _| true,
        )
        .unwrap();
        assert_eq!(generator.1.get(), 2 * chunks);
        let prepared = prepare(&store, &cfg, AutoOr::Value(3), &chunking(), &generator).unwrap();
        build(
            &store,
            &artifacts,
            prepared,
            &generator,
            false,
            &mut |_, _| true,
        )
        .unwrap();
        // Only one more query per chunk was asked for.
        assert_eq!(generator.1.get(), 3 * chunks);
        assert_eq!(generator.0.get(), 2 * chunks);
    }
}
