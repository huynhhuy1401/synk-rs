//! Read-only views of finished training runs: listing them, inspecting one
//! query by query, and comparing two.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use serde_json::{Value, json};
use synk_core::{ErrorCode, Result, SynkError};

use crate::artifact::{Artifacts, Manifest};
use crate::dataset;
use crate::ids::{self, Kind};
use crate::train::{self, EVAL_KIND, PER_QUERY_FILE, QueryRanks, TRAIN_KIND};

/// Longest text shown for a chunk.
const SNIPPET_CHARS: usize = 240;

/// A finished run by ID (see [`ids`]); the latest when `id` is `None`.
pub fn find_run(artifacts: &Artifacts, id: Option<&str>) -> Result<Manifest> {
    if artifacts.list(TRAIN_KIND)?.is_empty() {
        return Err(
            SynkError::new(ErrorCode::RunNotFound, "no finished training run yet")
                .with_next("synk train --dry-run"),
        );
    }
    ids::resolve(artifacts, Kind::Run, id.unwrap_or("latest"))
}

/// The readable ID of a run.
pub fn run_id(artifacts: &Artifacts, run: &Manifest) -> Result<String> {
    ids::id(artifacts, Kind::Run, run)
}

fn dataset_key(run: &Manifest) -> Option<&str> {
    run.inputs.iter().find_map(|i| i.strip_prefix("dataset:"))
}

/// The readable ID of the dataset a run was trained on.
pub fn dataset_id(artifacts: &Artifacts, run: &Manifest) -> Result<Option<String>> {
    let Some(key) = dataset_key(run) else {
        return Ok(None);
    };
    match artifacts.manifest(dataset::KIND, key)? {
        Some(m) => ids::id(artifacts, Kind::Dataset, &m).map(Some),
        None => Ok(Some(key.to_owned())),
    }
}

/// Every finished run, newest first, with its settings and verdict.
pub fn runs(artifacts: &Artifacts) -> Result<Vec<Value>> {
    let latest = artifacts.latest(TRAIN_KIND)?.map(|m| m.key);
    let all = artifacts.list(TRAIN_KIND)?;
    all.iter()
        .map(|run| {
            let verdict = train::eval_for(artifacts, run)?.map(|e| train::eval_verdict(&e));
            let p = &run.params;
            Ok(json!({
                "run": ids::id_among(Kind::Run, run, &all),
                "key": run.key,
                "latest": latest.as_deref() == Some(run.key.as_str()),
                "created_at": run.created_at,
                "base_model": run.summary["base_model"],
                "dataset": dataset_id(artifacts, run)?,
                "settings": {
                    "epochs": p["epochs"],
                    "batch_size": p["batch_size"],
                    "learning_rate": p["learning_rate"],
                    "negatives": p["negatives"],
                    "query_filter": p["query_filter"],
                    "reranker": p["reranker"]["catalog_id"].as_str()
                        .or(p["reranker"]["name_or_path"].as_str()),
                },
                "examples": run.summary["examples"],
                "steps": run.summary["steps"],
                "verdict": verdict.map(|v| json!({
                    "outcome": v.outcome,
                    "query_kind": v.query_kind,
                    "base": v.base,
                    "tuned": v.tuned,
                    "delta": v.delta,
                    "ci_low": v.ci_low,
                    "ci_high": v.ci_high,
                    "test_queries": v.test_queries,
                    "test_documents": v.test_documents,
                    "min_detectable_delta": v.min_detectable_delta,
                })),
            }))
        })
        .collect()
}

/// Per-query ranks of a run's evaluation.
fn per_query(artifacts: &Artifacts, run: &Manifest) -> Result<(Manifest, Vec<QueryRanks>)> {
    let missing = || {
        SynkError::new(
            ErrorCode::RunNotFound,
            format!(
                "run {} has no evaluation with per-query ranks (it was evaluated by an older synk)",
                run.key
            ),
        )
        .with_next("synk train --force train")
    };
    let eval = train::eval_for(artifacts, run)?.ok_or_else(missing)?;
    let path = artifacts.dir(EVAL_KIND, &eval.key).join(PER_QUERY_FILE);
    if !path.is_file() {
        return Err(missing());
    }
    let rows = train::read_per_query(&path)?;
    Ok((eval, rows))
}

fn ndcg(rank: Option<&u64>) -> f64 {
    match rank {
        Some(r) if *r <= 10 => 1.0 / ((*r + 1) as f64).log2(),
        _ => 0.0,
    }
}

fn snippet(text: &str) -> String {
    let text = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if text.chars().count() <= SNIPPET_CHARS {
        return text;
    }
    let cut: String = text.chars().take(SNIPPET_CHARS).collect();
    format!("{cut}…")
}

/// Chunk ID to (document, text), from a dataset's corpus.
fn corpus(dir: &Path) -> Result<BTreeMap<String, (String, String)>> {
    let path = dir.join("corpus.jsonl");
    let text = fs::read_to_string(&path).map_err(|e| SynkError::io(path.display(), e))?;
    Ok(text
        .lines()
        .filter_map(|l| serde_json::from_str::<Value>(l).ok())
        .filter_map(|v| {
            Some((
                v["id"].as_str()?.to_owned(),
                (
                    v["doc"].as_str()?.to_owned(),
                    v["text"].as_str()?.to_owned(),
                ),
            ))
        })
        .collect())
}

/// The queries whose rank changed most between the base and tuned model,
/// with the chunk that should have come first and the one that did.
pub fn inspect(artifacts: &Artifacts, run: &Manifest, worst: usize) -> Result<Value> {
    let (eval, rows) = per_query(artifacts, run)?;
    let chunks = match dataset_key(run) {
        Some(key) => corpus(&artifacts.dir(dataset::KIND, key)).unwrap_or_default(),
        None => BTreeMap::new(),
    };
    let chunk = |id: &str| {
        chunks
            .get(id)
            .map(|(doc, text)| json!({"chunk": id, "doc": doc, "text": snippet(text)}))
    };
    let mut changes: Vec<(f64, &QueryRanks)> = rows
        .iter()
        .map(|r| (ndcg(r.ranks.get("tuned")) - ndcg(r.ranks.get("base")), r))
        .collect();
    changes.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.id.cmp(&b.1.id)));
    let item = |(delta, r): &(f64, &QueryRanks)| {
        json!({
            "id": r.id,
            "query": r.query,
            "kind": r.kind,
            "lang": r.lang,
            "ranks": r.ranks,
            "delta": (delta * 10_000.0).round() / 10_000.0,
            "expected": chunk(&r.chunk),
            "tuned_top": r.top.get("tuned").filter(|t| **t != r.chunk).and_then(|t| chunk(t)),
        })
    };
    let losses: Vec<Value> = changes
        .iter()
        .filter(|c| c.0 < 0.0)
        .take(worst)
        .map(item)
        .collect();
    let gains: Vec<Value> = changes
        .iter()
        .rev()
        .filter(|c| c.0 > 0.0)
        .take(worst)
        .map(item)
        .collect();
    let count = |f: &dyn Fn(f64) -> bool| changes.iter().filter(|c| f(c.0)).count();
    Ok(json!({
        "run": run_id(artifacts, run)?,
        "key": run.key,
        "eval": eval.key,
        "queries": rows.len(),
        "improved": count(&|d| d > 0.0),
        "worse": count(&|d| d < 0.0),
        "unchanged": count(&|d| d == 0.0),
        "losses": losses,
        "gains": gains,
    }))
}

/// A paired comparison of two tuned models evaluated on the same dataset.
pub fn compare(artifacts: &Artifacts, a: &Manifest, b: &Manifest) -> Result<Value> {
    let (id_a, id_b) = (run_id(artifacts, a)?, run_id(artifacts, b)?);
    if dataset_key(a) != dataset_key(b) {
        return Err(SynkError::new(
            ErrorCode::RunsNotComparable,
            format!(
                "{id_a} and {id_b} were tested on different datasets ({} and {}), so their scores aren't paired",
                dataset_id(artifacts, a)?.unwrap_or_else(|| "?".to_owned()),
                dataset_id(artifacts, b)?.unwrap_or_else(|| "?".to_owned()),
            ),
        )
        .with_next("synk run list"));
    }
    let (_, rows_a) = per_query(artifacts, a)?;
    let (_, rows_b) = per_query(artifacts, b)?;
    let verdict = train::compare(&rows_a, &rows_b, &id_a, &id_b);
    Ok(json!({
        "a": id_a,
        "b": id_b,
        "dataset": dataset_id(artifacts, a)?,
        "verdict": verdict,
    }))
}
