//! A stand-in for the Python trainer, for tests: `SYNK_FAKE_TRAINER=1`.
//!
//! It speaks the same request/result contract with deterministic answers,
//! so the train/fetch/wait flow can be tested without Python or a GPU. The
//! tuned model always ranks the right chunk first and the base model third,
//! so the verdict is `better` once there are enough test documents.
//! `SYNK_FAKE_TRAINER_SLOW=<ms>` waits that long per training step, to test
//! cancelling.

use std::fs;
use std::path::Path;
use std::time::Duration;

use serde_json::{Value, json};
use synk_core::{ErrorCode, Result, SynkError};

pub const ENV: &str = "SYNK_FAKE_TRAINER";

pub fn enabled() -> bool {
    std::env::var(ENV).is_ok_and(|v| v == "1")
}

pub fn describe() -> Value {
    json!({
        "protocol": [crate::sidecar::PROTOCOL_VERSION],
        "tasks": ["eval", "fetch", "train"],
        "extras": [],
        "torch": "fake",
        "transformers": "fake",
        "sentence_transformers": "fake",
        "devices": ["cpu"],
        "gpu": null,
        "fake": true,
    })
}

fn stopped(request: &Value) -> bool {
    request["stop_file"]
        .as_str()
        .is_some_and(|p| Path::new(p).exists())
}

fn cancelled() -> SynkError {
    SynkError::new(ErrorCode::Cancelled, "cancelled")
}

fn lines(path: &Path) -> Vec<Value> {
    fs::read_to_string(path)
        .unwrap_or_default()
        .lines()
        .filter_map(|l| serde_json::from_str(l).ok())
        .collect()
}

pub fn run(request: &Value, on_event: &mut dyn FnMut(&Value)) -> Result<Value> {
    let io = |e: std::io::Error| SynkError::io("fake trainer", e);
    match request["task"].as_str() {
        Some("fetch") => Ok(json!({"path": null, "revision": request["revision"]})),
        Some("train") => {
            let dataset = Path::new(request["dataset_dir"].as_str().unwrap_or_default());
            let out = Path::new(request["output_dir"].as_str().unwrap_or_default());
            let corpus = lines(&dataset.join("corpus.jsonl"));
            let train_chunks: Vec<&str> = corpus
                .iter()
                .filter(|c| c["split"] == "train")
                .filter_map(|c| c["id"].as_str())
                .collect();
            let examples = lines(&dataset.join("queries.jsonl"))
                .iter()
                .filter(|q| q["kind"] != "real")
                .filter(|q| {
                    q["chunk"]
                        .as_str()
                        .is_some_and(|c| train_chunks.contains(&c))
                })
                .count();
            let steps = examples.div_ceil(16).max(1) as u64;
            let slow = std::env::var("SYNK_FAKE_TRAINER_SLOW")
                .ok()
                .and_then(|v| v.parse::<u64>().ok());
            for step in 1..=steps {
                if let Some(ms) = slow {
                    // Sleep in slices so a cancel is noticed quickly.
                    for _ in 0..ms / 50 {
                        if stopped(request) {
                            return Err(cancelled());
                        }
                        std::thread::sleep(Duration::from_millis(50));
                    }
                }
                if stopped(request) {
                    return Err(cancelled());
                }
                on_event(
                    &json!({"event": "progress", "stage": "train", "step": step, "total": steps, "unit": "steps"}),
                );
            }
            let model = out.join("model");
            fs::create_dir_all(&model).map_err(io)?;
            fs::write(model.join("fake-model.txt"), "fake").map_err(io)?;
            Ok(json!({
                "examples": examples,
                "train_queries": examples,
                "filtered_queries": 0,
                "dev_queries": 0,
                "negatives_per_query": 3,
                "mined_negatives": examples,
                "unmined_queries": 0,
                "steps": steps,
                "epochs": 1,
                "best_epoch": 1,
                "finetune": request["params"]["finetune"],
                "prompts": {"query": "query: ", "document": "passage: "},
            }))
        }
        Some("eval") => {
            if stopped(request) {
                return Err(cancelled());
            }
            let dataset = Path::new(request["dataset_dir"].as_str().unwrap_or_default());
            let out = Path::new(request["output_dir"].as_str().unwrap_or_default());
            let corpus = lines(&dataset.join("corpus.jsonl"));
            let doc_of = |chunk: &str| {
                corpus
                    .iter()
                    .find(|c| c["id"] == chunk)
                    .and_then(|c| c["doc"].as_str())
                    .unwrap_or_default()
                    .to_owned()
            };
            let test: Vec<&str> = corpus
                .iter()
                .filter(|c| c["split"] == "test")
                .filter_map(|c| c["id"].as_str())
                .collect();
            let mut rows = Vec::new();
            for q in lines(&dataset.join("queries.jsonl")) {
                let chunk = q["chunk"].as_str().unwrap_or_default();
                if q["kind"] != "real" && !test.contains(&chunk) {
                    continue;
                }
                rows.push(json!({
                    "id": q["id"], "query": q["text"], "kind": q["kind"].as_str().unwrap_or("synthetic"),
                    "chunk": chunk, "doc": doc_of(chunk), "lang": q["lang"],
                    "ranks": {"base": 3, "tuned": 1},
                    "top": {"base": corpus.first().map_or(Value::Null, |c| c["id"].clone()), "tuned": chunk},
                }));
            }
            let text: String = rows.iter().map(|r| format!("{r}\n")).collect();
            fs::write(out.join(crate::train::PER_QUERY_FILE), text).map_err(io)?;
            let m = |ndcg: f64| json!({"overall": {"queries": rows.len(), "ndcg@10": ndcg, "recall@1": ndcg, "recall@5": 1.0, "recall@10": 1.0, "mrr@10": ndcg}, "by_lang": {}});
            Ok(
                json!({"device": "cpu", "metrics": {"base": m(0.5), "tuned": m(1.0)}, "real_metrics": null}),
            )
        }
        Some("general") => {
            let out = Path::new(request["output_dir"].as_str().unwrap_or_default());
            let rows: String = (0..20)
                .map(|i| {
                    format!(
                        "{}
",
                        json!({"id": format!("en:{i}"), "query": "q", "kind": "general", "chunk": "", "doc": format!("en:{i}"), "lang": "en", "ranks": {"base": 2, "tuned": 2}, "top": {}})
                    )
                })
                .collect();
            fs::write(out.join("general.jsonl"), rows).map_err(io)?;
            Ok(json!({"sets": [{"name": "en", "queries": 20, "corpus": 100}]}))
        }
        other => Err(SynkError::new(
            ErrorCode::SidecarProtocol,
            format!("the fake trainer has no task {other:?}"),
        )),
    }
}
