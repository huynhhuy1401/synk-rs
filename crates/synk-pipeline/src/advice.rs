//! Advice: what an agent (or a person) could change to get a better result.
//!
//! Every entry has a stable `code`, like an error code, and says how safe it
//! is to act on without looking (`applicability`, modelled on rustc's
//! suggestions). Advice never blocks a command; it rides along in its output.

use serde::Serialize;
use serde_json::{Map, Value, json};

use crate::config::PipelineConfig;
use crate::train::Verdict;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Severity {
    Info,
    Warning,
}

pub use synk_core::Applicability;

/// Every advice code with what it means, for `synk schema` and docs.
pub const CODES: &[(&str, &str)] = &[
    (
        "CHUNKS_TRUNCATED",
        "Many chunks are longer than the model reads",
    ),
    (
        "CHUNK_CHARS_DEPRECATED",
        "dataset.chunk_chars is deprecated; chunks are measured in tokens",
    ),
    ("FEW_QUERIES", "Too few training queries for a fair chance"),
    (
        "FEW_TRAINING_STEPS",
        "Too few training steps for a reliable gain",
    ),
    ("LANG_REGRESSION", "One language got worse"),
    (
        "TEST_SET_UNDERPOWERED",
        "The test is too small to see small changes",
    ),
    (
        "TEST_SET_TOO_EASY",
        "The base model already scores near 1.0",
    ),
    (
        "QUERIES_FILTERED_HEAVILY",
        "The query filter dropped many generated queries",
    ),
    (
        "MANY_UNMINED_NEGATIVES",
        "Many queries had no safe hard negative",
    ),
    (
        "GENERAL_REGRESSION",
        "The tuned model got worse at ordinary search (general check)",
    ),
    (
        "PREFIX_DROPPED_BY_TARGET",
        "The export target doesn't add the model's query/document prefixes",
    ),
    (
        "REAL_QUERIES_FEW_DOCUMENTS",
        "The real searches cover too few documents to decide the verdict",
    ),
    (
        "SYNTHETIC_REAL_DISAGREE",
        "Generated test queries and real searches reach different verdicts",
    ),
];

#[derive(Debug, Clone, Serialize)]
pub struct Action {
    /// The command to run, after any config change.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub command: Option<String>,
    /// `synk.toml` changes as dotted paths; `null` means remove the setting.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub config: Option<Map<String, Value>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Advice {
    pub code: &'static str,
    pub severity: Severity,
    pub message: String,
    pub evidence: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub action: Option<Action>,
    pub applicability: Applicability,
}

const MIN_TRAIN_QUERIES: u64 = 300;
const MIN_STEPS: u64 = 100;
const MAX_CI_HALF_WIDTH: f64 = 0.05;
/// Above this base nDCG@10 there is little room to improve, or to measure it.
const EASY_BASE_SCORE: f64 = 0.95;
const MAX_FILTERED_SHARE: f64 = 0.4;
const MAX_UNMINED_SHARE: f64 = 0.2;
const MAX_TRUNCATED_SHARE: f64 = 0.05;
const MAX_QUERIES_PER_CHUNK: u64 = 8;
const ADD_DOCUMENTS: &str = "synk source add <path> && synk train --dry-run";
const WRITE_SEARCHES: &str = "synk queries template --out searches.jsonl";
/// Spread of per-document changes assumed before a run has measured one.
pub const DEFAULT_DOCUMENT_SD: f64 = 0.10;

/// Before training: warns when the test will be too small for a verdict.
/// `sd` is the spread of per-document changes from the project's last run,
/// if any.
pub fn predicted_power(test_documents: usize, sd: Option<f64>) -> Option<Advice> {
    let assumed = sd.unwrap_or(DEFAULT_DOCUMENT_SD);
    let md = crate::stats::min_detectable(test_documents, assumed);
    if md.is_some_and(|md| md <= crate::train::TYPICAL_GAIN) {
        return None;
    }
    let see = md.map_or_else(
        || "it can't tell any difference".to_owned(),
        |md| format!("it could only see a change of about {md:.2} or more"),
    );
    Some(Advice {
        code: "TEST_SET_UNDERPOWERED",
        severity: Severity::Warning,
        message: format!(
            "Only {test_documents} document(s) will be held out for the test, so {see}, and tuning usually brings about {:.2}. The verdict will probably be inconclusive even if tuning helps. Write real searches (synk queries template), add documents, or raise dataset.test_percent.",
            crate::train::TYPICAL_GAIN
        ),
        evidence: json!({
            "predicted": true,
            "test_documents": test_documents,
            "min_detectable_delta": md.map(|m| (m * 1000.0).round() / 1000.0),
            "assumed_sd": assumed,
            "sd_from": if sd.is_some() { "last run" } else { "default" },
        }),
        action: command(WRITE_SEARCHES).map(|c| Action {
            command: Some(c),
            config: None,
        }),
        applicability: Applicability::Manual,
    })
}

fn config(pairs: &[(&str, Value)]) -> Option<Map<String, Value>> {
    Some(
        pairs
            .iter()
            .map(|(k, v)| ((*k).to_owned(), v.clone()))
            .collect(),
    )
}

fn command(c: &str) -> Option<String> {
    Some(c.to_owned())
}

fn share(part: u64, whole: u64) -> f64 {
    if whole == 0 {
        0.0
    } else {
        part as f64 / whole as f64
    }
}

/// Advice about a dataset, from its summary (or its plan, which has the same
/// fields) and the current settings.
pub fn dataset(summary: &Value, cfg: &PipelineConfig) -> Vec<Advice> {
    let mut out = Vec::new();
    let chunks = summary["chunks"].as_u64().unwrap_or(0);
    if let Some(truncated) = summary["truncated_chunks"].as_u64()
        && share(truncated, chunks) > MAX_TRUNCATED_SHARE
    {
        out.push(Advice {
                code: "CHUNKS_TRUNCATED",
                severity: Severity::Warning,
                message: format!(
                    "{truncated} of {chunks} chunks are longer than the model reads ({} tokens), so their ends are invisible to training and evaluation.",
                    cfg.trainer.max_seq_length
                ),
                evidence: json!({"truncated_chunks": truncated, "chunks": chunks, "max_seq_length": cfg.trainer.max_seq_length}),
                action: Some(Action {
                    command: command("synk train --dry-run"),
                    config: config(&[("dataset.chunk_chars", Value::Null), ("dataset.chunk_tokens", json!("auto"))]),
                }),
                applicability: Applicability::Machine,
            });
    }
    if let Some(chars) = cfg.dataset.chunk_chars {
        out.push(Advice {
            code: "CHUNK_CHARS_DEPRECATED",
            severity: Severity::Info,
            message: "[dataset] chunk_chars is deprecated: chunks are now measured in the model's tokens (chunk_tokens = \"auto\").".to_owned(),
            evidence: json!({"chunk_chars": chars}),
            action: Some(Action {
                command: command("synk train --dry-run"),
                config: config(&[("dataset.chunk_chars", Value::Null)]),
            }),
            applicability: Applicability::Machine,
        });
    }
    let train_queries = summary["train_queries"].as_u64().unwrap_or(0);
    if train_queries < MIN_TRAIN_QUERIES {
        let per_chunk = summary["queries_per_chunk"]
            .as_u64()
            .or(cfg.generator.queries_per_chunk.value().map(|n| n as u64))
            .unwrap_or(3);
        let raise = (per_chunk + 2).min(MAX_QUERIES_PER_CHUNK);
        let (action, applicability, how) = if per_chunk < MAX_QUERIES_PER_CHUNK {
            (
                Action {
                    command: command("synk train --dry-run"),
                    config: config(&[("generator.queries_per_chunk", json!(raise))]),
                },
                Applicability::Machine,
                format!("raise queries_per_chunk to {raise}, or add documents"),
            )
        } else {
            (
                Action {
                    command: command(ADD_DOCUMENTS),
                    config: None,
                },
                Applicability::Manual,
                "add documents".to_owned(),
            )
        };
        out.push(Advice {
            code: "FEW_QUERIES",
            severity: Severity::Info,
            message: format!(
                "{train_queries} training queries; at least {MIN_TRAIN_QUERIES} give fine-tuning a fair chance. To get more, {how}."
            ),
            evidence: json!({"train_queries": train_queries, "queries_per_chunk": per_chunk}),
            action: Some(action),
            applicability,
        });
    }
    out
}

/// Advice about a finished training run and its verdict.
/// A general-check loss at least this large (and statistically clear) is
/// worth a warning. **[unverified threshold]**
const GENERAL_REGRESSION_DELTA: f64 = 0.02;

/// The general check's verdicts from an evaluation summary.
pub fn general_verdicts(eval: &Value) -> Vec<(String, Verdict)> {
    eval["general_check"]["verdicts"]
        .as_object()
        .into_iter()
        .flatten()
        .filter_map(|(k, v)| Some((k.clone(), serde_json::from_value(v.clone()).ok()?)))
        .collect()
}

pub fn train(
    train: &Value,
    verdict: &Verdict,
    general: &[(String, Verdict)],
    cfg: &PipelineConfig,
) -> Vec<Advice> {
    let mut out = Vec::new();
    for (lang, v) in &verdict.by_lang {
        if v.outcome == "worse" {
            out.push(Advice {
                code: "LANG_REGRESSION",
                severity: Severity::Warning,
                message: format!(
                    "Queries in `{lang}` got worse (search quality {:+.2}), even if the overall result did not. Check whether that language has enough documents.",
                    v.delta
                ),
                evidence: json!({"lang": lang, "delta": v.delta, "ci_low": v.ci_low, "ci_high": v.ci_high, "test_queries": v.test_queries}),
                action: None,
                applicability: Applicability::Review,
            });
        }
    }
    out.extend(steps(train, cfg));
    for (set, v) in general.iter() {
        if v.outcome == "worse" && v.delta <= -GENERAL_REGRESSION_DELTA {
            out.push(Advice {
                code: "GENERAL_REGRESSION",
                severity: Severity::Warning,
                message: format!(
                    "The tuned model got worse at ordinary {} search (general check: {:.2} -> {:.2}). It may have over-fitted your documents; a gain on them can still be worth it, but look at both before using it.",
                    if set == "vi" { "Vietnamese" } else { "English" },
                    v.base,
                    v.tuned
                ),
                evidence: json!({"set": set, "base": v.base, "tuned": v.tuned, "delta": v.delta, "ci_low": v.ci_low, "ci_high": v.ci_high}),
                action: None,
                applicability: Applicability::Review,
            });
        }
    }
    if let (Some(real), Some(_)) = (
        verdict.by_kind.get("real"),
        verdict.by_kind.get("synthetic"),
    ) && verdict.query_kind != "real"
    {
        let covered = real.test_documents;
        out.push(Advice {
            code: "REAL_QUERIES_FEW_DOCUMENTS",
            severity: Severity::Info,
            message: format!(
                "Your {} real searches are about {covered} document(s); they decide the verdict once there are {} or more about {}+ documents, because searches about one document rise and fall together. Write a few about other documents.",
                real.test_queries,
                crate::train::MIN_REAL_QUERIES,
                crate::train::MIN_REAL_DOCUMENTS,
            ),
            evidence: json!({"real_queries": real.test_queries, "documents": covered, "min_queries": crate::train::MIN_REAL_QUERIES, "min_documents": crate::train::MIN_REAL_DOCUMENTS}),
            action: command(WRITE_SEARCHES).map(|c| Action {
                command: Some(c),
                config: None,
            }),
            applicability: Applicability::Manual,
        });
    }
    if verdict.agreement.as_deref() == Some("disagree") {
        let kind = |k: &str| verdict.by_kind.get(k).map_or("?", |v| v.outcome.as_str());
        out.push(Advice {
            code: "SYNTHETIC_REAL_DISAGREE",
            severity: Severity::Warning,
            message: format!(
                "Generated test queries say {} but your real searches say {}. Real searches are how you'll use the model, so trust them; the generated test may reward the LLM's style of question.",
                kind("synthetic"),
                kind("real")
            ),
            evidence: json!({"synthetic": kind("synthetic"), "real": kind("real"), "decided_by": verdict.query_kind}),
            action: None,
            applicability: Applicability::Review,
        });
    }
    let half_width = match (verdict.ci_low, verdict.ci_high) {
        (Some(lo), Some(hi)) => Some((hi - lo) / 2.0),
        _ => None,
    };
    if half_width.is_none_or(|w| w > MAX_CI_HALF_WIDTH) {
        out.push(Advice {
            code: "TEST_SET_UNDERPOWERED",
            severity: Severity::Warning,
            message: format!(
                "The test has only {} documents, too few to see changes in search quality under {}; the verdict can only see large differences. Add documents, or write real searches (synk queries template).",
                verdict.test_documents,
                verdict.min_detectable_delta.map_or("any size".to_owned(), |d| format!("{d:.2}")),
            ),
            evidence: json!({"test_documents": verdict.test_documents, "test_queries": verdict.test_queries, "ci_half_width": half_width, "min_detectable_delta": verdict.min_detectable_delta}),
            action: Some(Action {
                command: command(ADD_DOCUMENTS),
                config: None,
            }),
            applicability: Applicability::Manual,
        });
    }
    if verdict.base >= EASY_BASE_SCORE {
        out.push(Advice {
            code: "TEST_SET_TOO_EASY",
            severity: Severity::Warning,
            message: format!(
                "The base model already scores {:.2} search quality (out of 1) on the test documents, so there is little room to improve or to measure it. More documents (especially similar ones that are easy to confuse) make a harder test.",
                verdict.base
            ),
            evidence: json!({"base": verdict.base, "test_queries": verdict.test_queries}),
            action: Some(Action {
                command: command(ADD_DOCUMENTS),
                config: None,
            }),
            applicability: Applicability::Manual,
        });
    }
    let before = train["train_queries"].as_u64().unwrap_or(0);
    let filtered = train["filtered_queries"].as_u64().unwrap_or(0);
    if share(filtered, before) > MAX_FILTERED_SHARE {
        out.push(Advice {
            code: "QUERIES_FILTERED_HEAVILY",
            severity: Severity::Warning,
            message: format!(
                "The query filter dropped {filtered} of {before} training queries because the base model could not find their chunk; many generated queries may be off-topic. Try a stronger generator model."
            ),
            evidence: json!({"filtered_queries": filtered, "train_queries": before, "generator_model": cfg.generator.model}),
            action: None,
            applicability: Applicability::Review,
        });
    }
    let examples = train["examples"].as_u64().unwrap_or(0);
    let unmined = train["unmined_queries"].as_u64().unwrap_or(0);
    if share(unmined, examples) > MAX_UNMINED_SHARE {
        out.push(Advice {
            code: "MANY_UNMINED_NEGATIVES",
            severity: Severity::Info,
            message: format!(
                "{unmined} of {examples} training queries had no safe hard negative, so they trained without one. More documents give more hard negatives."
            ),
            evidence: json!({"unmined_queries": unmined, "examples": examples}),
            action: Some(Action {
                command: command(ADD_DOCUMENTS),
                config: None,
            }),
            applicability: Applicability::Manual,
        });
    }
    out
}

/// `FEW_TRAINING_STEPS`, from a run's summary or a training plan.
pub fn steps(train: &Value, cfg: &PipelineConfig) -> Option<Advice> {
    let steps = train["steps"].as_u64().unwrap_or(0);
    if steps < MIN_STEPS {
        let evidence = json!({"steps": steps, "examples": train["examples"], "epochs": train["epochs"], "batch_size": cfg.trainer.batch_size});
        if cfg.trainer.epochs.is_auto() {
            Some(Advice {
                code: "FEW_TRAINING_STEPS",
                severity: Severity::Warning,
                message: format!(
                    "The run has {steps} training steps, even at the most epochs synk picks; at least {MIN_STEPS} are needed for a reliable gain. Add documents or queries."
                ),
                evidence,
                action: Some(Action {
                    command: command(ADD_DOCUMENTS),
                    config: None,
                }),
                applicability: Applicability::Manual,
            })
        } else {
            Some(Advice {
                code: "FEW_TRAINING_STEPS",
                severity: Severity::Warning,
                message: format!(
                    "The run has {steps} training steps; at least {MIN_STEPS} are recommended. epochs = \"auto\" picks enough epochs for this dataset."
                ),
                evidence,
                action: Some(Action {
                    command: command("synk train --dry-run"),
                    config: config(&[("trainer.epochs", json!("auto"))]),
                }),
                applicability: Applicability::Machine,
            })
        }
    } else {
        None
    }
}

/// The `next` hint from the first warning an agent can apply as is.
pub fn next(advice: &[Advice]) -> Option<String> {
    let a = advice
        .iter()
        .find(|a| a.severity == Severity::Warning && a.applicability == Applicability::Machine)?;
    let action = a.action.as_ref()?;
    let command = action.command.clone().unwrap_or_default();
    let Some(config) = &action.config else {
        return Some(command);
    };
    // Only settings to set: try them as a variant first; that's read-only.
    if command == "synk train --dry-run" && config.values().all(|v| !v.is_null()) {
        let sets: Vec<String> = config
            .iter()
            .map(|(k, v)| match v {
                Value::String(s) => format!("--set {k}={s}"),
                v => format!("--set {k}={v}"),
            })
            .collect();
        return Some(format!("synk train --dry-run {}", sets.join(" ")));
    }
    let changes: Vec<String> = config
        .iter()
        .map(|(k, v)| match v {
            Value::Null => format!("remove {k}"),
            v => format!("set {k} = {v}"),
        })
        .collect();
    Some(format!(
        "In synk.toml, {}; then: {command}",
        changes.join(", ")
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::AutoOr;

    fn verdict(ci: Option<(f64, f64)>) -> Verdict {
        Verdict {
            outcome: "no_clear_change".into(),
            metric: "ndcg@10".into(),
            base: 0.5,
            tuned: 0.5,
            delta: 0.0,
            test_queries: 40,
            confident: false,
            message: String::new(),
            ci_low: ci.map(|c| c.0),
            ci_high: ci.map(|c| c.1),
            p_value: None,
            test_documents: 4,
            min_detectable_delta: None,
            by_lang: Default::default(),
            query_kind: "synthetic".into(),
            by_kind: Default::default(),
            reason: None,
            method: "paired_t".into(),
            permutation_p: None,
            permutation_exact: None,
            agreement: None,
        }
    }

    #[test]
    fn few_steps_with_fixed_epochs_is_machine_applicable() {
        let mut cfg = PipelineConfig::default();
        cfg.trainer.epochs = AutoOr::Value(1);
        let train = json!({"steps": 20, "examples": 300, "train_queries": 300, "filtered_queries": 10, "unmined_queries": 0});
        let advice = super::train(&train, &verdict(Some((-0.01, 0.01))), &[], &cfg);
        let codes: Vec<_> = advice.iter().map(|a| a.code).collect();
        assert_eq!(codes, ["FEW_TRAINING_STEPS"]);
        assert_eq!(advice[0].applicability, Applicability::Machine);
        assert_eq!(
            next(&advice).unwrap(),
            "synk train --dry-run --set trainer.epochs=auto"
        );

        cfg.trainer.epochs = AutoOr::Auto;
        let advice = super::train(&train, &verdict(None), &[], &cfg);
        assert_eq!(advice[0].applicability, Applicability::Manual);
        assert!(advice.iter().any(|a| a.code == "TEST_SET_UNDERPOWERED"));
        assert!(next(&advice).is_none());
    }

    #[test]
    fn dataset_advice_covers_truncation_and_small_datasets() {
        let mut cfg = PipelineConfig::default();
        cfg.dataset.chunk_chars = Some(1000);
        let summary = json!({"chunks": 100, "truncated_chunks": 30, "train_queries": 120});
        let advice = dataset(&summary, &cfg);
        let codes: Vec<_> = advice.iter().map(|a| a.code).collect();
        assert_eq!(
            codes,
            ["CHUNKS_TRUNCATED", "CHUNK_CHARS_DEPRECATED", "FEW_QUERIES"]
        );
        assert_eq!(
            next(&advice).unwrap(),
            "In synk.toml, remove dataset.chunk_chars, set dataset.chunk_tokens = \"auto\"; then: synk train --dry-run"
        );
        let value = serde_json::to_value(&advice[2]).unwrap();
        assert_eq!(value["action"]["config"]["generator.queries_per_chunk"], 5);
        assert_eq!(value["applicability"], "machine");
    }
}
