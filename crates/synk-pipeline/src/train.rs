//! Training a model on a dataset, then comparing it with its base model.
//!
//! Both steps are artifacts. A training run's key covers the dataset, the
//! base model, the training settings and the library versions, so changing
//! any of them trains again, and repeating a finished run is free. An
//! interrupted run keeps its checkpoints in `<key>.tmp/` and can be resumed.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use synk_core::{ErrorCode, Result, SynkError};

use crate::artifact::{self, Artifacts, Manifest};
use crate::catalog::{ModelSpec, RerankerSpec};
use crate::config::{AutoOr, TrainerConfig};
use crate::sidecar::{PROTOCOL_VERSION, TrainerEnv};
use crate::stats;

pub const TRAIN_KIND: &str = "train";
const TRAIN_VERSION: u32 = 3;
pub const EVAL_KIND: &str = "eval";
// 4: verdict reason, sign-flip p, agreement; real queries need 10 documents.
const EVAL_VERSION: u32 = 4;

/// One public set of the general check, pinned to a commit.
pub struct GeneralSet {
    pub name: &'static str,
    pub repo: &'static str,
    pub revision: &'static str,
    /// `beir` (corpus/queries/qrels) or `squad` (question + context rows).
    pub format: &'static str,
    pub file: &'static str,
    pub allow_patterns: &'static [&'static str],
    pub max_queries: Option<u32>,
}

/// The general check: English web search (NanoMSMARCO, a 50-query BEIR
/// subset) and Vietnamese reading comprehension as retrieval (XQuAD vi,
/// 300 of its questions over its 240 paragraphs).
pub const GENERAL_SETS: &[GeneralSet] = &[
    GeneralSet {
        name: "en",
        repo: "zeta-alpha-ai/NanoMSMARCO",
        revision: "7b8ff22f2771dc65ac5b439f222eb19a1f56abda",
        format: "beir",
        file: "",
        allow_patterns: &["corpus/*", "queries/*", "qrels/*"],
        max_queries: None,
    },
    GeneralSet {
        name: "vi",
        repo: "google/xquad",
        revision: "51adfef1c1287aab1d2d91b5bead9bcfb9c68583",
        format: "squad",
        file: "xquad.vi/validation-00000-of-00001.parquet",
        allow_patterns: &["xquad.vi/*"],
        max_queries: Some(300),
    },
];

/// The general sets as the trainer reads them.
fn general_request_sets() -> Value {
    json!(
        GENERAL_SETS
            .iter()
            .map(|s| json!({
                "name": s.name, "repo": s.repo, "revision": s.revision, "format": s.format,
                "file": s.file, "allow_patterns": s.allow_patterns, "max_queries": s.max_queries,
            }))
            .collect::<Vec<_>>()
    )
}
const MODEL_DIR: &str = "model";
/// Ranks of every test query under each model, written by the sidecar.
pub const PER_QUERY_FILE: &str = "per_query.jsonl";

/// `epochs = "auto"` trains at least this many steps...
const MIN_STEPS: u64 = 300;
/// ...but never more than this many epochs.
const MAX_AUTO_EPOCHS: u64 = 5;
/// The round-trip filter keeps a query only if the base model ranks its
/// chunk this high or better.
const FILTER_K: u32 = 20;
/// The verdict is decided on real queries once there are this many.
pub const MIN_REAL_QUERIES: usize = 30;
/// ...and they must cover at least this many documents: queries about one
/// document rise and fall together, so 30 about 2 documents are 2 data
/// points.
pub const MIN_REAL_DOCUMENTS: u64 = 10;
/// A typical gain from tuning on one's own documents (nDCG@10). A test that
/// can't see a change this size is too small to give a verdict.
pub const TYPICAL_GAIN: f64 = 0.05;
/// Units for which an exact sign-flip test can't reach p < 0.05 (its
/// smallest p is 2/2^n).
const SIGN_FLIP_BLIND_UP_TO: u64 = 5;

pub struct Options {
    pub resume: bool,
    pub force: bool,
    pub device: Option<String>,
    pub reranker: Option<RerankerSpec>,
    /// A file whose existence asks the trainer to stop after a checkpoint.
    pub stop_file: Option<PathBuf>,
    /// Also run the general check.
    pub general_check: bool,
    /// Evaluate again even if the evaluation exists.
    pub force_eval: bool,
}

pub struct Outcome {
    pub train: Manifest,
    pub train_reused: bool,
    pub eval: Manifest,
    pub model_dir: PathBuf,
    pub verdict: Verdict,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Verdict {
    /// `better`, `worse` or `no_clear_change`.
    pub outcome: String,
    pub metric: String,
    pub base: f64,
    pub tuned: f64,
    pub delta: f64,
    pub test_queries: u64,
    /// True when the 95% interval for the change excludes 0.
    pub confident: bool,
    pub message: String,
    /// 95% confidence interval for `delta`; `None` with under 2 test documents.
    #[serde(default)]
    pub ci_low: Option<f64>,
    #[serde(default)]
    pub ci_high: Option<f64>,
    #[serde(default)]
    pub p_value: Option<f64>,
    #[serde(default)]
    pub test_documents: u64,
    /// The smallest change this test set detects reliably (80% power).
    #[serde(default)]
    pub min_detectable_delta: Option<f64>,
    /// The same verdict for each query language.
    #[serde(default)]
    pub by_lang: BTreeMap<String, Verdict>,
    /// Which queries decided: `synthetic` (held-out documents) or `real`
    /// (imported, once there are enough).
    #[serde(default = "synthetic")]
    pub query_kind: String,
    /// The verdict on each kind of query, when real queries exist.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub by_kind: BTreeMap<String, Verdict>,
    /// Why there's no clear change: `no_difference` (the test could have
    /// seen a typical gain and didn't) or `too_few_test_documents` (it
    /// couldn't: the verdict is inconclusive). Only with `no_clear_change`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    /// The decision rule: `paired_t` (a paired t-test on per-document
    /// changes).
    #[serde(default = "paired_t")]
    pub method: String,
    /// A cross-check: the two-sided sign-flip p-value on the same
    /// per-document changes, which assumes nothing about their shape.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub permutation_p: Option<f64>,
    /// Whether `permutation_p` enumerated every flip (up to 20 documents).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub permutation_exact: Option<bool>,
    /// With real queries: whether the synthetic and the real verdict have
    /// the same outcome (`agree` or `disagree`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agreement: Option<String>,
}

fn paired_t() -> String {
    "paired_t".to_owned()
}

fn synthetic() -> String {
    "synthetic".to_owned()
}

/// The parameters, inputs and key of a training run.
fn run_key(
    dataset: &Manifest,
    model: &ModelSpec,
    reranker: Option<&RerankerSpec>,
    cfg: &TrainerConfig,
    env: &TrainerEnv,
) -> (Value, Vec<String>, String) {
    let mut params = json!({
        "base_model": model,
        "epochs": cfg.epochs,
        "min_steps": MIN_STEPS,
        "max_auto_epochs": MAX_AUTO_EPOCHS,
        "batch_size": cfg.batch_size,
        "learning_rate": cfg.learning_rate,
        "max_seq_length": cfg.max_seq_length,
        "seed": cfg.seed,
        "warmup": 0.1,
        "negatives": cfg.negatives,
        "negative_margin": 0.05,
        "query_filter": cfg.query_filter,
        "filter_k": FILTER_K,
        "dev_percent": 10,
        "dev_min_documents": 20,
        "finetune": finetune(model, cfg),
        "precision": cfg.precision,
        "gradient_checkpointing": "auto",
        "cached_loss_above_batch": 32,
        "mrl_dims": mrl_dims(model),
    });
    if params["finetune"] != "full" {
        params["lora"] = json!({
            "r": cfg.lora_r,
            "alpha": cfg.lora_alpha,
            "dropout": 0.05,
            "learning_rate": cfg.lora_learning_rate,
        });
    }
    // Only present when set, so runs without one keep their keys.
    if let Some(reranker) = reranker {
        params["reranker"] = json!(reranker);
        params["reranker_max_prob"] = json!(0.5);
    }
    let env_key = blake3::hash(env.fingerprint().to_string().as_bytes()).to_hex()[..16].to_owned();
    let inputs = vec![format!("dataset:{}", dataset.key), format!("env:{env_key}")];
    let key = artifact::key(TRAIN_KIND, TRAIN_VERSION, &params, &inputs);
    (params, inputs, key)
}

/// `full` or `lora` for catalog models (LoRA from 300M parameters); `auto`
/// for others, which the trainer decides by counting parameters.
pub fn finetune(model: &ModelSpec, cfg: &TrainerConfig) -> String {
    if cfg.finetune != "auto" {
        return cfg.finetune.clone();
    }
    let millions = model
        .catalog_id
        .as_deref()
        .and_then(|id| crate::catalog::models().into_iter().find(|m| m.id == id))
        .and_then(|m| crate::catalog::parameters_millions(&m.parameters));
    match millions {
        Some(m) if m >= 300.0 => "lora".to_owned(),
        Some(_) => "full".to_owned(),
        None => "auto".to_owned(),
    }
}

/// The Matryoshka sizes of a catalog model, if it has them.
fn mrl_dims(model: &ModelSpec) -> Vec<u32> {
    model
        .catalog_id
        .as_deref()
        .and_then(|id| crate::catalog::models().into_iter().find(|m| m.id == id))
        .map(|m| m.mrl_dims)
        .unwrap_or_default()
}

fn device(opts_device: Option<&str>, cfg: &TrainerConfig) -> Option<String> {
    opts_device
        .map(str::to_owned)
        .or_else(|| cfg.device.clone())
}

/// Epochs and steps a run will train, from the number of examples. The
/// sidecar makes the same choice after filtering, so this is an estimate.
pub fn schedule(examples: u64, cfg: &TrainerConfig) -> (u64, u64) {
    let batch = u64::from(cfg.batch_size).min(examples).max(1);
    let per_epoch = examples.div_ceil(batch).max(1);
    let epochs = match cfg.epochs {
        AutoOr::Auto => MIN_STEPS.div_ceil(per_epoch).clamp(1, MAX_AUTO_EPOCHS),
        AutoOr::Value(n) => u64::from(n),
    };
    (epochs, per_epoch * epochs)
}

/// What `run` would do, without doing it. `env` is `None` when the training
/// environment isn't set up yet.
pub fn plan(
    artifacts: &Artifacts,
    dataset: &Manifest,
    model: &ModelSpec,
    reranker: Option<&RerankerSpec>,
    cfg: &TrainerConfig,
    env: Option<&TrainerEnv>,
    opts_device: Option<&str>,
) -> Result<Value> {
    let examples = dataset.summary["train_queries"].as_u64().unwrap_or(0);
    let (epochs, steps) = schedule(examples, cfg);
    let (run, existing, interrupted) = match env {
        Some(env) => {
            let (_, _, key) = run_key(dataset, model, reranker, cfg, env);
            let existing = artifacts.manifest(TRAIN_KIND, &key)?.is_some();
            let interrupted = !existing && artifacts.tmp_dir(TRAIN_KIND, &key).exists();
            (Some(key), existing, interrupted)
        }
        None => (None, false, false),
    };
    let device = device(opts_device, cfg)
        .or_else(|| env.and_then(|e| e.describe["devices"][0].as_str().map(str::to_owned)));
    Ok(json!({
        "run": run,
        "existing": existing,
        "interrupted": interrupted,
        "dataset": dataset.key,
        "base_model": model.label(),
        "examples": examples,
        "epochs": epochs,
        "steps": steps,
        "batch_size": cfg.batch_size,
        "device": device,
        "query_filter": cfg.query_filter,
        "negatives": cfg.negatives,
        "reranker": reranker.map(|r| r.catalog_id.clone().unwrap_or_else(|| r.name_or_path.clone())),
        "trainer_ready": env.is_some(),
        "estimate": "examples, epochs and steps are before the query filter, which can remove some queries",
    }))
}

#[allow(clippy::too_many_arguments)]
pub fn run(
    artifacts: &Artifacts,
    dataset: &Manifest,
    dataset_dir: &Path,
    model: &ModelSpec,
    cfg: &TrainerConfig,
    env: &TrainerEnv,
    opts: &Options,
    on_event: &mut dyn FnMut(&Value),
) -> Result<Outcome> {
    let device = device(opts.device.as_deref(), cfg);
    let (params, inputs, key) = run_key(dataset, model, opts.reranker.as_ref(), cfg, env);

    if opts.force {
        artifacts.remove(TRAIN_KIND, &key)?;
    }
    let (train, train_reused) = match artifacts.manifest(TRAIN_KIND, &key)? {
        Some(manifest) => (manifest, true),
        None => {
            let tmp = artifacts.tmp_dir(TRAIN_KIND, &key);
            if tmp.exists() && !opts.resume {
                return Err(SynkError::new(
                    ErrorCode::RunIncomplete,
                    "a training run with these settings was interrupted",
                )
                .with_hint("run synk train again to continue it"));
            }
            let tmp = artifacts.begin(TRAIN_KIND, &key, !opts.resume)?;
            let request = json!({
                "v": PROTOCOL_VERSION,
                "task": "train",
                "device": device,
                "base_model": model,
                "dataset_dir": dataset_dir,
                "output_dir": tmp,
                "params": params,
                "resume": opts.resume,
                "stop_file": opts.stop_file,
            });
            let mut result = env.run(&request, &tmp, on_event)?;
            // Paths inside the work directory are wrong after the commit.
            result["model_dir"] = json!(MODEL_DIR);
            result["base_model"] = json!(model.label());
            let manifest = Manifest::new(TRAIN_KIND, &key, params, inputs, result);
            artifacts.commit(&manifest)?;
            (manifest, false)
        }
    };
    artifacts.set_latest(TRAIN_KIND, &key)?;
    let model_dir = artifacts.dir(TRAIN_KIND, &key).join(MODEL_DIR);

    let eval = evaluate(
        artifacts,
        dataset,
        dataset_dir,
        model,
        &train,
        &model_dir,
        cfg,
        device,
        env,
        opts.force || opts.force_eval,
        opts.stop_file.as_deref(),
        opts.general_check,
        on_event,
    )?;
    let verdict = eval_verdict(&eval);
    Ok(Outcome {
        train,
        train_reused,
        eval,
        model_dir,
        verdict,
    })
}

#[allow(clippy::too_many_arguments)]
fn evaluate(
    artifacts: &Artifacts,
    dataset: &Manifest,
    dataset_dir: &Path,
    model: &ModelSpec,
    train: &Manifest,
    model_dir: &Path,
    cfg: &TrainerConfig,
    device: Option<String>,
    env: &TrainerEnv,
    force: bool,
    stop_file: Option<&Path>,
    general: bool,
    on_event: &mut dyn FnMut(&Value),
) -> Result<Manifest> {
    let (params, inputs, key) =
        eval_key(&train.key, &dataset.key, json!(cfg.max_seq_length), general);
    // A forced run trains a new model under the same key, so its old
    // evaluation describes a model that no longer exists.
    if force {
        artifacts.remove(EVAL_KIND, &key)?;
    } else if let Some(manifest) = artifacts.manifest(EVAL_KIND, &key)? {
        artifacts.set_latest(EVAL_KIND, &key)?;
        return Ok(manifest);
    }

    let tmp = artifacts.begin(EVAL_KIND, &key, true)?;
    // The tuned model was saved with the prompts training used.
    let tuned = json!({
        "label": "tuned",
        "name_or_path": model_dir,
        "revision": null,
        "query_prompt": train.summary["prompts"]["query"],
        "document_prompt": train.summary["prompts"]["document"],
    });
    let mut base = serde_json::to_value(model).expect("model spec serializes");
    base["label"] = json!("base");
    let request = json!({
        "v": PROTOCOL_VERSION,
        "task": "eval",
        "device": device,
        "dataset_dir": dataset_dir,
        "output_dir": tmp,
        "split": "test",
        "max_seq_length": cfg.max_seq_length,
        "models": [base, tuned],
        "stop_file": stop_file,
    });
    let mut result = env.run(&request, &tmp, on_event)?;
    let rows = read_per_query(&tmp.join(PER_QUERY_FILE))?;
    result["verdict"] = serde_json::to_value(verdict(&rows)).expect("verdict serializes");
    if general {
        let request = json!({
            "v": PROTOCOL_VERSION,
            "task": "general",
            "device": request["device"],
            "output_dir": tmp,
            "max_seq_length": cfg.max_seq_length,
            "models": request["models"],
            "sets": general_request_sets(),
            "stop_file": stop_file,
        });
        let summary = env.run(&request, &tmp, on_event)?;
        let rows = read_per_query(&tmp.join("general.jsonl"))?;
        let mut by_set = serde_json::Map::new();
        for set in GENERAL_SETS {
            let set_rows: Vec<QueryRanks> = rows
                .iter()
                .filter(|r| r.lang.as_deref() == Some(set.name))
                .cloned()
                .collect();
            if !set_rows.is_empty() {
                let mut v = verdict(&set_rows);
                v.by_lang.clear();
                by_set.insert(
                    set.name.to_owned(),
                    serde_json::to_value(v).expect("verdict serializes"),
                );
            }
        }
        result["general_check"] = json!({ "sets": summary["sets"], "verdicts": by_set });
    }
    let manifest = Manifest::new(EVAL_KIND, &key, params, inputs, result);
    artifacts.commit(&manifest)?;
    artifacts.set_latest(EVAL_KIND, &key)?;
    Ok(manifest)
}

/// Whether the evaluation of run `train_key` on dataset `dataset_key` exists.
pub fn eval_cached(
    artifacts: &Artifacts,
    train_key: &str,
    dataset_key: &str,
    cfg: &TrainerConfig,
    general: bool,
) -> Result<bool> {
    let (_, _, key) = eval_key(train_key, dataset_key, json!(cfg.max_seq_length), general);
    Ok(artifacts.manifest(EVAL_KIND, &key)?.is_some())
}

fn eval_key(
    train: &str,
    dataset: &str,
    max_seq_length: Value,
    general: bool,
) -> (Value, Vec<String>, String) {
    let mut params = json!({ "split": "test", "max_seq_length": max_seq_length });
    if general {
        params["general"] = general_request_sets();
    }
    let inputs = vec![format!("train:{train}"), format!("dataset:{dataset}")];
    let key = artifact::key(EVAL_KIND, EVAL_VERSION, &params, &inputs);
    (params, inputs, key)
}

/// The evaluation of a finished training run, if there is one.
pub fn eval_for(artifacts: &Artifacts, train: &Manifest) -> Result<Option<Manifest>> {
    let Some(dataset) = train.inputs.iter().find_map(|i| i.strip_prefix("dataset:")) else {
        return Ok(None);
    };
    for general in [true, false] {
        let (_, _, key) = eval_key(
            &train.key,
            dataset,
            train.params["max_seq_length"].clone(),
            general,
        );
        if let Some(eval) = artifacts.manifest(EVAL_KIND, &key)? {
            return Ok(Some(eval));
        }
    }
    // Evaluations made by an older synk have another key; the latest one
    // still names its run.
    Ok(artifacts
        .latest(EVAL_KIND)?
        .filter(|e| e.inputs.contains(&format!("train:{}", train.key))))
}

/// One test query's rank under the base and the tuned model.
#[derive(Debug, Clone, Deserialize)]
pub struct QueryRanks {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub query: String,
    /// `synthetic` or `real`.
    #[serde(default = "synthetic")]
    pub kind: String,
    #[serde(default)]
    pub chunk: String,
    pub doc: String,
    #[serde(default)]
    pub lang: Option<String>,
    pub ranks: BTreeMap<String, u64>,
    /// The chunk each model ranked first.
    #[serde(default)]
    pub top: BTreeMap<String, String>,
}

pub fn read_per_query(path: &Path) -> Result<Vec<QueryRanks>> {
    let text = fs::read_to_string(path).map_err(|e| SynkError::io(path.display(), e))?;
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            serde_json::from_str(l).map_err(|e| {
                SynkError::new(
                    ErrorCode::SidecarProtocol,
                    format!("{}: {e}", path.display()),
                )
            })
        })
        .collect()
}

/// The verdict saved with an evaluation, or one computed from its metrics
/// for evaluations made before verdicts were saved.
pub fn eval_verdict(eval: &Manifest) -> Verdict {
    serde_json::from_value(eval.summary["verdict"].clone())
        .unwrap_or_else(|_| legacy_verdict(&eval.summary))
}

fn ndcg_at_10(rank: u64) -> f64 {
    if rank <= 10 {
        1.0 / ((rank + 1) as f64).log2()
    } else {
        0.0
    }
}

fn round4(x: f64) -> f64 {
    (x * 10_000.0).round() / 10_000.0
}

/// Compares the tuned model with the base model, overall and per language.
/// Real queries decide once there are enough of them; synthetic queries on
/// held-out documents decide otherwise. Each kind is also reported.
pub fn verdict(rows: &[QueryRanks]) -> Verdict {
    decide(rows, &Names::tuned_vs_base())
}

/// Compares two tuned models on the same evaluation queries: `a` is the
/// reference, and "better" means `b` is better.
pub fn compare(a: &[QueryRanks], b: &[QueryRanks], a_key: &str, b_key: &str) -> Verdict {
    let b_by_id: BTreeMap<&str, &QueryRanks> = b.iter().map(|r| (r.id.as_str(), r)).collect();
    let rows: Vec<QueryRanks> = a
        .iter()
        .filter_map(|ra| {
            let rb = b_by_id.get(ra.id.as_str())?;
            let rank = |r: &QueryRanks| r.ranks.get("tuned").copied();
            Some(QueryRanks {
                ranks: [
                    ("base".to_owned(), rank(ra)?),
                    ("tuned".to_owned(), rank(rb)?),
                ]
                .into(),
                ..ra.clone()
            })
        })
        .collect();
    decide(&rows, &Names::runs(a_key, b_key))
}

/// How a verdict's message names the two sides.
struct Names {
    better: String,
    worse: String,
    neither: String,
}

impl Names {
    fn tuned_vs_base() -> Self {
        Self {
            better: "The tuned model is better".to_owned(),
            worse: "The tuned model is worse".to_owned(),
            neither: "No clear difference between the tuned and base models".to_owned(),
        }
    }

    fn runs(a: &str, b: &str) -> Self {
        Self {
            better: format!("Run {b} is better than run {a}"),
            worse: format!("Run {b} is worse than run {a}"),
            neither: format!("No clear difference between runs {a} and {b}"),
        }
    }
}

/// Whether this many real queries, about this many documents, decide the
/// verdict instead of the generated test.
pub fn real_queries_decide(queries: usize, documents: u64) -> bool {
    queries >= MIN_REAL_QUERIES && documents >= MIN_REAL_DOCUMENTS
}

fn decide(rows: &[QueryRanks], names: &Names) -> Verdict {
    let (real, synthetic): (Vec<&QueryRanks>, Vec<&QueryRanks>) =
        rows.iter().partition(|r| r.kind == "real");
    let mut synthetic_v = judge_with_langs(&synthetic, names, "test queries");
    if real.is_empty() {
        return synthetic_v;
    }
    let mut real_v = judge_with_langs(&real, names, "real queries");
    real_v.query_kind = "real".to_owned();
    let mut chosen = if real_queries_decide(real.len(), real_v.test_documents) {
        real_v.clone()
    } else {
        let mut v = synthetic_v.clone();
        let _ = write!(
            v.message,
            " {} real queries about {} documents are reported separately; {MIN_REAL_QUERIES} or more about {MIN_REAL_DOCUMENTS}+ documents decide the verdict.",
            real.len(),
            real_v.test_documents
        );
        v
    };
    chosen.agreement = Some(
        if synthetic_v.outcome == real_v.outcome {
            "agree"
        } else {
            "disagree"
        }
        .to_owned(),
    );
    synthetic_v.query_kind = "synthetic".to_owned();
    chosen.by_kind = [
        ("synthetic".to_owned(), synthetic_v),
        ("real".to_owned(), real_v),
    ]
    .into();
    chosen
}

fn judge_with_langs(rows: &[&QueryRanks], names: &Names, noun: &str) -> Verdict {
    let mut overall = judge(rows, names, noun);
    let mut langs: BTreeMap<String, Vec<&QueryRanks>> = BTreeMap::new();
    for row in rows {
        langs
            .entry(row.lang.clone().unwrap_or_else(|| "unknown".to_owned()))
            .or_default()
            .push(row);
    }
    overall.by_lang = langs
        .into_iter()
        .map(|(lang, rows)| (lang, judge(&rows, names, noun)))
        .collect();
    overall
}

/// A paired t-test on per-document mean nDCG@10 changes. The document is
/// the unit because its queries share one text and are not independent.
fn judge(rows: &[&QueryRanks], names: &Names, noun: &str) -> Verdict {
    let score =
        |r: &QueryRanks, label: &str| ndcg_at_10(r.ranks.get(label).copied().unwrap_or(u64::MAX));
    let n = rows.len().max(1) as f64;
    let base = rows.iter().map(|r| score(r, "base")).sum::<f64>() / n;
    let tuned = rows.iter().map(|r| score(r, "tuned")).sum::<f64>() / n;
    let mut by_doc: BTreeMap<&str, (f64, u32)> = BTreeMap::new();
    for r in rows {
        let entry = by_doc.entry(&r.doc).or_default();
        entry.0 += score(r, "tuned") - score(r, "base");
        entry.1 += 1;
    }
    let diffs: Vec<f64> = by_doc
        .values()
        .map(|(sum, k)| sum / f64::from(*k))
        .collect();
    let test = stats::paired_t(&diffs);
    let flips = stats::sign_flip_p(&diffs);
    let outcome = match test {
        Some(t) if t.ci_low > 0.0 => "better",
        Some(t) if t.ci_high < 0.0 => "worse",
        _ => "no_clear_change",
    };
    let delta = tuned - base;
    let (queries, documents) = (rows.len(), by_doc.len());
    let headline = match outcome {
        "better" => &names.better,
        "worse" => &names.worse,
        _ => &names.neither,
    };
    let mut message = format!("{headline}: nDCG@10 {base:.3} -> {tuned:.3} ({delta:+.3}");
    match test {
        Some(_) if diffs.iter().all(|d| *d == 0.0) => {
            let _ = write!(
                message,
                ") on {queries} {noun} from {documents} documents. Both models rank every query the same, so these queries can't tell them apart."
            );
        }
        Some(t) => {
            let _ = write!(
                message,
                ", 95% CI {:+.3} to {:+.3}) on {queries} {noun} from {documents} documents.",
                t.ci_low, t.ci_high
            );
            if outcome == "no_clear_change" {
                let _ = write!(
                    message,
                    " This test set can only detect changes of about {:.3} or more.",
                    t.min_detectable
                );
                if t.min_detectable > TYPICAL_GAIN {
                    let _ = write!(
                        message,
                        " Tuning usually brings about {TYPICAL_GAIN:.2}, so this is inconclusive: add documents, or write real searches (synk queries template)."
                    );
                }
            } else if documents as u64 <= SIGN_FLIP_BLIND_UP_TO {
                let _ = write!(
                    message,
                    " With only {documents} test documents, a test that assumes nothing about the data can't be sure (its smallest p is {:.3}); this rests on the t-test's assumption that changes are roughly normal.",
                    2.0 / f64::from(1u32 << documents)
                );
            }
        }
        None => {
            let _ = write!(
                message,
                ") on {queries} {noun} from {documents} document(s), too few to tell; add documents for a verdict."
            );
        }
    }
    Verdict {
        outcome: outcome.to_owned(),
        metric: "ndcg@10".to_owned(),
        base: round4(base),
        tuned: round4(tuned),
        delta: round4(delta),
        test_queries: queries as u64,
        confident: outcome != "no_clear_change",
        message,
        ci_low: test.map(|t| round4(t.ci_low)),
        ci_high: test.map(|t| round4(t.ci_high)),
        p_value: test.map(|t| round4(t.p_value)),
        test_documents: documents as u64,
        min_detectable_delta: test.map(|t| round4(t.min_detectable)),
        by_lang: BTreeMap::new(),
        query_kind: synthetic(),
        by_kind: BTreeMap::new(),
        reason: (outcome == "no_clear_change").then(|| {
            match test {
                Some(t) if t.min_detectable <= TYPICAL_GAIN => "no_difference",
                _ => "too_few_test_documents",
            }
            .to_owned()
        }),
        method: paired_t(),
        permutation_p: flips.map(|(p, _)| round4(p)),
        permutation_exact: flips.map(|(_, exact)| exact),
        agreement: None,
    }
}

/// The verdict rule before per-query ranks were saved: a fixed 0.01 margin.
fn legacy_verdict(eval: &Value) -> Verdict {
    let metric = "ndcg@10";
    let get = |label: &str| {
        eval["metrics"][label]["overall"][metric]
            .as_f64()
            .unwrap_or(0.0)
    };
    let (base, tuned) = (get("base"), get("tuned"));
    let test_queries = eval["metrics"]["base"]["overall"]["queries"]
        .as_u64()
        .unwrap_or(0);
    let delta = tuned - base;
    let outcome = if delta >= 0.01 {
        "better"
    } else if delta <= -0.01 {
        "worse"
    } else {
        "no_clear_change"
    };
    Verdict {
        outcome: outcome.to_owned(),
        metric: metric.to_owned(),
        base,
        tuned,
        delta: round4(delta),
        test_queries,
        confident: false,
        message: format!(
            "nDCG@10 {base:.3} -> {tuned:.3} ({delta:+.3}) on {test_queries} test queries. Evaluated by an older synk without a significance test; run synk train --force eval for one."
        ),
        ci_low: None,
        ci_high: None,
        p_value: None,
        test_documents: 0,
        min_detectable_delta: None,
        by_lang: BTreeMap::new(),
        query_kind: synthetic(),
        by_kind: BTreeMap::new(),
        reason: None,
        method: "margin".to_owned(),
        permutation_p: None,
        permutation_exact: None,
        agreement: None,
    }
}

/// A README for an exported model. It describes the run and its verdict
/// and, unlike sentence-transformers' own model card, holds no text from the
/// training data.
pub fn model_card(train: &Manifest, eval: Option<&Manifest>) -> String {
    let s = &train.summary;
    let mut card = format!(
        "# Tuned embedding model\n\nFine-tuned with synk {} from `{}`.\n\n\
         | Setting | Value |\n|---|---|\n\
         | Base model | `{}` |\n| Training run | `{}` |\n| Training examples | {} |\n\
         | Epochs | {} (kept epoch {}) |\n| Steps | {} |\n| Hard negatives per query | {} |\n\
         | Batch size | {} |\n| Learning rate | {} |\n| Max sequence length | {} |\n",
        train.synk_version,
        s["base_model"].as_str().unwrap_or("?"),
        s["base_model"].as_str().unwrap_or("?"),
        train.key,
        s["examples"],
        s["epochs"],
        s["best_epoch"],
        s["steps"],
        s["negatives_per_query"],
        train.params["batch_size"],
        train.params["learning_rate"],
        train.params["max_seq_length"],
    );
    if let Some(eval) = eval {
        let v = eval_verdict(eval);
        let _ = write!(
            card,
            "\n## Evaluation\n\nOn documents held out from training: {}\n",
            v.message
        );
    }
    let _ = write!(
        card,
        "\n## Usage\n\n```python\nfrom sentence_transformers import SentenceTransformer\n\n\
         model = SentenceTransformer(\"path/to/this/folder\")\n\
         # encode_query / encode_document add the query and document prompts\n\
         # saved with this model; it was trained and tested with them.\n\
         queries = model.encode_query([\"...\"], normalize_embeddings=True)\n\
         documents = model.encode_document([\"...\"], normalize_embeddings=True)\n```\n"
    );
    card
}

/// Copies a tuned model (a sentence-transformers folder) to `out`, which must
/// not exist or be empty, with `readme` as its README.md.
pub fn export_folder(model_dir: &Path, out: &Path, readme: &str) -> Result<()> {
    if out.exists()
        && fs::read_dir(out)
            .map_err(|e| SynkError::io(out.display(), e))?
            .next()
            .is_some()
    {
        return Err(SynkError::new(
            ErrorCode::ExportExists,
            format!("{} already exists and is not empty", out.display()),
        )
        .with_hint("pick another --out directory, or add --force to replace it"));
    }
    copy_dir(model_dir, out)?;
    let path = out.join("README.md");
    fs::write(&path, readme).map_err(|e| SynkError::io(path.display(), e))
}

fn copy_dir(from: &Path, to: &Path) -> Result<()> {
    fs::create_dir_all(to).map_err(|e| SynkError::io(to.display(), e))?;
    for entry in fs::read_dir(from).map_err(|e| SynkError::io(from.display(), e))? {
        let entry = entry.map_err(|e| SynkError::io(from.display(), e))?;
        let target = to.join(entry.file_name());
        if entry.path().is_dir() {
            copy_dir(&entry.path(), &target)?;
        } else {
            fs::copy(entry.path(), &target).map_err(|e| SynkError::io(target.display(), e))?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(doc: &str, lang: &str, base: u64, tuned: u64) -> QueryRanks {
        QueryRanks {
            id: format!("{doc}-{lang}-{base}-{tuned}"),
            query: String::new(),
            kind: "synthetic".into(),
            chunk: String::new(),
            doc: doc.into(),
            lang: Some(lang.into()),
            ranks: [("base".to_owned(), base), ("tuned".to_owned(), tuned)].into(),
            top: BTreeMap::new(),
        }
    }

    #[test]
    fn enough_real_queries_decide_the_verdict() {
        let mut rows: Vec<QueryRanks> = (0..6).map(|d| row(&format!("d{d}"), "en", 1, 1)).collect();
        let real = |n: usize, docs: usize| -> Vec<QueryRanks> {
            (0..n)
                .map(|i| QueryRanks {
                    kind: "real".into(),
                    id: format!("real-{i}"),
                    ..row(&format!("d{}", i % docs), "vi", 4, 1)
                })
                .collect()
        };
        rows.extend(real(5, 5));
        let few = verdict(&rows);
        assert_eq!(few.query_kind, "synthetic");
        assert!(
            few.message
                .contains("5 real queries about 5 documents are reported separately"),
            "{}",
            few.message
        );
        assert_eq!(few.by_kind["real"].outcome, "better");
        // Synthetic says no change, real says better.
        assert_eq!(few.agreement.as_deref(), Some("disagree"));

        // 40 queries, but about 8 documents only: still reported, not deciding.
        rows.truncate(6);
        rows.extend(real(40, 8));
        assert_eq!(verdict(&rows).query_kind, "synthetic");

        rows.truncate(6);
        rows.extend(real(40, 10));
        let many = verdict(&rows);
        assert_eq!(
            (many.query_kind.as_str(), many.outcome.as_str()),
            ("real", "better")
        );
        assert!(many.message.contains("real queries"));
    }

    #[test]
    fn no_clear_change_says_whether_the_test_could_see_a_gain() {
        // Two documents: the test can't see a typical gain: inconclusive.
        let tiny = verdict(&[row("d0", "en", 2, 1), row("d1", "en", 1, 2)]);
        assert_eq!(tiny.outcome, "no_clear_change");
        assert_eq!(tiny.reason.as_deref(), Some("too_few_test_documents"));
        assert!(tiny.message.contains("inconclusive"), "{}", tiny.message);
        // Many documents with tiny, mixed changes: it could have seen one.
        let big: Vec<_> = (0..200)
            .map(|d| {
                let t = if d % 2 == 0 { 1 } else { 2 };
                row(&format!("d{d}"), "en", 1, if d % 50 == 0 { t } else { 1 })
            })
            .collect();
        let v = verdict(&big);
        assert_eq!(v.outcome, "no_clear_change", "{}", v.message);
        assert_eq!(v.reason.as_deref(), Some("no_difference"), "{}", v.message);
        // Better and worse have no reason; the sign-flip p is reported.
        let better: Vec<_> = (0..4).map(|d| row(&format!("d{d}"), "en", 3, 1)).collect();
        let v = verdict(&better);
        assert_eq!(v.reason, None);
        assert_eq!(v.permutation_exact, Some(true));
        assert!(v.permutation_p.unwrap() > 0.05);
    }

    #[test]
    fn compare_pairs_two_runs_by_query() {
        let a: Vec<_> = (0..6).map(|d| row(&format!("d{d}"), "en", 9, 5)).collect();
        let b: Vec<_> = (0..6)
            .map(|d| QueryRanks {
                ranks: [("base".to_owned(), 9), ("tuned".to_owned(), 1)].into(),
                ..row(&format!("d{d}"), "en", 9, 5)
            })
            .collect();
        let v = compare(&a, &b, "aaa", "bbb");
        assert_eq!(v.outcome, "better");
        assert!(
            v.message.starts_with("Run bbb is better than run aaa"),
            "{}",
            v.message
        );
    }

    #[test]
    fn verdict_is_better_only_when_the_interval_excludes_zero() {
        // Every document improves from rank 3 to rank 1.
        let better: Vec<_> = (0..8)
            .flat_map(|d| {
                [
                    row(&format!("d{d}"), "en", 3, 1),
                    row(&format!("d{d}"), "vi", 2, 1),
                ]
            })
            .collect();
        let v = verdict(&better);
        assert_eq!((v.outcome.as_str(), v.confident), ("better", true));
        assert!(v.ci_low.unwrap() > 0.0);
        assert_eq!(v.test_documents, 8);
        assert_eq!(v.by_lang["vi"].outcome, "better");

        // Mixed results: half the documents improve, half get worse.
        let mixed: Vec<_> = (0..8)
            .map(|d| {
                let (b, t) = if d % 2 == 0 { (3, 1) } else { (1, 2) };
                row(&format!("d{d}"), "en", b, t)
            })
            .collect();
        let v = verdict(&mixed);
        assert_eq!(
            (v.outcome.as_str(), v.confident),
            ("no_clear_change", false)
        );
        assert!(v.message.contains("can only detect"));

        let worse: Vec<_> = (0..6).map(|d| row(&format!("d{d}"), "en", 1, 20)).collect();
        assert_eq!(verdict(&worse).outcome, "worse");

        let same: Vec<_> = (0..4).map(|d| row(&format!("d{d}"), "en", 1, 1)).collect();
        let v = verdict(&same);
        assert_eq!(v.outcome, "no_clear_change");
        assert!(v.message.contains("can't tell them apart"), "{}", v.message);

        // One document: no interval, so no verdict either way.
        let single = verdict(&[row("d0", "en", 5, 1), row("d0", "en", 4, 1)]);
        assert_eq!(single.outcome, "no_clear_change");
        assert!(single.ci_low.is_none());
        assert!(single.message.contains("too few to tell"));
    }

    #[test]
    fn auto_epochs_reach_the_step_target_within_the_cap() {
        let cfg = TrainerConfig::default();
        assert_eq!(schedule(960, &cfg), (5, 300));
        assert_eq!(schedule(100, &cfg), (5, 35));
        assert_eq!(schedule(10_000, &cfg), (1, 625));
        let fixed = TrainerConfig {
            epochs: AutoOr::Value(2),
            ..TrainerConfig::default()
        };
        assert_eq!(schedule(160, &fixed), (2, 20));
    }

    #[test]
    fn old_evaluations_still_get_a_verdict() {
        let eval = Manifest::new(
            EVAL_KIND,
            "k",
            json!({}),
            vec![],
            json!({"metrics": {
                "base": {"overall": {"ndcg@10": 0.7, "queries": 10}},
                "tuned": {"overall": {"ndcg@10": 0.8, "queries": 10}},
            }}),
        );
        let v = eval_verdict(&eval);
        assert_eq!((v.outcome.as_str(), v.confident), ("better", false));
    }
}
