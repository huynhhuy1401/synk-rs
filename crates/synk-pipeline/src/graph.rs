//! Previews: which steps are stale, why, and what running them costs.
//!
//! The pipeline is a fixed chain of [`Step`]s. A [`Preview`] marks each one
//! `cached` (its artifact exists for the current inputs), `run`, `blocked`
//! (something only a person can fix), or `skipped` (not needed for the
//! target). It lists everything that needs consent: downloads, each with
//! the name `synk fetch` takes, and document text sent to a server off this
//! machine. Making a preview never generates queries and never sends
//! document text anywhere; it may ask Hugging Face and the Ollama registry
//! for model sizes (sending only model names). `synk train --dry-run` shows
//! one; `train` and `dataset` refuse to start while it lists a download.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use synk_core::{ErrorCode, Result, SourceSpec, SynkError};
use synk_store::Store;

use crate::PipelineConfig;
use crate::artifact::Artifacts;
use crate::catalog::{self, ModelSpec, RerankerSpec};
use crate::dataset::{self, Chunking};
use crate::generator::{self, OpenAiGenerator, QueryGenerator};
use crate::hub::{self, RepoType};
use crate::ops::{Ops, now};
use crate::sidecar::{self, TrainerEnv};
use crate::tokenizer::TokenCounter;
use crate::{sync, train};

/// Rough download sizes of the trainer environment, by backend.
const TRAINER_BYTES_CPU: u64 = 1_000_000_000;
const TRAINER_BYTES_GPU: u64 = 3_500_000_000;
/// Defaults when no speed has been measured on this machine yet.
const DEFAULT_SECONDS_PER_LLM_CALL: f64 = 4.0;
const DEFAULT_SECONDS_PER_STEP_GPU: f64 = 0.4;
const DEFAULT_SECONDS_PER_STEP_CPU: f64 = 2.5;

/// The steps, in the order they run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Step {
    Trainer,
    Models,
    Generator,
    Sync,
    Dataset,
    Train,
    Eval,
}

impl Step {
    pub const ALL: [Step; 7] = [
        Step::Trainer,
        Step::Models,
        Step::Generator,
        Step::Sync,
        Step::Dataset,
        Step::Train,
        Step::Eval,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Trainer => "trainer",
            Self::Models => "models",
            Self::Generator => "generator",
            Self::Sync => "sync",
            Self::Dataset => "dataset",
            Self::Train => "train",
            Self::Eval => "eval",
        }
    }

    pub fn parse(s: &str) -> Result<Self> {
        Self::ALL
            .into_iter()
            .find(|step| step.as_str() == s)
            .ok_or_else(|| {
                SynkError::new(
                    ErrorCode::InvalidArgs,
                    format!(
                        "`{s}` is not a step; steps are: {}",
                        Self::ALL.map(Step::as_str).join(", ")
                    ),
                )
            })
    }

    /// The steps `target` needs, including itself.
    pub fn needed_for(target: Step) -> Vec<Step> {
        match target {
            Step::Trainer => vec![Step::Trainer],
            Step::Models => vec![Step::Trainer, Step::Models],
            Step::Generator => vec![Step::Generator],
            Step::Sync => vec![Step::Sync],
            Step::Dataset => vec![Step::Generator, Step::Sync, Step::Dataset],
            Step::Train | Step::Eval => Step::ALL.to_vec(),
        }
    }
}

/// Everything a preview or a run needs to know about the project.
pub struct Workspace {
    pub root: PathBuf,
    /// `.synk/`.
    pub data_dir: PathBuf,
    /// The effective settings: `synk.toml` plus `--set` overrides.
    pub cfg: PipelineConfig,
    pub overrides: BTreeMap<String, Value>,
    /// Sources with absolute paths.
    pub sources: Vec<SourceSpec>,
    /// `[trainer] python`, resolved.
    pub trainer_python: Option<PathBuf>,
    /// `[trainer] uv`, resolved.
    pub uv: Option<PathBuf>,
}

impl Workspace {
    pub fn artifacts(&self) -> Artifacts {
        Artifacts::new(self.data_dir.join("artifacts"))
    }

    pub fn store_path(&self) -> PathBuf {
        self.data_dir.join("synk.db")
    }

    pub fn ops(&self) -> Ops {
        Ops::new(self.data_dir.clone())
    }

    /// The store, created on first use.
    pub fn open_store(&self) -> Result<Store> {
        Store::open(&self.store_path())
    }

    pub fn config_hash(&self) -> String {
        let text = serde_json::to_string(&self.cfg).expect("config serializes");
        short(&text, 12)
    }

    pub fn model(&self) -> Result<ModelSpec> {
        catalog::resolve(&self.cfg.model.base, &self.root)
    }

    pub fn reranker(&self) -> Result<Option<RerankerSpec>> {
        self.cfg
            .trainer
            .reranker
            .as_deref()
            .map(|r| catalog::resolve_reranker(r, &self.root))
            .transpose()
    }

    pub fn trainer(&self) -> Result<TrainerEnv> {
        TrainerEnv::find(self.trainer_python.as_deref())
    }
}

fn short(text: &str, len: usize) -> String {
    blake3::hash(text.as_bytes()).to_hex()[..len].to_owned()
}

/// What a preview covers and how.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Options {
    pub target: Step,
    /// Run this step even if its artifact exists.
    pub force: Option<Step>,
    /// Build the dataset from at most this many chunks.
    pub sample: Option<usize>,
    /// Ask Hugging Face and the Ollama registry for sizes and versions.
    /// `synk status` turns it off to stay quick.
    #[serde(skip, default = "yes")]
    pub network: bool,
}

fn yes() -> bool {
    true
}

impl Default for Options {
    fn default() -> Self {
        Self {
            target: Step::Eval,
            force: None,
            sample: None,
            network: true,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Action {
    Cached,
    Run,
    Blocked,
    Skipped,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepPlan {
    pub step: Step,
    pub action: Action,
    pub why: Vec<String>,
    #[serde(default, skip_serializing_if = "Value::is_null")]
    pub detail: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub est_seconds: Option<u64>,
    /// `history` (measured on this machine before) or `default`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub estimate_basis: Option<String>,
}

/// Something running the preview does that needs the user's consent.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Consent {
    /// `download` or `egress` (document text sent off this machine).
    pub kind: String,
    pub what: String,
    /// For a download, the name `synk fetch` takes: a catalog ID or
    /// `hf:org/name` with `@` and the first 7 characters of the pinned
    /// commit, `ollama:TAG`, `general-check` or `trainer`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bytes: Option<u64>,
    /// Where it comes from or goes to.
    pub host: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chunks: Option<usize>,
}

/// A Hugging Face repository to download, at a fixed commit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fetch {
    /// The fetch name it belongs to (see [`Consent::name`]).
    pub name: String,
    pub repo: String,
    pub revision: String,
    pub repo_type: String,
    pub ignore_patterns: Vec<String>,
    #[serde(default)]
    pub allow_patterns: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Preview {
    pub v: u32,
    pub created_at: u64,
    pub options: Options,
    pub config_hash: String,
    pub overrides: BTreeMap<String, Value>,
    pub steps: Vec<StepPlan>,
    pub consent: Vec<Consent>,
    /// Why blocked steps can't run, as error objects.
    pub blockers: Vec<SynkError>,
    pub totals: Value,
    /// The base model, with its version pinned.
    pub model: ModelSpec,
    pub fetch: Vec<Fetch>,
    /// The dataset the dataset step would build or reuse, when known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dataset_key: Option<String>,
    /// The training run's artifact key, when the dataset and trainer
    /// exist already.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub run_key: Option<String>,
    /// Advice known before anything runs, e.g. a test too small for a
    /// verdict (`TEST_SET_UNDERPOWERED` with `evidence.predicted`).
    #[serde(default, skip_deserializing)]
    pub advice: Vec<crate::advice::Advice>,
    /// Free space against what the run will write, per drive.
    #[serde(default, skip_deserializing)]
    pub disk: Vec<crate::disk::Volume>,
}

impl Preview {
    pub fn step(&self, step: Step) -> Option<&StepPlan> {
        self.steps.iter().find(|s| s.step == step)
    }

    /// True when nothing needs to run.
    pub fn up_to_date(&self) -> bool {
        self.steps.iter().all(|s| s.action != Action::Run)
    }

    /// The fetch names of everything to download, without repeats.
    pub fn fetch_names(&self) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        for name in self.consent.iter().filter_map(|c| c.name.clone()) {
            if !out.contains(&name) {
                out.push(name);
            }
        }
        out
    }

    /// The downloads, if any.
    pub fn downloads(&self) -> impl Iterator<Item = &Consent> {
        self.consent.iter().filter(|c| c.kind == "download")
    }

    /// Document text sent off this machine, if any.
    pub fn egress(&self) -> impl Iterator<Item = &Consent> {
        self.consent.iter().filter(|c| c.kind == "egress")
    }
}

/// The fetch name of a Hugging Face model: its catalog ID or `hf:org/name`,
/// with `@` and the start of its commit when pinned.
pub fn fetch_name(spec: &ModelSpec) -> String {
    let base = spec
        .catalog_id
        .clone()
        .unwrap_or_else(|| format!("hf:{}", spec.name_or_path));
    match &spec.revision {
        Some(rev) => format!("{base}@{}", &rev[..rev.len().min(7)]),
        None => base,
    }
}

/// The fetch name of the trainer.
pub const TRAINER: &str = "trainer";
/// The fetch name of the general check's public test sets.
pub const GENERAL_CHECK: &str = "general-check";

/// The fetch name of an Ollama model.
pub fn ollama_name(tag: &str) -> String {
    format!("ollama:{tag}")
}

/// Measured speeds on this machine, for estimates.
fn history_path() -> PathBuf {
    sidecar::synk_home().join("throughput.json")
}

fn history() -> Value {
    fs::read_to_string(history_path())
        .ok()
        .and_then(|t| serde_json::from_str(&t).ok())
        .unwrap_or_else(|| json!({}))
}

/// Records a measured speed (seconds per unit) under `key`.
pub fn record_speed(key: &str, seconds_per_unit: f64) {
    if !seconds_per_unit.is_finite() || seconds_per_unit <= 0.0 {
        return;
    }
    let mut h = history();
    h[key] = json!((seconds_per_unit * 1000.0).round() / 1000.0);
    let path = history_path();
    if let Some(dir) = path.parent() {
        let _ = fs::create_dir_all(dir);
    }
    let _ = fs::write(path, h.to_string());
}

pub fn llm_speed_key(cfg: &PipelineConfig) -> String {
    format!("llm:{}", cfg.generator.model)
}

pub fn train_speed_key(model: &ModelSpec, device: &str) -> String {
    format!("train:{}:{device}", model.label())
}

fn estimate(key: &str, default: f64, units: u64) -> (Option<u64>, Option<String>) {
    let (per, basis) = match history()[key].as_f64() {
        Some(s) => (s, "history"),
        None => (default, "default"),
    };
    (
        Some((per * units as f64).round() as u64),
        Some(basis.to_owned()),
    )
}

/// A repository with its pinned revision, its metadata when the Hub
/// answered, and the error when it didn't and nothing is cached.
type Pinned = (
    String,
    Option<String>,
    Option<hub::RepoInfo>,
    Option<SynkError>,
);

/// Pins a Hugging Face model to a commit: the catalog's, the Hub's answer
/// for its branch, or the commit the local cache points at.
pub fn pin(
    spec: &ModelSpec,
    network: bool,
) -> (ModelSpec, Option<hub::RepoInfo>, Option<SynkError>) {
    let mut spec = spec.clone();
    if Path::new(&spec.name_or_path).is_dir() {
        return (spec, None, None);
    }
    let revision = spec.revision.clone().unwrap_or_else(|| "main".to_owned());
    let info = if network {
        Some(hub::info(RepoType::Model, &spec.name_or_path, &revision))
    } else {
        None
    };
    match info {
        Some(Ok(info)) => {
            spec.revision = Some(info.commit.clone());
            (spec, Some(info), None)
        }
        Some(Err(e)) => {
            if let Some(snap) = hub::snapshot(RepoType::Model, &spec.name_or_path, &revision) {
                spec.revision = snap.file_name().map(|n| n.to_string_lossy().into_owned());
                (spec, None, None)
            } else {
                (spec, None, Some(e))
            }
        }
        None => {
            if let Some(snap) = hub::snapshot(RepoType::Model, &spec.name_or_path, &revision) {
                spec.revision = snap.file_name().map(|n| n.to_string_lossy().into_owned());
            }
            (spec, None, None)
        }
    }
}

/// The dataset step's inputs: the chunker and a connected generator.
pub fn dataset_inputs(
    ws: &Workspace,
    store: &Store,
    model: &ModelSpec,
    sample: Option<usize>,
) -> Result<(crate::config::DatasetConfig, Chunking, OpenAiGenerator)> {
    let mut dataset_cfg = ws.cfg.dataset.clone();
    if sample.is_some() {
        dataset_cfg.max_chunks = sample;
    }
    let counter = if dataset_cfg.chunk_chars.is_some() {
        TokenCounter::load(model).ok()
    } else {
        Some(TokenCounter::load(model)?)
    };
    let chunking = Chunking::new(&dataset_cfg, ws.cfg.trainer.max_seq_length, counter)?;
    let mut generator = OpenAiGenerator::new(&ws.cfg.generator)?;
    generator.connect()?;
    // The caller's store: a preview looks inside a transaction it rolls
    // back, and a second connection couldn't write while it's open.
    generator.set_style(
        dataset::style_pool(store)?
            .into_iter()
            .map(|q| q.query)
            .collect(),
    );
    Ok((dataset_cfg, chunking, generator))
}

struct Builder {
    steps: Vec<StepPlan>,
    consent: Vec<Consent>,
    blockers: Vec<SynkError>,
}

impl Builder {
    fn add(
        &mut self,
        step: Step,
        action: Action,
        why: Vec<String>,
        detail: Value,
    ) -> &mut StepPlan {
        self.steps.push(StepPlan {
            step,
            action,
            why,
            detail,
            est_seconds: None,
            estimate_basis: None,
        });
        self.steps.last_mut().expect("just pushed")
    }

    fn block(&mut self, step: Step, err: SynkError) {
        let why = vec![err.message.clone()];
        self.blockers.push(err);
        self.add(step, Action::Blocked, why, Value::Null);
    }
}

/// Makes a preview. Read-only: nothing is downloaded, generated or written.
pub fn make(ws: &Workspace, opts: &Options) -> Result<Preview> {
    let needed = Step::needed_for(opts.target);
    let forced = |step: Step| opts.force == Some(step);
    let mut b = Builder {
        steps: Vec::new(),
        consent: Vec::new(),
        blockers: Vec::new(),
    };
    let artifacts = ws.artifacts();
    let model_unpinned = ws.model()?;
    let (model, model_info, model_err) = pin(&model_unpinned, opts.network);
    let reranker = ws.reranker()?;

    // trainer
    let env = ws.trainer().ok();
    if needed.contains(&Step::Trainer) {
        match (&env, &ws.trainer_python) {
            (Some(env), _) => {
                b.add(Step::Trainer, Action::Cached, vec![], json!({"devices": env.describe["devices"]}));
            }
            (None, Some(python)) => b.block(
                Step::Trainer,
                SynkError::new(
                    ErrorCode::SidecarNotFound,
                    format!(
                        "[trainer] python is {}, which can't run the trainer",
                        python.display()
                    ),
                )
                .with_hint("install torch and sentence-transformers[train] there, or remove trainer.python"),
            ),
            (None, None) => {
                let (backend, why) = sidecar::detect_backend();
                let bytes = if backend == sidecar::Backend::Cpu {
                    TRAINER_BYTES_CPU
                } else {
                    TRAINER_BYTES_GPU
                };
                b.consent.push(Consent {
                    kind: "download".into(),
                    name: Some(TRAINER.into()),
                    what: format!("trainer (Python, PyTorch {})", backend.as_str()),
                    bytes: Some(bytes),
                    host: "pypi.org, download.pytorch.org".into(),
                    chunks: None,
                });
                let s = b.add(
                    Step::Trainer,
                    Action::Run,
                    vec!["not installed".into()],
                    json!({"backend": backend.as_str(), "why_backend": why, "bytes_approximate": true}),
                );
                s.est_seconds = Some(300);
                s.estimate_basis = Some("default".into());
            }
        }
    }

    // models
    let mut fetch = Vec::new();
    if needed.contains(&Step::Models) {
        let mut why = Vec::new();
        let mut bytes_total = 0u64;
        let mut missing_size = false;
        let mut repos: Vec<(String, Pinned)> = vec![(
            fetch_name(&model),
            (
                model.name_or_path.clone(),
                model.revision.clone(),
                model_info,
                model_err,
            ),
        )];
        if let Some(r) = &reranker {
            let spec = ModelSpec {
                name_or_path: r.name_or_path.clone(),
                revision: r.revision.clone(),
                query_prompt: None,
                document_prompt: None,
                catalog_id: r.catalog_id.clone(),
            };
            let (pinned, info, err) = pin(&spec, opts.network);
            repos.push((
                fetch_name(&pinned),
                (pinned.name_or_path, pinned.revision, info, err),
            ));
        }
        let mut blocked = None;
        for (name, (repo, revision, info, err)) in repos {
            if Path::new(&repo).is_dir() {
                continue;
            }
            let rev = revision.clone().unwrap_or_else(|| "main".into());
            if hub::snapshot(RepoType::Model, &repo, &rev).is_some() && !forced(Step::Models) {
                continue;
            }
            if let Some(err) = err {
                blocked = Some(err);
                break;
            }
            let has_safetensors = info
                .as_ref()
                .is_some_and(|i| i.files.iter().any(|(p, _)| p.ends_with(".safetensors")));
            match info.as_ref().map(hub::RepoInfo::bytes) {
                Some(bytes) => bytes_total += bytes,
                None => missing_size = true,
            }
            why.push(format!(
                "{repo}@{} not downloaded",
                &rev[..rev.len().min(7)]
            ));
            b.consent.push(Consent {
                kind: "download".into(),
                name: Some(name.clone()),
                what: repo.clone(),
                bytes: info.as_ref().map(hub::RepoInfo::bytes),
                host: hub::endpoint().trim_start_matches("https://").into(),
                chunks: None,
            });
            fetch.push(Fetch {
                name,
                repo,
                revision: rev,
                repo_type: "model".into(),
                ignore_patterns: hub::ignore_patterns(has_safetensors || info.is_none())
                    .into_iter()
                    .map(str::to_owned)
                    .collect(),
                allow_patterns: Vec::new(),
            });
        }
        // The general check's public sets.
        if needed.contains(&Step::Eval) && ws.cfg.eval.general_check && blocked.is_none() {
            for set in train::GENERAL_SETS {
                if hub::snapshot(RepoType::Dataset, set.repo, set.revision).is_some() {
                    continue;
                }
                let bytes = if opts.network {
                    hub::info(RepoType::Dataset, set.repo, set.revision)
                        .ok()
                        .map(|i| {
                            i.files
                                .iter()
                                .filter(|(p, _)| {
                                    set.allow_patterns
                                        .iter()
                                        .any(|pat| p.starts_with(pat.trim_end_matches('*')))
                                })
                                .map(|(_, b)| b)
                                .sum::<u64>()
                        })
                } else {
                    None
                };
                match bytes {
                    Some(b) => bytes_total += b,
                    None => missing_size = true,
                }
                why.push(format!("general check set `{}` not downloaded", set.name));
                b.consent.push(Consent {
                    kind: "download".into(),
                    name: Some(GENERAL_CHECK.into()),
                    what: format!("{} (general check)", set.repo),
                    bytes,
                    host: hub::endpoint().trim_start_matches("https://").into(),
                    chunks: None,
                });
                fetch.push(Fetch {
                    name: GENERAL_CHECK.into(),
                    repo: set.repo.to_owned(),
                    revision: set.revision.to_owned(),
                    repo_type: "dataset".into(),
                    ignore_patterns: Vec::new(),
                    allow_patterns: set.allow_patterns.iter().map(|s| (*s).to_owned()).collect(),
                });
            }
        }
        match blocked {
            Some(err) => b.block(Step::Models, err),
            None if fetch.is_empty() => {
                b.add(
                    Step::Models,
                    Action::Cached,
                    vec![],
                    json!({"base": model.label()}),
                );
            }
            None => {
                b.add(
                    Step::Models,
                    Action::Run,
                    why,
                    json!({"base": model.label(), "bytes": (!missing_size).then_some(bytes_total)}),
                );
            }
        }
    }

    // sync
    let mut store = ws.open_store()?;
    let preview = sync::preview(&ws.sources, &store)?;
    let sync_changes = preview.changed();
    let mut checks = Vec::new();
    for w in &preview.warnings {
        checks.push(serde_json::to_value(w).expect("error serializes"));
    }
    let sync_detail = json!({
        "new": preview.totals.added,
        "changed": preview.totals.updated,
        "deleted": preview.totals.deleted,
        "unchanged": preview.totals.unchanged,
        "skipped": preview.totals.skipped,
        "removed_sources": preview.removed_sources,
    });

    // dataset (decides whether the generator is needed)
    let mut dataset_plan: Option<(Action, Vec<String>, Value, u64)> = None;
    let mut dataset_key = None;
    let mut generator_error = None;
    let mut advice = Vec::new();
    // Test documents, known from the documents alone: also on a first run,
    // before the LLM is there to plan the rest of the dataset.
    let mut test_documents = None;
    let mut style_change = false;
    // The documents' text, for the dataset's disk estimate.
    let mut text_bytes = 0;
    if needed.contains(&Step::Dataset) {
        // The dataset as it will be after the sync: sync inside a savepoint,
        // look, and roll back. Nothing is saved.
        let looked = store.dry_run(|store| {
            sync::run(&ws.sources, store)?;
            let live = store.live_documents()?;
            if live.len() < 2 {
                return Ok(None);
            }
            style_change = dataset::style_examples_change(store)?;
            text_bytes = live.iter().map(|d| d.text.len() as u64).sum();
            test_documents = dataset::test_document_ids(store, &live, ws.cfg.dataset.test_percent)
                .ok()
                .map(|t| dataset::sampled_test_documents(t.len(), opts.sample));
            Ok(Some(match dataset_inputs(ws, store, &model, opts.sample) {
                Err(e) => Err(e),
                Ok((dataset_cfg, chunking, generator)) => {
                    let prepared = dataset::prepare(
                        store,
                        &dataset_cfg,
                        ws.cfg.generator.queries_per_chunk,
                        &chunking,
                        &generator,
                    )?;
                    let p = dataset::plan(store, &artifacts, &prepared, &generator, 0)?;
                    Ok((p, prepared.key().to_owned(), generator.key()))
                }
            }))
        })?;
        match looked {
            None => {
                dataset_plan = Some((
                    Action::Blocked,
                    vec!["needs at least 2 documents".into()],
                    Value::Null,
                    0,
                ));
            }
            Some(Err(e)) => generator_error = Some(e),
            Some(Ok((p, key, generator_key))) => {
                let calls = p["llm_calls"].as_u64().unwrap_or(0);
                let detail = json!({
                    "chunks": p["chunks"],
                    "train_chunks": p["train_chunks"],
                    "test_chunks": p["test_chunks"],
                    "train_documents": p["train_documents"],
                    "test_documents": p["test_documents"],
                    "llm_calls": calls,
                    "chunks_with_queries": p["cached_chunks"],
                    "generator": ws.cfg.generator.model,
                    "style_examples_change": style_change,
                });
                let existing = p["existing"] == true;
                let (action, mut why) = if forced(Step::Dataset) {
                    (Action::Run, vec!["forced".to_owned()])
                } else if existing {
                    (Action::Cached, vec![])
                } else if sync_changes {
                    (Action::Run, vec!["documents changed".to_owned()])
                } else {
                    (
                        Action::Run,
                        vec!["settings changed since the last dataset".to_owned()],
                    )
                };
                if style_change && action == Action::Run {
                    why.push(format!(
                        "style examples are chosen from your real searches, so the LLM is asked again for every chunk ({calls} calls)"
                    ));
                }
                let _ = generator_key;
                if let Some(n) = p["test_documents"].as_u64() {
                    test_documents = Some(n as usize);
                }
                dataset_key = Some(key);
                let calls = if action == Action::Cached { 0 } else { calls };
                dataset_plan = Some((action, why, detail, calls));
            }
        }
    }

    if let Some(n) = test_documents {
        advice.extend(crate::advice::predicted_power(
            n,
            last_document_sd(&artifacts),
        ));
    }

    // generator
    if needed.contains(&Step::Generator) {
        let wants_queries = dataset_plan
            .as_ref()
            .is_some_and(|(a, _, _, calls)| *a == Action::Run && *calls > 0)
            || (dataset_plan.is_none() && generator_error.is_some())
            || opts.target == Step::Generator;
        let probe = generator::probe(&ws.cfg.generator);
        let remote = probe.reachable.is_none();
        if !wants_queries {
            b.add(
                Step::Generator,
                Action::Skipped,
                vec!["no queries to write".into()],
                Value::Null,
            );
        } else if remote && !ws.cfg.generator.allow_remote {
            b.block(
                Step::Generator,
                OpenAiGenerator::new(&ws.cfg.generator)
                    .err()
                    .unwrap_or_else(|| SynkError::new(ErrorCode::RemoteNotAllowed, "remote")),
            );
        } else if remote {
            b.add(
                Step::Generator,
                Action::Cached,
                vec!["remote; not checked".into()],
                json!({"model": ws.cfg.generator.model}),
            );
        } else {
            match (probe.reachable, probe.model_ready) {
                (Some(false), _) => b.block(
                    Step::Generator,
                    SynkError::new(
                        ErrorCode::GeneratorUnavailable,
                        format!("can't reach the LLM server: {}", probe.message),
                    )
                    .with_action(synk_core::ErrorAction::run(
                        "ollama serve",
                        synk_core::Applicability::Review,
                        "start Ollama",
                    ))
                    .with_hint("or point generator.base_url at a running OpenAI-compatible server"),
                ),
                (_, Some(false)) if probe.ollama => {
                    let bytes = if opts.network {
                        generator::ollama_size(&ws.cfg.generator.model)
                    } else {
                        None
                    };
                    b.consent.push(Consent {
                        kind: "download".into(),
                        name: Some(ollama_name(&ws.cfg.generator.model)),
                        what: ws.cfg.generator.model.clone(),
                        bytes,
                        host: "ollama.com (through your Ollama)".into(),
                        chunks: None,
                    });
                    b.add(
                        Step::Generator,
                        Action::Run,
                        vec![format!("{} isn't downloaded in Ollama", ws.cfg.generator.model)],
                        json!({"model": ws.cfg.generator.model, "bytes": bytes}),
                    );
                }
                (_, Some(false)) => b.block(
                    Step::Generator,
                    SynkError::new(
                        ErrorCode::GeneratorUnavailable,
                        probe.message.clone(),
                    )
                    .with_hint("load that model in your LLM server, or set generator.model to one it has"),
                ),
                _ => {
                    b.add(Step::Generator, Action::Cached, vec![], json!({"model": ws.cfg.generator.model}));
                }
            }
        }
        if let Some(err) = generator_error.take() {
            // The generator problem was already reported above when it is
            // about the server; anything else blocks the dataset.
            if !matches!(
                err.code,
                ErrorCode::GeneratorUnavailable | ErrorCode::RemoteNotAllowed
            ) {
                dataset_plan = Some((Action::Blocked, vec![err.message.clone()], Value::Null, 0));
                b.blockers.push(err);
            } else if b.step_blocked(Step::Generator) {
                dataset_plan = Some((
                    Action::Blocked,
                    vec!["the LLM server isn't ready".into()],
                    Value::Null,
                    0,
                ));
            } else {
                // The model is only being downloaded: the dataset runs after.
                dataset_plan = Some((
                    Action::Run,
                    vec!["the LLM model is downloaded first".into()],
                    Value::Null,
                    0,
                ));
            }
        }
    }

    if needed.contains(&Step::Sync) {
        let action = if sync_changes || forced(Step::Sync) {
            Action::Run
        } else {
            Action::Cached
        };
        let why = if sync_changes {
            vec![format!(
                "{} new, {} changed, {} deleted",
                preview.totals.added, preview.totals.updated, preview.totals.deleted
            )]
        } else {
            vec![]
        };
        b.add(Step::Sync, action, why, sync_detail);
    }

    if let Some((action, why, detail, calls)) = dataset_plan {
        if action == Action::Blocked && !why.is_empty() && b.blockers.is_empty() {
            b.blockers.push(
                SynkError::new(ErrorCode::DatasetTooSmall, why[0].clone())
                    .with_hint("add documents: synk source add <path>"),
            );
        }
        let step = b.add(Step::Dataset, action, why, detail);
        if action == Action::Run && calls > 0 {
            let (s, basis) = estimate(&llm_speed_key(&ws.cfg), DEFAULT_SECONDS_PER_LLM_CALL, calls);
            step.est_seconds = s;
            step.estimate_basis = basis;
        }
        if action == Action::Run && calls > 0 && !generator::is_local(&ws.cfg.generator.base_url) {
            b.consent.push(Consent {
                kind: "egress".into(),
                name: None,
                what: "document chunks, to write queries".into(),
                bytes: None,
                host: ws.cfg.generator.base_url.clone(),
                chunks: Some(calls as usize),
            });
        }
    }

    // train and eval
    let mut run_key = None;
    if needed.contains(&Step::Train) {
        let dataset_ready = dataset_key
            .as_deref()
            .and_then(|k| artifacts.manifest(dataset::KIND, k).ok().flatten());
        let upstream_runs = b.steps.iter().any(|s| {
            matches!(s.step, Step::Dataset | Step::Models | Step::Trainer)
                && s.action == Action::Run
        });
        let blocked_upstream = b.steps.iter().any(|s| s.action == Action::Blocked);
        let device = ws
            .cfg
            .trainer
            .device
            .clone()
            .or_else(|| {
                env.as_ref()
                    .and_then(|e| e.describe["devices"][0].as_str().map(str::to_owned))
            })
            .unwrap_or_else(|| "cpu".to_owned());
        let (train_action, train_why, train_detail, steps_count, eval_cached) = match (
            &dataset_ready,
            &env,
        ) {
            _ if blocked_upstream => (
                Action::Blocked,
                vec!["an earlier step is blocked".to_owned()],
                Value::Null,
                0,
                false,
            ),
            (Some(ds), Some(env)) if !upstream_runs => {
                let p = train::plan(
                    &artifacts,
                    ds,
                    &model,
                    reranker.as_ref(),
                    &ws.cfg.trainer,
                    Some(env),
                    None,
                )?;
                let key = p["run"].as_str().unwrap_or_default().to_owned();
                run_key = Some(key.clone());
                let run_key = key;
                let existing = p["existing"] == true;
                let (action, why) = if forced(Step::Train) {
                    (Action::Run, vec!["forced".to_owned()])
                } else if existing {
                    (Action::Cached, vec![])
                } else if p["interrupted"] == true {
                    (
                        Action::Run,
                        vec!["continues an interrupted run from its checkpoint".to_owned()],
                    )
                } else {
                    (
                        Action::Run,
                        vec!["no run with these settings yet".to_owned()],
                    )
                };
                let eval_cached = existing
                    && !forced(Step::Train)
                    && train::eval_cached(
                        &artifacts,
                        &run_key,
                        &ds.key,
                        &ws.cfg.trainer,
                        ws.cfg.eval.general_check,
                    )?;
                {
                    let steps = p_steps(&p);
                    (action, why, p, steps, eval_cached)
                }
            }
            _ => {
                let examples = dataset_ready
                    .as_ref()
                    .and_then(|d| d.summary["train_queries"].as_u64())
                    .unwrap_or(0);
                let (epochs, steps) = train::schedule(examples.max(1), &ws.cfg.trainer);
                (
                    Action::Run,
                    vec!["an earlier step runs first".to_owned()],
                    json!({"base_model": model.label(), "epochs": epochs, "steps": steps, "estimate": "before the dataset exists"}),
                    steps,
                    false,
                )
            }
        };
        let step = b.add(Step::Train, train_action, train_why, train_detail);
        if train_action == Action::Run {
            let default = if device == "cpu" {
                DEFAULT_SECONDS_PER_STEP_CPU
            } else {
                DEFAULT_SECONDS_PER_STEP_GPU
            };
            let (s, basis) = estimate(&train_speed_key(&model, &device), default, steps_count);
            step.est_seconds = s;
            step.estimate_basis = basis;
        }
        if needed.contains(&Step::Eval) {
            let action = match train_action {
                Action::Blocked => Action::Blocked,
                Action::Cached if eval_cached && !forced(Step::Eval) => Action::Cached,
                _ => Action::Run,
            };
            let why = match action {
                Action::Run if train_action == Action::Run => {
                    vec!["the model is trained first".to_owned()]
                }
                Action::Run => vec!["not tested yet".to_owned()],
                _ => vec![],
            };
            let step = b.add(Step::Eval, action, why, Value::Null);
            if action == Action::Run {
                step.est_seconds = Some(60);
                step.estimate_basis = Some("default".into());
            }
        }
    }

    // Steps not needed for the target.
    for step in Step::ALL {
        if !b.steps.iter().any(|s| s.step == step) {
            b.add(
                step,
                Action::Skipped,
                vec![format!("not needed for --target {}", opts.target.as_str())],
                Value::Null,
            );
        }
    }
    b.steps.sort_by_key(|s| s.step);

    let disk = disk_needs(ws, &model, &b.consent, &b.steps, text_bytes);
    for volume in &disk {
        if volume.short() && !crate::disk::warn_only() {
            b.blockers.push(insufficient_disk(volume));
        } else if volume.short() || volume.tight() {
            advice.push(crate::advice::disk_tight(volume));
        }
    }

    let download_bytes: u64 = b.consent.iter().filter_map(|c| c.bytes).sum();
    let est_seconds: u64 = b.steps.iter().filter_map(|s| s.est_seconds).sum();
    let preview = Preview {
        v: 1,
        created_at: now(),
        options: opts.clone(),
        config_hash: ws.config_hash(),
        overrides: ws.overrides.clone(),
        steps: b.steps,
        consent: b.consent,
        blockers: b.blockers,
        totals: json!({
            "download_bytes": download_bytes,
            "est_seconds": est_seconds,
            "warnings": checks,
        }),
        model,
        fetch,
        dataset_key,
        run_key,
        advice,
        disk,
    };
    Ok(preview)
}

/// What the run will write, where: downloads (the trainer grows when
/// installed) and, when training runs, its checkpoints and models.
fn disk_needs(
    ws: &Workspace,
    model: &ModelSpec,
    consent: &[Consent],
    steps: &[StepPlan],
    text_bytes: u64,
) -> Vec<crate::disk::Volume> {
    use crate::disk::{self, Need};
    let mut needs = Vec::new();
    let local_llm = crate::generator::is_local(&ws.cfg.generator.base_url);
    for c in consent.iter().filter(|c| c.kind == "download") {
        let (Some(name), Some(bytes)) = (c.name.as_deref(), c.bytes) else {
            continue;
        };
        let place = if name == "trainer" {
            Some((sidecar::synk_home(), bytes * disk::TRAINER_UNPACKED))
        } else if name.starts_with("ollama:") {
            local_llm
                .then(disk::ollama_models_dir)
                .flatten()
                .map(|d| (d, bytes))
        } else {
            hub::cache_dir().map(|d| (d, bytes))
        };
        if let Some((path, bytes)) = place {
            needs.push(Need {
                what: c.what.clone(),
                path,
                bytes,
            });
        }
    }
    let runs = |step| {
        steps
            .iter()
            .any(|s| s.step == step && s.action == Action::Run)
    };
    if runs(Step::Dataset) {
        needs.push(Need {
            what: "the dataset".to_owned(),
            path: ws.data_dir.join("artifacts"),
            bytes: text_bytes * disk::DATASET_PER_TEXT_BYTE + disk::DATASET_FIXED_BYTES,
        });
    }
    let trains = runs(Step::Train);
    if trains {
        needs.extend(training_need(ws, model));
    }
    disk::check(needs)
}

/// What training writes: checkpoints and models, from the model's parameter
/// count: the catalog's, or a local folder's weight files at 4 bytes per
/// parameter (unknown for other custom models).
pub fn training_need(ws: &Workspace, model: &ModelSpec) -> Option<crate::disk::Need> {
    let catalog = model
        .catalog_id
        .as_deref()
        .and_then(|id| catalog::models().into_iter().find(|m| m.id == id))
        .and_then(|m| crate::disk::parse_parameters(&m.parameters));
    let parameters = catalog.or_else(|| {
        let weights: u64 = fs::read_dir(&model.name_or_path)
            .ok()?
            .flatten()
            .filter(|e| {
                let name = e.file_name().to_string_lossy().to_lowercase();
                name.ends_with(".safetensors") || name.ends_with(".bin")
            })
            .filter_map(|e| e.metadata().ok())
            .map(|m| m.len())
            .sum();
        (weights > 0).then_some(weights / 4)
    })?;
    Some(crate::disk::Need {
        what: "training checkpoints and models".to_owned(),
        path: ws.data_dir.join("artifacts"),
        bytes: crate::disk::training_bytes(parameters, &ws.cfg.trainer.finetune),
    })
}

/// Before training starts (the dataset step may have taken hours): fails
/// with `INSUFFICIENT_DISK` if its checkpoints won't fit.
pub fn check_training_space(ws: &Workspace, model: &ModelSpec) -> Result<()> {
    if crate::disk::warn_only() {
        return Ok(());
    }
    let volumes = crate::disk::check(training_need(ws, model).into_iter().collect());
    match volumes.iter().find(|v| v.short()) {
        Some(v) => Err(insufficient_disk(v)),
        None => Ok(()),
    }
}

fn insufficient_disk(volume: &crate::disk::Volume) -> SynkError {
    use crate::disk::gb;
    let what: Vec<String> = volume
        .needs
        .iter()
        .map(|n| format!("{} ({})", n.what, gb(n.bytes)))
        .collect();
    SynkError::new(
        ErrorCode::InsufficientDisk,
        format!(
            "not enough disk space on the drive of {}: the run needs about {} ({}) and {} is free",
            volume.path.display(),
            gb(volume.need_bytes),
            what.join(", "),
            gb(volume.free_bytes)
        ),
    )
    .with_hint(format!(
        "free about {} there, or set SYNK_DISK_CHECK=warn if this estimate is wrong for you",
        gb(volume.need_bytes - volume.free_bytes)
    ))
    .with_details(serde_json::to_value(volume).expect("volume serializes"))
}

/// The spread of per-document changes measured by the project's latest
/// evaluation, to predict the next test's power.
fn last_document_sd(artifacts: &Artifacts) -> Option<f64> {
    let eval = artifacts.latest(train::EVAL_KIND).ok()??;
    let v = train::eval_verdict(&eval);
    let n = usize::try_from(v.test_documents).ok()?;
    crate::stats::sd_from_min_detectable(n, v.min_detectable_delta?)
}

fn p_steps(p: &Value) -> u64 {
    p["steps"].as_u64().unwrap_or(0)
}

impl Builder {
    fn step_blocked(&self, step: Step) -> bool {
        self.steps
            .iter()
            .any(|s| s.step == step && s.action == Action::Blocked)
    }
}
