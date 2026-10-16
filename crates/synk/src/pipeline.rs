//! Commands after sync: setup, models, dataset, train, export.

use std::fmt::Write as _;
use std::io::{IsTerminal as _, Write as _};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use serde_json::{Value, json};
use synk_core::Result;
use synk_pipeline::advice::{self};
use synk_pipeline::artifact::Manifest;
use synk_pipeline::ids::{self, Kind};
use synk_pipeline::sidecar::{self, Backend, TrainerEnv};
use synk_pipeline::{catalog, dataset, train};

use crate::config::Project;
use crate::human;
use crate::output::{Output, verbose};

/// How long-running commands report progress.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum ProgressMode {
    /// A live line on stderr, unless --json is set
    Auto,
    /// One JSON event per line on stderr
    Jsonl,
    /// Nothing
    None,
}

/// Version of the `--progress jsonl` event format.
const PROGRESS_VERSION: u32 = 1;

/// How often plain progress (stderr not a terminal) prints a line while a
/// stage runs, so a watching agent or log sees it isn't hung.
const PLAIN_EVERY: Duration = Duration::from_secs(30);

/// Progress on stderr. stdout stays reserved for the result.
pub struct Progress {
    mode: Mode,
    line_open: bool,
    /// Plain mode: the stage last printed and when.
    last: Option<(String, Instant)>,
}

#[derive(PartialEq)]
enum Mode {
    /// A line redrawn in place, for a person at a terminal.
    Live,
    /// Whole lines only: stderr is a pipe, a file, or a dumb terminal.
    Plain,
    Jsonl,
    Off,
}

/// Whether stderr can redraw a line in place.
fn live_terminal() -> bool {
    std::io::stderr().is_terminal() && std::env::var("TERM").map_or(true, |t| t != "dumb")
}

impl Progress {
    pub fn new(mode: ProgressMode, json: bool) -> Self {
        let mode = match mode {
            ProgressMode::Auto if !json && live_terminal() => Mode::Live,
            ProgressMode::Auto if !json => Mode::Plain,
            ProgressMode::Jsonl => Mode::Jsonl,
            _ => Mode::Off,
        };
        Self::with_mode(mode)
    }

    fn with_mode(mode: Mode) -> Self {
        Self {
            mode,
            line_open: false,
            last: None,
        }
    }

    /// Whether anything is shown at all.
    pub fn is_off(&self) -> bool {
        self.mode == Mode::Off
    }

    /// `stage` is a sidecar stage such as `mine:queries`, or `generate`.
    fn update(&mut self, stage: &str, step: u64, total: u64) {
        match self.mode {
            Mode::Live => {
                eprint!("\r{}: {}   ", stage_label(stage), count(step, total));
                let _ = std::io::stderr().flush();
                self.line_open = true;
            }
            Mode::Plain => {
                let due = match &self.last {
                    Some((last, at)) => last != stage || at.elapsed() >= PLAIN_EVERY,
                    None => true,
                };
                if due || (step == total && total > 0) {
                    eprintln!("{}: {}", stage_label(stage), count(step, total));
                    self.last = Some((stage.to_owned(), Instant::now()));
                }
            }
            Mode::Jsonl => jsonl(&json!({
                "v": PROGRESS_VERSION,
                "event": "progress",
                "stage": stage.split(':').next().unwrap_or(stage),
                "detail": stage,
                "step": step,
                "total": total,
            })),
            Mode::Off => {}
        }
    }

    pub fn line(&mut self, message: &str) {
        match self.mode {
            Mode::Live | Mode::Plain => {
                self.end();
                eprintln!("{message}");
            }
            Mode::Jsonl => {
                jsonl(&json!({"v": PROGRESS_VERSION, "event": "log", "message": message}))
            }
            Mode::Off => {}
        }
    }

    pub fn end(&mut self) {
        if self.line_open {
            eprintln!();
            self.line_open = false;
        }
    }

    /// Shows a sidecar event.
    pub fn event(&mut self, event: &Value) {
        match event["event"].as_str() {
            Some("progress") => self.update(
                event["stage"].as_str().unwrap_or_default(),
                event["step"].as_u64().unwrap_or(0),
                event["total"].as_u64().unwrap_or(0),
            ),
            Some("log") => self.line(event["message"].as_str().unwrap_or_default()),
            _ => {}
        }
    }
}

/// `12/40 (30%)`.
fn count(step: u64, total: u64) -> String {
    if total == 0 {
        return step.to_string();
    }
    format!("{step}/{total} ({}%)", step * 100 / total)
}

fn jsonl(event: &Value) {
    let mut err = std::io::stderr().lock();
    let _ = writeln!(err, "{event}");
    let _ = err.flush();
}

fn stage_label(stage: &str) -> String {
    match stage.split(':').next().unwrap_or(stage) {
        "generate" => "Writing queries (chunks)".to_owned(),
        "download" => "Downloading (bytes)".to_owned(),
        "general" => "General check".to_owned(),
        "export" => "Exporting".to_owned(),
        "mine" => "Filtering queries and mining hard negatives".to_owned(),
        "rerank" => "Checking hard negatives with the reranker".to_owned(),
        "train" => "Training (steps)".to_owned(),
        "dev" => "Scoring the epoch on dev documents".to_owned(),
        "eval" => {
            let model = stage.split(':').nth(1).unwrap_or_default();
            format!("Evaluating {model} model")
        }
        _ => stage.to_owned(),
    }
}

pub fn setup(
    project: Result<Project>,
    backend: &str,
    force: bool,
    mut progress: Progress,
) -> Result<Output> {
    let backend = match backend {
        "auto" => None,
        other => Some(Backend::parse(other)?),
    };
    let uv = project
        .ok()
        .and_then(|p| p.pipeline().trainer.uv.as_ref().map(|uv| p.root.join(uv)));
    let report = sidecar::setup(backend, force, uv.as_deref(), &mut |m| progress.line(m))?;
    let env = &report.env;
    let backend = env.backend.map_or("custom", Backend::as_str);
    let devices: Vec<&str> = env.describe["devices"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .collect();

    let mut warnings = Vec::new();
    if backend == "cuda" && !devices.contains(&"cuda") {
        warnings.push(
            "PyTorch's CUDA build is installed but sees no usable GPU, so training will run on the CPU. PyTorch's CUDA 13.0 build needs NVIDIA driver 580 or newer.".to_owned(),
        );
    }
    let next = "synk train --dry-run";
    let mut human = format!(
        "Trainer {} ({backend}: {})\n  trains on: {}",
        if report.reused {
            "already installed"
        } else {
            "installed"
        },
        report.reason,
        devices.join(", "),
    );
    if let Some(gpu) = env.describe["gpu"].as_str() {
        let _ = write!(human, " ({gpu})");
    }
    if verbose() {
        let _ = write!(
            human,
            "\n  python: {}\n  torch {}, sentence-transformers {}",
            env.python.display(),
            env.describe["torch"].as_str().unwrap_or("?"),
            env.describe["sentence_transformers"]
                .as_str()
                .unwrap_or("?"),
        );
    }
    for w in &warnings {
        let _ = write!(human, "\nwarning: {w}");
    }
    let _ = write!(human, "\nNext: {next}");
    Ok(Output {
        human,
        data: json!({
            "backend": backend,
            "reason": report.reason,
            "reused": report.reused,
            "python": env.python.display().to_string(),
            "environment": env.describe,
            "warnings": warnings,
            "next": next,
        }),
    })
}

pub fn models(project: Result<Project>) -> Result<Output> {
    let configured = project
        .ok()
        .map(|p| p.pipeline().model.base.clone())
        .unwrap_or_else(|| catalog::DEFAULT_MODEL.to_owned());
    let models = catalog::models();
    let mut human = String::new();
    for m in &models {
        let marker = if m.id == configured { "*" } else { " " };
        let _ = writeln!(
            human,
            "{marker} {:<24} {:>5} params  {:>4} dims  {}\n    {}",
            m.id, m.parameters, m.dims, m.hf, m.notes
        );
    }
    let _ = write!(
        human,
        "Configured: {configured}\nOther models: set [model] base = \"hf:<org>/<name>\" or a folder path (custom models are not tested by synk)\n\nRerankers (optional, [trainer] reranker = \"<id>\"):"
    );
    let rerankers = catalog::rerankers();
    for r in &rerankers {
        let _ = write!(
            human,
            "\n  {:<24} {:>5} params  {}\n    {}",
            r.id, r.parameters, r.hf, r.notes
        );
    }
    Ok(Output {
        human,
        data: json!({ "models": models, "configured": configured, "rerankers": rerankers }),
    })
}

pub fn show_model(project: Result<Project>, model: &str) -> Result<Output> {
    let root = project
        .as_ref()
        .map_or_else(|_| PathBuf::from("."), |p| p.root.clone());
    let configured = project
        .as_ref()
        .map(|p| p.pipeline().model.base.clone())
        .unwrap_or_else(|_| catalog::DEFAULT_MODEL.to_owned());
    if let Some(r) = catalog::rerankers().into_iter().find(|r| r.id == model) {
        let human = format!(
            "{} (reranker, tested by synk)\n  {} @ {}\n  {} parameters; languages: {}\n  {}",
            r.id, r.hf, r.revision, r.parameters, r.languages, r.notes
        );
        return Ok(Output {
            human,
            data: json!({ "kind": "reranker", "catalog": true, "model": r }),
        });
    }
    let spec = catalog::resolve(model, &root)?;
    let entry = spec
        .catalog_id
        .as_deref()
        .and_then(|id| catalog::models().into_iter().find(|m| m.id == id));
    let human = match &entry {
        Some(m) => format!(
            "{} (base model, tested by synk){}\n  {} @ {}\n  {} parameters, {} dimensions; languages: {}\n  {}",
            m.id,
            if m.id == configured {
                "; configured"
            } else {
                ""
            },
            m.hf,
            m.revision,
            m.parameters,
            m.dims,
            m.languages,
            m.notes
        ),
        None => format!(
            "{} (custom base model: synk trains it but hasn't tested it){}",
            spec.label(),
            spec.revision
                .as_deref()
                .map_or_else(String::new, |r| format!("\n  revision: {r}")),
        ),
    };
    Ok(Output {
        human,
        data: json!({
            "kind": "base",
            "catalog": entry.is_some(),
            "configured": spec.catalog_id.as_deref() == Some(configured.as_str()) || model == configured,
            "model": entry,
            "spec": spec,
        }),
    })
}

pub fn metrics_tables(eval: &Value) -> String {
    let mut out = metrics_table(&eval["metrics"]);
    if eval["real_metrics"].is_object() {
        out.push_str("\n\nReal queries (all documents):\n");
        out.push_str(&metrics_table(&eval["real_metrics"]));
    }
    out
}

fn metrics_table(metrics: &Value) -> String {
    let columns = ["recall@1", "recall@5", "recall@10", "mrr@10", "ndcg@10"];
    let mut out = format!("{:<20}", "");
    for c in columns {
        let _ = write!(out, "{c:>10}");
    }
    let mut rows = vec![(
        "overall".to_owned(),
        &metrics["base"]["overall"],
        &metrics["tuned"]["overall"],
    )];
    if let Some(langs) = metrics["base"]["by_lang"].as_object()
        && langs.len() > 1
    {
        for lang in langs.keys() {
            rows.push((
                format!("lang {lang}"),
                &metrics["base"]["by_lang"][lang],
                &metrics["tuned"]["by_lang"][lang],
            ));
        }
    }
    for (name, base, tuned) in rows {
        for (label, m) in [("base", base), ("tuned", tuned)] {
            let _ = write!(out, "\n{:<20}", format!("{name} {label}"));
            for c in columns {
                let _ = write!(out, "{:>10.3}", m[c].as_f64().unwrap_or(0.0));
            }
        }
    }
    out
}

pub struct ExportArgs {
    pub format: String,
    pub quant: Option<String>,
    pub ollama: Option<String>,
    pub force: bool,
}

pub fn export(
    project: &Project,
    out: &Path,
    run: Option<&str>,
    args: &ExportArgs,
    mut progress: Progress,
) -> Result<Output> {
    let artifacts = project.artifacts();
    let manifest: Manifest = synk_pipeline::report::find_run(&artifacts, run)?;
    let run_id = ids::id(&artifacts, Kind::Run, &manifest)?;
    let out: PathBuf = std::env::current_dir()
        .map(|cwd| cwd.join(out))
        .unwrap_or_else(|_| out.to_path_buf());
    let cfg = project.pipeline();
    let base = cfg
        .generator
        .base_url
        .trim_end_matches('/')
        .trim_end_matches("/v1");
    let ollama_url = if synk_pipeline::generator::is_local(base) {
        base.to_owned()
    } else {
        "http://localhost:11434".to_owned()
    };
    let request = synk_pipeline::export::Request {
        format: synk_pipeline::export::Format::parse(&args.format)?,
        quant: args.quant.clone(),
        ollama: args.ollama.clone(),
        force: args.force,
        ollama_url,
    };
    let exported = synk_pipeline::export::export(
        &artifacts,
        &manifest,
        &out,
        &request,
        || TrainerEnv::find(project.trainer_python().as_deref()),
        &mut |e| progress.event(e),
    );
    progress.end();
    let exported = exported?;
    let shown = crate::output::show_path(&exported.path);
    let mut human = format!(
        "Exported {run_id} as {} to {shown}",
        request.format.as_str()
    );
    match exported.check["status"].as_str() {
        Some("passed") => {
            if let Some(c) = exported.check["min_cosine"].as_f64() {
                let _ = write!(
                    human,
                    "
Check: passed (vectors match the original, cosine {c:.4})"
                );
            }
        }
        _ => {
            let _ = write!(
                human,
                "
Check: not done ({})",
                exported.check["reason"].as_str().unwrap_or("")
            );
        }
    }
    if let Some(o) = &exported.ollama {
        let _ = write!(
            human,
            "
Ollama model: {}",
            o["name"].as_str().unwrap_or("")
        );
    }
    for (k, v) in exported.usage.as_object().into_iter().flatten() {
        if let Some(text) = v.as_str().filter(|t| !t.is_empty()) {
            let _ = write!(
                human,
                "
{k}: {text}"
            );
        }
    }
    human.push_str(&human::advice(&exported.advice));
    Ok(Output {
        human,
        data: json!({
            "run": run_id,
            "key": manifest.key,
            "path": exported.path.display().to_string(),
            "format": request.format.as_str(),
            "quant": request.quant,
            "check": exported.check,
            "ollama": exported.ollama,
            "usage": exported.usage,
            "advice": exported.advice,
        }),
    })
}

/// Pipeline state for `synk status`: the latest dataset and run, if any,
/// and the `next` hint their advice suggests.
pub fn pipeline_status(project: &Project) -> Result<(Value, Option<String>)> {
    let cfg = project.pipeline();
    let artifacts = project.artifacts();
    let dataset = artifacts.latest(dataset::KIND)?;
    let run = artifacts.latest(train::TRAIN_KIND)?;
    let eval = match &run {
        Some(run) => train::eval_for(&artifacts, run)?,
        None => None,
    };
    let verdict = eval.as_ref().map(train::eval_verdict);
    let general = eval
        .as_ref()
        .map(|e| advice::general_verdicts(&e.summary))
        .unwrap_or_default();
    let mut advice = dataset
        .as_ref()
        .map(|d| advice::dataset(&d.summary, cfg))
        .unwrap_or_default();
    if let (Some(run), Some(verdict)) = (&run, &verdict) {
        advice.extend(advice::train(&run.summary, verdict, &general, cfg));
    }
    let next = advice::next(&advice);
    let dataset_json = match &dataset {
        Some(d) => {
            let mut v = dataset::summary(d);
            v["id"] = json!(ids::id(&artifacts, Kind::Dataset, d)?);
            Some(v)
        }
        None => None,
    };
    let run_json = match &run {
        Some(r) => Some(json!({
            "id": ids::id(&artifacts, Kind::Run, r)?,
            "key": r.key,
            "created_at": r.created_at,
            "base_model": r.summary["base_model"],
        })),
        None => None,
    };
    let trainer_ready =
        project.trainer_python().is_some() || sidecar::synk_home().join("trainer.json").is_file();
    let probe = synk_pipeline::generator::probe(&cfg.generator);
    let missing_sources: Vec<String> = project
        .resolved_sources()
        .iter()
        .filter(|s| !s.path.exists())
        .map(|s| s.name.clone())
        .collect();
    let condition = |kind: &str, status: Option<bool>, reason: &str, message: String| {
        json!({
            "type": kind,
            "status": match status { Some(true) => "True", Some(false) => "False", None => "Unknown" },
            "reason": reason,
            "message": message,
        })
    };
    let conditions = vec![
        condition(
            "SourcesFound",
            Some(missing_sources.is_empty()),
            if missing_sources.is_empty() {
                "AllPresent"
            } else {
                "PathMissing"
            },
            if missing_sources.is_empty() {
                format!("{} sources", project.sources().len())
            } else {
                format!("missing: {}", missing_sources.join(", "))
            },
        ),
        condition(
            "TrainerInstalled",
            Some(trainer_ready),
            if trainer_ready {
                "Found"
            } else {
                "NotInstalled"
            },
            if trainer_ready {
                "the Python trainer is set up".to_owned()
            } else {
                "not installed; synk fetch trainer (or synk setup) installs it".to_owned()
            },
        ),
        condition(
            "GeneratorReachable",
            probe.reachable,
            match probe.reachable {
                Some(true) => "Answered",
                Some(false) => "NoAnswer",
                None => "Remote",
            },
            probe.message.clone(),
        ),
        condition(
            "GeneratorModelReady",
            probe.model_ready,
            match probe.model_ready {
                Some(true) => "Found",
                Some(false) => "NotPulled",
                None => "Unknown",
            },
            cfg.generator.model.clone(),
        ),
    ];
    Ok((
        json!({
            "conditions": conditions,
            "dataset": dataset_json,
            "run": run_json,
            "verdict": verdict,
            "advice": advice,
            "trainer_ready": trainer_ready,
        }),
        next,
    ))
}
