//! `synk train`, `synk dataset`, `synk fetch`, and `wait`, `logs`, `cancel`.
//!
//! `train` and `dataset` never download or send document text off this
//! machine unless `--yes` is given: when the preview lists a download they
//! stop with `CONSENT_REQUIRED` and the exact `synk fetch` command. Running
//! that command is the consent (ADR 0002).

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use serde_json::{Value, json};
use synk_core::{Applicability, ErrorAction, ErrorCode, Result, SynkError};
use synk_pipeline::graph::{self, Action, Options, Preview, Step, Workspace};
use synk_pipeline::ops::{self, NewOp, OpKind, OpRecord, Ops, State};
use synk_pipeline::sidecar::Backend;
use synk_pipeline::{PipelineConfig, exec, fetch, ids};

use crate::config::Project;
use crate::human;
use crate::output::{Output, verbose};
use crate::pipeline::{Progress, ProgressMode};

/// What `train` and `dataset` take from the command line.
pub struct TrainArgs {
    /// Continue (or show) this run, with the settings it started with.
    pub run: Option<String>,
    pub set: Vec<String>,
    pub name: Option<String>,
    pub sample: Option<usize>,
    pub force: Option<String>,
    pub dry_run: bool,
    pub yes: bool,
}

/// How a long command shows itself while it runs.
pub struct Display {
    pub progress: Progress,
    pub json: bool,
    pub mode: ProgressMode,
}

fn gb(bytes: u64) -> String {
    if bytes >= 1_000_000_000 {
        format!("{:.1} GB", bytes as f64 / 1e9)
    } else {
        format!("{} MB", (bytes as f64 / 1e6).ceil())
    }
}

fn minutes(seconds: u64) -> String {
    match seconds {
        0..90 => format!("{seconds} s"),
        90..5400 => format!("{} min", (seconds + 30) / 60),
        _ => format!("{:.1} h", seconds as f64 / 3600.0),
    }
}

/// `synk fetch a b c`, as a command line.
fn fetch_command(names: &[String]) -> String {
    format!("synk fetch {}", names.join(" "))
}

fn download_bytes(preview: &Preview) -> (u64, bool) {
    let mut total = 0;
    let mut known = true;
    for c in preview.downloads() {
        match c.bytes {
            Some(b) => total += b,
            None => known = false,
        }
    }
    (total, known)
}

/// One line per step: what runs, why, and what it costs.
fn step_line(preview: &Preview, s: &graph::StepPlan) -> String {
    let action = match s.action {
        Action::Cached => "cached",
        Action::Run if matches!(s.step, Step::Trainer | Step::Models | Step::Generator) => "fetch",
        Action::Run => "run",
        Action::Blocked => "blocked",
        Action::Skipped => "skipped",
    };
    let d = &s.detail;
    let what = match (s.step, s.action) {
        (_, Action::Skipped | Action::Blocked) => s.why.join("; "),
        (Step::Trainer, Action::Run) => format!(
            "Python training environment ({})",
            d["backend"].as_str().unwrap_or("cpu")
        ),
        (Step::Trainer, _) => "installed".to_owned(),
        (Step::Models, _) => format!(
            "{}{}",
            preview.model.label(),
            if s.why.is_empty() {
                String::new()
            } else {
                format!(": {}", s.why.join(", "))
            }
        ),
        (Step::Generator, Action::Run) => {
            format!("download {} in Ollama", d["model"].as_str().unwrap_or("?"))
        }
        (Step::Generator, _) => d["model"].as_str().unwrap_or("").to_owned(),
        (Step::Sync, Action::Run) => s.why.join(", "),
        (Step::Sync, _) => format!("{} documents unchanged", d["unchanged"]),
        (Step::Dataset, Action::Run) if d["llm_calls"].as_u64().unwrap_or(0) > 0 => format!(
            "{} chunks; the LLM writes queries for {} of them ({})",
            d["chunks"],
            d["llm_calls"],
            d["generator"].as_str().unwrap_or("")
        ),
        (Step::Dataset, _) => {
            let mut t = format!("{} chunks", d["chunks"]);
            if !s.why.is_empty() {
                let _ = write!(t, "; {}", s.why.join(", "));
            }
            t
        }
        (Step::Train, Action::Run) => format!(
            "{}, {} steps{}",
            d["base_model"].as_str().unwrap_or(&preview.model.label()),
            d["steps"],
            if s.why.is_empty() {
                String::new()
            } else {
                format!("; {}", s.why.join(", "))
            }
        ),
        (Step::Train, _) => "already trained with these settings".to_owned(),
        (Step::Eval, Action::Run) => "test against the base model".to_owned(),
        (Step::Eval, _) => "already tested".to_owned(),
    };
    let mut cost = String::new();
    if let Some(c) = preview.downloads().find(|c| match s.step {
        Step::Trainer => c.name.as_deref() == Some(graph::TRAINER),
        Step::Models => {
            c.what != preview.model.label() && preview.fetch.iter().any(|f| f.repo == c.what)
        }
        Step::Generator => Some(c.what.as_str()) == s.detail["model"].as_str(),
        _ => false,
    }) {
        let _ = write!(
            cost,
            "download {}",
            c.bytes.map_or("size unknown".to_owned(), gb)
        );
    }
    if let Some(sec) = s.est_seconds {
        if !cost.is_empty() {
            cost.push_str(", ");
        }
        let _ = write!(
            cost,
            "~{}{}",
            minutes(sec),
            if s.estimate_basis.as_deref() == Some("default") {
                " (rough)"
            } else {
                ""
            }
        );
    }
    format!(
        "  {action:<8} {:<10} {what}{}",
        s.step.as_str(),
        if cost.is_empty() {
            String::new()
        } else {
            format!("   [{cost}]")
        }
    )
}

/// What sending text off this machine means, in words.
fn egress_line(preview: &Preview) -> String {
    let egress: Vec<&graph::Consent> = preview.egress().collect();
    if egress.is_empty() {
        return "Nothing leaves this machine.".to_owned();
    }
    egress
        .iter()
        .map(|e| {
            format!(
                "Sends {} chunks of your documents to {}.",
                e.chunks.unwrap_or(0),
                e.host
            )
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn render_preview(preview: &Preview, would: &str) -> String {
    let mut out = format!("Would: {would}");
    if !preview.overrides.is_empty() {
        let sets: Vec<String> = preview
            .overrides
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect();
        let _ = write!(out, "   (with {})", sets.join(", "));
    }
    for s in &preview.steps {
        if s.action == Action::Skipped && !verbose() {
            continue;
        }
        let _ = write!(out, "\n{}", step_line(preview, s));
    }
    out.push('\n');
    let names = preview.fetch_names();
    if !names.is_empty() {
        let (bytes, known) = download_bytes(preview);
        let _ = write!(
            out,
            "Downloads {}{}: {}",
            gb(bytes),
            if known { "" } else { " or more" },
            fetch_command(&names)
        );
        out.push('\n');
    }
    out.push_str(&egress_line(preview));
    let total = preview.totals["est_seconds"].as_u64().unwrap_or(0);
    if total > 0 {
        let _ = write!(out, " About {} in all.", minutes(total));
    }
    for b in &preview.blockers {
        let _ = write!(out, "\nblocked[{}]: {}", b.code.as_str(), b.message);
        for a in &b.actions {
            match &a.argv {
                Some(argv) => {
                    let _ = write!(out, "\n  {}: {}", a.why, argv.join(" "));
                }
                None => {
                    let _ = write!(out, "\n  hint: {}", a.why);
                }
            }
        }
    }
    if preview.up_to_date() && preview.blockers.is_empty() {
        out.push_str("\nEverything is up to date.");
    }
    out
}

fn preview_json(preview: &Preview) -> Value {
    let mut v = serde_json::to_value(preview).expect("preview serializes");
    let names = preview.fetch_names();
    v["fetch_names"] = json!(names);
    v["fetch"] = if names.is_empty() {
        Value::Null
    } else {
        json!(fetch_command(&names))
    };
    v["egress"] = json!(preview.egress().collect::<Vec<_>>());
    v
}

/// Prints a line every 30 s while a long command runs, so shells that kill
/// silent commands see it is alive.
struct Heartbeat {
    stop: Arc<AtomicBool>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl Heartbeat {
    fn start(json: bool, mode: ProgressMode) -> Option<Self> {
        let enabled = json && mode != ProgressMode::None;
        if !enabled {
            return None;
        }
        let stop = Arc::new(AtomicBool::new(false));
        let flag = stop.clone();
        let started = Instant::now();
        let handle = std::thread::spawn(move || {
            let mut last = Instant::now();
            while !flag.load(Ordering::Relaxed) {
                std::thread::sleep(Duration::from_millis(250));
                if last.elapsed() >= Duration::from_secs(30) {
                    last = Instant::now();
                    eprintln!(
                        "{}",
                        json!({"v": 1, "event": "heartbeat", "elapsed_seconds": started.elapsed().as_secs()})
                    );
                }
            }
        });
        Some(Self {
            stop,
            handle: Some(handle),
        })
    }
}

impl Drop for Heartbeat {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

/// Tells whoever watches stderr which operation started: with
/// `--progress jsonl` an event, otherwise a line.
fn announce(display: &mut Display, record: &OpRecord) {
    if display.mode == ProgressMode::Jsonl || (display.json && display.mode == ProgressMode::Auto) {
        eprintln!(
            "{}",
            json!({"v": 1, "event": "op", "kind": record.kind, "id": record.id, "attempt": record.attempt})
        );
    } else if !display.progress.is_off() {
        let again = if record.attempt > 1 {
            format!(" (attempt {})", record.attempt)
        } else {
            String::new()
        };
        display.progress.line(&format!("{}{again}", record.id));
    }
}

/// A JSON value written as TOML for `--set`.
fn toml_value(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

fn describe_consent(c: &graph::Consent) -> String {
    match c.kind.as_str() {
        "egress" => format!(
            "send {} chunks of your documents to {}",
            c.chunks.unwrap_or(0),
            c.host
        ),
        _ => format!(
            "download {} ({}) from {}",
            c.name.as_deref().unwrap_or(&c.what),
            c.bytes.map_or("size unknown".to_owned(), gb),
            c.host
        ),
    }
}

/// The error `train` and `dataset` stop with when the preview needs
/// consent: the exact fetch command, and `--yes`.
fn consent_required(preview: &Preview, retry: &str) -> SynkError {
    let names = preview.fetch_names();
    let list: Vec<String> = preview.consent.iter().map(describe_consent).collect();
    let (bytes, known) = download_bytes(preview);
    let message = if names.is_empty() {
        format!("needs your consent first: {}", list.join("; "))
    } else {
        format!(
            "needs downloads first ({}{}): {}",
            gb(bytes),
            if known { "" } else { " or more" },
            names.join(", ")
        )
    };
    let mut err = SynkError::new(ErrorCode::ConsentRequired, message);
    if !names.is_empty() {
        err = err.with_action(ErrorAction::run(
            &fetch_command(&names),
            Applicability::Manual,
            format!("Download exactly these (ask the user first), then run `{retry}` again"),
        ));
    }
    let yes = format!("{retry} --yes");
    err.with_action(ErrorAction::run(
        &yes,
        Applicability::Manual,
        if preview.egress().next().is_some() {
            "Or run it with consent (ask the user first; this sends document text to the server named above)"
        } else {
            "Or download and run in one go (ask the user first)"
        },
    ))
    .with_details(json!({
        "fetch": names,
        "downloads": preview.downloads().collect::<Vec<_>>(),
        "egress": preview.egress().collect::<Vec<_>>(),
        "dry_run": preview_json(preview),
    }))
}

/// The settings a run is started with, and their hash.
fn settings_of(ws: &Workspace, args: &TrainArgs) -> (Value, String) {
    let settings = json!({
        "overrides": ws.overrides,
        "config": ws.cfg,
        "sample": args.sample,
        "force": args.force,
    });
    let hash = ops::settings_hash(&settings);
    (settings, hash)
}

/// The workspace a run was started with: its settings, not `synk.toml`'s.
fn frozen_workspace(project: &Project, run: &OpRecord) -> Result<(Workspace, TrainArgs)> {
    let s = &run.settings;
    let cfg: PipelineConfig = serde_json::from_value(s["config"].clone()).map_err(|_| {
        SynkError::new(
            ErrorCode::RunNotResumable,
            format!("{} has no saved settings to continue with", run.id),
        )
        .with_next("synk train --dry-run")
    })?;
    let overrides: BTreeMap<String, Value> =
        serde_json::from_value(s["overrides"].clone()).unwrap_or_default();
    let ws = project.workspace_with(cfg, overrides)?;
    let args = TrainArgs {
        run: Some(run.id.clone()),
        set: Vec::new(),
        name: run.name.clone(),
        sample: s["sample"].as_u64().map(|n| n as usize),
        force: s["force"].as_str().map(str::to_owned),
        dry_run: false,
        yes: false,
    };
    Ok((ws, args))
}

/// Downloads `names` as a `fetch:N` operation.
fn run_fetch(
    ws: &Workspace,
    items: &[fetch::Item],
    backend: Option<Backend>,
    display: &mut Display,
) -> Result<(OpRecord, Vec<Value>)> {
    let ops = ws.ops();
    let names: Vec<&str> = items.iter().map(|i| i.name.as_str()).collect();
    let mut op = ops.start(NewOp {
        kind: OpKind::Fetch,
        label: "",
        name: None,
        settings: json!({ "items": names }),
        settings_hash: String::new(),
        n: None,
    })?;
    announce(display, &op.record());
    let _heartbeat = Heartbeat::start(display.json, display.mode);
    let progress = &mut display.progress;
    let result = fetch::run(ws, items, backend, &mut op, &mut |_, e| progress.event(e));
    progress.end();
    match result {
        Ok(done) => {
            let record = op.finish(Ok(json!({ "items": done })));
            Ok((record, done))
        }
        Err(err) => {
            let record = op.finish(Err(&err));
            Err(with_op(err, &record))
        }
    }
}

/// Adds the operation's ID to an error, and how to continue it.
fn with_op(mut err: SynkError, record: &OpRecord) -> SynkError {
    if err.details.is_null() {
        err.details = json!({});
    }
    err.details["id"] = json!(record.id);
    if err.code == ErrorCode::Cancelled || err.retryable {
        err = err.with_action(ErrorAction::run(
            &record.kind.continue_command(&record.id),
            Applicability::Review,
            "run it again; finished steps are kept and training continues from its checkpoint",
        ));
    }
    err
}

/// `synk train` (`kind` Run) and `synk dataset` (`kind` Dataset).
pub fn train(
    project: &Project,
    kind: OpKind,
    args: &TrainArgs,
    mut display: Display,
) -> Result<Output> {
    let ops = Ops::new(project.data_dir()?);
    ops.migrate(&project.artifacts())?;
    let verb = if kind == OpKind::Run {
        "train"
    } else {
        "dataset"
    };

    // `train RUN`: that run, with the settings it started with.
    let mut continuing = None;
    let (ws, args_owned);
    let args = match &args.run {
        Some(input) => {
            if !args.set.is_empty()
                || args.name.is_some()
                || args.sample.is_some()
                || args.force.is_some()
            {
                return Err(SynkError::new(
                    ErrorCode::InvalidArgs,
                    "a run keeps the settings it started with: don't combine it with --set, --name, --sample or --force",
                ));
            }
            let run = ops
                .find(input)?
                .filter(|r| r.kind == OpKind::Run)
                .ok_or_else(|| {
                    SynkError::new(ErrorCode::RunNotFound, format!("no run matches `{input}`"))
                        .with_next("synk run list")
                })?;
            match run.state {
                State::Succeeded if !args.dry_run => return Ok(reused_output(run)),
                State::Running => {
                    return Err(SynkError::new(
                        ErrorCode::JobRunning,
                        format!("{} is running", run.id),
                    )
                    .with_next("synk wait"));
                }
                _ => {}
            }
            let (w, mut a) = frozen_workspace(project, &run)?;
            a.dry_run = args.dry_run;
            a.yes = args.yes;
            ws = w;
            args_owned = a;
            continuing = Some(run);
            &args_owned
        }
        None => {
            ws = project.workspace(&args.set)?;
            args
        }
    };

    let target = if kind == OpKind::Run {
        Step::Eval
    } else {
        Step::Dataset
    };
    let force = args.force.as_deref().map(Step::parse).transpose()?;
    if let Some(step) = force
        && !matches!(step, Step::Dataset | Step::Train | Step::Eval)
    {
        return Err(SynkError::new(
            ErrorCode::InvalidArgs,
            format!(
                "--force takes dataset, train or eval, not {}",
                step.as_str()
            ),
        ));
    }
    let opts = Options {
        target,
        force,
        sample: args.sample,
        network: true,
    };
    let mut preview = graph::make(&ws, &opts)?;
    let (settings, settings_hash) = settings_of(&ws, args);

    // Which run this would be.
    let reuse = if kind == OpKind::Run && continuing.is_none() {
        reusable_run(&ops, &preview, force)?
    } else {
        None
    };
    let resume = match (&continuing, kind) {
        (Some(run), _) => Some(run.clone()),
        (None, OpKind::Run) if reuse.is_none() => ops
            .list_kind(OpKind::Run)?
            .into_iter()
            .find(|r| r.settings_hash == settings_hash)
            .filter(|r| {
                r.unfinished()
                    && args
                        .name
                        .as_deref()
                        .is_none_or(|n| r.name.as_deref() == Some(n))
            }),
        _ => None,
    };
    let retry = match &continuing {
        Some(run) => format!("synk train {}", ops::short_id(&run.id)),
        None => {
            let mut words = vec![format!("synk {verb}")];
            for set in &args.set {
                words.push(format!("--set {set}"));
            }
            if let Some(n) = &args.name {
                words.push(format!("--name {n}"));
            }
            if let Some(n) = args.sample {
                words.push(format!("--sample {n}"));
            }
            if let Some(f) = &args.force {
                words.push(format!("--force {f}"));
            }
            words.join(" ")
        }
    };

    if args.dry_run {
        let would = match (&reuse, &resume, kind) {
            (Some(r), _, _) => format!("nothing to do; {} has these results", r.id),
            (_, Some(r), _) => format!("continue {}", r.id),
            (_, _, OpKind::Run) => "a new run".to_owned(),
            _ => "build the dataset".to_owned(),
        };
        let mut human = render_preview(&preview, &would);
        let names = preview.fetch_names();
        let next = if !preview.blockers.is_empty() {
            preview.blockers[0].next.clone()
        } else if !names.is_empty() || preview.egress().next().is_some() || reuse.is_some() {
            None
        } else {
            Some(retry.clone())
        };
        if !names.is_empty() {
            let _ = write!(
                human,
                "\nNext (ask first): {}  or: {retry} --yes",
                fetch_command(&names)
            );
        } else if preview.egress().next().is_some() {
            let _ = write!(human, "\nNext (ask first): {retry} --yes");
        } else if let Some(next) = &next {
            let _ = write!(human, "\nNext: {next}");
        }
        let mut data = preview_json(&preview);
        data["would"] = json!(would);
        data["reuses"] = json!(reuse.as_ref().map(|r| &r.id));
        data["continues"] = json!(resume.as_ref().map(|r| &r.id));
        data["next"] = json!(next);
        return Ok(Output { human, data });
    }

    if let Some(b) = preview.blockers.first() {
        return Err(b.clone());
    }
    if let Some(run) = reuse {
        return Ok(reused_output(run));
    }
    let needs_consent = !preview.consent.is_empty();
    if needs_consent && !args.yes {
        return Err(consent_required(&preview, &retry));
    }
    // `--yes`: download first, then look again.
    let mut fetched = Vec::new();
    let names = preview.fetch_names();
    if !names.is_empty() {
        let items = fetch::resolve(&ws, &names)?;
        let (_, done) = run_fetch(&ws, &items, None, &mut display)?;
        fetched = done;
        preview = graph::make(&ws, &opts)?;
        if let Some(b) = preview.blockers.first() {
            return Err(b.clone());
        }
        if !preview.fetch_names().is_empty() {
            return Err(consent_required(&preview, &retry));
        }
    }

    let mut op = match (resume, kind) {
        (Some(record), _) => ops.resume(record)?,
        (None, OpKind::Run) => ops.start(NewOp {
            kind,
            label: &ids::model_label(&preview.model.label()),
            name: args.name.as_deref(),
            settings,
            settings_hash,
            n: None,
        })?,
        (None, _) => {
            let n = match &preview.dataset_key {
                Some(key) => ops.dataset_number(key, true)?,
                None => None,
            };
            let existing = match n {
                Some(n) => ops.find(&format!("dataset:{n}"))?,
                None => None,
            };
            match existing {
                Some(record) => ops.resume(record)?,
                None => ops.start(NewOp {
                    kind,
                    label: &ids::slug(&ws.cfg.generator.model),
                    name: None,
                    settings,
                    settings_hash,
                    n,
                })?,
            }
        }
    };
    announce(&mut display, &op.record());
    let _heartbeat = Heartbeat::start(display.json, display.mode);
    let mut current: Option<Step> = None;
    let progress = &mut display.progress;
    let result = exec::run(&ws, &preview, &mut op, &mut |step, event| {
        if current != Some(step) {
            current = Some(step);
            if !progress.is_off() {
                progress.line(&format!("-- {}", step.as_str()));
            }
        }
        progress.event(event);
    });
    progress.end();
    match result {
        Ok(outcome) => {
            let result = outcome.to_json();
            let record = op.finish(Ok(result.clone()));
            let mut data = result;
            data["id"] = json!(record.id);
            data["reused"] = json!(false);
            data["fetched"] = json!(fetched);
            data["op"] = serde_json::to_value(&record).expect("record serializes");
            Ok(Output {
                human: render_result(&record, &outcome),
                data,
            })
        }
        Err(err) => {
            let record = op.finish(Err(&err));
            Err(with_op(err, &record))
        }
    }
}

/// The finished run that already has what the preview would make.
fn reusable_run(ops: &Ops, preview: &Preview, force: Option<Step>) -> Result<Option<OpRecord>> {
    let cached = |step: Step| {
        preview
            .step(step)
            .is_some_and(|s| s.action == Action::Cached)
    };
    if force.is_some()
        || !preview.consent.is_empty()
        || !(cached(Step::Train) && cached(Step::Eval))
    {
        return Ok(None);
    }
    let Some(key) = &preview.run_key else {
        return Ok(None);
    };
    Ok(ops
        .run_for_key(key)?
        .filter(|r| r.state == State::Succeeded))
}

/// A finished operation returned as is.
fn reused_output(record: OpRecord) -> Output {
    let mut data = record.result.clone().unwrap_or_else(|| json!({}));
    if data["run"].is_null() && record.kind == OpKind::Run {
        data["run"] = json!(record.id);
        data["key"] = json!(record.key);
    }
    data["id"] = json!(record.id);
    data["reused"] = json!(true);
    data["op"] = serde_json::to_value(&record).expect("record serializes");
    let next = data["next"]
        .as_str()
        .map(str::to_owned)
        .or_else(|| (record.kind == OpKind::Run).then(|| format!("synk run show {}", record.id)));
    data["next"] = json!(next);
    let mut human = format!(
        "{} already has these results: nothing changed since it ran.",
        record.id
    );
    if !data["verdict"].is_null() {
        let _ = write!(human, "\nTest: {}", human::verdict(&data["verdict"]));
    }
    if let Some(next) = &next {
        let _ = write!(human, "\nNext: {next}");
    }
    Output { human, data }
}

fn render_result(record: &OpRecord, outcome: &exec::Outcome) -> String {
    let mut out = String::new();
    for s in &record.steps {
        let time = s
            .seconds
            .map(|t| format!("  {}", minutes(t)))
            .unwrap_or_default();
        let _ = writeln!(out, "  {:<8} {:<10}{time}", s.status, s.step);
    }
    if let Some(sync) = &outcome.sync {
        let t = &sync.totals;
        let _ = writeln!(
            out,
            "Documents: {} new, {} changed, {} deleted, {} unchanged",
            t.added, t.updated, t.deleted, t.unchanged
        );
        render_skipped(&mut out, sync);
    }
    if let Some((id, _)) = &outcome.dataset {
        let _ = writeln!(out, "Dataset: {id}");
    }
    if let Some(e) = &outcome.egress {
        let _ = writeln!(
            out,
            "Sent {} chunks of your documents to {}",
            e["chunks"],
            e["host"].as_str().unwrap_or("?")
        );
    }
    if let (Some((id, run)), Some(v)) = (&outcome.run, &outcome.verdict) {
        let _ = writeln!(
            out,
            "Run: {id} ({}, {} queries)",
            run.summary["base_model"].as_str().unwrap_or("?"),
            run.summary["examples"]
        );
        if let Some(same) = &outcome.same_as {
            let _ = writeln!(out, "Same inputs as {same}, so it is the same model.");
        }
        let _ = writeln!(out, "Test: {}", human::verdict(v));
        for (set, g) in &outcome.general {
            let _ = writeln!(
                out,
                "General check ({}): {:.2} -> {:.2}, {}",
                human::language(set),
                g.base,
                g.tuned,
                g.outcome.replace('_', " ")
            );
        }
        let _ = writeln!(out, "{}", human::QUALITY_NOTE);
    }
    let mut out = out.trim_end().to_owned();
    out.push_str(&human::advice(&outcome.advice));
    match outcome.next() {
        Some(next) => {
            let _ = write!(out, "\nNext: {next}");
        }
        None => out.push_str("\nDone."),
    }
    out
}

pub fn render_skipped(out: &mut String, sync: &synk_pipeline::sync::Outcome) {
    let skipped: Vec<_> = sync.sources.iter().flat_map(|r| r.skipped.iter()).collect();
    for item in skipped.iter().take(crate::commands::MAX_SKIPPED_SHOWN) {
        let _ = writeln!(
            out,
            "  skipped {}: {}",
            crate::output::show_path(std::path::Path::new(&item.path)),
            crate::commands::skip_reason(&item.reason)
        );
    }
    if skipped.len() > crate::commands::MAX_SKIPPED_SHOWN {
        let _ = writeln!(
            out,
            "  ... and {} more skipped (see --json)",
            skipped.len() - crate::commands::MAX_SKIPPED_SHOWN
        );
    }
    for w in &sync.warnings {
        let _ = writeln!(out, "warning[{}]: {}", w.code.as_str(), w.message);
    }
}

/// `synk fetch [NAME…]`.
pub fn fetch(
    project: &Project,
    names: &[String],
    sets: &[String],
    backend: &str,
    mut display: Display,
) -> Result<Output> {
    let backend = match backend {
        "auto" => None,
        other => Some(Backend::parse(other)?),
    };
    let ws = project.workspace(sets)?;
    let names = if names.is_empty() {
        let preview = graph::make(&ws, &Options::default())?;
        if let Some(b) = preview.blockers.first()
            && preview.fetch_names().is_empty()
        {
            return Err(b.clone());
        }
        preview.fetch_names()
    } else {
        names.to_vec()
    };
    let items = fetch::resolve(&ws, &names)?;
    let next = "synk train --dry-run";
    if items.iter().all(|i| i.present) {
        let mut human = if items.is_empty() {
            "Nothing to download: training has everything it needs.".to_owned()
        } else {
            let list: Vec<&str> = items.iter().map(|i| i.name.as_str()).collect();
            format!("Already here: {}", list.join(", "))
        };
        let _ = write!(human, "\nNext: {next}");
        return Ok(Output {
            human,
            data: json!({
                "id": null,
                "items": items.iter().map(fetch::Item::to_json).collect::<Vec<_>>(),
                "next": next,
            }),
        });
    }
    let (record, done) = run_fetch(&ws, &items, backend, &mut display)?;
    let mut human = record.id.clone();
    for d in &done {
        let _ = write!(
            human,
            "\n  {:<8} {}",
            d["status"].as_str().unwrap_or(""),
            d["name"].as_str().unwrap_or("")
        );
    }
    let _ = write!(human, "\nNext: {next}");
    Ok(Output {
        human,
        data: json!({
            "id": record.id,
            "items": done,
            "op": record,
            "next": next,
        }),
    })
}

fn render_op(r: &OpRecord) -> String {
    let mut out = format!(
        "{}: {}, started {} ago",
        r.id,
        r.state.as_str(),
        minutes(ops::now().saturating_sub(r.started_at))
    );
    if r.attempt > 1 {
        let _ = write!(out, " (attempt {})", r.attempt);
    }
    for s in &r.steps {
        let _ = write!(out, "\n  {:<8} {}", s.status, s.step);
    }
    if let Some(p) = &r.progress {
        let pct = (p.completed * 100).checked_div(p.total).unwrap_or(0);
        let _ = write!(
            out,
            "\n  now: {} {}/{} {} ({pct}%)",
            p.step, p.completed, p.total, p.unit
        );
    }
    if let Some(reason) = &r.reason {
        let _ = write!(out, "\n  {reason}");
    }
    if let Some(e) = &r.error {
        let _ = write!(
            out,
            "\n  error[{}]: {}",
            e["code"].as_str().unwrap_or("?"),
            e["message"].as_str().unwrap_or("")
        );
    }
    if let Some(res) = &r.result
        && !res["verdict"].is_null()
    {
        let _ = write!(out, "\n  test: {}", human::verdict(&res["verdict"]));
    }
    out
}

fn op_next(r: &OpRecord) -> Option<String> {
    match r.state {
        State::Running => Some("synk wait --timeout 9m".to_owned()),
        State::Succeeded => r
            .result
            .as_ref()
            .and_then(|res| res["next"].as_str().map(str::to_owned)),
        State::Interrupted | State::Cancelled | State::Failed => {
            Some(r.kind.continue_command(&r.id))
        }
    }
}

fn op_output(r: OpRecord) -> Output {
    let next = op_next(&r);
    let mut human = render_op(&r);
    if let Some(next) = &next {
        let _ = write!(human, "\nNext: {next}");
    }
    let mut data = serde_json::to_value(&r).expect("record serializes");
    data["next"] = json!(next);
    Output { human, data }
}

/// Waits up to `timeout` for an operation to finish. Running out of time is
/// not an error: `done` is false and `next` waits again.
pub fn wait(
    project: &Project,
    id: Option<&str>,
    timeout: Duration,
    json: bool,
    mode: ProgressMode,
) -> Result<Output> {
    let ops = Ops::new(project.data_dir()?);
    let target = ops.get(id)?.id;
    let _heartbeat = Heartbeat::start(json, mode);
    let started = Instant::now();
    loop {
        let record = ops.get(Some(&target))?;
        if record.done || started.elapsed() >= timeout {
            return Ok(op_output(record));
        }
        std::thread::sleep(Duration::from_millis(500));
    }
}

pub fn logs(project: &Project, id: Option<&str>, after: u64, limit: usize) -> Result<Output> {
    let ops = Ops::new(project.data_dir()?);
    let record = ops.get(id)?;
    let (events, more) = ops.events(&record.id, after, limit)?;
    let last = events
        .last()
        .and_then(|e| e["seq"].as_u64())
        .unwrap_or(after);
    let mut human = String::new();
    for e in &events {
        let line = match e["event"].as_str() {
            Some("progress") => format!(
                "{} {}/{} {}",
                e["stage"].as_str().unwrap_or(""),
                e["step"],
                e["total"],
                e["unit"].as_str().unwrap_or("")
            ),
            _ => e["message"].as_str().unwrap_or("").to_owned(),
        };
        let _ = writeln!(
            human,
            "[{}] {:<9} {line}",
            e["seq"],
            e["step"].as_str().unwrap_or("")
        );
    }
    let next = more.then(|| format!("synk logs {} --after {last}", record.id));
    if let Some(n) = &next {
        let _ = write!(human, "Next: {n}");
    }
    if human.is_empty() {
        human = format!("{}: no events yet", record.id);
    }
    Ok(Output {
        human: human.trim_end().to_owned(),
        data: json!({ "id": record.id, "events": events, "last_seq": last, "more": more, "next": next }),
    })
}

pub fn cancel(project: &Project, id: Option<&str>) -> Result<Output> {
    let ops = Ops::new(project.data_dir()?);
    let record = ops.cancel(id)?;
    let running = record.state == State::Running;
    let next = running.then(|| "synk wait --timeout 9m".to_owned());
    Ok(Output {
        human: if running {
            format!(
                "Asked {} to stop; it stops at the next safe point (training saves a checkpoint first).\nNext: {}",
                record.id,
                next.as_deref().unwrap_or_default()
            )
        } else {
            format!("{} isn't running ({})", record.id, record.state.as_str())
        },
        data: json!({ "id": record.id, "requested": running, "next": next }),
    })
}

/// The running operation, else the newest; used by `status`.
pub fn status_op(project: &Project) -> Option<OpRecord> {
    let ops = Ops::new(project.data_dir().ok()?);
    ops.running()
        .ok()
        .flatten()
        .or_else(|| ops.list().ok()?.into_iter().next())
}

/// `--set` values of a run, for `run list`.
pub fn sets_of(run: &OpRecord) -> Vec<String> {
    run.settings["overrides"]
        .as_object()
        .map(|m| {
            m.iter()
                .map(|(k, v)| format!("{k}={}", toml_value(v)))
                .collect()
        })
        .unwrap_or_default()
}
