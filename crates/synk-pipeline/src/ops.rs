//! Operations: the long commands (`train`, `dataset`, `fetch`), one at a
//! time per project, each with a record any shell can read.
//!
//! An operation lives in `.synk/ops/<kind>-<n>/`: `op.json` (state, steps,
//! progress, heartbeat, settings, result or error) and `events.jsonl` (every
//! progress and log event, numbered). `.synk/lock` names the running one.
//! Everything is plain files, so it works the same on every OS:
//! - IDs: `run:<n>-<label>`, `dataset:<n>-<label>`, `fetch:<n>`, with `n`
//!   counting up per kind and never reused (`.synk/ops/seq.json`);
//! - liveness: the running process rewrites `heartbeat_at` every few
//!   seconds; one whose heartbeat is older than [`STALE_AFTER`] is
//!   `interrupted` and its lock is free;
//! - cancel: `synk cancel` creates `cancel` in the operation's folder; it is
//!   checked between steps, and by the trainer at every training step.
//!
//! A training run's record is the run: its ID is what people name, and the
//! train artifact's key is stored in it once known.

use std::collections::BTreeMap;
use std::fs::{self, OpenOptions};
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use synk_core::{ErrorCode, Result, SynkError};

use crate::artifact::Artifacts;

/// How often a running operation rewrites its heartbeat.
const HEARTBEAT: Duration = Duration::from_secs(5);
/// A running operation whose heartbeat is older than this is dead.
pub const STALE_AFTER: u64 = 120;
/// Finished fetch and dataset records older than this many days are removed
/// when a new operation starts. Run records are kept: they are the runs.
const RETENTION_DAYS: u64 = 30;

pub fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

/// What an operation does.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OpKind {
    /// A training run: sync, dataset, train, eval.
    Run,
    /// Sync and the dataset only.
    Dataset,
    /// Downloads.
    Fetch,
}

impl OpKind {
    pub const ALL: [OpKind; 3] = [OpKind::Run, OpKind::Dataset, OpKind::Fetch];

    pub fn prefix(self) -> &'static str {
        match self {
            Self::Run => "run",
            Self::Dataset => "dataset",
            Self::Fetch => "fetch",
        }
    }

    fn parse(prefix: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|k| k.prefix() == prefix)
    }

    /// The command that continues an unfinished operation of this kind.
    pub fn continue_command(self, id: &str) -> String {
        match self {
            Self::Run => format!("synk train {}", short_id(id)),
            Self::Dataset => "synk dataset".to_owned(),
            Self::Fetch => "synk fetch".to_owned(),
        }
    }
}

/// A short hash of a run's settings, to recognise the same ones again.
pub fn settings_hash(settings: &Value) -> String {
    blake3::hash(settings.to_string().as_bytes()).to_hex()[..16].to_owned()
}

/// `run:3` for `run:3-e5-small`: enough to find it again.
pub fn short_id(id: &str) -> String {
    match id.split_once(':') {
        Some((kind, rest)) => {
            let n = rest.split('-').next().unwrap_or(rest);
            format!("{kind}:{n}")
        }
        None => id.to_owned(),
    }
}

/// `running`, then `succeeded`, `failed`, `cancelled` or `interrupted`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum State {
    Running,
    Succeeded,
    Failed,
    Cancelled,
    Interrupted,
}

impl State {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Running => "running",
            Self::Succeeded => "succeeded",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
            Self::Interrupted => "interrupted",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepRecord {
    pub step: String,
    /// `ran`, `cached`, `skipped` or `failed`.
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seconds: Option<u64>,
    #[serde(default, skip_serializing_if = "Value::is_null")]
    pub detail: Value,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Progress {
    pub step: String,
    pub stage: String,
    pub completed: u64,
    pub total: u64,
    pub unit: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpRecord {
    pub v: u32,
    pub id: String,
    pub kind: OpKind,
    pub n: u64,
    /// `--name`, for runs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    pub state: State,
    pub done: bool,
    /// 1, plus one each time the operation is continued.
    #[serde(default = "one")]
    pub attempt: u32,
    pub started_at: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finished_at: Option<u64>,
    pub heartbeat_at: u64,
    pub pid: u32,
    /// What it was started with: `--set` overrides, the effective settings,
    /// `--sample`, `--force`, and for fetches the items.
    #[serde(default, skip_serializing_if = "Value::is_null")]
    pub settings: Value,
    /// A hash of `settings`, to find an unfinished run with the same ones.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub settings_hash: String,
    /// The dataset it used or built, once known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dataset: Option<String>,
    /// The train artifact's key (runs) or the dataset's (dataset builds),
    /// once known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub key: Option<String>,
    pub steps: Vec<StepRecord>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub progress: Option<Progress>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<Value>,
    /// Why an operation is `interrupted` or failed without an error.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    /// An earlier run that turned out to have the same inputs; its model
    /// and test results are this run's.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub same_as: Option<String>,
}

fn one() -> u32 {
    1
}

impl OpRecord {
    /// A `running` record whose process stopped writing its heartbeat is
    /// reported as `interrupted`.
    fn settle(mut self) -> Self {
        if self.state == State::Running && now().saturating_sub(self.heartbeat_at) > STALE_AFTER {
            self.state = State::Interrupted;
            self.done = true;
            self.reason = Some(format!(
                "the process stopped (no heartbeat for {} s); continue with: {}",
                now().saturating_sub(self.heartbeat_at),
                self.kind.continue_command(&self.id)
            ));
        }
        self
    }

    /// Finished without success: it can be continued.
    pub fn unfinished(&self) -> bool {
        matches!(
            self.state,
            State::Failed | State::Cancelled | State::Interrupted
        )
    }
}

/// What starting an operation needs.
pub struct NewOp<'a> {
    pub kind: OpKind,
    /// The readable part after the number, e.g. the base model.
    pub label: &'a str,
    pub name: Option<&'a str>,
    pub settings: Value,
    pub settings_hash: String,
    /// Use this number instead of the next one (a dataset's number).
    pub n: Option<u64>,
}

/// The operations of one project, under `.synk/`.
pub struct Ops {
    data_dir: PathBuf,
}

impl Ops {
    pub fn new(data_dir: PathBuf) -> Self {
        Self { data_dir }
    }

    fn ops_dir(&self) -> PathBuf {
        self.data_dir.join("ops")
    }

    fn lock_path(&self) -> PathBuf {
        self.data_dir.join("lock")
    }

    fn seq_path(&self) -> PathBuf {
        self.ops_dir().join("seq.json")
    }

    fn datasets_path(&self) -> PathBuf {
        self.ops_dir().join("datasets.json")
    }

    /// `run:3-e5-small` -> `.synk/ops/run-3`.
    pub fn dir(&self, id: &str) -> PathBuf {
        let (kind, rest) = id.split_once(':').unwrap_or(("op", id));
        let n = rest.split('-').next().unwrap_or(rest);
        self.ops_dir().join(format!("{kind}-{n}"))
    }

    fn read(&self, dir: &Path) -> Option<OpRecord> {
        let text = fs::read_to_string(dir.join("op.json")).ok()?;
        serde_json::from_str::<OpRecord>(&text)
            .ok()
            .map(OpRecord::settle)
    }

    /// Every operation, newest first.
    pub fn list(&self) -> Result<Vec<OpRecord>> {
        let mut out = Vec::new();
        let Ok(entries) = fs::read_dir(self.ops_dir()) else {
            return Ok(out);
        };
        for entry in entries.flatten() {
            if let Some(record) = self.read(&entry.path()) {
                out.push(record);
            }
        }
        out.sort_by(|a, b| {
            b.started_at
                .cmp(&a.started_at)
                .then(b.kind.cmp(&a.kind))
                .then(b.n.cmp(&a.n))
        });
        Ok(out)
    }

    /// Operations of one kind, newest first.
    pub fn list_kind(&self, kind: OpKind) -> Result<Vec<OpRecord>> {
        let mut all: Vec<OpRecord> = self
            .list()?
            .into_iter()
            .filter(|r| r.kind == kind)
            .collect();
        all.sort_by_key(|r| std::cmp::Reverse(r.n));
        Ok(all)
    }

    /// The running operation, if any.
    pub fn running(&self) -> Result<Option<OpRecord>> {
        let Ok(id) = fs::read_to_string(self.lock_path()) else {
            return Ok(None);
        };
        match self.read(&self.dir(id.trim())) {
            Some(record) if record.state == State::Running => Ok(Some(record)),
            _ => Ok(None),
        }
    }

    /// An operation by:
    /// - `None`, `current` or `latest`: the running one, else the newest;
    /// - an ID or a unique start of one (`run:3`, `fetch:2`, `run:3-e5`);
    /// - a bare number (a run), or a run's `--name`.
    pub fn get(&self, input: Option<&str>) -> Result<OpRecord> {
        let input = input.map(str::trim).filter(|s| !s.is_empty());
        match input.unwrap_or("current") {
            "current" | "latest" | "last" => {
                if let Some(r) = self.running()? {
                    return Ok(r);
                }
                self.list()?.into_iter().next().ok_or_else(|| {
                    SynkError::new(
                        ErrorCode::JobNotFound,
                        "nothing has run in this project yet",
                    )
                    .with_next("synk train --dry-run")
                })
            }
            other => self.find(other)?.ok_or_else(|| not_found(other)),
        }
    }

    /// An operation by ID, number or name; `None` when nothing matches.
    pub fn find(&self, input: &str) -> Result<Option<OpRecord>> {
        let all = self.list()?;
        let (kind, rest) = match input.split_once(':') {
            Some((k, rest)) => match OpKind::parse(k) {
                Some(kind) => (Some(kind), rest),
                None => (None, input),
            },
            None => (None, input),
        };
        let of_kind = |r: &&OpRecord| kind.is_none_or(|k| r.kind == k);
        // A bare number is the sequence number, of a run unless prefixed.
        if let Ok(n) = rest.parse::<u64>() {
            let kind = kind.unwrap_or(OpKind::Run);
            return Ok(all.into_iter().find(|r| r.kind == kind && r.n == n));
        }
        if kind.is_none()
            && let Some(r) = all
                .iter()
                .find(|r| r.kind == OpKind::Run && r.name.as_deref() == Some(input))
        {
            return Ok(Some(r.clone()));
        }
        let found: Vec<&OpRecord> = all
            .iter()
            .filter(of_kind)
            .filter(|r| {
                let readable = r.id.split_once(':').map_or(r.id.as_str(), |(_, x)| x);
                readable == rest || readable.starts_with(&format!("{rest}-")) || r.id == input
            })
            .collect();
        match found.as_slice() {
            [] => Ok(None),
            [one] => Ok(Some((*one).clone())),
            many => Err(SynkError::new(
                ErrorCode::IdAmbiguous,
                format!(
                    "`{input}` matches {} operations; use more of the ID",
                    many.len()
                ),
            )
            .with_details(json!({
                "candidates": many.iter().map(|r| &r.id).collect::<Vec<_>>(),
            }))),
        }
    }

    /// The run that owns train artifact `key`: the first one that produced
    /// it.
    pub fn run_for_key(&self, key: &str) -> Result<Option<OpRecord>> {
        let mut runs: Vec<OpRecord> = self
            .list_kind(OpKind::Run)?
            .into_iter()
            .filter(|r| r.key.as_deref() == Some(key))
            .collect();
        runs.sort_by_key(|r| (r.same_as.is_some(), r.state != State::Succeeded, r.n));
        Ok(runs.into_iter().next())
    }

    /// The number of dataset `key`, assigning the next one when `assign`.
    pub fn dataset_number(&self, key: &str, assign: bool) -> Result<Option<u64>> {
        let path = self.datasets_path();
        let mut map: BTreeMap<String, u64> = read_json(&path).unwrap_or_default();
        if let Some(n) = map.get(key) {
            return Ok(Some(*n));
        }
        if !assign {
            return Ok(None);
        }
        let n = self.next_number(OpKind::Dataset, map.values().copied().max().unwrap_or(0))?;
        map.insert(key.to_owned(), n);
        write_json(&path, &map)?;
        Ok(Some(n))
    }

    /// The dataset key numbered `n`.
    pub fn dataset_key(&self, n: u64) -> Option<String> {
        let map: BTreeMap<String, u64> = read_json(&self.datasets_path()).unwrap_or_default();
        map.into_iter().find(|(_, m)| *m == n).map(|(k, _)| k)
    }

    /// Takes the next number of `kind`: above every number ever handed out.
    fn next_number(&self, kind: OpKind, at_least: u64) -> Result<u64> {
        let path = self.seq_path();
        let mut seq: BTreeMap<String, u64> = read_json(&path).unwrap_or_default();
        let existing = self.list_kind(kind)?.iter().map(|r| r.n).max().unwrap_or(0);
        let n = seq
            .get(kind.prefix())
            .copied()
            .unwrap_or(0)
            .max(existing)
            .max(at_least)
            + 1;
        seq.insert(kind.prefix().to_owned(), n);
        write_json(&path, &seq)?;
        Ok(n)
    }

    /// Asks a running operation to stop. Returns its record.
    pub fn cancel(&self, id: Option<&str>) -> Result<OpRecord> {
        let record = self.get(id)?;
        if record.state == State::Running {
            let path = self.dir(&record.id).join("cancel");
            fs::write(&path, b"").map_err(|e| SynkError::io(path.display(), e))?;
        }
        Ok(record)
    }

    /// Events of an operation after sequence number `after`, at most
    /// `limit`.
    pub fn events(&self, id: &str, after: u64, limit: usize) -> Result<(Vec<Value>, bool)> {
        let path = self.dir(id).join("events.jsonl");
        let text = fs::read_to_string(&path).unwrap_or_default();
        let mut events: Vec<Value> = text
            .lines()
            .filter_map(|l| serde_json::from_str::<Value>(l).ok())
            .filter(|e| e["seq"].as_u64().unwrap_or(0) > after)
            .collect();
        let more = events.len() > limit;
        events.truncate(limit);
        Ok((events, more))
    }

    fn check_free(&self) -> Result<()> {
        if let Some(running) = self.running()? {
            return Err(SynkError::new(
                ErrorCode::JobRunning,
                format!(
                    "{} is already running in this project (started {} s ago)",
                    running.id,
                    now().saturating_sub(running.started_at)
                ),
            )
            .with_next("synk wait")
            .with_details(json!({ "running": running.id })));
        }
        Ok(())
    }

    /// Starts a new operation, taking the project's lock.
    pub fn start(&self, new: NewOp<'_>) -> Result<Op> {
        self.check_free()?;
        if let Some(name) = new.name {
            let taken = self
                .list_kind(OpKind::Run)?
                .into_iter()
                .find(|r| r.name.as_deref() == Some(name));
            if let Some(r) = taken {
                return Err(SynkError::new(
                    ErrorCode::RunNameTaken,
                    format!("`{name}` already names {}", r.id),
                )
                .with_hint("pick another --name, or continue that run with synk train <run>"));
            }
            if name.parse::<u64>().is_ok() || name.contains(':') {
                return Err(SynkError::new(
                    ErrorCode::InvalidArgs,
                    format!("`{name}` can't be a name: it looks like an ID"),
                ));
            }
        }
        self.prune();
        let n = match new.n {
            Some(n) => n,
            None => self.next_number(new.kind, 0)?,
        };
        let id = match new.kind {
            OpKind::Fetch => format!("fetch:{n}"),
            kind => format!(
                "{}:{n}-{}",
                kind.prefix(),
                new.name
                    .map_or_else(|| new.label.to_owned(), crate::ids::slug)
            ),
        };
        let started = now();
        let record = OpRecord {
            v: 1,
            id,
            kind: new.kind,
            n,
            name: new.name.map(str::to_owned),
            state: State::Running,
            done: false,
            attempt: 1,
            started_at: started,
            finished_at: None,
            heartbeat_at: started,
            pid: std::process::id(),
            settings: new.settings,
            settings_hash: new.settings_hash,
            dataset: None,
            key: None,
            steps: Vec::new(),
            progress: None,
            result: None,
            error: None,
            reason: None,
            same_as: None,
        };
        self.begin(record, 0)
    }

    /// Continues an unfinished operation under the same ID.
    pub fn resume(&self, mut record: OpRecord) -> Result<Op> {
        self.check_free()?;
        let dir = self.dir(&record.id);
        let _ = fs::remove_file(dir.join("cancel"));
        let seq =
            fs::read_to_string(dir.join("events.jsonl")).map_or(0, |t| t.lines().count() as u64);
        let started = now();
        record.attempt += 1;
        record.state = State::Running;
        record.done = false;
        record.finished_at = None;
        record.heartbeat_at = started;
        record.pid = std::process::id();
        record.progress = None;
        record.result = None;
        record.error = None;
        record.reason = None;
        self.begin(record, seq)
    }

    fn begin(&self, record: OpRecord, seq: u64) -> Result<Op> {
        let dir = self.dir(&record.id);
        fs::create_dir_all(&dir).map_err(|e| SynkError::io(dir.display(), e))?;
        write_record(&dir, &record)?;
        let lock = self.lock_path();
        fs::write(&lock, &record.id).map_err(|e| SynkError::io(lock.display(), e))?;

        let shared = Arc::new(Mutex::new(record));
        let stop = Arc::new(AtomicBool::new(false));
        let heartbeat = {
            let (shared, stop, dir) = (shared.clone(), stop.clone(), dir.clone());
            std::thread::spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    std::thread::sleep(Duration::from_millis(500));
                    let mut record = shared.lock().expect("op record lock");
                    if now().saturating_sub(record.heartbeat_at) >= HEARTBEAT.as_secs() {
                        record.heartbeat_at = now();
                        let _ = write_record(&dir, &record);
                    }
                }
            })
        };
        Ok(Op {
            dir,
            lock,
            record: shared,
            stop,
            heartbeat: Some(heartbeat),
            seq,
        })
    }

    /// Removes finished fetch and dataset records older than the retention
    /// period.
    fn prune(&self) {
        let cutoff = now().saturating_sub(RETENTION_DAYS * 86_400);
        for op in self.list().unwrap_or_default() {
            if op.kind != OpKind::Run && op.done && op.started_at < cutoff {
                let _ = fs::remove_dir_all(self.dir(&op.id));
            }
        }
    }

    /// Gives numbers to runs and datasets made before runs had records:
    /// oldest first. Does nothing once any run has a number.
    pub fn migrate(&self, artifacts: &Artifacts) -> Result<()> {
        let seq: BTreeMap<String, u64> = read_json(&self.seq_path()).unwrap_or_default();
        if seq.contains_key(OpKind::Run.prefix()) || !self.list_kind(OpKind::Run)?.is_empty() {
            return Ok(());
        }
        let mut datasets = artifacts.list(crate::dataset::KIND)?;
        datasets.sort_by_key(|m| m.created_at);
        for d in &datasets {
            self.dataset_number(&d.key, true)?;
        }
        let mut runs = artifacts.list(crate::train::TRAIN_KIND)?;
        if runs.is_empty() {
            return Ok(());
        }
        runs.sort_by_key(|m| m.created_at);
        for m in &runs {
            let n = self.next_number(OpKind::Run, 0)?;
            let label =
                crate::ids::model_label(m.summary["base_model"].as_str().unwrap_or("model"));
            let dataset = m
                .inputs
                .iter()
                .find_map(|i| i.strip_prefix("dataset:"))
                .and_then(|k| self.dataset_number(k, false).ok().flatten().map(|n| (k, n)));
            let record = OpRecord {
                v: 1,
                id: format!("run:{n}-{label}"),
                kind: OpKind::Run,
                n,
                name: None,
                state: State::Succeeded,
                done: true,
                attempt: 1,
                started_at: m.created_at,
                finished_at: Some(m.created_at),
                heartbeat_at: m.created_at,
                pid: 0,
                settings: json!({ "params": m.params }),
                settings_hash: String::new(),
                dataset: dataset.map(|(_, n)| format!("dataset:{n}")),
                key: Some(m.key.clone()),
                steps: Vec::new(),
                progress: None,
                result: None,
                error: None,
                reason: Some("made before runs had numbers".to_owned()),
                same_as: None,
            };
            let dir = self.dir(&record.id);
            fs::create_dir_all(&dir).map_err(|e| SynkError::io(dir.display(), e))?;
            write_record(&dir, &record)?;
        }
        Ok(())
    }
}

fn not_found(id: &str) -> SynkError {
    SynkError::new(ErrorCode::JobNotFound, format!("nothing matches `{id}`"))
        .with_next("synk run list")
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Option<T> {
    serde_json::from_str(&fs::read_to_string(path).ok()?).ok()
}

fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir).map_err(|e| SynkError::io(dir.display(), e))?;
    }
    let text = serde_json::to_string_pretty(value).expect("json serializes");
    fs::write(path, text).map_err(|e| SynkError::io(path.display(), e))
}

fn write_record(dir: &Path, record: &OpRecord) -> Result<()> {
    let path = dir.join("op.json");
    let tmp = dir.join("op.json.tmp");
    let text = serde_json::to_string_pretty(record).expect("op record serializes");
    fs::write(&tmp, text).map_err(|e| SynkError::io(tmp.display(), e))?;
    // On Windows a reader may hold the file open for a moment.
    for _ in 0..20 {
        if fs::rename(&tmp, &path).is_ok() {
            return Ok(());
        }
        std::thread::sleep(Duration::from_millis(25));
    }
    fs::rename(&tmp, &path).map_err(|e| SynkError::io(path.display(), e))
}

/// The running operation, owned by the command that started it.
pub struct Op {
    dir: PathBuf,
    lock: PathBuf,
    record: Arc<Mutex<OpRecord>>,
    stop: Arc<AtomicBool>,
    heartbeat: Option<std::thread::JoinHandle<()>>,
    seq: u64,
}

impl Op {
    pub fn id(&self) -> String {
        self.record.lock().expect("op record lock").id.clone()
    }

    pub fn record(&self) -> OpRecord {
        self.record.lock().expect("op record lock").clone()
    }

    /// The file whose existence asks the operation to stop. The trainer gets
    /// its path too.
    pub fn cancel_file(&self) -> PathBuf {
        self.dir.join("cancel")
    }

    /// A folder inside the operation for a trainer task's request and log.
    pub fn work_dir(&self, name: &str) -> Result<PathBuf> {
        let dir = self.dir.join("work").join(name);
        fs::create_dir_all(&dir).map_err(|e| SynkError::io(dir.display(), e))?;
        Ok(dir)
    }

    pub fn cancel_requested(&self) -> bool {
        self.cancel_file().exists()
    }

    /// Fails with `CANCELLED` if a cancel was requested.
    pub fn check_cancel(&self) -> Result<()> {
        if self.cancel_requested() {
            Err(SynkError::new(
                ErrorCode::Cancelled,
                format!("{} was cancelled", self.id()),
            ))
        } else {
            Ok(())
        }
    }

    /// Changes the record and saves it.
    pub fn update(&self, change: impl FnOnce(&mut OpRecord)) {
        let mut record = self.record.lock().expect("op record lock");
        change(&mut record);
        let _ = write_record(&self.dir, &record);
    }

    /// Appends an event (progress or log) and tracks progress.
    pub fn event(&mut self, step: &str, event: &Value) {
        self.seq += 1;
        let mut line = event.clone();
        line["seq"] = json!(self.seq);
        line["step"] = json!(step);
        line["at"] = json!(now());
        if let Ok(mut file) = OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.dir.join("events.jsonl"))
        {
            let _ = writeln!(file, "{line}");
        }
        if event["event"] == "progress" {
            let progress = Progress {
                step: step.to_owned(),
                stage: event["stage"].as_str().unwrap_or(step).to_owned(),
                completed: event["step"]
                    .as_u64()
                    .or_else(|| event["completed"].as_u64())
                    .unwrap_or(0),
                total: event["total"].as_u64().unwrap_or(0),
                unit: event["unit"].as_str().unwrap_or("").to_owned(),
            };
            // Writing on every progress event would be slow; the heartbeat
            // persists it within a few seconds.
            self.record.lock().expect("op record lock").progress = Some(progress);
        }
    }

    pub fn step(&self, step: &str, status: &str, seconds: Option<u64>, detail: Value) {
        self.update(|r| {
            r.steps.retain(|s| s.step != step);
            r.steps.push(StepRecord {
                step: step.to_owned(),
                status: status.to_owned(),
                seconds,
                detail,
            });
            r.progress = None;
        });
    }

    /// Ends the operation and releases the lock.
    pub fn finish(mut self, outcome: std::result::Result<Value, &SynkError>) -> OpRecord {
        self.update(|r| {
            r.done = true;
            r.finished_at = Some(now());
            r.progress = None;
            match outcome {
                Ok(result) => {
                    r.state = State::Succeeded;
                    r.result = Some(result);
                }
                Err(err) => {
                    r.state = if err.code == ErrorCode::Cancelled {
                        State::Cancelled
                    } else {
                        State::Failed
                    };
                    r.error = Some(serde_json::to_value(err).expect("error serializes"));
                }
            }
        });
        self.shutdown();
        self.record()
    }

    fn shutdown(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.heartbeat.take() {
            let _ = handle.join();
        }
        let _ = fs::remove_file(&self.lock);
    }
}

impl Drop for Op {
    /// An operation dropped without `finish` (a panic) is marked failed.
    fn drop(&mut self) {
        if self.heartbeat.is_some() {
            self.update(|r| {
                if !r.done {
                    r.state = State::Failed;
                    r.done = true;
                    r.finished_at = Some(now());
                    r.reason = Some("synk stopped unexpectedly".to_owned());
                }
            });
            self.shutdown();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn new_run<'a>(label: &'a str, name: Option<&'a str>) -> NewOp<'a> {
        NewOp {
            kind: OpKind::Run,
            label,
            name,
            settings: json!({}),
            settings_hash: "h".into(),
            n: None,
        }
    }

    #[test]
    fn one_at_a_time_numbered_and_records_survive() {
        let dir = tempfile::tempdir().unwrap();
        let ops = Ops::new(dir.path().to_path_buf());
        let mut op = ops.start(new_run("e5-small", None)).unwrap();
        let id = op.id();
        assert_eq!(id, "run:1-e5-small");
        assert_eq!(
            ops.start(new_run("x", None)).err().unwrap().code,
            ErrorCode::JobRunning
        );
        assert_eq!(ops.get(Some("current")).unwrap().id, id);

        op.event(
            "dataset",
            &json!({"event": "progress", "stage": "generate", "step": 3, "total": 9}),
        );
        op.step("sync", "ran", Some(1), Value::Null);
        assert!(!op.cancel_requested());
        ops.cancel(None).unwrap();
        assert_eq!(op.check_cancel().unwrap_err().code, ErrorCode::Cancelled);
        let record = op.finish(Ok(json!({"run": "run:1"})));
        assert_eq!(record.state, State::Succeeded);

        let read = ops.get(Some("1")).unwrap();
        assert!(read.done);
        assert_eq!(read.steps[0].step, "sync");
        assert_eq!(ops.get(Some("run:1")).unwrap().id, id);
        assert_eq!(ops.get(Some("run:1-e5-small")).unwrap().id, id);
        let (events, more) = ops.events(&id, 0, 10).unwrap();
        assert_eq!(events.len(), 1);
        assert!(!more);
        assert!(ops.running().unwrap().is_none());

        // Numbers count up per kind; names are unique and resolve.
        let second = ops.start(new_run("bge-m3", Some("bge"))).unwrap();
        assert_eq!(second.id(), "run:2-bge");
        second.finish(Ok(Value::Null));
        assert_eq!(ops.get(Some("bge")).unwrap().n, 2);
        assert_eq!(
            ops.start(new_run("x", Some("bge"))).err().unwrap().code,
            ErrorCode::RunNameTaken
        );
        let fetch = ops
            .start(NewOp {
                kind: OpKind::Fetch,
                label: "",
                name: None,
                settings: json!({}),
                settings_hash: String::new(),
                n: None,
            })
            .unwrap();
        assert_eq!(fetch.id(), "fetch:1");
        fetch.finish(Ok(Value::Null));
        assert_eq!(ops.get(Some("fetch:1")).unwrap().kind, OpKind::Fetch);
        assert_eq!(ops.get(None).unwrap().id, "fetch:1");
    }

    #[test]
    fn a_stale_heartbeat_reads_as_interrupted_and_resumes() {
        let dir = tempfile::tempdir().unwrap();
        let ops = Ops::new(dir.path().to_path_buf());
        let mut op = ops.start(new_run("e5-small", None)).unwrap();
        op.event("train", &json!({"event": "log", "message": "x"}));
        let id = op.id();
        // Simulate a killed process: stop the heartbeat, then age the record.
        op.stop.store(true, Ordering::Relaxed);
        op.heartbeat.take().unwrap().join().unwrap();
        op.update(|r| r.heartbeat_at = now() - STALE_AFTER - 10);
        std::mem::forget(op);
        let read = ops.get(Some(&id)).unwrap();
        assert_eq!(read.state, State::Interrupted);
        assert!(read.unfinished());
        assert!(ops.running().unwrap().is_none());

        let mut again = ops.resume(read).unwrap();
        again.event("train", &json!({"event": "log", "message": "y"}));
        let record = again.finish(Ok(Value::Null));
        assert_eq!((record.id.as_str(), record.attempt), (id.as_str(), 2));
        let (events, _) = ops.events(&id, 0, 10).unwrap();
        assert_eq!(events[1]["seq"], 2, "numbering continues");
    }

    #[test]
    fn datasets_get_stable_numbers() {
        let dir = tempfile::tempdir().unwrap();
        let ops = Ops::new(dir.path().to_path_buf());
        assert_eq!(ops.dataset_number("k1", false).unwrap(), None);
        assert_eq!(ops.dataset_number("k1", true).unwrap(), Some(1));
        assert_eq!(ops.dataset_number("k2", true).unwrap(), Some(2));
        assert_eq!(ops.dataset_number("k1", true).unwrap(), Some(1));
        assert_eq!(ops.dataset_key(2).as_deref(), Some("k2"));
    }

    #[test]
    fn old_runs_get_numbers_oldest_first() {
        use crate::artifact::Manifest;
        let dir = tempfile::tempdir().unwrap();
        let artifacts = Artifacts::new(dir.path().join("artifacts"));
        for (key, model, at) in [
            ("bbbb", "BAAI/bge-m3", 20),
            ("aaaa", "intfloat/multilingual-e5-small", 10),
        ] {
            let mut m = Manifest::new(
                crate::train::TRAIN_KIND,
                key,
                json!({}),
                vec![],
                json!({"base_model": model}),
            );
            m.created_at = at;
            artifacts
                .begin(crate::train::TRAIN_KIND, key, true)
                .unwrap();
            artifacts.commit(&m).unwrap();
        }
        let ops = Ops::new(dir.path().to_path_buf());
        ops.migrate(&artifacts).unwrap();
        ops.migrate(&artifacts).unwrap();
        let runs = ops.list_kind(OpKind::Run).unwrap();
        let ids: Vec<&str> = runs.iter().map(|r| r.id.as_str()).collect();
        assert_eq!(ids, ["run:2-bge-m3", "run:1-e5-small"]);
        assert_eq!(ops.run_for_key("aaaa").unwrap().unwrap().n, 1);
    }
}
