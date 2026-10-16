//! Running a preview's steps as an operation: sync, dataset, train, eval.
//!
//! Downloads are not done here: `synk fetch` does them, and `train` and
//! `dataset` refuse to start while the preview still lists one. A cached
//! step is recorded as `cached`; a step that runs is recorded with its
//! duration. Every progress and log event goes to the operation's event log
//! and to `render` (the CLI's progress display) as it happens. Cancellation
//! is checked between steps, between LLM calls, and by the trainer at every
//! training step.

use std::time::Instant;

use serde_json::{Value, json};
use synk_core::{ErrorCode, Result, SynkError};

use crate::advice::{self, Advice};
use crate::artifact::Manifest;
use crate::dataset;
use crate::generator;
use crate::graph::{self, Action, Preview, Step, Workspace};
use crate::ids;
use crate::ops::{Op, OpKind, State};
use crate::sync;
use crate::train::{self, Verdict};

/// What a run (or a dataset build) produced.
pub struct Outcome {
    pub dataset: Option<(String, Manifest)>,
    pub run: Option<(String, Manifest)>,
    pub verdict: Option<Verdict>,
    pub advice: Vec<Advice>,
    pub sync: Option<sync::Outcome>,
    /// The general check's verdict per set (`en`, `vi`).
    pub general: Vec<(String, Verdict)>,
    /// Document text sent off this machine: host and chunk count.
    pub egress: Option<Value>,
    /// An earlier run with exactly the same inputs, whose results these are.
    pub same_as: Option<String>,
}

impl Outcome {
    /// The result saved in the record and printed with `--json`.
    pub fn to_json(&self) -> Value {
        json!({
            "dataset": self.dataset.as_ref().map(|(id, _)| id),
            "dataset_summary": self.dataset.as_ref().map(|(_, m)| &m.summary),
            "run": self.run.as_ref().map(|(id, _)| id),
            "key": self.run.as_ref().map(|(_, m)| &m.key),
            "verdict": self.verdict,
            "general_check": self.general.iter().map(|(k, v)| (k.clone(), json!(v))).collect::<serde_json::Map<_, _>>(),
            "advice": self.advice,
            "sync": self.sync.as_ref().map(|s| json!({
                "totals": s.totals,
                "removed_sources": s.removed_sources,
                "warnings": s.warnings,
                "skipped": s.sources.iter().flat_map(|r| r.skipped.clone()).collect::<Vec<_>>(),
            })),
            "egress": self.egress,
            "same_as": self.same_as,
            "next": self.next(),
        })
    }

    /// Export the run when it is better; otherwise look at why.
    pub fn next(&self) -> Option<String> {
        if let Some(next) = advice::next(&self.advice) {
            return Some(next);
        }
        let (id, _) = self.run.as_ref()?;
        Some(match self.verdict.as_ref().map(|v| v.outcome.as_str()) {
            Some("better") => ids::export_command(id),
            _ => format!("synk run inspect {id}"),
        })
    }
}

type Render<'a> = &'a mut dyn FnMut(Step, &Value);

fn emit(op: &mut Op, render: &mut Render<'_>, step: Step, event: &Value) {
    op.event(step.as_str(), event);
    render(step, event);
}

/// Runs `preview`'s steps up to its target. `render` sees every event as it
/// happens.
pub fn run(
    ws: &Workspace,
    preview: &Preview,
    op: &mut Op,
    mut render: Render<'_>,
) -> Result<Outcome> {
    if let Some(blocker) = preview.blockers.first() {
        return Err(blocker.clone());
    }
    let mut out = Outcome {
        dataset: None,
        run: None,
        verdict: None,
        advice: Vec::new(),
        sync: None,
        general: Vec::new(),
        egress: None,
        same_as: None,
    };
    let force = preview.options.force;
    let render = &mut render;
    let kind = op.record().kind;

    for sp in &preview.steps {
        op.check_cancel()?;
        let step = sp.step;
        if sp.action == Action::Skipped {
            continue;
        }
        match step {
            // What `synk fetch` does. A preview that still lists a download
            // is refused before an operation starts; this is a safeguard.
            Step::Trainer | Step::Models | Step::Generator => {
                if sp.action == Action::Run {
                    return Err(SynkError::new(
                        ErrorCode::ConsentRequired,
                        format!("{} still needs a download", step.as_str()),
                    )
                    .with_next("synk train --dry-run"));
                }
                op.step(step.as_str(), "cached", None, json!({}));
                continue;
            }
            Step::Eval => continue,
            _ => {}
        }
        emit(
            op,
            render,
            step,
            &json!({"event": "step", "status": "start", "action": sp.action}),
        );
        let started = Instant::now();
        let elapsed = || Some(started.elapsed().as_secs());
        match step {
            Step::Sync => {
                let mut store = ws.open_store()?;
                let outcome = sync::run(&ws.sources, &mut store)?;
                op.step(
                    step.as_str(),
                    if outcome.changed() { "ran" } else { "cached" },
                    elapsed(),
                    json!(outcome.totals),
                );
                out.sync = Some(outcome);
            }
            Step::Dataset => {
                let store = ws.open_store()?;
                let artifacts = ws.artifacts();
                let (dataset_cfg, chunking, generator) =
                    graph::dataset_inputs(ws, &preview.model, preview.options.sample)?;
                let prepared = dataset::prepare(
                    &store,
                    &dataset_cfg,
                    ws.cfg.generator.queries_per_chunk,
                    &chunking,
                    &generator,
                )?;
                let outcome = dataset::build(
                    &store,
                    &artifacts,
                    prepared,
                    &generator,
                    force == Some(Step::Dataset),
                    &mut |done, total| {
                        emit(
                            op,
                            render,
                            step,
                            &json!({
                                "event": "progress", "stage": "generate",
                                "step": done, "total": total, "unit": "chunks",
                            }),
                        );
                        !op.cancel_requested()
                    },
                )?;
                let s = &outcome.manifest.summary;
                let calls = s["chunks"]
                    .as_u64()
                    .unwrap_or(0)
                    .saturating_sub(s["cached_chunks"].as_u64().unwrap_or(0));
                if !outcome.reused && calls > 0 {
                    graph::record_speed(
                        &graph::llm_speed_key(&ws.cfg),
                        started.elapsed().as_secs_f64() / calls as f64,
                    );
                    if !generator::is_local(&ws.cfg.generator.base_url) {
                        out.egress = Some(json!({
                            "host": ws.cfg.generator.base_url,
                            "chunks": calls,
                        }));
                    }
                }
                out.advice.extend(advice::dataset(s, &ws.cfg));
                let id = ids::number_dataset(&artifacts, &outcome.manifest)?;
                let key = outcome.manifest.key.clone();
                op.update(|r| {
                    r.dataset = Some(id.clone());
                    if kind == OpKind::Dataset {
                        r.key = Some(key);
                    }
                });
                op.step(
                    step.as_str(),
                    if outcome.reused { "cached" } else { "ran" },
                    (!outcome.reused).then(|| started.elapsed().as_secs()),
                    json!({"dataset": id, "queries": s["queries"]}),
                );
                out.dataset = Some((id, outcome.manifest));
            }
            Step::Train => {
                let artifacts = ws.artifacts();
                let Some((_, dataset)) = out.dataset.clone() else {
                    return Err(SynkError::new(
                        ErrorCode::DatasetNotFound,
                        "no dataset to train on",
                    ));
                };
                let dataset_dir = artifacts.dir(dataset::KIND, &dataset.key);
                let env = ws.trainer()?;
                let opts = train::Options {
                    resume: true,
                    force: force == Some(Step::Train),
                    device: None,
                    reranker: ws.reranker()?,
                    stop_file: Some(op.cancel_file()),
                    general_check: ws.cfg.eval.general_check,
                    force_eval: force == Some(Step::Eval),
                };
                let device = ws
                    .cfg
                    .trainer
                    .device
                    .clone()
                    .or_else(|| env.describe["devices"][0].as_str().map(str::to_owned))
                    .unwrap_or_else(|| "cpu".to_owned());
                let outcome = train::run(
                    &artifacts,
                    &dataset,
                    &dataset_dir,
                    &preview.model,
                    &ws.cfg.trainer,
                    &env,
                    &opts,
                    &mut |e| emit(op, render, step, e),
                )?;
                if !outcome.train_reused {
                    let steps = outcome.train.summary["steps"].as_u64().unwrap_or(0);
                    if steps > 0 {
                        graph::record_speed(
                            &graph::train_speed_key(&preview.model, &device),
                            started.elapsed().as_secs_f64() / steps as f64,
                        );
                    }
                }
                // Another run made this model already: say so, and keep
                // this run's own ID.
                let key = outcome.train.key.clone();
                let me = op.id();
                let earlier = ws
                    .ops()
                    .run_for_key(&key)?
                    .filter(|r| r.id != me && r.state == State::Succeeded);
                out.same_as = earlier.map(|r| r.id);
                let id = if kind == OpKind::Run {
                    let same_as = out.same_as.clone();
                    op.update(|r| {
                        r.key = Some(key);
                        r.same_as = same_as;
                    });
                    me
                } else {
                    ids::id(&artifacts, ids::Kind::Run, &outcome.train)?
                };
                op.step(
                    "train",
                    if outcome.train_reused {
                        "cached"
                    } else {
                        "ran"
                    },
                    (!outcome.train_reused).then(|| started.elapsed().as_secs()),
                    json!({"run": id}),
                );
                op.step(
                    "eval",
                    "ran",
                    None,
                    json!({"outcome": outcome.verdict.outcome}),
                );
                out.general = advice::general_verdicts(&outcome.eval.summary);
                out.advice.extend(advice::train(
                    &outcome.train.summary,
                    &outcome.verdict,
                    &out.general,
                    &ws.cfg,
                ));
                out.verdict = Some(outcome.verdict);
                out.run = Some((id, outcome.train));
            }
            Step::Trainer | Step::Models | Step::Generator | Step::Eval => {}
        }
        if preview.options.target == step {
            break;
        }
    }
    let mut seen = std::collections::HashSet::new();
    out.advice.retain(|a| seen.insert(a.code));
    Ok(out)
}
