//! `synk fetch`: downloads, by name, and nothing else.
//!
//! Running `synk fetch NAME…` is the user's consent to download exactly
//! those items (ADR 0002), so names are exact: a model name carries the
//! start of the commit it was pinned to when synk printed it, and `fetch`
//! refuses if that commit is not what it would download. Names:
//! - a catalog model or reranker ID, or `hf:org/name`, optionally `@<commit>`;
//! - `ollama:TAG`, a model your local Ollama downloads;
//! - `general-check`, the public test sets of the general check;
//! - `trainer`, the Python training environment.
//!
//! Items already on this machine are reported `present` and skipped.
//! Hugging Face and Ollama downloads resume where they stopped.

use std::path::Path;

use serde_json::{Value, json};
use synk_core::{ErrorCode, Result, SynkError};

use crate::catalog::{self, ModelSpec};
use crate::config::GeneratorConfig;
use crate::generator;
use crate::graph::{self, Fetch, GENERAL_CHECK, TRAINER, Workspace};
use crate::hub::{self, RepoType};
use crate::ops::Op;
use crate::sidecar::{self, Backend, PROTOCOL_VERSION};
use crate::train;

/// What fetching one name does.
#[derive(Debug, Clone)]
enum Work {
    Trainer,
    Hub(Vec<Fetch>),
    Ollama(GeneratorConfig),
}

/// One item, resolved: its exact name, whether it is here already, and
/// its size when known.
#[derive(Debug, Clone)]
pub struct Item {
    pub name: String,
    pub what: String,
    pub present: bool,
    pub bytes: Option<u64>,
    work: Work,
}

impl Item {
    pub fn to_json(&self) -> Value {
        json!({
            "name": self.name,
            "what": self.what,
            "present": self.present,
            "bytes": self.bytes,
        })
    }
}

fn bad_name(name: &str, why: &str) -> SynkError {
    SynkError::new(
        ErrorCode::InvalidArgs,
        format!("`{name}` can't be fetched: {why}"),
    )
    .with_hint("names look like multilingual-e5-small, hf:BAAI/bge-m3@5617a9f, ollama:qwen3:8b, general-check or trainer; synk train --dry-run lists the ones training needs")
}

/// A model name: the catalog's model or reranker, or `hf:org/name`.
fn model_spec(base: &str, root: &Path) -> Result<ModelSpec> {
    match catalog::resolve(base, root) {
        Ok(spec) => Ok(spec),
        Err(model_err) => match catalog::resolve_reranker(base, root) {
            Ok(r) => Ok(ModelSpec {
                name_or_path: r.name_or_path,
                revision: r.revision,
                query_prompt: None,
                document_prompt: None,
                catalog_id: r.catalog_id,
            }),
            Err(_) => Err(model_err),
        },
    }
}

/// Pins a model and checks the pin against `wanted`, the commit start the
/// name carried.
fn pin_model(name: &str, spec: &ModelSpec, wanted: Option<&str>) -> Result<Item> {
    if Path::new(&spec.name_or_path).is_dir() {
        return Err(bad_name(name, "it is a folder on this machine already"));
    }
    // The catalog pins its models: the name must agree.
    if let (Some(want), Some(_), Some(rev)) = (wanted, &spec.catalog_id, &spec.revision)
        && !rev.starts_with(want)
    {
        return Err(SynkError::new(
            ErrorCode::InvalidArgs,
            format!(
                "`{name}` asks for commit {want}, but this synk's catalog pins {}",
                &rev[..rev.len().min(7)]
            ),
        )
        .with_next("synk train --dry-run"));
    }
    let mut unpinned = spec.clone();
    // A custom model's `@rev` is checked against the branch head first.
    if spec.catalog_id.is_none() && wanted.is_some() {
        unpinned.revision = None;
    }
    let (mut pinned, mut info, mut err) = graph::pin(&unpinned, true);
    if let (Some(want), Some(rev)) = (wanted, pinned.revision.clone())
        && !rev.starts_with(want)
    {
        // Not the branch head: ask for that commit itself.
        let mut exact = spec.clone();
        exact.revision = Some(want.to_owned());
        (pinned, info, err) = graph::pin(&exact, true);
    }
    if let Some(err) = err {
        return Err(err);
    }
    let revision = pinned.revision.clone().unwrap_or_else(|| "main".to_owned());
    let present = hub::snapshot(RepoType::Model, &pinned.name_or_path, &revision).is_some();
    let has_safetensors = info
        .as_ref()
        .is_none_or(|i| i.files.iter().any(|(p, _)| p.ends_with(".safetensors")));
    let exact_name = graph::fetch_name(&pinned);
    Ok(Item {
        name: exact_name.clone(),
        what: pinned.name_or_path.clone(),
        present,
        bytes: info.as_ref().map(hub::RepoInfo::bytes),
        work: Work::Hub(vec![Fetch {
            name: exact_name,
            repo: pinned.name_or_path.clone(),
            revision,
            repo_type: "model".into(),
            ignore_patterns: hub::ignore_patterns(has_safetensors)
                .into_iter()
                .map(str::to_owned)
                .collect(),
            allow_patterns: Vec::new(),
        }]),
    })
}

fn general_check() -> Item {
    let mut fetches = Vec::new();
    let mut bytes = Some(0u64);
    let mut present = true;
    for set in train::GENERAL_SETS {
        if hub::snapshot(RepoType::Dataset, set.repo, set.revision).is_some() {
            continue;
        }
        present = false;
        let size = hub::info(RepoType::Dataset, set.repo, set.revision)
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
            });
        bytes = bytes.zip(size).map(|(a, b)| a + b);
        fetches.push(Fetch {
            name: GENERAL_CHECK.into(),
            repo: set.repo.to_owned(),
            revision: set.revision.to_owned(),
            repo_type: "dataset".into(),
            ignore_patterns: Vec::new(),
            allow_patterns: set.allow_patterns.iter().map(|s| (*s).to_owned()).collect(),
        });
    }
    Item {
        name: GENERAL_CHECK.into(),
        what: "the general check's public test sets".into(),
        present,
        bytes: if present { Some(0) } else { bytes },
        work: Work::Hub(fetches),
    }
}

fn ollama(ws: &Workspace, tag: &str) -> Result<Item> {
    let mut cfg = ws.cfg.generator.clone();
    cfg.model = tag.to_owned();
    if !generator::is_local(&cfg.base_url) {
        return Err(SynkError::new(
            ErrorCode::GeneratorUnavailable,
            format!(
                "ollama:{tag} needs an Ollama on this machine; generator.base_url is {}",
                cfg.base_url
            ),
        ));
    }
    let probe = generator::probe(&cfg);
    if probe.reachable == Some(false) {
        return Err(SynkError::new(
            ErrorCode::GeneratorUnavailable,
            format!("can't reach the LLM server: {}", probe.message),
        )
        .with_action(synk_core::ErrorAction::run(
            "ollama serve",
            synk_core::Applicability::Review,
            "start Ollama",
        )));
    }
    let present = probe.model_ready == Some(true);
    if !present && !probe.ollama {
        return Err(SynkError::new(
            ErrorCode::GeneratorUnavailable,
            format!(
                "{} is not Ollama, so synk can't have it download `{tag}`",
                cfg.base_url
            ),
        )
        .with_hint("load the model in your LLM server yourself"));
    }
    Ok(Item {
        name: graph::ollama_name(tag),
        what: format!("{tag} (in your Ollama)"),
        present,
        bytes: if present {
            Some(0)
        } else {
            generator::ollama_size(tag)
        },
        work: Work::Ollama(cfg),
    })
}

/// Resolves fetch names: checks each one and finds what is present.
/// Contacts Hugging Face and Ollama for versions and sizes (sending only
/// model names); downloads nothing.
pub fn resolve(ws: &Workspace, names: &[String]) -> Result<Vec<Item>> {
    let mut out: Vec<Item> = Vec::new();
    for raw in names {
        let name = raw.trim();
        let item = if name == TRAINER {
            Item {
                name: TRAINER.into(),
                what: "the Python trainer (PyTorch, sentence-transformers)".into(),
                present: ws.trainer().is_ok(),
                bytes: None,
                work: Work::Trainer,
            }
        } else if name == GENERAL_CHECK {
            general_check()
        } else if let Some(tag) = name.strip_prefix("ollama:") {
            if tag.is_empty() {
                return Err(bad_name(name, "it has no tag"));
            }
            ollama(ws, tag)?
        } else {
            // `hf:org/name@rev` is split by the catalog; a catalog ID's
            // `@rev` is split here.
            let (base, wanted) = match name.rsplit_once('@') {
                Some((b, r)) if !r.is_empty() && !r.contains('/') => (b, Some(r)),
                _ => (name, None),
            };
            if base.is_empty() {
                return Err(bad_name(name, "it has no model"));
            }
            let spec = model_spec(base, &ws.root).map_err(|e| bad_name(name, &e.message))?;
            pin_model(name, &spec, wanted)?
        };
        if !out.iter().any(|o| o.name == item.name) {
            out.push(item);
        }
    }
    // The trainer first: Hugging Face downloads run inside it.
    out.sort_by_key(|i| !matches!(i.work, Work::Trainer));
    Ok(out)
}

type Render<'a> = &'a mut dyn FnMut(&str, &Value);

fn emit(op: &mut Op, render: &mut Render<'_>, step: &str, event: &Value) {
    op.event(step, event);
    render(step, event);
}

/// Downloads every item that isn't present, as operation `op`. Returns one
/// entry per item: its name and `fetched` or `present`.
pub fn run(
    ws: &Workspace,
    items: &[Item],
    backend: Option<Backend>,
    op: &mut Op,
    mut render: Render<'_>,
) -> Result<Vec<Value>> {
    let render = &mut render;
    let mut out = Vec::new();
    for item in items {
        op.check_cancel()?;
        if item.present {
            op.step(&item.name, "present", None, json!({}));
            out.push(json!({"name": item.name, "status": "present"}));
            continue;
        }
        let started = std::time::Instant::now();
        let step = item.name.as_str();
        emit(
            op,
            render,
            step,
            &json!({"event": "log", "message": format!("Downloading {}", item.what)}),
        );
        match &item.work {
            Work::Trainer => {
                sidecar::setup(backend, false, ws.uv.as_deref(), &mut |m| {
                    emit(op, render, step, &json!({"event": "log", "message": m}));
                })?;
            }
            Work::Hub(fetches) => {
                let env = ws.trainer().map_err(|e| {
                    e.with_hint("models are downloaded with the trainer: fetch `trainer` too")
                })?;
                for (i, f) in fetches.iter().enumerate() {
                    let work = op.work_dir(&format!("{}-{i}", crate::ids::slug(&f.name)))?;
                    let request = json!({
                        "v": PROTOCOL_VERSION,
                        "task": "fetch",
                        "repo": f.repo,
                        "revision": f.revision,
                        "repo_type": f.repo_type,
                        "ignore_patterns": f.ignore_patterns,
                        "allow_patterns": f.allow_patterns,
                        "stop_file": op.cancel_file(),
                    });
                    env.run(&request, &work, &mut |e| emit(op, render, step, e))?;
                }
            }
            Work::Ollama(cfg) => {
                let mut last = u64::MAX;
                generator::pull(cfg, &mut |done, total| {
                    // A line every 2% is plenty.
                    let pct = (done * 50).checked_div(total).unwrap_or(0);
                    if pct != last {
                        last = pct;
                        emit(
                            op,
                            render,
                            step,
                            &json!({
                                "event": "progress", "stage": "download",
                                "step": done, "total": total, "unit": "bytes",
                            }),
                        );
                    }
                })?;
            }
        }
        op.step(
            &item.name,
            "fetched",
            Some(started.elapsed().as_secs()),
            json!({"bytes": item.bytes}),
        );
        out.push(json!({"name": item.name, "status": "fetched", "bytes": item.bytes}));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_say_what_is_wrong() {
        let root = Path::new(".");
        assert_eq!(
            model_spec("nope", root).unwrap_err().code,
            ErrorCode::ModelNotFound
        );
        assert_eq!(
            model_spec("bge-reranker-v2-m3", root).unwrap().name_or_path,
            "BAAI/bge-reranker-v2-m3"
        );
        let spec = model_spec("multilingual-e5-small", root).unwrap();
        let err = pin_model("multilingual-e5-small@0000000", &spec, Some("0000000")).unwrap_err();
        assert_eq!(err.code, ErrorCode::InvalidArgs, "{err:?}");
        assert!(err.message.contains("catalog pins"), "{}", err.message);
    }
}
