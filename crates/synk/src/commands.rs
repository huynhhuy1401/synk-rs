use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use serde_json::{Value, json};
use synk_connectors::{SUPPORTED_EXTENSIONS, is_supported};
use synk_core::{ErrorCode, Result, SourceSpec, SynkError};

use crate::config::{Project, existing_path};
use crate::human;
use crate::output::{Output, show_path};

pub fn init(dir: &Path) -> Result<Output> {
    let project = Project::init(dir)?;
    let config = project.config_path.display().to_string();
    Ok(Output {
        human: format!(
            "Created {}\nNext: synk source add <path>",
            show_path(&project.config_path)
        ),
        data: json!({ "config": config }),
    })
}

/// Validates every path before saving, so a bad path adds nothing.
pub fn add(
    mut project: Project,
    paths: &[PathBuf],
    name: Option<&str>,
    exclude: &[String],
) -> Result<Output> {
    if name.is_some() && paths.len() > 1 {
        return Err(SynkError::new(
            ErrorCode::InvalidArgs,
            "--name can only be used with a single path",
        ));
    }

    let mut added = Vec::new();
    for path in paths {
        let abs = existing_path(path)?;
        let is_file = abs.is_file();
        if is_file && !is_supported(&abs) {
            return Err(SynkError::new(
                ErrorCode::UnsupportedFormat,
                format!("{} is not a supported file type", abs.display()),
            )
            .with_hint(format!(
                "supported extensions: {}",
                SUPPORTED_EXTENSIONS.join(", ")
            )));
        }
        check_overlap(&project, &abs)?;

        let base = name.map_or_else(|| default_name(&abs), str::to_owned);
        let spec = SourceSpec::folder(project.unique_name(&base), abs, exclude.to_vec());
        project.add_source(&spec)?;
        added.push(json!({
            "name": spec.name,
            "path": spec.path.display().to_string(),
            "kind": kind(&spec.path),
        }));
    }
    project.save()?;

    let mut human = String::new();
    for source in &added {
        let _ = writeln!(
            human,
            "Added {} ({}): {}",
            source["name"].as_str().unwrap_or_default(),
            source["kind"].as_str().unwrap_or_default(),
            show_path(Path::new(source["path"].as_str().unwrap_or_default()))
        );
    }
    human.push_str("Next: synk train --dry-run");
    Ok(Output {
        human,
        data: json!({ "added": added, "next": "synk train --dry-run" }),
    })
}

/// `synk source sync`: reads the sources into the store now.
pub fn sync(project: &Project) -> Result<Output> {
    let ops = synk_pipeline::ops::Ops::new(project.data_dir()?);
    if let Some(running) = ops.running()? {
        return Err(SynkError::new(
            ErrorCode::JobRunning,
            format!("{} is running and syncs by itself", running.id),
        )
        .with_next("synk wait"));
    }
    let mut store = project.open_store()?;
    let outcome = synk_pipeline::sync::run(&project.resolved_sources(), &mut store)?;
    let t = &outcome.totals;
    let mut human = format!(
        "Documents: {} new, {} changed, {} deleted, {} unchanged\n",
        t.added, t.updated, t.deleted, t.unchanged
    );
    crate::work::render_skipped(&mut human, &outcome);
    let next = "synk train --dry-run";
    let _ = write!(human, "Next: {next}");
    Ok(Output {
        human,
        data: json!({
            "sync": {
                "totals": outcome.totals,
                "removed_sources": outcome.removed_sources,
                "warnings": outcome.warnings,
                "skipped": outcome.sources.iter().flat_map(|r| r.skipped.clone()).collect::<Vec<_>>(),
            },
            "next": next,
        }),
    })
}

pub fn remove(mut project: Project, target: &str) -> Result<Output> {
    let target_path = dunce::canonicalize(target).ok();
    let name = project
        .resolved_sources()
        .into_iter()
        .find(|s| s.name == target || target_path.as_deref() == Some(s.path.as_path()))
        .map(|s| s.name)
        .ok_or_else(|| {
            SynkError::new(
                ErrorCode::SourceNotFound,
                format!("no source named or located at `{target}`"),
            )
            .with_next("synk source list")
        })?;

    project.remove_source(&name)?;
    project.save()?;
    let tombstoned = match project.existing_store()? {
        Some(store) => store.tombstone_source(&name)?,
        None => 0,
    };
    Ok(Output {
        human: format!("Removed {name} ({tombstoned} documents marked deleted)"),
        data: json!({ "removed": name, "documents_deleted": tombstoned }),
    })
}

pub fn sources(project: &Project) -> Result<Output> {
    let rows: Vec<Value> = project
        .sources()
        .iter()
        .map(|spec| {
            let resolved = project.resolve(spec);
            json!({
                "name": spec.name,
                "path": spec.path.display().to_string(),
                "resolved": resolved.path.display().to_string(),
                "kind": kind(&resolved.path),
                "exclude": spec.exclude,
            })
        })
        .collect();

    let human = if rows.is_empty() {
        "No sources. Next: synk source add <path>".to_owned()
    } else {
        rows.iter()
            .map(|r| {
                format!(
                    "{:<20} {:<8} {}",
                    r["name"].as_str().unwrap_or_default(),
                    r["kind"].as_str().unwrap_or_default(),
                    r["resolved"].as_str().unwrap_or_default()
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    };
    Ok(Output {
        human,
        data: json!({ "sources": rows }),
    })
}

pub fn status(project: &Project) -> Result<Output> {
    let counts = match project.existing_store()? {
        Some(store) => store.counts()?,
        None => Vec::new(),
    };
    let count_of = |name: &str| {
        counts
            .iter()
            .find(|(n, _)| n == name)
            .map_or(0, |(_, c)| *c)
    };

    let rows: Vec<Value> = project
        .resolved_sources()
        .iter()
        .map(|s| json!({ "name": s.name, "kind": kind(&s.path), "documents": count_of(&s.name) }))
        .collect();
    let total: usize = counts.iter().map(|(_, c)| c).sum();
    let (pipeline, advice_next) = crate::pipeline::pipeline_status(project)?;
    let run_id = pipeline["run"]["id"].as_str().unwrap_or_default();
    let op = crate::work::status_op(project);
    // A quick preview without network calls: which steps are out of date.
    let plan = project.workspace(&[]).ok().and_then(|ws| {
        let opts = synk_pipeline::graph::Options {
            network: false,
            ..Default::default()
        };
        synk_pipeline::graph::make(&ws, &opts).ok()
    });
    let pending: Vec<&str> = plan
        .as_ref()
        .map(|p| {
            p.steps
                .iter()
                .filter(|s| s.action == synk_pipeline::graph::Action::Run)
                .map(|s| s.step.as_str())
                .collect()
        })
        .unwrap_or_default();
    let running = op
        .as_ref()
        .filter(|j| j.state == synk_pipeline::ops::State::Running);
    let next: Option<String> = if rows.is_empty() {
        None
    } else if running.is_some() {
        Some("synk wait --timeout 9m".to_owned())
    } else if !pending.is_empty() || plan.as_ref().is_some_and(|p| !p.blockers.is_empty()) {
        Some("synk train --dry-run".to_owned())
    } else if let Some(next) = &advice_next {
        Some(next.clone())
    } else if pipeline["verdict"]["outcome"] == "better" {
        Some(synk_pipeline::ids::export_command(run_id))
    } else if !run_id.is_empty() {
        Some(format!("synk run inspect {run_id}"))
    } else {
        Some("synk train --dry-run".to_owned())
    };

    let mut human = format!("Project: {}\n", show_path(&project.config_path));
    for row in &rows {
        let _ = writeln!(
            human,
            "  {:<20} {:<8} {} documents",
            row["name"].as_str().unwrap_or_default(),
            row["kind"].as_str().unwrap_or_default(),
            row["documents"]
        );
    }
    let _ = write!(human, "Total: {total} documents");
    for c in pipeline["conditions"].as_array().into_iter().flatten() {
        let mark = match c["status"].as_str() {
            Some("True") => continue,
            Some("False") => "✗",
            _ => "?",
        };
        let label = match c["type"].as_str().unwrap_or_default() {
            "SourcesFound" => "sources",
            "TrainerInstalled" => "trainer",
            "GeneratorReachable" => "LLM server",
            "GeneratorModelReady" => "LLM model",
            other => other,
        };
        let _ = write!(
            human,
            "\n{mark} {label}: {}",
            c["message"].as_str().unwrap_or_default()
        );
    }
    if !pipeline["dataset"].is_null() {
        let s = &pipeline["dataset"]["summary"];
        let _ = write!(
            human,
            "\nDataset: {} chunks, {} queries",
            s["chunks"], s["queries"]
        );
        human.push_str(&human::id("dataset", &pipeline["dataset"]["key"]));
    }
    if !pipeline["run"].is_null() {
        let _ = write!(
            human,
            "\nLatest tuning: {}",
            human::verdict(&pipeline["verdict"])
        );
        human.push_str(&human::id("run", &pipeline["run"]["key"]));
    }
    if let Some(j) = running {
        let _ = write!(human, "\nRunning: {}", j.id);
        if let Some(p) = &j.progress {
            let _ = write!(
                human,
                " ({} {}/{} {})",
                p.step, p.completed, p.total, p.unit
            );
        }
    } else if !pending.is_empty() {
        let _ = write!(human, "\nOut of date: {}", pending.join(", "));
    }
    human.push_str(&human::advice(
        pipeline["advice"].as_array().map_or(&[][..], Vec::as_slice),
    ));
    let _ = write!(
        human,
        "\nNext: {}",
        next.as_deref().unwrap_or("synk source add <path>")
    );

    Ok(Output {
        human,
        data: json!({
            "config": project.config_path.display().to_string(),
            "store": project.store_path().display().to_string(),
            "sources": rows,
            "total_documents": total,
            "pipeline": pipeline,
            "op": op,
            "steps": plan.as_ref().map(|p| p.steps.iter().map(|s| json!({"step": s.step, "action": s.action, "why": s.why})).collect::<Vec<_>>()),
            "next": next,
        }),
    })
}

fn check_overlap(project: &Project, abs: &Path) -> Result<()> {
    for existing in project.resolved_sources() {
        if existing.path == abs {
            return Err(SynkError::new(
                ErrorCode::SourceExists,
                format!("{} is already source `{}`", abs.display(), existing.name),
            )
            .with_next("synk source list"));
        }
        if abs.starts_with(&existing.path) || existing.path.starts_with(abs) {
            return Err(SynkError::new(
                ErrorCode::SourceOverlap,
                format!(
                    "{} overlaps source `{}` ({})",
                    abs.display(),
                    existing.name,
                    existing.path.display()
                ),
            )
            .with_hint("leave the inner folder out of the outer source with --exclude")
            .with_action(synk_core::ErrorAction::run(
                &format!("synk source remove {}", existing.name),
                synk_core::Applicability::Review,
                "or remove the existing source",
            )));
        }
    }
    Ok(())
}

fn default_name(path: &Path) -> String {
    path.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "root".to_owned())
}

fn kind(path: &Path) -> &'static str {
    if path.is_file() {
        "file"
    } else if path.is_dir() {
        "folder"
    } else {
        "missing"
    }
}

/// Skipped files listed by name in human output; the rest are counted.
pub const MAX_SKIPPED_SHOWN: usize = 10;

/// A skip reason in plain words.
pub fn skip_reason(reason: &str) -> String {
    use synk_connectors::readers::reason as r;
    match reason {
        r::NO_TEXT_LAYER => "no text to read (a scanned PDF? synk doesn't do OCR)".to_owned(),
        r::GARBLED_TEXT => {
            "the text comes out garbled (the PDF's fonts can't be decoded)".to_owned()
        }
        r::NOT_UTF8 => "not UTF-8 text".to_owned(),
        r::UNREADABLE => "can't be read (damaged or not really that file type)".to_owned(),
        r::EMPTY => "empty".to_owned(),
        other => other.to_owned(),
    }
}
