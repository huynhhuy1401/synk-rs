//! Commands that look at results or add evaluation data: queries, runs,
//! inspect, compare.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::fs;
use std::path::Path;

use serde_json::{Value, json};
use synk_core::{ErrorCode, Result, SynkError};
use synk_pipeline::ops::{OpKind, OpRecord, Ops};
use synk_pipeline::{advice, queries, report, train};

use crate::config::Project;
use crate::human;
use crate::output::Output;

pub fn import_queries(
    project: &Project,
    file: &Path,
    replace: bool,
    restyle: bool,
) -> Result<Output> {
    use synk_pipeline::train::{MIN_REAL_DOCUMENTS, MIN_REAL_QUERIES};
    let text = fs::read_to_string(file).map_err(|e| SynkError::io(file.display(), e))?;
    let mut store = project.open_store()?;
    let parsed = queries::parse_import(&text, &store.live_documents()?)?;
    let stats = store.import_real_queries(&parsed.rows, replace)?;
    if restyle {
        // The next build chooses the style examples again.
        store.set_style_queries(&[])?;
    }
    let real = store.real_queries()?;
    let total = real.len();
    let documents = distinct_documents(&real);
    // Style examples are left out of evaluation, so only the rest count.
    let evaluated = synk_pipeline::dataset::evaluated_real_queries(&store)?;
    let next = "synk train --dry-run";
    // A forecast: the dataset also skips searches about a document with no
    // text left to test on.
    let decides =
        train::real_queries_decide(evaluated.len(), distinct_documents(&evaluated) as u64);
    let mut human = format!(
        "Real queries: {} added, {} already there, {} removed; {total} in total, about {documents} documents.",
        stats.added, stats.unchanged, stats.removed,
    );
    if parsed.skipped_empty > 0 {
        let _ = write!(human, " {} left empty were skipped.", parsed.skipped_empty);
    }
    let _ = write!(
        human,
        "\nThey are used to evaluate, never to train, except {} kept as examples of your style; {MIN_REAL_QUERIES} or more of the rest about {MIN_REAL_DOCUMENTS}+ documents decide the verdict{}.",
        total - evaluated.len(),
        if decides { " (yours do)" } else { "" }
    );
    if restyle {
        human.push_str(
            "\nThe style examples will be chosen again (the next dataset re-asks the LLM).",
        );
    }
    let _ = write!(human, "\nNext: {next}");
    Ok(Output {
        human,
        data: json!({
            "added": stats.added,
            "unchanged": stats.unchanged,
            "removed": stats.removed,
            "skipped_empty": parsed.skipped_empty,
            "total": total,
            "documents": documents,
            "evaluated": evaluated.len(),
            "decides_verdict": decides,
            "restyled": restyle,
            "next": next,
        }),
    })
}

fn distinct_documents(queries: &[synk_store::RealQuery]) -> usize {
    queries
        .iter()
        .map(|q| q.doc.as_str())
        .collect::<std::collections::HashSet<_>>()
        .len()
}

/// Writes a file listing some documents with an empty search each, for the
/// user to fill in without opening them. Test documents come first: real
/// queries about them also show whether the generated test agrees.
pub fn queries_template(
    project: &Project,
    out: &Path,
    documents: usize,
    force: bool,
) -> Result<Output> {
    if out.exists() && !force {
        return Err(SynkError::new(
            ErrorCode::InvalidArgs,
            format!("{} already exists", out.display()),
        )
        .with_hint("pick another --out, or add --force to overwrite it"));
    }
    let store = project.open_store()?;
    let docs = store.live_documents()?;
    if docs.is_empty() {
        return Err(
            SynkError::new(ErrorCode::DatasetTooSmall, "no documents synced yet")
                .with_next("synk source sync"),
        );
    }
    let tests = synk_pipeline::dataset::test_document_ids(
        &store,
        &docs,
        project.pipeline().dataset.test_percent,
    )
    .unwrap_or_default();
    let mut order: Vec<&synk_core::Document> = docs.iter().collect();
    // Test documents first, then the rest; both in a stable shuffled order.
    order.sort_by_key(|d| {
        (
            !tests.contains(&d.id),
            blake3::hash(d.id.as_bytes()).to_hex().to_string(),
        )
    });
    let chosen: Vec<(String, String)> = order
        .into_iter()
        .take(documents.max(1))
        .map(|d| (d.id.clone(), d.title.clone()))
        .collect();
    let in_test = chosen.iter().filter(|(id, _)| tests.contains(id)).count();
    let text = queries::template(&chosen);
    fs::write(out, text).map_err(|e| SynkError::io(out.display(), e))?;
    let path = crate::output::show_path(out);
    let next = format!("synk queries import {}", crate::output::shell_word(&path));
    Ok(Output {
        human: format!(
            "Wrote {path} with {} documents. For each, write how you'd search for something you know is in it, without opening it. Then: {next}",
            chosen.len()
        ),
        data: json!({
            "path": path,
            "documents": chosen.len(),
            "test_documents": in_test,
            "next": next,
        }),
    })
}

/// Where `next` hints suggest writing real searches.
const SEARCHES_FILE: &str = "searches.jsonl";

pub fn list_queries(project: &Project) -> Result<Output> {
    let queries = match project.existing_store()? {
        Some(store) => store.real_queries()?,
        None => Vec::new(),
    };
    let mut human = format!("{} real queries", queries.len());
    for q in &queries {
        let _ = write!(human, "\n  {:<40} -> {}", q.query, q.doc);
        if q.style {
            human.push_str("   (style example, not evaluated)");
        }
    }
    let next = if queries.is_empty() {
        // A template written earlier is imported, not overwritten.
        if Path::new(SEARCHES_FILE).exists() {
            "synk queries import searches.jsonl"
        } else {
            "synk queries template --out searches.jsonl"
        }
    } else {
        "synk train --dry-run"
    };
    let _ = write!(human, "\nNext: {next}");
    Ok(Output {
        human,
        data: json!({ "queries": queries, "next": next }),
    })
}

fn verdict_line(v: &Value) -> String {
    human::verdict(v)
        .lines()
        .next()
        .unwrap_or_default()
        .to_owned()
}

/// Dotted keys of a settings object: `{"trainer": {"epochs": 2}}` ->
/// `trainer.epochs = 2`.
fn flatten(v: &Value, prefix: &str, out: &mut BTreeMap<String, Value>) {
    match v.as_object() {
        Some(map) => {
            for (k, v) in map {
                let key = if prefix.is_empty() {
                    k.clone()
                } else {
                    format!("{prefix}.{k}")
                };
                flatten(v, &key, out);
            }
        }
        None => {
            out.insert(prefix.to_owned(), v.clone());
        }
    }
}

/// The settings of `run` that differ from `current`, as `key=value`.
fn differs(run: &OpRecord, current: &BTreeMap<String, Value>) -> Vec<String> {
    let mut theirs = BTreeMap::new();
    flatten(&run.settings["config"], "", &mut theirs);
    theirs
        .into_iter()
        .filter(|(k, v)| current.get(k).is_some_and(|c| c != v))
        .map(|(k, v)| match v {
            Value::String(s) => format!("{k}={s}"),
            v => format!("{k}={v}"),
        })
        .collect()
}

pub fn runs(project: &Project) -> Result<Output> {
    let artifacts = project.artifacts();
    let ops = Ops::new(project.data_dir()?);
    ops.migrate(&artifacts)?;
    let mut by_key: BTreeMap<String, Value> = report::runs(&artifacts)?
        .into_iter()
        .filter_map(|r| Some((r["key"].as_str()?.to_owned(), r)))
        .collect();
    let mut current = BTreeMap::new();
    flatten(
        &serde_json::to_value(project.pipeline()).unwrap_or_default(),
        "",
        &mut current,
    );
    let mut runs: Vec<Value> = Vec::new();
    for r in ops.list_kind(OpKind::Run)? {
        let mut row = r
            .key
            .as_ref()
            .and_then(|k| by_key.get(k).cloned())
            .unwrap_or_else(|| json!({"key": r.key}));
        row["run"] = json!(r.id);
        row["name"] = json!(r.name);
        row["state"] = json!(r.state);
        row["attempt"] = json!(r.attempt);
        row["started_at"] = json!(r.started_at);
        row["same_as"] = json!(r.same_as);
        row["sets"] = json!(crate::work::sets_of(&r));
        row["differs"] = json!(differs(&r, &current));
        if row["base_model"].is_null() {
            row["base_model"] = r.settings["config"]["model"]["base"].clone();
        }
        runs.push(row);
    }
    // Trained models no run record names (shouldn't happen; kept visible).
    let named: Vec<String> = runs
        .iter()
        .filter_map(|r| r["key"].as_str().map(str::to_owned))
        .collect();
    by_key.retain(|k, _| !named.contains(k));
    runs.extend(by_key.into_values());

    let mut human = format!("{} runs", runs.len());
    for r in &runs {
        let state = r["state"].as_str().unwrap_or("succeeded");
        let verdict = if r["verdict"].is_null() {
            state.to_owned()
        } else {
            verdict_line(&r["verdict"])
        };
        let _ = write!(
            human,
            "\n{} {:<20} {:<10} {}",
            if r["latest"] == true { "*" } else { " " },
            r["run"].as_str().unwrap_or_default(),
            if state == "succeeded" { "" } else { state },
            r["base_model"].as_str().unwrap_or_default(),
        );
        let _ = write!(human, "\n    {verdict}");
        let differs: Vec<&str> = r["differs"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .collect();
        if !differs.is_empty() {
            let _ = write!(
                human,
                "\n    differs from synk.toml: {}",
                differs.join(", ")
            );
        }
        if let Some(same) = r["same_as"].as_str() {
            let _ = write!(human, "\n    same model as {same}");
        }
    }
    // Compare the newest finished run with the newest older one on the
    // same dataset; runs on different datasets can't be compared.
    let id = |r: &Value| r["run"].as_str().unwrap_or_default().to_owned();
    let finished: Vec<&Value> = runs.iter().filter(|r| !r["verdict"].is_null()).collect();
    let newest = finished.first();
    let partner = newest.and_then(|n| {
        finished[1..]
            .iter()
            .find(|r| r["dataset"] == n["dataset"] && r["key"] != n["key"])
    });
    let next = match (newest, partner) {
        (None, _) => "synk train --dry-run".to_owned(),
        (Some(n), Some(p)) => format!("synk run compare {} {}", id(p), id(n)),
        (Some(n), None) => format!("synk run inspect {}", id(n)),
    };
    let _ = write!(human, "\nNext: {next}");
    Ok(Output {
        human,
        data: json!({ "runs": runs, "next": next }),
    })
}

pub fn inspect(project: &Project, run: Option<&str>, worst: usize) -> Result<Output> {
    let artifacts = project.artifacts();
    let run = report::find_run(&artifacts, run)?;
    let report = report::inspect(&artifacts, &run, worst)?;
    let mut human = format!(
        "Run {}: of {} evaluation queries, {} ranked better, {} worse, {} the same.",
        run.key, report["queries"], report["improved"], report["worse"], report["unchanged"]
    );
    for (title, key) in [("Biggest losses", "losses"), ("Biggest gains", "gains")] {
        let items = report[key].as_array().cloned().unwrap_or_default();
        if items.is_empty() {
            continue;
        }
        let _ = write!(human, "\n\n{title}:");
        for item in &items {
            let _ = write!(
                human,
                "\n  [{}] \"{}\"  rank {} -> {}",
                item["kind"].as_str().unwrap_or_default(),
                item["query"].as_str().unwrap_or_default(),
                item["ranks"]["base"],
                item["ranks"]["tuned"],
            );
            if let Some(text) = item["expected"]["text"].as_str() {
                let _ = write!(human, "\n      expected: {text}");
            }
            if let Some(text) = item["tuned_top"]["text"].as_str() {
                let _ = write!(human, "\n      tuned top: {text}");
            }
        }
    }
    let next = "synk run list";
    let _ = write!(human, "\nNext: {next}");
    let mut data = report;
    data["next"] = json!(next);
    Ok(Output { human, data })
}

pub fn compare(project: &Project, a: &str, b: &str) -> Result<Output> {
    let artifacts = project.artifacts();
    let run_a = report::find_run(&artifacts, Some(a))?;
    let run_b = report::find_run(&artifacts, Some(b))?;
    let mut data = report::compare(&artifacts, &run_a, &run_b)?;
    let next = match data["verdict"]["outcome"].as_str() {
        Some("better") => {
            synk_pipeline::ids::export_command(data["b"].as_str().unwrap_or_default())
        }
        Some("worse") => synk_pipeline::ids::export_command(data["a"].as_str().unwrap_or_default()),
        _ => format!(
            "synk run inspect {}",
            data["b"].as_str().unwrap_or_default()
        ),
    };
    let human = format!(
        "{}\nNext: {next}",
        data["verdict"]["message"].as_str().unwrap_or_default()
    );
    data["next"] = json!(next);
    Ok(Output { human, data })
}

pub fn show_run(project: &Project, run: Option<&str>) -> Result<Output> {
    let artifacts = project.artifacts();
    let run = report::find_run(&artifacts, run)?;
    let id = report::run_id(&artifacts, &run)?;
    let eval = train::eval_for(&artifacts, &run)?;
    let verdict = eval.as_ref().map(train::eval_verdict);
    let general = eval
        .as_ref()
        .map(|e| advice::general_verdicts(&e.summary))
        .unwrap_or_default();
    let advice = verdict
        .as_ref()
        .map(|v| advice::train(&run.summary, v, &general, project.pipeline()))
        .unwrap_or_default();
    let next = match verdict.as_ref().map(|v| v.outcome.as_str()) {
        Some("better") => synk_pipeline::ids::export_command(&id),
        _ => format!("synk run inspect {id}"),
    };
    let mut human = format!(
        "{id}\n  base model: {}\n  dataset: {}\n  trained on {} queries, {} epochs\nTest: {}\n{}",
        run.summary["base_model"].as_str().unwrap_or("?"),
        report::dataset_id(&artifacts, &run)?.unwrap_or_else(|| "?".to_owned()),
        run.summary["examples"],
        run.summary["epochs"],
        verdict
            .as_ref()
            .map_or_else(|| "not tested".to_owned(), human::verdict),
        human::QUALITY_NOTE,
    );
    if crate::output::verbose()
        && let Some(e) = &eval
    {
        let _ = write!(human, "\n\n{}", crate::pipeline::metrics_tables(&e.summary));
    }
    human.push_str(&human::advice(&advice));
    let _ = write!(human, "\nNext: {next}");
    Ok(Output {
        human,
        data: json!({
            "run": id,
            "key": run.key,
            "dataset": report::dataset_id(&artifacts, &run)?,
            "base_model": run.summary["base_model"],
            "created_at": run.created_at,
            "params": run.params,
            "summary": run.summary,
            "verdict": verdict,
            "advice": advice,
            "next": next,
        }),
    })
}

pub fn html_report(project: &Project, run: Option<&str>, out: &Path) -> Result<Output> {
    let artifacts = project.artifacts();
    let run = report::find_run(&artifacts, run)?;
    let id = report::run_id(&artifacts, &run)?;
    let html = synk_pipeline::report_html::render(&artifacts, &run, &id, |v, general| {
        advice::train(&run.summary, v, general, project.pipeline())
    })?;
    fs::write(out, html).map_err(|e| SynkError::io(out.display(), e))?;
    let shown = crate::output::show_path(out);
    Ok(Output {
        human: format!(
            "Wrote the report for {id} to {shown}. Open it in a browser.\nIt contains passages of your documents."
        ),
        data: json!({ "run": id, "path": out.display().to_string() }),
    })
}
