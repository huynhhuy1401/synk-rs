//! End-to-end tests of the JSON contract agents rely on.
//!
//! Every command runs with the fake trainer (`SYNK_FAKE_TRAINER=1`) and a
//! private `SYNK_HOME`, so the train/fetch/wait flow is tested without Python;
//! the one test that trains for real is `#[ignore]`d.

use std::fs;
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use serde_json::Value;

/// The environment every test command gets, plus `extra`.
fn command(dir: &Path, extra: &[(&str, &str)]) -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_synk"));
    cmd.current_dir(dir)
        .env("SYNK_HOME", dir.join(".synk-home"))
        .env("SYNK_FAKE_TRAINER", "1")
        .env_remove("SYNK_FAKE_TRAINER_SLOW")
        .env_remove("SYNK_FAKE_TRAINER_STUCK")
        .env_remove("SYNK_TEST_INTERRUPT_AFTER_MS")
        .env_remove("SYNK_TEST_INTERRUPT_FILE")
        .env_remove("SYNK_TEST_FREE_BYTES")
        .env_remove("SYNK_DISK_CHECK");
    for (k, v) in extra {
        cmd.env(k, v);
    }
    cmd
}

fn parse(out: &std::process::Output) -> Value {
    let value: Value = serde_json::from_slice(&out.stdout).unwrap_or_else(|e| {
        panic!(
            "stdout is not JSON ({e})\nstdout: {}\nstderr: {}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        )
    });
    assert_eq!(
        out.status.success(),
        value["ok"] == true,
        "exit code must match ok: {value}"
    );
    assert_eq!(value["v"], 1, "every object carries the contract version");
    value
}

/// Runs `synk --json <args>` in `dir` and parses stdout.
fn synk(dir: &Path, args: &[&str]) -> Value {
    synk_env(dir, args, &[])
}

fn synk_env(dir: &Path, args: &[&str], extra: &[(&str, &str)]) -> Value {
    let out = command(dir, extra)
        .arg("--json")
        .args(args)
        .output()
        .unwrap();
    parse(&out)
}

fn error_code(value: &Value) -> &str {
    value["error"]["code"]
        .as_str()
        .unwrap_or_else(|| panic!("expected an error: {value}"))
}

/// Asserts that every dotted path exists. Fields agents parse may be added
/// but never removed, so each command's list only grows.
fn assert_fields(value: &Value, paths: &[&str]) {
    for path in paths {
        let found = path
            .split('.')
            .try_fold(value, |v, key| v.get(key))
            .is_some();
        assert!(found, "missing field `{path}` in {value}");
    }
}

fn project_with_notes() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    fs::create_dir(dir.path().join("notes")).unwrap();
    fs::write(dir.path().join("notes/a.md"), "# A\n\nalpha").unwrap();
    fs::write(dir.path().join("notes/b.txt"), "beta").unwrap();
    fs::write(dir.path().join("handbook.md"), "# Handbook").unwrap();
    assert_eq!(synk(dir.path(), &["init"])["ok"], true);
    // Keep the general check's public sets off the network in tests.
    append_config(dir.path(), "\n[eval]\ngeneral_check = false\n");
    dir
}

fn append_config(root: &Path, extra: &str) {
    let config = root.join("synk.toml");
    let mut text = fs::read_to_string(&config).unwrap();
    text.push_str(extra);
    fs::write(config, text).unwrap();
}

/// `source sync`: reads the sources.
fn sync(root: &Path) -> Value {
    let out = synk(root, &["source", "sync"]);
    assert_eq!(out["ok"], true, "{out}");
    out
}

fn sync_totals(root: &Path) -> Value {
    sync(root)["data"]["sync"]["totals"].clone()
}

#[test]
fn full_flow_add_sync_status_remove() {
    let dir = project_with_notes();
    let root = dir.path();

    let added = synk(root, &["source", "add", "notes", "handbook.md"]);
    assert_eq!(added["data"]["added"].as_array().unwrap().len(), 2);
    let config = fs::read_to_string(root.join("synk.toml")).unwrap();
    assert!(
        config.contains("path = \"notes\""),
        "paths inside the project are relative:\n{config}"
    );

    assert_eq!(sync_totals(root)["added"], 3);
    let second = sync_totals(root);
    assert_eq!(second["added"], 0);
    assert_eq!(second["unchanged"], 3);

    fs::write(root.join("notes/a.md"), "# A\n\nalpha, edited").unwrap();
    fs::remove_file(root.join("notes/b.txt")).unwrap();
    let third = sync_totals(root);
    assert_eq!(third["updated"], 1);
    assert_eq!(third["deleted"], 1);

    let status = synk(root, &["status"]);
    assert_eq!(status["data"]["total_documents"], 2);

    let removed = synk(root, &["source", "remove", "handbook.md"]);
    assert_eq!(removed["data"]["documents_deleted"], 1);
    assert_eq!(
        synk(root, &["source", "list"])["data"]["sources"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
}

#[test]
fn hand_edited_config_is_preserved() {
    let dir = project_with_notes();
    let config = dir.path().join("synk.toml");
    fs::write(&config, "# my comment\n").unwrap();

    synk(dir.path(), &["source", "add", "notes"]);

    assert!(
        fs::read_to_string(&config)
            .unwrap()
            .starts_with("# my comment\n")
    );
}

#[test]
fn without_project_commands_explain_how_to_start() {
    let dir = tempfile::tempdir().unwrap();
    let out = synk(dir.path(), &["train", "--dry-run"]);
    assert_eq!(error_code(&out), "CONFIG_NOT_FOUND");
    assert_eq!(out["error"]["next"], "synk init");
}

#[test]
fn add_rejects_bad_sources_without_saving() {
    let dir = project_with_notes();
    let root = dir.path();
    fs::write(root.join("photo.png"), [0u8]).unwrap();
    synk(root, &["source", "add", "notes"]);

    assert_eq!(
        error_code(&synk(root, &["source", "add", "notes"])),
        "SOURCE_EXISTS"
    );
    assert_eq!(
        error_code(&synk(root, &["source", "add", "notes/a.md"])),
        "SOURCE_OVERLAP"
    );
    assert_eq!(
        error_code(&synk(root, &["source", "add", "photo.png"])),
        "UNSUPPORTED_FORMAT"
    );
    assert_eq!(
        error_code(&synk(root, &["source", "add", "missing"])),
        "SOURCE_MISSING"
    );
    // The valid path in a mixed batch must not be saved either.
    assert_eq!(
        error_code(&synk(root, &["source", "add", "handbook.md", "missing"])),
        "SOURCE_MISSING"
    );
    assert_eq!(
        synk(root, &["source", "list"])["data"]["sources"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
}

#[test]
fn missing_source_warns_and_keeps_documents() {
    let dir = project_with_notes();
    let root = dir.path();
    fs::create_dir(root.join("drive")).unwrap();
    fs::write(root.join("drive/x.md"), "x").unwrap();
    synk(root, &["source", "add", "notes", "drive"]);
    sync(root);

    fs::rename(root.join("drive"), root.join("unmounted")).unwrap();
    let out = sync(root);
    assert_eq!(out["data"]["sync"]["warnings"][0]["code"], "SOURCE_MISSING");
    assert_eq!(synk(root, &["status"])["data"]["total_documents"], 3);
}

#[test]
fn duplicate_names_get_a_suffix() {
    let dir = project_with_notes();
    let root = dir.path();
    fs::create_dir_all(root.join("other/notes")).unwrap();
    synk(root, &["source", "add", "notes"]);
    let out = synk(root, &["source", "add", "other/notes"]);
    assert_eq!(out["data"]["added"][0]["name"], "notes-2");
}

// --- pipeline -------------------------------------------------------------

/// A minimal OpenAI-compatible server that answers every chat completion
/// with two queries unique to the request. Returns its base URL.
fn fake_llm() -> String {
    serve_llm(fake_reply)
}

/// The fake LLM's answer: two queries named after a hash of the request.
fn fake_reply(body: &[u8]) -> (u16, String) {
    {
        // FNV-1a of the request, so each chunk gets its own queries.
        let id = body.iter().fold(0xcbf29ce484222325u64, |h, b| {
            (h ^ u64::from(*b)).wrapping_mul(0x100000001b3)
        }) % 100_000;
        let content = format!(r#"{{"queries": ["where is note {id}", "facts about {id}"]}}"#);
        let reply = serde_json::json!({
            "choices": [{"message": {"role": "assistant", "content": content}}]
        });
        (200, reply.to_string())
    }
}

/// An OpenAI-compatible server on a free port: `reply` answers each POST
/// body with a status and a JSON body; other requests get 200 and `{}`.
fn serve_llm(reply: fn(&[u8]) -> (u16, String)) -> String {
    use std::io::{BufRead, BufReader, Read, Write};
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}/v1", listener.local_addr().unwrap());
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { continue };
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut request_line = String::new();
            let _ = reader.read_line(&mut request_line);
            let mut length = 0;
            loop {
                let mut line = String::new();
                if reader.read_line(&mut line).unwrap_or(0) == 0 || line == "\r\n" {
                    break;
                }
                if let Some(v) = line.to_ascii_lowercase().strip_prefix("content-length:") {
                    length = v.trim().parse().unwrap_or(0);
                }
            }
            let mut body = vec![0; length];
            let _ = reader.read_exact(&mut body);
            let (status, text) = if request_line.starts_with("POST") {
                reply(&body)
            } else {
                (200, "{}".to_owned())
            };
            let _ = write!(
                stream,
                "HTTP/1.1 {status} X\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{text}",
                text.len()
            );
        }
    });
    url
}

fn with_generator(root: &Path, base_url: &str) {
    append_config(
        root,
        &format!("\n[generator]\nbase_url = \"{base_url}\"\nmodel = \"fake\"\n"),
    );
}

/// A generator plus a local base model whose word-level tokenizer needs no
/// download. The fake trainer "trains" it.
fn with_generator_and_tiny_model(root: &Path, base_url: &str) {
    with_generator(root, base_url);
    fs::create_dir(root.join("tiny-model")).unwrap();
    fs::write(
        root.join("tiny-model/tokenizer.json"),
        synk_pipeline::tokenizer::word_level_json(),
    )
    .unwrap();
    append_config(root, "\n[model]\nbase = \"tiny-model\"\n");
}

fn synced_project() -> tempfile::TempDir {
    let dir = project_with_notes();
    synk(dir.path(), &["source", "add", "notes", "handbook.md"]);
    sync(dir.path());
    dir
}

/// A project with more documents, so the test split has several and the
/// verdict can be decided.
fn bigger_project() -> tempfile::TempDir {
    let dir = project_with_notes();
    let root = dir.path();
    for i in 0..20 {
        fs::write(
            root.join(format!("notes/n{i}.md")),
            format!("# Note {i}\n\nTopic number {i} is about subject {i}."),
        )
        .unwrap();
    }
    synk(root, &["source", "add", "notes", "handbook.md"]);
    with_generator_and_tiny_model(root, &fake_llm());
    dir
}

#[test]
fn existing_commands_keep_their_fields() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    fs::create_dir(root.join("notes")).unwrap();
    fs::write(root.join("notes/a.md"), "alpha").unwrap();

    assert_fields(&synk(root, &["init"]), &["data.config"]);
    assert_fields(
        &synk(root, &["source", "add", "notes"]),
        &["data.added", "data.next"],
    );
    let sources = synk(root, &["source", "list"]);
    assert_fields(
        &sources["data"]["sources"][0],
        &["name", "path", "resolved", "kind", "exclude"],
    );
    let synced = sync(root);
    assert_fields(
        &synced,
        &[
            "data.sync.totals.added",
            "data.sync.totals.updated",
            "data.sync.totals.deleted",
            "data.sync.totals.unchanged",
            "data.sync.totals.skipped",
            "data.sync.warnings",
            "data.sync.removed_sources",
            "data.next",
        ],
    );
    assert_fields(
        &synk(root, &["status"]),
        &[
            "data.config",
            "data.store",
            "data.sources",
            "data.total_documents",
            "data.pipeline.conditions",
            "data.op",
            "data.steps",
            "data.next",
        ],
    );
    assert_fields(
        &synk(root, &["source", "remove", "notes"]),
        &["data.removed", "data.documents_deleted"],
    );
}

#[test]
fn models_lists_the_catalog_without_a_project() {
    let dir = tempfile::tempdir().unwrap();
    let out = synk(dir.path(), &["model", "list"]);
    assert_fields(&out, &["data.models", "data.configured"]);
    assert_eq!(out["data"]["configured"], "multilingual-e5-small");
    assert_fields(
        &out["data"]["models"][0],
        &[
            "id",
            "hf",
            "revision",
            "query_prompt",
            "document_prompt",
            "dims",
            "parameters",
            "ollama",
        ],
    );
    let ids: Vec<&str> = out["data"]["models"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|m| m["id"].as_str())
        .collect();
    assert!(
        ids.contains(&"bge-m3") && ids.contains(&"embeddinggemma-300m"),
        "{ids:?}"
    );
}

#[test]
fn train_runs_everything_and_reuses_it() {
    let dir = bigger_project();
    let root = dir.path();

    let preview = synk(root, &["train", "--dry-run"]);
    assert_fields(
        &preview,
        &[
            "data.steps",
            "data.consent",
            "data.blockers",
            "data.totals.download_bytes",
            "data.totals.est_seconds",
            "data.fetch_names",
            "data.would",
            "data.next",
        ],
    );
    assert_eq!(preview["data"]["next"], "synk train");
    assert_eq!(preview["data"]["would"], "a new run");
    let action = |p: &Value, step: &str| {
        p["data"]["steps"]
            .as_array()
            .unwrap()
            .iter()
            .find(|s| s["step"] == step)
            .unwrap()["action"]
            .clone()
    };
    assert_eq!(action(&preview, "sync"), "run");
    assert_eq!(action(&preview, "dataset"), "run");
    assert_eq!(action(&preview, "train"), "run");
    assert_eq!(action(&preview, "trainer"), "cached");
    assert!(
        preview["data"]["consent"].as_array().unwrap().is_empty()
            && preview["data"]["fetch"].is_null(),
        "nothing to download or send: {preview}"
    );
    // A dry run changes nothing.
    assert!(!root.join(".synk/ops").exists());
    assert_eq!(synk(root, &["status"])["data"]["total_documents"], 0);

    let trained = synk(root, &["train"]);
    assert_fields(
        &trained,
        &[
            "data.id",
            "data.reused",
            "data.op.state",
            "data.op.steps",
            "data.op.settings",
            "data.run",
            "data.key",
            "data.dataset",
            "data.dataset_summary.queries",
            "data.verdict.outcome",
            "data.advice",
            "data.next",
        ],
    );
    let run = "run:1-tiny-model";
    assert_eq!(trained["data"]["id"], run);
    assert_eq!(trained["data"]["run"], run);
    assert_eq!(trained["data"]["reused"], false);
    assert_eq!(trained["data"]["op"]["state"], "succeeded");
    assert!(
        trained["data"]["dataset"]
            .as_str()
            .unwrap()
            .starts_with("dataset:1-"),
        "{trained}"
    );
    assert_eq!(trained["data"]["verdict"]["outcome"], "better");
    assert_eq!(
        trained["data"]["next"],
        format!("synk export {run} --out ./tiny-model-model")
    );

    // Nothing changed: the same run comes back, and no new number.
    let again = synk(root, &["train", "--dry-run"]);
    for step in ["sync", "dataset", "train", "eval"] {
        assert_eq!(action(&again, step), "cached", "{step}: {again}");
    }
    assert_eq!(again["data"]["reuses"], run);
    assert!(again["data"]["next"].is_null());
    let reused = synk(root, &["train"]);
    assert_eq!(reused["data"]["id"], run);
    assert_eq!(reused["data"]["reused"], true);
    assert_eq!(reused["data"]["verdict"]["outcome"], "better");

    // `--set` makes a variant without editing synk.toml.
    let tried = synk(
        root,
        &["train", "--dry-run", "--set", "trainer.negatives=1"],
    );
    assert_eq!(tried["data"]["overrides"]["trainer.negatives"], 1);
    assert_eq!(action(&tried, "train"), "run");
    assert_eq!(action(&tried, "dataset"), "cached");
    let second = synk(
        root,
        &["train", "--set", "trainer.negatives=1", "--name", "neg1"],
    );
    assert_eq!(second["data"]["id"], "run:2-neg1", "{second}");
    assert!(
        !fs::read_to_string(root.join("synk.toml"))
            .unwrap()
            .contains("negatives = 1")
    );
    let taken = synk(
        root,
        &["train", "--set", "trainer.negatives=2", "--name", "neg1"],
    );
    assert_eq!(error_code(&taken), "RUN_NAME_TAKEN");

    let runs = synk(root, &["run", "list"]);
    let list = runs["data"]["runs"].as_array().unwrap();
    assert_eq!(list.len(), 2, "{runs}");
    assert_eq!(list[0]["run"], "run:2-neg1");
    assert_eq!(list[0]["name"], "neg1");
    assert_eq!(list[0]["state"], "succeeded");
    assert_eq!(list[0]["sets"], serde_json::json!(["trainer.negatives=1"]));
    assert_eq!(
        list[0]["differs"],
        serde_json::json!(["trainer.negatives=1"])
    );
    assert_eq!(list[1]["differs"], serde_json::json!([]));
    assert_eq!(
        runs["data"]["next"],
        format!("synk run compare {run} run:2-neg1")
    );
    let compared = synk(root, &["run", "compare", "1", "neg1"]);
    assert_fields(
        &compared,
        &["data.a", "data.b", "data.verdict.outcome", "data.next"],
    );
    assert_eq!(compared["data"]["a"], run);

    // Results views take numbers, names and IDs.
    assert_fields(
        &synk(root, &["run", "show", "best"]),
        &["data.run", "data.verdict", "data.advice", "data.next"],
    );
    assert_eq!(synk(root, &["run", "show", "1"])["data"]["run"], run);
    assert_eq!(
        synk(root, &["run", "show", "run:2"])["data"]["run"],
        "run:2-neg1"
    );
    let inspect = synk(root, &["run", "inspect", "--limit", "3"]);
    assert_fields(
        &inspect,
        &[
            "data.run",
            "data.queries",
            "data.improved",
            "data.gains",
            "data.losses",
        ],
    );
    let report = synk(root, &["run", "report", run, "--out", "report.html"]);
    assert_eq!(report["data"]["run"], run);
    let html = fs::read_to_string(root.join("report.html")).unwrap();
    assert!(
        html.contains("<!doctype html>") && html.contains(run),
        "{html}"
    );

    // Export: the fake model is copied with synk's README.
    let exported = synk(root, &["export", "1", "--out", "exported"]);
    assert_fields(
        &exported,
        &[
            "data.run",
            "data.path",
            "data.format",
            "data.check",
            "data.usage",
        ],
    );
    assert_eq!(exported["data"]["run"], run);
    assert!(root.join("exported/fake-model.txt").is_file());
    assert!(root.join("exported/README.md").is_file());
    // Copy-paste setup for the tools people embed with, with the model's
    // own prefixes and a re-index reminder.
    let usage = &exported["data"]["usage"];
    for key in [
        "sentence_transformers",
        "llamaindex",
        "langchain",
        "open_webui_env",
        "reindex",
    ] {
        assert!(!usage[key].is_null(), "usage.{key} missing: {usage}");
    }
    assert!(
        usage["llamaindex"]
            .as_str()
            .unwrap()
            .contains("query_instruction"),
        "{usage}"
    );
    assert_eq!(
        usage["open_webui_env"]["RAG_EMBEDDING_QUERY_PREFIX"],
        usage["query_prefix"]
    );
    let again = synk(root, &["export", run, "--out", "exported"]);
    assert_eq!(error_code(&again), "EXPORT_EXISTS");
    let forced = synk(root, &["export", run, "--out", "exported", "--force"]);
    assert_eq!(forced["ok"], true);
    assert_eq!(
        error_code(&synk(root, &["export", run, "--out", "x", "--ollama", "n"])),
        "INVALID_ARGS",
        "--ollama only applies to gguf"
    );

    // Runs made before numbering get numbers, oldest first.
    fs::remove_dir_all(root.join(".synk/ops")).unwrap();
    let renumbered = synk(root, &["run", "list"]);
    let ids: Vec<&str> = renumbered["data"]["runs"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|r| r["run"].as_str())
        .collect();
    assert_eq!(
        ids,
        ["run:2-tiny-model", "run:1-tiny-model"],
        "{renumbered}"
    );
}

#[test]
fn fetch_checks_names_and_skips_what_is_here() {
    let dir = bigger_project();
    let root = dir.path();
    let bad = synk(root, &["fetch", "no-such-model"]);
    assert_eq!(error_code(&bad), "INVALID_ARGS");
    // A catalog model pinned to another commit than the catalog's.
    let pinned = synk(root, &["fetch", "multilingual-e5-small@0000000"]);
    assert_eq!(error_code(&pinned), "INVALID_ARGS");
    assert!(
        pinned["error"]["message"]
            .as_str()
            .unwrap()
            .contains("catalog pins")
    );
    // Everything training needs is here: nothing to download.
    let nothing = synk(root, &["fetch"]);
    assert_eq!(nothing["ok"], true, "{nothing}");
    assert!(nothing["data"]["id"].is_null());
    assert!(nothing["data"]["items"].as_array().unwrap().is_empty());
    let trainer = synk(root, &["fetch", "trainer"]);
    assert_eq!(trainer["data"]["items"][0]["present"], true);
    assert_eq!(trainer["data"]["next"], "synk train --dry-run");
}

#[test]
fn downloads_need_fetch_or_yes() {
    let dir = bigger_project();
    let root = dir.path();
    // A real (absent) trainer: training must install it first.
    let no_trainer = [("SYNK_FAKE_TRAINER", "0")];
    let preview = synk_env(root, &["train", "--dry-run"], &no_trainer);
    assert_eq!(
        preview["data"]["fetch_names"],
        serde_json::json!(["trainer"])
    );
    assert_eq!(preview["data"]["fetch"], "synk fetch trainer");
    assert!(preview["data"]["next"].is_null(), "an agent must ask first");
    assert!(
        preview["data"]["consent"]
            .as_array()
            .unwrap()
            .iter()
            .any(|c| c["kind"] == "download" && c["name"] == "trainer"),
        "{preview}"
    );

    let out = synk_env(root, &["train"], &no_trainer);
    assert_eq!(error_code(&out), "CONSENT_REQUIRED");
    assert!(out["error"]["next"].is_null(), "an agent must ask first");
    let actions = out["error"]["actions"].as_array().unwrap();
    assert_eq!(
        actions[0]["argv"],
        serde_json::json!(["synk", "fetch", "trainer"])
    );
    assert_eq!(actions[0]["applicability"], "manual");
    assert_eq!(
        actions[1]["argv"],
        serde_json::json!(["synk", "train", "--yes"])
    );
    assert_eq!(
        out["error"]["details"]["fetch"],
        serde_json::json!(["trainer"])
    );
    let variant = synk_env(
        root,
        &["train", "--set", "trainer.negatives=1"],
        &no_trainer,
    );
    assert_eq!(
        variant["error"]["actions"][1]["argv"],
        serde_json::json!(["synk", "train", "--set", "trainer.negatives=1", "--yes"])
    );
    // Nothing ran, and no run got a number.
    assert!(
        synk(root, &["run", "list"])["data"]["runs"]
            .as_array()
            .unwrap()
            .is_empty()
    );
    assert_eq!(error_code(&synk(root, &["wait"])), "JOB_NOT_FOUND");
}

#[test]
fn runs_can_be_waited_cancelled_and_continued() {
    let dir = bigger_project();
    let root = dir.path();
    let dataset = synk(root, &["dataset"]);
    assert_eq!(dataset["data"]["op"]["kind"], "dataset", "{dataset}");

    // A slow training step, in the background.
    let child = command(root, &[("SYNK_FAKE_TRAINER_SLOW", "20000")])
        .args(["--json", "train"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let started = Instant::now();
    let running = loop {
        let j = synk(root, &["wait", "--timeout", "0s"]);
        if j["ok"] == true
            && j["data"]["state"] == "running"
            && j["data"]["kind"] == "run"
            && j["data"]["steps"]
                .as_array()
                .is_some_and(|s| s.iter().any(|s| s["step"] == "dataset"))
        {
            break j;
        }
        assert!(
            started.elapsed() < Duration::from_secs(30),
            "the run never started: {j}"
        );
        std::thread::sleep(Duration::from_millis(100));
    };
    let id = running["data"]["id"].as_str().unwrap().to_owned();
    assert_eq!(id, "run:1-tiny-model");
    assert_eq!(running["data"]["done"], false);
    assert_eq!(running["data"]["next"], "synk wait --timeout 9m");

    // One thing at a time.
    let blocked = synk(root, &["train", "--set", "trainer.negatives=1"]);
    assert_eq!(error_code(&blocked), "JOB_RUNNING");
    assert_eq!(blocked["error"]["next"], "synk wait");

    // Waiting with a short timeout is not an error.
    let waited = synk(root, &["wait", "current", "--timeout", "1s"]);
    assert_eq!(waited["ok"], true);
    assert_eq!(waited["data"]["done"], false);

    let cancel = synk(root, &["cancel"]);
    assert_eq!(cancel["data"]["requested"], true);
    let out = child.wait_with_output().unwrap();
    let trained = parse(&out);
    assert_eq!(error_code(&trained), "CANCELLED");
    assert_eq!(trained["error"]["details"]["id"], id.as_str());
    assert!(
        trained["error"]["actions"]
            .as_array()
            .unwrap()
            .iter()
            .any(|a| a["argv"] == serde_json::json!(["synk", "train", "run:1"])),
        "{trained}"
    );
    // The first stderr event names the run.
    let first: Value =
        serde_json::from_str(String::from_utf8_lossy(&out.stderr).lines().next().unwrap()).unwrap();
    assert_eq!(first["event"], "op");
    assert_eq!(first["id"], id.as_str());
    let done = synk(root, &["wait", &id, "--timeout", "10s"]);
    assert_eq!(done["data"]["state"], "cancelled");
    assert_eq!(done["data"]["done"], true);
    assert_eq!(done["data"]["next"], "synk train run:1");

    let logs = synk(root, &["logs", "1"]);
    assert_fields(&logs, &["data.events", "data.last_seq", "data.more"]);
    assert!(!logs["data"]["events"].as_array().unwrap().is_empty());

    // Editing synk.toml makes plain `train` a new run, while `train run:1`
    // continues run 1 with the settings it started with.
    append_config(root, "\n[trainer]\nnegatives = 1\n");
    let preview = synk(root, &["train", "--dry-run"]);
    assert_eq!(preview["data"]["would"], "a new run");
    let resumed = synk(root, &["train", "run:1"]);
    assert_eq!(resumed["data"]["id"], id.as_str(), "{resumed}");
    assert_eq!(resumed["data"]["op"]["attempt"], 2);
    assert_eq!(resumed["data"]["op"]["state"], "succeeded");
    let shown = synk(root, &["run", "show", "1"]);
    assert_ne!(shown["data"]["params"]["negatives"], 1);
    assert_eq!(
        error_code(&synk(root, &["train", "1", "--set", "trainer.epochs=2"])),
        "INVALID_ARGS"
    );
}

/// Starts `synk --json train` with a slow fake training step, waits until
/// it is training, then presses Ctrl-C `presses` times (through the test
/// hook), and returns its result and exit code.
fn train_interrupted(root: &Path, presses: usize, extra: &[(&str, &str)]) -> (Value, Option<i32>) {
    let presses_file = root.join("presses.txt");
    let mut vars = vec![
        ("SYNK_FAKE_TRAINER_SLOW", "20000"),
        ("SYNK_TEST_INTERRUPT_FILE", presses_file.to_str().unwrap()),
    ];
    vars.extend_from_slice(extra);
    let child = command(root, &vars)
        .args(["--json", "--progress", "jsonl", "train"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let started = Instant::now();
    loop {
        let j = synk(root, &["wait", "--timeout", "0s"]);
        let training = j["data"]["progress"]["step"] == "train"
            || j["data"]["steps"]
                .as_array()
                .is_some_and(|s| s.iter().any(|s| s["step"] == "dataset"));
        if j["data"]["state"] == "running" && training {
            break;
        }
        assert!(
            started.elapsed() < Duration::from_secs(60),
            "never trained: {j}"
        );
        std::thread::sleep(Duration::from_millis(100));
    }
    for _ in 0..presses {
        let mut text = fs::read_to_string(&presses_file).unwrap_or_default();
        text.push_str("press\n");
        fs::write(&presses_file, text).unwrap();
        std::thread::sleep(Duration::from_millis(150));
    }
    let out = child.wait_with_output().unwrap();
    let value: Value = serde_json::from_slice(&out.stdout).unwrap_or_else(|e| {
        panic!(
            "stdout must be exactly one JSON object ({e}): {}",
            String::from_utf8_lossy(&out.stdout)
        )
    });
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.lines().any(|l| l.contains(r#""event":"interrupt""#)),
        "{stderr}"
    );
    (value, out.status.code())
}

#[test]
fn ctrl_c_stops_at_a_safe_point_and_the_run_continues() {
    let dir = bigger_project();
    let root = dir.path();
    synk(root, &["dataset"]);
    let (stopped, code) = train_interrupted(root, 1, &[]);
    assert_eq!(code, Some(130), "{stopped}");
    assert_eq!(error_code(&stopped), "CANCELLED");
    assert_eq!(stopped["error"]["details"]["how"], "safe_stop", "{stopped}");
    let record = synk(root, &["wait", "run:1", "--timeout", "0s"]);
    assert_eq!(record["data"]["state"], "cancelled");
    assert_eq!(record["data"]["next"], "synk train run:1");
    // The lock is free, and the obvious retry continues the same run.
    let again = synk(root, &["train"]);
    assert_eq!(again["data"]["id"], "run:1-tiny-model", "{again}");
    assert_eq!(again["data"]["op"]["attempt"], 2);
}

#[test]
fn a_second_ctrl_c_stops_at_once() {
    let dir = bigger_project();
    let root = dir.path();
    synk(root, &["dataset"]);
    let (stopped, code) = train_interrupted(root, 2, &[("SYNK_FAKE_TRAINER_STUCK", "1")]);
    assert_eq!(code, Some(130), "{stopped}");
    assert_eq!(error_code(&stopped), "CANCELLED");
    assert_eq!(stopped["error"]["details"]["how"], "immediate", "{stopped}");
    assert_eq!(stopped["error"]["next"], "synk train run:1");
    let record = synk(root, &["wait", "run:1", "--timeout", "0s"]);
    assert_eq!(record["data"]["state"], "cancelled", "{record}");
    let again = synk(root, &["train"]);
    assert_eq!(again["data"]["op"]["attempt"], 2, "{again}");
}

#[test]
fn a_killed_run_reads_as_interrupted_at_once_and_continues() {
    let dir = bigger_project();
    let root = dir.path();
    synk(root, &["dataset"]);
    let mut child = command(root, &[("SYNK_FAKE_TRAINER_SLOW", "20000")])
        .args(["--json", "train"])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let started = Instant::now();
    while synk(root, &["wait", "--timeout", "0s"])["data"]["state"] != "running" {
        assert!(started.elapsed() < Duration::from_secs(30), "never started");
        std::thread::sleep(Duration::from_millis(100));
    }
    child.kill().unwrap();
    child.wait().unwrap();
    // No 2-minute heartbeat wait: the OS released the lock with the process.
    let record = synk(root, &["wait", "run:1", "--timeout", "0s"]);
    assert_eq!(record["data"]["state"], "interrupted", "{record}");
    let again = synk(root, &["train"]);
    assert_eq!(again["data"]["id"], "run:1-tiny-model", "{again}");
    assert_eq!(again["data"]["op"]["attempt"], 2);
}

#[test]
fn the_dry_run_warns_when_the_test_is_too_small_for_a_verdict() {
    let dir = synced_project();
    let root = dir.path();
    with_generator_and_tiny_model(root, &fake_llm());
    let preview = synk(root, &["train", "--dry-run"]);
    let dataset = preview["data"]["steps"]
        .as_array()
        .unwrap()
        .iter()
        .find(|s| s["step"] == "dataset")
        .unwrap()
        .clone();
    let tests = dataset["detail"]["test_documents"].as_u64().unwrap();
    assert!(tests >= 1, "{dataset}");
    assert!(dataset["detail"]["train_documents"].as_u64().unwrap() >= 1);
    let advice = preview["data"]["advice"].as_array().unwrap();
    let warning = advice
        .iter()
        .find(|a| a["code"] == "TEST_SET_UNDERPOWERED")
        .unwrap_or_else(|| panic!("no warning: {preview}"));
    assert_eq!(warning["evidence"]["predicted"], true);
    assert_eq!(warning["evidence"]["test_documents"], tests);
    assert_eq!(
        warning["action"]["command"],
        "synk queries template --out searches.jsonl"
    );

    // After the run, the verdict says why there's no clear change.
    let trained = synk(root, &["train"]);
    let verdict = &trained["data"]["verdict"];
    assert_eq!(verdict["method"], "paired_t", "{verdict}");
    if verdict["outcome"] == "no_clear_change" {
        assert_eq!(verdict["reason"], "too_few_test_documents", "{verdict}");
    }

    // test_percent takes "auto" (the default) or a number.
    assert_eq!(
        synk(root, &["config", "set", "dataset.test_percent", "50"])["ok"],
        true
    );
    assert_eq!(
        synk(root, &["config", "set", "dataset.test_percent", "auto"])["ok"],
        true
    );
}

#[test]
fn a_template_of_searches_imports_and_style_examples_stay_fixed() {
    let dir = bigger_project();
    let root = dir.path();
    sync(root);
    let made = synk(
        root,
        &[
            "queries",
            "template",
            "--out",
            "searches.jsonl",
            "--documents",
            "12",
        ],
    );
    assert_eq!(made["data"]["documents"], 12, "{made}");
    assert!(made["data"]["test_documents"].as_u64().unwrap() >= 1);
    assert_eq!(made["data"]["next"], "synk queries import searches.jsonl");
    let again = synk(root, &["queries", "template", "--out", "searches.jsonl"]);
    assert_eq!(error_code(&again), "INVALID_ARGS");

    // Fill in 8 of the 12, as a person would; leave the rest empty.
    let text = fs::read_to_string(root.join("searches.jsonl")).unwrap();
    let mut filled = 0;
    let lines: Vec<String> = text
        .lines()
        .map(|l| {
            if l.contains("\"query\":\"\"") && filled < 8 {
                filled += 1;
                l.replace("\"query\":\"\"", &format!("\"query\":\"search {filled}\""))
            } else {
                l.to_owned()
            }
        })
        .collect();
    fs::write(root.join("searches.jsonl"), lines.join("\n")).unwrap();
    let imported = synk(root, &["queries", "import", "searches.jsonl"]);
    assert_eq!(imported["data"]["added"], 8, "{imported}");
    assert_eq!(imported["data"]["skipped_empty"], 4);
    assert_eq!(imported["data"]["documents"], 8);
    assert_eq!(imported["data"]["decides_verdict"], false);

    // The dry run says the examples are about to be chosen.
    let step = dataset_step(&synk(root, &["dataset", "--dry-run"]));
    assert_eq!(step["detail"]["style_examples_change"], true, "{step}");
    assert!(
        step["why"]
            .as_array()
            .unwrap()
            .iter()
            .any(|w| w.as_str().unwrap().contains("style examples")),
        "{step}"
    );

    // Build once: 2 of the 8 become style examples, then stay.
    synk(root, &["dataset"]);
    let listed = synk(root, &["queries", "list"]);
    let style = |v: &Value| {
        v["data"]["queries"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|q| q["style"] == true)
            .map(|q| q["query"].as_str().unwrap().to_owned())
            .collect::<Vec<_>>()
    };
    let before = style(&listed);
    assert_eq!(before.len(), 2, "{listed}");
    // More searches: the examples don't change, so the LLM isn't re-asked.
    fs::write(
        root.join("more.jsonl"),
        "{\"query\": \"another search\", \"doc\": \"n1.md\"}\n{\"query\": \"and one more\", \"doc\": \"n2.md\"}\n{\"query\": \"a third\", \"doc\": \"n3.md\"}\n{\"query\": \"a fourth\", \"doc\": \"n4.md\"}\n",
    )
    .unwrap();
    synk(root, &["queries", "import", "more.jsonl"]);
    assert_eq!(style(&synk(root, &["queries", "list"])), before);
    let preview = synk(root, &["dataset", "--dry-run"]);
    let dataset = preview["data"]["steps"]
        .as_array()
        .unwrap()
        .iter()
        .find(|s| s["step"] == "dataset")
        .unwrap()
        .clone();
    assert_eq!(dataset["detail"]["llm_calls"], 0, "{dataset}");
    // --restyle chooses again.
    let restyled = synk(root, &["queries", "import", "more.jsonl", "--restyle"]);
    assert_eq!(restyled["data"]["restyled"], true);
}

/// A project of `n` notes, each its own topic, with the fake generator and
/// the tiny model.
fn project_of(n: usize) -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    fs::create_dir(root.join("notes")).unwrap();
    add_notes(root, 0..n);
    synk(root, &["init"]);
    append_config(root, "\n[eval]\ngeneral_check = false\n");
    synk(root, &["source", "add", "notes"]);
    with_generator_and_tiny_model(root, &fake_llm());
    dir
}

fn add_notes(root: &Path, range: std::ops::Range<usize>) {
    for i in range {
        fs::write(
            root.join(format!("notes/n{i}.md")),
            format!("# Note {i}\n\nTopic number {i} is about subject {i}."),
        )
        .unwrap();
    }
}

/// The test documents of the dataset `synk dataset` just built.
fn built_test_documents(root: &Path) -> Vec<String> {
    let built = synk(root, &["dataset"]);
    assert_eq!(built["ok"], true, "{built}");
    let key = built["data"]["op"]["key"].as_str().unwrap();
    let corpus =
        fs::read_to_string(root.join(format!(".synk/artifacts/dataset/{key}/corpus.jsonl")))
            .unwrap();
    let mut docs: Vec<String> = corpus
        .lines()
        .map(|l| serde_json::from_str::<Value>(l).unwrap())
        .filter(|c| c["split"] == "test")
        .map(|c| c["doc"].as_str().unwrap().to_owned())
        .collect();
    docs.sort();
    docs.dedup();
    docs
}

fn dataset_step(preview: &Value) -> Value {
    preview["data"]["steps"]
        .as_array()
        .unwrap_or_else(|| panic!("no steps: {preview}"))
        .iter()
        .find(|s| s["step"] == "dataset")
        .unwrap()
        .clone()
}

#[test]
fn auto_holds_out_enough_test_documents_and_adding_documents_keeps_them() {
    let dir = project_of(11);
    let root = dir.path();
    // 11 documents: 5 test documents (at most half), not 20%'s 2.
    let first = built_test_documents(root);
    assert_eq!(first.len(), 5, "{first:?}");
    // More documents: every test document stays one, and the test grows
    // to 10.
    add_notes(root, 11..26);
    let second = built_test_documents(root);
    assert!(
        first.iter().all(|d| second.contains(d)),
        "{first:?} -> {second:?}"
    );
    assert!(second.len() >= 10, "{second:?}");
}

#[test]
fn a_big_project_keeps_the_plain_twenty_percent_split() {
    let dir = project_of(200);
    let root = dir.path();
    let auto = dataset_step(&synk(root, &["train", "--dry-run"]));
    let plain = dataset_step(&synk(
        root,
        &["train", "--dry-run", "--set", "dataset.test_percent=20"],
    ));
    assert!(auto["detail"]["test_documents"].as_u64().unwrap() >= 10);
    for field in ["test_documents", "test_chunks", "train_chunks"] {
        assert_eq!(auto["detail"][field], plain["detail"][field], "{field}");
    }
}

#[test]
fn a_sample_keeps_every_test_document() {
    let dir = project_of(30);
    let root = dir.path();
    let full = synk(root, &["train", "--dry-run"]);
    let tests = dataset_step(&full)["detail"]["test_documents"].clone();
    assert_eq!(tests, 10, "{full}");
    let sampled = synk(root, &["train", "--dry-run", "--sample", "14"]);
    let step = dataset_step(&sampled);
    assert_eq!(step["detail"]["chunks"], 14, "{step}");
    assert_eq!(step["detail"]["test_documents"], tests, "{step}");
}

#[test]
fn of_two_trains_started_together_exactly_one_runs() {
    let dir = bigger_project();
    let root = dir.path();
    synk(root, &["dataset"]);
    let start = || {
        command(root, &[("SYNK_FAKE_TRAINER_SLOW", "1000")])
            .args(["--json", "train"])
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .unwrap()
    };
    let (a, b) = (start(), start());
    let results = [a, b].map(|c| parse(&c.wait_with_output().unwrap()));
    let blocked: Vec<&Value> = results.iter().filter(|r| r["ok"] == false).collect();
    assert_eq!(blocked.len(), 1, "{results:?}");
    assert_eq!(error_code(blocked[0]), "JOB_RUNNING", "{}", blocked[0]);
}

#[test]
fn real_searches_decide_the_verdict_at_thirty_about_ten_documents() {
    let dir = project_of(30);
    let root = dir.path();
    sync(root);
    // The template lists the test documents first.
    synk(
        root,
        &[
            "queries",
            "template",
            "--out",
            "t.jsonl",
            "--documents",
            "10",
        ],
    );
    let docs: Vec<String> = fs::read_to_string(root.join("t.jsonl"))
        .unwrap()
        .lines()
        .skip(1)
        .map(|l| {
            serde_json::from_str::<Value>(l).unwrap()["doc"]
                .as_str()
                .unwrap()
                .to_owned()
        })
        .collect();
    let searches = |docs: &[String]| {
        let mut text = String::new();
        for doc in docs {
            for k in 0..5 {
                text.push_str(&format!(
                    "{{\"query\": \"{doc} search {k}\", \"doc\": \"{doc}\"}}\n"
                ));
            }
        }
        text
    };
    // 45 searches about 9 documents: they don't decide yet.
    fs::write(root.join("s.jsonl"), searches(&docs[..9])).unwrap();
    synk(root, &["queries", "import", "s.jsonl"]);
    let trained = synk(root, &["train"]);
    assert_ne!(
        trained["data"]["verdict"]["query_kind"], "real",
        "{trained}"
    );
    let advice = trained["data"]["advice"]
        .as_array()
        .unwrap()
        .iter()
        .find(|a| a["code"] == "REAL_QUERIES_FEW_DOCUMENTS")
        .unwrap_or_else(|| panic!("no advice: {trained}"))
        .clone();
    assert_eq!(advice["evidence"]["more_documents"], 1, "{advice}");
    // The tenth document's searches: now they decide.
    fs::write(root.join("s.jsonl"), searches(&docs)).unwrap();
    synk(root, &["queries", "import", "s.jsonl"]);
    let trained = synk(root, &["train"]);
    assert_eq!(
        trained["data"]["verdict"]["query_kind"], "real",
        "{trained}"
    );
}

/// A real Ctrl-C (SIGINT), not the test hook.
#[cfg(unix)]
#[test]
fn sigint_stops_at_a_safe_point() {
    let dir = bigger_project();
    let root = dir.path();
    synk(root, &["dataset"]);
    let child = command(root, &[("SYNK_FAKE_TRAINER_SLOW", "20000")])
        .args(["--json", "train"])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let started = Instant::now();
    loop {
        let j = synk(root, &["wait", "--timeout", "0s"]);
        let training = j["data"]["progress"]["step"] == "train"
            || j["data"]["steps"]
                .as_array()
                .is_some_and(|s| s.iter().any(|s| s["step"] == "dataset"));
        if j["data"]["state"] == "running" && training {
            break;
        }
        assert!(started.elapsed() < Duration::from_secs(60), "never trained");
        std::thread::sleep(Duration::from_millis(100));
    }
    let killed = Command::new("kill")
        .args(["-INT", &child.id().to_string()])
        .status()
        .unwrap();
    assert!(killed.success());
    let out = child.wait_with_output().unwrap();
    assert_eq!(out.status.code(), Some(130));
    let stopped: Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(error_code(&stopped), "CANCELLED", "{stopped}");
    assert_eq!(stopped["error"]["details"]["how"], "safe_stop");
}

#[test]
fn the_dry_run_checks_free_disk_space() {
    let dir = synced_project();
    let root = dir.path();
    with_generator_and_tiny_model(root, &fake_llm());
    // 4 KB of weights: about 1,000 parameters, so training writes ~20 KB.
    fs::write(root.join("tiny-model/model.safetensors"), vec![0u8; 4000]).unwrap();
    let train_args = ["train", "--dry-run"];

    let plenty = synk_env(
        root,
        &train_args,
        &[("SYNK_TEST_FREE_BYTES", "100000000000")],
    );
    assert!(
        plenty["data"]["blockers"].as_array().unwrap().is_empty(),
        "{plenty}"
    );
    let disk = &plenty["data"]["disk"][0];
    let needs: Vec<(&str, u64)> = disk["needs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|n| (n["what"].as_str().unwrap(), n["bytes"].as_u64().unwrap()))
        .collect();
    // The dataset (twice the documents' text, plus 1 MB) and training.
    assert_eq!(needs[0].0, "the dataset", "{plenty}");
    assert!(needs[0].1 > 1_000_000);
    assert_eq!(needs[1], ("training checkpoints and models", 20_000));
    assert_eq!(disk["need_bytes"], needs[0].1 + 20_000);

    let full = synk_env(root, &train_args, &[("SYNK_TEST_FREE_BYTES", "1000")]);
    let blocker = &full["data"]["blockers"][0];
    assert_eq!(blocker["code"], "INSUFFICIENT_DISK", "{full}");
    assert_eq!(blocker["class"], "resource");
    // Running refuses before anything starts.
    let run = synk_env(root, &["train"], &[("SYNK_TEST_FREE_BYTES", "1000")]);
    assert_eq!(error_code(&run), "INSUFFICIENT_DISK", "{run}");

    // The escape hatch turns it into a warning.
    let warned = synk_env(
        root,
        &train_args,
        &[
            ("SYNK_TEST_FREE_BYTES", "1000"),
            ("SYNK_DISK_CHECK", "warn"),
        ],
    );
    assert!(warned["data"]["blockers"].as_array().unwrap().is_empty());
    assert!(
        warned["data"]["advice"]
            .as_array()
            .unwrap()
            .iter()
            .any(|a| a["code"] == "DISK_SPACE_TIGHT"),
        "{warned}"
    );
}

#[test]
fn text_is_normalized_and_an_old_store_is_read_again_once() {
    let dir = project_with_notes();
    let root = dir.path();
    // "Tiếng Việt" typed as base letters plus combining marks (NFD).
    fs::write(
        root.join("notes/nfd.md"),
        "# Tie\u{0302}\u{0301}ng Vie\u{0323}\u{0302}t\n\nGhi chu\u{0301} ve\u{0300} vie\u{0323}c sao lu\u{031B}u.",
    )
    .unwrap();
    synk(root, &["source", "add", "notes"]);
    let first = synk(root, &["source", "sync"]);
    assert_eq!(first["data"]["sync"]["reread"], false, "{first}");
    let db = rusqlite::Connection::open(root.join(".synk/synk.db")).unwrap();
    let text: String = db
        .query_row(
            "SELECT text FROM documents WHERE id LIKE '%nfd.md'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert!(text.contains("Tiếng Việt"), "stored text is NFC: {text:?}");

    // A store written by an older reader: every file is read again, once.
    db.execute(
        "UPDATE meta SET value = '1' WHERE key = 'reader_version'",
        [],
    )
    .unwrap();
    drop(db);
    let again = synk(root, &["source", "sync"]);
    assert_eq!(again["data"]["sync"]["reread"], true, "{again}");
    assert!(again["data"]["sync"]["totals"]["updated"].as_u64().unwrap() >= 1);
    let third = synk(root, &["source", "sync"]);
    assert_eq!(third["data"]["sync"]["reread"], false);
    assert_eq!(third["data"]["sync"]["totals"]["updated"], 0);
}

#[test]
fn the_test_size_warning_comes_before_the_llm_is_there() {
    // A first run: the LLM isn't reachable yet (nothing listens on port 9),
    // so the dataset can't be planned, but its test documents are known.
    let dir = synced_project();
    let root = dir.path();
    with_generator_and_tiny_model(root, "http://127.0.0.1:9/v1");
    let preview = synk(root, &["train", "--dry-run"]);
    assert!(
        preview["data"]["advice"]
            .as_array()
            .unwrap()
            .iter()
            .any(|a| a["code"] == "TEST_SET_UNDERPOWERED"),
        "{preview}"
    );
}

#[test]
fn a_failing_generator_passes_on_the_servers_message() {
    let dir = synced_project();
    let root = dir.path();
    let url = serve_llm(|_| {
        (
            500,
            r#"{"error":"llama runner process has terminated"}"#.to_owned(),
        )
    });
    with_generator_and_tiny_model(root, &url);
    let out = synk(root, &["dataset"]);
    assert_eq!(error_code(&out), "GENERATOR_UNAVAILABLE");
    let message = out["error"]["message"].as_str().unwrap();
    assert!(message.contains("HTTP 500"), "{message}");
    assert!(
        message.contains("llama runner process has terminated"),
        "{message}"
    );
}

#[test]
fn a_chunk_the_server_fails_on_is_skipped() {
    let dir = project_with_notes();
    let root = dir.path();
    for i in 0..20 {
        fs::write(
            root.join(format!("notes/n{i}.md")),
            format!("# Note {i}\n\nTopic number {i} is about subject {i}."),
        )
        .unwrap();
    }
    synk(root, &["source", "add", "notes", "handbook.md"]);
    // Ollama answers 500 when a model loops on one chunk.
    let url = serve_llm(|body| {
        if String::from_utf8_lossy(body).contains("about subject 7.") {
            (
                500,
                r#"{"error":"prediction aborted, token repeat limit reached"}"#.to_owned(),
            )
        } else {
            fake_reply(body)
        }
    });
    with_generator_and_tiny_model(root, &url);
    let out = synk(root, &["dataset"]);
    assert_eq!(out["ok"], true, "{out}");
    let text = out.to_string();
    assert!(text.contains(r#""failed_chunks":1"#), "{text}");
}

#[test]
fn blocked_runs_say_why_and_how_to_fix() {
    let dir = synced_project();
    let root = dir.path();
    // Nothing listens on port 9 (discard), so the connection is refused.
    with_generator_and_tiny_model(root, "http://127.0.0.1:9/v1");
    let preview = synk(root, &["train", "--dry-run"]);
    let blocker = &preview["data"]["blockers"][0];
    assert_eq!(blocker["code"], "GENERATOR_UNAVAILABLE");
    assert_eq!(
        blocker["actions"][0]["argv"],
        serde_json::json!(["ollama", "serve"])
    );
    let out = synk(root, &["train", "--yes"]);
    assert_eq!(error_code(&out), "GENERATOR_UNAVAILABLE");
    assert_eq!(out["error"]["retryable"], true);

    let config = root.join("synk.toml");
    let text = fs::read_to_string(&config)
        .unwrap()
        .replace("127.0.0.1:9", "llm.example.com");
    fs::write(&config, text).unwrap();
    let preview = synk(root, &["train", "--dry-run"]);
    let blocker = &preview["data"]["blockers"][0];
    assert_eq!(blocker["code"], "REMOTE_NOT_ALLOWED");
    // Sending documents away is the user's call: no automatic `next`.
    assert!(blocker["next"].is_null());
    assert_eq!(blocker["actions"][0]["applicability"], "manual");
}

#[test]
fn a_dataset_needs_two_documents() {
    let dir = project_with_notes();
    let root = dir.path();
    synk(root, &["source", "add", "handbook.md"]);
    sync(root);
    with_generator_and_tiny_model(root, &fake_llm());
    let preview = synk(root, &["train", "--dry-run"]);
    assert_eq!(preview["data"]["blockers"][0]["code"], "DATASET_TOO_SMALL");
    assert_eq!(
        error_code(&synk(root, &["train", "--yes"])),
        "DATASET_TOO_SMALL"
    );
}

#[test]
fn custom_ollama_tags_are_rejected_before_training() {
    let dir = synced_project();
    let root = dir.path();
    with_generator(root, &fake_llm());
    append_config(root, "\n[model]\nbase = \"nomic-embed-text:latest\"\n");
    assert_eq!(
        error_code(&synk(root, &["train", "--dry-run"])),
        "MODEL_NOT_TRAINABLE"
    );
}

#[test]
fn setup_rejects_backends_pytorch_does_not_build() {
    let dir = tempfile::tempdir().unwrap();
    let out = synk(dir.path(), &["setup", "--backend", "tpu"]);
    assert_eq!(error_code(&out), "INVALID_ARGS");
    // AMD and Apple GPUs aren't supported until they can be tested.
    for untested in ["rocm", "mps"] {
        let out = synk_env(
            dir.path(),
            &["setup", "--backend", untested],
            &[("SYNK_FAKE_TRAINER", "0")],
        );
        assert_eq!(error_code(&out), "DEVICE_UNSUPPORTED");
        assert_eq!(out["error"]["next"], "synk setup --backend cpu");
    }
}

#[test]
fn a_dataset_reports_advice_that_points_back_to_train() {
    let dir = synced_project();
    let root = dir.path();
    with_generator_and_tiny_model(root, &fake_llm());
    let built = synk(root, &["dataset"]);
    let result = &built["data"];
    assert_fields(
        result,
        &[
            "id",
            "op.kind",
            "dataset",
            "dataset_summary.documents",
            "dataset_summary.chunks",
            "dataset_summary.queries",
            "dataset_summary.queries_per_chunk",
            "dataset_summary.duplicate_queries",
            "dataset_summary.chunking",
            "advice",
        ],
    );
    assert_eq!(result["id"], "dataset:1-fake");
    assert_eq!(result["dataset"], "dataset:1-fake");
    assert!(result["run"].is_null());
    assert_eq!(result["dataset_summary"]["documents"], 3);
    assert_eq!(result["dataset_summary"]["queries"], 6);
    let few = result["advice"]
        .as_array()
        .unwrap()
        .iter()
        .find(|a| a["code"] == "FEW_QUERIES")
        .unwrap();
    assert_fields(
        few,
        &[
            "severity",
            "message",
            "evidence",
            "action.command",
            "applicability",
        ],
    );
    // `auto` already asks for the most queries per chunk, so the advice is
    // to add documents, then look again.
    assert!(
        few["action"]["command"]
            .as_str()
            .unwrap()
            .ends_with("synk train --dry-run")
    );
    assert_eq!(few["applicability"], "manual");
    // The dataset is reused, and the preview says so.
    let preview = synk(root, &["dataset", "--dry-run"]);
    let dataset = preview["data"]["steps"]
        .as_array()
        .unwrap()
        .iter()
        .find(|s| s["step"] == "dataset")
        .unwrap();
    assert_eq!(dataset["action"], "cached");
    assert_eq!(
        synk(root, &["status"])["data"]["pipeline"]["dataset"]["id"],
        result["dataset"]
    );
    let waited = synk(root, &["wait", "dataset:1"]);
    assert_eq!(waited["data"]["state"], "succeeded");
    assert_eq!(
        synk(root, &["run", "list"])["data"]["runs"],
        serde_json::json!([])
    );
}

#[test]
fn deprecated_chunk_chars_still_works_with_advice() {
    let dir = synced_project();
    let root = dir.path();
    with_generator_and_tiny_model(root, &fake_llm());
    append_config(root, "\n[dataset]\nchunk_chars = 500\n");
    let out = synk(root, &["dataset"]);
    let result = &out["data"];
    assert_eq!(result["dataset_summary"]["chunking"]["unit"], "chars");
    let codes: Vec<&str> = result["advice"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|a| a["code"].as_str())
        .collect();
    assert!(codes.contains(&"CHUNK_CHARS_DEPRECATED"), "{codes:?}");
}

#[test]
fn invalid_settings_are_config_errors() {
    let dir = synced_project();
    let root = dir.path();
    append_config(root, "\n[trainer]\nepochs = \"many\"\n");
    assert_eq!(error_code(&synk(root, &["status"])), "CONFIG_INVALID");
    let dir = synced_project();
    append_config(dir.path(), "\n[trainer]\nfinetune = \"partial\"\n");
    assert_eq!(
        error_code(&synk(dir.path(), &["train", "--dry-run"])),
        "CONFIG_INVALID"
    );
}

#[test]
fn progress_jsonl_goes_to_stderr() {
    let dir = synced_project();
    let root = dir.path();
    with_generator_and_tiny_model(root, &fake_llm());
    let out = command(root, &[])
        .args(["--json", "--progress", "jsonl", "dataset"])
        .output()
        .unwrap();
    let result = parse(&out);
    assert_eq!(result["ok"], true);
    let events: Vec<Value> = String::from_utf8_lossy(&out.stderr)
        .lines()
        .map(|l| serde_json::from_str(l).unwrap_or_else(|e| panic!("{e}: {l}")))
        .collect();
    assert!(
        events
            .iter()
            .any(|e| e["event"] == "progress" && e["stage"] == "generate")
    );
    assert!(events.iter().all(|e| e["v"] == 1));
    assert_eq!(events[0]["event"], "op");
    assert_eq!(events[0]["id"], result["data"]["id"]);
}

#[test]
fn real_queries_import_atomically_and_reach_the_dataset() {
    let dir = synced_project();
    let root = dir.path();
    with_generator_and_tiny_model(root, &fake_llm());
    fs::write(
        root.join("bad.jsonl"),
        "{\"query\": \"alpha?\", \"doc\": \"a.md\"}\n{\"query\": \"nope\", \"doc\": \"missing.md\"}\n",
    )
    .unwrap();
    let bad = synk(root, &["queries", "import", "bad.jsonl"]);
    assert_eq!(error_code(&bad), "QUERIES_INVALID");
    assert!(bad["error"]["message"].as_str().unwrap().contains("line 2"));
    assert_eq!(
        synk(root, &["queries", "list"])["data"]["queries"]
            .as_array()
            .unwrap()
            .len(),
        0,
        "a bad line rejects the whole file"
    );

    fs::write(
        root.join("good.jsonl"),
        "{\"query\": \"what is alpha\", \"doc\": \"a.md\"}\n{\"query\": \"beta?\", \"text\": \"BETA\"}\n",
    )
    .unwrap();
    let good = synk(root, &["queries", "import", "good.jsonl"]);
    assert_fields(
        &good,
        &[
            "data.added",
            "data.unchanged",
            "data.removed",
            "data.total",
            "data.next",
        ],
    );
    assert_eq!(good["data"]["added"], 2);
    assert_eq!(
        synk(root, &["queries", "import", "good.jsonl"])["data"]["unchanged"],
        2
    );
    let list = synk(root, &["queries", "list"]);
    assert_eq!(list["data"]["queries"][1]["doc"], "notes/b.txt");

    let dataset = synk(root, &["dataset"]);
    let summary = &dataset["data"]["dataset_summary"];
    assert_eq!(summary["real_queries"], 2);
    assert_eq!(summary["queries"], 6, "real queries don't train");
}

#[test]
fn run_views_explain_what_is_missing() {
    let dir = synced_project();
    let root = dir.path();
    let runs = synk(root, &["run", "list"]);
    assert_eq!(runs["data"]["runs"].as_array().unwrap().len(), 0);
    let out = synk(root, &["run", "inspect"]);
    assert_eq!(error_code(&out), "RUN_NOT_FOUND");
    assert_eq!(out["error"]["next"], "synk train --dry-run");
    let out = synk(root, &["export", "--out", "model"]);
    assert_eq!(error_code(&out), "RUN_NOT_FOUND");
    let models = synk(root, &["model", "list"]);
    assert_fields(
        &models["data"]["rerankers"][0],
        &["id", "hf", "revision", "parameters"],
    );
}

#[test]
fn schema_and_skill_describe_the_contract() {
    let dir = tempfile::tempdir().unwrap();
    let schema = synk(dir.path(), &["schema"]);
    assert_fields(
        &schema,
        &[
            "data.contract_version",
            "data.commands.subcommands",
            "data.schemas.error",
            "data.schemas.preview",
            "data.schemas.op",
            "data.error_codes",
            "data.advice_codes",
        ],
    );
    let codes: Vec<&str> = schema["data"]["error_codes"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|c| c["code"].as_str())
        .collect();
    for code in [
        "RUN_NAME_TAKEN",
        "CONSENT_REQUIRED",
        "JOB_RUNNING",
        "EXPORT_CHECK_FAILED",
    ] {
        assert!(codes.contains(&code), "{code} missing");
    }
    let train = synk(dir.path(), &["schema", "train"]);
    assert_eq!(train["data"]["commands"]["name"], "train");
    assert_eq!(
        error_code(&synk(dir.path(), &["schema", "nope"])),
        "INVALID_ARGS"
    );

    let skill = synk(dir.path(), &["skill", "--out", "skills/synk"]);
    assert_eq!(skill["ok"], true);
    let text = fs::read_to_string(dir.path().join("skills/synk/SKILL.md")).unwrap();
    assert!(text.starts_with("---\nname: synk\n"), "{text}");
    assert!(dir.path().join("skills/synk/references/codes.md").is_file());
}

/// Trains for real. Needs a Python with torch, sentence-transformers[train]
/// and peft, and downloads multilingual-e5-small and the general check's
/// sets on first run:
///
///     SYNK_E2E_PYTHON=/path/to/python cargo test -- --ignored
#[test]
#[ignore = "needs SYNK_E2E_PYTHON and network access to download a model"]
fn train_evaluates_and_exports() {
    let python = std::env::var("SYNK_E2E_PYTHON").expect("set SYNK_E2E_PYTHON");
    let real = [("SYNK_FAKE_TRAINER", "0")];
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    fs::create_dir(root.join("notes")).unwrap();
    for (i, text) in [
        "Backups run every night at two with pg_dump, kept for thirty days.",
        "Nginx keepalive_timeout is set to 75 seconds for the API servers.",
        "Sao lưu cơ sở dữ liệu chạy lúc hai giờ sáng mỗi ngày.",
        "Tăng giới hạn kết nối của PostgreSQL lên 500 khi tải cao.",
        "Deploys go through the staging cluster first, then production.",
        "Mật khẩu VPN được đổi mỗi quý bởi đội hạ tầng.",
        "The on-call rotation changes every Monday at ten.",
        "Nhật ký ứng dụng được giữ 14 ngày trong Loki.",
    ]
    .iter()
    .enumerate()
    {
        fs::write(
            root.join(format!("notes/n{i}.md")),
            format!("# Note {i}\n\n{text}"),
        )
        .unwrap();
    }
    synk(root, &["init"]);
    synk(root, &["source", "add", "notes"]);
    with_generator(root, &fake_llm());
    append_config(
        root,
        &format!(
            "\n[trainer]\npython = '{python}'\nbatch_size = 2\nreranker = \"mmarco-minilm-l12\"\nfinetune = \"lora\"\n"
        ),
    );
    fs::write(
        root.join("real.jsonl"),
        "{\"query\": \"backup schedule\", \"doc\": \"n0.md\"}\n{\"query\": \"nginx timeout\", \"doc\": \"n1.md\"}\n",
    )
    .unwrap();
    synk(root, &["queries", "import", "real.jsonl"]);

    let preview = synk_env(root, &["train", "--dry-run"], &real);
    assert!(
        preview["data"]["blockers"].as_array().unwrap().is_empty(),
        "{preview}"
    );
    let out = synk_env(root, &["train", "--yes"], &real);
    assert_eq!(out["data"]["op"]["state"], "succeeded", "{out}");
    let result = &out["data"];
    assert_fields(
        result,
        &[
            "run",
            "verdict.outcome",
            "verdict.ci_low",
            "verdict.by_lang",
            "general_check.en",
            "general_check.vi",
            "advice",
        ],
    );
    let run = result["run"].as_str().unwrap().to_owned();
    let shown = synk_env(root, &["run", "show", &run], &real);
    assert_eq!(shown["data"]["summary"]["finetune"], "lora");

    let exported = synk_env(root, &["export", &run, "--out", "tuned"], &real);
    assert_eq!(exported["data"]["check"]["status"], "passed");
    assert!(root.join("tuned/modules.json").is_file());
    let readme = fs::read_to_string(root.join("tuned/README.md")).unwrap();
    for private in ["pg_dump", "Nginx", "where is note"] {
        assert!(
            !readme.contains(private),
            "README leaks `{private}`:\n{readme}"
        );
    }
    let onnx = synk_env(
        root,
        &["export", &run, "--out", "tuned-onnx", "--format", "onnx"],
        &real,
    );
    assert_eq!(onnx["data"]["check"]["status"], "passed", "{onnx}");

    // A second run with another setting, compared with the first.
    let second = synk_env(root, &["train", "--set", "trainer.negatives=1"], &real);
    let second_run = second["data"]["run"].as_str().unwrap().to_owned();
    let compared = synk_env(root, &["run", "compare", &run, &second_run], &real);
    assert_fields(&compared, &["data.a", "data.b", "data.verdict.outcome"]);
}

// --- human output ----------------------------------------------------------

/// Runs `synk <args>` without `--json`: (stdout, stderr, exit code).
fn synk_human(dir: &Path, args: &[&str]) -> (String, String, i32) {
    let out = command(dir, &[]).args(args).output().unwrap();
    (
        String::from_utf8(out.stdout).expect("stdout is UTF-8"),
        String::from_utf8(out.stderr).expect("stderr is UTF-8"),
        out.status.code().unwrap_or(-1),
    )
}

#[test]
fn usage_errors_exit_2_and_suggest_the_command() {
    let dir = tempfile::tempdir().unwrap();
    for json in [true, false] {
        let mut cmd = command(dir.path(), &[]);
        if json {
            cmd.arg("--json");
        }
        let out = cmd.arg("trian").output().unwrap();
        assert_eq!(out.status.code(), Some(2), "json: {json}");
        if json {
            let value: Value = serde_json::from_slice(&out.stdout).unwrap();
            assert_eq!(error_code(&value), "INVALID_ARGS");
            assert_eq!(value["error"]["next"], "synk train");
        } else {
            let stderr = String::from_utf8_lossy(&out.stderr);
            assert!(stderr.contains("Usage: synk "), "{stderr}");
        }
    }
}

#[test]
fn vietnamese_names_and_queries_print_intact() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    fs::create_dir(root.join("ghi chú")).unwrap();
    fs::write(
        root.join("ghi chú/cấu hình.md"),
        "# Cấu hình nginx\n\nTăng keepalive_timeout lên 75 giây.",
    )
    .unwrap();
    fs::write(root.join("ghi chú/sao lưu.md"), "Sao lưu hằng đêm.").unwrap();
    synk(root, &["init"]);
    let (out, _, code) = synk_human(root, &["source", "add", "ghi chú"]);
    assert_eq!(code, 0);
    assert!(out.contains("ghi chú"), "{out}");
    let (out, err, code) = synk_human(root, &["source", "sync"]);
    assert_eq!(code, 0, "{err}");
    assert!(out.contains("Documents: 2 new"), "{out}");

    fs::write(
        root.join("q.jsonl"),
        "{\"query\": \"tăng thời gian chờ\", \"doc\": \"cấu hình.md\"}\n",
    )
    .unwrap();
    synk(root, &["queries", "import", "q.jsonl"]);
    let (out, _, _) = synk_human(root, &["queries", "list"]);
    assert!(out.contains("tăng thời gian chờ"), "{out}");
    let list = synk(root, &["queries", "list"]);
    assert_eq!(list["data"]["queries"][0]["query"], "tăng thời gian chờ");
}

#[test]
fn human_output_is_plain_and_progress_has_no_carriage_returns() {
    let dir = synced_project();
    let root = dir.path();
    with_generator_and_tiny_model(root, &fake_llm());
    let (out, _, _) = synk_human(root, &["train", "--dry-run"]);
    assert!(out.starts_with("Would: a new run"), "{out}");
    assert!(out.contains("Nothing leaves this machine."), "{out}");
    let (out, err, code) = synk_human(root, &["dataset"]);
    assert_eq!(code, 0, "{err}");
    assert!(
        !err.contains('\r'),
        "no carriage returns when piped: {err:?}"
    );
    assert!(err.contains("Writing queries"), "{err}");
    assert!(out.contains("Dataset: dataset:"), "{out}");
    assert!(!out.contains("[FEW_QUERIES]"), "codes only with -v: {out}");
    let (out, _, _) = synk_human(root, &["-v", "dataset", "--force", "dataset"]);
    assert!(out.contains("note[FEW_QUERIES]"), "{out}");
}

// --- contract v1 -----------------------------------------------------------

#[test]
fn every_error_carries_a_class_and_runnable_next() {
    let dir = tempfile::tempdir().unwrap();
    let err = synk(dir.path(), &["status"]);
    assert_eq!(error_code(&err), "CONFIG_NOT_FOUND");
    assert_eq!(err["error"]["class"], "config");
    assert_eq!(err["error"]["retryable"], false);
    assert_eq!(err["error"]["next"], "synk init");
    // `next` is runnable as is; alternatives with placeholders are actions.
    let actions = err["error"]["actions"].as_array().unwrap();
    assert_eq!(actions[0]["argv"], serde_json::json!(["synk", "init"]));
    assert_eq!(actions[1]["applicability"], "manual");
}

#[test]
fn config_set_validates_and_keeps_comments() {
    let dir = project_with_notes();
    let root = dir.path();
    append_config(
        root,
        "\n[generator]\n# my favourite model\nmodel = \"qwen3:4b\"\n",
    );

    let got = synk(root, &["config", "get", "generator.model"]);
    assert_eq!(got["data"]["value"], "qwen3:4b");
    assert_eq!(got["data"]["set_in_file"], true);
    let default = synk(root, &["config", "get", "trainer.batch_size"]);
    assert_eq!(default["data"]["value"], 16);
    assert_eq!(default["data"]["set_in_file"], false);

    let set = synk(root, &["config", "set", "generator.queries_per_chunk", "5"]);
    assert_eq!(set["data"]["old"], "auto");
    assert_eq!(set["data"]["new"], 5);
    let set = synk(
        root,
        &["config", "set", "generator.languages", "[\"vi\", \"en\"]"],
    );
    assert_eq!(set["data"]["new"], serde_json::json!(["vi", "en"]));
    let set = synk(root, &["config", "set", "model.base", "hf:BAAI/bge-m3"]);
    assert_eq!(set["data"]["new"], "hf:BAAI/bge-m3");
    let text = fs::read_to_string(root.join("synk.toml")).unwrap();
    assert!(text.contains("# my favourite model"), "{text}");
    assert!(text.contains("queries_per_chunk = 5"), "{text}");

    let bad = synk(root, &["config", "set", "trainer.epochs", "many"]);
    assert_eq!(error_code(&bad), "CONFIG_INVALID");
    assert!(
        !fs::read_to_string(root.join("synk.toml"))
            .unwrap()
            .contains("many")
    );
    let unknown = synk(root, &["config", "set", "trainer.epoch", "3"]);
    assert_eq!(error_code(&unknown), "CONFIG_KEY_UNKNOWN");
    assert!(
        unknown["error"]["details"]["known"]
            .as_array()
            .unwrap()
            .iter()
            .any(|k| k == "trainer.epochs")
    );

    let unset = synk(root, &["config", "unset", "generator.queries_per_chunk"]);
    assert_eq!(unset["data"]["new"], "auto");
    let shown = synk(root, &["config", "show"]);
    assert_eq!(shown["data"]["settings"]["generator"]["model"], "qwen3:4b");
    assert!(
        shown["data"]["set_in_file"]
            .as_array()
            .unwrap()
            .iter()
            .any(|k| k == "model.base")
    );
}

#[test]
fn model_show_describes_catalog_and_custom_models() {
    let dir = tempfile::tempdir().unwrap();
    let e5 = synk(dir.path(), &["model", "show", "multilingual-e5-small"]);
    assert_eq!(e5["data"]["catalog"], true);
    assert_eq!(e5["data"]["kind"], "base");
    let custom = synk(dir.path(), &["model", "show", "hf:BAAI/bge-m3@abc"]);
    assert_eq!(custom["data"]["catalog"], false);
    assert_eq!(custom["data"]["spec"]["revision"], "abc");
    let reranker = synk(dir.path(), &["model", "show", "mmarco-minilm-l12"]);
    assert_eq!(reranker["data"]["kind"], "reranker");
}
