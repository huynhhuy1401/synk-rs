//! `synk schema` and `synk skill`: how agents learn the interface.
//!
//! Both are generated from the code (the clap tree, `ErrorCode::ALL`, the
//! advice codes), so they can't drift from what the binary does.

use std::fmt::Write as _;
use std::fs;
use std::path::Path;

use serde_json::{Value, json};
use synk_core::{ErrorCode, Result, SynkError};
use synk_pipeline::advice;

use crate::output::{CONTRACT_VERSION, Output};

const SKILL: &str = include_str!("../skill/SKILL.md");

/// Describes one command: its arguments and subcommands.
fn describe(cmd: &clap::Command) -> Value {
    let args: Vec<Value> = cmd
        .get_arguments()
        .filter(|a| !a.is_global_set() && a.get_id() != "help" && a.get_id() != "version")
        .map(|a| {
            json!({
                "name": a.get_long().map_or_else(|| a.get_id().to_string(), |l| format!("--{l}")),
                "positional": a.is_positional(),
                "required": a.is_required_set(),
                "repeatable": matches!(a.get_action(), clap::ArgAction::Append),
                "takes_value": a.get_action().takes_values(),
                "default": a.get_default_values().first().map(|v| v.to_string_lossy().into_owned()),
                "help": a.get_help().map(ToString::to_string),
            })
        })
        .collect();
    let subcommands: Vec<Value> = cmd
        .get_subcommands()
        .filter(|c| c.get_name() != "help")
        .map(describe)
        .collect();
    json!({
        "name": cmd.get_name(),
        "about": cmd.get_about().map(ToString::to_string),
        "args": args,
        "subcommands": subcommands,
    })
}

fn error_codes() -> Vec<Value> {
    ErrorCode::ALL
        .iter()
        .map(|c| {
            json!({
                "code": c.as_str(),
                "class": c.class(),
                "retryable": c.retryable(),
            })
        })
        .collect()
}

fn advice_codes() -> Vec<Value> {
    advice::CODES
        .iter()
        .map(|(code, meaning)| json!({"code": code, "meaning": meaning}))
        .collect()
}

/// JSON Schemas of the envelope and of the objects every command shares.
fn shared_schemas() -> Value {
    let action = json!({
        "type": "object",
        "required": ["applicability", "why"],
        "properties": {
            "argv": {"type": "array", "items": {"type": "string"}},
            "applicability": {"enum": ["machine", "review", "manual"]},
            "why": {"type": "string"},
        },
    });
    json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "envelope": {
            "type": "object",
            "required": ["v", "ok"],
            "properties": {
                "v": {"const": CONTRACT_VERSION},
                "ok": {"type": "boolean"},
                "data": {"type": "object"},
                "error": {"$ref": "#/error"},
            },
        },
        "error": {
            "type": "object",
            "required": ["code", "class", "retryable", "message"],
            "properties": {
                "code": {"type": "string", "description": "stable; see error_codes. New codes may be added"},
                "class": {"enum": ["usage", "config", "not_found", "conflict", "prerequisite", "unavailable", "data", "resource", "failed", "internal"]},
                "retryable": {"type": "boolean"},
                "message": {"type": "string"},
                "next": {"type": "string", "description": "a command that runs exactly as written"},
                "actions": {"type": "array", "items": action.clone()},
                "details": {"type": "object"},
            },
        },
        "advice": {
            "type": "object",
            "required": ["code", "severity", "message", "applicability"],
            "properties": {
                "code": {"type": "string"},
                "severity": {"enum": ["warning", "info"]},
                "message": {"type": "string"},
                "evidence": {"type": "object"},
                "action": {"type": "object", "properties": {
                    "command": {"type": "string"},
                    "config": {"type": "object", "description": "settings to change first, as {key: value}"},
                }},
                "applicability": {"enum": ["machine", "review", "manual"]},
            },
        },
        "preview": {
            "type": "object",
            "description": "train/dataset --dry-run; also error.details.dry_run of CONSENT_REQUIRED",
            "required": ["steps", "consent", "blockers", "totals", "fetch_names"],
            "properties": {
                "steps": {"type": "array", "items": {"type": "object", "properties": {
                    "step": {"enum": ["trainer", "models", "generator", "sync", "dataset", "train", "eval"]},
                    "action": {"enum": ["cached", "run", "blocked", "skipped"]},
                    "why": {"type": "array", "items": {"type": "string"}},
                    "est_seconds": {"type": "integer"},
                    "estimate_basis": {"enum": ["history", "default"]},
                }}},
                "consent": {"type": "array", "items": {"type": "object", "properties": {
                    "kind": {"enum": ["download", "egress"]},
                    "name": {"type": "string", "description": "downloads: the name synk fetch takes"},
                    "what": {"type": "string"},
                    "bytes": {"type": "integer"},
                    "host": {"type": "string"},
                    "chunks": {"type": "integer"},
                }}},
                "fetch_names": {"type": "array", "items": {"type": "string"}},
                "fetch": {"type": ["string", "null"], "description": "the synk fetch command for everything missing"},
                "egress": {"type": "array"},
                "blockers": {"type": "array", "items": {"$ref": "#/error"}},
                "totals": {"type": "object"},
                "would": {"type": "string"},
                "reuses": {"type": ["string", "null"]},
                "continues": {"type": ["string", "null"]},
                "next": {"type": ["string", "null"]},
            },
        },
        "op": {
            "type": "object",
            "description": "the record of a run, dataset build or fetch (synk wait, data.op of train/dataset/fetch)",
            "required": ["id", "kind", "n", "state", "done", "attempt", "steps"],
            "properties": {
                "id": {"type": "string", "pattern": "^(run|dataset):[0-9]+-|^fetch:[0-9]+$"},
                "kind": {"enum": ["run", "dataset", "fetch"]},
                "n": {"type": "integer"},
                "name": {"type": "string"},
                "state": {"enum": ["running", "succeeded", "failed", "cancelled", "interrupted"]},
                "done": {"type": "boolean"},
                "attempt": {"type": "integer"},
                "heartbeat_at": {"type": "integer", "description": "unix seconds; stale after 120 s means interrupted"},
                "settings": {"type": "object", "description": "overrides, config, sample, force as started"},
                "dataset": {"type": "string"},
                "key": {"type": "string"},
                "same_as": {"type": "string"},
                "progress": {"type": "object"},
                "result": {"type": "object"},
                "error": {"$ref": "#/error"},
                "next": {"type": "string"},
            },
        },
        "event": {
            "type": "object",
            "description": "stderr with --progress jsonl (and op starts under --json), and synk logs",
            "properties": {
                "v": {"const": 1},
                "event": {"enum": ["op", "progress", "log", "heartbeat"]},
                "id": {"type": "string", "description": "op events: the run, dataset or fetch ID"},
                "kind": {"enum": ["run", "dataset", "fetch"]},
                "attempt": {"type": "integer"},
                "stage": {"type": "string"},
                "step": {"type": "integer"},
                "total": {"type": "integer"},
                "unit": {"type": "string"},
                "message": {"type": "string"},
            },
        },
        "ids": {
            "run": "run:<n>-<base model or --name>",
            "dataset": "dataset:<n>-<generator>",
            "fetch": "fetch:<n>",
            "older": "run:<yyyy-mm-dd>-<base model>-<hash> still resolves",
            "aliases": {"run": ["<n>", "<name>", "latest", "best"], "op": ["current", "latest"]},
        },
    })
}

pub fn schema(command: &str) -> Result<Output> {
    let mut target = crate::cli_command();
    for part in command.split_whitespace() {
        let next = target
            .get_subcommands()
            .find(|c| c.get_name() == part)
            .cloned()
            .ok_or_else(|| {
                SynkError::new(ErrorCode::InvalidArgs, format!("no command `{command}`"))
                    .with_next("synk schema")
            })?;
        target = next;
    }
    let data = json!({
        "contract_version": CONTRACT_VERSION,
        "synk_version": env!("CARGO_PKG_VERSION"),
        "commands": describe(&target),
        "schemas": shared_schemas(),
        "error_codes": error_codes(),
        "advice_codes": advice_codes(),
        "stability": "Within contract v1, fields, codes and enum values are only added. Ignore what you don't know.",
    });
    let mut human = format!(
        "synk JSON contract v{CONTRACT_VERSION} ({} error codes, {} advice codes). Use --json for the full schema.\n",
        ErrorCode::ALL.len(),
        advice::CODES.len()
    );
    for c in target.get_subcommands().filter(|c| c.get_name() != "help") {
        let _ = writeln!(
            human,
            "  {:<9} {}",
            c.get_name(),
            c.get_about().map(ToString::to_string).unwrap_or_default()
        );
    }
    Ok(Output {
        human: human.trim_end().to_owned(),
        data,
    })
}

/// `references/codes.md`, generated.
fn codes_md() -> String {
    let mut out = String::from(
        "# synk codes\n\nGenerated by `synk skill`. Branch on `code`; on `class` for codes you don't know.\n\n## Error codes\n\n| Code | Class | Retryable |\n|---|---|---|\n",
    );
    for c in ErrorCode::ALL {
        let _ = writeln!(
            out,
            "| `{}` | {} | {} |",
            c.as_str(),
            serde_json::to_value(c.class())
                .ok()
                .and_then(|v| v.as_str().map(str::to_owned))
                .unwrap_or_default(),
            if c.retryable() { "yes" } else { "no" }
        );
    }
    out.push_str("\n## Advice codes\n\n| Code | Meaning |\n|---|---|\n");
    for (code, meaning) in advice::CODES {
        let _ = writeln!(out, "| `{code}` | {meaning} |");
    }
    out
}

pub fn skill(out: Option<&Path>) -> Result<Output> {
    let Some(dir) = out else {
        return Ok(Output {
            human: SKILL.trim_end().to_owned(),
            data: json!({ "skill": SKILL, "references": { "codes.md": codes_md() } }),
        });
    };
    let refs = dir.join("references");
    fs::create_dir_all(&refs).map_err(|e| SynkError::io(refs.display(), e))?;
    let skill_path = dir.join("SKILL.md");
    fs::write(&skill_path, SKILL).map_err(|e| SynkError::io(skill_path.display(), e))?;
    let codes_path = refs.join("codes.md");
    fs::write(&codes_path, codes_md()).map_err(|e| SynkError::io(codes_path.display(), e))?;
    let written = [
        skill_path.display().to_string(),
        codes_path.display().to_string(),
    ];
    Ok(Output {
        human: format!("Wrote {}", written.join(" and ")),
        data: json!({ "written": written }),
    })
}
