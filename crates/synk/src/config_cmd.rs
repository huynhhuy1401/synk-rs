//! `synk config show|get|set|unset`: settings in synk.toml, edited so that
//! comments survive and invalid values are refused before anything is saved.

use std::fmt::Write as _;

use serde_json::{Value, json};
use synk_core::Result;

use crate::config::Project;
use crate::output::{Output, show_path};

/// A value as it would be written in synk.toml.
fn toml_text(v: &Value) -> String {
    match v {
        Value::Null => "(not set)".to_owned(),
        Value::String(s) => format!("\"{s}\""),
        other => other.to_string(),
    }
}

pub fn show(project: &Project) -> Result<Output> {
    let effective = project.effective();
    let in_file = project.keys_in_file();
    let mut human = format!(
        "Settings ({}; * = set there):",
        show_path(&project.config_path)
    );
    for (section, keys) in effective.as_object().into_iter().flatten() {
        let _ = write!(human, "\n[{section}]");
        for (key, value) in keys.as_object().into_iter().flatten() {
            let full = format!("{section}.{key}");
            let mark = if in_file.contains(&full) { "*" } else { " " };
            let _ = write!(human, "\n {mark} {key:<18} {}", toml_text(value));
        }
    }
    Ok(Output {
        human,
        data: json!({
            "config": show_path(&project.config_path),
            "settings": effective,
            "set_in_file": in_file,
        }),
    })
}

pub fn get(project: &Project, key: &str) -> Result<Output> {
    let value = project.get(key)?;
    let set = project.keys_in_file().iter().any(|k| k == key);
    Ok(Output {
        human: toml_text(&value),
        data: json!({ "key": key, "value": value, "set_in_file": set }),
    })
}

pub fn set(mut project: Project, key: &str, value: &str) -> Result<Output> {
    let (old, new) = project.set(key, value)?;
    changed(key, &old, &new)
}

pub fn unset(mut project: Project, key: &str) -> Result<Output> {
    let (old, new) = project.unset(key)?;
    changed(key, &old, &new)
}

fn changed(key: &str, old: &Value, new: &Value) -> Result<Output> {
    let next = "synk status";
    Ok(Output {
        human: if old == new {
            format!("{key} = {} (unchanged)\nNext: {next}", toml_text(new))
        } else {
            format!(
                "{key}: {} -> {}\nNext: {next}",
                toml_text(old),
                toml_text(new)
            )
        },
        data: json!({ "key": key, "old": old, "new": new, "next": next }),
    })
}
