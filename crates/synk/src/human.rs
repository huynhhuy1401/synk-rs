//! Plain-language wording shared by commands. IDs, metric names and advice
//! codes appear only with `-v`; `--json` always has them.

use std::fmt::Write as _;

use serde::Serialize;
use serde_json::Value;

use crate::output::verbose;

/// What "search quality" means, printed once under a verdict.
pub const QUALITY_NOTE: &str =
    "Search quality is 0 to 1; 1 means the right chunk always comes first.";

/// A verdict in plain words, e.g. `better: search quality 0.64 -> 0.78, on
/// 60 test queries from 12 documents`, plus a per-language line when the
/// test set has more than one language. With `-v`, the statistical message.
pub fn verdict(v: &impl Serialize) -> String {
    let v = serde_json::to_value(v).unwrap_or_default();
    if v.is_null() {
        return "not tested".to_owned();
    }
    let num = |k: &str| v[k].as_f64().unwrap_or(0.0);
    let (base, tuned) = (num("base"), num("tuned"));
    let queries = v["test_queries"].as_u64().unwrap_or(0);
    let documents = v["test_documents"].as_u64().unwrap_or(0);
    let noun = if v["query_kind"] == "real" {
        "of your real queries"
    } else {
        "test queries"
    };
    let scores = format!("search quality {base:.2} -> {tuned:.2}");
    let on = format!("{queries} {noun} from {documents} documents");
    let mut out = match (v["outcome"].as_str(), v["ci_low"].as_f64()) {
        (_, None) if documents == 0 => {
            // Evaluated by an older synk, without the statistical test.
            format!("{scores} (tested by an older synk; train again for a verdict)")
        }
        (_, None) => format!("too few test documents to tell: {scores}, on {on}; add documents"),
        (Some("better"), _) => format!("better: {scores}, on {on}"),
        (Some("worse"), _) => format!("worse: {scores}, on {on}"),
        _ if v["ci_low"] == 0.0 && v["ci_high"] == 0.0 => format!(
            "no clear change: both models rank all {on} the same, so this test can't tell them apart"
        ),
        _ => format!(
            "no clear change: {scores}, within what chance can explain on {on} (changes under about {:.2} can't be seen with this many)",
            num("min_detectable_delta")
        ),
    };
    if let Some(langs) = v["by_lang"].as_object().filter(|l| l.len() > 1) {
        let parts: Vec<String> = langs
            .iter()
            .map(|(lang, l)| {
                format!(
                    "{} {:.2} -> {:.2}",
                    language(lang),
                    l["base"].as_f64().unwrap_or(0.0),
                    l["tuned"].as_f64().unwrap_or(0.0)
                )
            })
            .collect();
        let _ = write!(out, "\n  {}", parts.join(", "));
    }
    if verbose() {
        let _ = write!(out, "\n  {}", v["message"].as_str().unwrap_or_default());
    }
    out
}

/// A language name for a code the detector uses.
pub fn language(code: &str) -> String {
    match code {
        "vi" => "Vietnamese",
        "en" => "English",
        "zh" => "Chinese",
        "ja" => "Japanese",
        "ko" => "Korean",
        "fr" => "French",
        "de" => "German",
        "es" => "Spanish",
        "unknown" => "other",
        other => other,
    }
    .to_owned()
}

/// Advice as lines: `warning: ...` / `note: ...`, with the code under `-v`.
pub fn advice(advice: &[impl Serialize]) -> String {
    let mut out = String::new();
    for a in advice {
        let a = serde_json::to_value(a).unwrap_or_default();
        let level = if a["severity"] == "warning" {
            "warning"
        } else {
            "note"
        };
        let code = if verbose() {
            format!("[{}]", a["code"].as_str().unwrap_or_default())
        } else {
            String::new()
        };
        let _ = write!(
            out,
            "\n{level}{code}: {}",
            a["message"].as_str().unwrap_or_default()
        );
    }
    out
}

/// `  (id: <key>)` under `-v`, nothing otherwise.
pub fn id(label: &str, key: &Value) -> String {
    match key.as_str() {
        Some(key) if verbose() => format!("\n  {label}: {key}"),
        _ => String::new(),
    }
}
