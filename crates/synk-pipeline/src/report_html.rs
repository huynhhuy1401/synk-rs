//! `synk run report`: one self-contained HTML page about a training run,
//! for a person: the verdict in words, the numbers, and real examples of
//! searches that got better or worse. No scripts, no external files.
//!
//! The page contains snippets of the user's documents, so it is written
//! where they ask and never into an exported model.

use std::fmt::Write as _;

use serde_json::Value;
use synk_core::Result;

use crate::advice::{self, Advice};
use crate::artifact::{Artifacts, Manifest};
use crate::report;
use crate::train::{self, Verdict};

fn esc(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

fn language(code: &str) -> &str {
    match code {
        "vi" => "Vietnamese",
        "en" => "English",
        "unknown" => "other",
        other => other,
    }
}

fn verdict_words(v: &Verdict) -> (&'static str, String) {
    let class = match v.outcome.as_str() {
        "better" => "good",
        "worse" => "bad",
        _ => "neutral",
    };
    let headline = match v.outcome.as_str() {
        "better" => "The tuned model finds the right passage more often.",
        "worse" => "The tuned model is worse than the original.",
        _ => "No clear change: the difference is within what chance can explain.",
    };
    (class, headline.to_owned())
}

fn item_html(out: &mut String, item: &Value) {
    let ranks = &item["ranks"];
    let _ = write!(
        out,
        "<li><p class=\"q\">{}</p><p class=\"meta\">{} · rank {} → {}</p>",
        esc(item["query"].as_str().unwrap_or("")),
        esc(item["kind"].as_str().unwrap_or("")),
        ranks["base"],
        ranks["tuned"],
    );
    if let Some(text) = item["expected"]["text"].as_str() {
        let _ = write!(
            out,
            "<p class=\"chunk\"><b>Right passage</b> ({}): {}</p>",
            esc(item["expected"]["doc"].as_str().unwrap_or("")),
            esc(text)
        );
    }
    if let Some(text) = item["tuned_top"]["text"].as_str() {
        let _ = write!(
            out,
            "<p class=\"chunk other\"><b>Tuned model's first result</b> ({}): {}</p>",
            esc(item["tuned_top"]["doc"].as_str().unwrap_or("")),
            esc(text)
        );
    }
    out.push_str("</li>");
}

/// The page for `run`, or an error if it has no evaluation with per-query
/// ranks.
pub fn render(
    artifacts: &Artifacts,
    run: &Manifest,
    run_id: &str,
    advice_for: impl Fn(&Verdict, &[(String, Verdict)]) -> Vec<Advice>,
) -> Result<String> {
    let inspect = report::inspect(artifacts, run, 8)?;
    let eval = train::eval_for(artifacts, run)?;
    let verdict = eval.as_ref().map(train::eval_verdict);
    let general = eval
        .as_ref()
        .map(|e| advice::general_verdicts(&e.summary))
        .unwrap_or_default();
    let advice = verdict
        .as_ref()
        .map(|v| advice_for(v, &general))
        .unwrap_or_default();

    let mut body = String::new();
    let _ = write!(
        body,
        "<header><h1>Search model report</h1><p class=\"meta\">{} · base model {} · {} training queries</p></header>",
        esc(run_id),
        esc(run.summary["base_model"].as_str().unwrap_or("?")),
        run.summary["examples"],
    );
    if let Some(v) = &verdict {
        let (class, headline) = verdict_words(v);
        let _ = write!(
            body,
            "<section class=\"verdict {class}\"><h2>{headline}</h2>\
             <p class=\"big\">{:.2} → {:.2}</p>\
             <p>Search quality (nDCG@10, 0 to 1; 1 means the right passage always comes first), on {} test queries from {} documents the model never saw.",
            v.base, v.tuned, v.test_queries, v.test_documents
        );
        if let (Some(lo), Some(hi)) = (v.ci_low, v.ci_high) {
            let _ = write!(
                body,
                " The change is {:+.3}, and with 95% confidence between {lo:+.3} and {hi:+.3}.",
                v.delta
            );
        }
        body.push_str("</p>");
        if v.by_lang.len() > 1 {
            body.push_str(
                "<table><tr><th>Language</th><th>Before</th><th>After</th><th>Verdict</th></tr>",
            );
            for (lang, l) in &v.by_lang {
                let _ = write!(
                    body,
                    "<tr><td>{}</td><td>{:.2}</td><td>{:.2}</td><td>{}</td></tr>",
                    language(lang),
                    l.base,
                    l.tuned,
                    esc(&l.outcome.replace('_', " "))
                );
            }
            body.push_str("</table>");
        }
        body.push_str("</section>");
    }
    if !general.is_empty() {
        body.push_str("<section><h2>General check</h2><p>Public search tests, to see whether tuning hurt ordinary search.</p><table><tr><th>Set</th><th>Before</th><th>After</th><th>Verdict</th></tr>");
        for (set, v) in &general {
            let _ = write!(
                body,
                "<tr><td>{}</td><td>{:.2}</td><td>{:.2}</td><td>{}</td></tr>",
                language(set),
                v.base,
                v.tuned,
                esc(&v.outcome.replace('_', " "))
            );
        }
        body.push_str("</table></section>");
    }
    if !advice.is_empty() {
        body.push_str("<section><h2>What to try</h2><ul>");
        for a in &advice {
            let _ = write!(body, "<li>{}</li>", esc(&a.message));
        }
        body.push_str("</ul></section>");
    }
    let _ = write!(
        body,
        "<section><h2>Searches</h2><p>{} test queries: {} ranked their right passage higher after tuning, {} lower, {} the same.</p>",
        inspect["queries"], inspect["improved"], inspect["worse"], inspect["unchanged"]
    );
    for (title, key) in [("Biggest gains", "gains"), ("Biggest losses", "losses")] {
        let items = inspect[key].as_array().cloned().unwrap_or_default();
        if items.is_empty() {
            continue;
        }
        let _ = write!(body, "<h3>{title}</h3><ol>");
        for item in &items {
            item_html(&mut body, item);
        }
        body.push_str("</ol>");
    }
    body.push_str("</section><footer>Made by synk. This page contains passages of your documents; share it with care.</footer>");

    Ok(format!(
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">\
         <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\
         <title>synk report: {}</title><style>{CSS}</style></head><body>{body}</body></html>\n",
        esc(run_id)
    ))
}

const CSS: &str = "
:root{--bg:#fff;--fg:#1d1f23;--muted:#5d6470;--card:#f5f6f8;--good:#1a7f37;--bad:#c62828;--line:#dfe3e8}
@media (prefers-color-scheme:dark){:root{--bg:#15171a;--fg:#e6e8eb;--muted:#9aa3ad;--card:#1f2226;--good:#4cc36b;--bad:#ef6b6b;--line:#30353b}}
body{background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,-apple-system,'Segoe UI',sans-serif;max-width:860px;margin:0 auto;padding:24px 16px}
h1{margin:0 0 4px}h2{margin:0 0 8px}.meta{color:var(--muted);font-size:14px;margin:2px 0}
section{background:var(--card);border-radius:12px;padding:16px 20px;margin:16px 0}
.verdict.good h2{color:var(--good)}.verdict.bad h2{color:var(--bad)}
.big{font-size:32px;font-weight:600;margin:4px 0}
table{border-collapse:collapse;margin-top:8px}td,th{padding:4px 14px 4px 0;text-align:left;border-bottom:1px solid var(--line)}
ol{padding-left:20px}li{margin:12px 0}.q{font-weight:600;margin:0}
.chunk{margin:4px 0;font-size:14px;border-left:3px solid var(--good);padding-left:8px}
.chunk.other{border-color:var(--bad)}footer{color:var(--muted);font-size:13px;margin-top:24px}
";
