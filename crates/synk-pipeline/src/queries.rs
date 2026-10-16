//! Real queries: searches a person actually made, imported from a file.
//!
//! Each line of the file is a JSON object with a `query` and what answers
//! it: a `doc` (a document ID, or a unique end of one such as `a.md`), a
//! `text` (a passage from the answering document), or both. Real queries
//! are only used to evaluate, never to train, so they measure the model on
//! the searches that matter.

use std::fmt::Write as _;

use serde_json::Value;
use synk_core::{Document, ErrorCode, Result, SynkError};

/// A validated row: query, document ID, and the passage if one was given.
pub type ImportRow = (String, String, Option<String>);

/// What a file held: the rows to import, and how many template lines were
/// left unfilled (a `doc` with an empty `query`).
#[derive(Debug)]
pub struct Parsed {
    pub rows: Vec<ImportRow>,
    pub skipped_empty: usize,
}

/// The first line of a template: instructions, skipped on import.
pub const TEMPLATE_HELP: &str = "For each document below, write in \"query\" how you would search for something you know is in it, without opening it (reading it first copies its words and makes the test too easy). Leave a line's query empty to skip it. Then: synk queries import <this file>";

/// At most this many problems are listed in one error.
const MAX_REPORTED: usize = 10;

/// Lowercase with whitespace collapsed, so a passage copied from a rendered
/// page still matches the stored text.
pub fn normalize(text: &str) -> String {
    text.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

/// Parses and checks a JSONL import against the live documents. Fails as a
/// whole if any line is bad, so a half-imported file never happens.
pub fn parse_import(text: &str, docs: &[Document]) -> Result<Parsed> {
    let normalized: Vec<String> = docs.iter().map(|d| normalize(&d.text)).collect();
    let mut rows = Vec::new();
    let mut problems = Vec::new();
    let mut skipped_empty = 0;
    for (n, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        // A template's help line, and its lines nobody filled in.
        if let Ok(value) = serde_json::from_str::<Value>(line) {
            if value.get("_help").is_some() {
                continue;
            }
            let unfilled = value["query"].as_str().is_some_and(|q| q.trim().is_empty());
            if unfilled && value.get("doc").is_some() {
                skipped_empty += 1;
                continue;
            }
        }
        match parse_row(line, docs, &normalized) {
            Ok(row) => rows.push(row),
            Err(why) => problems.push(format!("line {}: {why}", n + 1)),
        }
    }
    if !problems.is_empty() {
        let mut message = format!(
            "{} line(s) can't be imported; nothing was imported:",
            problems.len()
        );
        for p in problems.iter().take(MAX_REPORTED) {
            let _ = write!(message, "\n  {p}");
        }
        if problems.len() > MAX_REPORTED {
            let _ = write!(message, "\n  ...");
        }
        return Err(SynkError::new(ErrorCode::QueriesInvalid, message).with_next(
            "Fix those lines: each needs \"query\" and \"doc\" (a document ID) and/or \"text\" (a passage that answers it)",
        ));
    }
    if rows.is_empty() {
        let message = if skipped_empty > 0 {
            format!("none of the {skipped_empty} searches in the file is filled in")
        } else {
            "the file has no queries".to_owned()
        };
        return Err(SynkError::new(ErrorCode::QueriesInvalid, message));
    }
    Ok(Parsed {
        rows,
        skipped_empty,
    })
}

/// The lines of a template for `docs` (ID, title), help line first.
pub fn template(docs: &[(String, String)]) -> String {
    let mut out = serde_json::json!({ "_help": TEMPLATE_HELP }).to_string();
    out.push('\n');
    for (id, title) in docs {
        out.push_str(&serde_json::json!({ "doc": id, "title": title, "query": "" }).to_string());
        out.push('\n');
    }
    out
}

fn parse_row(line: &str, docs: &[Document], normalized: &[String]) -> Result<ImportRow, String> {
    let value: Value = serde_json::from_str(line).map_err(|e| format!("not JSON ({e})"))?;
    let field = |name: &str| -> Result<Option<String>, String> {
        match &value[name] {
            Value::Null => Ok(None),
            Value::String(s) if !s.trim().is_empty() => Ok(Some(s.trim().to_owned())),
            _ => Err(format!("\"{name}\" must be a non-empty string")),
        }
    };
    let nfc = synk_connectors::readers::nfc;
    let query = nfc(&field("query")?.ok_or("missing \"query\"")?);
    let doc = field("doc")?;
    let passage = field("text")?.map(|p| nfc(&p));
    let doc = match (doc, &passage) {
        (Some(doc), _) => find_doc(&doc, docs)?,
        (None, Some(passage)) => {
            let needle = normalize(passage);
            let found: Vec<usize> = (0..docs.len())
                .filter(|i| normalized[*i].contains(&needle))
                .collect();
            match found.as_slice() {
                [i] => *i,
                [] => return Err("no document contains \"text\"".to_owned()),
                _ => {
                    return Err(format!(
                        "\"text\" is in {} documents; add \"doc\" to say which",
                        found.len()
                    ));
                }
            }
        }
        (None, None) => return Err("needs \"doc\" or \"text\"".to_owned()),
    };
    if let Some(passage) = &passage
        && !normalized[doc].contains(&normalize(passage))
    {
        return Err(format!("{} doesn't contain \"text\"", docs[doc].id));
    }
    Ok((query, docs[doc].id.clone(), passage))
}

/// A document by its ID, or by a unique ending of its ID (`a.md`,
/// `notes/a.md`).
fn find_doc(name: &str, docs: &[Document]) -> Result<usize, String> {
    let name = name.replace('\\', "/");
    if let Some(i) = docs.iter().position(|d| d.id == name) {
        return Ok(i);
    }
    let suffix = format!("/{}", name.trim_start_matches("./"));
    let found: Vec<usize> = (0..docs.len())
        .filter(|i| docs[*i].id.ends_with(&suffix))
        .collect();
    match found.as_slice() {
        [i] => Ok(*i),
        [] => Err(format!("no document `{name}`")),
        _ => Err(format!(
            "`{name}` matches {} documents; use the full ID",
            found.len()
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use synk_core::FileState;

    fn doc(id: &str, text: &str) -> Document {
        Document {
            id: id.into(),
            source: "notes".into(),
            rel_path: id.into(),
            title: String::new(),
            text: text.into(),
            state: FileState {
                size: 0,
                modified: 0,
                hash: String::new(),
            },
        }
    }

    #[test]
    fn rows_resolve_by_id_suffix_or_passage() {
        let docs = [
            doc("notes/a.md", "Tăng keepalive_timeout\nlên 75s."),
            doc("notes/b.md", "Vacuum weekly."),
            doc("other/b.md", "Something else."),
        ];
        let text = r#"
{"query": "nginx timeout", "doc": "a.md"}
{"query": "postgres bloat", "text": "vacuum   WEEKLY"}
{"query": "x", "doc": "notes/b.md", "text": "Vacuum weekly."}
"#;
        let rows = parse_import(text, &docs).unwrap().rows;
        assert_eq!(rows[0], ("nginx timeout".into(), "notes/a.md".into(), None));
        assert_eq!(rows[1].1, "notes/b.md");
        assert_eq!(rows[2].2.as_deref(), Some("Vacuum weekly."));
    }

    #[test]
    fn a_template_imports_its_filled_lines_only() {
        let docs = [doc("notes/a.md", "x"), doc("notes/b.md", "y")];
        let docs_ids = [
            ("notes/a.md".to_owned(), "A".to_owned()),
            ("notes/b.md".to_owned(), "B".to_owned()),
        ];
        let blank = template(&docs_ids);
        assert!(blank.lines().next().unwrap().contains("without opening it"));
        assert_eq!(
            parse_import(&blank, &docs).err().unwrap().message,
            "none of the 2 searches in the file is filled in"
        );
        let filled = blank.replacen("\"query\":\"\"", "\"query\":\"nginx timeout\"", 1);
        let parsed = parse_import(&filled, &docs).unwrap();
        assert_eq!(parsed.rows.len(), 1);
        assert_eq!(parsed.rows[0].1, "notes/a.md");
        assert_eq!(parsed.skipped_empty, 1);
    }

    #[test]
    fn any_bad_line_rejects_the_file() {
        let docs = [doc("notes/b.md", "x"), doc("other/b.md", "y")];
        let text = "{\"query\": \"q\", \"doc\": \"b.md\"}\nnot json\n{\"doc\": \"notes/b.md\"}\n{\"query\": \"q\", \"text\": \"zzz\"}";
        let err = parse_import(text, &docs).unwrap_err();
        assert_eq!(err.code, ErrorCode::QueriesInvalid);
        for expected in [
            "line 1: `b.md` matches 2",
            "line 2: not JSON",
            "line 3: missing",
            "line 4: no document",
        ] {
            assert!(
                err.message.contains(expected),
                "{expected}: {}",
                err.message
            );
        }
    }
}
