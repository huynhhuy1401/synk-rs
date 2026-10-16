//! Readable IDs for datasets and training runs.
//!
//! A run's ID is its record's (see [`crate::ops`]): `run:<n>-<label>`, e.g.
//! `run:3-e5-small`, where `n` counts up per project and the label is the
//! base model or the run's `--name`. A dataset's is `dataset:<n>-<label>`,
//! numbered when it is built, with the query generator as label. The
//! artifact key stays the true identity; the rest is for reading.
//!
//! Artifacts made before numbering have the older form
//! `<prefix>:<date>-<label>-<hash>` until [`Ops::migrate`] numbers them.
//!
//! Lookups accept the full ID, `run:3`, a bare number, a run's name, the ID
//! without its prefix, any unique start of an older ID, a bare key prefix,
//! `latest`, and for runs `best`.

use serde_json::json;
use synk_core::{ErrorCode, Result, SynkError};

use crate::artifact::{Artifacts, Manifest};
use crate::dataset;
use crate::ops::{OpKind, Ops};
use crate::train::TRAIN_KIND;

const MIN_HASH: usize = 4;

/// The kinds that have readable IDs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Dataset,
    Run,
}

impl Kind {
    pub fn prefix(self) -> &'static str {
        match self {
            Self::Dataset => "dataset",
            Self::Run => "run",
        }
    }

    pub fn artifact_kind(self) -> &'static str {
        match self {
            Self::Dataset => dataset::KIND,
            Self::Run => TRAIN_KIND,
        }
    }

    fn not_found(self) -> ErrorCode {
        match self {
            Self::Dataset => ErrorCode::DatasetNotFound,
            Self::Run => ErrorCode::RunNotFound,
        }
    }

    fn noun(self) -> &'static str {
        match self {
            Self::Dataset => "dataset",
            Self::Run => "training run",
        }
    }
}

/// `2026-10-15` for seconds since the Unix epoch (UTC).
pub fn date(secs: u64) -> String {
    // Howard Hinnant's days-to-civil algorithm.
    let z = (secs / 86_400) as i64 + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!("{year:04}-{month:02}-{day:02}")
}

/// `2026-10-15-141203` (UTC), for job IDs.
pub fn datetime(secs: u64) -> String {
    let rest = secs % 86_400;
    format!(
        "{}-{:02}{:02}{:02}",
        date(secs),
        rest / 3600,
        rest % 3600 / 60,
        rest % 60
    )
}

/// Lowercase letters, digits and single dashes, at most 24 characters.
pub fn slug(text: &str) -> String {
    let mut out = String::new();
    for c in text.chars().flat_map(char::to_lowercase) {
        if c.is_ascii_alphanumeric() {
            out.push(c);
        } else if !out.ends_with('-') && !out.is_empty() {
            out.push('-');
        }
    }
    let out: String = out.trim_end_matches('-').chars().take(24).collect();
    let out = out.trim_end_matches('-').to_owned();
    if out.is_empty() { "x".to_owned() } else { out }
}

/// The readable middle part of an ID.
fn label(kind: Kind, m: &Manifest) -> String {
    let raw = match kind {
        // `qwen3:8b at http://localhost:11434/v1` -> `qwen3:8b`.
        Kind::Dataset => m.summary["generator"]
            .as_str()
            .and_then(|g| g.split(" at ").next())
            .unwrap_or("dataset"),
        Kind::Run => m.summary["base_model"].as_str().unwrap_or("model"),
    };
    model_label(raw)
}

/// A short label for a model name or path:
/// `intfloat/multilingual-e5-small` -> `e5-small`.
pub fn model_label(raw: &str) -> String {
    let name = raw.rsplit(['/', '\\']).next().unwrap_or(raw);
    slug(name.strip_prefix("multilingual-").unwrap_or(name))
}

fn render(kind: Kind, m: &Manifest, hash_len: usize) -> String {
    format!(
        "{}:{}-{}-{}",
        kind.prefix(),
        date(m.created_at),
        label(kind, m),
        &m.key[..hash_len.min(m.key.len())]
    )
}

/// Whether `readable` (an ID without its prefix) has the older,
/// date-based form.
fn dated(readable: &str) -> bool {
    readable.len() > 11 && readable.as_bytes()[4] == b'-' && readable[..4].parse::<u16>().is_ok()
}

/// A runnable export command for a run, writing next to the project.
pub fn export_command(run_id: &str) -> String {
    let readable = run_id.split_once(':').map_or(run_id, |(_, r)| r);
    let label = if dated(readable) {
        readable
            .splitn(4, '-')
            .nth(3)
            .unwrap_or("tuned")
            .rsplit_once('-')
            .map_or("tuned", |(label, _)| label)
    } else {
        readable.split_once('-').map_or("tuned", |(_, label)| label)
    };
    format!("synk export {run_id} --out ./{label}-model")
}

/// The project's operations, with older runs numbered.
fn ops(artifacts: &Artifacts) -> Result<Ops> {
    let ops = Ops::new(artifacts.data_dir());
    ops.migrate(artifacts)?;
    Ok(ops)
}

/// The ID of `m` among `all` (every artifact of the same kind).
pub fn id_among(kind: Kind, m: &Manifest, all: &[Manifest]) -> String {
    let mut len = MIN_HASH;
    while len < m.key.len()
        && all
            .iter()
            .any(|o| o.key != m.key && o.key.starts_with(&m.key[..len]))
    {
        len += 1;
    }
    render(kind, m, len)
}

/// The ID of `m`: its run's or its number, else the older form.
pub fn id(artifacts: &Artifacts, kind: Kind, m: &Manifest) -> Result<String> {
    let ops = ops(artifacts)?;
    match kind {
        Kind::Run => {
            if let Some(run) = ops.run_for_key(&m.key)? {
                return Ok(run.id);
            }
        }
        Kind::Dataset => {
            if let Some(n) = ops.dataset_number(&m.key, false)? {
                return Ok(format!("dataset:{n}-{}", label(kind, m)));
            }
        }
    }
    Ok(id_among(kind, m, &artifacts.list(kind.artifact_kind())?))
}

/// Numbers dataset `m` if it has no number yet, and returns its ID.
pub fn number_dataset(artifacts: &Artifacts, m: &Manifest) -> Result<String> {
    ops(artifacts)?.dataset_number(&m.key, true)?;
    id(artifacts, Kind::Dataset, m)
}

/// Finds an artifact by ID, prefix, key prefix, `latest` or (runs) `best`.
pub fn resolve(artifacts: &Artifacts, kind: Kind, input: &str) -> Result<Manifest> {
    let all = artifacts.list(kind.artifact_kind())?;
    let input = input.trim();
    let missing = || {
        SynkError::new(
            kind.not_found(),
            format!("no {} matches `{input}`", kind.noun()),
        )
        .with_next(format!("synk {} list", kind.prefix()))
    };
    match input {
        "latest" => {
            return artifacts
                .latest(kind.artifact_kind())?
                .or_else(|| all.first().cloned())
                .ok_or_else(missing);
        }
        "best" if kind == Kind::Run => {
            return best_run(artifacts, &all)?.ok_or_else(|| {
                SynkError::new(
                    ErrorCode::RunNotFound,
                    "no training run has a `better` verdict yet",
                )
                .with_next("synk run list")
            });
        }
        _ => {}
    }
    if let Some(m) = numbered(artifacts, kind, input)? {
        return Ok(m);
    }
    let bare = input
        .strip_prefix(&format!("{}:", kind.prefix()))
        .unwrap_or(input);
    let ids: Vec<(String, &Manifest)> = all.iter().map(|m| (id_among(kind, m, &all), m)).collect();
    let mut found: Vec<&(String, &Manifest)> = ids
        .iter()
        .filter(|(id, m)| {
            let readable = id.split_once(':').map_or(id.as_str(), |(_, r)| r);
            m.key.starts_with(bare) || readable.starts_with(bare)
        })
        .collect();
    // An exact ID wins over longer IDs it happens to be a prefix of.
    if let Some(exact) = found
        .iter()
        .find(|(id, m)| m.key == bare || id.split_once(':').is_some_and(|(_, r)| r == bare))
    {
        found = vec![*exact];
    }
    match found.as_slice() {
        [] => Err(missing()),
        [(_, m)] => Ok((*m).clone()),
        many => Err(SynkError::new(
            ErrorCode::IdAmbiguous,
            format!(
                "`{input}` matches {} {}s; use more of the ID",
                many.len(),
                kind.noun()
            ),
        )
        .with_details(json!({
            "candidates": many.iter().map(|(id, _)| id).collect::<Vec<_>>(),
        }))),
    }
}

/// An artifact found by run record (number, name, new-style ID) or by
/// dataset number.
fn numbered(artifacts: &Artifacts, kind: Kind, input: &str) -> Result<Option<Manifest>> {
    let ops = ops(artifacts)?;
    match kind {
        Kind::Run => {
            let Some(run) = ops.find(input)?.filter(|r| r.kind == OpKind::Run) else {
                return Ok(None);
            };
            let found = match &run.key {
                Some(key) => artifacts.manifest(kind.artifact_kind(), key)?,
                None => None,
            };
            match found {
                Some(m) => Ok(Some(m)),
                None => {
                    let mut err = SynkError::new(
                        ErrorCode::RunIncomplete,
                        format!("{} has no trained model ({})", run.id, run.state.as_str()),
                    );
                    if run.unfinished() {
                        err = err.with_next(run.kind.continue_command(&run.id));
                    }
                    Err(err)
                }
            }
        }
        Kind::Dataset => {
            let rest = input.strip_prefix("dataset:").unwrap_or(input);
            if dated(rest) {
                return Ok(None);
            }
            let Ok(n) = rest.split('-').next().unwrap_or(rest).parse::<u64>() else {
                return Ok(None);
            };
            match ops.dataset_key(n) {
                Some(key) => artifacts.manifest(kind.artifact_kind(), &key),
                None => Ok(None),
            }
        }
    }
}

/// The run with the largest gain among runs whose verdict is `better`.
fn best_run(artifacts: &Artifacts, all: &[Manifest]) -> Result<Option<Manifest>> {
    let mut best: Option<(f64, &Manifest)> = None;
    for m in all {
        let Some(eval) = crate::train::eval_for(artifacts, m)? else {
            continue;
        };
        let v = crate::train::eval_verdict(&eval);
        if v.outcome == "better" && best.is_none_or(|(d, _)| v.delta > d) {
            best = Some((v.delta, m));
        }
    }
    Ok(best.map(|(_, m)| m.clone()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn manifest(key: &str, model: &str, created_at: u64) -> Manifest {
        let mut m = Manifest::new(
            TRAIN_KIND,
            key,
            json!({}),
            vec![],
            json!({"base_model": model}),
        );
        m.created_at = created_at;
        m
    }

    #[test]
    fn dates_are_utc_calendar_days() {
        assert_eq!(date(0), "1970-01-01");
        assert_eq!(date(1_760_486_400), "2025-10-15");
        assert_eq!(date(951_782_400), "2000-02-29");
    }

    #[test]
    fn ids_read_well_and_grow_on_collision() {
        let a = manifest("3f2a9c0000000000", "multilingual-e5-small", 1_760_486_400);
        let b = manifest("3f2b110000000000", "BAAI/bge-m3", 1_760_486_400);
        let c = manifest("3f2a9d0000000000", "./models/my model", 1_760_486_400);
        let all = vec![a.clone(), b.clone(), c.clone()];
        assert_eq!(id_among(Kind::Run, &b, &all), "run:2025-10-15-bge-m3-3f2b");
        assert_eq!(
            id_among(Kind::Run, &a, &all),
            "run:2025-10-15-e5-small-3f2a9c"
        );
        assert_eq!(
            id_among(Kind::Run, &c, &all),
            "run:2025-10-15-my-model-3f2a9d"
        );
    }

    #[test]
    fn resolve_accepts_ids_prefixes_and_keys() {
        let dir = tempfile::tempdir().unwrap();
        let artifacts = Artifacts::new(dir.path().join("artifacts"));
        for m in [
            manifest("3f2a9c0000000000", "multilingual-e5-small", 1_760_486_400),
            manifest("aa00000000000000", "BAAI/bge-m3", 1_760_572_800),
        ] {
            artifacts.begin(TRAIN_KIND, &m.key, true).unwrap();
            artifacts.commit(&m).unwrap();
        }
        let key = |s: &str| resolve(&artifacts, Kind::Run, s).map(|m| m.key);
        assert_eq!(
            key("run:2025-10-15-e5-small-3f2a").unwrap(),
            "3f2a9c0000000000"
        );
        assert_eq!(key("2025-10-16").unwrap(), "aa00000000000000");
        assert_eq!(key("3f2a").unwrap(), "3f2a9c0000000000");
        assert_eq!(key("latest").unwrap(), "aa00000000000000");
        let err = key("2025-10").unwrap_err();
        assert_eq!(err.code, ErrorCode::IdAmbiguous);
        assert_eq!(err.details["candidates"].as_array().unwrap().len(), 2);
        assert_eq!(key("zz").unwrap_err().code, ErrorCode::RunNotFound);
    }
}
