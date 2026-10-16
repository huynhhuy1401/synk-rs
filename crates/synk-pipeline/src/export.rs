//! Exporting a tuned model: a sentence-transformers folder, ONNX, or GGUF
//! (optionally created in the local Ollama), each with a vector check.
//!
//! The check embeds a few of the dataset's chunks with the original
//! sentence-transformers model and with the exported one, and fails below a
//! cosine threshold, so a broken conversion is never reported as success.

use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde_json::{Value, json};
use sha2::Digest;
use synk_core::{Applicability, ErrorAction, ErrorCode, Result, SynkError};

use crate::advice::{Advice, Severity};
use crate::artifact::{Artifacts, Manifest};
use crate::dataset;
use crate::sidecar::{PROTOCOL_VERSION, TrainerEnv, synk_home};
use crate::train;

/// llama.cpp's converter, at the commit synk's GGUF fixes were made for.
/// Its files ship inside synk (`vendor/llama.cpp`, ADR 0004).
pub const LLAMA_CPP_COMMIT: &str = "711780ffc75bbc5645173ed8647a7b3f328c052a";

mod converter {
    include!(concat!(env!("OUT_DIR"), "/converter_files.rs"));
}

/// Writes the shipped converter under SYNK_HOME once (a folder per content
/// hash) and returns it. Nothing is downloaded.
pub fn converter_dir() -> Result<PathBuf> {
    converter_dir_in(&synk_home())
}

fn converter_dir_in(home: &Path) -> Result<PathBuf> {
    let mut hasher = blake3::Hasher::new();
    for (path, bytes) in converter::FILES {
        hasher.update(path.as_bytes());
        hasher.update(bytes);
    }
    let hash = hasher.finalize().to_hex();
    let dir = home.join("converter").join(format!(
        "llama.cpp-{}-{}",
        &LLAMA_CPP_COMMIT[..12],
        &hash[..8]
    ));
    let done = dir.join(".complete");
    if done.is_file() {
        return Ok(dir);
    }
    let tmp = dir.with_extension("tmp");
    let _ = fs::remove_dir_all(&tmp);
    for (path, bytes) in converter::FILES {
        let dest = tmp.join(path);
        if let Some(parent) = dest.parent() {
            fs::create_dir_all(parent).map_err(|e| SynkError::io(parent.display(), e))?;
        }
        fs::write(&dest, bytes).map_err(|e| SynkError::io(dest.display(), e))?;
    }
    fs::write(tmp.join(".complete"), LLAMA_CPP_COMMIT)
        .map_err(|e| SynkError::io(tmp.display(), e))?;
    let _ = fs::remove_dir_all(&dir);
    fs::rename(&tmp, &dir).map_err(|e| SynkError::io(dir.display(), e))?;
    Ok(dir)
}
/// Texts embedded for the vector check.
const CHECK_TEXTS: usize = 16;
/// Lowest acceptable cosine between the original and the exported vectors.
/// **[unverified for q8_0]**: measured 0.99999 for f16 on e5-small.
const MIN_COSINE_EXACT: f64 = 0.999;
const MIN_COSINE_Q8: f64 = 0.995;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    /// A sentence-transformers folder.
    St,
    Onnx,
    Gguf,
}

impl Format {
    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "st" | "sentence-transformers" => Ok(Self::St),
            "onnx" => Ok(Self::Onnx),
            "gguf" => Ok(Self::Gguf),
            other => Err(SynkError::new(
                ErrorCode::InvalidArgs,
                format!("unknown format `{other}`; use st, onnx or gguf"),
            )),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::St => "st",
            Self::Onnx => "onnx",
            Self::Gguf => "gguf",
        }
    }
}

pub struct Request {
    pub format: Format,
    /// GGUF: `q8_0` (default) or `f16`.
    pub quant: Option<String>,
    /// GGUF: create this model in the local Ollama.
    pub ollama: Option<String>,
    pub force: bool,
    /// Ollama's API root, e.g. `http://localhost:11434`.
    pub ollama_url: String,
}

pub struct Exported {
    pub path: PathBuf,
    pub check: Value,
    pub ollama: Option<Value>,
    pub usage: Value,
    pub advice: Vec<Advice>,
}

/// Empties `out` when allowed, or fails if it holds anything.
fn prepare_out(out: &Path, force: bool) -> Result<()> {
    let non_empty = out.exists()
        && fs::read_dir(out)
            .map_err(|e| SynkError::io(out.display(), e))?
            .next()
            .is_some();
    if non_empty {
        if !force {
            return Err(SynkError::new(
                ErrorCode::ExportExists,
                format!("{} already exists and is not empty", out.display()),
            )
            .with_hint("pick another --out folder, or add --force to replace it"));
        }
        fs::remove_dir_all(out).map_err(|e| SynkError::io(out.display(), e))?;
    }
    fs::create_dir_all(out).map_err(|e| SynkError::io(out.display(), e))
}

/// A few chunks of the run's dataset to embed for the check.
fn check_texts(artifacts: &Artifacts, run: &Manifest) -> Vec<String> {
    let Some(key) = run.inputs.iter().find_map(|i| i.strip_prefix("dataset:")) else {
        return vec!["hello".to_owned()];
    };
    let path = artifacts.dir(dataset::KIND, key).join("corpus.jsonl");
    let mut texts: Vec<String> = fs::read_to_string(path)
        .unwrap_or_default()
        .lines()
        .filter_map(|l| serde_json::from_str::<Value>(l).ok())
        .filter_map(|v| v["text"].as_str().map(str::to_owned))
        .take(CHECK_TEXTS)
        .collect();
    if texts.is_empty() {
        texts.push("hello".to_owned());
    }
    texts
}

fn cosine(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

fn check_failed(min: f64, threshold: f64, format: Format) -> SynkError {
    let err = SynkError::new(
        ErrorCode::ExportCheckFailed,
        format!(
            "the exported model's vectors differ from the original (cosine {min:.4}, needs {threshold})"
        ),
    );
    if format == Format::Gguf {
        err.with_action(ErrorAction::run(
            "synk export --format gguf --quant f16",
            Applicability::Review,
            "export without 8-bit quantization (bigger, exact); add your run and --out",
        ))
    } else {
        err
    }
}

/// Runs a trainer export task in a scratch folder.
fn trainer_export(
    env: &TrainerEnv,
    request: &Value,
    key: &str,
    on_event: &mut dyn FnMut(&Value),
) -> Result<Value> {
    let work = std::env::temp_dir().join(format!("synk-export-{key}"));
    fs::create_dir_all(&work).map_err(|e| SynkError::io(work.display(), e))?;
    let result = env.run(request, &work, on_event);
    let _ = fs::remove_dir_all(&work);
    result
}

#[allow(clippy::too_many_arguments)]
pub fn export(
    artifacts: &Artifacts,
    run: &Manifest,
    out: &Path,
    req: &Request,
    env: impl FnOnce() -> Result<TrainerEnv>,
    on_event: &mut dyn FnMut(&Value),
) -> Result<Exported> {
    let model_dir = artifacts.dir(train::TRAIN_KIND, &run.key).join("model");
    if req.format != Format::Gguf && (req.ollama.is_some() || req.quant.is_some()) {
        return Err(SynkError::new(
            ErrorCode::InvalidArgs,
            "--ollama and --quant only apply to --format gguf",
        ));
    }
    prepare_out(out, req.force)?;
    let eval = train::eval_for(artifacts, run)?;
    let readme = train::model_card(run, eval.as_ref());
    let prompts = &run.summary["prompts"];
    let has_prompts = ["query", "document"]
        .iter()
        .any(|k| prompts[k].as_str().is_some_and(|p| !p.is_empty()));
    let mut advice = Vec::new();

    let (path, check, ollama) = match req.format {
        Format::St => {
            fs::remove_dir(out).ok();
            train::export_folder(&model_dir, out, &readme)?;
            (
                out.to_path_buf(),
                json!({"status": "passed", "method": "exact copy"}),
                None,
            )
        }
        Format::Onnx => {
            let env = env()?;
            let result = trainer_export(
                &env,
                &json!({
                    "v": PROTOCOL_VERSION, "task": "export", "format": "onnx",
                    "model_dir": model_dir, "out": out, "check_texts": check_texts(artifacts, run),
                }),
                &run.key,
                on_event,
            )?;
            let min = result["check"]["min_cosine"].as_f64().unwrap_or(0.0);
            if min < MIN_COSINE_EXACT {
                return Err(check_failed(min, MIN_COSINE_EXACT, Format::Onnx));
            }
            write_readme(out, &readme)?;
            (
                out.to_path_buf(),
                json!({"status": "passed", "min_cosine": min, "texts": result["check"]["texts"]}),
                None,
            )
        }
        Format::Gguf => {
            let env = env()?;
            let quant = req.quant.clone().unwrap_or_else(|| "q8_0".to_owned());
            if quant != "q8_0" && quant != "f16" {
                return Err(SynkError::new(
                    ErrorCode::InvalidArgs,
                    format!("--quant `{quant}`: use q8_0 or f16"),
                ));
            }
            let result = trainer_export(
                &env,
                &json!({
                    "v": PROTOCOL_VERSION, "task": "export", "format": "gguf", "quant": quant,
                    "model_dir": model_dir, "out": out, "check_texts": check_texts(artifacts, run),
                    "converter_dir": converter_dir()?,
                }),
                &run.key,
                on_event,
            )?;
            let gguf = PathBuf::from(result["path"].as_str().unwrap_or_default());
            let modelfile = out.join("Modelfile");
            fs::write(&modelfile, "FROM ./model.gguf\n")
                .map_err(|e| SynkError::io(modelfile.display(), e))?;
            write_readme(out, &readme)?;
            if has_prompts {
                advice.push(Advice {
                    code: "PREFIX_DROPPED_BY_TARGET",
                    severity: Severity::Warning,
                    message: format!(
                        "This model expects \"{}\" before searches and \"{}\" before documents. Ollama doesn't add them, so apps that embed through Ollama should add them themselves, or search quality will be lower than synk measured.",
                        prompts["query"].as_str().unwrap_or(""),
                        prompts["document"].as_str().unwrap_or("")
                    ),
                    evidence: json!({"prompts": prompts}),
                    action: None,
                    applicability: Applicability::Manual,
                });
            }
            let threshold = if quant == "f16" {
                MIN_COSINE_EXACT
            } else {
                MIN_COSINE_Q8
            };
            let reference: Vec<Vec<f64>> =
                serde_json::from_value(result["reference"].clone()).unwrap_or_default();
            let texts: Vec<String> =
                serde_json::from_value(result["texts"].clone()).unwrap_or_default();
            let ollama_up = ollama_reachable(&req.ollama_url);
            let (check, ollama) = match (&req.ollama, ollama_up) {
                (Some(name), false) => {
                    return Err(SynkError::new(
                        ErrorCode::ExportUnsupported,
                        format!(
                            "can't reach Ollama at {} to create `{name}`; the GGUF file is in {}",
                            req.ollama_url,
                            out.display()
                        ),
                    )
                    .with_action(ErrorAction::run(
                        "ollama serve",
                        Applicability::Review,
                        "start Ollama, then export again",
                    )));
                }
                (name, true) => {
                    let temporary = name.is_none();
                    let name = name.clone().unwrap_or_else(|| {
                        format!("synk-check-{}", &run.key[..8.min(run.key.len())])
                    });
                    ollama_create(&req.ollama_url, &name, &gguf)?;
                    let got = ollama_embed(&req.ollama_url, &name, &texts);
                    if temporary {
                        ollama_delete(&req.ollama_url, &name);
                    }
                    let got = got?;
                    let min = reference
                        .iter()
                        .zip(&got)
                        .map(|(a, b)| cosine(a, b))
                        .fold(f64::INFINITY, f64::min);
                    if !min.is_finite() || min < threshold {
                        if !temporary {
                            ollama_delete(&req.ollama_url, &name);
                        }
                        return Err(check_failed(
                            if min.is_finite() { min } else { 0.0 },
                            threshold,
                            Format::Gguf,
                        ));
                    }
                    (
                        json!({"status": "passed", "method": "Ollama /api/embed", "min_cosine": (min * 1e6).round() / 1e6, "texts": texts.len(), "threshold": threshold}),
                        (!temporary).then(|| json!({"name": name, "created": true})),
                    )
                }
                (None, false) => (
                    json!({"status": "not_checked", "reason": "Ollama isn't running, so the GGUF file couldn't be tried"}),
                    None,
                ),
            };
            (gguf, check, ollama)
        }
    };
    let usage = usage(req.format, &path, ollama.as_ref(), prompts);
    Ok(Exported {
        path,
        check,
        ollama,
        usage,
        advice,
    })
}

fn write_readme(out: &Path, readme: &str) -> Result<()> {
    let path = out.join("README.md");
    fs::write(&path, readme).map_err(|e| SynkError::io(path.display(), e))
}

fn usage(format: Format, path: &Path, ollama: Option<&Value>, prompts: &Value) -> Value {
    let q = prompts["query"].as_str().unwrap_or("");
    let d = prompts["document"].as_str().unwrap_or("");
    let p = path.display();
    match format {
        Format::St => json!({
            "python": format!("SentenceTransformer(r\"{p}\").encode(texts, prompt_name=\"query\")  # \"document\" for documents"),
        }),
        Format::Onnx => json!({
            "onnxruntime": format!(
                "InferenceSession(r\"{}\"): inputs input_ids and attention_mask (tokenize with the tokenizer in the folder), output sentence_embedding",
                path.join("model.onnx").display()
            ),
            "query_prefix": q,
            "document_prefix": d,
        }),
        Format::Gguf => json!({
            "ollama": ollama.map_or_else(
                || format!("ollama create <name> -f {}", path.with_file_name("Modelfile").display()),
                |o| format!("ollama run {}  (embeddings: POST /api/embed)", o["name"].as_str().unwrap_or("")),
            ),
            "query_prefix": q,
            "document_prefix": d,
        }),
    }
}

fn agent(timeout: Duration) -> ureq::Agent {
    ureq::Agent::config_builder()
        .timeout_global(Some(timeout))
        .build()
        .into()
}

fn ollama_reachable(url: &str) -> bool {
    agent(Duration::from_secs(3))
        .get(&format!("{url}/api/tags"))
        .call()
        .is_ok()
}

fn ollama_error(what: &str, e: impl std::fmt::Display) -> SynkError {
    SynkError::new(ErrorCode::ExportUnsupported, format!("Ollama {what}: {e}"))
}

/// Uploads the GGUF file as a blob and creates `name` from it.
fn ollama_create(url: &str, name: &str, gguf: &Path) -> Result<()> {
    let mut file = fs::File::open(gguf).map_err(|e| SynkError::io(gguf.display(), e))?;
    let mut hasher = sha2::Sha256::new();
    let mut buf = vec![0; 1 << 20];
    loop {
        let n = file
            .read(&mut buf)
            .map_err(|e| SynkError::io(gguf.display(), e))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    let digest = format!("sha256:{:x}", hasher.finalize());
    let file = fs::File::open(gguf).map_err(|e| SynkError::io(gguf.display(), e))?;
    let long = agent(Duration::from_secs(3600));
    long.post(&format!("{url}/api/blobs/{digest}"))
        .send(ureq::SendBody::from_reader(&mut std::io::BufReader::new(
            file,
        )))
        .map_err(|e| ollama_error("rejected the model file", e))?;
    long.post(&format!("{url}/api/create"))
        .send_json(json!({"model": name, "files": {"model.gguf": digest}, "stream": false}))
        .map_err(|e| ollama_error(&format!("couldn't create `{name}`"), e))?;
    Ok(())
}

fn ollama_embed(url: &str, name: &str, texts: &[String]) -> Result<Vec<Vec<f64>>> {
    let body: Value = agent(Duration::from_secs(600))
        .post(&format!("{url}/api/embed"))
        .send_json(json!({"model": name, "input": texts}))
        .map_err(|e| ollama_error("couldn't embed with the new model", e))?
        .body_mut()
        .read_json()
        .map_err(|e| ollama_error("answered oddly", e))?;
    serde_json::from_value(body["embeddings"].clone())
        .map_err(|e| ollama_error("returned no embeddings", e))
}

fn ollama_delete(url: &str, name: &str) {
    let _ = agent(Duration::from_secs(30))
        .delete(&format!("{url}/api/delete"))
        .force_send_body()
        .send_json(json!({"model": name}));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_gguf_converter_ships_inside_synk() {
        let home = tempfile::tempdir().unwrap();
        let dir = converter_dir_in(home.path()).unwrap();
        for file in [
            "convert_hf_to_gguf.py",
            "gguf-py/gguf/__init__.py",
            "LICENSE",
        ] {
            assert!(dir.join(file).is_file(), "{file} missing");
        }
        let version = fs::read_to_string(dir.join("VENDORED.toml")).unwrap();
        assert!(
            version.contains(LLAMA_CPP_COMMIT),
            "VENDORED.toml names another commit"
        );
        // Written once, then reused.
        fs::write(dir.join("marker"), "x").unwrap();
        assert_eq!(converter_dir_in(home.path()).unwrap(), dir);
        assert!(dir.join("marker").is_file());
    }
}
