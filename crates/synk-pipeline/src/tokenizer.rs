//! The base model's tokenizer, for measuring chunks in the model's own
//! tokens.
//!
//! Only `tokenizer.json` is needed. synk looks in a local model folder, then
//! the Hugging Face cache (where the trainer keeps its models), then its own
//! cache, and downloads the file from Hugging Face as a last resort. The
//! download sends only the model's name, never user data.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

use synk_core::{ErrorCode, Result, SynkError};
use tokenizers::Tokenizer;

use crate::catalog::ModelSpec;
use crate::sidecar::synk_home;

const FILE: &str = "tokenizer.json";
/// Qwen3's tokenizer.json is about 11 MB; ureq's default limit is 10 MB.
const MAX_BYTES: u64 = 100 * 1024 * 1024;

pub struct TokenCounter {
    tokenizer: Tokenizer,
    /// Tokens the trainer adds around every chunk: the document prompt and
    /// special tokens such as `<s>` and `</s>`.
    overhead: usize,
    /// Hash of `tokenizer.json`, so a different tokenizer changes dataset keys.
    id: String,
}

impl TokenCounter {
    pub fn load(model: &ModelSpec) -> Result<Self> {
        let path = locate(model)?;
        let bytes = fs::read(&path).map_err(|e| SynkError::io(path.display(), e))?;
        Self::from_bytes(&bytes, model.document_prompt.as_deref().unwrap_or(""))
            .map_err(|e| e.with_next(format!("Delete {} and try again", path.display())))
    }

    pub fn from_bytes(bytes: &[u8], document_prompt: &str) -> Result<Self> {
        let mut tokenizer = Tokenizer::from_bytes(bytes).map_err(|e| {
            SynkError::new(
                ErrorCode::ModelNotFound,
                format!("can't read the model's tokenizer: {e}"),
            )
        })?;
        // Counting must see every token, whatever the file configures.
        tokenizer.with_padding(None);
        tokenizer
            .with_truncation(None)
            .expect("disabling truncation can't fail");
        let mut counter = Self {
            tokenizer,
            overhead: 0,
            id: blake3::hash(bytes).to_hex()[..16].to_owned(),
        };
        counter.overhead = counter.encode_len(document_prompt, true);
        Ok(counter)
    }

    /// Tokens in `text` on its own, without prompt or special tokens.
    pub fn count(&self, text: &str) -> usize {
        self.encode_len(text, false)
    }

    pub fn overhead(&self) -> usize {
        self.overhead
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    fn encode_len(&self, text: &str, special: bool) -> usize {
        self.tokenizer
            .encode_fast(text, special)
            .map_or(0, |e| e.get_ids().len())
    }
}

fn locate(model: &ModelSpec) -> Result<PathBuf> {
    let local = Path::new(&model.name_or_path);
    if local.is_dir() {
        let path = local.join(FILE);
        return if path.is_file() {
            Ok(path)
        } else {
            Err(SynkError::new(
                ErrorCode::ModelNotFound,
                format!("{} has no {FILE}", local.display()),
            ))
        };
    }
    let repo = &model.name_or_path;
    let revision = model.revision.as_deref().unwrap_or("main");
    if let Some(path) = hf_cache(repo, revision) {
        return Ok(path);
    }
    let cached = synk_home()
        .join("tokenizers")
        .join(repo.replace('/', "--"))
        .join(revision)
        .join(FILE);
    if cached.is_file() {
        return Ok(cached);
    }
    download(repo, revision, &cached)?;
    Ok(cached)
}

/// The file in the Hugging Face cache, if the trainer already downloaded it.
fn hf_cache(repo: &str, revision: &str) -> Option<PathBuf> {
    let hub = std::env::var_os("HF_HUB_CACHE")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HF_HOME").map(|h| PathBuf::from(h).join("hub")))
        .or_else(|| dirs::home_dir().map(|h| h.join(".cache").join("huggingface").join("hub")))?;
    let model_dir = hub.join(format!("models--{}", repo.replace('/', "--")));
    // A branch name like `main` maps to a commit through refs/.
    let commit = fs::read_to_string(model_dir.join("refs").join(revision))
        .map(|c| c.trim().to_owned())
        .unwrap_or_else(|_| revision.to_owned());
    let path = model_dir.join("snapshots").join(commit).join(FILE);
    path.is_file().then_some(path)
}

fn download(repo: &str, revision: &str, dest: &Path) -> Result<()> {
    let fail = |why: String| {
        SynkError::new(
            ErrorCode::ModelNotFound,
            format!("could not download the tokenizer of {repo}: {why}"),
        )
        .with_next("Check your network connection and run the command again")
    };
    if std::env::var("HF_HUB_OFFLINE").is_ok_and(|v| v == "1") {
        return Err(fail("HF_HUB_OFFLINE is set".to_owned()));
    }
    let endpoint =
        std::env::var("HF_ENDPOINT").unwrap_or_else(|_| "https://huggingface.co".to_owned());
    let url = format!(
        "{}/{repo}/resolve/{revision}/{FILE}",
        endpoint.trim_end_matches('/')
    );
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_global(Some(Duration::from_secs(120)))
        .build()
        .into();
    let mut request = agent.get(&url);
    if let Ok(token) = std::env::var("HF_TOKEN") {
        request = request.header("Authorization", &format!("Bearer {token}"));
    }
    let bytes = request
        .call()
        .map_err(|e| fail(e.to_string()))?
        .body_mut()
        .with_config()
        .limit(MAX_BYTES)
        .read_to_vec()
        .map_err(|e| fail(e.to_string()))?;
    // Check it parses before caching it.
    Tokenizer::from_bytes(&bytes).map_err(|e| fail(e.to_string()))?;
    let dir = dest.parent().expect("cache path has a parent");
    fs::create_dir_all(dir).map_err(|e| SynkError::io(dir.display(), e))?;
    let tmp = dest.with_extension("json.tmp");
    fs::write(&tmp, &bytes).map_err(|e| SynkError::io(tmp.display(), e))?;
    fs::rename(&tmp, dest).map_err(|e| SynkError::io(dest.display(), e))
}

/// A tiny word-level tokenizer (one token per whitespace-separated word,
/// plus `[CLS]` and `[SEP]`), for tests.
pub fn word_level_json() -> String {
    serde_json::json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [],
        "normalizer": null,
        "pre_tokenizer": {"type": "WhitespaceSplit"},
        "post_processor": {
            "type": "TemplateProcessing",
            "single": [
                {"SpecialToken": {"id": "[CLS]", "type_id": 0}},
                {"Sequence": {"id": "A", "type_id": 0}},
                {"SpecialToken": {"id": "[SEP]", "type_id": 0}}
            ],
            "pair": [
                {"Sequence": {"id": "A", "type_id": 0}},
                {"Sequence": {"id": "B", "type_id": 1}}
            ],
            "special_tokens": {
                "[CLS]": {"id": "[CLS]", "ids": [1], "tokens": ["[CLS]"]},
                "[SEP]": {"id": "[SEP]", "ids": [2], "tokens": ["[SEP]"]}
            }
        },
        "decoder": null,
        "model": {
            "type": "WordLevel",
            "vocab": {"[UNK]": 0, "[CLS]": 1, "[SEP]": 2},
            "unk_token": "[UNK]"
        }
    })
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_words_and_prompt_overhead() {
        let counter = TokenCounter::from_bytes(word_level_json().as_bytes(), "passage: ").unwrap();
        assert_eq!(counter.count("one two three"), 3);
        // "passage:" plus [CLS] and [SEP].
        assert_eq!(counter.overhead(), 3);
        let other = TokenCounter::from_bytes(word_level_json().as_bytes(), "").unwrap();
        assert_eq!(other.overhead(), 2);
        assert_eq!(counter.id(), other.id());
    }

    #[test]
    fn local_folder_without_tokenizer_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let spec = ModelSpec {
            name_or_path: dir.path().display().to_string(),
            revision: None,
            query_prompt: None,
            document_prompt: None,
            catalog_id: None,
        };
        assert_eq!(
            TokenCounter::load(&spec).err().unwrap().code,
            ErrorCode::ModelNotFound
        );
    }
}
