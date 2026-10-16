//! The model catalog, and resolving a configured base model to something
//! the trainer can load.

use std::path::Path;

use serde::{Deserialize, Serialize};
use synk_core::{ErrorCode, Result, SynkError};

pub const DEFAULT_MODEL: &str = "multilingual-e5-small";
const CATALOG: &str = include_str!("../catalog.toml");
const CATALOG_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatalogModel {
    pub id: String,
    pub hf: String,
    pub revision: String,
    pub query_prompt: String,
    pub document_prompt: String,
    pub dims: u32,
    pub parameters: String,
    pub languages: String,
    pub notes: String,
    #[serde(default)]
    pub ollama: Vec<String>,
    /// Embedding sizes the model was trained to work at (Matryoshka); synk
    /// keeps them working while tuning.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub mrl_dims: Vec<u32>,
    /// Needs an accepted license and `HF_TOKEN` to download.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub gated: bool,
}

/// `118M` / `0.6B` -> millions of parameters.
pub fn parameters_millions(text: &str) -> Option<f64> {
    let t = text.trim();
    let (num, mult) = if let Some(n) = t.strip_suffix('B') {
        (n, 1000.0)
    } else {
        let n = t.strip_suffix('M')?;
        (n, 1.0)
    };
    num.trim().parse::<f64>().ok().map(|n| n * mult)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatalogReranker {
    pub id: String,
    pub hf: String,
    pub revision: String,
    pub parameters: String,
    pub languages: String,
    pub notes: String,
}

#[derive(Deserialize)]
struct Catalog {
    catalog_version: u32,
    models: Vec<CatalogModel>,
    #[serde(default)]
    rerankers: Vec<CatalogReranker>,
}

fn catalog() -> Catalog {
    let catalog: Catalog = toml::from_str(CATALOG).expect("embedded catalog is valid");
    assert_eq!(catalog.catalog_version, CATALOG_VERSION);
    catalog
}

pub fn models() -> Vec<CatalogModel> {
    catalog().models
}

pub fn rerankers() -> Vec<CatalogReranker> {
    catalog().rerankers
}

/// A reranker the trainer can load. It is sent to the sidecar as is.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RerankerSpec {
    pub name_or_path: String,
    pub revision: Option<String>,
    pub catalog_id: Option<String>,
}

/// Resolves `[trainer] reranker`: a catalog ID, `hf:org/name[@revision]`,
/// or a folder (resolved against `root`).
pub fn resolve_reranker(name: &str, root: &Path) -> Result<RerankerSpec> {
    if let Some(r) = rerankers().into_iter().find(|r| r.id == name) {
        return Ok(RerankerSpec {
            name_or_path: r.hf,
            revision: Some(r.revision),
            catalog_id: Some(r.id),
        });
    }
    if let Some(hf) = name.strip_prefix("hf:") {
        let (name, revision) = match hf.split_once('@') {
            Some((name, rev)) => (name, Some(rev.to_owned())),
            None => (hf, None),
        };
        return Ok(RerankerSpec {
            name_or_path: name.to_owned(),
            revision,
            catalog_id: None,
        });
    }
    let path = root.join(name);
    if path.is_dir() {
        let path = dunce::canonicalize(&path).unwrap_or(path);
        return Ok(RerankerSpec {
            name_or_path: path.display().to_string(),
            revision: None,
            catalog_id: None,
        });
    }
    Err(SynkError::new(
        ErrorCode::ModelNotFound,
        format!("reranker `{name}` is not in the catalog, `hf:<org>/<name>` or an existing folder"),
    )
    .with_next("synk model list"))
}

/// A base model the trainer can load. It is sent to the sidecar as is.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelSpec {
    /// A Hugging Face ID or a local folder.
    pub name_or_path: String,
    pub revision: Option<String>,
    /// `None` means: use the prompts saved with the model, if any.
    pub query_prompt: Option<String>,
    pub document_prompt: Option<String>,
    /// Set for catalog models; `None` marks a custom model.
    pub catalog_id: Option<String>,
}

impl ModelSpec {
    fn from_catalog(m: &CatalogModel) -> Self {
        Self {
            name_or_path: m.hf.clone(),
            revision: Some(m.revision.clone()),
            query_prompt: Some(m.query_prompt.clone()),
            document_prompt: Some(m.document_prompt.clone()),
            catalog_id: Some(m.id.clone()),
        }
    }

    fn custom(name_or_path: String, revision: Option<String>) -> Self {
        Self {
            name_or_path,
            revision,
            query_prompt: None,
            document_prompt: None,
            catalog_id: None,
        }
    }

    pub fn label(&self) -> String {
        self.catalog_id
            .clone()
            .unwrap_or_else(|| self.name_or_path.clone())
    }
}

/// Resolves `base`, which is one of: a catalog ID, the Ollama tag of a
/// catalog model, `hf:org/name[@revision]`, an existing folder (resolved
/// against `root`), or a bare `org/name` Hugging Face ID.
pub fn resolve(base: &str, root: &Path) -> Result<ModelSpec> {
    let catalog = models();
    if let Some(m) = catalog
        .iter()
        .find(|m| m.id == base || m.ollama.iter().any(|tag| tag == base))
    {
        return Ok(ModelSpec::from_catalog(m));
    }
    if let Some(hf) = base.strip_prefix("hf:") {
        let (name, revision) = match hf.split_once('@') {
            Some((name, rev)) => (name, Some(rev.to_owned())),
            None => (hf, None),
        };
        return Ok(ModelSpec::custom(name.to_owned(), revision));
    }
    let path = root.join(base);
    if path.is_dir() {
        let path = dunce::canonicalize(&path).unwrap_or(path);
        return Ok(ModelSpec::custom(path.display().to_string(), None));
    }
    // An Ollama tag of a model outside the catalog. Ollama stores quantized
    // GGUF weights, which can't be trained, and synk can't know which
    // Hugging Face model they came from.
    if base.contains(':') && !base.contains('/') {
        return Err(SynkError::new(
            ErrorCode::ModelNotTrainable,
            format!("`{base}` looks like an Ollama tag of a model that is not in the catalog; Ollama models can't be trained directly"),
        )
        .with_next("synk model list")
        .with_hint("or set model.base to the Hugging Face model it was built from: synk config set model.base hf:<org>/<name>"));
    }
    if base.split('/').count() == 2 && !base.starts_with('.') {
        return Ok(ModelSpec::custom(base.to_owned(), None));
    }
    Err(SynkError::new(
        ErrorCode::ModelNotFound,
        format!("`{base}` is not a catalog model, a Hugging Face ID or an existing folder"),
    )
    .with_next("synk model list"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_parses_and_has_the_default() {
        let models = models();
        assert!(models.iter().any(|m| m.id == DEFAULT_MODEL));
        assert!(models.iter().all(|m| m.revision.len() == 40));
        assert!(rerankers().iter().all(|r| r.revision.len() == 40));
        let root = Path::new(".");
        let bge = resolve_reranker("bge-reranker-v2-m3", root).unwrap();
        assert_eq!(bge.name_or_path, "BAAI/bge-reranker-v2-m3");
        assert_eq!(
            resolve_reranker("hf:org/x@r", root)
                .unwrap()
                .revision
                .as_deref(),
            Some("r")
        );
        assert_eq!(
            resolve_reranker("nope", root).unwrap_err().code,
            ErrorCode::ModelNotFound
        );
    }

    #[test]
    fn resolves_every_kind_of_base() {
        let root = Path::new(".");
        let e5 = resolve("multilingual-e5-small", root).unwrap();
        assert_eq!(e5.catalog_id.as_deref(), Some("multilingual-e5-small"));
        assert_eq!(e5.query_prompt.as_deref(), Some("query: "));

        let by_tag = resolve("qwen3-embedding:0.6b", root).unwrap();
        assert_eq!(by_tag.name_or_path, "Qwen/Qwen3-Embedding-0.6B");

        let hf = resolve("hf:BAAI/bge-m3@abc", root).unwrap();
        assert_eq!(
            (hf.name_or_path.as_str(), hf.revision.as_deref()),
            ("BAAI/bge-m3", Some("abc"))
        );
        assert!(hf.catalog_id.is_none());

        assert_eq!(
            resolve("BAAI/bge-m3", root).unwrap().name_or_path,
            "BAAI/bge-m3"
        );
        assert_eq!(
            resolve("nomic-embed-text:latest", root).unwrap_err().code,
            ErrorCode::ModelNotTrainable
        );
        assert_eq!(
            resolve("nope", root).unwrap_err().code,
            ErrorCode::ModelNotFound
        );
    }
}
