//! The pipeline sections of `synk.toml`. Every field has a default, so a
//! project with only `[[sources]]` works as is.

use std::fmt;
use std::path::PathBuf;

use serde::de::{self, Deserializer, Visitor};
use serde::{Deserialize, Serialize, Serializer};

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(default)]
pub struct PipelineConfig {
    pub model: ModelConfig,
    pub generator: GeneratorConfig,
    pub dataset: DatasetConfig,
    pub trainer: TrainerConfig,
    pub eval: EvalConfig,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct EvalConfig {
    /// Also test both models on public English and Vietnamese sets, so
    /// tuning can't quietly hurt ordinary search (about 10 MB, once).
    pub general_check: bool,
}

impl Default for EvalConfig {
    fn default() -> Self {
        Self {
            general_check: true,
        }
    }
}

impl PipelineConfig {
    /// Checks values serde can't: the allowed words of string settings.
    pub fn validate(&self) -> Result<(), String> {
        let t = &self.trainer;
        if !["auto", "full", "lora"].contains(&t.finetune.as_str()) {
            return Err(format!(
                "trainer.finetune is `{}`; use auto, full or lora",
                t.finetune
            ));
        }
        if !["auto", "fp32", "fp16", "bf16"].contains(&t.precision.as_str()) {
            return Err(format!(
                "trainer.precision is `{}`; use auto, fp32, fp16 or bf16",
                t.precision
            ));
        }
        Ok(())
    }
}

/// A setting that is either `"auto"` (synk picks) or a fixed value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AutoOr<T> {
    #[default]
    Auto,
    Value(T),
}

impl<T: Copy> AutoOr<T> {
    pub fn value(self) -> Option<T> {
        match self {
            Self::Auto => None,
            Self::Value(v) => Some(v),
        }
    }

    pub fn is_auto(self) -> bool {
        matches!(self, Self::Auto)
    }
}

impl<T: Serialize> Serialize for AutoOr<T> {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Auto => s.serialize_str("auto"),
            Self::Value(v) => v.serialize(s),
        }
    }
}

impl<'de, T: TryFrom<u64>> Deserialize<'de> for AutoOr<T> {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct V<T>(std::marker::PhantomData<T>);
        impl<T: TryFrom<u64>> Visitor<'_> for V<T> {
            type Value = AutoOr<T>;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("\"auto\" or a positive whole number")
            }
            fn visit_str<E: de::Error>(self, v: &str) -> Result<Self::Value, E> {
                if v == "auto" {
                    Ok(AutoOr::Auto)
                } else {
                    Err(E::invalid_value(de::Unexpected::Str(v), &self))
                }
            }
            fn visit_u64<E: de::Error>(self, v: u64) -> Result<Self::Value, E> {
                if v == 0 {
                    return Err(E::invalid_value(de::Unexpected::Unsigned(v), &self));
                }
                T::try_from(v)
                    .map(AutoOr::Value)
                    .map_err(|_| E::invalid_value(de::Unexpected::Unsigned(v), &self))
            }
            fn visit_i64<E: de::Error>(self, v: i64) -> Result<Self::Value, E> {
                u64::try_from(v)
                    .map_err(|_| E::invalid_value(de::Unexpected::Signed(v), &self))
                    .and_then(|v| self.visit_u64(v))
            }
        }
        d.deserialize_any(V(std::marker::PhantomData))
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct ModelConfig {
    /// A catalog ID, the Ollama tag of a catalog model, `hf:org/name[@rev]`,
    /// or a path to a sentence-transformers folder.
    pub base: String,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            base: crate::catalog::DEFAULT_MODEL.to_owned(),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct GeneratorConfig {
    /// An OpenAI-compatible API root. Ollama serves one at `/v1`.
    pub base_url: String,
    pub model: String,
    /// Same-language queries per chunk. `auto` aims for about 1,500
    /// training queries: 3 to 8 per chunk.
    pub queries_per_chunk: AutoOr<usize>,
    /// Extra languages to write one query in per chunk, for cross-language
    /// search, e.g. `["vi", "en"]`. Empty means same-language queries only.
    pub languages: Vec<String>,
    /// Must be true to send chunks to a server that is not on this machine.
    pub allow_remote: bool,
    /// Name of an environment variable holding an API key, if the server needs one.
    pub api_key_env: Option<String>,
}

impl Default for GeneratorConfig {
    fn default() -> Self {
        Self {
            base_url: "http://localhost:11434/v1".to_owned(),
            model: "qwen3:8b".to_owned(),
            queries_per_chunk: AutoOr::Auto,
            languages: Vec::new(),
            allow_remote: false,
            api_key_env: None,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct DatasetConfig {
    /// Largest chunk in tokens of the base model's tokenizer. `auto` fills
    /// the trainer's `max_seq_length`, so the model sees every chunk whole.
    pub chunk_tokens: AutoOr<usize>,
    /// Deprecated: chunk size in characters. When set, it wins over
    /// `chunk_tokens`.
    pub chunk_chars: Option<usize>,
    /// Share of documents held out for evaluation, in percent.
    pub test_percent: u8,
    /// Use at most this many chunks (a deterministic sample). Handy for a
    /// quick first run.
    pub max_chunks: Option<usize>,
}

impl Default for DatasetConfig {
    fn default() -> Self {
        Self {
            chunk_tokens: AutoOr::Auto,
            chunk_chars: None,
            test_percent: 20,
            max_chunks: None,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct TrainerConfig {
    /// Use this Python instead of the environment `synk setup` creates. It
    /// needs torch and sentence-transformers[train] installed.
    pub python: Option<PathBuf>,
    /// Path to `uv`, if it is not on PATH.
    pub uv: Option<PathBuf>,
    /// `cpu` or `cuda` (NVIDIA). Default: the NVIDIA GPU if the environment
    /// has one, else the CPU.
    pub device: Option<String>,
    /// `auto`: enough epochs for at least 300 training steps, at most 5.
    pub epochs: AutoOr<u32>,
    pub batch_size: u32,
    pub learning_rate: f64,
    pub max_seq_length: u32,
    pub seed: u64,
    /// Hard negatives per training example.
    pub negatives: u32,
    /// Drop training queries whose chunk the base model ranks below 20th
    /// (a round-trip filter for bad synthetic queries).
    pub query_filter: bool,
    /// A cross-encoder that checks mined hard negatives, removing ones that
    /// likely answer the query too. Off by default: it is an extra download.
    pub reranker: Option<String>,
    /// `auto` (LoRA for models of 300M parameters or more, else full),
    /// `full` or `lora`.
    pub finetune: String,
    /// `auto` (bf16 or fp16 on NVIDIA GPUs, fp32 on the CPU), `fp32`,
    /// `fp16` or `bf16`.
    pub precision: String,
    pub lora_r: u32,
    pub lora_alpha: u32,
    pub lora_learning_rate: f64,
}

impl Default for TrainerConfig {
    fn default() -> Self {
        Self {
            python: None,
            uv: None,
            device: None,
            epochs: AutoOr::Auto,
            batch_size: 16,
            learning_rate: 2e-5,
            max_seq_length: 256,
            seed: 42,
            negatives: 3,
            query_filter: true,
            reranker: None,
            finetune: "auto".to_owned(),
            precision: "auto".to_owned(),
            lora_r: 32,
            lora_alpha: 64,
            lora_learning_rate: 1e-4,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_or_accepts_auto_and_numbers_only() {
        let cfg: TrainerConfig = toml::from_str("epochs = \"auto\"").unwrap();
        assert!(cfg.epochs.is_auto());
        let cfg: TrainerConfig = toml::from_str("epochs = 2").unwrap();
        assert_eq!(cfg.epochs, AutoOr::Value(2));
        assert!(toml::from_str::<TrainerConfig>("epochs = \"many\"").is_err());
        assert!(toml::from_str::<TrainerConfig>("epochs = 0").is_err());
        assert!(toml::from_str::<TrainerConfig>("epochs = -1").is_err());
        assert_eq!(
            serde_json::to_value(AutoOr::<u32>::Auto).unwrap(),
            serde_json::json!("auto")
        );
    }
}
