//! Finding, reading and editing `synk.toml`.
//!
//! Edits go through `toml_edit` so hand-written comments and formatting
//! survive `synk add` / `synk remove`.

use std::collections::BTreeMap;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

use serde::Deserialize;
use synk_core::{ErrorCode, Result, SourceSpec, SynkError};
use synk_pipeline::PipelineConfig;
use synk_pipeline::artifact::Artifacts;
use synk_pipeline::graph::Workspace;
use synk_store::Store;
use toml_edit::{Array, ArrayOfTables, DocumentMut, Item, Table, value};

pub const CONFIG_FILE: &str = "synk.toml";
const DATA_DIR: &str = ".synk";
const STORE_FILE: &str = "synk.db";
const ARTIFACTS_DIR: &str = "artifacts";

const TEMPLATE: &str = r#"# synk project. Edit by hand, or with `synk source ...` and `synk config set`.
# Then: synk train --dry-run (what would run, and what it needs), synk train.
#
# Optional settings, with their defaults (all: synk config show):
#
# [model]
# base = "multilingual-e5-small"   # see `synk model list`, or "hf:org/name", or a folder
#
# [generator]                      # writes queries; any OpenAI-compatible server
# base_url = "http://localhost:11434/v1"
# model = "qwen3:8b"
# queries_per_chunk = "auto"       # ~1,500 training queries: 3 to 8 per chunk
# languages = []                   # e.g. ["vi", "en"] also writes cross-language queries
#
# [dataset]
# chunk_tokens = "auto"            # fill what the model reads (trainer max_seq_length)
# test_percent = "auto"          # 20%, or more so the test has 10+ documents
#
# [trainer]
# epochs = "auto"                  # enough for ~300 steps, at most 5
# batch_size = 16
# negatives = 3                    # hard negatives per query
# finetune = "auto"                # full below 300M parameters, LoRA from there
# reranker = "mmarco-minilm-l12"   # double-check hard negatives (extra download)
#
# [eval]
# general_check = true             # also test on public English and Vietnamese sets
"#;

#[derive(Deserialize)]
struct ConfigFile {
    #[serde(default)]
    sources: Vec<SourceSpec>,
    #[serde(flatten)]
    pipeline: PipelineConfig,
}

pub struct Project {
    /// Directory containing `synk.toml`. Relative source paths resolve against it.
    pub root: PathBuf,
    pub config_path: PathBuf,
    doc: DocumentMut,
    /// Sources as written in the file (paths may be relative).
    sources: Vec<SourceSpec>,
    pipeline: PipelineConfig,
}

impl Project {
    pub fn init(dir: &Path) -> Result<Self> {
        let dir = existing_path(dir)?;
        let config_path = dir.join(CONFIG_FILE);
        if config_path.exists() {
            return Err(SynkError::new(
                ErrorCode::ConfigExists,
                format!("{} already exists", config_path.display()),
            )
            .with_next("synk status"));
        }
        fs::write(&config_path, TEMPLATE).map_err(|e| SynkError::io(config_path.display(), e))?;
        Self::load(config_path)
    }

    /// Loads `explicit`, or the nearest `synk.toml` in the current directory or its parents.
    pub fn find(explicit: Option<&Path>) -> Result<Self> {
        if let Some(path) = explicit {
            if !path.is_file() {
                return Err(SynkError::new(
                    ErrorCode::ConfigNotFound,
                    format!("config file not found: {}", path.display()),
                ));
            }
            return existing_path(path).and_then(Self::load);
        }
        let cwd = std::env::current_dir().map_err(|e| SynkError::io("current directory", e))?;
        let found = cwd
            .ancestors()
            .map(|dir| dir.join(CONFIG_FILE))
            .find(|path| path.is_file())
            .ok_or_else(|| {
                SynkError::new(
                    ErrorCode::ConfigNotFound,
                    format!("no {CONFIG_FILE} in {} or its parents", cwd.display()),
                )
                .with_next("synk init")
                .with_hint("then add your documents: synk source add <path>")
            })?;
        existing_path(&found).and_then(Self::load)
    }

    fn load(config_path: PathBuf) -> Result<Self> {
        let text = fs::read_to_string(&config_path)
            .map_err(|e| SynkError::io(config_path.display(), e))?;
        let doc: DocumentMut = text.parse().map_err(|e| invalid(&config_path, e))?;
        let ConfigFile { sources, pipeline } =
            toml::from_str(&text).map_err(|e| invalid(&config_path, e))?;
        pipeline.validate().map_err(|e| invalid(&config_path, e))?;
        if let Some(bad) = sources.iter().find(|s| s.kind != synk_core::FOLDER) {
            return Err(invalid(
                &config_path,
                format!(
                    "source `{}` has type `{}`; synk can read: {}",
                    bad.name,
                    bad.kind,
                    synk_core::FOLDER
                ),
            ));
        }
        let root = config_path
            .parent()
            .expect("config file has a parent")
            .to_path_buf();
        Ok(Self {
            root,
            config_path,
            doc,
            sources,
            pipeline,
        })
    }

    /// The `[model]`, `[generator]`, `[dataset]` and `[trainer]` settings.
    pub fn pipeline(&self) -> &PipelineConfig {
        &self.pipeline
    }

    /// The Python from `[trainer] python`, resolved against the project.
    pub fn trainer_python(&self) -> Option<PathBuf> {
        self.pipeline
            .trainer
            .python
            .as_ref()
            .map(|p| self.root.join(p))
    }

    /// Saved pipeline outputs. Nothing is created until a step writes one;
    /// `open_store` has created `.synk/` by then.
    pub fn artifacts(&self) -> Artifacts {
        Artifacts::new(self.root.join(DATA_DIR).join(ARTIFACTS_DIR))
    }

    pub fn sources(&self) -> &[SourceSpec] {
        &self.sources
    }

    pub fn resolve(&self, spec: &SourceSpec) -> SourceSpec {
        SourceSpec {
            path: self.root.join(&spec.path),
            ..spec.clone()
        }
    }

    /// Sources with absolute paths.
    pub fn resolved_sources(&self) -> Vec<SourceSpec> {
        self.sources.iter().map(|s| self.resolve(s)).collect()
    }

    pub fn unique_name(&self, base: &str) -> String {
        unique_name(&self.sources, base)
    }

    /// Adds a source whose path is absolute. Paths inside the project are
    /// written relative to it, so the project folder stays portable.
    pub fn add_source(&mut self, spec: &SourceSpec) -> Result<()> {
        let path = self.portable_path(&spec.path);
        let mut table = Table::new();
        table["name"] = value(spec.name.as_str());
        table["path"] = value(path.as_str());
        if !spec.exclude.is_empty() {
            table["exclude"] = value(spec.exclude.iter().map(String::as_str).collect::<Array>());
        }
        // In a file with only comments, toml_edit keeps them as trailing
        // text, which would push them below the new table. Move them above.
        let prefix = if self.doc.is_empty() {
            let comments = self.doc.trailing().as_str().unwrap_or_default().to_owned();
            self.doc.set_trailing("");
            comments
        } else {
            "\n".to_owned()
        };
        table.decor_mut().set_prefix(prefix);
        self.sources_table()?.push(table);
        self.sources.push(SourceSpec {
            path: PathBuf::from(path),
            ..spec.clone()
        });
        Ok(())
    }

    pub fn remove_source(&mut self, name: &str) -> Result<()> {
        let tables = self.sources_table()?;
        let index = tables
            .iter()
            .position(|t| t.get("name").and_then(Item::as_str) == Some(name));
        if let Some(index) = index {
            tables.remove(index);
        }
        self.sources.retain(|s| s.name != name);
        Ok(())
    }

    pub fn save(&self) -> Result<()> {
        fs::write(&self.config_path, self.doc.to_string())
            .map_err(|e| SynkError::io(self.config_path.display(), e))
    }

    pub fn store_path(&self) -> PathBuf {
        self.root.join(DATA_DIR).join(STORE_FILE)
    }

    /// Opens the store, creating `.synk/` (git-ignored) on first use.
    pub fn open_store(&self) -> Result<Store> {
        self.data_dir()?;
        Store::open(&self.store_path())
    }

    /// `.synk/`, created with a `.gitignore` on first use.
    pub fn data_dir(&self) -> Result<PathBuf> {
        let data_dir = self.root.join(DATA_DIR);
        if !data_dir.exists() {
            fs::create_dir_all(&data_dir).map_err(|e| SynkError::io(data_dir.display(), e))?;
            let gitignore = data_dir.join(".gitignore");
            fs::write(&gitignore, "*\n").map_err(|e| SynkError::io(gitignore.display(), e))?;
        }
        Ok(data_dir)
    }

    /// Opens the store only if a sync has already created it.
    pub fn existing_store(&self) -> Result<Option<Store>> {
        let path = self.store_path();
        if path.exists() {
            Store::open(&path).map(Some)
        } else {
            Ok(None)
        }
    }

    /// Every setting with its effective value, defaults included, as
    /// `{"section": {"key": value}}`.
    pub fn effective(&self) -> serde_json::Value {
        serde_json::to_value(&self.pipeline).unwrap_or_default()
    }

    /// The keys written in `synk.toml` (the rest come from defaults).
    pub fn keys_in_file(&self) -> Vec<String> {
        let mut keys = Vec::new();
        for (section, item) in self.doc.iter() {
            if let Some(table) = item.as_table() {
                if section == "sources" {
                    continue;
                }
                for (key, _) in table.iter() {
                    keys.push(format!("{section}.{key}"));
                }
            }
        }
        keys
    }

    /// The effective value of `section.key`.
    pub fn get(&self, key: &str) -> Result<serde_json::Value> {
        let (section, name) = split_key(key)?;
        let value = &self.effective()[section][name];
        if self.effective()[section].get(name).is_none() {
            return Err(unknown_key(key, &self.effective()));
        }
        Ok(value.clone())
    }

    /// Sets `section.key` in `synk.toml`. `value` is TOML (`5`, `true`,
    /// `["vi", "en"]`, `"text"`); anything that doesn't parse as TOML is
    /// taken as a string. The file is only saved if the result is valid.
    /// Returns the old and new effective values.
    pub fn set(
        &mut self,
        key: &str,
        value: &str,
    ) -> Result<(serde_json::Value, serde_json::Value)> {
        let old = self.get(key)?;
        self.edit(|doc| set_in(doc, key, value))?;
        Ok((old, self.get(key)?))
    }

    /// The settings with `sets` (`key=value` pairs, as for `config set`)
    /// applied on top of synk.toml, without saving anything; and the
    /// overrides as `{key: value}`.
    pub fn with_overrides(
        &self,
        sets: &[String],
    ) -> Result<(PipelineConfig, BTreeMap<String, serde_json::Value>)> {
        let mut doc = self.doc.clone();
        let mut keys = Vec::new();
        for set in sets {
            let (key, value) = set.split_once('=').ok_or_else(|| {
                SynkError::new(
                    ErrorCode::InvalidArgs,
                    format!("--set `{set}`: use KEY=VALUE, e.g. --set model.base=bge-m3"),
                )
            })?;
            let key = key.trim();
            self.get(key)?;
            set_in(&mut doc, key, value.trim())?;
            keys.push(key.to_owned());
        }
        let ConfigFile { pipeline, .. } = parse(&doc.to_string())?;
        let effective = serde_json::to_value(&pipeline).unwrap_or_default();
        let overrides = keys
            .into_iter()
            .map(|k| {
                let (section, name) = k.split_once('.').expect("checked key");
                let v = effective[section][name].clone();
                (k, v)
            })
            .collect();
        Ok((pipeline, overrides))
    }

    /// Everything a run needs, with `sets` applied.
    pub fn workspace(&self, sets: &[String]) -> Result<Workspace> {
        let (cfg, overrides) = self.with_overrides(sets)?;
        self.workspace_with(cfg, overrides)
    }

    /// Everything a run needs, with these settings instead of `synk.toml`'s
    /// (a run continued with the settings it started with).
    pub fn workspace_with(
        &self,
        cfg: PipelineConfig,
        overrides: BTreeMap<String, serde_json::Value>,
    ) -> Result<Workspace> {
        let data_dir = self.data_dir()?;
        Ok(Workspace {
            root: self.root.clone(),
            data_dir,
            uv: cfg.trainer.uv.as_ref().map(|p| self.root.join(p)),
            trainer_python: cfg.trainer.python.as_ref().map(|p| self.root.join(p)),
            cfg,
            overrides,
            sources: self.resolved_sources(),
        })
    }

    /// Removes `section.key` from `synk.toml`, so its default applies.
    pub fn unset(&mut self, key: &str) -> Result<(serde_json::Value, serde_json::Value)> {
        let old = self.get(key)?;
        let (section, name) = split_key(key)?;
        self.edit(|doc| {
            if let Some(table) = doc.get_mut(section).and_then(Item::as_table_mut) {
                table.remove(name);
            }
            Ok(())
        })?;
        Ok((old, self.get(key)?))
    }

    /// Applies `change` to a copy of the file, validates the result, and
    /// only then saves it and reloads.
    fn edit(&mut self, change: impl FnOnce(&mut DocumentMut) -> Result<()>) -> Result<()> {
        let mut doc = self.doc.clone();
        change(&mut doc)?;
        let text = doc.to_string();
        let ConfigFile { sources, pipeline } = parse(&text)?;
        fs::write(&self.config_path, &text)
            .map_err(|e| SynkError::io(self.config_path.display(), e))?;
        self.doc = doc;
        self.sources = sources;
        self.pipeline = pipeline;
        Ok(())
    }

    fn portable_path(&self, abs: &Path) -> String {
        match abs.strip_prefix(&self.root) {
            Ok(rel) if rel.as_os_str().is_empty() => ".".to_owned(),
            Ok(rel) => rel.to_string_lossy().replace('\\', "/"),
            Err(_) => abs.to_string_lossy().into_owned(),
        }
    }

    fn sources_table(&mut self) -> Result<&mut ArrayOfTables> {
        self.doc
            .entry("sources")
            .or_insert(Item::ArrayOfTables(ArrayOfTables::new()))
            .as_array_of_tables_mut()
            .ok_or_else(|| {
                SynkError::new(
                    ErrorCode::ConfigInvalid,
                    "`sources` must be an array of tables ([[sources]])",
                )
            })
    }
}

/// Canonicalizes a path that must exist, without Windows `\\?\` prefixes.
pub fn existing_path(path: &Path) -> Result<PathBuf> {
    dunce::canonicalize(path).map_err(|_| {
        SynkError::new(
            ErrorCode::SourceMissing,
            format!("path does not exist: {}", path.display()),
        )
    })
}

/// `base`, or `base-2`, `base-3`, ... if the name is taken.
pub fn unique_name(existing: &[SourceSpec], base: &str) -> String {
    let taken = |name: &str| existing.iter().any(|s| s.name == name);
    if !taken(base) {
        return base.to_owned();
    }
    (2..)
        .map(|i| format!("{base}-{i}"))
        .find(|name| !taken(name))
        .expect("unbounded range")
}

/// Parses and validates a whole synk.toml.
fn parse(text: &str) -> Result<ConfigFile> {
    let file: ConfigFile = toml::from_str(text).map_err(|e| {
        SynkError::new(
            ErrorCode::ConfigInvalid,
            format!("that value is not valid: {e}"),
        )
    })?;
    file.pipeline
        .validate()
        .map_err(|e| SynkError::new(ErrorCode::ConfigInvalid, e))?;
    Ok(file)
}

/// Sets `section.key` in a document. `value` is TOML when it parses as
/// TOML, a string otherwise.
fn set_in(doc: &mut DocumentMut, key: &str, value: &str) -> Result<()> {
    let (section, name) = split_key(key)?;
    let parsed: toml_edit::Value = value
        .parse()
        .unwrap_or_else(|_| toml_edit::Value::from(value));
    let table = doc
        .entry(section)
        .or_insert(Item::Table(Table::new()))
        .as_table_mut()
        .ok_or_else(|| {
            SynkError::new(
                ErrorCode::ConfigInvalid,
                format!("`{section}` in synk.toml is not a table"),
            )
        })?;
    table[name] = Item::Value(parsed);
    Ok(())
}

/// `generator.model` -> (`generator`, `model`).
fn split_key(key: &str) -> Result<(&str, &str)> {
    key.split_once('.')
        .filter(|(s, k)| !s.is_empty() && !k.is_empty() && !k.contains('.'))
        .ok_or_else(|| {
            SynkError::new(
                ErrorCode::ConfigKeyUnknown,
                format!("`{key}` is not a setting; settings look like `generator.model`"),
            )
            .with_next("synk config show")
        })
}

fn unknown_key(key: &str, effective: &serde_json::Value) -> SynkError {
    let known: Vec<String> = effective
        .as_object()
        .into_iter()
        .flatten()
        .flat_map(|(section, keys)| {
            keys.as_object()
                .into_iter()
                .flatten()
                .map(move |(k, _)| format!("{section}.{k}"))
        })
        .collect();
    SynkError::new(
        ErrorCode::ConfigKeyUnknown,
        format!("`{key}` is not a setting"),
    )
    .with_next("synk config show")
    .with_details(serde_json::json!({ "known": known }))
}

fn invalid(path: &Path, err: impl fmt::Display) -> SynkError {
    SynkError::new(
        ErrorCode::ConfigInvalid,
        format!("{}: {err}", path.display()),
    )
}
