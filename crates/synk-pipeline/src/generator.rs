//! Writing synthetic queries for chunks with an LLM.
//!
//! One client covers every supported server: Ollama, llama.cpp's
//! `llama-server`, vLLM and LM Studio all serve the OpenAI-compatible
//! `/chat/completions` endpoint.

use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use synk_core::{ErrorCode, Result, SynkError};
use whatlang::Lang;

use crate::config::GeneratorConfig;
use crate::lang;

/// Bump when the prompt or the parsing changes, so cached queries are not
/// reused for a different recipe.
const PROMPT_VERSION: u32 = 3;
const MAX_QUERY_CHARS: usize = 300;
const SEED: u64 = 42;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GeneratedQuery {
    pub text: String,
    /// ISO code of the language the query was asked to be in, if known.
    pub lang: Option<String>,
    /// True for a query written in another language than the chunk.
    pub cross: bool,
}

/// What to ask for one chunk.
pub struct Ask<'a> {
    pub title: &'a str,
    /// The headings above the chunk, e.g. `Ops > Backups`.
    pub heading: &'a str,
    pub chunk: &'a str,
    pub chunk_lang: Option<Lang>,
    /// Same-language queries wanted.
    pub n: usize,
    /// Languages that still need one cross-language query each.
    pub cross: &'a [Lang],
    /// Queries this chunk already has; new ones must differ.
    pub avoid: &'a [String],
}

pub trait QueryGenerator {
    /// Identifies everything that changes the output except how many
    /// queries are asked for, for caching: queries are cached one by one,
    /// so asking for more later only generates the missing ones.
    fn key(&self) -> String;

    /// The configured cross-language languages.
    fn languages(&self) -> &[Lang];

    /// Queries for one chunk. `Ok(None)` means the LLM gave no usable answer
    /// for this chunk; an error means generation can't continue at all.
    fn generate(&self, ask: &Ask<'_>) -> Result<Option<Vec<GeneratedQuery>>>;

    /// Short description for reports, e.g. `qwen3:8b at http://localhost:11434/v1`.
    fn describe(&self) -> String;

    /// Called when generation is done, so the server can free the memory
    /// the model holds (the GPU the trainer needs next). Best effort.
    fn release(&self) {}
}

pub struct OpenAiGenerator {
    agent: ureq::Agent,
    base_url: String,
    model: String,
    languages: Vec<Lang>,
    /// A few of the user's real searches, shown as examples of their style.
    style: Vec<String>,
    api_key: Option<String>,
    /// The model's content digest, when the server is Ollama. Part of the
    /// cache key, so re-pulling a tag with new weights writes new queries.
    digest: Option<String>,
    /// Chunks in a row the server failed on (HTTP 5xx). One bad chunk is
    /// skipped; several in a row mean the server itself is broken.
    server_failures: AtomicU32,
}

/// Chunks in a row the server may fail on before generation stops.
const MAX_SERVER_FAILURES: u32 = 3;

/// What `synk status` can tell about the query generator without sending
/// any text: whether the server answers and whether it has the model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Probe {
    /// `None` when not checked (a remote server).
    pub reachable: Option<bool>,
    /// `None` when the server doesn't list its models.
    pub model_ready: Option<bool>,
    /// True when the server is Ollama, which can download models itself.
    pub ollama: bool,
    pub message: String,
}

/// Asks the server for its model list, with a short timeout. Never sends
/// document text, and never contacts a remote server.
pub fn probe(cfg: &GeneratorConfig) -> Probe {
    let base_url = cfg.base_url.trim_end_matches('/');
    if !is_local(base_url) {
        return Probe {
            reachable: None,
            model_ready: None,
            ollama: false,
            message: format!("{base_url} is not on this machine; not checked"),
        };
    }
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_global(Some(Duration::from_secs(3)))
        .build()
        .into();
    let wanted = [cfg.model.clone(), format!("{}:latest", cfg.model)];
    let root = base_url.trim_end_matches("/v1");
    // Ollama lists models at /api/tags, OpenAI-compatible servers at /models.
    for (ollama, url, list, keys) in [
        (
            true,
            format!("{root}/api/tags"),
            "models",
            ["name", "model"],
        ),
        (false, format!("{base_url}/models"), "data", ["id", "id"]),
    ] {
        match agent.get(&url).call() {
            Ok(mut response) => {
                let Ok(body) = response.body_mut().read_json::<Value>() else {
                    continue;
                };
                let Some(models) = body[list].as_array() else {
                    continue;
                };
                let found = models.iter().any(|m| {
                    keys.iter()
                        .any(|k| m[k].as_str().is_some_and(|n| wanted.iter().any(|w| w == n)))
                });
                return Probe {
                    reachable: Some(true),
                    model_ready: Some(found),
                    ollama,
                    message: if found {
                        format!("{} at {base_url}", cfg.model)
                    } else {
                        format!("{base_url} has no model `{}`", cfg.model)
                    },
                };
            }
            Err(ureq::Error::StatusCode(_)) => continue,
            Err(_) => {
                return Probe {
                    reachable: Some(false),
                    model_ready: None,
                    ollama: false,
                    message: format!("nothing answers at {base_url}"),
                };
            }
        }
    }
    Probe {
        reachable: Some(true),
        model_ready: None,
        ollama: false,
        message: format!("{base_url} answers but doesn't list its models"),
    }
}

/// The download size of an Ollama model, from the public registry's
/// manifest (sends only the model name). `None` when unknown.
pub fn ollama_size(model: &str) -> Option<u64> {
    let (name, tag) = model.split_once(':').unwrap_or((model, "latest"));
    let path = if name.contains('/') {
        name.to_owned()
    } else {
        format!("library/{name}")
    };
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_global(Some(Duration::from_secs(5)))
        .build()
        .into();
    let manifest: Value = agent
        .get(&format!(
            "https://registry.ollama.ai/v2/{path}/manifests/{tag}"
        ))
        .header(
            "Accept",
            "application/vnd.docker.distribution.manifest.v2+json",
        )
        .call()
        .ok()?
        .body_mut()
        .read_json()
        .ok()?;
    let layers = manifest["layers"].as_array()?;
    Some(layers.iter().filter_map(|l| l["size"].as_u64()).sum())
}

/// Has Ollama download `cfg.model`, reporting bytes done and total. Ollama
/// resumes an interrupted pull by itself.
pub fn pull(cfg: &GeneratorConfig, on_progress: &mut dyn FnMut(u64, u64)) -> Result<()> {
    use std::io::BufRead as _;
    let root = cfg.base_url.trim_end_matches('/').trim_end_matches("/v1");
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_global(None)
        .build()
        .into();
    let unavailable = |e: String| {
        SynkError::new(
            ErrorCode::GeneratorUnavailable,
            format!("Ollama couldn't download `{}`: {e}", cfg.model),
        )
    };
    let response = agent
        .post(&format!("{root}/api/pull"))
        .send_json(json!({ "model": cfg.model, "stream": true }))
        .map_err(|e| unavailable(e.to_string()))?;
    let reader = std::io::BufReader::new(response.into_body().into_reader());
    for line in reader.lines() {
        let line = line.map_err(|e| unavailable(e.to_string()))?;
        let Ok(event) = serde_json::from_str::<Value>(&line) else {
            continue;
        };
        if let Some(err) = event["error"].as_str() {
            return Err(unavailable(err.to_owned()));
        }
        if let (Some(done), Some(total)) = (event["completed"].as_u64(), event["total"].as_u64()) {
            on_progress(done, total);
        }
        if event["status"] == "success" {
            return Ok(());
        }
    }
    Err(unavailable("the download stopped early".to_owned()))
}

impl OpenAiGenerator {
    pub fn new(cfg: &GeneratorConfig) -> Result<Self> {
        let base_url = cfg.base_url.trim_end_matches('/').to_owned();
        if !is_local(&base_url) && !cfg.allow_remote {
            return Err(SynkError::new(
                ErrorCode::RemoteNotAllowed,
                format!("{base_url} is not on this machine, and generating queries would send your documents to it"),
            )
            .with_action(synk_core::ErrorAction::run(
                "synk config set generator.allow_remote true",
                synk_core::Applicability::Manual,
                "only if the person whose documents these are agrees to send them there",
            )));
        }
        let languages = cfg
            .languages
            .iter()
            .map(|l| lang::parse(l))
            .collect::<Result<Vec<_>>>()?;
        let api_key = match &cfg.api_key_env {
            Some(var) => Some(std::env::var(var).map_err(|_| {
                SynkError::new(
                    ErrorCode::ConfigInvalid,
                    format!("[generator] api_key_env names `{var}`, but it is not set"),
                )
            })?),
            None => None,
        };
        let agent = ureq::Agent::config_builder()
            .timeout_global(Some(Duration::from_secs(600)))
            .build()
            .into();
        Ok(Self {
            agent,
            base_url,
            model: cfg.model.clone(),
            languages,
            style: Vec::new(),
            api_key,
            digest: None,
            server_failures: AtomicU32::new(0),
        })
    }

    /// Asks the server which weights it serves for the model. Ollama lists
    /// them with a digest at `/api/tags`; other servers have no such
    /// endpoint, and the model name is all synk can key on.
    pub fn connect(&mut self) -> Result<()> {
        let root = self.base_url.trim_end_matches("/v1");
        let mut response = match self.agent.get(&format!("{root}/api/tags")).call() {
            Ok(response) => response,
            Err(ureq::Error::StatusCode(_)) => return Ok(()),
            Err(e) => return Err(self.request_error(e)),
        };
        let Ok(tags) = response.body_mut().read_json::<Value>() else {
            return Ok(());
        };
        let Some(models) = tags["models"].as_array() else {
            return Ok(());
        };
        let wanted = [self.model.clone(), format!("{}:latest", self.model)];
        let found = models.iter().find(|m| {
            ["name", "model"]
                .iter()
                .any(|k| m[k].as_str().is_some_and(|n| wanted.iter().any(|w| w == n)))
        });
        match found {
            Some(m) => {
                self.digest = m["digest"].as_str().map(str::to_owned);
                Ok(())
            }
            None => Err(SynkError::new(
                ErrorCode::GeneratorUnavailable,
                format!("Ollama at {root} has no model `{}`", self.model),
            )
            .with_action(synk_core::ErrorAction::run(
                &format!("ollama pull {}", self.model),
                synk_core::Applicability::Review,
                "download the model (often several GB)",
            ))),
        }
    }

    /// Real queries to show as examples of how this user searches.
    pub fn set_style(&mut self, examples: Vec<String>) {
        self.style = examples;
    }

    fn prompt(&self, ask: &Ask<'_>) -> String {
        let lang_hint = ask
            .chunk_lang
            .map_or(String::new(), |l| format!(" ({})", l.eng_name()));
        let mut prompt = format!("Document title: {}\n", ask.title);
        if !ask.heading.is_empty() {
            prompt.push_str(&format!("Section: {}\n", ask.heading));
        }
        prompt.push_str(&format!("\nPassage:\n\"\"\"\n{}\n\"\"\"\n\n", ask.chunk));
        if ask.n > 0 {
            prompt.push_str(&format!(
                "Write {n} different search queries that someone would type into a search box to find this passage later.\n\
                 - Write them in the same language as the passage{lang_hint}. If the passage mixes in English technical terms, the queries may too.\n\
                 - Keep them short and realistic: a few keywords or a short question.\n\
                 - Each query must be answered by this passage, but must not copy a sentence from it.\n",
                n = ask.n,
            ));
        }
        if !self.style.is_empty() {
            prompt
                .push_str("- This person searches like this (match the style, not the topics):\n");
            for example in &self.style {
                prompt.push_str(&format!("  - {example}\n"));
            }
        }
        if !ask.avoid.is_empty() {
            prompt.push_str("- Don't repeat these existing queries; write different ones:\n");
            for q in ask.avoid {
                prompt.push_str(&format!("  - {q}\n"));
            }
        }
        let schema = if ask.cross.is_empty() {
            r#"{"queries": ["...", "..."]}"#.to_owned()
        } else {
            let names: Vec<&str> = ask.cross.iter().map(|l| l.eng_name()).collect();
            prompt.push_str(&format!(
                "- {} one search query in each of these languages, as someone searching in that language would: {}.\n",
                if ask.n > 0 { "Also write" } else { "Write" },
                names.join(", ")
            ));
            r#"{"queries": ["...", "..."], "cross": [{"language": "...", "query": "..."}]}"#
                .to_owned()
        };
        prompt.push_str(&format!("\nReply with JSON only: {schema}"));
        prompt
    }

    /// The model's reply. `Ok(Err(..))` is the server failing on this
    /// prompt (HTTP 5xx, e.g. Ollama's "token repeat limit reached" when the
    /// model loops); other errors mean the server can't be used at all.
    fn complete(
        &self,
        prompt: &str,
        attempt: u64,
    ) -> Result<std::result::Result<String, SynkError>> {
        let body = json!({
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You create search queries for training a search engine. Reply with a single JSON object and nothing else."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.7,
            "seed": SEED + attempt,
            "response_format": {"type": "json_object"},
            // Reasoning models (e.g. qwen3) otherwise spend minutes thinking
            // per chunk. Servers that don't know the field ignore it.
            "reasoning_effort": "none",
        });
        let url = format!("{}/chat/completions", self.base_url);
        // Error statuses are read here, so the server's own message reaches
        // the user instead of only its status code.
        let mut request = self
            .agent
            .post(&url)
            .config()
            .http_status_as_error(false)
            .build();
        if let Some(key) = &self.api_key {
            request = request.header("Authorization", &format!("Bearer {key}"));
        }
        let mut response = request
            .send_json(&body)
            .map_err(|e| self.request_error(e))?;
        let status = response.status().as_u16();
        if status >= 400 {
            let text = response.body_mut().read_to_string().unwrap_or_default();
            let err = self.request_error(ureq::Error::StatusCode(status));
            let detail: String = text.trim().chars().take(300).collect();
            let err = if detail.is_empty() {
                err
            } else {
                let message = format!("{}: {detail}", err.message);
                SynkError { message, ..err }
            };
            return if status >= 500 {
                Ok(Err(err))
            } else {
                Err(err)
            };
        }
        let response: Value = response
            .body_mut()
            .read_json()
            .map_err(|e| self.request_error(e))?;
        Ok(Ok(response["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or_default()
            .to_owned()))
    }

    fn request_error(&self, err: ureq::Error) -> SynkError {
        match err {
            ureq::Error::StatusCode(404) => SynkError::new(
                ErrorCode::GeneratorUnavailable,
                format!(
                    "the server at {} has no model `{}`",
                    self.base_url, self.model
                ),
            )
            .with_action(synk_core::ErrorAction::run(
                &format!("ollama pull {}", self.model),
                synk_core::Applicability::Review,
                "download the model (often several GB)",
            )),
            ureq::Error::StatusCode(code) => SynkError::new(
                ErrorCode::GeneratorUnavailable,
                format!(
                    "the query generator at {} answered HTTP {code}",
                    self.base_url
                ),
            ),
            other => SynkError::new(
                ErrorCode::GeneratorUnavailable,
                format!(
                    "could not reach the query generator at {}: {other}",
                    self.base_url
                ),
            )
            .with_action(synk_core::ErrorAction::run(
                "ollama serve",
                synk_core::Applicability::Review,
                "start Ollama",
            ))
            .with_hint("or point generator.base_url at a running OpenAI-compatible server"),
        }
    }
}

impl QueryGenerator for OpenAiGenerator {
    fn key(&self) -> String {
        let mut languages: Vec<&str> = self.languages.iter().map(|l| l.code()).collect();
        languages.sort_unstable();
        let canonical = json!({
            "prompt_version": PROMPT_VERSION,
            "model": self.model,
            "languages": languages,
            "digest": self.digest,
            "style": self.style,
        });
        blake3::hash(canonical.to_string().as_bytes()).to_hex()[..16].to_owned()
    }

    fn languages(&self) -> &[Lang] {
        &self.languages
    }

    fn generate(&self, ask: &Ask<'_>) -> Result<Option<Vec<GeneratedQuery>>> {
        let prompt = self.prompt(ask);
        for attempt in 0..2 {
            // A server error on this chunk: try another seed, then skip the
            // chunk, unless the server failed on several chunks in a row.
            let reply = match self.complete(&prompt, attempt)? {
                Ok(reply) => reply,
                Err(err) if attempt == 1 => {
                    let failures = self.server_failures.fetch_add(1, Ordering::Relaxed) + 1;
                    if failures >= MAX_SERVER_FAILURES {
                        return Err(err);
                    }
                    return Ok(None);
                }
                Err(_) => continue,
            };
            self.server_failures.store(0, Ordering::Relaxed);
            let queries: Vec<GeneratedQuery> =
                parse_reply(&reply, ask.chunk, ask.chunk_lang, ask.cross, ask.n)
                    .into_iter()
                    .filter(|q| !ask.avoid.iter().any(|a| a.eq_ignore_ascii_case(&q.text)))
                    .collect();
            if !queries.is_empty() {
                return Ok(Some(queries));
            }
        }
        Ok(None)
    }

    fn describe(&self) -> String {
        format!("{} at {}", self.model, self.base_url)
    }

    /// Ollama keeps a model loaded for minutes after its last request,
    /// holding GPU memory the trainer needs; `keep_alive: 0` unloads it now.
    /// Only for Ollama (known by its digest); errors are ignored.
    fn release(&self) {
        if self.digest.is_none() {
            return;
        }
        let root = self.base_url.trim_end_matches("/v1");
        let _ = self
            .agent
            .post(&format!("{root}/api/generate"))
            .send_json(json!({"model": self.model, "keep_alive": 0}));
    }
}

/// Extracts and cleans the queries from an LLM reply. Tolerates reasoning
/// blocks and text around the JSON object.
fn parse_reply(
    reply: &str,
    chunk: &str,
    chunk_lang: Option<Lang>,
    cross: &[Lang],
    max_same_lang: usize,
) -> Vec<GeneratedQuery> {
    let reply = match reply.rfind("</think>") {
        Some(end) => &reply[end + "</think>".len()..],
        None => reply,
    };
    let (Some(start), Some(end)) = (reply.find('{'), reply.rfind('}')) else {
        return Vec::new();
    };
    let Ok(value) = serde_json::from_str::<Value>(&reply[start..=end]) else {
        return Vec::new();
    };

    let mut out: Vec<GeneratedQuery> = Vec::new();
    let same_lang = chunk_lang.map(|l| lang::code(l).to_owned());
    for q in value["queries"].as_array().into_iter().flatten() {
        // LLMs sometimes write more than they were asked for.
        if out.len() == max_same_lang {
            break;
        }
        if let Some(text) = q.as_str() {
            push_clean(&mut out, text, same_lang.clone(), false, chunk);
        }
    }
    for item in value["cross"].as_array().into_iter().flatten() {
        let Some(text) = item["query"].as_str() else {
            continue;
        };
        let requested = item["language"]
            .as_str()
            .and_then(|l| lang::parse(l).ok())
            .filter(|l| cross.contains(l));
        // Only keep cross-language queries for languages that were asked for.
        if let Some(l) = requested {
            push_clean(&mut out, text, Some(lang::code(l).to_owned()), true, chunk);
        }
    }
    out
}

fn push_clean(
    out: &mut Vec<GeneratedQuery>,
    text: &str,
    lang: Option<String>,
    cross: bool,
    chunk: &str,
) {
    let text = synk_connectors::readers::nfc(text.trim().trim_matches('"').trim());
    let text = text.as_str();
    let chars = text.chars().count();
    if !(2..=MAX_QUERY_CHARS).contains(&chars) {
        return;
    }
    // A long query found verbatim in the chunk is a copied sentence, which
    // teaches the model nothing about how people search.
    if chars > 40 && chunk.contains(text) {
        return;
    }
    if out
        .iter()
        .any(|q| q.text.to_lowercase() == text.to_lowercase())
    {
        return;
    }
    out.push(GeneratedQuery {
        text: text.to_owned(),
        lang,
        cross,
    });
}

/// True when `base_url` points at this machine. Parsed by the same URL
/// parser the requests use, so the host checked is the host contacted
/// (`http://evil.com?@127.0.0.1` is evil.com).
pub fn is_local(base_url: &str) -> bool {
    let Ok(uri) = base_url.parse::<ureq::http::Uri>() else {
        return false;
    };
    let Some(host) = uri.host() else {
        return false;
    };
    let host = host.trim_start_matches('[').trim_end_matches(']');
    host.eq_ignore_ascii_case("localhost")
        || host
            .parse::<std::net::IpAddr>()
            .is_ok_and(|ip| ip.is_loopback())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_loopback_hosts_are_local() {
        assert!(is_local("http://localhost:11434/v1"));
        assert!(is_local("http://127.0.0.1:8080/v1"));
        assert!(is_local("http://[::1]:8080/v1"));
        assert!(!is_local("https://api.example.com/v1"));
        assert!(!is_local("http://localhost.example.com/v1"));
        assert!(!is_local("http://192.168.1.5:11434/v1"));
        assert!(!is_local("http://127.evil.example/v1"));
        assert!(!is_local("http://127.0.0.1.example.com/v1"));
        assert!(!is_local("http://evil.com?@127.0.0.1"));
        assert!(!is_local("http://evil.com#@127.0.0.1/v1"));
        assert!(is_local("http://user:pw@127.0.0.1:8080/v1"));
    }

    #[test]
    fn remote_server_needs_opt_in() {
        let cfg = GeneratorConfig {
            base_url: "https://api.example.com/v1".into(),
            ..GeneratorConfig::default()
        };
        assert_eq!(
            OpenAiGenerator::new(&cfg).err().unwrap().code,
            ErrorCode::RemoteNotAllowed
        );
        let allowed = GeneratorConfig {
            allow_remote: true,
            ..cfg
        };
        assert!(OpenAiGenerator::new(&allowed).is_ok());
    }

    #[test]
    fn parses_replies_with_reasoning_and_cross_language_queries() {
        let chunk = "Deploy lên server bị lỗi timeout, phải tăng keepalive lên 75s trong nginx.";
        let reply = r#"<think>hmm</think> Sure: {"queries": ["lỗi timeout khi deploy", "LỖI TIMEOUT KHI DEPLOY", "", "Deploy lên server bị lỗi timeout, phải tăng keepalive lên 75s trong nginx."],
            "cross": [{"language": "English", "query": "nginx keepalive timeout on deploy"}, {"language": "French", "query": "délai"}]}"#;
        let queries = parse_reply(reply, chunk, Some(Lang::Vie), &[Lang::Eng], 3);
        assert_eq!(
            queries,
            [
                GeneratedQuery {
                    text: "lỗi timeout khi deploy".into(),
                    lang: Some("vi".into()),
                    cross: false
                },
                GeneratedQuery {
                    text: "nginx keepalive timeout on deploy".into(),
                    lang: Some("en".into()),
                    cross: true
                },
            ]
        );
        assert!(parse_reply("no json here", chunk, None, &[], 3).is_empty());

        let many = r#"{"queries": ["a1", "a2", "a3", "a4"]}"#;
        let capped = parse_reply(many, chunk, None, &[], 2);
        assert_eq!(
            capped.iter().map(|q| q.text.as_str()).collect::<Vec<_>>(),
            ["a1", "a2"]
        );
    }

    #[test]
    fn prompt_carries_heading_style_cross_and_queries_to_avoid() {
        let mut generator = OpenAiGenerator::new(&GeneratorConfig::default()).unwrap();
        let key_without_style = generator.key();
        generator.set_style(vec!["nginx timeout?".into()]);
        assert_ne!(
            generator.key(),
            key_without_style,
            "style changes the prompt"
        );
        let avoid = vec!["tăng timeout".to_owned()];
        let ask = Ask {
            title: "t",
            heading: "Ops > Nginx",
            chunk: "passage",
            chunk_lang: Some(Lang::Vie),
            n: 2,
            cross: &[Lang::Eng],
            avoid: &avoid,
        };
        let prompt = generator.prompt(&ask);
        assert!(prompt.contains("Section: Ops > Nginx"), "{prompt}");
        assert!(
            prompt.contains("Write 2 different search queries"),
            "{prompt}"
        );
        assert!(prompt.contains("  - nginx timeout?"), "{prompt}");
        assert!(prompt.contains("  - tăng timeout"), "{prompt}");
        assert!(
            prompt.contains("in that language would: English."),
            "{prompt}"
        );
        let only_cross = Ask { n: 0, ..ask };
        let prompt = generator.prompt(&only_cross);
        assert!(!prompt.contains("different search queries"), "{prompt}");
        assert!(
            prompt.contains("- Write one search query in each"),
            "{prompt}"
        );
    }
}
