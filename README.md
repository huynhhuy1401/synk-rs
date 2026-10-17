# synk

Fine-tune a small embedding (search) model on your own documents, locally,
and prove with a statistical test whether it beats the original. Built in
Rust, designed to be driven by AI agents, comfortable for developers.

- **Your documents stay on your machine.** A local LLM (Ollama, LM Studio,
  llama.cpp, vLLM) writes practice queries; a local trainer tunes the model.
- **An honest verdict.** "Better" only when a 95% interval says so, overall,
  per language, on your real searches, plus a general check that ordinary
  search didn't get worse.
- **Agent-first.** One JSON object per command, stable codes, nothing
  downloaded without consent, and runs an agent can wait on.

## How it works

```
synk.toml (+ --set) ──synk train──▶ sync → dataset → train → eval
         │                                                  │
   missing model, LLM or trainer?             run:3-e5-small: verdict, advice
   → stops: "synk fetch <names>"                            │
                                 synk export <run> (folder · ONNX · GGUF/Ollama)
```

1. **sync** reads your documents (Markdown, text, HTML, Word, PDF).
2. **dataset** cuts them into chunks and has the LLM write the searches each
   chunk answers. Whole documents are held out for the test.
3. **train** drops queries the base model can't connect to their chunk,
   mines hard negatives (chunks that look right but aren't), and fine-tunes:
   every weight for small models, LoRA from 300M parameters.
4. **eval** compares the tuned and the base model on the held-out documents,
   your real searches, and public English and Vietnamese sets.
5. **export** writes the model you choose, and checks its vectors.

## Install

Prebuilt binaries for Windows, Linux (x86_64, arm64) and Apple Silicon Macs
are on the
[releases page](https://github.com/huynhhuy1401/synk-rs/releases), with
one-line installers (no Rust needed):

```sh
# macOS, Linux
curl --proto '=https' --tlsv1.2 -LsSf https://github.com/huynhhuy1401/synk-rs/releases/latest/download/synk-installer.sh | sh
```

```powershell
# Windows
powershell -ExecutionPolicy ByPass -c "irm https://github.com/huynhhuy1401/synk-rs/releases/latest/download/synk-installer.ps1 | iex"
```

Or from source: `cargo install --path crates/synk`.

## Quick start

```sh
synk init
synk source add ./notes
synk train --dry-run          # what would run, what it needs, how long; changes nothing
synk train --yes              # download what's missing, then run
synk export best --out ./notes-model
```

`synk train --dry-run` prints something like:

```
Would: a new run
  fetch    trainer    Python training environment (cpu)      [download 1.0 GB, ~5 min (rough)]
  fetch    models     multilingual-e5-small: intfloat/multilingual-e5-small@614241f not downloaded
  cached   generator  qwen3:4b
  run      sync       42 new, 0 changed, 0 deleted
  run      dataset    127 chunks; the LLM writes queries for 127 of them (qwen3:4b)   [~8 min (rough)]
  run      train      multilingual-e5-small, 320 steps; an earlier step runs first   [~13 min (rough)]
  run      eval       test against the base model           [~1 min (rough)]
Downloads 1.5 GB: synk fetch trainer multilingual-e5-small@614241f
Nothing leaves this machine. About 27 min in all.
Next (ask first): synk fetch trainer multilingual-e5-small@614241f  or: synk train --yes
```

Without `--yes`, `synk train` never downloads: when something is missing it
stops and prints the exact `synk fetch` command. Run that (it downloads
exactly those names, at those versions), then `synk train`.

Every `synk train` is a **run** with a short ID, `run:1-e5-small`. Running it
again with nothing changed returns the same run; an interrupted run continues
where it stopped. `synk wait`, `synk logs` and `synk cancel` act on whatever is
running. Ctrl-C stops a run at the next safe point (the trainer saves a
checkpoint first); a second Ctrl-C stops it at once. Either way `synk train`
continues it later.

`synk status` says where the project stands and what to do next, and checks
what synk needs: the sources exist, the trainer is installed, the LLM server
answers and has the model.

## Requirements

- **An LLM server** for writing queries: [Ollama](https://ollama.com) by
  default (`qwen3:8b`; any model works, e.g. `synk config set generator.model
  qwen3:4b`). With Ollama, synk has it download the model
  (`synk fetch ollama:qwen3:8b`, or `synk train --yes`).
- **[uv](https://docs.astral.sh/uv/)**, which synk uses to create the
  Python training environment (`synk fetch trainer`, or `synk setup` to pick a
  backend).

| Hardware | Training |
|---|---|
| CPU (any OS) | Supported. Fine for `multilingual-e5-small`. |
| NVIDIA GPU, Linux or Windows | Supported (CUDA 13.0): bf16 on RTX 30 and newer, fp16 on older cards. Trains on one GPU (the first, or the one in `CUDA_VISIBLE_DEVICES`). Tested on a Tesla T4 on Linux; Windows with CUDA is untested. |
| AMD GPU, Apple Silicon | Trains on CPU. AMD (ROCm) and Apple (MPS) GPUs aren't supported yet because they haven't been tested ([ADR 0003](docs/adr/0003-only-tested-gpus.md)). |

The CUDA 13.0 build of PyTorch needs NVIDIA driver 580 or newer.

## Consent

Downloads (the trainer, the LLM, models, the general check's sets) and
sending document text to a server off this machine need consent:

- **`synk fetch NAME…`** downloads exactly the names given, at the versions
  in the names (`multilingual-e5-small@614241f`, `hf:BAAI/bge-m3@5617a9f`,
  `ollama:qwen3:8b`, `general-check`, `trainer`); `synk fetch` alone fetches
  what `train --dry-run` lists. `synk setup` installs the trainer.
- **`--yes`** on `train` or `dataset` downloads whatever is missing first.

Without either, `train` fails with `CONSENT_REQUIRED`; `error.details.fetch`
lists the names. `--dry-run` never sends document text anywhere; it may ask
Hugging Face and the Ollama registry for model sizes, sending only model
names. Sending chunks to a remote LLM needs `generator.allow_remote = true`
and `--yes` on each run, and the run records what was sent.

## Models

`synk model list` and `synk model show <id>`:

| ID | Size | Notes |
|---|---|---|
| `multilingual-e5-small` (default) | 118M | ~100 languages. Fine-tunes on a CPU. |
| `multilingual-e5-base` | 278M | Better quality; a GPU helps. |
| `bge-m3` | 568M | 100+ languages, 8192-token inputs; LoRA; needs a GPU. |
| `embeddinggemma-300m` | 308M | Gated: accept Google's terms and set `HF_TOKEN`. LoRA. |
| `qwen3-embedding-0.6b` | 0.6B | Strong multilingual; LoRA; needs a GPU. |

Any sentence-transformers model works as a custom model: `hf:org/name`
(optionally `@revision`) or a local folder. Every download is pinned to a
commit. `HF_ENDPOINT` points at a mirror and `HF_TOKEN` reads private or
gated models, as for every Hugging Face tool. Models that need
`trust_remote_code` aren't supported.

Optional **rerankers** (`trainer.reranker`) double-check mined hard negatives
and drop the ones that probably answer the query too: `bge-reranker-v2-m3`
(strongest) or `mmarco-minilm-l12` (small, fine on a CPU).

Try a variant without editing `synk.toml`. It becomes its own run; then
compare:

```sh
synk train --set model.base=bge-m3 --name bge --yes    # -> run:2-bge
synk run compare 1 bge
```

## Results

```sh
synk run list                      # every run: state, verdict, settings that differ from synk.toml
synk run show best                 # one run: settings, verdict, advice (-v: metric tables)
synk run inspect                   # the searches the latest run helped or hurt most
synk run compare <a> <b>           # is B better than A? (paired test)
synk run report best --out r.html  # a one-page report to open in a browser
```

Runs are numbered per project: `run:3-e5-small`, or `run:4-bge` with
`--name bge`. Commands take `run:3`, `3`, the name, `latest` or `best` (the
largest `better` gain). Datasets are `dataset:2-qwen3-4b`. `synk train run:3`
continues run 3 with the settings it started with, even if `synk.toml`
changed since.

**The general check** (`eval.general_check`, on by default, about 10 MB once)
also tests both models on NanoMSMARCO (English web search) and XQuAD
(Vietnamese), so you see if tuning hurt ordinary search (advice
`GENERAL_REGRESSION`).

**Real queries.** Your own searches are the most honest test: they're used
to test, never to train, and 30 or more about 10+ documents decide the
verdict. A few (one in four, at most 5) become style examples for the LLM
instead and don't count toward the 30 (`queries import` reports `evaluated`
and `decides_verdict`); they're chosen once
and kept (`queries import --restyle` chooses again). Choosing them changes
every chunk's prompt, so the dry run says when it will happen and how many LLM
calls that costs (`style_examples_change`). To write some, let synk
list documents for you, and for each write how you'd search for something you
know is in it, **without opening it** (reading it first copies its words and
makes the test too easy). About 15 minutes:

```sh
synk queries template --out searches.jsonl     # 15 documents, test ones first
# fill in the empty "query" fields, then:
synk queries import searches.jsonl
```

Or write the file yourself:

```json
{"query": "tăng timeout nginx", "doc": "notes/nginx.md"}
{"query": "how do we vacuum postgres", "text": "Run VACUUM ANALYZE weekly"}
```

## Export

```sh
synk export best --out ./m                           # sentence-transformers folder
synk export best --out ./m-onnx --format onnx        # ONNX Runtime
synk export best --out ./m-gguf --format gguf --ollama notes-embed   # Ollama
```

Every export checks that the exported model's vectors match the original
(GGUF through Ollama's `/api/embed`; `q8_0` by default, `--quant f16` for an
exact copy). GGUF uses llama.cpp's converter, shipped inside synk; the result
(`data.converter`) and the model's README name its commit. The README synk
writes describes the run and holds none of your text. Ollama doesn't add a model's `query: `/`passage: ` prefixes, so apps
embedding through Ollama should add them (advice `PREFIX_DROPPED_BY_TARGET`).

**Using the model.** The model was trained and tested with its query and
document prefixes (for e5: `query: ` and `passage: `); an app that leaves them
out gets a quietly worse model. `synk export --json` prints `data.usage` with
copy-paste setup that keeps them, filled in with your path and prefixes:

| Tool | What synk gives you |
|---|---|
| sentence-transformers | `usage.sentence_transformers`: `encode_query` / `encode_document` |
| LlamaIndex | `usage.llamaindex`: `HuggingFaceEmbedding(..., query_instruction=, text_instruction=)`; without them it drops the prefixes |
| LangChain | `usage.langchain`: `HuggingFaceEmbeddings(..., encode_kwargs=, query_encode_kwargs=)` |
| Open WebUI | `usage.open_webui_env`: `RAG_EMBEDDING_QUERY_PREFIX` / `RAG_EMBEDDING_CONTENT_PREFIX` (built-in engine with the folder, or `ollama` with a GGUF export) |
| Ollama API | `usage.ollama_api`: `/api/embed` calls with the prefixes added |
| AnythingLLM | Its Ollama embedder can't add prefixes; expect lower quality than synk measured |

After switching models, re-index your documents in the app: vectors from two
models can't be compared.

## Configuration

`synk.toml` is the config. Edit it by hand, or with `synk config
set|unset|get|show` and `synk source add|remove`, which keep your comments and
refuse unknown keys and invalid values. Defaults:

```toml
[[sources]]
name = "notes"
path = "notes"              # relative to synk.toml, or absolute
exclude = ["archive/**"]    # globs relative to the source

[model]
base = "multilingual-e5-small"

[generator]                 # any OpenAI-compatible server
base_url = "http://localhost:11434/v1"
model = "qwen3:8b"
queries_per_chunk = "auto"  # aims for ~1,500 training queries: 3 to 8 per chunk
languages = []              # e.g. ["vi", "en"]: also one query per chunk in each other language
allow_remote = false        # must be true to send chunks to a server not on this machine
# api_key_env = "MY_KEY"

[dataset]
chunk_tokens = "auto"       # fills what the model reads (trainer.max_seq_length)
test_percent = "auto"       # at least 10 test documents where possible (20%-50%); or a number
# max_chunks = 200          # a sample, for a quick run (or: --sample 200)

[trainer]
epochs = "auto"             # enough for ~300 steps, at most 5
batch_size = 16             # above 32, a cached loss keeps memory flat
learning_rate = 2e-5
max_seq_length = 256
negatives = 3
query_filter = true
finetune = "auto"           # full below 300M parameters, LoRA from there; or "full" / "lora"
precision = "auto"          # bf16/fp16 on NVIDIA GPUs, fp32 on the CPU
lora_r = 32
lora_alpha = 64
lora_learning_rate = 1e-4
# reranker = "mmarco-minilm-l12"
# device = "cuda"
# python = "/path/to/python"  # your own environment instead of synk's

[eval]
general_check = true
```

Raising `queries_per_chunk` only asks the LLM for the missing queries: they
are cached one by one.

Synced documents live in `.synk/synk.db`; every step's output in
`.synk/artifacts/`, keyed by everything that went into it; runs, dataset
builds and fetches in `.synk/ops/`. Shared environments, model
weights and speed history live in `$SYNK_HOME` (default
`%LOCALAPPDATA%\synk` / `~/.local/share/synk`) and the Hugging Face cache.

**Disk space.** `train --dry-run` lists what the run will write on each drive
(`data.disk`: downloads, the dataset, about twice your documents' text, and
training checkpoints, about 20 bytes per parameter for full fine-tuning, 8 for
LoRA) against the space free there, and blocks with `INSUFFICIENT_DISK` when it
won't fit. The sizes are estimates; if one is wrong for you,
`SYNK_DISK_CHECK=warn` turns the block into a warning. If a disk fills up
anyway, the error's `details` name the step and the path.

## For agents

Load the skill: `synk skill --out .claude/skills/synk` (or wherever your
agent reads skills). `synk schema --json` describes every command, the
envelope, the error, preview, run record and event objects, and every error
and advice code.

Every command accepts `--json` and prints exactly one object on stdout:

```json
{"v": 1, "ok": true, "data": {...}}
{"v": 1, "ok": false, "error": {"code": "CONSENT_REQUIRED", "class": "prerequisite", "retryable": false,
  "message": "needs downloads first (470 MB): multilingual-e5-small@614241f",
  "actions": [{"argv": ["synk", "fetch", "multilingual-e5-small@614241f"], "applicability": "manual", "why": "..."},
              {"argv": ["synk", "train", "--yes"], "applicability": "manual", "why": "..."}],
  "details": {"fetch": ["multilingual-e5-small@614241f"], "downloads": [...], "egress": [], "dry_run": {...}}}}
```

- `v` is the contract version. Within `v: 1`, fields, codes and enum values
  are only ever added; ignore what you don't know.
- Exit code is `0` exactly when `ok` is `true`; `2` for usage errors.
- `next`, when present, runs exactly as written. Other fixes (with
  placeholders, or only for a person) are in `error.actions` with an
  `applicability`: `machine` (an agent may), `review`, `manual`. Downloads
  and remote sends are never `machine`.
- `error.class` (`usage`, `config`, `not_found`, `conflict`,
  `prerequisite`, `unavailable`, `data`, `resource`, `failed`, `internal`)
  for codes you don't know; `retryable` says whether the same command may
  succeed later.
- A verdict other than `better` is a result, not an error.
- No command prompts. Downloading commands are `fetch`, `setup`, and `train`
  / `dataset` with `--yes`; `export` writes files (and an Ollama model).
  Everything else is read-only or edits `synk.toml`.

**The loop.** `synk train --dry-run --json` → if `data.fetch_names` or
`data.egress` isn't empty, show the user the sizes and hosts and ask →
`synk fetch <names> --json` → `synk train --json`. `train` blocks until done
and writes a run record; its first stderr line is
`{"v": 1, "event": "op", "id": "run:3-e5-small", ...}`. If your shell times
out: `synk wait --timeout 9m --json` (`ok: true, done: false` when time runs
out), `synk logs --after <seq>`, `synk cancel`. One run, dataset build or
fetch runs per project (`JOB_RUNNING`). `synk train` again continues an
interrupted run (`attempt` + 1) or returns the finished one (`reused: true`).
With `--json`, a heartbeat line goes to stderr every 30 s so shells that kill
silent commands don't.

**Progress.** `--progress jsonl` writes events to stderr:
`{"v": 1, "event": "op" | "progress" | "log" | "step" | "heartbeat" | "interrupt", ...}`
(`interrupt` carries `count`: 1 = stopping at a safe point, 2 = stopping now).

**Verdict** (`data.verdict`): `outcome` is `better`, `worse` or
`no_clear_change`. A `no_clear_change` has a `reason`: `no_difference` (the
test could have seen a typical gain of about 0.05 and didn't) or
`too_few_test_documents` (it couldn't: the verdict is inconclusive).
`permutation_p` is an exact sign-flip p-value next to the t-test's
`p_value`, and with real searches `agreement` says whether they agree with
the generated test. Real searches decide the verdict once there are 30 or
more about 10+ documents, besides the style examples.

**Advice** (`data.advice` of `train`, `data.pipeline.advice`, and of
`train --dry-run`) suggests how to get
a better result: `code`, `severity`, `message`, `evidence`, `action`
(`command`, and `config` to set first), `applicability`.

| Advice code | When |
|---|---|
| `FEW_QUERIES` | Under 300 training queries |
| `CHUNKS_TRUNCATED` | Over 5% of chunks are longer than the model reads |
| `CHUNK_CHARS_DEPRECATED` | `dataset.chunk_chars` is set |
| `FEW_TRAINING_STEPS` | Training ran under 100 steps |
| `TEST_SET_UNDERPOWERED` | The verdict's 95% interval is wider than ±0.05; also in `train --dry-run` (`evidence.predicted`) when the test will have too few documents to see a typical gain |
| `TEST_SET_TOO_EASY` | The base model already scores 0.95+ on the test |
| `QUERIES_FILTERED_HEAVILY` | The query filter dropped over 40% of training queries |
| `MANY_UNMINED_NEGATIVES` | Over 20% of training queries had no safe hard negative |
| `LANG_REGRESSION` | One language got worse |
| `GENERAL_REGRESSION` | The general check got clearly worse (≥0.02) |
| `PREFIX_DROPPED_BY_TARGET` | The export target won't add the model's prefixes |
| `DISK_SPACE_TIGHT` | Less than 2 GB would be left on a drive the run writes to (or too little, with `SYNK_DISK_CHECK=warn`) |
| `REAL_QUERIES_FEW_DOCUMENTS` | Real searches exist but are under 30, or about under 10 documents, so they don't decide yet |
| `SYNTHETIC_REAL_DISAGREE` | Generated test queries and real searches reach different verdicts |

| Error code | Meaning |
|---|---|
| `INVALID_ARGS` | Unknown or conflicting flags, or an invalid value |
| `CONFIG_NOT_FOUND` / `CONFIG_EXISTS` / `CONFIG_INVALID` | No `synk.toml`; `init` in a project; a file or value that can't be parsed |
| `CONFIG_KEY_UNKNOWN` | `synk config` with a key synk doesn't define; `details.known` lists them |
| `SOURCE_MISSING` / `SOURCE_EXISTS` / `SOURCE_OVERLAP` / `SOURCE_NOT_FOUND` | Source path problems |
| `UNSUPPORTED_FORMAT` | A file type synk can't read |
| `IO` / `STORE` | Filesystem or database failure |
| `CONSENT_REQUIRED` | `train`/`dataset` would download or send without `--yes`; `details.fetch` lists the names for `synk fetch` |
| `JOB_RUNNING` / `JOB_NOT_FOUND` | A run, dataset build or fetch is already running in this project; nothing matches that ID |
| `CANCELLED` | It was cancelled (`synk cancel`, or Ctrl-C: exit code 130, `details.how` is `safe_stop` or `immediate`); `synk train <run>` continues it |
| `RUN_NAME_TAKEN` | `--name` is already used by another run |
| `RUN_NOT_RESUMABLE` | `train <run>` on a run without saved settings to continue with |
| `SIDECAR_NOT_FOUND` / `SIDECAR_PROTOCOL` / `SIDECAR_FAILED` / `SETUP_FAILED` | Trainer environment problems |
| `GENERATOR_UNAVAILABLE` / `REMOTE_NOT_ALLOWED` | The LLM can't be reached or lacks the model; a remote LLM without `allow_remote` |
| `MODEL_NOT_FOUND` / `MODEL_NOT_TRAINABLE` / `MODEL_ACCESS_DENIED` | No such model; an Ollama tag outside the catalog; a gated or private repo without access |
| `DOWNLOAD_FAILED` | A download failed after retries |
| `DEVICE_UNSUPPORTED` | No PyTorch build or device for the requested backend |
| `OUT_OF_MEMORY` | Training ran out of memory; `actions` suggest a smaller batch or LoRA |
| `INSUFFICIENT_DISK` | A drive the run writes to lacks space (a `train --dry-run` blocker with `details` per drive), or filled up during the run (`details.step`, `details.path`) |
| `DATASET_NOT_FOUND` / `DATASET_TOO_SMALL` | No dataset; too few documents or queries |
| `RUN_INCOMPLETE` / `RUN_NOT_FOUND` / `RUNS_NOT_COMPARABLE` | Run lookups and comparisons |
| `QUERIES_INVALID` | `queries import`: a bad line; nothing was imported |
| `ID_AMBIGUOUS` | An ID prefix matches several; `details.candidates` lists them |
| `EXPORT_EXISTS` / `EXPORT_CHECK_FAILED` / `EXPORT_UNSUPPORTED` | Export folder in use; vectors don't match; the format can't convert this model |

## Privacy

synk sends no telemetry. The only network calls it makes: package and model
downloads (with consent), model-size lookups on Hugging Face and the Ollama
registry (model names only), your LLM server (document chunks; remote only
with `allow_remote` and consent), and your local Ollama on export.

## Workspace

| Crate | Role |
|---|---|
| `synk` | CLI: commands, rendering, `schema`, `skill` |
| `synk-core` | Shared types, the `Connector` trait, error codes |
| `synk-connectors` | Sources and file readers (md, txt, html, docx, pdf) |
| `synk-store` | SQLite storage with incremental sync, tombstones and migrations |
| `synk-pipeline` | Steps, the preview graph, runs and fetches, model catalog, Hub access, the Python trainer and export |

## Development

```sh
cargo test                     # unit + CLI tests (fake trainer, no Python or LLM)
SYNK_E2E_PYTHON=/path/to/python cargo test -- --ignored   # trains a real model
```

## License

MIT. See [LICENSE](LICENSE).
