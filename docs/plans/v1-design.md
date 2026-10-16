# synk v1: the full design

Date: 2026-10-15. Status: **accepted** by the owner on 2026-10-15 (all decisions in §12 as recommended); being built, M1 first. It replaces the scope parts of
v1-improvements.md and cli-experience.md,
and turns agent-first-design.md into a buildable plan.

The owner asked for this: *"I don't want a simple version anymore. Plan a full-fledged feature set, but simple
enough for v1. You can start from scratch for everything."* This doc designs v1 as one product.
Today's code is reused only where it is already the best choice, and each reuse is named (§10, §11).

**Labels.**
- **[observed]**: I read it in this repo or ran it here (Windows 11, working tree on top of `79a1dfd`). `cargo test` passes with 74 tests and 1 ignored.
- **[unverified]**: I found no primary source.
- Every other external claim links to a primary source.

Versions checked on 2026-10-15:

| Component | Version |
|---|---|
| sentence-transformers | 6.1.0 |
| torch | 2.14.1 |
| huggingface_hub | 2.2.0 |
| peft | 0.21.2 |
| mteb | 2.25.0 (released 2026-10-15) |
| llama.cpp | `master` 620d222 |
| dist (cargo-dist) | v0.33.0 (2026-09-11) |
| crates | rusqlite 0.40.2, schemars 1.2.3, tokenizers 0.23.2, pdf-extract 0.12.1, htmd 0.5.6, docx-lite 0.2.0, zip 9.0.3, quick-xml 0.42.0 |

---

## 0. TL;DR

1. **What synk v1 does:** it fine-tunes a small multilingual embedding model on your own documents, locally, and proves with a statistical test whether the tuned model is better than the base model:
   - overall, per language, and on real queries;
   - with a general check that the model didn't get worse at ordinary search.

   It then exports the model as a sentence-transformers folder, ONNX, or GGUF registered in Ollama, and checks the exported vectors.
2. **One interface for agents and people: `config → plan → apply → job`.**
   - `synk.toml` is the config.
   - `synk plan` is read-only and states the downloads, the remote sends and the time.
   - `synk apply` runs the out-of-date steps up to the verdict, as a **job** with a durable record.
   - `synk export` is a separate command, so you can keep many runs and export the one you want.
3. **Consent is a plan ID or `--yes`.** There are no spending limits. `plan` never sends text to an LLM. A timing probe needs `--measure` and runs only against a local server.
4. **Long work: blocking `apply`, plus `job status/wait/logs/cancel`.** There is no `--detach` in v1, because agent harnesses already background long commands. A heartbeat keeps Gemini-style inactivity killers away.
5. **Data in:** Markdown, text, HTML, DOCX and PDF (text layer only, no OCR). Exact-duplicate removal and a split-leak guard are added. Connectors for Slack, Drive and Lark come later, behind a cursor seam that is added now.
6. **Training data:** today's recipe is kept (LLM queries, round-trip filter, 3 hard negatives, optional reranker, dev-epoch choice). Three things are added:
   - `queries_per_chunk = "auto"`;
   - few-shot style examples from real queries;
   - an incremental query cache, so raising `queries_per_chunk` only generates the missing queries (fixes review finding 7).
7. **Models:**
   - a v1 catalog of five models, chosen for Vietnamese/English;
   - base models from the catalog, any Hugging Face repo (including mirrors, private and gated repos) or a local folder;
   - every download is pinned to a commit;
   - LoRA (r=32, α=64) is chosen automatically for models of 300M+ parameters;
   - Matryoshka loss for models that support it, bf16/fp16 on GPUs, gradient checkpointing.
8. **Architecture stays Rust + Python sidecar** ([ADR 0001](../adr/0001-python-training-sidecar.md)). Protocol v2 makes the sidecar task-based (`describe`, `fetch`, `train`, `eval`, `export`) and adds a stop file for cancellation.
9. **Breaking the JSON contract once, now.** synk has never been released: there are no tags and no crates.io entry **[observed]**. v1 freezes a new contract with `"v": 1`, and the additive-only rule starts from there.
10. **Six milestones, each shippable** (§10). Roughly 55% of today's code is reused as is or with small changes. The rewritten parts are the CLI layer, the step orchestration and the sidecar protocol.

---

## 1. Product

**One sentence:** *give synk a folder of notes; it tells you, with evidence, whether a tuned search model beats the one you have, and hands you the better one in the format your RAG app uses.*

| Principle | Consequence |
|---|---|
| Local by default | No telemetry. The only network calls are the ones listed in §7, and each needs consent or sends only names |
| Honest results | A verdict other than `better` is a normal result. Every claimed gain carries a 95% interval, and only the deciding number is shown to people |
| Agent-first, expert-comfortable | Every command returns one JSON object and has stable codes. Humans see the same objects rendered as text. There are no prompts and no wizard |
| One way to do each thing | No aliases for the same action, and spending happens only in `apply`, `setup` and `export` |
| Reproducible | Content-addressed artifacts, pinned models and pinned Python packages, and dataset snapshots that hold their own text |

**Who it is for:** a developer or an AI agent working for one. Users need not know ML: advice codes say what to change. Experts get every knob in `synk.toml` and `-v`.

**Non-goals for v1:**
- serving embeddings or acting as a vector DB;
- an MCP server (cli-fit-for-synk.md §3);
- cloud training;
- hyperparameter search;
- a GUI.

---

## 2. Scope: IN v1 and LATER

| Area | IN v1 | LATER (trigger to revisit) | Why this cut |
|---|---|---|---|
| **A. Sources** | Folders and files of `.md`, `.txt`, `.html`, `.docx`, `.pdf` (text layer) | Code files (tree-sitter), CSV/XLSX, OCR, Slack, Drive, Lark, Notion | The owner's notes are documents. API connectors need OAuth and run into rate limits (feasibility §4) |
| A. Sync | Incremental sync, tombstones, exact duplicate removal, split-leak guard | Near-duplicate removal (MinHash) | Exact duplicates are cheap to remove and they leak test data |
| A. Language | Per-chunk detection (whatlang), reported per language | Mixed-language chunk splitting, `lingua` | Today's detector already handles vi and en **[observed]** in `lang.rs` tests |
| A. Chunking | Token-aware, heading-aware (heading path given to the generator) | Semantic chunking | Token chunking exists and fixed a measured truncation |
| **B. Queries** | OpenAI-compatible LLM, local by default; cross-lingual queries; `queries_per_chunk = "auto"`; few-shot from real queries; incremental cache | LLM-as-judge filter, GPL soft labels, Vietnamese no-diacritic augmentation | Each IN item has evidence or fixes a bug. The LATER items need measurement first |
| B. Filters and negatives | Round-trip filter, 3 hard negatives, optional reranker check, dedup | MarginMSE/GPL | Already built and evidence-backed (gap analysis §2) |
| B. Real queries | Import (eval-only), small style pool for few-shot | Click-log adapters | |
| **C. Models** | Catalog v1 (5 base, 2 rerankers); `hf:org/name@rev`; mirrors (`endpoint`); private/gated (token from env); local folders; fetch with progress, resume and pinning | `trust_remote_code` models, a remote catalog refresh | Running remote code is a security decision the owner should make later |
| C. Training | Full FT below 300M; LoRA r=32 α=64 at 300M+ (merged on export); Matryoshka for MRL models; bf16/fp16; gradient checkpointing; CachedMNRL above batch 32; OOM handling | QLoRA, Unsloth backend, model merge with base (LM-Cocktail), candle trainer | The IN items are what sentence-transformers and PEFT do natively. Unsloth adds a second dependency stack |
| C. GPUs | NVIDIA (CUDA 13.0), Apple (MPS), AMD on ROCm Linux for listed GPUs, CPU everywhere | ROCm on Windows | Same as today's README, no hacks |
| **D. Eval** | nDCG@10 verdict (paired t per document), per language, by query kind, real queries, **general check** (English NanoBEIR + Vietnamese XQuAD), `run inspect/compare`, **static HTML report** | BM25 baseline, scoring without prompts | The general check protects users. The HTML report is cheap because it reads `per_query.jsonl` |
| **E. Export** | ST folder; ONNX; GGUF (q8_0 default, f16 option) with Ollama registration; vector check; private model card with usage snippets | Pure-Rust GGUF writer, publishing to the HF Hub | GGUF reaches Ollama, Open WebUI and AnythingLLM (cli-fit §5.4) |
| **F. Interface** | `plan/apply/job`, plan IDs, `--yes`, `--set`, readable IDs, error `class/retryable/actions`, `synk schema`, `synk skill`, grouped help, `config get/set/unset/show` | `--detach`, `explain`, `--fields`, cursors, MCP, shell completions | Review finding 5: the LATER items add little once errors carry actions and the skill exists |
| **G. Architecture** | Sidecar protocol v2, job records, step graph, schemars snapshots, fake trainer for tests, cargo-dist installers | Plugin executables, WASM, a DAG engine | extensible-architecture.md |

---

## 3. Review findings on agent-first-design.md, resolved

| # | Finding | Decision | Rationale |
|---|---|---|---|
| 1 | `plan` timed the LLM on 3 real chunks, which sends text to a remote LLM before consent. Confirmed **[observed]**: `dataset::plan` calls the generator for up to `samples` uncached chunks, and only `allow_remote` gates it (`dataset.rs:336`, `generator.rs:64`) | **`plan` never generates.** It estimates from the throughput recorded by earlier runs (`$SYNK_HOME/throughput.json`, keyed by generator), or else from a per-token default, and marks the estimate `basis: "measured" \| "history" \| "default"`. `plan --measure` times up to 3 chunks, **only when the server is local**, and caches the queries it gets. Against a remote server, `--measure` returns `REMOTE_NOT_ALLOWED` | Agents re-plan after every config change, so plan must be cheap, deterministic and free of side effects. A cold local model also distorts a 3-call timing. This keeps the owner's lean (time only when local), but makes it opt-in |
| 2 | Export inside the apply graph conflicts with experiments (`notes-model`, `notes-model-2`) | **`export` is its own command.** The graph ends at `eval`. `export RUN --out DIR` writes once and fails with `EXPORT_EXISTS` if the folder is in use (unless `--force`) | An export is a choice between runs, not a step that follows from the config. Several runs can live side by side under readable IDs |
| 3 | User-level spending limits are complex for little security | **No limits in v1.** Consent is a plan ID (exact) or `--yes` ("whatever the plan is now"). Remote sends also need `generator.allow_remote = true` | Claude Code's docs say an easily routed rule "isn't a security boundary" ([permissions](https://code.claude.com/docs/en/permissions)). The real gate is the harness rule on `synk apply` |
| 4 | `--detach` is unverified on Windows; harnesses already background (Claude Code moves to background at the timeout, Codex polls, Gemini kills after 300 s of silence) | **No `--detach`.** `apply` blocks and writes a **durable job record**. `job show/wait/logs/cancel` work from any other shell. A heartbeat every 30 s on stderr with `--progress jsonl`, which is on by default with `--json` when stderr is not a terminal | cli-fit §1.1, §2.3: option C works in every harness. Revisit `--detach` if Codex's one-shot mode (killed after 10 s) turns out to matter |
| 5 | v1 scope risk: plan engine, job system, schema, explain, IDs and skill | **Keep:** plan, apply, job, IDs, schema (generated), skill. **Cut:** `explain`, `--fields`, cursors, `job list` filters, `status --watch` | `explain` duplicates the README tables and the skill's references. `--limit` covers output caps. Schema stays because schemars generates it and snapshot tests then guard the contract for free |
| 6 | Glossary overlap: spec vs config, job vs training run, plan; "Tune" is now dead | **One word each:** *Config* (not spec), *Plan*, *Job*, *Step*. A *Training run* stays distinct: a job runs zero or one training runs per base model. **Remove "Tune"** (§8) | One word per concept (cli-design-three-audiences TL;DR 6–7) |
| 7 | `queries_per_chunk` is part of the generator cache key, so raising it regenerates every query. Confirmed **[observed]**: `OpenAiGenerator::key` hashes `queries_per_chunk` (`generator.rs:227–237`) | **Make the cache incremental** (§4.2): rows keyed by `(chunk, generator identity, slot)`. Raising N asks only for the missing `N − have` queries, with "different from: …" in the prompt. Lowering N reuses a prefix. Bump `PROMPT_VERSION` to 3 and add a new migration | Turns "3→5" from hours into minutes and makes `"auto"` affordable |

---

## 4. Pipeline design

```
sources ─sync─▶ store (SQLite) ─dataset─▶ dataset artifact ─train─▶ run artifact ─eval─▶ verdict
   (A)              │                (B)                       (C)                  (D)
                    └ query cache (per chunk, per slot)                      export (E) on request
```

Step graph for `plan`/`apply`, in order:
- **Machine-level** (shared by all projects): `trainer` (Python env), `models` (base model, reranker, general-check data), `generator` (the LLM present on its server).
- **Per project:** `sync` → `dataset` → `train` → `eval`.

### 4.1 Data in (A)

| Item | Design | Source / evidence |
|---|---|---|
| Formats | `md/markdown/txt` (today); `html/htm` → Markdown via `htmd` 0.5.6 (Apache-2.0); `docx` text via `zip` + `quick-xml` (about 150 lines, own code), with `docx-lite` 0.2.0 as a fallback; `pdf` text layer via `pdf-extract` 0.12.1 (MIT, 6.4M downloads) | crates.io API, 2026-10-15. PDF tools in Python are worse fits: PyMuPDF is AGPL ([PyPI](https://pypi.org/project/pymupdf/)), and docling and markitdown are MIT but need the Python env before `sync` ([docling](https://pypi.org/project/docling/), [markitdown](https://pypi.org/project/markitdown/)) |
| PDF quality gate | Skip a PDF with a reason when it has under 50 extracted characters per page (probably scanned) or over 2% U+FFFD / control characters (broken font maps). It shows in `sync.skipped[]` as `{path, reason: "no_text_layer" \| "garbled_text"}` | Vietnamese PDFs with legacy font encodings may extract as mojibake **[unverified]**. Spike in M1 with 20 real vi PDFs; thresholds are **[unverified]** |
| Why the parsers are in Rust | `sync` must work before `synk setup`, and the trainer env is about 3 GB | ADR 0001 |
| Connector seam | `Connector::scan(&SourceState) -> SyncBatch`. `SourceState { known, cursor: Option<Value> }`, and `SourceSpec.type` defaults to `"folder"` | extensible-architecture §4. One-way door: add it before the first API connector |
| Document identity | Keep `<source>/<rel_path>`. Splits key on the source-relative path hash, which already ignores source names **[observed]** (`dataset.rs:515–557`) | |
| Dedup | (1) **Document:** identical `content_hash` across sources → keep the first, report `duplicates[]`. (2) **Chunk:** a normalized text hash (NFC, whitespace collapsed, lowercase). A chunk that appears in both splits is dropped from **test**, and the count is reported as `leaked_chunks_removed` | A duplicate in both splits inflates the verdict, and the test split must be clean (glossary: *Dev split*) |
| Language | whatlang per chunk at confidence ≥ 0.5 (today). Chunks below that are `und` and grouped as "other" in reports | **[observed]** `lang.rs:110` |
| Chunking | Token-aware with the base model's tokenizer (today). New: the heading path (`# A > ## B`) goes into the generator prompt beside the title. It is not added to the chunk text, so what gets embedded doesn't change | |

### 4.2 Training data (B)

| Item | Design | Evidence |
|---|---|---|
| Generator | One OpenAI-compatible client covers Ollama, llama.cpp `llama-server`, vLLM and LM Studio (today). JSON mode, `reasoning_effort: none`, seeded | **[observed]** `generator.rs:166–180` |
| Remote | Allowed only with `allow_remote = true` **and** consent in the plan. The plan's `consent[]` names the host and the chunk count | finding 1 |
| Ollama model missing | The `generator` step pulls it via `/api/pull` as a `download` consent item. Other servers get a `manual` action | Ollama pull resumes ([api.md](https://raw.githubusercontent.com/ollama/ollama/main/docs/api.md), via agent-first D8) |
| `queries_per_chunk = "auto"` (new default) | `clamp(ceil(1500 / train_chunks), 3, 8)`, plus one per other configured language | Small corpora want more queries per passage ([GPL Table 3](https://arxiv.org/abs/2112.07577)). The 1,500 target is **[unverified]**: measure it in M3 |
| Incremental cache | Table `query_slots(chunk_hash, gen_identity, slot, lang, text, created_at)`. `gen_identity` = prompt version + model + digest + language set, **without N**. Missing slots are generated with the existing queries shown as "do not repeat". Old `query_cache` rows are imported as slots 0..n−1 by the new migration | finding 7 |
| Few-shot from real queries | When real queries exist, up to 5 are sampled into a **style pool** that is excluded from evaluation and shown in the prompt as "examples of how this user searches". `PROMPT_VERSION` 3 | Promptagator: few-shot examples close most of the gap to supervised training ([2209.11755](https://arxiv.org/abs/2209.11755)). The pool is held out so the verdict isn't scored on the same queries |
| Cross-lingual | `[generator] languages = ["vi","en"]` (today): one extra query per other language | **[observed]** `generator.rs:135–162` |
| Filters | (1) Syntactic: length, copied sentences, in-chunk dedup (today). (2) Cross-dataset query dedup (today). (3) Round-trip: drop train queries whose chunk ranks below 20 for the base model (today) | Promptagator +2.5; InPars −4 without a filter (gap analysis §2) |
| Hard negatives | 3 per query, mined from other documents with a 95% relative margin. With a reranker set, candidates with p ≥ 0.5 or ≥ the positive's score are dropped (today). No random fallback | NV-Retriever, GPL (gap analysis) |
| Splits | Test = 20% of documents by path hash. Dev = training documents (needs 20+). Real queries are eval-only. The style pool is excluded from eval | **[observed]** |

### 4.3 Models (C)

**Base-model sources** (`[model] base = …`):

| Form | Example | Resolution |
|---|---|---|
| Catalog ID | `multilingual-e5-small` | The pinned commit from `catalog.toml` |
| HF repo | `hf:BAAI/bge-m3` or `hf:org/name@<commit>` | Without `@rev`, `plan` resolves `main` to a commit and records it in the plan and run manifest. Advice `MODEL_NOT_PINNED` (machine: `config set model.base hf:…@<sha>`) |
| Mirror | `[models] endpoint = "https://hf-mirror.example"` | Passed as `HF_ENDPOINT` to the sidecar and the tokenizer fetch. `HF_ENDPOINT` is a supported huggingface_hub setting (see `hf-hub` "Override the Hub base URL", [docs.rs](https://docs.rs/hf-hub/1.0.0/hf_hub/)). A mirror receives only model names |
| Private or gated | `[models] token_env = "HF_TOKEN"` (default `HF_TOKEN`) | Tokens are never stored in `synk.toml`. `HF_TOKEN` "overwrite[s] the token stored on the machine" ([env vars](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables)). A gated repo without access → `MODEL_ACCESS_DENIED` with a manual action ("accept the license on the model page") |
| Local folder | `./models/my-st-model` | Must be a sentence-transformers folder. Prompts and pooling come from its config |
| Ollama tag | `qwen3-embedding:0.6b` | Mapped through the catalog's `ollama` field, otherwise `MODEL_NOT_TRAINABLE` (today) |

**Fetcher.** The sidecar's `fetch` task calls `huggingface_hub.snapshot_download(repo, revision=<commit>, allow_patterns=[weights, configs, tokenizer, ST modules])`.
- **Cache and resume:** files go into the standard HF cache, so other tools reuse them. Interrupted downloads leave `.incomplete` blobs ([manage-cache](https://huggingface.co/docs/huggingface_hub/guides/manage-cache)).
- **Completeness:** an incomplete snapshot raises `IncompleteSnapshotError` instead of returning a partial folder (same page).
- **Progress:** a byte-count `progress` event about once per second.
- **Size in the plan:** `plan` gets sizes from the Hub API (metadata only) or from the catalog when offline.
- **Tokenizer:** it keeps today's Rust fetch, a single small file, so `sync`/`dataset` work before `setup`.

**Catalog v1** (`catalog_version = 1`, additive). Vietnamese coverage is checked against model-card metadata (HF API, 2026-10-15):

| ID | Params | vi evidence | Train mode | Notes |
|---|---|---|---|---|
| `multilingual-e5-small` (default) | 118M | `vi` in card language list (94 languages) | full, CPU OK | prompts `query: `/`passage: `; MIT |
| `multilingual-e5-base` | 278M | `vi` in list | full | MIT |
| `bge-m3` | 568M | MKQA vi Recall@100 76.6 (dense) in the [M3 paper, Table 2](https://arxiv.org/abs/2402.03216); card says "more than 100 working languages" ([card](https://huggingface.co/BAAI/bge-m3)) | LoRA | MIT, 8192 tokens, no query instruction needed. GGUF via the registered XLM-R converter (cli-fit §5.2) |
| `qwen3-embedding-0.6b` | 0.6B | "support for over 100 languages"; vi not named **[unverified for vi]** ([card](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)) | LoRA | Apache-2.0; MRL 32–1024; instruction prompt; transformers ≥ 4.51 |
| `embeddinggemma-300m` | 300M | "100+ spoken languages"; vi not named **[unverified for vi]** ([card](https://huggingface.co/google/embeddinggemma-300m)) | LoRA | **Gated**, Gemma license; MRL 768/512/256/128; "activations do not support `float16`", so it runs in bf16/fp32 only; GGUF needs `--sentence-transformers-dense-modules` |
| ~~`gte-multilingual-base`~~ | 305M | `vi` in its 75-language list | — | **Excluded:** needs `trust_remote_code=True` ([card](https://huggingface.co/Alibaba-NLP/gte-multilingual-base)). LATER, behind an explicit opt-in |

Rerankers stay `bge-reranker-v2-m3` and `mmarco-minilm-l12` (today). Catalog entries gain `finetune = "full" | "lora"`, `mrl_dims`, `dtype_gpu`, `gguf = { ok, flags }`, `min_vram_gb` (estimates, **[unverified]**) and `gated`.

**Training recipe:**

| Knob | Default | Source |
|---|---|---|
| Loss | MNRL; `CachedMNRL` above batch 32; wrapped in `MatryoshkaLoss(dims = catalog mrl_dims)` for MRL models | [ST losses](https://sbert.net/docs/package_reference/sentence_transformer/losses.html); keeping MRL in the loss protects truncation (feasibility §3) |
| `finetune = "auto"` | full below 300M; LoRA at 300M+ | LoRA ≈ full quality (0.4705 vs 0.4728 NanoBEIR) ([ST PEFT](https://sbert.net/examples/sentence_transformer/training/peft/README.html)) |
| LoRA | r = 32, α = 64, attention projections; merged on export | Unsloth's EmbeddingGemma and BGE-M3 notebooks use `r = 32`, `lora_alpha = 64` ([EmbeddingGemma nb](https://github.com/unslothai/notebooks/blob/main/nb/EmbeddingGemma_(300M).ipynb), [BGE-M3 nb](https://github.com/unslothai/notebooks/blob/main/nb/BGE_M3.ipynb), read 2026-10-15) |
| Precision | CUDA/ROCm: bf16 if supported, else fp16, except models flagged no-fp16 (EmbeddingGemma) → fp32. MPS and CPU: fp32 | The Unsloth notebooks use `bf16 = is_bf16_supported(), fp16 = not is_bf16_supported()` (same notebooks); [TrainingArguments](https://huggingface.co/docs/transformers/main_classes/trainer). bf16 on MPS **[unverified]**: `describe` probes it, fp32 until proven |
| Gradient checkpointing | On for LoRA models on GPU; off otherwise | The notebooks set it per model (`"unsloth"` for EmbeddingGemma, `False` for BGE-M3). Unsloth says it had to patch it in for some architectures ([guide](https://unsloth.ai/docs/basics/embedding-finetuning)), so support varies by model **[unverified per model]** |
| Epochs, lr, batch | `epochs = "auto"` (≥300 steps, ≤5); lr 2e-5 full / 1e-4 LoRA **[unverified]**; batch 16 CPU / 64 GPU | today + E5/BGE recipes (gap analysis) |
| OOM | Catch CUDA/MPS out-of-memory → `OUT_OF_MEMORY` with machine actions: halve `trainer.batch_size`, then `trainer.finetune = "lora"` | new |
| Device | auto from `describe` (today) | |

**GPU backends** (unchanged policy, [README](../../README.md), extensible-architecture §5):
- CUDA 13.0 wheels on Linux and Windows;
- MPS on macOS 14+;
- ROCm 7.2 on Linux for GPUs in AMD's list;
- everything else on CPU.

No `HSA_OVERRIDE_GFX_VERSION`. `describe` reports `{device, name, vram_bytes, bf16}`, and `plan` uses it to warn early (`MODEL_TOO_LARGE_FOR_DEVICE`, review).

### 4.4 Evaluation (D)

| Item | Design | Evidence |
|---|---|---|
| Metric that decides | nDCG@10 on the test split. Also recall@1/5/10 and MRR@10 | today |
| Verdict | Paired t on per-document mean Δ. `better` / `worse` / `no_clear_change` from the 95% interval. Also per language and per query kind (synthetic, real) | today; [Urbano et al. 2019](https://arxiv.org/abs/1905.11096) |
| Real queries | 30+ → they decide the verdict (today) | |
| **General check** (new) | Base and tuned are scored on fixed public sets: **NanoBEIR** (English, via ST's `NanoBEIREvaluator`, [ST eval](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html)) and **XQuADRetrieval vi** (CC-BY-SA-4.0 per [MTEB's task file](https://github.com/embeddings-benchmark/mteb/blob/main/mteb/tasks/retrieval/multilingual/x_qu_ad_retrieval.py)). Sets are downloaded on consent and never bundled (license). Reported as `general_check: {en, vi}`, each with Δ and CI | Fine-tuning lowered other tasks 49.81 → 47.49 in [LM-Cocktail Table 2](https://arxiv.org/abs/2311.13534). NanoBEIR has no vi (gap analysis §5). MTEB also has larger vi sets (VN-MTEB, e.g. `SciFact-VN`, CC-BY-SA-4.0, [task](https://github.com/embeddings-benchmark/mteb/blob/main/mteb/tasks/retrieval/vie/sci_fact_vn_retrieval.py)) for LATER |
| General regression | Advice `GENERAL_REGRESSION` (review) when a general-check CI is entirely below 0 **and** Δ < −0.02 (threshold **[unverified]**). It doesn't change the verdict; `export` repeats it as a warning | Domain gain and general loss are separate facts |
| Inspect, compare | `run inspect` (biggest gains and losses with texts, `--limit`). `run compare A B`: paired **by query** when both runs used the same dataset, otherwise **by test document** (`paired_by` field). The split depends only on paths, so two base models with different chunkings share test documents **[observed]** | today + the experiment flow in transcript 2 |
| HTML report | `run report RUN --out report.html`: one static file, no JS and no network. It holds the verdict, intervals, per-language table, general check, and the 10 biggest losses and gains. Contains document text, so it is written only where asked | clean-slate-ux suggested a browser report for results |
| Advice added | `GENERAL_REGRESSION`, `MODEL_NOT_PINNED`, `PREFIX_DROPPED_BY_TARGET`, `OUT_OF_MEMORY_RETRY` (machine), `BIGGER_MODEL_FITS` (info: a GPU is present and the default model is CPU-sized) | |

### 4.5 Export and use (E)

`synk export [RUN] --out DIR [--format st|onnx|gguf] [--quant q8_0|f16] [--ollama NAME] [--force]`

| Format | How | Checked by | Notes |
|---|---|---|---|
| `st` (default) | Copy the run's model; LoRA is merged first (`merge_and_unload`) | The model loads and embeds 32 dev chunks identically (cos ≥ 0.9999) | Canonical and lossless. Open WebUI loads it by path (cli-fit §5.4) |
| `onnx` | ST `backend="onnx"` + `save_pretrained`; `sentence-transformers[onnx]` in the env | onnxruntime with pooling + normalize vs ST, cos ≥ 0.999 | ST's ONNX "only converts the Transformer component", so pooling must be re-applied by the user ([efficiency](https://sbert.net/docs/sentence_transformer/usage/efficiency.html)). The model card says so, with code |
| `gguf` | llama.cpp `convert_hf_to_gguf.py` vendored at a pinned commit, `--outtype q8_0` by default (choices include `q8_0`, `f16`, `bf16`, [source](https://github.com/ggml-org/llama.cpp/blob/master/convert_hf_to_gguf.py)). Two fixes are applied: the e5-small relabel + position-offset fix (measured cos 0.99999) and `--sentence-transformers-dense-modules` for EmbeddingGemma | With `--ollama NAME`: `ollama create`, then `/api/embed` vs ST on 32 dev chunks; cos ≥ 0.995 for q8_0 and ≥ 0.999 for f16 (q8_0 threshold **[unverified]**). Without Ollama: `check: "not_run"` plus advice | q8_0 is Unsloth's default ("we default save it to `q8_0`", EmbeddingGemma notebook). Prefixes are lost in Ollama (cli-fit §5.3), so the card prints the exact prefix and advice `PREFIX_DROPPED_BY_TARGET` is added |
| Model card | `README.md`: base, settings, verdict, general check, prompts, usage snippets for ST, Ollama `/api/embed`, LangChain/LlamaIndex via ST. **No document text** | — | Today's private card **[observed]** `train.rs:625` |

Export writes into `.synk/artifacts/export/<key>/`, keyed by (run key, format, quant, converter commit), and then copies it to `--out`. Exporting the same run again is instant.

---

## 5. Interface

### 5.1 Rules

1. **stdout is exactly one JSON object with `--json`**: `{"v":1,"ok":true,"data":…}` or `{"v":1,"ok":false,"error":…}`. Exit codes: 0 when `ok`, 2 for usage errors, 1 otherwise.
2. **Spending verbs:** only `apply`, `setup` and `export` download, send text out, or use the GPU for long. Everything else is read-only or edits config, so harness rules work: allow `synk status|plan|job|run|model|schema *`, ask on `synk apply *`.
3. **Consent:** an `apply` whose plan has `consent[]` needs `plan:<id>` or `--yes`. A plan with no consent items runs without either.
4. **Plans are deterministic:** `plan:<hash of (config hash, input fingerprints, env fingerprint, steps)>`. `apply plan:X` recomputes the plan.
   - If anything changed, the result is `PLAN_STALE`, with the fresh plan and `consent_unchanged`.
   - Plans are stored in `.synk/plans/` for 30 days.
5. **`--set KEY=VALUE` on `plan`/`apply`** overrides config for one plan without editing `synk.toml`. It is recorded in the plan and in the run manifest, and is what experiments use.
6. **IDs:** `<kind>:<label>-<hash4+>`, e.g. `run:2026-10-15-e5small-3f2a`. Any unique prefix, the bare hash, `latest` or `best` resolve to an ID. Several matches give `ID_AMBIGUOUS` with candidates.
7. **Errors** carry `code`, `class`, `retryable`, `message`, `next` (always runnable as is, or absent), `actions[]` (`argv` or `config`, plus `applicability` and `why`) and `details`.
8. **Advice** keeps today's shape (`code`, `severity`, `message`, `evidence`, `action`, `applicability`).
9. **No prompts, ever.** Humans see the same objects as text. `-v` adds IDs, metric names and codes. Error codes always show.
10. **Progress:** `--progress auto|jsonl|none`. Events go to stderr as `{"v":1,"event":"progress|log|heartbeat|step", …}`. A heartbeat every 30 s while a step runs.

### 5.2 Command tree

```
synk [--json] [--progress auto|jsonl|none] [-v] [--config PATH]
│
├─ Work ─────────────────────────────────────────────────────────────────────────────
│  status                                   config, conditions, steps, running job, latest run, next
│  plan  [--target STEP] [--force STEP] [--set K=V]… [--sample N] [--measure]
│  apply [plan:ID] [--yes] [--target STEP] [--force STEP] [--set K=V]… [--sample N]
│  job   show [JOB] | wait [JOB] [--timeout 9m] | logs [JOB] [--after SEQ] [--limit N] | cancel [JOB] | list [--limit N]
│
├─ Results ──────────────────────────────────────────────────────────────────────────
│  run   list [--limit N] | show [RUN] | inspect [RUN] [--limit N] | compare A B | report [RUN] --out FILE
│  export [RUN] --out DIR [--format st|onnx|gguf] [--quant q8_0|f16] [--ollama NAME] [--force]
│
├─ Config ───────────────────────────────────────────────────────────────────────────
│  init [DIR]
│  source add PATH… [--name N] [--exclude GLOB]… | remove NAME | list
│  config show | get KEY | set KEY VALUE | unset KEY
│  queries import FILE [--replace] | list [--limit N]
│
├─ Machine ──────────────────────────────────────────────────────────────────────────
│  setup [--backend auto|cpu|cuda|rocm|mps] [--force]     the Python env (consent by invocation)
│  model list [--kind base|reranker] | show MODEL
│
└─ Reference ────────────────────────────────────────────────────────────────────────
   schema [COMMAND]                         JSON Schema 2020-12 of inputs, data, events, codes
   skill [--out DIR]                        prints SKILL.md, or writes SKILL.md + references/
```

STEP ∈ `trainer, models, generator, sync, dataset, train, eval`. `--target eval` is the default. `--sample N` builds the dataset from N chunks for a quick run (today's `--limit`).

Removed relative to today **[observed]** in `main.rs`:
- `add`, `remove`, `sources` → `source …`
- `sync`, `dataset`, `train` → `apply --target …`
- `runs`, `inspect`, `compare` → `run …`
- `models` → `model list`
- the `--plan` flags → `plan`
- `--resume` → resuming is automatic
- `sync --source` (try without a project) → LATER

### 5.3 JSON sketches (`data` only, trimmed)

```jsonc
// synk status
{"config": {"path": "synk.toml", "hash": "a91f"},
 "conditions": [{"type": "TrainerInstalled", "status": "True", "reason": "Found", "message": "cuda, torch 2.14.1"},
                {"type": "GeneratorReachable", "status": "False", "reason": "ConnectionRefused", "message": "http://localhost:11434"}],
 "steps": [{"step": "sync", "state": "current"}, {"step": "dataset", "state": "stale", "why": ["3 documents changed"]}],
 "job": null,
 "latest_run": {"id": "run:2026-10-15-e5small-3f2a", "outcome": "better"},
 "advice": [], "next": "synk plan"}

// synk plan
{"id": "plan:7c1e9a", "config_hash": "a91f", "overrides": {},
 "checks": [{"condition": "DiskSpace", "status": "True", "message": "41 GB free, needs 9.1 GB"}],
 "steps": [{"step": "trainer", "action": "run", "why": ["not installed"], "est_seconds": 240},
           {"step": "models", "action": "run", "why": ["multilingual-e5-small@614241f not cached", "general check data not cached"]},
           {"step": "dataset", "action": "run", "detail": {"chunks": 127, "queries_to_generate": 508, "queries_cached": 0},
            "est_seconds": 1500, "estimate_basis": "default"},
           {"step": "train", "action": "run", "detail": {"device": "cuda", "finetune": "full", "precision": "bf16", "steps": 320}},
           {"step": "eval", "action": "run"}],
 "consent": [{"kind": "download", "what": "trainer", "bytes": 3100000000, "from": "pypi.org, download.pytorch.org"},
             {"kind": "download", "what": "qwen3:8b", "bytes": 5200000000, "from": "ollama"},
             {"kind": "download", "what": "multilingual-e5-small", "bytes": 470000000, "from": "huggingface.co"},
             {"kind": "download", "what": "general-check", "bytes": 9000000, "from": "huggingface.co"}],
 "totals": {"download_bytes": 8779000000, "est_seconds": 2800, "disk_bytes": 9100000000},
 "egress": [],
 "next": "synk apply plan:7c1e9a"}

// synk apply plan:7c1e9a  → the finished job (blocking)
{"job": {"id": "job:2026-10-15-1412-7c1e", "plan": "plan:7c1e9a", "state": "succeeded", "done": true,
  "steps": [{"step": "trainer", "status": "ran", "seconds": 212}, {"step": "sync", "status": "cached"}],
  "result": {"run": "run:2026-10-15-e5small-3f2a", "dataset": "dataset:2026-10-15-qwen3-8b-b418",
    "verdict": {"outcome": "better", "metric": "ndcg@10", "base": 0.641, "tuned": 0.778, "ci95": [0.081, 0.193],
                "by_lang": {"vi": {"outcome": "better"}, "en": {"outcome": "better"}}, "query_kind": "synthetic"},
    "general_check": {"en": {"delta": -0.004, "ci95": [-0.015, 0.007]}, "vi": {"delta": 0.003, "ci95": [-0.01, 0.016]}},
    "advice": []}},
 "next": "synk export run:2026-10-15-e5small-3f2a --out ./notes-model"}

// synk job wait --timeout 9m   (not finished: ok:true, done:false)
{"id": "job:…-7c1e", "state": "running", "done": false, "heartbeat_at": "2026-10-15T14:21:01Z",
 "progress": {"step": "dataset", "completed": 96, "total": 508, "unit": "queries", "eta_seconds": 1130},
 "next": "synk job wait job:2026-10-15-1412-7c1e --timeout 9m"}

// synk export run:…-3f2a --out ./notes-model --format gguf --ollama notes-e5
{"run": "run:2026-10-15-e5small-3f2a", "path": "notes-model", "format": "gguf", "quant": "q8_0",
 "ollama": {"name": "notes-e5", "created": true},
 "check": {"status": "passed", "min_cosine": 0.9987, "texts": 32},
 "usage": {"query_prefix": "query: ", "document_prefix": "passage: "},
 "advice": [{"code": "PREFIX_DROPPED_BY_TARGET", "severity": "warning", "applicability": "manual", "…": "…"}]}

// error
{"v": 1, "ok": false, "error": {"code": "PLAN_STALE", "class": "conflict", "retryable": true,
  "message": "inputs changed since plan:d20b44: 2 documents changed in source notes",
  "next": "synk apply plan:5e0a17",
  "details": {"changed": ["documents: notes ~2"], "consent_unchanged": true, "plan": {"id": "plan:5e0a17", "…": "…"}}}}
```

### 5.4 Jobs

| Aspect | Design |
|---|---|
| Record | `.synk/jobs/<id>/job.json` (state, plan, steps, progress, heartbeat_at, pid, result or error) and `events.jsonl` (every progress/log event, with `seq`) |
| States | `running` → `succeeded \| failed \| cancelled \| interrupted`. `done` = not running ([AIP-151](https://google.aip.dev/151)) |
| Lock | `.synk/lock` = job ID. Another `apply` gets `JOB_RUNNING` with `next: synk job wait <id>` |
| Liveness | `heartbeat_at` is written every 10 s. No heartbeat for 2 min → `interrupted`, and the lock is released. Uses no OS-specific process APIs |
| Wait | `job wait --timeout` (default 9m, under Claude Code's 10-min cap, cli-fit §1.1). A timeout returns `ok:true, done:false` |
| Cancel | `job cancel` writes `.synk/jobs/<id>/cancel`. The running `apply` checks it between LLM calls and forwards it as `stop` into the sidecar's work dir. The sidecar's trainer callback checks `stop` each step, saves a checkpoint and exits. Ctrl-C does the same in-process. Works the same on every OS |
| Resume | `apply` again (same or fresh plan). Finished steps are `cached`, cached queries are reused, and `train` continues from its checkpoint |
| Retention | Job records 30 days (AIP-151 rule of thumb); artifacts are kept until a later `gc` |

### 5.5 Errors: new and changed codes

| Code | Class | When |
|---|---|---|
| `CONSENT_REQUIRED` | consent | `apply` without `plan:ID`/`--yes` and the plan needs consent; `details.plan` |
| `PLAN_STALE` / `PLAN_NOT_FOUND` | conflict / not_found | finding 3, rule 4 |
| `JOB_RUNNING` / `JOB_NOT_FOUND` | conflict / not_found | §5.4 |
| `ID_AMBIGUOUS` | usage | rule 6 |
| `INSUFFICIENT_DISK` | resource | plan check or mid-job |
| `OUT_OF_MEMORY` | resource | GPU/CPU OOM in the trainer; actions: batch size, LoRA |
| `MODEL_ACCESS_DENIED` | prerequisite | gated or private repo without a token or access |
| `DOWNLOAD_FAILED` | unavailable | network error after retries; retryable |
| `EXPORT_EXISTS` | conflict | `--out` not empty and no `--force` |
| `EXPORT_CHECK_FAILED` | failed | exported vectors differ from ST; action: `--quant f16` |
| `CONFIG_KEY_UNKNOWN` | config | `config set` with an unknown key |
| `CANCELLED` | failed | the job error when cancelled |

Kept from today **[observed]** in `error.rs`: all 26 codes except `RUN_INCOMPLETE`, which is dropped because resuming is automatic.

### 5.6 Discoverability

- **`synk schema [CMD]`:** JSON Schema 2020-12 generated with `schemars` 1.2.3 and snapshot-tested, so a removed field fails CI (extensible-architecture §2).
- **`synk skill`:** prints an embedded `SKILL.md` of about 150 lines, plus `references/codes.md` and `references/contract.md`. It covers the plan → ask the user → apply → wait loop, the harness permission rules from rule 2, and branches on code and `class`. The skill format loads only its name and description until used ([agentskills.io spec](https://agentskills.io/specification)).
- **Help:** a hand-written `help_template` with the five groups of §5.2 (clap has a single subcommand heading, e2e-cli-ux TL;DR 7), with 3–5 examples per subcommand. A test keeps the template in sync with clap.

### 5.7 Human rendering (same objects)

```
$ synk plan
plan:7c1e9a                                                     (nothing runs until: synk apply plan:7c1e9a)
  run     trainer      Python training environment     download 3.1 GB        ~4 min
  run     generator    qwen3:8b on Ollama              download 5.2 GB        ~5 min
  run     models       multilingual-e5-small + checks  download 0.5 GB        ~1 min
  cached  sync         42 documents
  run     dataset      127 chunks → ~508 queries                              ~25 min (rough: no history)
  run     train        full fine-tune on cuda (bf16)                          ~11 min
  run     eval         test split + general check                             ~2 min
Downloads 8.8 GB. Nothing leaves this machine. Total about 48 min.
```

`apply` prints one line per step and a live bar only on a TTY (not under `TERM=dumb`), then the verdict sentence, the general check and `next`.

### 5.8 Config editing

`config set` goes through `Project` (`toml_edit`), so user comments survive (AGENTS.md rule). It returns `{key, old, new, affects: ["dataset","train","eval"], next: "synk plan"}`. `affects` comes from comparing the step keys before and after. Agents may also edit `synk.toml` directly: `plan` hashes the parsed config. Precedence: `--set` > `synk.toml` > defaults. There are no `SYNK_*` overrides except `SYNK_HOME`.

### 5.9 Contract versioning

`"v": 1` on every stdout object and every event. Within v1, fields, codes, advice and `applicability` values may only be added, and consumers must ignore unknown ones (Terraform/LSP rule, extensible-architecture §2). A breaking change means `v: 2` plus `--json-version 1` support for one major release. File contracts keep their own versions: sidecar `PROTOCOL_VERSION`, dataset rows `v`, manifest `schema`, `catalog_version`.

---

## 6. Architecture (G)

### 6.1 Crates

| Crate | v1 role | Change |
|---|---|---|
| `synk-core` | Types, `Connector` + `SourceState`, `ErrorCode` (`#[non_exhaustive]`) + `ErrorClass`, IDs, sidecar protocol types | extend |
| `synk-connectors` | `folder` connector + format readers (`md`, `txt`, `html`, `docx`, `pdf`) behind a `Reader` trait chosen by extension | extend |
| `synk-store` | SQLite: documents, `query_slots`, real queries, style pool; append-only `MIGRATIONS` | extend (migrations 4–5) |
| `synk-pipeline` | Steps (`sync`, `dataset`, `train`, `eval`) as pure functions over artifacts; `graph.rs` (step keys, staleness, plan); `jobs.rs`; `fetch.rs`; `export.rs`; `general.rs`; `report_html.rs`; advice; catalog; sidecar | restructure: step functions kept, orchestration new |
| `synk` (CLI) | clap tree, renderers (`human.rs`, `output.rs`), `schema`, `skill`; **no pipeline logic** | rewrite of the command layer |

### 6.2 Sidecar protocol v2

| Element | v2 |
|---|---|
| Invocation | `python synk_sidecar.py run <work_dir>/request.json`. Script embedded with `include_str!` (today) |
| Request | `{"v":2, "task": "describe\|fetch\|train\|eval\|export", …}` |
| Events (stdout JSONL) | `progress {stage, completed, total, unit}`, `log`, `metric`, `result`, `error {code, message, details}` |
| Stop | The sidecar checks `<work_dir>/stop` each step / batch / file and exits after a checkpoint with `error {code: "CANCELLED"}` |
| `describe` | protocol versions, tasks, packages, devices `{type, name, vram_bytes, bf16}`, extras (`onnx`, `gguf`) |
| `fetch` | `snapshot_download` with byte progress; returns `{path, commit, bytes}` |
| `train` | today's train + `finetune`, `lora{r, alpha, targets}`, `precision`, `grad_checkpointing`, `mrl_dims`, `loss` |
| `eval` | today's eval + `general_sets[]` |
| `export` | `{format, quant, out, check_texts[]}` → `{path, check}` |
| Env | `torch==2.14.1`, `sentence-transformers[train,onnx]==6.1.0`, `peft==0.21.2`, `gguf==0.19.0` + vendored converter (pinned llama.cpp commit + 2 patches); `EXCLUDE_NEWER` pin (today) |

### 6.3 Storage layout

```
synk.toml                              config (only edited through Project/toml_edit)
.synk/synk.db                          documents, query_slots, real_queries, style_pool
.synk/artifacts/<kind>/<key>/          dataset | train | eval | export; manifest.json {schema, kind, key, inputs, params}
.synk/plans/<id>.json                  30 days
.synk/jobs/<id>/{job.json,events.jsonl,cancel}
.synk/lock
$SYNK_HOME/envs/<hash>/                Python envs, shared by projects
$SYNK_HOME/throughput.json             generator speed history (estimates only)
$HF_HOME/hub                           model weights and check sets (shared with other HF tools)
```

Keys are blake3 over (kind, KIND_VERSION, canonical params, input keys), as today **[observed]** in `artifact.rs:23`. The `generator` and `models` steps have no artifact: their state is checked live (Ollama `/api/tags`, the HF cache).

### 6.4 Extension points

| Seam | Mechanism | Who extends |
|---|---|---|
| Connectors | `Connector` trait + registry on `SourceSpec.type` (compiled-in) | Rust PRs |
| Format readers | `Reader` trait by extension | Rust PRs |
| Query generator | `QueryGenerator` trait; one real impl (OpenAI-compatible) + a fake | — |
| Trainer | the protocol v2 file contract (a candle trainer would implement the same tasks) | LATER |
| Exporters | `ExportFormat` enum → sidecar `export` task | — |
| Models | `catalog.toml` data | data PRs |

External plugin executables and WASM stay deferred (extensible-architecture: revisit when outside authors ask).

### 6.5 Testing

| Layer | What | How |
|---|---|---|
| Unit | chunking, readers, dedup, stats, keys, staleness, ID resolution | `cargo test` |
| Contract | every command's JSON (success + main errors) | `tests/cli.rs` + golden files; `synk schema` snapshot. A removed field fails |
| Graph | plan/apply/stale/cancel/resume without Python or an LLM | **fake trainer** (`SYNK_FAKE_TRAINER=1` → a Rust impl of protocol v2 that writes deterministic ranks) and a fake OpenAI server (a local `TcpListener` stub) |
| Parsers | 1 fixture per format, incl. Vietnamese DOCX/PDF/HTML, a scanned PDF and a broken-font PDF | fixtures in `tests/data` |
| E2E (ignored) | real training on CPU with e5-small on a 30-doc vi/en fixture; export to all 3 formats and check | `SYNK_E2E_PYTHON=… cargo test -- --ignored` (today's mechanism) |
| GPU matrix | CUDA (owner's machine), MPS, ROCm: a manual checklist per release | no GPU CI **[unverified availability]** |

### 6.6 Packaging and distribution

- **Binaries:** dist (cargo-dist) v0.33.0 generates shell, PowerShell, Homebrew and MSI installers from one config ([dist installers](https://axodotdev.github.io/cargo-dist/book/installers/index.html), via e2e-cli-ux TL;DR 8). `cargo binstall` picks the binaries up.
- **Python env:** `synk setup` (or the plan's `trainer` step) uses `uv`, which the user installs. If `uv` is missing, `SIDECAR_NOT_FOUND` names the official installer. The env is shared through `$SYNK_HOME/envs/<hash>` (today).
- **Telemetry:** none, and huggingface_hub's own telemetry is turned off by passing `HF_HUB_DISABLE_TELEMETRY=1` to the sidecar ([env vars](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables)).

---

## 7. Network calls synk may make (complete list)

| Call | When | Sends | Gate |
|---|---|---|---|
| PyPI / download.pytorch.org (via uv) | `trainer` step, `setup` | package names | consent (download) |
| huggingface.co or the mirror | `models` step, tokenizer fetch | model / dataset names, token if set | consent (download); the tokenizer is exempt as today (small, ADR 0002) |
| Hub metadata | `plan` (sizes, resolving `main`) | model names | none: read-only metadata **[decision for owner #8]** |
| Ollama `/api/tags`, `/api/pull` | plan checks; `generator` step | model tag | pull: consent (download) |
| Generator `/chat/completions` | `dataset` step, `plan --measure` (local only) | **chunk text** | local by default; remote: `allow_remote` + consent (egress) |
| Ollama `/api/create`, `/api/embed` | `export --ollama` | the model file, 32 dev chunks | local only |

---

## 8. Glossary update (GLOSSARY.md)

| Change | Term | Definition (proposed) | _Avoid_ |
|---|---|---|---|
| **remove** | Tune | — (no all-in-one command; `apply` replaces it) | — |
| add | **Config** | `synk.toml`: what the user wants (sources, models, settings). The desired state that plans are made from | spec, settings file |
| add | **Step** | One stage of the pipeline that `plan` can mark current or stale: trainer, models, generator, sync, dataset, train, eval | stage, task |
| add | **Plan** | What `synk apply` would do now: the stale steps, why, the downloads, the remote sends and the time. Identified by a hash of its inputs; a stale plan can't be applied | dry run, preview |
| add | **Consent** | Approval to download or send text off the machine, given by applying a plan ID or with `--yes` | confirmation, approval |
| add | **Job** | One execution of `synk apply`, with a durable record that can be shown, waited on and cancelled. A job runs zero or one training runs | task, operation |
| add | **General check** | Evaluating the base and tuned models on public datasets, to see whether tuning hurt ordinary search | forgetting test, benchmark |
| add | **Export** | Writing a tuned model in a format other programs load (sentence-transformers folder, ONNX, GGUF) and checking its vectors | publish, save |
| add | **Style pool** | A few real queries shown to the query generator as examples, and therefore left out of evaluation | few-shot set |
| keep | Training run | (unchanged) — add: "Several training runs can coexist; `export` picks one." | job (now its own term) |
| keep | all others | unchanged | |

---

## 9. End-to-end agent transcripts

`$` lines are what the agent runs. JSON is trimmed. Sizes and times are illustrative.

### 9.1 First run on a new machine

```
$ synk init --json && synk source add ./notes --json
{"v":1,"ok":true,"data":{"config":"synk.toml"}}            # no runnable next without a path, so `next` is omitted
{"v":1,"ok":true,"data":{"added":[{"id":"source:notes","path":"notes","documents_found":42}],"next":"synk plan"}}

$ synk plan --json
{"v":1,"ok":true,"data":{"id":"plan:7c1e9a",
  "checks":[{"condition":"GeneratorReachable","status":"True","reason":"Ollama"},
            {"condition":"TrainerInstalled","status":"False","reason":"Missing"}],
  "steps":[…trainer run, generator run (qwen3:8b), models run, sync run, dataset run, train run, eval run…],
  "consent":[{"kind":"download","what":"trainer","bytes":3100000000},
             {"kind":"download","what":"qwen3:8b","bytes":5200000000,"from":"ollama"},
             {"kind":"download","what":"multilingual-e5-small","bytes":470000000},
             {"kind":"download","what":"general-check","bytes":9000000}],
  "egress":[],"totals":{"download_bytes":8779000000,"est_seconds":2800},
  "next":"synk apply plan:7c1e9a"}}
```

The agent asks: *"synk needs 8.8 GB of downloads and about 47 minutes. Your notes stay on this machine. OK?"*. The user says yes, and the harness's `ask` rule on `synk apply` shows the command.

```
$ synk apply plan:7c1e9a --json --progress jsonl          # Claude Code moves it to the background after 2 min
stderr: {"v":1,"event":"step","step":"trainer","status":"running"} … {"v":1,"event":"heartbeat"} …

$ synk job wait --timeout 9m --json                        # from the agent's next turn
{"v":1,"ok":true,"data":{"id":"job:2026-10-15-1412-7c1e","state":"running","done":false,
  "progress":{"step":"dataset","completed":96,"total":508,"unit":"queries","eta_seconds":1130},
  "next":"synk job wait job:2026-10-15-1412-7c1e --timeout 9m"}}
  … repeated …
$ synk job wait --timeout 9m --json
{"v":1,"ok":true,"data":{"state":"succeeded","done":true,"result":{"run":"run:2026-10-15-e5small-3f2a",
  "verdict":{"outcome":"better","base":0.641,"tuned":0.778,"ci95":[0.081,0.193]},
  "general_check":{"en":{"delta":-0.004},"vi":{"delta":0.003}},"advice":[]},
  "next":"synk export run:2026-10-15-e5small-3f2a --out ./notes-model"}}

$ synk export latest --out ./notes-model --json
{"v":1,"ok":true,"data":{"path":"notes-model","format":"st","check":{"status":"passed","min_cosine":1.0},
  "usage":{"query_prefix":"query: ","document_prefix":"passage: "}}}
```

### 9.2 Experiment with two base models, then export the better one as GGUF for Ollama

```
$ synk plan --set model.base=multilingual-e5-base --json
{"v":1,"ok":true,"data":{"id":"plan:41b0d2","overrides":{"model.base":"multilingual-e5-base"},
  "steps":[{"step":"models","action":"run","why":["multilingual-e5-base@d128750 not cached"]},
           {"step":"dataset","action":"run","why":["tokenizer changed: chunks re-cut"],
            "detail":{"chunks":131,"queries_to_generate":61,"queries_cached":463}},     # cache hits per chunk text
           {"step":"train","action":"run","detail":{"finetune":"full","precision":"bf16"}},{"step":"eval","action":"run"}],
  "consent":[{"kind":"download","what":"multilingual-e5-base","bytes":1110000000}],
  "next":"synk apply plan:41b0d2"}}

$ synk apply plan:41b0d2 --json        # (the user approved 1.1 GB)
… → {"result":{"run":"run:2026-10-15-e5base-9d10","verdict":{"outcome":"better","base":0.668,"tuned":0.812}}}

$ synk run compare run:2026-10-15-e5small-3f2a run:2026-10-15-e5base-9d10 --json
{"v":1,"ok":true,"data":{"a":"run:…-3f2a","b":"run:…-9d10","paired_by":"document","test_documents":9,
  "outcome":"no_clear_change","delta":0.034,"ci95":[-0.012,0.080],
  "advice":[{"code":"TEST_SET_UNDERPOWERED","applicability":"manual","message":"9 test documents; import real queries or add documents to tell these apart."}]}}
```

The agent reports that the bigger model isn't clearly better on 9 test documents. The user picks the small one, because it is faster on their RAG box.

```
$ synk export run:2026-10-15-e5small-3f2a --out ./notes-e5-gguf --format gguf --ollama notes-e5 --json
{"v":1,"ok":true,"data":{"format":"gguf","quant":"q8_0","ollama":{"name":"notes-e5","created":true},
  "check":{"status":"passed","min_cosine":0.9987,"texts":32},
  "usage":{"query_prefix":"query: ","document_prefix":"passage: "},
  "advice":[{"code":"PREFIX_DROPPED_BY_TARGET","severity":"warning","applicability":"manual",
    "message":"Ollama does not add \"query: \"/\"passage: \". Add them in your app, or search quality drops."}]}}
```

`synk.toml` was never edited. Both runs stay listed in `synk run list`.

### 9.3 Failure and recovery

```
$ synk apply --json
{"v":1,"ok":false,"error":{"code":"CONSENT_REQUIRED","class":"consent","retryable":false,
  "message":"this plan downloads 1.1 GB (bge-reranker-v2-m3)","next":"synk apply plan:a3f19c",
  "details":{"plan":{"id":"plan:a3f19c","consent":[{"kind":"download","what":"bge-reranker-v2-m3","bytes":1140000000}]}}}}
   (the agent asks the user, who says yes)
$ synk apply plan:a3f19c --json
{"v":1,"ok":false,"error":{"code":"OUT_OF_MEMORY","class":"resource","retryable":true,
  "message":"the GPU ran out of memory while training bge-m3 at batch 64 (8 GB card)",
  "actions":[{"config":{"trainer.batch_size":32},"applicability":"machine","why":"halve the batch"},
             {"config":{"trainer.max_seq_length":256},"applicability":"review","why":"shorter inputs use less memory; chunks get re-cut"}],
  "next":"synk config set trainer.batch_size 32"}}
$ synk config set trainer.batch_size 32 --json
{"v":1,"ok":true,"data":{"key":"trainer.batch_size","old":64,"new":32,"affects":["train","eval"],"next":"synk plan"}}
$ synk apply plan:a3f19c --json
{"v":1,"ok":false,"error":{"code":"PLAN_STALE","retryable":true,"next":"synk apply plan:c88e01",
  "details":{"changed":["config trainer.batch_size 64 → 32"],"consent_unchanged":true}}}
$ synk apply plan:c88e01 --json         # consent unchanged: no need to ask again
   … dataset and reranker cached; training runs …
   (the laptop sleeps and the process dies)
$ synk status --json
{"v":1,"ok":true,"data":{"job":{"id":"job:…-c88e","state":"interrupted","done":true,
  "reason":"no heartbeat since 2026-10-15T15:02:11Z","progress":{"step":"train","completed":190,"total":320}},
  "next":"synk apply plan:c88e01"}}
$ synk apply plan:c88e01 --json         # resumes at checkpoint 190/320
… {"state":"succeeded","result":{"verdict":{"outcome":"better"}}}
```

---

## 10. Milestones

Each milestone ends with green tests, a README update and something a user can run.

| # | Milestone | Contents | Reused | Rewritten / new | Size |
|---|---|---|---|---|---|
| **M1** | **Contract reset + data in** | Envelope `v:1`; error `class/retryable/actions`; IDs; `source …`, `config get/set/unset/show`, `status` (conditions); clap tree + help template; HTML/DOCX/PDF readers + quality gate; doc/chunk dedup + leak guard; `SourceState` cursor; golden JSON tests; `schema` snapshots | store, `fs.rs` walker, chunker, tokenizer, lang, `config.rs` (`Project`), `output.rs` | `main.rs` tree, `commands.rs` → `source.rs`/`config_cmd.rs`, readers | M, ~2.5k LOC, 2–3 weeks |
| **M2** | **Plan / apply / job** | `graph.rs` (step keys, staleness, `why`); `plan` (read-only, `--measure` local only, throughput history); consent + plan IDs + `PLAN_STALE`; `--set`; job records, lock, heartbeat, `job show/wait/logs/cancel`; stop file; Ollama pull; protocol v2 `describe`/`fetch`; fake trainer + fake LLM server; `setup` as consent | `dataset::build`, `train::run`, `evaluate`, artifacts, `sidecar.rs` env setup, progress events | orchestration in `pipeline.rs` → `apply.rs`/`jobs.rs`; sidecar task dispatch | L, ~3.5k LOC, 3–4 weeks |
| **M3** | **Training data v2** | `query_slots` migration + import from `query_cache`; incremental generation; `queries_per_chunk = "auto"`; style pool from real queries; heading path in the prompt; `PROMPT_VERSION` 3 | generator client, parse/filters, round-trip, mining, reranker, `queries import` | cache layer, prompt | S–M, ~1k LOC, 1–2 weeks |
| **M4** | **Models + trainer v2** | Catalog v1 (bge-m3, EmbeddingGemma gated, Qwen3) with new fields; `hf:` pin resolution, mirror, token; `MODEL_ACCESS_DENIED`; LoRA auto + merge; Matryoshka; precision auto; gradient checkpointing; CachedMNRL; OOM handling; `model show` | catalog loader, `pick_device`, train loop, dev-epoch selection | sidecar `train` parameters, catalog schema additions | L, ~1.5k LOC (half Python), 2–3 weeks + GPU testing |
| **M5** | **Evaluation + results** | General check (NanoBEIR + XQuAD vi) with download consent; `GENERAL_REGRESSION`; `run list/show/inspect/compare` (with `paired_by`); `run report` HTML; new advice codes | `stats.rs`, verdict, `report.rs`, `per_query.jsonl`, `advice.rs` | general eval in the sidecar, HTML renderer | M, ~1.5k LOC, 2 weeks |
| **M6** | **Export + ship** | `export` st/onnx/gguf; vendored converter with the e5 and Gemma fixes; `--ollama` + vector check; model card with usage; `PREFIX_DROPPED_BY_TARGET`; `synk skill` + SKILL.md; dist installers; README rewrite; e2e tests for all formats | `export_folder`, `model_card` | sidecar `export` task, converter vendoring | L, ~2k LOC, 3 weeks |

**Progress (2026-10-16).** M1–M6 are built; all tests pass with the fake
trainer. Real-model testing is next. Where the build differs from this
design:
- **Plan preview.** `plan` previews the dataset by running the sync inside a
  SQLite savepoint and rolling it back, so it sees the post-sync documents
  exactly and saves nothing.
- **Query cache.** `query_slots` plus `query_asks` (what was already asked
  per chunk), so an LLM that writes fewer queries than asked isn't asked
  again on every build. The old `query_cache` rows aren't imported:
  `PROMPT_VERSION` 3 changes the prompt (section heading, style examples),
  so they would not match anyway.
- **Style pool.** Not a table: the first quarter (at most 5) of the real
  queries by a hash of their text, so the choice is stable without
  migration 5.
- **Schema.** `synk schema` is generated from the clap tree, `ErrorCode::ALL`
  and the advice codes, with hand-written JSON Schemas of the shared objects
  (envelope, error, advice, plan, job, event). Per-command output schemas
  via schemars are left for later.
- **Disk check.** Not built: `INSUFFICIENT_DISK` is defined but not emitted.
- **GGUF.** The e5 fix is a config relabel in a temporary copy (architecture
  `XLMRobertaModel`, no `pad_token_id`), not a converter patch. The
  converter is downloaded once per pinned llama.cpp commit when exporting.
- **Ctrl-C** during `apply` isn't caught: the job reads as `interrupted`
  once its heartbeat is two minutes old, and applying again resumes.
- **Vietnamese PDF spike** still needs real files from the owner.

**Changed after M6 (2026-10-16): only tested GPUs.** ROCm and MPS were
dropped; synk trains on NVIDIA (CUDA) or CPU. See
[ADR 0003](../adr/0003-only-tested-gpus.md). Rows below that mention ROCm or
MPS are superseded.

**Changed after M6 (2026-10-16): fetch, then train.** The owner found plan
IDs a poor way to name attempts. Plan/apply/job were replaced by
`synk train` (runs numbered `run:<n>-<label>`), `synk fetch NAME…` as the
consent for downloads, `--dry-run` as the preview, and top-level
`wait`/`logs`/`cancel`. See [fetch-then-train.md](fetch-then-train.md) and
[ADR 0002](../adr/0002-fetch-then-train.md). Sections 5.2–5.5, 6.3 and 9
below still describe plan/apply/job; read them through ADR 0002.

Total: roughly 12–17 weeks for one developer **[estimate]**.

Order: M1 and M2 come first because everything else plugs into the step graph and the contract. M3 and M4 can be swapped. M6 last, because its checks reuse M5's embedding code.

Reused as is or with small changes, about 55% of today's code **[observed]**, by file size:
- `chunk.rs`, `tokenizer.rs`, `lang.rs`, `stats.rs`, `artifact.rs`, `catalog.rs`, `queries.rs`, `advice.rs`
- most of `dataset.rs` and `train.rs`
- `fs.rs`, the store, the sidecar env setup
- the Python train/eval core

Rewritten: `main.rs`, `commands.rs`, `pipeline.rs`, `views.rs` (CLI layer), the sidecar's task entry, and the generator cache.

---

## 11. What is reused from today, and why

| Today's piece **[observed]** | Verdict | Why |
|---|---|---|
| Rust orchestrator + Python sidecar via uv | **keep** | Only stack covering CUDA/ROCm/MPS/CPU with every target model (ADR 0001) |
| Content-addressed artifacts, `KIND_VERSION`s | **keep** | They are what makes plan/apply cheap (agent-first §10) |
| SQLite with append-only migrations | **keep** | Migrations 1–3 exist; v1 appends 4 (`query_slots`) and 5 (`style_pool`) |
| Token-aware chunking, path-hash splits, dev split | **keep** | Measured fixes; split already ignores source names |
| Paired-t verdict per language and kind, `per_query.jsonl` | **keep** | Trustworthy verdict; feeds inspect, compare and the report |
| Advice object | **keep** | Matches rustc applicability; agents already have the shape |
| `--plan` flags, step commands, `runs/inspect/compare` names | **replace** | They become `plan`, `apply --target` and `run …` |
| Generator cache key with `queries_per_chunk` | **replace** | Finding 7 |
| `plan` sending sample chunks | **replace** | Finding 1 |
| Human renderer (`human.rs`) | **keep, extend** | Plain words, `-v`, TTY rules already right |

---

## 12. Decisions for the owner

| # | Decision | Recommendation |
|---|---|---|
| 1 | Break the JSON contract once, before the first release, and freeze `v:1`? | **Yes.** Nothing is published **[observed]**, so this is the cheapest moment. Old command names are not kept as aliases |
| 2 | `plan` never sends text; timing only with `--measure` against a local LLM? | **Yes.** |
| 3 | Export is a separate command and never automatic? | **Yes.** It allows several runs and an explicit choice |
| 4 | No spending limits; consent = plan ID or `--yes`; remote also needs `allow_remote`? | **Yes.** |
| 5 | No `--detach` in v1, with job records + wait/logs/cancel? | **Yes.** Revisit if Codex one-shot users report killed runs |
| 6 | PDF/DOCX/HTML in v1 (no OCR), parsed in Rust? | **Yes**, after the M1 spike on 20 Vietnamese PDFs. If PDF text is mojibake too often, ship PDF as "experimental" in the docs |
| 7 | General check on by default (about 9 MB download, a few minutes of eval)? | **Yes.** It is the forgetting guard the owner asked for. `[eval] general_check = false` opts out |
| 8 | Allow `plan` to call the HF Hub for metadata (sizes, resolving `main`) without consent? | **Yes.** It sends only model names, the same as the exempt tokenizer fetch today |
| 9 | Catalog v1: add bge-m3 and EmbeddingGemma (gated); exclude `trust_remote_code` models? | **Yes.** |
| 10 | LoRA r=32/α=64 automatically at 300M+ params? | **Yes**, with `trainer.finetune = "full"` to override |
| 11 | GGUF default `q8_0` with a vector check, `f16` as the fallback? | **Yes.** Smaller files and Unsloth's default; the check catches bad quantization |
| 12 | `queries_per_chunk = "auto"` as the default? | **Yes**, now that raising it is incremental |
| 13 | MCP server in v1? | **No.** The CLI + skill reach every shell agent (cli-fit §3) |
| 14 | Record the plan/apply/job model, the dropped `tune` and the export split as an ADR, replacing the old `--download` flag? | **Yes.** |

---

## 13. Unverified items

- `pdf-extract` text quality on Vietnamese PDFs, and the 50-chars/page and 2% garbage thresholds (M1 spike).
- The `queries_per_chunk = "auto"` target of 1,500 training queries; the `GENERAL_REGRESSION` threshold of −0.02; the q8_0 cosine threshold of 0.995.
- Vietnamese quality of Qwen3-Embedding-0.6B and EmbeddingGemma: their cards say 100+ languages without naming vi.
- bf16 training on Apple MPS; gradient checkpointing support per architecture; `min_vram_gb` values; LoRA learning rate 1e-4.
- Download sizes and times in the transcripts are illustrative (sizes from cli-experience.md, not re-measured).
- GGUF conversion of fine-tuned e5-base, bge-m3, EmbeddingGemma and Qwen3 (read from source in cli-fit §5.2, not run).
- Codex one-shot exec (10 s kill) and how often agents hit it without `--detach`.
- Availability of GPU CI runners for the e2e matrix.
