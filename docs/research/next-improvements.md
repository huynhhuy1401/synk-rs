# What should synk's next improvements be, after v0.1.0?

Checked: 2026-10-17
Status: research + recommendation, not accepted

Method: the repo first (code, ADRs, plans, the field notes from the Kaggle
T4 run), then primary sources: papers, model cards, official docs and
library source. Every outside claim links its source. **[unverified]** marks
claims that come from search snippets or third-party pages, or that nobody
has measured yet. Code references are `file:line` at commit `9fcd4d5`.

Input notes: [local-rag-tooling.md](local-rag-tooling.md) (the landscape,
2026-10-16) and [github-workflows.md](github-workflows.md) (CI). This note
does not repeat them; it ranks what to do next.

## TL;DR

1. **The verdict can't be decisive on small corpora, by design. Fix the
   test design first.** The verdict's unit is a document
   (`train.rs:661`), and only ~20% of documents are tested. The Kaggle run
   had 11 documents, so 1 test document: no test can say anything from one
   document. To see a +0.05 gain you need roughly 30 test documents, which
   means ~150 documents at `test_percent = 20`. Most personal note folders
   are smaller. Do three things: warn in `--dry-run` *before* spending hours;
   an `auto` test share that aims for at least ~10 test documents; and opt-in
   **cross-validation over documents** (`eval.folds`), so every document is
   tested once.
2. **Make real queries easy to get.** Real searches are the only honest
   answer to "is it better for *me*", and 30 of them already decide the
   verdict. Add `synk queries template` (a file listing documents; the user
   writes how they'd search for each) and say "spread them over 10+
   documents". This is 15 minutes of the user's time and beats any
   statistical trick.
3. **Fix the honesty and consent leaks found in the code.** GGUF export
   downloads llama.cpp from GitHub without consent (`synk_sidecar.py:902-914`);
   importing a few real queries can silently invalidate the whole LLM query
   cache (`generator.rs:476-484`); the project lock has a race
   (`ops.rs:461`, `ops.rs:567`).
4. **Catch Ctrl-C, and check disk space before downloads and training.**
   Today Ctrl-C leaves the run "running" for up to 120 s (`ops.rs:36`), so
   the obvious retry fails with `JOB_RUNNING`. A full-model checkpoint holds
   the optimizer state, so a run needs several GB of free space that
   nothing checks.
5. **Vietnamese text hygiene:** Unicode NFC normalization at read time, a
   Vietnamese detector that works on mixed vi/en technical text, and the
   still-open Vietnamese PDF spike with real files. These are cheap and
   protect the owner's own data.
6. **Tell users exactly how to use the model after export.** LlamaIndex and
   LangChain need explicit settings to apply the `query: `/`passage: `
   prefixes, and Open WebUI has `RAG_EMBEDDING_QUERY_PREFIX` /
   `RAG_EMBEDDING_CONTENT_PREFIX`. Put copy-paste snippets in the export
   output and README.
7. **Publish one honest benchmark:** run synk on public corpora that have
   *human* test queries, and report whether synk's synthetic-query verdict
   agrees with the human-query result. This is the claim users most need
   checked, and nobody else publishes it.
8. **Add `granite-embedding-311m-multilingual-r2` to the catalog as a
   candidate** (Apache 2.0, Vietnamese in its 52 tuned languages, no remote
   code), after one real run. Keep e5-small as the default.

## 1. Result quality and honesty

### 1.1 Why the Kaggle verdict was "no clear change"

What the code does:

- The dataset splits whole documents into train and test by a hash of the
  path (`dataset.rs:749`). With `test_percent = 20` and 11 documents, about
  2 land in test on average; the Kaggle run got 1.
- The verdict is a paired t-test on **per-document** mean nDCG@10 changes
  (`train.rs:661`, `stats.rs:22`). That is the right unit: queries about
  the same document share its text, so they are not independent. With one
  document there is no interval at all (`stats.rs:24`), so the verdict is
  "too few to tell" no matter how good the model is.
- With fewer than 20 training documents, there is no dev split
  (`synk_sidecar.py:386`, `dev_min_documents: 20` in `train.rs`), so
  `epochs = "auto"` simply trains up to 5 epochs (`train.rs:236`) with no
  check for over-fitting. That is the "0 dev queries" in the log.
- `--sample N` samples **chunks**, not documents (`dataset.rs:543`), so a
  sample can drop test documents and shrink the test further.
- `--dry-run` reports `test_chunks` but not test documents
  (`dataset.rs:605`), so nobody can see the problem before training.

How big the test must be. The code already computes the smallest change a
test can see 80% of the time: `(t₀.₉₇₅ + t₀.₈₀) × sd / √n`
(`stats.rs:49`). The spread of per-document changes (`sd`) is unknown for
real corpora; the table uses 0.10 and 0.15 as plausible guesses
**[unverified: measure it from real runs]**:

| Test documents (n) | Smallest visible change, sd 0.10 | sd 0.15 |
|---|---|---|
| 3 | 0.31 | 0.47 |
| 5 | 0.17 | 0.25 |
| 10 | 0.10 | 0.15 |
| 20 | 0.066 | 0.10 |
| 30 | 0.053 | 0.079 |
| 50 | 0.040 | 0.061 |

The Kaggle gain (+0.055) is a typical size for this kind of tuning:
GPL reports up to +9.3 nDCG@10 points
([arXiv 2112.07577](https://arxiv.org/abs/2112.07577v3)) and Databricks saw
anything from 0 to large gains depending on the data
([Databricks](https://www.databricks.com/blog/improving-retrieval-and-rag-embedding-model-finetuning)).
So to *see* a typical gain, synk needs about 30 test documents. That is the
core finding of this note: **synk's honest verdict and small personal
corpora currently don't fit together.** Each fix below attacks a different
part of it.

### 1.2 Fixes, cheapest first

**a. Predict the problem in `--dry-run` (S).** Count test documents in the
dataset plan (`dataset.rs:565`, add `test_documents`), and raise
`TEST_SET_UNDERPOWERED` as dataset advice *before* training when there are
fewer than ~8 test documents (it fires today only after the run,
`advice.rs:252`). Message, plain: "Only 2 documents are held out for the
test. The verdict will probably be 'no clear change' even if tuning helps.
Add documents, set `dataset.test_percent` higher, import real queries, or
use `eval.folds`." Verify: a CLI test with 6 documents expects the advice
code in `train --dry-run --json`.

**b. `test_percent = "auto"` (S).** Aim for at least 10 test documents, at
least 20%, at most 50%. On 11 documents this gives 5 test documents instead
of 1 or 2: still weak (table above), but no longer hopeless. Cost: fewer
training documents. Must keep the split stable under document additions
(today's hash split is; a threshold that moves with the corpus size would
move documents between splits, so pick it once per dataset and record it
in the dataset params). Bump the dataset `KIND_VERSION`.

**c. `--sample` keeps every test document (S).** Sample training chunks
freely, but take at most k chunks from *each* test document instead of the
first N test chunks (`dataset.rs:543`). Power comes from the number of test
documents, not the number of test queries, so this keeps the verdict alive
while keeping the LLM bill small.

**d. Cross-validation over documents: `eval.folds = 5` (M).** Train K
models, each with a different fifth of the documents held out; every
document is tested exactly once by a model that never saw it; pool the
per-document differences into one verdict. On 11 documents that is n = 11
instead of n = 1. Cost: K trainings (Kaggle: ~5 × 190 s on a T4; on CPU
~K × the README's 13 min for e5-small). Two honest caveats to print:

- The folds' training sets overlap, so the per-document differences are
  not fully independent, and no estimate of the variance of K-fold
  cross-validation is unbiased in general
  ([Bengio & Grandvalet, JMLR 2004](https://www.jmlr.org/papers/v5/grandvalet04a.html)).
  The interval is therefore somewhat optimistic **[size of the effect here
  unverified]**. Say so in the verdict message and in `GLOSSARY.md`.
- The verdict is then about *the recipe on this corpus*, not about one
  particular saved model. Pair it with item (f).

Where: a fold index in the dataset (`split` becomes `fold` in
`corpus.jsonl`, a new field; JSONL `v` rules allow adding it), one train
step per fold in `graph.rs`, `judge` already pools by document. Each fold
is its own cached artifact, so an interrupted CV run resumes. Verify: the
fake trainer test for "every document appears in exactly one test fold",
then one Kaggle run on synk's own docs.

**e. A permutation test as a cross-check at small n (S).** With few
documents the t-test leans on a normality assumption. In IR evaluation,
the t-test, bootstrap and randomization tests agree closely at normal
topic counts ([Smucker, Allan, Carterette, CIKM 2007](https://ciir-publications.cs.umass.edu/getpdf.php?id=744)),
but disagree more as the number of topics falls
([SIGIR 2009 follow-up](https://uwaterloo.ca/data-systems-group/node/4913)).
A sign-flip permutation test on the per-document differences is exact,
needs no library, and for n ≤ 15 can enumerate all 2ⁿ flips. Report its
p-value next to the t-test's in `verdict` (new field); keep the t-test as
the decision rule so verdicts don't change meaning. Note that with n ≤ 5
the two-sided permutation test can't reach p < 0.05 at all (its smallest
possible p-value is 2/2ⁿ = 0.0625 at n = 5). When the t-test says
`better` there and the permutation test can't, the result rests on the
normality assumption alone — a clear, honest sentence to show users.

**f. "Refit on everything" before export (M, after d).** For small
corpora, holding out 20–50% of documents also means the exported model
never trained on them. Offer `synk export <run> --refit`: train once more
on all documents with the same settings, label the export "refit on all
documents; the verdict was measured on run:N's held-out documents". This
is standard practice after cross-validation, but it is a weaker promise,
so it must be opt-in and clearly labelled.

**g. Real queries, made easy (S–M).** Real queries already decide the
verdict at 30 or more (`train.rs:623`), and they are searched over the
whole corpus, which is exactly how the model will be used. The gap is
getting them. Proposals:

- `synk queries template --out searches.jsonl --documents 15` writes one
  line per document (title + path, no text), with an empty `query`. The
  user fills in how they would search for something they know is in it,
  *without opening it* (writing while reading the text copies its words
  and makes the test too easy). `synk queries import` already reads the
  result.
- Advice when real queries cover few documents: the real-query verdict
  also uses documents as units, and the code accepts 30 queries from just
  2 documents (`train.rs:623`). Say "spread them over 10+ documents".
- Later: the opt-in query log from `synk serve` (local-rag-tooling.md #3).

Why this matters: synthetic test queries come from the same LLM and prompt
as the training queries, so a gain on them partly measures "better at this
LLM's style". Fully synthetic test collections rank systems sensibly but
are biased toward LLM-based systems
([arXiv 2405.07767](https://arxiv.org/abs/2405.07767v1),
[arXiv 2506.10301](https://arxiv.org/abs/2506.10301v2)), and Chroma saw no
significant gain from synthetic positives for its adapters
([Chroma](https://www.trychroma.com/research/embedding-adapters)). Real
queries remove this doubt.

**h. Research option, not for v0.2: held-out *queries* on a shared
corpus.** GPL and similar work train on the target corpus itself and test
with that dataset's own (human) queries
([UKPLab/gpl README](https://github.com/UKPLab/gpl)); the corpus is shared
between training and test. synk's document hold-out is stricter than that.
A "shared corpus, held-out queries" mode would test every document, but
with synthetic queries it leaks: a held-out query is often a paraphrase of
a training query for the same chunk. Only consider it for real queries
(where it is what (g) already does), or after the benchmark in §4.3 shows
how much it inflates results.

### 1.3 Better synthetic queries

What synk does well already: style examples from real queries (Promptagator
uses about 8 examples, [arXiv 2209.11755](https://arxiv.org/abs/2209.11755));
a round-trip filter; cross-language queries; per-chunk caching; thinking
turned off with `reasoning_effort: "none"` (`generator.rs:393`), which
Ollama maps to "thinking off" for boolean thinking models such as qwen3
([Ollama OpenAI compatibility](https://docs.ollama.com/api/openai-compatibility)).
Open LLMs from 3B up work as query generators
([arXiv 2510.02241](https://arxiv.org/abs/2510.02241)).

Improvements:

- **Vietnamese without diacritics (S, measure first).** Many people type
  Vietnamese without tone marks ("tang timeout nginx") **[unverified: no
  primary source found for how common this is]**. Measure before training
  on it: add an eval slice that strips diacritics from Vietnamese test
  queries (deterministic, no LLM) and report base vs tuned on it. If the
  base model drops sharply there, add a training option that strips
  diacritics from a share of Vietnamese training queries. Where:
  `train.rs` verdict `by_kind` (new key), sidecar `evaluate`.
- **Ask for a mix of query types (S).** Today the prompt asks for N
  "different" queries (`generator.rs:327`). Ask explicitly for a keyword
  query, a question, and one that uses *none* of the passage's distinctive
  words. Bump `PROMPT_VERSION`. Measure with a cheap lexical-overlap
  statistic per dataset (share of query tokens found in the chunk) and
  report it in the dataset summary.
- **Summarize-then-ask for low-resource languages (M, experiment).**
  SWIM-IR found that having the LLM summarize the passage first, then ask,
  helped it write good queries in 33 languages
  ([arXiv 2311.05800](https://arxiv.org/abs/2311.05800v2)). Try it as a
  prompt variant on Vietnamese chunks; keep it only if the verdict
  improves.

### 1.4 Hard negatives

What synk does: mines with the base model, skips chunks from the same
document, and drops candidates scoring above 95% of the positive
(`synk_sidecar.py:288-304`). That is NV-Retriever's best setting,
TopK-PercPos at 95% ([arXiv 2407.15831](https://arxiv.org/html/2407.15831v2)).

Why `MANY_UNMINED_NEGATIVES` fired on Kaggle, and what to change:

- **Same-document chunks are always skipped** (`synk_sidecar.py:304`). On a
  small corpus with long documents, most of the closest chunks are from
  the same document, so few candidates are left. Allow same-document
  negatives **only when the reranker confirms them** (reranker probability
  below `reranker_max_prob`). Without a reranker, keep today's rule.
- **The miner is the small base model.** NV-Retriever found the best hard
  negatives came from strong 7B teachers (e5-mistral-7b), not from small
  models (same paper). Option `trainer.miner = "<catalog id>"` (e.g.
  `qwen3-embedding-0.6b` or `bge-m3`) to mine with a stronger model. It
  is a download, so it needs a `Consent` item and a fetch name.
- **e5 scores are compressed.** multilingual-e5 cosine scores sit around
  0.7–1.0 because of its low training temperature
  ([model card FAQ](https://huggingface.co/intfloat/multilingual-e5-small)).
  A 5% relative margin is therefore a large cut in e5's useful range. Log
  the distribution of (negative score / positive score) in `prep.json` so
  this can be tuned per model from data, not guessed **[unverified that
  it matters]**.
- Verify each change with `synk run compare` on two real corpora; keep
  only what wins. Bump the train step's `KIND_VERSION`.

### 1.5 Base models for Vietnamese + English (October 2026)

| Model | Size | License | Notes | Source |
|---|---|---|---|---|
| multilingual-e5-small (default) | 118M | MIT | VN-MTEB retrieval 34.12; trains on CPU; 512-token limit | [model card](https://huggingface.co/intfloat/multilingual-e5-small), [VN-MTEB](https://arxiv.org/html/2507.21500v1) |
| multilingual-e5-base | 278M | MIT | VN-MTEB retrieval 34.50 (barely above small) | [VN-MTEB](https://arxiv.org/html/2507.21500v1) |
| bge-m3 | 568M | MIT | VN-MTEB retrieval 39.84 | [model card](https://huggingface.co/BAAI/bge-m3), [VN-MTEB](https://arxiv.org/html/2507.21500v1) |
| Qwen3-Embedding-0.6B | 0.6B | Apache 2.0 | MMTEB 64.33; needs the query instruction (1–5% drop without) | [model card](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) |
| EmbeddingGemma-300m | 308M | Gemma terms, gated | Already in catalog | [Google](https://ai.google.dev/gemma/docs/embeddinggemma) |
| **granite-embedding-311m-multilingual-r2** | 311M | Apache 2.0 | Released 2026-04-29; Vietnamese among its 52 retrieval-tuned languages; 32K context; Matryoshka 768→128; MTEB multilingual retrieval 65.2 (vendor-reported); ModernBERT, no `auto_map` (no remote code) | [model card](https://huggingface.co/ibm-granite/granite-embedding-311m-multilingual-r2), [config](https://huggingface.co/ibm-granite/granite-embedding-311m-multilingual-r2/raw/main/config.json) |
| gte-multilingual-base | 305M | — | VN-MTEB retrieval 38.38, but needs `trust_remote_code` (custom `NewModel`), so synk can't use it | [config](https://huggingface.co/Alibaba-NLP/gte-multilingual-base/raw/main/config.json) |
| AITeamVN/Vietnamese_Embedding | 568M | Apache 2.0 | bge-m3 tuned on ~300K Vietnamese triplets; author-reported Acc@1 0.727 vs 0.568 (bge-m3) on Zalo Legal | [model card](https://huggingface.co/AITeamVN/Vietnamese_Embedding) |

Recommendations:

- **Keep e5-small as the default.** It trains on a CPU, and e5-base gains
  almost nothing on Vietnamese retrieval (34.12 → 34.50).
- **Trial granite r2 (S once a GPU run is available).** Add to the catalog
  only after one real run passes train + eval + the three exports. Open
  questions: whether llama.cpp's GGUF converter supports this ModernBERT
  model **[unverified]**, and whether 311M should be LoRA (the catalog rule
  is LoRA from 300M, `train.rs` `finetune`).
- **Offer a "compare untuned models" step (M).** Before tuning, the cheapest
  win may be switching models. `eval.baselines = ["bge-m3", "bm25"]` would
  evaluate untuned models (and BM25, a strong baseline in BEIR,
  [arXiv 2104.08663](https://arxiv.org/abs/2104.08663v1)) on the same test
  set with the same paired test. Advice `BASELINE_BETTER` when an untuned
  model beats the tuned one. This answers "is tuning worth it here?"
  honestly. Costs evaluation time and downloads (consent items).

### 1.6 When tuning is not worth it (tell the user)

synk should say so, with evidence, rather than leave it to the user:

- The base already scores ≥ 0.95 (`TEST_SET_TOO_EASY`, exists).
- A bigger untuned model beats the tuned small one (`BASELINE_BETTER`,
  proposed above).
- The corpus is too small to train or test (the dry-run advice in 1.2a).
- Retrieval isn't the bottleneck: Databricks found end-to-end RAG improved
  only where retrieval was the problem
  ([Databricks](https://www.databricks.com/blog/improving-retrieval-and-rag-embedding-model-finetuning)).
  synk can't measure this; README should say so plainly.

## 2. Product and UX for a non-expert

### 2.1 First run

- **The first run downloads a lot and nothing checks space**: qwen3:8b
  (~5 GB), the trainer (1.0 GB CPU / 3.5 GB CUDA estimated,
  `graph.rs:34-35`), the model. See §3.2.
- **Estimates are guesses until synk has history.** Defaults are 4 s per
  LLM call and 0.4 / 2.5 s per training step on GPU / CPU, for every model
  (`graph.rs:37-39`). The Kaggle run trained ~275 steps in 190 s
  (874 kept queries, batch 16, 5 epochs; derived from the field notes), so
  ~0.7 s/step: the GPU default is ~1.7× too low for e5-small with 3
  negatives. The default does not scale with model size, and the history
  key (`graph.rs:363`) ignores batch size, sequence length and negatives,
  so a `--set trainer.batch_size=64` variant reuses a wrong speed. Fix:
  defaults per catalog model (measured on CPU and T4), store speed per
  *sequence* (batch × (2 + negatives)) instead of per step, and keep a
  running median instead of the last value (`graph.rs:342`).
- **LLM time on a laptop CPU is the biggest unknown** and the default 4 s
  per call can be far off **[unverified]**. Option `--dry-run --measure`:
  time the local LLM on a built-in sample passage (no user text), so the
  estimate is measured without breaking "dry-run never sends document
  text". Opt-in, because it loads the model.
- **uv is a prerequisite the user must install by hand** (`sidecar.rs`
  `find_uv`). Consider `synk fetch uv` (a consented download of uv's
  standalone binary) **[effort and licensing check needed]**.

### 2.2 After export: integration snippets (S)

The sentence-transformers folder stores the prompts, but popular wrappers
don't use them unless told:

- **LlamaIndex** `HuggingFaceEmbedding` builds its own prompts dict from
  `query_instruction` / `text_instruction` and encodes with
  `prompt_name="query"` / `"text"`
  ([source](https://raw.githubusercontent.com/run-llama/llama_index/main/llama-index-integrations/embeddings/llama-index-embeddings-huggingface/llama_index/embeddings/huggingface/base.py)).
  For a local folder its defaults are probably empty **[unverified]**, so
  users must pass `query_instruction="query: ", text_instruction="passage: "`.
- **LangChain** `HuggingFaceEmbeddings` takes `query_encode_kwargs` for
  queries and `encode_kwargs` for documents
  ([source](https://raw.githubusercontent.com/langchain-ai/langchain/master/libs/partners/huggingface/langchain_huggingface/embeddings/huggingface.py)):
  `query_encode_kwargs={"prompt_name": "query"}`,
  `encode_kwargs={"prompt_name": "document"}`.
- **Open WebUI** has `RAG_EMBEDDING_QUERY_PREFIX` and
  `RAG_EMBEDDING_CONTENT_PREFIX`
  ([config.py](https://raw.githubusercontent.com/open-webui/open-webui/main/backend/open_webui/config.py)).
  Which embedding engines apply them is **[unverified]**; test with the
  Ollama engine and the built-in one before documenting.
- Every one of these apps needs a full re-index after a model change
  ([Open WebUI](https://docs.openwebui.com/troubleshooting/rag),
  [AnythingLLM](https://docs.anythingllm.com/setup/embedder-configuration/overview)).
  Say so in the export output.

Where: `export.rs:339` `usage()` gains `llamaindex`, `langchain`,
`open_webui_env` entries (add-only fields); README "Export" gets the same
snippets. Verify: a Python test in the e2e job that loads the export with
each wrapper's settings and checks the vectors match
`SentenceTransformer.encode(..., prompt_name=...)`.

### 2.3 When documents change

Retraining is already cheap where it matters: queries are cached per chunk,
so only new or changed chunks call the LLM, and the split is stable per
path. What's missing is *knowing* when to retrain. Add to `synk status`: "N
documents changed since run:3 (x% of chunks are new to it)" and advice
`RUN_STALE` above a threshold. (local-rag-tooling.md #6, `synk drift`, is
the fuller version.)

### 2.4 Connectors

Stay with files. The owner's data is wiki/notes/docs, PDF and DOCX, which
the `fs` connector covers. Network connectors (Confluence, Notion API,
Google Drive) add auth, tokens and network reads; defer them, and if ever
added, make them explicit opt-in and read-only (AGENTS.md rules). Better
value now: reader quality (§3.4) and maybe `.pptx`/`.odt`/`.epub` only on
user demand.

### 2.5 Small UX gaps

- `synk run list` has no `--limit` (`views.rs:106`, plan §12). Add
  `--limit N` (default 20) and `--all`; JSON keeps `runs` and adds
  `total`. S.
- The style pool changes as real queries are added (`dataset.rs:233`:
  "one in four, at most five, picked by hash"), and the generator's cache
  key includes the style examples (`generator.rs:484`). So importing a few
  more real queries can change the pool and **re-ask the LLM for every
  chunk** (hours on a CPU). The dry-run shows the LLM calls, but nothing
  says why. Fix: freeze the style pool once chosen (store which queries
  are style examples), or change it only with an explicit flag; and add a
  dry-run reason "style examples changed". Bump `PROMPT_VERSION` only if
  the prompt changes.

## 3. Reliability and operations (ranked by user impact)

### 3.1 Ctrl-C and killed processes (high impact, S)

- synk installs no Ctrl-C handler (no `ctrlc`/`signal-hook` in the
  workspace). On Ctrl-C the record stays `running` until its heartbeat is
  120 s old (`ops.rs:36`). During that time the natural retry,
  `synk train`, fails with `JOB_RUNNING`. Agents whose shell timed out
  hit the same wall.
- Fix: the `ctrlc` crate (covers Windows console events too) sets a flag;
  the running op writes its `cancel` file (the sidecar already stops at a
  safe point and checkpoints when that file appears, `synk_sidecar.py`
  `stop_file`), then finishes as `cancelled`. A second Ctrl-C exits at
  once. Also: in `running()`, treat a record whose `pid` no longer exists
  as interrupted immediately, instead of waiting 120 s (`sysinfo` or
  platform calls).
- The sidecar dies with its parent only indirectly: its next write to the
  closed stdout pipe fails. That works because progress is frequent, but
  add a test that kills synk mid-training and checks the Python process is
  gone within a few seconds (Linux CI, CPU).

### 3.2 Disk space (high impact, S–M)

Nothing checks free space. What a first run needs, roughly:

- Downloads: the LLM (~5 GB for qwen3:8b), the trainer (1–3.5 GB, more
  after unpacking **[unverified]**), the model.
- During full fine-tuning, a checkpoint holds the weights plus AdamW's two
  moment buffers, so roughly 3× the fp32 model size
  (e5-small: 118M × 4 bytes × 3 ≈ 1.4 GB); with `save_total_limit=1` the
  old one is removed only after the new one is saved, plus the best-epoch
  copy and the final model **[estimate; measure on a real run]**.
- On Windows without Developer Mode, the Hugging Face cache stores files
  without symlinks, so repeated revisions duplicate files
  ([HF cache docs](https://huggingface.co/docs/huggingface_hub/guides/manage-cache)).

Fix: in `graph.rs`, sum the bytes each step needs *per volume* (HF cache,
`$SYNK_HOME`, the Ollama models dir, `.synk/artifacts`) and compare with
free space (`fs4::available_space` or `sysinfo`). Not enough → a dry-run
blocker with a new error code (e.g. `DISK_SPACE_LOW`, class `resource`),
with `details.needed`/`details.free`/`details.path`. Update the README
error table. Also map "No space left on device" / Windows `ERROR_DISK_FULL`
during training to that code instead of `SIDECAR_FAILED`. Verify: a unit
test with a fake free-space provider; a manual run on a small tmpfs.

### 3.3 The project lock race (medium impact, S)

`check_free` reads the lock, then `begin` writes it with `fs::write`
(`ops.rs:461`, `ops.rs:567`). Two `synk train` started together (an agent
retrying while the first is still starting) can both pass. Fix: create the
lock with `OpenOptions::new().write(true).create_new(true)`; if it exists
and its record is stale or its pid is dead, remove and retry once. Verify:
a test that starts two ops from two threads and expects one `JOB_RUNNING`.

### 3.4 Vietnamese text extraction (medium–high impact for the owner, S + spike)

- **NFC normalization.** `normalize` (`readers.rs:119`) trims lines but
  doesn't normalize Unicode. Vietnamese can be stored precomposed (NFC) or
  decomposed (NFD), and the same word then hashes, dedups, and may
  tokenize differently (depends on the model's tokenizer) **[tokenizer
  behaviour unverified per model]**. Apply NFC
  ([UAX #15](https://unicode.org/reports/tr15/)) to all reader output with
  the `unicode-normalization` crate. This changes document hashes once:
  every chunk's queries are re-asked unless the change ships with the
  next breaking dataset bump; plan it with 1.2b.
- **Legacy encodings.** Older Vietnamese documents use TCVN3/VNI fonts;
  their PDF/DOCX text comes out as Latin-looking mojibake that the current
  garbled-text check (U+FFFD and control characters only,
  `readers.rs:292`) does not catch **[unverified on real files]**. Add a
  skip reason when the text has no Vietnamese letters but many characters
  typical of TCVN3 output, after collecting examples.
- **The PDF spike.** `pdf-extract` 0.12 is the only PDF reader. Collect
  10–20 real Vietnamese PDFs (public legal documents are a good, shareable
  source), extract with pdf-extract and with pdfium, and compare against a
  hand check. Decide from that; don't switch blindly.
- **Language detection on mixed text.** `lang::detect` is whatlang with a
  0.5 confidence floor (`lang.rs:110`). Vietnamese notes full of English
  technical terms may be detected as English or as unknown, and then the
  prompt asks for English queries. Vietnamese has letters no other common
  language uses (ă â đ ê ô ơ ư plus tone marks); a rule like "≥ 2% of
  letters are Vietnamese-specific → vi" before whatlang is cheap. Verify
  with fixtures from the owner's style of notes.

### 3.5 Windows + CUDA is untested (medium impact, M)

- PyTorch publishes the exact wheel synk installs for Windows:
  `torch-2.14.1+cu130-cp312-cp312-win_amd64.whl`
  ([PyTorch index](https://download.pytorch.org/whl/cu130/torch/)), so the
  install path should work; the training path hasn't run.
- Options: (1) GitHub's GPU larger runners, generally available for
  Windows and Linux with a T4
  ([changelog](https://github.blog/changelog/2024-07-08-github-actions-gpu-hosted-runners-are-now-generally-available)),
  set up through an organization's runner group; the repo is under a
  personal account, so this needs an org and a paid plan **[plan
  requirements and price unverified]**. (2) Rent a Windows GPU VM by the
  hour and run the `#[ignore]`d e2e test once per release
  **[cost unverified]**. (3) A volunteer with a Windows + RTX machine
  running a scripted checklist (`SYNK_E2E_PYTHON=... cargo test --
  --ignored` plus one export of each format).
- Until one of these has run, the README's "Windows with CUDA is untested"
  stays. Good.

### 3.6 Remote-LLM egress has no end-to-end test (low–medium, S)

The consent rule for remote sends is the most sensitive promise synk
makes, and it is untested (plan §12). Add a CLI test with a tiny
OpenAI-compatible server on a local `TcpListener`, plus a test-only way to
make `is_local` treat it as remote (an env var read only in debug builds,
or a hostname that resolves to loopback but isn't on the loopback list).
Assert: without `allow_remote` → `REMOTE_NOT_ALLOWED`; with it and no
`--yes` → `CONSENT_REQUIRED` with an `egress` item; with both → the run
record's `result.egress` lists the host and chunk count.

### 3.7 No GPU CI, no e2e job (low–medium, S)

`ci.yml` runs build/clippy/tests on four OSes; the e2e workflow proposed
in github-workflows.md §3 isn't in `.github/workflows/` yet. Add it (CPU,
weekly + on tags). GPU stays a manual Kaggle checklist per release.

### 3.8 Thresholds nobody has checked

`AUTO_TARGET_QUERIES = 1500` (`dataset.rs:212`), `GENERAL_REGRESSION_DELTA
= 0.02` (`advice.rs:196`), `MIN_COSINE_Q8 = 0.995` (`export.rs:30`),
`dev_min_documents = 20`, `MIN_REAL_QUERIES = 30`. Don't tune them by
feel. The benchmark in §4.3 produces the data; record each decision in an
ADR.

## 4. Distribution and adoption

### 4.1 Install paths

- dist (cargo-dist 0.32) builds shell and PowerShell installers
  (`dist-workspace.toml`). dist also supports Homebrew formulas and MSI;
  winget is listed only as "requested", and Scoop isn't mentioned
  ([dist installers](https://axodotdev.github.io/cargo-dist/book/installers/index.html)).
- Recommendation: add `homebrew` (needs a tap repo and a token in the
  release workflow) when there are Mac users asking; a Scoop bucket is a
  one-file JSON manifest and suits Windows developers; winget needs a PR
  to microsoft/winget-pkgs per release **[tooling unverified]**. None of
  these is urgent: the one-line installers work, and the real friction is
  uv and the ~5 GB of downloads, not the 10 MB binary.

### 4.2 Docs and demo

- A 5-minute walkthrough on a small *public* corpus (e.g. synk's own docs)
  showing the honest "no clear change, too few documents" outcome *and*
  how to fix it (real queries or folds). Showing a non-`better` verdict is
  on-brand.
- A "Use the model" page: the snippets from §2.2.
- A "Troubleshooting" page keyed by error code (the codes exist; the
  prose doesn't).

### 4.3 One honest benchmark to publish (M)

Question: **does synk's synthetic-query verdict agree with human queries?**

- Corpora with human test queries over a fixed corpus: from BEIR
  ([arXiv 2104.08663](https://arxiv.org/abs/2104.08663v1)) a small one such
  as SciFact or NFCorpus; for Vietnamese, VieQuADRetrieval (an MTEB task
  built from a Vietnamese reading-comprehension set,
  [dataset card](https://huggingface.co/datasets/mteb/VieQuADRetrieval/blob/main/README.md))
  **[licenses to check per set]**. Avoid machine-translated sets as the
  main Vietnamese evidence; VN-MTEB's data is translated
  ([arXiv 2507.21500](https://arxiv.org/abs/2507.21500)).
- Protocol: run synk on the corpus only (no labels), get its verdict; then
  import the human queries as real queries and get the human verdict.
  Repeat at several corpus sizes (sample 20, 50, 150, 500 documents) to
  show where the synthetic verdict becomes trustworthy, and with folds on
  vs off.
- Publish the numbers, the commands, and the failures. This also gives the
  data to set the thresholds in §3.8.

## 5. Risky or half-done code

| Where | What | Fix |
|---|---|---|
| `synk_sidecar.py:902-914` | GGUF export downloads llama.cpp's converter from GitHub with no consent item and no fetch name, then runs its Python. README's network list doesn't mention it. Breaks the AGENTS.md consent rule. | Make it a fetch name (`llama-cpp-converter@711780f`), listed by `export --dry-run`, downloaded only by `synk fetch` or `export --yes`; record a SHA-256 of the tarball and check it. |
| `generator.rs:476-484`, `dataset.rs:233` | Style examples are part of the LLM cache key and the pool shifts as real queries are added → silent full regeneration. | Freeze the pool; show the reason in the dry-run (§2.5). |
| `ops.rs:461`, `ops.rs:567` | Lock check-then-write race. | `create_new` (§3.3). |
| `ops.rs:36` + no signal handler | Ctrl-C → 120 s of `JOB_RUNNING`. | §3.1. |
| `sidecar.rs:428-432` | The torch install step has no `--exclude-newer` (the next step has it, `sidecar.rs:442`), so torch's own dependencies can resolve differently over time, though the comment says every install resolves the same. | Add `--exclude-newer` to the torch step, or install from a pinned lock file (`uv pip compile` output with hashes, shipped in the binary). |
| `dataset.rs:543` | `--sample` drops whole test documents. | §1.2c. |
| `dataset.rs:605` | Dry-run shows test chunks, not test documents. | §1.2a. |
| `train.rs:623` | 30 real queries from 2 documents decide the verdict. | Advice to spread them; consider requiring ≥ 5 documents. |
| `graph.rs:37-39`, `graph.rs:342` | Speed defaults don't depend on model; history keeps only the last value. | §2.1. |
| `export.rs:28-30` | q8_0 threshold unverified; the check compares 16 vectors, not retrieval quality. | Re-run the test-set eval on the exported model (local-rag-tooling.md #1) and report retention with an interval. |
| `readers.rs:119`, `lang.rs:110` | No NFC; mixed vi/en detection. | §3.4. |

## Prioritized list

Effort: S ≤ 3 days, M 1–3 weeks, L more, for one person who knows the code.

| # | Item | Why / evidence | User impact | Effort | Risk |
|---|---|---|---|---|---|
| 1 | Dry-run power warning + `test_documents` in the plan | Kaggle: 1 test document → no verdict possible; nothing warned before training (`dataset.rs:605`, `advice.rs:252`) | High: saves hours on doomed runs | S | Low |
| 2 | `synk queries template` + "spread over 10+ docs" advice | Real queries decide at 30 (`train.rs:623`); synthetic test sets are biased ([2405.07767](https://arxiv.org/abs/2405.07767v1)) | High: the most honest verdict, cheaply | S | Low |
| 3 | GGUF converter download behind consent + checksum | Unconsented download and code execution (`synk_sidecar.py:902`) | Medium (trust, rules) | S | Low |
| 4 | Ctrl-C handler + dead-pid detection | 120 s `JOB_RUNNING` after Ctrl-C (`ops.rs:36`) | High (everyone hits it) | S | Low |
| 5 | Disk-space check + `DISK_SPACE_LOW` | Several GB needed, nothing checks; Windows HF cache duplicates ([HF](https://huggingface.co/docs/huggingface_hub/guides/manage-cache)) | High on laptops | S–M | Low |
| 6 | Atomic project lock | Race (`ops.rs:461/567`) | Medium (agents retry) | S | Low |
| 7 | `--sample` keeps all test documents; `test_percent = "auto"` | Power depends on test documents (§1.1 table) | High on small corpora | S | Medium: dataset version bump re-keys datasets (queries stay cached) |
| 8 | NFC + Vietnamese detection rule | Owner's data; mixed vi/en notes | Medium–high for vi users | S | Medium: hash change re-asks the LLM once |
| 9 | Integration snippets in export output + README | LlamaIndex/LangChain/Open WebUI need explicit prefix settings (sources §2.2) | High: wrong prefixes silently lose quality | S | Low |
| 10 | Freeze the style pool | Silent full LLM regeneration (`generator.rs:484`) | Medium | S | Low |
| 11 | Permutation p-value alongside the t-test | Tests disagree at small n ([SIGIR 2009](https://uwaterloo.ca/data-systems-group/node/4913)) | Low–medium (clearer small-n messages) | S | Low |
| 12 | Remote-egress end-to-end test | The most sensitive promise is untested (plan §12) | Medium (trust) | S | Low |
| 13 | e2e CPU workflow; `run list --limit` | Known gaps | Low–medium | S | Low |
| 14 | Cross-validation over documents (`eval.folds`) | n = all documents instead of 20% ([JMLR 2004](https://www.jmlr.org/papers/v5/grandvalet04a.html) caveat) | High on small corpora | M | Medium: K× time; optimistic intervals must be stated |
| 15 | Honest benchmark: synthetic vs human verdict | Nobody publishes this; sets thresholds (§3.8) | High (credibility) | M | Low |
| 16 | Hard negatives: same-doc negatives with reranker; stronger miner option | `MANY_UNMINED_NEGATIVES` on Kaggle; NV-Retriever teachers ([2407.15831](https://arxiv.org/html/2407.15831v2)) | Medium | M | Medium: must win in `run compare` |
| 17 | Untuned baselines (`eval.baselines`, BM25) + `BASELINE_BETTER` | "Is tuning worth it?" answered with the same test | Medium–high | M | Low |
| 18 | Estimates: per-model defaults, per-sequence speed, `--measure` | GPU default ~1.7× low on Kaggle (derived) | Medium | S–M | Low |
| 19 | Vietnamese PDF spike with real files | Open since v0.1 | Medium for the owner | S (spike) | Low |
| 20 | Diacritic-free Vietnamese eval slice, then training option | Plausible user behaviour **[unverified]** | Medium for vi users | S then M | Low |
| 21 | Prompt: mix of query types; overlap statistic | Lexical-copy risk in synthetic queries | Medium | S | Low (PROMPT_VERSION bump) |
| 22 | granite-embedding-311m-multilingual-r2 in catalog | Apache 2.0, vi, no remote code ([card](https://huggingface.co/ibm-granite/granite-embedding-311m-multilingual-r2)) | Medium | S after a GPU run | Medium: GGUF support unverified |
| 23 | Windows + CUDA run (rented VM or volunteer) | Wheel exists ([index](https://download.pytorch.org/whl/cu130/torch/)), path untested | Medium (Windows RTX users) | M | Low |
| 24 | `export --refit` | Small corpora lose 20–50% of training docs | Medium | M | Medium: weaker promise, label clearly |
| 25 | Pinned trainer lock file (or `--exclude-newer` on torch) | `sidecar.rs:428` | Low–medium | S | Low |
| 26 | Homebrew / Scoop / winget | dist supports Homebrew/MSI only ([docs](https://axodotdev.github.io/cargo-dist/book/installers/index.html)) | Low | S each | Low |
| 27 | From local-rag-tooling.md: export retention check, `synk serve`, query adapters, `synk bench` | See that note | High but larger | M–L | Medium |

## Suggested order

### v0.2.0 — "honest on small corpora, and doesn't trip people up"

Theme: make the first run either succeed or say clearly, *before* spending
time, why it can't give a verdict.

1. Reliability quick wins: #4 Ctrl-C, #6 lock, #5 disk check, #3 GGUF
   consent, #10 style pool, #25 torch pin. Each has a unit or CLI test.
2. Verdict power: #1 dry-run warning, #7 sampling and `test_percent =
   "auto"`, #11 permutation p-value, #2 `queries template`. One dataset
   `KIND_VERSION` bump covers #7 and #8 together.
3. Vietnamese hygiene: #8 NFC + detection, #19 PDF spike (decide the PDF
   reader from it).
4. Use after export: #9 snippets.
5. Tests and ops: #12 egress test, #13 e2e workflow and `run list --limit`.
6. Release check: one Kaggle T4 run on synk's own docs with `--sample`
   (expect the dry-run to warn), plus one run with 30 real queries over
   10+ documents (expect a decisive verdict either way). Write the
   results into the release notes, including failures.

### v0.3.0 — "a verdict you can trust, and know when tuning isn't worth it"

1. #14 cross-validation over documents, then #24 refit.
2. #15 the public benchmark (synthetic vs human verdict); use its data to
   settle the thresholds in §3.8 and record them in an ADR.
3. #17 untuned baselines and BM25; #16 hard-negative changes, kept only if
   they win on the benchmark corpora.
4. #22 granite r2 trial; #20 and #21 query-generation experiments,
   judged on the benchmark.
5. #23 a Windows + CUDA run; drop "untested" from the README only after it.
6. Start the larger items from local-rag-tooling.md (retention check first,
   then `synk serve` with its opt-in query log, which feeds #2).

## Sources

Papers
- GPL — https://arxiv.org/abs/2112.07577v3 ; README — https://github.com/UKPLab/gpl
- Promptagator — https://arxiv.org/abs/2209.11755 ; with open LLMs — https://arxiv.org/abs/2510.02241
- NV-Retriever — https://arxiv.org/html/2407.15831v2
- SWIM-IR — https://arxiv.org/abs/2311.05800v2
- BEIR — https://arxiv.org/abs/2104.08663v1
- VN-MTEB — https://arxiv.org/abs/2507.21500 , https://arxiv.org/html/2507.21500v1
- Synthetic test collections — https://arxiv.org/abs/2405.07767v1 ; bias — https://arxiv.org/abs/2506.10301v2
- Smucker, Allan, Carterette, CIKM 2007 — https://ciir-publications.cs.umass.edu/getpdf.php?id=744 ; SIGIR 2009 follow-up — https://uwaterloo.ca/data-systems-group/node/4913
- Bengio & Grandvalet, JMLR 2004 — https://www.jmlr.org/papers/v5/grandvalet04a.html

Model cards and configs
- multilingual-e5-small — https://huggingface.co/intfloat/multilingual-e5-small
- bge-m3 — https://huggingface.co/BAAI/bge-m3
- Qwen3-Embedding-0.6B — https://huggingface.co/Qwen/Qwen3-Embedding-0.6B
- granite-embedding-311m-multilingual-r2 — https://huggingface.co/ibm-granite/granite-embedding-311m-multilingual-r2 ; config — https://huggingface.co/ibm-granite/granite-embedding-311m-multilingual-r2/raw/main/config.json
- gte-multilingual-base config — https://huggingface.co/Alibaba-NLP/gte-multilingual-base/raw/main/config.json
- AITeamVN/Vietnamese_Embedding — https://huggingface.co/AITeamVN/Vietnamese_Embedding
- EmbeddingGemma — https://ai.google.dev/gemma/docs/embeddinggemma

Docs and source
- Ollama OpenAI compatibility — https://docs.ollama.com/api/openai-compatibility
- Hugging Face cache (Windows symlinks) — https://huggingface.co/docs/huggingface_hub/guides/manage-cache
- PyTorch cu130 wheel index — https://download.pytorch.org/whl/cu130/torch/
- Open WebUI config.py — https://raw.githubusercontent.com/open-webui/open-webui/main/backend/open_webui/config.py ; RAG troubleshooting — https://docs.openwebui.com/troubleshooting/rag
- AnythingLLM embedder — https://docs.anythingllm.com/setup/embedder-configuration/overview
- LlamaIndex HuggingFaceEmbedding — https://raw.githubusercontent.com/run-llama/llama_index/main/llama-index-integrations/embeddings/llama-index-embeddings-huggingface/llama_index/embeddings/huggingface/base.py
- LangChain HuggingFaceEmbeddings — https://raw.githubusercontent.com/langchain-ai/langchain/master/libs/partners/huggingface/langchain_huggingface/embeddings/huggingface.py
- dist installers — https://axodotdev.github.io/cargo-dist/book/installers/index.html
- GitHub GPU runners — https://github.blog/changelog/2024-07-08-github-actions-gpu-hosted-runners-are-now-generally-available
- Unicode normalization (UAX #15) — https://unicode.org/reports/tr15/
- VieQuADRetrieval — https://huggingface.co/datasets/mteb/VieQuADRetrieval/blob/main/README.md
- Databricks embedding fine-tuning — https://www.databricks.com/blog/improving-retrieval-and-rag-embedding-model-finetuning
- Chroma embedding adapters — https://www.trychroma.com/research/embedding-adapters
