# Changelog

## Unreleased (v0.2.0): honest on small corpora, and doesn't trip people up

**Upgrading:** the first `synk train` after upgrading rebuilds the dataset and
trains again, once: the test split, sampling and text normalization changed.
Generated queries stay cached, so the LLM isn't asked again (except for the
few chunks whose detected language changed). Existing trainers keep working;
`synk setup --force` moves to the new pinned one.

### Results you can trust on small projects

- `train --dry-run` shows how many **test documents** the run will have, and
  warns (`TEST_SET_UNDERPOWERED`, `evidence.predicted`) when that's too few to
  see a typical gain, before any training.
- `dataset.test_percent = "auto"` (the new default) holds out at least 10
  documents where possible (20%–50%); projects with ~50+ documents keep
  exactly the old 20% split. `--sample` keeps every test document.
- A `no_clear_change` verdict says why: `reason` is `no_difference` or
  `too_few_test_documents` (inconclusive).
- The verdict reports an exact sign-flip p-value next to the t-test
  (`permutation_p`).
- **`synk queries template`** writes a file of documents to fill in with
  your own searches. Real searches decide the verdict at 30+ about 10+
  documents; `agreement` and new advice (`REAL_QUERIES_FEW_DOCUMENTS`,
  `SYNTHETIC_REAL_DISAGREE`) compare them with the generated test.
- Style examples for the LLM are chosen once and kept, so importing more
  real searches no longer re-asks the LLM for every chunk
  (`queries import --restyle` chooses again).

### Reliability

- **Ctrl-C** stops at the next safe point (the trainer saves a checkpoint);
  a second Ctrl-C stops at once. Exit code 130, one `CANCELLED` result with
  `details.how`; `synk train` continues the run.
- The project lock is an OS file lock: two commands started together can't
  both run, a killed run reads `interrupted` at once, and a laptop that slept
  keeps its run.
- The trainer stops when synk stops (a stdin lifeline), and runs in its own
  process group.
- **Disk space** is checked before downloads and training (`data.disk`,
  `INSUFFICIENT_DISK`, `DISK_SPACE_TIGHT`; `SYNK_DISK_CHECK=warn`), and a
  full disk mid-run is reported as `INSUFFICIENT_DISK`.
- GGUF export ships llama.cpp's converter inside synk: no download (it used
  to fetch the whole repository without asking).
- The trainer installs from hashed lock files: the same packages every time.
- The query generator skips a chunk the LLM fails on (e.g. Ollama's "token
  repeat limit reached") instead of failing the run, and passes on the
  server's error text; Ollama's model is unloaded before training so the GPU
  is free; one GPU is used even when several are visible; fp16 on GPUs
  without native bf16 (T4, RTX 20).
- Every `next` hint runs as written.

### Vietnamese and mixed text

- Text is normalized to Unicode NFC when read (files, imported and generated
  queries); the first sync after upgrading reads every file again once
  (`reread`).
- Vietnamese is recognized by its own letters before the general detector,
  so mixed Vietnamese/English notes are no longer labelled French.

### Using the model

- `synk export --json` prints copy-paste setup that keeps the model's
  query/document prefixes for sentence-transformers, LlamaIndex, LangChain,
  Open WebUI, Ollama and AnythingLLM (`data.usage`).

## v0.1.0 (2026-10-16)

First release. Trains on NVIDIA GPUs (CUDA) or the CPU.
