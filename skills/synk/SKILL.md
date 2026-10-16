---
name: synk
description: Fine-tune a small embedding (search) model on a user's own documents, locally, and prove with a statistical test whether it beats the original. Use when the user wants better RAG or semantic search over their notes, docs or wiki, mentions embedding models, fine-tuning, retrieval quality, or "synk".
---

# synk

synk is a CLI. Every command takes `--json` and prints exactly one object on
stdout: `{"v": 1, "ok": true, "data": ...}` or `{"v": 1, "ok": false, "error": ...}`.
Exit code is 0 exactly when `ok` is true (2 for usage errors). Parse JSON,
never the human text. Progress and heartbeats go to stderr.

## The loop

```
synk init                                   # once, creates synk.toml
synk source add <folder>                    # the user's documents
synk train --dry-run --json                 # read-only: what would run, what it needs, how long
# -> data.fetch_names / data.egress not empty? show the user sizes and hosts, and ask
synk fetch <the names> --json               # after the user agrees: downloads exactly those
synk train --json                           # a run: sync, queries, train, test
synk export <run> --out ./model --json      # when data.verdict.outcome is "better"
```

- **Ask the user before downloading or sending text away.** `synk train`
  without `--yes` never does either: it fails with `CONSENT_REQUIRED`, and
  `error.details.fetch` lists the names (trainer ~1-4 GB, LLM ~2-5 GB, models
  0.1-2 GB; `details.downloads` has sizes). After the user agrees, run
  `synk fetch <names>` (exactly what they saw), then `synk train` again.
  Never pass `--yes` without the user's agreement; with a remote LLM it also
  sends document text off the machine.
- Read-only: `train --dry-run`, `dataset --dry-run`, `status`, `wait`, `logs`,
  `run`, `model`, `schema`, `config get|show`. Downloading: `fetch`, `setup`,
  `train --yes`. Suggested harness rules: allow `synk train --dry-run *`,
  `synk status *`, `synk wait *`, `synk logs *`, `synk run *`; ask for
  `synk fetch *`, `synk setup *`, `synk train *`.
- Every `synk train` is a run with an ID like `run:3-e5-small` (or
  `run:4-bge` with `--name bge`). The first stderr line is
  `{"event": "op", "id": ...}`. Commands take `run:3`, `3` or the name.
- `train` blocks until done (minutes to hours). If your shell times out or
  backgrounds it: `synk wait --timeout 9m --json` (repeat while `data.done`
  is false), `synk logs --after <seq>`, `synk cancel`.
- Retrying is safe: `synk train` with nothing changed returns the same run
  (`data.reused: true`); an interrupted or cancelled run continues
  (`data.op.attempt` + 1). `synk train run:3` continues run 3 with the
  settings it started with.

## Reading results

- `data.verdict.outcome`: `better`, `worse` or `no_clear_change`. A verdict
  that isn't `better` is a normal result, not an error. Explain it to the
  user with `data.advice` (each has `code`, `message`, `applicability`, and
  often `action`).
- `synk run list --json` lists every run with its state, verdict and the
  settings that differ from synk.toml (`differs`).
- `synk run inspect <run> --json` shows the queries that improved or got worse,
  with the chunk that should have come first and the one that did.
- `synk run compare <a> <b> --json`: is run B better than run A?
- `synk run report <run> --out report.html` writes a page for the user.

## Changing things

- `synk config set <key> <value>` edits synk.toml safely (unknown keys and
  invalid values are refused). `synk config show --json` lists every key.
- `synk train --set key=value --name NAME` tries a setting as a new run
  without editing synk.toml, e.g. `--set model.base=bge-m3 --name bge`; then
  `synk run compare <a> bge`. Preview it first with `--dry-run`: a new base
  model usually needs a fetch.
- Only follow an advice `action` or error `actions` entry automatically when
  its `applicability` is `machine`. `review` needs a look; `manual` needs the
  user.

## Errors

Branch on `error.code`; for codes you don't know, on `error.class` (`usage`,
`config`, `not_found`, `conflict`, `prerequisite`, `unavailable`, `data`,
`resource`, `failed`, `internal`). `error.retryable` says whether the same
command may succeed later. `error.next`, when present, runs as is.
`error.actions` lists other fixes. See `references/codes.md`, or
`synk schema --json`.

## Exporting

- `--format st` (default): a sentence-transformers folder. Use the model's
  prompts: `encode(texts, prompt_name="query")` for searches,
  `prompt_name="document"` for documents.
- `--format onnx`: for ONNX Runtime.
- `--format gguf --ollama NAME`: creates NAME in the local Ollama, for apps
  like Open WebUI and AnythingLLM. Ollama doesn't add the query/document
  prefixes (advice `PREFIX_DROPPED_BY_TARGET`); tell the user.
- Every export checks that the exported model's vectors match the original.
