# Plan: fetch, then train (design D)

Status: **built** 2026-10-16 (see §12 for where the build differs). Date: 2026-10-16. Replaces the plan/apply/job surface of
[v1-design.md](v1-design.md) §5.
Background: run-identity-and-approval.md,
design D plus §4 there.

## 0. In one paragraph

A person or an agent types `synk train`. If everything it needs is already on the machine, it runs and prints one ID: `run:3-e5-small`. If something must be downloaded first, it changes nothing and says exactly what to fetch: `synk fetch multilingual-e5-small qwen3:8b general-check`. Running that command is the approval. `--yes` approves and runs in one go. Trying a variant is `synk train --set model.base=bge-m3`, and the result is another run. `synk run compare 3 4` says which variant won. While a run is going, `synk wait`, `synk logs` and `synk cancel` act on it without needing its ID. Plan IDs, `apply` and job IDs go away.

## 1. Goals and non-goals

**Goals**
- One ID per attempt, the run, readable and short enough to say aloud ("run 3 vs run 4").
- No step that only exists to produce a token. Approval is either the literal list of names being downloaded, or `--yes`.
- No "plan is stale" loop.
- Several variants are easy to try and compare.
- The agent contract stays strict: one stdout object, `next` always runnable, no prompts, progress on stderr.

**Non-goals (v1)**
- A queue or batches running in the background (design C). It can be added on top later.
- Pending runs waiting for approval (design B).
- `synk config set --from RUN`. Later, but the run record keeps its settings so it can be added.

## 2. The command surface

```
Work:
  status    Where the project stands, and what to do next
  train     Sync, write queries, train and test: one run
  dataset   Sync and build the training data only
  fetch     Download what training needs: models, the LLM, test sets, the trainer
  wait      Wait for the current run to finish
  logs      Progress and log lines of a run
  cancel    Ask the current run to stop

Results:
  run       List, show, inspect, compare and report on runs
  export    Save a tuned model: a folder, ONNX, or GGUF for Ollama

Config:   init, source, config, queries        (unchanged)
Machine:  setup, model                          (setup kept; see §4.4)
Reference: schema, skill                        (unchanged)
```

Removed: `plan`, `apply`, `job` (all subcommands).

### 2.1 `synk train`

```
synk train [RUN] [--set K=V]… [--name NAME] [--sample N] [--force STEP]
           [--dry-run] [--yes]
```

| Form | What it does |
|---|---|
| `synk train` | Runs sync → dataset → train → eval with `synk.toml`. It downloads nothing and sends nothing off the machine unless `--yes` is given. |
| `synk train --set model.base=bge-m3 --name bge` | Same, with overrides for this run only. `--name` must be unique in the project. |
| `synk train --dry-run` | Read-only preview: the steps and why each runs, what is missing as one `fetch` command, any remote sends, and an estimate. Mints nothing. This replaces `synk plan`. |
| `synk train --yes` | Fetches what is missing, then runs. Remote sends also need `generator.allow_remote = true` (§4.2). |
| `synk train run:3` | Continues run 3 with the settings it was **started with**, ignoring later edits to `synk.toml`. If run 3 already succeeded, it returns run 3 (`reused: true`). `--set`/`--name` with `RUN` is `INVALID_ARGS`. |
| `--force STEP` | Kept from today (`dataset`, `train`, `eval`): redo that step even when its result exists. |
| `--sample N` | Kept: a quick dataset from at most N chunks. |

`--target` goes. Its uses become verbs: `--target dataset` is `synk dataset`; `--target models|generator|trainer` is `synk fetch`. `--target sync` has no replacement, because `status` shows what changed and `dataset` syncs.

### 2.2 `synk dataset`

```
synk dataset [--set K=V]… [--sample N] [--force dataset] [--dry-run] [--yes]
```
Sync plus the dataset step. It doesn't create a run (a run is a training attempt; see the glossary). It does create an **operation** so that `wait`, `logs` and `cancel` work (§3.4). Its result names the dataset: `dataset:2-qwen3-8b`.

### 2.3 `synk fetch`

```
synk fetch [NAME…] [--set K=V]… [--backend B]
```
| Form | What it does |
|---|---|
| `synk fetch NAME…` | Downloads exactly these items and nothing else. |
| `synk fetch` (no names) | Downloads everything `synk train` would need right now with the current settings and `--set`. This is the same list `train --dry-run` prints. Running the command is the consent, as with `synk setup` today. |

Item names, as `train --dry-run`, `status` and `CONSENT_REQUIRED` print them:

| Item | Name form | Example |
|---|---|---|
| Base model or reranker from the catalog | catalog ID, with `@rev` | `multilingual-e5-small@614241f` |
| Any Hugging Face model | `hf:org/name@rev` | `hf:BAAI/bge-m3@5617a9f` |
| LLM in Ollama | `ollama:TAG` | `ollama:qwen3:8b` |
| The general check's public test sets | `general-check` | `general-check` |
| The Python trainer | `trainer` | `trainer` |

- **The revision is part of the name** when synk printed it. This binds the approval: `fetch` downloads that commit, so the thing approved is the thing fetched. Without `@rev`, `fetch` pins the way `train` would.
- `fetch` is idempotent. Items already present are reported as `present`. Hugging Face and Ollama downloads resume where they stopped.
- `fetch` runs as an **operation** (§3.4), so a 5 GB download can be waited on, followed and cancelled like a run.

### 2.4 `wait`, `logs`, `cancel`

```
synk wait [ID] [--timeout 9m]      # default: the current operation, else the latest
synk logs [ID] [--after SEQ] [--limit N]
synk cancel [ID]                   # default: the current operation
```
- `ID` is a run (`run:3`, `3`, a name, `latest`), a dataset, or `fetch:N`.
- These replace `job wait|logs|cancel` with the same semantics: running out of time is `ok: true, done: false`, with a `next` that waits again.
- These three are top-level because they act on "whatever is running", and only one thing runs per project. `run wait` is not added: one way to do each thing.

### 2.5 `run …` (results, now also covering unfinished runs)

```
synk run list [--limit N] [--all]      # every run: number, name, state, base model, verdict, settings that differ
synk run show [RUN]                    # settings, state, steps, verdict, advice
synk run inspect | compare A B | report     (unchanged)
```
- `run list` reads run records (§3), so failed, cancelled, interrupted and running runs are listed too.
- `--all` includes runs whose artifacts were cleaned up.
- A new column, "settings that differ", lists the overrides and changed config keys against the project's current `synk.toml`. This is DVC `exp show`'s best idea, and the point of having variants.

## 3. Runs: IDs, records, lifecycle

### 3.1 ID format

`run:<n>-<label>`, for example `run:3-e5-small`, or `run:4-bge` with `--name bge`.
- `n` is a per-project sequence. It is allocated under `.synk/lock`, so there are no races.
- `label` is the slug of `--name` when given, otherwise the slug of the base model. That is today's `ids::label` logic.
- Datasets: `dataset:<n>-<generator>`, for example `dataset:2-qwen3-8b`. Fetches: `fetch:<n>`. Each has its own sequence.
- **Lookup** accepts:
  - `run:3-e5-small`, `run:3`, `3`;
  - the name (`bge`);
  - `latest` and `best`;
  - a unique artifact-key prefix (as today).
  - A bare number always means the sequence number.
- The artifact `key` stays in every output next to the ID (AGENTS.md rule).
- Today's date-based IDs (`run:2026-10-15-e5-small-3f2a`) **stop being printed** but still resolve, through the migration (§3.5).

### 3.2 The run record

Today's `JobRecord` becomes the run record, stored in `.synk/runs/<n>/run.json` with `events.jsonl` and `work/` beside it. Fields:

```jsonc
{
  "v": 1,
  "id": "run:3-e5-small", "n": 3, "name": null,
  "state": "running",            // running | succeeded | failed | cancelled | interrupted
  "done": false,
  "attempt": 1,                  // +1 each time `train` continues it
  "started_at": 0, "finished_at": null, "heartbeat_at": 0, "pid": 0,
  "settings": {                  // frozen at start; what `train RUN` reuses
    "overrides": {"model.base": "bge-m3"},
    "config": { /* effective PipelineConfig */ },
    "config_hash": "…", "sample": null, "force": null,
    "base_model": {"name_or_path": "…", "revision": "…"}   // pinned
  },
  "settings_hash": "…",          // hash of `settings`; matches retries (§3.3)
  "dataset": {"id": "dataset:2-qwen3-8b", "key": "…"},    // once known
  "key": "…",                    // the train artifact key, once known
  "steps": [ /* as JobRecord.steps */ ],
  "progress": null,
  "result": { /* verdict, general check, advice, next */ },
  "error": null, "reason": null,
  "same_as": null                // §3.3, rare
}
```

Operations of the other kinds (`dataset:N`, `fetch:N`) use the same struct with `"kind": "dataset" | "fetch"`, in `.synk/ops/<kind>-<n>/`. `jobs.rs` is renamed to `ops.rs` and keeps its machinery: lock, heartbeat, cancel file, events, `settle()` to interrupted, pruning.

### 3.3 When `synk train` makes a new run, and when it doesn't

The goal: retries must not create clones (research §3: "idempotent creation").

1. **`train RUN`**: always that run. It continues if unfinished (`attempt += 1`) and returns it if succeeded.
2. **`train` (no ID), and the latest run with the same `settings_hash` is unfinished** (interrupted, failed, cancelled): continue that run (`attempt += 1`). This replaces "apply the same plan again to resume".
3. **`train`, and the graph shows train and eval `cached`** (identical inputs, already done): return the run that owns that train key, with `reused: true`. No new number is minted.
4. **Otherwise**: mint a new run.
5. **Rare late match**: a new run whose train key turns out to equal a finished run's key, because the dataset came out identical. It finishes as `succeeded` with `same_as: "run:2-…"` and copies its result. The number is not reused. This is documented, not hidden.

`settings_hash` covers the overrides, the effective config and `--sample`. It doesn't cover documents. So a run interrupted while documents changed continues on the current documents. That is the same as applying again today.

### 3.4 One thing at a time

`.synk/lock` holds the ID of the running operation (a run, `dataset:N` or `fetch:N`). Starting another one fails with `JOB_RUNNING`. The code is kept for compatibility, and its message now names the run, with `next: synk wait`. `status` shows the current operation.

### 3.5 Migration of existing data (pre-release, but cheap to do right)

The first time a command opens the runs store and `.synk/runs/` is missing, it mints a record for each existing train artifact:
- in `created_at` order, numbering 1..k;
- `state: succeeded`;
- `settings` copied from the manifest params;
- `key` set.

Old `.synk/jobs/` and `.synk/plans/` are left alone; `status --verbose` mentions that they can be deleted. This is not a SQLite migration, because records are files, as jobs are today.

## 4. Consent

### 4.1 Rules

| Action | Downloads | Sends text off the machine |
|---|---|---|
| `synk fetch NAME…` | exactly those names | never |
| `synk fetch` | what `train --dry-run` lists now | never |
| `synk setup` | the trainer env | never |
| `synk train` / `synk dataset` | **never**; a missing item → `CONSENT_REQUIRED` | **never** without `--yes` |
| `synk train --yes` / `synk dataset --yes` | whatever is missing | yes, if `generator.allow_remote = true` |
| `synk train --dry-run` | never (asks HF/Ollama for sizes, sending only model names, as `plan` does today) | never |
| `synk export --ollama` | never (local Ollama) | never |

- Remote sends keep **two locks**: the config opt-in and `--yes` on each run. This is apt's carve-out (research §2.3), and it is the deliberate weak spot of D. The `--dry-run` result shows the host and the chunk count. The run record stores the egress actually done (`result.egress: {host, chunks}`), so it can be audited afterwards.

### 4.2 `CONSENT_REQUIRED` from `train`/`dataset`

```jsonc
{"v":1,"ok":false,"error":{
  "code":"CONSENT_REQUIRED","category":"prerequisite","retryable":false,
  "message":"needs downloads first (5.7 GB): multilingual-e5-small, qwen3:8b, general-check",
  "details":{
    "fetch":["multilingual-e5-small@614241f","ollama:qwen3:8b","general-check"],
    "downloads":[{"name":"multilingual-e5-small@614241f","bytes":470000000,"host":"huggingface.co"}, …],
    "egress":[],                       // or [{"host":"api.example.com","chunks":412}]
    "dry_run":{ /* the full --dry-run data */ }
  },
  "actions":[
    {"argv":["synk","fetch","multilingual-e5-small@614241f","ollama:qwen3:8b","general-check"],
     "applicability":"manual","why":"downloads exactly these; then run synk train"},
    {"argv":["synk","train","--yes"],"applicability":"manual","why":"downloads and runs in one go"}
  ],
  "next":null}}
```
- `next` is null: a download is never `machine` (AGENTS.md).
- When only egress is pending, the actions are `synk train --yes` alone, plus a hint about `generator.allow_remote`.
- The plain-text output ends with the two commands, as in research §5 D's transcript.

### 4.3 What stays from today's graph

`graph::make` stays as the single source of truth for "what is missing, what is stale, what it costs". It is renamed in place, not rewritten:
- `Plan` → `Preview`. It loses `id`, `fingerprint` and `save`/`load`, and gains `fetch_names: Vec<String>`.
- `Consent` items are unchanged; each one gains a `name` (the fetch name).
- `Options.target` stays internally. `train` uses `Eval`, `dataset` uses `Dataset`, and `fetch` uses a new internal `Fetch` target meaning "trainer + models + generator".

### 4.4 `setup` vs `fetch trainer`

Both exist:
- `synk fetch trainer` installs the trainer with the auto-detected backend, which is what `train --yes` does.
- `synk setup [--backend B] [--force]` stays as the explicit machine command, for choosing or reinstalling a backend.
- `fetch --backend B` is passed through when `trainer` is among the items.

This avoids breaking `setup`, which the README, the installer docs and the skill already use.

## 5. Output contracts

### 5.1 `synk train --json` (success)

```jsonc
{"v":1,"ok":true,"data":{
  "run":{ /* the run record, §3.2 */ },
  "id":"run:3-e5-small","key":"…",
  "reused":false,
  "dataset":"dataset:2-qwen3-8b",
  "verdict":{…},"general_check":{…},"advice":[…],"sync":{…},
  "next":"synk export run:3-e5-small --out ./e5-small-model"}}
```
- The first stderr progress event is `{"v":1,"event":"run","id":"run:3-e5-small","attempt":1}`. An agent whose shell is backgrounded still learns the ID.
- Then come the step, progress, log and heartbeat events, as today.

### 5.2 `synk train --dry-run --json`

Today's plan JSON minus `id`, `fingerprint` and `next: synk apply …`, plus:
- `fetch` (the argv string);
- `fetch_names`;
- `would_create`: `"new run"`, `"continues run:3-e5-small"` or `"reuses run:2-e5-small"`;
- `next`: when nothing is missing, `synk train` (it spends nothing). Otherwise null, with the fetch action in `actions`.

### 5.3 `synk wait --json`

Like `job wait` today, with `id` meaning the run, dataset or fetch ID:
```jsonc
{"id":"run:3-e5-small","kind":"run","state":"running","done":false,"attempt":1,
 "progress":{…},"next":"synk wait --timeout 9m"}
```

### 5.4 Codes

**Error codes.** No variant is removed (add-only rule):
- `PLAN_STALE` and `PLAN_NOT_FOUND` were removed (never released).
- `JOB_RUNNING` stays and is still produced when an operation holds the lock. Its message says "run" or "fetch".
- New: `RUN_NAME_TAKEN` (Conflict), for `--name` already used.
- New: `RUN_NOT_RESUMABLE` (Conflict), for `train RUN` on a run whose base model or trainer can no longer be pinned.

**Advice codes.** Any advice whose `command` was `synk plan` becomes `synk train --dry-run`. Advice with a `--set` suggestion now suggests `synk train --set …`, a variant, instead of "edit, then plan".

## 6. Code changes, file by file

**`crates/synk-pipeline`**

| File | Change |
|---|---|
| `graph.rs` | `Plan` → `Preview`. Drop the ID, fingerprint, `save`/`load` and `PLAN_RETENTION_DAYS`. Add a fetch `name` to `Consent` and `fetch_names` to `Preview`. Add an internal `Target::Fetch`. Keep `covers_consent_of` (used by `fetch` to check names against needs). |
| `jobs.rs` → `ops.rs` | `JobRecord` → `OpRecord` with the `kind`, `n`, `name`, `attempt`, `settings`, `settings_hash`, `key`, `dataset` and `same_as` fields of §3.2. The sequence allocator works under the lock. `get()` resolves numbers, names, `latest`, `current` and prefixes. Add a `migrate_legacy_runs()` (§3.5). |
| `apply.rs` → `exec.rs` | `run(ws, preview, op, render)`. The trainer, models and generator steps run only if the caller passed `allow_downloads`. Otherwise `make` reports them as missing and the CLI refuses before starting. Also: record egress, and fill in the run record's `dataset` and `key` as they become known. |
| `fetch.rs` (new) | Parse fetch names. Map them to `Fetch` entries (HF), Ollama pulls, the general-check sets and the trainer. Execute them as a `fetch:N` op, reusing the code moved out of `apply.rs` (Models/Generator/Trainer arms). |
| `ids.rs` | Readable IDs come from op records (`run:<n>-<label>`). Keep the old date-based renderer only to resolve legacy input. `resolve()` checks op records first, then artifacts. `export_command` takes the new format. |
| `train.rs` | Hints: "apply the plan again" → "run `synk train run:N` to continue". The run key logic is unchanged. |
| `advice.rs` | Command strings (§5.4). |

**`crates/synk`**

| File | Change |
|---|---|
| `main.rs` | New `Train`, `Dataset`, `Fetch`, `Wait`, `Logs`, `Cancel` commands. Remove `Plan`, `Apply` and `Job`. Update `HELP_TEMPLATE` (the test keeps it in sync). Update the `after_help` examples. |
| `work.rs` | `plan`/`apply` → `train`, `dataset` and `dry_run` around one shared `start(kind, …)`. `job_*` → `wait`, `logs`, `cancel`. `render_plan` becomes `render_preview`, which ends with the fetch command. The run-reuse rules of §3.3 live here. |
| `views.rs` | `run list`/`show` read op records and add the state and "differs" columns. |
| `commands.rs` | `status`: show the current op; `next` is `synk wait`, `synk train --dry-run`, or the `fetch` command. `source add` → `next: synk train --dry-run`. |
| `pipeline.rs` | `setup` → `next: synk train --dry-run`. The model-status text mentions `synk fetch`. |
| `reference.rs`, `skill/SKILL.md` | New flow; the agent recipe is in §8. |

## 7. Tests

These rewrite the 6 plan/apply/job tests in `crates/synk/tests/cli.rs`, using the fake trainer and fake LLM.

1. **`train_runs_everything_and_reuses_it`**: `train --yes` returns `run:1-…`. A second `train` returns the same run with `reused: true` and no new number.
2. **`variants_get_their_own_runs`**: `train --set trainer.epochs=2 --name e2` gives `run:2-e2`. `run list` shows both with the differing setting. `run compare 1 e2` works.
3. **`downloads_need_fetch_or_yes`**: on an empty cache, `train` → `CONSENT_REQUIRED` with `details.fetch` and two manual actions, and no run minted. Then `fetch <those names>` → `train` runs without `--yes`.
4. **`fetch_names_bind_revisions`**: `fetch hf:org/x@abc` fetches that revision. A malformed name → `INVALID_ARGS`.
5. **`dry_run_changes_nothing`**: no op record, no run number, no store writes. Compare the file tree before and after.
6. **`runs_can_be_waited_cancelled_and_continued`**: start, `cancel`, state `cancelled`. `train` (same settings) continues the same run with `attempt: 2`. `train run:1` after editing `synk.toml` uses the frozen settings.
7. **`one_operation_at_a_time`**: a running `fetch` blocks `train` with `JOB_RUNNING`, `next: synk wait`.
8. **`remote_sends_need_yes`**: an egress-only preview: `train` refuses, `train --yes` with `allow_remote` runs, and the record has `result.egress`.
9. **`legacy_runs_are_numbered`**: a project with old artifacts and no `.synk/runs/` → `run list` shows `run:1`, `run:2`, and old date IDs still resolve.
10. **`dataset_only`**: `dataset --yes` → `dataset:1-…`, no run. `wait` works on it.
11. Update `existing_commands_keep_their_fields`, `schema_and_skill_describe_the_contract` and the ignored real e2e test (`train --yes`, then the variant, then compare).
12. Unit tests: `ops.rs` (sequence under the lock, resolve by number, name and prefix, migration) and `fetch.rs` (name parsing).

## 8. Agent and human flows (for SKILL.md and the README)

**Human**
```
$ synk init && synk source add ./notes
$ synk train
Needs downloads first (5.7 GB): synk fetch multilingual-e5-small@614241f ollama:qwen3:8b general-check
Or download and run in one go: synk train --yes
$ synk train --yes
run:1-e5-small   sync → queries (25 min) → train (11 min) → test
run:1-e5-small: better (nDCG@10 0.64 → 0.78)
Next: synk export run:1-e5-small --out ./e5-small-model
$ synk train --set model.base=bge-m3 --name bge --yes
$ synk run compare 1 bge
```

**Agent**
```
synk train --dry-run --json            # read-only; safe to allow
  → data.fetch / data.egress non-empty?  ask the user, quoting sizes and hosts
synk fetch <exact names> --json        # what the user approved
synk train --json                      # first stderr event carries the run ID
  (shell times out) → synk wait --timeout 9m --json   until done
synk run show <id> --json → follow next
```
Suggested harness rules for the skill:
- allow `synk train --dry-run *`, `synk status *` and `synk run *`;
- ask on `synk fetch *`, `synk setup *` and `synk train *`.

`synk train` without `--yes` can't spend. "Ask" is still the safe default, because prefix rules can't see `--yes` placed later in the command.

## 9. Docs to update

- **ADR 0002** "Fetch, then train; the run is the only handle" (now ADR 0002).
- **v1-design.md**: TL;DR, §3, §5.1–5.5 and §6.3 storage. Add a pointer to this plan in the Progress section.
- **GLOSSARY.md**:
  - remove Plan and Job;
  - add Fetch, Dry run and Operation (internal; "a run, a dataset build or a fetch");
  - Training run gains the states, the number and the name;
  - Consent is redefined.
- **AGENTS.md**:
  - consent: "running `synk setup`/`synk fetch`, or `--yes`";
  - "`plan` must stay read-only" → "`--dry-run` must stay read-only";
  - the layout entries for `ops.rs`, `exec.rs` and `fetch.rs`.
- **README.md**: quick start, the command table, the error-code table (retired codes, the two new ones), and the consent section.
- **`synk skill`**: the flow and rules of §8.

## 10. Order of work

Each step leaves `cargo test` and clippy green.

1. **Ops store.**
   - Rename `jobs.rs` → `ops.rs`.
   - Add the sequence, kinds, settings, and resolve by number and name.
   - Add the migration.
   - Unit tests.
   - The CLI still uses it through `apply`.
2. **Preview.** `Plan` → `Preview`, with no ID or saving, and fetch names on consent items. Unit tests on names.
3. **`fetch`.** `fetch.rs`, the `fetch` command, and the `fetch:N` op. Tests 4 and 7.
4. **`train` / `dataset` / `--dry-run`.** Replace `plan`/`apply`, with the reuse rules of §3.3 and `CONSENT_REQUIRED` with fetch actions. Tests 1, 2, 3, 5, 8 and 10.
5. **`wait` / `logs` / `cancel`.** Remove `job`. Test 6.
6. **Views and status.** `run list`/`show` from records, the differs column, and status. Test 9.
7. **Text sweep.** Every `synk plan`/`synk apply` string in advice, hints, setup and source add. `grep -rn "synk plan\|synk apply\|synk job"` must come back empty outside the retired code table.
8. **Docs** (§9), plus the skill and README. Tests 11 and 12.
9. **Real e2e** with the cpu env: `train --yes`, a variant, compare, then export. Then a manual check in the scratchpad `gguf/` project.

Rough size: steps 1–6 are most of it, about 1,500 changed lines, half of them tests. **[unverified]**

## 11. Risks and open points

| Risk | Mitigation |
|---|---|
| Remote LLM users have to pass `--yes` on every run | Intentional (two locks). `--dry-run` shows host and chunk count, and `result.egress` records what was sent. Revisit if users complain. |
| `fetch` without names is a blanket approval | Same as `synk setup` today. The names are printed before anything downloads, and `--dry-run` shows them first. |
| Number gaps from a late `same_as` match | Rare. Documented. The number isn't reused. |
| Continuing a run on changed documents | Same as today's apply-again. The run's `attempt` and steps show it. |
| Harness rules can't see `--yes` mid-command | The skill recommends "ask" on `synk train *`. |
| Bigger change before first release | The `--json` contract is not released yet. After this, it freezes at release. |

**Decisions I made that you may want to change**
1. **`dataset` stays a top-level verb.** The alternative is `train --until dataset`, which would make a run that has no model.
2. **`setup` stays** beside `fetch trainer`.
3. **`wait`/`logs`/`cancel` are top-level**, not `run wait`, because they also cover fetches and dataset builds.
4. **Run numbers are per project**, starting at 1, and aren't reused after old records are pruned.

## 12. As built

All of §2–§9 is built, with these differences:
- **Where records live.** Every operation is in `.synk/ops/<kind>-<n>/op.json`
  (runs, dataset builds and fetches alike), with `seq.json` (numbers ever
  handed out) and `datasets.json` (dataset key → number). `jobs.rs` became
  `ops.rs`, `apply.rs` became `exec.rs`.
- **`train --yes`** runs the downloads as a `fetch:N` operation first, then
  makes a fresh preview and starts the run. `exec.rs` no longer downloads
  anything; a preview that still lists a download there is refused.
- **`synk source sync`** was added: tests and agents need to read sources
  without building a dataset, and `--target sync` is gone.
- **Dataset builds** use the dataset's number: `synk dataset` on a dataset
  that already has a record continues that record (`attempt` + 1) instead of
  making a second `dataset:N`.
- **`run list`** has no `--limit`/`--all` yet; it lists every run.
- **Advice** whose action is "set these settings" now gives a runnable `next`:
  `synk train --dry-run --set key=value`, a read-only look at the variant.
- **Not tested end to end:** a remote LLM's egress (`result.egress`); there's
  no fake remote server in the tests. The code path is the same `--yes` and
  `allow_remote` checks as before.
- **Verified for real** (scratchpad project, qwen3:4b, CPU trainer): an old
  run numbered `run:1-e5-small` by the migration; a variant
  `--set trainer.epochs=2 --name e2` became `run:2-e2`, and `run list` showed
  `differs from synk.toml: trainer.epochs=2`; `synk train` with nothing
  changed returned run 1; a missing Hugging Face model stopped with
  `CONSENT_REQUIRED` and the exact `synk fetch hf:…@1110a24`, which then
  downloaded it as `fetch:1`; a wrong `@commit` was refused.
