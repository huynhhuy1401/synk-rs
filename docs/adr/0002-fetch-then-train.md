# Fetch, then train; the run is the only handle

Status: accepted, 2026-10-16. Replaces an earlier plan/apply/job design.

Each attempt now gets one ID that people keep: the run, `run:<n>-<label>`
(`run:3-e5-small`, or `run:4-bge` with `--name bge`), numbered per project.
`synk train` runs sync, dataset, train and test with `synk.toml` plus any
`--set` overrides, and never downloads or sends document text off the
machine on its own. When the preview lists a download, it stops with
`CONSENT_REQUIRED` and the exact `synk fetch <names>` command; running that
command is the consent, or `--yes` gives it in one go. Sending text to a
remote LLM also needs `generator.allow_remote = true`. `train --dry-run` is
the read-only preview. `wait`, `logs` and `cancel` act on the one operation
running in the project (a run, a dataset build, or a fetch) without an ID.
`train RUN` continues a run with the settings it started with.

Plan IDs, `apply`, and job IDs are gone. See
[fetch-then-train.md](../plans/fetch-then-train.md) and
run-identity-and-approval.md,
design D.

## Why

- One attempt showed three IDs (`plan:`, `job:`, `run:`). Only the run is
  something people talk about later; every tool studied (MLflow, W&B, DVC,
  Nextflow, Docker, GitHub runs) names the run.
- A plan ID went stale on any document edit, which gave agents a retry loop
  and people a token to copy.
- A list of model names is something a non-expert can judge and approve,
  can't go stale, and binds exactly: names carry the commit they were pinned
  to (`multilingual-e5-small@614241f`). This is how `ollama pull` and
  `docker pull` work.

## Considered Options

- **Keep plan/apply, merge job into run** (design A): cheapest, but each
  attempt still shows a plan ID and a run ID.
- **Pending runs that wait for approval** (design B, HCP Terraform's model):
  the exactest approval, including remote sends, but adds states
  (`needs_approval`, `expired`, `discarded`), clutter, and makes proposing
  and spending the same command prefix for harness permission rules.
- **Experiments and a queue** (design C, DVC): best for overnight batches;
  the most new vocabulary and needs a background runner. It can be added on
  top of this later.

## Consequences

- Downloads happen only by invoking `synk fetch` or `synk setup`, or with
  `--yes`. Remote sends need the config opt-in and `--yes` on each run; the
  run records what it sent (`result.egress`). That is looser than a frozen
  approval, deliberately.
- Retrying is idempotent: `synk train` with nothing changed returns the run
  that has those results (`reused: true`), and the latest unfinished run with
  the same settings is continued (`attempt` + 1) instead of a new one.
- Runs made before numbering are numbered oldest first the first time a
  command looks; older date-based IDs still resolve.
- `JOB_RUNNING` means "something is running in this project" (a run, a
  dataset build or a fetch); `JOB_NOT_FOUND` means nothing matches an ID.
- `synk source sync` exists so sources can be read without building a
  dataset; `train` and `dataset` sync by themselves.
