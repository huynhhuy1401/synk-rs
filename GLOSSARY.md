# synk

synk fine-tunes a small embedding model on a user's own documents, so their
retrieval (RAG) understands their vocabulary, and proves the tuned model is
better before they use it.

## Language

### Data

**Source**:
A configured place documents come from, such as a folder or a single file.
_Avoid_: input, dataset, connection

**Document**:
One normalized text item read from a source.
_Avoid_: file, record, item

**Chunk**:
A span of a document small enough to be one unit of retrieval.
_Avoid_: passage, segment, split

### Training data

**Query**:
A search string that should retrieve a given chunk. A **synthetic query** is
written by an LLM from the chunk; a **real query** is written by a person.
_Avoid_: question, anchor, prompt

**Hard negative**:
A chunk that looks relevant to a query but does not answer it.
_Avoid_: distractor, negative sample

**Reranker**:
A model that reads a query and a chunk together and scores how well the chunk
answers it. Slower than an embedding model but more accurate; synk uses one,
optionally, to check hard negatives.
_Avoid_: cross-encoder (in user-facing text), judge

**Training example**:
One query, the chunk it should retrieve, and its hard negatives.
_Avoid_: pair, triplet, sample

**Query filter**:
The step that drops synthetic queries the base model can't connect to their
own chunk, because such queries are usually off-topic or wrong.
_Avoid_: cleaning, validation

**Dev split**:
A few training documents held out of training to choose the best epoch. The
test split is never used for choices.
_Avoid_: validation set (ambiguous with the test split)

**Dataset**:
The chunks and queries built from a fixed snapshot of documents, divided into
a train split and a test split by document.
_Avoid_: corpus (when meaning the whole dataset)

**Test document**:
A document in the test split: never trained on, used only to test. The
verdict counts test documents, not test queries, because queries about one
document rise and fall together.
_Avoid_: test file, eval doc

**Template** (queries):
A file listing some documents, each with an empty search for the user to
fill in without opening the document. Imported, the searches become real
queries.
_Avoid_: form, questionnaire

**Query generator**:
The LLM that writes synthetic queries.
_Avoid_: teacher, LLM backend

### Models

**Base model**:
The embedding model a training run starts from.
_Avoid_: pretrained model, original model

**Tuned model**:
The embedding model a training run produces.
_Avoid_: fine-tuned model, custom model, new model

**Model catalog**:
The list of base models synk has tested and ships settings for.
_Avoid_: presets, supported models

**Custom model**:
A base model that is not in the model catalog. synk trains it but makes no
promise about the result.
_Avoid_: BYO model, unsupported model

**Training run** (run):
One attempt to produce a tuned model from one dataset and one base model,
started by `synk train`. Its ID, `run:<n>-<label>`, is numbered per project
and is the one handle people keep; `--name` replaces the label. A run is
running, succeeded, failed, cancelled or interrupted, and keeps the settings
it started with. Several can coexist; `synk export` picks one.
_Avoid_: job, experiment, session, attempt (except for a run's retries)

**Variant**:
A run started with different settings (`synk train --set ...`), to compare
with another run.
_Avoid_: experiment, trial

**Verdict**:
The outcome of comparing a tuned model with its base model on the test split:
better, worse, or no clear change. Better or worse only when a statistical
test says so. Real queries decide it when there are enough of them, spread
over enough documents.
_Avoid_: score, result

**Inconclusive**:
A "no clear change" verdict whose test was too small to see a typical gain:
too few test documents. Different from "no clear change" after a test big
enough to see one, which means tuning likely didn't help.
_Avoid_: failed, unknown

**Smallest detectable change**:
The true gain in search quality the test would catch 8 times out of 10.
When it is bigger than what tuning usually brings, the test is too small
and the verdict will be inconclusive.
_Avoid_: margin of error, minimum effect

**Advice**:
A suggestion, with a stable code, for getting a better result, and how safe
it is to apply without a person looking.
_Avoid_: hint, warning, recommendation

### Pipeline

**Artifact**:
The saved output of one pipeline step, identified by a key derived from
everything that went into it.
_Avoid_: cache entry, output folder

**Config**:
`synk.toml`: what the user wants (sources, models, settings). Runs start
from it, plus any `--set` overrides.
_Avoid_: spec, settings file

**Step**:
One stage of the pipeline that a dry run can mark current or stale: trainer,
models, generator, sync, dataset, train, eval.
_Avoid_: stage, task

**Dry run**:
What `synk train` (or `synk dataset`) would do now, without doing it: the
stale steps and why, what to fetch, the text that would leave the machine,
and the time. `--dry-run`.
_Avoid_: plan, preview (in user-facing text)

**Fetch**:
Downloading something training needs, by name: a model, an Ollama model,
the general check's test sets, or the trainer. `synk fetch NAME...`.
_Avoid_: pull, install (except for the trainer)

**Consent**:
Approval to download, or to send text off the machine, given by running
`synk fetch` or `synk setup`, or with `--yes`.
_Avoid_: confirmation, approval

**Operation**:
A long command with a record any shell can wait on, follow or cancel: a
training run, a dataset build (`dataset:<n>`), or a fetch (`fetch:<n>`). One
runs at a time per project.
_Avoid_: job, task

**General check**:
Evaluating the base and tuned models on public datasets, to see whether
tuning hurt ordinary search.
_Avoid_: forgetting test, benchmark

**Export**:
Writing a tuned model in a format other programs load (sentence-transformers
folder, ONNX, GGUF) and checking that its vectors match.
_Avoid_: publish, save

**Style examples** (together, the _style pool_):
A few real queries shown to the query generator as examples of how the user
searches, and therefore left out of evaluation. Chosen once, then kept.
_Avoid_: few-shot set

**Trainer**:
The component that runs training and evaluation for synk, separate from the
synk program itself.
_Avoid_: backend, worker
