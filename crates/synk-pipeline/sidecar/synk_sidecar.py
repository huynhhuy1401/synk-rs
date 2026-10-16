"""synk training sidecar.

synk launches this script in its own Python environment. It speaks protocol v2:

    python synk_sidecar.py describe
    python synk_sidecar.py run <request.json>

Every line on stdout is one JSON event: {"v": 2, "event": "progress" | "log" |
"result" | "error", ...}. The last event is either "result" or "error". Anything
else (library warnings, progress bars) goes to stderr, which synk saves to a log.

Tasks: fetch (download a model or dataset at a pinned commit), train, eval,
general (the general check), export. A request may name a `stop_file`: when
that file appears, the task stops at the next safe point (after saving a
checkpoint when training) and reports the error code CANCELLED.

This file is embedded in the synk binary and written next to the environment
on every run, so it always matches the synk version that calls it.
"""

import json
import math
import os
import random
import shutil
import sys
import time

PROTOCOL_VERSION = 2

# Set from the request; checked at every progress report.
STOP_FILE = None


def emit(event, **fields):
    sys.stdout.write(json.dumps({"v": PROTOCOL_VERSION, "event": event, **fields}) + "\n")
    sys.stdout.flush()


# Set when synk holds a lifeline: the process then ends with os._exit.
LIFELINE = False


def watch_lifeline():
    """synk keeps this process's stdin open while it runs. When synk is gone
    (Ctrl-C twice, killed, crashed), stdin closes and the trainer exits
    instead of training on as an orphan.

    The pipe is moved off stdin first (stdin becomes the null device): on
    Windows, a read pending on stdin makes other threads that touch it hang
    (`import torch` does). A thread blocked inside sys.stdin would also hold
    its lock, which Python's shutdown waits for, so the process ends with
    os._exit."""
    import threading

    global LIFELINE
    LIFELINE = True
    pipe = os.dup(0)
    null = os.open(os.devnull, os.O_RDONLY)
    os.dup2(null, 0)
    os.close(null)
    if os.name == "nt":
        import ctypes
        import msvcrt

        ctypes.windll.kernel32.SetStdHandle(-10, msvcrt.get_osfhandle(0))  # STD_INPUT_HANDLE

    def watch():
        try:
            while os.read(pipe, 4096):
                pass
        except OSError:
            pass
        os._exit(3)

    threading.Thread(target=watch, daemon=True).start()


def stop_requested():
    return bool(STOP_FILE) and os.path.exists(STOP_FILE)


def check_stop():
    if stop_requested():
        raise SidecarError("CANCELLED", "the job was cancelled")


def progress(stage, step, total, unit=None):
    fields = {"stage": stage, "step": step, "total": total}
    if unit:
        fields["unit"] = unit
    emit("progress", **fields)
    check_stop()


class SidecarError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code
        self.message = message


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


# --- describe -------------------------------------------------------------


def describe():
    import platform

    import sentence_transformers
    import torch
    import transformers

    devices = ["cpu"]
    if nvidia_gpu():
        devices.insert(0, "cuda")
    gpu = None
    vram = None
    bf16 = False
    if nvidia_gpu():
        gpu = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory
        bf16 = native_bf16()
    extras = []
    for module, name in (("peft", "lora"), ("onnxruntime", "onnx"), ("gguf", "gguf")):
        try:
            __import__(module)
            extras.append(name)
        except ImportError:
            pass
    return {
        "protocol": [PROTOCOL_VERSION],
        "tasks": sorted(TASKS),
        "extras": extras,
        "vram_bytes": vram,
        "bf16": bf16,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "sentence_transformers": sentence_transformers.__version__,
        # ROCm builds of PyTorch would expose AMD GPUs through the "cuda"
        # device; synk doesn't use them (not tested yet).
        "accelerator": "rocm" if getattr(torch.version, "hip", None) else ("cuda" if torch.version.cuda else None),
        "devices": devices,
        "gpu": gpu,
    }


# --- shared helpers -------------------------------------------------------


def nvidia_gpu():
    """An NVIDIA GPU through a CUDA build of PyTorch. AMD (ROCm) and Apple
    (MPS) GPUs aren't supported until they can be tested."""
    import torch

    return bool(torch.version.cuda) and torch.cuda.is_available()


def native_bf16():
    """bf16 in hardware: NVIDIA Ampere (RTX 30, A100) and newer. Older cards
    (T4, RTX 20) only emulate it, slowly, so they train in fp16."""
    import torch

    return nvidia_gpu() and torch.cuda.get_device_capability(0)[0] >= 8


def pick_device(requested):
    if requested == "cuda" and not nvidia_gpu():
        raise SidecarError("DEVICE_UNSUPPORTED", "this environment has no NVIDIA GPU available to a CUDA build of PyTorch")
    if requested not in (None, "cpu", "cuda"):
        raise SidecarError(
            "DEVICE_UNSUPPORTED",
            f"device `{requested}` isn't supported; synk trains on `cuda` (NVIDIA) or `cpu`",
        )
    if requested:
        return requested
    return "cuda" if nvidia_gpu() else "cpu"


def load_model(spec, device, max_seq_length=None):
    from sentence_transformers import SentenceTransformer

    try:
        model = SentenceTransformer(
            spec["name_or_path"],
            revision=spec.get("revision"),
            device=device,
            trust_remote_code=False,
        )
    except Exception as e:  # noqa: BLE001 - any load failure means the same thing to synk
        raise SidecarError("MODEL_NOT_FOUND", f"could not load {spec['name_or_path']}: {e}") from e
    if max_seq_length:
        model.max_seq_length = max_seq_length
    return model


def resolve_prompts(model, spec):
    """The (query, document) prompts to use. synk sends them for catalog
    models; for custom models (null) the prompts saved with the model apply."""
    saved = getattr(model, "prompts", None) or {}
    query = spec.get("query_prompt")
    document = spec.get("document_prompt")
    if query is None:
        query = saved.get("query", "")
    if document is None:
        document = saved.get("document", saved.get("passage", ""))
    return query, document


def encode(model, texts, prompt, stage, batch_size=32):
    import torch

    chunks = []
    total = len(texts)
    for start in range(0, total, batch_size):
        batch = texts[start : start + batch_size]
        chunks.append(
            model.encode(
                batch,
                prompt=prompt or None,
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        )
        progress(stage, min(start + batch_size, total), total)
    if not chunks:
        return torch.empty(0)
    return torch.cat(chunks)


def load_dataset_dir(dataset_dir):
    corpus = read_jsonl(os.path.join(dataset_dir, "corpus.jsonl"))
    queries = read_jsonl(os.path.join(dataset_dir, "queries.jsonl"))
    return corpus, queries


# --- fetch ----------------------------------------------------------------


def fetch(req):
    """Downloads a repository at a pinned commit into the shared Hugging Face
    cache, reporting bytes. huggingface_hub resumes partial downloads."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError
    from tqdm.auto import tqdm

    repo = req["repo"]

    class Bytes(tqdm):
        def __init__(self, *args, **kwargs):
            kwargs["disable"] = False
            kwargs["file"] = sys.stderr
            super().__init__(*args, **kwargs)

        def update(self, n=1):
            out = super().update(n)
            if self.unit == "B" and self.total:
                progress("download", int(self.n), int(self.total), unit="bytes")
            return out

    try:
        path = snapshot_download(
            repo_id=repo,
            revision=req.get("revision"),
            repo_type=req.get("repo_type", "model"),
            ignore_patterns=req.get("ignore_patterns") or None,
            allow_patterns=req.get("allow_patterns") or None,
            tqdm_class=Bytes,
        )
    except GatedRepoError as e:
        raise SidecarError("MODEL_ACCESS_DENIED", f"{repo} is gated: accept its terms on Hugging Face and set HF_TOKEN ({e})") from e
    except RepositoryNotFoundError as e:
        raise SidecarError("MODEL_NOT_FOUND", f"{repo} doesn't exist, or is private and HF_TOKEN can't read it") from e
    except Exception as e:  # noqa: BLE001 - network errors of every kind
        raise SidecarError("DOWNLOAD_FAILED", f"downloading {repo} failed: {type(e).__name__}: {e}") from e
    return {"path": path, "revision": req.get("revision")}


# --- train ----------------------------------------------------------------


def rank_of(row, pos):
    """1-based rank of column `pos` in a row of scores."""
    return int((row > row[pos]).sum()) + 1


def filter_and_mine(model, prompts, examples, chunks, params, reranker=None):
    """The round-trip query filter and hard-negative mining, sharing one
    encoding pass with the base model.

    Filter: a query is kept only if the base model ranks its own chunk within
    the top `filter_k` training chunks. Queries it can't connect to their
    chunk at all are usually off-topic or wrong (Promptagator's round-trip
    consistency).

    Mining: chunks from other documents that rank highest but score below
    `(1 - margin) *` the positive's score, so chunks that likely answer the
    query too are not used as negatives. Chunks from the same document are
    skipped for the same reason. Queries with no safe negative train on
    in-batch negatives only.

    With a reranker, four times as many candidates are mined and the
    reranker (a cross-encoder, which reads query and chunk together) removes
    those it thinks answer the query: probability 0.5 or more, or at least
    the positive's. The first `negatives` survivors are kept.
    """
    chunk_index = {c["id"]: i for i, c in enumerate(chunks)}
    q_emb = encode(model, [e["query"] for e in examples], prompts[0], "mine:queries")
    c_emb = encode(model, [c["text"] for c in chunks], prompts[1], "mine:chunks")
    scores = q_emb @ c_emb.T
    wanted = params.get("negatives", 3)
    pool = wanted * 4 if reranker is not None else wanted
    margin = params.get("negative_margin", 0.05)
    filter_k = params.get("filter_k", 20) if params.get("query_filter", True) else None
    top_k = min(len(chunks), max(50, pool * 10))
    kept = []
    stats = {"filtered_queries": 0, "unmined_queries": 0, "mined_negatives": 0, "reranker_dropped": 0}
    for i, ex in enumerate(examples):
        row = scores[i]
        pos = chunk_index[ex["chunk"]]
        if filter_k is not None and rank_of(row, pos) > filter_k:
            stats["filtered_queries"] += 1
            continue
        pos_score = float(row[pos])
        candidates = []
        if pool > 0:
            for j in row.topk(top_k).indices.tolist():
                cand = chunks[j]
                if cand["doc"] == ex["doc"]:
                    continue
                if float(row[j]) <= pos_score * (1 - margin):
                    candidates.append(cand)
                    if len(candidates) == pool:
                        break
        kept.append((ex, candidates))
        if (i + 1) % 100 == 0 or i + 1 == len(examples):
            progress("mine", i + 1, len(examples))

    if reranker is not None:
        kept = rerank_candidates(reranker, kept, params, stats)
    out = []
    for ex, candidates in kept:
        negatives = candidates[:wanted]
        if not negatives:
            stats["unmined_queries"] += 1
        stats["mined_negatives"] += len(negatives)
        out.append(
            {
                **ex,
                "negatives": [c["text"] for c in negatives],
                "negative_chunks": [c["id"] for c in negatives],
            }
        )
    return out, stats


def load_reranker(spec, device):
    from sentence_transformers import CrossEncoder

    try:
        reranker = CrossEncoder(
            spec["name_or_path"],
            revision=spec.get("revision"),
            device=device,
            trust_remote_code=False,
        )
    except Exception as e:  # noqa: BLE001 - any load failure means the same thing to synk
        raise SidecarError("MODEL_NOT_FOUND", f"could not load reranker {spec['name_or_path']}: {e}") from e
    if getattr(reranker, "num_labels", 1) != 1:
        raise SidecarError("MODEL_NOT_FOUND", f"{spec['name_or_path']} is not a reranker: it outputs {reranker.num_labels} scores, not 1")
    return reranker


def rerank_candidates(reranker, kept, params, stats, batch=256):
    import torch

    max_prob = params.get("reranker_max_prob", 0.5)
    pairs = []
    for ex, candidates in kept:
        pairs.append((ex["query"], ex["positive"]))
        pairs.extend((ex["query"], c["text"]) for c in candidates)
    probs = []
    for start in range(0, len(pairs), batch):
        probs.extend(
            float(p)
            for p in reranker.predict(
                pairs[start : start + batch],
                batch_size=32,
                show_progress_bar=False,
                activation_fn=torch.nn.Sigmoid(),
            )
        )
        progress("rerank", min(start + batch, len(pairs)), len(pairs))
    out = []
    k = 0
    for ex, candidates in kept:
        p_pos = probs[k]
        k += 1
        survivors = []
        for c in candidates:
            p = probs[k]
            k += 1
            if p < min(max_prob, p_pos):
                survivors.append(c)
            else:
                stats["reranker_dropped"] += 1
        out.append((ex, survivors))
    return out


def pick_dev_docs(docs, params, seed):
    """Training documents held out to pick the best epoch. Only worth it when
    there is more than one epoch to choose from, and enough documents that
    the dev score isn't one document's luck."""
    percent = params.get("dev_percent", 0)
    if not percent or params.get("epochs") == 1 or len(docs) < params.get("dev_min_documents", 20):
        return set()
    n = max(1, round(len(docs) * percent / 100))
    return set(random.Random(seed).sample(sorted(docs), n))


def dev_score(model, prompts, dev_examples, chunks):
    """Mean nDCG@10 of the dev queries, searching all training chunks."""
    chunk_index = {c["id"]: i for i, c in enumerate(chunks)}
    c_emb = encode(model, [c["text"] for c in chunks], prompts[1], "dev:chunks")
    q_emb = encode(model, [e["query"] for e in dev_examples], prompts[0], "dev:queries")
    scores = q_emb @ c_emb.T
    ranks = [rank_of(scores[i], chunk_index[e["chunk"]]) for i, e in enumerate(dev_examples)]
    return ir_metrics(ranks)["ndcg@10"]


def last_complete_checkpoint(checkpoint_dir):
    """The newest checkpoint that finished saving, or None. The HF Trainer
    writes trainer_state.json last, so a checkpoint without it was cut off
    mid-save (e.g. by Ctrl-C); those are deleted."""
    if not os.path.isdir(checkpoint_dir):
        return None
    complete = []
    for name in os.listdir(checkpoint_dir):
        path = os.path.join(checkpoint_dir, name)
        if not name.startswith("checkpoint-"):
            continue
        if os.path.isfile(os.path.join(path, "trainer_state.json")):
            complete.append((int(name.split("-")[1]), path))
        else:
            shutil.rmtree(path, ignore_errors=True)
    return max(complete)[1] if complete else None


def merge_lora(model):
    """Folds LoRA weights into the base weights, so the saved model is a
    plain model every tool can load."""
    for module in model:
        inner = getattr(module, "auto_model", None)
        if inner is not None and hasattr(inner, "merge_and_unload"):
            module.auto_model = inner.merge_and_unload()


def save_model(model, path, merged=False):
    """Saves without sentence-transformers' model card, which would copy
    training examples (the user's text) into README.md. With `merged`, a
    LoRA model is saved as a copy with its adapters folded in, and keeps
    training."""
    tmp = path + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    if merged:
        import copy

        clone = copy.deepcopy(model)
        merge_lora(clone)
        clone.save(tmp, create_model_card=False)
        del clone
    else:
        model.save(tmp, create_model_card=False)
    shutil.rmtree(path, ignore_errors=True)
    os.replace(tmp, path)


def train(req):
    import torch
    from datasets import Dataset, DatasetDict
    from sentence_transformers import (
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
        losses,
    )
    from sentence_transformers.training_args import BatchSamplers
    from transformers import TrainerCallback

    params = req["params"]
    spec = req["base_model"]
    out_dir = req["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    seed = params.get("seed", 42)
    random.seed(seed)
    torch.manual_seed(seed)

    device = pick_device(req.get("device"))
    emit("log", message=f"device: {device}")
    model = load_model(spec, device, params.get("max_seq_length"))
    n_params = sum(p.numel() for p in model.parameters())
    finetune = params.get("finetune", "auto")
    if finetune == "auto":
        finetune = "lora" if n_params >= 300_000_000 else "full"
    precision = params.get("precision", "auto")
    if precision == "auto":
        if device == "cuda":
            precision = "bf16" if native_bf16() else "fp16"
        else:
            # Half precision on CPU is slow.
            precision = "fp32"
    grad_checkpointing = params.get("gradient_checkpointing", "auto")
    if grad_checkpointing == "auto":
        grad_checkpointing = n_params >= 300_000_000
    emit("log", message=f"{n_params / 1e6:.0f}M parameters: {finetune} fine-tuning, {precision}"
         + (", gradient checkpointing" if grad_checkpointing else ""))
    query_prompt, document_prompt = resolve_prompts(model, spec)
    prompts = (query_prompt, document_prompt)
    # Saved with the model, so anything that loads it with
    # sentence-transformers applies them without being told.
    model.prompts = {"query": query_prompt, "document": document_prompt}

    corpus, queries = load_dataset_dir(req["dataset_dir"])
    # Only train-split chunks may appear in training, as positives or negatives.
    chunks = [c for c in corpus if c["split"] == "train"]
    by_id = {c["id"]: c for c in chunks}
    dev_docs = pick_dev_docs({c["doc"] for c in chunks}, params, seed)
    all_examples = [
        {"query": q["text"], "chunk": q["chunk"], "doc": by_id[q["chunk"]]["doc"], "positive": by_id[q["chunk"]]["text"]}
        for q in queries
        # Real queries are for evaluation only.
        if q["chunk"] in by_id and q.get("kind") != "real"
    ]
    dev_examples = [e for e in all_examples if e["doc"] in dev_docs]
    examples = [e for e in all_examples if e["doc"] not in dev_docs]
    if len(examples) < 2:
        raise SidecarError("DATASET_TOO_SMALL", f"only {len(examples)} training examples in the train split")
    candidates = [c for c in chunks if c["doc"] not in dev_docs]

    # Filtering and mining are deterministic per dataset, but their results
    # are saved so a resumed run trains on exactly the same examples.
    examples_path = os.path.join(out_dir, "examples.jsonl")
    prep_path = os.path.join(out_dir, "prep.json")
    if os.path.exists(examples_path) and os.path.exists(prep_path):
        kept = read_jsonl(examples_path)
        with open(prep_path, encoding="utf-8") as f:
            prep = json.load(f)
    else:
        reranker = load_reranker(params["reranker"], device) if params.get("reranker") else None
        kept, prep = filter_and_mine(model, prompts, examples, candidates, params, reranker)
        if reranker is not None:
            del reranker
            if device == "cuda":
                torch.cuda.empty_cache()
        write_jsonl(examples_path, kept)
        with open(prep_path, "w", encoding="utf-8") as f:
            json.dump(prep, f)
    emit(
        "log",
        message=f"{len(kept)} of {len(examples)} queries kept by the filter, "
        f"{prep['mined_negatives']} hard negatives, {len(dev_examples)} dev queries",
    )
    if len(kept) < 2:
        raise SidecarError(
            "DATASET_TOO_SMALL",
            f"the query filter kept {len(kept)} of {len(examples)} training queries; "
            "set query_filter = false under [trainer], or generate queries with a stronger model",
        )

    # Every row of a dataset needs the same columns, so examples are grouped
    # by how many negatives they got; batches never mix groups.
    groups = {}
    for e in kept:
        groups.setdefault(len(e["negatives"]), []).append(e)

    def to_dataset(rows, k):
        columns = {"anchor": [r["query"] for r in rows], "positive": [r["positive"] for r in rows]}
        for n in range(k):
            columns[f"negative_{n + 1}"] = [r["negatives"][n] for r in rows]
        return Dataset.from_dict(columns)

    if len(groups) == 1:
        (k, rows), = groups.items()
        train_ds = to_dataset(rows, k)
    else:
        train_ds = DatasetDict({f"negatives_{k}": to_dataset(rows, k) for k, rows in sorted(groups.items())})
    column_prompts = {"anchor": query_prompt, "positive": document_prompt}
    for n in range(max(groups)):
        column_prompts[f"negative_{n + 1}"] = document_prompt

    learning_rate = params.get("learning_rate", 2e-5)
    if finetune == "lora":
        from peft import LoraConfig, TaskType, get_peft_model

        lora = params.get("lora") or {}
        config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora.get("r", 32),
            lora_alpha=lora.get("alpha", 64),
            lora_dropout=lora.get("dropout", 0.05),
            target_modules="all-linear",
        )
        transformer = next(m for m in model if hasattr(m, "auto_model"))
        transformer.auto_model = get_peft_model(transformer.auto_model, config)
        if grad_checkpointing:
            # Frozen embeddings + checkpointing need inputs that require grad.
            transformer.auto_model.enable_input_require_grads()
        learning_rate = lora.get("learning_rate", 1e-4)

    batch_size = min(params.get("batch_size", 16), len(kept))
    steps_per_epoch = sum(math.ceil(len(rows) / batch_size) for rows in groups.values())
    epochs = params.get("epochs", 1)
    if epochs == "auto":
        epochs = min(max(1, math.ceil(params.get("min_steps", 300) / steps_per_epoch)), params.get("max_auto_epochs", 5))
    total_steps = steps_per_epoch * epochs
    checkpoint_dir = os.path.join(out_dir, "checkpoints")
    best_dir = os.path.join(out_dir, "best")
    best_path = os.path.join(out_dir, "best.json")
    best = {"epoch": None, "score": None, "scores": {}}
    if req.get("resume") and os.path.exists(best_path):
        with open(best_path, encoding="utf-8") as f:
            best = json.load(f)

    stopped = {"value": False}

    class Progress(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            emit("progress", stage="train", step=state.global_step, total=state.max_steps or total_steps, unit="steps")
            if stop_requested():
                # Save a checkpoint, then stop; the next apply resumes here.
                stopped["value"] = True
                control.should_save = True
                control.should_training_stop = True

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and "loss" in logs:
                emit("log", message=f"step {state.global_step}: loss {logs['loss']:.4f}")

        def on_epoch_end(self, args, state, control, **kwargs):
            # Keep the epoch that does best on documents training never saw.
            if not dev_examples:
                return
            epoch = round(state.epoch)
            score = dev_score(model, prompts, dev_examples, chunks)
            best["scores"][str(epoch)] = score
            emit("log", message=f"epoch {epoch}: dev nDCG@10 {score:.4f}")
            # On a tie, the later epoch: more training, no measured harm.
            if best["score"] is None or score >= best["score"]:
                best["epoch"], best["score"] = epoch, score
                save_model(model, best_dir, merged=finetune == "lora")
            with open(best_path, "w", encoding="utf-8") as f:
                json.dump(best, f)

    args = SentenceTransformerTrainingArguments(
        output_dir=checkpoint_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        bf16=precision == "bf16",
        fp16=precision == "fp16",
        gradient_checkpointing=bool(grad_checkpointing),
        warmup_steps=params.get("warmup", 0.1),
        seed=seed,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        prompts=column_prompts,
        save_strategy="steps",
        save_steps=max(10, total_steps // 10),
        save_total_limit=1,
        logging_steps=max(1, total_steps // 20),
        report_to=[],
        disable_tqdm=True,
        use_cpu=device == "cpu",
    )
    # In-batch negatives: a bigger batch means more of them. Above
    # `cached_loss_above_batch` the cached variant (GradCache) keeps memory
    # flat. Models trained with Matryoshka keep working at smaller sizes.
    if batch_size > params.get("cached_loss_above_batch", 32):
        loss = losses.CachedMultipleNegativesRankingLoss(model, mini_batch_size=16)
    else:
        loss = losses.MultipleNegativesRankingLoss(model)
    mrl_dims = params.get("mrl_dims") or []
    if mrl_dims:
        loss = losses.MatryoshkaLoss(model, loss, matryoshka_dims=mrl_dims)
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        loss=loss,
        callbacks=[Progress()],
    )
    checkpoint = last_complete_checkpoint(checkpoint_dir) if req.get("resume") else None
    resumed = checkpoint is not None
    if resumed:
        emit("log", message=f"resuming from {os.path.basename(checkpoint)}")
    started = time.time()
    try:
        trainer.train(resume_from_checkpoint=checkpoint)
    except torch.cuda.OutOfMemoryError as e:
        raise SidecarError(
            "OUT_OF_MEMORY",
            f"the GPU ran out of memory at batch size {batch_size}: {e}".splitlines()[0],
        ) from e
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            raise SidecarError("OUT_OF_MEMORY", f"ran out of memory at batch size {batch_size}") from e
        raise
    if stopped["value"]:
        raise SidecarError("CANCELLED", "the job was cancelled; training stopped after a checkpoint")
    seconds = time.time() - started
    if finetune == "lora":
        merge_lora(model)

    model_dir = os.path.join(out_dir, "model")
    best_epoch = epochs
    if best["epoch"] is not None and best["epoch"] != epochs and os.path.isdir(best_dir):
        best_epoch = best["epoch"]
        shutil.rmtree(model_dir, ignore_errors=True)
        os.replace(best_dir, model_dir)
    else:
        save_model(model, model_dir)
    # Checkpoints (with optimizer state, several times the model's size)
    # only exist to resume an interrupted run.
    shutil.rmtree(checkpoint_dir, ignore_errors=True)
    shutil.rmtree(best_dir, ignore_errors=True)
    return {
        "model_dir": model_dir,
        "device": device,
        "examples": len(kept),
        "train_queries": len(examples),
        "filtered_queries": prep["filtered_queries"],
        "dev_queries": len(dev_examples),
        "dev_documents": len(dev_docs),
        "negatives_per_query": params.get("negatives", 3),
        "mined_negatives": prep["mined_negatives"],
        "unmined_queries": prep["unmined_queries"],
        "random_negatives": 0,
        "reranker_dropped": prep.get("reranker_dropped", 0),
        "steps": total_steps,
        "epochs": epochs,
        "best_epoch": best_epoch,
        "dev_scores": best["scores"],
        "resumed": resumed,
        "train_seconds": round(seconds, 1),
        "finetune": finetune,
        "precision": precision,
        "gradient_checkpointing": bool(grad_checkpointing),
        "parameters": n_params,
        "loss": type(loss).__name__,
        "prompts": {"query": query_prompt, "document": document_prompt},
    }


# --- eval -----------------------------------------------------------------

K_VALUES = (1, 5, 10)


def ir_metrics(ranks):
    """Metrics for queries with exactly one relevant chunk. `ranks` holds the
    1-based rank of that chunk for each query."""
    n = len(ranks)
    if n == 0:
        return {"queries": 0}
    out = {"queries": n}
    for k in K_VALUES:
        out[f"recall@{k}"] = sum(1 for r in ranks if r <= k) / n
    out["mrr@10"] = sum(1 / r for r in ranks if r <= 10) / n
    out["ndcg@10"] = sum(1 / math.log2(r + 1) for r in ranks if r <= 10) / n
    return {key: round(v, 4) if isinstance(v, float) else v for key, v in out.items()}


def evaluate(req):
    corpus, queries = load_dataset_dir(req["dataset_dir"])
    split = req.get("split", "test")
    split_chunks = {c["id"] for c in corpus if c["split"] == split}
    test_queries = [q for q in queries if q.get("kind") != "real" and q["chunk"] in split_chunks]
    # Real queries are evaluated over every document: they are the searches
    # the user actually makes, and training never saw them.
    real_queries = [q for q in queries if q.get("kind") == "real"]
    if not test_queries:
        raise SidecarError("DATASET_TOO_SMALL", f"no queries in the {split} split")
    all_queries = test_queries + real_queries
    chunk_index = {c["id"]: i for i, c in enumerate(corpus)}
    relevant = [[chunk_index[c] for c in (q.get("relevant_chunks") or [q["chunk"]])] for q in all_queries]
    doc_of = {c["id"]: c["doc"] for c in corpus}
    device = pick_device(req.get("device"))
    # Each query's rank under every model, and what each ranked first: synk's
    # verdict tests the paired differences, and `synk inspect` shows them.
    per_query = [
        {
            "id": q["id"],
            "query": q["text"],
            "kind": q.get("kind") or "synthetic",
            "chunk": q["chunk"],
            "doc": doc_of[q["chunk"]],
            "lang": q.get("lang"),
            "ranks": {},
            "top": {},
        }
        for q in all_queries
    ]

    results, real_results = {}, {}
    for spec in req["models"]:
        label = spec["label"]
        model = load_model(spec, device, req.get("max_seq_length"))
        query_prompt, document_prompt = resolve_prompts(model, spec)
        # The whole corpus is searched, so train chunks act as distractors.
        c_emb = encode(model, [c["text"] for c in corpus], document_prompt, f"eval:{label}:chunks")
        q_emb = encode(model, [q["text"] for q in all_queries], query_prompt, f"eval:{label}:queries")
        scores = q_emb @ c_emb.T
        ranks = []
        for i in range(len(all_queries)):
            # With several right chunks, the rank of the best-placed one.
            best = scores[i, relevant[i]].max()
            rank = int((scores[i] > best).sum()) + 1
            ranks.append(rank)
            per_query[i]["ranks"][label] = rank
            per_query[i]["top"][label] = corpus[int(scores[i].argmax())]["id"]
        for target, qs, rs in (
            (results, test_queries, ranks[: len(test_queries)]),
            (real_results, real_queries, ranks[len(test_queries) :]),
        ):
            if not qs:
                continue
            groups = {}
            for q, r in zip(qs, rs):
                groups.setdefault(q.get("lang") or "unknown", []).append(r)
            target[label] = {
                "overall": ir_metrics(rs),
                "by_lang": {lang: ir_metrics(g) for lang, g in sorted(groups.items())},
            }
        del model
    if req.get("output_dir"):
        write_jsonl(os.path.join(req["output_dir"], "per_query.jsonl"), per_query)
    return {
        "device": device,
        "corpus_chunks": len(corpus),
        "metrics": results,
        "real_metrics": real_results or None,
    }


# --- general check --------------------------------------------------------


def general_sets(req):
    """Public retrieval sets, from the Hugging Face cache (the plan's models
    step downloaded them): queries, a corpus, and each query's right chunks."""
    import pyarrow.parquet as pq
    from huggingface_hub import snapshot_download

    out = []
    for spec in req["sets"]:
        path = snapshot_download(
            repo_id=spec["repo"],
            revision=spec["revision"],
            repo_type="dataset",
            allow_patterns=spec.get("allow_patterns") or None,
            local_files_only=True,
        )

        def table(rel):
            return pq.read_table(os.path.join(path, rel)).to_pylist()

        if spec["format"] == "beir":
            corpus = {str(r["_id"]): r["text"] for r in table("corpus/train-00000-of-00001.parquet")}
            queries = {str(r["_id"]): r["text"] for r in table("queries/train-00000-of-00001.parquet")}
            relevant = {}
            for r in table("qrels/train-00000-of-00001.parquet"):
                relevant.setdefault(str(r["query-id"]), set()).add(str(r["corpus-id"]))
            items = [(qid, queries[qid], rel) for qid, rel in relevant.items() if qid in queries]
        else:  # squad-style: one question per row, its context is the answer
            rows = table(spec["file"])
            contexts = {}
            for r in rows:
                contexts.setdefault(r["context"], f"c{len(contexts)}")
            corpus = {cid: text for text, cid in contexts.items()}
            items = [(r["id"], r["question"], {contexts[r["context"]]}) for r in rows]
        items.sort(key=lambda x: x[0])
        limit = spec.get("max_queries")
        if limit and len(items) > limit:
            items = random.Random(0).sample(items, limit)
        out.append((spec, corpus, items))
    return out


def general(req):
    """The base and tuned models on public sets, so tuning can't quietly
    hurt ordinary search. Per-query ranks feed synk's paired test."""
    device = pick_device(req.get("device"))
    sets = general_sets(req)
    rows = []
    for spec, corpus, items in sets:
        ids = list(corpus)
        index = {cid: i for i, cid in enumerate(ids)}
        for label_spec in req["models"]:
            label = label_spec["label"]
            model = load_model(label_spec, device, req.get("max_seq_length"))
            qp, dp = resolve_prompts(model, label_spec)
            c_emb = encode(model, [corpus[c] for c in ids], dp, f"general:{spec['name']}:{label}")
            q_emb = encode(model, [q for _, q, _ in items], qp, f"general:{spec['name']}:{label}")
            scores = q_emb @ c_emb.T
            for i, (qid, query, rel) in enumerate(items):
                cols = [index[c] for c in rel if c in index]
                if not cols:
                    continue
                best = scores[i, cols].max()
                rank = int((scores[i] > best).sum()) + 1
                key = f"{spec['name']}:{qid}"
                row = next((r for r in rows if r["id"] == key), None)
                if row is None:
                    row = {"id": key, "query": query, "kind": "general", "chunk": "", "doc": key,
                           "lang": spec["name"], "ranks": {}, "top": {}}
                    rows.append(row)
                row["ranks"][label] = rank
            del model
    write_jsonl(os.path.join(req["output_dir"], "general.jsonl"), rows)
    return {"sets": [{"name": s["name"], "queries": len(items), "corpus": len(c)} for s, c, items in sets]}


# --- export ---------------------------------------------------------------

def gguf_source(model_dir, work):
    """A copy of the model the converter reads correctly. multilingual-e5
    models are BERT with an XLM-R tokenizer: labelled XLM-R, and without
    `pad_token_id` (which would shift BERT's position embeddings by
    RoBERTa's offset), they convert exactly."""
    with open(os.path.join(model_dir, "config.json"), encoding="utf-8") as f:
        config = json.load(f)
    tok_class = ""
    tok_path = os.path.join(model_dir, "tokenizer_config.json")
    if os.path.exists(tok_path):
        with open(tok_path, encoding="utf-8") as f:
            tok_class = json.load(f).get("tokenizer_class", "")
    if config.get("model_type") == "bert" and "XLMRoberta" in tok_class:
        src = os.path.join(work, "source")
        shutil.rmtree(src, ignore_errors=True)
        shutil.copytree(model_dir, src)
        config["architectures"] = ["XLMRobertaModel"]
        config.pop("pad_token_id", None)
        with open(os.path.join(src, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        emit("log", message="BERT model with an XLM-R tokenizer: converting as XLM-R with BERT positions")
        return src
    return model_dir


def cosines(a, b):
    import torch

    a = torch.nn.functional.normalize(torch.as_tensor(a, dtype=torch.float32), dim=-1)
    b = torch.nn.functional.normalize(torch.as_tensor(b, dtype=torch.float32), dim=-1)
    return (a * b).sum(-1).tolist()


def export_onnx(model, out, texts, reference):
    """A model.onnx that maps token IDs straight to sentence embeddings
    (transformer, pooling and any dense layers inside), next to the
    tokenizer files, checked with ONNX Runtime."""
    import numpy as np
    import onnxruntime
    import torch

    class Embed(torch.nn.Module):
        def __init__(self, st):
            super().__init__()
            self.st = st

        def forward(self, input_ids, attention_mask):
            return self.st({"input_ids": input_ids, "attention_mask": attention_mask})["sentence_embedding"]

    model.eval()
    sample = model.tokenize(texts[:2])
    args = (sample["input_ids"], sample["attention_mask"])
    path = os.path.join(out, "model.onnx")
    axes = {"input_ids": {0: "batch", 1: "tokens"}, "attention_mask": {0: "batch", 1: "tokens"}, "sentence_embedding": {0: "batch"}}
    with torch.no_grad():
        try:
            torch.onnx.export(Embed(model), args, path, input_names=["input_ids", "attention_mask"],
                              output_names=["sentence_embedding"], dynamic_axes=axes, opset_version=18, dynamo=False)
        except Exception as first:  # noqa: BLE001 - try the other exporter
            emit("log", message=f"TorchScript ONNX exporter failed ({type(first).__name__}); trying the dynamo exporter")
            try:
                torch.onnx.export(Embed(model), args, path, input_names=["input_ids", "attention_mask"],
                                  output_names=["sentence_embedding"], dynamic_axes=axes, dynamo=True)
            except Exception as e:  # noqa: BLE001
                raise SidecarError("EXPORT_UNSUPPORTED", f"ONNX export failed: {type(e).__name__}: {e}") from e
    model.tokenizer.save_pretrained(out)
    with open(os.path.join(out, "synk_onnx.json"), "w", encoding="utf-8") as f:
        json.dump({"inputs": ["input_ids", "attention_mask"], "output": "sentence_embedding",
                   "max_seq_length": model.max_seq_length, "prompts": getattr(model, "prompts", {})}, f, indent=2)
    session = onnxruntime.InferenceSession(path, providers=["CPUExecutionProvider"])
    got = []
    for text in texts:
        t = model.tokenize([text])
        (vec,) = session.run(["sentence_embedding"], {"input_ids": t["input_ids"].numpy().astype(np.int64),
                                                        "attention_mask": t["attention_mask"].numpy().astype(np.int64)})
        got.append(vec[0])
    cos = cosines(reference, np.stack(got))
    return {"path": out, "check": {"min_cosine": round(min(cos), 6), "texts": len(texts)}}


def export(req):
    """ONNX or GGUF from a tuned model folder. Returns reference vectors of
    `check_texts` (no prompts), so the caller can compare what a runtime
    produces with what sentence-transformers produces."""
    import subprocess

    from sentence_transformers import SentenceTransformer

    fmt = req["format"]
    src = req["model_dir"]
    out = req["out"]
    texts = req.get("check_texts") or ["hello"]
    os.makedirs(out, exist_ok=True)
    reference_model = SentenceTransformer(src, device="cpu")
    reference = reference_model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    if fmt == "onnx":
        return export_onnx(reference_model, out, texts, reference)
    if fmt == "gguf":
        # llama.cpp's converter, shipped inside synk and written out by it.
        root = req["converter_dir"]
        work = os.path.join(out, ".synk-work")
        os.makedirs(work, exist_ok=True)
        source = gguf_source(src, work)
        outfile = os.path.join(out, "model.gguf")
        cmd = [sys.executable, os.path.join(root, "convert_hf_to_gguf.py"), source,
               "--outfile", outfile, "--outtype", req.get("quant") or "q8_0"]
        if any(name.endswith("_Dense") for name in os.listdir(src)):
            # EmbeddingGemma: without this flag the Dense layers are dropped.
            cmd.append("--sentence-transformers-dense-modules")
        emit("log", message="Converting to GGUF")
        # Offline: the converter has code paths that fetch from Hugging Face;
        # an export must never download, so they fail instead.
        env = dict(os.environ, PYTHONPATH=os.path.join(root, "gguf-py"), HF_HUB_OFFLINE="1")
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env, encoding="utf-8", errors="replace")
        sys.stderr.write(proc.stdout + proc.stderr)
        shutil.rmtree(work, ignore_errors=True)
        if proc.returncode != 0:
            tail = "\n".join((proc.stderr or proc.stdout).strip().splitlines()[-5:])
            raise SidecarError("EXPORT_UNSUPPORTED", f"llama.cpp can't convert this model to GGUF:\n{tail}")
        return {"path": outfile, "reference": reference.tolist(), "texts": texts}
    raise SidecarError("SIDECAR_PROTOCOL", f"unknown export format {fmt!r}")


# --- entry point ----------------------------------------------------------

TASKS = {"fetch": fetch, "train": train, "eval": evaluate, "general": general, "export": export}


def main(argv):
    try:
        if len(argv) == 2 and argv[1] == "describe":
            emit("result", data=describe())
            return 0
        if len(argv) == 3 and argv[1] == "run":
            with open(argv[2], encoding="utf-8") as f:
                req = json.load(f)
            global STOP_FILE
            STOP_FILE = req.get("stop_file")
            if req.get("lifeline"):
                watch_lifeline()
            if req.get("v") != PROTOCOL_VERSION:
                raise SidecarError("SIDECAR_PROTOCOL", f"unsupported request version {req.get('v')}")
            task = TASKS.get(req.get("task"))
            if task is None:
                raise SidecarError("SIDECAR_PROTOCOL", f"unknown task {req.get('task')!r}")
            emit("result", data=task(req))
            return 0
        raise SidecarError("SIDECAR_PROTOCOL", "usage: synk_sidecar.py describe | run <request.json>")
    except SidecarError as e:
        emit("error", code=e.code, message=e.message)
        return 1
    except KeyboardInterrupt:
        # Run on its own, the trainer may still get a Ctrl-C.
        emit("error", code="CANCELLED", message="the trainer was interrupted")
        return 1
    except Exception as e:  # noqa: BLE001 - report every crash as an event
        import traceback

        traceback.print_exc()
        emit("error", code="SIDECAR_FAILED", message=f"{type(e).__name__}: {e}")
        return 1


if __name__ == "__main__":
    code = main(sys.argv)
    if LIFELINE:
        # Skip interpreter shutdown, which would wait on the lifeline thread.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)
    sys.exit(code)
