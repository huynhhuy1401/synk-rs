# Train with a Python sidecar, not in Rust

synk is written in Rust, but training, hard-negative mining and evaluation run
in a separate Python process (PyTorch + sentence-transformers) that synk
launches as a subprocess. The two talk only through files and a versioned
protocol: a request JSON in, progress events as JSON lines on stdout, and a
result JSON plus a model folder out. We chose this because it is the only stack
that today covers CUDA, ROCm, Apple MPS and CPU with every multilingual
embedding model; candle has no ROCm backend, and Burn has no ready-made
embedding models to train.

## Considered Options

- **Embedding Python in the synk process (PyO3).** Rejected: it ships the same
  multi-GB Python dependencies, makes packaging harder, and runs an hours-long,
  crash-prone job (out of memory, Ctrl-C) inside the CLI.
- **Pure Rust on candle or Burn.** Deferred, not rejected. A Rust trainer must
  implement the same protocol and match the Python trainer's evaluation scores
  on the same dataset before it can become the default.

## Consequences

- Users run `synk setup` once to download the Python environment. synk never
  downloads it as a side effect of another command. (Amended by
  [ADR 0002](0002-fetch-then-train.md): `synk fetch trainer` and `--yes` also
  download it.)
- AMD GPUs train on ROCm only where PyTorch ships ROCm builds: Linux, on cards
  AMD supports. Other AMD cards, and every AMD card on Windows, train on CPU.
  synk does not apply unofficial workarounds such as `HSA_OVERRIDE_GFX_VERSION`.
  (Amended by [ADR 0003](0003-only-tested-gpus.md): ROCm and MPS are not
  supported until they can be tested; those GPUs train on CPU.)
