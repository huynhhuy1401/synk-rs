# Ship llama.cpp's GGUF converter inside synk

Status: accepted, 2026-10-17. Applies [ADR 0002](0002-fetch-then-train.md)
(downloads need consent) to export.

`synk export --format gguf` used to download llama.cpp's whole repository
(about 178 MB unpacked) from GitHub on first use, without a consent step or
a checksum, then ran its converter script. That broke the rule that every
download is listed by `--dry-run` and approved with `synk fetch` or `--yes`.

Now the converter's files (`convert_hf_to_gguf.py`, `conversion/`,
`gguf-py/`, about 2 MB) and llama.cpp's MIT license live in
`crates/synk-pipeline/vendor/llama.cpp`, at the commit synk already pinned
(`LLAMA_CPP_COMMIT`). `build.rs` embeds them in the binary; export writes
them under `SYNK_HOME/converter/` once and runs the converter with
`HF_HUB_OFFLINE=1`, so its optional Hugging Face code paths fail instead of
downloading.

## Considered Options

- **A fetch name** (`synk fetch llama-converter`, pinned commit + sha256):
  consistent with other downloads, but one more thing for users to approve
  and a download that can fail, for 2 MB of Python.
- **Drop GGUF export**: loses Ollama, Open WebUI and AnythingLLM users.

## Consequences

- GGUF export works offline and never downloads.
- Updating llama.cpp means copying the files again (see
  `vendor/llama.cpp/VENDORED.toml`) and running the real end-to-end test,
  the same work the pinned commit already needed.
- The repository carries ~2 MB of third-party Python under its own license.
