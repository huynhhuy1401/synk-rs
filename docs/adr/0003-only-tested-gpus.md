# Train only on hardware we can test: NVIDIA (CUDA) and CPU

Status: accepted, 2026-10-16. Amends [ADR 0001](0001-python-training-sidecar.md).

synk trains on NVIDIA GPUs through PyTorch's CUDA 13.0 build, or on the CPU.
AMD GPUs (ROCm) and Apple GPUs (MPS) are not supported: `synk setup
--backend rocm|mps` fails with `DEVICE_UNSUPPORTED` and points to `--backend
cpu`, auto-detection never picks them, and the trainer uses the GPU only
through a CUDA build of PyTorch (`torch.version.cuda`), so a ROCm or MPS
build in a custom `trainer.python` trains on the CPU.

The owner has no AMD or Apple hardware, and there is no GPU CI. CPU is tested
on every change; CUDA can be tested on a rented or free cloud GPU (Kaggle).
Claiming support for a path nobody has run would be a promise synk can't
keep, and ROCm in particular depends on per-card support lists and, under
WSL, on AMD's own PyTorch builds.

## Consequences

- AMD and Apple users train on the CPU, which is fine for
  `multilingual-e5-small` and slow for larger models.
- Adding ROCm or MPS back needs someone to run the end-to-end test on that
  hardware first; the code paths removed were small (a wheel index, a
  detection probe, a device name).
