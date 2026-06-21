# AGENTS.md

## Cursor Cloud specific instructions

This repo is a **CUDA GEMM (SGEMM) optimization walkthrough**: a set of standalone
CUDA C++ programs in `gemm/` (9 progressively optimized kernels), plus diagnostic
utilities in `device_info/` and `warp_granularity.cu` at the repo root. Each `.cu`
file is an independent `main()` that benchmarks a custom kernel against cuBLAS and
verifies correctness. There is no server/service, package manager, or database.

### Toolchain (installed by the update script)
- CUDA Toolkit 12.0 via Ubuntu's `nvidia-cuda-toolkit` apt package.
  - `nvcc` is at `/usr/bin/nvcc`; cuBLAS headers in `/usr/include`, libs in
    `/usr/lib/x86_64-linux-gnu` (`-lcublas` resolves from the default linker path).

### Building
The `gemm/Makefile` defaults to `CUDA_PATH ?= /data/cuda-12.6`, which does not exist
here. Override it to point at the apt-installed toolkit (which has `bin/nvcc`):

```bash
cd gemm
make CUDA_PATH=/usr            # builds all 9 GEMM variants
make clean CUDA_PATH=/usr      # remove binaries + .o files
```

Diagnostic utilities have no Makefile; compile directly, e.g.:
```bash
nvcc warp_granularity.cu -o warp_granularity
nvcc device_info/device_info.cu -o device_info_bin
```

There is no lint or automated test framework; "testing" a kernel means running its
binary, which prints GFLOPS/bandwidth and a correctness check vs cuBLAS.

### IMPORTANT: no GPU in the Cloud VM
The Cloud Agent VM has **no NVIDIA GPU** (no `nvidia-smi`, no `/dev/nvidia*`, no
NVIDIA PCI device, no driver). Code **compiles and links** fine, but the resulting
binaries **cannot run end-to-end** — they abort at CUDA/cuBLAS init, e.g.:

```
M=1024, N=1024, K=1024
CUBLAS initialization failure
```

To actually execute and benchmark the kernels you need a machine with an NVIDIA GPU
and matching driver. In this environment, validate changes via successful compilation
(`make CUDA_PATH=/usr`) only.
