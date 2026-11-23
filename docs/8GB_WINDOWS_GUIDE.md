# Strata `strata-inkling` — 8 GiB Windows CUDA Optimization Guide

This fork (`poog26/Strata`) stacks five clean commits on top of upstream `Niko1221/Strata` (`c6f6427`) to maximize **Inkling-2-Mini** (`SQ-10`) throughput and context length on 8 GiB NVIDIA GPUs under Windows 11 WDDM (tested on **RTX 3070 Ti Laptop 8 GiB / `sm_86`**, i9-12900H, 16 GB RAM).

---

## What Changed (Commit Stack on `main`)

1. **`b0728b3`** — Upstream PR #6586: fix `ep_dual_gpu.cu` null dereference (`--kv k8v8`) and `d_dense_x_` overrun (`--kv auto`).
2. **`c23f3da`** — Upstream PR #6593: eliminate `sm_86` register and local-memory stack spills in `ink_kernels.cuh`, `ink_moe_cuda.cuh`, and `ink_runner.cuh`.
3. **`ee11149`** — 8 GiB hybrid offload optimizations:
   - Keep single-GPU `--kv`, `--cpu-threads`, and `--vram-reserve-mib` runs on fast `InkCUDA` instead of diverting to slow `ep_dual_gpu`.
   - Route `M=1` (`--draft off`) through GPU Ping-Pong Async DMA (6.3× faster than Windows WDDM CPU doorbell).
   - Tight context-aware VRAM packing + wired host-tracked dynamic expert promotion ring (`step_promote_host`) + hybrid `moe_ratio=9.0` for DSpark.
4. **`2eeef70`** — Per-group-32 4-bit KV cache (`--kv k4v4`, **9.0 KiB/tok** across the 8 global layers) + ring-buffered DSpark KV/`d_f_rnn` (~36 MiB fixed regardless of `max_ctx`, down from 4.20 GiB at 200k) + `unembed_screen` reserve guard. Enables **128k–256k unclamped context** on 8 GiB VRAM.
5. **`879604d`** — Windows NVML full-VRAM utilization + post-init dynamic promo ring scoop:
   - Queries `nvml.dll` in `ink_cuda_mem_info()` on Windows WDDM to reclaim ~550–650 MiB hidden below `cudaMemGetInfo()`'s conservative WDDM cap.
   - Defers dynamic promo ring allocation until after `runner` + `DSpark` init so all remaining dedicated VRAM above the ~104 MiB WDDM DWM safety floor is packed cleanly (~99% dedicated VRAM utilization, zero shared-RAM paging).
   - Splits DSpark's ~64 MiB reserve so `--draft off` gives freed VRAM to extra experts.

---

## Verified Benchmarks (`Inkling-2-Mini-SQ-10.strata`, RTX 3070 Ti Laptop 8 GiB)

256-token real English multi-turn chat prompts:

| Config | VRAM Experts + Promo | VRAM Used / Free | Turn 1 | Turn 2 | PP |
|---|---:|---|---:|---:|---:|
| **8k `fp8`, `--draft auto`** *(max speed)* | `1,766 + 40` (`1,806`) | `7,933 / 85 MiB` (`98.9%`) | **29.52 t/s** | **32.03 t/s** | `169.5 t/s` |
| **200k `k4v4`, `--draft off`** *(steady long ctx)* | `1,192 + 49` (`1,241`) | `7,941 / 77 MiB` (`99.0%`) | **21.52 t/s** | **21.55 t/s** | `118.2 t/s` |
| **200k `k4v4`, `--draft auto`** *(predictable/code)* | `1,169 + 43` (`1,212`) | `7,950 / 68 MiB` (`99.1%`) | **19.60 t/s** | **21.66 t/s** | `121.7 t/s` |

> **Tip on DSpark (`--draft auto` vs `--draft off`):** Keep `--draft auto` at short context (8k–32k) where ~55–60% of experts live in VRAM. At 200k+ context (~40% experts in VRAM), `--draft off` puts freed VRAM into more experts and gives faster Turn 1 and steadier throughput across all prompt types.

---

## Ready-to-Run Commands

### 1. Max-Speed Short Context (8k `fp8`, ~32 tok/s)
```powershell
strata-inkling.exe --serve --native "path\to\Inkling-2-Mini-SQ-10.strata" --kv fp8 --max-context 8192 --draft auto
```

### 2. Long Context (200k `k4v4`, ~21.5 tok/s)
```powershell
strata-inkling.exe --serve --native "path\to\Inkling-2-Mini-SQ-10.strata" --kv k4v4 --max-context 204800 --draft off
```
- Supports `--max-context 131072` (128k), `204800` (200k), or `262144` (256k) unclamped.
- Optional `--vram-reserve-mib N` overrides the default `104 MiB` WDDM safety floor (avoid going below ~80 MiB on Windows 11 WDDM, as crossing the DWM floor triggers silent shared-RAM paging).

---

## Rebuilding on Windows 11 (MSVC + CUDA 13 + Ninja)

```powershell
# 1. Clone (or checkout the verified tag)
git clone https://github.com/poog26/Strata.git
cd Strata
# Optional: git checkout v0.2.19-8gb-opt

# 2. Configure & build using Visual Studio's bundled CMake + Ninja
#    (adjust vcvars64.bat / cmake.exe paths and CMAKE_CUDA_ARCHITECTURES for your GPU)
$vcvars = "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
$cmake  = "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"

cmd.exe /c "`"$vcvars`" && `"$cmake`" -S engine -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 && `"$cmake`" --build build --target strata-inkling -j 12"
```

Place `cublas64_13.dll`, `cublasLt64_13.dll`, and `cudart64_13.dll` (from CUDA Toolkit 13.x `bin\`) alongside `strata-inkling.exe` if CUDA `bin\` is not on `PATH`.

---

## Rebasing onto Future Upstream Releases

```powershell
git remote add upstream https://github.com/Niko1221/Strata.git
git fetch upstream
git checkout -b rebase-upstream main
git rebase upstream/main
```
*(Once upstream merges PR #6586 and PR #6593, `git rebase` will automatically skip `b0728b3` and `c23f3da` and cleanly replay only the 8 GiB hybrid/K4V4/NVML commits.)*
