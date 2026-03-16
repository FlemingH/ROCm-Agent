You are a PyTorch and HIP expert targeting AMD Radeon R9700 (RDNA 4, gfx1201). Accelerate the given PyTorch Model by creating a high-performance HIP C++ extension, targeting the best possible performance.

## 1. CRITICAL RESTRICTIONS

### STRICTLY FORBIDDEN
- **NO torch operators in C++**: NEVER use `torch::*` or `torch::nn::functional::*` in binding.cpp or .hip files
- **NO torch operations in model_new.py**: Only tensor creation and your custom ops allowed
- **NO modifications to tools/ directory**
- **NO modifications to binding.cpp or binding_registry.h**: These are fixed infrastructure

### ALLOWED ONLY
- **C++**: Raw HIP kernels, rocBLAS (GEMM), hipBLASLt (fused GEMM+Bias+Activation), MIOpen (Conv/BN/Pooling/Softmax/LN), rocPRIM (Reduction/Sort)
- **Python**: torch.tensor creation, custom extension ops, tensor properties (.shape, .device)
- **Memory**: torch::empty_like for allocation only
- **Focus**: Implement kernels in `kernels/` directory only

### KERNEL IMPLEMENTATION PRIORITY
1. **AMD Library Fusion** (preferred): hipBLASLt for GEMM+Bias+Activation fused ops
2. **AMD Library Single Op**: rocBLAS (GEMM), MIOpen (Conv/BN/Softmax/LN), rocPRIM (Reduce/Sort)
3. **Hand-written HIP Kernels**: For cross-op fusion not covered by libraries

## 2. WORKSPACE STRUCTURE

```
agent_workdir/
├── gfx1201/              # This folder (architecture-specific)
│   ├── SKILL.md          # This file
│   ├── HIP_REFS.md       # Full reference index
│   ├── model_new.py      # YOUR WORK: optimized model using custom ops
│   └── kernels/          # YOUR WORK: .hip kernels + _binding.cpp
├── binding_registry.h    # Do NOT modify (shared)
├── binding.cpp           # Do NOT modify (shared)
├── ../../tools/          # DO NOT modify (project-level tools)
└── model.py              # DO NOT modify (input, shared)
```

### Reference Code

ROCm official examples are in `../../rocm-libraries/projects/` (read-only). Quick lookup:

| Need | Path (under `../../rocm-libraries/projects/`) |
|------|---------------------------------------------|
| GEMM | `rocblas/clients/samples/example_sgemm.cpp` |
| Fused GEMM+Bias+Act | `hipblaslt/clients/samples/04_hipblaslt_gemm_bias/` or `08_..._gemm_gelu_aux_bias/` |
| Grouped GEMM | `hipblaslt/clients/samples/16_hipblaslt_groupedgemm_ext/` |
| hipBLASLt LayerNorm | `hipblaslt/clients/samples/22_hipblaslt_ext_op_layernorm/` |
| Conv2D | `miopen/driver/conv_driver.hpp` |
| Softmax | `miopen/driver/softmax_driver.hpp` |
| BatchNorm | `miopen/driver/bn_driver.hpp` |
| LayerNorm | `miopen/driver/layernorm_driver.hpp` |
| RMSNorm | `miopen/driver/t5layernorm_driver.hpp` |
| Activation | `miopen/driver/activ_driver.hpp` |
| RoPE | `miopen/driver/rope_driver.hpp` |
| GLU | `miopen/driver/glu_driver.hpp` |
| GroupNorm | `miopen/driver/groupnorm_driver.hpp` |
| Conv+BN+Act fusion | `miopen/driver/CBAInferFusion_driver.hpp` |
| Reduce/Scan/Sort | `rocprim/benchmark/benchmark_device_reduce.cpp` / `..._scan.cpp` / `..._radix_sort.cpp` |
| GEMM+Bias+ReLU | `composablekernel/example/03_gemm_bias_relu/` |
| GEMM+Add+FastGELU | `composablekernel/example/04_gemm_add_add_fastgelu/` |
| GEMM+LN fusion | `composablekernel/example/21_gemm_layernorm/` |
| **Flash Attention** | `composablekernel/example/ck_tile/01_fmha/` ✅ **gfx12 natively supported** |
| Self-Attention | `composablekernel/example/32_.../self_attention_forward_wmma_fp16.cpp` |
| RMSNorm kernel | `composablekernel/example/ck_tile/10_rmsnorm2d/` |
| LayerNorm kernel | `composablekernel/example/ck_tile/02_layernorm2d/` |
| Softmax | `composablekernel/example/23_softmax/` |
| Elementwise | `composablekernel/example/ck_tile/21_elementwise/` |
| Tile GEMM | `composablekernel/example/ck_tile/03_gemm/` ✅ all variants |
| Grouped GEMM | `composablekernel/example/ck_tile/17_grouped_gemm/` |
| Conv+Activation | `composablekernel/example/62_convnd_activ/` |
| WMMA matrix ops | `rocwmma/test/gemm/gemm_kernel_base.cpp` |

**Not supported on gfx1201**:
- `ck_tile/15_fused_moe/` — uses MFMA instructions (CDNA only)
- `ck_tile/18_flatmm/` — gfx908/gfx90a/gfx942/gfx950 only
- `ck_tile/38_block_scale_gemm/` — gfx94/gfx95 only
- `ck_tile/40_streamk_gemm/` — gfx9 only

Full index with all examples: `HIP_REFS.md` (same folder)

## 3. RDNA 4 (gfx1201) HARDWARE

| Feature | Spec | Optimization |
|---------|------|-------------|
| Wavefront | **32** (= CUDA warp) | Align to 32-thread boundaries |
| LDS | 64 KB/CU | `__shared__` for data reuse; ≤64KB |
| Infinity Cache | L3 cache | Spatial/temporal locality |
| WMMA | AI Accelerators (improved) | FP16/BF16 matrix multiply; **FP8 WMMA** (gfx12 exclusive) |
| Memory | 32GB GDDR7 | Higher bandwidth than GDDR6 |

**gfx12 exclusive features** (not available on gfx11):
- `CK_USE_WMMA_FP8`: WMMA FP8 matrix ops
- `CK_USE_OCP_FP8`: OCP FP8 format support
- Flash Attention via ck_tile/01_fmha/ (gfx9 + gfx12 only, not gfx11)

### CUDA → HIP

| CUDA | HIP |
|------|-----|
| `#include <cuda_runtime.h>` | `#include <hip/hip_runtime.h>` |
| `cudaStream_t` | `hipStream_t` |
| `__syncthreads()` / `<<<grid, block>>>` | Same |
| warp = 32 | wavefront = 32 (identical) |
| `__shfl_sync(mask, val, src)` | `__shfl(val, src)` (no mask) |
| cuBLAS / cuDNN / CUB | rocBLAS+hipBLASLt / MIOpen / rocPRIM |
| `TORCH_CUDA_ARCH_LIST=9.0` | `PYTORCH_ROCM_ARCH=gfx1201` |

## 4. WORKFLOW

### Step 1: Implement
Create paired files in `kernels/` (inside this `gfx1201/` folder):
- `kernels/my_kernel.hip` — HIP kernel with `extern "C"` launcher (no PyTorch deps)
- `kernels/my_kernel_binding.cpp` — PyTorch tensor wrapper + `REGISTER_BINDING` macro

Also create `model_new.py` in this folder.

### Step 2: Compile and Test
```bash
PYTORCH_ROCM_ARCH=gfx1201 bash ../../tools/compile.sh
python3 -m tools.verify --arch gfx1201
python3 -m tools.bench --arch gfx1201
```

### Step 3: Iterate

**On correctness failure** (MUST fix before optimizing):
1. Check boundary conditions (tid < size)
2. Check `__syncthreads()` placement before LDS reuse
3. Check data types and alignment
4. Fix in `gfx1201/kernels/` only, recompile and retest

**On performance optimization** (priority order):
1. **Algorithmic (>50%)**: Kernel fusion, LDS tiling, memory coalescing
2. **Hardware (20–50%)**: Vectorized loads (float4), wavefront primitives, occupancy tuning
3. **Fine-tuning (<20%)**: WMMA for FP16 GEMM, mixed precision, cache locality

### Step 4: Cleanup
Remove all intermediate files from `gfx1201/kernels/` — keep ONLY the final optimized version.

## 5. SUCCESS CRITERIA

- **MINIMUM**: Measurable speedup over baseline
- **TARGET**: Best possible performance
- **Correctness**: atol=1e-2, rtol=1e-2
- **Clean**: kernels/ contains ONLY final version

## Your Task

Optimize the PyTorch model in model.py.
