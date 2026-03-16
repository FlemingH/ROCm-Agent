# ROCm Libraries HIP Reference Index

Source: [rocm-libraries rocm-7.2.0](https://github.com/ROCm/rocm-libraries/releases/tag/rocm-7.2.0). All paths relative to `../../rocm-libraries/projects/`.

> **Current target: gfx1201 (RDNA 4)**
> - ✅ = Supported on gfx1201
> - ⚠️ = Not supported (restriction noted); may serve as API reference
> - CK examples: WMMA variants work on gfx12; MFMA = CDNA only
> - **gfx12 gains Flash Attention support** (ck_tile/01_fmha/)
> - **gfx12 gains full Tile GEMM** (ck_tile/03_gemm/ all variants)
> - **gfx12 exclusive**: WMMA FP8 (`CK_USE_WMMA_FP8`) + OCP FP8 (`CK_USE_OCP_FP8`)
> - **MIOpen note**: `ALL_GPU_DATABASES` does not include gfx12 — first-run kernel compilation may be slower as MIOpen builds kernels on-the-fly instead of using pre-compiled databases

---

## 1. hipBLASLt — Fused GEMM (Priority 1)

43 samples in `hipblaslt/clients/samples/`. Common helper: `common/helper.h`.

### Basic GEMM

| Sample | Description | gfx1201 |
|--------|-------------|---------|
| `01_hipblaslt_gemm/` | Basic GEMM | ✅ |
| `02_hipblaslt_gemm_batched/` | Batched GEMM | ✅ |
| `24_hipblaslt_gemm_with_TF32/` | TF32 precision GEMM | ✅ |

### Epilogue Fusion (Bias / Activation)

| Sample | Fusion pattern | gfx1201 |
|--------|---------------|---------|
| `04_hipblaslt_gemm_bias/` | GEMM + Bias | ✅ |
| `08_hipblaslt_gemm_gelu_aux_bias/` | GEMM + GELU + Bias | ✅ |
| `26_hipblaslt_gemm_swish_bias/` | GEMM + Swish + Bias | ✅ |
| `27_hipblaslt_gemm_clamp_bias/` | GEMM + Clamp + Bias | ✅ |

### Grouped GEMM

| Sample | Description | gfx1201 |
|--------|-------------|---------|
| `16_hipblaslt_groupedgemm_ext/` | Grouped GEMM | ✅ |

### Extension Ops

| Sample | Description | gfx1201 |
|--------|-------------|---------|
| `22_hipblaslt_ext_op_layernorm/` | **hipBLASLt LayerNorm** | ✅ |
| `23_hipblaslt_ext_op_amax/` | hipBLASLt AMax | ✅ |

---

## 2. rocBLAS — GEMM (Priority 2)

| File | Description | gfx1201 |
|------|-------------|---------|
| `example_sgemm.cpp` | **SGEMM intro** | ✅ |
| `example_sgemm_strided_batched.cpp` | Strided Batched SGEMM | ✅ |
| `example_user_driven_tuning.cpp` | GEMM tuning | ✅ |
| `example_sscal.cpp` | BLAS1 vector scaling | ✅ |

---

## 3. MIOpen — Conv / BN / Softmax / LN / Activation (Priority 2)

### Core Operators

| File | Operator | gfx1201 |
|------|----------|---------|
| `driver/conv_driver.hpp` | **Conv2D/3D** | ✅ |
| `driver/softmax_driver.hpp` | **Softmax** | ✅ |
| `driver/bn_driver.hpp` | **BatchNorm** | ✅ |
| `driver/layernorm_driver.hpp` | **LayerNorm** | ✅ |
| `driver/pool_driver.hpp` | **Pooling** | ✅ |
| `driver/activ_driver.hpp` | **Activation** | ✅ |
| `driver/groupnorm_driver.hpp` | **GroupNorm** | ✅ |
| `driver/reduce_driver.hpp` | **Reduce** | ✅ |

### Transformer Operators

| File | Operator | gfx1201 |
|------|----------|---------|
| `driver/addlayernorm_driver.hpp` | **Add + LayerNorm** | ✅ |
| `driver/t5layernorm_driver.hpp` | **RMSNorm** | ✅ |
| `driver/rope_driver.hpp` | **RoPE** | ✅ |
| `driver/glu_driver.hpp` | **GLU** | ✅ |

### Fusion

| File | Description | gfx1201 |
|------|-------------|---------|
| `driver/CBAInferFusion_driver.hpp` | **Conv + BN + Activation** | ✅ |

---

## 4. rocPRIM — Reduce / Scan / Sort (Priority 2)

| File | Description | gfx1201 |
|------|-------------|---------|
| `benchmark/benchmark_device_reduce.cpp` | **Reduce** | ✅ |
| `benchmark/benchmark_device_scan.cpp` | **Scan** | ✅ |
| `benchmark/benchmark_device_radix_sort.cpp` | **Radix Sort** | ✅ |
| `benchmark/benchmark_device_transform.cpp` | **Transform** | ✅ |
| `benchmark/benchmark_block_reduce.cpp` | Block Reduce | ✅ |
| `benchmark/benchmark_warp_reduce.cpp` | Warp Reduce | ✅ |

---

## 5. Composable Kernel — Advanced Fused Kernels (Priority 3)

### GEMM Family

| Sample | Fusion pattern | gfx1201 |
|--------|---------------|---------|
| `01_gemm/` | Basic GEMM | ✅ |
| `03_gemm_bias_relu/` | **GEMM + Bias + ReLU** | ✅ |
| `04_gemm_add_add_fastgelu/` | **GEMM + Add + FastGELU** | ✅ |
| `21_gemm_layernorm/` | **GEMM + Bias + ReLU + Add + LN** | ✅ |
| `64_fpAintB_gemm/` | **FP×INT mixed-precision** | ✅ |
| `68_gemm_add/` | GEMM + Add | ✅ |
| `69_gemm_add_relu/` | GEMM + Add + ReLU | ✅ |

### Attention / Softmax

| Sample | Fusion pattern | gfx1201 |
|--------|---------------|---------|
| `23_softmax/` | **Softmax** | ✅ |
| `32_batched_gemm_scale_softmax_gemm/` | **Self-attention** | ✅ |
| `47_gemm_bias_softmax_gemm_permute/` | GEMM + Softmax + GEMM | ✅ |

### Conv Family

| Sample | Description | gfx1201 |
|--------|-------------|---------|
| `09_convnd_fwd/` | N-D Conv forward | ✅ |
| `11_convnd_fwd_bias/` | Conv + Bias | ✅ |
| `62_convnd_activ/` | **Conv + Activation** | ✅ |

### Normalization / Reduce / Pooling

| Sample | Description | gfx1201 |
|--------|-------------|---------|
| `12_reduce/` | Reduce | ✅ |
| `13_pool2d_fwd/` | 2D Pooling | ✅ |
| `27_layernorm2d_fwd/` | LayerNorm2D | ✅ |
| `34_batchnorm/` | BatchNorm | ✅ |
| `42_groupnorm_fwd/` | GroupNorm | ✅ |

### CK Tile Examples

| Sample | Description | gfx1201 |
|--------|-------------|---------|
| `01_fmha/` | **Flash Multi-Head Attention** | ✅ |
| `02_layernorm2d/` | **LayerNorm2D** | ✅ |
| `03_gemm/` | **Tile GEMM (all variants)** | ✅ |
| `09_topk_softmax/` | **TopK + Softmax** | ✅ |
| `10_rmsnorm2d/` | **RMSNorm2D** | ✅ |
| `11_add_rmsnorm2d_rdquant/` | **Add + RMSNorm + quant** | ✅ |
| `12_smoothquant/` | SmoothQuant | ✅ |
| `15_fused_moe/` | Fused MoE | ⚠️ MFMA only |
| `17_grouped_gemm/` | **Grouped GEMM** | ✅ |
| `18_flatmm/` | Flat MatMul | ⚠️ gfx9 only |
| `21_elementwise/` | **Elementwise ops** | ✅ |
| `38_block_scale_gemm/` | Block Scale GEMM | ⚠️ gfx94/95 only |
| `40_streamk_gemm/` | Stream-K GEMM | ⚠️ gfx9 only |

---

## 6. rocWMMA — Matrix Acceleration

| Path | Description | gfx1201 |
|------|-------------|---------|
| `test/gemm/gemm_kernel_base.cpp` | WMMA GEMM kernel | ✅ |

---

## By Training Difficulty

### Easy (~30%): Single operator
- rocBLAS: `example_sgemm.cpp`, `example_sscal.cpp`
- rocPRIM: `benchmark_device_reduce.cpp`, `benchmark_device_scan.cpp`
- CK: `ck_tile/21_elementwise/`, `01_gemm/`
- MIOpen: `activ_driver.hpp`, `softmax_driver.hpp`

### Medium (~45%): 2–3 operators with GEMM/Conv
- hipBLASLt: `04_gemm_bias/`, `08_gemm_gelu_aux_bias/`, `26_gemm_swish_bias/`
- CK: `03_gemm_bias_relu/`, `04_gemm_add_add_fastgelu/`, `68_gemm_add/`
- MIOpen: `conv_driver.hpp` + `bn_driver.hpp`, `CBAInferFusion_driver.hpp`

### Hard (~25%): 4–5 operators or full modules
- CK: `ck_tile/01_fmha/` ✅ **Flash Attention (gfx12 native!)**
- CK: `32_batched_gemm_scale_softmax_gemm/` (self-attention)
- CK: `21_gemm_layernorm/` (GEMM + BN + ReLU + Add + LN)
- CK: `47_gemm_bias_softmax_gemm_permute/`
- MIOpen: `addlayernorm_driver.hpp` + `rope_driver.hpp`
