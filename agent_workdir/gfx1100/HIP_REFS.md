# ROCm Libraries HIP Reference Index

Source: [rocm-libraries rocm-7.2.0](https://github.com/ROCm/rocm-libraries/releases/tag/rocm-7.2.0). All paths relative to `../../rocm-libraries/projects/`.

> **Current target: gfx1100 (RDNA 3)**
> - ✅ = Supported on gfx1100
> - ⚠️ = Not fully supported (restriction noted); may serve as API reference
> - CK examples: use `*_wmma*` variants on gfx11 (MFMA = CDNA only, WMMA = RDNA 3/4)
> - gfx11 lacks FP16/BF16 atomic add; split-K GEMM is limited
> - **CMake removing GPU restrictions ≠ runtime support** — some CK examples compile for gfx11 but use MFMA instructions internally

---

## 1. hipBLASLt — Fused GEMM (Priority 1)

43 samples in `hipblaslt/clients/samples/`. Common helper: `common/helper.h`.

### Basic GEMM

| Sample | Description |
|--------|-------------|
| `01_hipblaslt_gemm/` | Basic GEMM (EPILOGUE_DEFAULT) |
| `01_hipblaslt_gemm_ext/` | Basic GEMM (ext API) |
| `02_hipblaslt_gemm_batched/` | Batched GEMM |
| `02_hipblaslt_gemm_batched_ext/` | Batched GEMM (ext API) |
| `24_hipblaslt_gemm_with_TF32/` | TF32 precision GEMM |

### Epilogue Fusion (Bias / Activation)

| Sample | Fusion pattern |
|--------|---------------|
| `04_hipblaslt_gemm_bias/` | GEMM + Bias |
| `04_hipblaslt_gemm_bias_ext/` | GEMM + Bias (ext API) |
| `08_hipblaslt_gemm_gelu_aux_bias/` | GEMM + GELU + Bias |
| `08_hipblaslt_gemm_gelu_aux_bias_ext/` | GEMM + GELU + Bias (ext API) |
| `11_hipblaslt_gemm_bgradb/` | Bias gradient |
| `11_hipblaslt_gemm_ext_bgradb/` | Bias gradient (ext API) |
| `12_hipblaslt_gemm_dgelu_bgrad/` | GELU gradient + Bias gradient |
| `12_hipblaslt_gemm_dgelu_bgrad_ext/` | GELU gradient + Bias gradient (ext API) |
| `26_hipblaslt_gemm_swish_bias/` | GEMM + Swish + Bias |
| `27_hipblaslt_gemm_clamp_bias/` | GEMM + Clamp + Bias |

RELU epilogue: see `HIPBLASLT_EPILOGUE_RELU_BIAS` in `hipblaslt/clients/common/include/testing_matmul.hpp`.

### Quantization / Scaling

| Sample | Description |
|--------|-------------|
| `07_hipblaslt_gemm_alphavec_ext/` | Alpha vector scaling |
| `09_hipblaslt_gemm_amax/` | AMax output |
| `09_hipblaslt_gemm_amax_ext/` | AMax (ext API) |
| `10_hipblaslt_gemm_amax_with_scale/` | AMax + Scale |
| `10_hipblaslt_gemm_amax_with_scale_ext/` | AMax + Scale (ext API) |
| `15_hipblaslt_gemm_with_scale_a_b/` | Scale A/B |
| `15_hipblaslt_gemm_with_scale_a_b_ext/` | Scale A/B (ext API) |
| `15_hipblaslt_gemm_with_scale_a_b_vector/` | Scale A/B vector |
| `19_hipblaslt_gemm_mix_precision/` | Mixed precision GEMM |
| `19_hipblaslt_gemm_mix_precision_ext/` | Mixed precision (ext API) |
| `20_hipblaslt_gemm_mix_precision_with_amax_ext/` | Mixed precision + AMax |

### Grouped GEMM

| Sample | Description |
|--------|-------------|
| `16_hipblaslt_groupedgemm_ext/` | Grouped GEMM |
| `17_hipblaslt_groupedgemm_fixed_mk_ext/` | Grouped GEMM (fixed M×K) |
| `18_hipblaslt_groupedgemm_get_all_algos_ext/` | Grouped GEMM algorithm enumeration |

### Tuning / Layout

| Sample | Description |
|--------|-------------|
| `03_hipblaslt_gemm_tuning_splitk_ext/` | Split-K tuning |
| `05_hipblaslt_gemm_get_all_algos/` | Algorithm enumeration |
| `05_hipblaslt_gemm_get_all_algos_ext/` | Algorithm enumeration (ext API) |
| `06_hipblaslt_gemm_get_algo_by_index_ext/` | Select algorithm by index |
| `13_hipblaslt_gemm_is_tuned_ext/` | Check if tuned |
| `14_hipblaslt_gemm_tuning_wgm_ext/` | Workgroup tuning |
| `21_hipblaslt_gemm_attr_tciA_tciB/` | TCI attributes |
| `25_hipblaslt_gemm_swizzle_a/` | Swizzle A |
| `25_hipblaslt_gemm_swizzle_b/` | Swizzle B |
| `25_hipblaslt_gemm_bias_swizzle_a_ext/` | Bias + Swizzle A |
| `25_hipblaslt_weight_swizzle_padding/` | Weight Swizzle Padding |

### Extension Ops

| Sample | Description |
|--------|-------------|
| `22_hipblaslt_ext_op_layernorm/` | **hipBLASLt LayerNorm extension op** |
| `23_hipblaslt_ext_op_amax/` | hipBLASLt AMax extension op |

---

## 2. rocBLAS — GEMM (Priority 2)

Located in `rocblas/clients/samples/`.

| File | Description |
|------|-------------|
| `example_sgemm.cpp` | **SGEMM intro** (~190 lines, full workflow) |
| `example_sgemm_strided_batched.cpp` | Strided Batched SGEMM |
| `example_sgemm_multiple_strided_batch.cpp` | Multiple Strided Batched SGEMM |
| `example_gemm_strided_batched_ex_xdl_math_op.cpp` | Extended GEMM + XDL math mode |
| `example_user_driven_tuning.cpp` | GEMM implementation enumeration & tuning |
| `example_sscal.cpp` | BLAS1 vector scaling |
| `example_scal_template.cpp` | Templated SCAL |
| `example_scal_multiple_strided_batch.cpp` | Multi-batch SCAL |
| `example_gemv_graph_capture.cpp` | GEMV + HIP Graph capture |
| `example_hip_complex_her2.cpp` | Complex HER2 |
| `example_openmp.cpp` | OpenMP multi-thread multi-handle |
| `example_solver_rocblas.cpp` | Device memory management |

---

## 3. MIOpen — Conv / BN / Softmax / LN / Activation (Priority 2)

Drivers in `miopen/driver/`, tests in `miopen/test/`.

### Core Operators

| File | Operator |
|------|----------|
| `driver/conv_driver.hpp` | **Conv2D/3D** (Forward/Backward/WeightGrad) |
| `driver/softmax_driver.hpp` | **Softmax** |
| `driver/bn_driver.hpp` | **BatchNorm** (Inference/Training) |
| `driver/layernorm_driver.hpp` | **LayerNorm** |
| `driver/pool_driver.hpp` | **Pooling** (Max/Avg) |
| `driver/activ_driver.hpp` | **Activation** (ReLU/Sigmoid/Tanh etc.) |
| `driver/dropout_driver.hpp` | **Dropout** |
| `driver/groupnorm_driver.hpp` | **GroupNorm** |
| `driver/reduce_driver.hpp` | **Reduce** |
| `driver/gemm_driver.hpp` | GEMM (via MIOpen) |

### Transformer Operators

| File | Operator |
|------|----------|
| `driver/addlayernorm_driver.hpp` | **Add + LayerNorm** (residual + LN fusion) |
| `driver/t5layernorm_driver.hpp` | **T5 LayerNorm (RMSNorm)** |
| `driver/rope_driver.hpp` | **RoPE (Rotary Position Embedding)** |
| `driver/glu_driver.hpp` | **GLU (Gated Linear Unit)** |
| `driver/cat_driver.hpp` | Concat |
| `driver/prelu_driver.hpp` | PReLU |

### Fusion

| File | Description |
|------|-------------|
| `driver/CBAInferFusion_driver.hpp` | **Conv + BatchNorm + Activation fused inference** |

### Tests & Docs

| Path | Description |
|------|-------------|
| `test/main.cpp` | Conv2D full example |
| `test/gtest/conv_api_strided_tensors.cpp` | Strided tensor Conv |
| `test/gtest/softmax_find20.cpp` | Softmax Find 2.0 API |
| `test/gtest/bn.hpp` | BN multi-version API |
| `test/gtest/layernorm.hpp` | LayerNorm C++ wrapper |
| `docs/conceptual/porting-guide.rst` | **CUDA → MIOpen migration guide** |

---

## 4. rocPRIM — Reduce / Scan / Sort (Priority 2)

Examples in `rocprim/example/`, benchmarks in `rocprim/benchmark/`.

### Device-level

| File | Description |
|------|-------------|
| `benchmark/benchmark_device_reduce.cpp` | **Reduce (sum, etc.)** |
| `benchmark/benchmark_device_reduce_by_key.cpp` | Reduce by key |
| `benchmark/benchmark_device_scan.cpp` | **Scan (prefix sum)** |
| `benchmark/benchmark_device_scan_by_key.cpp` | Scan by key |
| `benchmark/benchmark_device_radix_sort.cpp` | **Radix Sort** |
| `benchmark/benchmark_device_radix_sort_onesweep.cpp` | Radix Sort (Onesweep) |
| `benchmark/benchmark_device_merge_sort.cpp` | Merge Sort |
| `benchmark/benchmark_device_merge.cpp` | Merge |
| `benchmark/benchmark_device_select.cpp` | Select |
| `benchmark/benchmark_device_partition.cpp` | Partition |
| `benchmark/benchmark_device_histogram.cpp` | Histogram |
| `benchmark/benchmark_device_transform.cpp` | **Transform (general)** |
| `benchmark/benchmark_device_binary_search.cpp` | Binary Search |
| `benchmark/benchmark_device_search.cpp` | Search |
| `benchmark/benchmark_device_nth_element.cpp` | Nth element |
| `benchmark/benchmark_device_segmented_reduce.cpp` | Segmented Reduce |
| `benchmark/benchmark_device_batch_memcpy.cpp` | Batch memcpy |

### Block / Warp Primitives

| File | Description |
|------|-------------|
| `benchmark/benchmark_block_reduce.cpp` | Block Reduce |
| `benchmark/benchmark_block_scan.cpp` | Block Scan |
| `benchmark/benchmark_block_sort.cpp` | Block Sort |
| `benchmark/benchmark_block_radix_sort.cpp` | Block Radix Sort |
| `benchmark/benchmark_block_exchange.cpp` | Block Exchange |
| `benchmark/benchmark_block_histogram.cpp` | Block Histogram |
| `benchmark/benchmark_warp_reduce.cpp` | Warp Reduce |
| `benchmark/benchmark_warp_scan.cpp` | Warp Scan |
| `benchmark/benchmark_warp_sort.cpp` | Warp Sort |

### Examples & Utilities

| File | Description |
|------|-------------|
| `example/rocprim/device/example_device_search.cpp` | Device Search example |
| `example/extra/example_temporary_storage.cpp` | Temporary storage management |
| `example/extra/example_type_traits_interface.cpp` | Type traits interface |

---

## 5. Composable Kernel — Advanced Fused Kernels (Priority 3)

### Classic Examples (`composablekernel/example/`)

#### GEMM Family

| Sample | Fusion pattern |
|--------|---------------|
| `01_gemm/` | Basic GEMM |
| `02_gemm_bilinear/` | GEMM + Bilinear |
| `03_gemm_bias_relu/` | **GEMM + Bias + ReLU** |
| `04_gemm_add_add_fastgelu/` | **GEMM + Add + Add + FastGELU** |
| `14_gemm_quantization/` | GEMM quantization |
| `15_grouped_gemm/` | Grouped GEMM |
| `16_gemm_multi_d_multi_reduces/` | GEMM + multi-D + multi-Reduce |
| `18_batched_gemm_reduce/` | Batched GEMM + Reduce |
| `21_gemm_layernorm/` | **GEMM + Bias + ReLU + Add + LayerNorm** |
| `22_cgemm/` | Complex GEMM |
| `24_batched_gemm/` | Batched GEMM |
| `25_gemm_bias_e_permute/` | GEMM + Bias + Permute |
| `28_grouped_gemm_bias_e_permute/` | Grouped GEMM + Bias + Permute |
| `29_batched_gemm_bias_e_permute/` | Batched GEMM + Bias + Permute |
| `31_batched_gemm_gemm/` | Batched GEMM + GEMM |
| `46_gemm_add_multiply/` | GEMM + Add + Multiply |
| `59_grouped_gemm_multi_ABD/` | Grouped GEMM multi-ABD |
| `60_gemm_multi_ABD/` | GEMM multi-ABD |
| `64_fpAintB_gemm/` | **FP×INT mixed-precision GEMM** |
| `65_gemm_multiply_multiply/` | GEMM + Multiply + Multiply |
| `67_gemm_microscaling/` | GEMM Microscaling |
| `68_gemm_add/` | GEMM + Add |
| `69_gemm_add_relu/` | **GEMM + Add + ReLU** |

#### Attention / Softmax

| Sample | Fusion pattern | gfx1100 |
|--------|---------------|---------|
| `23_softmax/` | **Softmax** | ✅ |
| `32_batched_gemm_scale_softmax_gemm/` | **Self-attention** — use `*_wmma_fp16.cpp` | ✅ WMMA |
| `37_batched_gemm_add_add_relu_gemm_add/` | Batched GEMM multi-step fusion | ✅ |
| `47_gemm_bias_softmax_gemm_permute/` | GEMM + Bias + Softmax + GEMM + Permute | ✅ XDL |

#### Conv Family

| Sample | Description |
|--------|-------------|
| `09_convnd_fwd/` | N-D Conv forward |
| `10_convnd_fwd_multiple_d_multiple_reduce/` | Conv + multi-D + multi-Reduce |
| `11_convnd_fwd_bias/` | Conv + Bias |
| `17_convnd_bwd_data/` | Conv backward (data) |
| `20_grouped_conv_bwd_weight/` | Grouped Conv backward (weight) |
| `30_grouped_conv_fwd_multiple_d/` | Grouped Conv forward + multi-D |
| `38_grouped_conv_bwd_data_multiple_d/` | Grouped Conv backward (data) + multi-D |
| `40_conv2d_fwd_quantization/` | Conv2D forward quantization |
| `41_grouped_conv_conv_fwd/` | Grouped Conv + Conv forward |
| `62_convnd_activ/` | **Conv + Activation** (gfx11 specific) |

#### Normalization / Reduce / Pooling

| Sample | Description |
|--------|-------------|
| `12_reduce/` | **Reduce** |
| `13_pool2d_fwd/` | **2D Pooling forward** |
| `27_layernorm2d_fwd/` | **LayerNorm2D forward** |
| `33_multiple_reduce/` | Multiple Reduce |
| `34_batchnorm/` | **BatchNorm** |
| `42_groupnorm_fwd/` | **GroupNorm forward** |
| `45_elementwise_normalization/` | Elementwise normalization |
| `48_pool3d_fwd/` | 3D Pooling forward |
| `49_maxpool2d_bwd/` | MaxPool2D backward |
| `51_avgpool3d_bwd/` | AvgPool3D backward |
| `53_layernorm2d_bwd/` | LayerNorm2D backward |
| `54_groupnorm_bwd/` | GroupNorm backward |
| `63_layernorm4d_fwd/` | LayerNorm4D forward |

#### Other

| Sample | Description | gfx1100 |
|--------|-------------|---------|
| `19_binary_elementwise/` | Binary elementwise ops | ✅ |
| `26_contraction/` | Tensor contraction | ✅ (FP32/64 limited) |
| `35_splitK_gemm/` | Split-K GEMM | ✅ (no FP16 atomic) |
| `36_sparse_embedding/` | Sparse embedding | ✅ |
| `39_permute/` | Permute | ✅ |
| `43_splitk_gemm_bias_e_permute/` | Split-K GEMM + Bias + Permute | ✅ |
| `44_elementwise_permute/` | Elementwise + Permute | ✅ |
| `50_put_element/` | Put Element | ✅ |
| `52_im2col_col2im/` | Im2Col / Col2Im | ✅ |
| `61_contraction_multi_ABD/` | Multi-ABD contraction | ✅ |
| `66_complex_contraction_bilinear/` | Complex contraction + Bilinear | ✅ |

### CK Tile Examples (`composablekernel/example/ck_tile/`)

| Sample | Description | gfx1100 |
|--------|-------------|---------|
| `01_fmha/` | Flash Multi-Head Attention (fwd + bwd) | ⚠️ gfx9/gfx12 only |
| `02_layernorm2d/` | **LayerNorm2D** | ✅ |
| `03_gemm/` | **Tile GEMM** | ⚠️ WMMA variants only (`universal_gemm`, `splitk`, `preshuffle`); `gemm_basic` uses MFMA |
| `04_img2col/` | Im2Col | ✅ |
| `05_reduce/` | Reduce | ✅ |
| `06_permute/` | Permute | ✅ |
| `09_topk_softmax/` | **TopK + Softmax** | ✅ |
| `10_rmsnorm2d/` | **RMSNorm2D** | ✅ |
| `11_add_rmsnorm2d_rdquant/` | **Add + RMSNorm + RD quant** | ✅ |
| `12_smoothquant/` | **SmoothQuant** | ✅ |
| `13_moe_sorting/` | MoE Sorting | ✅ |
| `14_moe_smoothquant/` | MoE + SmoothQuant | ✅ |
| `15_fused_moe/` | Fused MoE | ⚠️ uses MFMA instructions (CDNA only, despite no CMake restriction) |
| `16_batched_gemm/` | Batched GEMM | ✅ |
| `17_grouped_gemm/` | **Grouped GEMM** | ✅ (defaults to WMMA config) |
| `18_flatmm/` | Flat MatMul | ⚠️ gfx908/90a/942/950 only |
| `19_gemm_multi_d/` | GEMM + multi-D | ✅ |
| `20_grouped_convolution/` | Grouped Conv + Bias + Clamp | ✅ |
| `21_elementwise/` | **General elementwise ops** | ✅ |
| `22_gemm_multi_abd/` | GEMM multi-ABD | ✅ |
| `35_batched_transpose/` | Batched Transpose | ✅ (no matrix instructions) |
| `36_pooling/` | Pooling | ✅ |
| `37_transpose/` | Transpose | ✅ |
| `38_block_scale_gemm/` | Block Scale GEMM | ⚠️ gfx94/gfx95 only |
| `39_copy/` | Copy kernel | ✅ |
| `40_streamk_gemm/` | Stream-K GEMM | ⚠️ gfx9 only |
| `41_batched_contraction/` | Batched tensor contraction | ✅ |

### CK Tutorials (`composablekernel/tutorial/ck_tile/`)

| Sample | Description |
|--------|-------------|
| `00_copy_kernel/` | **Tile model intro** |
| `01_naive_gemm/` | Simple GEMM exercise |

---

## 6. rocWMMA — Matrix Acceleration

Located in `rocwmma/test/`.

| Path | Description |
|------|-------------|
| `test/gemm/gemm_kernel_base.cpp` | WMMA GEMM kernel base (16×16, 32×32 tile) |
| `test/gemm/gemm_resource.cpp` | GEMM test resource |
| `test/gemm/gemm_PGR1_LB2_MP0_MB_CP/test/workgroup/` | Various GEMM configurations |
| `test/unit/fill_fragment_test/` | Fragment initialization |
| `test/unit/transforms_test/` | Layout transforms |

---

## By Training Difficulty

### Easy (~30%): Single operator
- rocBLAS: `example_sgemm.cpp`, `example_sscal.cpp`
- rocPRIM: `benchmark_device_reduce.cpp`, `benchmark_device_scan.cpp`, `benchmark_device_transform.cpp`
- CK: `ck_tile/21_elementwise/`, `tutorial/01_naive_gemm/`, `01_gemm/`
- MIOpen: `activ_driver.hpp`, `softmax_driver.hpp`

### Medium (~45%): 2–3 operators with GEMM/Conv
- hipBLASLt: `04_gemm_bias/`, `08_gemm_gelu_aux_bias/`, `26_gemm_swish_bias/`
- CK: `03_gemm_bias_relu/`, `04_gemm_add_add_fastgelu/`, `68_gemm_add/`, `69_gemm_add_relu/`
- MIOpen: `conv_driver.hpp` + `bn_driver.hpp`, `CBAInferFusion_driver.hpp`

### Hard (~25%): 4–5 operators or full modules
- CK: `32_batched_gemm_scale_softmax_gemm/` with `*_wmma_fp16.cpp` (self-attention ✅)
- CK: `21_gemm_layernorm/` (GEMM + BN + ReLU + Add + LN ✅)
- CK: `47_gemm_bias_softmax_gemm_permute/` (✅)
- CK: ~~`ck_tile/15_fused_moe/`~~ (⚠️ MFMA only, not supported on gfx1100)
- CK: ~~`ck_tile/01_fmha/`~~ (⚠️ Flash Attention not supported on gfx1100, API reference only)
- MIOpen: `addlayernorm_driver.hpp` (residual + LN ✅) + `rope_driver.hpp` (RoPE ✅)
