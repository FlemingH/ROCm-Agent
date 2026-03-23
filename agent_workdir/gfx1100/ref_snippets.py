"""Operator-to-reference-code mapping, extracted live from rocm-libraries 7.2.0.

Reads actual source files from rocm-libraries/projects/ and extracts relevant
code snippets for each operator type. Used by prepare_data.py to inject
targeted HIP kernel references into training prompts.
"""

import re
from pathlib import Path

ROCM_LIBS = Path(__file__).resolve().parent.parent.parent / "rocm-libraries" / "projects"


def _read_lines(rel_path: str, start: int, end: int) -> str:
    """Read lines [start, end] (1-indexed inclusive) from a rocm-libraries file."""
    fp = ROCM_LIBS / rel_path
    if not fp.exists():
        return ""
    lines = fp.read_text(errors="replace").splitlines()
    return "\n".join(lines[start - 1 : end])


def _extract_struct(rel_path: str, struct_name: str, max_lines: int = 30) -> str:
    """Extract a struct/class body by name from a C++ file."""
    fp = ROCM_LIBS / rel_path
    if not fp.exists():
        return ""
    text = fp.read_text(errors="replace")
    pattern = re.compile(
        rf"struct\s+{struct_name}\b.*?\{{(.*?)\n\}};",
        re.DOTALL,
    )
    m = pattern.search(text)
    if not m:
        return ""
    body = m.group(0)
    lines = body.splitlines()[:max_lines]
    return "\n".join(lines)


# --- Lazy-loaded snippet cache ---

_cache: dict[str, str] = {}


def _load_snippets() -> dict[str, str]:
    if _cache:
        return _cache

    CK_ELEM = "composablekernel/include/ck_tile/ops/elementwise/unary_element_wise_operation.hpp"
    CK_BIN = "composablekernel/include/ck_tile/ops/elementwise/binary_elementwise_operation.hpp"
    BN_INFER = "miopen/src/kernels/MIOpenBatchNormActivInfer.cl"
    LN_REF = "composablekernel/include/ck_tile/host/reference/reference_layernorm2d_fwd.hpp"
    REDUCE_KERN = "rocprim/test/rocprim/test_block_reduce.kernels.hpp"
    GEMM_SAMPLE = "rocblas/clients/samples/example_sgemm.cpp"

    # --- Activations from CK ---
    for name in ["Relu", "Sigmoid", "Silu", "TanH", "Elu", "Gelu"]:
        code = _extract_struct(CK_ELEM, name)
        if code:
            header = f"// {name} activation — from composablekernel unary_element_wise_operation.hpp"
            _cache[f"_act_{name.lower()}"] = f"{header}\n{code}"

    # --- BatchNorm inference from MIOpen ---
    bn_code = _read_lines(BN_INFER, 32, 105)
    if bn_code:
        _cache["_batchnorm"] = (
            "// BatchNorm inference — from miopen MIOpenBatchNormActivInfer.cl\n"
            "// Core: invVariance = rsqrt(var + eps); out = scale * (in - mean) * invVariance + bias\n"
            + bn_code
        )

    # --- LayerNorm from CK reference ---
    ln_code = _read_lines(LN_REF, 48, 90)
    if ln_code:
        _cache["_layernorm"] = (
            "// LayerNorm — from composablekernel reference_layernorm2d_fwd.hpp (Welford)\n"
            + ln_code
        )

    # --- Block Reduce from rocPRIM ---
    reduce_code = _read_lines(REDUCE_KERN, 43, 62)
    if reduce_code:
        _cache["_reduce"] = (
            "// Block Reduce — from rocprim test_block_reduce.kernels.hpp\n"
            + reduce_code
        )

    # --- GEMM from rocBLAS sample ---
    gemm_fp = ROCM_LIBS / GEMM_SAMPLE
    if gemm_fp.exists():
        text = gemm_fp.read_text(errors="replace")
        m = re.search(r"(rocblas_sgemm\(.*?\);)", text, re.DOTALL)
        if m:
            _cache["_gemm"] = (
                "// GEMM — from rocblas example_sgemm.cpp\n"
                "// rocblas_sgemm(handle, transA, transB, M, N, K, &alpha, A, lda, B, ldb, &beta, C, ldc)\n"
                f"{m.group(1)}"
            )

    # --- Binary elementwise from CK ---
    for name in ["Add", "Subtract", "Multiply"]:
        code = _extract_struct(CK_BIN, name, max_lines=15)
        if code:
            _cache[f"_bin_{name.lower()}"] = (
                f"// {name} — from composablekernel binary_elementwise_operation.hpp\n{code}"
            )

    # --- Extra unary ops from CK ---
    for name in ["UnaryAbs", "Exp", "Log", "Neg"]:
        code = _extract_struct(CK_ELEM, name, max_lines=15)
        if code:
            key = name.replace("Unary", "").lower()
            _cache[f"_act_{key}"] = (
                f"// {name} — from composablekernel unary_element_wise_operation.hpp\n{code}"
            )

    # --- RMSNorm from CK reference ---
    RMSNORM_REF = "composablekernel/include/ck_tile/host/reference/reference_rmsnorm2d_fwd.hpp"
    rms_code = _read_lines(RMSNORM_REF, 50, 97)
    if rms_code:
        _cache["_rmsnorm"] = (
            "// RMSNorm — from composablekernel reference_rmsnorm2d_fwd.hpp\n"
            "// Core: rms = sqrt(mean(x^2) + eps); out = x / rms * gamma\n"
            + rms_code
        )

    # --- Softmax from MIOpen ---
    SOFTMAX_KERN = "miopen/src/kernels/MIOpenSoftmax.cl"
    sm_code = _read_lines(SOFTMAX_KERN, 239, 330)
    if sm_code:
        _cache["_softmax"] = (
            "// Softmax — from miopen MIOpenSoftmax.cl\n"
            "// Algorithm: max_reduce → exp(x - max) → sum_reduce → divide\n"
            + sm_code
        )

    # --- GroupNorm from MIOpen ---
    GN_KERN = "miopen/src/kernels/MIOpenGroupNorm.cpp"
    gn_code = _read_lines(GN_KERN, 35, 121)
    if gn_code:
        _cache["_groupnorm"] = (
            "// GroupNorm — from miopen MIOpenGroupNorm.cpp\n"
            + gn_code
        )

    # --- MaxPool2D from CK reference ---
    POOL_REF = "composablekernel/include/ck_tile/host/reference/reference_pool.hpp"
    pool_code = _read_lines(POOL_REF, 52, 84)
    if pool_code:
        _cache["_pool"] = (
            "// MaxPool2D — from composablekernel reference_pool.hpp\n"
            + pool_code
        )

    # --- Conv1x1 from MIOpen ---
    CONV1X1 = "miopen/src/kernels/MIOpenConv1x1.cl"
    conv_code = _read_lines(CONV1X1, 303, 343)
    if conv_code:
        _cache["_conv"] = (
            "// Conv1x1 core loop — from miopen MIOpenConv1x1.cl\n"
            + conv_code
        )

    # --- Simple elementwise ops (hardcoded, no rocm-libraries source) ---
    _cache["_clamp"] = (
        "// Clamp: out = min(max(x, lo), hi)\n"
        "__global__ void clamp_kernel(float* out, const float* in, float lo, float hi, int n) {\n"
        "    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
        "    if (i < n) out[i] = fminf(fmaxf(in[i], lo), hi);\n"
        "}"
    )
    _cache["_where"] = (
        "// Where: out = cond ? x : y\n"
        "__global__ void where_kernel(float* out, const bool* cond, const float* x, const float* y, int n) {\n"
        "    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
        "    if (i < n) out[i] = cond[i] ? x[i] : y[i];\n"
        "}"
    )
    _cache["_pow"] = (
        "// Pow: out = pow(x, p)\n"
        "__global__ void pow_kernel(float* out, const float* in, float p, int n) {\n"
        "    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
        "    if (i < n) out[i] = powf(in[i], p);\n"
        "}"
    )
    _cache["_reciprocal"] = (
        "// Reciprocal: out = 1 / x\n"
        "__global__ void reciprocal_kernel(float* out, const float* in, int n) {\n"
        "    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
        "    if (i < n) out[i] = 1.0f / in[i];\n"
        "}"
    )
    _cache["_sign"] = (
        "// Sign: out = (x > 0) - (x < 0)\n"
        "__global__ void sign_kernel(float* out, const float* in, int n) {\n"
        "    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
        "    if (i < n) { float x = in[i]; out[i] = (x > 0.0f) - (x < 0.0f); }\n"
        "}"
    )
    _cache["_act_leaky_relu"] = (
        "// LeakyReLU: out = x > 0 ? x : alpha * x\n"
        "__global__ void leaky_relu_kernel(float* out, const float* in, float alpha, int n) {\n"
        "    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
        "    if (i < n) { float x = in[i]; out[i] = x > 0.0f ? x : alpha * x; }\n"
        "}"
    )
    _cache["_act_hardswish"] = (
        "// HardSwish: out = x * clamp(x+3, 0, 6) / 6\n"
        "__global__ void hardswish_kernel(float* out, const float* in, int n) {\n"
        "    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
        "    if (i < n) { float x = in[i]; out[i] = x * fminf(fmaxf(x + 3.0f, 0.0f), 6.0f) / 6.0f; }\n"
        "}"
    )
    _cache["_act_hardsigmoid"] = (
        "// HardSigmoid: out = clamp(x/6 + 0.5, 0, 1)\n"
        "__global__ void hardsigmoid_kernel(float* out, const float* in, int n) {\n"
        "    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
        "    if (i < n) out[i] = fminf(fmaxf(in[i] / 6.0f + 0.5f, 0.0f), 1.0f);\n"
        "}"
    )
    _cache["_act_mish"] = (
        "// Mish: out = x * tanh(softplus(x)) = x * tanh(log(1 + exp(x)))\n"
        "__global__ void mish_kernel(float* out, const float* in, int n) {\n"
        "    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
        "    if (i < n) { float x = in[i]; out[i] = x * tanhf(logf(1.0f + expf(x))); }\n"
        "}"
    )

    return _cache


# --- Operator name to snippet key mapping ---

_OP_MAP: dict[str, str] = {
    # Activations
    "relu": "_act_relu", "torch.relu": "_act_relu", "F.relu": "_act_relu", "nn.ReLU": "_act_relu",
    "elu": "_act_elu", "torch.elu": "_act_elu", "F.elu": "_act_elu", "nn.ELU": "_act_elu",
    "sigmoid": "_act_sigmoid", "torch.sigmoid": "_act_sigmoid", "F.sigmoid": "_act_sigmoid", "nn.Sigmoid": "_act_sigmoid",
    "silu": "_act_silu", "torch.silu": "_act_silu", "F.silu": "_act_silu", "nn.SiLU": "_act_silu",
    "gelu": "_act_gelu", "torch.gelu": "_act_gelu", "F.gelu": "_act_gelu", "nn.GELU": "_act_gelu",
    "tanh": "_act_tanh", "torch.tanh": "_act_tanh", "F.tanh": "_act_tanh", "nn.Tanh": "_act_tanh",

    # Unary math
    "torch.abs": "_act_abs", "torch.neg": "_act_neg",
    "torch.exp": "_act_exp", "torch.log": "_act_log",

    # Normalization
    "nn.BatchNorm1d": "_batchnorm", "nn.BatchNorm2d": "_batchnorm", "nn.BatchNorm3d": "_batchnorm",
    "batch_norm": "_batchnorm", "F.batch_norm": "_batchnorm",
    "nn.LayerNorm": "_layernorm", "layer_norm": "_layernorm", "F.layer_norm": "_layernorm",
    "nn.GroupNorm": "_groupnorm", "F.group_norm": "_groupnorm",
    "nn.RMSNorm": "_rmsnorm",

    # Softmax
    "F.softmax": "_softmax", "nn.Softmax": "_softmax", "torch.softmax": "_softmax",
    "F.log_softmax": "_softmax", "nn.LogSoftmax": "_softmax",

    # Compute
    "nn.Linear": "_gemm", "F.linear": "_gemm", "torch.mm": "_gemm", "torch.matmul": "_gemm",
    "torch.bmm": "_gemm", "torch.addmm": "_gemm",
    "nn.Conv1d": "_conv", "nn.Conv2d": "_conv", "nn.Conv3d": "_conv",
    "F.conv1d": "_conv", "F.conv2d": "_conv", "F.conv3d": "_conv",
    "nn.ConvTranspose1d": "_conv", "nn.ConvTranspose2d": "_conv",

    # Reduction
    "torch.sum": "_reduce", "torch.mean": "_reduce", "torch.max": "_reduce",
    "torch.min": "_reduce", "torch.prod": "_reduce",

    # Pooling
    "nn.MaxPool2d": "_pool", "F.max_pool2d": "_pool",
    "nn.AvgPool2d": "_pool", "F.avg_pool2d": "_pool",
    "nn.AdaptiveAvgPool2d": "_pool", "F.adaptive_avg_pool2d": "_pool",
    "nn.AdaptiveMaxPool2d": "_pool",

    # Binary elementwise
    "torch.add": "_bin_add", "add": "_bin_add",
    "torch.mul": "_bin_multiply", "mul": "_bin_multiply", "torch.multiply": "_bin_multiply",
    "torch.sub": "_bin_subtract", "sub": "_bin_subtract",

    # Simple unary math
    "torch.clamp": "_clamp", "torch.clip": "_clamp", "F.hardtanh": "_clamp",
    "torch.where": "_where",
    "torch.pow": "_pow", "torch.square": "_pow",
    "torch.reciprocal": "_reciprocal",
    "torch.sign": "_sign",

    # Extra activations
    "F.leaky_relu": "_act_leaky_relu", "nn.LeakyReLU": "_act_leaky_relu",
    "F.hardswish": "_act_hardswish", "nn.Hardswish": "_act_hardswish",
    "F.hardsigmoid": "_act_hardsigmoid", "nn.Hardsigmoid": "_act_hardsigmoid",
    "F.mish": "_act_mish", "nn.Mish": "_act_mish",
}


def get_ref_code(ops: list[str], max_snippets: int = 3) -> str:
    """Return deduplicated reference snippets for a list of operator names."""
    snippets = _load_snippets()
    seen_keys: set[str] = set()
    parts: list[str] = []

    for op in ops:
        key = _OP_MAP.get(op)
        if key is None:
            for map_op, map_key in _OP_MAP.items():
                if map_op.lower() in op.lower() or op.lower() in map_op.lower():
                    key = map_key
                    break
        if key and key not in seen_keys and key in snippets:
            seen_keys.add(key)
            parts.append(snippets[key])
            if len(parts) >= max_snippets:
                break

    return "\n\n".join(parts)
