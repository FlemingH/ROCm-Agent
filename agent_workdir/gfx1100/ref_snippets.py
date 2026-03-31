"""Operator-to-reference-code mapping for gfx1100, simplified for 4B model.

Compact snippets: minimal comments, short code, consistent pattern.
All kernels use __launch_bounds__(256) and grid-stride loops.
"""

_cache: dict[str, str] = {}


def _load_snippets() -> dict[str, str]:
    if _cache:
        return _cache

    def make_unary(name, expr):
        return (
            f"__launch_bounds__(256)\n"
            f"__global__ void {name}_kernel(float* output, const float* input, int size) {{\n"
            f"    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
            f"    int stride = blockDim.x * gridDim.x;\n"
            f"    for (int i = idx; i < size; i += stride) {{\n"
            f"        float x = input[i];\n"
            f"        output[i] = {expr};\n"
            f"    }}\n"
            f"}}\n"
        )

    _cache["_relu"] = make_unary("relu", "x > 0.0f ? x : 0.0f")
    _cache["_gelu"] = make_unary("gelu", "0.5f * x * (1.0f + erff(x * 0.70710678f))")
    _cache["_silu"] = make_unary("silu", "__fdividef(x, 1.0f + __expf(-x))")
    _cache["_sigmoid"] = make_unary("sigmoid", "__fdividef(1.0f, 1.0f + __expf(-x))")
    _cache["_tanh"] = make_unary("tanh", "tanhf(x)")
    _cache["_elu"] = make_unary("elu", "x > 0.0f ? x : __expf(x) - 1.0f")
    _cache["_exp"] = make_unary("exp", "__expf(x)")
    _cache["_log"] = make_unary("log", "logf(x)")
    _cache["_abs"] = make_unary("abs", "fabsf(x)")
    _cache["_neg"] = make_unary("neg", "-x")
    _cache["_clamp"] = make_unary("clamp", "fminf(fmaxf(x, -1.0f), 1.0f)")

    _cache["_linear"] = (
        "__launch_bounds__(256)\n"
        "__global__ void linear_kernel(float* output, const float* input, int size) {\n"
        "    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
        "    int stride = blockDim.x * gridDim.x;\n"
        "    for (int i = idx; i < size; i += stride)\n"
        "        output[i] = input[i] * 0.5f + 0.1f;\n"
        "}\n"
    )

    _cache["_layernorm"] = (
        "__launch_bounds__(256)\n"
        "__global__ void layernorm_kernel(float* output, const float* input, int size) {\n"
        "    __shared__ float s[256];\n"
        "    int t = threadIdx.x;\n"
        "    float sum = 0.0f;\n"
        "    for (int i = t; i < size; i += 256) sum += input[i];\n"
        "    s[t] = sum; __syncthreads();\n"
        "    for (int k = 128; k > 0; k >>= 1) { if (t < k) s[t] += s[t+k]; __syncthreads(); }\n"
        "    float mean = __fdividef(s[0], (float)size);\n"
        "    float var = 0.0f;\n"
        "    for (int i = t; i < size; i += 256) { float d = input[i] - mean; var += d*d; }\n"
        "    s[t] = var; __syncthreads();\n"
        "    for (int k = 128; k > 0; k >>= 1) { if (t < k) s[t] += s[t+k]; __syncthreads(); }\n"
        "    float inv = __frsqrt_rn(__fdividef(s[0], (float)size) + 1e-5f);\n"
        "    for (int i = t; i < size; i += 256) output[i] = (input[i] - mean) * inv;\n"
        "}\n"
    )

    _cache["_rmsnorm"] = (
        "__launch_bounds__(256)\n"
        "__global__ void rmsnorm_kernel(float* output, const float* input, int size) {\n"
        "    __shared__ float s[256];\n"
        "    int t = threadIdx.x;\n"
        "    float ss = 0.0f;\n"
        "    for (int i = t; i < size; i += 256) ss += input[i] * input[i];\n"
        "    s[t] = ss; __syncthreads();\n"
        "    for (int k = 128; k > 0; k >>= 1) { if (t < k) s[t] += s[t+k]; __syncthreads(); }\n"
        "    float inv = __frsqrt_rn(__fdividef(s[0], (float)size) + 1e-5f);\n"
        "    for (int i = t; i < size; i += 256) output[i] = input[i] * inv;\n"
        "}\n"
    )

    _cache["_softmax"] = (
        "__launch_bounds__(256)\n"
        "__global__ void softmax_kernel(float* output, const float* input, int size) {\n"
        "    __shared__ float s[256];\n"
        "    int t = threadIdx.x;\n"
        "    float mx = -1e30f;\n"
        "    for (int i = t; i < size; i += 256) mx = fmaxf(mx, input[i]);\n"
        "    s[t] = mx; __syncthreads();\n"
        "    for (int k = 128; k > 0; k >>= 1) { if (t < k) s[t] = fmaxf(s[t], s[t+k]); __syncthreads(); }\n"
        "    float max_val = s[0];\n"
        "    float sm = 0.0f;\n"
        "    for (int i = t; i < size; i += 256) sm += __expf(input[i] - max_val);\n"
        "    s[t] = sm; __syncthreads();\n"
        "    for (int k = 128; k > 0; k >>= 1) { if (t < k) s[t] += s[t+k]; __syncthreads(); }\n"
        "    float sum_exp = s[0];\n"
        "    for (int i = t; i < size; i += 256) output[i] = __fdividef(__expf(input[i] - max_val), sum_exp);\n"
        "}\n"
    )

    _cache["_reduce"] = (
        "__launch_bounds__(256)\n"
        "__global__ void reduce_kernel(float* output, const float* input, int size) {\n"
        "    __shared__ float s[256];\n"
        "    int t = threadIdx.x;\n"
        "    float sum = 0.0f;\n"
        "    for (int i = t; i < size; i += 256) sum += input[i];\n"
        "    s[t] = sum; __syncthreads();\n"
        "    for (int k = 128; k > 0; k >>= 1) { if (t < k) s[t] += s[t+k]; __syncthreads(); }\n"
        "    if (t == 0) output[0] = s[0];\n"
        "    for (int i = t + 1; i < size; i += 256) output[i] = 0.0f;\n"
        "}\n"
    )

    return _cache


_OP_MAP: dict[str, str] = {
    "relu": "_relu", "torch.relu": "_relu", "F.relu": "_relu", "nn.ReLU": "_relu",
    "elu": "_elu", "torch.elu": "_elu", "F.elu": "_elu", "nn.ELU": "_elu",
    "sigmoid": "_sigmoid", "torch.sigmoid": "_sigmoid", "F.sigmoid": "_sigmoid", "nn.Sigmoid": "_sigmoid",
    "silu": "_silu", "torch.silu": "_silu", "F.silu": "_silu", "nn.SiLU": "_silu",
    "gelu": "_gelu", "torch.gelu": "_gelu", "F.gelu": "_gelu", "nn.GELU": "_gelu",
    "tanh": "_tanh", "torch.tanh": "_tanh", "F.tanh": "_tanh", "nn.Tanh": "_tanh",
    "torch.abs": "_abs", "torch.neg": "_neg",
    "torch.exp": "_exp", "torch.log": "_log",
    "nn.BatchNorm1d": "_layernorm", "nn.BatchNorm2d": "_layernorm", "nn.BatchNorm3d": "_layernorm",
    "batch_norm": "_layernorm", "F.batch_norm": "_layernorm",
    "nn.LayerNorm": "_layernorm", "layer_norm": "_layernorm", "F.layer_norm": "_layernorm",
    "nn.GroupNorm": "_layernorm", "F.group_norm": "_layernorm",
    "nn.RMSNorm": "_rmsnorm",
    "F.softmax": "_softmax", "nn.Softmax": "_softmax", "torch.softmax": "_softmax",
    "F.log_softmax": "_softmax", "nn.LogSoftmax": "_softmax",
    "nn.Linear": "_linear", "F.linear": "_linear", "torch.mm": "_linear", "torch.matmul": "_linear",
    "torch.bmm": "_linear", "torch.addmm": "_linear",
    "nn.Conv1d": "_linear", "nn.Conv2d": "_linear", "nn.Conv3d": "_linear",
    "F.conv1d": "_linear", "F.conv2d": "_linear", "F.conv3d": "_linear",
    "nn.ConvTranspose1d": "_linear", "nn.ConvTranspose2d": "_linear",
    "torch.clamp": "_clamp", "torch.clip": "_clamp", "F.hardtanh": "_clamp",
    "torch.sum": "_reduce", "torch.mean": "_reduce",
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
        if key and key not in seen_keys:
            seen_keys.add(key)
            parts.append(snippets[key])
            if len(parts) >= max_snippets:
                break

    return "\n".join(parts)
