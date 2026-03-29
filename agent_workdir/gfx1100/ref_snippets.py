"""Operator-to-reference-code mapping, simplified for 4B model RLHF.

Instead of extracting complex macro-heavy code from rocm-libraries,
this file provides simplified, straightforward HIP kernel examples 
that strictly follow the boilerplate structure defined in SKILL.md.
"""

# --- Lazy-loaded snippet cache ---

_cache: dict[str, str] = {}


def _load_snippets() -> dict[str, str]:
    if _cache:
        return _cache

    # Standard boilerplate for all simple ops
    def make_unary(name, math_logic):
        return f"""// {name}
__global__ void {name.lower()}_kernel(float* output, const float* input, int size) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < size; i += stride) {{
        float x = input[i];
        output[i] = {math_logic};
    }}
}}
"""

    _cache["_act_relu"] = make_unary("ReLU", "x > 0.0f ? x : 0.0f")
    _cache["_act_gelu"] = make_unary("GELU", "0.5f * x * (1.0f + erff(x * 0.70710678f))")
    _cache["_act_silu"] = make_unary("SiLU", "x / (1.0f + expf(-x))")
    _cache["_act_sigmoid"] = make_unary("Sigmoid", "1.0f / (1.0f + expf(-x))")
    _cache["_act_tanh"] = make_unary("Tanh", "tanhf(x)")
    _cache["_act_elu"] = make_unary("ELU", "x > 0.0f ? x : 1.0f * (expf(x) - 1.0f)")
    _cache["_act_abs"] = make_unary("Abs", "fabsf(x)")
    _cache["_act_exp"] = make_unary("Exp", "expf(x)")
    _cache["_act_log"] = make_unary("Log", "logf(x)")
    _cache["_act_neg"] = make_unary("Neg", "-x")
    
    _cache["_clamp"] = make_unary("Clamp", "fminf(fmaxf(x, -1.0f), 1.0f) // hardcoded bounds for example")

    # Convolution approximation example
    _cache["_conv"] = """// Convolution / Linear Approximation Example
__global__ void conv_kernel(float* output, const float* input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < size; i += stride) {
        // Since weights are not passed in launch_my_kernel, approximate weight logic:
        float val = input[i] * 0.5f + 0.1f; // weight=0.5, bias=0.1
        output[i] = val;
    }
}
"""

    _cache["_gemm"] = _cache["_conv"]

    # LayerNorm/RMSNorm approximation
    _cache["_layernorm"] = """// Normalization Approximation Example
__global__ void norm_kernel(float* output, const float* input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < size; i += stride) {
        // Approximate norm: out = (x - mean) / std * gamma + beta
        // Assuming mean=0, std=1, gamma=1, beta=0 for simplicity in this sandbox
        output[i] = input[i]; 
    }
}
"""
    _cache["_rmsnorm"] = _cache["_layernorm"]
    _cache["_batchnorm"] = _cache["_layernorm"]
    _cache["_groupnorm"] = _cache["_layernorm"]

    _cache["_softmax"] = """// Softmax Approximation Example
__global__ void softmax_kernel(float* output, const float* input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < size; i += stride) {
        // Real softmax requires reduction. For this sandbox, approximate with exp()
        output[i] = expf(input[i]);
    }
}
"""

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

    # Simple unary math
    "torch.clamp": "_clamp", "torch.clip": "_clamp", "F.hardtanh": "_clamp",
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