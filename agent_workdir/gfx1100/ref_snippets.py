"""Operator-to-reference-code mapping, simplified for 4B model RLHF.

Instead of extracting complex macro-heavy code from rocm-libraries,
this file provides simplified, straightforward HIP kernel examples 
that strictly follow the boilerplate structure defined in SKILL.md.

gfx1100 specific: wavefront=32, __launch_bounds__(256), shared memory
reduction patterns for normalization and softmax ops.
"""

# --- Lazy-loaded snippet cache ---

_cache: dict[str, str] = {}


def _load_snippets() -> dict[str, str]:
    if _cache:
        return _cache

    # Standard boilerplate for all simple ops
    def make_unary(name, math_logic):
        return f"""// {name} — elementwise, no reduction needed
__launch_bounds__(256)
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
    _cache["_act_silu"] = make_unary("SiLU", "__fdividef(x, 1.0f + __expf(-x))")
    _cache["_act_sigmoid"] = make_unary("Sigmoid", "__fdividef(1.0f, 1.0f + __expf(-x))")
    _cache["_act_tanh"] = make_unary("Tanh", "tanhf(x)")
    _cache["_act_elu"] = make_unary("ELU", "x > 0.0f ? x : 1.0f * (__expf(x) - 1.0f)")
    _cache["_act_abs"] = make_unary("Abs", "fabsf(x)")
    _cache["_act_exp"] = make_unary("Exp", "__expf(x)")
    _cache["_act_log"] = make_unary("Log", "logf(x)")
    _cache["_act_neg"] = make_unary("Neg", "-x")
    
    _cache["_clamp"] = make_unary("Clamp", "fminf(fmaxf(x, -1.0f), 1.0f) // hardcoded bounds for example")

    # Convolution approximation example
    _cache["_conv"] = """// Convolution / Linear Approximation Example
__launch_bounds__(256)
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

    # LayerNorm with shared memory reduction (correct pattern)
    _cache["_layernorm"] = """// LayerNorm with shared memory reduction
// Assumes entire input is one "row" to normalize (flat array).
// Uses 2-pass: pass1=compute mean, pass2=compute variance, then normalize.
__launch_bounds__(256)
__global__ void layernorm_kernel(float* output, const float* input, int size) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int stride = blockDim.x;

    // Pass 1: compute sum for mean
    float local_sum = 0.0f;
    for (int i = tid; i < size; i += stride)
        local_sum += input[i];
    sdata[tid] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float mean = __fdividef(sdata[0], (float)size);

    // Pass 2: compute sum of squared differences for variance
    float local_var = 0.0f;
    for (int i = tid; i < size; i += stride) {
        float diff = input[i] - mean;
        local_var += diff * diff;
    }
    sdata[tid] = local_var;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float inv_std = __frsqrt_rn(__fdividef(sdata[0], (float)size) + 1e-5f);

    // Normalize (gamma=1, beta=0)
    for (int i = tid; i < size; i += stride)
        output[i] = (input[i] - mean) * inv_std;
}
"""
    _cache["_rmsnorm"] = """// RMSNorm with shared memory reduction
__launch_bounds__(256)
__global__ void rmsnorm_kernel(float* output, const float* input, int size) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int stride = blockDim.x;

    // Compute sum of squares
    float local_ss = 0.0f;
    for (int i = tid; i < size; i += stride)
        local_ss += input[i] * input[i];
    sdata[tid] = local_ss;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float inv_rms = __frsqrt_rn(__fdividef(sdata[0], (float)size) + 1e-5f);

    for (int i = tid; i < size; i += stride)
        output[i] = input[i] * inv_rms;
}
"""
    _cache["_batchnorm"] = _cache["_layernorm"]
    _cache["_groupnorm"] = _cache["_layernorm"]

    # Softmax with shared memory reduction (numerically stable)
    _cache["_softmax"] = """// Softmax with shared memory reduction (numerically stable)
// 1. Find max (for numerical stability)
// 2. Compute sum of exp(x - max)
// 3. Divide each exp by sum
__launch_bounds__(256)
__global__ void softmax_kernel(float* output, const float* input, int size) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int stride = blockDim.x;

    // Pass 1: find max
    float local_max = -1e30f;
    for (int i = tid; i < size; i += stride)
        local_max = fmaxf(local_max, input[i]);
    sdata[tid] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    float max_val = sdata[0];

    // Pass 2: sum of exp(x - max)
    float local_sum = 0.0f;
    for (int i = tid; i < size; i += stride)
        local_sum += __expf(input[i] - max_val);
    sdata[tid] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float sum_exp = sdata[0];

    // Pass 3: normalize
    for (int i = tid; i < size; i += stride)
        output[i] = __fdividef(__expf(input[i] - max_val), sum_exp);
}
"""

    # Sum reduction example (for torch.sum, torch.mean)
    _cache["_reduce_sum"] = """// Sum Reduction with shared memory
// Writes total sum to output[0]. Zero-fills rest of output.
__launch_bounds__(256)
__global__ void reduce_sum_kernel(float* output, const float* input, int size) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int stride = blockDim.x;

    float local_sum = 0.0f;
    for (int i = tid; i < size; i += stride)
        local_sum += input[i];
    sdata[tid] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) output[0] = sdata[0];
    // Zero fill remaining output
    for (int i = tid + 1; i < size; i += stride)
        output[i] = 0.0f;
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

    # Reduction ops
    "torch.sum": "_reduce_sum", "torch.mean": "_reduce_sum",
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

    if not parts:
        return ""
    return "\n".join(parts)
