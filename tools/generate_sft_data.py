#!/usr/bin/env python3
"""Generate SFT training data: frontier model → HIP kernel → compile → verify.

Pipeline:
  1. Load parquet, filter to binding-compatible (single-input, no learnable params)
  2. Call vLLM frontier model to generate HIP kernel code
  3. Parse kernel's launch_fused_kernel signature → dynamic binding + model_new
  4. Compile with hipcc in a sandbox
  5. Verify correctness: run Model vs ModelNew on GPU, compare outputs
  6. Save compile+verify passing pairs as SFT parquet

Usage:
    python tools/generate_sft_data.py \
        --input data/rocm_agent_ops_v4/train.parquet \
        --output data/sft_data \
        --arch gfx1100 \
        --vllm-url http://localhost:8001/v1 \
        --candidates 3 \
        --verify-gpu 0
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import requests

ROOT = Path(__file__).resolve().parent.parent

# ---------- Filtering ----------

PARAM_KEYWORDS = [
    "nn.Linear", "nn.Conv", "nn.BatchNorm", "nn.LayerNorm",
    "nn.GroupNorm", "nn.RMSNorm", "self.weight", "self.bias", "nn.Embedding",
    "nn.LSTM", "nn.GRU", "nn.RNN", "nn.Transformer",
    "nn.MultiheadAttention", "nn.Parameter",
]


def is_binding_compatible(model_code: str) -> bool:
    """Return True if the model has no learnable params and single forward input."""
    has_param = any(kw in model_code for kw in PARAM_KEYWORDS)
    if has_param:
        return False
    fwd = re.search(r"def forward\(self,\s*(.*?)\)", model_code)
    if not fwd:
        return False
    args = [a.strip() for a in fwd.group(1).split(",") if a.strip()]
    return len(args) == 1


# Operations that change output shape — incompatible with empty_like(input) binding
_NON_ELEMENTWISE_PATTERNS = [
    # Reductions
    r'torch\.mean\s*\(', r'torch\.sum\s*\(', r'torch\.prod\s*\(',
    r'torch\.norm\s*\(', r'torch\.var\s*\(', r'torch\.std\s*\(',
    r'torch\.argmax', r'torch\.argmin', r'torch\.median',
    r'torch\.max\s*\(', r'torch\.min\s*\(',
    r'\.cumsum\s*\(', r'\.cumprod\s*\(',
    r'F\.softmax\s*\(', r'F\.log_softmax\s*\(',
    r'\.softmax\s*\(', r'\.log_softmax\s*\(',
    # Shape / layout changes
    r'\.view\s*\(', r'\.reshape\s*\(',
    r'\.permute\s*\(', r'\.transpose\s*\(',
    r'\.unsqueeze\s*\(', r'\.squeeze\s*\(',
    r'\.flatten\s*\(', r'\.expand\s*\(',
    r'torch\.cat\s*\(', r'torch\.stack\s*\(',
    r'torch\.flip\s*\(', r'torch\.roll\s*\(',
    r'torch\.gather\s*\(', r'torch\.scatter\s*\(',
    r'torch\.index_select\s*\(',
    # Interpolation / resampling
    r'F\.interpolate\s*\(', r'\.upsample\s*\(',
    r'nn\.Upsample', r'F\.upsample',
    # Matrix ops
    r'torch\.matmul\s*\(', r'torch\.bmm\s*\(', r'torch\.mm\s*\(',
    r'torch\.einsum\s*\(',
    # FFT / spectral
    r'torch\.stft', r'torch\.fft', r'torch\.istft',
    # Pooling / conv
    r'F\.avg_pool', r'F\.max_pool', r'F\.adaptive_',
    r'F\.conv\b', r'F\.linear\b',
    # Triangular
    r'torch\.tril\s*\(', r'torch\.triu\s*\(',
    # Sorting
    r'torch\.sort\s*\(', r'torch\.topk\s*\(',
    # Type changes
    r'torch\.bucketize\s*\(', r'\.to\s*\(torch\.',
    # Complex / polar
    r'torch\.polar\s*\(', r'torch\.complex\s*\(',
    # Normalization (requires cross-element computation)
    r'\.instance_norm\s*\(', r'nn\.InstanceNorm',
    r'F\.instance_norm\s*\(', r'\.norm\s*\(',
    # Tensor creation in forward (can't handle in binding)
    r'torch\.randn\s*\(', r'torch\.zeros\s*\(', r'torch\.ones\s*\(',
    r'torch\.arange\s*\(', r'torch\.linspace\s*\(',
]


def is_elementwise_compatible(model_code: str) -> bool:
    """Return True if forward body uses only elementwise operations."""
    fwd = re.search(
        r'def forward\(self,.*?\):\s*\n(.*?)(?:\ndef |\nclass |\Z)',
        model_code, re.DOTALL,
    )
    if not fwd:
        return False
    body = fwd.group(1)
    return not any(re.search(p, body) for p in _NON_ELEMENTWISE_PATTERNS)


def batch_check_output_shapes(model_codes: list[str]) -> list[bool]:
    """Check output shape == input shape for a batch of models in ONE subprocess.

    Much faster than spawning per-sample because torch is imported only once.
    """
    if not model_codes:
        return []

    SENTINEL = "###__SHAPE_CHECK_SEPARATOR__###"
    # Build script as string concat to avoid .format() brace conflicts
    batch_script = (
        "import sys, torch, torch.nn as nn, torch.nn.functional as F\n"
        "SENTINEL = " + repr(SENTINEL) + "\n"
        "codes = open(sys.argv[1]).read().split(SENTINEL)\n"
        "for i, code in enumerate(codes):\n"
        "    code = code.strip()\n"
        "    if not code:\n"
        "        continue\n"
        "    try:\n"
        "        ns = {}\n"
        "        exec(code, ns)\n"
        "        Model = ns['Model']\n"
        "        get_inputs = ns['get_inputs']\n"
        "        get_init_inputs = ns['get_init_inputs']\n"
        "        init_inputs = get_init_inputs()\n"
        "        if not isinstance(init_inputs, (list, tuple)): init_inputs = [init_inputs]\n"
        "        model = Model(*init_inputs).eval()\n"
        "        inputs = get_inputs()\n"
        "        if not isinstance(inputs, (list, tuple)): inputs = [inputs]\n"
        "        with torch.no_grad():\n"
        "            output = model(*inputs)\n"
        "        if (isinstance(output, torch.Tensor)\n"
        "                and isinstance(inputs[0], torch.Tensor)\n"
        "                and output.shape == inputs[0].shape\n"
        "                and output.dtype == torch.float32):\n"
        "            print(str(i) + ':OK')\n"
        "        else: print(str(i) + ':MISMATCH')\n"
        "    except Exception: print(str(i) + ':ERROR')\n"
    )

    datafile = Path(tempfile.mktemp(suffix=".txt", prefix="shape_data_"))
    scriptfile = Path(tempfile.mktemp(suffix=".py", prefix="shape_batch_"))
    try:
        scriptfile.write_text(batch_script)
        datafile.write_text(SENTINEL.join(model_codes))
        result = subprocess.run(
            [sys.executable, str(scriptfile), str(datafile)],
            capture_output=True, text=True,
            timeout=max(120, len(model_codes) * 8),
            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        )
        ok_set = set()
        for line in result.stdout.strip().splitlines():
            if ":OK" in line:
                idx = int(line.split(":")[0])
                ok_set.add(idx)
        return [i in ok_set for i in range(len(model_codes))]
    except Exception:
        return [False] * len(model_codes)
    finally:
        scriptfile.unlink(missing_ok=True)
        datafile.unlink(missing_ok=True)

def patch_system_prompt(messages: list, model_code: str) -> list:
    """Patch system prompt to enforce hardcoded constants and 4-param kernel."""
    init_params = parse_init_params(model_code)
    init_values_match = re.search(
        r'def get_init_inputs\(\):\s*\n\s*return\s*\[(.*?)\]', model_code, re.DOTALL,
    )
    const_hint = ""
    if init_params and init_values_match:
        vals = [v.strip() for v in init_values_match.group(1).split(",") if v.strip()]
        pairs = []
        for name, val in zip(init_params, vals):
            pairs.append(f"{name}={val}")
        if pairs:
            const_hint = (
                "\n- Concrete constant values to hardcode: "
                + ", ".join(pairs)
                + "."
            )

    new_rule = (
        "- Always use exactly 4 parameters: "
        "`(float* output, const float* input, int size, hipStream_t stream)`.\n"
        "- NEVER add extra parameters to `launch_fused_kernel`.\n"
        "- HARDCODE all runtime constants from `__init__`/`get_init_inputs()` "
        "as C literal values directly in the kernel (e.g. `0.1f`, `2.0f`)."
        + const_hint
    )

    patched = []
    for msg in messages:
        if msg["role"] == "system":
            content = msg["content"]
            content = content.replace(
                "- Always use exactly 4 parameters: "
                "`(float* output, const float* input, int size, hipStream_t stream)`.",
                new_rule,
            )
            patched.append({**msg, "content": content})
        else:
            patched.append(msg)
    return patched


# ---------- vLLM generation ----------

def call_vllm(url: str, messages: list, temperature: float, max_tokens: int,
              stop: list | None = None) -> str:
    """Call vLLM OpenAI-compatible /chat/completions endpoint."""
    payload = {
        "model": "default",
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stop": stop or ["<END_OF_OUTPUT>"],
    }
    resp = requests.post(f"{url}/chat/completions", json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


# ---------- Code extraction ----------

def extract_hip_code(text: str) -> str | None:
    """Extract the fused_kernel.hip code block from model output."""
    m = re.search(
        r"\*\*kernels/fused_kernel\.hip\*\*\s*```(?:cpp|c\+\+|hip)?\s*\n(.*?)```",
        text, re.DOTALL,
    )
    if m:
        return m.group(1).strip()
    blocks = re.findall(r"```(?:cpp|c\+\+|hip)?\s*\n(.*?)```", text, re.DOTALL)
    for b in blocks:
        if "__global__" in b or "hip_runtime" in b:
            return b.strip()
    return None


# ---------- Dynamic binding generation ----------

# C++ type → pybind11 type mapping (for common kernel param types)
CPP_TYPE_MAP = {
    "float": "float",
    "double": "double",
    "int": "int64_t",
    "int32_t": "int64_t",
    "int64_t": "int64_t",
    "bool": "bool",
    "unsigned int": "int64_t",
    "size_t": "int64_t",
}


def parse_kernel_extra_params(hip_code: str) -> list[tuple[str, str]]:
    """Parse launch_fused_kernel signature to extract extra params.

    Returns list of (cpp_type, name) for params between 'int size' and
    'hipStream_t stream'.  Returns [] for fixed 4-param signature.
    """
    m = re.search(
        r'void\s+launch_fused_kernel\s*\((.*?)\)',
        hip_code, re.DOTALL,
    )
    if not m:
        return []

    params_str = m.group(1)
    # Normalize whitespace
    params_str = re.sub(r'\s+', ' ', params_str)
    params = [p.strip() for p in params_str.split(',')]

    # Find indices of the fixed boundary params
    size_idx = None
    stream_idx = None
    for i, p in enumerate(params):
        if re.search(r'\bint\s+\w*size\w*\b', p, re.IGNORECASE):
            size_idx = i
        if re.search(r'hipStream_t', p):
            stream_idx = i

    if size_idx is None or stream_idx is None or stream_idx <= size_idx + 1:
        return []

    extra_params = []
    for p in params[size_idx + 1:stream_idx]:
        p = p.strip()
        # Match "type name" patterns like "float slope", "int dim"
        m2 = re.match(r'((?:unsigned\s+)?(?:float|double|int|int32_t|int64_t|bool|size_t))\s+(\w+)', p)
        if m2:
            extra_params.append((m2.group(1), m2.group(2)))
        else:
            # Unknown type, skip
            return []

    return extra_params


def generate_dynamic_binding_cpp(extra_params: list[tuple[str, str]]) -> str:
    """Generate fused_kernel_binding.cpp matching the kernel's actual signature."""

    # --- extern "C" declaration ---
    extern_parts = ["float* output", "const float* input", "int size"]
    for ctype, name in extra_params:
        extern_parts.append(f"{ctype} {name}")
    extern_parts.append("hipStream_t stream")
    extern_decl = ", ".join(extern_parts)

    # --- Python-facing function signature ---
    py_parts = ["torch::Tensor input"]
    for ctype, name in extra_params:
        # Use the pybind-compatible type
        py_type = CPP_TYPE_MAP.get(ctype, ctype)
        py_parts.append(f"{py_type} {name}")
    py_sig = ", ".join(py_parts)

    # --- launch call arguments ---
    call_parts = ["output.data_ptr<float>()", "input.data_ptr<float>()", "input.numel()"]
    for ctype, name in extra_params:
        # Cast back to the kernel's expected type if needed
        py_type = CPP_TYPE_MAP.get(ctype, ctype)
        if py_type != ctype:
            call_parts.append(f"static_cast<{ctype}>({name})")
        else:
            call_parts.append(name)
    call_parts.append("stream")
    call_args = ", ".join(call_parts)

    return f"""\
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <hip/hip_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "binding_registry.h"

extern "C" void launch_fused_kernel({extern_decl});

torch::Tensor fused_kernel_forward({py_sig}) {{
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");

    auto output = torch::empty_like(input);
    hipStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    launch_fused_kernel({call_args});
    return output;
}}

void register_fused_kernel(pybind11::module& m) {{
    m.def("fused_kernel_forward", &fused_kernel_forward, "Fused kernel forward");
}}
REGISTER_BINDING(fused_kernel, register_fused_kernel);
"""


def parse_init_params(model_code: str) -> list[str]:
    """Extract __init__ parameter names (excluding self) from model code."""
    m = re.search(r'def __init__\(self(?:,\s*(.*?))?\)\s*:', model_code)
    if not m or not m.group(1):
        return []
    raw = m.group(1)
    params = []
    for p in raw.split(','):
        p = p.strip()
        if not p:
            continue
        name = p.split('=')[0].strip()
        # Remove type annotations
        name = name.split(':')[0].strip()
        params.append(name)
    return params


def generate_model_new(model_code: str, num_kernel_extra_params: int) -> str:
    """Generate ModelNew that stores __init__ args and passes them to kernel.

    Stores all __init__ params as self._kp_<name> so they can be passed
    to the HIP kernel via hip_extension.fused_kernel_forward().
    """
    init_params = parse_init_params(model_code)

    # Build __init__ signature
    if init_params:
        init_sig = "self, " + ", ".join(init_params)
    else:
        init_sig = "self"

    # Store all init params for kernel use
    store_lines = []
    for name in init_params:
        store_lines.append(f"        self._kp_{name} = {name}")
    if not store_lines:
        store_lines.append("        pass")
    store_block = "\n".join(store_lines)

    # Build forward call — pass stored init params to kernel
    # The kernel expects them in order, matching __init__ param order
    if init_params and num_kernel_extra_params > 0:
        if len(init_params) != num_kernel_extra_params:
            # Count mismatch - cannot reliably map params
            return None
        param_args = ", ".join(f"self._kp_{p}" for p in init_params)
        fwd_call = f"hip_extension.fused_kernel_forward(args[0].contiguous(), {param_args})"
    else:
        fwd_call = "hip_extension.fused_kernel_forward(args[0].contiguous())"

    return f"""import torch
import torch.nn as nn

class ModelNew(nn.Module):
    def __init__({init_sig}):
        super().__init__()
{store_block}

    def forward(self, *args, **kwargs):
        import hip_extension
        return {fwd_call}
"""


# ---------- Sandbox setup ----------

def setup_sandbox(hip_code: str, model_code: str, arch: str) -> Path:
    """Create a temporary sandbox with all files needed for compile + verify.

    Dynamically generates binding and model_new based on the kernel's
    actual launch_fused_kernel signature.
    """
    sandbox = Path(tempfile.mkdtemp(prefix="sft_sandbox_"))
    workdir = sandbox / "agent_workdir"
    workdir.mkdir()
    arch_dir = workdir / arch
    arch_dir.mkdir()
    kernels_dir = arch_dir / "kernels"
    kernels_dir.mkdir()

    # 1. Binding infrastructure
    shutil.copy2(ROOT / "agent_workdir" / "binding.cpp", workdir / "binding.cpp")
    shutil.copy2(ROOT / "agent_workdir" / "binding_registry.h", workdir / "binding_registry.h")

    # 2. Parse kernel signature → dynamic binding
    extra_params = parse_kernel_extra_params(hip_code)
    binding_cpp = generate_dynamic_binding_cpp(extra_params)

    # 3. Write kernel + binding
    (kernels_dir / "fused_kernel.hip").write_text(hip_code)
    (kernels_dir / "fused_kernel_binding.cpp").write_text(binding_cpp)

    # 4. model.py (original) for verify baseline
    (workdir / "model.py").write_text(model_code)

    # 5. model_new.py — dynamic, passes init params to kernel
    model_new_code = generate_model_new(model_code, len(extra_params))
    if model_new_code is None:
        shutil.rmtree(sandbox, ignore_errors=True)
        raise ValueError("param count mismatch in generate_model_new")
    (arch_dir / "model_new.py").write_text(model_new_code)

    # 6. Copy tools/ so sandbox can run compile + verify
    tools_src = ROOT / "tools"
    shutil.copytree(
        tools_src, sandbox / "tools",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )

    return sandbox


# ---------- Compile + Verify ----------

def try_compile(sandbox: Path, arch: str) -> tuple[bool, str]:
    """Compile HIP kernel in sandbox. Returns (success, output)."""
    env = {**os.environ, "PYTORCH_ROCM_ARCH": arch}
    result = subprocess.run(
        [sys.executable, "-m", "tools.compile", "--arch", arch],
        cwd=str(sandbox),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return result.returncode == 0, result.stdout + result.stderr


def try_verify(sandbox: Path, arch: str, gpu_id: int) -> tuple[bool, str]:
    """Run correctness verification on GPU. Returns (success, output)."""
    env = {
        **os.environ,
        "PYTORCH_ROCM_ARCH": arch,
        "HIP_VISIBLE_DEVICES": str(gpu_id),
        "ROCR_VISIBLE_DEVICES": str(gpu_id),
    }
    result = subprocess.run(
        [sys.executable, "-m", "tools.verify", "--arch", arch],
        cwd=str(sandbox),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    output = result.stdout + result.stderr
    passed = result.returncode == 0 or "[PARTIAL_PASS]" in output
    return passed, output


def try_compile_and_verify(
    hip_code: str, model_code: str, arch: str, gpu_id: int, compile_only: bool = False
) -> tuple[str, str]:
    """Full pipeline: setup → compile → verify. Returns (status, detail).

    Status: "verified" | "compile_pass" | "compile_fail" | "verify_fail" | "param_mismatch" | "error"
    """
    # Validate param counts before building sandbox
    extra_params = parse_kernel_extra_params(hip_code)
    init_params = parse_init_params(model_code)
    if extra_params and len(extra_params) != len(init_params):
        return "param_mismatch", f"init={len(init_params)} kernel={len(extra_params)}"

    sandbox = setup_sandbox(hip_code, model_code, arch)
    try:
        ok, msg = try_compile(sandbox, arch)
        if not ok:
            return "compile_fail", msg[-500:]

        if compile_only:
            return "compile_pass", ""

        ok, msg = try_verify(sandbox, arch, gpu_id)
        if ok:
            return "verified", msg[-300:]
        else:
            return "verify_fail", msg[-500:]

    except subprocess.TimeoutExpired:
        return "error", "timeout"
    except Exception as e:
        return "error", str(e)[:300]
    finally:
        shutil.rmtree(sandbox, ignore_errors=True)


# ---------- Main pipeline ----------

def process_sample(
    idx: int,
    prompt_messages: list,
    model_code: str,
    url: str,
    temperature: float,
    max_tokens: int,
    candidates: int,
    arch: str,
    gpu_id: int,
    compile_only: bool,
) -> dict | None:
    """Generate kernel candidates, return first one passing compile+verify."""
    # Patch prompt to enforce hardcoded constants + 4-param kernel
    patched_messages = patch_system_prompt(prompt_messages, model_code)
    for attempt in range(candidates):
        try:
            raw = call_vllm(url, patched_messages, temperature, max_tokens)
        except Exception as e:
            print(f"  [{idx}] vLLM error (attempt {attempt+1}): {e}")
            continue

        hip_code = extract_hip_code(raw)
        if not hip_code:
            print(f"  [{idx}] No HIP code extracted (attempt {attempt+1})")
            continue

        if "launch_fused_kernel" not in hip_code:
            print(f"  [{idx}] Missing launch_fused_kernel (attempt {attempt+1})")
            continue
        if "__global__" not in hip_code:
            print(f"  [{idx}] Missing __global__ (attempt {attempt+1})")
            continue

        status, detail = try_compile_and_verify(
            hip_code, model_code, arch, gpu_id, compile_only
        )

        if status in ("verified", "compile_pass"):
            completion = (
                f"**kernels/fused_kernel.hip**\n"
                f"```cpp\n{hip_code}\n```\n"
                f"<END_OF_OUTPUT>"
            )
            return {
                "completion": completion,
                "hip_code": hip_code,
                "attempts": attempt + 1,
                "status": status,
            }
        else:
            tag = {"compile_fail": "Compile", "verify_fail": "Verify",
                    "param_mismatch": "ParamMismatch"}.get(status, status.title())
            print(f"  [{idx}] {tag} failed (attempt {attempt+1}): {detail[:120]}")

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Generate SFT data: frontier model + compile + verify"
    )
    parser.add_argument("--input", required=True,
                        help="Input parquet (e.g. data/rocm_agent_ops_v4/train.parquet)")
    parser.add_argument("--output", required=True,
                        help="Output directory for SFT data")
    parser.add_argument("--arch", default="gfx1100",
                        help="Target GPU architecture")
    parser.add_argument("--vllm-url", default="http://localhost:8001/v1",
                        help="vLLM server base URL")
    parser.add_argument("--candidates", type=int, default=3,
                        help="Max generation attempts per sample")
    parser.add_argument("--temperature", type=float, default=0.3,
                        help="Generation temperature")
    parser.add_argument("--max-tokens", type=int, default=600,
                        help="Max output tokens")
    parser.add_argument("--val-ratio", type=float, default=0.1,
                        help="Validation split ratio")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit number of samples (0=all)")
    parser.add_argument("--skip-filter", action="store_true",
                        help="Skip binding compatibility filter")
    parser.add_argument("--compile-only", action="store_true",
                        help="Only check compilation, skip GPU verification")
    parser.add_argument("--verify-gpu", type=int, default=0,
                        help="GPU ID for running verification (default: 0)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    mode = "compile-only" if args.compile_only else "compile+verify"
    print(f"Mode: {mode} (dynamic binding)")
    if not args.compile_only:
        print(f"Verify GPU: {args.verify_gpu}")

    # 1. Load and filter
    print(f"\nLoading {args.input}...")
    df = pq.read_table(args.input).to_pandas()
    print(f"  Total samples: {len(df)}")

    if not args.skip_filter:
        # Phase 1: fast regex filter
        compat_mask = []
        for _, row in df.iterrows():
            kwargs = json.loads(row["interaction_kwargs"])
            mc = kwargs["model_code"]
            compat_mask.append(
                is_binding_compatible(mc) and is_elementwise_compatible(mc)
            )
        df = df[compat_mask].reset_index(drop=True)
        print(f"  After binding + elementwise filter: {len(df)}")

        # Apply --limit BEFORE expensive shape check (limit * 3 to have margin)
        if args.limit > 0:
            margin = min(len(df), args.limit * 5)
            df = df.head(margin)
            print(f"  Pre-limit for shape check: {margin}")

        # Phase 2: shape-check filter (batch, single subprocess)
        print(f"  Running batch shape-check on {len(df)} samples...")
        model_codes = [json.loads(row["interaction_kwargs"])["model_code"]
                       for _, row in df.iterrows()]
        shape_mask = batch_check_output_shapes(model_codes)
        df = df[shape_mask].reset_index(drop=True)
        print(f"  After shape-check filter: {len(df)}")

    if args.limit > 0:
        df = df.head(args.limit)
        print(f"  After --limit: {len(df)}")

    # 2. Check vLLM server
    print(f"\nChecking vLLM server at {args.vllm_url}...")
    try:
        health_url = args.vllm_url.replace("/v1", "/health")
        r = requests.get(health_url, timeout=5)
        if r.status_code == 200:
            print("  vLLM server is ready.")
        else:
            print(f"  Warning: health check returned {r.status_code}")
    except Exception as e:
        print(f"  ERROR: Cannot reach vLLM server: {e}")
        print("  Please start the vLLM server first.")
        sys.exit(1)

    # 3. Generate
    print(f"\nGenerating completions ({args.candidates} candidates/sample, "
          f"temp={args.temperature}, mode={mode})...")
    results = []
    stats = {"verified": 0, "compile_pass": 0, "compile_fail": 0,
             "verify_fail": 0, "param_mismatch": 0, "no_code": 0}
    t0 = time.time()

    for i, (_, row) in enumerate(df.iterrows()):
        prompt_messages = json.loads(row["prompt"])
        kwargs = json.loads(row["interaction_kwargs"])
        model_code = kwargs["model_code"]

        result = process_sample(
            idx=i,
            prompt_messages=prompt_messages,
            model_code=model_code,
            url=args.vllm_url,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            candidates=args.candidates,
            arch=args.arch,
            gpu_id=args.verify_gpu,
            compile_only=args.compile_only,
        )

        if result:
            stats[result["status"]] += 1
            results.append({
                "prompt": row["prompt"],
                "completion": result["completion"],
                "interaction_kwargs": row["interaction_kwargs"],
                "difficulty": row["difficulty"],
                "data_source": row["data_source"],
                "verify_status": result["status"],
            })
            status_str = f"{result['status'].upper()} (attempt {result['attempts']})"
        else:
            stats["no_code"] += 1
            status_str = "FAIL (all attempts)"

        total_pass = stats["verified"] + stats["compile_pass"]
        total_done = i + 1
        elapsed = time.time() - t0
        eta = elapsed / total_done * (len(df) - total_done)
        print(f"[{total_done}/{len(df)}] {status_str}  "
              f"pass={total_pass} fail={total_done - total_pass}  "
              f"rate={total_pass/total_done*100:.0f}%  "
              f"ETA={eta/60:.0f}m")

    elapsed_total = time.time() - t0

    # 4. Split and save
    import random
    random.seed(args.seed)
    random.shuffle(results)

    val_size = int(len(results) * args.val_ratio)
    val_data = results[:val_size]
    train_data = results[val_size:]

    columns = ["prompt", "completion", "interaction_kwargs", "difficulty",
               "data_source", "verify_status"]

    def to_table(rows):
        if not rows:
            return pa.table({k: [] for k in columns})
        return pa.table({k: [r[k] for r in rows] for k in columns})

    pq.write_table(to_table(train_data), out_dir / "train.parquet")
    pq.write_table(to_table(val_data), out_dir / "val.parquet")

    total_pass = stats["verified"] + stats["compile_pass"]
    print(f"\n{'='*60}")
    print(f"DONE in {elapsed_total/60:.1f} minutes")
    print(f"  Input samples:    {len(df)}")
    print(f"  Verified pass:    {stats['verified']}")
    print(f"  Compile-only pass:{stats['compile_pass']}")
    print(f"  Verify fail:      {stats['verify_fail']}")
    print(f"  Compile fail:     {stats['compile_fail']}")
    print(f"  Param mismatch:   {stats['param_mismatch']}")
    print(f"  No code:          {stats['no_code']}")
    print(f"  Total pass:       {total_pass} ({total_pass/max(len(df),1)*100:.1f}%)")
    print(f"  Train samples:    {len(train_data)}")
    print(f"  Val samples:      {len(val_data)}")
    print(f"  Output dir:       {out_dir}")


if __name__ == "__main__":
    main()
