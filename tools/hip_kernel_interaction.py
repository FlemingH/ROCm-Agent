"""HIP kernel interaction agent: compile, verify, profile, and score.

Reward scheme (graduated):
  -1.0   no code files found
  -0.9   partial files only
  -0.5   compiled failed (syntax/linker error)
   0.0   compiled OK, verification failed
  +0.3   compiled OK, output shape correct but values wrong
  +0.5   compiled OK, some checks fully pass (partial verify)
  +1.0   verification passed (all checks)
  +2.0   faster than Eager by >5%
  +3.0   faster than both Eager and torch.compile by >5%
"""

import asyncio
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

# ---------- Dynamic binding helpers ----------

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

    params_str = re.sub(r'\s+', ' ', m.group(1))
    params = [p.strip() for p in params_str.split(',')]

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
        m2 = re.match(r'((?:unsigned\s+)?(?:float|double|int|int32_t|int64_t|bool|size_t))\s+(\w+)', p)
        if m2:
            extra_params.append((m2.group(1), m2.group(2)))
        else:
            return []

    return extra_params


def generate_dynamic_binding_cpp(extra_params: list[tuple[str, str]]) -> str:
    """Generate fused_kernel_binding.cpp matching the kernel's actual signature."""
    extern_parts = ["float* output", "const float* input", "int size"]
    for ctype, name in extra_params:
        extern_parts.append(f"{ctype} {name}")
    extern_parts.append("hipStream_t stream")
    extern_decl = ", ".join(extern_parts)

    py_parts = ["torch::Tensor input"]
    for ctype, name in extra_params:
        py_type = CPP_TYPE_MAP.get(ctype, ctype)
        py_parts.append(f"{py_type} {name}")
    py_sig = ", ".join(py_parts)

    call_parts = ["output.data_ptr<float>()", "input.data_ptr<float>()", "input.numel()"]
    for ctype, name in extra_params:
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
        name = p.split('=')[0].strip().split(':')[0].strip()
        params.append(name)
    return params


def generate_dynamic_model_new(model_code: str, num_kernel_extra_params: int) -> str | None:
    """Generate ModelNew that stores __init__ args and passes them to kernel."""
    init_params = parse_init_params(model_code)

    if init_params:
        init_sig = "self, " + ", ".join(init_params)
    else:
        init_sig = "self"

    store_lines = []
    for name in init_params:
        store_lines.append(f"        self._kp_{name} = {name}")
    if not store_lines:
        store_lines.append("        pass")
    store_block = "\n".join(store_lines)

    if init_params and num_kernel_extra_params > 0:
        if len(init_params) != num_kernel_extra_params:
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


class HipKernelInteraction:
    """Single- or multi-turn agent that compiles/verifies/profiles HIP kernels."""

    def __init__(self, config: dict):
        self.arch = config.get("arch", "gfx1201")
        self.workdir_template = config.get("workdir", "agent_workdir")
        self.compile_timeout = config.get("compile_timeout", 60)
        self.verify_timeout = config.get("verify_timeout", 30)
        self.profile_timeout = config.get("profile_timeout", 60)
        self.max_iterations = config.get("max_iterations", 20)
        self.eval_gpu = config.get("eval_gpu", None)
        self._instances: Dict[str, dict] = {}

    async def start_interaction(self, instance_id: Optional[str] = None,
                                 model_code: str = "") -> str:
        if instance_id is None:
            instance_id = str(uuid4())
        sandbox = Path(tempfile.mkdtemp(prefix=f"hip_sandbox_{instance_id[:8]}_"))
        self._setup_sandbox(sandbox, model_code)
        self._instances[instance_id] = {
            "sandbox": sandbox,
            "iteration": 0,
            "model_code": model_code,
        }
        return instance_id

    async def generate_response(self, instance_id: str,
                                 messages: List[Dict[str, Any]],
                                 ) -> Tuple[bool, str, float, Dict[str, Any]]:
        inst = self._instances[instance_id]
        inst["iteration"] += 1
        sandbox = inst["sandbox"]

        assistant_code = self._extract_last_assistant(messages)
        if not assistant_code:
            return False, "No code found.", -1.0, {}

        self._write_agent_output(sandbox, assistant_code, inst.get("model_code", ""))

        parsed = self._parse_code_blocks(assistant_code)
        has_hip = any(k.endswith(".hip") for k in parsed)

        if not has_hip:
            should_stop = inst["iteration"] >= self.max_iterations
            return should_stop, "No code files found.", -1.0, {"stage": "no_code"}

        compile_ok, compile_msg = await self._run_compile(sandbox)
        if not compile_ok:
            if has_hip:
                reward = -0.5
            else:
                reward = -0.9
            should_stop = inst["iteration"] >= self.max_iterations
            return should_stop, f"Compilation failed (reward {reward:.2f}):\n{compile_msg}", reward, {"stage": "compile_error"}

        verify_ok, verify_msg = await self._run_verify(sandbox)
        if not verify_ok:
            reward = self._parse_verify_reward(verify_msg)
            stage = "verify_partial" if reward > 0 else "verify_error"
            should_stop = inst["iteration"] >= self.max_iterations
            label = "partial" if reward > 0 else "failed"
            return should_stop, f"Compiled OK, verification {label} (reward {reward:.1f}):\n{verify_msg}", reward, {"stage": stage}

        profile_ok, profile_result = await self._run_profile(sandbox)
        if not profile_ok:
            should_stop = inst["iteration"] >= self.max_iterations
            return should_stop, "Verification passed, profiling failed. Reward: +1.", 1.0, {"stage": "profile_error"}

        reward = self._compute_reward(profile_result)
        feedback = self._format_profile_feedback(profile_result, reward)
        should_stop = reward >= 3.0 or inst["iteration"] >= self.max_iterations
        return should_stop, feedback, reward, {"stage": "profiled", **profile_result}

    async def finalize_interaction(self, instance_id: str) -> None:
        inst = self._instances.pop(instance_id, None)
        if inst and inst.get("sandbox"):
            shutil.rmtree(inst["sandbox"], ignore_errors=True)

    # --- Sandbox setup ---

    def _setup_sandbox(self, sandbox: Path, model_code: str):
        template = Path(self.workdir_template)
        workdir = sandbox / "agent_workdir"
        workdir.mkdir(exist_ok=True)

        for f in ["binding.cpp", "binding_registry.h"]:
            src = template / f
            if src.exists():
                shutil.copy2(src, workdir / f)

        tools_src = Path(__file__).resolve().parent
        if tools_src.is_dir():
            shutil.copytree(tools_src, sandbox / "tools",
                            ignore=shutil.ignore_patterns("__pycache__"))

        (workdir / "model.py").write_text(model_code)
        arch_dir = workdir / self.arch
        arch_dir.mkdir(exist_ok=True)
        (arch_dir / "kernels").mkdir(exist_ok=True)

    # --- Output parsing ---

    def _extract_last_assistant(self, messages: List[Dict[str, Any]]) -> str:
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                return msg.get("content", "")
        return ""

    def _write_agent_output(self, sandbox: Path, code: str, model_code: str = ""):
        workdir = sandbox / "agent_workdir"
        arch_dir = workdir / self.arch
        kernels_dir = arch_dir / "kernels"
        for f in kernels_dir.glob("*"):
            f.unlink()

        parsed = self._parse_code_blocks(code)

        # Auto-generate binding from kernel signature if not provided
        hip_code = parsed.get("fused_kernel.hip", "")
        if "fused_kernel_binding.cpp" not in parsed and hip_code:
            extra_params = parse_kernel_extra_params(hip_code)
            parsed["fused_kernel_binding.cpp"] = generate_dynamic_binding_cpp(extra_params)

        # Auto-generate model_new.py if not provided
        if "model_new.py" not in parsed and hip_code:
            if not model_code:
                model_file = workdir / "model.py"
                if model_file.exists():
                    model_code = model_file.read_text()
            if model_code:
                extra_params = parse_kernel_extra_params(hip_code)
                model_new = generate_dynamic_model_new(model_code, len(extra_params))
                if model_new:
                    parsed["model_new.py"] = model_new

        for filename, content in parsed.items():
            if filename == "fused_kernel.hip" and "hip_runtime.h" not in content:
                content = "#include <hip/hip_runtime.h>\n" + content
            if filename == "model_new.py":
                (arch_dir / filename).write_text(content)
            elif filename.endswith(".hip") or filename.endswith(".cpp"):
                (kernels_dir / filename).write_text(content)

    @staticmethod
    def _normalize_named_file_headers(text: str) -> str:
        """Normalize small markdown drift in exact file header lines."""
        header_map = {
            "fused_kernel.hip": "**kernels/fused_kernel.hip**",
            "fused_kernel_binding.cpp": "**kernels/fused_kernel_binding.cpp**",
            "model_new.py": "**model_new.py**",
        }
        for filename, canonical in header_map.items():
            text = re.sub(
                rf"(?m)^[ \t*#`-]*(?:kernels/)?{re.escape(filename)}[ \t*#`:\-()]*$",
                canonical,
                text,
            )
        return text

    @staticmethod
    def _extract_named_file_blocks(text: str) -> Dict[str, str]:
        """Extract known file blocks in order, keeping first valid occurrence."""
        blocks: Dict[str, str] = {}
        header_to_filename = {
            "**kernels/fused_kernel.hip**": "fused_kernel.hip",
            "**kernels/fused_kernel_binding.cpp**": "fused_kernel_binding.cpp",
            "**model_new.py**": "model_new.py",
        }
        lines = text.splitlines()
        i = 0
        while i < len(lines):
            header = lines[i].strip()
            filename = header_to_filename.get(header)
            if not filename:
                i += 1
                continue

            j = i + 1
            while j < len(lines) and lines[j].strip() == "":
                j += 1
            if j >= len(lines) or not lines[j].lstrip().startswith("```"):
                i += 1
                continue

            code_lines = []
            k = j + 1
            while k < len(lines):
                if lines[k].lstrip().startswith("```"):
                    if filename not in blocks:
                        blocks[filename] = "\n".join(code_lines)
                    i = k
                    break
                code_lines.append(lines[k])
                k += 1
            else:
                if filename not in blocks:
                    blocks[filename] = "\n".join(code_lines)
                break

            i += 1
        return blocks

    def _parse_code_blocks(self, text: str) -> Dict[str, str]:
        """Extract code blocks — try named headers first, then infer by content."""
        text = self._normalize_named_file_headers(text)
        blocks = self._extract_named_file_blocks(text)
        if blocks:
            return blocks

        all_blocks = re.findall(r'```(\w*)\n(.*?)\n```', text, re.DOTALL)
        py = [(l, c) for l, c in all_blocks if l in ('python', '')]
        cpp = [(l, c) for l, c in all_blocks if l in ('cpp', 'c', 'c++', 'hip')]

        for _, c in py:
            if 'ModelNew' in c or 'hip_extension' in c:
                blocks["model_new.py"] = c
                break
        for _, c in cpp:
            if 'REGISTER_BINDING' in c or 'binding_registry' in c:
                blocks["fused_kernel_binding.cpp"] = c
            elif '__global__' in c or 'hip_runtime' in c:
                blocks["fused_kernel.hip"] = c
        if "fused_kernel_binding.cpp" not in blocks:
            for _, c in cpp:
                if 'torch/extension' in c or 'torch/types' in c:
                    blocks["fused_kernel_binding.cpp"] = c
                    break
        return blocks

    # --- Subprocess execution ---

    async def _run_compile(self, sandbox: Path) -> Tuple[bool, str]:
        import sys
        return await self._run_cmd(
            ["bash", "-c", f"cd {sandbox} && PYTORCH_ROCM_ARCH={self.arch} {sys.executable} -m tools.compile --arch {self.arch}"],
            self.compile_timeout,
        )

    async def _run_verify(self, sandbox: Path) -> Tuple[bool, str]:
        import sys
        return await self._run_cmd(
            ["bash", "-c", f"cd {sandbox} && {sys.executable} -m tools.verify --arch {self.arch}"],
            self.verify_timeout, use_eval_gpu=True,
        )

    async def _run_profile(self, sandbox: Path) -> Tuple[bool, dict]:
        import sys
        ok, output = await self._run_cmd(
            ["bash", "-c", f"cd {sandbox} && {sys.executable} -m tools.bench --arch {self.arch} --iters 10"],
            self.profile_timeout, use_eval_gpu=True,
        )
        if not ok:
            return False, {}
        result = self._parse_profile_output(output)
        return (True, result) if result else (False, {})

    async def _run_cmd(self, cmd: list, timeout: int, use_eval_gpu: bool = False) -> Tuple[bool, str]:
        try:
            import sys
            env = {**os.environ, "PYTORCH_ROCM_ARCH": self.arch}

            # Make sure ninja from conda env is in PATH
            conda_bin = os.path.dirname(sys.executable)
            if conda_bin not in env.get("PATH", ""):
                env["PATH"] = f"{conda_bin}:{env.get('PATH', '')}"

            if use_eval_gpu and self.eval_gpu is not None:
                # PyTorch on ROCm uses CUDA_VISIBLE_DEVICES for tensor allocations.
                # Setting HIP_VISIBLE_DEVICES alongside CUDA_VISIBLE_DEVICES causes
                # a bug in PyTorch where it says "No HIP GPUs are available".
                env["CUDA_VISIBLE_DEVICES"] = str(self.eval_gpu)
                # MUST remove these to prevent PyTorch ROCm backend confusion
                env.pop("HIP_VISIBLE_DEVICES", None)
                env.pop("ROCR_VISIBLE_DEVICES", None)
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env=env,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            return proc.returncode == 0, stdout.decode(errors="replace")
        except asyncio.TimeoutError:
            proc.kill()
            return False, f"Timed out after {timeout}s"
        except Exception as e:
            return False, str(e)

    # --- Reward computation ---

    _PROFILE_RE = re.compile(
        r'Torch Baseline:\s*([\d.]+)us.*Torch Compile:\s*([\d.]+)us.*HIP Extension:\s*([\d.]+)us'
    )

    def _parse_profile_output(self, output: str) -> dict | None:
        m = self._PROFILE_RE.search(output)
        if not m:
            return None
        return {
            "torch_baseline_us": float(m.group(1)),
            "torch_compile_us": float(m.group(2)),
            "hip_extension_us": float(m.group(3)),
        }

    @staticmethod
    def _parse_verify_reward(verify_msg: str) -> float:
        """Extract intermediate reward from verify output signals."""
        if "[PARTIAL_PASS]" in verify_msg:
            return 0.5
        if "[SHAPE_OK]" in verify_msg:
            return 0.3
        return 0.0

    @staticmethod
    def _compute_reward(profile: dict) -> float:
        baseline, compile_t, hip = profile["torch_baseline_us"], profile["torch_compile_us"], profile["hip_extension_us"]
        if hip < baseline * 0.95 and hip < compile_t * 0.95:
            return 3.0
        elif hip < baseline * 0.95:
            return 2.0
        return 1.0

    @staticmethod
    def _format_profile_feedback(profile: dict, reward: float) -> str:
        b, c, h = profile["torch_baseline_us"], profile["torch_compile_us"], profile["hip_extension_us"]
        return (
            f"Torch Baseline: {b:.3f}us, Torch Compile: {c:.3f}us, HIP Extension: {h:.3f}us | "
            f"Speedup: {b/h:.2f}x eager, {c/h:.2f}x compile | Reward: {reward:+.0f}"
        )
