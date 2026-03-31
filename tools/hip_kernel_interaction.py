"""HIP kernel interaction agent: compile, verify, profile, and score.

Reward scheme (graduated):
  -1.0   no code files found
  -0.9   partial files only
  -0.75  model_new + one of hip/binding
  -0.5   all 3 files, syntax error
  -0.25  all 3 files, linker error
   0.0   compiled OK, verification failed
  +1.0   verification passed
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

        self._write_agent_output(sandbox, assistant_code)

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
            should_stop = inst["iteration"] >= self.max_iterations
            return should_stop, f"Compiled OK, verification failed (reward 0.0):\n{verify_msg}", 0.0, {"stage": "verify_error"}

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

    def _write_agent_output(self, sandbox: Path, code: str):
        workdir = sandbox / "agent_workdir"
        arch_dir = workdir / self.arch
        kernels_dir = arch_dir / "kernels"
        for f in kernels_dir.glob("*"):
            f.unlink()
        
        parsed = self._parse_code_blocks(code)

        if "fused_kernel_binding.cpp" not in parsed:
            parsed["fused_kernel_binding.cpp"] = """#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <hip/hip_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "binding_registry.h"

extern "C" void launch_fused_kernel(float* output, const float* input, int size, hipStream_t stream);

torch::Tensor fused_kernel_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");

    auto output = torch::empty_like(input);
    
    // Support outputs that need contiguous float32 memory matching the input elements
    // In some cases input is modified, so we just pass data ptrs
    hipStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    launch_fused_kernel(output.data_ptr<float>(), input.data_ptr<float>(), input.numel(), stream);
    return output;
}

void register_fused_kernel(pybind11::module& m) {
    m.def("fused_kernel_forward", &fused_kernel_forward, "Fused kernel forward");
}
REGISTER_BINDING(fused_kernel, register_fused_kernel);
"""

        if "model_new.py" not in parsed:
            model_file = workdir / "model.py"
            if model_file.exists():
                model_code = model_file.read_text()
                # Instead of just replacing the body with return hip_extension.fused_kernel_forward(x)
                # which leaves other submodules intact and causes verify.py to trip on them,
                # we replace the entire body to clear out old torch logic while keeping signature
                import re
                
                # First rename the class
                new_code = model_code.replace("class Model(", "class ModelNew(")
                
                # Then regex to replace the forward method. We use a more aggressive regex 
                # that replaces everything from def forward to the end of the class/file.
                new_code = re.sub(
                    r'(\s+)def forward\(.*',
                    r'\1def forward(self, *args, **kwargs):\n\1    import hip_extension\n\1    return hip_extension.fused_kernel_forward(args[0].contiguous())\n',
                    new_code,
                    flags=re.DOTALL
                )
                parsed["model_new.py"] = new_code

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
            # Ninja is required by torch C++ extensions, add it to PATH if not already
            import os
            import sys
            env = {**os.environ, "PYTORCH_ROCM_ARCH": self.arch}
            
            # Make sure ninja from conda env is in PATH
            conda_bin = os.path.dirname(sys.executable)
            if conda_bin not in env.get("PATH", ""):
                env["PATH"] = f"{conda_bin}:{env.get('PATH', '')}"
                
            if use_eval_gpu and self.eval_gpu is not None:
                env["HIP_VISIBLE_DEVICES"] = str(self.eval_gpu)
                env["ROCR_VISIBLE_DEVICES"] = str(self.eval_gpu)
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
