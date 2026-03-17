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
        has_model_new = "model_new.py" in parsed
        has_hip = any(k.endswith(".hip") for k in parsed)
        has_binding = any(k.endswith("_binding.cpp") for k in parsed)

        if not has_model_new and not has_hip:
            should_stop = inst["iteration"] >= self.max_iterations
            return should_stop, "No code files found.", -1.0, {"stage": "no_code"}

        compile_ok, compile_msg = await self._run_compile(sandbox)
        if not compile_ok:
            if has_model_new and has_hip and has_binding:
                reward = -0.25 if ("undefined" in compile_msg.lower() or "linker" in compile_msg.lower()) else -0.5
            elif has_model_new and (has_hip or has_binding):
                reward = -0.75
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
        for filename, content in self._parse_code_blocks(code).items():
            if filename == "model_new.py":
                (arch_dir / filename).write_text(content)
            elif filename.endswith(".hip") or filename.endswith(".cpp"):
                (kernels_dir / filename).write_text(content)

    def _parse_code_blocks(self, text: str) -> Dict[str, str]:
        """Extract code blocks — try named headers first, then infer by content."""
        blocks = {}
        named = re.compile(
            r'(?:\*\*|#+\s*)(?:kernels/)?(\S+\.(?:hip|cpp|py))(?:\*\*)?\s*\n```\w*\n(.*?)\n```',
            re.DOTALL,
        )
        for m in named.finditer(text):
            blocks[m.group(1).strip()] = m.group(2)
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
        return await self._run_cmd(
            ["bash", "-c", f"cd {sandbox} && PYTORCH_ROCM_ARCH={self.arch} python3 -m tools.compile --arch {self.arch}"],
            self.compile_timeout,
        )

    async def _run_verify(self, sandbox: Path) -> Tuple[bool, str]:
        return await self._run_cmd(
            ["bash", "-c", f"cd {sandbox} && python3 -m tools.verify --arch {self.arch}"],
            self.verify_timeout,
        )

    async def _run_profile(self, sandbox: Path) -> Tuple[bool, dict]:
        ok, output = await self._run_cmd(
            ["bash", "-c", f"cd {sandbox} && python3 -m tools.bench --arch {self.arch} --iters 10"],
            self.profile_timeout,
        )
        if not ok:
            return False, {}
        result = self._parse_profile_output(output)
        return (True, result) if result else (False, {})

    async def _run_cmd(self, cmd: list, timeout: int) -> Tuple[bool, str]:
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env={**os.environ, "PYTORCH_ROCM_ARCH": self.arch},
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
