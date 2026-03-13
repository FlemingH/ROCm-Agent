"""HIP kernel generation interaction agent for verl multi-turn Agentic RL.

Implements BaseInteraction: the Agent generates HIP kernel code, this interaction
compiles it, verifies correctness, profiles performance, and returns reward + feedback.

Reward scheme:
  -1  compile or verification failure
  +1  correct output (passes verification)
  +2  faster than Eager baseline by >5%
  +3  faster than both Eager and torch.compile by >5%
"""

import asyncio
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

try:
    from verl.interactions.base import BaseInteraction
except ImportError:
    class BaseInteraction:
        """Fallback for standalone testing without verl installed."""
        def __init__(self, config: dict):
            self.config = config
            self.name = config.get("name", "interaction_agent")


class HipKernelInteraction(BaseInteraction):
    """Multi-turn interaction agent that compiles/verifies/profiles HIP kernels."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.arch = config.get("arch", "gfx1100")
        self.workdir_template = config.get("workdir", "agent_workdir")
        self.compile_timeout = config.get("compile_timeout", 60)
        self.verify_timeout = config.get("verify_timeout", 30)
        self.profile_timeout = config.get("profile_timeout", 60)
        self.max_iterations = config.get("max_iterations", 20)
        self._instances: Dict[str, dict] = {}

    async def start_interaction(self, instance_id: Optional[str] = None,
                                 model_code: str = "", **kwargs) -> str:
        if instance_id is None:
            instance_id = str(uuid4())

        sandbox = Path(tempfile.mkdtemp(prefix=f"hip_sandbox_{instance_id[:8]}_"))
        self._setup_sandbox(sandbox, model_code)

        self._instances[instance_id] = {
            "sandbox": sandbox,
            "model_code": model_code,
            "iteration": 0,
            "last_reward": 0.0,
            "best_reward": 0.0,
            "compile_ok": False,
            "verify_ok": False,
            "profile_result": None,
        }
        return instance_id

    async def generate_response(self, instance_id: str,
                                 messages: List[Dict[str, Any]],
                                 **kwargs) -> Tuple[bool, str, float, Dict[str, Any]]:
        inst = self._instances[instance_id]
        inst["iteration"] += 1
        sandbox = inst["sandbox"]

        assistant_code = self._extract_last_assistant(messages)
        if not assistant_code:
            return False, "No code found in your response. Please provide HIP kernel code.", -1.0, {}

        self._write_agent_output(sandbox, assistant_code)

        compile_ok, compile_msg = await self._run_compile(sandbox)
        if not compile_ok:
            inst["compile_ok"] = False
            inst["last_reward"] = -1.0
            feedback = f"Compilation failed:\n{compile_msg}\n\nFix the errors and try again."
            should_stop = inst["iteration"] >= self.max_iterations
            return should_stop, feedback, -1.0, {"stage": "compile_error"}

        inst["compile_ok"] = True

        verify_ok, verify_msg = await self._run_verify(sandbox)
        if not verify_ok:
            inst["verify_ok"] = False
            inst["last_reward"] = -1.0
            feedback = f"Verification failed:\n{verify_msg}\n\nFix correctness issues and try again."
            should_stop = inst["iteration"] >= self.max_iterations
            return should_stop, feedback, -1.0, {"stage": "verify_error"}

        inst["verify_ok"] = True

        profile_ok, profile_result = await self._run_profile(sandbox)
        if not profile_ok:
            inst["last_reward"] = 1.0
            feedback = "Verification passed but profiling failed. Reward: +1 (correct)."
            should_stop = inst["iteration"] >= self.max_iterations
            return should_stop, feedback, 1.0, {"stage": "profile_error"}

        reward = self._compute_reward(profile_result)
        inst["last_reward"] = reward
        inst["best_reward"] = max(inst["best_reward"], reward)
        inst["profile_result"] = profile_result

        feedback = self._format_profile_feedback(profile_result, reward)
        should_stop = reward >= 3.0 or inst["iteration"] >= self.max_iterations
        return should_stop, feedback, reward, {"stage": "profiled", **profile_result}

    async def calculate_score(self, instance_id: str, **kwargs) -> float:
        inst = self._instances.get(instance_id, {})
        return inst.get("best_reward", 0.0)

    async def finalize_interaction(self, instance_id: str, **kwargs) -> None:
        inst = self._instances.pop(instance_id, None)
        if inst and inst.get("sandbox"):
            shutil.rmtree(inst["sandbox"], ignore_errors=True)

    def _setup_sandbox(self, sandbox: Path, model_code: str):
        template = Path(self.workdir_template)

        for f in ["binding.cpp", "binding_registry.h"]:
            src = template / f
            if src.exists():
                shutil.copy2(src, sandbox / f)

        tools_dst = sandbox / "tools"
        tools_src = Path(__file__).resolve().parent
        if tools_src.is_dir():
            shutil.copytree(tools_src, tools_dst, ignore=shutil.ignore_patterns("__pycache__"))

        (sandbox / "model.py").write_text(model_code)

        arch_dir = sandbox / self.arch
        arch_dir.mkdir(exist_ok=True)
        (arch_dir / "kernels").mkdir(exist_ok=True)

    def _extract_last_assistant(self, messages: List[Dict[str, Any]]) -> str:
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                return msg.get("content", "")
        return ""

    def _write_agent_output(self, sandbox: Path, code: str):
        """Parse agent output and write model_new.py + kernel files."""
        arch_dir = sandbox / self.arch
        kernels_dir = arch_dir / "kernels"

        for f in kernels_dir.glob("*"):
            if f.name.endswith("_hip.hip"):
                continue
            f.unlink()

        sections = self._parse_code_blocks(code)

        for filename, content in sections.items():
            if filename == "model_new.py":
                (arch_dir / filename).write_text(content)
            elif filename.endswith(".hip") or filename.endswith(".cpp"):
                (kernels_dir / filename).write_text(content)

        if "model_new.py" not in sections:
            model_new_code = self._extract_fenced_block(code, "model_new.py")
            if model_new_code:
                (arch_dir / "model_new.py").write_text(model_new_code)

    def _parse_code_blocks(self, text: str) -> Dict[str, str]:
        """Extract named code blocks from agent output.

        Looks for patterns like:
          **kernels/my_kernel.hip**
          ```cpp
          ...code...
          ```
        or:
          # model_new.py
          ```python
          ...code...
          ```
        """
        import re
        blocks = {}
        pattern = re.compile(
            r'(?:\*\*|#+\s*)'
            r'(?:kernels/)?'
            r'(\S+\.(?:hip|cpp|py))'
            r'(?:\*\*)?'
            r'\s*\n'
            r'```\w*\n'
            r'(.*?)'
            r'\n```',
            re.DOTALL
        )
        for m in pattern.finditer(text):
            filename = m.group(1).strip()
            content = m.group(2)
            blocks[filename] = content
        return blocks

    def _extract_fenced_block(self, text: str, hint: str) -> str | None:
        import re
        pattern = re.compile(r'```(?:python)?\n(.*?)\n```', re.DOTALL)
        for m in pattern.finditer(text):
            if hint.lower() in text[max(0, m.start()-100):m.start()].lower():
                return m.group(1)
        return None

    async def _run_compile(self, sandbox: Path) -> Tuple[bool, str]:
        cmd = ["bash", "-c", f"cd {sandbox} && PYTORCH_ROCM_ARCH={self.arch} python3 -m tools.compile --arch {self.arch}"]
        return await self._run_cmd(cmd, self.compile_timeout)

    async def _run_verify(self, sandbox: Path) -> Tuple[bool, str]:
        cmd = ["bash", "-c", f"cd {sandbox} && python3 -m tools.verify --arch {self.arch}"]
        return await self._run_cmd(cmd, self.verify_timeout)

    async def _run_profile(self, sandbox: Path) -> Tuple[bool, dict]:
        cmd = ["bash", "-c", f"cd {sandbox} && python3 -m tools.profile --arch {self.arch} --iters 10"]
        ok, output = await self._run_cmd(cmd, self.profile_timeout)
        if not ok:
            return False, {}

        result = self._parse_profile_output(output)
        if not result:
            return False, {}
        return True, result

    async def _run_cmd(self, cmd: list, timeout: int) -> Tuple[bool, str]:
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env={**os.environ, "PYTORCH_ROCM_ARCH": self.arch},
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            output = stdout.decode(errors="replace")
            return proc.returncode == 0, output
        except asyncio.TimeoutError:
            proc.kill()
            return False, f"Command timed out after {timeout}s"
        except Exception as e:
            return False, str(e)

    def _parse_profile_output(self, output: str) -> dict | None:
        """Parse 'Torch Baseline: X.XXXus, Torch Compile: X.XXXus, HIP Extension: X.XXXus'"""
        import re
        pattern = re.compile(
            r'Torch Baseline:\s*([\d.]+)us.*'
            r'Torch Compile:\s*([\d.]+)us.*'
            r'HIP Extension:\s*([\d.]+)us'
        )
        m = pattern.search(output)
        if not m:
            return None
        return {
            "torch_baseline_us": float(m.group(1)),
            "torch_compile_us": float(m.group(2)),
            "hip_extension_us": float(m.group(3)),
        }

    def _compute_reward(self, profile: dict) -> float:
        baseline = profile["torch_baseline_us"]
        compile_t = profile["torch_compile_us"]
        hip = profile["hip_extension_us"]

        if hip < baseline * 0.95 and hip < compile_t * 0.95:
            return 3.0
        elif hip < baseline * 0.95:
            return 2.0
        else:
            return 1.0

    def _format_profile_feedback(self, profile: dict, reward: float) -> str:
        baseline = profile["torch_baseline_us"]
        compile_t = profile["torch_compile_us"]
        hip = profile["hip_extension_us"]

        speedup_eager = baseline / hip if hip > 0 else 0
        speedup_compile = compile_t / hip if hip > 0 else 0

        lines = [
            "Verification passed. Performance results:",
            f"  Torch Baseline:  {baseline:.3f} us",
            f"  Torch Compile:   {compile_t:.3f} us",
            f"  HIP Extension:   {hip:.3f} us",
            f"  Speedup vs Eager:   {speedup_eager:.2f}x",
            f"  Speedup vs Compile: {speedup_compile:.2f}x",
            f"  Reward: {reward:+.0f}",
        ]

        if reward < 3.0:
            lines.append("")
            lines.append("Try to optimize further for better performance.")

        return "\n".join(lines)
