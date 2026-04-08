#!/usr/bin/env python3
"""Iterative HIP kernel generation agent using trained LoRA model.

Two-phase optimization:
  Phase 1 (correctness): Generate a compiling+verifying HIP kernel (reward >= 1.0)
  Phase 2 (performance): Use rocprofv3 hardware counters (L2 cache hit, occupancy,
           VGPR usage, etc.) as feedback to optimize until reward >= target

Usage:
    python tools/iterative_agent.py \
      --base-model models/Jan-code-4b \
      --adapter checkpoints/grpo-jan-code-4b-b26 \
      --model-code examples/swish_mish_fused.py \
      --max-iters 10 --arch gfx1100 --eval-gpu 1 --gen-gpu 0
"""

import argparse
import asyncio
import csv
import json
import os
import re
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from hip_kernel_interaction import HipKernelInteraction


# ---------------------------------------------------------------------------
# rocprofv3 hardware counter profiling
# ---------------------------------------------------------------------------

# Counter groups — each pmc line must fit in one hardware pass
# rocprofv3 on gfx1100 can do ~1-2 derived metrics per pass
ROCPROF_INPUT = """\
pmc: SQ_WAVES_sum SQ_INSTS_VALU_sum
pmc: GL2C_HIT_sum GL2C_MISS_sum GL2C_MC_RDREQ_sum
"""



TIMING_RE = re.compile(
    r'TIMING: Torch Baseline:\s*([\d.]+)us.*Torch Compile:\s*([\d.]+)us.*HIP Extension:\s*([\d.]+)us'
)


def write_rocprof_input(path: Path):
    """Write rocprofv3 input .txt file with PMC counter groups."""
    path.write_text(ROCPROF_INPUT)


async def run_rocprof(sandbox: Path, arch: str, eval_gpu: str,
                      bench_script_text: str, timeout: int = 120) -> dict:
    """Run rocprofv3 on bench script and parse results."""
    input_file = sandbox / "rocprof_input.txt"
    output_prefix = sandbox / "rocprof_out"
    script_file = sandbox / "rocprof_bench.py"

    write_rocprof_input(input_file)
    script_file.write_text(bench_script_text)

    env = {**os.environ, "PYTORCH_ROCM_ARCH": arch}
    conda_bin = os.path.dirname(sys.executable)
    if conda_bin not in env.get("PATH", ""):
        env["PATH"] = f"{conda_bin}:{env.get('PATH', '')}"
    env["CUDA_VISIBLE_DEVICES"] = str(eval_gpu)
    env.pop("HIP_VISIBLE_DEVICES", None)
    env.pop("ROCR_VISIBLE_DEVICES", None)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"

    cmd = [
        "rocprofv3",
        "-i", str(input_file),
        "--kernel-trace",
        "-o", str(output_prefix),
        "-f", "csv",
        "--", sys.executable, str(script_file),
    ]

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=env,
            cwd=str(sandbox),
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        output_text = stdout.decode(errors="replace")

        if proc.returncode != 0:
            return {"error": f"rocprofv3 rc={proc.returncode}: {output_text[-500:]}"}

        result = parse_rocprof_results(sandbox, output_prefix)
        # Also parse self-timing from stdout
        tm = TIMING_RE.search(output_text)
        if tm:
            result["self_torch_baseline_us"] = float(tm.group(1))
            result["self_torch_compile_us"] = float(tm.group(2))
            result["self_hip_extension_us"] = float(tm.group(3))
        return result
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except Exception:
            pass
        return {"error": f"rocprofv3 timed out after {timeout}s"}
    except Exception as e:
        return {"error": str(e)}


def parse_rocprof_results(sandbox: Path, output_prefix: Path) -> dict:
    """Parse rocprofv3 CSV outputs into a single dict of metrics."""
    result = {}

    # Parse counter_collection CSV
    cc_path = Path(str(output_prefix) + "_counter_collection.csv")
    if cc_path.exists():
        try:
            with open(cc_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    kname = row.get("Kernel_Name", "")
                    # Only collect counters for our fused_kernel
                    if "fused_kernel" not in kname:
                        continue
                    cname = row.get("Counter_Name", "")
                    cval = row.get("Counter_Value", "")
                    if cname and cval:
                        try:
                            result[cname] = float(cval)
                        except ValueError:
                            result[cname] = cval
                    # Also grab register counts from the row
                    for col in ["VGPR_Count", "Accum_VGPR_Count", "SGPR_Count",
                                "LDS_Block_Size", "Scratch_Size", "Workgroup_Size",
                                "Grid_Size"]:
                        v = row.get(col, "")
                        if v and col not in result:
                            try:
                                result[col] = int(v)
                            except ValueError:
                                result[col] = v
        except Exception as e:
            result["csv_error"] = str(e)

    # Parse kernel_trace CSV for timing
    kt_path = Path(str(output_prefix) + "_kernel_trace.csv")
    if kt_path.exists():
        try:
            with open(kt_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    kname = row.get("Kernel_Name", "")
                    if "fused_kernel" not in kname:
                        continue
                    start = row.get("Start_Timestamp", "")
                    end = row.get("End_Timestamp", "")
                    if start and end:
                        try:
                            duration_ns = int(end) - int(start)
                            result["kernel_duration_ns"] = duration_ns
                            result["kernel_duration_us"] = duration_ns / 1000.0
                        except ValueError:
                            pass
                    for col in ["VGPR_Count", "Accum_VGPR_Count", "SGPR_Count",
                                "LDS_Block_Size", "Scratch_Size", "Workgroup_Size",
                                "Grid_Size"]:
                        v = row.get(col, "")
                        if v and col not in result:
                            try:
                                result[col] = int(v)
                            except ValueError:
                                result[col] = v
        except Exception as e:
            result["trace_error"] = str(e)

    # Compute derived metrics
    gl2c_hit = result.get("GL2C_HIT_sum", 0)
    gl2c_miss = result.get("GL2C_MISS_sum", 0)
    if gl2c_hit + gl2c_miss > 0:
        result["L2_hit_rate"] = 100.0 * gl2c_hit / (gl2c_hit + gl2c_miss)

    return result


def format_rocprof_feedback(metrics: dict, bench_feedback: str = "") -> str:
    """Format rocprof metrics into a feedback string for the model."""
    if "error" in metrics:
        return f"rocprofv3 error: {metrics['error']}"

    lines = ["[AMD gfx1100 Hardware Profiling Results]"]

    # Register pressure
    vgpr = metrics.get("VGPR_Count")
    sgpr = metrics.get("SGPR_Count")
    if vgpr is not None:
        # gfx1100 RDNA3: 1536 VGPRs per SIMD, wave32, max 16 waves/SIMD
        # Each wave uses (ceil(vgpr/8)*8) VGPRs if using 128+ VGPRs, occupancy drops
        max_waves = min(16, 1536 // max(vgpr, 1))
        occ_pct = 100.0 * max_waves / 16
        quality = ("LOW - reduce VGPRs to increase occupancy" if occ_pct < 50
                   else "MEDIUM" if occ_pct < 75 else "GOOD")
        lines.append(f"  VGPR Count: {vgpr} (max waves/SIMD: {max_waves}, est. occupancy: {occ_pct:.0f}% - {quality})")
    if sgpr is not None:
        lines.append(f"  SGPR Count: {sgpr}")

    # L2 Cache
    l2_rate = metrics.get("L2_hit_rate")
    if l2_rate is not None:
        quality = ("POOR - improve memory coalescing" if l2_rate < 50
                   else "OK" if l2_rate < 80 else "GOOD")
        lines.append(f"  L2 Cache Hit Rate: {l2_rate:.1f}% ({quality})")
    elif "GL2C_HIT_sum" in metrics:
        hits = metrics['GL2C_HIT_sum']
        misses = metrics.get('GL2C_MISS_sum', 0)
        mc_rd = metrics.get('GL2C_MC_RDREQ_sum', 0)
        lines.append(f"  L2 Hits: {hits:.0f}, Misses: {misses:.0f}, DRAM Reads: {mc_rd:.0f}")

    # Wavefronts & instructions
    waves = metrics.get("SQ_WAVES_sum")
    if waves is not None:
        lines.append(f"  Total Wavefronts: {waves:.0f}")

    valu = metrics.get("SQ_INSTS_VALU_sum")
    if valu is not None:
        lines.append(f"  VALU Instructions (total): {valu:.0f}")
        if waves and waves > 0:
            lines.append(f"  VALU Insts/Wavefront: {valu/waves:.1f}")

    # LDS / Scratch
    lds = metrics.get("LDS_Block_Size")
    if lds is not None and lds > 0:
        lines.append(f"  LDS per Workgroup: {lds} bytes")
    scratch = metrics.get("Scratch_Size")
    if scratch is not None and scratch > 0:
        lines.append(f"  Scratch per Wave: {scratch} bytes (SPILLING - reduce VGPRs!)")

    # Kernel timing from rocprof
    dur = metrics.get("kernel_duration_us")
    if dur is not None:
        lines.append(f"  Kernel Duration (rocprof): {dur:.2f} us")

    # Workgroup info
    wg = metrics.get("Workgroup_Size")
    gs = metrics.get("Grid_Size")
    if wg is not None:
        lines.append(f"  Workgroup Size: {wg}, Grid Size: {gs or '?'}")

    # Add bench timing if available
    if bench_feedback:
        lines.append(f"\n[Benchmark Timing]\n  {bench_feedback}")

    # Optimization suggestions
    lines.append("\n[Optimization Hints]")
    if vgpr and vgpr > 48:
        lines.append("  - High VGPR usage limits occupancy. Try reducing temporaries or using simpler math.")
    if l2_rate is not None and l2_rate < 50:
        lines.append("  - Low L2 hit rate. Use float4 vectorized loads for better coalescing.")
    if scratch and scratch > 0:
        lines.append("  - Register spilling to scratch memory. Reduce local variables or simplify computation.")
    if valu and waves and waves > 0 and valu / waves > 100:
        lines.append("  - High instruction count per wave. Look for redundant computations to eliminate.")

    return "\n".join(lines)


def generate_rocprof_bench_script(sandbox: Path, arch: str, ext_name: str) -> str:
    """Generate a Python script for rocprofv3 that also prints timing comparison.

    Output line parsed by run_rocprof:
      TIMING: Torch Baseline: XXus, Torch Compile: XXus, HIP Extension: XXus
    """
    return f"""import sys, os, torch, importlib
sys.path.insert(0, '{sandbox}/agent_workdir')
sys.path.insert(0, '{sandbox}')
os.environ['PYTORCH_ROCM_ARCH'] = '{arch}'

spec = importlib.util.spec_from_file_location('model', '{sandbox}/agent_workdir/model.py')
model_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_mod)
Model = model_mod.Model
get_inputs = model_mod.get_inputs
get_init_inputs = model_mod.get_init_inputs

init_inputs = get_init_inputs()
if not isinstance(init_inputs, (list, tuple)):
    init_inputs = [init_inputs]
torch_model = Model(*init_inputs).eval().cuda()

hip_ext = importlib.import_module('{ext_name}')
def _get_weights(m):
    return [v.contiguous() for v in m.state_dict().values() if v.is_floating_point()]

class DynModelNew(Model):
    def forward(self, *args, **kwargs):
        weights = _get_weights(self)
        inputs = [a.contiguous() for a in args if isinstance(a, torch.Tensor)]
        return hip_ext.fused_kernel_forward(*inputs, *weights)

hip_model = DynModelNew(*init_inputs).eval().cuda()
hip_model.load_state_dict(torch_model.state_dict(), strict=False)

inputs = get_inputs()
if not isinstance(inputs, (list, tuple)):
    inputs = [inputs]
inputs = [x.cuda() if isinstance(x, torch.Tensor) else x for x in inputs]

def _bench(model, inp, warmup=5, iters=20):
    with torch.no_grad():
        for _ in range(warmup):
            model(*inp)
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            model(*inp)
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) * 1000.0 / iters  # us

with torch.no_grad():
    # Warmup + profiled dispatch (for rocprofv3 counter collection)
    for _ in range(3):
        hip_model(*inputs)
    torch.cuda.synchronize()
    hip_model(*inputs)
    torch.cuda.synchronize()

    # Self-timing comparison
    hip_us = _bench(hip_model, inputs)
    torch_us = _bench(torch_model, inputs)
    try:
        compiled_model = torch.compile(torch_model)
        compile_us = _bench(compiled_model, inputs)
    except Exception:
        compile_us = torch_us
    print(f"TIMING: Torch Baseline: {{torch_us:.3f}}us, Torch Compile: {{compile_us:.3f}}us, HIP Extension: {{hip_us:.3f}}us")
"""


# ---------------------------------------------------------------------------
# Model loading & generation helpers
# ---------------------------------------------------------------------------

def load_model(base_model_path: str, adapter_path: str = None, gpu_id: int = 0):
    """Load base model with optional LoRA adapter."""
    print(f"Loading tokenizer from {base_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    print(f"Loading base model on GPU {gpu_id}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map={"": gpu_id},
        torch_dtype=torch.bfloat16,
    )

    if adapter_path:
        print(f"Loading LoRA adapter from {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
        print("LoRA merged into base model")

    model.eval()
    return model, tokenizer


def fix_hip_compat(code: str) -> str:
    """Fix common CUDA-specific math functions to HIP equivalents."""
    # CUDA device math intrinsics that don't exist on HIP/RDNA3
    replacements = {
        '__tanhf(': 'tanhf(',
        '__tanh(': 'tanh(',
        '__sinf(': 'sinf(',
        '__cosf(': 'cosf(',
        '__logf(': 'logf(',
        '__log2f(': 'log2f(',
        '__sqrtf(': 'sqrtf(',
        '__powf(': 'powf(',
    }
    for old, new in replacements.items():
        code = code.replace(old, new)
    return code


def extract_compile_errors(feedback: str) -> str:
    """Extract key error lines from verbose compile output."""
    lines = feedback.splitlines()
    key_lines = []
    for line in lines:
        stripped = line.strip()
        if ": error:" in stripped or ": warning:" in stripped:
            key_lines.append(stripped)
        elif "note: candidate function not viable" in stripped:
            key_lines.append(stripped)
    if key_lines:
        return "\n".join(key_lines[:10])
    meaningful = [l for l in lines if l.strip() and not l.startswith(" ")]
    return "\n".join(meaningful[-5:])


def extract_hip_code(agent_output: str) -> str:
    """Extract the HIP kernel code from model output."""
    m = re.search(r'```(?:cpp|c\+\+|hip)?\n(.*?)\n```', agent_output, re.DOTALL)
    return m.group(1).strip() if m else ""


def build_prompt(tokenizer, skill_text: str, model_code: str,
                 feedback: str = None, prev_kernel: str = None,
                 perf_feedback: str = None):
    """Build chat prompt. Three modes:
      1) No feedback: initial generation
      2) feedback + prev_kernel: correctness fix
      3) perf_feedback + prev_kernel: performance optimization
    """
    user_msg = (
        f"Optimize this model with a single fused HIP kernel for gfx1100.\n\n"
        f"```python\n{model_code.strip()}\n```\n"
    )

    messages = [
        {"role": "system", "content": skill_text},
        {"role": "user", "content": user_msg},
    ]

    if perf_feedback and prev_kernel:
        messages.append({
            "role": "assistant",
            "content": f"**kernels/fused_kernel.hip**\n```cpp\n{prev_kernel}\n```\n<END_OF_OUTPUT>",
        })
        messages.append({
            "role": "user",
            "content": (
                f"The kernel above is functionally correct but slow. "
                f"Here are the AMD gfx1100 hardware profiling results:\n\n"
                f"```\n{perf_feedback}\n```\n\n"
                f"Optimize the kernel for better performance. Consider:\n"
                f"- Improve memory coalescing for higher L2 cache hit rate\n"
                f"- Reduce VGPR pressure to increase wavefront occupancy\n"
                f"- Use vectorized loads (float4) for bandwidth\n"
                f"- Minimize dependency chains for instruction-level parallelism\n"
                f"- Use shared memory (LDS) if data is reused\n\n"
                f"Output the optimized **kernels/fused_kernel.hip** file."
            ),
        })
    elif feedback and prev_kernel:
        messages.append({
            "role": "assistant",
            "content": f"**kernels/fused_kernel.hip**\n```cpp\n{prev_kernel}\n```\n<END_OF_OUTPUT>",
        })
        clean = extract_compile_errors(feedback) if "error:" in feedback else feedback[:800]
        messages.append({
            "role": "user",
            "content": (
                f"The kernel above has an issue:\n```\n{clean}\n```\n\n"
                f"Fix the kernel and output the corrected **kernels/fused_kernel.hip** file."
            ),
        })
    elif feedback:
        clean = extract_compile_errors(feedback) if "error:" in feedback else feedback[:800]
        messages.append({"role": "assistant", "content": "(previous attempt failed)"})
        messages.append({
            "role": "user",
            "content": (
                f"Your previous kernel had an error:\n```\n{clean}\n```\n\n"
                f"Output the corrected **kernels/fused_kernel.hip** file."
            ),
        })

    try:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        parts = []
        for m in messages:
            parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>")
        prompt = "\n".join(parts) + "\n<|im_start|>assistant\n"

    return prompt


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 1024,
             temperature: float = 0.3):
    """Generate completion."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            top_p=0.95 if temperature > 0 else 1.0,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)


async def evaluate(model_code: str, agent_output: str, arch: str, eval_gpu: str):
    """Evaluate agent output in sandbox."""
    workdir_abs = str(PROJECT_ROOT / "agent_workdir")
    interaction = HipKernelInteraction({
        "arch": arch,
        "workdir": workdir_abs,
        "compile_timeout": 90,
        "verify_timeout": 30,
        "profile_timeout": 60,
        "max_iterations": 1,
        "eval_gpu": eval_gpu,
    })

    iid = await interaction.start_interaction(model_code=model_code)
    sandbox_path = interaction._instances[iid]["sandbox"]
    msgs = [{"role": "assistant", "content": agent_output}]
    should_stop, feedback, reward, metadata = await interaction.generate_response(
        iid, msgs)
    # Return sandbox info for rocprof; caller must finalize
    return feedback, reward, metadata, sandbox_path, interaction, iid


def compute_self_reward(metrics: dict) -> float:
    """Compute reward from rocprof self-timing data.
    Same scheme as HipKernelInteraction._compute_reward:
      1.0 = correct, 2.0 = faster than eager, 3.0 = faster than eager + compile
    """
    baseline = metrics.get("self_torch_baseline_us")
    compile_t = metrics.get("self_torch_compile_us")
    hip = metrics.get("self_hip_extension_us")
    if baseline is None or hip is None:
        return 1.0  # Can't compute, keep at 1.0
    reward = 1.0
    if hip < baseline:
        reward = 2.0
    if compile_t is not None and hip < compile_t:
        reward = 3.0
    return reward


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

async def run_agent(args):
    """Main agent loop: generate -> evaluate -> rocprof -> optimize."""
    skill_path = PROJECT_ROOT / "agent_workdir" / args.arch / "SKILL.md"
    skill_text = skill_path.read_text() if skill_path.exists() else ""

    model_code = Path(args.model_code).read_text()
    model, tokenizer = load_model(args.base_model, args.adapter, gpu_id=args.gen_gpu)

    print(f"\n{'='*60}")
    print(f"Iterative HIP Kernel Agent (2-phase)")
    print(f"  Base model: {args.base_model}")
    print(f"  Adapter: {args.adapter or '(none)'}")
    print(f"  Target arch: {args.arch}")
    print(f"  Max iterations: {args.max_iters}")
    print(f"  Target reward: {args.target_reward}")
    print(f"  Model code: {args.model_code}")
    print(f"{'='*60}\n")

    feedback = None
    prev_kernel = None
    perf_feedback = None
    best_reward = -2.0
    best_output = None
    best_kernel = None
    stage = "unknown"
    iteration = 0
    phase = 1  # 1=correctness, 2=performance

    for iteration in range(1, args.max_iters + 1):
        phase_str = "CORRECTNESS" if phase == 1 else "PERFORMANCE"
        print(f"\n--- Iteration {iteration}/{args.max_iters} [Phase {phase}: {phase_str}] ---")

        # Generate
        t0 = time.time()
        prompt = build_prompt(tokenizer, skill_text, model_code,
                              feedback if phase == 1 else None,
                              prev_kernel,
                              perf_feedback if phase == 2 else None)
        agent_output = generate(model, tokenizer, prompt,
                                max_new_tokens=args.max_tokens,
                                temperature=args.temperature)
        gen_time = time.time() - t0
        print(f"Generation: {gen_time:.1f}s, {len(agent_output)} chars")

        # Show snippet
        hip_code = extract_hip_code(agent_output)
        if hip_code:
            code_lines = hip_code.split('\n')
            print(f"  HIP code: {len(code_lines)} lines")
            for line in code_lines[:5]:
                print(f"    {line}")
            if len(code_lines) > 5:
                print(f"    ... ({len(code_lines)-5} more lines)")
        else:
            print("  WARNING: No HIP code block found in output")
            print(f"  Output preview: {agent_output[:300]}")

        # Evaluate (compile + verify + bench)
        t0 = time.time()
        # Fix CUDA -> HIP math function compatibility
        agent_output = fix_hip_compat(agent_output)

        feedback, reward, metadata, sandbox_path, interaction_obj, iid = \
            await evaluate(model_code, agent_output, args.arch, str(args.eval_gpu))
        eval_time = time.time() - t0
        stage = metadata.get("stage", "unknown")
        print(f"Evaluation: {eval_time:.1f}s, stage={stage}, reward={reward:+.2f}")

        # --- rocprofv3 profiling for verified kernels ---
        rocprof_text = ""
        if reward >= 1.0:
            print(f"  Running rocprofv3 hardware counter analysis...")
            ext_name = "hip_ext_" + sandbox_path.name.replace("-", "_")
            bench_script = generate_rocprof_bench_script(
                sandbox_path, args.arch, ext_name)
            rocprof_metrics = await run_rocprof(
                sandbox_path, args.arch, str(args.eval_gpu),
                bench_script, timeout=120)

            # Build timing string: prefer HipKernelInteraction metadata, fallback to self-timing
            bench_timing = ""
            if "torch_baseline_us" in metadata:
                bench_timing = (
                    f"Torch Baseline: {metadata['torch_baseline_us']:.1f}us, "
                    f"Torch Compile: {metadata['torch_compile_us']:.1f}us, "
                    f"HIP Extension: {metadata['hip_extension_us']:.1f}us"
                )

            if "error" not in rocprof_metrics:
                # Use self-timing from rocprof bench script if bench.py failed
                if not bench_timing and "self_torch_baseline_us" in rocprof_metrics:
                    bench_timing = (
                        f"Torch Baseline: {rocprof_metrics['self_torch_baseline_us']:.1f}us, "
                        f"Torch Compile: {rocprof_metrics['self_torch_compile_us']:.1f}us, "
                        f"HIP Extension: {rocprof_metrics['self_hip_extension_us']:.1f}us"
                    )
                rocprof_text = format_rocprof_feedback(rocprof_metrics, bench_timing)
                print(rocprof_text)

                # Upgrade reward using self-timing if bench.py failed (profile_error)
                if stage == "profile_error" and "self_torch_baseline_us" in rocprof_metrics:
                    self_reward = compute_self_reward(rocprof_metrics)
                    if self_reward > reward:
                        reward = self_reward
                        print(f"  >> Reward upgraded to {reward:+.1f} via self-timing")
            else:
                print(f"  rocprofv3 note: {rocprof_metrics.get('error', '')[:200]}")
                if bench_timing:
                    rocprof_text = f"[Benchmark Timing]\n  {bench_timing}"

        # Finalize sandbox
        await interaction_obj.finalize_interaction(iid)

        # Track best
        if reward > best_reward:
            best_reward = reward
            best_output = agent_output
            best_kernel = hip_code

        # Check target
        if reward >= args.target_reward:
            print(f"\n*** TARGET REACHED at iteration {iteration}! Reward: {reward:+.1f} ***")
            break

        # Phase transition logic
        if reward >= 1.0 and phase == 1:
            phase = 2
            feedback = None
            prev_kernel = hip_code or best_kernel
            perf_feedback = rocprof_text or feedback
            print(f"  >> Switching to Phase 2: Performance Optimization")
            continue

        if phase == 2:
            if reward >= 1.0:
                # Still correct, update perf feedback for next iteration
                prev_kernel = hip_code or best_kernel
                perf_feedback = rocprof_text or feedback
            else:
                # Regression — back to correctness phase, but keep best kernel
                phase = 1
                perf_feedback = None
                prev_kernel = best_kernel  # Use the last known-good kernel
                print(f"  >> Regression! Back to Phase 1: Correctness")
        else:
            # Phase 1: show error for debugging
            if iteration < args.max_iters:
                clean = (extract_compile_errors(feedback) if "error:" in feedback
                         else feedback[:500])
                print(f"  Feedback: {clean[:300]}")

    if iteration == args.max_iters and best_reward < args.target_reward:
        print(f"\n*** Max iterations reached. Best reward: {best_reward:+.2f} ***")

    # Summary
    print(f"\n{'='*60}")
    print(f"RESULT SUMMARY")
    print(f"  Iterations used: {iteration}")
    print(f"  Best reward: {best_reward:+.2f}")
    print(f"  Final stage: {stage}")
    print(f"  Final phase: {phase} ({'correctness' if phase == 1 else 'performance'})")
    if best_reward >= 3.0:
        print(f"  Status: OPTIMAL (faster than eager + torch.compile)")
    elif best_reward >= 2.0:
        print(f"  Status: FAST (faster than eager)")
    elif best_reward >= 1.0:
        print(f"  Status: CORRECT (verified, not yet fast)")
    elif best_reward > 0:
        print(f"  Status: PARTIAL (compiled, values approximate)")
    elif best_reward >= -0.5:
        print(f"  Status: COMPILED (verification failed)")
    else:
        print(f"  Status: FAILED")
    print(f"{'='*60}")

    if best_output and args.save_output:
        out_path = Path(args.save_output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(best_output)
        print(f"Best output saved to: {out_path}")

    return best_reward


def parse_args():
    p = argparse.ArgumentParser(description="Iterative HIP kernel generation agent")
    p.add_argument("--base-model", required=True, help="Path to base model")
    p.add_argument("--adapter", default=None, help="Path to LoRA adapter checkpoint")
    p.add_argument("--model-code", required=True, help="Path to PyTorch model .py file")
    p.add_argument("--arch", default="gfx1100", help="GPU architecture")
    p.add_argument("--max-iters", type=int, default=10)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--gen-gpu", type=int, default=0)
    p.add_argument("--eval-gpu", type=int, default=0)
    p.add_argument("--target-reward", type=float, default=3.0,
                   help="Target reward (3.0 = beats eager + torch.compile)")
    p.add_argument("--save-output", default=None, help="Save best output to file")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    reward = asyncio.run(run_agent(args))
    sys.exit(0 if reward >= args.target_reward else 1)
