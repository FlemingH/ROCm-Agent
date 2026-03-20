#!/usr/bin/env python3
"""GRPO training with parallel HIP kernel evaluation.

Usage:
    python3 tools/train_grpo.py --model models/Qwen3-8B --max-steps 5
"""

import argparse
import asyncio
import json
import os
import random
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch
import pyarrow.parquet as pq
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import GRPOConfig, GRPOTrainer

from hip_kernel_interaction import HipKernelInteraction

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_dataset_from_parquet(path: str) -> Dataset:
    table = pq.read_table(path)
    data = table.to_pydict()
    prompts = [json.loads(raw) for raw in data["prompt"]]
    return Dataset.from_dict({"prompt": prompts})


def _compile_only(args_tuple: tuple) -> tuple:
    """Subprocess worker: parse + compile only (CPU, no GPU). Returns (sandbox_path, reward_so_far, stage)."""
    model_code, agent_output, arch, workdir_abs = args_tuple
    interaction = HipKernelInteraction({
        "arch": arch,
        "workdir": workdir_abs,
        "compile_timeout": 90,
        "verify_timeout": 30,
        "profile_timeout": 60,
        "max_iterations": 1,
    })
    loop = asyncio.new_event_loop()
    try:
        iid = loop.run_until_complete(interaction.start_interaction(model_code=model_code))
        msgs = [{"role": "assistant", "content": agent_output}]

        inst = interaction._instances[iid]
        inst["iteration"] += 1
        sandbox = inst["sandbox"]

        code = interaction._extract_last_assistant(msgs)
        if not code:
            loop.run_until_complete(interaction.finalize_interaction(iid))
            return None, -1.0, "no_code"

        interaction._write_agent_output(sandbox, code)
        parsed = interaction._parse_code_blocks(code)
        has_model_new = "model_new.py" in parsed
        has_hip = any(k.endswith(".hip") for k in parsed)
        has_binding = any(k.endswith("_binding.cpp") for k in parsed)

        if not has_model_new and not has_hip:
            loop.run_until_complete(interaction.finalize_interaction(iid))
            return None, -1.0, "no_code"

        compile_ok, compile_msg = loop.run_until_complete(interaction._run_compile(sandbox))
        if not compile_ok:
            if has_model_new and has_hip and has_binding:
                reward = -0.25 if ("undefined" in compile_msg.lower() or "linker" in compile_msg.lower()) else -0.5
            elif has_model_new and (has_hip or has_binding):
                reward = -0.75
            else:
                reward = -0.9
            loop.run_until_complete(interaction.finalize_interaction(iid))
            return None, reward, "compile_error"

        return str(sandbox), 0.0, "compiled"
    except Exception:
        return None, -1.0, "error"
    finally:
        loop.close()


def _gpu_eval(sandbox_path: str, arch: str) -> float:
    """Sequential GPU evaluation: verify + profile. Called in main process."""
    interaction = HipKernelInteraction({
        "arch": arch, "workdir": "agent_workdir",
        "verify_timeout": 30, "profile_timeout": 60,
    })
    loop = asyncio.new_event_loop()
    try:
        sandbox = Path(sandbox_path)
        verify_ok, verify_msg = loop.run_until_complete(interaction._run_verify(sandbox))
        if not verify_ok:
            return 0.0

        profile_ok, profile_result = loop.run_until_complete(interaction._run_profile(sandbox))
        if not profile_ok:
            return 1.0

        return interaction._compute_reward(profile_result)
    except Exception:
        return 0.0
    finally:
        loop.close()
        import shutil
        shutil.rmtree(sandbox_path, ignore_errors=True)


def make_reward_fn(arch: str, workdir_abs: str, num_workers: int = 4,
                   reward_noise: float = 0.1):
    """Compile-parallel, GPU-serial reward function."""
    executor = ProcessPoolExecutor(max_workers=num_workers)

    def reward_fn(completions: list[str], prompts: list = None, **kwargs) -> list[float]:
        tasks = []
        for i, completion in enumerate(completions):
            text = completion if isinstance(completion, str) else (
                completion[-1]["content"] if completion else ""
            )
            if not text or len(text.strip()) < 20:
                tasks.append(None)
                continue

            model_code = ""
            for msg in (prompts[i] if prompts else []):
                if msg.get("role") == "user":
                    m = re.search(r'```python\n(.*?)\n```', msg["content"], re.DOTALL)
                    if m:
                        model_code = m.group(1)
                        break
            if not model_code:
                tasks.append(None)
                continue
            tasks.append((model_code, text, arch, workdir_abs))

        # Phase 1: Compile in parallel (CPU only, no GPU)
        futures = [executor.submit(_compile_only, t) if t else None for t in tasks]
        compile_results = []
        for f in futures:
            if f is None:
                compile_results.append((None, -1.0, "skip"))
            else:
                try:
                    compile_results.append(f.result(timeout=180))
                except Exception:
                    compile_results.append((None, -1.0, "timeout"))

        # Phase 2: GPU eval sequentially (verify + profile, one at a time)
        rewards = []
        for sandbox_path, reward_so_far, stage in compile_results:
            if stage != "compiled" or sandbox_path is None:
                rewards.append(reward_so_far + random.uniform(-reward_noise, reward_noise))
            else:
                gpu_reward = _gpu_eval(sandbox_path, arch)
                rewards.append(gpu_reward + random.uniform(-reward_noise, reward_noise))

        return rewards

    return reward_fn


def load_model(model_path: str):
    print(f"Loading model from {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map={"": 0}, dtype=torch.bfloat16,
    )
    print(f"Model loaded on GPU 0: {torch.cuda.memory_allocated(0) / 1e9:.1f} GB")
    return model


def apply_lora(model, r: int = 8, alpha: int = 16, dropout: float = 0.05):
    config = LoraConfig(
        r=r, lora_alpha=alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=dropout,
        task_type="CAUSAL_LM",
    )
    model.enable_input_require_grads()
    model = get_peft_model(model, config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"LoRA: {trainable / 1e6:.1f}M trainable / {total / 1e6:.1f}M total ({100 * trainable / total:.2f}%)")
    return model


def parse_args():
    p = argparse.ArgumentParser(description="GRPO training with parallel HIP eval")
    p.add_argument("--model", required=True)
    p.add_argument("--train-data", default="data/rocm_agent_ops/train.parquet")
    p.add_argument("--output-dir", default="checkpoints/grpo")
    p.add_argument("--max-completion-length", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--gradient-accumulation", type=int, default=4)
    p.add_argument("--num-generations", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--save-steps", type=int, default=100)
    p.add_argument("--logging-steps", type=int, default=1)
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--arch", default="gfx1201")
    p.add_argument("--reward-workers", type=int, default=4)
    p.add_argument("--temperature", type=float, default=1.0)
    return p.parse_args()


def main():
    args = parse_args()
    os.chdir(PROJECT_ROOT)
    workdir_abs = str(PROJECT_ROOT / "agent_workdir")

    dataset = load_dataset_from_parquet(args.train_data)
    print(f"Loaded {len(dataset)} training samples")

    model = load_model(args.model)
    model.generation_config.temperature = args.temperature
    model.generation_config.top_k = 0
    model.generation_config.top_p = 1.0
    model.generation_config.do_sample = True
    model = apply_lora(model, r=args.lora_r, alpha=args.lora_alpha)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    reward_fn = make_reward_fn(args.arch, workdir_abs, args.reward_workers)

    gen_batch = max(args.batch_size, args.num_generations)
    gen_batch -= gen_batch % args.num_generations

    config = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        num_generations=args.num_generations,
        generation_batch_size=gen_batch,
        max_completion_length=args.max_completion_length,
        learning_rate=args.lr,
        temperature=args.temperature,
        bf16=True,
        use_vllm=False,
        gradient_checkpointing=True,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        train_dataset=dataset,
        args=config,
        processing_class=tokenizer,
    )

    eff = args.batch_size * args.gradient_accumulation
    steps = len(dataset) // eff
    total = steps * args.epochs if args.max_steps < 0 else args.max_steps
    print(f"GRPO: model={args.model}, arch={args.arch}")
    print(f"  epochs={args.epochs}, steps/epoch={steps}, total={total}")
    print(f"  batch={args.batch_size}, grad_accum={args.gradient_accumulation}, num_gen={args.num_generations}")
    print(f"  max_len={args.max_completion_length}, temp={args.temperature}, workers={args.reward_workers}")

    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"Training complete. Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
