#!/usr/bin/env python3
"""GRPO training: single-GPU + LoRA + parallel local HIP eval on R9700.

Optimized for ~2 day training on R9700 (32GB GDDR7, gfx1201):
  - BF16 model (3-7B) + LoRA training
  - Parallel reward computation (multi-process hipcc compile/verify/profile)

Usage:
    python3 tools/train_grpo.py \
      --model models/Qwen2.5-Coder-3B-Instruct \
      --max-steps 5
"""

import argparse
import asyncio
import json
import os
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
    prompts = []
    for raw_prompt in data["prompt"]:
        messages = json.loads(raw_prompt)
        prompts.append(messages)
    return Dataset.from_dict({"prompt": prompts})


def _evaluate_single(args_tuple: tuple) -> float:
    """Worker function for parallel reward. Runs in a subprocess."""
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
        iid = loop.run_until_complete(
            interaction.start_interaction(model_code=model_code)
        )
        messages = [{"role": "assistant", "content": agent_output}]
        _, _, reward, _ = loop.run_until_complete(
            interaction.generate_response(iid, messages)
        )
        loop.run_until_complete(interaction.finalize_interaction(iid))
        return max(reward, -1.0)
    except Exception:
        return -1.0
    finally:
        loop.close()


def make_reward_fn(arch: str, workdir_abs: str, num_workers: int = 4):
    """Create a parallel reward function using ProcessPoolExecutor."""
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

            prompt_msgs = prompts[i] if prompts else []
            model_code = ""
            for msg in prompt_msgs:
                if msg.get("role") == "user":
                    code_match = re.search(r'```python\n(.*?)\n```', msg["content"], re.DOTALL)
                    if code_match:
                        model_code = code_match.group(1)
                        break

            if not model_code:
                tasks.append(None)
                continue

            tasks.append((model_code, text, arch, workdir_abs))

        futures = []
        for task in tasks:
            if task is None:
                futures.append(None)
            else:
                futures.append(executor.submit(_evaluate_single, task))

        rewards = []
        for f in futures:
            if f is None:
                rewards.append(-1.0)
            else:
                try:
                    rewards.append(f.result(timeout=180))
                except Exception:
                    rewards.append(-1.0)

        return rewards

    return reward_fn


def resolve_model_path(model_path: str) -> str:
    """Resolve to GPTQ variant if available."""
    gptq_path = model_path.rstrip("/") + "-GPTQ-4bit"
    if Path(gptq_path).exists():
        return gptq_path
    return model_path


def load_model_single_gpu(model_path: str):
    """Load model on a single GPU."""
    resolved = resolve_model_path(model_path)
    print(f"Loading model from {resolved}")
    model = AutoModelForCausalLM.from_pretrained(
        resolved,
        device_map={"": 0},
        dtype=torch.bfloat16,
    )
    mem_gb = torch.cuda.memory_allocated(0) / 1e9
    print(f"Model loaded on GPU 0: {mem_gb:.1f} GB")
    return model


def apply_lora(model, r: int = 8, alpha: int = 16, dropout: float = 0.05):
    lora_config = LoraConfig(
        r=r, lora_alpha=alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=dropout,
        task_type="CAUSAL_LM",
    )
    model.enable_input_require_grads()
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"LoRA: {trainable / 1e6:.1f}M trainable / {total / 1e6:.1f}M total ({100 * trainable / total:.2f}%)")
    return model


def parse_args():
    parser = argparse.ArgumentParser(description="GRPO training (single GPU, parallel eval)")
    parser.add_argument("--model", default="models/Qwen2.5-Coder-3B-Instruct",
                        help="Model path (GPTQ variant auto-detected)")
    parser.add_argument("--train-data", default="data/rocm_agent_ops/train.parquet")
    parser.add_argument("--output-dir", default="checkpoints/grpo")
    parser.add_argument("--max-completion-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--num-generations", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--arch", default="gfx1201")
    parser.add_argument("--reward-workers", type=int, default=4,
                        help="Parallel workers for reward computation")
    return parser.parse_args()


def main():
    args = parse_args()

    os.chdir(PROJECT_ROOT)
    workdir_abs = str(PROJECT_ROOT / "agent_workdir")

    dataset = load_dataset_from_parquet(args.train_data)
    print(f"Loaded {len(dataset)} training samples")

    resolved_model_path = resolve_model_path(args.model)

    model = load_model_single_gpu(args.model)
    model = apply_lora(model, r=args.lora_r, alpha=args.lora_alpha)

    tokenizer = AutoTokenizer.from_pretrained(resolved_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    reward_fn = make_reward_fn(args.arch, workdir_abs, args.reward_workers)

    gen_batch = max(args.batch_size, args.num_generations)
    gen_batch = gen_batch - (gen_batch % args.num_generations)

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

    eff_batch = args.batch_size * args.gradient_accumulation
    steps_per_epoch = len(dataset) // eff_batch
    total_steps = steps_per_epoch * args.epochs if args.max_steps < 0 else args.max_steps
    print(f"Starting GRPO: model={resolved_model_path}, arch={args.arch}")
    print(f"  epochs={args.epochs}, steps/epoch={steps_per_epoch}, total={total_steps}")
    print(f"  batch={args.batch_size}, grad_accum={args.gradient_accumulation}, eff_batch={eff_batch}")
    print(f"  num_gen={args.num_generations}, max_len={args.max_completion_length}")
    print(f"  reward_workers={args.reward_workers}")

    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"Training complete. Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
