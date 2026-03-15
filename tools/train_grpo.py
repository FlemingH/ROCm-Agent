#!/usr/bin/env python3
"""GRPO training script using TRL GRPOTrainer.

Phase 1: Single-turn GRPO warmup. The model generates HIP kernel code
for a given PyTorch model, and a reward function scores correctness/performance.

Usage:
    # Single GPU
    python3 tools/train_grpo.py --model Qwen/Qwen2.5-0.5B-Instruct

    # Multi-GPU (4x W7800)
    accelerate launch --num_processes=4 tools/train_grpo.py \
        --model Qwen/Qwen2.5-0.5B-Instruct

    # Full training with 30B model (needs quantization)
    accelerate launch --num_processes=4 tools/train_grpo.py \
        --model models/Qwen3-Coder-30B-A3B-Instruct \
        --max-prompt-length 4096 --max-completion-length 8192 \
        --batch-size 8 --epochs 5
"""

import argparse
import json
import re
import subprocess
import tempfile
from pathlib import Path

import pyarrow.parquet as pq
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer


def load_dataset_from_parquet(path: str) -> Dataset:
    table = pq.read_table(path)
    data = table.to_pydict()

    prompts = []
    for raw_prompt in data["prompt"]:
        messages = json.loads(raw_prompt)
        prompts.append(messages)

    return Dataset.from_dict({"prompt": prompts})


def hip_kernel_reward(completions: list[str], prompts: list[str] = None, **kwargs) -> list[float]:
    """Reward function for HIP kernel generation.

    Scoring:
      0.0  empty or unparseable response
      0.1  has code blocks but won't compile
      0.5  compiles but not verified
      1.0  correct output (verification passes)

    For the smoke test, we use a lightweight heuristic.
    Full compile/verify/profile rewards via hip_kernel_interaction
    are used in Phase 3 multi-turn training.
    """
    rewards = []
    for completion in completions:
        text = completion if isinstance(completion, str) else completion[-1]["content"] if completion else ""

        if not text or len(text.strip()) < 20:
            rewards.append(0.0)
            continue

        has_hip_code = bool(re.search(r'__global__|#include.*hip|hipStream_t|__shared__', text))
        has_model_new = bool(re.search(r'class\s+ModelNew|import\s+hip_extension', text))
        has_code_blocks = bool(re.search(r'```(?:cpp|python|hip)', text))

        score = 0.0
        if has_code_blocks:
            score += 0.1
        if has_hip_code:
            score += 0.4
        if has_model_new:
            score += 0.5

        rewards.append(min(score, 1.0))

    return rewards


def parse_args():
    parser = argparse.ArgumentParser(description="GRPO training for HIP kernel generation")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--train-data", default="data/rocm_agent_ops/train.parquet")
    parser.add_argument("--output-dir", default="checkpoints/phase1_grpo")
    parser.add_argument("--max-prompt-length", type=int, default=512)
    parser.add_argument("--max-completion-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--num-generations", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--logging-steps", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()

    dataset = load_dataset_from_parquet(args.train_data)
    print(f"Loaded {len(dataset)} training samples")

    config = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        learning_rate=args.lr,
        bf16=True,
        use_vllm=False,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=hip_kernel_reward,
        train_dataset=dataset,
        args=config,
    )

    print(f"Starting GRPO training: model={args.model}, epochs={args.epochs}, batch={args.batch_size}")
    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"Training complete. Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
