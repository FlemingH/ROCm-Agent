#!/usr/bin/env python3
"""Convert CUDA-Agent-Ops-6K dataset to chat format with interaction_kwargs.

Reads the raw parquet (columns: ops, data_source, code) and produces
train/val parquet files in chat format. Each sample becomes a system+user
prompt asking the agent to optimize the given model.py.

Usage:
    python3 tools/prepare_data.py \
        --input data/CUDA-Agent-Ops-6K/data.parquet \
        --output data/rocm_agent_ops/ \
        --arch gfx1201 \
        --skill-path agent_workdir/gfx1201/SKILL.md \
        --val-ratio 0.1
"""

import argparse
import json
import random
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def load_skill(skill_path: str) -> str:
    return Path(skill_path).read_text()


def classify_difficulty(ops: list[str]) -> str:
    n = len(ops)
    if n <= 1:
        return "easy"
    elif n <= 3:
        return "medium"
    else:
        return "hard"


def make_chat_sample(code: str, ops: list[str], data_source: str,
                     skill_text: str, arch: str) -> dict:
    if isinstance(ops, list):
        ops_list = ops
    elif isinstance(ops, str) and ops.startswith("["):
        ops_list = json.loads(ops)
    else:
        ops_list = [ops] if ops else []
    difficulty = classify_difficulty(ops_list)

    system_msg = skill_text

    user_msg = (
        f"Below is the PyTorch model you need to optimize. "
        f"Create high-performance HIP kernels for {arch}.\n\n"
        f"```python\n{code.strip()}\n```"
    )

    prompt = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    interaction_kwargs = {
        "name": "hip_kernel",
        "model_code": code,
        "ops": json.dumps(ops_list) if isinstance(ops_list, list) else ops_list,
        "difficulty": difficulty,
        "data_source": data_source,
    }

    return {
        "prompt": json.dumps(prompt),
        "interaction_kwargs": json.dumps(interaction_kwargs),
        "difficulty": difficulty,
        "data_source": data_source,
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare CUDA-Agent-Ops-6K for GRPO training")
    parser.add_argument("--input", required=True, help="Path to raw data.parquet")
    parser.add_argument("--output", required=True, help="Output directory for train/val parquet")
    parser.add_argument("--arch", default="gfx1201", help="Target GPU architecture")
    parser.add_argument("--skill-path", default="agent_workdir/gfx1201/SKILL.md",
                        help="Path to SKILL.md for system prompt")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    skill_text = load_skill(args.skill_path)
    table = pq.read_table(args.input)
    data = table.to_pydict()
    n = len(data["code"])
    print(f"Loaded {n} samples from {args.input}")

    samples = []
    for i in range(n):
        sample = make_chat_sample(
            code=data["code"][i],
            ops=data["ops"][i],
            data_source=data["data_source"][i],
            skill_text=skill_text,
            arch=args.arch,
        )
        samples.append(sample)

    random.seed(args.seed)
    random.shuffle(samples)

    val_size = int(n * args.val_ratio)
    val_samples = samples[:val_size]
    train_samples = samples[val_size:]

    diff_counts = {}
    for s in train_samples:
        diff_counts[s["difficulty"]] = diff_counts.get(s["difficulty"], 0) + 1

    print(f"Train: {len(train_samples)}, Val: {val_size}")
    print(f"Difficulty distribution (train): {diff_counts}")

    def to_table(rows):
        cols = {k: [r[k] for r in rows] for k in rows[0]}
        return pa.table(cols)

    pq.write_table(to_table(train_samples), out_dir / "train.parquet")
    pq.write_table(to_table(val_samples), out_dir / "val.parquet")
    print(f"Written to {out_dir}/train.parquet and {out_dir}/val.parquet")


if __name__ == "__main__":
    main()
