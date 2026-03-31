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


_ref_code_fn = None

def _get_ref_code_fn(arch: str):
    global _ref_code_fn
    if _ref_code_fn is None:
        import importlib.util
        ref_path = Path(__file__).resolve().parent.parent / "agent_workdir" / arch / "ref_snippets.py"
        spec = importlib.util.spec_from_file_location("ref_snippets", ref_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _ref_code_fn = mod.get_ref_code
    return _ref_code_fn


def make_chat_sample(code: str, ops: list[str], data_source: str,
                     skill_text: str, arch: str) -> dict:
    get_ref_code = _get_ref_code_fn(arch)

    if isinstance(ops, list):
        ops_list = ops
    elif isinstance(ops, str) and ops.startswith("["):
        ops_list = json.loads(ops)
    else:
        ops_list = [ops] if ops else []
    difficulty = classify_difficulty(ops_list)

    ref_code = get_ref_code(ops_list, max_snippets=3)
    ref_section = ""
    if ref_code:
        ref_section = (
            f"\n\nReference HIP kernels from AMD rocm-libraries:\n"
            f"```cpp\n{ref_code}\n```"
        )

    user_msg = (
        f"Optimize this model with HIP kernels for {arch}. "
        f"Output exactly 1 file.\n\n"
        f"```python\n{code.strip()}\n```"
        f"{ref_section}"
    )

    prompt = [
        {"role": "system", "content": skill_text},
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
    parser.add_argument("--skill-path", default=None,
                        help="Path to SKILL.md (default: agent_workdir/<arch>/SKILL.md)")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    if args.skill_path is None:
        args.skill_path = f"agent_workdir/{args.arch}/SKILL.md"

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
