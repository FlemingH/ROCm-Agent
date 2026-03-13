#!/usr/bin/env python3
"""Filter high-quality trajectories from Phase 1 GRPO for RFT supervised fine-tuning.

Reads trajectory logs (jsonl or parquet), keeps only those with reward >= threshold,
and outputs a parquet file suitable for verl SFT training.

Usage:
    python3 tools/filter_trajectories.py \
        --input checkpoints/phase1_moe_grpo/trajectories/ \
        --output data/rocm_agent_ops/rft_filtered.parquet \
        --min-reward 2
"""

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def load_trajectories(input_path: str) -> list[dict]:
    p = Path(input_path)
    rows = []

    if p.is_file() and p.suffix == ".parquet":
        table = pq.read_table(p)
        rows = table.to_pydict()
        rows = [dict(zip(rows.keys(), vals)) for vals in zip(*rows.values())]

    elif p.is_file() and p.suffix in (".jsonl", ".json"):
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))

    elif p.is_dir():
        for f in sorted(p.glob("*.jsonl")):
            with open(f) as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
        for f in sorted(p.glob("*.parquet")):
            table = pq.read_table(f)
            d = table.to_pydict()
            rows.extend(dict(zip(d.keys(), vals)) for vals in zip(*d.values()))
    else:
        raise FileNotFoundError(f"Input path not found: {input_path}")

    return rows


def extract_sft_sample(traj: dict) -> dict | None:
    """Extract a (prompt, response) pair from a trajectory record.

    verl stores trajectories in various formats depending on version.
    This handles common patterns:
      - {"prompt": [...], "response": "...", "reward": float}
      - {"messages": [...], "reward": float}
    """
    reward = traj.get("reward", traj.get("score", None))
    if reward is None:
        return None

    if "prompt" in traj and "response" in traj:
        prompt = traj["prompt"]
        if isinstance(prompt, str):
            prompt = json.loads(prompt)
        messages = prompt + [{"role": "assistant", "content": traj["response"]}]
    elif "messages" in traj:
        messages = traj["messages"]
        if isinstance(messages, str):
            messages = json.loads(messages)
    else:
        return None

    return {
        "messages": json.dumps(messages),
        "reward": float(reward),
    }


def main():
    parser = argparse.ArgumentParser(description="Filter trajectories for RFT")
    parser.add_argument("--input", required=True, help="Trajectory file or directory")
    parser.add_argument("--output", required=True, help="Output parquet path")
    parser.add_argument("--min-reward", type=float, default=2.0,
                        help="Minimum reward to keep (default: 2.0 = faster than Eager)")
    args = parser.parse_args()

    trajectories = load_trajectories(args.input)
    print(f"Loaded {len(trajectories)} trajectories from {args.input}")

    filtered = []
    reward_dist = {}
    for traj in trajectories:
        sample = extract_sft_sample(traj)
        if sample is None:
            continue
        r = sample["reward"]
        bucket = int(r)
        reward_dist[bucket] = reward_dist.get(bucket, 0) + 1
        if r >= args.min_reward:
            filtered.append(sample)

    print(f"Reward distribution: {dict(sorted(reward_dist.items()))}")
    print(f"Filtered: {len(filtered)} samples (reward >= {args.min_reward})")

    if not filtered:
        print("WARNING: No samples passed the filter. Try lowering --min-reward.")
        return

    cols = {k: [s[k] for s in filtered] for k in filtered[0]}
    table = pa.table(cols)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out_path)
    print(f"Written to {out_path}")


if __name__ == "__main__":
    main()
