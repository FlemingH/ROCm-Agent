#!/usr/bin/env python3
"""GRPO training with parallel compile + GPU-isolated eval.

Supports two generation modes:
  1. Local generation (default): model.generate() on the training GPU.
  2. vLLM server mode (--use-vllm): generation offloaded to an external
     vLLM server, with automatic weight sync after each step.

Usage:
    # Single GPU, local generation:
    python3 tools/train_grpo.py --model models/Qwen3-8B --max-steps 5

    # 4-GPU with vLLM (GPU 0+1 vLLM, GPU 2 train, GPU 3 eval):
    python3 tools/vllm_serve.py ...          # starts vLLM on GPU 0+1
    python3 tools/train_grpo.py \\
      --model models/Qwen3-8B --use-vllm --train-gpu 2 --eval-gpu 3 \\
      --arch gfx1100
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
STRICT_OUTPUT_STOP_MARKER = "<END_OF_OUTPUT>"
STRICT_ONE_FILE_OUTPUT_REGEX = (
    r"\*\*kernels/fused_kernel\.hip\*\*\n```cpp\n[\s\S]*?\n```\n<END_OF_OUTPUT>"
)


def load_dataset_from_parquet(path: str) -> Dataset:
    table = pq.read_table(path)
    data = table.to_pydict()
    prompts = [json.loads(raw) for raw in data["prompt"]]
    return Dataset.from_dict({"prompt": prompts})


def _evaluate_single(args_tuple: tuple) -> float:
    """Subprocess worker: compile (CPU) + verify/bench (eval GPU)."""
    model_code, agent_output, arch, workdir_abs, eval_gpu = args_tuple
    interaction = HipKernelInteraction({
        "arch": arch, "workdir": workdir_abs,
        "compile_timeout": 90, "verify_timeout": 30,
        "profile_timeout": 60, "max_iterations": 1,
        "eval_gpu": eval_gpu,
    })
    loop = asyncio.new_event_loop()
    try:
        iid = loop.run_until_complete(interaction.start_interaction(model_code=model_code))
        msgs = [{"role": "assistant", "content": agent_output}]
        _, _, reward, _ = loop.run_until_complete(interaction.generate_response(iid, msgs))
        loop.run_until_complete(interaction.finalize_interaction(iid))
        return max(reward, -1.0)
    except Exception:
        return -1.0
    finally:
        loop.close()


def make_reward_fn(arch: str, workdir_abs: str, eval_gpu: str,
                   num_workers: int = 4, reward_noise: float = 0.1):
    """Parallel reward: compile on CPU, verify/bench on eval GPU."""
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
            tasks.append((model_code, text, arch, workdir_abs, eval_gpu))

        futures = [executor.submit(_evaluate_single, t) if t else None for t in tasks]

        rewards = []
        for f in futures:
            if f is None:
                rewards.append(-1.0)
            else:
                try:
                    rewards.append(f.result(timeout=180) + random.uniform(-reward_noise, reward_noise))
                except Exception:
                    rewards.append(-1.0)
        return rewards

    return reward_fn


def load_model(model_path: str, gpu_id: int = 0):
    print(f"Loading model from {model_path} on GPU {gpu_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map={"": gpu_id}, dtype=torch.bfloat16,
    )
    print(f"Model loaded on GPU {gpu_id}: {torch.cuda.memory_allocated(gpu_id) / 1e9:.1f} GB")
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
    p = argparse.ArgumentParser(description="GRPO training with GPU-isolated eval")
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
    p.add_argument("--save-steps", type=int, default=50)
    p.add_argument("--logging-steps", type=int, default=1)
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--arch", default="gfx1201")
    p.add_argument("--reward-workers", type=int, default=4)
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--beta", type=float, default=0.04,
                   help="KL regularization coefficient (0=no KL penalty, 0.04=standard)")
    p.add_argument("--eval-gpu", default=None,
                   help="GPU id for verify/bench (e.g. '3'). If unset, uses same GPU as training.")
    p.add_argument("--use-vllm", action="store_true",
                   help="Use external vLLM server for generation (start with scripts/start-vllm.sh)")
    p.add_argument("--train-gpu", type=int, default=0,
                   help="GPU id for training model (default 0; set to 2 for 4-GPU vLLM setup)")
    p.add_argument("--vllm-port", type=int, default=8000,
                   help="vLLM server port (default 8000)")
    p.add_argument(
        "--conservative-eos-stop",
        action="store_true",
        help="Add conservative explicit EOS/stop controls for generation using tokenizer eos_token_id/eos_token.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    os.chdir(PROJECT_ROOT)
    workdir_abs = str(PROJECT_ROOT / "agent_workdir")

    eval_gpu = args.eval_gpu if args.eval_gpu else str(args.train_gpu)

    dataset = load_dataset_from_parquet(args.train_data)
    print(f"Loaded {len(dataset)} training samples")

    model = load_model(args.model, gpu_id=args.train_gpu)
    if not args.use_vllm:
        model.generation_config.temperature = args.temperature
        model.generation_config.top_k = 0
        model.generation_config.top_p = 1.0
        model.generation_config.do_sample = True
    model = apply_lora(model, r=args.lora_r, alpha=args.lora_alpha)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    generation_kwargs = None
    if args.conservative_eos_stop:
        generation_kwargs = {"ignore_eos": False}
        if tokenizer.eos_token_id is not None:
            generation_kwargs["stop_token_ids"] = [int(tokenizer.eos_token_id)]
        if tokenizer.eos_token:
            generation_kwargs["stop"] = [tokenizer.eos_token]
            generation_kwargs["include_stop_str_in_output"] = False
        print(f"Explicit EOS/stop enabled: {generation_kwargs}")

    reward_fn = make_reward_fn(args.arch, workdir_abs, eval_gpu, args.reward_workers)

    gen_batch = max(args.batch_size, args.num_generations)
    gen_batch -= gen_batch % args.num_generations

    grpo_kwargs = dict(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        learning_rate=args.lr,
        temperature=args.temperature,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        report_to="none",
        remove_unused_columns=False,
        beta=args.beta,
    )

    if args.use_vllm:
        if generation_kwargs is None:
            generation_kwargs = {}
        stops = list(generation_kwargs.get("stop", []))
        if STRICT_OUTPUT_STOP_MARKER not in stops:
            stops.append(STRICT_OUTPUT_STOP_MARKER)
        generation_kwargs["stop"] = stops
        generation_kwargs["include_stop_str_in_output"] = False
        grpo_kwargs.update(
            use_vllm=True,
            vllm_mode="server",
            vllm_server_port=args.vllm_port,
            vllm_structured_outputs_regex=STRICT_ONE_FILE_OUTPUT_REGEX,
            vllm_importance_sampling_correction=False,
        )
        if generation_kwargs is not None:
            grpo_kwargs["generation_kwargs"] = generation_kwargs
        print(f"vLLM server mode: port={args.vllm_port}")
        print("Structured outputs enabled: strict one-file regex")
        print("vLLM importance sampling correction: disabled")
    else:
        grpo_kwargs.update(
            use_vllm=False,
            generation_batch_size=gen_batch,
        )

    config = GRPOConfig(**grpo_kwargs)

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
    mode = "vLLM server" if args.use_vllm else "local generation"
    print(f"GRPO: model={args.model}, arch={args.arch}, mode={mode}")
    print(f"  epochs={args.epochs}, steps/epoch={steps}, total={total}")
    print(f"  batch={args.batch_size}, grad_accum={args.gradient_accumulation}, num_gen={args.num_generations}")
    print(f"  max_len={args.max_completion_length}, temp={args.temperature}, workers={args.reward_workers}")
    print(f"  train_gpu={args.train_gpu}, eval_gpu={eval_gpu}")

    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"Training complete. Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
