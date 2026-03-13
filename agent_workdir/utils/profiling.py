#!/usr/bin/env python3
"""Performance profiling: Eager vs torch.compile vs HIP extension."""

import argparse
import importlib
import os
import sys
import torch
from pathlib import Path


def transform_tensors(tensors, fn):
    if isinstance(tensors, torch.Tensor):
        return fn(tensors)
    if isinstance(tensors, (list, tuple)):
        return [transform_tensors(x, fn) for x in tensors]
    if isinstance(tensors, dict):
        return {k: transform_tensors(v, fn) for k, v in tensors.items()}
    return tensors


def get_prof_ctx():
    return torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False,
        with_stack=False,
    )


def load_arch_module(arch: str):
    arch_path = Path(arch)
    if str(arch_path) not in sys.path:
        sys.path.insert(0, str(arch_path))
    spec = importlib.util.spec_from_file_location("model_new", arch_path / "model_new.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.ModelNew


def benchmark_model(model, inputs, warmup_iters, run_iters):
    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = model(*inputs)

        with get_prof_ctx() as ctx:
            torch.cuda.synchronize()
            for _ in range(run_iters):
                _ = model(*inputs)
            torch.cuda.synchronize()

    return (
        sum(e.device_time for e in ctx.events() if e.device_type.name == "CUDA")
        / run_iters
    )


def main():
    parser = argparse.ArgumentParser(
        description="Profile hip_extension vs torch baseline and torch.compile."
    )
    parser.add_argument('--arch', default=os.environ.get('PYTORCH_ROCM_ARCH', 'gfx1100'))
    parser.add_argument("--iters", type=int, default=10, help="Benchmark iterations")
    parser.add_argument("--single-run", type=str,
        help="Run once for targets: torch_baseline,torch_compile,hip_extension")
    args = parser.parse_args()

    from model import Model, get_inputs, get_init_inputs
    ModelNew = load_arch_module(args.arch)

    init_inputs = get_init_inputs()
    if not isinstance(init_inputs, (list, tuple)):
        init_inputs = [init_inputs]

    torch_model = Model(*init_inputs).eval().cuda()
    hip_model = ModelNew(*init_inputs).eval().cuda()
    hip_model.load_state_dict(torch_model.state_dict())

    torch_inputs = get_inputs()
    if not isinstance(torch_inputs, (list, tuple)):
        torch_inputs = [torch_inputs]
    torch_inputs = transform_tensors(torch_inputs, lambda x: x.cuda())
    hip_inputs = transform_tensors(torch_inputs, lambda x: x.clone())

    if args.single_run:
        targets = [x.strip() for x in args.single_run.split(",") if x.strip()]
        with torch.no_grad():
            if "torch_baseline" in targets:
                _ = torch_model(*torch_inputs)
            if "torch_compile" in targets:
                _ = torch.compile(torch_model)(*torch_inputs)
            if "hip_extension" in targets:
                _ = hip_model(*hip_inputs)
        print("[DONE] single-run completed")
        return

    torch_compile_model = torch.compile(torch_model)
    warmup_iters = 5
    run_iters = args.iters

    hip_time = benchmark_model(hip_model, hip_inputs, warmup_iters, run_iters)
    torch_time = benchmark_model(torch_model, torch_inputs, warmup_iters, run_iters)
    compile_time = benchmark_model(torch_compile_model, torch_inputs, warmup_iters, run_iters)

    print(
        f"Torch Baseline: {torch_time:.3f}us, "
        f"Torch Compile: {compile_time:.3f}us, "
        f"HIP Extension: {hip_time:.3f}us"
    )


if __name__ == "__main__":
    main()
