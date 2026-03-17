#!/usr/bin/env python3
"""Performance profiling: Eager vs torch.compile vs HIP extension."""

import argparse
import importlib
import os
import sys
import torch
from pathlib import Path

WORKDIR = Path('agent_workdir')


def transform_tensors(tensors, fn):
    if isinstance(tensors, torch.Tensor):
        return fn(tensors)
    if isinstance(tensors, (list, tuple)):
        return [transform_tensors(x, fn) for x in tensors]
    if isinstance(tensors, dict):
        return {k: transform_tensors(v, fn) for k, v in tensors.items()}
    return tensors


def load_module_from_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def benchmark_model(model, inputs, warmup_iters, run_iters):
    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = model(*inputs)

        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA],
            record_shapes=False, with_stack=False,
        ) as ctx:
            torch.cuda.synchronize()
            for _ in range(run_iters):
                _ = model(*inputs)
            torch.cuda.synchronize()

    return (
        sum(e.device_time for e in ctx.events() if e.device_type.name == "CUDA")
        / run_iters
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', default=os.environ.get('PYTORCH_ROCM_ARCH', 'gfx1201'))
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args()

    sys.path.insert(0, str(WORKDIR.resolve()))

    model_mod = load_module_from_file("model", WORKDIR / "model.py")
    Model, get_inputs, get_init_inputs = model_mod.Model, model_mod.get_inputs, model_mod.get_init_inputs

    model_new_mod = load_module_from_file("model_new", WORKDIR / args.arch / "model_new.py")
    ModelNew = model_new_mod.ModelNew

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

    torch_compile_model = torch.compile(torch_model)

    hip_time = benchmark_model(hip_model, hip_inputs, 5, args.iters)
    torch_time = benchmark_model(torch_model, torch_inputs, 5, args.iters)
    compile_time = benchmark_model(torch_compile_model, torch_inputs, 5, args.iters)

    print(
        f"Torch Baseline: {torch_time:.3f}us, "
        f"Torch Compile: {compile_time:.3f}us, "
        f"HIP Extension: {hip_time:.3f}us"
    )


if __name__ == "__main__":
    main()
