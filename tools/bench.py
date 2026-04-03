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


def _get_weight_tensors(model):
    """Extract float32 weight tensors from state_dict in deterministic order.

    Skips non-float tensors (e.g. BatchNorm's num_batches_tracked is int64).
    """
    return [v.contiguous() for v in model.state_dict().values()
            if v.is_floating_point()]


def _make_dynamic_model_new(Model, ext_name: str = "hip_extension"):
    """Create a ModelNew class that inherits Model and overrides forward().

    The new forward() extracts all float32 weight tensors from the model and
    passes them (along with forward inputs) to hip_extension.fused_kernel_forward().

    This ensures load_state_dict(strict=True) works because DynamicModelNew
    has the exact same parameter structure as Model.
    """
    class DynamicModelNew(Model):
        def forward(self, *args, **kwargs):
            import importlib
            hip_extension = importlib.import_module(ext_name)
            weights = _get_weight_tensors(self)
            input_tensors = [a.contiguous() for a in args
                             if isinstance(a, torch.Tensor)]
            return hip_extension.fused_kernel_forward(*input_tensors, *weights)

    DynamicModelNew.__name__ = "ModelNew"
    DynamicModelNew.__qualname__ = "ModelNew"
    return DynamicModelNew


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


def run(sandbox_dir: Path, arch: str, iters: int = 10, ext_name: str = "hip_extension") -> tuple[bool, str]:
    sys.path.insert(0, str(sandbox_dir.resolve()))
    try:
        model_mod = load_module_from_file("model", sandbox_dir / "model.py")
        Model, get_inputs, get_init_inputs = model_mod.Model, model_mod.get_inputs, model_mod.get_init_inputs

        model_new_path = sandbox_dir / arch / "model_new.py"
        if model_new_path.exists():
            model_new_mod = load_module_from_file("model_new", model_new_path)
            ModelNew = model_new_mod.ModelNew
        else:
            ModelNew = _make_dynamic_model_new(Model, ext_name)

        init_inputs = get_init_inputs()
        if not isinstance(init_inputs, (list, tuple)):
            init_inputs = [init_inputs]

        torch_model = Model(*init_inputs).eval().cuda()
        hip_model = ModelNew(*init_inputs).eval().cuda()
        hip_model.load_state_dict(torch_model.state_dict(), strict=False)

        torch_inputs = get_inputs()
        if not isinstance(torch_inputs, (list, tuple)):
            torch_inputs = [torch_inputs]
        torch_inputs = transform_tensors(torch_inputs, lambda x: x.cuda())
        hip_inputs = transform_tensors(torch_inputs, lambda x: x.clone())

        torch_compile_model = torch.compile(torch_model)

        hip_time = benchmark_model(hip_model, hip_inputs, 5, iters)
        torch_time = benchmark_model(torch_model, torch_inputs, 5, iters)
        compile_time = benchmark_model(torch_compile_model, torch_inputs, 5, iters)

        result_str = (
            f"Torch Baseline: {torch_time:.3f}us, "
            f"Torch Compile: {compile_time:.3f}us, "
            f"HIP Extension: {hip_time:.3f}us"
        )
        return True, result_str
    except Exception as e:
        import traceback
        return False, f"Benchmarking failed:\n{traceback.format_exc()}"
    finally:
        sys.path.remove(str(sandbox_dir.resolve()))
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', default=os.environ.get('PYTORCH_ROCM_ARCH', 'gfx1201'))
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument('--ext-name', default='hip_extension')
    args = parser.parse_args()

    ok, msg = run(WORKDIR, args.arch, args.iters, ext_name=args.ext_name)
    print(msg)
    if not ok:
        sys.exit(1)
