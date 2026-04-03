#!/usr/bin/env python3
"""Correctness verification: compare Model (baseline) vs ModelNew (HIP extension).

Runs from project root. Loads model.py from agent_workdir/ and model_new.py
from agent_workdir/<arch>/.
"""

import argparse
import importlib
import os
import sys
import torch
import torch.nn.functional as F
from contextlib import contextmanager
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


def compute_mse(actual, expected):
    if isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor):
        if actual.is_floating_point() and expected.is_floating_point():
            return F.mse_loss(actual, expected).item()
        else:
            return (actual.float() - expected.float()).pow(2).mean().item()
    return 1e9


def check_equal(actual, expected):
    assert type(actual) == type(expected), f"{type(actual)=} != {type(expected)=}"
    if isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected), f"{len(actual)=} != {len(expected)=}"
        for x, y in zip(actual, expected):
            check_equal(x, y)
    elif isinstance(actual, dict):
        for key, val in expected.items():
            assert key in actual, f"Missing key in output: {key}"
            check_equal(actual[key], val)
    elif isinstance(actual, (str, float, int)):
        assert actual == expected, f"{actual=} != {expected=}"
    elif isinstance(actual, torch.Tensor):
        torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)
    else:
        raise TypeError(f"Unsupported output type: {type(actual)}")


@contextmanager
def block_torch_functional(excludes=None):
    if excludes is None:
        excludes = set()
    originals = {}
    for name in dir(F):
        attr = getattr(F, name)
        if callable(attr) and not name.startswith("_") and name not in excludes:
            originals[name] = attr
            def wrapper(*args, __name=name, **kwargs):
                raise RuntimeError(
                    f"Function torch.nn.functional.{__name} is not allowed in this context."
                )
            setattr(F, name, wrapper)
    try:
        yield
    finally:
        for name, attr in originals.items():
            setattr(F, name, attr)


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


import io
import contextlib

def run(sandbox_dir: Path, arch: str, ext_name: str = "hip_extension") -> tuple[bool, str]:
    sys.path.insert(0, str(sandbox_dir.resolve()))
    try:
        model_mod = load_module_from_file("model", sandbox_dir / "model.py")
        Model = model_mod.Model
        get_inputs = model_mod.get_inputs
        get_init_inputs = model_mod.get_init_inputs

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

        num_checks = 5
        pass_count = 0
        shape_ok_count = 0
        total_mse = 0.0
        
        output_lines = []
        with torch.no_grad():
            for i in range(num_checks):
                torch_inputs = get_inputs()
                if not isinstance(torch_inputs, (list, tuple)):
                    torch_inputs = [torch_inputs]
                torch_inputs = transform_tensors(torch_inputs, lambda x: x.cuda())
                hip_inputs = transform_tensors(torch_inputs, lambda x: x.clone())

                torch_output = torch_model(*torch_inputs)
                try:
                    with block_torch_functional():
                        hip_output = hip_model(*hip_inputs)
                except Exception as e:
                    output_lines.append(f"[FAIL] check {i + 1}/{num_checks}: runtime error: {e}")
                    continue

                shape_match = False
                if isinstance(hip_output, torch.Tensor) and isinstance(torch_output, torch.Tensor):
                    shape_match = hip_output.shape == torch_output.shape and hip_output.dtype == torch_output.dtype
                elif type(hip_output) == type(torch_output):
                    shape_match = True
                if shape_match:
                    shape_ok_count += 1

                try:
                    check_equal(hip_output, torch_output)
                    pass_count += 1
                    output_lines.append(f"[PASS] check {i + 1}/{num_checks}")
                except Exception as e:
                    if shape_match:
                        mse = compute_mse(hip_output, torch_output)
                        total_mse += mse
                        output_lines.append(f"[SHAPE_OK] check {i + 1}/{num_checks}: values differ: MSE={mse}")
                    else:
                        output_lines.append(f"[FAIL] check {i + 1}/{num_checks}: {e}")

        torch.cuda.synchronize()
        output_str = "\n".join(output_lines)
        if pass_count == num_checks:
            return True, output_str + "\n[PASS] verify success"
        elif pass_count > 0:
            return False, output_str + f"\n[PARTIAL_PASS] {pass_count}/{num_checks} passed, {shape_ok_count}/{num_checks} shape-correct"
        elif shape_ok_count > 0:
            avg_mse = total_mse / shape_ok_count
            return False, output_str + f"\n[SHAPE_OK] {shape_ok_count}/{num_checks} shape-correct, AVG_MSE={avg_mse}"
        else:
            return False, output_str + "\n[FAIL] verify failed"
    except Exception as e:
        import traceback
        return False, f"Verification script error:\n{traceback.format_exc()}"
    finally:
        sys.path.remove(str(sandbox_dir.resolve()))
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', default=os.environ.get('PYTORCH_ROCM_ARCH', 'gfx1201'))
    parser.add_argument('--ext-name', default='hip_extension')
    args = parser.parse_args()

    ok, msg = run(WORKDIR, args.arch, ext_name=args.ext_name)
    print(msg)
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
