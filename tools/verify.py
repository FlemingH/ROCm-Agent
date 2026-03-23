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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', default=os.environ.get('PYTORCH_ROCM_ARCH', 'gfx1201'))
    args = parser.parse_args()

    sys.path.insert(0, str(WORKDIR.resolve()))

    model_mod = load_module_from_file("model", WORKDIR / "model.py")
    Model = model_mod.Model
    get_inputs = model_mod.get_inputs
    get_init_inputs = model_mod.get_init_inputs

    model_new_mod = load_module_from_file("model_new", WORKDIR / args.arch / "model_new.py")
    ModelNew = model_new_mod.ModelNew

    init_inputs = get_init_inputs()
    if not isinstance(init_inputs, (list, tuple)):
        init_inputs = [init_inputs]

    torch_model = Model(*init_inputs).eval().cuda()
    hip_model = ModelNew(*init_inputs).eval().cuda()
    hip_model.load_state_dict(torch_model.state_dict())

    num_checks = 5
    with torch.no_grad():
        for i in range(num_checks):
            torch_inputs = get_inputs()
            if not isinstance(torch_inputs, (list, tuple)):
                torch_inputs = [torch_inputs]
            torch_inputs = transform_tensors(torch_inputs, lambda x: x.cuda())
            hip_inputs = transform_tensors(torch_inputs, lambda x: x.clone())

            torch_output = torch_model(*torch_inputs)
            with block_torch_functional():
                hip_output = hip_model(*hip_inputs)
            check_equal(hip_output, torch_output)
            print(f"[PASS] check {i + 1}/{num_checks}")

    torch.cuda.synchronize()
    print("[PASS] verify success")


if __name__ == "__main__":
    main()
