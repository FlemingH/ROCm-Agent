#!/usr/bin/env python3
"""Compile HIP/C++ sources into hip_extension.so for a target architecture.

Runs from project root. Sources come from:
  - agent_workdir/*.cpp (binding.cpp)
  - agent_workdir/<arch>/kernels/*.hip and *.cpp
Output goes to agent_workdir/hip_extension.so
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

import torch.utils.cpp_extension as cpp_ext

WORKDIR = Path('agent_workdir')


def find_sources(arch: str) -> list[str]:
    root_sources = [str(p) for p in WORKDIR.glob('*.cpp')]
    kernels_dir = WORKDIR / arch / 'kernels'
    kernel_sources = []
    if kernels_dir.is_dir():
        kernel_sources = (
            [str(p) for p in kernels_dir.glob('*.hip')]
            + [str(p) for p in kernels_dir.glob('*.cpp')]
        )
    return sorted(set(root_sources + kernel_sources))


def compile_kernels(arch: str) -> int:
    build_dir = WORKDIR / 'build' / arch
    output_so = WORKDIR / 'hip_extension.so'
    sources = find_sources(arch)

    if not sources:
        print(f'Error: no source files found in {WORKDIR}/*.cpp or {WORKDIR}/{arch}/kernels/')
        return 1

    print(f'[{arch}] Compiling {len(sources)} files: {", ".join(sources)}')

    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)

    if output_so.exists():
        output_so.unlink()

    extra_include = [str(WORKDIR.resolve())]

    try:
        cpp_ext.load(
            name='hip_extension',
            sources=sources,
            build_directory=str(build_dir),
            verbose=False,
            with_cuda=True,
            extra_cflags=['-O3', '-std=c++17'],
            extra_cuda_cflags=['-O3', '-ffast-math'],
            extra_include_paths=extra_include,
        )
    except Exception as exc:
        print('Compilation failed.')
        print(str(exc))
        return 1

    built_so = build_dir / 'hip_extension.so'
    if built_so.exists():
        shutil.copy2(built_so, output_so)
        print(f'Compile success: {output_so}')
        return 0

    so_files = list(build_dir.glob('hip_extension*.so'))
    if so_files:
        shutil.copy2(so_files[0], output_so)
        print(f'Compile success: {output_so}')
        return 0

    print('Compilation finished but hip_extension.so was not generated.')
    return 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', default=os.environ.get('PYTORCH_ROCM_ARCH', 'gfx1201'))
    args = parser.parse_args()
    return compile_kernels(args.arch)


if __name__ == '__main__':
    sys.exit(main())
