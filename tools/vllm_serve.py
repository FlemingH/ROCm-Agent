#!/usr/bin/env python3
"""Host-mode vLLM serve wrapper with spawn multiprocessing fix for ROCm."""
import multiprocessing
import sys

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    from trl import TrlParser
    from trl.scripts.vllm_serve import ScriptArguments, main
    (script_args,) = TrlParser((ScriptArguments,)).parse_args_and_config()
    main(script_args)
