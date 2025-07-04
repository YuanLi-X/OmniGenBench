# -*- coding: utf-8 -*-
# file: auto_bench_cli.py
# time: 21:06 31/01/2025
# author: YANG, HENG <hy345@exeter.ac.uk> (Yang Heng)
# Homepage: https://yangheng95.github.io
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2025. All Rights Reserved.
import argparse
import os
import platform
import sys
import time
from pathlib import Path

from ....auto.auto_bench.auto_bench import AutoBench
from ....src.misc.utils import fprint
from ..base import BaseCommand


class BenchCommand(BaseCommand):
    """
    Command class to handle the 'autobench' sub-command for running genomic model benchmarks.

    Provides command-line argument parsing, special handling for specific model types,
    initialization and execution of the AutoBench benchmarking process,
    and logs management for benchmark runs.
    """
    @classmethod
    def register_command(cls, subparsers):
        """
        Register the 'autobench' command and its arguments to the CLI subparsers.

        Args:
            subparsers (argparse._SubParsersAction): Subparsers object from argparse
                to which the 'autobench' command is added.

        Sets up arguments including:
            - benchmark root directory choice,
            - tokenizer path,
            - model path (required),
            - overwrite flag,
            - batch size scale,
            - trainer backend selection,
            plus additional common arguments via `add_common_arguments`.
        """
        parser = subparsers.add_parser(
            "autobench",
            help="Run Auto-benchmarking for Genomic Foundation Models.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        # Required argument
        parser.add_argument(
            "-b",
            "--benchmark",
            type=str,
            default="RGB",
            choices=["RGB", "PGB", "GUE", "GB", "BEACON"],
            help="Path to the BEACON benchmark root directory.",
        )
        parser.add_argument(
            "-t",
            "--tokenizer",
            type=str,
            default=None,
            help="Path to the tokenizer to use (HF tokenizer ID or local path).",
        )

        parser.add_argument(
            "-m",
            "--model",
            type=str,
            required=True,
            help="Path to the model to evaluate (HF model ID or local path).",
        )

        # Optional arguments
        parser.add_argument(
            "--overwrite",
            type=bool,
            default=False,
            help="Overwrite existing bench results, otherwise resume from benchmark checkpoint.",
        )
        parser.add_argument(
            "--bs_scale",
            type=int,
            default=1,
            help="Batch size scale factor. To increase GPU memory utilization, set to 2 or 4, etc.",
        )
        parser.add_argument(
            "--trainer",
            type=str,
            default="accelerate",
            choices=["native", "accelerate", "hf_trainer"],
            help="Trainer to use for training. \n"
            "Use 'accelerate' for distributed training. Set to false to disable. "
            "You can use 'accelerate config' to customize behavior.\n"
            "Use 'hf_trainer' for Hugging Face Trainer. \n"
            "Set to 'native' to use native PyTorch training loop.\n",
        )

        cls.add_common_arguments(parser)
        parser.set_defaults(func=cls.execute)

    @staticmethod
    def execute(args: argparse.Namespace):
        """
        Execute the autobench command with the provided arguments.

        - Prints user info messages about logs and usage.
        - Handles special loading for 'multimolecule' models.
        - Creates and runs the AutoBench instance with given parameters.
        - Manages log directory and file naming.
        - Constructs and runs the benchmark command line, redirecting output to logs,
          with platform-specific handling for Windows and Unix-like systems.

        Args:
            args (argparse.Namespace): Parsed command-line arguments.
        """
        fprint("Running benchmark, this may take a while, please be patient...")
        fprint("You can find the logs in the 'autobench_logs' directory.")
        fprint("You can find the metrics in the 'autobench_evaluations' directory.")
        fprint(
            "If you don't intend to use accelerate, please add '--use_accelerate false' to the command."
        )
        fprint(
            "If you want to alter accelerate's behavior, please refer to 'accelerate config' command."
        )
        fprint(
            "If you encounter any issues, please report them on the GitHub repository."
        )
        # 特殊模型处理
        if "multimolecule" in args.model:
            from multimolecule import RnaTokenizer, AutoModelForTokenPrediction

            tokenizer = RnaTokenizer.from_pretrained(args.model)
            model = AutoModelForTokenPrediction.from_pretrained(
                args.model, trust_remote_code=True
            ).base_model
        else:
            tokenizer = args.tokenizer
            model = args.model

        autobench = AutoBench(
            benchmark=args.benchmark,
            model_name_or_path=model,
            tokenizer=tokenizer,
            overwrite=args.overwrite,
            trainer=args.trainer,
        )
        autobench.run(**vars(args))
        log_dir = Path(args.output_dir) / "autobench_evaluations"
        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        log_file = log_dir / f"bench_{args.benchmark}_{timestamp}.log"

        cmd_base = f"{sys.executable} -m omnigenome_cli.bench_internal " + " ".join(
            f"--{k}={v}" if v is not None else f"--{k}"
            for k, v in vars(args).items()
            if k not in {"func", "output_dir", "log_level"}
        )

        if platform.system() == "Windows":
            return f"{cmd_base} 2>&1 | powershell -Command \"tee-object -FilePath '{log_file}'\""
        os.system(f"{cmd_base} 2>&1 | tee {log_file}")


def register_command(subparsers):
    """
    Register the BenchCommand with the CLI subparsers.

    Args:
        subparsers (argparse._SubParsersAction): Subparsers object to register the command.
    """
    BenchCommand.register_command(subparsers)
