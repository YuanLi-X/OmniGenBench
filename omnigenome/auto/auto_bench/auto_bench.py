# -*- coding: utf-8 -*-
# file: auto_bench.py
# time: 11:54 14/04/2024
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2024. All Rights Reserved.

import os
import time
import warnings

import findfile
import torch
from metric_visualizer import MetricVisualizer

from transformers import TrainingArguments, Trainer as HFTrainer
from ...src.abc.abstract_tokenizer import OmniGenomeTokenizer
from ...src.lora.lora_model import OmniLoraModel
from ...src.misc.utils import (
    seed_everything,
    fprint,
    load_module_from_path,
    check_bench_version,
    clean_temp_checkpoint,
)
from ...src.trainer.trainer import Trainer
from ...src.trainer.accelerate_trainer import AccelerateTrainer
from ...utility.hub_utils import download_benchmark
from ... import __version__ as omnigenome_version


class AutoBench:
    """
    A class for automatically benchmarking models on a given benchmark suite.

    Attributes:
        benchmark (str): Path to the benchmark directory.
        model_name_or_path (str): Model name or path.
        tokenizer (str or Tokenizer, optional): Tokenizer instance or path.
        autocast (str): Autocast mode, e.g., 'fp16'.
        overwrite (bool): Whether to overwrite existing metric visualizations.
        trainer (str): Trainer type, e.g., 'native', 'hf_trainer', or 'accelerate'.
        mv_path (str): Path to save metric visualizer results.
        mv (MetricVisualizer): Metric visualizer instance.
        bench_metadata: Benchmark metadata module.
    """
    def __init__(
        self,
        benchmark,
        model_name_or_path,
        tokenizer=None,
        **kwargs,
    ):
        """
        Initialize the AutoBench class.

        Args:
            benchmark (str): Benchmark directory or name.
            model_name_or_path (str or model): Model name or preloaded model.
            tokenizer (str or tokenizer, optional): Tokenizer or tokenizer path.
            **kwargs: Additional options like 'autocast', 'overwrite', 'trainer'.
        """
        self.benchmark = benchmark.rstrip("/")
        self.autocast = kwargs.pop("autocast", "fp16")
        self.overwrite = kwargs.pop("overwrite", False)
        self.trainer = kwargs.pop("trainer", "native")

        self.model_name_or_path = model_name_or_path
        self.tokenizer = tokenizer
        if isinstance(self.model_name_or_path, str):
            self.model_name_or_path = self.model_name_or_path.rstrip("/")
            self.model_name = self.model_name_or_path.split("/")[-1]
        else:
            self.model_name = self.model_name_or_path.__class__.__name__
        if isinstance(tokenizer, str):
            self.tokenizer = tokenizer.rstrip("/")
        os.makedirs("./autobench_evaluations", exist_ok=True)
        time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        mv_name = f"{benchmark}-{self.model_name}"
        self.mv_path = f"./autobench_evaluations/{mv_name}-{time_str}.mv"

        mv_paths = findfile.find_files(
            "./autobench_evaluations",
            [benchmark, self.model_name, ".mv"],
        )
        if mv_paths and not self.overwrite:
            self.mv = MetricVisualizer.load(mv_paths[-1])
            self.mv.summary(round=4)
        else:
            self.mv = MetricVisualizer(self.mv_path)
        if not os.path.exists(self.benchmark):
            fprint(
                "Benchmark:",
                benchmark,
                "does not exist. Search online for available benchmarks.",
            )
            self.benchmark = download_benchmark(self.benchmark)

        # Import benchmark list
        self.bench_metadata = load_module_from_path(
            f"bench_metadata", f"{self.benchmark}/metadata.py"
        )
        check_bench_version(
            self.bench_metadata.__omnigenome_version__, omnigenome_version
        )
        fprint("Loaded benchmarks: ", self.bench_metadata.bench_list)
        self.bench_info()

    def bench_info(self):
        """
        Print benchmark configuration information.

        Returns:
            str: A string summarizing benchmark information.
        """
        ...
        info = f"Benchmark Root: {self.benchmark}\n"
        info += f"Benchmark List: {self.bench_metadata.bench_list}\n"
        info += f"Model Name or Path: {self.model_name}\n"
        info += f"Tokenizer: {self.tokenizer}\n"
        info += f"Metric Visualizer Path: {self.mv_path}\n"
        info += f"BenchConfig Details: {self.bench_metadata}\n"
        fprint(info)
        return info

    def run(self, **kwargs):
        """
        Run the benchmark evaluation across all tasks defined in the benchmark metadata.

        This includes:
        - Loading and overriding configuration for each task
        - Initializing model and tokenizer
        - Training using the specified trainer (native/hf_trainer/accelerate)
        - Evaluating and logging metrics to MetricVisualizer
        - Optionally applying LoRA modules
        - Saving predictions if configured

        Args:
            **kwargs: Optional overrides for benchmark config parameters. Can include
                      training hyperparameters, data options, or LoRA configuration.

        Raises:
            ValueError: If model_name_or_path is not provided.

        Returns:
            None. Metrics and predictions are saved to disk.
        """
        bs_scale = kwargs.pop("bs_scale", 1)
        # Import benchmark config
        for _, bench in enumerate(self.bench_metadata.bench_list):
            clean_temp_checkpoint(1)  # clean temp checkpoint older than 1 day
            fprint(
                ">" * 80,
                f"\nRunning evaluation for task: {bench}",
                "Progress: ",
                _ + 1,
                "/",
                len(self.bench_metadata.bench_list),
                f"{(_ + 1) * 100 / len(self.bench_metadata.bench_list)}%",
            )
            _kwargs = kwargs.copy()
            bench_config_path = findfile.find_file(
                self.benchmark, f"{self.benchmark}.{bench}.config".split(".")
            )
            config = load_module_from_path("config", bench_config_path)
            bench_config = config.bench_config
            fprint(f"Loaded config for {bench} from {bench_config_path}")
            fprint(bench_config)

            for key, value in _kwargs.items():
                if key in bench_config:
                    fprint(
                        "Override", key, "with", value, "according to the input kwargs"
                    )
                    bench_config.update({key: value})

                else:
                    warnings.warn(
                        f"kwarg: {key} not found in bench_config while setting {key} = {value}"
                    )
                    bench_config.update({key: value})

            for key, value in bench_config.items():
                if key in bench_config and key in _kwargs:
                    _kwargs.pop(key)

            fprint(
                f"AutoBench Config for {bench}:",
                "\n".join([f"{k}: {v}" for k, v in bench_config.items()]),
            )

            # Init Tokenizer and Model
            if not self.tokenizer:
                tokenizer = OmniGenomeTokenizer.from_pretrained(
                    self.model_name_or_path,
                    trust_remote_code=bench_config.get("trust_remote_code", True),
                    **bench_config,
                )
            else:
                tokenizer = self.tokenizer

            if not isinstance(bench_config["seeds"], list):
                bench_config["seeds"] = [bench_config["seeds"]]

            for seed in bench_config["seeds"]:
                batch_size = (
                    bench_config["batch_size"] if "batch_size" in bench_config else 8
                ) * bs_scale

                record_name = f"{self.benchmark}-{bench}-{self.model_name}".split("/")[
                    -1
                ]
                # check if the record exists
                if record_name in self.mv.transpose() and len(
                    list(self.mv.transpose()[record_name].values())[0]
                ) >= len(bench_config["seeds"]):
                    continue

                seed_everything(seed)
                if self.model_name_or_path:
                    model_cls = bench_config["model_cls"]
                    model = model_cls(
                        self.model_name_or_path,
                        tokenizer=tokenizer,
                        label2id=bench_config.label2id,
                        num_labels=bench_config["num_labels"],
                        trust_remote_code=True,
                        ignore_mismatched_sizes=True,
                    )
                else:
                    raise ValueError(
                        "model_name_or_path is not specified. Please provide a valid model name or path."
                    )

                fprint(f"\n{model}")

                if kwargs.get("lora_config", None) is not None:
                    fprint("Applying LoRA to the model with config:", kwargs["lora_config"])
                    model = OmniLoraModel(model, **kwargs.get("lora_config", {}))

                # Init Trainer
                dataset_cls = bench_config["dataset_cls"]

                if hasattr(model.config, "max_position_embeddings"):
                    max_length = min(
                        bench_config["max_length"],
                        model.config.max_position_embeddings,
                    )
                else:
                    max_length = bench_config["max_length"]

                train_set = dataset_cls(
                    data_source=bench_config["train_file"],
                    tokenizer=tokenizer,
                    label2id=bench_config["label2id"],
                    max_length=max_length,
                    structure_in=bench_config.get("structure_in", False),
                    max_examples=bench_config.get("max_examples", None),
                    shuffle=bench_config.get("shuffle", True),
                    drop_long_seq=bench_config.get("drop_long_seq", False),
                    **_kwargs,
                )
                test_set = dataset_cls(
                    data_source=bench_config["test_file"],
                    tokenizer=tokenizer,
                    label2id=bench_config["label2id"],
                    max_length=max_length,
                    structure_in=bench_config.get("structure_in", False),
                    max_examples=bench_config.get("max_examples", None),
                    shuffle=False,
                    drop_long_seq=bench_config.get("drop_long_seq", False),
                    **_kwargs,
                )
                valid_set = dataset_cls(
                    data_source=bench_config["valid_file"],
                    tokenizer=tokenizer,
                    label2id=bench_config["label2id"],
                    max_length=max_length,
                    structure_in=bench_config.get("structure_in", False),
                    max_examples=bench_config.get("max_examples", None),
                    shuffle=False,
                    drop_long_seq=bench_config.get("drop_long_seq", False),
                    **_kwargs,
                )

                if self.trainer == "hf_trainer":
                    # Set up HuggingFace Trainer
                    hf_kwargs = {
                        k: v
                        for k, v in kwargs.items()
                        if hasattr(TrainingArguments, k) and k != "output_dir"
                    }
                    training_args = TrainingArguments(
                        output_dir=f"./autobench_evaluations/{self.model_name}-{bench}",
                        num_train_epochs=hf_kwargs.pop(
                            "num_train_epochs", bench_config["epochs"]
                        ),
                        per_device_train_batch_size=hf_kwargs.pop(
                            "batch_size", batch_size
                        ),
                        per_device_eval_batch_size=hf_kwargs.pop(
                            "batch_size", batch_size
                        ),
                        gradient_accumulation_steps=hf_kwargs.pop(
                            "gradient_accumulation_steps", 1
                        ),
                        learning_rate=hf_kwargs.pop("learning_rate", 2e-5),
                        weight_decay=hf_kwargs.pop("weight_decay", 0),
                        eval_strategy=hf_kwargs.pop("eval_strategy", "epoch"),
                        save_strategy=hf_kwargs.pop("save_strategy", "epoch"),
                        fp16=hf_kwargs.pop("fp16", True),
                        remove_unused_columns=False,
                        label_names=["labels"],
                        **hf_kwargs,
                    )

                    valid_set = valid_set if len(valid_set) else test_set

                    if len(bench_config["compute_metrics"]) > 1:
                        fprint(
                            "Multiple metrics not supported by HFTrainer, using the first one metric only."
                        )
                    trainer = HFTrainer(
                        model=model,
                        args=training_args,
                        train_dataset=train_set,
                        eval_dataset=valid_set,
                        compute_metrics=(
                            bench_config["compute_metrics"][0]
                            if isinstance(bench_config["compute_metrics"], list)
                            else bench_config["compute_metrics"]
                        ),
                    )

                    # Train and evaluate
                    eval_result = trainer.evaluate(
                        valid_set if len(valid_set) else test_set
                    )
                    print(eval_result)
                    train_result = trainer.train()
                    eval_result = trainer.evaluate()
                    test_result = trainer.evaluate(
                        test_set if len(test_set) else valid_set
                    )

                    metrics = {
                        "train": train_result.metrics,
                        "eval": eval_result,
                        "test": test_result,
                    }
                    fprint(metrics)
                else:
                    optimizer = torch.optim.AdamW(
                        filter(lambda p: p.requires_grad, model.parameters()),
                        lr=(
                            bench_config["learning_rate"]
                            if "learning_rate" in bench_config
                            else 2e-5
                        ),
                        weight_decay=(
                            bench_config["weight_decay"]
                            if "weight_decay" in bench_config
                            else 0
                        ),
                    )
                    if self.trainer == "accelerate":
                        trainer_cls = AccelerateTrainer
                    else:
                        trainer_cls = Trainer
                    fprint(f"Using Trainer: {trainer_cls}")
                    trainer = trainer_cls(
                        model=model,
                        train_dataset=train_set,
                        eval_dataset=valid_set,
                        test_dataset=test_set,
                        batch_size=batch_size,
                        patience=(
                            bench_config["patience"]
                            if "patience" in bench_config
                            else 3
                        ),
                        epochs=bench_config["epochs"],
                        gradient_accumulation_steps=bench_config.get(
                            "gradient_accumulation_steps", 1
                        ),
                        optimizer=optimizer,
                        loss_fn=(
                            bench_config["loss_fn"]
                            if "loss_fn" in bench_config
                            else None
                        ),
                        compute_metrics=bench_config["compute_metrics"],
                        seed=seed,
                        autocast=self.autocast,
                        **_kwargs,
                    )
                    metrics = trainer.train()

                    predictions = trainer.predictions

                    if bench_config.get("save_predictions", False):
                        os.makedirs(f"predictions/{bench}", exist_ok=True)
                        import numpy as np

                        for split in predictions.keys():
                            with open(
                                f"predictions/{bench}/{split}.npy",
                                "wb",
                            ) as f:
                                np.save(f, predictions[split])

                    if metrics:
                        for key, value in metrics["test"][-1].items():
                            try:
                                value = float(value)
                            except:
                                pass  # ignore non-float values
                            self.mv.log(f"{record_name}", f"{key}", value)
                        # for key, value in metrics['test'][-1].items():
                        #     self.mv.log(f'{record_name}', f'test_{key}', value)
                        # for i, valid_metrics in enumerate(metrics["valid"]):
                        #     for key, value in valid_metrics.items():
                        #         self.mv.log(f'{record_name}', f'valid_epoch_{i}_{key}', value)

                        self.mv.summary(round=4)
                        self.mv.dump(self.mv_path)
                        self.mv.to_csv(self.mv_path.replace(".mv", ".csv"))
                    del model, trainer, optimizer
                    torch.cuda.empty_cache()
