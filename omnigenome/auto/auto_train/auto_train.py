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

from ...src.lora.lora_model import OmniLoraModel
from ...src.abc.abstract_tokenizer import OmniGenomeTokenizer
from ...src.misc.utils import (
    seed_everything,
    fprint,
    load_module_from_path,
    clean_temp_checkpoint,
)
from ...src.trainer.accelerate_trainer import AccelerateTrainer
from ...src.trainer.trainer import Trainer

autotrain_evaluations = "./autotrain_evaluations"


class AutoTrain:
    """
    Automated training and evaluation handler for genomic foundation models.

    This class manages dataset loading, model/tokenizer initialization,
    training with different trainer backends (accelerate, HFTrainer, native),
    metric logging, and saving results for benchmarking.

    Attributes:
        dataset (str): Dataset root directory or name.
        model_name_or_path (str or model object): Model identifier or path.
        tokenizer (str or tokenizer object): Tokenizer identifier or object.
        autocast (str): Mixed precision mode (e.g., 'fp16').
        overwrite (bool): Whether to overwrite existing results.
        trainer (str): Trainer backend to use ('accelerate', 'hf_trainer', 'native').
        model_name (str): Extracted model name from path or object.
        mv (MetricVisualizer): Metric visualizer instance for logging and plotting.
        mv_path (str): File path to save metric visualizer data.
    """
    def __init__(
        self,
        dataset,
        model_name_or_path,
        tokenizer=None,
        **kwargs,
    ):
        """
        Initialize AutoTrain instance.

        Args:
            dataset (str): Dataset directory or name.
            model_name_or_path (str or model): Model path or model object.
            tokenizer (optional, str or tokenizer): Tokenizer path or object.
            **kwargs: Additional optional parameters:
                - autocast (str): Mixed precision mode, default 'fp16'.
                - overwrite (bool): Overwrite existing evaluation results.
                - trainer (str): Trainer type to use.
        """
        self.dataset = dataset.rstrip("/")
        self.autocast = kwargs.pop("autocast", "fp16")
        self.overwrite = kwargs.pop("overwrite", False)
        self.trainer = kwargs.pop("trainer", "accelerate")

        self.model_name_or_path = model_name_or_path
        self.tokenizer = tokenizer
        if isinstance(self.model_name_or_path, str):
            self.model_name_or_path = self.model_name_or_path.rstrip("/")
            self.model_name = self.model_name_or_path.split("/")[-1]
        else:
            self.model_name = self.model_name_or_path.__class__.__name__
        if isinstance(tokenizer, str):
            self.tokenizer = tokenizer.rstrip("/")
        os.makedirs(autotrain_evaluations, exist_ok=True)
        time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        mv_name = f"{dataset}-{self.model_name}"
        self.mv_path = f"{autotrain_evaluations}/{mv_name}-{time_str}.mv"

        mv_paths = findfile.find_files(
            autotrain_evaluations,
            [dataset, self.model_name, ".mv"],
        )
        if mv_paths and not self.overwrite:
            self.mv = MetricVisualizer.load(mv_paths[-1])
            self.mv.summary(round=4)
        else:
            self.mv = MetricVisualizer(self.mv_path)
        self.bench_info()

    def bench_info(self):
        """
        Print and return benchmark info summary.

        Includes dataset path, model name, tokenizer info, and metric visualizer path.

        Returns:
            str: Formatted information string.
        """
        info = f"Dataset Root: {self.dataset}\n"
        info += f"Model Name or Path: {self.model_name}\n"
        info += f"Tokenizer: {self.tokenizer}\n"
        info += f"Metric Visualizer Path: {self.mv_path}\n"
        fprint(info)
        return info

    def run(self, **kwargs):
        """
        Run training and evaluation based on dataset config and input overrides.

        Loads dataset configuration, applies any parameter overrides passed in kwargs,
        initializes tokenizer and model, supports LoRA application,
        sets up training and validation datasets,
        selects trainer backend and runs training loops,
        logs metrics with MetricVisualizer, and saves outputs.

        Args:
            **kwargs: Configuration parameters that override dataset config values,
                      such as learning rate, batch size, epochs, lora_config, etc.

        Raises:
            ValueError: If model_name_or_path is not provided.

        Returns:
            None
        """
        """

        :param kwargs: parameters in kwargs will be used to override the default parameters in the dataset config
        :return:
        """

        clean_temp_checkpoint(1)  # clean temp checkpoint older than 1 day

        _kwargs = kwargs.copy()
        bench_config_path = findfile.find_file(
            self.dataset, f"{self.dataset}.config".split(".")
        )
        config = load_module_from_path("config", bench_config_path)
        bench_config = config.bench_config
        fprint(f"Loaded config for {self.dataset} from {bench_config_path}")
        fprint(bench_config)

        for key, value in _kwargs.items():
            if key in bench_config:
                fprint("Override", key, "with", value, "according to the input kwargs")
                bench_config.update({key: value})

            else:
                warnings.warn(
                    f"kwarg: {key} not found in config while setting {key} = {value}"
                )
                bench_config.update({key: value})

        for key, value in bench_config.items():
            if key in bench_config and key in _kwargs:
                _kwargs.pop(key)

        fprint(
            f"AutoTrain Config for {self.dataset}:",
            "\n".join([f"{k}: {v}" for k, v in bench_config.items()]),
        )

        # Init Tokenizer and Model
        if not self.tokenizer:
            tokenizer = OmniGenomeTokenizer.from_pretrained(
                self.model_name_or_path, trust_remote_code=True
            )
        else:
            tokenizer = self.tokenizer

        if not isinstance(bench_config["seeds"], list):
            bench_config["seeds"] = [bench_config["seeds"]]

        for seed in bench_config["seeds"]:
            batch_size = (
                bench_config["batch_size"] if "batch_size" in bench_config else 8
            )

            record_name = f"{os.path.basename(self.dataset)}-{self.model_name}".split(
                "/"
            )[-1]
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
                    output_dir=f"./autotrain_evaluations/{self.model_name}",
                    num_train_epochs=hf_kwargs.pop(
                        "num_train_epochs", bench_config["epochs"]
                    ),
                    per_device_train_batch_size=hf_kwargs.pop("batch_size", batch_size),
                    per_device_eval_batch_size=hf_kwargs.pop("batch_size", batch_size),
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
                test_result = trainer.evaluate(test_set if len(test_set) else valid_set)

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
                        bench_config["patience"] if "patience" in bench_config else 3
                    ),
                    epochs=bench_config["epochs"],
                    gradient_accumulation_steps=bench_config.get(
                        "gradient_accumulation_steps", 1
                    ),
                    optimizer=optimizer,
                    loss_fn=(
                        bench_config["loss_fn"] if "loss_fn" in bench_config else None
                    ),
                    compute_metrics=bench_config["compute_metrics"],
                    seed=seed,
                    autocast=self.autocast,
                    **_kwargs,
                )
                metrics = trainer.train()
                save_path = os.path.join(
                    autotrain_evaluations, self.dataset, self.model_name
                )
                trainer.save_model(save_path)

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
