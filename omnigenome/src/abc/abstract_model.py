# -*- coding: utf-8 -*-
# file: omnigenome_model.py
# time: 18:36 06/04/2024
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2024. All Rights Reserved.
import json
import os
import shutil
import warnings
import inspect
from importlib import import_module

import dill
import findfile
import torch
from transformers import AutoModel, AutoConfig, AutoTokenizer, BatchEncoding

from ..misc.utils import fprint, env_meta_info

warnings.filterwarnings("once")


def count_parameters(model):
    """
    Count the number of trainable parameters in the model.

    Args:
        model (torch.nn.Module): The PyTorch model.

    Returns:
        int: Number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class OmniGenomeModel(torch.nn.Module):
    """
    A general PyTorch model wrapper for OmniGenome that supports flexible
    model loading, saving, forward pass, loss computation, and inference.

    This class integrates HuggingFace transformers models with additional
    features such as dynamic config loading, tokenizer integration, and
    metadata management.

    Attributes:
        model (torch.nn.Module): The core transformer model.
        tokenizer: Tokenizer associated with the model.
        loss_fn: Optional loss function for training.
        config: Model configuration object.
        metadata (dict): Environment and model metadata.
    """
    def __init__(self, config_or_model_model, tokenizer, *args, **kwargs):
        """
        Initialize the OmniGenomeModel.

        Args:
            config_or_model_model (str, torch.nn.Module, or AutoConfig):
                Model identifier or pre-built model or config.
            tokenizer: Tokenizer object.
            *args, **kwargs: Additional arguments and keyword arguments.

        Keyword Args:
            label2id (dict, optional): Mapping from label names to IDs.
            trust_remote_code (bool, optional): Whether to trust remote code.
            num_labels (int, optional): Number of classification labels.
            ignore_mismatched_sizes (bool, optional): Allow size mismatch when loading.
            dropout (float, optional): Dropout rate.
        """
        self.loss_fn = None

        label2id = kwargs.pop("label2id", None)
        trust_remote_code = kwargs.pop("trust_remote_code", True)
        num_labels = kwargs.pop("num_labels", None)
        ignore_mismatched_sizes = kwargs.pop("ignore_mismatched_sizes", False)

        if label2id is not None and num_labels is None:
            num_labels = len(label2id)
        elif num_labels is not None and label2id is None:
            label2id = {str(i): i for i in range(num_labels)}

        # do not change the order of the following lines
        super().__init__(*args, **kwargs)

        if isinstance(config_or_model_model, str):
            config = AutoConfig.from_pretrained(
                config_or_model_model,
                num_labels=num_labels,
                label2id=label2id,
                trust_remote_code=trust_remote_code,
            )
            # Load the model from either `architectures` or `auto_map`
            if hasattr(config, "auto_map") and config.auto_map:
                architectures = list(set(config.auto_map.keys()) - set(["AutoConfig"]))
                if architectures:
                    model_cls_name = (
                        "AutoModel"
                        if "AutoModel" in architectures
                        else architectures[-1]
                    )
                    model_cls = getattr(import_module(f"transformers"), model_cls_name)

                    model = model_cls.from_pretrained(
                        config_or_model_model,
                        config=config,
                        trust_remote_code=trust_remote_code,
                        ignore_mismatched_sizes=ignore_mismatched_sizes,
                    ).base_model
                else:
                    raise ValueError(
                        f"The model cannot be instantiated from {config_or_model_model}. "
                        f"Please check the model configuration contains the architectures or auto_map."
                    )
            elif hasattr(config, "architectures") and config.architectures:
                model_cls_name = (
                    AutoModel
                    if "AutoModel" in config.architectures
                    else config.architectures[-1]
                )
                model_cls = getattr(import_module(f"transformers"), model_cls_name)
                model = model_cls.from_pretrained(
                    config_or_model_model,
                    config=config,
                    trust_remote_code=trust_remote_code,
                    ignore_mismatched_sizes=ignore_mismatched_sizes,
                ).base_model
            else:
                raise ValueError(
                    "Neither `architectures` nor `auto_map` is defined in the config."
                )
            self.model = model
            self.model.config = config
            del model_cls
        elif isinstance(config_or_model_model, torch.nn.Module):
            self.model = config_or_model_model
            self.model.config.num_labels = (
                num_labels if len(label2id) == num_labels else len(label2id)
            )
            self.model.config.label2id = label2id
        elif isinstance(config_or_model_model, AutoConfig):
            config = config_or_model_model
            config.num_labels = (
                num_labels if len(label2id) == num_labels else len(label2id)
            )
            config.label2id = label2id
            self.model = AutoModel.from_config(config)
            self.model.config = config
        else:
            raise ValueError(
                "The config_or_model_model should be either a string, a torch.nn.Module or a AutoConfig object."
            )

        # Update the config
        self.config = self.model.config
        if isinstance(label2id, dict):
            self.config.label2id = label2id
            self.config.id2label = {v: k for k, v in label2id.items()}
        if (
            not hasattr(self.config, "num_labels")
            or len(self.config.id2label) != self.config.num_labels
        ):
            fprint(
                "Warning: The number of labels in the config is not equal to the number of labels in the label2id dictionary. "
            )
            fprint(
                "Please check the label2id dictionary and the num_labels parameter in the config."
            )
            self.config.num_labels = len(self.config.id2label)

        # The metadata of the model
        self.metadata = env_meta_info()
        self.metadata["model_cls"] = self.__class__.__name__

        # The config of the model
        if hasattr(self.config, "n_embd") and self.config.n_embd:
            self.config.hidden_size = self.config.n_embd
        elif hasattr(self.config, "d_model") and self.config.d_model:
            self.config.hidden_size = self.config.d_model
        elif hasattr(self.config, "hidden_size") and self.config.hidden_size:
            self.config.hidden_size = self.config.hidden_size
        else:
            raise RuntimeError(
                "The hidden size of the model is not found in the config."
            )

        # The tokenizer of the model
        self.tokenizer = tokenizer
        self.metadata["tokenizer_cls"] = self.tokenizer.__class__.__name__
        if hasattr(self.tokenizer, "base_tokenizer"):
            self.pad_token_id = self.tokenizer.base_tokenizer.pad_token_id
        else:
            self.pad_token_id = self.tokenizer.pad_token_id

        self.dropout = torch.nn.Dropout(kwargs.get("dropout", 0.0))
        self.activation = torch.nn.Tanh()

    def last_hidden_state_forward(self, **inputs):
        """
        Forward pass to retrieve last hidden states from the model.

        Args:
            **inputs: Keyword arguments containing inputs for the model.

        Returns:
            torch.Tensor: Last hidden state tensor from the model.
        """
        model = self.model
        input_mapping = {}
        inputs["output_hidden_states"] = True
        inputs["x"] = inputs["input_ids"]  # For compatibility with Evo models
        if isinstance(inputs, BatchEncoding) or isinstance(inputs, dict):
            # Determine the input parameter names of the model's forward method
            forward_params = inspect.signature(model.forward).parameters
            # Map the inputs to the forward method parameters
            for param in forward_params:
                if param in inputs:
                    input_mapping[param] = inputs[param]
            # 对于未在模型签名中声明的关键参数，可以给出警告或日志
            ignored_keys = set(inputs.keys()) - set(input_mapping.keys())
            if ignored_keys:
                warnings.warn(f"Warning: Ignored keys in inputs: {ignored_keys}")

            inputs = input_mapping
        elif isinstance(inputs, tuple):
            input_ids = inputs[0]
            attention_mask = inputs[1] if len(inputs) > 1 else None
            inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        elif isinstance(inputs, torch.Tensor):
            shape = inputs.shape
            try:
                if len(shape) == 3:
                    if shape[1] == 2:
                        input_ids = inputs[:, 0]
                        attention_mask = inputs[:, 1]
                    else:
                        input_ids = inputs[0]
                        attention_mask = inputs[1] if len(inputs) > 1 else None
                elif len(shape) == 2:
                    input_ids = inputs
                    attention_mask = None
                else:
                    raise ValueError(
                        f"Failed to get the input_ids and attention_mask from the inputs, got shape {shape}."
                    )
            except:
                raise ValueError(
                    f"Failed to get the input_ids and attention_mask from the inputs, got shape {shape}."
                )
            inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        else:
            raise ValueError(
                f"The inputs should be a tuple, BatchEncoding or a dictionary-like object, got {type(inputs)}."
            )

        # 执行模型
        outputs = model(**inputs)

        if not hasattr(outputs, "last_hidden_state"):
            warnings.warn(
                f"last_hidden_state not found in the outputs from the {model.__class__.__name__} model."
            )

        if hasattr(outputs, "last_hidden_state"):
            last_hidden_state = outputs.last_hidden_state
        elif isinstance(outputs, dict) and "last_hidden_state" in outputs:
            last_hidden_state = outputs["last_hidden_state"]
        elif hasattr(outputs, "hidden_states"):
            last_hidden_state = outputs.hidden_states[-1]
        elif isinstance(outputs, (list, tuple, torch.Tensor)):
            if len(outputs) <= 2:
                # For Evo models that return a tuple of (last_hidden_state, logits)
                last_hidden_state = outputs[0]
            elif len(outputs) >= 3:
                last_hidden_state = outputs[-1]
        else:
            raise ValueError(
                f"Cannot find the last hidden state in the outputs from the {model.__class__.__name__} model, "
                f"please check the model architecture."
            )

        return last_hidden_state

    def loss_function(self, logits, labels):
        """
        Placeholder for loss computation function.

        Override this method in subclasses to implement custom loss.

        Args:
            logits: Model output logits.
            labels: Ground truth labels.

        Raises:
            NotImplementedError: Always, if not overridden.
        """
        raise NotImplementedError(
            "The loss_function() function should be implemented for your model."
        )

    def set_loss_fn(self, loss_function):
        """
        Set the loss function used during training.

        Args:
            loss_function (callable): Loss function to be set.
        """
        self.loss_fn = loss_function

    def predict(self, sequence_or_inputs, **kwargs):
        """
        Predict outputs from raw input sequences or tokenized inputs.

        Args:
            sequence_or_inputs: Raw sequences or tokenized inputs.
            **kwargs: Additional arguments passed to tokenizer.

        Returns:
            Model outputs without gradient computation.
        """
        # Please implement the predict() function for your model
        raw_outputs = self._forward_from_raw_input(sequence_or_inputs, **kwargs)
        return raw_outputs

    def inference(self, sequence_or_inputs, **kwargs):
        """
        Alias to predict() for inference mode.

        Args and Returns are the same as predict().
        """
        raw_outputs = self._forward_from_raw_input(sequence_or_inputs, **kwargs)
        return raw_outputs

    def __call__(self, **inputs):
        """
        Forward call for integration with transformers Trainer or native training loop.

        Args:
            **inputs: Tokenized inputs including optional labels.

        Returns:
            dict: Model outputs including optional loss.
        """
        # For transformer trainer integration, we need to pop the "inputs" to be a tokenized inputs object.
        # For native trainer, the inputs are already tokenized inputs object
        labels = inputs.pop("labels", None)
        inputs = inputs.pop("inputs", inputs)
        inputs["labels"] = labels
        if isinstance(inputs, dict):

            labels = inputs.get("labels", None)
            label = inputs.get("label", None)
            labels = labels if labels is not None else label
            # if labels is None:
            #     warnings.warn(
            #         "No labels are provided in the inputs, the model will not calculate the loss."
            #     )
        elif isinstance(inputs, tuple):
            labels = inputs[1]
            inputs = inputs[0]
        elif labels is not None:
            labels = labels
        outputs = self.forward(**inputs)

        if labels is not None:
            outputs["loss"] = self._calculate_loss(outputs, labels)
        else:
            outputs["loss"] = None
        return outputs

    def _calculate_loss(self, outputs, labels):
        """
        Calculate loss based on model outputs and labels.

        Args:
            outputs (dict): Outputs from forward pass.
            labels: Ground truth labels.

        Returns:
            Computed loss tensor.

        Raises:
            RuntimeError: If outputs do not contain logits or loss.
        """
        loss = outputs.get("loss", None)
        if loss is not None:
            return outputs

        logits = outputs["logits"]
        if logits is not None or labels is not None:
            loss = self.loss_function(logits, labels)
            return loss
        else:
            raise RuntimeError(
                "The output of the forward() function should be a dictionary-like objective"
                " and have either 'loss', or 'logits' and 'labels' attribute."
            )

    def save(self, path, overwrite=False, dtype=torch.float16, **kwargs):
        """
        Save the entire model, tokenizer, metadata, and state dict to the specified path.

        Args:
            path (str): Directory to save the model.
            overwrite (bool): Whether to overwrite existing directory. Default False.
            dtype (torch.dtype): Data type to cast the model before saving. Default torch.float16.
            **kwargs: Additional kwargs (currently unused).
        """
        self.eval()

        if os.path.exists(path) and not overwrite:
            raise FileExistsError(
                f"The path {path} already exists, please set overwrite=True to overwrite it."
            )

        if not os.path.exists(path):
            os.makedirs(path)

        for file in findfile.find_files(
            self.config.name_or_path,
            or_key=["bin", "json", "txt", "py"],
            exclude_key=["pytorch_model.bin", "model.safetensors"],
        ):
            shutil.copyfile(file, f"{path}/{os.path.basename(file)}")

        _device = self.model.device
        _dtype = self.model.dtype
        self.model.to(dtype).to("cpu")
        self.tokenizer.save_pretrained(path)

        # Save metadata including information about the loss function
        metadata = self.metadata.copy()
        if self.loss_fn is not None:
            metadata["loss_fn_class"] = self.loss_fn.__class__.__name__
            metadata["loss_fn_module"] = self.loss_fn.__class__.__module__

        with open(f"{path}/metadata.json", "w", encoding="utf8") as f:
            json.dump(metadata, f)
        with open(f"{path}/tokenizer.bin", "wb", encoding="utf8") as f:
            dill.dump(self.tokenizer, f)
        self.model.save_pretrained(
            f"{path}", safe_serialization=False
        )  # do not remove this line, used to save customized model scripts

        # Save complete state dict including all components
        with open(f"{path}/pytorch_model.bin", "wb") as f:
            torch.save(self.state_dict(), f)

        self.model.to(_dtype).to(_device)
        fprint(f"The model is saved to {path}.")

    def load(self, path, **kwargs):
        """
        Load the model from the given directory.

        Args:
            path (str): Directory where the model is saved.
            **kwargs: Additional args passed to AutoConfig.from_pretrained (e.g., device).

        Returns:
            self: The loaded model instance.
        """
        with open(f"{path}/metadata.json", "r", encoding="utf8") as f:
            metadata = json.load(f)

        if metadata["model_cls"] != self.__class__.__name__:  # Check the model class
            raise ValueError(
                f"The model class in the loaded model is {metadata['model_cls']}, "
                f"but the current model class is {self.__class__.__name__}."
            )
        config = AutoConfig.from_pretrained(path, trust_remote_code=True, **kwargs)

        for key, value in config.__dict__.items():
            if key not in self.config.__dict__ or self.config.__dict__[key] != value:
                fprint(
                    f"Warning: The value of the key {key} in the loaded model is {value}, "
                    f"but the current value is {self.config.__dict__.get(key, None)}."
                )

        # Attempt to restore any saved loss function
        if "loss_fn_class" in metadata and "loss_fn_module" in metadata:
            try:
                loss_module = import_module(metadata["loss_fn_module"])
                loss_class = getattr(loss_module, metadata["loss_fn_class"])
                # Initialize loss function if possible (parameters will be loaded with state dict)
                self.loss_fn = loss_class()
                fprint(
                    f"Restored loss function: {metadata['loss_fn_class']} from {metadata['loss_fn_module']}"
                )
            except (ImportError, AttributeError) as e:
                warnings.warn(f"Could not restore loss function: {e}")

        with open(f"{path}/pytorch_model.bin", "rb") as f:
            loaded_state_dict = torch.load(f, map_location=kwargs.get("device", "cpu"))

            # Check if keys match between current and loaded state dict
            current_keys = set(self.state_dict().keys())
            loaded_keys = set(loaded_state_dict.keys())
            missing_keys = current_keys - loaded_keys
            unexpected_keys = loaded_keys - current_keys

            if missing_keys:
                warnings.warn(f"Missing keys in loaded weights: {missing_keys}")
            if unexpected_keys:
                warnings.warn(f"Unexpected keys in loaded weights: {unexpected_keys}")

            self.load_state_dict(loaded_state_dict, strict=False)
        # Load the tokenizer
        if os.path.exists(f"{path}/tokenizer.bin"):
            with open(f"{path}/tokenizer.bin", "rb") as f:
                self.tokenizer = dill.load(f)

        return self

    def _forward_from_raw_input(self, sequence_or_inputs, **kwargs):
        """
        Convert raw input sequences into tokenized tensors and run model inference.

        Args:
            sequence_or_inputs (list or BatchEncoding or dict):
                Raw sequences or already tokenized inputs.
            **kwargs:
                Arguments for tokenizer (padding, truncation, max_length, etc.).

        Returns:
            dict: Model outputs including logits and other relevant tensors.
        """
        if not isinstance(sequence_or_inputs, BatchEncoding) and not isinstance(
            sequence_or_inputs, dict
        ):
            inputs = self.tokenizer(
                sequence_or_inputs,
                padding=kwargs.pop("padding", True),
                max_length=kwargs.pop("max_length", 1024),
                truncation=kwargs.pop("truncation", True),
                return_tensors=kwargs.pop("return_tensors", "pt"),
                **kwargs,
            )
        else:
            inputs = sequence_or_inputs
        inputs = inputs.to(self.model.device)
        with torch.no_grad():
            raw_outputs = self(**inputs)
            raw_outputs["inputs"] = inputs
        return raw_outputs

    @staticmethod
    def from_pretrained(model_name_or_path, tokenizer, *args, **kwargs):
        """
        Factory method to instantiate OmniGenomeModel from pretrained weights.

        Args:
            model_name_or_path (str): Pretrained model path or model name.
            tokenizer: Tokenizer object or None to load default.
            *args, **kwargs: Additional arguments for config and model loading.

        Returns:
            OmniGenomeModel: Initialized model instance.
        """
        config = kwargs.pop("config", None)
        if config is None:
            config = AutoConfig.from_pretrained(model_name_or_path, **kwargs)
        base_model = AutoModel.from_pretrained(model_name_or_path, **kwargs)
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(base_model, **kwargs)
        return OmniGenomeModel(config, base_model, tokenizer, *args, **kwargs)

    def model_info(self):
        """
        Print and return a summary string of the model including name, metadata,
        config, architecture, and parameter count.

        Returns:
            str: Summary info string.
        """
        info = f"Model Name: {self.__class__.__name__}\n"
        info += f"Model Metadata: {self.metadata}\n"
        info += f"Base Model Name: {self.config.name_or_path}\n"
        info += f"Model Type: {self.config.model_type}\n"
        info += f"Model Architecture: {self.config.architectures}\n"
        info += f"Model Parameters: {count_parameters(self.model) / 1e6} M\n"
        info += f"Model Config: {self.config}\n"
        fprint(info)
        return info
