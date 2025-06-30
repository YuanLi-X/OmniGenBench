# -*- coding: utf-8 -*-
# file: lora_model.py
# time: 12:36 11/06/2025
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# homepage: https://yangheng95.github.io
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2025. All Rights Reserved.
import torch
from torch import nn
from omnigenome.src.misc.utils import fprint

def find_linear_target_modules(model, keyword_filter=None, use_full_path=True):
    """
    Find and return the names of all nn.Linear modules in the given model.

    Args:
        model (torch.nn.Module): The PyTorch model to search.
        keyword_filter (str or list/tuple of str, optional): If provided, only
            modules whose names match any of these keywords (case-insensitive) will be returned.
            Defaults to None (no filtering).
        use_full_path (bool, optional): If True, return full module names (e.g. "layer1.linear"),
            otherwise return only the last component of the name (e.g. "linear"). Defaults to True.

    Returns:
        list[str]: Sorted list of module names that are instances of nn.Linear and match the filter.
    """
    import re
    from torch import nn

    if keyword_filter is not None:
        if isinstance(keyword_filter, str):
            keyword_filter = [keyword_filter]
        elif not isinstance(keyword_filter, (list, tuple)):
            raise TypeError("keyword_filter must be None, str, or a list/tuple of str")

        pattern = '|'.join(map(re.escape, keyword_filter))

    linear_modules = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if keyword_filter is None or re.search(pattern, name, re.IGNORECASE):
                linear_modules.add(name if use_full_path else name.split('.')[-1])

    return sorted(linear_modules)

def auto_lora_model(model, **kwargs):
    """
    Automatically create and apply a LoRA (Low-Rank Adaptation) PEFT model wrapper to the given model.

    This function configures LoRA parameters, identifies target modules for LoRA injection
    (defaulting to linear layers), freezes the original model parameters, and returns
    a LoRA-adapted model.

    Args:
        model (torch.nn.Module): The base model to be adapted with LoRA.
        **kwargs: Additional LoRA configuration arguments including:
            - target_modules (list[str], optional): Names of modules to apply LoRA to.
            - use_rslora (bool, optional): Whether to use rSLoRA variant (default True).
            - bias (str, optional): Bias mode for LoRA ("none", "all", or "lora_only").
            - r (int, optional): LoRA rank.
            - lora_alpha (int, optional): LoRA alpha scaling.
            - lora_dropout (float, optional): Dropout rate for LoRA layers.
            - keyword_filter (str or list[str], optional): Keyword filter for target module names.

    Returns:
        nn.Module: The LoRA-adapted model.

    Raises:
        AssertionError: If no target modules are found for LoRA injection.
    """
    from peft import LoraConfig, get_peft_model
    from transformers import PretrainedConfig

    # A bad case for the EVO-1 model, which has a custom config class
    ######################
    if hasattr(model, 'config') and not isinstance(model.config, PretrainedConfig):
        delattr(model.config, 'Loader')
        model.config = PretrainedConfig.from_dict(dict(model.config))
    #######################

    target_modules = kwargs.pop("target_modules", None)
    use_rslora = kwargs.pop("use_rslora", True)
    bias = kwargs.pop("bias", "none")
    r = kwargs.pop("r", 32)
    lora_alpha = kwargs.pop("lora_alpha", 256)
    lora_dropout = kwargs.pop("lora_dropout", 0.1)

    if target_modules is None:
        target_modules = find_linear_target_modules(model, keyword_filter=kwargs.get("keyword_filter", None))
    assert target_modules is not None, "No target modules found for LoRA injection."
    config = LoraConfig(
        target_modules=target_modules,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias=bias,
        use_rslora=use_rslora,
        **kwargs,
    )

    for param in model.parameters():
        param.requires_grad = False

    lora_model = get_peft_model(model, config)
    trainable_params, all_param = lora_model.get_nb_trainable_parameters()
    fprint(
        f"trainable params: {trainable_params:,d} || all params: {all_param:,d}"
        f" || trainable%: {100 * trainable_params / all_param:.4f}"
    )
    return lora_model

class OmniLoraModel(nn.Module):
    """
    A wrapper PyTorch module that applies LoRA adaptation to a given base model.

    This class initializes the LoRA model using `auto_lora_model`, moves the model to CPU by default,
    and provides convenience methods forwarding calls to the underlying LoRA-adapted base model.

    Args:
        model (torch.nn.Module): The base model to be adapted with LoRA.
        **kwargs: Additional LoRA configuration arguments passed to `auto_lora_model`.
            Must include 'target_modules' specifying which modules to adapt.

    Raises:
        ValueError: If 'target_modules' argument is not provided.

    Methods:
        to(*args, **kwargs): Override of `nn.Module.to` to move LoRA model and keep track of device/dtype.
        forward(*args, **kwargs): Forward pass through the LoRA-adapted model.
        predict(*args, **kwargs): Calls the base model's predict method.
        save(*args, **kwargs): Calls the base model's save method.
        model_info(): Returns information from the base model.
        set_loss_fn(fn): Sets loss function in the base model.
        last_hidden_state_forward(**kwargs): Calls base model method for last hidden state forward pass.
        tokenizer(): Returns tokenizer from base model.
        config(): Returns configuration from base model.
        model(): Returns underlying base model instance.
    """
    def __init__(self, model, **kwargs):
        super(OmniLoraModel, self).__init__()
        target_modules = kwargs.get("target_modules", None)
        if target_modules is None:
            raise ValueError(
                "No target modules found for LoRA injection. To perform LoRA adaptation fine-tuning, "
                "please specify the target modules using the 'target_modules' argument. "
                "The target modules depend on the model architecture, such as 'query', 'value', etc. ")

        self.lora_model = auto_lora_model(model, **kwargs)

        fprint(
            "To reduce GPU memory occupation, "
            "you should avoid include non-trainable parameters into optimizers, "
            "e.g.,  optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), ...), "
            "AVOIDING: optimizer = torch.optim.AdamW(model.parameters(), ...)"
        )

        self.config = model.config
        self.to('cpu')  # Move the model to CPU initially
        fprint(
            "LoRA model initialized with the following configuration:\n",
            self.lora_model
        )


    def to(self, *args, **kwargs):
        """
        Override the to method to ensure the lora_model is moved to the correct device and dtype.
        """
        self.lora_model.to(*args, **kwargs)
        try:
            # For evo-1 and similar models, we need to set the device and dtype
            for param in self.parameters():
                self.device = param.device
                self.dtype = param.dtype
                break
            for module in self.lora_model.modules():
                module.device = self.device
                if hasattr(module, 'dtype'):
                    module.dtype = self.dtype
        except Exception as e:
            pass # Ignore errors if parameters are not available
        return self

    def forward(self, *args, **kwargs):
        return self.lora_model(*args, **kwargs)

    def predict(self, *args, **kwargs):
        return self.lora_model.base_model.predict(*args, **kwargs)

    def save(self, *args, **kwargs):
        return self.lora_model.base_model.save(*args, **kwargs)

    def model_info(self):
        return self.lora_model.base_model.model_info()

    def set_loss_fn(self, fn):
        return self.lora_model.base_model.set_loss_fn(fn)

    def last_hidden_state_forward(self, **kwargs):
        return self.lora_model.base_model.last_hidden_state_forward(**kwargs)

    def tokenizer(self):
        return self.lora_model.base_model.tokenizer

    def config(self):
        return self.lora_model.base_model.config

    def model(self):
        return self.lora_model.base_model.model