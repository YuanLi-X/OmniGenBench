# -*- coding: utf-8 -*-
# file: omnigenome_wrapper.py
# time: 18:37 06/04/2024
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2024. All Rights Reserved.
import warnings

from transformers import AutoTokenizer

from ..misc.utils import env_meta_info, load_module_from_path


class OmniGenomeTokenizer:
    """
    A wrapper class for tokenizers used in the OmniGenome framework.

    It encapsulates a base tokenizer and provides extended functionality such as
    custom tokenization behavior, metadata storage, and delegation of attribute
    access to the base tokenizer when needed.

    Args:
        base_tokenizer (PreTrainedTokenizer or None): The underlying tokenizer to wrap.
        max_length (int): The maximum sequence length for tokenization (default 512).
        **kwargs: Additional optional parameters to customize tokenizer behavior and metadata.
    """
    def __init__(self, base_tokenizer=None, max_length=512, **kwargs):
        """
        Initialize the OmniGenomeTokenizer.

        Stores environment metadata and custom options such as u2t, t2u, and add_whitespace.

        Args:
            base_tokenizer: The base tokenizer instance to wrap.
            max_length: Maximum sequence length for tokenization.
            **kwargs: Additional parameters for metadata and flags.
        """
        self.metadata = env_meta_info()

        self.base_tokenizer = base_tokenizer
        self.max_length = max_length

        for key, value in kwargs.items():
            self.metadata[key] = value

        self.u2t = kwargs.get("u2t", False)
        self.t2u = kwargs.get("t2u", False)
        self.add_whitespace = kwargs.get("add_whitespace", False)

    @staticmethod
    def from_pretrained(model_name_or_path, **kwargs):
        """
        Load a pretrained tokenizer, optionally using a wrapper if available.

        Attempts to load a custom tokenizer wrapper from the model directory. If none
        is found, falls back to the default AutoTokenizer.

        Args:
            model_name_or_path (str): Path or name of the pretrained model.
            **kwargs: Additional arguments to pass to the tokenizer loader.

        Returns:
            tokenizer: An instance of the tokenizer (wrapped or base).
        """
        wrapper_path = f"{model_name_or_path.rstrip('/')}/omnigenome_wrapper.py"
        try:
            tokenizer_cls = load_module_from_path(
                "OmniGenomeTokenizerWrapper", wrapper_path
            ).Tokenizer
            tokenizer = tokenizer_cls(
                AutoTokenizer.from_pretrained(model_name_or_path, **kwargs), **kwargs
            )
        except Exception as e:
            warnings.warn(
                f"No tokenizer wrapper found in {wrapper_path} -> Exception: {e}"
            )

            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **kwargs)

        return tokenizer

    def save_pretrained(self, save_directory):
        """
        Save the wrapped base tokenizer to the specified directory.

        Args:
            save_directory (str): Directory path to save the tokenizer files.
        """
        self.base_tokenizer.save_pretrained(save_directory)

    def __call__(self, *args, **kwargs):
        """
        Tokenize input sequences using the wrapped base tokenizer with default
        settings for padding, truncation, max length, and tensor return type.

        Args:
            *args: Positional arguments passed to the base tokenizer.
            **kwargs: Keyword arguments passed to the base tokenizer, with defaults:
                padding=True, truncation=True, max_length=self.max_length or 512,
                return_tensors='pt'.

        Returns:
            Tokenized output as returned by the base tokenizer.
        """
        padding = kwargs.pop("padding", True)
        truncation = kwargs.pop("truncation", True)
        max_length = kwargs.pop(
            "max_length", self.max_length if self.max_length else 512
        )
        return_tensor = kwargs.pop("return_tensors", "pt")
        return self.base_tokenizer(
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors=return_tensor,
            *args,
            **kwargs,
        )

    def tokenize(self, sequence, **kwargs):
        """
        Abstract method for tokenizing a sequence.

        Must be implemented specifically for different models.

        Args:
            sequence (str): Input sequence to tokenize.
            **kwargs: Additional arguments for tokenization.

        Raises:
            NotImplementedError: Always, this method must be overridden.
        """
        raise NotImplementedError(
            "The tokenize() function should be adapted for different models,"
            " please implement it for your model."
        )

    def encode(self, sequence, **kwargs):
        """
        Abstract method for encoding a sequence into token IDs.

        Must be implemented specifically for different models.

        Args:
            sequence (str): Input sequence to encode.
            **kwargs: Additional arguments for encoding.

        Raises:
            NotImplementedError: Always, this method must be overridden.
        """
        raise NotImplementedError(
            "The encode() function should be adapted for different models,"
            " please implement it for your model."
        )

    def decode(self, sequence, **kwargs):
        """
        Abstract method for decoding token IDs back into a sequence.

        Must be implemented specifically for different models.

        Args:
            sequence (list or tensor): Token IDs to decode.
            **kwargs: Additional arguments for decoding.

        Raises:
            NotImplementedError: Always, this method must be overridden.
        """
        raise NotImplementedError(
            "The decode() function should be adapted for different models,"
            " please implement it for your model."
        )

    def __getattribute__(self, item):
        """
        Override attribute access to delegate to the base tokenizer if attribute
        is not found in this wrapper.

        Args:
            item (str): Attribute name.

        Returns:
            Attribute value if found.

        Raises:
            AttributeError: If the attribute is not found in both this class and the base tokenizer.
        """
        try:
            return super().__getattribute__(item)
        except AttributeError:
            try:
                return self.base_tokenizer.__getattribute__(item)
            except (AttributeError, RecursionError) as e:
                raise AttributeError(
                    f"'{self.__class__.__name__}' object has no attribute '{item}'"
                ) from e
