# -*- coding: utf-8 -*-
# file: abstract_dataset.py
# time: 14:13 06/04/2024
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2024. All Rights Reserved.
import json

import numpy as np
import torch

from ..abc.abstract_dataset import OmniGenomeDataset
from ..misc.utils import fprint
from ... import __name__, __version__


class OmniGenomeDatasetForTokenClassification(OmniGenomeDataset):
    """
    Dataset class for token-level classification tasks in the OmniGenome framework.

    This class processes input data where each token in a sequence has an associated label.

    Args:
        data_source: The source of data (list, file path, etc.).
        tokenizer: Tokenizer instance used for sequence tokenization.
        max_length: Maximum token length for sequences.
        **kwargs: Additional metadata or configuration parameters.

    Methods:
        prepare_input(instance, **kwargs):
            Prepares tokenized input and aligned labels for a single data instance.
    """
    def __init__(self, data_source, tokenizer, max_length=None, **kwargs):
        super(OmniGenomeDatasetForTokenClassification, self).__init__(
            data_source, tokenizer, max_length, **kwargs
        )
        self.metadata.update(
            {
                "library_name": __name__,
                "omnigenome_version": __version__,
                "task": "genome_token_classification",
            }
        )

        for key, value in kwargs.items():
            self.metadata[key] = value

    def prepare_input(self, instance, **kwargs):
        """
        Prepare tokenized inputs and token-level labels from a data instance.

        Args:
            instance (str or dict): Input sequence or dict containing sequence and labels.
            **kwargs: Optional arguments for tokenization settings (padding, truncation).

        Returns:
            dict: Tokenized inputs with corresponding 'labels' tensor for token classification.
        """
        labels = -100
        if isinstance(instance, str):
            sequence = instance
        elif isinstance(instance, dict):
            sequence = (
                instance.get("seq", None)
                if "seq" in instance
                else instance.get("sequence", None)
            )
            label = instance.get("label", None)
            labels = instance.get("labels", None)
            labels = labels if labels is not None else label
            if not sequence:
                raise Exception(
                    "The input instance must contain a 'seq' or 'sequence' key."
                )
        else:
            raise Exception("Unknown instance format.")

        tokenized_inputs = self.tokenizer(
            sequence,
            padding=kwargs.get("padding", "do_not_pad"),
            truncation=kwargs.get("truncation", True),
            max_length=self.max_length,
            return_tensors="pt",
        )
        for col in tokenized_inputs:
            tokenized_inputs[col] = tokenized_inputs[col].squeeze()

        if labels is not None:
            if len(set(self.label2id.keys()) | set([str(l) for l in labels])) != len(
                set(self.label2id.keys())
            ):
                fprint(
                    f"Warning: The labels <{labels}> in the input instance do not match the label2id mapping."
                )
            labels = (
                [-100]
                + [self.label2id.get(str(l), -100) for l in labels][
                    : self.max_length - 2
                ]
                + [-100]
            )

        tokenized_inputs["labels"] = torch.tensor(labels)
        return tokenized_inputs


class OmniGenomeDatasetForSequenceClassification(OmniGenomeDataset):
    """
    Dataset class for sequence-level classification tasks in the OmniGenome framework.

    Each entire sequence has a single label.

    Args:
        data_source: Source of data.
        tokenizer: Tokenizer instance for sequences.
        max_length: Maximum sequence token length.
        **kwargs: Additional metadata or config.

    Methods:
        prepare_input(instance, **kwargs):
            Tokenizes the input sequence and prepares the sequence-level label.
    """
    def __init__(self, data_source, tokenizer, max_length=None, **kwargs):
        super(OmniGenomeDatasetForSequenceClassification, self).__init__(
            data_source, tokenizer, max_length, **kwargs
        )

        self.metadata.update(
            {
                "library_name": __name__,
                "omnigenome_version": __version__,
                "task": "genome_sequence_classification",
            }
        )
        for key, value in kwargs.items():
            self.metadata[key] = value

    def prepare_input(self, instance, **kwargs):
        """
        Prepare tokenized inputs and a single label for sequence classification.

        Args:
            instance (str or dict): Input sequence or dict with sequence and label(s).
            **kwargs: Optional tokenization parameters.

        Returns:
            dict: Tokenized inputs including a 'labels' tensor for sequence classification.
        """
        labels = -100
        if isinstance(instance, str):
            sequence = instance
        elif isinstance(instance, dict):
            sequence = (
                instance.get("seq", None)
                if "seq" in instance
                else instance.get("sequence", None)
            )
            label = instance.get("label", None)
            labels = instance.get("labels", None)
            labels = labels if labels is not None else label
        else:
            raise Exception("Unknown instance format.")

        tokenized_inputs = self.tokenizer(
            sequence,
            padding=kwargs.get("padding", "do_not_pad"),
            truncation=kwargs.get("truncation", True),
            max_length=self.max_length,
            return_tensors="pt",
        )
        for col in tokenized_inputs:
            tokenized_inputs[col] = tokenized_inputs[col].squeeze()

        if labels is not None:
            if len(set(self.label2id.keys()) | set([str(labels)])) != len(
                set(self.label2id.keys())
            ):
                fprint(
                    f"Warning: The labels <{labels}> in the input instance do not match the label2id mapping."
                )
            labels = self.label2id.get(str(labels), -100) if self.label2id else labels
            try:
                labels = int(labels)
            except Exception as e:
                # Will be error if your misused data class,
                # check if you are looking for a token classification task
                raise Exception(
                    "The input instance must contain a 'label' or 'labels' key. And the label must be an integer."
                )
        tokenized_inputs["labels"] = torch.tensor(labels)
        return tokenized_inputs


class OmniGenomeDatasetForTokenRegression(OmniGenomeDataset):
    """
    Dataset class for token-level regression tasks in OmniGenome.

    Each token in the input sequence is assigned a continuous target value.

    Args:
        data_source: Source data.
        tokenizer: Tokenizer instance.
        max_length: Max token length.
        **kwargs: Additional metadata or configs.

    Methods:
        prepare_input(instance, **kwargs):
            Tokenizes the input and processes token-level continuous labels.
    """
    def __init__(self, data_source, tokenizer, max_length=None, **kwargs):
        super(OmniGenomeDatasetForTokenRegression, self).__init__(
            data_source, tokenizer, max_length, **kwargs
        )

        self.metadata.update(
            {
                "library_name": __name__,
                "omnigenome_version": __version__,
                "task": "genome_token_regression",
            }
        )

        for key, value in kwargs.items():
            self.metadata[key] = value

    def prepare_input(self, instance, **kwargs):
        """
        Prepare tokenized inputs and continuous token-level regression labels.

        Args:
            instance (str or dict): Input sequence or dict with sequence and regression labels.
            **kwargs: Tokenization options.

        Returns:
            dict: Tokenized inputs including continuous 'labels' tensor.
        """
        labels = -100
        if isinstance(instance, str):
            sequence = instance
        elif isinstance(instance, dict):
            sequence = (
                instance.get("seq", None)
                if "seq" in instance
                else instance.get("sequence", None)
            )
            label = instance.get("label", None)
            labels = instance.get("labels", None)
            labels = labels if labels is not None else label
        else:
            raise Exception("Unknown instance format.")

        tokenized_inputs = self.tokenizer(
            sequence,
            padding=kwargs.get("padding", "do_not_pad"),
            truncation=kwargs.get("truncation", True),
            max_length=self.max_length,
            return_tensors="pt",
        )
        for col in tokenized_inputs:
            tokenized_inputs[col] = tokenized_inputs[col].squeeze()

        if labels is not None:
            # Will be error if your misused data class,
            # check if you are looking for a sequence classification task
            try:
                _labels = json.loads(labels)
            except:
                seps = [" ", ",", ";", "\t"]
                for sep in seps:
                    _labels = labels.split(sep)
                    if len(_labels) > 1:
                        break
                labels = [l for l in _labels]
            labels = np.array(labels, dtype=np.float32)[: self.max_length - 2]
            if labels.ndim == 1:
                labels = labels.reshape(-1)
                labels = np.concatenate([[-100], labels, [-100]])
            elif labels.ndim == 2:
                labels = labels.reshape(1, -1)
                labels = np.zeros(
                    (labels.shape[0] + 2, labels.shape[1]), dtype=np.float32
                )
                for i, label in enumerate(labels):
                    labels[i] = np.concatenate(
                        [[-100] * label.shape[1], label, [-100] * label.shape[1]]
                    )
        tokenized_inputs["labels"] = torch.tensor(labels, dtype=torch.float32)
        return tokenized_inputs


class OmniGenomeDatasetForSequenceRegression(OmniGenomeDataset):
    """
    Dataset class for sequence-level regression tasks in OmniGenome.

    Each entire sequence is assigned a continuous regression value.

    Args:
        data_source: Data source.
        tokenizer: Tokenizer instance.
        max_length: Maximum sequence length.
        **kwargs: Additional metadata or configurations.

    Methods:
        prepare_input(instance, **kwargs):
            Tokenizes input and prepares continuous regression label for the sequence.
    """
    def __init__(self, data_source, tokenizer, max_length=None, **kwargs):
        super(OmniGenomeDatasetForSequenceRegression, self).__init__(
            data_source, tokenizer, max_length, **kwargs
        )

        self.metadata.update(
            {
                "library_name": __name__,
                "omnigenome_version": __version__,
                "task": "genome_sequence_regression",
            }
        )

        for key, value in kwargs.items():
            self.metadata[key] = value

    def prepare_input(self, instance, **kwargs):
        """
        Prepare tokenized inputs and sequence-level continuous regression label.

        Args:
            instance (str or dict): Input sequence or dict with sequence and label(s).
            **kwargs: Tokenization options.

        Returns:
            dict: Tokenized inputs including 'labels' tensor of floats.
        """
        labels = -100
        if isinstance(instance, str):
            sequence = instance
        elif isinstance(instance, dict):
            sequence = (
                instance.get("seq", None)
                if "seq" in instance
                else instance.get("sequence", None)
            )
            label = instance.get("label", None)
            labels = instance.get("labels", None)
            labels = labels if labels is not None else label
        else:
            raise Exception("Unknown instance format.")

        tokenized_inputs = self.tokenizer(
            sequence,
            padding=kwargs.get("padding", "do_not_pad"),
            truncation=kwargs.get("truncation", True),
            max_length=self.max_length,
            return_tensors="pt",
        )
        for col in tokenized_inputs:
            tokenized_inputs[col] = tokenized_inputs[col].squeeze()

        if labels is not None:
            labels = np.array(labels, dtype=np.float32)
            if labels.ndim == 1:
                labels = labels.reshape(-1)
            elif labels.ndim == 2:
                labels = labels.reshape(1, -1)

        tokenized_inputs["labels"] = torch.tensor(labels, dtype=torch.float32)

        return tokenized_inputs
