# -*- coding: utf-8 -*-
# file: abstract_metric.py
# time: 12:58 09/04/2024
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2024. All Rights Reserved.
import numpy as np
import sklearn.metrics as metrics

from ..misc.utils import env_meta_info


class OmniGenomeMetric:
    """
    Abstract base class for evaluation metrics, leveraging sklearn.metrics functions.

    This class serves as a blueprint for metric implementations used to evaluate
    model predictions against true labels. It also dynamically imports all
    sklearn.metrics functions as class attributes for convenience.

    Attributes:
        metric_func (callable or None): Specific metric function to use for computation.
        ignore_y (optional): Labels or values to ignore during metric calculation.
        metadata (dict): Environment metadata information.
    """

    def __init__(self, metric_func=None, ignore_y=None, *args, **kwargs):
        """
        Initialize the metric class.

        Args:
            metric_func (callable or None): The metric function to be used for evaluation.
            ignore_y: Optional values in y_true to ignore during evaluation.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        self.metric_func = metric_func
        self.ignore_y = ignore_y

        for metric in metrics.__dict__.keys():
            setattr(self, metric, metrics.__dict__[metric])

        self.metadata = env_meta_info()

    def compute(self, y_true, y_pred) -> dict:
        """
        Abstract method to compute the metric.

        This method should be overridden in subclasses to perform the actual
        metric calculation given true and predicted values.

        Args:
            y_true (array-like): Ground truth labels.
            y_pred (array-like): Predicted labels or values.

        Returns:
            dict: A dictionary mapping metric names to computed values,
                  e.g., {'accuracy': 0.9}.

        Raises:
            NotImplementedError: If the method is not implemented in the subclass.
        """
        raise NotImplementedError(
            "Method compute() is not implemented in the child class. "
            "This function returns a dict containing the metric name and value."
            "e.g. {'accuracy': 0.9}"
        )

    @staticmethod
    def flatten(y_true, y_pred):
        """
        Utility method to flatten nested true and predicted value arrays into 1D arrays.

        Args:
            y_true (array-like): Ground truth labels, possibly multi-dimensional.
            y_pred (array-like): Predicted values, possibly multi-dimensional.

        Returns:
            tuple: Flattened numpy arrays (y_true_flat, y_pred_flat).
        """
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        return y_true, y_pred
