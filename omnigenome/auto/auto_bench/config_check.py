# -*- coding: utf-8 -*-
# file: config_verification.py
# time: 02/11/2022 17:05
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# GScholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# ResearchGate: https://www.researchgate.net/profile/Heng-Yang-17/research
# Copyright (C) 2022. All Rights Reserved.

one_shot_messages = set()


def config_check(args):
    """
    Validate the training configuration parameters.

    This function performs sanity checks on selected keys in the training
    configuration dictionary. For example, it verifies that `use_amp` is
    either `True` or `False`. If a check fails, it raises a RuntimeError
    with an explanatory message.

    Parameters:
        args (dict): A dictionary containing configuration options for training.

    Raises:
        RuntimeError: If a configuration value is invalid.
    """
    try:
        if "use_amp" in args:
            assert args["use_amp"] in {True, False}
        # if "patience" in args:
        #     assert args["patience"] > 0

    except AssertionError as e:
        raise RuntimeError(
            "Exception: {}. Some parameters are not valid, please see the main example.".format(
                e
            )
        )
