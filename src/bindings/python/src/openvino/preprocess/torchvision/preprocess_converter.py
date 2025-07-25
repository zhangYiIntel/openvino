# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Union
from collections.abc import Callable
import logging

from openvino import Model


class PreprocessConverter():
    def __init__(self, model: Model):
        self._model = model

    @staticmethod
    def from_torchvision(model: Model, transform: Callable, input_example: Any,
                         input_name: Union[str, None] = None) -> Model:
        """Embed torchvision preprocessing in an OpenVINO model.

        Arguments:
            model (Model):
                Result name
            transform (Callable):
                torchvision transform to convert
            input_example (torch.Tensor or np.ndarray or PIL.Image):
                Example of input data for transform to trace its structure.
                Don't confuse with the model input.
            input_name (str, optional):
                Name of the current model's input node to connect with preprocessing.
                Not needed if the model has one input.

        Returns:
            Model: OpenVINO Model object with embedded preprocessing
        Example:
            >>> model = PreprocessorConvertor.from_torchvision(model, "input", transform, input_example)
        """
        try:
            import PIL
            import torch
            from torchvision import transforms
            from .torchvision_preprocessing import _from_torchvision
            return _from_torchvision(model, transform, input_example, input_name)
        except ImportError as e:
            raise ImportError(f"Please install torch, torchvision and pillow packages:\n{e}")
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            raise e
