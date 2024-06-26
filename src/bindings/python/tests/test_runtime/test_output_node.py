# -*- coding: utf-8 -*-
# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import openvino.runtime.opset13 as ops
from openvino import Type


def test_output_replace(device):
    param = ops.parameter([1, 64], Type.i64)
    param.output(0).get_tensor().set_names({"a", "b"})
    relu = ops.relu(param)
    relu.output(0).get_tensor().set_names({"c", "d"})

    new_relu = ops.relu(param)
    new_relu.output(0).get_tensor().set_names({"f"})

    relu.output(0).replace(new_relu.output(0))

    assert new_relu.output(0).get_tensor().get_names() == {"c", "d", "f"}


def test_output_names():
    param = ops.parameter([1, 64], Type.i64)

    names = {"param1", "data1"}
    param.output(0).set_names(names)
    assert param.output(0).get_names() == names

    more_names = {"yet_another_name", "input1"}
    param.output(0).add_names(more_names)
    assert param.output(0).get_names() == names.union(more_names)
