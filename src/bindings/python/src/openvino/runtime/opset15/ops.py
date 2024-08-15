# -*- coding: utf-8 -*-
# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Factory functions for ops added to openvino opset15."""
from functools import partial
from typing import List, Literal, Optional

import numpy as np
from openvino.runtime import Node, Type
from openvino.runtime.opset1 import convert_like
from openvino.runtime.opset14 import constant
from openvino.runtime.opset_utils import _get_node_factory
from openvino.runtime.utils.decorators import nameable_op
from openvino.runtime.utils.types import NodeInput, as_nodes

_get_node_factory_opset15 = partial(_get_node_factory, "opset15")

# -------------------------------------------- ops ------------------------------------------------


@nameable_op
def scatter_nd_update(
    data: NodeInput,
    indices: NodeInput,
    updates: NodeInput,
    reduction: Optional[Literal["none", "sum", "sub", "prod", "min", "max"]] = None,
    name: Optional[str] = None,
) -> Node:
    """Return a node which performs ScatterNDUpdate.

    :param data: Node input representing the tensor to be updated.
    :param indices: Node input representing the indices at which updates will be applied.
    :param updates: Node input representing the updates to be applied.
    :param reduction: The type of operation to perform on the inputs. One of "none", "sum",
                      "sub", "prod", "min", "max".
    :param name: Optional name for the output node.
    :return: New node performing the ScatterNDUpdate.
    """
    inputs = as_nodes(data, indices, updates, name=name)
    attributes = {}
    if reduction:
        attributes["reduction"] = reduction
    return _get_node_factory_opset15().create("ScatterNDUpdate", inputs, attributes)


@nameable_op
def col2im(
    data: NodeInput,
    output_size: NodeInput,
    kernel_size: NodeInput,
    strides: Optional[List[int]] = None,
    dilations: Optional[List[int]] = None,
    pads_begin: Optional[List[int]] = None,
    pads_end: Optional[List[int]] = None,
    name: Optional[str] = None,
) -> Node:
    """Perform data movement operation which combines sliding blocks into an image tensor.

    :param  data:                The node providing input data.
    :param  output_size:         Shape of the spatial dimensions of the output image.
    :param  kernel_size:         Size of the sliding blocks.
    :param  strides:             Stride on the sliding blocks in the input spatial dimensions. Defaults to [1, 1].
    :param  dilations:           The dilation of filter elements (distance between elements). Defaults to [1, 1].
    :param  pads_begin:          The number of pixels added at the beginning along each axis. Defaults to [0, 0].
    :param  pads_end:            The number of pixels added at the end along each axis. Defaults to [0, 0].
    :param  name:                The optional name for the created output node.

    :return:   The new node performing Col2Im operation.
    """
    if strides is None:
        strides = [1, 1]
    if dilations is None:
        dilations = [1, 1]
    if pads_begin is None:
        pads_begin = [0, 0]
    if pads_end is None:
        pads_end = [0, 0]
    return _get_node_factory_opset15().create(
        "Col2Im",
        as_nodes(data, output_size, kernel_size, name=name),
        {
            "strides": strides,
            "dilations": dilations,
            "pads_begin": pads_begin,
            "pads_end": pads_end,
        },
    )


@nameable_op
def embedding_bag_offsets(
    emb_table: NodeInput,
    indices: NodeInput,
    offsets: NodeInput,
    default_index: Optional[NodeInput] = None,
    per_sample_weights: Optional[NodeInput] = None,
    reduction: Literal["sum", "mean"] = "sum",
    name: Optional[str] = None,
) -> Node:
    """Return a node which performs sums or means of bags of embeddings without the intermediate embeddings.

    :param emb_table: Tensor containing the embedding lookup table.
    :param indices: 1D Tensor with indices.
    :param offsets: 1D Tensor containing the starting index positions of each bag in indices.
    :param per_sample_weights: Tensor with weights for each sample.
    :param default_index: Scalar containing default index in embedding table to fill empty bags.
                          If unset or set to -1, empty bags will be filled with 0.
                          Reverse indexing using negative indices is not supported.
    :param reduction: String to select algorithm used to perform reduction of elements in bag.
    :param name: Optional name for output node.
    :return: The new node performing EmbeddingBagOffsets operation.
    """
    inputs = [emb_table, indices, offsets]
    if default_index is not None:
        inputs.append(default_index)
    elif per_sample_weights is not None:
        inputs.append(convert_like(constant(np.array(-1, np.int32)), inputs[1]))
    if per_sample_weights is not None:
        inputs.append(per_sample_weights)

    return _get_node_factory_opset15().create("EmbeddingBagOffsets", as_nodes(*inputs, name=name), {"reduction": reduction})


@nameable_op
def embedding_bag_packed(
    emb_table: NodeInput,
    indices: NodeInput,
    per_sample_weights: Optional[NodeInput] = None,
    reduction: Literal["sum", "mean"] = "sum",
    name: Optional[str] = None,
) -> Node:
    """Return a node which performs sums or means of "bags" of embeddings, without the intermediate embeddings.

    :param emb_table: Tensor containing the embedding lookup table.
    :param indices: 2D Tensor of shape [batch, indices_per_bag] with indices.
    :param per_sample_weights: Tensor of weights to be multiplied with embedding table with same shape as indices.
    :param reduction: Operator to perform reduction of elements in bag.
    :param name: Optional name for output node.
    :return: The new node performing EmbeddingBagPacked operation.
    """
    inputs = [emb_table, indices]
    if per_sample_weights is not None:
        inputs.append(per_sample_weights)

    return _get_node_factory_opset15().create("EmbeddingBagPacked", as_nodes(*inputs, name=name), {"reduction": reduction})


@nameable_op
def roi_align_rotated(
    data: NodeInput,
    rois: NodeInput,
    batch_indices: NodeInput,
    pooled_h: int,
    pooled_w: int,
    sampling_ratio: int,
    spatial_scale: float,
    clockwise_mode: bool,
    name: Optional[str] = None,
) -> Node:
    """Return a node which performs ROIAlignRotated operation.

    :param data: Input data.
    :param rois: RoIs (Regions of Interest) to pool over.
    :param batch_indices: Tensor with each element denoting the index of
                          the corresponding image in the batch.
    :param pooled_h: Height of the ROI output feature map.
    :param pooled_w: Width of the ROI output feature map.
    :param sampling_ratio: Number of bins over height and width to use to calculate
                           each output feature map element.
    :param spatial_scale: Multiplicative spatial scale factor to translate ROI coordinates.
    :param clockwise_mode:  If true, rotation angle is interpreted as clockwise,
                            otherwise as counterclockwise
    :param name: The optional name for the output node

    :return: The new node which performs ROIAlignRotated
    """
    return _get_node_factory_opset15().create(
        "ROIAlignRotated",
        as_nodes(data, rois, batch_indices, name=name),
        {
            "pooled_h": pooled_h,
            "pooled_w": pooled_w,
            "sampling_ratio": sampling_ratio,
            "spatial_scale": spatial_scale,
            "clockwise_mode": clockwise_mode,
        },
    )
