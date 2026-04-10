// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pyopenvino/graph/ops/paged_gated_delta_net.hpp"

#include "openvino/op/op.hpp"
#include "openvino/op/paged_gated_delta_net.hpp"
#include "pyopenvino/core/common.hpp"

namespace py = pybind11;

void regclass_graph_op_PagedGatedDeltaNet(py::module m) {
    using ov::op::internal::PagedGatedDeltaNet;
    py::class_<PagedGatedDeltaNet, std::shared_ptr<PagedGatedDeltaNet>, ov::Node> cls(
        m,
        "_PagedGatedDeltaNet");
    cls.doc() = "Experimental extention for PagedGatedDeltaNet operation. Use with care: no backward compatibility is "
                "guaranteed in future releases.";
    cls.def(py::init<const ov::OutputVector&, bool, float, float>(),
            py::arg("args"),
            py::arg("fuse_qk_l2norm") = false,
            py::arg("q_l2_norm_eps") = 1e-6F,
            py::arg("k_l2_norm_eps") = 1e-6F);
    cls.def_property_readonly("fuse_qk_l2norm", &PagedGatedDeltaNet::get_fuse_qk_l2norm);
    cls.def_property_readonly("q_l2_norm_eps", &PagedGatedDeltaNet::get_q_l2_norm_eps);
    cls.def_property_readonly("k_l2_norm_eps", &PagedGatedDeltaNet::get_k_l2_norm_eps);
}
