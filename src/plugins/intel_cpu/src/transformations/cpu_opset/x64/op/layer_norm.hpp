// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/node.hpp>
#include <ngraph/op/op.hpp>

namespace ov {
namespace intel_cpu {

class LayerNorm : public ngraph::op::Op {
public:
    OPENVINO_OP("LayerNorm", "cpu_plugin_opset");

    LayerNorm() = default;

    LayerNorm(const ngraph::Output<ngraph::Node> &A, const ngraph::Output<ngraph::Node> &B, const ngraph::Output<ngraph::Node> &C);
    
    void validate_and_infer_types() override;

    bool visit_attributes(ngraph::AttributeVisitor &visitor) override;

    std::shared_ptr<ngraph::Node> clone_with_new_inputs(const ngraph::OutputVector &new_args) const override;
public:
    bool fuse_gelu = false;
};

}   // namespace intel_cpu
}   // namespace ov
