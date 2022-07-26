// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/node.hpp>
#include <ngraph/op/op.hpp>

namespace ov {
namespace intel_cpu {

class InteractionNode : public ngraph::op::Op {
public:
    OPENVINO_OP("Interaction", "cpu_plugin_opset");

    InteractionNode() = default;

    InteractionNode(const OutputVector& args);

    InteractionNode(const NodeVector& args);

    bool visit_attributes(ngraph::AttributeVisitor &visitor) override;

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const ngraph::OutputVector& new_args) const override;

private:
    ngraph::element::Type m_output_type;
};

}   // namespace intel_cpu
}   // namespace ov
