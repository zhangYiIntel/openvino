// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace intel_cpu {

class FlashAttentionNode : public ov::op::Op {
public:
    OPENVINO_OP("FlashAttention", "cpu_plugin_opset");

    FlashAttentionNode() = default;

    FlashAttentionNode(const ngraph::Output<ngraph::Node>& in0,
                       const ngraph::Output<ngraph::Node>& in1,
                       const ngraph::Output<ngraph::Node>& in2,
                       const ngraph::Output<ngraph::Node>& in3,
                       const std::vector<float>& mul_scales,
                       bool is_mul_first,
                       const ngraph::element::Type output_type);
    bool visit_attributes(ov::AttributeVisitor &visitor) override;

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;
    ov::element::Type get_output_type() const { return m_output_type; }
private:
    ov::element::Type m_output_type;
    bool is_mul_first;
    std::vector<float> mul_scales;
};

}   // namespace intel_cpu
}   // namespace ov
