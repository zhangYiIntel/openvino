// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/node.hpp>
#include <ngraph/op/op.hpp>

namespace ov {
namespace intel_cpu {

class AddCustom : public ngraph::op::Op {
public:
    OPENVINO_OP("AddCustom", "cpu_plugin_opset");

    AddCustom() = default;

    AddCustom(const ngraph::Output<ngraph::Node> &A, const ngraph::Output<ngraph::Node> &B);
    
    void validate_and_infer_types() override;

    bool visit_attributes(ngraph::AttributeVisitor &visitor) override;

    std::shared_ptr<ngraph::Node> clone_with_new_inputs(const ngraph::OutputVector &new_args) const override;
};

}   // namespace intel_cpu
}   // namespace ov
