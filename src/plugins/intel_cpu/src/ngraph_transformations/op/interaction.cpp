// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "interaction.hpp"
#include "../itt.hpp"

ov::intel_cpu::InteractionNode::InteractionNode(const OutputVector& args) {
    validate_and_infer_types();
}

ov::intel_cpu::InteractionNode::InteractionNode(const NodeVector& args) {
    validate_and_infer_types();
}

std::shared_ptr<ngraph::Node> ov::intel_cpu::InteractionNode::clone_with_new_inputs(const ngraph::OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(InteractionNode_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<ov::intel_cpu::InteractionNode>(new_args);
    throw ngraph::ngraph_error("Unsupported number of arguments for FullyConnected operation");
}

void ov::intel_cpu::InteractionNode::validate_and_infer_types() {
    INTERNAL_OP_SCOPE(InteractionNode_validate_and_infer_types);
    //TODO
    return;
}

bool ov::intel_cpu::InteractionNode::visit_attributes(ngraph::AttributeVisitor &visitor) {
    INTERNAL_OP_SCOPE(InteractionNode_visit_attributes);
    //TODO current no attributes provided
    return true;
}
