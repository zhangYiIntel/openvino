// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_norm.hpp"
#include "transformations/itt.hpp"

ov::intel_cpu::LayerNorm::LayerNorm(const ngraph::Output<ngraph::Node> &node1, const ngraph::Output<ngraph::Node> &node2, const ngraph::Output<ngraph::Node> &node3) :
    op::Op({node1, node2, node3}) {
    validate_and_infer_types();
}

std::shared_ptr<ngraph::Node> ov::intel_cpu::LayerNorm::clone_with_new_inputs(const ngraph::OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(LayerNorm_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<ov::intel_cpu::LayerNorm>(new_args.at(0), new_args.at(1), new_args.at(2));
}

void ov::intel_cpu::LayerNorm::validate_and_infer_types() {
    INTERNAL_OP_SCOPE(LayerNorm_validate_and_infer_types);
    auto ps = get_input_partial_shape(0);
    set_output_type(0, get_input_element_type(0), ps);
    return;
}

bool ov::intel_cpu::LayerNorm::visit_attributes(ngraph::AttributeVisitor &visitor) {
    INTERNAL_OP_SCOPE(LayerNorm_visit_attributes);
    return true;
}
