// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "add_custom.hpp"
#include "transformations/itt.hpp"

ov::intel_cpu::AddCustom::AddCustom(const ngraph::Output<ngraph::Node> &node1, const ngraph::Output<ngraph::Node> &node2) :
    op::Op({node1, node2}) {
    validate_and_infer_types();
}

std::shared_ptr<ngraph::Node> ov::intel_cpu::AddCustom::clone_with_new_inputs(const ngraph::OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(AddCustom_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<ov::intel_cpu::AddCustom>(new_args.at(0), new_args.at(1));
}

void ov::intel_cpu::AddCustom::validate_and_infer_types() {
    INTERNAL_OP_SCOPE(AddCustom_validate_and_infer_types);
    auto ps = get_input_partial_shape(0);
    set_output_type(0, get_input_element_type(0), ps);
    return;
}

bool ov::intel_cpu::AddCustom::visit_attributes(ngraph::AttributeVisitor &visitor) {
    INTERNAL_OP_SCOPE(AddCustom_visit_attributes);
    return true;
}
