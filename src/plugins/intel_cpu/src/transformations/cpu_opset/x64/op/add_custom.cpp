// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "add_custom.hpp"
#include "transformations/itt.hpp"
#include "utils.hpp"

ov::intel_cpu::AddCustom::AddCustom(const ngraph::Output<ngraph::Node> &node1, const ngraph::Output<ngraph::Node> &node2, bool fuse_gelu) :
    op::Op({node1, node2}), fuse_gelu(fuse_gelu) {
    validate_and_infer_types();
}

ov::intel_cpu::AddCustom::AddCustom(const ngraph::Output<ngraph::Node> &node1, const ngraph::Output<ngraph::Node> &node2, const ngraph::Output<ngraph::Node> &node3) :
    op::Op({node1, node2, node3}) {
    validate_and_infer_types();
}

std::shared_ptr<ngraph::Node> ov::intel_cpu::AddCustom::clone_with_new_inputs(const ngraph::OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(AddCustom_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<ov::intel_cpu::AddCustom>(new_args.at(0), new_args.at(1), fuse_gelu);
}

void ov::intel_cpu::AddCustom::validate_and_infer_types() {
    INTERNAL_OP_SCOPE(AddCustom_validate_and_infer_types);
    std::tuple<ov::element::Type, ov::PartialShape> args_et_pshape;
    auto shapeA = get_input_partial_shape(0);
    auto shapeB = get_input_partial_shape(1);
    if (shapeA.rank().get_length() >= shapeB.rank().get_length()) {
         args_et_pshape = {get_input_element_type(0), get_input_partial_shape(0)};
    } else {
        args_et_pshape = {get_input_element_type(1), get_input_partial_shape(1)};
    }
    set_output_type(0, std::get<0>(args_et_pshape), std::get<1>(args_et_pshape));
    return;
}

bool ov::intel_cpu::AddCustom::visit_attributes(ngraph::AttributeVisitor &visitor) {
    INTERNAL_OP_SCOPE(AddCustom_visit_attributes);
    return true;
}
