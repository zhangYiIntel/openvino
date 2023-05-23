// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "add_custom.hpp"
#include "transformations/itt.hpp"

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


template <class OpType, class T>
void addcustom_shape_infer(const OpType* op, const std::vector<T>& input_shapes, std::vector<T>& output_shapes) {
    NODE_VALIDATION_CHECK(op,
                          input_shapes.size() == 2 && output_shapes.size() == 1,
                          "Incorrect number of input/output shapes");
    auto output_shape = input_shapes[0];
    const auto& autob = op->get_autob();
    if (autob.m_type == ov::op::AutoBroadcastType::NONE) {
        NODE_VALIDATION_CHECK(op, T::merge_into(output_shape, input_shapes[1]), "Argument shapes are inconsistent.");
    } else if (autob.m_type == ov::op::AutoBroadcastType::NUMPY || autob.m_type == ov::op::AutoBroadcastType::PDPD) {
        NODE_VALIDATION_CHECK(op,
                              T::broadcast_merge_into(output_shape, input_shapes[1], autob),
                              "Argument shapes are inconsistent.");
    } else {
        NODE_VALIDATION_CHECK(op, false, "Unsupported auto broadcast specification");
    }
    output_shapes[0] = output_shape;
}

void ov::intel_cpu::AddCustom::validate_and_infer_types() {
    INTERNAL_OP_SCOPE(AddCustom_validate_and_infer_types);
    std::tuple<ov::element::Type, ov::PartialShape> args_et_pshape;
    auto shapeA = get_input_partial_shape(0);
    auto shapeB = get_input_partial_shape(1);
    std::vector<ov::PartialShape> in_shapes{shapeA, shapeB};
    if (shapeA.rank().get_length() >= shapeB.rank().get_length()) {
         args_et_pshape = {get_input_element_type(0), get_input_partial_shape(0)};
    } else {
        args_et_pshape = {get_input_element_type(1), get_input_partial_shape(1)};
    }
    std::vector<ov::PartialShape> out_shapes{ov::PartialShape{}};
    addcustom_shape_infer(this, in_shapes, out_shapes);
    set_output_type(0, get_input_element_type(0), out_shapes[0]);
    return;
}

bool ov::intel_cpu::AddCustom::visit_attributes(ngraph::AttributeVisitor &visitor) {
    INTERNAL_OP_SCOPE(AddCustom_visit_attributes);
    return true;
}
