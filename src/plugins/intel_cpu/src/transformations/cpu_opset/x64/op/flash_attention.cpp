// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "flash_attention.hpp"

#include <matmul_shape_inference.hpp>
#include <openvino/opsets/opset3.hpp>

#include "transformations/itt.hpp"

ov::intel_cpu::FlashAttentionNode::FlashAttentionNode(const ngraph::Output<ngraph::Node>& in0,
                                                      const ngraph::Output<ngraph::Node>& in1,
                                                      const ngraph::Output<ngraph::Node>& in2,
                                                      const ngraph::Output<ngraph::Node>& in3,
                                                      const std::vector<float>& mul_scales,
                                                      bool is_mul_first,
                                                      const ngraph::element::Type output_type)
    : Op({in0, in1, in2, in3}),
      m_output_type(output_type),
      is_mul_first(is_mul_first) {
    validate_and_infer_types();
    this->mul_scales = mul_scales;
}

std::shared_ptr<ov::Node> ov::intel_cpu::FlashAttentionNode::clone_with_new_inputs(
    const ov::OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(FlashAttentionNode_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<ov::intel_cpu::FlashAttentionNode>(new_args.at(0),
                                                               new_args.at(1),
                                                               new_args.at(2),
                                                               new_args.at(3),
                                                               mul_scales,
                                                               is_mul_first,
                                                               m_output_type);
}

void ov::intel_cpu::FlashAttentionNode::validate_and_infer_types() {
    INTERNAL_OP_SCOPE(FlashAttentionNode_validate_and_infer_types);
    try {
        auto transpose = [](const ov::PartialShape& shape, const std::vector<size_t>& order) -> ov::PartialShape {
            auto new_shape = ov::PartialShape::dynamic(shape.rank());
            for (int i = 0; i < shape.size(); i++) {
                new_shape[i] = shape[order[i]];
            }
            return new_shape;
        };

        const auto matmul0_shape0 = transpose(get_input_partial_shape(0), {0, 2, 1, 3});
        const auto matmul0_shape1 = transpose(get_input_partial_shape(1), {0, 2, 3, 1});

        auto matmul0_in0 = std::make_shared<ov::opset3::Parameter>(ov::element::f32, matmul0_shape0);
        auto matmul0_in1 = std::make_shared<ov::opset3::Parameter>(ov::element::f32, matmul0_shape1);
        auto matmul0 = std::make_shared<ov::opset3::MatMul>(matmul0_in0, matmul0_in1);

        std::vector<ov::PartialShape> matmul0_input_shapes = {matmul0_shape0, matmul0_shape1};
        std::vector<ov::PartialShape> matmul0_output_shapes = {ov::PartialShape{}};

        shape_infer(matmul0.get(), matmul0_input_shapes, matmul0_output_shapes);

        const auto matmul1_shape0 = matmul0_output_shapes[0];
        const auto matmul1_shape1 = transpose(get_input_partial_shape(3), {0, 2, 1, 3});

        auto matmul1_in0 = std::make_shared<ov::opset3::Parameter>(ov::element::f32, matmul1_shape0);
        auto matmul1_in1 = std::make_shared<ov::opset3::Parameter>(ov::element::f32, matmul1_shape1);
        auto matmul1 = std::make_shared<ov::opset3::MatMul>(matmul1_in0, matmul1_in1);

        std::vector<ov::PartialShape> matmul1_input_shapes = {matmul1_shape0, matmul1_shape1};
        std::vector<ov::PartialShape> matmul1_output_shapes = {ov::PartialShape{}};

        shape_infer(matmul1.get(), matmul1_input_shapes, matmul1_output_shapes);

        const auto output_shape = transpose(matmul1_output_shapes[0], {0, 2, 1, 3});

        set_output_type(0,
                        m_output_type == ov::element::undefined || m_output_type == ov::element::dynamic
                            ? get_input_element_type(0)
                            : m_output_type,
                        output_shape);
    } catch(std::exception& ex) {
        std::cout << __LINE__ << ex.what() << std::endl;
    }
}

bool ov::intel_cpu::FlashAttentionNode::visit_attributes(ov::AttributeVisitor& visitor) {
    INTERNAL_OP_SCOPE(FlashAttentionNode_visit_attributes);
    visitor.on_attribute("out-type", m_output_type);
    return true;
}
