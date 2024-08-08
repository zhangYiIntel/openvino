// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/kv_cache.hpp"
#include "gather_shape_inference.hpp"
#include "concat_shape_inference.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/concat.hpp"

namespace ov {
namespace intel_gpu {
namespace op {

KVCache::KVCache(const Output<Node>& past,
                 const Output<Node>& new_token_data,
                 const Output<Node>& beam_idx,
                 const std::shared_ptr<ov::op::util::Variable>& past_variable,
                 int64_t concat_axis,
                 int64_t gather_axis,
                 const ov::element::Type output_type)
    : Op({past, new_token_data, beam_idx})
    , m_concat_axis(concat_axis)
    , m_gather_axis(gather_axis)
    , m_indirect(true)
    , m_output_type(output_type) {
    m_variable = past_variable;
    if (m_indirect)
        set_output_size(2);
    validate_and_infer_types();
}

KVCache::KVCache(const Output<Node>& past,
                 const Output<Node>& new_token_data,
                 const std::shared_ptr<ov::op::util::Variable>& past_variable,
                 int64_t concat_axis,
                 const ov::element::Type output_type)
    : Op({past, new_token_data})
    , m_concat_axis(concat_axis)
    , m_gather_axis(0)
    , m_indirect(false)
    , m_output_type(output_type) {
    m_variable = past_variable;
    validate_and_infer_types();
}

bool KVCache::visit_attributes(ov::AttributeVisitor& visitor) {
    visitor.on_attribute("concat_axis", m_concat_axis);
    visitor.on_attribute("gather_axis", m_gather_axis);
    visitor.on_attribute("indirect", m_indirect);
    visitor.on_attribute("output_type", m_output_type);
    return true;
}

void KVCache::validate_and_infer_types() {
    auto output_type = m_output_type == ov::element::undefined ? get_input_element_type(0) : m_output_type;
    std::vector<ov::PartialShape> input_shapes = {m_variable->get_info().data_shape, get_input_partial_shape(1)};
    if (get_output_size() == 2)
        input_shapes.push_back(get_input_partial_shape(2));
    auto shapes = shape_infer(this, input_shapes);
    set_output_type(0, output_type, shapes[0]);
    if (m_indirect) {
        set_output_type(1, get_input_element_type(2), shapes[1]);
    }
}

std::shared_ptr<Node> KVCache::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    if (new_args.size() == 2) {
        return std::make_shared<KVCache>(new_args.at(0),
                                         new_args.at(1),
                                         m_variable,
                                         m_concat_axis,
                                         m_output_type);

    } else {
        return std::make_shared<KVCache>(new_args.at(0),
                                         new_args.at(1),
                                         new_args.at(2),
                                         m_variable,
                                         m_concat_axis,
                                         m_gather_axis,
                                         m_output_type);
    }
}

std::vector<ov::PartialShape> shape_infer(const KVCache* op, std::vector<ov::PartialShape> input_shapes) {
    std::vector<ov::PartialShape> out_shapes;
    out_shapes.resize(op->get_output_size());

    const auto& gather_axis = op->get_gather_axis();
    const auto& concat_axis = ov::util::normalize(op->get_concat_axis(), input_shapes[0].size());
    if (op->get_output_size() == 2) {
        out_shapes[0] = input_shapes[0];
        out_shapes[0][gather_axis] = input_shapes[2][0];
        out_shapes[0][concat_axis] += input_shapes[1][concat_axis];

        std::vector<ov::Dimension> dims(out_shapes[0].size(), 1);
        dims[gather_axis] = out_shapes[0][gather_axis];
        dims[concat_axis] = out_shapes[0][concat_axis];
        out_shapes[1] = dims;
    } else {
        out_shapes[0] = input_shapes[0];
        out_shapes[0][concat_axis] += input_shapes[1][concat_axis];
    }

    return out_shapes;
}

}  // namespace op
}  // namespace intel_gpu
}  // namespace ov
