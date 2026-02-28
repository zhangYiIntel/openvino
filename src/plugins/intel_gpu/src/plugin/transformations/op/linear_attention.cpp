// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/linear_attention.hpp"

#include "openvino/core/validation_util.hpp"
#include "openvino/util/common_util.hpp"

namespace {
// Validates input rank and type for a node input.
// We consider that dynamic rank/type are always valid case.
// Empty {} means any rank/type
inline void input_check(const ov::Node* node,
                        size_t idx,
                        const std::string_view input_name,
                        std::initializer_list<ov::Rank>&& allowed_ranks,
                        const std::vector<ov::element::Type>& allowed_types) {
    using namespace ov;
    using namespace ov::util;
    using namespace ov::element;

    const auto& rank = node->get_input_partial_shape(idx).rank();
    const auto& tp = node->get_input_element_type(idx);

    auto rank_check = [&](const Rank& rank) {
        return rank.is_dynamic() || empty(allowed_ranks) || is_rank_compatible_any_of(rank.get_length(), allowed_ranks);
    };

    auto type_check = [&](const Type& type) {
        auto it = std::find(allowed_types.begin(), allowed_types.end(), tp);
        return type.is_dynamic() || allowed_types.empty() || it != allowed_types.end();
    };

    NODE_VALIDATION_CHECK(node,
                          rank_check(rank),
                          "Rank of `",
                          input_name,
                          "` input should be in [dynamic, ",
                          join(allowed_ranks),
                          "] list, but it is ",
                          rank,
                          ".");

    NODE_VALIDATION_CHECK(node,
                          type_check(tp),
                          "Element type of `",
                          input_name,
                          "` input should be in [dynamic, ",
                          join(allowed_types),
                          "] list, but it is ",
                          tp,
                          ".");
}
}  // namespace

namespace ov::intel_gpu::op {

LinearAttention::LinearAttention(const ov::OutputVector& args,
                                 const std::shared_ptr<ov::op::util::Variable>& state_variable,
                                 bool output_state)
    : ov::op::Op(args) {
    m_variable = state_variable;
    m_output_state = output_state;
    constructor_validate_and_infer_types();
}

void LinearAttention::validate_and_infer_types() {
    NODE_VALIDATION_CHECK(this,
                          get_input_size() == 6,
                          "LinearAttention expects 6 inputs, but it has ",
                          get_input_size());

    input_check(this, 0, "query", {4}, {});
    input_check(this, 1, "key", {4}, {});
    input_check(this, 2, "value", {4}, {});
    input_check(this, 3, "beta", {3}, {});
    input_check(this, 4, "g", {3}, {});
    input_check(this, 5, "initial_states", {4}, {});

    const auto& q_ps = get_input_partial_shape(0);
    const auto& v_ps = get_input_partial_shape(2);
    const auto& h_ps = get_input_partial_shape(5);

    ov::PartialShape out_ps = v_ps;
    if (out_ps.rank().is_static() && q_ps.rank().is_static() && out_ps.rank().get_length() == 4 &&
        q_ps.rank().get_length() == 4) {
        out_ps[0] = q_ps[0];
        out_ps[1] = q_ps[1];
    }
    set_output_type(0, get_input_element_type(0), out_ps);
    if (m_output_state) {
        set_output_type(1, get_input_element_type(5), h_ps);
    }
}

std::shared_ptr<ov::Node> LinearAttention::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<LinearAttention>(new_args, m_variable, m_output_state);
}

}  // namespace ov::intel_gpu::op
