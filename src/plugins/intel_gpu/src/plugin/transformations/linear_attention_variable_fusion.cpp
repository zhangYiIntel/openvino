// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "linear_attention_variable_fusion.hpp"

#include "intel_gpu/op/linear_attention.hpp"
#include "intel_gpu/op/read_value.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/op/assign.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/linear_attn.hpp"
#include "openvino/op/read_value.hpp"
#include "openvino/op/util/variable.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

namespace ov::intel_gpu {

LinearAttentionVariableFusionMatcher::LinearAttentionVariableFusionMatcher() {
    using namespace ov::pass::pattern;

    auto past = wrap_type<ov::op::v6::ReadValue>();
    auto past_convert = wrap_type<ov::op::v0::Convert>({past});
    auto past_input = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{past, past_convert});

    auto linear_attn = wrap_type<ov::op::LinearAttention>({any_input(), any_input(), any_input(), any_input(), any_input(), past_input});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        if (transformation_callback(m.get_match_root()))
            return false;

        const auto& pattern_map = m.get_pattern_value_map();
        auto linear_attn_node = ov::as_type_ptr<ov::op::LinearAttention>(m.get_match_root());
        if (!linear_attn_node)
            return false;

        auto past_node = ov::as_type_ptr<ov::op::v6::ReadValue>(pattern_map.at(past).get_node_shared_ptr());
        if (!past_node)
            return false;

        // Find Assign connected to LinearAttention output(1)
        std::shared_ptr<ov::op::v6::Assign> assign_node = nullptr;
        for (auto& target_input : linear_attn_node->output(1).get_target_inputs()) {
            auto user = target_input.get_node()->shared_from_this();
            if (auto assign = ov::as_type_ptr<ov::op::v6::Assign>(user)) {
                assign_node = assign;
                break;
            }
            if (auto convert = ov::as_type_ptr<ov::op::v0::Convert>(user)) {
                for (auto& convert_target : convert->output(0).get_target_inputs()) {
                    if (auto assign = ov::as_type_ptr<ov::op::v6::Assign>(convert_target.get_node()->shared_from_this())) {
                        assign_node = assign;
                        break;
                    }
                }
            }
        }

        if (!assign_node)
            return false;

        if (past_node->get_variable_id() != assign_node->get_variable_id())
            return false;

        auto variable = past_node->get_variable();

        std::shared_ptr<ov::Node> variable_initializer = nullptr;
        if (past_node->get_input_size() == 1) {
            variable_initializer = past_node->get_input_node_shared_ptr(0);
        }

        // Replace common ReadValue op with a custom one as common one expects paired Assign operation
        auto new_read_value_node = variable_initializer
            ? std::make_shared<ov::intel_gpu::op::ReadValue>(variable_initializer->output(0), variable)
            : std::make_shared<ov::intel_gpu::op::ReadValue>(variable);
        new_read_value_node->set_friendly_name(past_node->get_friendly_name());
        ov::copy_runtime_info(past_node, new_read_value_node);
        ov::replace_node(past_node, new_read_value_node);

        ov::OutputVector inputs;
        inputs.reserve(6);
        inputs.push_back(linear_attn_node->input_value(0));
        inputs.push_back(linear_attn_node->input_value(1));
        inputs.push_back(linear_attn_node->input_value(2));
        inputs.push_back(linear_attn_node->input_value(3));
        inputs.push_back(linear_attn_node->input_value(4));
        inputs.push_back(linear_attn_node->input_value(5));

        auto internal_la = std::make_shared<ov::intel_gpu::op::LinearAttention>(inputs, variable);
        internal_la->set_friendly_name(linear_attn_node->get_friendly_name());

        ov::copy_runtime_info(m.get_matched_nodes(), internal_la);
        ov::replace_node(linear_attn_node, internal_la);

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(linear_attn, "LinearAttentionVariableFusionMatcher");
    this->register_matcher(m, callback);
}

LinearAttentionVariableFusion::LinearAttentionVariableFusion() {
    add_matcher<ov::intel_gpu::LinearAttentionVariableFusionMatcher>();
}

bool LinearAttentionVariableFusion::run_on_model(const std::shared_ptr<ov::Model>& m) {
    bool res = pass::GraphRewrite::run_on_model(m);
    if (res) {
        ov::SinkVector sinks = m->get_sinks();
        for (auto& sink : sinks) {
            if (!sink)
                continue;
            auto input_node = sink->get_input_node_shared_ptr(0);
            if (ov::is_type<ov::op::v6::Assign>(sink) &&
                (ov::is_type<ov::intel_gpu::op::LinearAttention>(input_node) ||
                 (ov::is_type<ov::op::v0::Convert>(input_node) &&
                  ov::is_type<ov::intel_gpu::op::LinearAttention>(input_node->get_input_node_shared_ptr(0))))) {
                m->remove_sink(sink);
            }
        }
    }

    return res;
}

}  // namespace ov::intel_gpu
