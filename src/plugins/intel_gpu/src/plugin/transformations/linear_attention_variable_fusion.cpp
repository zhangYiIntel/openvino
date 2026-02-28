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

#include <unordered_map>

namespace ov::intel_gpu {

namespace {
struct VariableUsageInfo {
    size_t read_value_count = 0;
    size_t assign_count = 0;
};

thread_local std::unordered_map<std::string, VariableUsageInfo> g_variable_usage;
}  // namespace

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

        // Find Assign connected to LinearAttention output(1) and ensure it's the only consumer
        std::shared_ptr<ov::op::v6::Assign> assign_node = nullptr;
        size_t assign_consumers = 0;
        for (auto& target_input : linear_attn_node->output(1).get_target_inputs()) {
            auto user = target_input.get_node()->shared_from_this();
            if (auto assign = ov::as_type_ptr<ov::op::v6::Assign>(user)) {
                assign_node = assign;
                assign_consumers++;
                continue;
            }
            if (auto convert = ov::as_type_ptr<ov::op::v0::Convert>(user)) {
                // Only allow no-op Convert (same input/output type)
                if (convert->get_input_element_type(0) != convert->get_output_element_type(0))
                    return false;

                bool convert_only_assign = true;
                for (auto& convert_target : convert->output(0).get_target_inputs()) {
                    auto convert_user = convert_target.get_node()->shared_from_this();
                    if (auto assign = ov::as_type_ptr<ov::op::v6::Assign>(convert_user)) {
                        assign_node = assign;
                        assign_consumers++;
                    } else {
                        convert_only_assign = false;
                    }
                }
                if (!convert_only_assign)
                    return false;
                continue;
            }

            // Any other consumer of output(1) makes fusion unsafe
            return false;
        }

        if (assign_consumers != 1)
            return false;

        if (!assign_node)
            return false;

        if (past_node->get_variable_id() != assign_node->get_variable_id())
            return false;

        // Disallow type-conversion on the state input to avoid mismatched variable updates
        auto past_input_node = linear_attn_node->input_value(5).get_node_shared_ptr();
        if (past_input_node.get() != past_node.get()) {
            // Allow only no-op Convert directly on ReadValue
            auto past_convert = ov::as_type_ptr<ov::op::v0::Convert>(past_input_node);
            if (!past_convert)
                return false;

            if (past_convert->get_input_node_shared_ptr(0).get() != past_node.get())
                return false;

            if (past_convert->get_input_element_type(0) != past_convert->get_output_element_type(0))
                return false;
        }

        auto usage_it = g_variable_usage.find(past_node->get_variable_id());
        if (usage_it == g_variable_usage.end())
            return false;

        // Only fuse when a variable is used by a single ReadValue and a single Assign
        if (usage_it->second.read_value_count != 1 || usage_it->second.assign_count != 1)
            return false;

        // Ensure ReadValue is only used by this LinearAttention (or through a single Convert into it)
        for (auto& past_target : past_node->output(0).get_target_inputs()) {
            auto past_user = past_target.get_node()->shared_from_this();
            if (past_user.get() == linear_attn_node.get())
                continue;

            if (auto convert = ov::as_type_ptr<ov::op::v0::Convert>(past_user)) {
                bool convert_only_la = true;
                for (auto& convert_target : convert->output(0).get_target_inputs()) {
                    if (convert_target.get_node() != linear_attn_node.get()) {
                        convert_only_la = false;
                        break;
                    }
                }
                if (!convert_only_la)
                    return false;
                continue;
            }

            return false;
        }

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

        auto internal_la = std::make_shared<ov::intel_gpu::op::LinearAttention>(inputs, variable, false);
        internal_la->set_friendly_name(linear_attn_node->get_friendly_name());

        ov::copy_runtime_info(m.get_matched_nodes(), internal_la);

        // Replace only output(0)
        linear_attn_node->output(0).replace(internal_la->output(0));

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(linear_attn, "LinearAttentionVariableFusionMatcher");
    this->register_matcher(m, callback);
}

LinearAttentionVariableFusion::LinearAttentionVariableFusion() {
    add_matcher<ov::intel_gpu::LinearAttentionVariableFusionMatcher>();
}

bool LinearAttentionVariableFusion::run_on_model(const std::shared_ptr<ov::Model>& m) {
    g_variable_usage.clear();
    for (const auto& op : m->get_ops()) {
        if (auto rv = ov::as_type_ptr<ov::op::v6::ReadValue>(op)) {
            g_variable_usage[rv->get_variable_id()].read_value_count++;
        } else if (auto assign = ov::as_type_ptr<ov::op::v6::Assign>(op)) {
            g_variable_usage[assign->get_variable_id()].assign_count++;
        }
    }

    bool res = pass::GraphRewrite::run_on_model(m);
    if (res) {
        ov::SinkVector sinks = m->get_sinks();
        for (auto& sink : sinks) {
            if (!sink)
                continue;
            if (!ov::is_type<ov::op::v6::Assign>(sink))
                continue;

            auto src_output = sink->input_value(0);
            auto src_node = src_output.get_node_shared_ptr();

            // Remove Assigns that consume orphaned LinearAttention output(1)
            if (auto la = ov::as_type_ptr<ov::op::LinearAttention>(src_node)) {
                if (src_output.get_index() == 1 && la->output(0).get_target_inputs().empty()) {
                    m->remove_sink(sink);
                }
                continue;
            }

            // Handle Assign fed by a Convert of LinearAttention output(1)
            if (auto convert = ov::as_type_ptr<ov::op::v0::Convert>(src_node)) {
                auto conv_src = convert->input_value(0);
                auto conv_src_node = conv_src.get_node_shared_ptr();
                if (auto la = ov::as_type_ptr<ov::op::LinearAttention>(conv_src_node)) {
                    if (conv_src.get_index() == 1 && la->output(0).get_target_inputs().empty()) {
                        m->remove_sink(sink);
                    }
                }
            }
        }
    }

    return res;
}

}  // namespace ov::intel_gpu
