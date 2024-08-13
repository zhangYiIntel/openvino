// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "mark_inference_precision.hpp"

#include "itt.hpp"
#include "openvino/opsets/opset1.hpp"
#include "transformations/utils.hpp"
#include "transformations/utils/utils.hpp"
#include "transformations/rt_info/keep_const_precision.hpp"
#include "utils/cpu_utils.hpp"
#include "utils/general_utils.h"
#include "openvino/core/graph_util.hpp"
#include "openvino/op/util/precision_sensitive_attribute.hpp"
#include "transformations/common_optimizations/mark_precision_sensitive_shapeof_subgraphs.hpp"
#include "openvino/pass/pass.hpp"
#include "openvino/pass/manager.hpp"

namespace ov {
namespace intel_cpu {
bool MarkInferencePrecision::run_on_model(const std::shared_ptr<ov::Model>& model) {
    RUN_ON_MODEL_SCOPE(MarkInferencePrecision);
    using OVNodePtr = std::shared_ptr<ov::Node>;
    std::function<void(const OVNodePtr&, std::unordered_set<OVNodePtr>& skipNodes)> dfs_mark;
    dfs_mark = [&](const OVNodePtr& node, std::unordered_set<OVNodePtr>& skipped_nodes) {
        for (size_t i = 0; i < node->input_values().size(); i++) {
            const auto& parent_node = node->input_values()[i].get_node_shared_ptr();
            if (ov::is_type<ov::op::v0::MatMul>(parent_node) ||         // bert nets
                ov::is_type<ov::op::v1::Convolution>(parent_node) ||    // conv / bert nets
                ov::is_type<ov::op::util::RNNCellBase>(parent_node) ||  // recurent nets
                ov::is_type<ov::op::v0::ROIPooling>(parent_node) ||     // object detection nets
                ov::is_type<ov::op::v0::Interpolate>(parent_node) ||    // super resolution nets
                ov::is_type<ov::op::util::InterpolateBase>(parent_node)) {
                continue;
            }
            const auto res = skipped_nodes.insert(parent_node);
            if (res.second)  // node not visited yet
                dfs_mark(parent_node, skipped_nodes);
        }
    };
    // search from output to skip none-siginificant nodes
    std::unordered_set<OVNodePtr> skipped_nodes;
    for (const auto& result : model->get_results()) {
        dfs_mark(result, skipped_nodes);
    }

    for (auto& node : model->get_ordered_ops()) {
        if (ov::is_type<ov::op::v0::Parameter>(node) || ov::is_type<ov::op::v0::Result>(node) ||
            ov::is_type<ov::op::v0::Constant>(node) || ov::is_type<ov::op::v3::ReadValue>(node) ||
            ov::is_type<ov::op::v3::Assign>(node)) {
            continue;
        }
        //Skip Converts at input/output, which is handled by ConverPrecision
        if (ov::is_type<ov::op::v0::Convert>(node)) {
            const auto& parent_node = node->input_values()[0].get_node_shared_ptr();
            bool is_parent_convert = ov::is_type<ov::op::v0::Parameter>(parent_node);
            auto consumers = node->output(0).get_target_inputs();
            bool is_child_result = false;
            if (consumers.size() == 1) {
                if (ov::is_type<ov::op::v0::Result>(consumers.begin()->get_node()))
                    is_child_result = true;
            }
            if (is_parent_convert || is_child_result)
                continue;
        }

        auto check_enforce_tail = [](const std::shared_ptr<ov::Node>& node) {
            const auto rt_info = node->get_rt_info();
            const auto it = rt_info.find("enforceBF16evenForGraphTail");
            bool enforce_bf16_for_graph_tail = false;
            if (it != rt_info.end())
                enforce_bf16_for_graph_tail = it->second.as<bool>();
            return enforce_bf16_for_graph_tail;
        };

        bool enforce_bf16_for_graph_tail = check_enforce_tail(node);
        // mark node from skipped_nodes
        if (skipped_nodes.count(node) && !enforce_bf16_for_graph_tail) {
            if (!ov::is_type<ov::op::v0::Constant>(node)) {
                ov::disable_fp16_compression(node);
            }
        }
        // mark node's input
        for (size_t i = 0; i < node->input_values().size(); i++) {
            const auto& parent_node = node->input_values()[i].get_node_shared_ptr();
            if (ov::is_type<ov::op::v0::Constant>(parent_node) && ov::is_type<ov::op::util::BroadcastBase>(node)) {
                continue;
            } else if (node->input_values()[i].get_element_type() != ov::element::f32) {
                ov::disable_fp16_compression(parent_node);
                continue;
            } else if (ov::is_type<ov::op::v0::Constant>(parent_node) && !ov::is_type<ov::op::v0::Concat>(node)) {
                ov::disable_fp16_compression(parent_node);
                ov::enable_keep_const_precision(parent_node);
                continue;
            } else if ((ov::is_type<ov::op::v0::Constant>(parent_node) ||
                        ov::is_type<ov::op::v0::Parameter>(parent_node)) &&
                       ov::is_type<ov::op::util::BinaryElementwiseArithmetic>(node)) {
                ov::enable_keep_const_precision(parent_node);
                ov::disable_fp16_compression(parent_node);
                continue;
            } else if (ov::is_type<ov::op::v4::Range>(parent_node) && ov::is_type<ov::op::v0::Convert>(node)) {
                ov::disable_fp16_compression(parent_node);
                continue;
            } else if (ov::is_type<ov::op::v0::Constant>(parent_node) && ov::is_type<ov::op::v4::Range>(node)) {
                ov::enable_keep_const_precision(parent_node);
                continue;
            } else if (ov::is_type<ov::op::v0::Convert>(parent_node) && ov::is_type<ov::op::v4::Range>(node)) {
                ov::disable_fp16_compression(parent_node);
                continue;
            }
        }
    }
    ov::pass::Manager manager(get_pass_config(), "MarkInferencePrecision");
    REGISTER_PASS(manager, ov::pass::MarkPrecisionSensitiveShapeOfSubgraphs);
    manager.run_passes(model);
    return true;
}

}  // namespace intel_cpu
}  // namespace ov