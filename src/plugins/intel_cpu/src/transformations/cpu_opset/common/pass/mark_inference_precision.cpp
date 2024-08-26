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
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/depth_to_space.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/exp.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/interpolate.hpp"
#include "openvino/op/max_pool.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/mvn.hpp"
#include "openvino/op/normalize_l2.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/reduce_min.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shuffle_channels.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/strided_slice.hpp"
#include "openvino/op/tile.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/broadcast_base.hpp"
#include "openvino/op/util/pad_base.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/pass/pass.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/visualize_tree.hpp"
#include "transformations/rt_info/dequantization_node.hpp"
#include "transformations/rt_info/decompression.hpp"

namespace ov {
namespace intel_cpu {

void mark_fq_path(const std::shared_ptr<Node>& node) {
    node->get_rt_info().emplace("fq_path", true);
}
bool is_fq_path(const std::shared_ptr<const Node>& node) {
    return node->get_rt_info().count("fq_path");
}

void erase_fq_path(const std::shared_ptr<Node>& node) {
    auto& rt_info = node->get_rt_info();
    rt_info.erase("fq_path");
}
class PropagateDownDisableSensitivityForQuantized : public pass::MatcherPass {
public:
    OPENVINO_RTTI("DisableMarkingForQuantizedNodes", "0");
    PropagateDownDisableSensitivityForQuantized() {
        MATCHER_SCOPE(PropagateDownDisableSensitivityForQuantized);
        using namespace ov::pass;
        // through this nodes
        const std::shared_ptr<Node> quantization_propagating_nodes = pattern::wrap_type<ov::op::v0::Squeeze,
                                                                                        ov::op::v0::Unsqueeze,
                                                                                        ov::op::v0::FakeQuantize,
                                                                                        ov::op::v1::Reshape,
                                                                                        op::util::BroadcastBase,
                                                                                        ov::op::v0::DepthToSpace,
                                                                                        ov::op::v0::Interpolate,
                                                                                        ov::op::v4::Interpolate,
                                                                                        ov::op::v11::Interpolate,
                                                                                        ov::op::v1::MaxPool,
                                                                                        ov::op::v8::MaxPool,
                                                                                        op::util::PadBase,
                                                                                        ov::op::v1::ReduceMax,
                                                                                        ov::op::v1::ReduceMin,
                                                                                        ov::op::v0::Relu,
                                                                                        ov::op::v1::Transpose,
                                                                                        ov::op::v0::ShuffleChannels,
                                                                                        ov::op::v1::StridedSlice,
                                                                                        ov::op::v8::Slice,
                                                                                        ov::op::v1::VariadicSplit,
                                                                                        ov::op::v1::Split,
                                                                                        ov::op::v0::Concat,
                                                                                        ov::op::v0::Tile>();

        matcher_pass_callback callback = [=](pattern::Matcher& m) {
            const auto& node = m.get_match_root();
            if (!node)
                return false;

            auto is_quantize = as_type_ptr<ov::op::v0::FakeQuantize>(node);
            if (is_quantize) {
                mark_fq_path(node);
                return true;
            }

            bool is_changed = false;

            for (const auto& in_node_output : node->input_values()) {
                auto input_node = in_node_output.get_node_shared_ptr();
                auto is_quantize = as_type_ptr<ov::op::v0::FakeQuantize>(input_node);
                if (is_quantize || is_fq_path(input_node)) {
                    mark_fq_path(node);
                    enable_fp16_compression(node);
                    is_changed = true;
                }
            }

            return is_changed;
        };
        auto m = std::make_shared<pattern::Matcher>(quantization_propagating_nodes, matcher_name);
        register_matcher(m, callback);
    }
};

bool MarkInferencePrecision::run_on_model(const std::shared_ptr<ov::Model>& model) {
    RUN_ON_MODEL_SCOPE(MarkInferencePrecision);
    using OVNodePtr = std::shared_ptr<ov::Node>;
    std::function<void(const OVNodePtr&, std::unordered_set<OVNodePtr>& skipNodes)> dfs_mark;
    dfs_mark = [&](const OVNodePtr& node, std::unordered_set<OVNodePtr>& skipped_nodes) {
        for (size_t i = 0; i < node->input_values().size(); i++) {
            const auto& parent_node = node->input_values()[i].get_node_shared_ptr();
            if (ov::is_type<ov::op::v0::MatMul>(parent_node) ||         // bert nets
                ov::is_type<ov::op::util::ConvolutionFwdPropBase>(parent_node) ||    // conv / bert nets
                ov::is_type<ov::op::util::ConvolutionBackPropBase>(parent_node) ||
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
            if (is_dequantization_node(node) || is_decompression(node)) {
                ov::disable_fp16_compression(node);
                ov::enable_keep_const_precision(node);
                ov::disable_fp16_compression(parent_node);
                ov::enable_keep_const_precision(parent_node);
                continue;
            } else if (ov::is_type<ov::op::v0::Constant>(parent_node) && ov::is_type<ov::op::v1::Select>(node)) {
                continue;
            } else if (ov::is_type<ov::op::v0::Constant>(parent_node) && ov::is_type<ov::op::util::BroadcastBase>(node)) {
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
    REGISTER_PASS(manager, PropagateDownDisableSensitivityForQuantized);

    REGISTER_PASS(manager, ov::pass::MarkPrecisionSensitiveShapeOfSubgraphs);
    manager.run_passes(model);
    return true;
}

}  // namespace intel_cpu
}  // namespace ov