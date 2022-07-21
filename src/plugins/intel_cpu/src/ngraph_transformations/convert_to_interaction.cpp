// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convert_to_interaction.hpp"
#include "op/interaction.hpp"
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <transformations/utils/utils.hpp>

#include "itt.hpp"

ov::intel_cpu::ConvertToInteraction::ConvertToInteraction() {
    MATCHER_SCOPE(ConvertToInteraction);
    using namespace ngraph::pattern;
    auto dense_feature = any_input(has_static_rank());
    OutputVector features{dense_feature};
    for (size_t i = 0; i < 26; i++) {
        features.push_back(wrap_type<ngraph::opset8::EmbeddingBagOffsetsSum>({any_input(),
            any_input(), any_input()})->output(0));
    }
    auto concat_m = wrap_type<ngraph::opset1::Concat>(features, has_static_rank());
    auto reshape_m = wrap_type<ngraph::opset1::Reshape>({concat_m->output(0), any_input()->output(0)});
    auto transpose_m = wrap_type<ngraph::opset1::Transpose>({reshape_m->output(0), any_input()->output(0)});
    auto matmul_m = wrap_type<ngraph::opset1::Transpose>({reshape_m, transpose_m});

    matcher_pass_callback callback = [=](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        MATCHER_SCOPE_ENABLE(ConvertToInteraction);
        auto concat_node = pattern_map.at(concat_m).get_node_shared_ptr();
        auto dense_feature_node = concat_node->input_value(0).get_node_shared_ptr();
        auto get_consumers = [](std::shared_ptr<Node>& node) {
            auto inputs = node->output(0).get_target_inputs();
            std::vector<std::shared_ptr<Node>> consumers;
            for (auto& input : inputs) {
                consumers.push_back(input.get_node()->shared_from_this());
            }
            return consumers;
        };
        auto dense_feature_consumers = get_consumers(dense_feature_node);
        std::shared_ptr<Node> final_concat = nullptr;
        for (auto& node : dense_feature_consumers) {
            if (!ov::is_type<ngraph::opset8::Concat>(node) ||
                node  == concat_node)
                continue;
            final_concat = node;
        }
        auto interaction_node = std::make_shared<InteractionNode>(concat_node->input_values());
        replace_node(final_concat, interaction_node);

        return true;
    };

    auto m = std::make_shared<Matcher>(concat_m, matcher_name);
    this->register_matcher(m, callback);
}
