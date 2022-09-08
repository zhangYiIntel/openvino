// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <ngraph/rt_info.hpp>
#include <openvino/op/util/multi_subgraph_base.hpp>
#include <openvino/opsets/opset8.hpp>
#include <vector>

#include "itt.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/common_optimizations/eliminate_dublicated_subgraph_inputs.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov::op::util;
ov::pass::EliminateDublicatedSubgraphOpInputs::EliminateDublicatedSubgraphOpInputs() {
    MATCHER_SCOPE(EliminateDublicatedSubgraphOpInputs);
    auto subgraph_op = pattern::wrap_type<op::util::SubGraphOp>();
    ov::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto subgraph_op = std::dynamic_pointer_cast<op::util::SubGraphOp>(m.get_match_root());
        if (subgraph_op == nullptr) {
            return false;
        }

        std::vector<std::shared_ptr<SubGraphOp::InputDescription>> should_stay;
        std::map<std::shared_ptr<SubGraphOp::InputDescription>,
                 std::vector<std::shared_ptr<SubGraphOp::InputDescription>>>
            need_to_eliminate;
        auto input_descs = subgraph_op->get_input_descriptions();
        for (int i = 0; i < input_descs.size(); i++) {
            auto& key = input_descs[i];
            auto is_equal = [&](const std::shared_ptr<SubGraphOp::InputDescription>& input) -> bool {
                if (subgraph_op->input_value(input->m_input_index) == subgraph_op->input_value(key->m_input_index)) {
                    auto invariant_l = std::dynamic_pointer_cast<SubGraphOp::InvariantInputDescription>(input);
                    auto invariant_r = std::dynamic_pointer_cast<SubGraphOp::InvariantInputDescription>(key);
                    if (invariant_l && invariant_r) {
                        return true;
                    }

                    auto slice_l = std::dynamic_pointer_cast<SubGraphOp::SliceInputDescription>(input);
                    auto slice_r = std::dynamic_pointer_cast<SubGraphOp::SliceInputDescription>(key);

                    if (slice_l && slice_r) {
                        return slice_l->m_axis == slice_r->m_axis && slice_l->m_start == slice_r->m_start &&
                               slice_l->m_end == slice_r->m_end && slice_l->m_part_size == slice_r->m_part_size &&
                               slice_l->m_stride == slice_r->m_stride;
                    }

                    auto merged_l = std::dynamic_pointer_cast<SubGraphOp::MergedInputDescription>(input);
                    auto merged_r = std::dynamic_pointer_cast<SubGraphOp::MergedInputDescription>(key);

                    if (merged_l && merged_r) {
                        return merged_l->m_body_value_index == merged_r->m_body_value_index;
                    }
                }
                return false;
            };
            auto it = std::find_if(should_stay.begin(), should_stay.end(), is_equal);
            if (it == should_stay.end()) {
                should_stay.push_back(key);
            } else {
                need_to_eliminate[*it].push_back(key);
            }
        }

        auto new_ti = std::make_shared<opset8::TensorIterator>();
        auto body = subgraph_op->get_function();
        new_ti->set_body(body);
        new_ti->set_output_descriptions(0, subgraph_op->get_output_descriptions());
        if (!need_to_eliminate.empty()) {
            for (const auto& it : need_to_eliminate) {
                for (const auto& redundant : it.second) {
                    auto parameters = body->get_parameters();
                    parameters[redundant->m_body_parameter_index]->output(0).replace(
                        parameters[it.first->m_body_parameter_index]);
                }
            }
        } else {
            return false;
        }

        ov::ParameterVector new_params;
        for (const auto& remain : should_stay) {
            auto par = body->get_parameters()[remain->m_body_parameter_index];
            new_params.push_back(par);
        }
        auto new_body = std::make_shared<ov::Model>(body->get_results(), new_params);
        new_ti->set_body(new_body);

        for (const auto& remain : should_stay) {
            auto par = body->get_parameters()[remain->m_body_parameter_index];
            auto in = subgraph_op->input_value(remain->m_input_index);
            if (auto invariant = std::dynamic_pointer_cast<SubGraphOp::InvariantInputDescription>(remain)) {
                new_ti->set_invariant_input(par, in);
            } else if (auto merged = std::dynamic_pointer_cast<SubGraphOp::MergedInputDescription>(remain)) {
                auto results = body->get_results();
                new_ti->set_merged_input(par, in, results[merged->m_body_value_index]);
            } else if (auto slice = std::dynamic_pointer_cast<SubGraphOp::SliceInputDescription>(remain)) {
                new_ti->set_sliced_input(par,
                                         in,
                                         slice->m_start,
                                         slice->m_stride,
                                         slice->m_part_size,
                                         slice->m_end,
                                         slice->m_axis);
            }
        }
        new_ti->validate_and_infer_types();

        copy_runtime_info(subgraph_op, new_ti);
        replace_node(subgraph_op, new_ti);
        return true;
    };
    auto m = std::make_shared<ngraph::pattern::Matcher>(subgraph_op, matcher_name);
    this->register_matcher(m, callback);
}
