// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/convert_paged_la_inputs.hpp"

#include <cstdint>
#include <memory>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/paged_gated_delta_net.hpp"
#include "openvino/op/paged_causal_conv1d.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/util/log.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov::pass;

namespace v0 = ov::op::v0;

namespace ov::pass {

ConvertPagedLAInputs::ConvertPagedLAInputs(ov::element::Type cache_precision)
    : m_cache_precision(cache_precision)
{
    MATCHER_SCOPE(ConvertPagedLAInputs);

    auto query = pattern::any_input(pattern::has_static_rank());
    auto key = pattern::any_input(pattern::has_static_rank());
    auto value = pattern::any_input(pattern::has_static_rank());
    auto recurrent_state_table = pattern::wrap_type<v0::Parameter>({});
    auto gate = pattern::any_input(pattern::has_static_rank());
    auto beta = pattern::any_input(pattern::has_static_rank());
    auto subsequence_begins = pattern::any_input(pattern::has_static_rank());
    auto block_indices = pattern::any_input(pattern::has_static_rank());
    auto block_indices_begins = pattern::any_input(pattern::has_static_rank());
    auto past_lens = pattern::any_input(pattern::has_static_rank());
    auto cache_interval = pattern::any_input(pattern::has_static_rank());

    auto input_embeds = pattern::any_input(pattern::has_static_rank());
    auto conv_state_table = pattern::wrap_type<v0::Parameter>({});
    auto conv_weight = pattern::any_input(pattern::has_static_rank());
    auto conv_bias = pattern::any_input(pattern::has_static_rank());
    auto conv_subsequence_begins = pattern::any_input(pattern::has_static_rank());
    auto conv_block_indices = pattern::any_input(pattern::has_static_rank());
    auto conv_block_indices_begins = pattern::any_input(pattern::has_static_rank());
    auto conv_past_lens = pattern::any_input(pattern::has_static_rank());
    auto conv_cache_interval = pattern::any_input(pattern::has_static_rank());

    auto paged_gated_delta_net = pattern::wrap_type<ov::op::internal::PagedGatedDeltaNet>({query,
                                                                                           key,
                                                                                           value,
                                                                                           recurrent_state_table,
                                                                                           gate,
                                                                                           beta,
                                                                                           subsequence_begins,
                                                                                           block_indices,
                                                                                           block_indices_begins,
                                                                                           past_lens,
                                                                                           cache_interval});
    auto paged_causal_conv1d = pattern::wrap_type<ov::op::internal::PagedCausalConv1D>({input_embeds,
                                                                                        conv_state_table,
                                                                                        conv_weight,
                                                                                        conv_bias,
                                                                                        conv_subsequence_begins,
                                                                                        conv_block_indices,
                                                                                        conv_block_indices_begins,
                                                                                        conv_past_lens,
                                                                                        conv_cache_interval});
    auto result = std::make_shared<pattern::op::Or>(OutputVector{paged_gated_delta_net, paged_causal_conv1d});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](pattern::Matcher& m) {
        const auto root = m.get_match_root();
        if (const auto gdn = ov::as_type_ptr<ov::op::internal::PagedGatedDeltaNet>(root)) {
            auto recurrent_state_table_param =
                ov::as_type_ptr<v0::Parameter>(gdn->get_input_node_shared_ptr(3));
            if (!recurrent_state_table_param) {
                return false;
            }
            recurrent_state_table_param->set_element_type(m_cache_precision);
            recurrent_state_table_param->validate_and_infer_types();
            std::cout << "Goding to change PagedGatedDeltaNet precision|m_cache_precision|" << m_cache_precision << std::endl;
            return true;
        }

        // if (const auto conv = ov::as_type_ptr<ov::op::internal::PagedCausalConv1D>(root)) {
        //     auto conv_state_table_param = ov::as_type_ptr<v0::Parameter>(conv->get_input_node_shared_ptr(1));
        //     if (!conv_state_table_param) {
        //         return false;
        //     }
        //     conv_state_table_param->set_element_type(m_cache_precision);
        //     conv_state_table_param->validate_and_infer_types();
        //     return true;
        // }

        return false;
    };

    auto m = std::make_shared<pattern::Matcher>(result, matcher_name);
    this->register_matcher(m, callback);
}


}  // namespace ov::pass
