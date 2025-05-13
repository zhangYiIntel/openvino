// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/paged_attn_fuse_rope.hpp"

#include <cstdint>
#include <memory>
#include <transformations/utils/gen_pattern.hpp>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/paged_attention.hpp"
#include "openvino/util/log.hpp"
#include "transformations/utils/utils.hpp"
#include "ov_ops/rotary_positional_embeddings.hpp"
#include "openvino/core/graph_util.hpp"
using namespace ov::gen_pattern;

ov::pass::PagedAttnFuseRope::PagedAttnFuseRope() {
    MATCHER_SCOPE(PagedAttnFuseRope);
    auto Q = ov::pass::pattern::any_input(ov::pass::pattern::has_static_rank());
    auto cos_table = ov::pass::pattern::any_input(ov::pass::pattern::has_static_rank());
    auto sin_table = ov::pass::pattern::any_input(ov::pass::pattern::has_static_rank());
    auto K = ov::pass::pattern::any_input(ov::pass::pattern::has_static_rank());

    auto Q_rope = makePattern<ov::op::internal::RoPE>({Q, cos_table, sin_table});
    auto K_rope = makePattern<ov::op::internal::RoPE>({K, cos_table, sin_table});
    auto reshaped_Q = makePattern<ov::op::v1::Reshape>({Q_rope, ov::pass::pattern::any_input()});
    auto reshaped_K = makePattern<ov::op::v1::Reshape>({K_rope, ov::pass::pattern::any_input()});
    auto V = ov::pass::pattern::any_input(ov::pass::pattern::has_static_rank());
    auto key_cache_0 = makePattern<ov::op::v0::Parameter>({});
    auto value_cache_0 = makePattern<ov::op::v0::Parameter>({});
    auto past_lens = ov::pass::pattern::any_input(ov::pass::pattern::has_static_rank());
    auto subsequence_begins = ov::pass::pattern::any_input(ov::pass::pattern::has_static_rank());
    auto block_indices = ov::pass::pattern::any_input(ov::pass::pattern::has_static_rank());
    auto block_indices_begins = ov::pass::pattern::any_input(ov::pass::pattern::has_static_rank());
    auto scale = ov::pass::pattern::any_input(ov::pass::pattern::has_static_rank());
    auto sliding_window = ov::pass::pattern::any_input(ov::pass::pattern::has_static_rank());
    auto alibi_slopes = ov::pass::pattern::any_input(ov::pass::pattern::has_static_rank());
    auto max_context_len = ov::pass::pattern::any_input(ov::pass::pattern::has_static_rank());
    auto rotated_block_indices = ov::pass::pattern::any_input(ov::pass::pattern::has_static_rank());
    auto rotation_deltas = ov::pass::pattern::any_input(ov::pass::pattern::has_static_rank());
    auto rotation_trig_lut = ov::pass::pattern::any_input(ov::pass::pattern::has_static_rank());

    auto pa_1 = makePattern<op::PagedAttentionExtension>({reshaped_Q,
                                                          reshaped_K,
                                                          V,
                                                          key_cache_0,
                                                          value_cache_0,
                                                          past_lens,
                                                          subsequence_begins,
                                                          block_indices,
                                                          block_indices_begins,
                                                          scale,
                                                          sliding_window,
                                                          alibi_slopes,
                                                          max_context_len});

    auto pa_2 = makePattern<op::PagedAttentionExtension>({reshaped_Q,
                                                          reshaped_K,
                                                          V,
                                                          key_cache_0,
                                                          value_cache_0,
                                                          past_lens,
                                                          subsequence_begins,
                                                          block_indices,
                                                          block_indices_begins,
                                                          scale,
                                                          sliding_window,
                                                          alibi_slopes,
                                                          max_context_len,
                                                          rotated_block_indices,
                                                          rotation_deltas,
                                                          rotation_trig_lut});
    auto result = pa_1 | pa_2;
    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_map();
        const std::shared_ptr<ov::Node> pa_op = m.get_match_root();
        auto cos_table_out = pattern_map.at(cos_table);
        auto sin_table_out = pattern_map.at(sin_table);
        auto q = pattern_map.at(Q);
        auto k = pattern_map.at(K);
        auto v = pattern_map.at(V);
        auto Q_rope_node = pattern_map.at(Q_rope);
        auto K_rope_node = pattern_map.at(K_rope);
        auto pa_inputs = pa_op->input_values();
        ov::replace_output_update_name(Q_rope_node->output(0), Q_rope_node->input_value(0));
        ov::replace_output_update_name(K_rope_node->output(0), K_rope_node->input_value(0));
        pa_inputs.push_back(cos_table_out);
        pa_inputs.push_back(sin_table_out);
        for (size_t i = 0; i < pa_inputs.size(); i++) {
            std::cout << pa_op->get_friendly_name() << "|" << pa_inputs[i].get_node()->get_friendly_name() << std::endl;
        }
        std::cout << "PagedAttnFuseRope|" << pa_op->get_friendly_name() << std::endl;
        auto pa_with_rope = std::make_shared<op::PagedAttentionExtension>(pa_inputs, true);
        auto rope = ov::as_type_ptr<ov::op::internal::RoPE>(Q_rope_node);
        const auto& rope_config = rope->get_config();
        pa_with_rope->m_rope_config = rope_config;
        std::cout << "PagedAttnFuseRope" << "|" << rope_config.slice_start << "|" << rope_config.slice_start << std::endl;

        pa_with_rope->set_friendly_name(pa_op->get_friendly_name());
        ov::copy_runtime_info(pa_op, pa_with_rope);

        // Step 3. Replace Negative operation with Multiply operation
        ov::replace_node(pa_op, pa_with_rope);

        return false;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(result, matcher_name);
    this->register_matcher(m, callback);
}