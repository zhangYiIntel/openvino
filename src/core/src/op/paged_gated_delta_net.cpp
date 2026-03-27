// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/paged_gated_delta_net.hpp"

#include "dimension_util.hpp"
#include "itt.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/op.hpp"

namespace {

inline void input_check(const ov::Node* node,
                        size_t idx,
                        const std::string_view input_name,
                        std::initializer_list<ov::Rank>&& allowed_ranks,
                        const std::vector<ov::element::Type>& allowed_types) {
    using namespace ov;
    using namespace ov::util;

    const auto& rank = node->get_input_partial_shape(idx).rank();
    const auto& tp = node->get_input_element_type(idx);

    auto rank_check = [&](const Rank& r) {
        return !r.is_dynamic() && is_rank_compatible_any_of(r.get_length(), allowed_ranks);
    };

    auto type_check = [&](const element::Type& type) {
        auto it = std::find(allowed_types.begin(), allowed_types.end(), tp);
        return !type.is_dynamic() && (allowed_types.empty() || it != allowed_types.end());
    };

    NODE_VALIDATION_CHECK(node,
                          rank_check(rank),
                          "Rank of `",
                          input_name,
                          "` input should be in [",
                          join(allowed_ranks),
                          "] list, but it is ",
                          rank,
                          ".");

    NODE_VALIDATION_CHECK(node,
                          type_check(tp),
                          "Element type of `",
                          input_name,
                          "` input should be in [",
                          join(allowed_types),
                          "] list, but it is ",
                          tp,
                          ".");
}

}  // namespace

namespace ov::op::internal {

PagedGatedDeltaNet::PagedGatedDeltaNet(const Output<Node>& query,
                                       const Output<Node>& key,
                                       const Output<Node>& value,
                                       const Output<Node>& recurrent_state_table,
                                       const Output<Node>& gate,
                                       const Output<Node>& beta,
                                       const Output<Node>& subsequence_begins,
                                       const Output<Node>& block_indices,
                                       const Output<Node>& block_indices_begins,
                                       const Output<Node>& past_lens,
                                       const Output<Node>& cache_interval)
    : Op({query,
          key,
          value,
          recurrent_state_table,
          gate,
          beta,
          subsequence_begins,
          block_indices,
          block_indices_begins,
          past_lens,
          cache_interval}) {
    constructor_validate_and_infer_types();
}

PagedGatedDeltaNet::PagedGatedDeltaNet(const ov::OutputVector& args) : ov::op::Op(args) {
    constructor_validate_and_infer_types();
}

void PagedGatedDeltaNet::validate_and_infer_types() {
    NODE_VALIDATION_CHECK(this,
                          get_input_size() == 11,
                          "PagedGatedDeltaNet expects 11 inputs, but it has ",
                          get_input_size());

    input_check(this, 0, "query", {3}, {ov::element::f32, ov::element::f16, ov::element::bf16});
    input_check(this, 1, "key", {3}, {ov::element::f32, ov::element::f16, ov::element::bf16});
    input_check(this, 2, "value", {3}, {ov::element::f32, ov::element::f16, ov::element::bf16});
    input_check(this, 3, "recurrent_state_table", {4}, {ov::element::f32, ov::element::f16, ov::element::bf16});
    input_check(this, 4, "gate", {2}, {ov::element::f32, ov::element::f16, ov::element::bf16});
    input_check(this, 5, "beta", {2}, {ov::element::f32, ov::element::f16, ov::element::bf16});
    input_check(this, 6, "subsequence_begins", {1}, {ov::element::i32});
    input_check(this, 7, "block_indices", {1}, {ov::element::i32});
    input_check(this, 8, "block_indices_begins", {1}, {ov::element::i32});
    input_check(this, 9, "past_lens", {1}, {ov::element::i32});
    input_check(this, 10, "cache_interval", {1}, {ov::element::i32});

    const auto query_ps = get_input_partial_shape(0);
    const auto key_ps = get_input_partial_shape(1);
    const auto value_ps = get_input_partial_shape(2);
    const auto state_ps = get_input_partial_shape(3);
    const auto gate_ps = get_input_partial_shape(4);
    const auto beta_ps = get_input_partial_shape(5);
    const auto subseq_ps = get_input_partial_shape(6);
    const auto block_begins_ps = get_input_partial_shape(8);
    const auto past_lens_ps = get_input_partial_shape(9);
    const auto cache_interval_ps = get_input_partial_shape(10);

    NODE_VALIDATION_CHECK(this,
                          query_ps[1].compatible(key_ps[1]),
                          "The number of heads in query and key should be the same, but got ",
                          query_ps[1],
                          " and ",
                          key_ps[1],
                          ".");
    NODE_VALIDATION_CHECK(this,
                          query_ps[2].compatible(key_ps[2]),
                          "The head size in query and key should be the same, but got ",
                          query_ps[2],
                          " and ",
                          key_ps[2],
                          ".");

    NODE_VALIDATION_CHECK(this,
                          gate_ps[1].compatible(beta_ps[1]) && gate_ps[1].compatible(value_ps[1]),
                          "The number of heads in gate, beta and value should be the same, but got ",
                          gate_ps[1],
                          ", ",
                          beta_ps[1],
                          " and ",
                          value_ps[1],
                          ".");

    NODE_VALIDATION_CHECK(this,
                          state_ps[1].compatible(value_ps[1]),
                          "The number of heads in recurrent_state_table and value should be the same, but got ",
                          state_ps[1],
                          " and ",
                          value_ps[1],
                          ".");
    NODE_VALIDATION_CHECK(this,
                          state_ps[2].compatible(key_ps[2]),
                          "The dim at shape[-2] of recurrent_state_table and key head size should be the same, but got ",
                          state_ps[2],
                          " and ",
                          key_ps[2],
                          ".");
    NODE_VALIDATION_CHECK(this,
                          state_ps[3].compatible(value_ps[2]),
                          "The dim at shape[-1] of recurrent_state_table and value head size should be the same, but got ",
                          state_ps[3],
                          " and ",
                          value_ps[2],
                          ".");

    NODE_VALIDATION_CHECK(this,
                          query_ps[0].compatible(key_ps[0]) && query_ps[0].compatible(value_ps[0]) &&
                              query_ps[0].compatible(gate_ps[0]) && query_ps[0].compatible(beta_ps[0]),
                          "The token dimension of query, key, value, gate and beta should be the same.");

    // sequence-related vectors
    NODE_VALIDATION_CHECK(this,
                          subseq_ps[0].compatible(block_begins_ps[0]),
                          "subsequence_begins and block_indices_begins should have the same length, but got ",
                          subseq_ps[0],
                          " and ",
                          block_begins_ps[0],
                          ".");

    const auto sequences_num = subseq_ps[0] - 1;
    NODE_VALIDATION_CHECK(this,
                          past_lens_ps[0].compatible(sequences_num),
                          "past_lens size should be (subsequence_begins size - 1), but got ",
                          past_lens_ps[0],
                          " and ",
                          sequences_num,
                          ".");
    NODE_VALIDATION_CHECK(this,
                          cache_interval_ps[0].compatible(sequences_num),
                          "cache_interval size should be (subsequence_begins size - 1), but got ",
                          cache_interval_ps[0],
                          " and ",
                          sequences_num,
                          ".");

    auto output_shape = value_ps;
    if (output_shape.rank().is_static() && query_ps.rank().is_static()) {
        output_shape[0] = query_ps[0];
    }

    set_output_type(0, get_input_element_type(2), output_shape);
}

std::shared_ptr<ov::Node> PagedGatedDeltaNet::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    return std::make_shared<PagedGatedDeltaNet>(new_args);
}

}  // namespace ov::op::internal
