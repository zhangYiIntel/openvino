// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convert_qsa_indexer.hpp"

#include <algorithm>
#include <cstddef>
#include <memory>

#include "openvino/cc/pass/itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/node_input.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/type.hpp"
#include "openvino/op/qsa_indexer.hpp"
#include "openvino/op/sparse_attention.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/util/pp.hpp"
#include "transformations/cpu_opset/x64/op/qsa_indexer.hpp"
#include "transformations/cpu_opset/x64/op/sparse_attention_with_kv_cache.hpp"

namespace {

// Port of the selected blocks of SparseAttention and of SparseAttentionWithKVCache.
constexpr size_t SPARSE_ATTENTION_BLOCK_INDICES = 3;

// The result of SparseAttention does not depend on the order of the selected blocks of a row, and its CPU kernels
// prefer ascending block ids. Anything else that reads the selection sees the order of SparseAttentionIndexer-17.
bool selection_feeds_only_attention(const ov::Output<ov::Node>& selection) {
    const auto consumers = selection.get_target_inputs();
    return std::all_of(consumers.begin(), consumers.end(), [](const ov::Input<ov::Node>& consumer) {
        const auto* node = consumer.get_node();
        const bool is_attention = ov::is_type<ov::op::v17::SparseAttention>(node) ||
                                  ov::is_type<ov::intel_cpu::SparseAttentionWithKVCache>(node);
        return is_attention && consumer.get_index() == SPARSE_ATTENTION_BLOCK_INDICES;
    });
}

}  // namespace

ov::intel_cpu::ConvertQSAIndexer::ConvertQSAIndexer() {
    MATCHER_SCOPE(ConvertQSAIndexer);
    using namespace ov::pass::pattern;

    auto p_indexer = wrap_type<ov::op::internal::QSAIndexer>({});

    matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        const auto indexer = ov::as_type_ptr<ov::op::internal::QSAIndexer>(m.get_match_root());
        if (!indexer || transformation_callback(indexer)) {
            return false;
        }
        const auto& c = indexer->get_config();
        QSAIndexerNode::Config config;
        config.compress_ratio = c.key_block_size;
        config.block_topk = c.block_topk;
        config.h_sel = c.num_selection_heads;
        config.indexer_rotary_dim = c.indexer_rotary_dim;
        config.indexer_head_dim = c.indexer_head_dim;
        config.eps = c.epsilon;
        config.scale = c.scale;
        // The node applies the query RMSNorm and RoPE only if they are not done in front of the op.
        config.query_norm = c.query_norm;
        config.query_rope = c.query_rope;
        config.order_by_score = !selection_feeds_only_attention(
            indexer->output(ov::op::internal::QSAIndexer::OutputIdx::SELECTED_BLOCK_INDICES));

        const auto lowered = std::make_shared<QSAIndexerNode>(indexer->input_values(), config);
        lowered->set_friendly_name(indexer->get_friendly_name());
        ov::copy_runtime_info(indexer, lowered);
        ov::replace_node(indexer, lowered);
        return true;
    };

    auto m = std::make_shared<Matcher>(p_indexer, matcher_name);
    this->register_matcher(m, callback);
}
