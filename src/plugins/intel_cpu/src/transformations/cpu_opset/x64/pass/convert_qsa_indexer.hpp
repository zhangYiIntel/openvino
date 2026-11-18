// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"

namespace ov::intel_cpu {

/// \brief Lowers the common ov::op::internal::QSAIndexer, produced by ov::pass::FuseQSAIndexer and
/// ov::pass::FuseQSAIndexerQuery, to the QSAIndexerNode of the CPU opset.
///
/// The two ops share the ports and the attributes, including query_norm and query_rope: the node applies the query
/// RMSNorm and RoPE only if the op has absorbed them. It chooses the order of the selected blocks: ascending block ids
/// unless something other than SparseAttention reads the selection, which then sees the order of
/// SparseAttentionIndexer-17 (descending score).
class ConvertQSAIndexer : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ConvertQSAIndexer");
    ConvertQSAIndexer();
};

}  // namespace ov::intel_cpu
