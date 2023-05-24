// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/pass/graph_rewrite.hpp>
#include <openvino/opsets/opset4.hpp>

namespace ov {
namespace intel_cpu {

class FlashAttentionFuseBase : public ov::pass::MatcherPass {
protected:
    bool valid_transpose_order(const std::shared_ptr<ngraph::Node>& node, const std::vector<int64_t>& expected_order) {
        if (auto transpose_pattern = ngraph::as_type_ptr<ov::opset4::Constant>(node)) {
            if (transpose_pattern->cast_vector<int64_t>() != expected_order) {
                return false;
            }
        } else {
            return false;
        }

        return true;
    }
};

class FlashFloatAttentionFusion: public FlashAttentionFuseBase {
public:
    OPENVINO_RTTI("FlashFloatAttentionFusion", "0");
    FlashFloatAttentionFusion();
};

class FlashAttentionFusion : public ov::pass::GraphRewrite {
public:
    OPENVINO_RTTI("FlashAttentionFusion", "0");
    FlashAttentionFusion() {
        add_matcher<FlashFloatAttentionFusion>();
    }
};

}   // namespace intel_cpu
}   // namespace ov
