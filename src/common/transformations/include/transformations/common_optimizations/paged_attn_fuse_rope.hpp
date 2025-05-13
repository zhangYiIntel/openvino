// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/type/element_type.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {
class TRANSFORMATIONS_API PagedAttnFuseRope;

/**
 * @ingroup ov_transformation_common_api
 * @brief Set precision and shape of KV cache in PagedAttn op based runtime options
 */

class PagedAttnFuseRope : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("PagedAttnFuseRope");
    PagedAttnFuseRope();
};

}  // namespace pass
}  // namespace ov
