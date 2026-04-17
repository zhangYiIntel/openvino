// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/type/element_type.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {
class TRANSFORMATIONS_API ConvertPagedLAInputs;

/**
 * @ingroup ov_transformation_common_api
 * @brief Set precision and shape of KV cache in PagedLA op based runtime options
 */

class ConvertPagedLAInputs : public ov::pass::MatcherPass {
public:
    using UpdateShapeFunc = std::function<void(const ov::element::Type, const bool, const size_t, int64_t&, int64_t&)>;
    using UpdatePrecisionFunc = std::function<void(ov::element::Type&)>;

    OPENVINO_MATCHER_PASS_RTTI("ConvertPagedLAInputs");
    ConvertPagedLAInputs(ov::element::Type cache_precision = ov::element::f32);

private:
    ov::element::Type m_cache_precision;
};

}  // namespace pass
}  // namespace ov
