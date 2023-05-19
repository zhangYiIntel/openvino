// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace ov {
namespace intel_cpu {

class ConvertToLayerNorm: public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertToLayerNorm", "0");
    ConvertToLayerNorm();
};

}   // namespace intel_cpu
}   // namespace ov
