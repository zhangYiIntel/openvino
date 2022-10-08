// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace ov {
namespace intel_cpu {

class ConvertToInteraction: public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertToInteraction", "0");
    ConvertToInteraction();
};

class FuseFQtoInteraction: public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("FuseFQtoInteraction", "0");
    FuseFQtoInteraction();
};

}   // namespace intel_cpu
}   // namespace ov
