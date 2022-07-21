// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <chrono>
#include <cmath>
#include <string>
#include <vector>

#include "ie_parallel.hpp"
#include "ngraph_transformations/op/interaction.hpp"
#include "interaction.hpp"
#include "utils/general_utils.h"

namespace ov {
namespace intel_cpu {
namespace node {

Interaction::Interaction(const std::shared_ptr<ngraph::Node>& op, const dnnl::engine& eng, WeightsSharing::Ptr &cache)
        : Node(op, eng, cache) {
    std::string errorMessage;
    if (isSupportedOperation(op, errorMessage)) {
        errorPrefix = "Interaction node with name '" + getName() + "'";
    } else {
        IE_THROW(NotImplemented) << errorMessage;
    }
}

void Interaction::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;
}

void Interaction::execute(dnnl::stream strm) {
    return;
}

bool Interaction::created() const {
    return getType() == Type::Interaction;
}

bool Interaction::isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op,
        std::string& errorMessage) noexcept {
    //TODO
    return true;
}


}   // namespace node
}   // namespace intel_cpu
}   // namespace ov