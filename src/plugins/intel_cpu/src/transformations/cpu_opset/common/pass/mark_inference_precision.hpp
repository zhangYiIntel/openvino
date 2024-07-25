// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov {
namespace intel_cpu {

class MarkInferencePrecision : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("MarkInferencePrecision", "0");
    MarkInferencePrecision() : ModelPass() {}
    bool run_on_model(const std::shared_ptr<ov::Model> &model) override;
};

}   // namespace intel_cpu
}   // namespace ov
