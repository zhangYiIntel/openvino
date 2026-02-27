// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>

#include "common_test_utils/ov_test_utils.hpp"
#include "plugin/transformations/linear_attention_variable_fusion.hpp"

#include "intel_gpu/op/linear_attention.hpp"
#include "intel_gpu/op/read_value.hpp"
#include "openvino/op/assign.hpp"
#include "openvino/op/linear_attn.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/read_value.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/sink.hpp"

using namespace testing;
using namespace ov::intel_gpu;

namespace {
std::shared_ptr<ov::Model> build_model() {
    auto variable = std::make_shared<ov::op::util::Variable>(
        ov::op::util::VariableInfo{{-1, 32, 128, 128}, ov::element::f16, "la_state"});

    auto query = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, -1, 32, 128});
    auto key = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, -1, 32, 128});
    auto value = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, -1, 32, 128});
    auto g = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, -1, 32});
    auto beta = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, -1, 32});

    auto past = std::make_shared<ov::op::v6::ReadValue>(variable);
    auto la = std::make_shared<ov::op::LinearAttention>(ov::OutputVector{query, key, value, g, beta, past});

    auto assign = std::make_shared<ov::op::v6::Assign>(la->output(1), variable);
    auto result = std::make_shared<ov::op::v0::Result>(la->output(0));

    return std::make_shared<ov::Model>(ov::ResultVector{result},
                                       ov::SinkVector{assign},
                                       ov::ParameterVector{query, key, value, g, beta});
}

std::shared_ptr<ov::Model> build_reference_model() {
    auto variable = std::make_shared<ov::op::util::Variable>(
        ov::op::util::VariableInfo{{-1, 32, 128, 128}, ov::element::f16, "la_state"});

    auto query = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, -1, 32, 128});
    auto key = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, -1, 32, 128});
    auto value = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, -1, 32, 128});
    auto g = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, -1, 32});
    auto beta = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, -1, 32});

    auto past = std::make_shared<ov::intel_gpu::op::ReadValue>(variable);
    auto la = std::make_shared<ov::intel_gpu::op::LinearAttention>(
        ov::OutputVector{query, key, value, g, beta, past}, variable);

    auto result = std::make_shared<ov::op::v0::Result>(la->output(0));

    return std::make_shared<ov::Model>(ov::ResultVector{result},
                                       ov::ParameterVector{query, key, value, g, beta});
}
}  // namespace

TEST_F(TransformationTestsF, LinearAttentionVariableFusion_basic) {
    disable_rt_info_check();

    model = build_model();
    manager.register_pass<LinearAttentionVariableFusion>();

    model_ref = build_reference_model();
}
