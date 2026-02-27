// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "openvino/op/op.hpp"
#include "openvino/op/util/variable.hpp"
#include "openvino/op/util/variable_extension.hpp"

namespace ov::intel_gpu::op {

/// \brief Internal LinearAttention op with Variable support
/// This operation updates data of the corresponding Variable
class LinearAttention : public ov::op::Op, public ov::op::util::VariableExtension {
public:
    OPENVINO_OP("LinearAttention", "gpu_opset");

    LinearAttention() = default;

    LinearAttention(const ov::OutputVector& args, const std::shared_ptr<ov::op::util::Variable>& state_variable);

    void validate_and_infer_types() override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

    std::string get_variable_id() const override {
        OPENVINO_ASSERT(m_variable, "Variable is not initialized. Variable_id is unavailable");
        return m_variable->get_info().variable_id;
    }
};

}  // namespace ov::intel_gpu::op
