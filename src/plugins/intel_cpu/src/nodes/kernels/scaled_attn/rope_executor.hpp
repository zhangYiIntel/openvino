// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ov_ops/rotary_positional_embeddings.hpp"
#include "utils/plain_tensor.hpp"

namespace ov {
namespace intel_cpu {
namespace paged_attn {
template <typename T>
struct RoPEExecutorRotateHalf;
template <typename T>
struct RoPEExecutorInterleaved;
template <typename T>
struct RoPEExecutorChatGLM;
template <typename T>
struct RoPEExecutorQwen;
struct RopeExecutor {
    virtual void execute(std::vector<PlainTensor>& inputs, std::vector<PlainTensor>& outputs) = 0;
    virtual ~RopeExecutor() = default;
};
std::shared_ptr<ov::intel_cpu::paged_attn::RopeExecutor> make_rope_executor(const ov::element::Type srcPrecision,
                                                                            const op::internal::RoPE::Config& config,
                                                                            bool& can_inplace);
}  // namespace paged_attn

}  // namespace intel_cpu
}  // namespace ov
