// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "openvino/op/op.hpp"
#include "ov_ops/rotary_positional_embeddings.hpp"

namespace ov {
namespace op {

// This is an experimental operation that is implemented in the plugins.
// Do not use in user applications, backward compatibility is not guaranteed in future releases.
class OPENVINO_API PagedAttentionExtension : public ov::op::Op {
public:
    OPENVINO_OP("PagedAttentionExtension");

    PagedAttentionExtension(const ov::OutputVector& args);
    PagedAttentionExtension(const ov::OutputVector& args, bool fuse_rope);
    void validate_and_infer_types() override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

    void set_out_type(int index, const ov::element::Type& output_type);

    bool m_fuse_rope = false;
    op::internal::RoPE::Config m_rope_config;

protected:
    std::vector<ov::element::Type> m_output_type = {ov::element::dynamic, ov::element::dynamic};
};

}  // namespace op
}  // namespace ov
