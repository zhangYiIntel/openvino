// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/op/op.hpp>

namespace ov {
namespace intel_cpu {

class MHA2Node : public ngraph::op::Op {
public:
    OPENVINO_OP("MHA2", "cpu_plugin_opset");

    MHA2Node() = default;

    MHA2Node(const ngraph::Output<ngraph::Node> &q,
             const ngraph::Output<ngraph::Node> &k,
             const ngraph::Output<ngraph::Node> &v,
             bool is_causal,
             bool kv_head_transposed,
             const std::string & name);

    MHA2Node(const ngraph::Output<ngraph::Node> &q,
             const ngraph::Output<ngraph::Node> &k,
             const ngraph::Output<ngraph::Node> &v,
             const ngraph::Output<ngraph::Node> &pastk,
             const ngraph::Output<ngraph::Node> &pastv,
             bool is_causal,
             const std::string & name);

    void validate_and_infer_types() override;

    bool visit_attributes(ngraph::AttributeVisitor &visitor) override;

    std::shared_ptr<ngraph::Node> clone_with_new_inputs(const ngraph::OutputVector &new_args) const override;

    bool get_causal_mask() const {
        return is_causal;
    }
    bool get_kv_cache() const {
        return with_kv_cache;
    }
    bool get_kv_head_transposed() const {
        return kv_head_transposed;
    }
private:
    bool is_causal;
    bool with_kv_cache;
    bool kv_head_transposed;
};

}   // namespace intel_cpu
}   // namespace ov
