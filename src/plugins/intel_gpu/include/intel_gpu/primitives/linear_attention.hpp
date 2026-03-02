// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"
#include "intel_gpu/graph/topology.hpp"
#include "intel_gpu/graph/program.hpp"
#include "openvino/op/linear_attn.hpp"
#include <vector>

namespace cldnn {

using LinearAttention = ov::op::LinearAttention;

/// @brief linear_attention primitive
/// @details Performs linear_attention
struct linear_attention : public primitive_base<linear_attention> {
    CLDNN_DECLARE_PRIMITIVE(linear_attention)

    linear_attention() : primitive_base("", {}) {}

    /// @brief Constructs linear_attention primitive / layer.
    ///
    /// @param id                 An identifier of new primitive.
    /// @param inputs             A list of Input primitive ids (inputs).
    linear_attention(const primitive_id& id,
            const std::vector<input_info>& inputs,
            const std::string& variable_id = "")
        : primitive_base(id, inputs)
        , variable_id(variable_id) {
    }

    std::string variable_id;

    size_t hash() const override {
        size_t seed = primitive::hash();
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        return true;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<linear_attention>::save(ob);
        // ob << variable_id;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<linear_attention>::load(ib);
        // ib >> variable_id;
    }

    // size_t k_head_size = 0;
    // size_t v_head_size = 0;
    // size_t k_heads_num = 0;
    // size_t v_heads_num = 0;
};

}  // namespace cldnn
