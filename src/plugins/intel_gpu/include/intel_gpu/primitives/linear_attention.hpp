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
            const std::string& variable_id = "",
            const ov::element::Type& user_specified_type = ov::element::dynamic)
        : primitive_base(id, inputs)
        , variable_id(variable_id)
        , user_specified_type(user_specified_type) {
    }

    std::string variable_id;
    ov::element::Type user_specified_type;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, variable_id);
        seed = hash_combine(seed, user_specified_type.hash());
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const linear_attention>(rhs);
        return variable_id == rhs_casted.variable_id &&
               user_specified_type == rhs_casted.user_specified_type;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<linear_attention>::save(ob);
        ov::element::Type_t data_type = user_specified_type;
        ob << variable_id;
        ob << make_data(&data_type, sizeof(ov::element::Type_t));
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<linear_attention>::load(ib);
        ov::element::Type_t data_type = ov::element::Type_t::dynamic;
        ib >> variable_id;
        ib >> make_data(&data_type, sizeof(ov::element::Type_t));
        user_specified_type = data_type;
    }

    // size_t k_head_size = 0;
    // size_t v_head_size = 0;
    // size_t k_heads_num = 0;
    // size_t v_heads_num = 0;
};

}  // namespace cldnn
