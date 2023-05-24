// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "flash_attn_fusion.hpp"

#include <openvino/opsets/opset1.hpp>
#include <openvino/opsets/opset3.hpp>
#include <openvino/opsets/opset4.hpp>
#include <openvino/core/rt_info.hpp>
#include <openvino/pass/pattern/op/wrap_type.hpp>
#include "transformations/cpu_opset/x64/op/flash_attention.hpp"

#include "itt.hpp"


ov::intel_cpu::FlashFloatAttentionFusion::FlashFloatAttentionFusion() {
    MATCHER_SCOPE(FlashFloatAttentionFusion);

    auto in0 = ov::pass::pattern::any_input(ov::pass::pattern::has_static_rank());
    auto in1 = ov::pass::pattern::any_input(ov::pass::pattern::has_static_rank());
    auto in3 = ov::pass::pattern::any_input(ov::pass::pattern::has_static_rank());
    auto in4 = ov::pass::pattern::wrap_type<ov::opset4::Constant>();
    auto in5 = ov::pass::pattern::wrap_type<ov::opset4::Constant>();
    auto in6 = ov::pass::pattern::wrap_type<ov::opset4::Constant>();
    auto in7 = ov::pass::pattern::wrap_type<ov::opset4::Constant>();
    auto in8 = ov::pass::pattern::any_input(ov::pass::pattern::has_static_rank());
    auto in9 = ov::pass::pattern::wrap_type<ov::opset4::Constant>();
    auto in10 = ov::pass::pattern::wrap_type<ov::opset4::Constant>();
    auto transpose0 = std::make_shared<ov::opset3::Transpose>(in0, in4);
    auto transpose1 = std::make_shared<ov::opset3::Transpose>(in1, in5);
    auto matmul0 = std::make_shared<ov::opset3::MatMul>(transpose0, transpose1);
    auto add = std::make_shared<ov::opset4::Add>(matmul0, in3);
    auto softmax = std::make_shared<ov::opset1::Softmax>(add);
    auto transpose2 = std::make_shared<ov::opset3::Transpose>(in8, in9);
    auto matmul1 = std::make_shared<ov::opset3::MatMul>(softmax, transpose2);
    auto transpose3 = std::make_shared<ov::opset3::Transpose>(matmul1, in10);

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        auto& pattern_to_output = m.get_pattern_value_map();
        auto transpose0_in = pattern_to_output.at(in0);
        auto transpose1_in = pattern_to_output.at(in1);
        auto add_in1 = pattern_to_output.at(in3);
        auto transpose2_in = pattern_to_output.at(in8);

        // if (transpose0_in.get_shape() != transpose1_in.get_shape() || transpose0_in.get_shape() != transpose2_in.get_shape()) {
        //     return false;
        // }

        // if (transpose0_in.get_shape().size() != 4) {
        //     return false;
        // }
        //TO DO check dims in partial shape
        // auto expected_add_shape = Shape({transpose0_in.get_shape()[0], 1, 1, transpose0_in.get_shape()[1]});
        // if (add_in1.get_shape() != expected_add_shape) {
        //     return false;
        // }

        if (!valid_transpose_order(pattern_to_output.at(in4).get_node_shared_ptr(), {0, 2, 1, 3})) return false;
        if (!valid_transpose_order(pattern_to_output.at(in5).get_node_shared_ptr(), {0, 2, 3, 1})) return false;
        if (!valid_transpose_order(pattern_to_output.at(in9).get_node_shared_ptr(), {0, 2, 1, 3})) return false;
        if (!valid_transpose_order(pattern_to_output.at(in10).get_node_shared_ptr(), {0, 2, 1, 3})) return false;

        auto matmul0_node = ov::as_type_ptr<ov::opset3::MatMul>(pattern_to_output.at(matmul0).get_node_shared_ptr());
        if (!matmul0_node)
            return false;
        if (matmul0_node->get_transpose_a() || matmul0_node->get_transpose_b())
            return false;

        auto softmax_node = ov::as_type_ptr<ov::opset1::Softmax>(pattern_to_output.at(softmax).get_node_shared_ptr());
        if (!softmax_node)
            return false;
        if (softmax_node->get_axis() != 3)
            return false;

        auto matmul1_node = ov::as_type_ptr<ov::opset3::MatMul>(pattern_to_output.at(matmul1).get_node_shared_ptr());
        if (!matmul1_node)
            return false;
        if (matmul1_node->get_transpose_a() || matmul1_node->get_transpose_b())
            return false;

        auto transpose3_node = pattern_to_output.at(transpose3).get_node_shared_ptr();
        auto mha = std::make_shared<ov::intel_cpu::FlashAttentionNode>(transpose0_in, transpose1_in, add_in1, transpose2_in, std::vector<float>(), false,
                                                            transpose3_node->get_output_element_type(0));
        mha->set_friendly_name(m.get_match_root()->get_friendly_name());
        ov::copy_runtime_info({pattern_to_output.at(transpose0).get_node_shared_ptr(),
                                   pattern_to_output.at(transpose1).get_node_shared_ptr(),
                                   pattern_to_output.at(matmul0).get_node_shared_ptr(),
                                   pattern_to_output.at(add).get_node_shared_ptr(),
                                   pattern_to_output.at(softmax).get_node_shared_ptr(),
                                   pattern_to_output.at(transpose2).get_node_shared_ptr(),
                                   pattern_to_output.at(matmul1).get_node_shared_ptr(),
                                   pattern_to_output.at(transpose3).get_node_shared_ptr(),
                                  },
                                  mha);

        if (transformation_callback(mha)) {
            return false;
        }

        ov::replace_node(m.get_match_root(), mha);

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(transpose3, matcher_name);
    this->register_matcher(m, callback);
}