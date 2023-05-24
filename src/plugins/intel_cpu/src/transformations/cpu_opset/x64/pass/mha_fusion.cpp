// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mha_fusion.hpp"

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset3.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/pattern/op/or.hpp>
#include "transformations/cpu_opset/x64/op/mha.hpp"
#include "transformations/cpu_opset/x64/op/mha2.hpp"
#include "simplify_fakequantize.hpp"

#include "itt.hpp"

// TODO: draw pattern
ov::intel_cpu::MHAFloatFusion::MHAFloatFusion() {
    MATCHER_SCOPE(MHAFloatFusion);

    auto in0 = ngraph::pattern::any_input(ngraph::pattern::has_static_shape());
    auto in1 = ngraph::pattern::any_input(ngraph::pattern::has_static_shape());
    auto in2 = ngraph::pattern::wrap_type<ngraph::opset4::Constant>();
    auto in3 = ngraph::pattern::any_input(ngraph::pattern::has_static_shape());
    auto in4 = ngraph::pattern::wrap_type<ngraph::opset4::Constant>();
    auto in5 = ngraph::pattern::wrap_type<ngraph::opset4::Constant>();
    auto in6 = ngraph::pattern::wrap_type<ngraph::opset4::Constant>();
    auto in7 = ngraph::pattern::wrap_type<ngraph::opset4::Constant>();
    auto in8 = ngraph::pattern::any_input(ngraph::pattern::has_static_shape());
    auto in9 = ngraph::pattern::wrap_type<ngraph::opset4::Constant>();
    auto in10 = ngraph::pattern::wrap_type<ngraph::opset4::Constant>();
    auto transpose0 = std::make_shared<ngraph::opset3::Transpose>(in0, in4);
    auto transpose1 = std::make_shared<ngraph::opset3::Transpose>(in1, in5);
    auto mul = std::make_shared<ngraph::opset3::Multiply>(transpose1, in2);
    auto matmul0 = std::make_shared<ngraph::opset3::MatMul>(transpose0, mul);
    auto add = std::make_shared<ngraph::opset4::Add>(matmul0, in3);
    auto reshape0 = std::make_shared<ngraph::opset1::Reshape>(add, in6, true);
    auto softmax = std::make_shared<ngraph::opset1::Softmax>(reshape0);
    auto reshape1 = std::make_shared<ngraph::opset1::Reshape>(softmax, in7, true);
    auto transpose2 = std::make_shared<ngraph::opset3::Transpose>(in8, in9);
    auto matmul1 = std::make_shared<ngraph::opset3::MatMul>(reshape1, transpose2);
    auto transpose3 = std::make_shared<ngraph::opset3::Transpose>(matmul1, in10);

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        auto& pattern_to_output = m.get_pattern_value_map();
        auto transpose0_in = pattern_to_output.at(in0);
        auto transpose1_in = pattern_to_output.at(in1);
        auto mul_in1 = pattern_to_output.at(in2);
        auto add_in1 = pattern_to_output.at(in3);
        auto transpose2_in = pattern_to_output.at(in8);

        if (transpose0_in.get_shape() != transpose1_in.get_shape() || transpose0_in.get_shape() != transpose2_in.get_shape()) {
            return false;
        }

        if (transpose0_in.get_shape().size() != 4) {
            return false;
        }

        auto expected_add_shape = Shape({transpose0_in.get_shape()[0], 1, 1, transpose0_in.get_shape()[1]});
        if (add_in1.get_shape() != expected_add_shape) {
            return false;
        }

        if (!valid_transpose_order(pattern_to_output.at(in4).get_node_shared_ptr(), {0, 2, 1, 3})) return false;
        if (!valid_transpose_order(pattern_to_output.at(in5).get_node_shared_ptr(), {0, 2, 3, 1})) return false;
        if (!valid_transpose_order(pattern_to_output.at(in9).get_node_shared_ptr(), {0, 2, 1, 3})) return false;
        if (!valid_transpose_order(pattern_to_output.at(in10).get_node_shared_ptr(), {0, 2, 1, 3})) return false;

        std::vector<float> mul_scales;
        if (auto mul_node = ngraph::as_type_ptr<ngraph::opset3::Multiply>(pattern_to_output.at(mul).get_node_shared_ptr())) {
            mul_scales = ngraph::as_type_ptr<ngraph::opset4::Constant>(mul_node->get_input_node_shared_ptr(1))->cast_vector<float>();

            auto expected_shape = ngraph::Shape({1, transpose0_in.get_shape()[2], 1, 1});
            if (mul_scales.size() != 1 && mul_node->get_input_shape(1) != expected_shape) {
                return false;
            }
        } else {
            return false;
        }

        auto matmul0_node = ngraph::as_type_ptr<ngraph::opset3::MatMul>(pattern_to_output.at(matmul0).get_node_shared_ptr());
        if (!matmul0_node)
            return false;
        if (matmul0_node->get_transpose_a() || matmul0_node->get_transpose_b())
            return false;

        auto reshape0_node = ngraph::as_type_ptr<ngraph::opset1::Reshape>(pattern_to_output.at(reshape0).get_node_shared_ptr());
        if (!reshape0_node)
            return false;

        if (auto reshape_pattern = ngraph::as_type_ptr<ngraph::opset4::Constant>(pattern_to_output.at(in6).get_node_shared_ptr())) {
            if (reshape0_node->get_input_shape(0).size() != 4) {
                return false;
            }

            std::vector<int64_t> reshapeConstData = {static_cast<int64_t>(reshape0_node->get_input_shape(0)[0] *
                                                                          reshape0_node->get_input_shape(0)[1] *
                                                                          reshape0_node->get_input_shape(0)[2]),
                                                     -1};

            if (reshape_pattern->cast_vector<int64_t>() != reshapeConstData) {
                return false;
            }
        } else {
            return false;
        }

        if (auto reshape1_node = ngraph::as_type_ptr<ngraph::opset1::Reshape>(pattern_to_output.at(reshape1).get_node_shared_ptr())) {
            if (reshape0_node->get_input_shape(0) != reshape1_node->get_output_shape(0)) {
                return false;
            }
        } else {
            return false;
        }

        auto softmax_node = ngraph::as_type_ptr<ngraph::opset1::Softmax>(pattern_to_output.at(softmax).get_node_shared_ptr());
        if (!softmax_node)
            return false;
        if (softmax_node->get_axis() != 1)
            return false;

        auto matmul1_node = ngraph::as_type_ptr<ngraph::opset3::MatMul>(pattern_to_output.at(matmul1).get_node_shared_ptr());
        if (!matmul1_node)
            return false;
        if (matmul1_node->get_transpose_a() || matmul1_node->get_transpose_b())
            return false;

        bool is_mul_first = true;
        auto transpose3_node = pattern_to_output.at(transpose3).get_node_shared_ptr();
        auto mha = std::make_shared<ov::intel_cpu::MHANode>(transpose0_in, transpose1_in, add_in1, transpose2_in, mul_scales, is_mul_first,
                                                            transpose3_node->get_output_element_type(0));
        mha->set_friendly_name(m.get_match_root()->get_friendly_name());
        ngraph::copy_runtime_info({pattern_to_output.at(transpose0).get_node_shared_ptr(),
                                   pattern_to_output.at(transpose1).get_node_shared_ptr(),
                                   pattern_to_output.at(mul).get_node_shared_ptr(),
                                   pattern_to_output.at(matmul0).get_node_shared_ptr(),
                                   pattern_to_output.at(add).get_node_shared_ptr(),
                                   pattern_to_output.at(reshape0).get_node_shared_ptr(),
                                   pattern_to_output.at(softmax).get_node_shared_ptr(),
                                   pattern_to_output.at(reshape1).get_node_shared_ptr(),
                                   pattern_to_output.at(transpose2).get_node_shared_ptr(),
                                   pattern_to_output.at(matmul1).get_node_shared_ptr(),
                                   pattern_to_output.at(transpose3).get_node_shared_ptr(),
                                  },
                                  mha);

        if (transformation_callback(mha)) {
            return false;
        }

        ngraph::replace_node(m.get_match_root(), mha);

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(transpose3, matcher_name);
    this->register_matcher(m, callback);
}

ov::intel_cpu::MHAFloatFusion2::MHAFloatFusion2() {
    MATCHER_SCOPE(MHAFloatFusion2);

    auto in0 = ngraph::pattern::any_input(ngraph::pattern::has_static_shape());
    auto in1 = ngraph::pattern::any_input(ngraph::pattern::has_static_shape());
    auto in3 = ngraph::pattern::any_input(ngraph::pattern::has_static_shape());
    auto in4 = ngraph::pattern::wrap_type<ngraph::opset4::Constant>();
    auto in5 = ngraph::pattern::wrap_type<ngraph::opset4::Constant>();
    auto in6 = ngraph::pattern::wrap_type<ngraph::opset4::Constant>();
    auto in7 = ngraph::pattern::wrap_type<ngraph::opset4::Constant>();
    auto in8 = ngraph::pattern::any_input(ngraph::pattern::has_static_shape());
    auto in9 = ngraph::pattern::wrap_type<ngraph::opset4::Constant>();
    auto in10 = ngraph::pattern::wrap_type<ngraph::opset4::Constant>();
    auto transpose0 = std::make_shared<ngraph::opset3::Transpose>(in0, in4);
    auto transpose1 = std::make_shared<ngraph::opset3::Transpose>(in1, in5);
    auto matmul0 = std::make_shared<ngraph::opset3::MatMul>(transpose0, transpose1);
    auto add = std::make_shared<ngraph::opset4::Add>(matmul0, in3);
    auto softmax = std::make_shared<ngraph::opset1::Softmax>(add);
    auto transpose2 = std::make_shared<ngraph::opset3::Transpose>(in8, in9);
    auto matmul1 = std::make_shared<ngraph::opset3::MatMul>(softmax, transpose2);
    auto transpose3 = std::make_shared<ngraph::opset3::Transpose>(matmul1, in10);

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        auto& pattern_to_output = m.get_pattern_value_map();
        auto transpose0_in = pattern_to_output.at(in0);
        auto transpose1_in = pattern_to_output.at(in1);
        auto add_in1 = pattern_to_output.at(in3);
        auto transpose2_in = pattern_to_output.at(in8);

        if (transpose0_in.get_shape() != transpose1_in.get_shape() || transpose0_in.get_shape() != transpose2_in.get_shape()) {
            return false;
        }

        if (transpose0_in.get_shape().size() != 4) {
            return false;
        }

        auto expected_add_shape = Shape({transpose0_in.get_shape()[0], 1, 1, transpose0_in.get_shape()[1]});
        if (add_in1.get_shape() != expected_add_shape) {
            return false;
        }

        if (!valid_transpose_order(pattern_to_output.at(in4).get_node_shared_ptr(), {0, 2, 1, 3})) return false;
        if (!valid_transpose_order(pattern_to_output.at(in5).get_node_shared_ptr(), {0, 2, 3, 1})) return false;
        if (!valid_transpose_order(pattern_to_output.at(in9).get_node_shared_ptr(), {0, 2, 1, 3})) return false;
        if (!valid_transpose_order(pattern_to_output.at(in10).get_node_shared_ptr(), {0, 2, 1, 3})) return false;

        auto matmul0_node = ngraph::as_type_ptr<ngraph::opset3::MatMul>(pattern_to_output.at(matmul0).get_node_shared_ptr());
        if (!matmul0_node)
            return false;
        if (matmul0_node->get_transpose_a() || matmul0_node->get_transpose_b())
            return false;

        auto softmax_node = ngraph::as_type_ptr<ngraph::opset1::Softmax>(pattern_to_output.at(softmax).get_node_shared_ptr());
        if (!softmax_node)
            return false;
        if (softmax_node->get_axis() != 3)
            return false;

        auto matmul1_node = ngraph::as_type_ptr<ngraph::opset3::MatMul>(pattern_to_output.at(matmul1).get_node_shared_ptr());
        if (!matmul1_node)
            return false;
        if (matmul1_node->get_transpose_a() || matmul1_node->get_transpose_b())
            return false;

        auto transpose3_node = pattern_to_output.at(transpose3).get_node_shared_ptr();
        auto mha = std::make_shared<ov::intel_cpu::MHANode>(transpose0_in, transpose1_in, add_in1, transpose2_in, std::vector<float>(), false,
                                                            transpose3_node->get_output_element_type(0));
        mha->set_friendly_name(m.get_match_root()->get_friendly_name());
        ngraph::copy_runtime_info({pattern_to_output.at(transpose0).get_node_shared_ptr(),
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

        ngraph::replace_node(m.get_match_root(), mha);

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(transpose3, matcher_name);
    this->register_matcher(m, callback);
}

/*
B: batch size
M: tokens in query
N: tokens in key & value
H: head count (6/8/...)
K: number of state(64)

    Q: [B, M, H*K] => reshape => [B, M, H, K] => Transpose<0,2,1,3> => reshape => [B*H, M, K] => Matmul_qk_in0
with_kv_cache = False
    K: [B, N, H*K] => reshape => [B, N, H, K] => Transpose<0,2,1,3> => reshape => [B*H, N, K] => Matmul_qk_in1
    V: [B, N, H*K] => reshape => [B, N, H, K] => Transpose<0,2,1,3> => reshape => [B*H, N, K] => Matmul_wv_in1
with_kv_cache = True
    kt: [B, 1, H*K] => Reshape => [B, 1, H, K] => Transpose<0,2,1,3> => [B, H, 1, K] => Concat to pastK
    vt: [B, 1, H*K] => Reshape => [B, 1, H, K] => Transpose<0,2,1,3> => [B, H, 1, K] => Concat to pastV
    
    pastK: [B, H,N-1,K] => Concat<axis=2> => [B,H,N,64] => Reshape => [B*H,N,64] => Matmul_qk_in1
    pastV: [B, H,N-1,K] => Concat<axis=2> => [B,H,N,64] => Reshape => [B*H,N,64] => Matmul_wv_in1

    Concat result of K & V is also output

    Matmul_qk<transpose_a=0 transpose_b=1> => Softmax<axis=2> => Matmul_wv<transpose_a=0 transpose_b=0>
                                           => [B*H, M, K] => reshape => [B, H, M, K] => Transpose<0,2,1,3>
                                           => [B, M, H, K] => reshape => [B, M, H*K]
*/

ov::intel_cpu::MHAFloatFusionWhisper::MHAFloatFusionWhisper() {
    MATCHER_SCOPE(MHAFloatFusionWhisper);
    bool b_special_zero = true;
    auto q = ngraph::pattern::any_input();  // [B, M, H*K]
    auto k = ngraph::pattern::any_input();  // [B, N, H*K]
    auto v = ngraph::pattern::any_input();  // [B, N, H*K]
    auto shape_q = ngraph::pattern::any_input();
    auto shape_k = ngraph::pattern::any_input();
    auto shape_v = ngraph::pattern::any_input();
    auto order_0213_q = ngraph::pattern::wrap_type<ngraph::opset4::Constant>();
    auto order_0213_k = ngraph::pattern::wrap_type<ngraph::opset4::Constant>();
    auto order_0213_v = ngraph::pattern::wrap_type<ngraph::opset4::Constant>();
    auto order_0213_wv = ngraph::pattern::wrap_type<ngraph::opset4::Constant>();
    auto shape_q2 = ngraph::pattern::any_input();
    auto shape_k2 = ngraph::pattern::any_input();
    auto shape_v2 = ngraph::pattern::any_input();
    auto shape_wv1 = ngraph::pattern::any_input();
    auto shape_wv2 = ngraph::pattern::any_input();

    auto q1 = std::make_shared<ngraph::opset3::Reshape>(q, shape_q, b_special_zero);          // [B, M, H, K]
    auto q2 = std::make_shared<ngraph::opset3::Transpose>(q1, order_0213_q);  // [B, H, M, K]
    auto q3 = std::make_shared<ngraph::opset3::Reshape>(q2, shape_q2, b_special_zero);        // [B*H, M, K]

    auto k1 = std::make_shared<ngraph::opset3::Reshape>(k, shape_k, b_special_zero);  // [B, N, H, K]
    auto k2 = std::make_shared<ngraph::opset3::Transpose>(k1, order_0213_k); // [B, H, N, K]

    // from model input (encoder attention)
    auto k_encoder = std::make_shared<ngraph::opset3::Parameter>();

    auto k3 = std::make_shared<ngraph::opset3::Reshape>(
        std::make_shared<ngraph::pattern::op::Or>(OutputVector{k2, k_encoder}),
        shape_k2,
        b_special_zero);  // [B*H, N, K]

    auto v1 = std::make_shared<ngraph::opset3::Reshape>(v, shape_v, b_special_zero);    // [B, N, H, K]
    auto v2 = std::make_shared<ngraph::opset3::Transpose>(v1, order_0213_v);            // [B, H, N, K]
    // from model input (encoder attention)
    auto v_encoder = std::make_shared<ngraph::opset3::Parameter>();

    auto v3 = std::make_shared<ngraph::opset3::Reshape>(
        std::make_shared<ngraph::pattern::op::Or>(OutputVector{v2, v_encoder}),
        shape_v2,
        b_special_zero);  // [B*H, N, K]

    // with_kv_cache
    auto k_cache = ngraph::pattern::any_input();        // [B, H, N-1, 64]
    auto v_cache = ngraph::pattern::any_input();        // [B, H, N-1, 64]
    auto kt = ngraph::pattern::any_input();                 // [B, 1, H*K]
    auto vt = ngraph::pattern::any_input();                 // [B, 1, H*K]
    auto kt1 = std::make_shared<ngraph::opset3::Reshape>(kt, ngraph::pattern::any_input(), b_special_zero); // [B, 1, H, K]
    auto vt1 = std::make_shared<ngraph::opset3::Reshape>(vt, ngraph::pattern::any_input(), b_special_zero); // [B, 1, H, K]
    auto order_0213_kt1 = ngraph::pattern::wrap_type<ngraph::opset4::Constant>();
    auto order_0213_vt1 = ngraph::pattern::wrap_type<ngraph::opset4::Constant>();
    auto kt2 = std::make_shared<ngraph::opset3::Transpose>(kt1, order_0213_kt1); // [B, H, 1, K]
    auto vt2 = std::make_shared<ngraph::opset3::Transpose>(vt1, order_0213_vt1); // [B, H, 1, K]

    auto newk = std::make_shared<ngraph::opset3::Concat>(ngraph::NodeVector{k_cache, kt2}, 2); // [B,H, N,64]
    auto k5 = std::make_shared<ngraph::opset3::Reshape>(newk, ngraph::pattern::any_input(), b_special_zero);            // [B*H, N, K]

    auto newv = std::make_shared<ngraph::opset3::Concat>(ngraph::NodeVector{v_cache, vt2}, 2); // [B,H, N,64]
    auto v5 = std::make_shared<ngraph::opset3::Reshape>(newv, ngraph::pattern::any_input(), b_special_zero);            // [B*H, N, K]

    auto qe = q3->output(0);
    // kv may from decoder's past_key_values input
    auto ke = std::make_shared<ngraph::pattern::op::Or>(OutputVector{k3->output(0), k5->output(0)});
    auto ve = std::make_shared<ngraph::pattern::op::Or>(OutputVector{v3->output(0), v5->output(0)});

    auto qk = std::make_shared<ngraph::opset3::MatMul>(qe, ke);               // [B*H, M, N]

    // alternative path, with causal mask
    auto qk1 = std::make_shared<ngraph::opset3::Reshape>(qk, ngraph::pattern::any_input(), b_special_zero);
    auto add_causal_mask = std::make_shared<ngraph::opset3::Add>(qk1, ngraph::pattern::any_input());
    auto qk3 = std::make_shared<ngraph::opset3::Reshape>(add_causal_mask, ngraph::pattern::any_input(), b_special_zero);

    auto w = std::make_shared<ngraph::opset1::Softmax>(
        std::make_shared<ngraph::pattern::op::Or>(OutputVector{qk, qk3})
    ); // [B*H, M, N]

    auto wv1 = std::make_shared<ngraph::opset3::MatMul>(w, ve);                             // [B*H, M, K]
    auto wv2 = std::make_shared<ngraph::opset3::Reshape>(wv1, shape_wv1, b_special_zero);   // [B, H, M, K]
    auto wv3 = std::make_shared<ngraph::opset3::Transpose>(wv2, order_0213_wv);             // [B, M, H, K]
    auto wv4 = std::make_shared<ngraph::opset3::Reshape>(wv3, shape_wv2, b_special_zero);   // [B, M, H*K]

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        auto& pvmap = m.get_pattern_value_map();
        auto check_order_0213 = [&](Output<Node> & out) {
            auto order = ngraph::as_type_ptr<ngraph::opset4::Constant>(out.get_node_shared_ptr())->cast_vector<int>();
            return (order == std::vector<int>({0, 2, 1, 3}));
        };

        if (!check_order_0213(pvmap[order_0213_q]))
            return false;
        if (pvmap.count(order_0213_k) && !check_order_0213(pvmap[order_0213_k]))
            return false;
        if (pvmap.count(order_0213_v) && !check_order_0213(pvmap[order_0213_v]))
            return false;
        if (!check_order_0213(pvmap[order_0213_wv]))
            return false;
        // replace
        //std::cout << pvmap[w].get_node_shared_ptr()->get_friendly_name() << " is found!" << std::endl;

        std::shared_ptr<ov::intel_cpu::MHA2Node> mha;
        bool with_causal_mask = (pvmap.find(add_causal_mask) != pvmap.end());
        bool with_kv_cache = (pvmap.find(k_cache) != pvmap.end());

        if (!with_kv_cache) {
            // no kv_cache concat path
            Output<Node> k_input;
            Output<Node> v_input;
            bool kv_head_transposed;
            if (pvmap.count(k_encoder) && pvmap.count(v_encoder)) {
                // k & v from encoder [B, H, N, 64]
                k_input = pvmap[k_encoder];
                v_input = pvmap[v_encoder];
                kv_head_transposed = true;
            } else if (pvmap.count(k) && pvmap.count(v)) {
                // k & v from FC: [B, N, H*K]
                k_input = pvmap[k];
                v_input = pvmap[v];
                kv_head_transposed = false;
            }

            mha = std::make_shared<ov::intel_cpu::MHA2Node>(pvmap[q],
                                                            k_input,
                                                            v_input,
                                                            with_causal_mask,
                                                            kv_head_transposed,
                                                            m.get_match_root()->get_friendly_name());

            ngraph::copy_runtime_info({pvmap.at(qk).get_node_shared_ptr(),
                                       pvmap.at(w).get_node_shared_ptr(),
                                       pvmap.at(wv1).get_node_shared_ptr()},
                                       mha);
        } else {
            // with kv_cache concat path
            mha = std::make_shared<ov::intel_cpu::MHA2Node>(pvmap[q],
                                            pvmap[kt], pvmap[vt],
                                            pvmap[k_cache], pvmap[v_cache],
                                            with_causal_mask,
                                            m.get_match_root()->get_friendly_name());

            ngraph::copy_runtime_info({pvmap.at(qk).get_node_shared_ptr(),
                                       pvmap.at(w).get_node_shared_ptr(),
                                       pvmap.at(wv1).get_node_shared_ptr()},
                                       mha);
        }

        if (transformation_callback(mha))
            return false;

        if (std::getenv("SKIP_MHA2") && atoi(std::getenv("SKIP_MHA2")))
            return false;
        ngraph::replace_node(m.get_match_root(), {mha->output(0)});

        if (with_kv_cache) {
            ngraph::replace_node(pvmap.at(newk).get_node_shared_ptr(), {mha->output(1)});
            ngraph::replace_node(pvmap.at(newv).get_node_shared_ptr(), {mha->output(2)});
            // auto& newk_tensor = mha->output(1).get_tensor();
            // auto& newv_tensor = mha->output(2).get_tensor();
            // ov::descriptor::set_ov_tensor_legacy_name(newk_tensor, pvmap.at(newk).get_node_shared_ptr()->get_friendly_name());
            // ov::descriptor::set_ov_tensor_legacy_name(newv_tensor, pvmap.at(newv).get_node_shared_ptr()->get_friendly_name());
        }
        return false;
    };
    auto m = std::make_shared<ngraph::pattern::Matcher>(wv4, matcher_name);
    this->register_matcher(m, callback);
}

// TODO: draw pattern
ov::intel_cpu::MHAQuantFusion::MHAQuantFusion() {
    MATCHER_SCOPE(MHAQuantFusion);

    auto in0 = ngraph::pattern::any_input(ngraph::pattern::has_static_shape());
    auto in1 = ngraph::pattern::any_input(ngraph::pattern::has_static_shape());
    auto in2 = ngraph::pattern::wrap_type<ngraph::opset4::Constant>();
    auto in3 = ngraph::pattern::any_input(ngraph::pattern::has_static_shape());
    auto in4 = ngraph::pattern::wrap_type<ngraph::opset4::Constant>();
    auto in5 = ngraph::pattern::wrap_type<ngraph::opset4::Constant>();
    auto in6 = ngraph::pattern::wrap_type<ngraph::opset4::Constant>();
    auto in7 = ngraph::pattern::wrap_type<ngraph::opset4::Constant>();
    auto in8 = ngraph::pattern::any_input(ngraph::pattern::has_static_shape());
    auto in9 = ngraph::pattern::wrap_type<ngraph::opset4::Constant>();
    auto in10 = ngraph::pattern::wrap_type<ngraph::opset4::Constant>();
    auto transpose0 = std::make_shared<ngraph::opset3::Transpose>(in0, in4);
    auto transpose1 = std::make_shared<ngraph::opset3::Transpose>(in1, in5);
    auto matmul0 = std::make_shared<ngraph::opset3::MatMul>(transpose0, transpose1);
    auto fakeQuantize0 = ngraph::pattern::wrap_type<ngraph::opset1::FakeQuantize>({matmul0,
                                                                                   ngraph::pattern::wrap_type<ngraph::opset4::Constant>(),
                                                                                   ngraph::pattern::wrap_type<ngraph::opset4::Constant>(),
                                                                                   ngraph::pattern::wrap_type<ngraph::opset4::Constant>(),
                                                                                   ngraph::pattern::wrap_type<ngraph::opset4::Constant>()});
    auto add = std::make_shared<ngraph::opset4::Add>(fakeQuantize0, in3);
    auto mul = std::make_shared<ngraph::opset3::Multiply>(add, in2);
    auto reshape0 = std::make_shared<ngraph::opset1::Reshape>(mul, in6, true);
    auto softmax = std::make_shared<ngraph::opset1::Softmax>(reshape0);
    auto reshape1 = std::make_shared<ngraph::opset1::Reshape>(softmax, in7, true);
    auto fakeQuantize1 = ngraph::pattern::wrap_type<ngraph::opset1::FakeQuantize>({reshape1,
                                                                                   ngraph::pattern::wrap_type<ngraph::opset4::Constant>(),
                                                                                   ngraph::pattern::wrap_type<ngraph::opset4::Constant>(),
                                                                                   ngraph::pattern::wrap_type<ngraph::opset4::Constant>(),
                                                                                   ngraph::pattern::wrap_type<ngraph::opset4::Constant>()});
    auto transpose2 = std::make_shared<ngraph::opset3::Transpose>(in8, in9);
    auto matmul1 = std::make_shared<ngraph::opset3::MatMul>(fakeQuantize1, transpose2);
    auto fakeQuantize2 = ngraph::pattern::wrap_type<ngraph::opset1::FakeQuantize>({matmul1,
                                                                                   ngraph::pattern::wrap_type<ngraph::opset4::Constant>(),
                                                                                   ngraph::pattern::wrap_type<ngraph::opset4::Constant>(),
                                                                                   ngraph::pattern::wrap_type<ngraph::opset4::Constant>(),
                                                                                   ngraph::pattern::wrap_type<ngraph::opset4::Constant>()});
    auto transpose3 = std::make_shared<ngraph::opset3::Transpose>(fakeQuantize2, in10);

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        auto& pattern_to_output = m.get_pattern_value_map();
        auto transpose0_in = pattern_to_output.at(in0);
        auto transpose1_in = pattern_to_output.at(in1);
        auto add_in1 = pattern_to_output.at(in3);
        auto transpose2_in = pattern_to_output.at(in8);

        if (transpose0_in.get_shape() != transpose1_in.get_shape() || transpose0_in.get_shape() != transpose2_in.get_shape()) {
            return false;
        }

        if (transpose0_in.get_shape().size() != 4) {
            return false;
        }

        auto expected_add_shape = Shape({transpose0_in.get_shape()[0], 1, 1, transpose0_in.get_shape()[1]});
        if (add_in1.get_shape() != expected_add_shape) {
            return false;
        }

        std::vector<float> mul_scales;
        if (auto mul_node = ngraph::as_type_ptr<ngraph::opset3::Multiply>(pattern_to_output.at(mul).get_node_shared_ptr())) {
            mul_scales = ngraph::as_type_ptr<ngraph::opset4::Constant>(mul_node->get_input_node_shared_ptr(1))->cast_vector<float>();

            auto expected_shape = ngraph::Shape({1, transpose0_in.get_shape()[2], 1, 1});
            if (mul_scales.size() != 1 && mul_node->get_input_shape(1) != expected_shape) {
                return false;
            }
        } else {
            return false;
        }

        if (!valid_transpose_order(pattern_to_output.at(in4).get_node_shared_ptr(), {0, 2, 1, 3})) return false;
        if (!valid_transpose_order(pattern_to_output.at(in5).get_node_shared_ptr(), {0, 2, 3, 1})) return false;
        if (!valid_transpose_order(pattern_to_output.at(in9).get_node_shared_ptr(), {0, 2, 1, 3})) return false;
        if (!valid_transpose_order(pattern_to_output.at(in10).get_node_shared_ptr(), {0, 2, 1, 3})) return false;

        auto matmul0_node = ngraph::as_type_ptr<ngraph::opset3::MatMul>(pattern_to_output.at(matmul0).get_node_shared_ptr());
        if (!matmul0_node)
            return false;
        if (matmul0_node->get_transpose_a() || matmul0_node->get_transpose_b())
            return false;

        std::vector<float> fq0_scale;
        auto fq0_node = ngraph::as_type_ptr<ngraph::opset1::FakeQuantize>(pattern_to_output.at(fakeQuantize0).get_node_shared_ptr());
        if (fq0_node) {
            fq0_scale = simplifyToScale(fq0_node);
            if (!fq0_scale.size())
                return false;
        }

        auto reshape0_node = ngraph::as_type_ptr<ngraph::opset1::Reshape>(pattern_to_output.at(reshape0).get_node_shared_ptr());
        if (!reshape0_node)
            return false;

        if (auto reshape_pattern = ngraph::as_type_ptr<ngraph::opset4::Constant>(pattern_to_output.at(in6).get_node_shared_ptr())) {
            if (reshape0_node->get_input_shape(0).size() != 4) {
                return false;
            }

            std::vector<int64_t> reshapeConstData = {static_cast<int64_t>(reshape0_node->get_input_shape(0)[0] *
                                                                          reshape0_node->get_input_shape(0)[1] *
                                                                          reshape0_node->get_input_shape(0)[2]),
                                                     -1};

            if (reshape_pattern->cast_vector<int64_t>() != reshapeConstData) {
                return false;
            }
        } else {
            return false;
        }

        if (auto reshape1_node = ngraph::as_type_ptr<ngraph::opset1::Reshape>(pattern_to_output.at(reshape1).get_node_shared_ptr())) {
            if (reshape0_node->get_input_shape(0) != reshape1_node->get_output_shape(0)) {
                return false;
            }
        } else {\
            return false;
        }

        auto softmax_node = ngraph::as_type_ptr<ngraph::opset1::Softmax>(pattern_to_output.at(softmax).get_node_shared_ptr());
        if (!softmax_node)
            return false;
        if (softmax_node->get_axis() != 1)
            return false;

        std::vector<float> fq1_scale;
        auto fq1_node = ngraph::as_type_ptr<ngraph::opset1::FakeQuantize>(pattern_to_output.at(fakeQuantize1).get_node_shared_ptr());
        if (fq1_node) {
            fq1_scale = simplifyToScale(fq1_node);
            if (!fq1_scale.size())
                return false;
        } else {
            return false;
        }

        auto matmul1_node = ngraph::as_type_ptr<ngraph::opset3::MatMul>(pattern_to_output.at(matmul1).get_node_shared_ptr());
        if (!matmul1_node)
            return false;
        if (matmul1_node->get_transpose_a() || matmul1_node->get_transpose_b())
            return false;

        std::vector<float> fq2_scale;
        if (auto fq_node = ngraph::as_type_ptr<ngraph::opset1::FakeQuantize>(pattern_to_output.at(fakeQuantize2).get_node_shared_ptr())) {
            fq2_scale = simplifyToScale(fq_node);
            if (!fq2_scale.size())
                return false;
        }

        bool is_mul_first = false;
        auto transpose3_node = pattern_to_output.at(transpose3).get_node_shared_ptr();
        auto mha = std::make_shared<ov::intel_cpu::MHANode>(transpose0_in, transpose1_in, add_in1, transpose2_in, mul_scales, is_mul_first,
                                                            std::vector<float>(), fq0_scale, fq1_scale, fq2_scale,
                                                            ngraph::element::undefined,
                                                            fq0_node ? fq0_node->get_output_element_type(0) : ngraph::element::undefined,
                                                            fq1_node->get_output_element_type(0), transpose3_node->get_output_element_type(0));
        mha->set_friendly_name(m.get_match_root()->get_friendly_name());
        ngraph::copy_runtime_info({pattern_to_output.at(transpose0).get_node_shared_ptr(),
                                   pattern_to_output.at(transpose1).get_node_shared_ptr(),
                                   pattern_to_output.at(matmul0).get_node_shared_ptr(),
                                   pattern_to_output.at(fakeQuantize0).get_node_shared_ptr(),
                                   pattern_to_output.at(add).get_node_shared_ptr(),
                                   pattern_to_output.at(mul).get_node_shared_ptr(),
                                   pattern_to_output.at(reshape0).get_node_shared_ptr(),
                                   pattern_to_output.at(softmax).get_node_shared_ptr(),
                                   pattern_to_output.at(reshape1).get_node_shared_ptr(),
                                   pattern_to_output.at(fakeQuantize1).get_node_shared_ptr(),
                                   pattern_to_output.at(transpose2).get_node_shared_ptr(),
                                   pattern_to_output.at(matmul1).get_node_shared_ptr(),
                                   pattern_to_output.at(fakeQuantize2).get_node_shared_ptr(),
                                   pattern_to_output.at(transpose3).get_node_shared_ptr(),
                                  },
                                  mha);

        if (transformation_callback(mha)) {
            return false;
        }

        ngraph::replace_node(m.get_match_root(), mha);

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(transpose3, matcher_name);
    this->register_matcher(m, callback);
}

// TODO: draw pattern
ov::intel_cpu::MHAQuantFusion2::MHAQuantFusion2() {
    MATCHER_SCOPE(MHAQuantFusion2);

    auto in0 = ngraph::pattern::any_input(ngraph::pattern::has_static_shape());
    auto in1 = ngraph::pattern::any_input(ngraph::pattern::has_static_shape());
    auto in2 = ngraph::pattern::wrap_type<ngraph::opset4::Constant>();
    auto in3 = ngraph::pattern::any_input(ngraph::pattern::has_static_shape());
    auto in4 = ngraph::pattern::wrap_type<ngraph::opset4::Constant>();
    auto in5 = ngraph::pattern::wrap_type<ngraph::opset4::Constant>();
    auto in8 = ngraph::pattern::any_input(ngraph::pattern::has_static_shape());
    auto in9 = ngraph::pattern::wrap_type<ngraph::opset4::Constant>();
    auto in10 = ngraph::pattern::wrap_type<ngraph::opset4::Constant>();
    auto transpose0 = std::make_shared<ngraph::opset3::Transpose>(in0, in4);
    auto transpose1 = std::make_shared<ngraph::opset3::Transpose>(in1, in5);
    auto fakeQuantize0 = ngraph::pattern::wrap_type<ngraph::opset1::FakeQuantize>({transpose1,
                                                                                ngraph::pattern::wrap_type<ngraph::opset4::Constant>(),
                                                                                ngraph::pattern::wrap_type<ngraph::opset4::Constant>(),
                                                                                ngraph::pattern::wrap_type<ngraph::opset4::Constant>(),
                                                                                ngraph::pattern::wrap_type<ngraph::opset4::Constant>()});
    auto matmul0 = std::make_shared<ngraph::opset3::MatMul>(transpose0, fakeQuantize0);
    auto mul = std::make_shared<ngraph::opset3::Multiply>(matmul0, in2);
    auto add = std::make_shared<ngraph::opset4::Add>(mul, in3);
    auto softmax = std::make_shared<ngraph::opset1::Softmax>(add);
    auto transpose2 = std::make_shared<ngraph::opset3::Transpose>(in8, in9);
    auto matmul1 = std::make_shared<ngraph::opset3::MatMul>(softmax, transpose2);
    auto fakeQuantize1 = ngraph::pattern::wrap_type<ngraph::opset1::FakeQuantize>({matmul1,
                                                                                   ngraph::pattern::wrap_type<ngraph::opset4::Constant>(),
                                                                                   ngraph::pattern::wrap_type<ngraph::opset4::Constant>(),
                                                                                   ngraph::pattern::wrap_type<ngraph::opset4::Constant>(),
                                                                                   ngraph::pattern::wrap_type<ngraph::opset4::Constant>()});
    auto transpose3 = std::make_shared<ngraph::opset3::Transpose>(fakeQuantize1, in10);

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        auto& pattern_to_output = m.get_pattern_value_map();
        auto transpose0_in = pattern_to_output.at(in0);
        auto transpose1_in = pattern_to_output.at(in1);
        auto add_in1 = pattern_to_output.at(in3);
        auto transpose2_in = pattern_to_output.at(in8);

        if (transpose0_in.get_shape() != transpose1_in.get_shape() || transpose0_in.get_shape() != transpose2_in.get_shape()) {
            return false;
        }

        if (transpose0_in.get_shape().size() != 4) {
            return false;
        }

        auto expected_add_shape = Shape({transpose0_in.get_shape()[0], 1, 1, transpose0_in.get_shape()[1]});
        if (add_in1.get_shape() != expected_add_shape) {
            return false;
        }

        std::vector<float> mul_scales;
        if (auto mul_node = ngraph::as_type_ptr<ngraph::opset3::Multiply>(pattern_to_output.at(mul).get_node_shared_ptr())) {
            mul_scales = ngraph::as_type_ptr<ngraph::opset4::Constant>(mul_node->get_input_node_shared_ptr(1))->cast_vector<float>();

            auto expected_shape = ngraph::Shape({1, transpose0_in.get_shape()[2], 1, 1});
            if (mul_scales.size() != 1 && mul_node->get_input_shape(1) != expected_shape) {
                return false;
            }
        } else {
            return false;
        }

        if (!valid_transpose_order(pattern_to_output.at(in4).get_node_shared_ptr(), {0, 2, 1, 3})) return false;
        if (!valid_transpose_order(pattern_to_output.at(in5).get_node_shared_ptr(), {0, 2, 3, 1})) return false;
        if (!valid_transpose_order(pattern_to_output.at(in9).get_node_shared_ptr(), {0, 2, 1, 3})) return false;
        if (!valid_transpose_order(pattern_to_output.at(in10).get_node_shared_ptr(), {0, 2, 1, 3})) return false;

        auto matmul0_node = ngraph::as_type_ptr<ngraph::opset3::MatMul>(pattern_to_output.at(matmul0).get_node_shared_ptr());
        if (!matmul0_node)
            return false;
        if (matmul0_node->get_transpose_a() || matmul0_node->get_transpose_b())
            return false;

        std::vector<float> fq0_scale;
        auto fq0_node = ngraph::as_type_ptr<ngraph::opset1::FakeQuantize>(pattern_to_output.at(fakeQuantize0).get_node_shared_ptr());
        if (fq0_node) {
            fq0_scale = simplifyToScale(fq0_node);
            if (!fq0_scale.size())
                return false;
        } else {
            return false;
        }

        auto softmax_node = ngraph::as_type_ptr<ngraph::opset1::Softmax>(pattern_to_output.at(softmax).get_node_shared_ptr());
        if (!softmax_node)
            return false;
        if (softmax_node->get_axis() != 3)
            return false;

        std::vector<float> fq1_scale;
        if (auto fq_node = ngraph::as_type_ptr<ngraph::opset1::FakeQuantize>(pattern_to_output.at(fakeQuantize1).get_node_shared_ptr())) {
            fq1_scale = simplifyToScale(fq_node);
            if (!fq1_scale.size())
                return false;
        }

        auto matmul1_node = ngraph::as_type_ptr<ngraph::opset3::MatMul>(pattern_to_output.at(matmul1).get_node_shared_ptr());
        if (!matmul1_node)
            return false;
        if (matmul1_node->get_transpose_a() || matmul1_node->get_transpose_b())
            return false;

        bool is_mul_first = true;
        auto transpose3_node = pattern_to_output.at(transpose3).get_node_shared_ptr();
        auto mha = std::make_shared<ov::intel_cpu::MHANode>(transpose0_in, transpose1_in, add_in1, transpose2_in, mul_scales, is_mul_first,
                                                            fq0_scale, std::vector<float>(), std::vector<float>(), fq1_scale,
                                                            fq0_node->get_output_element_type(0), ngraph::element::undefined, ngraph::element::undefined,
                                                            transpose3_node->get_output_element_type(0));
        mha->set_friendly_name(m.get_match_root()->get_friendly_name());
        ngraph::copy_runtime_info({pattern_to_output.at(transpose0).get_node_shared_ptr(),
                                   pattern_to_output.at(transpose1).get_node_shared_ptr(),
                                   pattern_to_output.at(fakeQuantize0).get_node_shared_ptr(),
                                   pattern_to_output.at(matmul0).get_node_shared_ptr(),
                                   pattern_to_output.at(mul).get_node_shared_ptr(),
                                   pattern_to_output.at(add).get_node_shared_ptr(),
                                   pattern_to_output.at(softmax).get_node_shared_ptr(),
                                   pattern_to_output.at(transpose2).get_node_shared_ptr(),
                                   pattern_to_output.at(matmul1).get_node_shared_ptr(),
                                   pattern_to_output.at(fakeQuantize1).get_node_shared_ptr(),
                                   pattern_to_output.at(transpose3).get_node_shared_ptr(),
                                  },
                                  mha);

        if (transformation_callback(mha)) {
            return false;
        }

        ngraph::replace_node(m.get_match_root(), mha);

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(transpose3, matcher_name);
    this->register_matcher(m, callback);
}
