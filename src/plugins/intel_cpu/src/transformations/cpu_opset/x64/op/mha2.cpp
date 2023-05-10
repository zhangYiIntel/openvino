// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mha2.hpp"

#include <matmul_shape_inference.hpp>
#include <ngraph/opsets/opset3.hpp>

#include "transformations/itt.hpp"

ov::intel_cpu::MHA2Node::MHA2Node(const ngraph::Output<ngraph::Node>& q,
                                  const ngraph::Output<ngraph::Node>& k,
                                  const ngraph::Output<ngraph::Node>& v,
                                  bool is_causal,
                                  bool kv_head_transposed,
                                  const std::string& name)
    : Op({q, k, v}),
      is_causal(is_causal),
      with_kv_cache(false),
      kv_head_transposed(kv_head_transposed) {
    /*
    set_friendly_name(name);
    std::cout << "    q: " << q.get_node_shared_ptr()->get_friendly_name() << std::endl;
    std::cout << "    k: " << k.get_node_shared_ptr()->get_friendly_name() << std::endl;
    std::cout << "    v: " << v.get_node_shared_ptr()->get_friendly_name() << std::endl;
    */
    validate_and_infer_types();
}

ov::intel_cpu::MHA2Node::MHA2Node(const ngraph::Output<ngraph::Node>& q,
                                  const ngraph::Output<ngraph::Node>& k,
                                  const ngraph::Output<ngraph::Node>& v,
                                  const ngraph::Output<ngraph::Node>& pastk,
                                  const ngraph::Output<ngraph::Node>& pastv,
                                  bool is_causal,
                                  const std::string& name)
    : Op({q, k, v, pastk, pastv}),
      is_causal(is_causal),
      with_kv_cache(true),
      kv_head_transposed(false) {
    /*
    set_friendly_name(name);
    std::cout << "    q: " << q.get_node_shared_ptr()->get_friendly_name() << std::endl;
    std::cout << "    k: " << k.get_node_shared_ptr()->get_friendly_name() << std::endl;
    std::cout << "    v: " << v.get_node_shared_ptr()->get_friendly_name() << std::endl;
    std::cout << "    pastk: " << pastk.get_node_shared_ptr()->get_friendly_name() << std::endl;
    std::cout << "    pastv: " << pastv.get_node_shared_ptr()->get_friendly_name() << std::endl;
    */
    validate_and_infer_types();
}

std::shared_ptr<ngraph::Node> ov::intel_cpu::MHA2Node::clone_with_new_inputs(
    const ngraph::OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(MHA2Node_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    if (with_kv_cache) {
        return std::make_shared<ov::intel_cpu::MHA2Node>(new_args.at(0),
                                                         new_args.at(1),
                                                         new_args.at(2),
                                                         new_args.at(3),
                                                         new_args.at(4),
                                                         is_causal,
                                                         get_friendly_name());
    }
    return std::make_shared<ov::intel_cpu::MHA2Node>(new_args.at(0),
                                                     new_args.at(1),
                                                     new_args.at(2),
                                                     is_causal,
                                                     kv_head_transposed,
                                                     get_friendly_name());
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
void ov::intel_cpu::MHA2Node::validate_and_infer_types() {
    INTERNAL_OP_SCOPE(MHA2Node_validate_and_infer_types);

    auto shape_q = get_input_partial_shape(0);
    auto shape_k = get_input_partial_shape(1);
    auto shape_v = get_input_partial_shape(2);
    size_t q_rank = shape_q.size();
    size_t k_rank = shape_k.size();
    size_t v_rank = shape_v.size();

    NODE_VALIDATION_CHECK(this, (q_rank == 3), "expect shape of q to be [B, N, H*K].");
    if (kv_head_transposed) {
        NODE_VALIDATION_CHECK(this, (k_rank == 4), "expect shape of k to be [B, H, N, K].");
        NODE_VALIDATION_CHECK(this, (v_rank == 4), "expect shape of v to be [B, H, N, K].");
    } else {
        NODE_VALIDATION_CHECK(this, (k_rank == 3), "expect shape of k to be [B, N, H*K].");
        NODE_VALIDATION_CHECK(this, (v_rank == 3), "expect shape of v to be [B, N, H*K].");
    }
    set_output_type(0, get_input_element_type(0), shape_q);

    if (with_kv_cache) {
        auto shape_pastk = get_input_partial_shape(3);
        auto shape_pastv = get_input_partial_shape(4);
        size_t pastk_rank = shape_pastk.size();
        size_t pastv_rank = shape_pastv.size();
        NODE_VALIDATION_CHECK(this, (pastk_rank == 4), "rank of pastkeys is not 4.");
        NODE_VALIDATION_CHECK(this, (pastv_rank == 4), "rank of pastvalues is not 4.");

        // derive shape of presentk/presentv
        auto shape_presentk = shape_pastk;
        auto shape_presentv = shape_pastv;
        // concat:
        //  shape_pastk  : [B, H, N, K]
        //  shape_k      : [B, 1, H*K] view_as [B, H, 1, K]
        //  presentk     : [B, H, N+1, K]
        shape_presentk[2] += shape_k[1];
        shape_presentv[2] += shape_v[1];
        set_output_type(1, get_input_element_type(3), shape_presentk);
        set_output_type(2, get_input_element_type(4), shape_presentv);
    }
    /*
    std::cout << this->get_friendly_name()
                << "  q:" << get_input_partial_shape(0)
                << "  k:" << get_input_partial_shape(1)
                << "  v:" << get_input_partial_shape(2);
    if (with_kv_cache) {
        std::cout << " pastq:" << get_input_partial_shape(3)
                  << " pastk:" << get_input_partial_shape(4);
    }
    std::cout << " wv:" << get_output_partial_shape(0);
    if (with_kv_cache) {
        std::cout << " newq:" << get_output_partial_shape(1)
                  << " newk:" << get_output_partial_shape(2);
    }
    */
}

bool ov::intel_cpu::MHA2Node::visit_attributes(ngraph::AttributeVisitor& visitor) {
    INTERNAL_OP_SCOPE(MHA2Node_visit_attributes);
    visitor.on_attribute("causal", is_causal);
    return true;
}
