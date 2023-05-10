// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mha2.h"

#include <utils/general_utils.h>

#include <ie_ngraph_utils.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <string>
#include <vector>

#include "common/cpu_convert.h"
#include "common/cpu_memcpy.h"
#include "transformations/cpu_opset/x64/op/mha2.hpp"

using namespace InferenceEngine;
using namespace InferenceEngine::details;

namespace ov {
namespace intel_cpu {
namespace node {

bool MHA2::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto mha = std::dynamic_pointer_cast<const MHA2Node>(op);
        if (!mha) {
            errorMessage = "Only MHA2Node from CPU internal opset is supported";
            return false;
        }

        if (!(mha->get_input_element_type(0) == element::f32 && mha->get_input_element_type(1) == element::f32 &&
              mha->get_input_element_type(2) == element::f32)) {
            errorMessage = "Only support op with f32 inputs";
            return false;
        }
    } catch (...) {
        return false;
    }

    return true;
}

MHA2::MHA2(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr context)
    : Node(op, context, NgraphShapeInferFactory(op, EMPTY_PORT_MASK)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }

    mha2 = std::dynamic_pointer_cast<const MHA2Node>(op);
}

void MHA2::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    if (mha2->get_kv_cache()) {
        addSupportedPrimDesc(
            {
                {LayoutType::ncsp, Precision::FP32},
                {LayoutType::ncsp, Precision::FP32},
                {LayoutType::ncsp, Precision::FP32},
                {LayoutType::ncsp, Precision::FP32},    // pastk
                {LayoutType::ncsp, Precision::FP32},    // pastv
            },
            {
                {LayoutType::ncsp, Precision::FP32},
                {LayoutType::ncsp, Precision::FP32},    // newk
                {LayoutType::ncsp, Precision::FP32},    // newv
            },
            ref_any);
    } else {
        addSupportedPrimDesc(
            {
                {LayoutType::ncsp, Precision::FP32},
                {LayoutType::ncsp, Precision::FP32},
                {LayoutType::ncsp, Precision::FP32},
            },
            {{LayoutType::ncsp, Precision::FP32}},
            ref_any);
    }
}

/*
    q:{1, 1, 384} FP32 abc
    k:{1, 1, 384} FP32 abc
    v:{1, 1, 384} FP32 abc
    pastk:{1, 6, 4, 64} FP32 abcd
    pastv:{1, 6, 4, 64} FP32 abcd
    wv:{1, 1, 384} FP32 abc
    causal_mask: 0
    presentk: {1, 6, 5, 64} FP32 abcd
    presentv: {1, 6, 5, 64} FP32 abcd
*/
void MHA2::execute(dnnl::stream strm) {
    auto mem_q = getParentEdgeAt(0)->getMemoryPtr();
    auto mem_k = getParentEdgeAt(1)->getMemoryPtr();
    auto mem_v = getParentEdgeAt(2)->getMemoryPtr();
    auto mem_wv = getChildEdgeAt(0)->getMemoryPtr();
    MemoryPtr mem_pastk;
    MemoryPtr mem_pastv;
    MemoryPtr mem_presentk;
    MemoryPtr mem_presentv;

    bool with_kv_cache = mha2->get_kv_cache();
    bool with_causal_mask = mha2->get_causal_mask();
    bool kv_head_transposed = mha2->get_kv_head_transposed();
    if (with_kv_cache) {
        mem_pastk = getParentEdgeAt(3)->getMemoryPtr();
        mem_pastv = getParentEdgeAt(4)->getMemoryPtr();
        mem_presentk = getChildEdgeAt(1)->getMemoryPtr();
        mem_presentv = getChildEdgeAt(2)->getMemoryPtr();
    }
/*
    std::cout << "MHA2::execute " << getName() << std::endl;
    std::cout << "q:" << mem_q->getDesc() << std::endl;
    std::cout << "k:" << mem_k->getDesc() << std::endl;
    std::cout << "v:" << mem_v->getDesc() << std::endl;
    if (mem_pastk) std::cout << "pastk:" << mem_pastk->getDesc() << std::endl;
    if (mem_pastv) std::cout << "pastv:" << mem_pastv->getDesc() << std::endl;
    std::cout << "wv:" << mem_wv->getDesc() << std::endl;
    std::cout << "causal_mask: " << mha2->get_causal_mask() << std::endl;
    if (mem_presentk) std::cout << "presentk: " << mem_presentk->getDesc() << std::endl;
    if (mem_presentv) std::cout << "presentv: " << mem_presentv->getDesc() << std::endl;
*/
    MemoryPtr final_k;
    MemoryPtr final_v;

    auto shape_q = mem_q->getStaticDims(); // [B, M, H*K]
    int B = shape_q[0];
    int M = shape_q[1];
    int HK = shape_q[2];
    int K = 64;
    int H = HK/K;

    if (with_kv_cache) {
        mem_pastk = getParentEdgeAt(3)->getMemoryPtr();
        mem_pastv = getParentEdgeAt(4)->getMemoryPtr();
        mem_presentk = getChildEdgeAt(1)->getMemoryPtr();
        mem_presentv = getChildEdgeAt(2)->getMemoryPtr();

        final_k = mem_presentk;
        final_v = mem_presentv;
        kv_head_transposed = true;
    } else {
        // no kv_cache, kv_head_transposed can be true/false:
        //  true:  [B, H, N, K]
        //  false: [B, N, H*K]
        final_k = mem_k;
        final_v = mem_v;
    }

    auto pQ = reinterpret_cast<float*>(mem_q->GetPtr());
    auto pK = reinterpret_cast<float*>(final_k->GetPtr());
    auto pV = reinterpret_cast<float*>(final_v->GetPtr());
    auto pWV = reinterpret_cast<float*>(mem_wv->GetPtr());

    // q: [B, M, H*k]
    if (kv_head_transposed) {
        // k&v: [B, H, N, K]
        int N = final_k->getStaticDims()[2];
        int stride_b_q = M*H*K;
        int stride_b_kv = N*H*K;
        int stride_h_kv = N*K;
        int stride_h_q = K;
        int stride_bytes_hk = H*K*sizeof(float);
        int stride_bytes_K = K*sizeof(float);

        float* past_k;
        float* past_v;
        float* one_token_k;
        float* one_token_v;
        if (mem_pastk) past_k = reinterpret_cast<float*>(mem_pastk->GetPtr());
        if (mem_pastv) past_v = reinterpret_cast<float*>(mem_pastv->GetPtr());
        one_token_k = reinterpret_cast<float*>(mem_k->GetPtr());
        one_token_v = reinterpret_cast<float*>(mem_v->GetPtr());

        parallel_for2d(B, H, [&](size_t b, size_t h) {
            size_t tid = parallel_get_thread_num();
            auto bh = (b*H + h);
            auto * pheadK = pK + bh*N*K;
            auto * pheadV = pV + bh*N*K;
            if (with_kv_cache) {
                // generate presentk/presentv
                //  k,v : [B, 1, H*k] => [B, 1, H, K] => [B, H, 1, K]
                // pastk, paskv:                         [B, H, N-1, K]
                // presentk, presentv:                   [B, H, N, K]
                auto src1 = one_token_k + bh*K;
                auto src0 = past_k + bh*(N-1)*K;
                memcpy(pheadK, src0, (N-1)*K*sizeof(float));
                memcpy(pheadK + (N-1)*K, src1, K*sizeof(float));

                src1 = one_token_v + bh*K;
                src0 = past_v + bh*(N-1)*K;
                memcpy(pheadV, src0, (N-1)*K*sizeof(float));
                memcpy(pheadV + (N-1)*K, src1, K*sizeof(float));
            }
            //  q[b, 0:M, h, K] => MxK
            //  k[b, h, 0:N, K] => NxK
            //  v[b, h, 0:N, K] => NxK
            // wv[b, 0:M, h, K] => MxK
            tensor2D<float> q(M, K, pQ + b*stride_b_q + h*stride_h_q, stride_bytes_hk);
            tensor2D<float> k(N, K, pheadK, stride_bytes_K);
            tensor2D<float> v(N, K, pheadV, stride_bytes_K);
            tensor2D<float> wv(M, K, pWV + b*stride_b_q + h*stride_h_q, stride_bytes_hk);
            kernels.one_head_attention(tid, q, k, v, wv, with_causal_mask);
        });
    } else {
        // k&v: [B, N, H*K]
        int N = final_k->getStaticDims()[1];
        int stride_b_q = M*H*K;
        int stride_b_kv = N*H*K;
        int stride_bytes_hk = H*K*sizeof(float);
        int stride_h = K;
        parallel_for2d(B, H, [&](size_t b, size_t h) {
            size_t tid = parallel_get_thread_num();
            // M can also run in parallel, but since
            // it's range is small, if we run it in one core, it can share k
            //  q[b, 0:M, h, K] => MxK
            //  k[b, 0:N, h, K] => NxK
            //  v[b, 0:N, h, K] => NxK
            // wv[b, 0:M, h, K] => MxK
            tensor2D<float> q(M, K, pQ + b*stride_b_q + h*stride_h, stride_bytes_hk);
            tensor2D<float> k(N, K, pK + b*stride_b_kv + h*stride_h, stride_bytes_hk);
            tensor2D<float> v(N, K, pV + b*stride_b_kv + h*stride_h, stride_bytes_hk);
            tensor2D<float> wv(M, K, pWV + b*stride_b_q + h*stride_h, stride_bytes_hk);
            kernels.one_head_attention(tid, q, k, v, wv, with_causal_mask);
        });
    }
}

void MHA2::executeDynamicImpl(dnnl::stream strm) {
#if 0
    // shape infer & allocate output memory
    std::vector<VectorDims> outputShapes;
    auto mem_q = getParentEdgeAt(0)->getMemoryPtr();
    auto mem_k = getParentEdgeAt(1)->getMemoryPtr();
    auto mem_v = getParentEdgeAt(2)->getMemoryPtr();
    
    outputShapes.push_back(mem_q->getStaticDims());
    if (mha2->get_kv_cache()) {
        auto mem_pastk = getParentEdgeAt(3)->getMemoryPtr();
        auto mem_pastv = getParentEdgeAt(4)->getMemoryPtr();
        auto shape_curk = mem_pastk->getStaticDims();
        auto shape_curv = mem_pastv->getStaticDims();
        shape_curk[2] += mem_k->getStaticDims()[2];
        shape_curv[2] += mem_v->getStaticDims()[2];
        outputShapes.push_back(shape_curk);
        outputShapes.push_back(shape_curv);
    }
    Node::redefineOutputMemory(outputShapes);
#endif
    execute(strm);
}

bool MHA2::created() const {
    return getType() == Type::MHA2;
}

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
