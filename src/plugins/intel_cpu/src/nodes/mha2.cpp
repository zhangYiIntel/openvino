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
    verbose = (std::getenv("MHA2VERBOSE") && atoi(std::getenv("MHA2VERBOSE")));
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

// A preprocessor argument counter
#define COUNT(...) COUNT_I(__VA_ARGS__, 9, 8, 7, 6, 5, 4, 3, 2, 1,)
#define COUNT_I(_9,_8,_7,_6,_5,_4,_3,_2,_1,X,...) X
// Preprocessor paster
#define _GLUE(A,B) _GLUE_I(A,B)
#define _GLUE_I(A,B) A##B
// chained caller
#define NAMED_VALUES(...) _GLUE(NAMED_VALUES_,COUNT(__VA_ARGS__))(__VA_ARGS__)
// chain
#define NAMED_VALUES_1(a) #a,":",a
#define NAMED_VALUES_2(a,...) #a,":",a," ",NAMED_VALUES_1(__VA_ARGS__)
#define NAMED_VALUES_3(a,...) #a,":",a," ",NAMED_VALUES_2(__VA_ARGS__)
#define NAMED_VALUES_4(a,...) #a,":",a," ",NAMED_VALUES_3(__VA_ARGS__)
#define NAMED_VALUES_5(a,...) #a,":",a," ",NAMED_VALUES_4(__VA_ARGS__)
#define NAMED_VALUES_6(a,...) #a,":",a," ",NAMED_VALUES_5(__VA_ARGS__)
#define NAMED_VALUES_7(a,...) #a,":",a," ",NAMED_VALUES_6(__VA_ARGS__)
#define NAMED_VALUES_8(a,...) #a,":",a," ",NAMED_VALUES_7(__VA_ARGS__)
#define NAMED_VALUES_9(a,...) #a,":",a," ",NAMED_VALUES_8(__VA_ARGS__)

#ifndef UNUSED
#define UNUSED(x) (void)(x)
#endif

template<typename ... Args>
static void log(Args&& ... args) {
    std::stringstream ss;
    int dummy[] = {(ss << std::forward<Args>(args), 0)...};
    UNUSED(dummy);
    ss << std::endl;
    std::cout << ss.str();
}

#define NAMED_LOG(prefix, ...) log(prefix, NAMED_VALUES(__VA_ARGS__))

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

    MemoryPtr final_k;
    MemoryPtr final_v;

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
    auto shape_q = mem_q->getStaticDims(); // [B, M, H*K]
    int B = shape_q[0];
    int M = shape_q[1];
    int HK = shape_q[2];
    int K = 64;
    int H = HK/K;
    int N;    
    if (kv_head_transposed) {
        // k&v: [B, H, N, K]
        N = final_k->getStaticDims()[2];
    } else {
        // k&v: [B, N, H*K]
        N = final_k->getStaticDims()[1];
    }

    auto pQ = reinterpret_cast<float*>(mem_q->GetPtr());
    auto pK = reinterpret_cast<float*>(final_k->GetPtr());
    auto pV = reinterpret_cast<float*>(final_v->GetPtr());
    auto pWV = reinterpret_cast<float*>(mem_wv->GetPtr());

    bool splitN = parallel_get_max_threads() > B*H;

    if (verbose) {
        NAMED_LOG("MHA2::execute", getName(), *mha2);
        NAMED_LOG("\t",with_causal_mask, with_kv_cache, kv_head_transposed, parallel_get_max_threads());
        NAMED_LOG("\t",B,M,N,H,K);
        NAMED_LOG("\t",splitN);
        NAMED_LOG("\t",mem_q->getDesc());
        NAMED_LOG("\t",mem_k->getDesc());
        NAMED_LOG("\t",mem_v->getDesc());
        if (mem_pastk) NAMED_LOG("\t",mem_pastk->getDesc());
        if (mem_pastv) NAMED_LOG("\t",mem_pastv->getDesc());
        NAMED_LOG("\t",mem_wv->getDesc());
        if (mem_presentk) NAMED_LOG("\t",mem_presentk->getDesc());
        if (mem_presentv) NAMED_LOG("\t",mem_presentv->getDesc());
    }
    // q: [B, M, H*k]
    if (kv_head_transposed) {
        // k&v: [B, H, N, K]
        int N = final_k->getStaticDims()[2];
        int stride_b_q = M*H*K;
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
            kernels.one_head_attention(tid, q, k, v, wv, 0, with_causal_mask);
        });
    } else {
        // with_causal_mask:0   with_kv_cache:0   kv_head_transposed:0
        // parallel in B/H/M dimensions
        // k&v: [B, N, H*K]
        int N = final_k->getStaticDims()[1];
        int stride_bytes_hk = H*K*sizeof(float);

        // kernel register blocking on 6 rows
        const size_t work_amount = (size_t)B*H * M;

        auto coord2D = [&](size_t index, size_t D0, size_t D1, size_t &i0, size_t &i1) {
            // i = d0 * D1 + d1
            i0 = index/D1;
            i1 = index - i0*D1;
        };

        parallel_nt(0, [&](int ithr, int nthr) {
            size_t start{0}, end{0};
            splitter(work_amount, nthr, ithr, start, end);
            size_t bh0, mb0;
            size_t bh1, mb1;
            if (start == end) return;
            coord2D(start, B*H, M, bh0, mb0);
            coord2D(end, B*H, M, bh1, mb1);
            auto m_start = mb0;
            auto m_end = std::min(size_t(M), mb1);

            // first head
            auto m0 = m_start;
            auto m1 = m0;
            // bh = b*H + h
            for(auto bh = bh0; bh <= bh1; bh++) {
                // determine m1 for current head
                m1 = (bh == bh1) ? m_end : M;
                if (m1 <= m0) break;

                //  q[b, m0:m1, h, K] => (m1-m0)xK
                //  k[b, 0:N,   h, K] => NxK
                //  v[b, 0:N,   h, K] => NxK
                // wv[b, m0:m1, h, K] => (m1-m0)xK
                auto b = bh/H;
                auto h = bh - b*H;
                auto kv_off = b*(N*H*K) + h*K;
                auto q_off = b*(M*H*K) + m0*(H*K) + h*K;
                tensor2D<float> q(m1 - m0,  K, pQ + q_off, stride_bytes_hk);
                tensor2D<float> k(N,        K, pK + kv_off, stride_bytes_hk);
                tensor2D<float> v(N,        K, pV + kv_off, stride_bytes_hk);
                tensor2D<float> wv(m1 - m0, K, pWV + q_off, stride_bytes_hk);
                kernels.one_head_attention(ithr, q, k, v, wv, m0, with_causal_mask);

                // m0 for next head is always 0
                m0 = 0;
            }
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
