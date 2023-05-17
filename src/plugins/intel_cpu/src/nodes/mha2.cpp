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

#include "utils/profiler.hpp"

#include <sys/types.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <sched.h>

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

static size_t offset2coord(size_t off, size_t D, size_t &d) {
    auto next_off = off/D;
    d = off - next_off*D;
    return next_off;
}

template<typename ... Args>
static size_t offset2coord(size_t off, size_t D, size_t &d, Args&& ... args) {
    off = offset2coord(off, std::forward<Args>(args)...);
    auto next_off = off/D;
    d = off - next_off*D;
    return next_off;
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

        static int bN = std::getenv("bN") ? atoi(std::getenv("bN")) : 256;
        static int sss = 0;
        // if (N > bN) sss ++;
        if (with_kv_cache || with_causal_mask || (N < bN) || (sss & 3)==2) {
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
            // no with_kv_cache, no with_causal_mask
            //
            // kernel register blocking on 6x16, so N is split in unit of 256 columns
            // that means each token will be encoded by 256 key/values and finally combined
            // if N is smaller than 256, we don't split them
            //
            //
            int num_sub_states = (N + bN - 1) / bN;
            const size_t work_amount = (size_t)(B * H) * num_sub_states;
            
            //NAMED_LOG("======",num_sub_states, M, N, work_amount);
            sub_states.resize(B * M * H * num_sub_states, K);
            qk_max.resize(1, work_amount*M);
            qk_sum.resize(1, work_amount*M);

            auto _prof = Profile("MHA2_exe");
            parallel_nt(0, [&](int ithr, int nthr) {
                // each work item is doing  M x bN sub-states encoding
                // and finally, main thread will combine sub-states into one
                size_t start{0}, end{0};
                splitter(work_amount, nthr, ithr, start, end);
                if (start == end)
                    return;
                //std::stringstream ss; ss << ithr << "/" << nthr << std::endl;   std::cout << ss.str();
                // encoding sub-states one by one
                for (auto cur = start; cur < end; cur++) {
                    size_t h;
                    size_t nb;
                    auto b = offset2coord(cur, H, h, num_sub_states, nb);
                    auto n0 = nb * bN;
                    auto n1 = std::min(size_t(N), n0 + bN);

                    //  q[b, 0:M, h, K]   => M x K
                    //  k[b, h, n0:n1, K] => bN x K
                    //  v[b, h, n0:n1, K] => bN x K
                    //  s[b, 0:M, h, nb, K] => M x K
                    // wv[b, 0:M, h, K] => M x K

                    tensor2D<float> q(M, K, pQ + b * stride_b_q + h * stride_h_q, stride_bytes_hk);
                    tensor2D<float> k(n1 - n0, K, pK + (b * H + h) * N * K + n0 * K, stride_bytes_K);
                    tensor2D<float> v(n1 - n0, K, pV + (b * H + h) * N * K + n0 * K, stride_bytes_K);
                    tensor2D<float> s(M,
                                      K,
                                      &sub_states(b * (M * H * num_sub_states) + h * (num_sub_states) + nb, 0),
                                      bN*H*K*sizeof(float));
                    // tensor2D<float> wv(M, K, s, stride_bytes_hk);
                    kernels.one_head_attention(ithr, q, k, v, s, 0, with_causal_mask, &qk_max[cur * M], &qk_sum[cur * M]);
                   
                    //NAMED_LOG("======",cur, b, h, nb, n0, n1, qk_max[cur * M], qk_sum[cur * M]);
                }
            });

            // combine sub-states
            auto _prof2 = Profile("combine");
            for(int b = 0; b<B; b++) {
                for(int h = 0; h<H; h++) {
                    //  s[b, 0:M, h, nb, K] => M x nb x K
                    // wv[b, 0:M, h, K] => M x K
                    // qk_max [b,h,nb,M]
                    tensor2D<float> wv(M, K, pWV + b*stride_b_q + h*stride_h_q, stride_bytes_hk);
                
                    auto cur = (b * H + h) * num_sub_states;

                    //NAMED_LOG("======",b, h, cur);
                    // qk_max: [b, h, 0:num_sub_states, 0:M]
                    // qk_sum: [b, h, 0:num_sub_states, 0:M]
                    float* p_qk_max = &qk_max[cur];  // num_sub_states x M
                    float* p_qk_sum = &qk_sum[cur];  // num_sub_states x M
                    for (int m = 0; m < M; m++) {
                        float * p_wv = &wv(m, 0);
                        // get weights of sub-states :
                        //    tmax = max_i(qk_max_i)
                        //    
                        //    tsum_i = sum_i * exp(qk_max_i - tmax)
                        //    weight_i = tsum_i/sum_i(tsum_i)
                        //float tmax = std::numeric_limits<float>::lowest();
                        auto tmax = _mm256_set1_ps(std::numeric_limits<float>::lowest());
                        for (int nb = 0; nb < num_sub_states; nb++) {
                            auto sub_max = _mm256_broadcast_ss(&p_qk_max[nb*M + m]);
                            tmax = _mm256_max_ps(tmax, sub_max);
                        }

                        //NAMED_LOG("======", tmax);
                        auto tsum = _mm256_setzero_ps();
                        for (int nb = 0; nb < num_sub_states; nb++) {
                            auto sub_max = _mm256_broadcast_ss(&p_qk_max[nb*M + m]);
                            sub_max = _mm256_sub_ps(sub_max, tmax);
                            auto sub_sum = _mm256_broadcast_ss(&p_qk_sum[nb*M + m]);
                            avx2::functional::exp_ps(sub_max);
                            sub_sum = _mm256_mul_ps(sub_sum, sub_max);
                            p_qk_sum[nb*M + m] = _mm256_cvtss_f32(sub_sum);
                            tsum = _mm256_add_ps(tsum, sub_sum);
                            //p_qk_sum[nb*M + m] *= std::exp(p_qk_max[nb*M + m] - tmax);
                            //tsum += p_qk_sum[nb*M + m];
                        }
                        //NAMED_LOG("======", tsum);
                        static __m256 one = _mm256_castsi256_ps(_mm256_set1_epi32(0x3f800000)); // 1.0f
                        auto tweight_recip = _mm256_div_ps(one, tsum);                          // 1/sum_exp

                        // linear combine sub-states, 
                        __m256i wv_mask = _mm256_setzero_si256();
                        for (int nb = 0; nb < num_sub_states; nb++) {
                            //
                            float* p_sub = &sub_states(b * (M * H * num_sub_states) + h * (num_sub_states) + nb, 0);

                            auto x_weight = _mm256_broadcast_ss(&p_qk_sum[nb*M + m]);
                            x_weight = _mm256_mul_ps(x_weight, tweight_recip);
                            //auto x_weight = _mm256_set1_ps(p_qk_sum[nb*M + m] * tweight_recip);
                            // wv = substates * weight    for nb=0
                            // wv += substates * weight   otherwise
                            if (nb == 1) wv_mask = avx2::functional::get_mask(8);
                            int k;
                            for(k = 0; (k+8) <= K; k += 8) {
                                auto x_sub = _mm256_loadu_ps(p_sub + k);
                                auto x_new = _mm256_maskload_ps(p_wv + k, wv_mask);
                                x_new = _mm256_fmadd_ps(x_sub, x_weight, x_new);
                                _mm256_storeu_ps(p_wv + k, x_new);
                            }
                            if (k < K) {
                                auto mask = avx2::functional::get_mask(K&7);
                                auto x_sub = _mm256_maskload_ps(p_sub + k, mask);
                                auto x_new = _mm256_maskload_ps(p_wv + k, wv_mask);
                                x_new = _mm256_fmadd_ps(x_sub, x_weight, x_new);
                                _mm256_maskstore_ps(p_wv + k, mask, x_new);
                            }
                        }
                    }
                }
            }
        }
    } else {
        // with_causal_mask:0   with_kv_cache:0   kv_head_transposed:0
        // parallel in B/H/M dimensions
        // k&v: [B, N, H*K]
        int N = final_k->getStaticDims()[1];
        int stride_bytes_hk = H*K*sizeof(float);

        // kernel register blocking on 6 rows
        const size_t work_amount = (size_t)B*H * M;

        parallel_nt(0, [&](int ithr, int nthr) {
            size_t start{0}, end{0};
            splitter(work_amount, nthr, ithr, start, end);
            size_t bh0, mb0;
            size_t bh1, mb1;
            if (start == end) return;
            bh0 = offset2coord(start, M, mb0);
            bh1 = offset2coord(end, M, mb1);
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

struct cpu_thr_info {
    pid_t pid;
    cpu_set_t cpuset;
    friend inline std::ostream& operator<<(std::ostream& os, const cpu_thr_info & me) {
        os << "pid=" << long(me.pid) << " cpuset=";
        const long nCores = sysconf( _SC_NPROCESSORS_ONLN );
        for (long i = 0; i < nCores; i++) {
            if (CPU_ISSET(i, &me.cpuset))
                os << "X";
            else
                os << "_";
        }
        return os;
    }
    static void test() {
        static int test_cnt = 5;

        if (test_cnt <= 0) return;
        test_cnt--;

        cpu_thr_info all_thr[256];
        std::atomic<long> total_thr;
        
        parallel_nt(0, [&](int ithr, int nthr) {
            total_thr = nthr;
            auto & info = all_thr[ithr];
            info.pid = syscall(__NR_gettid);
            CPU_ZERO(&info.cpuset);
            sched_getaffinity(0, sizeof(info.cpuset), &info.cpuset);
            std::this_thread::sleep_for(std::chrono::seconds(1));
        });

        std::cout << "cpu_thr_info total_threads = " << total_thr << ", mappings:" << std::endl;
        for ( int i=0;i<total_thr; i++)
            std::cout << "\t thread " << i << ": " << all_thr[i] << std::endl;
    }
};

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

    //cpu_thr_info::test();
    execute(strm);
}

bool MHA2::created() const {
    return getType() == Type::MHA2;
}

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
