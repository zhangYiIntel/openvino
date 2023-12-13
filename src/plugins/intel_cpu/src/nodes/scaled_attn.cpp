// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "scaled_attn.h"

#include <dnnl_extension_utils.h>
#include <onednn/dnnl.h>

#include <algorithm>
#include <cpu/x64/cpu_isa_traits.hpp>
#include <cpu/x64/jit_generator.hpp>
#include <ie_ngraph_utils.hpp>
#include <string>
#include <shape_inference/shape_inference_internal_dyn.hpp>
#include <vector>

#include "openvino/core/parallel.hpp"
#include "memory_desc/cpu_memory_desc_utils.h"
#include "memory_desc/dnnl_blocked_memory_desc.h"
#include "utils/plain_tensor.hpp"
#include <openvino/op/scaled_dot_product_attention.hpp>
#include "common/arbitrary_order_desc_creator.h"

#ifdef OV_CPU_WITH_MLAS
#    include "mlas/sgemm.hpp"
#endif

#include "utils/plain_tensor.hpp"
#include "kernels/scaled_attn/softmax.hpp"
#include "kernels/scaled_attn/mha_single_token.hpp"
#include "kernels/scaled_attn/attn_memcpy.hpp"

#include <cpu/x64/brgemm/brgemm.hpp>
#include <cpu/x64/matmul/brgemm_matmul_copy_utils.hpp>
#include <cpu/x64/matmul/brgemm_matmul_utils.hpp>
#include <cpu/x64/amx_tile_configure.hpp>
#include <cstddef>


using namespace InferenceEngine;
using namespace InferenceEngine::Extensions::Cpu::XARCH;
using namespace dnnl::impl::cpu::x64;

using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64::matmul;
#define THROW_ERROR IE_THROW() << "oneDNN 1st token executor with name Init Failure'" << "' "
namespace ov {
namespace intel_cpu {
namespace node {

// default implementation: reference
template <ScaledDotProductAttention::KernelTypes KType, typename T>
struct MHAKernel {
    MHAKernel() = default;

    template <typename D>
    float dot_product(const D* a, const D* b, int len, int stride_b = 1) {
        float result = 0;
        if (stride_b == 1) {
            for (int i = 0; i < len; i++)
                result += static_cast<float>(a[i]) * static_cast<float>(b[i]);
        } else {
            for (int i = 0; i < len; i++)
                result += static_cast<float>(a[i]) * static_cast<float>(b[i * stride_b]);
        }
        return result;
    }

    void softmax(float* a, int len) {
        float max = *std::max_element(a, a + len);
        float sum = 0.0f;
        for (int i = 0; i < len; i++) {
            a[i] = exp(a[i] - max);
            sum += a[i];
        }
        float scale = 1.0f / sum;
        for (int i = 0; i < len; i++) {
            a[i] *= scale;
        }
    }

    template <typename D>
    void accumulate(float* acc, const D* v, int len, float weight = 1.0f) {
        for (int i = 0; i < len; i++) {
            acc[i] += static_cast<float>(v[i]) * weight;
        }
    }

    PlainTensor causal_mask;
    bool select_nfltmax_at_0;  // set attn_score to -FLT_MAX when causal_mask[...] equal to this
    void set_causal_mask(PlainTensor mask, bool _select_nfltmax_at_0) {
        causal_mask = mask;
        select_nfltmax_at_0 = _select_nfltmax_at_0;
    }

    // Q, K, V is ready, do attention
    // query         [B, H, q_len, S]
    // present_key   [B, H, kv_len, S]  stride of last dim maybe > 1
    // present_value [B, H, kv_len, S]
    // attention_mask [B, 1, q_len, kv_len]
    // output_emb    [B, q_len, H*S]
    void operator()(dnnl::stream strm,
                    PlainTensor& query,
                    PlainTensor& present_key,
                    PlainTensor& present_value,
                    const PlainTensor& alibi_mask,
                    const PlainTensor& attention_mask,
                    PlainTensor& output_emb,
                    bool has_out_transpose,
                    bool auto_causal,
                    float d_scale = 0.0f) {
        auto B = query.size(0);
        auto H = query.size(1);
        auto q_len = query.size(2);
        auto head_size = query.size(3);
        auto kv_len = present_key.size(2);

        if (d_scale == 0.0f)
            d_scale = 1.0f / sqrt(head_size);

        auto k_stride_s = present_key.stride(3);

        parallel_for2d(B, H, [&](size_t b, size_t h) {
            std::vector<float> attn_score(kv_len);
            std::vector<float> word_vec(head_size, 0.0f);

            for (size_t m = 0; m < q_len; m++) {
                // dot-product to get attention scores
                auto* q = &query.at<T>({b, h, m, 0});
                // how many key/values can be accessed causally
                auto ncausal = kv_len;
                // no causall mask is set and it's not fused into attention_mask
                if (auto_causal)
                    ncausal = kv_len - q_len + m + 1;
                for (size_t n = 0; n < ncausal; n++) {
                    auto* k = &present_key.at<T>({b, h, n, 0}, true);
                    attn_score[n] = dot_product(q, k, head_size, k_stride_s) * d_scale;

                    // apply alibi tensor
                    if (alibi_mask)
                        attn_score[n] += alibi_mask.at<float>({b, h, m, n}, true);

                    // apply attention mask (maybe combined with causal_mask)
                    if (attention_mask)
                        attn_score[n] += attention_mask.at<float>({b, h, m, n}, true);

                    // apply causal_mask
                    if (causal_mask) {
                        bool is_zero = causal_mask.at<uint8_t>({b, h, m, n}, true) == 0;
                        if (select_nfltmax_at_0) {
                            if (is_zero)
                                attn_score[n] = -FLT_MAX;
                        } else {
                            if (!is_zero) {
                                attn_score[n] = -FLT_MAX;
                            }
                        }
                    }
                }

                // softmax
                softmax(&attn_score[0], ncausal);

                // linearly combine value
                word_vec.assign(head_size, 0.0f);
                for (size_t n = 0; n < ncausal; n++) {
                    auto* v = &present_value.at<T>({b, h, n, 0}, true);
                    accumulate(word_vec.data(), v, head_size, attn_score[n]);
                }

                // output [B, L1, H*head_size]
                auto* out = has_out_transpose ? &output_emb.at<T>({b, m, h * head_size}) : &output_emb.at<T>({b, h, m});
                std::copy(word_vec.begin(), word_vec.end(), out);
            }
        });
    }
};

template <typename T>
struct MHAKernel<ScaledDotProductAttention::KT_ONEDNN, T> {
    // q: [B, H, q_len, S]
    // k: [B, H, kv_len, S]
    // v: [B, H, kv_len, S]
    dnnl::memory::desc q_md;
    dnnl::memory::desc k_md;
    dnnl::memory::desc weight_md;
    dnnl::memory::desc v_md;
    dnnl::memory::desc out_md;
    dnnl::memory attn_score;
    dnnl::memory attn_weight;
    dnnl::matmul qk_prim;
    dnnl::matmul wv_prim;
    using tag = dnnl::memory::format_tag;
    using dt = dnnl::memory::data_type;

    struct brgemmExecutor {
        brgemmExecutor(size_t M, size_t K, size_t N, size_t lda, size_t ldb, size_t ldc)
            : M(M),
              K(K),
              N(N),
              lda(lda),
              ldb(ldb),
              ldc(ldc) {
            // blocking M
            const size_t matmulOptimalM = 32;
            M_blk = matmulOptimalM;
            M_tail = M % M_blk;
            ov::element::Type brg0Prc = ov::element::bf16;
            brg0VnniFactor = 4 / brg0Prc.size();

            // blocing N
            N_blk = 32;
            N_tail = N % N_blk;
            // blocing N
            K_blk = 32;
            K_tail = K % K_blk;
            packedBSize = rnd_up(K, brg0VnniFactor) * rnd_up(N, N_blk) * brg0Prc.size();
            packedBData.resize(packedBSize);
            size_t brg0BaseIdx = std::numeric_limits<size_t>::max();
            for (size_t m = 0; m < 2; m++) {
                for (size_t k = 0; k < 2; k++) {
                    for (size_t n = 0; n < 2; n++) {
                        auto& brgemmCtx = brgCtxs0[getBrgIdx(m, k, n)];

                        auto M_ = m ? M_tail : M < M_blk ? 0 : M_blk;
                        auto N_ = n ? N_tail : N - N_tail;
                        auto K_ = k ? K_tail : K - K_tail;
                        auto beta = k && brgCtxs0[getBrgIdx(m, 0, n)].K != 0 ? 1.0f : 0.0f;

                        brgemmCtx.M = M_;
                        brgemmCtx.N = N_;
                        brgemmCtx.K = K_;
                        brgemmCtx.LDA = lda;
                        brgemmCtx.LDB = rnd_up(N, N_blk);  // ???
                        brgemmCtx.LDC = ldc;
                        brgemmCtx.dt_in0 =
                            static_cast<dnnl_data_type_t>(DnnlExtensionUtils::ElementTypeToDataType(brg0Prc));
                        brgemmCtx.dt_in1 =
                            static_cast<dnnl_data_type_t>(DnnlExtensionUtils::ElementTypeToDataType(brg0Prc));
                        brgemmCtx.beta = beta;
                        brgemmCtx.is_with_amx = true;

                        // don't create brgemm kernels for empty tiles
                        if (M_ != 0 && K_ != 0 && N_ != 0) {
                            if (brg0BaseIdx == std::numeric_limits<size_t>::max())
                                brg0BaseIdx = getBrgIdx(m, k, n);
                            init_brgemm(brgemmCtx, brgKernels[getBrgIdx(m, k, n)], true);
                        }
                    }
                }
            }

            auto& brgemmCtx0 = brgCtxs0[brg0BaseIdx];

            // TODO: matrix A copy should be performed to enable AMX matmuls for arbitrary shapes
            if (brgemmCtx0.is_with_amx && K_tail) {
                init_brgemm_copy_a(brgCopyAKernel, K, K_blk, K_tail, brgemmCtx0.LDA, brgemmCtx0.dt_in0);
            }

            if (brgemmCtx0.is_with_amx || brg0Prc == ov::element::i8 || brg0Prc == ov::element::bf16) {
                init_brgemm_copy_b(brgCopyBKernel,
                                   N,
                                   N_blk,
                                   N_tail,
                                   brgemmCtx0.LDB,
                                   brgemmCtx0.K,
                                   brgemmCtx0.is_with_amx,
                                   brgemmCtx0.dt_in0,
                                   brgemmCtx0.dt_in1,
                                   ldb == 1 ? true : false);
            }
        }
        size_t M = 0, M_blk = 0, M_tail = 0;
        size_t K = 0, K_blk = 0, K_tail = 0, N = 0, N_blk = 0, N_tail = 0;
        size_t lda = 0, ldb = 0, ldc = 0;
        size_t brg0VnniFactor = 0;
        size_t packedBSize = 0;
        std::vector<uint8_t> packedBData;
        static constexpr size_t MHA_BRGEMM_KERNELS_NUM = 8;
        struct brgemmCtx {
            size_t M = 0, N = 0, K = 0, LDA = 0, LDB = 0, LDC = 0;
            dnnl_data_type_t dt_in0 = dnnl_data_type_undef;
            dnnl_data_type_t dt_in1 = dnnl_data_type_undef;
            char palette[64];
            bool is_with_amx = false;
            bool is_with_comp = false;
            bool transpose_a = false;
            bool transpose_b = false;
            float beta = 0.0f;
        };
        brgemmCtx brgCtxs0[MHA_BRGEMM_KERNELS_NUM];
        std::unique_ptr<dnnl::impl::cpu::x64::brgemm_kernel_t> brgKernels[MHA_BRGEMM_KERNELS_NUM];
        std::unique_ptr<dnnl::impl::cpu::x64::matmul::jit_brgemm_matmul_copy_a_t> brgCopyAKernel;
        std::unique_ptr<dnnl::impl::cpu::x64::matmul::jit_brgemm_matmul_copy_b_t> brgCopyBKernel;
        size_t getBrgIdx(size_t mIdx, size_t kIdx, size_t nIdx) {
            return mIdx * 4 + kIdx * 2 + nIdx;
        }
        void init_brgemm(brgemmCtx& ctx,
                         std::unique_ptr<dnnl::impl::cpu::x64::brgemm_kernel_t>& brgKernel,
                         bool use_amx) {
#ifdef OPENVINO_ARCH_X86_64
            brgemm_t brgDesc;

            const bool is_int8 =
                one_of(ctx.dt_in0, data_type::u8, data_type::s8) && one_of(ctx.dt_in1, data_type::u8, data_type::s8);
            auto isa = use_amx                                     ? isa_undef
                       : ctx.dt_in0 == dnnl_data_type_t::dnnl_bf16 ? avx512_core_bf16
                                                                   : (is_int8 ? avx512_core_vnni : avx512_core);
            auto status = brgemm_desc_init(&brgDesc,
                                           isa,
                                           brgemm_addr,
                                           ctx.dt_in0,
                                           ctx.dt_in1,
                                           ctx.transpose_a,
                                           ctx.transpose_b,
                                           brgemm_row_major,
                                           1.f,
                                           ctx.beta,
                                           ctx.LDA,
                                           ctx.LDB,
                                           ctx.LDC,
                                           ctx.M,
                                           ctx.N,
                                           ctx.K,
                                           nullptr);
            if (status != dnnl_success) {
                THROW_ERROR << "cannot be executed due to invalid brgconv params";
            }

            ctx.is_with_amx = use_amx;
            status = brgemm_init_tiles(brgDesc, ctx.palette);
            if (use_amx) {
                amx_tile_configure(ctx.palette);
            }

            ctx.is_with_comp = ctx.dt_in0 == dnnl_data_type_t::dnnl_s8 && !ctx.is_with_amx;

            brgemm_kernel_t* brgKernel_ = nullptr;
            status = brgemm_kernel_create(&brgKernel_, brgDesc);
            if (status != dnnl_success) {
                THROW_ERROR << "cannot be executed due to invalid brgconv params";
            }
            brgKernel.reset(brgKernel_);
#else
            THROW_ERROR << "is not supported on non-x86_64";
#endif  // OPENVINO_ARCH_X86_64
        }
        void init_brgemm_copy_a(
            std::unique_ptr<dnnl::impl::cpu::x64::matmul::jit_brgemm_matmul_copy_a_t>& brgCopyKernel,
            size_t K,
            size_t K_blk,
            size_t K_tail,
            size_t LDA,
            dnnl_data_type_t dt_in0,
            bool transpose = false) {
            brgemm_matmul_conf_t brgCopyKernelConf;
            brgCopyKernelConf.src_tag = dnnl_abcd;
            brgCopyKernelConf.K = K;
            brgCopyKernelConf.K_tail = K_tail;
            brgCopyKernelConf.K_blk = K_blk;
            brgCopyKernelConf.use_buffer_a_tail_only = false;
            brgCopyKernelConf.LDA = LDA;
            brgCopyKernelConf.has_zero_point_b = false;
            brgCopyKernelConf.s8s8_compensation_required = false;
            brgCopyKernelConf.wei_zp_type = dnnl::impl::cpu::x64::none;
            brgCopyKernelConf.src_zp_type = dnnl::impl::cpu::x64::none;
            brgCopyKernelConf.src_dt = dt_in0;
            brgCopyKernelConf.a_dt_sz =
                DnnlExtensionUtils::sizeOfDataType(static_cast<dnnl::memory::data_type>(dt_in0));
            brgCopyKernelConf.transposed_A = transpose;

#if defined(OPENVINO_ARCH_X86_64)
            create_brgemm_matmul_copy_a(brgCopyKernel, &brgCopyKernelConf);
#endif  // OPENVINO_ARCH_X86_64
        }

        void init_brgemm_copy_b(
            std::unique_ptr<dnnl::impl::cpu::x64::matmul::jit_brgemm_matmul_copy_b_t>& brgCopyKernel,
            size_t N,
            size_t N_blk,
            size_t N_tail,
            size_t LDB,
            size_t K,
            bool is_with_amx,
            dnnl_data_type_t dt_in0,
            dnnl_data_type_t dt_in1,
            bool transpose = false) {
            brgemm_matmul_conf_t brgCopyKernelConf;
            brgCopyKernelConf.src_dt = dt_in0;
            brgCopyKernelConf.wei_dt = dt_in1;
            brgCopyKernelConf.wei_n_blk = N_blk;
            brgCopyKernelConf.wei_tag = transpose ? dnnl_ba : dnnl_ab;
            brgCopyKernelConf.copy_B_wei_stride = 0;
            brgCopyKernelConf.LDB = LDB;
            brgCopyKernelConf.N = N;
            brgCopyKernelConf.N_tail = N_tail;
            brgCopyKernelConf.N_blk = N_blk;
            brgCopyKernelConf.K = K;
            brgCopyKernelConf.K_blk = K;
            brgCopyKernelConf.N_chunk_elems = brgCopyKernelConf.N_blk;
            brgCopyKernelConf.b_dt_sz =
                DnnlExtensionUtils::sizeOfDataType(static_cast<dnnl::memory::data_type>(brgCopyKernelConf.src_dt));
            brgCopyKernelConf.tr_b_dt_sz =
                DnnlExtensionUtils::sizeOfDataType(static_cast<dnnl::memory::data_type>(brgCopyKernelConf.src_dt));
            brgCopyKernelConf.req_wei_vnni_downconvert = false;

            if (is_with_amx) {
                brgCopyKernelConf.isa = avx512_core_amx;
                brgCopyKernelConf.s8s8_compensation_required = false;
            } else {
                brgCopyKernelConf.isa = dt_in0 == dnnl_data_type_t::dnnl_bf16 ? avx512_core_bf16 : avx512_core_vnni;
                brgCopyKernelConf.s8s8_compensation_required = dt_in0 == dnnl_data_type_t::dnnl_s8;
            }

            brgCopyKernelConf.has_zero_point_a = false;
            brgCopyKernelConf.has_zero_point_b = false;
            brgCopyKernelConf.src_zp_type = dnnl::impl::cpu::x64::none;

#if defined(OPENVINO_ARCH_X86_64)
            auto ret = create_brgemm_matmul_copy_b(brgCopyKernel, &brgCopyKernelConf);
            if (ret != dnnl::impl::status_t::dnnl_success)
                THROW_ERROR << "cannot create_brgemm_matmul_copy_b kernel, dnnl_status: ";
#endif  // OPENVINO_ARCH_X86_64
        }

        void executeGemm(void* a, void* b, void* c) {
            auto ptr_a = reinterpret_cast<uint8_t*>(a);
            auto ptr_b = reinterpret_cast<uint8_t*>(b);
            auto ptr_c = reinterpret_cast<uint8_t*>(c);
            auto dataType = ov::element::bf16;
            if (brgCopyBKernel) {
                for (size_t nb = 0; nb < div_up(N, N_blk); nb++) {
                    auto pCopyKernel0In = ptr_b + nb * N_blk * dataType.size();
                    auto pCopyKernel0Out = packedBData.data() + nb * N_blk * brg0VnniFactor * dataType.size();

                    auto ctx = jit_brgemm_matmul_copy_b_t::ctx_t();

                    const bool is_N_tail = (N - nb * N_blk < N_blk);
                    ctx.current_N_blk = is_N_tail ? N_tail : N_blk;
                    ctx.src = pCopyKernel0In;
                    ctx.tr_src = pCopyKernel0Out;
                    ctx.compensation_ptr = nullptr;
                    ctx.zp_a_compensation_ptr = nullptr;
                    ctx.zp_a_neg_value_ptr = nullptr;
                    ctx.current_K_start = 0;
                    ctx.current_K_iters = K;

                    (*brgCopyBKernel)(&ctx);
                }
            }
            size_t brgIdx0 = getBrgIdx(0, 0, 0);
            // The step for matrix A over main K dimension
            size_t K0_step0 = brgCtxs0[brgIdx0].K;
            // The step for matrix B over main K dimension
            size_t K0_step1 = brgCtxs0[brgIdx0].K * brgCtxs0[brgIdx0].LDB;
            // The step for matrix B over N dimension
            size_t N0_step0 = brgCtxs0[brgIdx0].N * brg0VnniFactor;
            // The step for matrix C over N dimension
            size_t N0_step1 = brgCtxs0[brgIdx0].N;
            for (size_t mb = 0; mb < div_up(M, M_blk); mb++) {
                const bool is_M_tail = (M - mb * M_blk < M_blk);
                for (size_t n = 0; n < 2; n++) {
                    for (size_t k = 0; k < 2; k++) {
                        size_t mIdx = is_M_tail ? 1 : 0;
                        auto& brgemmCtx = brgCtxs0[getBrgIdx(mIdx, k, n)];

                        if (brgemmCtx.K != 0 && brgemmCtx.N != 0) {
                            callBrgemm(brgemmCtx,
                                    brgKernels[getBrgIdx(mIdx, k, n)],
                                    ptr_a + (mb * M_blk * brgemmCtx.LDA) * ov::element::bf16.size(),
                                    packedBData.data() + (k * K0_step1 + n * N0_step0) * ov::element::bf16.size(),
                                    ptr_c + (mb * M_blk * brgemmCtx.LDC + n * N0_step1) * ov::element::bf16.size(),
                                    nullptr);
                        }
                    }
                }
            }
        }
        void callBrgemm(brgemmCtx& ctx,
                        std::unique_ptr<dnnl::impl::cpu::x64::brgemm_kernel_t>& brgKernel,
                        const void* pin0,
                        const void* pin1,
                        void* pout,
                        void* wsp) {
#if defined(OPENVINO_ARCH_X86_64)
            if (ctx.is_with_amx)
                amx_tile_configure(ctx.palette);
            if (ctx.is_with_comp) {
                brgemm_post_ops_data_t post_ops_data;
                brgemm_kernel_execute_postops(brgKernel.get(), 1, pin0, pin1, nullptr, pout, pout, post_ops_data, wsp);
            } else {
                brgemm_batch_element_t addr_batch;
                addr_batch.ptr.A = pin0;
                addr_batch.ptr.B = pin1;
                brgemm_kernel_execute(brgKernel.get(), 1, &addr_batch, pout, nullptr, nullptr);
            }
#else
            THROW_ERROR("is not supported on non-x64 platforms");
#endif  // OPENVINO_ARCH_X86_64
        }
    };

    std::shared_ptr<brgemmExecutor> qk_gemm_ptr = nullptr;

    void prepare_multiquery_prim(dnnl::stream strm, PlainTensor& query, PlainTensor& present_key) {
        auto make_dnnl_dims = [](const std::vector<size_t>& dims) {
            dnnl::memory::dims dnnl_dims(dims.size());
            for (size_t i = 0; i < dims.size(); i++)
                dnnl_dims[i] = static_cast<dnnl::memory::dim>(dims[i]);
            return dnnl_dims;
        };
        auto B = query.size(0);
        auto H = query.size(1);
        auto q_len = query.size(2);
        auto head_size = query.size(3);
        auto Hk = present_key.size(1);
        auto kv_len = present_key.size(2);
        if (qk_gemm_ptr == nullptr) {
            qk_gemm_ptr = std::make_shared<brgemmExecutor>(q_len, head_size, kv_len, query.stride(2), query.stride(3), kv_len);
        }
        auto qkv_dt = precision_of<T>::value == ov::element::f32 ? dt::f32 : dt::bf16;
        dnnl::memory::desc attn_md(make_dnnl_dims({B, H, q_len, kv_len}), dt::f32, tag::abcd);
        weight_md = dnnl::memory::desc(make_dnnl_dims({B, H, q_len, kv_len}), qkv_dt, tag::abcd);
        if (!attn_score || attn_md.get_size() > attn_score.get_desc().get_size()) {
            attn_score = dnnl::memory(attn_md, strm.get_engine());
            attn_weight = dnnl::memory(weight_md, strm.get_engine());
        }
        return;
    }

    void exec_multiquery_qk(PlainTensor& query, PlainTensor& present_key) {
        const auto B = query.size(0);
        const auto H = query.size(1);
        const auto q_len = query.size(2);
        const auto head_size = query.size(3);
        const auto Hk = present_key.size(1);
        const auto kv_len = present_key.size(2);
        size_t h_each_group_len = H / Hk;
        PlainTensor score;
        score.resize({B, H, q_len, kv_len}, static_cast<float*>(attn_score.get_data_handle()));
        for (size_t b = 0; b < B; b++) {
            for (size_t h = 0; h < H; h++) {
                bfloat16* q_ptr = &query.at<bfloat16>({b, h, 0, 0});
                bfloat16* k_ptr = &present_key.at<bfloat16>({b, h / h_each_group_len, 0, 0});
                float* c_ptr = &score.at<float>({b, h, 0, 0});
                qk_gemm_ptr->executeGemm(q_ptr, k_ptr, c_ptr);
            }
        }
    }

    void prepare_prim(dnnl::stream strm, size_t B, size_t H, size_t Hk, size_t q_len, size_t kv_len, size_t S, bool has_out_transpose) {
        auto make_dnnl_dims = [](const std::vector<size_t>& dims) {
            dnnl::memory::dims dnnl_dims(dims.size());
            for (size_t i = 0; i < dims.size(); i++)
                dnnl_dims[i] = static_cast<dnnl::memory::dim>(dims[i]);
            return dnnl_dims;
        };
        auto qkv_dt = precision_of<T>::value == ov::element::f32 ? dt::f32 : dt::bf16;
        dnnl::memory::desc cur_q_md(make_dnnl_dims({B, H, q_len, S}), qkv_dt, tag::abcd);
        dnnl::memory::desc cur_k_md(make_dnnl_dims({B, Hk, kv_len, S}), qkv_dt, tag::abcd);
        if (cur_q_md == q_md && cur_k_md == k_md)
            return;

        q_md = cur_q_md;
        k_md = cur_k_md;
        dnnl::memory::desc attn_md(make_dnnl_dims({B, H, q_len, kv_len}), dt::f32, tag::abcd);
        k_md = k_md.permute_axes({0, 1, 3, 2});
        auto qk_pd = dnnl::matmul::primitive_desc(strm.get_engine(), q_md, k_md, attn_md);
        qk_prim = dnnl::matmul(qk_pd);

        weight_md = dnnl::memory::desc(make_dnnl_dims({B, H, q_len, kv_len}), qkv_dt, tag::abcd);
        v_md = dnnl::memory::desc(make_dnnl_dims({B, Hk, kv_len, S}), qkv_dt, tag::abcd);
        out_md = dnnl::memory::desc(make_dnnl_dims({B, H, q_len, S}), qkv_dt, tag::abcd);
        if (has_out_transpose)
            out_md = out_md.permute_axes({0, 2, 1, 3});
        auto wv_pd = dnnl::matmul::primitive_desc(strm.get_engine(), weight_md, v_md, out_md);
        wv_prim = dnnl::matmul(wv_pd);

        if (!attn_score || attn_md.get_size() > attn_score.get_desc().get_size()) {
            attn_score = dnnl::memory(attn_md, strm.get_engine());
            attn_weight = dnnl::memory(weight_md, strm.get_engine());
        }
    }

    void exec_qk(dnnl::stream strm, PlainTensor& query, PlainTensor& present_key) {
        dnnl::memory q(q_md, strm.get_engine(), query.data<T>());
        dnnl::memory k(k_md, strm.get_engine(), present_key.data<T>());
        qk_prim.execute(strm, {{DNNL_ARG_SRC, q},
                               {DNNL_ARG_WEIGHTS, k},
                               {DNNL_ARG_DST, attn_score}});
    }

    void exec_kv(dnnl::stream strm, PlainTensor& present_value, PlainTensor& output_emb) {
        dnnl::memory v(v_md, strm.get_engine(), present_value.data<T>());
        dnnl::memory out(out_md, strm.get_engine(), output_emb.data<T>());
        wv_prim.execute(strm, {{DNNL_ARG_SRC, attn_weight}, {DNNL_ARG_WEIGHTS, v}, {DNNL_ARG_DST, out}});
    }

    PlainTensor causal_mask;
    bool select_nfltmax_at_0 = false;  // set attn_score to -FLT_MAX when causal_mask[...] equal to this
    void set_causal_mask(PlainTensor mask, bool _select_nfltmax_at_0) {
        causal_mask = mask;
        select_nfltmax_at_0 = _select_nfltmax_at_0;
    }

    // Q, K, V is ready, do attention
    // query         [B, H, q_len, S]
    // present_key   [B, H, kv_len, S]  stride of last dim maybe > 1
    // present_value [B, H, kv_len, S]
    // attention_mask [B, 1, q_len, kv_len]
    // alibi          [B, H, q_len, kv_len]
    // output_emb    [B, L1, H*S]
    void operator()(dnnl::stream strm,
                    PlainTensor& query,
                    PlainTensor& present_key,
                    PlainTensor& present_value,
                    const PlainTensor& alibi_mask,
                    const PlainTensor& attention_mask,
                    PlainTensor& output_emb,
                    bool has_out_transpose,
                    bool auto_causal,
                    float d_scale = 0.0f) {
        auto B = query.size(0);
        auto H = query.size(1);
        auto q_len = query.size(2);
        auto head_size = query.size(3);
        auto Hk = present_key.size(1);
        auto kv_len = present_key.size(2);

        if (d_scale == 0.0f)
            d_scale = 1.0f / sqrt(head_size);
        // go to multiple-query case

        if (H != Hk) {
            prepare_multiquery_prim(strm, query, present_key);
            exec_multiquery_qk(query, present_key);
            std::cout << "Finish MultiQuery" << std::endl;
        } else {
            prepare_prim(strm, B, H, Hk, q_len, kv_len, head_size, has_out_transpose);
            exec_qk(strm, query, present_key);
        }

        PlainTensor score;
        score.resize({B, H, q_len, kv_len}, static_cast<float*>(attn_score.get_data_handle()));
        PlainTensor weight;
        weight.resize({B, H, q_len, kv_len}, static_cast<T*>(attn_weight.get_data_handle()));
        // softmax
        parallel_for3d(B, H, q_len, [&](size_t b, size_t h, size_t m) {
            // apply attention mask & sofmax
            auto ncausal = auto_causal ? (kv_len - q_len + m + 1) : kv_len;
            attn_softmax(&score.at<float>({b, h, m, 0}),
                         &weight.at<T>({b, h, m, 0}),
                         d_scale,
                         alibi_mask ? &alibi_mask.at<float>({b, h, m, 0}, true) : nullptr,
                         attention_mask ? &attention_mask.at<float>({b, h, m, 0}, true) : nullptr,
                         causal_mask ? &causal_mask.at<uint8_t>({b, h, m, 0}, true) : nullptr,
                         select_nfltmax_at_0,
                         ncausal,
                         kv_len,
                         precision_of<T>::value);
        });
        exec_kv(strm, present_value, output_emb);
    }
};

#ifdef OV_CPU_WITH_MLAS
template <>
struct MHAKernel<ScaledDotProductAttention::KT_MLAS, float> {
    size_t m_block_size;
    // buffer to hold qk temp
    std::vector<PlainTensor> qk_buffers;

    MHAKernel() {
        m_block_size = 4;
        select_nfltmax_at_0 = false;
        qk_buffers.resize(parallel_get_max_threads(), PlainTensor(true));
    }

    PlainTensor causal_mask;
    bool select_nfltmax_at_0;  // set attn_score to -FLT_MAX when causal_mask[...] equal to this
    void set_causal_mask(PlainTensor mask, bool _select_nfltmax_at_0) {
        causal_mask = mask;
        select_nfltmax_at_0 = _select_nfltmax_at_0;
    }

    // Q, K, V is ready, do attention
    // query         [B, H, q_len, S]
    // present_key   [B, H, kv_len, S]  stride of last dim maybe > 1
    // present_value [B, H, kv_len, S]
    // attention_mask [B, 1, q_len, kv_len]
    // alibi
    // output_emb    [B, L1, H*S]
    void operator()(dnnl::stream strm,
                    PlainTensor& query,
                    PlainTensor& present_key,
                    PlainTensor& present_value,
                    const PlainTensor& alibi_mask,
                    const PlainTensor& attention_mask,
                    PlainTensor& output_emb,
                    bool has_out_transpose,
                    bool auto_causal,
                    float d_scale = 0.0f) {
        auto B = query.size(0);
        auto H = query.size(1);
        auto q_len = query.size(2);
        auto head_size = query.size(3);
        auto kv_len = present_key.size(2);
        auto h_group_num = present_key.size(1);
        size_t h_each_group_len = H / h_group_num;

        if (d_scale == 0.0f)
            d_scale = 1.0f / sqrt(head_size);
        auto k_stride_s = present_key.stride(3);

        auto m_blocks = (q_len + m_block_size - 1) / m_block_size;

        parallel_for3d(B, H, m_blocks, [&](size_t b, size_t h, size_t m_blk) {
            auto thread_id = parallel_get_thread_num();
            if (thread_id < 0)
                OPENVINO_THROW("The calling thread isn't initialized!");
            auto& qk_buf = qk_buffers[thread_id];

            auto m_start = m_blk * m_block_size;
            auto m_end = std::min(m_start + m_block_size, q_len);
            auto m_cnt = m_end - m_start;

            auto kv_len_cache_align = (((kv_len * sizeof(float)) + 63) / 64 * 64) / sizeof(float);
            qk_buf.resize<float>({m_block_size, kv_len_cache_align});
            const float* q_ptr = &query.at<float>({b, h, m_start, 0});
            const float* k_ptr = &present_key.at<float>({b, h / h_each_group_len, 0, 0});
            const float* v_ptr = &present_value.at<float>({b, h / h_each_group_len, 0, 0});

            float* alibi_ptr = nullptr;
            auto alibi_stride = 0;
            if (alibi_mask) {
                alibi_ptr = &alibi_mask.at<float>({b, h, 0, 0}, true);
                if (alibi_mask.size(2) > 1)
                    alibi_stride = alibi_mask.stride(2);
            }
            float* attn_mask_ptr = nullptr;
            auto attn_mask_stride = 0;
            if (attention_mask) {
                attn_mask_ptr = &attention_mask.at<float>({b, h, 0, 0}, true);
                if (attention_mask.size(2) > 1)
                    attn_mask_stride = attention_mask.stride(2);
            }
            uint8_t* cmask_ptr = nullptr;
            auto cmask_stride = 0;
            if (causal_mask) {
                cmask_ptr = &causal_mask.at<uint8_t>({b, h, 0, 0}, true);
                if (causal_mask.size(2) > 1)
                    cmask_stride = causal_mask.stride(2);
            }

            float* qk = &(qk_buf.at<float>({0, 0}));
            auto qk_m_stride = qk_buf.stride(0);

            if (k_stride_s == 1)
                mlas_sgemm("N",
                           "T",
                           m_cnt,
                           kv_len,
                           head_size,
                           1.0f,
                           q_ptr,
                           query.stride(2),
                           k_ptr,
                           present_key.stride(2),
                           0.f,
                           qk,
                           qk_m_stride,
                           1);
            else
                mlas_sgemm("N",
                           "N",
                           m_cnt,
                           kv_len,
                           head_size,
                           1.0f,
                           q_ptr,
                           query.stride(2),
                           k_ptr,
                           present_key.stride(3),
                           0.f,
                           qk,
                           qk_m_stride,
                           1);

            for (size_t m = m_start; m < m_end; m++) {
                // apply attention mask & sofmax
                auto ncausal = auto_causal ? (kv_len - q_len + m + 1) : kv_len;
                attn_softmax(qk + (m - m_start) * qk_m_stride,
                             qk + (m - m_start) * qk_m_stride,
                             d_scale,
                             alibi_ptr + m * alibi_stride,
                             attn_mask_ptr + m * attn_mask_stride,
                             cmask_ptr + m * cmask_stride,
                             select_nfltmax_at_0,
                             ncausal,
                             kv_len,
                             ov::element::f32);
            }
            mlas_sgemm("N",
                       "N",
                       m_cnt,
                       head_size,
                       kv_len,
                       1.0f,
                       qk,
                       qk_m_stride,
                       v_ptr,
                       present_value.stride(2),
                       0.f,
                       has_out_transpose ? &output_emb.at<float>({b, m_start, h * head_size}) : &output_emb.at<float>({b, h, m_start}),
                       has_out_transpose ? output_emb.stride(1) : output_emb.stride(2),
                       1);
        });
    }
};
#endif

// 2nd token case : only 1 token in query
struct MHASingleToken {
    PlainTensor m_attn_w;
    PlainTensor m_temp;

    MHASingleToken() : m_attn_w(true), m_temp(true) {}

    // Q, K, V is ready, do attention
    // query         [B, H, q_len, S]
    // present_key   [B, H, kv_len, S]  stride of last dim maybe > 1
    // present_value [B, H, kv_len, S]
    // alibi
    // attention_mask [B, 1, q_len, kv_len]
    // output_emb    [B, L1, H, S]
    void operator()(PlainTensor& query,
                    PlainTensor& present_key,
                    PlainTensor& present_value,
                    const PlainTensor& alibi_mask,
                    const PlainTensor& attention_mask,
                    PlainTensor& output_emb,
                    const PlainTensor& beams,
                    bool has_out_transpose,
                    bool auto_causal,
                    float d_scale = 0.0f) {
        mha_single_token(query, present_key, present_value, alibi_mask, attention_mask, beams, output_emb,
            m_attn_w, m_temp, has_out_transpose, auto_causal, d_scale);
    }
};

template <ScaledDotProductAttention::KernelTypes KType, typename T>
struct ScaledDotProductAttention::AttentionExecutor : public ScaledDotProductAttention::Executor {
    PlainTensor q_input;           // f32[B, H, L1, S]
    PlainTensor k_input;           // f32[B, H|1, L1, S] / [B, H|1, L0+L1, S]
    PlainTensor v_input;           // f32[B, H|1, L1, S] / [B, H|1, L0+L1, S]
    PlainTensor beam_table;        // i32[B, max_kvLen]
    PlainTensor attn_buf;          // f32[[B|1],[H|1], L1|1, L0+L1]
    float scale_input = 0.0f;

    MHAKernel<KType, T> kernel;
    MHASingleToken kernel_single_token;

    size_t B, H, L1, L0, S;

    Config config;
    AttentionExecutor(const Config& _config) : attn_buf(true), config(_config) {}

    void prepare_attn_mask(MemoryPtr attn_input) {
        attn_buf.resize<float>(attn_input->getStaticDims());
        auto p = reinterpret_cast<uint8_t*>(attn_input->getData());
        for (size_t i = 0; i < attn_input->getSize(); i++)
            attn_buf.data<float>()[i] = p[i] ? 0.0f : -FLT_MAX;
    }

    void concat_pastkv(const std::vector<MemoryPtr>& inputs,
                       const std::vector<MemoryPtr>& outputs,
                       const PlainTensor& k_input,
                       const PlainTensor& v_input,
                       PlainTensor& past_k_output,
                       PlainTensor& past_v_output) {
        if (config.config.fuse_concat) {
            k_input.assert_dims({B, 0, L1, S}, true);
            v_input.assert_dims({B, 0, L1, S}, true);
            auto past_k_idx = inputs.size() - 2;
            auto past_k_mem = inputs[past_k_idx + 0];
            const auto& permute_axes = config.config.permute_axes;
            L0 = permute_axes.empty() ? past_k_mem->getStaticDims()[2] : past_k_mem->getStaticDims()[permute_axes[2]];
            // [B, H, L0, S]
            past_k_output.reset(outputs[1]);
            past_v_output.reset(outputs[2]);
            if (!permute_axes.empty()) {
                // [L, B, H, S] -> [B, H, L, S]
                past_k_output = past_k_output.permute(permute_axes);
                past_v_output = past_v_output.permute(permute_axes);
            }
            attn_memcpy(k_input, v_input, past_k_output.slice(2, L0, L0 + L1), past_v_output.slice(2, L0, L0 + L1));
            if (!config.is_concat_inplaced) {
                PlainTensor past_k_input, past_v_input;
                past_k_input.reset(past_k_mem);
                past_v_input.reset(inputs[past_k_idx + 1]);
                attn_memcpy(past_k_input, past_v_input, past_k_output, past_v_output);
            }
        } else {
            // k,v inputs are already concatenated
            L0 = k_input.size(2) - L1;
            k_input.assert_dims({B, 0, L0 + L1, S}, true);
            v_input.assert_dims({B, 0, L0 + L1, S}, true);
            past_k_output = k_input;
            past_v_output = v_input;
        }
    }

    void execute(dnnl::stream strm, const std::vector<MemoryPtr>& inputs, const std::vector<MemoryPtr>& outputs) override {
        bool has_out_transpose = config.config.output_BLHxS;
        bool fuse_causal_attn = config.config.fuse_causal_attn;
        bool is_causal = config.config.is_causal;
        const bool fuse_concat = config.config.fuse_concat;
        auto input_num = inputs.size() - (fuse_concat ? 2 : 0);

        q_input.reset(inputs[0]);
        k_input.reset(inputs[1]);
        v_input.reset(inputs[2]);
        PlainTensor attn_mask;
        if (input_num > 3) {
            // attn_mask
            if (inputs[3]->getDesc().getPrecision() == ov::element::u8) {
                // bool->f32
                prepare_attn_mask(inputs[3]);
                attn_mask = attn_buf;
            } else {
                attn_mask.reset(inputs[3]);
            }
            // if has scale, attn_mask must be present
            if (input_num > 4) {
                scale_input = *reinterpret_cast<float*>(inputs[4]->getData());
            }
        }

        // q: [B, H, L1, S]
        const auto & permute_axes = config.config.permute_axes;
        B = permute_axes.empty() ? q_input.size(0) : q_input.size(permute_axes[0]);
        H = permute_axes.empty() ? q_input.size(1) : q_input.size(permute_axes[1]);
        L1 = permute_axes.empty() ? q_input.size(2) : q_input.size(permute_axes[2]);
        S = q_input.size(-1);

        PlainTensor present_key, present_value;
        if (!permute_axes.empty()) {
            q_input = q_input.permute(permute_axes);
            k_input = k_input.permute(permute_axes);
            v_input = v_input.permute(permute_axes);
        }
        concat_pastkv(inputs, outputs, k_input, v_input, present_key, present_value);

        ov::intel_cpu::PlainTensor output_emb(outputs[0]);

        bool auto_causal;
        bool use_attn_mask;
        if (fuse_causal_attn) {
            assert(attn_mask);
            attn_mask.assert_dims({B, 1, L1, L0 + L1});
            auto_causal = true;
            use_attn_mask = true;
        } else {
            if (is_causal) {
                auto_causal = true;
                use_attn_mask = false;
            } else {
                // no attn_mask but has scale, there is a 1-d fake attn_mask
                if (input_num > 3 && attn_mask.m_rank > 1) {
                    assert(attn_mask);
                    // spec requires at least 3, but torch sl test does use rank 2
                    if (attn_mask.m_rank == 2)
                        attn_mask = attn_mask.reshape({1, 1, attn_mask.m_dims[0], attn_mask.m_dims[1]});
                    else if (attn_mask.m_rank == 3)
                        attn_mask = attn_mask.reshape({1, attn_mask.m_dims[0], attn_mask.m_dims[1], attn_mask.m_dims[2]});
                    auto_causal = false;
                    use_attn_mask = true;
                } else {
                    auto_causal = false;
                    use_attn_mask = false;
                }
            }
        }

        if (L1 > 1) {
            // multi-token version
            kernel(strm, q_input, k_input, v_input, {}, use_attn_mask ? attn_mask : PlainTensor(),
                   output_emb, has_out_transpose, auto_causal, scale_input);
        } else {
            // 1-token version
            // for second token, using a special AVX2/AVX512 float path:
            //  1, in matrix mutiply, using AMX is not efficency because the M dimension of A will alway be 1
            //  2, using float will save the repack cost which typically is required for bf16/int8 opt
            //  3, using dot product can leverage the SIMD while easily adapt to indirect kv cache
            kernel_single_token(q_input, present_key, present_value, {}, use_attn_mask ? attn_mask : PlainTensor(),
                        output_emb, beam_table, has_out_transpose, auto_causal, scale_input);
        }
    }
};

ScaledDotProductAttention::ScaledDotProductAttention(const std::shared_ptr<ngraph::Node>& op, const GraphContext::CPtr context)
    : Node(op, context, NgraphShapeInferFactory(op, EMPTY_PORT_MASK)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW("CPU: " + errorMessage);
    }

    const auto node = std::dynamic_pointer_cast<const ov::op::v13::ScaledDotProductAttention>(op);
    if (node) {
        m_config.config.is_causal = node->get_causal();
    } else {
        const auto node = std::dynamic_pointer_cast<const ScaledDotProductAttentionWithKVCache>(op);
        m_config.config = node->get_config();
    }
}

void ScaledDotProductAttention::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;
    auto rtPrecision = getOriginalInputPrecisionAtPort(0);
    auto orginSDPInputNumber = getOriginalInputsNumber() - (m_config.config.fuse_concat ? 2 : 0);

    bool enableKVCacheFP16 = m_config.config.fuse_concat && mayiuse(cpu_isa_t::avx2) && rtPrecision != ov::element::bf16;

    auto kvCachePrecision = enableKVCacheFP16 ? ov::element::f16 : rtPrecision;

    NodeConfig config;
    auto& creatorsMap = BlockedDescCreator::getCommonCreators();
    config.inConfs.resize(getOriginalInputsNumber());
    config.outConfs.resize(getOriginalOutputsNumber());
    config.inConfs[0].setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(
        rtPrecision, getInputShapeAtPort(0)));
    config.inConfs[1].setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(
        rtPrecision, getInputShapeAtPort(1)));
    config.inConfs[2].setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(
        rtPrecision, getInputShapeAtPort(2)));
    auto nextPortIdx = 3;
    if (orginSDPInputNumber > 3) {
        // attn_mask
        if (getOriginalInputPrecisionAtPort(nextPortIdx) == ov::element::u8) {
            config.inConfs[nextPortIdx].setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(
                ov::element::u8, getInputShapeAtPort(nextPortIdx)));
        } else {
            config.inConfs[nextPortIdx].setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(
                ov::element::f32, getInputShapeAtPort(nextPortIdx)));
        }
        nextPortIdx++;
    }
    if (orginSDPInputNumber > 4) {
        config.inConfs[nextPortIdx].setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(
            ov::element::f32, getInputShapeAtPort(nextPortIdx)));
    }

    if (m_config.config.fuse_concat) {
        ArbitraryOrderDescCreator cabdDescCreator({2, 0, 1, 3});
        const auto& permute_axes = m_config.config.permute_axes;
        if (!permute_axes.empty()) {
            // [L,B,H,S]->permute[1,2,0,3] ->[B,H,L,S]
            // The actual index of B is permute[0], H is permute[1], L is permute[2], S is permute[3]
            cabdDescCreator = ArbitraryOrderDescCreator({static_cast<size_t>(permute_axes[2]),
                static_cast<size_t>(permute_axes[0]),
                static_cast<size_t>(permute_axes[1]),
                static_cast<size_t>(permute_axes[3])});
        }
        config.inConfs[orginSDPInputNumber + 0].setMemDesc(cabdDescCreator.createSharedDesc(
            kvCachePrecision, getInputShapeAtPort(orginSDPInputNumber + 0)));
        config.inConfs[orginSDPInputNumber + 1].setMemDesc(cabdDescCreator.createSharedDesc(
            kvCachePrecision, getInputShapeAtPort(orginSDPInputNumber + 1)));

        config.outConfs[1].setMemDesc(cabdDescCreator.createSharedDesc(
            kvCachePrecision, getOutputShapeAtPort(1)));
        config.outConfs[1].inPlace(orginSDPInputNumber + 0);
        config.outConfs[2].setMemDesc(cabdDescCreator.createSharedDesc(
            kvCachePrecision, getOutputShapeAtPort(2)));
        config.outConfs[2].inPlace(orginSDPInputNumber + 1);
    }

    config.outConfs[0].setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(
        rtPrecision, getOutputShapeAtPort(0)));

    supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::ref_any);
    // may fallback to abcd without inplace
    if (m_config.config.fuse_concat) {
        config.inConfs[orginSDPInputNumber + 0].setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(
            kvCachePrecision, getInputShapeAtPort(orginSDPInputNumber + 0)));
        config.inConfs[orginSDPInputNumber + 1].setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(
            kvCachePrecision, getInputShapeAtPort(orginSDPInputNumber + 1)));
        config.outConfs[1].setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(
            kvCachePrecision, getOutputShapeAtPort(1)));
        config.outConfs[1].inPlace(-1);
        config.outConfs[2].setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(
            kvCachePrecision, getOutputShapeAtPort(2)));
        config.outConfs[2].inPlace(-1);
        supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::ref_any);
    }
}

void ScaledDotProductAttention::createPrimitive() {
    if (m_config.config.fuse_concat) {
        auto desc = getSelectedPrimitiveDescriptor();
        if (desc == nullptr)
            OPENVINO_THROW("has unidentified preferable primitive descriptor");

        m_config.is_concat_inplaced = desc->getConfig().outConfs[1].inPlace() >= 0;
    }
    auto rtPrecision = getOriginalInputPrecisionAtPort(0);

    if (rtPrecision == ov::element::bf16) {
        m_executor = std::make_shared<AttentionExecutor<KT_ONEDNN, ov::bfloat16>>(m_config);
    } else {
        // only support bf16/f32
        rtPrecision = ov::element::f32;
#ifdef OV_CPU_WITH_MLAS
        m_executor = std::make_shared<AttentionExecutor<KT_MLAS, float>>(m_config);
#else
        m_executor = std::make_shared<AttentionExecutor<KT_ONEDNN, float>>(m_config);
#endif
    }
}

void ScaledDotProductAttention::execute(dnnl::stream strm) {
    std::vector<MemoryPtr> inputs(getParentEdges().size()), outputs(getChildEdges().size());
    for (size_t i = 0; i < inputs.size(); i++) {
        inputs[i] = getParentEdgeAt(i)->getMemoryPtr();
    }
    for (size_t i = 0; i < outputs.size(); i++) {
        outputs[i] = getChildEdgeAt(i)->getMemoryPtr();
    }
    m_executor->execute(strm, inputs, outputs);
}

bool ScaledDotProductAttention::isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (!std::dynamic_pointer_cast<const ov::op::v13::ScaledDotProductAttention>(op) &&
            !std::dynamic_pointer_cast<const ScaledDotProductAttentionWithKVCache>(op)) {
            errorMessage = "Only ScaledDotProductAttention or ScaledDotProductAttentionWithKVCache operation are supported";
            return false;
        }
        // expect shape of q: [B, H, L, S]
        auto inRank = op->get_input_partial_shape(0).size();
        if (inRank != 4u) {
            errorMessage = "Doesn't support 'data' input with rank: " + std::to_string(inRank);
            return false;
        }
        int orgSDPAInput = static_cast<int>(op->get_input_size());
        const auto node = std::dynamic_pointer_cast<const ScaledDotProductAttentionWithKVCache>(op);
        if (node) {
            if (node->get_config().fuse_concat) {
                orgSDPAInput -= 2;
            }
        }
        if (orgSDPAInput > 3) {
            inRank = op->get_input_partial_shape(3).size();
            if (inRank > 4u) {
                errorMessage = "Doesn't support 'attention mask' with rank: " + std::to_string(inRank);
                return false;
            }
        }
        // using mha should be better for static shapes
        if (!op->is_dynamic()) {
            errorMessage = "Only run in dynamic mode";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
