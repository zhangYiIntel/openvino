
// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <dnnl_extension_utils.h>
#include <node.h>

#include <algorithm>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "openvino/core/parallel.hpp"
#include "utils/profiler.hpp"
#include "vnode_attn_softmax.hpp"
#ifdef _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#include <immintrin.h>
#endif
#ifdef OV_CPU_WITH_LLMDNN
#    include "llm_emb_gpt.hpp"
#    include "llm_mha_gpt.hpp"
#    include "llm_mm.hpp"
#endif
#ifdef OV_CPU_WITH_MLAS
#    include "mlas/sgemm.hpp"
#endif

#include "utils/plain_tensor.hpp"
namespace ov {
namespace intel_cpu {
namespace node {

//============================ kernels ============================
enum KernelTypes { KT_REF, KT_LLMDNN, KT_MLAS};

template <KernelTypes KType>
struct ktype_name_of {
    static constexpr char* value = "?";
};

template <>
struct ktype_name_of<KT_REF> {
    static constexpr char* value = "REF";
};
template <>
struct ktype_name_of<KT_LLMDNN> {
    static constexpr char* value = "LLMDNN";
};
template <>
struct ktype_name_of<KT_MLAS> {
    static constexpr char* value = "MLAS";
};

// default implementation: reference
template <KernelTypes KType, typename T>
struct RoPE_kernel {
    RoPE_kernel() = default;

    // groupped multi-query:
    //     G: groups of qkv, in each group 1 key/value is shared among gH query tokens
    // we don't have optimized MHA kernels for that yet, so RoPE still expand 1

    // w/o past_kv
    void operator()(PlainTensor<T>& cur_query,        // [B,L,H,S] or [B,L,G,gH,S]
                    PlainTensor<T>& cur_key,           // [B,L,H,S] or [B,L,G,1,S]
                    PlainTensor<T>& cur_value,           // [B,L,H,S] or [B,L,G,1,S]
                    PlainTensor<T>& query_emb,        // B, H, qLen, S
                    PlainTensor<T>& present_key,      // B, H, kvLen, S
                    PlainTensor<T>& present_value,    // B, H, kvLen, S
                    const PlainTensor<float>& cos_tab,// 1, 1, max_position_embeddings, rotary_dims
                    const PlainTensor<float>& sin_tab,
                    const PlainTensor<int32_t>& gather_idx = {}) { // [B, L]
        PROFILE(prof, "RoPE_REF");
        auto B = cur_query.size(0);
        auto L1 = cur_query.size(1);
        auto S = cur_query.size(-1);
        auto H = present_key.size(1);
        auto L0 = present_key.size(2) - L1;
        auto rotary_dims = cos_tab.size(3);
        auto half_rotary_dims = rotary_dims / 2;

        size_t G = 0;
        size_t gH = 0;
        assert(cur_key.m_rank == cur_query.m_rank);
        assert(cur_value.m_rank == cur_query.m_rank);
        if (cur_query.m_rank == 5) {
            // 5D case: multi-query
            G = cur_query.size(2);
            gH = cur_query.size(3);
            assert(cur_key.size(2) == G && cur_key.size(3) == 1);
            assert(cur_value.size(2) == G && cur_value.size(3) == 1);
        }
        // rotary embedding for word vector at each position p
        // meanwhile concat is performed
        parallel_for3d(B, H, L1, [&](size_t b, size_t h, size_t p) {
            auto p1 = p + L0;
            auto pos_cos_sin = p1;
            if (gather_idx) {
                pos_cos_sin = gather_idx.at({b, p}, true);
            }
            T* q;
            T* k;
            T* v;
            if (gH > 0) {
                // multi-query: h = G*gH
                size_t g = h / gH;
                size_t hg = h % gH;
                q = &cur_query.at({b, p, g, hg, 0});
                k = &cur_key.at({b, p, g, 0, 0});
                v = &cur_value.at({b, p, g, 0, 0});
            } else {
                q = &cur_query.at({b, p, h, 0});
                k = &cur_key.at({b, p, h, 0});
                v = &cur_value.at({b, p, h, 0});
            }
            auto* present_k = &present_key.at({b, h, p1, 0});    // f32[B, H, L0+L1, 64]
            auto* present_v = &present_value.at({b, h, p1, 0});  // f32[B, H, L0+L1, 64]
            auto* q_embed = &query_emb.at({b, h, p, 0});
            // q_embed = (q * cos) + (rotate_half(q) * sin)
            // k_embed = (k * cos) + (rotate_half(k) * sin)
            auto* cos = &cos_tab({0, 0, pos_cos_sin, 0});
            auto* sin = &sin_tab({0, 0, pos_cos_sin, 0});

            size_t s = 0;
            for (; s < half_rotary_dims; s++) {
                q_embed[s] = cos[s] * q[s] + sin[s] * (-q[s + half_rotary_dims]);
                present_k[s] = cos[s] * k[s] + sin[s] * (-k[s + half_rotary_dims]);
                present_v[s] = v[s];
            }
            for (; s < rotary_dims; s++) {
                q_embed[s] = cos[s] * q[s] + sin[s] * (q[s - half_rotary_dims]);
                present_k[s] = cos[s] * k[s] + sin[s] * (k[s - half_rotary_dims]);
                present_v[s] = v[s];
            }
            for (; s < S; s++) {
                q_embed[s] = q[s];
                present_k[s] = k[s];
                present_v[s] = v[s];
            }
        });
    }

    // [B, L] mapping position of tokens in current query/key to position in cos/sin table
    PlainTensor<int32_t> gather_pos_idx;

    void operator()(PlainTensor<T>& cur_query,   // [B,L,H,S] or [B,L,G,gH,S]
                    PlainTensor<T>& cur_key,     // [B,L,H,S] or [B,L,G,1,S]
                    PlainTensor<T>& cur_value,   // [B,L,H,S] or [B,L,G,1,S]
                    PlainTensor<T>& past_key,    // B,H,L0,S
                    PlainTensor<T>& past_value,  // B,H,L0,S
                    PlainTensor<T>& query_emb,   // B,H,L,S
                    PlainTensor<T>& present_key,
                    PlainTensor<T>& present_value,
                    PlainTensor<float>& rotary_emb_cos,
                    PlainTensor<float>& rotary_emb_sin) {
        auto B = cur_query.size(0);
        auto L1 = cur_query.size(1);
        auto H = past_key.size(1);
        auto L0 = past_key.size(2);
        auto S = past_key.size(3);
        size_t G = 0;
        size_t gH = 0;
        assert(cur_key.m_rank == cur_query.m_rank);
        assert(cur_value.m_rank == cur_query.m_rank);
        if (cur_query.m_rank == 5) {
            // 5D case: multi-query
            G = cur_query.size(2);
            gH = cur_query.size(3);
            assert(cur_key.size(2) == G && cur_key.size(3) == 1);
            assert(cur_value.size(2) == G && cur_value.size(3) == 1);
        }
        auto rotary_dims = rotary_emb_cos.size(3);
        auto half_rotary_dims = rotary_dims / 2;

        if (L0 > 0) {
            PROFILE(prof1, "copyPast");
            auto max_nt = parallel_get_max_threads();
            if (B * H < max_nt) {
                parallel_for3d(B, H, L0, [&](size_t b, size_t h, size_t l0) {
                    memcpy(&present_value.at({b, h, l0, 0}), &past_value.at({b, h, l0, 0}), sizeof(T) * S);
                    memcpy(&present_key.at({b, h, l0, 0}), &past_key.at({b, h, l0, 0}), sizeof(T) * S);
                });
            } else {
                parallel_for2d(B, H, [&](size_t b, size_t h) {
                    memcpy(&present_value.at({b, h, 0, 0}), &past_value.at({b, h, 0, 0}), sizeof(T) * L0 * S);
                    memcpy(&present_key.at({b, h, 0, 0}), &past_key.at({b, h, 0, 0}), sizeof(T) * L0 * S);
                });
            }
        }

        PROFILE(prof2, "RoPE_REF");
        // rotary embedding for word vector at each position p
        // meanwhile concat is performed
        parallel_for3d(B, H, L1, [&](size_t b, size_t h, size_t p) {
            auto p1 = p + L0;
            auto pos_cos_sin = p1;
            if (gather_pos_idx) {
                pos_cos_sin = gather_pos_idx.at({b, p}, true);
            }
            T* q;
            T* k;
            T* v;
            if (gH > 0) {
                // multi-query: h = G*gH
                size_t g = h / gH;
                size_t hg = h % gH;
                q = &cur_query.at({b, p, g, hg, 0});
                k = &cur_key.at({b, p, g, 0, 0});
                v = &cur_value.at({b, p, g, 0, 0});
            } else {
                q = &cur_query.at({b, p, h, 0});
                k = &cur_key.at({b, p, h, 0});
                v = &cur_value.at({b, p, h, 0});
            }
            auto* present_k = &present_key.at({b, h, p1, 0});    // f32[B, H, L0+L1, 64]
            auto* present_v = &present_value.at({b, h, p1, 0});  // f32[B, H, L0+L1, 64]
            auto* q_embed = &query_emb.at({b, h, p, 0});
            // q_embed = (q * cos) + (rotate_half(q) * sin)
            // k_embed = (k * cos) + (rotate_half(k) * sin)
            auto* cos = &rotary_emb_cos({0, 0, pos_cos_sin, 0});
            auto* sin = &rotary_emb_sin({0, 0, pos_cos_sin, 0});

            size_t s = 0;
            for (; s < half_rotary_dims; s++) {
                q_embed[s] = cos[s] * q[s] + sin[s] * (-q[s + half_rotary_dims]);
                present_k[s] = cos[s] * k[s] + sin[s] * (-k[s + half_rotary_dims]);
                present_v[s] = v[s];
            }
            for (; s < rotary_dims; s++) {
                q_embed[s] = cos[s] * q[s] + sin[s] * (q[s - half_rotary_dims]);
                present_k[s] = cos[s] * k[s] + sin[s] * (k[s - half_rotary_dims]);
                present_v[s] = v[s];
            }
            for (; s < S; s++) {
                q_embed[s] = q[s];
                present_k[s] = k[s];
                present_v[s] = v[s];
            }
        });
    }
};

// GPT-J style RoPE
template <KernelTypes KType, typename T>
struct RoPE2_kernel {
    RoPE2_kernel() = default;
    // w/o past_kv
    void operator()(PlainTensor<T>& rope_q,           // B,L,H,S
                    PlainTensor<T>& rope_k,           // B,L,H,S
                    PlainTensor<T>& rope_v,           // B,L,H,S
                    PlainTensor<T>& query_emb,           // B,H,L,S
                    PlainTensor<T>& present_key,         // B, H, L0+L1, S
                    PlainTensor<T>& present_value,       // B, H, L0+L1, S
                    const PlainTensor<float>& cos_tab,  // B, L1, half_rotary_dims
                    const PlainTensor<float>& sin_tab,
                    const PlainTensor<int32_t>& gather_idx = {}) {
        PROFILE(prof, "RoPE2");
        assert(!gather_idx);
        auto B = rope_q.size(0);
        auto L1 = rope_q.size(1);
        auto H = rope_q.size(2);
        auto S = rope_q.size(3);
        auto L0 = present_key.size(2) - L1;
        auto rotary_dims = cos_tab.size(2) * 2;
        parallel_for3d(B, H, L1, [&](size_t b, size_t h, size_t p) {
            auto p1 = p + L0;
            auto* q = &rope_q.at({b, p, h, 0});
            auto* k = &rope_k.at({b, p, h, 0});
            auto* v = &rope_v.at({b, p, h, 0});
            auto* present_k = &present_key.at({b, h, p1, 0});    // f32[B, H, L0+L1, 64]
            auto* present_v = &present_value.at({b, h, p1, 0});  // f32[B, H, L0+L1, 64]
            auto* q_embed = &query_emb.at({b, h, p, 0});
            auto* cos = &cos_tab({0, p, 0});
            auto* sin = &sin_tab({0, p, 0});

            size_t s = 0;
            size_t i = 0;

            for (; s < rotary_dims; i++, s += 2) {
                q_embed[s] = cos[i] * q[s] - sin[i] * q[s + 1];
                q_embed[s + 1] = cos[i] * q[s + 1] + sin[i] * q[s];

                present_k[s] = cos[i] * k[s] - sin[i] * k[s + 1];
                present_k[s + 1] = cos[i] * k[s + 1] + sin[i] * k[s];

                present_v[s] = v[s];
                present_v[s + 1] = v[s + 1];
            }

            for (; s < S; s++) {
                q_embed[s] = q[s];
                present_k[s] = k[s];
                present_v[s] = v[s];
            }
        });
    }

    void operator()(PlainTensor<T>& cur_query,           // B,L,H,S
                    PlainTensor<T>& cur_key,             // B,L,H,S
                    PlainTensor<T>& cur_value,           // B,L,H,S
                    PlainTensor<T>& past_key,            // B,H,L0,S
                    PlainTensor<T>& past_value,          // B,H,L0,S
                    PlainTensor<T>& query_emb,           // B,H,L,S
                    PlainTensor<T>& present_key,         // B, H, L0, S
                    PlainTensor<T>& present_value,       // B, H, L0, S
                    PlainTensor<float>& rotary_emb_cos,  // B, L1, half_rotary_dims
                    PlainTensor<float>& rotary_emb_sin) {
        auto B = cur_query.size(0);
        auto L1 = cur_query.size(1);
        auto H = past_key.size(1);
        auto L0 = past_key.size(2);
        auto S = past_key.size(3);
        auto half_rotary_dims = rotary_emb_cos.size(2);
        auto rotary_dims = half_rotary_dims * 2;
        // copy past kv into present
        if (L0 > 0) {
            PROFILE(prof, "copyPast");
            auto max_nt = parallel_get_max_threads();
            if (B * H < max_nt) {
                parallel_for3d(B, H, L0, [&](size_t b, size_t h, size_t l0) {
                    memcpy(&present_value.at({b, h, l0, 0}), &past_value.at({b, h, l0, 0}), sizeof(T) * S);
                    memcpy(&present_key.at({b, h, l0, 0}), &past_key.at({b, h, l0, 0}), sizeof(T) * S);
                });
            } else {
                parallel_for2d(B, H, [&](size_t b, size_t h) {
                    memcpy(&present_value.at({b, h, 0, 0}), &past_value.at({b, h, 0, 0}), sizeof(T) * L0 * S);
                    memcpy(&present_key.at({b, h, 0, 0}), &past_key.at({b, h, 0, 0}), sizeof(T) * L0 * S);
                });
            }
        }

        PROFILE(prof2, "RoPE2_REF");
        // rotary embedding for word vector at each position p
        // meanwhile concat is performed
        parallel_for3d(B, H, L1, [&](size_t b, size_t h, size_t p) {
            auto p1 = p + L0;
            auto* q = &cur_query.at({b, p, h, 0});
            auto* k = &cur_key.at({b, p, h, 0});
            auto* v = &cur_value.at({b, p, h, 0});
            auto* present_k = &present_key.at({b, h, p1, 0});    // f32[B, H, L0+L1, 64]
            auto* present_v = &present_value.at({b, h, p1, 0});  // f32[B, H, L0+L1, 64]
            auto* q_embed = &query_emb.at({b, h, p, 0});
            auto* cos = &rotary_emb_cos({0, p, 0});
            auto* sin = &rotary_emb_sin({0, p, 0});

            size_t s = 0;
            size_t i = 0;

            for (; s < rotary_dims; i++, s += 2) {
                q_embed[s] = cos[i] * q[s] - sin[i] * q[s + 1];
                q_embed[s + 1] = cos[i] * q[s + 1] + sin[i] * q[s];

                present_k[s] = cos[i] * k[s] - sin[i] * k[s + 1];
                present_k[s + 1] = cos[i] * k[s + 1] + sin[i] * k[s];

                present_v[s] = v[s];
                present_v[s + 1] = v[s + 1];
            }

            for (; s < S; s++) {
                q_embed[s] = q[s];
                present_k[s] = k[s];
                present_v[s] = v[s];
            }
        });
    }
};

template <typename RT>
struct CopyPastKernel {
    void operator()(const PlainTensor<RT>& past_key,      // "f32[B, H, L0, S]"
                    const PlainTensor<RT>& past_value,    // "f32[B, H, L0, S]"
                    const PlainTensor<RT>& present_key,   // "f32[B, H, L0+L1, S]"
                    const PlainTensor<RT>& present_value  // "f32[B, H, L0+L1, S]"
    ) {
        PROFILE(prof, "copyPast");
        auto B = past_key.size(0);
        auto H = past_key.size(1);
        auto L0 = past_key.size(2);
        auto S = past_key.size(3);
        //auto L1 = k_input.size(1);
        auto past_ks_stride = past_key.stride(3);

        assert(past_ks_stride == 1);

        if (past_ks_stride == 1) {
            auto max_nt = parallel_get_max_threads();
            if (B * H < max_nt) {
                parallel_for3d(B, H, L0, [&](size_t b, size_t h, size_t l0) {
                    memcpy(&present_value.at({b, h, l0, 0}), &past_value.at({b, h, l0, 0}), sizeof(RT) * S);
                    memcpy(&present_key.at({b, h, l0, 0}), &past_key.at({b, h, l0, 0}), sizeof(RT) * S);
                });
            } else {
                parallel_for2d(B, H, [&](size_t b, size_t h) {
                    memcpy(&present_value.at({b, h, 0, 0}), &past_value.at({b, h, 0, 0}), sizeof(RT) * L0 * S);
                    memcpy(&present_key.at({b, h, 0, 0}), &past_key.at({b, h, 0, 0}), sizeof(RT) * L0 * S);
                });
            }
        }
#if 0
        if (0) {
            parallel_for3d(2, B, H, [&](size_t kv_id, size_t b, size_t h) {
                if (kv_id == 0) {
                    if (past_ks_stride == 1) {
                        if (L0 > 0) {
                            memcpy(&present_key.at({b, h, 0, 0}), &past_key.at({b, h, 0, 0}), sizeof(RT) * L0 * S);
                        }
                        for (size_t p = 0; p < L1; p++) {
                            memcpy(&present_key.at({b, h, L0 + p, 0}), &k_input.at({b, p, h, 0}), sizeof(RT) * S);
                        }
                    } else {
                        // special layout for bloom past/present_key, [B, H, S, L0]
                        if (L0 > 0) {
                            for (size_t s = 0; s < S; s++) {
                                memcpy(&present_key.at({b, h, 0, s}), &past_key.at({b, h, 0, s}), sizeof(RT) * L0);
                            }
                        }
                        for (size_t s = 0; s < S; s++) {
                            for (size_t p = 0; p < L1; p++) {
                                present_key.at({b, h, L0 + p, s}) = k_input.at({b, p, h, s});
                            }
                        }
                    }
                } else {
                    // past_key/value
                    if (L0 > 0) {
                        memcpy(&present_value.at({b, h, 0, 0}), &past_value.at({b, h, 0, 0}), sizeof(RT) * L0 * S);
                    }
                    for (size_t p = 0; p < L1; p++) {
                        memcpy(&present_value.at({b, h, L0 + p, 0}), &v_input.at({b, p, h, 0}), sizeof(RT) * S);
                    }
                }
            });
        }
#endif
    }
};

#ifdef OV_CPU_WITH_LLMDNN
template <typename DT>
llmdnn::tensor Convert2LLMTensor(const PlainTensor<DT>& src) {
    llmdnn::tensor dst;
    dst.m_capacity = 0;
    dst.m_rank = src.m_rank;
    dst.m_ptr = src.m_ptr;
    memcpy(dst.m_dims, src.m_dims, sizeof(size_t) * src.m_rank);
    dst.m_element_size = sizeof(DT);
    dst.m_dtype = llmdnn::precision_of<DT>::value;
    for (size_t i = 0; i < src.m_rank; i++) {
        dst.m_strides[i] = src.m_strides[i] * sizeof(DT);
    }
    return std::move(dst);
}

// specialization on llmdnn
template <>
struct RoPE_kernel<KT_LLMDNN, ov::bfloat16> {
    RoPE_kernel() = default;

    void operator()(PlainTensor<ov::bfloat16>& cur_query,   // B,L,H,S
                    PlainTensor<ov::bfloat16>& cur_key,     // B,L,H,S
                    PlainTensor<ov::bfloat16>& cur_value,   // B,L,H,S
                    PlainTensor<ov::bfloat16>& past_key,    // f32[B, H, L0, S]
                    PlainTensor<ov::bfloat16>& past_value,  // f32[B, H, L0, S]
                    PlainTensor<ov::bfloat16>& query_emb,
                    PlainTensor<ov::bfloat16>& present_key,
                    PlainTensor<ov::bfloat16>& present_value,
                    PlainTensor<float>& rotary_emb_cos,
                    PlainTensor<float>& rotary_emb_sin) {
        PROFILE(prof2, "RoPE_LLMDNN");
        llmdnn::emb_gpt(Convert2LLMTensor(cur_query),   // B,L,H,S
                        Convert2LLMTensor(cur_key),     // B,L,H,S
                        Convert2LLMTensor(cur_value),   // B,L,H,S
                        Convert2LLMTensor(past_key),    // f32[B, H, L0, S]
                        Convert2LLMTensor(past_value),  // f32[B, H, L0, S]
                        Convert2LLMTensor(query_emb),
                        Convert2LLMTensor(present_key),
                        Convert2LLMTensor(present_value),
                        Convert2LLMTensor(rotary_emb_cos),
                        Convert2LLMTensor(rotary_emb_sin),
                        llmdnn::tensor());
    }
};
#endif

// default implementation: reference
template <KernelTypes KType, typename T>
struct MHA_kernel {
    MHA_kernel() = default;

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

    PlainTensor<uint8_t> causal_mask;
    bool select_nfltmax_at_0;  // set attn_score to -FLT_MAX when causal_mask[...] equal to this
    void set_causal_mask(PlainTensor<uint8_t> mask, bool _select_nfltmax_at_0) {
        causal_mask = mask;
        select_nfltmax_at_0 = _select_nfltmax_at_0;
    }

    // Q, K, V is ready, do attention
    // query         [B, H, q_len, S]
    // present_key   [B, H, kv_len, S]  stride of last dim maybe > 1
    // present_value [B, H, kv_len, S]
    // attention_mask [B, 1, q_len, kv_len]
    // output_emb    [B, q_len, H*S]
    void operator()(PlainTensor<T>& query,
                    PlainTensor<T>& present_key,
                    PlainTensor<T>& present_value,
                    const PlainTensor<float>& alibi_mask,
                    const PlainTensor<float>& attention_mask,
                    PlainTensor<T>& output_emb,
                    float d_scale = 0.0f) {
        PROFILE(prof, "MHA_REF");
        auto B = query.size(0);
        auto H = query.size(1);
        auto q_len = query.size(2);
        auto head_size = query.size(3);
        auto kv_len = present_key.size(2);

        if (d_scale == 0.0f)
            d_scale = 1.0f / sqrt(head_size);

        auto k_stride_s = present_key.stride(3);
        bool auto_causal = attention_mask.size(2) == 1 && !causal_mask;

        parallel_for2d(B, H, [&](size_t b, size_t h) {
            std::vector<float> attn_score(kv_len);
            std::vector<float> word_vec(head_size, 0.0f);

            // auto key = &present_key.at({b, h, 0, 0});
            // auto value = &present_value.at({b, h, 0, 0});
            // auto output = &output_emb.at({b, 0, h * head_size});
            for (size_t m = 0; m < q_len; m++) {
                // dot-product to get attention scores
                auto* q = &query.at({b, h, m, 0});
                // how many key/values can be accessed causally
                auto ncausal = kv_len;  // kv_len - q_len + m + 1;
                // no causall mask is set and it's not fused into attention_mask
                if (auto_causal)
                    ncausal = kv_len - q_len + m + 1;
                for (size_t n = 0; n < ncausal; n++) {
                    auto* k = &present_key.at({b, h, n, 0});
                    attn_score[n] = dot_product(q, k, head_size, k_stride_s) * d_scale;

                    // apply alibi tensor
                    if (alibi_mask)
                        attn_score[n] += alibi_mask.at({b, h, m, n}, true);

                    // apply attention mask (maybe combined with causal_mask)
                    if (attention_mask)
                        attn_score[n] += attention_mask.at({b, h, m, n}, true);

                    // apply causal_mask
                    if (causal_mask) {
                        bool is_zero = causal_mask.at({b, h, m, n}, true) == 0;
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
                    auto* v = &present_value.at({b, h, n, 0});
                    accumulate(word_vec.data(), v, head_size, attn_score[n]);
                }

                // output [B, L1, H*head_size]
                auto* out = &output_emb.at({b, m, h * head_size});
                std::copy(word_vec.begin(), word_vec.end(), out);
            }
        });
    }
};
#define OV_CPU_WITH_MLAS
#ifdef OV_CPU_WITH_MLAS
template <>
struct MHA_kernel<KT_MLAS, float> {
    size_t m_block_size;
    // buffer to hold qk temp
    std::vector<PlainTensor<float>> qk_buffers;

    MHA_kernel() {
        m_block_size = std::getenv("MBLK") ? atoi(std::getenv("MBLK")) : 4;
        qk_buffers.resize(parallel_get_max_threads(), PlainTensor<float>(true));
    }

    PlainTensor<uint8_t> causal_mask;
    bool select_nfltmax_at_0;  // set attn_score to -FLT_MAX when causal_mask[...] equal to this
    void set_causal_mask(PlainTensor<uint8_t> mask, bool _select_nfltmax_at_0) {
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
    void operator()(PlainTensor<float>& query,
                    PlainTensor<float>& present_key,
                    PlainTensor<float>& present_value,
                    const PlainTensor<float>& alibi_mask,
                    const PlainTensor<float>& attention_mask,
                    PlainTensor<float>& output_emb,
                    float d_scale = 0.0f) {
        PROFILE(prof, "MHA_MLAS");
        auto B = query.size(0);
        auto H = query.size(1);
        auto q_len = query.size(2);
        auto head_size = query.size(3);
        auto kv_len = present_key.size(2);

        if (d_scale == 0.0f)
            d_scale = 1.0f / sqrt(head_size);
        auto k_stride_s = present_key.stride(3);

        auto m_blocks = (q_len + m_block_size - 1) / m_block_size;
        bool auto_causal = attention_mask.size(2) == 1 && !causal_mask;

        parallel_for3d(B, H, m_blocks, [&](size_t b, size_t h, size_t m_blk) {
            size_t thread_id = static_cast<size_t>(parallel_get_thread_num());
            auto& qk_buf = qk_buffers[thread_id];

            auto m_start = m_blk * m_block_size;
            auto m_end = std::min(m_start + m_block_size, q_len);
            auto m_cnt = m_end - m_start;

            auto kv_len_cache_align = (((kv_len * sizeof(float)) + 63) / 64 * 64) / sizeof(float);
            qk_buf.resize({m_block_size, kv_len_cache_align});
            const float* q_ptr = &query.at({b, h, m_start, 0});
            const float* k_ptr = &present_key.at({b, h, 0, 0});
            const float* v_ptr = &present_value.at({b, h, 0, 0});

            float* alibi_ptr = nullptr;
            auto alibi_stride = 0;
            if (alibi_mask) {
                alibi_ptr = &alibi_mask.at({b, h, 0, 0}, true);
                if (alibi_mask.size(2) > 1)
                    alibi_stride = alibi_mask.stride(2);
            }
            float* attn_mask_ptr = nullptr;
            auto attn_mask_stride = 0;
            if (attention_mask) {
                attn_mask_ptr = &attention_mask.at({b, h, 0, 0}, true);
                if (attention_mask.size(2) > 1)
                    attn_mask_stride = attention_mask.stride(2);
            }
            uint8_t* cmask_ptr = nullptr;
            auto cmask_stride = 0;
            if (causal_mask) {
                cmask_ptr = &causal_mask.at({b, h, 0, 0}, true);
                if (causal_mask.size(2) > 1)
                    cmask_stride = causal_mask.stride(2);
            }

            float* qk = &(qk_buf.at({0, 0}));
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
                InferenceEngine::Extensions::Cpu::XARCH::attn_softmax(qk + (m - m_start) * qk_m_stride,
                                                                      d_scale,
                                                                      alibi_ptr + m * alibi_stride,
                                                                      attn_mask_ptr + m * attn_mask_stride,
                                                                      cmask_ptr + m * cmask_stride,
                                                                      select_nfltmax_at_0,
                                                                      ncausal,
                                                                      kv_len);
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
                       &output_emb.at({b, m_start, h * head_size}),
                       output_emb.stride(1),
                       1);
        });
    }
};
#endif

#ifdef __AVX512F__
#define ENABLE_AVX512_OPT
#endif

// 2nd token case : only 1 token in query
template <typename RT>
struct MHA_1Token {
    PlainTensor<float> m_attn_w;
    PlainTensor<float> m_temp;

    MHA_1Token() : m_temp(true), m_attn_w(true) {}

    PlainTensor<uint8_t> causal_mask;
    bool select_nfltmax_at_0;  // set attn_score to -FLT_MAX when causal_mask[...] equal to this
    void set_causal_mask(PlainTensor<uint8_t> mask, bool _select_nfltmax_at_0) {
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
    void operator()(PlainTensor<RT>& query,
                    PlainTensor<RT>& present_key,
                    PlainTensor<RT>& present_value,
                    const PlainTensor<float>& alibi_mask,
                    const PlainTensor<float>& attention_mask,
                    PlainTensor<RT>& output_emb,
                    const PlainTensor<int32_t>& beams,
                    float d_scale = 0.0f) {
        PROFILE(prof0, "MHA_1Token");
        auto B = query.size(0);
        auto H = query.size(1);
        auto q_len = query.size(2);
        auto S = query.size(3);
        auto kv_len = present_key.size(2);

        if (d_scale == 0.0f)
            d_scale = 1.0f / sqrt(S);
        auto k_stride_s = present_key.stride(3);

        assert(k_stride_s == 1);

        bool auto_causal = attention_mask.size(2) == 1 && !causal_mask;

        PROFILE(prof, "Q*K");
        // use per-token kernel, for each k,v token
        //  attn mask is a matrix of q_len(kv_len)
        m_attn_w.resize({B, H, q_len, kv_len});

        parallel_for3d(B, H, kv_len, [&](size_t b, size_t h, size_t pk) {
            // which batch item should be used at postion pk?
            auto b_kv = beams ? beams.at({b, pk}) : b;
            for (size_t pq = 0; pq < q_len; pq++) {
                auto sum = dot_product_opt(&query.at({b, h, pq, 0}), &present_key.at({b_kv, h, pk, 0}), S);
                m_attn_w.at({b, h, pq, pk}) = sum;
            }
        });

        PROFILE_NEXT(prof, "softmax");

        parallel_for3d(B, H, q_len, [&](size_t b, size_t h, size_t pq) {
            // apply attention mask & sofmax
            auto ncausal = auto_causal ? (kv_len - q_len + pq + 1) : kv_len;
            float* alibi_ptr = alibi_mask ? &alibi_mask.at({b, h, pq, 0}, true) : nullptr;
            float* attn_mask_ptr = attention_mask ? &attention_mask.at({b, h, pq, 0}, true) : nullptr;
            uint8_t* cmask_ptr = causal_mask ? &causal_mask.at({b, h, pq, 0}, true) : nullptr;
            InferenceEngine::Extensions::Cpu::XARCH::attn_softmax(&m_attn_w.at({b, h, pq, 0}),
                                                                  d_scale,
                                                                  alibi_ptr,
                                                                  attn_mask_ptr,
                                                                  cmask_ptr,
                                                                  select_nfltmax_at_0,
                                                                  ncausal,
                                                                  kv_len);
        });

        PROFILE_NEXT(prof, "W*V");
        // attn_w * V
        auto nthr = parallel_get_max_threads();
        m_temp.resize({nthr, B, q_len, H, S});
        // m_attn_w {B, H, q_len, kv_len}
        parallel_nt_static(nthr, [&](const size_t ithr, const size_t nthr) {
            size_t kv_start{0}, kv_end{0};
            splitter(kv_len, nthr, ithr, kv_start, kv_end);

            memset(&m_temp.at({ithr, 0, 0, 0, 0}), 0, m_temp.stride(0) * sizeof(float));

            for (size_t b = 0; b < B; b++)
                for (size_t h = 0; h < H; h++)
                    for (size_t pv = kv_start; pv < kv_end; pv++) {
                        // which batch item should be used at postion pv?
                        auto b_kv = beams ? beams.at({b, pv}) : b;
                        auto* v = &present_value.at({b_kv, h, pv, 0});
                        for (size_t pq = 0; pq < q_len; pq++) {
                            auto* out = &m_temp.at({ithr, b, pq, h, 0});
                            auto weight = m_attn_w.at({b, h, pq, pv});
                            accumulate_weighted_v(out, weight, v, S);
                        }
                    }
        });

        PROFILE_NEXT(prof, "Reduce");
        parallel_for3d(B, H, q_len, [&](size_t b, size_t h, size_t pq) {
            auto* temp = &m_temp.at({0, b, pq, h, 0});
            size_t temp_stride = m_temp.stride(0);
            auto* dst = &output_emb.at({b, pq, h*S});
            reduce_v(dst, temp, nthr, S, temp_stride);
        });
    }

    template <bool trans_B = false, class TC, class TA, class TB>
    void matmul(size_t M, size_t N, size_t K, TC* C, size_t ldC, TA* A, size_t ldA, TB* B, size_t ldB) {
        for (size_t m = 0; m < M; m++) {
            for (size_t n = 0; n < N; n++) {
                auto& sum = C[m * ldC + n];
                sum = 0;
                for (size_t k = 0; k < K; k++) {
                    if (trans_B) {
                        sum += A[m * ldA + k] * B[n * ldB + k];
                    } else {
                        sum += A[m * ldA + k] * B[k * ldB + n];
                    }
                }
            }
        }
    }

    inline __m512 mm512_uni_loadu_ps(ov::bfloat16* a) {
#ifdef __AVX512BF16__
        auto vec_bf16 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(a));
        __m512i y = _mm512_cvtepu16_epi32(vec_bf16);
        return _mm512_castsi512_ps(_mm512_slli_epi32(y, 16));
#endif
    }
    inline __m512 mm512_uni_loadu_ps(float* a) {
        return _mm512_loadu_ps(a);
    }
    inline void mm512_uni_storeu_ps(float* a,  __m512 v) {
        _mm512_storeu_ps(a, v);
    }
    inline void mm512_uni_storeu_ps(ov::bfloat16* a,  __m512 v) {
#ifdef __AVX512BF16__
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(a),
                            reinterpret_cast<__m256i>(_mm512_cvtneps_pbh(v)));
#endif
    }

#ifdef __AVX2__
inline void hsum(__m256& x) {
    __m256 y;                             // x:  0 1 2 3   4 5 6 7
    y = _mm256_permute_ps(x, 0x39);       // y:  1 2 3 0   5 6 7 4
    x = _mm256_add_ps(x, y);              // X:  01 12 23 30  45 56 67 74
    y = _mm256_permute_ps(x, 0x4e);       // y:  23 30 01 12  67 74 45 56
    x = _mm256_add_ps(x, y);              // x: 0123 x x x   4567 x x x
    y = _mm256_permute2f128_ps(x, x, 1);  // y: 4567 x x x  0123 x x x
    x = _mm256_add_ps(x, y);              // x: 01234567 x x x x x x x
}
#endif


    template<typename T>
    float dot_product_opt(T* a, T* b, size_t n) {
        size_t i = 0;
        float sum = 0.0f;
#ifdef ENABLE_AVX512_OPT
        auto vsum = _mm512_setzero_ps();
        for (; i <= n - 16; i += 16) {
            auto va = mm512_uni_loadu_ps(a + i);
            auto vb = mm512_uni_loadu_ps(b + i);
            vsum = _mm512_fmadd_ps(va, vb, vsum);
        }
        sum = _mm512_reduce_add_ps(vsum);
#elif defined(__AVX2__)
        auto vsum = _mm256_set1_ps(0.0f);
        for(; i < n - 8; i += 8) {
            auto va = _mm256_loadu_ps(a + i);
            auto vb = _mm256_loadu_ps(b + i);
            vsum = _mm256_fmadd_ps(va, vb, vsum);
        }
        sum = _mm256_cvtss_f32(v_sum);
#endif
        for (; i < n; i++) {
            sum += a[i] * b[i];
        }
        return sum;
    }

    template<typename TO, typename TI>
    void accumulate_weighted_v(TO* out, float weight, TI* v, size_t S) {
        size_t i = 0;
#ifdef ENABLE_AVX512_OPT
        auto attn_w_vec_fp32 = _mm512_set1_ps(weight);
        for (; i <= S - 16; i +=16) {
            auto v_value = mm512_uni_loadu_ps(v + i);
            auto v_out = mm512_uni_loadu_ps(out + i);
            v_out = _mm512_fmadd_ps(attn_w_vec_fp32, v_value, v_out);
            _mm512_storeu_ps(out + i, v_out);
        }
#elif defined(__AVX2__)
        auto attn_w_vec_fp32 = _mm256_set1_ps(weight);
        for(; i <= S - 8; i += 8) {
            auto v_value = _mm256_loadu_ps(v + i);
            auto v_out = _mm256_loadu_ps(out + i);
            v_out = _mm256_fmadd_ps(attn_w_vec_fp32, v_value, v_out);
            _mm256_storeu_ps(out + i, v_out);
        }
#endif
        for (; i < S; i++) {
            out[i] += weight * v[i];
        }
    }

    template<typename T>
    void reduce_v(T* dst, float* temp, size_t M, size_t S, size_t temp_stride) {
        size_t i = 0;
#ifdef ENABLE_AVX512_OPT
        for (; i <= S - 16; i+= 16) {
            auto* src = temp + i;
            auto result_vec_fp32 = _mm512_setzero_ps();
            for (size_t m = 0; m < M; m++) {
                //auto* temp = &m_temp.at({ithr, b, pq, h, 0});
                auto o_vec_fp32 = _mm512_loadu_ps(src);
                result_vec_fp32 = _mm512_add_ps(result_vec_fp32, o_vec_fp32);
                src += temp_stride;
            }
            // save to bf16
            mm512_uni_storeu_ps(dst + i, result_vec_fp32);
        }
#endif
        for (; i <S; i++) {
            auto* src = temp + i;
            float sum = 0.0f;
            // sum result from all threads partition
            for (size_t m = 0; m < M; m++) {
                sum += src[0];
                src += temp_stride;
            }
            dst[i] = sum;
        }
    }
};

#ifdef OV_CPU_WITH_LLMDNN
template <>
struct MHA_kernel<KT_LLMDNN, ov::bfloat16> {
    llmdnn::mha_gpt m_kernel;
    bool m_kernel_initialized = false;
    MHA_kernel() {}

    PlainTensor<uint8_t> causal_mask;
    bool select_nfltmax_at_0;  // set attn_score to -FLT_MAX when causal_mask[...] equal to this
    void set_causal_mask(PlainTensor<uint8_t> mask, bool _select_nfltmax_at_0) {
        causal_mask = mask;
        select_nfltmax_at_0 = _select_nfltmax_at_0;
    }

    void operator()(PlainTensor<ov::bfloat16>& query,
                    PlainTensor<ov::bfloat16>& present_key,
                    PlainTensor<ov::bfloat16>& present_value,
                    const PlainTensor<float>& alibi_mask,
                    const PlainTensor<float>& attention_mask,  // [batch, 1, query_seq_len, key_seq_len]
                    PlainTensor<ov::bfloat16>& attn_output,
                    float d_scale = 0.0f) {
        PROFILE(prof, "MHA_LLMDNN");
        auto head_size = query.size(3);
        if (d_scale == 0.0f)
            d_scale = 1.0f / sqrt(head_size);

        // llmdnn::tensor alibi;
        bool use_causal_mask = (!causal_mask) && (attention_mask.size(2) == 1);
        m_kernel.exec(Convert2LLMTensor(query),
                      Convert2LLMTensor(present_key),
                      Convert2LLMTensor(present_value),
                      Convert2LLMTensor(attn_output),
                      Convert2LLMTensor(attention_mask),
                      Convert2LLMTensor(alibi_mask),
                      Convert2LLMTensor(causal_mask),
                      select_nfltmax_at_0,
                      d_scale,
                      use_causal_mask);
    }
};
#endif

template <KernelTypes KType, typename RT>
struct generic_attention {
    RoPE2_kernel<KT_REF, RT> rope2_kernel;
    RoPE_kernel<KT_REF, RT> rope1_kernel;

    MHA_kernel<KType, RT> kernel;
    MHA_1Token<RT> kernel_1tok;
    CopyPastKernel<RT> cppast_kernel;

    PlainTensor<RT> m_query_emb;  // query with embed

    int m_flag;
    generic_attention() : m_query_emb(true) {
        auto* p_flag = std::getenv("MHA_FLAG");
        if (p_flag) {
            m_flag = atoi(p_flag);
        } else {
            m_flag = 0;
        }
    }

    // internal KV cache
    template<typename T>
    struct KVCache {
        struct CacheEntry {
            std::shared_ptr<PlainTensor<T>> key;
            std::shared_ptr<PlainTensor<T>> value;
            CacheEntry(size_t B, size_t H, size_t L, size_t S) {
                key = std::make_shared<PlainTensor<T>>(true);
                value = std::make_shared<PlainTensor<T>>(true);
                key->resize({B, H, L, S});
                value->resize({B, H, L, S});
            }
        };

        std::map<Node*, CacheEntry> m_cache;
        PlainTensor<int32_t> m_beam_idx_tab; //linked list
        PlainTensor<int32_t> m_beams;
        int32_t * m_beam_idx;

        KVCache(size_t B, size_t L) : m_beam_idx_tab(true), m_beams(true) {
            m_beam_idx_tab.resize({B, L});
            m_beams.resize({B, L});

            // kv_beam[b, l] is the item id for previous sequence position

            auto* p_beam_idx_addr = std::getenv("beam_idx_addr");
            if (p_beam_idx_addr) {
                m_beam_idx = reinterpret_cast<int*>(std::stoll(p_beam_idx_addr, nullptr, 0)) + 1;
            } else {
                m_beam_idx = nullptr;
            }
        }

        void update_beams(size_t past_kv_len, size_t q_len) {
            // present KV of length q_len will be stored into cache
            // beam_idx input will be:
            //   1. for 1st token(prompt input), it will be missing
            //   2. for 2nd(& rest) tokens, it likes a pointer of current
            //      input_id which points to where previous kv token is.
            if (m_beam_idx == nullptr || m_beam_idx[-1] == 0) {
                // no beam search, or it's first token
                auto B = m_beam_idx_tab.size(0);
                for (size_t b = 0; b < B; b++) {
                    auto * p = &m_beam_idx_tab.at({b, past_kv_len});
                    for (size_t pos = 0; pos < q_len; pos++) {
                        p[pos] = b;
                    }
                }
                //std::cout << past_kv_len << "+" << q_len << " A1 beam_idx_tab=" << *beam_idx_tab << std::endl;
            } else {
                // q_len must be 1, points to where previous token is
                // assert(q_len == 1);
                // assert(m_beam_idx[-1] == B);
                auto pos = past_kv_len;
                auto B = m_beam_idx_tab.size(0);
                for (size_t b = 0; b < B; b++) {
                    m_beam_idx_tab.at({b, pos}) = m_beam_idx[b];
                }
                //std::cout << past_kv_len << "+" << q_len << " B beam_idx_tab=" << *beam_idx_tab << std::endl;
            }

            auto beam_size = m_beams.size(0);
            auto kv_len = past_kv_len + q_len;
            for (size_t beam = 0; beam < beam_size; beam++) {
                // the last one is always in right order
                m_beams({beam, kv_len - 1}) = beam;
                for (int p = kv_len - 2; p >=0; p--) {
                    auto batch_id = m_beams({beam, p + 1});
                    m_beams({beam, p}) = m_beam_idx_tab({batch_id, p + 1});
                }
            }
            //std::cout << "beam_idx_tab A=" << beam_idx_tab << std::endl;
            if (0) {
                std::cout << "m_beam_idx_tab =" << m_beam_idx_tab << std::endl;
                std::cout << "m_beams        =" << m_beams << std::endl;
            }
        }

        CacheEntry& get(Node* node, size_t B, size_t H, size_t L, size_t S) {
            if (m_cache.count(node) == 0) {
                m_cache.emplace(std::make_pair(node, CacheEntry(B, H, L, S)));
            }
            return m_cache.at(node);
        }
    };

    template<typename T>
    KVCache<T>& get_kv_cache(size_t B, size_t L) {
        static KVCache<T> cache(B, L);
        return cache;
    }

    void operator()(Node* node,
                  int RoPEtype,
                  PlainTensor<RT>& q_proj,      // [B,L,H,S] or [B,L,G,gH,S]
                  PlainTensor<RT>& k_proj,      // [B,L,H,S] or [B,L,G,1,S]
                  PlainTensor<RT>& v_proj,      // [B,L,H,S] or [B,L,G,1,S]
                  PlainTensor<RT>& past_key,    // {B, H, kvLen, S}
                  PlainTensor<RT>& past_value,  // {B, H, kvLen, S}
                  PlainTensor<RT>& present_key,
                  PlainTensor<RT>& present_value,
                  const PlainTensor<float>& alibi_mask,
                  const PlainTensor<float>& attention_mask,
                  const PlainTensor<float>& cos_tab,
                  const PlainTensor<float>& sin_tab,
                  PlainTensor<RT>& output_emb,
                  const PlainTensor<int32_t>& gather_pos_idx = {}) {
        // B,H,L,S
        auto B = q_proj.size(0);
        auto qLen = q_proj.size(1);
        auto S = q_proj.size(-1);
        auto H = past_key.size(1);
        auto kvLen0 = past_key.size(2);
        auto kvLen = present_key.size(2);
        // this is a hack, assume one graph is used only for a infer-request
        auto& kv_cache = get_kv_cache<RT>(B, 2048);
        auto& kv_cache_ent = kv_cache.get(node, B, H, 2048, S);

        if (m_flag & 1) {
            // update KV cache beam table
            if (node->getOriginalLayers() == "VNode_0") {
                kv_cache.update_beams(kvLen0, qLen);
            }
            present_key = kv_cache_ent.key->slice(2, 0, kvLen);
            present_value = kv_cache_ent.value->slice(2, 0, kvLen);
        } else {
            if (kvLen0 > 0)
                cppast_kernel(past_key, past_value, present_key, present_value);
        }

        m_query_emb.resize({B, H, qLen, S});

        if (RoPEtype == 1) {
            // multi-query (5D q/k/v) is only supportted in RoPE1
            rope1_kernel(q_proj, k_proj, v_proj, m_query_emb, present_key, present_value, cos_tab, sin_tab, gather_pos_idx);
        }
        if (RoPEtype == 2) {
            rope2_kernel(q_proj, k_proj, v_proj, m_query_emb, present_key, present_value, cos_tab, sin_tab);
        }

        if (qLen > 1 || ((m_flag & 2) == 0)) {
            // multi-token version
            kernel(m_query_emb, present_key, present_value, alibi_mask, attention_mask, output_emb);
        } else {
            // 1-token version
            PlainTensor<int32_t> beams;
            if (m_flag & 1) beams = kv_cache.m_beams;
            kernel_1tok(m_query_emb,
                        present_key,
                        present_value,
                        alibi_mask,
                        attention_mask,
                        output_emb,
                        beams);
        }
    }
};

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov