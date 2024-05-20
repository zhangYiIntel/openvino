// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <float.h>

#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <type_traits>

#if defined(HAVE_AVX2) || defined(HAVE_AVX512F)
#    include <immintrin.h>
#endif

#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/parallel.hpp"
#include "mha_single_token.hpp"
#include "common.hpp"
#include "softmax_kernel.hpp"
#include "utils/profiler.hpp"

namespace ov {
namespace Extensions {
namespace Cpu {
namespace XARCH {

using namespace ov;

#if defined(HAVE_AVX2)

#define prefetch_bytes(bytes, sel, advance, src) {  \
    auto *p = reinterpret_cast<char *>(src);        \
    for (size_t i = 0; i < bytes; i += 64)          \
        _mm_prefetch(p + i + advance, sel);         \
}

#else

#define prefetch_bytes(bytes, sel, advance, src)

#endif

template<typename TA, typename TB>
void cvt_copy(TA* dst, TB* src, size_t n) {
    size_t i = 0;
#if defined(HAVE_AVX512F)
    for (; i + vec_len_f32_avx512 <= n; i += vec_len_f32_avx512) {
        auto vb = mm512_uni_loadu_ps(src + i);
        mm512_uni_storeu_ps(dst + i, vb);
    }
#elif defined(HAVE_AVX2)
    for (; i + vec_len_f32_avx2 <= n; i += vec_len_f32_avx2) {
        auto vb = mm256_uni_loadu_ps(src + i);
        mm256_uni_storeu_ps(dst + i, vb);
    }
#endif
    for (; i < n; i++) {
        dst[i] = src[i];
    }
}

template<typename T>
static void attn_acc_value_block(float* out, float* weight, T* v, size_t S, size_t block_size) {
#if defined(HAVE_AVX512F)
    size_t j = 0;
    for (; j + 4 <= block_size; j += 4) {
        auto attn_w_vec0 = _mm512_set1_ps(weight[0]);
        auto attn_w_vec1 = _mm512_set1_ps(weight[1]);
        auto attn_w_vec2 = _mm512_set1_ps(weight[2]);
        auto attn_w_vec3 = _mm512_set1_ps(weight[3]);
        size_t i = 0;
        for (; i + vec_len_f32_avx512 <= S; i += vec_len_f32_avx512) {
            auto v_out = mm512_uni_loadu_ps(out + i);
            v_out = _mm512_fmadd_ps(attn_w_vec0, mm512_uni_loadu_ps(v + i), v_out);
            v_out = _mm512_fmadd_ps(attn_w_vec1, mm512_uni_loadu_ps(v + i + S), v_out);
            v_out = _mm512_fmadd_ps(attn_w_vec2, mm512_uni_loadu_ps(v + i + S * 2), v_out);
            v_out = _mm512_fmadd_ps(attn_w_vec3, mm512_uni_loadu_ps(v + i + S * 3), v_out);

            _mm512_storeu_ps(out + i, v_out);
        }
        for (; i < S; i++) {
            out[i] += weight[0] * v[i];
            out[i] += weight[1] * v[i + S];
            out[i] += weight[2] * v[i + S * 2];
            out[i] += weight[3] * v[i + S * 3];
        }
        v += 4 * S;
        weight += 4;
    }
    if (j + 2 <= block_size) {
        auto attn_w_vec0 = _mm512_set1_ps(weight[0]);
        auto attn_w_vec1 = _mm512_set1_ps(weight[1]);
        size_t i = 0;
        for (; i + vec_len_f32_avx512 <= S; i += vec_len_f32_avx512) {
            auto v_out = mm512_uni_loadu_ps(out + i);
            v_out = _mm512_fmadd_ps(attn_w_vec0, mm512_uni_loadu_ps(v + i), v_out);
            v_out = _mm512_fmadd_ps(attn_w_vec1, mm512_uni_loadu_ps(v + i + S), v_out);

            _mm512_storeu_ps(out + i, v_out);
        }
        for (; i < S; i++) {
            out[i] += weight[0] * v[i];
            out[i] += weight[1] * v[i + S];
        }
        v += 2 * S;
        weight += 2;
        j += 2;
    }
    if (j < block_size) {
        auto attn_w_vec0 = _mm512_set1_ps(weight[0]);
        size_t i = 0;
        for (; i + vec_len_f32_avx512 <= S; i += vec_len_f32_avx512) {
            auto v_out = mm512_uni_loadu_ps(out + i);
            v_out = _mm512_fmadd_ps(attn_w_vec0, mm512_uni_loadu_ps(v + i), v_out);

            _mm512_storeu_ps(out + i, v_out);
        }
        for (; i < S; i++) {
            out[i] += weight[0] * v[i];
        }
    }
    return;
#elif defined(HAVE_AVX2)
    size_t j = 0;
    for (; j + 4 <= block_size; j += 4) {
        auto attn_w_vec0 = _mm256_set1_ps(weight[0]);
        auto attn_w_vec1 = _mm256_set1_ps(weight[1]);
        auto attn_w_vec2 = _mm256_set1_ps(weight[2]);
        auto attn_w_vec3 = _mm256_set1_ps(weight[3]);
        size_t i = 0;
        for (; i + vec_len_f32_avx2 <= S; i += vec_len_f32_avx2) {
            auto v_out = mm256_uni_loadu_ps(out + i);
            v_out = _mm256_fmadd_ps(attn_w_vec0, mm256_uni_loadu_ps(v + i), v_out);
            v_out = _mm256_fmadd_ps(attn_w_vec1, mm256_uni_loadu_ps(v + i + S), v_out);
            v_out = _mm256_fmadd_ps(attn_w_vec2, mm256_uni_loadu_ps(v + i + S * 2), v_out);
            v_out = _mm256_fmadd_ps(attn_w_vec3, mm256_uni_loadu_ps(v + i + S * 3), v_out);

            mm256_uni_storeu_ps(out + i, v_out);
        }
        for (; i < S; i++) {
            out[i] += weight[0] * v[i];
            out[i] += weight[1] * v[i + S];
            out[i] += weight[2] * v[i + S * 2];
            out[i] += weight[3] * v[i + S * 3];
        }
        v += 4 * S;
        weight += 4;
    }
    if (j + 2 <= block_size) {
        auto attn_w_vec0 = _mm256_set1_ps(weight[0]);
        auto attn_w_vec1 = _mm256_set1_ps(weight[1]);
        size_t i = 0;
        for (; i + vec_len_f32_avx2 <= S; i += vec_len_f32_avx2) {
            auto v_out = mm256_uni_loadu_ps(out + i);
            v_out = _mm256_fmadd_ps(attn_w_vec0, mm256_uni_loadu_ps(v + i), v_out);
            v_out = _mm256_fmadd_ps(attn_w_vec1, mm256_uni_loadu_ps(v + i + S), v_out);

            mm256_uni_storeu_ps(out + i, v_out);
        }
        for (; i < S; i++) {
            out[i] += weight[0] * v[i];
            out[i] += weight[1] * v[i + S];
        }
        v += 2 * S;
        weight += 2;
        j += 2;
    }
    if (j < block_size) {
        auto attn_w_vec0 = _mm256_set1_ps(weight[0]);
        size_t i = 0;
        for (; i + vec_len_f32_avx2 <= S; i += vec_len_f32_avx2) {
            auto v_out = mm256_uni_loadu_ps(out + i);
            v_out = _mm256_fmadd_ps(attn_w_vec0, mm256_uni_loadu_ps(v + i), v_out);

            mm256_uni_storeu_ps(out + i, v_out);
        }
        for (; i < S; i++) {
            out[i] += weight[0] * v[i];
        }
    }
    return;
#endif
    for (size_t j = 0; j < block_size; j++) {
        for (size_t i = 0; i < S; i++) {
            out[i] += weight[j] * v[i];
        }
        v += S;
    }
}

static void attn_acc_value_block(float* out, float* weight, uint8_t* v, size_t S, size_t block_size) {
    // The layout for per token per head:
    // |scale(f32)|zeropoint(f32)|quantized feature(u8,idx_1)|quantized feature(u8,idx_2)|...|quantized feature(u8,idx_S)|
    // The quantized feature will start from 8bytes=sizeof(float)+sizeof(float)
#if defined(HAVE_AVX512F)
    size_t j = 0;
    for (; j + 4 <= block_size; j += 4) {
        auto v_f0 = reinterpret_cast<float*>(v);
        auto v_f1 = reinterpret_cast<float*>(v + S + 8);
        auto v_f2 = reinterpret_cast<float*>(v + 2 * (S + 8));
        auto v_f3 = reinterpret_cast<float*>(v + 3 * (S + 8));
        auto attn_w_vec0 = _mm512_set1_ps(weight[0] * v_f0[0]);
        auto attn_w_vec1 = _mm512_set1_ps(weight[1] * v_f1[0]);
        auto attn_w_vec2 = _mm512_set1_ps(weight[2] * v_f2[0]);
        auto attn_w_vec3 = _mm512_set1_ps(weight[3] * v_f3[0]);
        auto zp0 = _mm512_set1_ps(v_f0[1]);
        auto zp1 = _mm512_set1_ps(v_f1[1]);
        auto zp2 = _mm512_set1_ps(v_f2[1]);
        auto zp3 = _mm512_set1_ps(v_f3[1]);
        size_t i = 0;
        v += 8;
        for (; i + vec_len_f32_avx512 <= S; i += vec_len_f32_avx512) {
            auto v_out = mm512_uni_loadu_ps(out + i);
            auto v0 = _mm512_sub_ps(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<__m128i*>(v + i)))), zp0);
            auto v1 = _mm512_sub_ps(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<__m128i*>(v + i + S + 8)))), zp1);
            auto v2 = _mm512_sub_ps(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<__m128i*>(v + i + 2 * (S + 8))))), zp2);
            auto v3 = _mm512_sub_ps(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<__m128i*>(v + i + 3 * (S + 8))))), zp3);
            v_out = _mm512_fmadd_ps(attn_w_vec0, v0, v_out);
            v_out = _mm512_fmadd_ps(attn_w_vec1, v1, v_out);
            v_out = _mm512_fmadd_ps(attn_w_vec2, v2, v_out);
            v_out = _mm512_fmadd_ps(attn_w_vec3, v3, v_out);

            _mm512_storeu_ps(out + i, v_out);
        }
        for (; i < S; i++) {
            out[i] += weight[0] * (v[i] - v_f0[1]) * v_f0[0];
            out[i] += weight[1] * (v[i + S + 8] - v_f1[1]) * v_f1[0];
            out[i] += weight[2] * (v[i + 2 * (S + 8)] - v_f2[1]) * v_f2[0];
            out[i] += weight[3] * (v[i + 3 * (S + 8)] - v_f3[1]) * v_f3[0];
        }
        v += 4 * (S + 8) - 8;
        weight += 4;
    }
    for (; j < block_size; j++) {
        auto v_f0 = reinterpret_cast<float*>(v);
        auto attn_w_vec0 = _mm512_set1_ps(weight[0] * v_f0[0]);
        auto zp0 = _mm512_set1_ps(v_f0[1]);
        size_t i = 0;
        v += 8;
        for (; i + vec_len_f32_avx512 <= S; i += vec_len_f32_avx512) {
            auto v_out = mm512_uni_loadu_ps(out + i);
            auto v0 = _mm512_sub_ps(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<__m128i*>(v + i)))), zp0);
            v_out = _mm512_fmadd_ps(attn_w_vec0, v0, v_out);

            _mm512_storeu_ps(out + i, v_out);
        }
        for (; i < S; i++) {
            out[i] += weight[0] * (v[i] - v_f0[1]) * v_f0[0];
        }
        v += S;
        weight++;
    }
    return;
#elif defined(HAVE_AVX2)
    size_t j = 0;
    for (; j < block_size; j++) {
        auto v_f0 = reinterpret_cast<float*>(v);
        auto attn_w_vec0 = _mm256_set1_ps(weight[0] * v_f0[0]);
        auto zp0 = _mm256_set1_ps(v_f0[1]);
        size_t i = 0;
        v += 8;
        for (; i + vec_len_f32_avx2 <= S; i += vec_len_f32_avx2) {
            auto v_out = mm256_uni_loadu_ps(out + i);
            auto v0 = _mm256_sub_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<__m128i*>(v + i)))), zp0);
            v_out = _mm256_fmadd_ps(attn_w_vec0, v0, v_out);

            mm256_uni_storeu_ps(out + i, v_out);
        }
        for (; i < S; i++) {
            out[i] += weight[0] * (v[i] - v_f0[1]) * v_f0[0];
        }
        v += S;
        weight++;
    }
    return;
#endif
    for (size_t j = 0; j < block_size; j++) {
        auto v0 = reinterpret_cast<float*>(v);
        v += 8;
        for (size_t i = 0; i < S; i++) {
            out[i] += weight[j] * (v[i] - v0[1]) * v0[0];
        }
        v += S;
    }
}

template<typename T>
static void attn_acc_value(float* out, float weight, T* v, size_t S, float* scale, float* zp) {
    size_t i = 0;
#if defined(HAVE_AVX512F)
    auto attn_w_vec_fp32 = _mm512_set1_ps(weight);
    for (; i + vec_len_f32_avx512 <= S; i += vec_len_f32_avx512) {
        auto v_value = mm512_uni_loadu_ps(v + i);
        auto v_out = mm512_uni_loadu_ps(out + i);
        v_out = _mm512_fmadd_ps(attn_w_vec_fp32, v_value, v_out);
        _mm512_storeu_ps(out + i, v_out);
    }
#elif defined(HAVE_AVX2)
    auto attn_w_vec_fp32 = _mm256_set1_ps(weight);
    for (; i + vec_len_f32_avx2 <= S; i += vec_len_f32_avx2) {
        auto v_value = mm256_uni_loadu_ps(v + i);
        auto v_out = mm256_uni_loadu_ps(out + i);
        v_out = _mm256_fmadd_ps(attn_w_vec_fp32, v_value, v_out);
        mm256_uni_storeu_ps(out + i, v_out);
    }
#endif
    for (; i < S; i++) {
        out[i] += weight * v[i];
    }
}

static void attn_acc_value(float* out, float weight, uint8_t* v, size_t S, float* scale, float* zp) {
    size_t i = 0;
    weight *= *scale;
#if defined(HAVE_AVX512F)
    auto attn_w_vec_fp32 = _mm512_set1_ps(weight);
    auto v_zp = _mm512_set1_ps(*zp);
    for (; i + 4 * vec_len_f32_avx512 <= S; i += 4 * vec_len_f32_avx512) {
        auto v0_128 = _mm_loadu_si128(reinterpret_cast<__m128i*>(v + i));
        auto v1_128 = _mm_loadu_si128(reinterpret_cast<__m128i*>(v + i + vec_len_f32_avx512));
        auto v2_128 = _mm_loadu_si128(reinterpret_cast<__m128i*>(v + i + vec_len_f32_avx512 * 2));
        auto v3_128 = _mm_loadu_si128(reinterpret_cast<__m128i*>(v + i + vec_len_f32_avx512 * 3));

        auto v0_out = mm512_uni_loadu_ps(out + i);
        auto v1_out = mm512_uni_loadu_ps(out + i + vec_len_f32_avx512);
        auto v2_out = mm512_uni_loadu_ps(out + i + vec_len_f32_avx512 * 2);
        auto v3_out = mm512_uni_loadu_ps(out + i + vec_len_f32_avx512 * 3);

        auto v0_256 = _mm512_cvtepu8_epi32(v0_128);
        auto v1_256 = _mm512_cvtepu8_epi32(v1_128);
        auto v2_256 = _mm512_cvtepu8_epi32(v2_128);
        auto v3_256 = _mm512_cvtepu8_epi32(v3_128);

        auto v0_value = _mm512_cvtepi32_ps(v0_256);
        auto v1_value = _mm512_cvtepi32_ps(v1_256);
        auto v2_value = _mm512_cvtepi32_ps(v2_256);
        auto v3_value = _mm512_cvtepi32_ps(v3_256);

        v0_value = _mm512_sub_ps(v0_value, v_zp);
        v1_value = _mm512_sub_ps(v1_value, v_zp);
        v2_value = _mm512_sub_ps(v2_value, v_zp);
        v3_value = _mm512_sub_ps(v3_value, v_zp);

        v0_out = _mm512_fmadd_ps(attn_w_vec_fp32, v0_value, v0_out);
        v1_out = _mm512_fmadd_ps(attn_w_vec_fp32, v1_value, v1_out);
        v2_out = _mm512_fmadd_ps(attn_w_vec_fp32, v2_value, v2_out);
        v3_out = _mm512_fmadd_ps(attn_w_vec_fp32, v3_value, v3_out);

        mm512_uni_storeu_ps(out + i + vec_len_f32_avx512 * 0, v0_out);
        mm512_uni_storeu_ps(out + i + vec_len_f32_avx512 * 1, v1_out);
        mm512_uni_storeu_ps(out + i + vec_len_f32_avx512 * 2, v2_out);
        mm512_uni_storeu_ps(out + i + vec_len_f32_avx512 * 3, v3_out);
    }
    if (i + 2 * vec_len_f32_avx512 <= S) {
        auto v0_128 = _mm_loadu_si128(reinterpret_cast<__m128i*>(v + i));
        auto v1_128 = _mm_loadu_si128(reinterpret_cast<__m128i*>(v + i + vec_len_f32_avx512));

        auto v0_out = mm512_uni_loadu_ps(out + i);
        auto v1_out = mm512_uni_loadu_ps(out + i + vec_len_f32_avx512);

        auto v0_256 = _mm512_cvtepu8_epi32(v0_128);
        auto v1_256 = _mm512_cvtepu8_epi32(v1_128);

        auto v0_value = _mm512_cvtepi32_ps(v0_256);
        auto v1_value = _mm512_cvtepi32_ps(v1_256);

        v0_value = _mm512_sub_ps(v0_value, v_zp);
        v1_value = _mm512_sub_ps(v1_value, v_zp);

        v0_out = _mm512_fmadd_ps(attn_w_vec_fp32, v0_value, v0_out);
        v1_out = _mm512_fmadd_ps(attn_w_vec_fp32, v1_value, v1_out);

        mm512_uni_storeu_ps(out + i + vec_len_f32_avx512 * 0, v0_out);
        mm512_uni_storeu_ps(out + i + vec_len_f32_avx512 * 1, v1_out);
        i += 2 * vec_len_f32_avx512;
    }
    if (i + vec_len_f32_avx512 <= S) {
        auto v0_128 = _mm_loadu_si128(reinterpret_cast<__m128i*>(v + i));
        auto v0_out = mm512_uni_loadu_ps(out + i);
        auto v0_256 = _mm512_cvtepu8_epi32(v0_128);
        auto v0_value = _mm512_cvtepi32_ps(v0_256);
        v0_value = _mm512_sub_ps(v0_value, v_zp);
        v0_out = _mm512_fmadd_ps(attn_w_vec_fp32, v0_value, v0_out);
        mm512_uni_storeu_ps(out + i + vec_len_f32_avx512 * 0, v0_out);
        i += vec_len_f32_avx512;
    }
#elif defined(HAVE_AVX2)
    auto attn_w_vec_fp32 = _mm256_set1_ps(weight);
    auto v_zp = _mm256_set1_ps(*zp);
    for (; i + 4 * vec_len_f32_avx2 <= S; i += 4 * vec_len_f32_avx2) {
        auto v0_128 = _mm_loadl_epi64(reinterpret_cast<__m128i*>(v + i));
        auto v1_128 = _mm_loadl_epi64(reinterpret_cast<__m128i*>(v + i + vec_len_f32_avx2));
        auto v2_128 = _mm_loadl_epi64(reinterpret_cast<__m128i*>(v + i + vec_len_f32_avx2 * 2));
        auto v3_128 = _mm_loadl_epi64(reinterpret_cast<__m128i*>(v + i + vec_len_f32_avx2 * 3));

        auto v0_out = mm256_uni_loadu_ps(out + i);
        auto v1_out = mm256_uni_loadu_ps(out + i + vec_len_f32_avx2);
        auto v2_out = mm256_uni_loadu_ps(out + i + vec_len_f32_avx2 * 2);
        auto v3_out = mm256_uni_loadu_ps(out + i + vec_len_f32_avx2 * 3);

        auto v0_256 = _mm256_cvtepu8_epi32(v0_128);
        auto v1_256 = _mm256_cvtepu8_epi32(v1_128);
        auto v2_256 = _mm256_cvtepu8_epi32(v2_128);
        auto v3_256 = _mm256_cvtepu8_epi32(v3_128);

        auto v0_value = _mm256_cvtepi32_ps(v0_256);
        auto v1_value = _mm256_cvtepi32_ps(v1_256);
        auto v2_value = _mm256_cvtepi32_ps(v2_256);
        auto v3_value = _mm256_cvtepi32_ps(v3_256);

        v0_value = _mm256_sub_ps(v0_value, v_zp);
        v1_value = _mm256_sub_ps(v1_value, v_zp);
        v2_value = _mm256_sub_ps(v2_value, v_zp);
        v3_value = _mm256_sub_ps(v3_value, v_zp);

        v0_out = _mm256_fmadd_ps(attn_w_vec_fp32, v0_value, v0_out);
        v1_out = _mm256_fmadd_ps(attn_w_vec_fp32, v1_value, v1_out);
        v2_out = _mm256_fmadd_ps(attn_w_vec_fp32, v2_value, v2_out);
        v3_out = _mm256_fmadd_ps(attn_w_vec_fp32, v3_value, v3_out);

        mm256_uni_storeu_ps(out + i + vec_len_f32_avx2 * 0, v0_out);
        mm256_uni_storeu_ps(out + i + vec_len_f32_avx2 * 1, v1_out);
        mm256_uni_storeu_ps(out + i + vec_len_f32_avx2 * 2, v2_out);
        mm256_uni_storeu_ps(out + i + vec_len_f32_avx2 * 3, v3_out);
    }
    if (i + 2 * vec_len_f32_avx2 <= S) {
        auto v0_128 = _mm_loadl_epi64(reinterpret_cast<__m128i*>(v + i));
        auto v1_128 = _mm_loadl_epi64(reinterpret_cast<__m128i*>(v + i + vec_len_f32_avx2));

        auto v0_out = mm256_uni_loadu_ps(out + i);
        auto v1_out = mm256_uni_loadu_ps(out + i + vec_len_f32_avx2);

        auto v0_256 = _mm256_cvtepu8_epi32(v0_128);
        auto v1_256 = _mm256_cvtepu8_epi32(v1_128);

        auto v0_value = _mm256_cvtepi32_ps(v0_256);
        auto v1_value = _mm256_cvtepi32_ps(v1_256);

        v0_value = _mm256_sub_ps(v0_value, v_zp);
        v1_value = _mm256_sub_ps(v1_value, v_zp);

        v0_out = _mm256_fmadd_ps(attn_w_vec_fp32, v0_value, v0_out);
        v1_out = _mm256_fmadd_ps(attn_w_vec_fp32, v1_value, v1_out);

        mm256_uni_storeu_ps(out + i + vec_len_f32_avx2 * 0, v0_out);
        mm256_uni_storeu_ps(out + i + vec_len_f32_avx2 * 1, v1_out);
        i += 2 * vec_len_f32_avx2;
    }
    if (i + vec_len_f32_avx2 <= S) {
        auto v0_128 = _mm_loadl_epi64(reinterpret_cast<__m128i*>(v + i));
        auto v0_out = mm256_uni_loadu_ps(out + i);
        auto v0_256 = _mm256_cvtepu8_epi32(v0_128);
        auto v0_value = _mm256_cvtepi32_ps(v0_256);
        v0_value = _mm256_sub_ps(v0_value, v_zp);
        v0_out = _mm256_fmadd_ps(attn_w_vec_fp32, v0_value, v0_out);
        mm256_uni_storeu_ps(out + i, v0_out);
        i += vec_len_f32_avx2;
    }
#endif
    for (; i < S; i++) {
        out[i] += weight * (v[i] - *zp);
    }
}

template<typename T>
static float sum_q_head(T* a, size_t n) {
    float sum = 0.0f;
    size_t i = 0;
#if defined(HAVE_AVX512F)
    auto vsum0 = _mm512_set1_ps(0.0f);
    auto vsum1 = _mm512_set1_ps(0.0f);
    auto vsum2 = _mm512_set1_ps(0.0f);
    auto vsum3 = _mm512_set1_ps(0.0f);
    for (; i + 4 * vec_len_f32_avx512 <= n; i += vec_len_f32_avx512 * 4) {
        auto va0 = mm512_uni_loadu_ps(a + i);
        auto va1 = mm512_uni_loadu_ps(a + i + vec_len_f32_avx512);
        auto va2 = mm512_uni_loadu_ps(a + i + vec_len_f32_avx512 * 2);
        auto va3 = mm512_uni_loadu_ps(a + i + vec_len_f32_avx512 * 3);

        vsum0 = _mm512_add_ps(va0, vsum0);
        vsum1 = _mm512_add_ps(va1, vsum1);
        vsum2 = _mm512_add_ps(va2, vsum2);
        vsum3 = _mm512_add_ps(va3, vsum3);
    }
    if (i + 2 * vec_len_f32_avx512 <= n) {
        auto va0 = mm512_uni_loadu_ps(a + i);
        auto va1 = mm512_uni_loadu_ps(a + i + vec_len_f32_avx512);

        vsum0 = _mm512_add_ps(va0, vsum0);
        vsum1 = _mm512_add_ps(va1, vsum1);
        i += 2 * vec_len_f32_avx512;
    }
    if (i + vec_len_f32_avx512 <= n) {
        auto va0 = mm512_uni_loadu_ps(a + i);
        vsum0 = _mm512_add_ps(va0, vsum0);
        i += vec_len_f32_avx512;
    }
    vsum0 = _mm512_add_ps(vsum0, vsum1);
    vsum2 = _mm512_add_ps(vsum2, vsum3);
    vsum0 = _mm512_add_ps(vsum0, vsum2);
    sum = _mm512_reduce_add_ps(vsum0);
#elif defined(HAVE_AVX2)
    auto vsum0 = _mm256_set1_ps(0.0f);
    auto vsum1 = _mm256_set1_ps(0.0f);
    auto vsum2 = _mm256_set1_ps(0.0f);
    auto vsum3 = _mm256_set1_ps(0.0f);
    for (; i + 4 * vec_len_f32_avx2 <= n; i += vec_len_f32_avx2 * 4) {
        auto va0 = mm256_uni_loadu_ps(a + i);
        auto va1 = mm256_uni_loadu_ps(a + i + vec_len_f32_avx2);
        auto va2 = mm256_uni_loadu_ps(a + i + vec_len_f32_avx2 * 2);
        auto va3 = mm256_uni_loadu_ps(a + i + vec_len_f32_avx2 * 3);

        vsum0 = _mm256_add_ps(va0, vsum0);
        vsum1 = _mm256_add_ps(va1, vsum1);
        vsum2 = _mm256_add_ps(va2, vsum2);
        vsum3 = _mm256_add_ps(va3, vsum3);
    }
    if (i + 2 * vec_len_f32_avx2 <= n) {
        auto va0 = mm256_uni_loadu_ps(a + i);
        auto va1 = mm256_uni_loadu_ps(a + i + vec_len_f32_avx2);

        vsum0 = _mm256_add_ps(va0, vsum0);
        vsum1 = _mm256_add_ps(va1, vsum1);
        i += 2 * vec_len_f32_avx2;
    }
    if (i + vec_len_f32_avx2 <= n) {
        auto va0 = mm256_uni_loadu_ps(a + i);
        vsum0 = _mm256_add_ps(va0, vsum0);
        i += vec_len_f32_avx2;
    }
    vsum0 = _mm256_add_ps(vsum0, vsum1);
    vsum2 = _mm256_add_ps(vsum2, vsum3);
    vsum0 = _mm256_add_ps(vsum0, vsum2);
    hsum(vsum0);
    sum = _mm256_cvtss_f32(vsum0);
#endif

    for (; i < n; i++) {
        float tmp = a[i];
        sum += tmp;
    }
    return sum;
}

template<typename TA, typename TB>
static float dot_product(TA* a, TB* b, size_t n, float* scale, float* zp, float* head_sum) {
    size_t i = 0;
    float sum = 0.0f;
#if defined(HAVE_AVX512F)
    auto vsum0 = _mm512_setzero_ps();
    auto vsum1 = _mm512_setzero_ps();
    auto vsum2 = _mm512_setzero_ps();
    auto vsum3 = _mm512_setzero_ps();
    for (; i + 4 * vec_len_f32_avx512 <= n; i += 4 * vec_len_f32_avx512) {
        auto va0 = mm512_uni_loadu_ps(a + i);
        auto va1 = mm512_uni_loadu_ps(a + i + vec_len_f32_avx512);
        auto va2 = mm512_uni_loadu_ps(a + i + vec_len_f32_avx512 * 2);
        auto va3 = mm512_uni_loadu_ps(a + i + vec_len_f32_avx512 * 3);

        auto vb0 = mm512_uni_loadu_ps(b + i);
        auto vb1 = mm512_uni_loadu_ps(b + i + vec_len_f32_avx512);
        auto vb2 = mm512_uni_loadu_ps(b + i + vec_len_f32_avx512 * 2);
        auto vb3 = mm512_uni_loadu_ps(b + i + vec_len_f32_avx512 * 3);

        vsum0 = _mm512_fmadd_ps(va0, vb0, vsum0);
        vsum1 = _mm512_fmadd_ps(va1, vb1, vsum1);
        vsum2 = _mm512_fmadd_ps(va2, vb2, vsum2);
        vsum3 = _mm512_fmadd_ps(va3, vb3, vsum3);
    }
    if (i + 2 * vec_len_f32_avx512 <= n) {
        auto va0 = mm512_uni_loadu_ps(a + i);
        auto va1 = mm512_uni_loadu_ps(a + i + vec_len_f32_avx512);

        auto vb0 = mm512_uni_loadu_ps(b + i);
        auto vb1 = mm512_uni_loadu_ps(b + i + vec_len_f32_avx512);

        vsum0 = _mm512_fmadd_ps(va0, vb0, vsum0);
        vsum1 = _mm512_fmadd_ps(va1, vb1, vsum1);
        i += 2 * vec_len_f32_avx512;
    }
    if (i + vec_len_f32_avx512 <= n) {
        auto va0 = mm512_uni_loadu_ps(a + i);
        auto vb0 = mm512_uni_loadu_ps(b + i);
        vsum0 = _mm512_fmadd_ps(va0, vb0, vsum0);
        i += vec_len_f32_avx512;
    }
    vsum0 = _mm512_add_ps(vsum0, vsum1);
    vsum2 = _mm512_add_ps(vsum2, vsum3);
    vsum0 = _mm512_add_ps(vsum0, vsum2);
    sum = _mm512_reduce_add_ps(vsum0);
#elif defined(HAVE_AVX2)
    auto vsum0 = _mm256_set1_ps(0.0f);
    auto vsum1 = _mm256_set1_ps(0.0f);
    auto vsum2 = _mm256_set1_ps(0.0f);
    auto vsum3 = _mm256_set1_ps(0.0f);
    for (; i + 4 * vec_len_f32_avx2 <= n; i += vec_len_f32_avx2 * 4) {
        auto va0 = mm256_uni_loadu_ps(a + i);
        auto va1 = mm256_uni_loadu_ps(a + i + vec_len_f32_avx2);
        auto va2 = mm256_uni_loadu_ps(a + i + vec_len_f32_avx2 * 2);
        auto va3 = mm256_uni_loadu_ps(a + i + vec_len_f32_avx2 * 3);

        auto vb0 = mm256_uni_loadu_ps(b + i);
        auto vb1 = mm256_uni_loadu_ps(b + i + vec_len_f32_avx2);
        auto vb2 = mm256_uni_loadu_ps(b + i + vec_len_f32_avx2 * 2);
        auto vb3 = mm256_uni_loadu_ps(b + i + vec_len_f32_avx2 * 3);

        vsum0 = _mm256_fmadd_ps(va0, vb0, vsum0);
        vsum1 = _mm256_fmadd_ps(va1, vb1, vsum1);
        vsum2 = _mm256_fmadd_ps(va2, vb2, vsum2);
        vsum3 = _mm256_fmadd_ps(va3, vb3, vsum3);
    }
    if (i + 2 * vec_len_f32_avx2 <= n) {
        auto va0 = mm256_uni_loadu_ps(a + i);
        auto va1 = mm256_uni_loadu_ps(a + i + vec_len_f32_avx2);

        auto vb0 = mm256_uni_loadu_ps(b + i);
        auto vb1 = mm256_uni_loadu_ps(b + i + vec_len_f32_avx2);

        vsum0 = _mm256_fmadd_ps(va0, vb0, vsum0);
        vsum1 = _mm256_fmadd_ps(va1, vb1, vsum1);
        i += 2 * vec_len_f32_avx2;
    }
    if (i + vec_len_f32_avx2 <= n) {
        auto va0 = mm256_uni_loadu_ps(a + i);
        auto vb0 = mm256_uni_loadu_ps(b + i);
        vsum0 = _mm256_fmadd_ps(va0, vb0, vsum0);
        i += vec_len_f32_avx2;
    }
    vsum0 = _mm256_add_ps(vsum0, vsum1);
    vsum2 = _mm256_add_ps(vsum2, vsum3);
    vsum0 = _mm256_add_ps(vsum0, vsum2);
    hsum(vsum0);
    sum = _mm256_cvtss_f32(vsum0);
#endif
    for (; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

template<typename TA>
static float dot_product(TA* a, uint8_t* b, size_t n, float* scale, float* zp, float* head_sum) {
    size_t i = 0;
    float sum = 0.0f;
#if defined(HAVE_AVX512F)
    auto vsum0 = _mm512_set1_ps(0.0f);
    auto vsum1 = _mm512_set1_ps(0.0f);
    auto vsum2 = _mm512_set1_ps(0.0f);
    auto vsum3 = _mm512_set1_ps(0.0f);
    auto v_zp = _mm512_set1_ps(*zp);
    for (; i + 4 * vec_len_f32_avx512 <= n; i += vec_len_f32_avx512 * 4) {
        auto va0 = mm512_uni_loadu_ps(a + i);
        auto va1 = mm512_uni_loadu_ps(a + i + vec_len_f32_avx512);
        auto va2 = mm512_uni_loadu_ps(a + i + vec_len_f32_avx512 * 2);
        auto va3 = mm512_uni_loadu_ps(a + i + vec_len_f32_avx512 * 3);

        auto vb0_128 = _mm_loadu_si128(reinterpret_cast<__m128i*>(b + i));
        auto vb1_128 = _mm_loadu_si128(reinterpret_cast<__m128i*>(b + i + vec_len_f32_avx512));
        auto vb2_128 = _mm_loadu_si128(reinterpret_cast<__m128i*>(b + i + vec_len_f32_avx512 * 2));
        auto vb3_128 = _mm_loadu_si128(reinterpret_cast<__m128i*>(b + i + vec_len_f32_avx512 * 3));

        auto vb0_256 = _mm512_cvtepu8_epi32(vb0_128);
        auto vb1_256 = _mm512_cvtepu8_epi32(vb1_128);
        auto vb2_256 = _mm512_cvtepu8_epi32(vb2_128);
        auto vb3_256 = _mm512_cvtepu8_epi32(vb3_128);

        auto vb0 = _mm512_cvtepi32_ps(vb0_256);
        auto vb1 = _mm512_cvtepi32_ps(vb1_256);
        auto vb2 = _mm512_cvtepi32_ps(vb2_256);
        auto vb3 = _mm512_cvtepi32_ps(vb3_256);

        vb0 = _mm512_sub_ps(vb0, v_zp);
        vb1 = _mm512_sub_ps(vb1, v_zp);
        vb2 = _mm512_sub_ps(vb2, v_zp);
        vb3 = _mm512_sub_ps(vb3, v_zp);

        vsum0 = _mm512_fmadd_ps(va0, vb0, vsum0);
        vsum1 = _mm512_fmadd_ps(va1, vb1, vsum1);
        vsum2 = _mm512_fmadd_ps(va2, vb2, vsum2);
        vsum3 = _mm512_fmadd_ps(va3, vb3, vsum3);
    }
    if (i + 2 * vec_len_f32_avx512 <= n) {
        auto va0 = mm512_uni_loadu_ps(a + i);
        auto va1 = mm512_uni_loadu_ps(a + i + vec_len_f32_avx512);

        auto vb0_128 = _mm_loadu_si128(reinterpret_cast<__m128i*>(b + i));
        auto vb1_128 = _mm_loadu_si128(reinterpret_cast<__m128i*>(b + i + vec_len_f32_avx512));

        auto vb0_256 = _mm512_cvtepu8_epi32(vb0_128);
        auto vb1_256 = _mm512_cvtepu8_epi32(vb1_128);

        auto vb0 = _mm512_cvtepi32_ps(vb0_256);
        auto vb1 = _mm512_cvtepi32_ps(vb1_256);

        vb0 = _mm512_sub_ps(vb0, v_zp);
        vb1 = _mm512_sub_ps(vb1, v_zp);

        vsum0 = _mm512_fmadd_ps(va0, vb0, vsum0);
        vsum1 = _mm512_fmadd_ps(va1, vb1, vsum1);
        i += 2 * vec_len_f32_avx512;
    }
    if (i + vec_len_f32_avx512 <= n) {
        auto va0 = mm512_uni_loadu_ps(a + i);
        auto vb0_128 = _mm_loadu_si128(reinterpret_cast<__m128i*>(b + i));
        auto vb0_256 = _mm512_cvtepu8_epi32(vb0_128);
        auto vb0 = _mm512_cvtepi32_ps(vb0_256);
        vb0 = _mm512_sub_ps(vb0, v_zp);
        vsum0 = _mm512_fmadd_ps(va0, vb0, vsum0);
        i += vec_len_f32_avx512;
    }
    vsum0 = _mm512_add_ps(vsum0, vsum1);
    vsum2 = _mm512_add_ps(vsum2, vsum3);
    vsum0 = _mm512_add_ps(vsum0, vsum2);
    sum = _mm512_reduce_add_ps(vsum0);
    for (; i < n; i++) {
        sum += a[i] * (b[i] - *zp);
    }
    return scale[0] * sum;

#elif defined(HAVE_AVX2)
    auto vsum0 = _mm256_set1_ps(0.0f);
    auto vsum1 = _mm256_set1_ps(0.0f);
    auto vsum2 = _mm256_set1_ps(0.0f);
    auto vsum3 = _mm256_set1_ps(0.0f);
    for (; i + 4 * vec_len_f32_avx2 <= n; i += vec_len_f32_avx2 * 4) {
        auto va0 = mm256_uni_loadu_ps(a + i);
        auto va1 = mm256_uni_loadu_ps(a + i + vec_len_f32_avx2);
        auto va2 = mm256_uni_loadu_ps(a + i + vec_len_f32_avx2 * 2);
        auto va3 = mm256_uni_loadu_ps(a + i + vec_len_f32_avx2 * 3);

        auto vb0_128 = _mm_loadl_epi64(reinterpret_cast<__m128i*>(b + i));
        auto vb1_128 = _mm_loadl_epi64(reinterpret_cast<__m128i*>(b + i + vec_len_f32_avx2));
        auto vb2_128 = _mm_loadl_epi64(reinterpret_cast<__m128i*>(b + i + vec_len_f32_avx2 * 2));
        auto vb3_128 = _mm_loadl_epi64(reinterpret_cast<__m128i*>(b + i + vec_len_f32_avx2 * 3));

        auto vb0_256 = _mm256_cvtepu8_epi32(vb0_128);
        auto vb1_256 = _mm256_cvtepu8_epi32(vb1_128);
        auto vb2_256 = _mm256_cvtepu8_epi32(vb2_128);
        auto vb3_256 = _mm256_cvtepu8_epi32(vb3_128);

        auto vb0 = _mm256_cvtepi32_ps(vb0_256);
        auto vb1 = _mm256_cvtepi32_ps(vb1_256);
        auto vb2 = _mm256_cvtepi32_ps(vb2_256);
        auto vb3 = _mm256_cvtepi32_ps(vb3_256);

        vsum0 = _mm256_fmadd_ps(va0, vb0, vsum0);
        vsum1 = _mm256_fmadd_ps(va1, vb1, vsum1);
        vsum2 = _mm256_fmadd_ps(va2, vb2, vsum2);
        vsum3 = _mm256_fmadd_ps(va3, vb3, vsum3);
    }
    if (i + 2 * vec_len_f32_avx2 <= n) {
        auto va0 = mm256_uni_loadu_ps(a + i);
        auto va1 = mm256_uni_loadu_ps(a + i + vec_len_f32_avx2);

        auto vb0_128 = _mm_loadl_epi64(reinterpret_cast<__m128i*>(b + i));
        auto vb1_128 = _mm_loadl_epi64(reinterpret_cast<__m128i*>(b + i + vec_len_f32_avx2));

        auto vb0_256 = _mm256_cvtepu8_epi32(vb0_128);
        auto vb1_256 = _mm256_cvtepu8_epi32(vb1_128);

        auto vb0 = _mm256_cvtepi32_ps(vb0_256);
        auto vb1 = _mm256_cvtepi32_ps(vb1_256);

        vsum0 = _mm256_fmadd_ps(va0, vb0, vsum0);
        vsum1 = _mm256_fmadd_ps(va1, vb1, vsum1);
        i += 2 * vec_len_f32_avx2;
    }
    if (i + vec_len_f32_avx2 <= n) {
        auto va0 = mm256_uni_loadu_ps(a + i);
        auto vb0_128 = _mm_loadl_epi64(reinterpret_cast<__m128i*>(b + i));
        auto vb0_256 = _mm256_cvtepu8_epi32(vb0_128);
        auto vb0 = _mm256_cvtepi32_ps(vb0_256);
        vsum0 = _mm256_fmadd_ps(va0, vb0, vsum0);
        i += vec_len_f32_avx2;
    }
    vsum0 = _mm256_add_ps(vsum0, vsum1);
    vsum2 = _mm256_add_ps(vsum2, vsum3);
    vsum0 = _mm256_add_ps(vsum0, vsum2);
    hsum(vsum0);
    sum = _mm256_cvtss_f32(vsum0);
    for (; i < n; i++) {
        sum += a[i] * b[i];
    }
    // B = scale * (b - zero)
    // Σ (A * B) = Σ (a * scale * (b - zero)) = scale * (Σ a * b - zero Σ a) = scale * (sum - zp * head_sum)
    return scale[0] * (sum - zp[0] * head_sum[0]);
#else
    for (; i < n; i++) {
        sum += a[i] * (b[i] - *zp);
    }
    return scale[0] * sum;
#endif
}

template<typename TA, typename TB>
static void dot_product_block(TA* a, TB* b, float* c, size_t n, size_t block_size) {
#if defined(HAVE_AVX512F)
    size_t j = 0;
    for (; j + 4 <= block_size; j += 4) {
        auto vsum0 = _mm512_setzero_ps();
        auto vsum1 = _mm512_setzero_ps();
        auto vsum2 = _mm512_setzero_ps();
        auto vsum3 = _mm512_setzero_ps();
        size_t i = 0;
        for (; i + vec_len_f32_avx512 <= n; i += vec_len_f32_avx512) {
            auto va = mm512_uni_loadu_ps(a + i);
            vsum0 = _mm512_fmadd_ps(va, mm512_uni_loadu_ps(b + i), vsum0);
            vsum1 = _mm512_fmadd_ps(va, mm512_uni_loadu_ps(b + i + n), vsum1);
            vsum2 = _mm512_fmadd_ps(va, mm512_uni_loadu_ps(b + i + 2 * n), vsum2);
            vsum3 = _mm512_fmadd_ps(va, mm512_uni_loadu_ps(b + i + 3 * n), vsum3);
        }
        float sum0 = _mm512_reduce_add_ps(vsum0);
        float sum1 = _mm512_reduce_add_ps(vsum1);
        float sum2 = _mm512_reduce_add_ps(vsum2);
        float sum3 = _mm512_reduce_add_ps(vsum3);
        for (; i < n; i++) {
            sum0 += a[i] * b[i];
            sum1 += a[i] * b[i + n];
            sum2 += a[i] * b[i + 2 * n];
            sum3 += a[i] * b[i + 3 * n];
        }
        c[0] = sum0;
        c[1] = sum1;
        c[2] = sum2;
        c[3] = sum3;
        c += 4;
        b +=  4 * n;
    }
    for (; j < block_size; j++) {
        auto vsum = _mm512_setzero_ps();
        size_t i = 0;
        for (; i + vec_len_f32_avx512 <= n; i += vec_len_f32_avx512) {
            auto va = mm512_uni_loadu_ps(a + i);
            vsum = _mm512_fmadd_ps(va, mm512_uni_loadu_ps(b + i), vsum);
        }
        float sum = _mm512_reduce_add_ps(vsum);
        for (; i < n; i++) {
            sum += a[i] * b[i];
        }
        b += n;
        *c++ = sum;
    }
    return;
#elif defined(HAVE_AVX2)
    size_t j = 0;
    for (; j + 4 <= block_size; j += 4) {
        auto vsum0 = _mm256_set1_ps(0.0f);
        auto vsum1 = _mm256_set1_ps(0.0f);
        auto vsum2 = _mm256_set1_ps(0.0f);
        auto vsum3 = _mm256_set1_ps(0.0f);
        size_t i = 0;
        for (; i + vec_len_f32_avx2 <= n; i += vec_len_f32_avx2) {
            auto va = mm256_uni_loadu_ps(a + i);
            vsum0 = _mm256_fmadd_ps(va, mm256_uni_loadu_ps(b + i), vsum0);
            vsum1 = _mm256_fmadd_ps(va, mm256_uni_loadu_ps(b + i + n), vsum1);
            vsum2 = _mm256_fmadd_ps(va, mm256_uni_loadu_ps(b + i + 2 * n), vsum2);
            vsum3 = _mm256_fmadd_ps(va, mm256_uni_loadu_ps(b + i + 3 * n), vsum3);
        }
        hsum(vsum0);
        hsum(vsum1);
        hsum(vsum2);
        hsum(vsum3);
        float sum0 = _mm256_cvtss_f32(vsum0);
        float sum1 = _mm256_cvtss_f32(vsum1);
        float sum2 = _mm256_cvtss_f32(vsum2);
        float sum3 = _mm256_cvtss_f32(vsum3);
        for (; i < n; i++) {
            sum0 += a[i] * b[i];
            sum1 += a[i] * b[i + n];
            sum2 += a[i] * b[i + 2 * n];
            sum3 += a[i] * b[i + 3 * n];
        }
        c[0] = sum0;
        c[1] = sum1;
        c[2] = sum2;
        c[3] = sum3;
        c += 4;
        b +=  4 * n;
    }
    for (; j < block_size; j++) {
        auto vsum = _mm256_set1_ps(0.0f);
        size_t i = 0;
        for (; i + vec_len_f32_avx2 <= n; i += vec_len_f32_avx2) {
            auto va = mm256_uni_loadu_ps(a + i);
            vsum = _mm256_fmadd_ps(va, mm256_uni_loadu_ps(b + i), vsum);
        }
        hsum(vsum);
        float sum = _mm256_cvtss_f32(vsum);
        for (; i < n; i++) {
            sum += a[i] * b[i];
        }
        b += n;
        *c++ = sum;
    }
    return;
#endif
    for (size_t j = 0; j < block_size; j++) {
        float sum = 0;
        for (size_t i = 0; i < n; i++) {
            sum += a[i] * b[i];
        }
        b += n;
        *c++ = sum;
    }
}

template<typename TA>
static void dot_product_block(TA* a, uint8_t* b, float* c, size_t n, size_t block_size) {
    // The layout for per token per head:
    // |scale(f32)|zeropoint(f32)|quantized feature(u8,idx_1)|quantized feature(u8,idx_2)|...|quantized feature(u8,idx_S)|
    // The quantized feature will start from 8bytes=sizeof(float)+sizeof(float)
#if defined(HAVE_AVX512F)
    size_t j = 0;
    for (; j + 4 <= block_size; j += 4) {
        auto vsum0 = _mm512_setzero_ps();
        auto vsum1 = _mm512_setzero_ps();
        auto vsum2 = _mm512_setzero_ps();
        auto vsum3 = _mm512_setzero_ps();
        auto b0 = reinterpret_cast<float*>(b);
        auto b1 = reinterpret_cast<float*>(b + n + 8);
        auto b2 = reinterpret_cast<float*>(b + (n + 8) * 2);
        auto b3 = reinterpret_cast<float*>(b + (n + 8) * 3);
        auto v_zp0 = _mm512_set1_ps(b0[1]);
        auto v_zp1 = _mm512_set1_ps(b1[1]);
        auto v_zp2 = _mm512_set1_ps(b2[1]);
        auto v_zp3 = _mm512_set1_ps(b3[1]);
        size_t i = 0;
        b += 8;
        for (; i + vec_len_f32_avx512 <= n; i += vec_len_f32_avx512) {
            auto va = mm512_uni_loadu_ps(a + i);
            auto vb0 = _mm512_sub_ps(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<__m128i*>(b + i)))), v_zp0);
            auto vb1 = _mm512_sub_ps(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<__m128i*>(b + i + n + 8)))), v_zp1);
            auto vb2 = _mm512_sub_ps(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<__m128i*>(b + i + 2 * (n + 8))))), v_zp2);
            auto vb3 = _mm512_sub_ps(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<__m128i*>(b + i + 3 * (n + 8))))), v_zp3);

            vsum0 = _mm512_fmadd_ps(va, vb0, vsum0);
            vsum1 = _mm512_fmadd_ps(va, vb1, vsum1);
            vsum2 = _mm512_fmadd_ps(va, vb2, vsum2);
            vsum3 = _mm512_fmadd_ps(va, vb3, vsum3);
        }
        float sum0 = _mm512_reduce_add_ps(vsum0);
        float sum1 = _mm512_reduce_add_ps(vsum1);
        float sum2 = _mm512_reduce_add_ps(vsum2);
        float sum3 = _mm512_reduce_add_ps(vsum3);
        for (; i < n; i++) {
            sum0 += a[i] * (b[i] - b0[1]);
            sum1 += a[i] * (b[i + n + 8] - b1[1]);
            sum2 += a[i] * (b[i + 2 * (n + 8)] - b2[1]);
            sum3 += a[i] * (b[i + 3 * (n + 8)] - b3[1]);
        }
        c[0] = sum0 * b0[0];
        c[1] = sum1 * b1[0];
        c[2] = sum2 * b2[0];
        c[3] = sum3 * b3[0];
        c += 4;
        b +=  4 * (n + 8) - 8;
    }
    for (; j < block_size; j++) {
        auto vsum = _mm512_setzero_ps();
        auto b0 = reinterpret_cast<float*>(b);
        auto v_zp = _mm512_set1_ps(b0[1]);
        size_t i = 0;
        b += 8;
        for (; i + vec_len_f32_avx512 <= n; i += vec_len_f32_avx512) {
            auto va = mm512_uni_loadu_ps(a + i);
            auto vb = _mm512_sub_ps(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<__m128i*>(b + i)))), v_zp);
            vsum = _mm512_fmadd_ps(va, vb, vsum);
        }
        float sum = _mm512_reduce_add_ps(vsum);
        for (; i < n; i++) {
            sum += a[i] * (b[i] - b0[1]);
        }
        b += n;
        *c++ = sum * b0[0];
    }
    return;
#elif defined(HAVE_AVX2)
    size_t j = 0;
    for (; j + 4 <= block_size; j += 4) {
        auto vsum0 = _mm256_setzero_ps();
        auto vsum1 = _mm256_setzero_ps();
        auto vsum2 = _mm256_setzero_ps();
        auto vsum3 = _mm256_setzero_ps();
        auto b0 = reinterpret_cast<float*>(b);
        auto b1 = reinterpret_cast<float*>(b + n + 8);
        auto b2 = reinterpret_cast<float*>(b + (n + 8) * 2);
        auto b3 = reinterpret_cast<float*>(b + (n + 8) * 3);
        auto v_zp0 = _mm256_set1_ps(b0[1]);
        auto v_zp1 = _mm256_set1_ps(b1[1]);
        auto v_zp2 = _mm256_set1_ps(b2[1]);
        auto v_zp3 = _mm256_set1_ps(b3[1]);
        size_t i = 0;
        b += 8;
        for (; i + vec_len_f32_avx2 <= n; i += vec_len_f32_avx2) {
            auto va = mm256_uni_loadu_ps(a + i);
            auto vb0 = _mm256_sub_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<__m128i*>(b + i)))), v_zp0);
            auto vb1 = _mm256_sub_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<__m128i*>(b + i + n + 8)))), v_zp1);
            auto vb2 = _mm256_sub_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<__m128i*>(b + i + 2 * (n + 8))))), v_zp2);
            auto vb3 = _mm256_sub_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<__m128i*>(b + i + 3 * (n + 8))))), v_zp3);

            vsum0 = _mm256_fmadd_ps(va, vb0, vsum0);
            vsum1 = _mm256_fmadd_ps(va, vb1, vsum1);
            vsum2 = _mm256_fmadd_ps(va, vb2, vsum2);
            vsum3 = _mm256_fmadd_ps(va, vb3, vsum3);
        }
        hsum(vsum0);
        hsum(vsum1);
        hsum(vsum2);
        hsum(vsum3);
        float sum0 = _mm256_cvtss_f32(vsum0);
        float sum1 = _mm256_cvtss_f32(vsum1);
        float sum2 = _mm256_cvtss_f32(vsum2);
        float sum3 = _mm256_cvtss_f32(vsum3);
        for (; i < n; i++) {
            sum0 += a[i] * (b[i] - b0[1]);
            sum1 += a[i] * (b[i + n + 8] - b1[1]);
            sum2 += a[i] * (b[i + 2 * (n + 8)] - b2[1]);
            sum3 += a[i] * (b[i + 3 * (n + 8)] - b3[1]);
        }
        c[0] = sum0 * b0[0];
        c[1] = sum1 * b1[0];
        c[2] = sum2 * b2[0];
        c[3] = sum3 * b3[0];
        c += 4;
        b +=  4 * (n + 8) - 8;
    }
    for (; j < block_size; j++) {
        auto vsum = _mm256_setzero_ps();
        auto b0 = reinterpret_cast<float*>(b);
        auto v_zp = _mm256_set1_ps(b0[1]);
        size_t i = 0;
        b += 8;
        for (; i + vec_len_f32_avx2 <= n; i += vec_len_f32_avx2) {
            auto va = mm256_uni_loadu_ps(a + i);
            auto vb = _mm256_sub_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<__m128i*>(b + i)))), v_zp);
            vsum = _mm256_fmadd_ps(va, vb, vsum);
        }
        hsum(vsum);
        float sum = _mm256_cvtss_f32(vsum);
        for (; i < n; i++) {
            sum += a[i] * (b[i] - b0[1]);
        }
        b += n;
        *c++ = sum * b0[0];
    }
    return;
#endif
    for (size_t j = 0; j < block_size; j++) {
        float sum = 0;
        auto b0 = reinterpret_cast<float*>(b);
        b += 8;
        for (size_t i = 0; i < n; i++) {
            sum += a[i] * (b[i] - b0[1]);
        }
        b += n;
        *c++ = sum * b0[0];
    }
}

template<typename T>
static void attn_reduce(T* dst, float* temp, size_t M, size_t S, size_t temp_stride) {
    size_t i = 0;
#if defined(HAVE_AVX512F)
    for (; i + vec_len_f32_avx512 <= S; i+= vec_len_f32_avx512) {
        auto* src = temp + i;
        auto result_vec_fp32 = _mm512_setzero_ps();
        for (size_t m = 0; m < M; m++) {
            auto o_vec_fp32 = _mm512_loadu_ps(src);
            result_vec_fp32 = _mm512_add_ps(result_vec_fp32, o_vec_fp32);
            src += temp_stride;
        }
        // save to bf16
        mm512_uni_storeu_ps(dst + i, result_vec_fp32);
    }
#elif defined(HAVE_AVX2)
    for (; i + vec_len_f32_avx2 <= S; i += vec_len_f32_avx2) {
        auto* src = temp + i;
        auto result_vec_fp32 = _mm256_set1_ps(0.0f);
        for (size_t m = 0; m < M; m++) {
            auto o_vec_fp32 = mm256_uni_loadu_ps(src);
            result_vec_fp32 = _mm256_add_ps(result_vec_fp32, o_vec_fp32);
            src += temp_stride;
        }
        mm256_uni_storeu_ps(dst + i, result_vec_fp32);
    }
#endif
    for (; i < S; i++) {
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

template <typename T, typename T2>
static void mha_single_token_kernel(const ov::intel_cpu::PlainTensor& query,
                             const ov::intel_cpu::PlainTensor& present_key,
                             const ov::intel_cpu::PlainTensor& present_value,
                             const ov::intel_cpu::PlainTensor& alibi_mask,
                             const ov::intel_cpu::PlainTensor& attention_mask,
                             const ov::intel_cpu::PlainTensor& beams,
                             size_t max_context_len,
                             const ov::intel_cpu::PlainTensor& context_lens,
                             ov::intel_cpu::PlainTensor& output_emb,
                             ov::intel_cpu::PlainTensor& buf_attn_w,
                             ov::intel_cpu::PlainTensor& buf_attn_score,
                             bool has_out_transpose,
                             bool auto_causal,
                             float d_scale,
                             const ov::intel_cpu::PlainTensor& past_k_scale_zp,
                             const ov::intel_cpu::PlainTensor& past_v_scale_zp,
                             ov::intel_cpu::PlainTensor& head_sum) {
    ov::intel_cpu::PlainTensor causal_mask;
    bool select_nfltmax_at_0 = false;
    auto B = query.size(0);
    auto H = query.size(1);
    auto q_len = query.size(2);
    auto S = query.size(3);
    auto h_group_num = present_value.size(1);
    size_t h_each_group_len = 1;
    bool is_pagedattn = context_lens;
    size_t block_size = present_value.size(2);
    if (h_group_num != H) {
        h_each_group_len = H / h_group_num;
    }
    if (d_scale == 0.0f)
        d_scale = 1.0f / sqrt(S);
    auto nthr = parallel_get_max_threads();
    size_t kv_len;
    if (is_pagedattn) {
        kv_len = max_context_len;
    } else {
        kv_len = present_key.size(2);
    }

#if defined(HAVE_AVX2) && !defined(HAVE_AVX512F)
    // avx2 will pre-compute the zero point and try to save the sub instruction in the dot_product,
    //  but it seems not necessary for avx512. Possible reason may be that for avx2 the cost of dot_product
    //  is larger than the memory access time, but for avx512 is not and the cost of pre-compute is a pure increase.
    bool pastkv_is_int8 = past_k_scale_zp;
    if (pastkv_is_int8 && !is_pagedattn) {
        // be sure no false sharing
        head_sum.resize<float>({B, H, q_len, 16});
        parallel_for3d(B, H, q_len, [&](size_t b, size_t h, size_t pq) {
            *head_sum.ptr<float>(b, h, pq) = sum_q_head(query.ptr<T>(b, h, pq), S);
        });
    }
#endif

    // TODO: refactor to seperate files
    if (is_pagedattn) {
        // if present_key is true, it means q*k is already computed in the caller
        if (present_key) {
            if (B >= static_cast<size_t>(nthr)) {
                parallel_for2d_dynamic(B, beams.m_dims[1], [&](size_t b, size_t pk_in_blocks) {
                    auto context_len = static_cast<size_t>(context_lens.ptr<int32_t>()[b]);
                    // kv_len must be valid
                    auto pk = pk_in_blocks * block_size;
                    if (pk < context_len) {
                        auto block_number = beams.ptr<int32_t>(b)[pk_in_blocks];
                        for (size_t h_group = 0; h_group < h_group_num; h_group++) {
                            for (size_t pq = 0; pq < q_len; pq++) {
                                for (size_t h = h_group * h_each_group_len; h < (h_group + 1) * h_each_group_len; h++) {
                                    dot_product_block(query.ptr<T>(b, h, pq), present_key.ptr<T2>(block_number, h_group),
                                        buf_attn_w.ptr<float>(b, h, pq) + pk, S, std::min(block_size, context_len - pk));
                                }
                            }
                        }
                    }
                });
            } else {
                parallel_for3d_dynamic(B, beams.m_dims[1], h_group_num, [&](size_t b, size_t pk_in_blocks, size_t h_group) {
                    auto context_len = static_cast<size_t>(context_lens.ptr<int32_t>()[b]);
                    // kv_len must be valid
                    auto pk = pk_in_blocks * block_size;
                    if (pk < context_len) {
                        auto block_number = beams.ptr<int32_t>(b)[pk_in_blocks];
                        for (size_t pq = 0; pq < q_len; pq++) {
                            for (size_t h = h_group * h_each_group_len; h < (h_group + 1) * h_each_group_len; h++) {
                                dot_product_block(query.ptr<T>(b, h, pq), present_key.ptr<T2>(block_number, h_group),
                                    buf_attn_w.ptr<float>(b, h, pq) + pk, S, std::min(block_size, context_len - pk));
                            }
                        }
                    }
                });
            }
        }

        parallel_for3d_dynamic(B, H, q_len, [&](size_t b, size_t h, size_t pq) {
            auto cur_kv_len = static_cast<size_t>(context_lens.ptr<int32_t>()[b]);
            auto ncausal = cur_kv_len;
            // apply attention mask & sofmax
            float* alibi_ptr = alibi_mask ? &alibi_mask.at<float>({b, h, pq, 0}, true) : nullptr;
            uint8_t* attn_mask_ptr = nullptr;
            auto attn_mask_prec = attention_mask.get_precision();
            if (attention_mask)
                attn_mask_ptr = reinterpret_cast<uint8_t*>(&attention_mask.at<T>({b, h, pq, 0}, true));
            uint8_t* cmask_ptr = causal_mask ? &causal_mask.at<uint8_t>({b, h, pq, 0}, true) : nullptr;
            attn_softmax_kernel(buf_attn_w.ptr<float>(b, h, pq),
                                buf_attn_w.ptr<float>(b, h, pq),
                                d_scale,
                                alibi_ptr,
                                attn_mask_ptr,
                                cmask_ptr,
                                select_nfltmax_at_0,
                                ncausal,
                                cur_kv_len,
                                attn_mask_prec,
                                ov::element::f32);
        });

        // attn_w * V
        // there are enough works for each thread
        if (B >= static_cast<size_t>(nthr)) {
            buf_attn_score.resize<float>({static_cast<size_t>(nthr), q_len, h_each_group_len, S});
            parallel_for2d_dynamic(B, h_group_num, [&](size_t b, size_t h_group) {
                auto ithr = parallel_get_thread_num();
                auto context_len = static_cast<size_t>(context_lens.ptr<int32_t>()[b]);
                memset(buf_attn_score.ptr<float>(ithr), 0, q_len * h_each_group_len * S * sizeof(float));
                for (size_t pv = 0; pv < context_len; pv += block_size) {
                    size_t pv_in_blocks = pv / block_size;
                    auto block_number = beams.ptr<int32_t>(b)[pv_in_blocks];
                    auto* v = present_value.ptr<T2>(block_number, h_group);
                    for (size_t pq = 0; pq < q_len; pq++) {
                        for (size_t h = h_group * h_each_group_len, group_idx = 0; h < (h_group + 1) * h_each_group_len; h++, group_idx++) {
                            attn_acc_value_block(buf_attn_score.ptr<float>(ithr, pq, group_idx),
                                                 buf_attn_w.ptr<float>(b, h, pq) + pv,
                                                 v,
                                                 S,
                                                 std::min(block_size, context_len - pv));
                        }
                    }
                }
                // convert to dst
                for (size_t pq = 0; pq < q_len; pq++)
                    for (size_t h = h_group * h_each_group_len, group_idx = 0; h < (h_group + 1) * h_each_group_len; h++, group_idx++)
                        cvt_copy(output_emb.ptr<T>(b, pq, h * S), buf_attn_score.ptr<float>(ithr, pq, group_idx), S);
            });
            return;
        }
        buf_attn_score.resize<float>({static_cast<size_t>(nthr), B, q_len, H, S});
        // buf_attn_w {B, H, q_len, kv_len}
        parallel_nt_static(nthr, [&](const size_t ithr, const size_t nthr) {
            memset(buf_attn_score.ptr<float>(ithr, 0, 0, 0, 0), 0, buf_attn_score.stride(0) * sizeof(float));
        });

        auto kv_len_in_blocks = beams.m_dims[1];
        parallel_for3d_dynamic(B, kv_len_in_blocks, h_group_num, [&](size_t b, size_t pv_in_blocks, size_t h_group) {
            auto ithr = parallel_get_thread_num();
            auto context_len = static_cast<size_t>(context_lens.ptr<int32_t>()[b]);
            auto pv = pv_in_blocks * block_size;
            // kv_len must be valid
            if (pv < context_len) {
                auto block_number = beams.ptr<int32_t>(b)[pv_in_blocks];
                auto* v = present_value.ptr<T2>(block_number, h_group);
                for (size_t pq = 0; pq < q_len; pq++) {
                    for (size_t h = h_group * h_each_group_len; h < (h_group + 1) * h_each_group_len; h++) {
                        attn_acc_value_block(buf_attn_score.ptr<float>(ithr, b, pq, h),
                                             buf_attn_w.ptr<float>(b, h, pq) + pv,
                                             v,
                                             S,
                                             std::min(block_size, context_len - pv));
                    }
                }
            }
        });
    } else {
        {
            auto _profile = ov::intel_cpu::Profile("QK_GEMM");
            parallel_nt_static(nthr, [&](const size_t ithr, const size_t nthr) {
                size_t start{0}, end{0};
                splitter(B * h_group_num * kv_len, nthr, ithr, start, end);

                size_t b, h_group, pk;
                if (start < end) {
                    parallel_it_init(start, pk, kv_len, b, B, h_group, h_group_num);
                    if (q_len == 1 && h_each_group_len == 1) {
                        if (B == 1) {
                            // the memory will be continuous when b==1
                            for (size_t iwork = start; iwork < end; ++iwork) {
                                auto p = past_k_scale_zp.ptr<float>(pk, 0, h_group);
                                auto p_k = present_key.ptr<T2>(0, h_group, pk);
                                prefetch_bytes(S, _MM_HINT_T0, 4096, p_k);
                                buf_attn_w.ptr<float>(0, h_group, 0)[pk] =
                                        dot_product(query.ptr<T>(0, h_group), p_k,
                                            S, p, p + 1, head_sum.ptr<float>(0, h_group));
                                parallel_it_step(pk, kv_len, b, B, h_group, h_group_num);
                            }
                        } else {
                            for (size_t iwork = start; iwork < end; ++iwork) {
                                auto b_kv = beams ? beams.ptr<int32_t>(b)[pk] : b;
                                auto p = past_k_scale_zp.ptr<float>(pk, b_kv, h_group);
                                auto p_k = present_key.ptr<T2>(b_kv, h_group, pk);
                                buf_attn_w.ptr<float>(b, h_group, 0)[pk] =
                                        dot_product(query.ptr<T>(b, h_group), p_k,
                                            S, p, p + 1, head_sum.ptr<float>(b, h_group));
                                parallel_it_step(pk, kv_len, b, B, h_group, h_group_num);
                            }
                        }
                    } else {
                        for (size_t iwork = start; iwork < end; ++iwork) {
                            auto b_kv = beams ? beams.ptr<int32_t>(b)[pk] : b;
                            for (size_t pq = 0; pq < q_len; pq++) {
                                auto p = past_k_scale_zp.ptr<float>(pk, b_kv, h_group);
                                for (size_t h = h_group * h_each_group_len; h < (h_group + 1) * h_each_group_len; h++) {
                                    buf_attn_w.ptr<float>(b, h, pq)[pk] =
                                            dot_product(query.ptr<T>(b, h, pq), present_key.ptr<T2>(b_kv, h_group, pk),
                                                S, p, p + 1, head_sum.ptr<float>(b, h, pq));
                                }
                            }
                            parallel_it_step(pk, kv_len, b, B, h_group, h_group_num);
                        }
                    }
                }
            });
        }
        {
            auto _profile = ov::intel_cpu::Profile("SOFTMAX");
            parallel_for3d(B, H, q_len, [&](size_t b, size_t h, size_t pq) {
                auto cur_kv_len = kv_len;
                auto ncausal = auto_causal ? (cur_kv_len - q_len + pq + 1) : cur_kv_len;
                // apply attention mask & sofmax
                float* alibi_ptr = alibi_mask ? &alibi_mask.at<float>({b, h, pq, 0}, true) : nullptr;
                uint8_t* attn_mask_ptr = nullptr;
                auto attn_mask_prec = attention_mask.get_precision();
                if (attention_mask)
                    attn_mask_ptr = reinterpret_cast<uint8_t*>(&attention_mask.at<T>({b, h, pq, 0}, true));
                uint8_t* cmask_ptr = causal_mask ? &causal_mask.at<uint8_t>({b, h, pq, 0}, true) : nullptr;
                attn_softmax_kernel(buf_attn_w.ptr<float>(b, h, pq),
                                    buf_attn_w.ptr<float>(b, h, pq),
                                    d_scale,
                                    alibi_ptr,
                                    attn_mask_ptr,
                                    cmask_ptr,
                                    select_nfltmax_at_0,
                                    ncausal,
                                    cur_kv_len,
                                    attn_mask_prec,
                                    ov::element::f32);
            });
        }
        // attn_w * V
        // Fast Path if there are enough works for each thread
        {
            auto _profile = ov::intel_cpu::Profile("WV_Fast_GEMM");
            if (B >= static_cast<size_t>(nthr)) {
                buf_attn_score.resize<float>({static_cast<size_t>(nthr), q_len, h_each_group_len, S});
                parallel_for2d(B, h_group_num, [&](size_t b, size_t h_group) {
                    auto ithr = parallel_get_thread_num();
                    memset(buf_attn_score.ptr<float>(ithr), 0, q_len * h_each_group_len * S * sizeof(float));
                    for (size_t pv = 0; pv < kv_len; pv++) {
                        auto b_kv = beams ? beams.ptr<int32_t>(b)[pv] : b;
                        auto* v = present_value.ptr<T2>(b_kv, h_group, pv);
                        auto p = past_v_scale_zp.ptr<float>(pv, b_kv, h_group);
                        for (size_t pq = 0; pq < q_len; pq++) {
                            for (size_t h = h_group * h_each_group_len, group_idx = 0; h < (h_group + 1) * h_each_group_len; h++, group_idx++) {
                                attn_acc_value(buf_attn_score.ptr<float>(ithr, pq, group_idx),
                                            buf_attn_w.ptr<float>(b, h, pq)[pv],
                                            v,
                                            S,
                                            p + 0,
                                            p + 1);
                            }
                        }
                    }
                    // convert to dst
                    for (size_t pq = 0; pq < q_len; pq++) {
                        for (size_t h = h_group * h_each_group_len, group_idx = 0; h < (h_group + 1) * h_each_group_len;
                            h++, group_idx++) {
                            auto* dst = has_out_transpose ? output_emb.ptr<T>(b, pq, h * S) : output_emb.ptr<T>(b, h, pq);
                            cvt_copy(dst, buf_attn_score.ptr<float>(ithr, pq, group_idx), S);
                        }
                    }
                });
                return;
            }
        }
        buf_attn_score.resize<float>({static_cast<size_t>(nthr), B, q_len, H, S});
        // buf_attn_w {B, H, q_len, kv_len}
        {
            auto _profile = ov::intel_cpu::Profile("WV_GEMM");
            parallel_nt_static(nthr, [&](const size_t ithr, const size_t nthr) {
                size_t start{0}, end{0};
                splitter(B * h_group_num * kv_len, nthr, ithr, start, end);

                memset(buf_attn_score.ptr<float>(ithr, 0, 0, 0, 0), 0, buf_attn_score.stride(0) * sizeof(float));

                size_t b, h_group, pv;
                if (start < end) {
                    parallel_it_init(start, pv, kv_len, b, B, h_group, h_group_num);
                    if (q_len == 1 && h_each_group_len == 1) {
                        for (size_t iwork = start; iwork < end; ++iwork) {
                            auto b_kv = beams ? beams.ptr<int32_t>(b)[pv] : b;
                            auto* v = present_value.ptr<T2>(b_kv, h_group, pv);
                            auto p = past_v_scale_zp.ptr<float>(pv, b_kv, h_group);
                            attn_acc_value(buf_attn_score.ptr<float>(ithr, b, 0, h_group),
                                        buf_attn_w.ptr<float>(b, h_group, 0, pv)[0],
                                        v,
                                        S,
                                        p + 0,
                                        p + 1);
                            parallel_it_step(pv, kv_len, b, B, h_group, h_group_num);
                        }
                    } else {
                        for (size_t iwork = start; iwork < end; ++iwork) {
                            auto b_kv = beams ? beams.ptr<int32_t>(b)[pv] : b;
                            auto* v = present_value.ptr<T2>(b_kv, h_group, pv);
                            auto p = past_v_scale_zp.ptr<float>(pv, b_kv, h_group);
                            for (size_t pq = 0; pq < q_len; pq++) {
                                for (size_t h = h_group * h_each_group_len; h < (h_group + 1) * h_each_group_len; h++) {
                                    attn_acc_value(buf_attn_score.ptr<float>(ithr, b, pq, h),
                                                buf_attn_w.ptr<float>(b, h, pq)[pv],
                                                v,
                                                S,
                                                p + 0,
                                                p + 1);
                                }
                            }
                            parallel_it_step(pv, kv_len, b, B, h_group, h_group_num);
                        }
                    }
                }
            });
        }
    }

    parallel_for3d(B, H, q_len, [&](size_t b, size_t h, size_t pq) {
        auto* temp = buf_attn_score.ptr<float>(0, b, pq, h);
        size_t temp_stride = buf_attn_score.stride(0);
        auto* dst = has_out_transpose ? output_emb.ptr<T>(b, pq, h * S) : output_emb.ptr<T>(b, h, pq);
        attn_reduce(dst, temp, nthr, S, temp_stride);
    });
}

void mha_single_token(const ov::intel_cpu::PlainTensor& query,
                      const ov::intel_cpu::PlainTensor& present_key,
                      const ov::intel_cpu::PlainTensor& present_value,
                      const ov::intel_cpu::PlainTensor& alibi_mask,
                      const ov::intel_cpu::PlainTensor& attention_mask,
                      const ov::intel_cpu::PlainTensor& beams,
                      size_t max_context_len,
                      const ov::intel_cpu::PlainTensor& context_lens,
                      ov::intel_cpu::PlainTensor& output_emb,
                      ov::intel_cpu::PlainTensor& buf_attn_w,
                      ov::intel_cpu::PlainTensor& buf_attn_score,
                      bool has_out_transpose,
                      bool auto_causal,
                      float d_scale,
                      const ov::intel_cpu::PlainTensor& past_k_scale_zp,
                      const ov::intel_cpu::PlainTensor& past_v_scale_zp,
                      ov::intel_cpu::PlainTensor& head_sum) {
    if (query.get_precision() == ov::element::bf16) {
        if (present_key.get_precision() == ov::element::u8) {
            mha_single_token_kernel<ov::bfloat16, uint8_t>(query,
                                                           present_key,
                                                           present_value,
                                                           alibi_mask,
                                                           attention_mask,
                                                           beams,
                                                           max_context_len,
                                                           context_lens,
                                                           output_emb,
                                                           buf_attn_w,
                                                           buf_attn_score,
                                                           has_out_transpose,
                                                           auto_causal,
                                                           d_scale,
                                                           past_k_scale_zp,
                                                           past_v_scale_zp,
                                                           head_sum);
        } else {
            mha_single_token_kernel<ov::bfloat16, ov::bfloat16>(query,
                                                                present_key,
                                                                present_value,
                                                                alibi_mask,
                                                                attention_mask,
                                                                beams,
                                                                max_context_len,
                                                                context_lens,
                                                                output_emb,
                                                                buf_attn_w,
                                                                buf_attn_score,
                                                                has_out_transpose,
                                                                auto_causal,
                                                                d_scale,
                                                                past_k_scale_zp,
                                                                past_v_scale_zp,
                                                                head_sum);
        }
    } else if (query.get_precision() == ov::element::f32) {
        if (present_key.get_precision() == ov::element::u8) {
            mha_single_token_kernel<float, uint8_t>(query,
                                                    present_key,
                                                    present_value,
                                                    alibi_mask,
                                                    attention_mask,
                                                    beams,
                                                    max_context_len,
                                                    context_lens,
                                                    output_emb,
                                                    buf_attn_w,
                                                    buf_attn_score,
                                                    has_out_transpose,
                                                    auto_causal,
                                                    d_scale,
                                                    past_k_scale_zp,
                                                    past_v_scale_zp,
                                                    head_sum);
        } else if (present_key.get_precision() == ov::element::f16) {
            mha_single_token_kernel<float, ov::float16>(query,
                                                        present_key,
                                                        present_value,
                                                        alibi_mask,
                                                        attention_mask,
                                                        beams,
                                                        max_context_len,
                                                        context_lens,
                                                        output_emb,
                                                        buf_attn_w,
                                                        buf_attn_score,
                                                        has_out_transpose,
                                                        auto_causal,
                                                        d_scale,
                                                        past_k_scale_zp,
                                                        past_v_scale_zp,
                                                        head_sum);
        } else {
            mha_single_token_kernel<float, float>(query,
                                                present_key,
                                                present_value,
                                                alibi_mask,
                                                attention_mask,
                                                beams,
                                                max_context_len,
                                                context_lens,
                                                output_emb,
                                                buf_attn_w,
                                                buf_attn_score,
                                                has_out_transpose,
                                                auto_causal,
                                                d_scale,
                                                past_k_scale_zp,
                                                past_v_scale_zp,
                                                head_sum);
        }
    } else {
        OPENVINO_THROW("Unsupported precision: ", query.get_precision());
    }
}
}  // namespace XARCH
}  // namespace Cpu
}  // namespace Extensions
}  // namespace ov