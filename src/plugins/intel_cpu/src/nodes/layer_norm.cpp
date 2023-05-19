// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>

#include "layer_norm.h"
#include "transformations/cpu_opset/x64/op/layer_norm.hpp"
#include <utils/shape_inference/shape_inference_internal_dyn.hpp>
#include <cpu/x64/jit_generator.hpp>
#include "ie_parallel.hpp"
#ifdef _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#include <immintrin.h>
#endif
using namespace InferenceEngine;
using namespace ov::intel_cpu;
using namespace ov::intel_cpu::node;
using namespace dnnl::impl::cpu::x64;
using namespace Xbyak;

#define THROW_ERROR IE_THROW() << getTypeStr() << " node with name '" << getName() << "' "
namespace {
inline __m256i get_mask(int N7) {
	static __m256i mask[] = {
		_mm256_set_epi32( 0, 0, 0, 0, 0, 0, 0, 0),
		_mm256_set_epi32( 0, 0, 0, 0, 0, 0, 0,-1),
		_mm256_set_epi32( 0, 0, 0, 0, 0, 0,-1,-1),
		_mm256_set_epi32( 0, 0, 0, 0, 0,-1,-1,-1),
		_mm256_set_epi32( 0, 0, 0, 0,-1,-1,-1,-1),
		_mm256_set_epi32( 0, 0, 0,-1,-1,-1,-1,-1),
		_mm256_set_epi32( 0, 0,-1,-1,-1,-1,-1,-1),
		_mm256_set_epi32( 0,-1,-1,-1,-1,-1,-1,-1),
		_mm256_set_epi32(-1,-1,-1,-1,-1,-1,-1,-1),
	};
	return _mm256_loadu_si256(&mask[N7]);
}

// https://stackoverflow.com/questions/23189488/horizontal-sum-of-32-bit-floats-in-256-bit-avx-vector
static inline float _mm256_reduce_add_ps(__m256 x) {
    /* ( x3+x7, x2+x6, x1+x5, x0+x4 ) */
    const __m128 x128 = _mm_add_ps(_mm256_extractf128_ps(x, 1), _mm256_castps256_ps128(x));
    /* ( -, -, x1+x3+x5+x7, x0+x2+x4+x6 ) */
    const __m128 x64 = _mm_add_ps(x128, _mm_movehl_ps(x128, x128));
    /* ( -, -, -, x0+x1+x2+x3+x4+x5+x6+x7 ) */
    const __m128 x32 = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
    /* Conversion to float is a no-op on x86-64 */
    return _mm_cvtss_f32(x32);
}

static inline float sum(float* src, size_t ele_num) {
    size_t i = 0;
    __m256 s;
    s = _mm256_xor_ps(s, s);
    for (; i < ele_num / 8 * 8; i += 8) {
        auto a0 = _mm256_loadu_ps(src);
        s = _mm256_add_ps(s, a0);
        src += 8;
    }
    if (i != ele_num) {
        auto msk = get_mask(ele_num - i);
        auto a0 = _mm256_maskload_ps(src, msk);
        s = _mm256_add_ps(s, a0);
    }
    return _mm256_reduce_add_ps(s);
}

static inline float sum_power2(float* src, float mean, size_t ele_num) {
    size_t i = 0;
    __m256 s, zero;
    s = _mm256_xor_ps(s, s);
    zero = _mm256_xor_ps(zero, zero);
    auto m = _mm256_set1_ps(mean);
    for (; i < ele_num / 8 * 8; i += 8) {
        auto a0 = _mm256_loadu_ps(src);
        a0 = _mm256_sub_ps(a0, m);
        s = _mm256_fmadd_ps(a0, a0, s);
        src += 8;
    }
    if (i != ele_num) {
        auto msk = get_mask(ele_num - i);
        auto a0 = _mm256_maskload_ps(src, msk);
        a0 = _mm256_sub_ps(a0, m);
        a0 = _mm256_blendv_ps(zero, a0, _mm256_castsi256_ps(msk));
        s = _mm256_fmadd_ps(a0, a0, s);
    }
    return _mm256_reduce_add_ps(s);
}

static inline void mvn(float* src, float mean, float var, size_t ele_num, float* dst) {
    size_t i = 0;
    auto m = _mm256_set1_ps(mean);
    auto v = _mm256_set1_ps(var);
    for (; i < ele_num / 8 * 8; i += 8) {
        auto a0_f = _mm256_loadu_ps(src);
        a0_f = _mm256_sub_ps(a0_f, m);
        a0_f = _mm256_mul_ps(a0_f, v);
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst), (__m256i)a0_f);

        src += 8;
        dst += 8;
    }
    if (i != ele_num) {
        auto msk = get_mask(ele_num - i);
        auto a0_f = _mm256_maskload_ps(src, msk);
        a0_f = _mm256_sub_ps(a0_f, m);
        a0_f = _mm256_mul_ps(a0_f, v);
        _mm256_maskstore_ps(dst, msk, a0_f);
    }
}

inline void mvn_line(float* src, size_t ele_num, float eps, bool inside_sqrt, float* dst) {
    // mean
    float mean = sum(src, ele_num) / ele_num;
    // var
    float var = sum_power2(src, mean, ele_num) / ele_num;
    var = 1.0f / (inside_sqrt ? std::sqrt(var + eps) : std::sqrt(var) + eps);
    // mvn
    mvn(src, mean, var, ele_num, dst);
}

static inline void mvn_scale_bias(float* src, float mean, float var, size_t ele_num, float* dst, float* scale, float* bias) {
    size_t i = 0;
    auto m = _mm256_set1_ps(mean);
    auto v = _mm256_set1_ps(var);
    for (; i < ele_num / 8 * 8; i += 8) {
        auto a0_f = _mm256_loadu_ps(src);
        auto b = _mm256_loadu_ps(bias);
        auto s = _mm256_loadu_ps(scale);
        a0_f = _mm256_sub_ps(a0_f, m);
        a0_f = _mm256_mul_ps(a0_f, v);
        a0_f = _mm256_fmadd_ps(a0_f, s, b);
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst), (__m256i)a0_f);

        src += 8;
        dst += 8;
        bias += 8;
        scale += 8;
    }
    if (i != ele_num) {
        auto msk = get_mask(ele_num - i);
        auto a0_f = _mm256_maskload_ps(src, msk);
        auto b = _mm256_maskload_ps(bias, msk);
        auto s = _mm256_maskload_ps(scale, msk);
        a0_f = _mm256_sub_ps(a0_f, m);
        a0_f = _mm256_mul_ps(a0_f, v);
        a0_f = _mm256_fmadd_ps(a0_f, s, b);
        _mm256_maskstore_ps(dst, msk, a0_f);
    }
}

inline void mvn_line_scale_bias(float* src, size_t ele_num, float eps, bool inside_sqrt, float *dst, float* scale, float* bias) {
    // mean
    float mean = sum(src, ele_num) / ele_num;
    // var
    float var = sum_power2(src, mean, ele_num) / ele_num;
    var = 1.0f / (inside_sqrt ? std::sqrt(var + eps) : std::sqrt(var) + eps);
    // mvn
    mvn_scale_bias(src, mean, var, ele_num, dst, scale, bias);
}
}

bool ov::intel_cpu::node::LayerNorm::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (!std::dynamic_pointer_cast<const ov::intel_cpu::LayerNorm>(op)) {
            errorMessage = "Not supported LayerNorm operation version";
            return false;
        }
    } catch (...) {
        return false;
    }

    return true;
}

ov::intel_cpu::node::LayerNorm::LayerNorm(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr context) :
        Node(op, context, InternalDynShapeInferFactory()) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }

    auto LayerNormPtr = std::dynamic_pointer_cast<const ov::intel_cpu::LayerNorm>(op);
}

void ov::intel_cpu::node::LayerNorm::initSupportedPrimitiveDescriptors() {
    dataPrecision = getOriginalInputPrecisionAtPort(0);

    addSupportedPrimDesc({{LayoutType::ncsp, dataPrecision},
                    {LayoutType::ncsp, dataPrecision},
                    {LayoutType::ncsp, dataPrecision}},
                    {{LayoutType::ncsp, dataPrecision}},
                    impl_desc_type::ref_any);
}

void ov::intel_cpu::node::LayerNorm::createPrimitive() {
    Node::createPrimitive();
}

void ov::intel_cpu::node::LayerNorm::prepareParams() {
    const auto& dims = getParentEdgeAt(0)->getMemoryPtr()->getStaticDims();
    totalElements =
        std::accumulate(dims.begin(), dims.end(), size_t(1), std::multiplies<size_t>());
    return;
}

void ov::intel_cpu::node::LayerNorm::executeDynamicImpl(dnnl::stream strm) {
    auto& dims = getParentEdgeAt(0)->getMemoryPtr()->getStaticDims();
    redefineOutputMemory({dims});

    execute(strm);
}

void ov::intel_cpu::node::LayerNorm::execute(dnnl::stream strm) {
    auto* src = reinterpret_cast<float*>(getParentEdgeAt(0)->getMemoryPtr()->GetPtr());
    auto* scale = reinterpret_cast<float*>(getParentEdgeAt(1)->getMemoryPtr()->GetPtr());
    float* bias = reinterpret_cast<float*>(getParentEdgeAt(2)->getMemoryPtr()->GetPtr());
    auto* dst = reinterpret_cast<float*>(getChildEdgeAt(0)->getMemoryPtr()->GetPtr());
    auto node0Dims = getParentEdgeAt(0)->getMemoryPtr()->getStaticDims();
    auto node1Dims = getParentEdgeAt(1)->getMemoryPtr()->getStaticDims();
    size_t channels = node1Dims.back();
    auto eps = 0.000009999999747378752f;
    bool inside_sqrt = true;
    auto total = totalElements;
    auto count = total / channels;
    // assume batch is 1
    parallel_for(count, [&](int i) {
        mvn_line_scale_bias(src + i * channels, channels, eps, true, dst + i * channels, scale, bias);
    });  
}
