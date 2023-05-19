// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>

#include "add_custom.h"
#include "transformations/cpu_opset/x64/op/add_custom.hpp"
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

inline __m256 gelu_erf_minmax_approx(__m256 & x) {
    auto x2 = _mm256_mul_ps(x, x); // x^2
    
    auto x_positive = _mm256_castsi256_ps(_mm256_and_si256(_mm256_castps_si256(x), _mm256_set1_epi32(0x7FFFFFFF)));    // clear sign mask
    auto x_half = _mm256_mul_ps(x, _mm256_set1_ps(0.5f));

    auto poly = _mm256_castsi256_ps(_mm256_set1_epi32(0x1f1c83fd));
    poly = _mm256_fmadd_ps(poly, x2, _mm256_castsi256_ps(_mm256_set1_epi32(0xa3198977))); // poly * x^2 + xxx
    poly = _mm256_fmadd_ps(poly, x2, _mm256_castsi256_ps(_mm256_set1_epi32(0x268a7927)));
    poly = _mm256_fmadd_ps(poly, x2, _mm256_castsi256_ps(_mm256_set1_epi32(0xa998c963)));
    poly = _mm256_fmadd_ps(poly, x2, _mm256_castsi256_ps(_mm256_set1_epi32(0x2c67ddb2)));
    poly = _mm256_fmadd_ps(poly, x2, _mm256_castsi256_ps(_mm256_set1_epi32(0xaf013b2c)));
    poly = _mm256_fmadd_ps(poly, x2, _mm256_castsi256_ps(_mm256_set1_epi32(0x315d4a4f)));
    poly = _mm256_fmadd_ps(poly, x2, _mm256_castsi256_ps(_mm256_set1_epi32(0xb3969b11)));
    poly = _mm256_fmadd_ps(poly, x2, _mm256_castsi256_ps(_mm256_set1_epi32(0x35a776e9)));
    poly = _mm256_fmadd_ps(poly, x2, _mm256_castsi256_ps(_mm256_set1_epi32(0xb79b0914)));
    poly = _mm256_fmadd_ps(poly, x2, _mm256_castsi256_ps(_mm256_set1_epi32(0x3970b255)));
    poly = _mm256_fmadd_ps(poly, x2, _mm256_castsi256_ps(_mm256_set1_epi32(0xbb1b7399)));
    poly = _mm256_fmadd_ps(poly, x2, _mm256_castsi256_ps(_mm256_set1_epi32(0x3ca3621f)));
    poly = _mm256_fmadd_ps(poly, x2, _mm256_castsi256_ps(_mm256_set1_epi32(0xbe082bc7)));
    poly = _mm256_fmadd_ps(poly, x2, _mm256_castsi256_ps(_mm256_set1_epi32(0x3f4c4228)));

    // 1.0f + erf(x * inv_sqrt2) = 1.0f + x * P(x^2)
    poly = _mm256_fmadd_ps(poly, x, _mm256_set1_ps(1.0f));
    // x*0.5*(1 + x*Polynomial(x^2))
    poly = _mm256_mul_ps(poly, x_half);

    // combine:
    // zone_id
    //  1 -inf; -saturation_lbound           : 0.0f
    //  2 -saturation_lbound; -linear_ubound : x*0.5*(1 + x*Polynomial(x^2))
    //  3 -linear_ubound, linear_ubound         : x*0.5
    //  4 linear_ubound : saturation_lbound     : x*0.5*(1 + x*Polynomial(x^2))
    //  5 saturation_lbound: +inf               : x
    constexpr int neg_saturation_lbound = 0xc0a00000;
    constexpr int linear_ubound = 0x33800000;
    constexpr int saturation_lbound = 0x40a00000;

    auto mask_x_not_zone1 = _mm256_cmp_ps(x, _mm256_castsi256_ps(_mm256_set1_epi32(neg_saturation_lbound)), _CMP_NLT_UQ);
    x = _mm256_blendv_ps(_mm256_setzero_ps(), x, mask_x_not_zone1); // not_zone1 => keep x

    auto mask_x_in_zone5 = _mm256_cmp_ps(x_positive, _mm256_castsi256_ps(_mm256_set1_epi32(saturation_lbound)), _CMP_NLE_UQ);
    poly = _mm256_blendv_ps(poly, x, mask_x_in_zone5);

    auto mask_x_in_zone3 = _mm256_cmp_ps(x_positive, _mm256_castsi256_ps(_mm256_set1_epi32(linear_ubound)), _CMP_LE_OQ);
    poly = _mm256_blendv_ps(poly, x_half, mask_x_in_zone3);
    return poly;
}

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
static void add2(float* a, float *b, float *dst, size_t ele_num) {
    size_t i = 0;
    for (; i < ele_num - 8; i += 8) {
        auto a0_f = _mm256_loadu_ps(a+i);
        auto b0_f = _mm256_loadu_ps(b+i);
        auto d_f = _mm256_add_ps(a0_f, b0_f);
        _mm256_storeu_ps(dst+i, d_f);
    }
    if (i < ele_num) {
        auto msk = get_mask(ele_num - i);
        auto a0_f = _mm256_maskload_ps(a+i, msk);
        auto b0_f = _mm256_maskload_ps(b+i, msk);
        auto d_f = _mm256_add_ps(a0_f, b0_f);
        _mm256_maskstore_ps(dst+i, msk, d_f);
    }
}

static void add3(float* a, float *b, float *c, float *dst, size_t ele_num) {
    size_t i = 0;
    for (; i < ele_num - 8; i += 8) {
        auto a0_f = _mm256_loadu_ps(a+i);
        auto b0_f = _mm256_loadu_ps(b+i);
        auto c0_f = _mm256_loadu_ps(c+i);
        auto d_f = _mm256_add_ps(a0_f, b0_f);
        d_f = _mm256_add_ps(d_f, c0_f);
        _mm256_storeu_ps(dst+i, d_f);
    }
    if (i < ele_num) {
        auto msk = get_mask(ele_num - i);
        auto a0_f = _mm256_maskload_ps(a+i, msk);
        auto b0_f = _mm256_maskload_ps(b+i, msk);
        auto c0_f = _mm256_maskload_ps(c+i, msk);
        auto d_f = _mm256_add_ps(a0_f, b0_f);
        d_f = _mm256_add_ps(d_f, c0_f);
        _mm256_maskstore_ps(dst+i, msk, d_f);
    }
}

static void add2_gelu(float* a, float *b, float *dst, size_t ele_num) {
    size_t i = 0;
    for (; i < ele_num - 8; i += 8) {
        auto a0_f = _mm256_loadu_ps(a+i);
        auto b0_f = _mm256_loadu_ps(b+i);
        auto d_f = _mm256_add_ps(a0_f, b0_f);
        auto act_f = gelu_erf_minmax_approx(d_f);
        _mm256_storeu_ps(dst+i, act_f);
    }
    if (i < ele_num) {
        auto msk = get_mask(ele_num - i);
        auto a0_f = _mm256_maskload_ps(a+i, msk);
        auto b0_f = _mm256_maskload_ps(b+i, msk);
        auto d_f = _mm256_add_ps(a0_f, b0_f);
        auto act_f = gelu_erf_minmax_approx(d_f);
        _mm256_maskstore_ps(dst+i, msk, act_f);
    }
}

}

bool ov::intel_cpu::node::AddCustom::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (!std::dynamic_pointer_cast<const ov::intel_cpu::AddCustom>(op)) {
            errorMessage = "Not supported AddCustom operation version. CPU plug-in supports only 10th version.";
            return false;
        }
    } catch (...) {
        return false;
    }

    return true;
}

ov::intel_cpu::node::AddCustom::AddCustom(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr context) :
        Node(op, context, InternalDynShapeInferFactory()) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }

    auto addCustomPtr = std::dynamic_pointer_cast<const ov::intel_cpu::AddCustom>(op);
    postGelu = addCustomPtr->fuse_gelu;
    witBiases = inputShapes.size() == 3;
}

void ov::intel_cpu::node::AddCustom::initSupportedPrimitiveDescriptors() {
    dataPrecision = getOriginalInputPrecisionAtPort(0);
    if (witBiases) {
        addSupportedPrimDesc({{LayoutType::ncsp, dataPrecision},
                        {LayoutType::ncsp, dataPrecision},
                        {LayoutType::ncsp, dataPrecision}},
                        {{LayoutType::ncsp, dataPrecision}},
                        impl_desc_type::ref_any);
    } else {
        addSupportedPrimDesc({{LayoutType::ncsp, dataPrecision},
                        {LayoutType::ncsp, dataPrecision}},
                        {{LayoutType::ncsp, dataPrecision}},
                        impl_desc_type::ref_any);
    }
}

void ov::intel_cpu::node::AddCustom::createPrimitive() {
    Node::createPrimitive();
}

void ov::intel_cpu::node::AddCustom::prepareParams() {
    const auto& dims = getParentEdgeAt(0)->getMemoryPtr()->getStaticDims();
    auto node0Dims = getParentEdgeAt(0)->getMemoryPtr()->getStaticDims();
    auto node1Dims = getParentEdgeAt(1)->getMemoryPtr()->getStaticDims();
    totalElements =
        std::accumulate(dims.begin(), dims.end(), size_t(1), std::multiplies<size_t>());
    if (node0Dims.size() == node1Dims.size() && node0Dims == node1Dims) {
        sameShape = true;
    } else {
        size_t rank_a = node0Dims.size();
        size_t rank_b = node1Dims.size();
        size_t rank_diff = rank_a > rank_b ? (rank_a - rank_b) : (rank_b - rank_a);
        size_t fisrt_dim = rank_a > rank_b ? node0Dims[0] : node0Dims[1];
        sameShape = (rank_diff == 1 && fisrt_dim == 1);
    }
}

void ov::intel_cpu::node::AddCustom::executeDynamicImpl(dnnl::stream strm) {
    auto& dims = getParentEdgeAt(0)->getMemoryPtr()->getStaticDims();
    redefineOutputMemory({dims});

    execute(strm);
}

void ov::intel_cpu::node::AddCustom::execute(dnnl::stream strm) {
    auto* node0 = reinterpret_cast<float*>(getParentEdgeAt(0)->getMemoryPtr()->GetPtr());
    auto* node1 = reinterpret_cast<float*>(getParentEdgeAt(1)->getMemoryPtr()->GetPtr());
    float* node2 = nullptr;
    if (witBiases)
        node2 = reinterpret_cast<float*>(getParentEdgeAt(2)->getMemoryPtr()->GetPtr());
    auto* dst = reinterpret_cast<float*>(getChildEdgeAt(0)->getMemoryPtr()->GetPtr());
    auto node0Dims = getParentEdgeAt(0)->getMemoryPtr()->getStaticDims();
    auto node1Dims = getParentEdgeAt(1)->getMemoryPtr()->getStaticDims();
    size_t channels = node1Dims.back();

    auto total = totalElements;
    auto count = total / channels;
    // assume batch is 1
    if (postGelu) {
        parallel_for(count, [&](int i) {
            add2_gelu(node0 + i * channels, node1, dst + i * channels, channels);
        });   
    } else if (witBiases) {
        parallel_for(count, [&](int i) {
            add3(node0 + i * channels, node1 + i * channels, node2, dst + i * channels, channels);
        }); 
    } else if (sameShape) {
        parallel_for(count, [&](int i) {
            add2(node0 + i * channels, node1 + i * channels, dst + i * channels, channels);
        });   
    } else {
        parallel_for(count, [&](int i) {
            add2(node0 + i * channels, node1, dst + i * channels, channels);
        });      
    }
}
