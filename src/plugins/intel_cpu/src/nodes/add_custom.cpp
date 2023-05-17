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
#include <x86intrin.h>
#include <immintrin.h>
using namespace InferenceEngine;
using namespace ov::intel_cpu;
using namespace ov::intel_cpu::node;
using namespace dnnl::impl::cpu::x64;
using namespace Xbyak;

#define THROW_ERROR IE_THROW() << getTypeStr() << " node with name '" << getName() << "' "
namespace {
static void add2(float* a, float *b, float *dst, size_t ele_num) {
    size_t i = 0;
    for (; i < ele_num - 8; i += 8) {
        auto a0_f = _mm256_loadu_ps(a+i);
        // auto b0_f = _mm256_loadu_ps(b);
        auto b0_f = _mm256_set1_ps(0.5);
        auto d_f = _mm256_add_ps(a0_f, b0_f);
        _mm256_storeu_ps(dst+i, d_f);
    }
    // std::cout << "process tail" << i << std::endl;
    for (; i < ele_num; i++) {
        *(dst+i) = a[i] + 0.5;
        // std::cout << a[i] << "|" << dst[i] << std::endl;
    }
    // if (i != ele_num) {
    //     auto msk = get_mask(ele_num - i);
    //     auto a0_f = _mm256_maskload_ps(src, msk);
    //     auto b0_f = _mm256_maskload_ps(b, msk);
    //     a0_f = _mm256_fmadd_ps(a0_f, s, b);
    //     _mm256_maskstore_ps(dst, msk, a0_f);
    // }
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
}

void ov::intel_cpu::node::AddCustom::initSupportedPrimitiveDescriptors() {
    dataPrecision = getOriginalInputPrecisionAtPort(0);

    addSupportedPrimDesc({{LayoutType::ncsp, dataPrecision},
                          {LayoutType::ncsp, dataPrecision}},
                         {{LayoutType::ncsp, dataPrecision}},
                          impl_desc_type::ref_any);
}

void ov::intel_cpu::node::AddCustom::createPrimitive() {
    Node::createPrimitive();
}

void ov::intel_cpu::node::AddCustom::prepareParams() {
    const auto& dims = getParentEdgeAt(0)->getMemoryPtr()->getStaticDims();
    totalElements =
        std::accumulate(dims.begin(), dims.end(), size_t(1), std::multiplies<size_t>());
}

void ov::intel_cpu::node::AddCustom::executeDynamicImpl(dnnl::stream strm) {
    auto& dims = getParentEdgeAt(0)->getMemoryPtr()->getStaticDims();
    redefineOutputMemory({dims});

    execute(strm);
}

void ov::intel_cpu::node::AddCustom::execute(dnnl::stream strm) {
    auto* node0 = reinterpret_cast<float*>(getParentEdgeAt(0)->getMemoryPtr()->GetPtr());
    auto* node1 = reinterpret_cast<float*>(getParentEdgeAt(1)->getMemoryPtr()->GetPtr());
    auto* dst = reinterpret_cast<float*>(getChildEdgeAt(0)->getMemoryPtr()->GetPtr());
    auto node0Dims = getParentEdgeAt(0)->getMemoryPtr()->getStaticDims();
    auto node1Dims = getParentEdgeAt(1)->getMemoryPtr()->getStaticDims();
    size_t channels = node1Dims.back();
    
    auto total = totalElements;
    auto count = total / channels;
    parallel_for(count, [&](int i) {
        add2(node0 + i * channels, node1, dst + i * channels, channels);
    });    
}
