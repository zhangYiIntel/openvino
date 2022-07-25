// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <chrono>
#include <cmath>
#include <string>
#include <vector>

#include "ie_parallel.hpp"
#include "ngraph_transformations/op/interaction.hpp"
#include "interaction.h"
#include "utils/general_utils.h"
#include <onednn/dnnl.h>
#include <dnnl_extension_utils.h>
#include <immintrin.h>
#include "nodes/common/cpu_convert.h"
#include "memory_desc/cpu_memory_desc_utils.h"
#include "memory_desc/dnnl_blocked_memory_desc.h"

namespace ov {
namespace intel_cpu {
namespace node {

Interaction::Interaction(const std::shared_ptr<ngraph::Node>& op, const dnnl::engine& eng, WeightsSharing::Ptr &cache)
        : Node(op, eng, cache) {
    std::string errorMessage;
    if (isSupportedOperation(op, errorMessage)) {
        errorPrefix = "Interaction node with name '" + getName() + "'";
    } else {
        IE_THROW(NotImplemented) << errorMessage;
    }
}

void Interaction::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;
    // initialize input ports
    std::vector<PortConfigurator> inPortConfigs;
    for (size_t i = 0; i < getParentEdges().size(); ++i) {
        inPortConfigs.emplace_back(
            LayoutType::ncsp,
            InferenceEngine::Precision::FP32,
            getInputShapeAtPort(i),
            false, -1);
    }
    // initialize output port
    std::vector<PortConfigurator> outPortConfigs = {
        PortConfigurator {
            LayoutType::ncsp,
            InferenceEngine::Precision::FP32,
            getOutputShapeAtPort(0),
            false,
            -1
        }
    };
    //add descriptor
    addSupportedPrimDesc(inPortConfigs, outPortConfigs, impl_desc_type::ref_any, true);
}

namespace ref {
    template <typename dst_type, typename src_type>
    inline void mov_ker(dst_type* inout, src_type* in, int len) {
    for (int i = 0; i < len; i++) {
        *(inout + i) = *(in + i);
    }
    }
}

static inline void move_ker(float* out, const float* in, int64_t len) {
  int64_t i = 0;
#if 1
#pragma unroll(4)
  for (i = 0; i < len - 15; i += 16) {
    auto in0 = _mm512_loadu_ps(in + i);
    _mm512_storeu_ps(out + i, in0);
  }

  if (i < len) {
    auto mask = ((1 << (len - i)) - 1);
    auto in0 = _mm512_maskz_loadu_ps(mask, in + i);
    _mm512_mask_storeu_ps(out + i, mask, in0);
  }
#else
  ref::mov_ker(out, in, len);
#endif
}

template <typename T>
static inline void cat(
    const T* in1,
    const T* in2,
    T* out,
    size_t in1_size,
    size_t in2_size) {
  move_ker(out, in1, in1_size);
  move_ker(&out[in1_size], in2, in2_size);
}

template <typename T>
static inline void cat(
    T* out,
    const std::vector<T*>& in,
    const std::vector<uint32_t>& feature_sizes,
    int64_t bs) {
  size_t offset = 0;
  for (int j = 0; j < feature_sizes.size(); j++) {
    move_ker(&out[offset], &in[j][bs * feature_sizes[j]], feature_sizes[j]);
    offset += feature_sizes[j];
  }
}

void Interaction::execute(dnnl::stream strm) {
    using tag = dnnl::memory::format_tag;
    using dt = dnnl::memory::data_type;
    using namespace dnnl;

    auto outFeaturesPtr = reinterpret_cast<float*>(getChildEdgesAtPort(0)[0]->getMemoryPtr()->GetPtr());
    for (int64_t start = 0; start < batchSize; start++) {
        // float catBuf[inputSizes * featureSize] __attribute__((aligned(64)));
        // float mmBuf[inputSizes * inputSizes] __attribute__((aligned(64)));
        std::vector<const float*> inputPtrs(inputSizes);
        for (uint32_t n = 0; n < inputSizes; n++) {
            auto inputPtr = reinterpret_cast<const float*>(getParentEdgeAt(n)->getMemoryPtr()->GetPtr());
            inputPtrs[n] = &inputPtr[start * featureSize];
        }
        std::unordered_map<int, memory> mem_ags {
            {DNNL_ARG_SRC, inputMemPtr->GetPrimitive()},
            {DNNL_ARG_WEIGHTS, inputMemPtr->GetPrimitive()},
            {DNNL_ARG_DST, outputMemPtr->GetPrimitive()}};
        (*prim).execute(strm, mem_ags);
        move_ker(&outFeaturesPtr[start * outputFeaturesLen], inputPtrs[0], featureSize);
    }
    return;
}

bool Interaction::created() const {
    return getType() == Type::Interaction;
}

void Interaction::prepareParams() {
    using tag = dnnl::memory::format_tag;
    using dt = dnnl::memory::data_type;
    using namespace dnnl;
    const auto& denseFeatureDims = getParentEdgeAt(0)->getMemory().getStaticDims();
    batchSize = denseFeatureDims[0];
    featureSize = denseFeatureDims[1];
    inputSizes = inputShapes.size();
    outputFeaturesLen = inputSizes * (inputSizes - 1) / 2 + featureSize;
    std::vector<int64_t> lhsShape({inputSizes, featureSize});
    std::vector<int64_t> lhsStride({featureSize, 1});
    std::vector<int64_t> rhsShape({featureSize, inputSizes});
    std::vector<int64_t> rhsStride({1, featureSize});
    std::vector<int64_t> resShape({inputSizes, inputSizes});
    std::vector<int64_t> resStride({inputSizes, 1});
    auto src_md = memory::desc(lhsShape, dt::f32, lhsStride);
    auto weights_md = memory::desc(rhsShape, dt::f32, rhsStride);
    auto dst_md = memory::desc(resShape, dt::f32, resStride);
    auto matmul_d = matmul::desc(src_md, weights_md, dst_md);
    primitive_attr matmul_attr;
    auto matmul_pd = matmul::primitive_desc(matmul_d, matmul_attr, getEngine());
    std::cout << "!!!!initialize prim" << std::endl;
    prim.reset(new matmul(matmul_pd));
    InferenceEngine::TensorDesc inputDesc(
        InferenceEngine::Precision::FP32,
        denseFeatureDims,
        InferenceEngine::Layout::HW);
    InferenceEngine::TensorDesc outputDesc(
        InferenceEngine::Precision::FP32,
        {inputShapes.size(), inputShapes.size()},
        InferenceEngine::Layout::HW);
    inputPtr = InferenceEngine::make_shared_blob<float>(inputDesc);
    inputPtr->allocate();
    outputPtr = InferenceEngine::make_shared_blob<float>(outputDesc);
    outputPtr->allocate();
    inputMemPtr = std::make_shared<Memory>(getEngine());
    outputMemPtr = std::make_shared<Memory>(getEngine());
    auto inDesc = MemoryDescUtils::convertToDnnlBlockedMemoryDesc(inputPtr->getTensorDesc());
    auto outDesc = MemoryDescUtils::convertToDnnlBlockedMemoryDesc(outputPtr->getTensorDesc());
    inputMemPtr->Create(inDesc, inputPtr->buffer());
    outputMemPtr->Create(outDesc, outputPtr->buffer());
    return;
}

void Interaction::executeDynamicImpl(dnnl::stream strm) {
    execute(strm);
}

bool Interaction::isExecutable() const {
    return true;
}

bool Interaction::isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op,
        std::string& errorMessage) noexcept {
    //TODO
    return true;
}


}   // namespace node
}   // namespace intel_cpu
}   // namespace ov