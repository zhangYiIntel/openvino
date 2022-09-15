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
#include "nodes/common/cpu_convert.h"
#include "memory_desc/cpu_memory_desc_utils.h"
#include "memory_desc/dnnl_blocked_memory_desc.h"
#include "common/bfloat16.hpp"
#include "common/cpu_memcpy.h"
#include <cpu/x64/cpu_isa_traits.hpp>
#include "fake_quantize.h"

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

void Interaction::getSupportedDescriptors() {
    dataPrecision = getOriginalInputPrecisionAtPort(0);
    // Current impl only support FP32 BF16, BF16 is preferred
    if (dataPrecision != InferenceEngine::Precision::FP32 && dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core_bf16)) {
        dataPrecision = InferenceEngine::Precision::BF16;
    } else {
        dataPrecision = InferenceEngine::Precision::FP32;
    }
    outputDataType = dataPrecision;
    if (!fusedWith.empty()) {
        outputDataType = fusedWith[fusedWith.size() - 1]->getOriginalOutputPrecisionAtPort(0);
    }
}

void Interaction::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;
    std::cout << "What is my ouput " << outputDataType << std::endl;
    // initialize input ports
    std::vector<PortConfigurator> inPortConfigs;
    for (size_t i = 0; i < getParentEdges().size(); ++i) {
        inPortConfigs.emplace_back(
            LayoutType::ncsp,
            dataPrecision,
            getInputShapeAtPort(i),
            false, -1);
    }
    // initialize output port
    std::vector<PortConfigurator> outPortConfigs = {
        PortConfigurator {
            LayoutType::ncsp,
            outputDataType,
            getOutputShapeAtPort(0),
            false,
            -1
        }
    };
    //add descriptor
    addSupportedPrimDesc(inPortConfigs, outPortConfigs, impl_desc_type::ref_any, true);
}

template <typename T>
inline void move_ker(T* out, const T* in, int64_t len) {
    cpu_memcpy(out, in, sizeof(T) * len);
}

template <typename InPrec, typename OutPrec>
inline void cat(const InPrec* in1, const InPrec* in2, OutPrec* out, size_t in1_size, size_t in2_size, float scale = 0.0) {
    move_ker(out, in1, in1_size);
    move_ker(&out[in1_size], in2, in2_size);
}

template <>
inline void cat<float, int8_t>(const float* in1, const float* in2, int8_t* out, size_t in1_size, size_t in2_size, float scale) {
    size_t index = 0;
    for (size_t i = 0; i < in1_size; i++) {
        float dst_val = dnnl::impl::nstl::min(static_cast<float>(5.0896),
            dnnl::impl::nstl::max(static_cast<float>(-5.12978), in1[i]));
        out[index++] = int8_t(roundf(dst_val * scale));
    }

    for (size_t i = 0; i < in2_size; i++) {
         float dst_val = dnnl::impl::nstl::min(static_cast<float>(5.0896),
            dnnl::impl::nstl::max(static_cast<float>(-5.12978), in2[i]));
        out[index++] = int8_t(roundf(dst_val * scale));
    }
}

template <typename T>
static inline void cat(T* out,
                       const std::vector<const T*>& in,
                       const std::vector<uint32_t>& feature_sizes,
                       int64_t bs) {
    size_t offset = 0;
    for (int j = 0; j < feature_sizes.size(); j++) {
        move_ker(&out[offset], &in[j][bs * feature_sizes[j]], feature_sizes[j]);
        offset += feature_sizes[j];
    }
}

template <typename T>
static inline void flat_triangle(const T* in, T* out, size_t size) {
    size_t offset = 0;
    for (int i = 1; i < size; i++) {
        move_ker(&out[offset], &in[i * size], i);
        offset += i;
    }
}

template <typename Prec, typename OutPrec>
void Interaction::run(dnnl::stream strm) {
    using tag = dnnl::memory::format_tag;
    using dt = dnnl::memory::data_type;
    using namespace dnnl;

    auto outFeaturesPtr = reinterpret_cast<OutPrec*>(getChildEdgesAtPort(0)[0]->getMemoryPtr()->GetPtr());
    std::vector<const Prec*> inputPtrs(inputSizes);
    for (uint32_t n = 0; n < inputSizes; n++) {
        auto inPtr = reinterpret_cast<const Prec*>(getParentEdgeAt(n)->getMemoryPtr()->GetPtr());
        inputPtrs[n] = inPtr;
    }
    for (int64_t start = 0; start < batchSize; start++) {
        cat<Prec>(inputPtr->buffer().as<Prec*>(), inputPtrs, featureSizes, start);
        std::unordered_map<int, memory> mem_ags {
            {DNNL_ARG_SRC, inputMemPtr->GetPrimitive()},
            {DNNL_ARG_WEIGHTS, inputMemPtr->GetPrimitive()},
            {DNNL_ARG_DST, outputMemPtr->GetPrimitive()}};
        (*prim).execute(strm, mem_ags);
        flat_triangle<Prec>(outputPtr->buffer().as<Prec*>(),
            flatPtr->buffer().as<Prec*>(), inputSizes);
        //in1 dense feature
        //in2 flatted interaction features
        cat<Prec, OutPrec>(
          &inputPtrs[0][start * featureSize],
          flatPtr->buffer().as<Prec*>(),
          &outFeaturesPtr[start * outputFeaturesLen],
          featureSize,
          interactFeatureSize,
          outputScale);
    }
}



void Interaction::execute(dnnl::stream strm) {
    if (outputDataType == InferenceEngine::Precision::FP32) {
        run<float, float>(strm);
    } else if (outputDataType == InferenceEngine::Precision::BF16) {
        run<int16_t, int16_t>(strm);
    } else if (outputDataType == InferenceEngine::Precision::I8) {
        run<float, int8_t>(strm);
    }
}

bool Interaction::created() const {
    return getType() == Type::Interaction;
}

void Interaction::setPostOps() {
    for (int i = 0; i < fusedWith.size(); i++) {
        auto& node = fusedWith[i];

        if (auto* fakeQuantizeNode = dynamic_cast<FakeQuantize *>(node.get())) {
            auto scale = fakeQuantizeNode->getInputScale();
            outputScale = scale[0];
            std::cout << "Interaction output scale|" << outputScale << "|" << scale.size() << std::endl;
            continue;
        }

        IE_THROW() << "Fusing of " << NameFromType(node->getType()) << " operation to " << NameFromType(this->getType()) << " node is not implemented";
    }
}

void Interaction::prepareParams() {
    using tag = dnnl::memory::format_tag;
    using dt = dnnl::memory::data_type;
    using namespace dnnl;
    const auto& denseFeatureDims = getParentEdgeAt(0)->getMemory().getStaticDims();
    batchSize = denseFeatureDims[0];
    featureSize = denseFeatureDims[1];
    inputSizes = inputShapes.size();
    interactFeatureSize = inputSizes * (inputSizes - 1) / 2;
    outputFeaturesLen = interactFeatureSize + featureSize;
    std::vector<int64_t> lhsShape({static_cast<int64_t>(inputSizes), static_cast<int64_t>(featureSize)});
    std::vector<int64_t> lhsStride({static_cast<int64_t>(featureSize), 1});
    std::vector<int64_t> rhsShape({static_cast<int64_t>(featureSize), static_cast<int64_t>(inputSizes)});
    std::vector<int64_t> rhsStride({1, static_cast<int64_t>(featureSize)});
    std::vector<int64_t> resShape({static_cast<int64_t>(inputSizes), static_cast<int64_t>(inputSizes)});
    std::vector<int64_t> resStride({static_cast<int64_t>(inputSizes), 1});
    auto dataType = DnnlExtensionUtils::IEPrecisionToDataType(dataPrecision);
    auto src_md = memory::desc(lhsShape, dataType, lhsStride);
    auto weights_md = memory::desc(rhsShape, dataType, rhsStride);
    auto dst_md = memory::desc(resShape, dataType, resStride);
    auto matmul_d = matmul::desc(src_md, weights_md, dst_md);
    primitive_attr matmul_attr;
    auto matmul_pd = matmul::primitive_desc(matmul_d, matmul_attr, getEngine());
    prim.reset(new matmul(matmul_pd));
    featureSizes.assign(inputSizes, featureSize);
    std::vector<InferenceEngine::TensorDesc> internalMemDesc = {
        InferenceEngine::TensorDesc(
            dataPrecision,
            {inputSizes, featureSize},
            InferenceEngine::Layout::HW),
        InferenceEngine::TensorDesc(
            dataPrecision,
            {inputShapes.size(), inputShapes.size()},
            InferenceEngine::Layout::HW),
        InferenceEngine::TensorDesc(
            dataPrecision,
            {interactFeatureSize},
            InferenceEngine::Layout::ANY)
    };

    if (dataPrecision == InferenceEngine::Precision::FP32) {
        initializeInternalMemory<float>(internalMemDesc);
    } else {
        initializeInternalMemory<int16_t>(internalMemDesc);
    }
    setPostOps();
    inputMemPtr = std::make_shared<Memory>(getEngine());
    outputMemPtr = std::make_shared<Memory>(getEngine());
    auto inDesc = MemoryDescUtils::convertToDnnlBlockedMemoryDesc(inputPtr->getTensorDesc());
    auto outDesc = MemoryDescUtils::convertToDnnlBlockedMemoryDesc(outputPtr->getTensorDesc());
    inputMemPtr->Create(inDesc, inputPtr->buffer());
    outputMemPtr->Create(outDesc, outputPtr->buffer());
}

void Interaction::executeDynamicImpl(dnnl::stream strm) {
    execute(strm);
}

bool Interaction::isExecutable() const {
    return true;
}

bool Interaction::isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op,
        std::string& errorMessage) noexcept {
    try {
        const auto interaction = std::dynamic_pointer_cast<const InteractionNode>(op);
        if (!interaction) {
            errorMessage = "Only Interaction operation is supported";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

bool Interaction::canFuse(const NodePtr& node) const {
    //experiment on FQ only
    if (node->getType() == Type::FakeQuantize) {
        bool ret = node->getAlgorithm() != Algorithm::FQBinarization;
        for (size_t i = 1; i < node->getParentEdges().size(); i++) {
            ret &= node->getParentEdgesAtPort(i)[0]->getParent()->getChildEdges().size() == 1;
        }
        return ret;
    } else {
        return false;
    }
}


}   // namespace node
}   // namespace intel_cpu
}   // namespace ov