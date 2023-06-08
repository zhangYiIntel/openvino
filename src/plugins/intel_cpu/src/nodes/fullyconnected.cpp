// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fullyconnected.h"

#include "eltwise.h"
#include "input.h"
#include "fake_quantize.h"
#include "input.h"
#include "reorder.h"
#include "transformations/cpu_opset/common/op/fully_connected.hpp"
#include "ngraph/opsets/opset1.hpp"
#include "dnnl_extension_utils.h"
#include "onednn/dnnl.h"
#include "utils/general_utils.h"
#include "memory_desc/cpu_memory_desc_utils.h"
#include "memory_desc/dnnl_blocked_memory_desc.h"
#include "utils/cpu_utils.hpp"

#include "onednn/dnnl.h"
#include "oneapi/dnnl/dnnl.hpp"
#include "cpu/x64/cpu_isa_traits.hpp"
#include "common/primitive_hashing_utils.hpp"
#include "common/primitive_desc.hpp"
#include "common/primitive_desc_iface.hpp"
#include "mlas.h"

#include <string>
#include <vector>

#ifdef OV_CPU_WITH_MLAS
#include "gemm/ov_cpu_gemm.h"
#endif

using namespace dnnl;
using namespace InferenceEngine;

namespace ov {
namespace intel_cpu {
namespace node {
namespace {

struct FCKey {
    DnnlMemoryDescCPtr inp0;
    DnnlMemoryDescCPtr inp1;
    DnnlMemoryDescCPtr bias;
    DnnlMemoryDescCPtr out;
    dnnl::primitive_attr attr;
    impl_desc_type implType;
    bool useConv1x1;

    size_t hash() const;
    bool operator==(const FCKey& rhs) const;
};

size_t FCKey::hash() const {
    using namespace dnnl::impl;
    using namespace dnnl::impl::primitive_hashing;

    size_t seed = 0;

    for (const auto& ptr : {inp0, inp1, bias, out}) {
        if (ptr) {
            seed = hash_combine(seed, get_md_hash(*ptr->getDnnlDesc().get()));
        }
    }

    seed = hash_combine(seed, get_attr_hash(*attr.get()));
    seed = hash_combine(seed, implType);
    seed = hash_combine(seed, useConv1x1);
    return seed;
}

bool FCKey::operator==(const FCKey &rhs) const {
    bool retVal = true;
    if (inp0 != rhs.inp0) {
        retVal = retVal && inp0 && rhs.inp0 && inp0->getDnnlDesc() == rhs.inp0->getDnnlDesc();
    }
    if (inp1 != rhs.inp1) {
        retVal = retVal && inp1 && rhs.inp1 && inp1->getDnnlDesc() == rhs.inp1->getDnnlDesc();
    }
    if (bias != rhs.bias) {
        retVal = retVal && bias && rhs.bias && bias->getDnnlDesc() == rhs.bias->getDnnlDesc();
    }
    if (out != rhs.out) {
        retVal = retVal && out && rhs.out && out->getDnnlDesc() == rhs.out->getDnnlDesc();
    }
    retVal = retVal && *attr.get() == *rhs.attr.get() &&
             implType == rhs.implType && useConv1x1 == rhs.useConv1x1;
    return retVal;
}

class FCShapeInfer : public ShapeInferEmptyPads {
public:
    FCShapeInfer(size_t outPut_rank) : out_rank(outPut_rank) {}
    Result infer(
        const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes,
        const std::unordered_map<size_t, MemoryPtr>& data_dependency) override {
        const VectorDims& activationShape = input_shapes[0].get();
        const VectorDims& weightShape = input_shapes[1].get();
        size_t activationRank = activationShape.size();
        size_t channelRank = weightShape.size() - 1;

        // activation   weight    output_shape
        // NCHW         CoCHW     NCo
        // TNC          CoC       TNCo
        // NC           CoC       NCo
        VectorDims outputShape(out_rank, 1);
        // set Co
        outputShape.back() = weightShape[0];
        // set batch dims
        size_t batchRank = activationRank - channelRank;
        size_t startIdx = out_rank - batchRank - 1;
        for (size_t i = 0; i < batchRank; i++) {
            outputShape[i + startIdx] = activationShape[i];
        }

        return {{std::move(outputShape)}, ShapeInferStatus::success};
    }

    port_mask_t get_port_mask() const override {
        return EMPTY_PORT_MASK;
    }

private:
    size_t out_rank = 0;
};

class FCShapeInferFactory : public ShapeInferFactory {
public:
    FCShapeInferFactory(std::shared_ptr<ov::Node> op) : m_op(op) {}
    ShapeInferPtr makeShapeInfer() const override {
        return std::make_shared<FCShapeInfer>(m_op->get_output_partial_shape(0).rank().get_length());
    }

private:
    std::shared_ptr<const ngraph::Node> m_op;
};

} // namespace

bool FullyConnected::isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto fc = std::dynamic_pointer_cast<const FullyConnectedNode>(op);
        if (!fc) {
            errorMessage = "Only legacy FullyConnected operation is supported";
            return false;
        }
        if (fc->get_input_size() == 3 && std::dynamic_pointer_cast<const ngraph::opset1::Constant>(fc->get_input_node_shared_ptr(BIAS_ID)) == nullptr) {
            errorMessage = "Only Constant operation on 'bias' input is supported";
            return false;
        }
        const auto inRank = fc->get_input_partial_shape(DATA_ID).size();
        const auto weightRank = fc->get_input_partial_shape(WEIGHTS_ID).size();
        if (!one_of(inRank, 2u, 3u, 4u)) {
            errorMessage = "Doesn't support 'data' input with rank: " + std::to_string(inRank);
            return false;
        }
        if ((one_of(inRank, 2u, 3u) && weightRank != 2) || (inRank == 4 && weightRank != 4)) {
            errorMessage = "Doesn't support 'data' input with rank: " + std::to_string(inRank) +
                           " and 'weight' input with rank: " + std::to_string(weightRank);
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

FullyConnected::FullyConnected(const std::shared_ptr<ngraph::Node>& op, const GraphContext::CPtr context)
        : Node(op, context, FCShapeInferFactory(op)), withBiases(false) {
    std::string errorMessage;
    if (isSupportedOperation(op, errorMessage)) {
        errorPrefix = "FullyConnected node with name '" + getName() + "'";

        withBiases = inputShapes.size() == 3;

        if (context->getConfig().fcSparseWeiDecompressionRate < 1.0f)
            minSparseRate = context->getConfig().fcSparseWeiDecompressionRate;
    } else {
        IE_THROW(NotImplemented) << errorMessage;
    }
}

std::vector<memory::format_tag> FullyConnected::getAvailableFormatsForDims(const Shape &dims) const {
    if (dims.getRank() == 0)
        return {memory::format_tag::x};
    else if (dims.getRank() == 1)
        return {memory::format_tag::x};
    else if (dims.getRank() == 2)
        return {memory::format_tag::nc};
    else if (dims.getRank() == 3)
        return {memory::format_tag::tnc};
    else if (dims.getRank() == 4)
        return {memory::format_tag::nChw8c, memory::format_tag::nChw16c, memory::format_tag::nhwc, memory::format_tag::nchw};
    else if (dims.getRank() == 5)
        return {memory::format_tag::nCdhw8c, memory::format_tag::nCdhw16c, memory::format_tag::ndhwc, memory::format_tag::ncdhw};
    return {memory::format_tag::any};
}

VectorDims FullyConnected::makeDummyInputDims() const {
    const auto& inShape = getInputShapeAtPort(DATA_ID);
    const auto& weightDims = getInputShapeAtPort(WEIGHTS_ID).getStaticDims();

    auto inMinDims = inShape.getMinDims();
    auto inMaxDims = inShape.getMaxDims();

    if (inMinDims.size() == 3) {
        inMinDims.back() = weightDims.back();
        inMaxDims.back() = weightDims.back();
    } else {
        for (size_t i = 1; i < inMinDims.size(); i++) {
            inMinDims[i] = weightDims[i];
            inMaxDims[i] = weightDims[i];
        }
    }
    return MemoryDescUtils::makeDummyShape(Shape(inMinDims, inMaxDims)).getStaticDims();
}

VectorDims FullyConnected::makeDummyOutputDims(const VectorDims& inDims) const {
    std::vector<Shape> inShapes = {Shape(inDims), getInputShapeAtPort(WEIGHTS_ID)};
    if (inputShapes.size() > 2) {
        inShapes.emplace_back(getInputShapeAtPort(BIAS_ID));
    }
    return shapeInferGeneric(inShapes).front();
}

void FullyConnected::getSupportedDescriptors() {
    if (getParentEdges().size() != 2 && getParentEdges().size() != 3)
        IE_THROW() << errorPrefix << " has incorrect number of input edges";
    if (getChildEdges().empty())
        IE_THROW()<< errorPrefix << " has incorrect number of output edges";

    useSparseWeights = useSparseWeightsDecompression();

    auto inputDataType = DnnlExtensionUtils::IEPrecisionToDataType(getOriginalInputPrecisionAtPort(DATA_ID));
    outputDataType = DnnlExtensionUtils::IEPrecisionToDataType(getOriginalOutputPrecisionAtPort(DATA_ID));

    if (!fusedWith.empty()) {
        outputDataType = DnnlExtensionUtils::IEPrecisionToDataType(fusedWith[fusedWith.size() - 1]->getOriginalOutputPrecisionAtPort(0));
    }
    auto weightsDataType = DnnlExtensionUtils::IEPrecisionToDataType(getOriginalInputPrecisionAtPort(WEIGHTS_ID));

    isINT8 = one_of(inputDataType, memory::data_type::u8, memory::data_type::s8) && weightsDataType == memory::data_type::s8;
    // revert back outputDataType on special cases
    if (inputDataType == memory::data_type::f32) {
        // oneDNN only support f32 output when input is f32, even if FQ is fused
        outputDataType = memory::data_type::f32;
    } else if (inputDataType == memory::data_type::bf16) {
        // bf16 input only supports bf16/f32 output, even if FQ is fused as post-ops
        if (one_of(outputDataType , memory::data_type::u8, memory::data_type::s8)) {
            outputDataType = memory::data_type::bf16;
        }
    } else if (one_of(inputDataType, memory::data_type::u8, memory::data_type::s8)) {
        if (weightsDataType != memory::data_type::s8) {
            // weight has to be s8 for INT8 mode, otherwise fallback to
            // f32 mode
            inputDataType = outputDataType = memory::data_type::f32;
        }
    } else {
        // s32/u32/... unsupported input data types, fallback to f32
        inputDataType = outputDataType = memory::data_type::f32;
    }
#ifdef OV_CPU_WITH_MLAS
    useMlas = !useSparseWeights && (inputDataType != memory::data_type::bf16) && getenv("ENABLE_MLAS");
    if (isINT8) {
        // strict the mlas usage in pure int8 models
        bool isStrictINT8 = (one_of(inputDataType, memory::data_type::u8, memory::data_type::s8)
            && weightsDataType == memory::data_type::s8);
        if (isStrictINT8) {
            aSigned = inputDataType == memory::data_type::s8 ? true : false;
            bSigned = true;
        }
}
#endif
    if (useMlas) return;
    inDims = isDynamicNode() ? makeDummyInputDims() : getInputShapeAtPort(DATA_ID).getStaticDims();
    outDims = isDynamicNode() ? makeDummyOutputDims(inDims) : getOutputShapeAtPort(0).getStaticDims();

    for (auto format : getAvailableFormatsForDims(getInputShapeAtPort(0))) {
        auto in_candidate = dnnl::memory::desc(DnnlExtensionUtils::convertToDnnlDims(inDims), inputDataType, format);
        auto out_candidate = dnnl::memory::desc(DnnlExtensionUtils::convertToDnnlDims(outDims), outputDataType, dnnl::memory::format_tag::any);

        createDescriptorInternal(in_candidate, out_candidate);
    }
}

void FullyConnected::createPrimitive() {
#ifdef OV_CPU_WITH_MLAS
    if (useMlas) {
        Node::createPrimitive();
        return;
    }
#endif
    setPostOps(attr, outDims);
    attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    Node::createPrimitive();
    appendPostOpArgs(attr, primArgs, postOpsArgs);
}

void FullyConnected::prepareParams() {
    auto srcMemPtr = getParentEdgesAtPort(0)[0]->getMemoryPtr();
    auto dstMemPtr = getChildEdgesAtPort(0)[0]->getMemoryPtr();
    if (!dstMemPtr || !dstMemPtr->isAllocated())
        IE_THROW() << "Destination memory hasn't been allocated.";
    if (!srcMemPtr || !srcMemPtr->isAllocated())
        IE_THROW() << "Input memory hasn't been allocated.";
    MemoryPtr biasMemPtr = nullptr;
    if (withBiases) {
        biasMemPtr = getParentEdgesAtPort(2)[0]->getMemoryPtr();
        if (!biasMemPtr || !biasMemPtr->isAllocated())
            IE_THROW() << "Input memory hasn't been allocated.";
    }

    NodeDesc *selected_pd = getSelectedPrimitiveDescriptor();
    if (selected_pd == nullptr)
        IE_THROW() << "Preferable primitive descriptor is not set for node " << getName() << ".";
#ifdef OV_CPU_WITH_MLAS
    if (useMlas) {
        auto& dims0 = getParentEdgeAt(0)->getMemoryPtr()->getStaticDims();
        auto& dims1 = getParentEdgeAt(1)->getMemoryPtr()->getStaticDims();
        //Weight is transpoed by MatMulConstTransposesExtraction
        K = dims0[dims0.size() - 1];
        N = dims1[dims1.size() - 2];
        M = std::accumulate(dims0.begin(), dims0.end() - 1, 1, std::multiplies<int64_t>());
        if (packedBPtr == nullptr) {
            if (isINT8) {
                const size_t packedBsize = MlasGemmPackBSize(N, K, aSigned, bSigned);
                uint8_t* weightPtr = reinterpret_cast<uint8_t*>(getParentEdgeAt(1)->getMemoryPtr()->GetPtr());
                packedBPtr.reset(new ngraph::runtime::AlignedBuffer(packedBsize));
                MlasGemmPackB(N, K, weightPtr, N, aSigned, bSigned, packedBPtr->get_ptr<uint8_t>());
            } else {
                auto packedBsize = ov_sgemm_pack_get_size("B", M, N, K);
                packedBPtr.reset(new ngraph::runtime::AlignedBuffer(packedBsize * sizeof(float)));
                float* weightPtr = reinterpret_cast<float*>(getParentEdgeAt(1)->getMemoryPtr()->GetPtr());
                size_t lda = M;
                size_t ldb = K;
                ov_sgemm_pack("B", "N", "T", M, N, K, lda, ldb, weightPtr, packedBPtr->get_ptr<float>());
            }
        }
        return;
    }
#endif
    DnnlMemoryDescPtr weightDesc = MemoryDescUtils::convertToDnnlMemoryDesc(weightDescIP);
    DnnlMemoryDescCPtr biasDesc = nullptr;
    if (biasMemPtr) {
        biasDesc = biasMemPtr->GetDescWithType<DnnlMemoryDesc>();
    }

    DnnlMemoryDescCPtr inDesc = srcMemPtr->GetDescWithType<DnnlMemoryDesc>();
    DnnlMemoryDescCPtr outDesc = dstMemPtr->GetDescWithType<DnnlMemoryDesc>();

    useConv1x1 = canBeExecutedInConv1x1();
    FCKey key = {inDesc,
                 weightDesc,
                 biasDesc,
                 outDesc,
                 attr,
                 implementationTypeIP,
                 useConv1x1};

    auto& engine = getEngine();

    auto builder = [&engine](const FCKey& key) -> executorPtr {
        // use conv1x1 primitive for computation
        if (key.useConv1x1) {
            auto prim_desc = createDescriptorInternalForConv(key.inp0, key.inp1, key.bias, key.out, key.attr, engine);
            const bool found = DnnlExtensionUtils::find_implementation(prim_desc, brgconv_avx512_1x1);

            if (found)
                return std::make_shared<DnnlExecutor>(prim_desc);
        }

        // fallback to normal convolution primitive
        auto inDesc = key.inp0->getDnnlDesc();
        const auto& inDims = inDesc.get_dims(); // @TODO query + copy might be slow
        if (inDims.size() == 3) {
            auto normalizedInDims = {inDims[0] * inDims[1], inDims[2]};
            inDesc = inDesc.reshape(normalizedInDims);
        }

        auto outDesc = key.out->getDnnlDesc();
        const auto& outDims = outDesc.get_dims(); // @TODO query + copy might be slow

        if (outDims.size() == 3) {
            auto normalizedOutDims = { outDims[0] * outDims[1], outDims[2] };
            outDesc = outDesc.reshape(normalizedOutDims);
        }

        dnnl::inner_product_forward::primitive_desc prim_desc;
        if (key.bias) {
            prim_desc = dnnl::inner_product_forward::primitive_desc(
                engine,
                dnnl::prop_kind::forward_inference,
                inDesc,
                key.inp1->getDnnlDesc(),
                key.bias->getDnnlDesc(),
                outDesc,
                key.attr);
        } else {
            prim_desc = dnnl::inner_product_forward::primitive_desc(
                engine,
                dnnl::prop_kind::forward_inference,
                inDesc,
                key.inp1->getDnnlDesc(),
                outDesc,
                key.attr);
        }

        const bool found = DnnlExtensionUtils::find_implementation(prim_desc, key.implType);

        if (!found)
            return nullptr;

        return std::make_shared<DnnlExecutor>(prim_desc);;
    };

    auto cache = context->getParamsCache();
    auto result = cache->getOrCreate(key, builder);

    if (!result.first) {
        IE_THROW() << "Primitive descriptor was not found for node " << getName() << ".";
    }

    auto prevExecPtr = execPtr;
    execPtr = result.first;

    if (execPtr) {
        if (execPtr->getSrcDesc()->isCompatible(*inDesc)) {
            primArgs[DNNL_ARG_SRC] = srcMemPtr->GetPrimitive();
        } else {
            primArgs[DNNL_ARG_SRC] = dnnl::memory(execPtr->getDnnlSrcDesc(), engine, srcMemPtr->GetData());
        }

        if (execPtr->getDstDesc()->isCompatible(*outDesc)) {
            primArgs[DNNL_ARG_DST] = dstMemPtr->GetPrimitive();
        } else {
            primArgs[DNNL_ARG_DST] = dnnl::memory(execPtr->getDnnlDstDesc(), engine, dstMemPtr->GetData());
        }

        if (!prevExecPtr || !execPtr->getWeightDesc()->isCompatible(*(prevExecPtr->getWeightDesc()))) {
            primArgs[DNNL_ARG_WEIGHTS] = prepareWeightMemory(execPtr->getWeightDesc())->GetPrimitive();
        }
        // changed shapes may also cause the kernel type changed
        selected_pd->setImplementationType(execPtr->getImplementationType());
        // WA: We update implType to know whether weights decompression was used inside the kernel
        if (selected_pd->getImplementationType() == ov::intel_cpu::brgemm_avx512_amx && useSparseWeights) {
            selected_pd->setImplementationType(ov::intel_cpu::brgemm_sparse_avx512_amx);
        }
        // maybe expected 1x1 conv is not created, update the flag depends on the real type
        useConv1x1 = execPtr->getImplementationType() == brgconv_avx512_1x1;

        if (withBiases) {
            primArgs[DNNL_ARG_BIAS] = biasMemPtr->GetPrimitive();
        }

        auto schratchpadMem = getScratchPadMem(execPtr->getScratchPadDesc());
        primArgs[DNNL_ARG_SCRATCHPAD] = schratchpadMem->GetPrimitive();
#ifdef CPU_DEBUG_CAPS
        if (result.second == CacheEntryBase::LookUpStatus::Miss) {
            auto pd = execPtr->getPrimitiveDesc();
            DEBUG_LOG("verbose##", getName(), "##", DnnlExtensionUtils::query_pd_info(pd), "\n");
        }
#endif
    } else {
        IE_THROW() << "Executor is not created for node " << getName() << ".";
    }
}

void FullyConnected::execute(dnnl::stream strm) {
#ifdef OV_CPU_WITH_MLAS
    if (useMlas) {
        auto& dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
        auto& src0MemPtr = getParentEdgeAt(0)->getMemoryPtr();
        if (isINT8) {
            auto& dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
            auto& src0MemPtr = getParentEdgeAt(0)->getMemoryPtr();

            MLAS_GEMM_QUANT_SHAPE_PARAMS gemm_shape;
            gemm_shape.AIsSigned = aSigned;
            gemm_shape.BIsSigned = bSigned;
            gemm_shape.K = static_cast<size_t>(K);
            gemm_shape.M = static_cast<size_t>(M);
            gemm_shape.N = static_cast<size_t>(N);

            MLAS_GEMM_QUANT_DATA_PARAMS gemm_param;
            gemm_param.A = reinterpret_cast<uint8_t*>(src0MemPtr->GetData());
            gemm_param.lda = gemm_shape.K;
            gemm_param.ZeroPointA = 0;

            gemm_param.B = packedBPtr->get_ptr<uint8_t>();
            gemm_param.ldb = gemm_shape.N;
            gemm_param.ZeroPointB = &bZp;
            gemm_param.C = reinterpret_cast<int32_t*>(dstMemPtr->GetData());
            gemm_param.ldc = gemm_shape.N;
            gemm_param.PerColumnZeroPoints = false;
            const auto& deq = getDQScales();
            auto scale_bias_proc_ptr = std::make_shared<MLAS_QGEMM_SCALE_BIAS_OUTPUT_PROCESSOR>(
            reinterpret_cast<float*>(dstMemPtr->GetData()),
            static_cast<size_t>(N),
            deq.data(),
            nullptr,
            MLAS_QGEMM_OUTPUT_MODE::ZeroMode,
            deq.size() > 1 ? MLAS_QUANTIZATION_GRANULARITY::PerColumn : MLAS_QUANTIZATION_GRANULARITY::PerMatrix);
            if (getenv("ENABLE_DEQ"))
                gemm_param.OutputProcessor = scale_bias_proc_ptr.get();
            MlasGemmBatch(gemm_shape, &gemm_param, 1, nullptr);
        } else {
            int64_t lda = K;
            int64_t ldb = K;
            int64_t ldc = N;
            ov_sgemm_pack_compute("N",
                                  "N",
                                  M,
                                  N,
                                  K,
                                  1.0f,
                                  reinterpret_cast<float*>(src0MemPtr->GetData()),
                                  lda,
                                  packedBPtr->get_ptr<float>(),
                                  ldb,
                                  0.0f,
                                  reinterpret_cast<float*>(dstMemPtr->GetData()),
                                  ldc);
        }
        return;
    }
#endif
    if (!execPtr) {
        IE_THROW() << "Can't execute FullyConnected node with name: " << getName() << ", because executor is not compiled";
    }

    // in cases parameter -> FullyConnected or dynamic shapes
    // we keep old pointer to data in primArgs on second iteration with same input shapes
    auto updateMemoryPtr = [this](int argType) {
        auto param = primArgs.find(argType);
        if (param != primArgs.end()) {
            if (argType == DNNL_ARG_SRC && (getInputShapeAtPort(DATA_ID).getRank() == 3 || useConv1x1)) {
                primArgs.at(argType).set_data_handle(getParentEdgesAtPort(0)[0]->getMemoryPtr()->GetData());
            }
            if (argType == DNNL_ARG_DST && (getOutputShapeAtPort(0).getRank() == 3 || useConv1x1)) {
                primArgs.at(argType).set_data_handle(getChildEdgesAtPort(0)[0]->getMemoryPtr()->GetData());
            }
        }
    };

    updateMemoryPtr(DNNL_ARG_SRC);
    updateMemoryPtr(DNNL_ARG_DST);

    execPtr->exec(primArgs, strm);
}

void FullyConnected::executeDynamicImpl(dnnl::stream strm) {
    execute(strm);
}

bool FullyConnected::canFuse(const NodePtr& node) const {
    if (!getenv("ENABLE_FCFUSE")) {
        std::cout << "!!!!DISBALE FC FUSE" << std::endl;
        return false;
    }
    if (node->getType() == Type::FakeQuantize) {
        bool ret = node->getAlgorithm() != Algorithm::FQBinarization;
        for (size_t i = 1; i < node->getParentEdges().size(); i++) {
            ret &= node->getParentEdgesAtPort(i)[0]->getParent()->getChildEdges().size() == 1;
        }
        return ret;
    } else if (node->getType() == Type::Eltwise) {
        return DnnlExtensionUtils::isUnarySupportedAsPostOp(node->getAlgorithm()) ||
            node->canBePerformedAsScaleShift(this);
    }
    return false;
}

void FullyConnected::setPostOps(dnnl::primitive_attr& attr, const VectorDims& dims_ext) {
    dnnl::post_ops ops;

    // accoridng to https://oneapi-src.github.io/oneDNN/dev_guide_inner_product.html
    // oneDNN inner product primitive's input & output tensors are always 2D:
    //   input: [N, IC]  weight: [OC, IC]   bias: [OC]   output:[N,OC]
    //
    // when input output tensors have spatial dimensions, they are flattened to 2D.
    // and following type of MatMul will be converted into FullyConnected inside CPU plugin:
    //    2D:   [X,Y] [Y,Z] =>   [X,Z]   with    N=X,IC=Y,OC=Z
    //    3D: [B,X,Y] [Y,Z] => [B,X,Z]   with  N=B*X,IC=Y,OC=Z

    VectorDims dims;
    if (dims_ext.size() == 2) {
        // 2D
        dims = dims_ext;
    } else if (dims_ext.size() == 3) {
        // 3D
        dims.push_back(dims_ext[0] * dims_ext[1]);
        dims.push_back(dims_ext[2]);
    } else {
        IE_THROW() << "Unexpected rank(" << dims_ext.size() << ") for output tensor of node: " << getName();
    }


    DnnlPostOpsComposer dnnlpoc(getEngine(), attr, ops, postOpsArgs, dims, dims.size() - 1, isINT8, 1 << 0,  getDQScales(), withBiases);

    for (size_t i = 0; i < fusedWith.size(); ++i) {
        auto& node = fusedWith[i];
        bool isLastPostOp = (i == (fusedWith.size() - 1));

        if (auto* fakeQuantizeNode = dynamic_cast<FakeQuantize*>(node.get())) {
            fakeQuantizeNode->appendAttrPostOps(dnnlpoc, isLastPostOp, outputDataType);
            continue;
        }

        if (auto* eltwiseNode = dynamic_cast<Eltwise*>(node.get())) {
            eltwiseNode->appendAttrPostOps(dnnlpoc, isLastPostOp, outputDataType);
            continue;
        }

        IE_THROW() << "Fusing of " << NameFromType(node->getType()) << " operation to " << NameFromType(this->getType())
                   << " node is not implemented";
    }

    attr.set_post_ops(ops);
}

bool FullyConnected::created() const {
    return getType() == Type::FullyConnected;
}

const std::vector<impl_desc_type>& FullyConnected::getPrimitivesPriority() {
    std::vector<impl_desc_type> priorities = {
            impl_desc_type::unknown,
            impl_desc_type::acl,
            impl_desc_type::brgemm_sparse_avx512_amx,
            impl_desc_type::brgemm_avx512_amx,
            impl_desc_type::brgemm_avx512,
            impl_desc_type::gemm_blas,
            impl_desc_type::gemm_avx512,
            impl_desc_type::gemm_avx2,
            impl_desc_type::gemm_avx,
            impl_desc_type::gemm_sse42,
            impl_desc_type::gemm_any,
            impl_desc_type::gemm,
            impl_desc_type::jit_gemm,
            impl_desc_type::jit_uni_dw,
            impl_desc_type::jit_uni_1x1,
            impl_desc_type::jit_uni,
            impl_desc_type::jit_avx512_dw,
            impl_desc_type::jit_avx512_1x1,
            impl_desc_type::jit_avx512,
            impl_desc_type::jit_avx2_dw,
            impl_desc_type::jit_avx2_1x1,
            impl_desc_type::jit_avx2,
            impl_desc_type::jit_avx_dw,
            impl_desc_type::jit_avx_1x1,
            impl_desc_type::jit_avx,
            impl_desc_type::jit_sse42_dw,
            impl_desc_type::jit_sse42_1x1,
            impl_desc_type::jit_sse42,
            impl_desc_type::ref,
    };

    for (const auto& impl : priorities) {
        if (std::find(implPriorities.begin(), implPriorities.end(), impl) == implPriorities.end())
            implPriorities.push_back(impl);
    }
    return implPriorities;
}

// WA: creation DnnlMemoryDesc with format == any is prohibited
// so we create dnnl::memory::desc directly
// we need specific method and can't remove createDescriptor from base class because its used into initDescriptor
void FullyConnected::createDescriptorInternal(const dnnl::memory::desc &inputDesc,
                                              const dnnl::memory::desc &outputDesc) {
    auto create2Dcandidate = [](const dnnl::memory::desc &desc) {
        if (desc.get_dims().size() != 3) // already 2D
            return desc;

        auto inDims = desc.get_dims();
        auto normalizedInDims = {inDims[0] * inDims[1], inDims[2]};

        return dnnl::memory::desc(normalizedInDims, desc.get_data_type(),
                                  DnnlExtensionUtils::GetPlainFormatByRank(normalizedInDims.size()));
    };

    const auto in_candidate  = create2Dcandidate(inputDesc);
    const auto out_candidate = create2Dcandidate(outputDesc);

    const dnnl::memory::data_type indt = inputDesc.get_data_type();
    const dnnl::memory::data_type outdt = outputDesc.get_data_type();
    dnnl::memory::data_type wdt = indt;
    dnnl::memory::data_type bdt = outdt;

    if (indt == dnnl::memory::data_type::bf16) {
        bdt = dnnl::memory::data_type::f32;
    } else if (indt == dnnl::memory::data_type::u8 || indt == dnnl::memory::data_type::s8) {
        wdt = memory::data_type::s8;
        if (withBiases)
            bdt = DnnlExtensionUtils::IEPrecisionToDataType(getOriginalInputPrecisionAtPort(BIAS_ID));
    }
    // We need to explicitly specify the memory descriptor to use sparse weights decompression
    dnnl::memory::desc wgh_candidate;
    if (useSparseWeights) {
        wgh_candidate = wgh_candidate.sparse_desc(DnnlExtensionUtils::convertToDnnlDims(getInputShapeAtPort(WEIGHTS_ID).getStaticDims()),
                                                  wdt);
    } else {
        wgh_candidate = { DnnlExtensionUtils::convertToDnnlDims(getInputShapeAtPort(WEIGHTS_ID).getStaticDims()),
                                        wdt, dnnl::memory::format_tag::any };
    }

    const dnnl::primitive_attr attr;

    if (withBiases) {
        dnnl::memory::desc bias_candidate(DnnlExtensionUtils::convertToDnnlDims(getInputShapeAtPort(BIAS_ID).getStaticDims()), bdt,
                                            dnnl::memory::format_tag::any);
        auto desc = inner_product_forward::primitive_desc(
            getEngine(),
            prop_kind::forward_inference,
            in_candidate,
            wgh_candidate,
            bias_candidate,
            out_candidate,
            attr,
            true);

        descs.push_back(desc);
    } else {
        auto desc = inner_product_forward::primitive_desc(
            getEngine(),
            prop_kind::forward_inference,
            in_candidate,
            wgh_candidate,
            out_candidate,
            attr,
            true);

        descs.push_back(desc);
    }
}

void FullyConnected::createDescriptor(const std::vector<MemoryDescPtr> &inputDesc,
                                      const std::vector<MemoryDescPtr> &outputDesc) {
    MemoryDescPtr inpDesc;
    if (inputDesc[0]->isDefined()) {
        inpDesc = inputDesc[0];
    } else {
        inpDesc = inputDesc[0]->cloneWithNewDims(inDims);
    }

    MemoryDescPtr outDesc;
    if (outputDesc[0]->isDefined()) {
        outDesc = outputDesc[0];
    } else {
        outDesc = outputDesc[0]->cloneWithNewDims(outDims);
    }
    createDescriptorInternal(MemoryDescUtils::convertToDnnlMemoryDesc(inpDesc)->getDnnlDesc(),
                             MemoryDescUtils::convertToDnnlMemoryDesc(outDesc)->getDnnlDesc());
}

void FullyConnected::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;
    if (useMlas) {
        auto dataPrecision = getOriginalInputPrecisionAtPort(0);
        auto weightPrecision = dataPrecision;
        if (isINT8) {
            weightPrecision = InferenceEngine::Precision(InferenceEngine::Precision::I8);
        }
        if (withBiases) {
            //To Do withBias is not supported for mlas now
            addSupportedPrimDesc({{LayoutType::ncsp, dataPrecision},
                            {LayoutType::ncsp, weightPrecision},
                            {LayoutType::ncsp, dataPrecision}},
                            {{LayoutType::ncsp, DnnlExtensionUtils::DataTypeToIEPrecision(outputDataType)}},
                            impl_desc_type::gemm_mlas);
        } else {
            addSupportedPrimDesc({{LayoutType::ncsp, dataPrecision},
                {LayoutType::ncsp, weightPrecision}},
                {{LayoutType::ncsp, DnnlExtensionUtils::DataTypeToIEPrecision(outputDataType)}},
                impl_desc_type::gemm_mlas);
        }
    } else {
        for (auto& desc : descs) {
            primitive_desc_iterator itpd = desc;
            while (static_cast<bool>(itpd)) {
                // 3D FC requires implicit reshape so strides should be defined
                auto supportsUndefStridesAndOffset = [&]() {
                    return getOutputShapeAtPort(0).getRank() == 2;
                };

                NodeConfig config;
                for (size_t i = 0; i < descInputNumbers(); i++) {
                    PortConfig portConfig;
                    portConfig.inPlace(-1);
                    portConfig.constant(false);
                    auto desc = getSrcMemDesc(itpd, i);
                    if (supportsUndefStridesAndOffset() && !(i == WEIGHTS_ID && useSparseWeights)) {
                        portConfig.setMemDesc(std::dynamic_pointer_cast<BlockedMemoryDesc>(desc), BLOCKED_DESC_EMPTY_MASK);
                    } else {
                        portConfig.setMemDesc(desc);
                    }
                    config.inConfs.push_back(portConfig);
                }

                for (size_t i = 0; i < descOutputNumbers(); i++) {
                    PortConfig portConfig;
                    portConfig.inPlace(canBeInPlace() ? 0 : -1);
                    portConfig.constant(false);
                    auto desc = getDstMemDesc(itpd, i);
                    if (supportsUndefStridesAndOffset()) {
                        portConfig.setMemDesc(std::dynamic_pointer_cast<BlockedMemoryDesc>(desc), BLOCKED_DESC_EMPTY_MASK);
                    } else {
                        portConfig.setMemDesc(desc);
                    }
                    config.outConfs.push_back(portConfig);
                }

                impl_desc_type impl_type = parse_impl_name(itpd.impl_info_str());

                supportedPrimitiveDescriptors.emplace_back(config, impl_type);

                if (!itpd.next_impl())
                    break;
            }
        }
    }
}

std::shared_ptr<MemoryDesc> FullyConnected::getSrcMemDesc(dnnl::primitive_desc_iterator &primitive_desc_it, size_t idx) {
    auto desc = idx > 0 ? primitive_desc_it.weights_desc(idx - 1) : primitive_desc_it.src_desc(idx);

    if (getInputShapeAtPort(idx).getRank() == 3) {
        return std::make_shared<CpuBlockedMemoryDesc>(
            DnnlExtensionUtils::DataTypeToIEPrecision(desc.get_data_type()), getInputShapeAtPort(idx));
    }

    if (getInputShapeAtPort(idx).isDynamic()) {
        return DnnlExtensionUtils::makeUndefinedDesc(desc, getInputShapeAtPort(idx));
    }

    return DnnlExtensionUtils::makeDescriptor(desc);
}

std::shared_ptr<MemoryDesc> FullyConnected::getDstMemDesc(dnnl::primitive_desc_iterator &primitive_desc_it, size_t idx) {
    auto desc = primitive_desc_it.dst_desc(idx);

    if (getOutputShapeAtPort(idx).getRank() == 3) {
        return std::make_shared<CpuBlockedMemoryDesc>(
            DnnlExtensionUtils::DataTypeToIEPrecision(desc.get_data_type()), getOutputShapeAtPort(idx));
    }

    if (getOutputShapeAtPort(idx).isDynamic()) {
        return DnnlExtensionUtils::makeUndefinedDesc(desc, getOutputShapeAtPort(idx));
    }

    return DnnlExtensionUtils::makeDescriptor(desc);
}

InferenceEngine::Precision FullyConnected::getRuntimePrecision() const {
    std::vector<InferenceEngine::Precision> inputPrecisions;
    // Don't take bias precision into account
    size_t inputsNumLimit = 2;
    for (size_t i = 0; i < std::min(getParentEdges().size(), inputsNumLimit); i++) {
        auto parentEdge = getParentEdgeAt(i);
        if (parentEdge && parentEdge->getStatus() == Edge::Status::Validated) {
            inputPrecisions.emplace_back(DnnlExtensionUtils::DataTypeToIEPrecision((parentEdge->getMemoryPtr()->GetDataType())));
        }
    }

    return getMaxPrecision(inputPrecisions);
}

void FullyConnected::initOptimalPrimitiveDescriptor() {
    Node::initOptimalPrimitiveDescriptor();
    auto selectedPD = getSelectedPrimitiveDescriptor();
    implementationTypeIP = selectedPD->getImplementationType();
    // if convolution selected the reorder for ip is useless. Will do the reoder for ip in prepareParams
    auto constParent = getParentEdgeAt(1)->getParent();
    auto selectedParentPD = constParent->getSelectedPrimitiveDescriptor();
    auto config = selectedPD->getConfig();
    weightDescIP = config.inConfs[1].getMemDesc();
    config.inConfs[1].setMemDesc(selectedParentPD->getConfig().outConfs[0].getMemDesc());
    selectedPD->setConfig(config);
}

dnnl::convolution_forward::primitive_desc
FullyConnected::createDescriptorInternalForConv(DnnlMemoryDescCPtr inputDescPtr,
                                                DnnlMemoryDescCPtr weightDescPtr,
                                                DnnlMemoryDescCPtr biasDescPtr,
                                                DnnlMemoryDescCPtr outputDescPtr,
                                                const dnnl::primitive_attr& attr,
                                                const dnnl::engine& engine) {
    const dnnl::memory::desc &inputDesc  = inputDescPtr->getDnnlDesc();
    const dnnl::memory::desc &outputDesc = outputDescPtr->getDnnlDesc();
    const dnnl::memory::desc &weightDesc = weightDescPtr->getDnnlDesc();

    // make a fake shape: N, IC, W
    auto inDims = inputDesc.get_dims();
    dnnl::memory::dims normalizedInDims;
    if (inDims.size() == 3) {
        normalizedInDims = {inDims[0], inDims[2], inDims[1]};
    } else if (inDims.size() == 2) {
        normalizedInDims = {dnnl::memory::dim{1}, inDims[1], inDims[0]};
    }
    auto convInDesc = dnnl::memory::desc(normalizedInDims, inputDesc.get_data_type(), memory::format_tag::nwc);

    // make a fake shape: N, OC, W
    auto outDims = outputDesc.get_dims();
    dnnl::memory::dims normalizedOutDims;
    if (outDims.size() == 3) {
        normalizedOutDims = { outDims[0], outDims[2], outDims[1]};
    } else if (outDims.size() == 2) {
        normalizedOutDims = { dnnl::memory::dim{1}, outDims[1], outDims[0]};
    }
    auto convOutDesc = dnnl::memory::desc(normalizedOutDims, outputDesc.get_data_type(), memory::format_tag::nwc);

    // make a fake shape: OC, IC, 1
    auto weightDims = weightDesc.get_dims();
    dnnl::memory::dims normalizedWeightDims;
    normalizedWeightDims = {static_cast<dnnl::memory::dim>(weightDims[0]),
                            static_cast<dnnl::memory::dim>(weightDims[1]),
                            dnnl::memory::dim{1}};
    auto convWeightDescAny = dnnl::memory::desc(normalizedWeightDims, weightDesc.get_data_type(), dnnl::memory::format_tag::any);

    if (biasDescPtr) {
        return dnnl::convolution_forward::primitive_desc(
            engine,
            prop_kind::forward_inference,
            dnnl::algorithm::convolution_direct,
            convInDesc, convWeightDescAny, biasDescPtr->getDnnlDesc(), convOutDesc,
            dnnl::memory::dims{1},   // stride
            dnnl::memory::dims{0},   // dilation
            dnnl::memory::dims{0},   // paddingL
            dnnl::memory::dims{0},   // paddingR
            attr);
    } else {
        return dnnl::convolution_forward::primitive_desc(
            engine,
            prop_kind::forward_inference, dnnl::algorithm::convolution_direct,
            convInDesc, convWeightDescAny, convOutDesc,
            dnnl::memory::dims{1},   // stride
            dnnl::memory::dims{0},   // dilation
            dnnl::memory::dims{0},   // paddingL
            dnnl::memory::dims{0},   // paddingR
            attr);
    }
}

bool FullyConnected::canBeExecutedInConv1x1() const {
    bool retVal = false;
    const auto inRank = getInputShapeAtPort(DATA_ID).getRank();
    const auto weightRank = getInputShapeAtPort(WEIGHTS_ID).getRank();
    // disable rank=4:
    // if layout is nhwc:
    //   A matrix: N * IC * H * W --> N * (IC*H*W), the M, N', K of matrix multiply will be:
    //   M = 1, K = (IC*H*W), when M = 1 it should not be efficient since acts as a vector multiply
    // if layout is nchw/nChw16c: brg1x1 not support. Although jit supports, it should have similar
    //   problems with the above.
    if (dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core) &&
        getOriginalInputPrecisionAtPort(DATA_ID) == InferenceEngine::Precision::FP32 &&
        one_of(inRank, 2u, 3u) && weightRank == 2) {
        auto dstMemPtr = getChildEdgesAtPort(0)[0]->getMemoryPtr();
        DnnlMemoryDescCPtr outDesc = dstMemPtr->GetDescWithType<DnnlMemoryDesc>();
        // brg convolution does not support stride
        dnnl::impl::memory_desc_wrapper wrapped(outDesc->getDnnlDesc().get());
        if (wrapped.offset0() == 0)
            retVal = true;
    }

    if (retVal) {
        auto srcMemPtr = getParentEdgesAtPort(0)[0]->getMemoryPtr();
        const auto& srcDims = srcMemPtr->getStaticDims();
        auto weightMemPtr = getParentEdgesAtPort(1)[0]->getMemoryPtr();
        const auto& weightDims = weightMemPtr->getStaticDims();
        // for original inner product semantics:
        //  when input is 2D tensor
        //    M in oneDNN will map to widthInConv
        //  when input is 3D tensor
        //    M in oneDNN will map to widthInConv*minibatch
        // currently nwc mapping in brg:
        //  when input is 2D tensor
        //    widthInConv will map to 'w', 'n' will be 1
        //  when input is 3D tensor
        //    widthInConv will map to 'w', 'n' will be minibatch
        Dim widthInConv, N, K;
        widthInConv = srcDims[inRank - 2];
        K = srcDims[inRank - 1];
        N = weightDims[0];

        if (!(widthInConv >= 2 && widthInConv <= 3136 &&
              K >= 96 && K <= 4096 &&
              N >= 96 && N <= K * 4))
            retVal = false;
    }

    return retVal;
}

bool FullyConnected::useSparseWeightsDecompression() {
    // minSparseRate == 1 means that sparse feature is switched off
    if (minSparseRate == 1.f) {
        return false;
    }

    if (!impl::cpu::x64::mayiuse(impl::cpu::x64::avx512_core_amx))
        return false;

    auto weiDims = getInputShapeAtPort(WEIGHTS_ID).getStaticDims();
    if (weiDims.size() != 2 || weiDims[0] % 64 != 0 || weiDims[1] % 64 != 0) {
        return false;
    }

    auto inputPrecision = getOriginalInputPrecisionAtPort(DATA_ID);
    auto weightsPrecision = getOriginalInputPrecisionAtPort(WEIGHTS_ID);
    if (!one_of(inputPrecision , Precision::U8, Precision::I8) || weightsPrecision != Precision::I8) {
        return false;
    }

    // calculate sparse rate
    const auto constNode = std::dynamic_pointer_cast<Input>(getParentEdgeAt(WEIGHTS_ID)->getParent());
    if (!constNode) {
        return false;
    }
    auto blb = constNode->getMemoryPtr();
    if (blb == nullptr)
        IE_THROW() << "Cannot get const blob for node " << getName() << ".";

    auto weightsData = reinterpret_cast<const int8_t*>(blb->GetPtr());
    auto elementsCount = blb->GetDescWithType<BlockedMemoryDesc>()->getPaddedElementsCount();
    size_t zerosCounts = 0;
    for (size_t i = 0; i < elementsCount; i++) {
        if (weightsData[i] == 0) {
            zerosCounts++;
        }
    }

    DEBUG_LOG(getName(), ", elementsCount = ", elementsCount, ", zerosCounts = ",
        zerosCounts, ", nnzCount = ", elementsCount - zerosCounts);

    weiSparseRate = static_cast<float>(zerosCounts) / static_cast<float>(elementsCount);

    DEBUG_LOG(getName(), " | sparse rate = ", weiSparseRate * 100, "%, min sparse rate = ",
        minSparseRate * 100, "%, use sparse weights = ", weiSparseRate >= minSparseRate);

    if (weiSparseRate < minSparseRate) {
        return false;
    }

    return true;
}

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
