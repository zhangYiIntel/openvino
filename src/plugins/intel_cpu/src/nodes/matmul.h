// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <node.h>
#include <ie_common.h>
#include <string>
#include <vector>
#include <array>
#include "memory_desc/dnnl_blocked_memory_desc.h"
#include "common/dnnl_executor.h"
#include "ngraph/runtime/aligned_buffer.hpp"

#include "kernels/x64/kernels_avx2.hpp"
#include "ie_parallel.hpp"
namespace ov {
namespace intel_cpu {
namespace node {

struct Avx2MatMul {
    std::vector<std::shared_ptr<avx2::Matmul>> ops;
    avx2::PP::None pp_none;

    Avx2MatMul() = default;

    void setup(bool b_is_const) {
        auto NT = parallel_get_num_threads();
        for(int i = 0; i < NT; i++) {
            ops.push_back(std::make_shared<avx2::Matmul>(b_is_const, false));
        }
    }

    void operator()(float * A, float * B, float * C,
                  size_t M, size_t K, size_t N) {
        constexpr int bN = 16;
        size_t Nb = (N + bN - 1)/bN;
        tensor2D<float> matA(M, K, A, K*sizeof(float));
        tensor2D<float> matB(K, N, B, N*sizeof(float));
        tensor2D<float> matC(M, N, C, N*sizeof(float));
        parallel_nt_static(0, [&](int ithr, int nthr) {
            // each work item is doing  M x bN sub-states encoding
            // and finally, main thread will combine sub-states into one
            size_t start{0}, end{0};
            splitter(Nb, nthr, ithr, start, end);
            auto n0 = start * bN;
            auto n1 = end * bN;
            if (n1 > N) n1 = N;
            (*ops[ithr])(matA, matB, matC, n0, n1, pp_none);
        });
    }
};

class MatMul : public Node {
public:
    Avx2MatMul kern;

    MatMul(const std::shared_ptr<ngraph::Node>& op, const GraphContext::CPtr context);

    void getSupportedDescriptors() override;
    void createDescriptor(const std::vector<MemoryDescPtr>& inputDesc,
                          const std::vector<MemoryDescPtr>& outputDesc) override;
    void initSupportedPrimitiveDescriptors() override;
    MemoryDescPtr getSrcMemDesc(dnnl::primitive_desc_iterator &primitive_desc_it, size_t idx) override;
    bool canFuse(const NodePtr& node) const override;
    bool created() const override;

    InferenceEngine::Precision getRuntimePrecision() const override;
    size_t descInputNumbers() override {
        return getOriginalInputsNumber();
    }

    int getFusingAxis() const override {
        return getOutputShapeAtPort(0).getRank() - 1;
    }

    void prepareParams() override;
    void execute(dnnl::stream strm) override;
    void executeDynamicImpl(dnnl::stream strm) override;

    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;
    const std::vector<impl_desc_type>& getPrimitivesPriority() override;

protected:
    AttrPtr initPrimitiveAttr() override;
    AttrPtr initPrimitiveAttr(const VectorDims& dims);

private:
    using executorPtr = std::shared_ptr<DnnlExecutor>;
    executorPtr execPtr = nullptr;
    dnnl::memory::desc getBiasDescFrom(const DnnlMemoryDescCPtr outMemDesc);
    std::pair<Shape, Shape> makeDummyInputShapes(const Shape& in0, const Shape& in1) const;

    bool withBiases;

    void setPostOps(dnnl::primitive_attr &attr, const VectorDims& dims, bool initWeights);

    std::string errorPrefix;

    /* whether to transpose input */
    std::array<bool, 2> transposeIn;

    std::array<DnnlBlockedMemoryDescPtr, 2> inDataDesc;
    DnnlBlockedMemoryDescPtr outDataDesc;
    std::shared_ptr<ngraph::runtime::AlignedBuffer> packedBPtr = nullptr;
    size_t K, N = 0;
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
