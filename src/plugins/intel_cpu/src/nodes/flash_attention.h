// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <node.h>
#include <memory>
#include <string>
#include <vector>

namespace ov {
namespace intel_cpu {
namespace node {

class FlashAttention : public Node {
public:
    FlashAttention(const std::shared_ptr<ngraph::Node>& op, const GraphContext::CPtr context);
    void getSupportedDescriptors() override {};
    void initSupportedPrimitiveDescriptors() override;
    void execute(dnnl::stream strm) override;
    bool created() const override;

    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;

    bool isExecutable() const override;
    void executeDynamicImpl(dnnl::stream strm) override;
    bool needShapeInfer() const override {
        return true;
    }
    void prepareParams() override;

private:
    std::string errorPrefix;
    int64_t qSplitSize;
    int64_t kvSplitSize;
    int64_t qSlice;
    int64_t qTail;
    int64_t kvSlice;
    int64_t kvTail;
    int64_t qStride;
    int64_t kStride;

    int64_t batchSize;
    int64_t headNum;
    int64_t headSize;
    int64_t hiddenSize;
    int64_t seqLen;

    MemoryPtr bufferQK = nullptr; // [threadNum, qSplitSize, kvSplitSize]
    MemoryPtr bufferQKMax = nullptr; // [threadNum, qSplitSize]
    MemoryPtr bufferQKSum = nullptr; // [threadNum, qSplitSize]
    MemoryPtr bufferPreOutput = nullptr; // [threadNum, qSplitSize, headSize]
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
