// Copyright (C) 2018-2022 Intel Corporation
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

class Interaction : public Node {
public:
    Interaction(const std::shared_ptr<ngraph::Node>& op, const dnnl::engine& eng, WeightsSharing::Ptr &cache);
    void getSupportedDescriptors() override {};
    void initSupportedPrimitiveDescriptors() override;
    void execute(dnnl::stream strm) override;
    bool created() const override;

    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;

    bool isExecutable() const override;
    void executeDynamicImpl(dnnl::stream strm) override;

    bool needShapeInfer() const override { return false; }
    void prepareParams() override;

private:
    void createDescriptorInternal(const dnnl::memory::desc &inputDesc,
                                  const dnnl::memory::desc &outputDesc);

    VectorDims makeDummyInputDims() const;
    VectorDims makeDummyOutputDims(const VectorDims& inDims) const;

    VectorDims inDims;
    VectorDims outDims;

    int64_t batchSize = 0;
    int64_t featureSize = 0;
    int64_t inputSizes = 0;
    int64_t outputFeaturesLen = 0;
    std::string errorPrefix;
    dnnl::memory::data_type outputDataType;
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
