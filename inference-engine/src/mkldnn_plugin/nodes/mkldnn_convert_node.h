// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <mkldnn_node.h>
#include <string>
#include <vector>

namespace MKLDNNPlugin {

class MKLDNNConvertNode : public MKLDNNNode {
public:
    MKLDNNConvertNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);
    MKLDNNConvertNode(const InferenceEngine::SizeVector &dims, const InferenceEngine::Precision &inPrc, const InferenceEngine::Precision &outPrc,
                      const std::string &nodeName, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;
    void execute(mkldnn::stream strm) override;
    bool created() const override;
    bool canBeInPlace() const override {
        return false;
    }

    // This is the interface extension designed to provide inp and output tensor descriptors without the CNNLayer.
    // In that case the Convert node is instantiated with default CNNLayer and inp/out tensor descriptors are set via this method.
    // This is useful if the Convert node is added to the graph as an auxiliary operation at the MKLDNNGraph
    // initialization stage.
    void setDescs(const MemoryDesc& input, const MemoryDesc& output) {
        this->input = input.clone();
        this->output = output.clone();
    }

    const MemoryDesc& getInput() const { return *input; }
    const MemoryDesc& getOutput() const { return *output; }

    static bool isSupportedOperation(const std::shared_ptr<ngraph::Node>& op, std::string& errorMessage) noexcept;

private:
    std::unique_ptr<MemoryDesc> input;
    std::unique_ptr<MemoryDesc> output;

    std::string errorPrefix;
};
}  // namespace MKLDNNPlugin

