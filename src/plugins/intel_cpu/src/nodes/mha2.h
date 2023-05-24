// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <node.h>
#include <memory>
#include <string>
#include <vector>
#include "transformations/cpu_opset/x64/op/mha2.hpp"
#include "kernels/x64/kernels_avx2.hpp"
#include "ie_parallel.hpp"
#include "utils/profiler.hpp"

#define PARALLEL_NT_STATIC(...) ov::parallel_nt_static(0, __VA_ARGS__)

#include "kernels/x64/kernels_mha.hpp"

namespace ov {
namespace intel_cpu {
namespace node {

class MHA2 : public Node {
public:
    MHA2(const std::shared_ptr<ngraph::Node>& op, const GraphContext::CPtr context);

    void getSupportedDescriptors() override {};
    void initSupportedPrimitiveDescriptors() override;
    void execute(dnnl::stream strm) override;
    bool created() const override;

    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;
    bool needShapeInfer() const override {
        return false;
    }

protected:
    std::shared_ptr<const ov::intel_cpu::MHA2Node> mha2;
    void executeDynamicImpl(dnnl::stream strm) override;
    bool needPrepareParams() const override { return false; }

    MHA2Kernels kernels;
    bool verbose;
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
