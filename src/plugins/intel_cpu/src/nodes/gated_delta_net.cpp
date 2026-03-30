// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gated_delta_net.h"

#include <common/utils.hpp>
#include <cstddef>
#include <cmath>
#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>
#include <vector>

#include "cpu_memory.h"
#include "graph_context.h"
#include "kernels/linear_attn/recurrent_linear_attn.hpp"
#include "memory_desc/cpu_blocked_memory_desc.h"
#include "memory_desc/cpu_memory_desc.h"
#include "node.h"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/gated_delta_net.hpp"
#include "shape_inference/shape_inference_cpu.hpp"
#include "utils/plain_tensor.hpp"
#if defined(OPENVINO_ARCH_X86_64)
#    include "cpu_parallel.hpp"
#    include "kernels/x64/gdn_jit_kernel.hpp"
#endif

using namespace ov::Extensions::Cpu;
using namespace ov::Extensions::Cpu::XARCH;

namespace ov::intel_cpu::node {

#if defined(OPENVINO_ARCH_X86_64)
namespace {
struct GatedDeltaNetKey {
    ov::element::Type precision;
    size_t qk_head_size;
    bool fuse_qk_l2norm;
    float q_l2_norm_eps;
    float k_l2_norm_eps;

    [[nodiscard]] size_t hash() const {
        size_t seed = 0;
        seed = dnnl::impl::hash_combine(seed, precision.hash());
        seed = dnnl::impl::hash_combine(seed, qk_head_size);
        seed = dnnl::impl::hash_combine(seed, fuse_qk_l2norm);
        seed = dnnl::impl::hash_combine(seed, q_l2_norm_eps);
        seed = dnnl::impl::hash_combine(seed, k_l2_norm_eps);
        return seed;
    }

    bool operator==(const GatedDeltaNetKey& rhs) const {
        return precision == rhs.precision && qk_head_size == rhs.qk_head_size &&
               fuse_qk_l2norm == rhs.fuse_qk_l2norm && q_l2_norm_eps == rhs.q_l2_norm_eps &&
               k_l2_norm_eps == rhs.k_l2_norm_eps;
    }
};

void recurrent_linear_attn_jit(const ov::intel_cpu::PlainTensor& query,
                               const ov::intel_cpu::PlainTensor& key,
                               const ov::intel_cpu::PlainTensor& value,
                               const ov::intel_cpu::PlainTensor& recurrent_state,
                               const ov::intel_cpu::PlainTensor& gate,
                               const ov::intel_cpu::PlainTensor& beta,
                               ov::intel_cpu::PlainTensor& output_attn,
                               ov::intel_cpu::PlainTensor& output_recurrent_state,
                               float* temp_buffer,
                               const ov::intel_cpu::CpuParallelPtr& cpu_parallel,
                               const std::shared_ptr<kernel::JitKernelBase>& jit_kernel) {
    OPENVINO_ASSERT(jit_kernel, "GDN JIT kernel is not created");

    const size_t B = query.m_dims[0];
    const size_t T = query.m_dims[1];
    const size_t H = query.m_dims[2];
    const size_t K = query.m_dims[3];
    const size_t V = value.m_dims[3];
    cpu_parallel->parallel_for3d(B, H, V, [&](size_t i_b, size_t i_h, size_t i_v) {
        const size_t tid = parallel_get_thread_num();
        float* init_state = temp_buffer + tid * 3 * K;
        float* b_k = temp_buffer + tid * 3 * K + K;
        float* b_q = temp_buffer + tid * 3 * K + 2 * K;

        float* q_ptr = query.ptr<float>(i_b, 0, i_h);
        float* k_ptr = key.ptr<float>(i_b, 0, i_h);
        float* v_ptr = value.ptr<float>(i_b, 0, i_h);
        float* out_ptr = output_attn.ptr<float>(i_b, 0, i_h) + i_v;

        float* gate_ptr = gate.ptr<float>(i_b, 0, i_h);
        float* beta_ptr = beta.ptr<float>(i_b, 0, i_h);

        for (size_t j = 0; j < K; j++) {
            init_state[j] = recurrent_state.at<float>({i_b, i_h, j, i_v});
        }

        kernel::jit_gdn_call_args args{};
        args.state = reinterpret_cast<uint8_t*>(init_state);
        args.key_seq = reinterpret_cast<const uint8_t*>(k_ptr);
        args.query_seq = reinterpret_cast<const uint8_t*>(q_ptr);
        args.value_seq = v_ptr + i_v;
        args.gate_seq = gate_ptr;
        args.beta_seq = beta_ptr;
        args.t_size = T;
        args.key_query_stride = H * K;
        args.gate_beta_stride = H;
        args.value_stride = H * V;
        args.output_stride = H * V;
        args.key_tmp = reinterpret_cast<uint8_t*>(b_k);
        args.query_tmp = reinterpret_cast<uint8_t*>(b_q);
        args.output_seq = out_ptr;
        (*jit_kernel)(&args);

        for (size_t j = 0; j < K; j++) {
            output_recurrent_state.at<float>({i_b, i_h, j, i_v}) = init_state[j];
        }
    });
}

}  // namespace
#endif

GatedDeltaNet::GatedDeltaNet(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, NgraphShapeInferFactory(op)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }
    const auto& gdn = ov::as_type_ptr<ov::op::internal::GatedDeltaNet>(op);
    m_fuse_qk_l2norm = gdn->get_fuse_qk_l2norm();
    m_q_l2_norm_eps = gdn->get_q_l2_norm_eps();
    m_k_l2_norm_eps = gdn->get_k_l2_norm_eps();
}

void GatedDeltaNet::initSupportedPrimitiveDescriptors() {
    // TODO: support other precision CVS-182464
    auto dataPrecision = ov::element::f32;
    std::vector<PortConfigurator> inPortConfigs;
    for (size_t i = 0; i < getParentEdges().size(); ++i) {
        inPortConfigs.emplace_back(LayoutType::ncsp, dataPrecision, getInputShapeAtPort(i), false, -1);
    }
    std::vector<PortConfigurator> outPortConfigs = {
        PortConfigurator{LayoutType::ncsp, dataPrecision, getOutputShapeAtPort(0), false, -1},
        PortConfigurator{LayoutType::ncsp, dataPrecision, getOutputShapeAtPort(1), false, -1}};
    addSupportedPrimDesc(inPortConfigs, outPortConfigs, impl_desc_type::ref_any);
}

void GatedDeltaNet::createPrimitive() {
    const auto precision = ov::element::f32;
    const auto queryDims = getInputShapeAtPort(0).getDims();
    auto headSize = *(queryDims.end() - 1);
    const auto numWorkerThreads = context->getCpuParallel()->get_num_worker_threads();
    auto newMemDesc = std::make_shared<CpuBlockedMemoryDesc>(
        precision,
        ov::intel_cpu::Shape{static_cast<size_t>(numWorkerThreads), 3 * headSize});
    m_tmpInpBuffer = context->getScratchPad()->createScratchPadMem(newMemDesc);
#if defined(OPENVINO_ARCH_X86_64)
    GatedDeltaNetKey key{precision, headSize, m_fuse_qk_l2norm, m_q_l2_norm_eps, m_k_l2_norm_eps};

    auto builder = [&](const GatedDeltaNetKey& compile_key) -> std::shared_ptr<kernel::JitKernelBase> {
        return kernel::create_gdn_jit_kernel(compile_key.precision,
                                             compile_key.qk_head_size,
                                             compile_key.fuse_qk_l2norm,
                                             compile_key.q_l2_norm_eps,
                                             compile_key.k_l2_norm_eps);
    };

    auto cache = context->getParamsCache();
    auto result = cache->getOrCreate(key, builder);
    m_gdnJitKernel = result.first;
#endif
}

void GatedDeltaNet::execute([[maybe_unused]] const dnnl::stream& strm) {
    auto originalInputNumber = getOriginalInputsNumber();
    std::vector<MemoryPtr> inputs(originalInputNumber);
    std::vector<MemoryPtr> outputs(2);

    for (size_t i = 0; i < originalInputNumber; i++) {
        inputs[i] = getSrcMemoryAtPort(i);
    }

    outputs[0] = getDstMemoryAtPort(0);
    outputs[1] = getDstMemoryAtPort(1);

    PlainTensor query(inputs[0]);
    PlainTensor key(inputs[1]);
    PlainTensor value(inputs[2]);
    PlainTensor recurrent_state(inputs[3]);
    PlainTensor gate(inputs[4]);
    PlainTensor beta(inputs[5]);
    PlainTensor output_attn(outputs[0]);
    PlainTensor output_recurrent_state(outputs[1]);

    auto* temp_buffer = m_tmpInpBuffer->getDataAs<float>();
#if defined(OPENVINO_ARCH_X86_64)
    if (m_gdnJitKernel) {
        recurrent_linear_attn_jit(query,
                                  key,
                                  value,
                                  recurrent_state,
                                  gate,
                                  beta,
                                  output_attn,
                                  output_recurrent_state,
                                  temp_buffer,
                                  context->getCpuParallel(),
                                  m_gdnJitKernel);
        return;
    }
#endif

    recurrent_linear_attn(query,
                          key,
                          value,
                          recurrent_state,
                          gate,
                          beta,
                          m_q_l2_norm_eps,
                          m_k_l2_norm_eps,
                          m_fuse_qk_l2norm,
                          output_attn,
                          output_recurrent_state,
                          temp_buffer,
                          context->getCpuParallel());
}

bool GatedDeltaNet::isSupportedOperation(const std::shared_ptr<const ov::Node>& op,
                                         std::string& errorMessage) noexcept {
    if (op == nullptr || !ov::is_type<ov::op::internal::GatedDeltaNet>(op)) {
        errorMessage = "Node is not an instance of ov::op::internal::GatedDeltaNet.";
        return false;
    }
    return true;
}

}  // namespace ov::intel_cpu::node
