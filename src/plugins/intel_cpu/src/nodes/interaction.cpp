// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <chrono>
#include <cmath>
#include <string>
#include <vector>

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
#include <ie_ngraph_utils.hpp>
#include <cpu/x64/cpu_isa_traits.hpp>
#include <cpu/x64/jit_generator.hpp>
#include "emitters/jit_dnnl_emitters.hpp"
#include "emitters/jit_load_store_emitters.hpp"
#include <fstream>

namespace ov {
namespace intel_cpu {
namespace node {
using namespace Xbyak;
#define THROW_ERROR IE_THROW() << getTypeStr() << " node with name '" << getName() << "' "
template <cpu_isa_t isa>
struct jit_move_scale_kernel : public jit_uni_move_scale_kernel, public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_move_scale_kernel)

    explicit jit_move_scale_kernel(const jit_move_scale_compile_params& jcp) : jit_uni_move_scale_kernel(jcp), jit_generator() {
        vec_size = dnnl::impl::cpu::x64::cpu_isa_traits<isa>::vlen / sizeof(float);
    }
    virtual ~jit_move_scale_kernel() {}

    void create_ker() override {
        jit_generator::create_kernel();
        auto ker = jit_ker();
        auto size = getSize();
        FILE* fp = ::fopen("move_scale_kernel", "wb");
        if (fp) {
            size_t unused = ::fwrite(ker, size, 1, fp);
            UNUSED(unused);
            ::fclose(fp);
        }
        ker_ = (decltype(ker_))jit_ker();
    }

private:
    using Vmm = typename dnnl::impl::utils::conditional3<isa == cpu_isa_t::sse41, Xmm, isa == cpu_isa_t::avx2, Ymm, Zmm>::type;

    void generate() override {
        this->preamble();

#define GET_OFF(field) offsetof(jit_move_scale_call_args, field)
        mov(reg_in, ptr[reg_params + GET_OFF(p_in)]);
        mov(reg_out, ptr[reg_params + GET_OFF(p_out)]);
        mov(reg_work_amount, jcp_.work_amount);

        if (jcp_.with_scales) {
            mov(reg_scales, ptr[reg_params + GET_OFF(p_scales)]);
        }

        Xbyak::Label move_scale_loop_label;
        Xbyak::Label move_scale_end_label;

        if (jcp_.with_scales && jcp_.broadcast_scales) {
            uni_vmovss(Xmm(vmm_scales.getIdx()), ptr[reg_scales]);
            uni_vbroadcastss(vmm_scales, Xmm(vmm_scales.getIdx()));
        }

        mov(reg_in_aux, reg_in);
        mov(reg_out_aux, reg_out);
        if (jcp_.with_scales && !jcp_.broadcast_scales) {
            mov(reg_scales, ptr[reg_params + GET_OFF(p_scales)]);
        }
        size_t tail_size = jcp_.work_amount % vec_size;
        L(move_scale_loop_label);
        {
            cmp(reg_work_amount, vec_size);
            jl(move_scale_end_label, T_NEAR);

            convert_reorder(vec_size);

            sub(reg_work_amount, vec_size);

            jmp(move_scale_loop_label, T_NEAR);
        }
        L(move_scale_end_label);
        if (tail_size) {
            convert_reorder(tail_size);
        }

        this->postamble();

        for (const auto& emitter : emitters) {
            if (emitter.second)
                emitter.second->emit_data();
        }
    }

    void convert_reorder(size_t step) {
        bool is_tail = step < vec_size;

        load(vmm_in, reg_in_aux, jcp_.src_prc, step, is_tail);

        if (jcp_.with_scales) {
            if (!jcp_.broadcast_scales) {
                load(vmm_scales, reg_scales, Precision::FP32, step, is_tail);
                add(reg_scales,  sizeof(float) * step);
            }
            uni_vmulps(vmm_in, vmm_in, vmm_scales);
        }

        store(reg_out_aux, vmm_in, jcp_.dst_prc, step);

        if (!is_tail) {
            add(reg_in_aux, jcp_.src_prc.size() * step);
            add(reg_out_aux, jcp_.dst_prc.size() * step);
        }
    }
#undef GET_OFF

    inline void load(const Vmm& vmm_dst, const Xbyak::Reg64& reg_src, Precision src_prc, const int& elt_num, bool fill) {
        const auto seed = load_emitter_params(src_prc, Precision::FP32, elt_num, fill, "float_min").hash();
        if (!emitters[seed]) {
            emitters[seed].reset(new jit_load_emitter(this, isa, src_prc, Precision::FP32, elt_num, Precision::FP32, fill, "float_min"));
        }

        emitters[seed]->emit_code({static_cast<size_t>(reg_src.getIdx()), 0}, {static_cast<size_t>(vmm_dst.getIdx())},
                                  pool_aux_vmm_idxs, pool_aux_gpr_idxs);
    }
    inline void store(const Xbyak::Reg64& reg_dst, const Vmm& vmm_src, Precision dst_prc, const int& elt_num) {
        const auto seed = store_emitter_params(Precision::FP32, dst_prc, elt_num).hash();
        if (!emitters[seed]) {
            emitters[seed].reset(new jit_store_emitter(this, isa, Precision::FP32, dst_prc, elt_num));
        }

        emitters[seed]->emit_code({static_cast<size_t>(vmm_src.getIdx()), 0}, {static_cast<size_t>(reg_dst.getIdx())},
                                  pool_aux_vmm_idxs, pool_aux_gpr_idxs);
    }

    size_t vec_size;

    Xmm xmm_tmp = Xmm(2);
    Vmm vmm_scales = Vmm(0);
    Vmm vmm_in = Vmm(1);

    Reg64 reg_in = r8;
    Reg64 reg_in_aux = r9;
    Reg64 reg_out = r10;
    Reg64 reg_out_aux = r11;
    Reg64 reg_scales = r12;
    Reg64 reg_work_amount = r14;
    Reg64 reg_params = abi_param1;

    const std::vector<size_t> pool_aux_gpr_idxs = { static_cast<size_t>(rsi.getIdx()), static_cast<size_t>(rbp.getIdx()) };
    const std::vector<size_t> pool_aux_vmm_idxs = { static_cast<size_t>(xmm_tmp.getIdx()) };

    std::unordered_map<size_t, std::unique_ptr<jit_emitter>> emitters;
};

Interaction::Interaction(const std::shared_ptr<ngraph::Node>& op, const dnnl::engine& eng, WeightsSharing::Ptr &cache)
        : Node(op, eng, cache) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }
    errorPrefix = "Interaction node with name '" + getName() + "'";
    const auto interaction = std::dynamic_pointer_cast<const InteractionNode>(op);
    const std::vector<float>& scales = interaction->get_output_scales();
    if (!scales.empty()) {
        fqScales = scales;
        outputDataType  = InferenceEngine::details::convertPrecision(interaction->get_fq_output_type());
    }
}

void Interaction::getSupportedDescriptors() {
    dataPrecision = getOriginalInputPrecisionAtPort(0);
    if (dataPrecision != InferenceEngine::Precision::FP32 && dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core_bf16)) {
        dataPrecision = InferenceEngine::Precision::BF16;
    } else {
        dataPrecision = InferenceEngine::Precision::FP32;
    }

    if (fqScales.empty()) {
        outputDataType = dataPrecision;;
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

static inline void cat(const uint8_t* in1, const uint8_t* in2, uint8_t* out,
    size_t in1Size, size_t in2Size, size_t elemSize) {
    cpu_memcpy(out, in1, in1Size * elemSize);
    cpu_memcpy(out + in1Size * elemSize, in2, in2Size * elemSize);
}

static inline void cat(uint8_t* out,
                       const std::vector<const uint8_t*>& in,
                       const std::vector<uint32_t>& feature_sizes,
                       int64_t bs,
                       size_t elemSize) {
    size_t offset = 0;
    for (int j = 0; j < feature_sizes.size(); j++) {
        cpu_memcpy(out + offset * elemSize, in[j] + bs * feature_sizes[j] * elemSize,
            feature_sizes[j] * elemSize);
        offset += feature_sizes[j];
    }
}


static inline void flat_triangle(const uint8_t* in, uint8_t* out, size_t size, size_t elemSize) {
    size_t offset = 0;
    for (int i = 1; i < size; i++) {
        cpu_memcpy(out + offset * elemSize, in + i * size * elemSize, i * elemSize);
        offset += i;
    }
}

void Interaction::execRef(dnnl::stream strm, bool fuseFQ) {
    using tag = dnnl::memory::format_tag;
    using dt = dnnl::memory::data_type;
    using namespace dnnl;

    uint8_t* outFeaturesPtr = reinterpret_cast<uint8_t*>(getChildEdgesAtPort(0)[0]->getMemoryPtr()->GetPtr());
    std::vector<const uint8_t*> inputPtrs(inputSizes);
    for (uint32_t n = 0; n < inputSizes; n++) {
        auto inPtr = reinterpret_cast<const uint8_t*>(getParentEdgeAt(n)->getMemoryPtr()->GetPtr());
        inputPtrs[n] = inPtr;
    }
    std::unordered_map<int, memory> mem_ags {
    {DNNL_ARG_SRC, inputMemPtr->GetPrimitive()},
    {DNNL_ARG_WEIGHTS, inputMemPtr->GetPrimitive()},
    {DNNL_ARG_DST, outputMemPtr->GetPrimitive()}};
    float* scales = fuseFQ ? fqScales.data() : nullptr;
    for (int64_t start = 0; start < batchSize; start++) {
        cat(reinterpret_cast<uint8_t*>(inputMemPtr->GetPtr()), inputPtrs, featureSizes, start, dataPrecision.size());
        (*prim).execute(strm, mem_ags);
        flat_triangle(reinterpret_cast<const uint8_t*>(outputMemPtr->GetPtr()),
            reinterpret_cast<uint8_t*>(flatMemPtr->GetPtr()), inputSizes, dataPrecision.size());
        //in1 dense feature
        //in2 flatted interaction features
        if (fuseFQ) {
            if (moveFeatureKernel) {
                jit_move_scale_call_args featArgs;
                featArgs.p_in = inputPtrs[0] + start * featureSize * dataPrecision.size();
                featArgs.p_out = outFeaturesPtr + start * outputFeaturesLen * dataPrecision.size();
                featArgs.p_scales = scales;
                (*moveFeatureKernel)(&featArgs);
            }
            if (moveInteractKernel) {
                jit_move_scale_call_args interArgs;
                interArgs.p_in = flatMemPtr->GetPtr();
                interArgs.p_out = outFeaturesPtr + (start * outputFeaturesLen + featureSize) * dataPrecision.size();
                interArgs.p_scales = scales;
                (*moveFeatureKernel)(&interArgs);
            }
        } else {
            cat(inputPtrs[0] + start * featureSize * dataPrecision.size(),
                reinterpret_cast<const uint8_t*>(flatMemPtr->GetPtr()),
                outFeaturesPtr + start * outputFeaturesLen * dataPrecision.size(),
                featureSize,
                interactFeatureSize,
                dataPrecision.size());
        }
    }
}



void Interaction::execute(dnnl::stream strm) {
    if (fqScales.empty())
        execRef(strm);
    else
        execRef(strm, true);
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
    auto initMemoryPtr = [&](const InferenceEngine::Precision &prc, const intel_cpu::Shape& shape,
        MemoryPtr& ptr) {
        ptr = std::make_shared<Memory>(getEngine());
        ptr->Create(intel_cpu::DnnlBlockedMemoryDesc(prc, shape));
    };
    initMemoryPtr(dataPrecision, intel_cpu::Shape{inputSizes, featureSize}, inputMemPtr);
    initMemoryPtr(dataPrecision, intel_cpu::Shape{inputShapes.size(), inputShapes.size()}, outputMemPtr);
    initMemoryPtr(dataPrecision, intel_cpu::Shape{interactFeatureSize}, flatMemPtr);

    jit_move_scale_compile_params jcp;
    jcp.src_prc = dataPrecision;
    jcp.dst_prc = getOriginalOutputPrecisionAtPort(0);
    jcp.with_scales = !fqScales.empty();
    jcp.broadcast_scales = fqScales.size() == 1;
    jcp.work_amount = featureSize;

    jit_move_scale_compile_params interJcp;
    interJcp.src_prc = dataPrecision;
    interJcp.dst_prc = getOriginalOutputPrecisionAtPort(0);
    interJcp.with_scales = !fqScales.empty();
    interJcp.broadcast_scales = fqScales.size() == 1;
    interJcp.work_amount = interactFeatureSize;
    if (!fqScales.empty()) {
        if (mayiuse(cpu_isa_t::avx512_core)) {
            moveFeatureKernel.reset(new jit_move_scale_kernel<cpu_isa_t::avx512_core>(jcp));
            moveInteractKernel.reset(new jit_move_scale_kernel<cpu_isa_t::avx512_core>(interJcp));
        } else if (mayiuse(cpu_isa_t::avx2)) {
            moveFeatureKernel.reset(new jit_move_scale_kernel<cpu_isa_t::avx2>(jcp));
            moveInteractKernel.reset(new jit_move_scale_kernel<cpu_isa_t::avx2>(interJcp));
        } else if (mayiuse(cpu_isa_t::sse41)) {
            moveFeatureKernel.reset(new jit_move_scale_kernel<cpu_isa_t::sse41>(jcp));
            moveInteractKernel.reset(new jit_move_scale_kernel<cpu_isa_t::sse41>(interJcp));
        } else {
            THROW_ERROR << "cannot create jit eltwise kernel";
        }

        if (moveFeatureKernel && moveInteractKernel) {
            moveFeatureKernel->create_ker();
            moveInteractKernel->create_ker();
        }
    }
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


}   // namespace node
}   // namespace intel_cpu
}   // namespace ov