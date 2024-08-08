// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "emitters/plugin/x64/jit_emitter.hpp"

#include <cpu/x64/matmul/brgemm_matmul_copy_utils.hpp>

#include "emitters/plugin/x64/jit_emitter.hpp"
#include "transformations/snippets/x64/op/brgemm_copy_b.hpp"

namespace ov {
namespace intel_cpu {

class jit_brgemm_copy_b_emitter : public jit_emitter {
public:
    jit_brgemm_copy_b_emitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa,
                              const ov::snippets::lowered::ExpressionPtr& expr);

    size_t get_inputs_num() const override {return 1;}
    static std::set<std::vector<element::Type>> get_supported_precisions(const std::shared_ptr<ov::Node>& node = nullptr) {
        return {{element::i8}, {element::bf16}, {element::f32}};
    }

private:
    void validate_arguments(const std::vector<size_t> &in, const std::vector<size_t> &out) const override;
    void emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const override;

    void init_brgemm_copy(std::unique_ptr<dnnl::impl::cpu::x64::matmul::jit_brgemm_matmul_copy_b_t>& kernel,
                          size_t N, size_t N_blk, size_t N_tail, size_t out_leading_dim, size_t K_blk, brgemm_utils::BRGEMM_TYPE brgemm_type,
                          const ov::element::Type& dt_in0, const ov::element::Type& dt_in1, size_t wei_stride, dnnl_format_tag_t format) const;
    void emit_kernel_call(const dnnl::impl::cpu::x64::matmul::jit_brgemm_matmul_copy_b_t* kernel,
                          Xbyak::Reg64 src, Xbyak::Reg64 dst, Xbyak::Reg64 comp, size_t N, size_t K,
                          size_t offset_in, size_t offset_out, size_t offset_comp) const;

    static void execute(dnnl::impl::cpu::x64::matmul::jit_brgemm_matmul_copy_b_t* kernel,
                        const void* src, const void* dst, const void* comp, size_t N, size_t K);

    std::unique_ptr<dnnl::impl::cpu::x64::matmul::jit_brgemm_matmul_copy_b_t> m_kernel;
    ov::element::Type m_brg_weight_etype;

    // Block size which is set by snippets: it is usually shared between brgemm and brgemm_copy_b nodes
    size_t m_N_blk = 0lu;
    // Block size which is used by the internal OneDNN implementation.
    // It is used in snippets emitter to iterate through input/output data and call OneDNN kernel
    size_t m_inner_N_block = 0lu;
    size_t m_inner_N_tail = 0lu;

    size_t m_K = 0lu;
    size_t m_K_blk = 0lu;
    size_t m_brgemmVNNIFactor = 0lu;

    size_t m_in_offset = 0lu;
    size_t m_out_offset = 0lu;
    size_t m_comp_offset = 0lu;

    bool m_with_comp = false;
    bool m_transpose = false;

#ifdef SNIPPETS_DEBUG_CAPS
    friend std::string init_info_jit_brgemm_copy_b_emitter(const jit_brgemm_copy_b_emitter *emitter);
#endif
};

}   // namespace intel_cpu
}   // namespace ov