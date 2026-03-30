// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <xbyak/xbyak.h>

#include <cpu/x64/cpu_isa_traits.hpp>
#include <cstddef>
#include <memory>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "emitters/plugin/x64/jit_emitter.hpp"
#include "jit_kernel_base.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov::intel_cpu::kernel {

struct jit_gdn_compile_params {
    ov::element::Type data_prc = ov::element::f32;
    size_t qk_head_size = 0;
    bool fuse_qk_l2norm = false;
    float q_l2_norm_eps = 1e-6F;
    float k_l2_norm_eps = 1e-6F;
    float q_scale = 1.0F;
};

struct jit_gdn_call_args {
    uint8_t* state;
    const uint8_t* key_seq;
    const uint8_t* query_seq;
    const float* value_seq;
    const float* gate_seq;
    const float* beta_seq;
    size_t t_size;
    size_t key_query_stride;
    size_t gate_beta_stride;
    size_t value_stride;
    size_t output_stride;
    uint8_t* key_tmp;
    uint8_t* query_tmp;
    float* output_seq;
};

template <dnnl::impl::cpu::x64::cpu_isa_t isa>
struct jit_gdn_kernel : public JitKernel<jit_gdn_compile_params, jit_gdn_call_args> {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_gdn_kernel)

    explicit jit_gdn_kernel(const jit_gdn_compile_params& jcp) : JitKernel(jit_name(), jcp, isa) {}

private:
    using Xmm = Xbyak::Xmm;
    using Vmm = std::conditional_t<isa == dnnl::impl::cpu::x64::avx2, Xbyak::Ymm, Xbyak::Zmm>;

    static constexpr size_t vec_size = dnnl::impl::cpu::x64::cpu_isa_traits_t<isa>::vlen / sizeof(float);
    static constexpr size_t vec_bytes = vec_size * sizeof(float);
    static constexpr int vec_shift = isa == dnnl::impl::cpu::x64::avx2 ? 3 : 4;

    // GPR map
    const Xbyak::Reg64 reg_args = rbx;
    const Xbyak::Reg64 reg_state = r8;
    const Xbyak::Reg64 reg_key_tmp = r9;
    const Xbyak::Reg64 reg_query_tmp = r10;
    const Xbyak::Reg64 reg_t = r12;
    const Xbyak::Reg64 reg_key_seq = r13;
    const Xbyak::Reg64 reg_query_seq = r14;
    const Xbyak::Reg64 reg_value_seq = r15;
    const Xbyak::Reg64 reg_aux = r11;
    const Xbyak::Reg64 reg_gate_seq = rsi;
    const Xbyak::Reg64 reg_beta_seq = rdi;
    const Xbyak::Reg64 reg_out_seq = rbp;
    const Xbyak::Reg64 reg_aux2 = rax;

    // XMM map
    const Xmm x_hk = Xmm(0);
    const Xmm x_tmp0 = Xmm(1);
    const Xmm x_tmp1 = Xmm(2);
    const Xmm x_delta = Xmm(3);
    const Xmm x_out = Xmm(4);
    const Xmm x_gate = Xmm(5);
    const Xmm x_beta = Xmm(6);
    const Xmm x_value = Xmm(7);
    const Xmm x_eps_k = Xmm(8);
    const Xmm x_eps_q = Xmm(9);
    const Xmm x_qscale = Xmm(10);

    const Vmm v_tmp0 = Vmm(x_tmp0.getIdx());
    const Vmm v_tmp1 = Vmm(x_tmp1.getIdx());
    const Vmm v_aux0 = Vmm(11);
    const Vmm v_aux1 = Vmm(12);
    const Vmm v_aux2 = Vmm(13);

    void generate() override;
    void load(const Vmm& vmm_dst,
              const Xbyak::Reg64& reg_src,
              ov::element::Type src_prc,
              const int& elt_num,
              bool fill,
              size_t offset = 0);
    void store(const Xbyak::Reg64& reg_dst,
               const Vmm& vmm_src,
               ov::element::Type dst_prc,
               const int& elt_num,
               size_t offset = 0);
    void reduce_zmm_f32_to_xmm_scalar(const Xbyak::Zmm& zmm_src, const Xbyak::Xmm& xmm_dst);
    void dot_product_scalar(const Xbyak::Xmm& xmm_dst,
                            const Xbyak::Reg64& reg_a,
                            const Xbyak::Reg64& reg_b,
                            size_t tail_count,
                            size_t base_off,
                            size_t elem_size);
    void dot_product_to_scalar(const Xbyak::Xmm& xmm_dst,
                               const Xbyak::Reg64& reg_a,
                               const Xbyak::Reg64& reg_b,
                               const Xbyak::Reg64& reg_aux);
    void multiply_scalar(const Xbyak::Reg64& reg_vec, const Xbyak::Xmm& xmm_scalar);
    void l2norm_inplace(const Xbyak::Reg64& reg_vec, const Xbyak::Xmm& xmm_eps,
                        const Xbyak::Xmm& xmm_tmp0,
                        const Xbyak::Xmm& xmm_tmp1,
                        const Xbyak::Xmm& xmm_sum);

    std::unordered_map<size_t, std::unique_ptr<jit_emitter>> emitters;
    const std::vector<size_t> pool_aux_gpr_idxs = {};
    const std::vector<size_t> pool_aux_vmm_idxs = {};
};

std::shared_ptr<JitKernelBase> create_gdn_jit_kernel(ov::element::Type data_prc = ov::element::f32,
                                                     size_t qk_head_size = 0,
                                                     bool fuse_qk_l2norm = false,
                                                     float q_l2_norm_eps = 1e-6F,
                                                     float k_l2_norm_eps = 1e-6F);

}  // namespace ov::intel_cpu::kernel
