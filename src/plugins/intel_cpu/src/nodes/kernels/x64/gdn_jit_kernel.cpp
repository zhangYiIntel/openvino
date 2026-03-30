// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gdn_jit_kernel.hpp"

#include <common/utils.hpp>
#include <cpu/x64/injectors/jit_uni_eltwise_injector.hpp>
#include <type_traits>

#include "emitters/plugin/x64/jit_load_store_emitters.hpp"

using namespace dnnl::impl::cpu;
using namespace dnnl::impl::cpu::x64;

namespace ov::intel_cpu::kernel {

#define GET_OFF(field) offsetof(jit_gdn_call_args, field)

template <cpu_isa_t isa>
void jit_gdn_kernel<isa>::load(const Vmm& vmm_dst,
                               const Xbyak::Reg64& reg_src,
                               ov::element::Type src_prc,
                               const int& elt_num,
                               bool fill,
                               size_t offset) {
    // Typed load helper (src_prc -> f32 VMM via jit emitter)
    const auto seed = load_emitter_params(src_prc, ov::element::f32, elt_num, fill, "float_min").hash();
    if (!emitters[seed]) {
        emitters[seed] = std::make_unique<jit_load_emitter>(this,
                                                            isa,
                                                            src_prc,
                                                            ov::element::f32,
                                                            elt_num,
                                                            ov::element::f32,
                                                            fill,
                                                            "float_min");
    }
    emitters[seed]->emit_code({static_cast<size_t>(reg_src.getIdx()), offset},
                              {static_cast<size_t>(vmm_dst.getIdx())},
                              pool_aux_vmm_idxs,
                              pool_aux_gpr_idxs);
}

template <cpu_isa_t isa>
void jit_gdn_kernel<isa>::store(const Xbyak::Reg64& reg_dst,
                                const Vmm& vmm_src,
                                ov::element::Type dst_prc,
                                const int& elt_num,
                                size_t offset) {
    // Typed store helper (f32 VMM -> dst_prc via jit emitter)
    const auto seed = store_emitter_params(ov::element::f32, dst_prc, elt_num).hash();
    if (!emitters[seed]) {
        emitters[seed] = std::make_unique<jit_store_emitter>(this, isa, ov::element::f32, dst_prc, elt_num);
    }
    emitters[seed]->emit_code({static_cast<size_t>(vmm_src.getIdx())},
                              {static_cast<size_t>(reg_dst.getIdx()), offset},
                              pool_aux_vmm_idxs,
                              pool_aux_gpr_idxs);
}

template <cpu_isa_t isa>
void jit_gdn_kernel<isa>::reduce_zmm_f32_to_xmm_scalar(const Xbyak::Zmm& zmm_src, const Xbyak::Xmm& xmm_dst) {
    // Horizontal reduce 16x f32 (ZMM) into scalar lane of xmm_dst
    vextractf32x8(Xbyak::Ymm(x_tmp1.getIdx()), zmm_src, 1);
    vaddps(Xbyak::Ymm(x_tmp0.getIdx()), Xbyak::Ymm(zmm_src.getIdx()), Xbyak::Ymm(x_tmp1.getIdx()));
    vextractf128(x_tmp1, Xbyak::Ymm(x_tmp0.getIdx()), 1);
    vaddps(x_tmp0, x_tmp0, x_tmp1);
    vhaddps(x_tmp0, x_tmp0, x_tmp0);
    vhaddps(x_tmp0, x_tmp0, x_tmp0);
    vaddss(xmm_dst, xmm_dst, x_tmp0);
}

template <cpu_isa_t isa>
void jit_gdn_kernel<isa>::dot_product_scalar(const Xbyak::Xmm& xmm_dst,
                                             const Xbyak::Reg64& reg_a,
                                             const Xbyak::Reg64& reg_b,
                                             size_t tail_count,
                                             size_t base_off,
                                             size_t elem_size) {
    // Scalar tail dot-product accumulation into xmm_dst
    for (size_t i = 0; i < tail_count; i++) {
        const size_t off = base_off + i * elem_size;
        load(v_tmp0, reg_a, m_jcp.data_prc, 1, false, off);
        load(v_tmp1, reg_b, m_jcp.data_prc, 1, false, off);
        vmulss(x_tmp0, x_tmp0, x_tmp1);
        vaddss(xmm_dst, xmm_dst, x_tmp0);
    }
}

template <cpu_isa_t isa>
void jit_gdn_kernel<isa>::dot_product_to_scalar(const Xbyak::Xmm& xmm_dst,
                                                const Xbyak::Reg64& reg_a,
                                                const Xbyak::Reg64& reg_b,
                                                const Xbyak::Reg64& reg_aux) {
    // Dot product dispatcher (bf16/f16/f32/small fallback) -> scalar xmm_dst
    uni_vpxor(xmm_dst, xmm_dst, xmm_dst);
    const size_t qk = m_jcp.qk_head_size;

    if (m_jcp.data_prc == ov::element::bf16 && mayiuse(avx512_core_bf16)) {
        // bf16 fast path (vdpbf16ps on 32 bf16 elems per vector)
        const size_t vec_elems = 32;
        const size_t vec_cnt = qk / vec_elems;
        const size_t tail = qk % vec_elems;

        uni_vpxor(v_aux0, v_aux0, v_aux0);

        for (size_t i = 0; i < vec_cnt; i++) {
            const size_t off = i * 64;
            vmovups(v_aux1, ptr[reg_a + off]);
            vmovups(v_aux2, ptr[reg_b + off]);
            vdpbf16ps(v_aux0, v_aux1, v_aux2);
        }

        reduce_zmm_f32_to_xmm_scalar(Xbyak::Zmm(v_aux0.getIdx()), xmm_dst);

        dot_product_scalar(xmm_dst, reg_a, reg_b, tail, vec_cnt * 64, sizeof(uint16_t));
    } else if (m_jcp.data_prc == ov::element::f16 && mayiuse(avx512_core_fp16)) {
        // f16 fast path (vfmadd231ph + fp16->fp32 accumulation)
        const size_t vec_elems = 32;
        const size_t vec_cnt = qk / vec_elems;
        const size_t tail = qk % vec_elems;

        uni_vpxor(v_aux0, v_aux0, v_aux0);

        for (size_t i = 0; i < vec_cnt; i++) {
            const size_t off = i * 64;
            vmovups(v_aux1, ptr[reg_a + off]);
            vmovups(v_aux2, ptr[reg_b + off]);
            uni_vpxor(v_tmp0, v_tmp0, v_tmp0);
            vfmadd231ph(v_tmp0, v_aux1, v_aux2);

            vextractf32x8(Xbyak::Ymm(v_aux2.getIdx()), Xbyak::Zmm(v_tmp0.getIdx()), 0);
            vcvtph2ps(v_aux1, Xbyak::Ymm(v_aux2.getIdx()));
            vaddps(v_aux0, v_aux0, v_aux1);

            vextractf32x8(Xbyak::Ymm(v_aux2.getIdx()), Xbyak::Zmm(v_tmp0.getIdx()), 1);
            vcvtph2ps(v_aux1, Xbyak::Ymm(v_aux2.getIdx()));
            vaddps(v_aux0, v_aux0, v_aux1);
        }

        reduce_zmm_f32_to_xmm_scalar(Xbyak::Zmm(v_aux0.getIdx()), xmm_dst);

        dot_product_scalar(xmm_dst, reg_a, reg_b, tail, vec_cnt * 64, sizeof(uint16_t));
    } else {
        // generic path (f32 vectorized, otherwise scalar)
        if (m_jcp.data_prc == ov::element::f32) {
            const size_t vec_cnt = qk / vec_size;
            const size_t tail = qk % vec_size;

            uni_vpxor(v_aux0, v_aux0, v_aux0);

            for (size_t i = 0; i < vec_cnt; i++) {
                const size_t off = i * vec_bytes;
                load(v_aux1, reg_a, ov::element::f32, static_cast<int>(vec_size), false, off);
                load(v_aux2, reg_b, ov::element::f32, static_cast<int>(vec_size), false, off);
                vfmadd231ps(v_aux0, v_aux1, v_aux2);
            }

            if constexpr (std::is_same_v<Vmm, Xbyak::Ymm>) {
                vextractf128(x_tmp0, v_aux0, 1);
                vaddps(Xbyak::Xmm(v_aux0.getIdx()), Xbyak::Xmm(v_aux0.getIdx()), x_tmp0);
                vhaddps(Xbyak::Xmm(v_aux0.getIdx()), Xbyak::Xmm(v_aux0.getIdx()), Xbyak::Xmm(v_aux0.getIdx()));
                vhaddps(Xbyak::Xmm(v_aux0.getIdx()), Xbyak::Xmm(v_aux0.getIdx()), Xbyak::Xmm(v_aux0.getIdx()));
                vaddss(xmm_dst, xmm_dst, Xbyak::Xmm(v_aux0.getIdx()));
            } else {
                reduce_zmm_f32_to_xmm_scalar(Xbyak::Zmm(v_aux0.getIdx()), xmm_dst);
            }

            dot_product_scalar(xmm_dst, reg_a, reg_b, tail, vec_cnt * vec_bytes, sizeof(float));
        } else {
            dot_product_scalar(xmm_dst, reg_a, reg_b, qk, 0, m_jcp.data_prc.size());
        }
    }
}

template <cpu_isa_t isa>
void jit_gdn_kernel<isa>::multiply_scalar(const Xbyak::Reg64& reg_vec, const Xbyak::Xmm& xmm_scalar) {
    // In-place vector scale: reg_vec[i] *= xmm_scalar
    const int elt_num = static_cast<int>(vec_size);
    const size_t elem_size = m_jcp.data_prc.size();
    const size_t step = static_cast<size_t>(elt_num) * elem_size;
    const size_t vec_cnt = m_jcp.qk_head_size / static_cast<size_t>(elt_num);
    const size_t tail = m_jcp.qk_head_size % static_cast<size_t>(elt_num);

    vbroadcastss(v_tmp1, xmm_scalar);

    for (size_t i = 0; i < vec_cnt; i++) {
        const size_t off = i * step;
        load(v_tmp0, reg_vec, m_jcp.data_prc, elt_num, false, off);
        vmulps(v_tmp0, v_tmp0, v_tmp1);
        store(reg_vec, v_tmp0, m_jcp.data_prc, elt_num, off);
    }
    for (size_t i = 0; i < tail; i++) {
        const size_t off = vec_cnt * step + i * elem_size;
        load(v_tmp0, reg_vec, m_jcp.data_prc, 1, false, off);
        vmulss(x_tmp0, x_tmp0, xmm_scalar);
        store(reg_vec, v_tmp0, m_jcp.data_prc, 1, off);
    }
}

template <cpu_isa_t isa>
void jit_gdn_kernel<isa>::l2norm_inplace(const Xbyak::Reg64& reg_vec,
                                         const Xbyak::Xmm& xmm_eps,
                                         const Xbyak::Xmm& xmm_tmp0,
                                         const Xbyak::Xmm& xmm_tmp1,
                                         const Xbyak::Xmm& xmm_sum) {
    // In-place L2 normalization over one head vector (with eps)
    const size_t qk = m_jcp.qk_head_size;
    const size_t vec_cnt = qk / vec_size;
    const size_t tail = qk % vec_size;

    uni_vpxor(xmm_sum, xmm_sum, xmm_sum);

    uni_vpxor(v_aux0, v_aux0, v_aux0);

    for (size_t i = 0; i < vec_cnt; i++) {
        const size_t off = i * vec_bytes;
        load(v_aux1, reg_vec, ov::element::f32, static_cast<int>(vec_size), false, off);
        vfmadd231ps(v_aux0, v_aux1, v_aux1);
    }

    if constexpr (std::is_same_v<Vmm, Xbyak::Ymm>) {
        vextractf128(x_tmp1, v_aux0, 1);
        vaddps(xmm_sum, xmm_sum, x_tmp1);
        vhaddps(xmm_sum, xmm_sum, xmm_sum);
        vhaddps(xmm_sum, xmm_sum, xmm_sum);
    } else {
        reduce_zmm_f32_to_xmm_scalar(Xbyak::Zmm(v_aux0.getIdx()), xmm_sum);
    }

    for (size_t i = 0; i < tail; i++) {
        const size_t off = vec_cnt * vec_bytes + i * sizeof(float);
        vmovss(xmm_tmp0, ptr[reg_vec + off]);
        vmulss(xmm_tmp0, xmm_tmp0, xmm_tmp0);
        vaddss(xmm_sum, xmm_sum, xmm_tmp0);
    }

    vaddss(xmm_sum, xmm_sum, xmm_eps);
    vsqrtss(xmm_sum, xmm_sum, xmm_sum);
    mov(reg_aux2.cvt32(), float2int(1.0F));
    vmovd(xmm_tmp1, reg_aux2.cvt32());
    vdivss(xmm_tmp1, xmm_tmp1, xmm_sum);

    vbroadcastss(v_tmp1, xmm_tmp1);

    for (size_t i = 0; i < vec_cnt; i++) {
        const size_t off = i * vec_bytes;
        load(v_tmp0, reg_vec, ov::element::f32, static_cast<int>(vec_size), false, off);
        vmulps(v_tmp0, v_tmp0, v_tmp1);
        store(reg_vec, v_tmp0, ov::element::f32, static_cast<int>(vec_size), off);
    }

    for (size_t i = 0; i < tail; i++) {
        const size_t off = vec_cnt * vec_bytes + i * sizeof(float);
        vmovss(xmm_tmp0, ptr[reg_vec + off]);
        vmulss(xmm_tmp0, xmm_tmp0, xmm_tmp1);
        vmovss(ptr[reg_vec + off], xmm_tmp0);
    }
}

template <cpu_isa_t isa>
void jit_gdn_kernel<isa>::generate() {
    // JIT codegen entry for one (B,H,V) sequence kernel
    auto exp_injector = std::make_shared<jit_uni_eltwise_injector_t<isa>>(this,
                                                                          dnnl::impl::alg_kind::eltwise_exp,
                                                                          0.F,
                                                                          0.F,
                                                                          1.F,
                                                                          dnnl::impl::data_type::f32,
                                                                          true,
                                                                          Xbyak::Reg64(Xbyak::Operand::RCX),
                                                                          Xbyak::Opmask(1),
                                                                          true,
                                                                          false,
                                                                          false,
                                                                          false);

    this->preamble();

    Xbyak::Label l_t_loop;
    Xbyak::Label l_end;

    mov(reg_args, abi_param1);

    mov(reg_state, ptr[reg_args + GET_OFF(state)]);
    mov(reg_key_tmp, ptr[reg_args + GET_OFF(key_tmp)]);
    mov(reg_query_tmp, ptr[reg_args + GET_OFF(query_tmp)]);
    mov(reg_key_seq, ptr[reg_args + GET_OFF(key_seq)]);
    mov(reg_query_seq, ptr[reg_args + GET_OFF(query_seq)]);
    mov(reg_value_seq, ptr[reg_args + GET_OFF(value_seq)]);
    mov(reg_gate_seq, ptr[reg_args + GET_OFF(gate_seq)]);
    mov(reg_beta_seq, ptr[reg_args + GET_OFF(beta_seq)]);
    mov(reg_out_seq, ptr[reg_args + GET_OFF(output_seq)]);
    mov(reg_t, ptr[reg_args + GET_OFF(t_size)]);

    // One-time setup before t-loop
    exp_injector->load_table_addr();

    test(reg_t, reg_t);
    jz(l_end, T_NEAR);

    L(l_t_loop);
    {
        // Per-step constants / pointers setup
        // Reload scalar constants each iteration, as helper injectors may clobber xmm regs.
        mov(reg_aux.cvt32(), float2int(m_jcp.k_l2_norm_eps));
        vmovd(x_eps_k, reg_aux.cvt32());
        mov(reg_aux.cvt32(), float2int(m_jcp.q_l2_norm_eps));
        vmovd(x_eps_q, reg_aux.cvt32());
        mov(reg_aux.cvt32(), float2int(m_jcp.q_scale));
        vmovd(x_qscale, reg_aux.cvt32());

        // Reset scratch pointers for this step before writing copied K/Q.
        mov(reg_key_tmp, ptr[reg_args + GET_OFF(key_tmp)]);
        mov(reg_query_tmp, ptr[reg_args + GET_OFF(query_tmp)]);

        // copy K/Q for current t into scratch
        const int copy_elt_num = static_cast<int>(vec_size);
        const size_t elem_size = m_jcp.data_prc.size();
        const size_t copy_step = static_cast<size_t>(copy_elt_num) * elem_size;
        const size_t vec_cnt = m_jcp.qk_head_size / static_cast<size_t>(copy_elt_num);
        const size_t tail = m_jcp.qk_head_size % static_cast<size_t>(copy_elt_num);

        for (size_t i = 0; i < vec_cnt; i++) {
            const size_t off = i * copy_step;
            load(v_tmp0, reg_key_seq, m_jcp.data_prc, copy_elt_num, false, off);
            store(reg_key_tmp, v_tmp0, m_jcp.data_prc, copy_elt_num, off);
            load(v_tmp1, reg_query_seq, m_jcp.data_prc, copy_elt_num, false, off);
            store(reg_query_tmp, v_tmp1, m_jcp.data_prc, copy_elt_num, off);
        }
        for (size_t i = 0; i < tail; i++) {
            const size_t off = vec_cnt * copy_step + i * elem_size;
            load(v_tmp0, reg_key_seq, m_jcp.data_prc, 1, false, off);
            store(reg_key_tmp, v_tmp0, m_jcp.data_prc, 1, off);
            load(v_tmp1, reg_query_seq, m_jcp.data_prc, 1, false, off);
            store(reg_query_tmp, v_tmp1, m_jcp.data_prc, 1, off);
        }
        add(reg_key_seq, m_jcp.qk_head_size * elem_size);
        add(reg_query_seq, m_jcp.qk_head_size * elem_size);

        // restore scratch pointers for this step
        mov(reg_key_tmp, ptr[reg_args + GET_OFF(key_tmp)]);
        mov(reg_query_tmp, ptr[reg_args + GET_OFF(query_tmp)]);

        if (m_jcp.fuse_qk_l2norm) {
            // Optional fused L2 norm for key/query
            l2norm_inplace(reg_key_tmp, x_eps_k, x_tmp0, x_tmp1, x_hk);
            l2norm_inplace(reg_query_tmp, x_eps_q, x_tmp0, x_tmp1, x_hk);
        }

        // query *= q_scale
        multiply_scalar(reg_query_tmp, x_qscale);

        // h *= exp(gate)
        uni_vpxor(x_gate, x_gate, x_gate);
        vmovss(x_gate, ptr[reg_gate_seq]);
        exp_injector->compute_vector_range(x_gate.getIdx(), x_gate.getIdx() + 1);
        multiply_scalar(reg_state, x_gate);

        // hk = dot(h, k_tmp)
        mov(reg_query_tmp, reg_state);
        mov(reg_key_tmp, ptr[reg_args + GET_OFF(key_tmp)]);
        dot_product_to_scalar(x_hk, reg_query_tmp, reg_key_tmp, reg_aux);

        // delta = (v - hk) * beta
        vmovss(x_value, ptr[reg_value_seq]);
        vmovss(x_beta, ptr[reg_beta_seq]);
        vsubss(x_delta, x_value, x_hk);
        vmulss(x_delta, x_delta, x_beta);

        // h += k_tmp * delta
        {
            const int update_elt_num = static_cast<int>(vec_size);
            const size_t elem_size = m_jcp.data_prc.size();
            const size_t update_step = static_cast<size_t>(update_elt_num) * elem_size;
            const size_t vec_cnt = m_jcp.qk_head_size / static_cast<size_t>(update_elt_num);
            const size_t tail = m_jcp.qk_head_size % static_cast<size_t>(update_elt_num);

            vbroadcastss(v_aux2, x_delta);

            mov(reg_aux2, ptr[reg_args + GET_OFF(key_tmp)]);
            for (size_t i = 0; i < vec_cnt; i++) {
                const size_t off = i * update_step;
                load(v_tmp0, reg_state, m_jcp.data_prc, update_elt_num, false, off);
                load(v_tmp1, reg_aux2, m_jcp.data_prc, update_elt_num, false, off);
                vfmadd231ps(v_tmp0, v_tmp1, v_aux2);
                store(reg_state, v_tmp0, m_jcp.data_prc, update_elt_num, off);
            }
            for (size_t i = 0; i < tail; i++) {
                const size_t off = vec_cnt * update_step + i * elem_size;
                load(v_tmp0, reg_state, m_jcp.data_prc, 1, false, off);
                load(v_tmp1, reg_aux2, m_jcp.data_prc, 1, false, off);
                vmulss(x_tmp1, x_tmp1, x_delta);
                vaddss(x_tmp0, x_tmp0, x_tmp1);
                store(reg_state, v_tmp0, m_jcp.data_prc, 1, off);
            }
        }

        // out = dot(h, q_tmp)
        mov(reg_query_tmp, reg_state);
        mov(reg_key_tmp, ptr[reg_args + GET_OFF(query_tmp)]);
        dot_product_to_scalar(x_out, reg_query_tmp, reg_key_tmp, reg_aux);
        vmovss(ptr[reg_out_seq], x_out);

        // advance to next t
        mov(reg_aux2, ptr[reg_args + GET_OFF(key_query_stride)]);
        sub(reg_aux2, m_jcp.qk_head_size);
        imul(reg_aux2, reg_aux2, m_jcp.data_prc.size());
        add(reg_key_seq, reg_aux2);
        add(reg_query_seq, reg_aux2);

        mov(reg_aux2, ptr[reg_args + GET_OFF(value_stride)]);
        shl(reg_aux2, 2);
        add(reg_value_seq, reg_aux2);

        mov(reg_aux2, ptr[reg_args + GET_OFF(gate_beta_stride)]);
        shl(reg_aux2, 2);
        add(reg_gate_seq, reg_aux2);
        add(reg_beta_seq, reg_aux2);

        mov(reg_aux2, ptr[reg_args + GET_OFF(output_stride)]);
        shl(reg_aux2, 2);
        add(reg_out_seq, reg_aux2);

        dec(reg_t);
        jnz(l_t_loop, T_NEAR);
    }

    L(l_end);
    this->postamble();

    exp_injector->prepare_table();
    for (const auto& emitter : emitters) {
        if (emitter.second) {
            emitter.second->emit_data();
        }
    }
}

std::shared_ptr<JitKernelBase> create_gdn_jit_kernel(ov::element::Type data_prc,
                                                     size_t qk_head_size,
                                                     bool fuse_qk_l2norm,
                                                     float q_l2_norm_eps,
                                                     float k_l2_norm_eps) {
    std::shared_ptr<JitKernelBase> res;
    jit_gdn_compile_params jcp;
    jcp.data_prc = data_prc;
    jcp.qk_head_size = qk_head_size;
    jcp.fuse_qk_l2norm = fuse_qk_l2norm;
    jcp.q_l2_norm_eps = q_l2_norm_eps;
    jcp.k_l2_norm_eps = k_l2_norm_eps;
    jcp.q_scale = 1.0F / std::sqrt(static_cast<float>(qk_head_size));

    if (data_prc != ov::element::f32 && data_prc != ov::element::bf16 && data_prc != ov::element::f16) {
        return res;
    }
    if (qk_head_size == 0) {
        return res;
    }

    if (data_prc == ov::element::bf16) {
        if (mayiuse(avx512_core_bf16)) {
            res = std::make_shared<jit_gdn_kernel<avx512_core_bf16>>(jcp);
        }
    } else if (data_prc == ov::element::f16) {
        if (mayiuse(avx512_core_fp16)) {
            res = std::make_shared<jit_gdn_kernel<avx512_core_fp16>>(jcp);
        }
    } else if (data_prc == ov::element::f32) {
        if (mayiuse(avx512_core)) {
            if ((qk_head_size % 16) == 0) {
                res = std::make_shared<jit_gdn_kernel<avx512_core>>(jcp);
            }
        } else if (mayiuse(avx2)) {
            if ((qk_head_size % 8) == 0) {
                res = std::make_shared<jit_gdn_kernel<avx2>>(jcp);
            }
        }
    }

    if (res) {
        res->create_kernel();
    }

    return res;
}

template struct jit_gdn_kernel<avx2>;
template struct jit_gdn_kernel<avx512_core>;
template struct jit_gdn_kernel<avx512_core_bf16>;
template struct jit_gdn_kernel<avx512_core_fp16>;

}  // namespace ov::intel_cpu::kernel
