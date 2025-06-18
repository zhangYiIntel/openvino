// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "rope_kernel.hpp"

#include <memory>

using namespace dnnl::impl::cpu::x64;

namespace ov::intel_cpu::kernel {

#define GET_OFF(field) offsetof(jit_rotary_call_args, field)

template <cpu_isa_t isa>
void jit_rotary_kernel<isa>::generate() {
    this->preamble();
    mov(reg_src, ptr[abi_param1 + GET_OFF(src)]);
    mov(reg_cos, ptr[abi_param1 + GET_OFF(cos)]);
    mov(reg_sin, ptr[abi_param1 + GET_OFF(sin)]);
    mov(reg_dst, ptr[abi_param1 + GET_OFF(dst)]);
    mov(reg_table, l_table);
    uni_vpxor(vmm_src0, vmm_src0, vmm_src0);
    uni_vpxor(vmm_src1, vmm_src1, vmm_src1);
    uni_vpxor(vmm_cos, vmm_cos, vmm_cos);
    uni_vpxor(vmm_sin, vmm_sin, vmm_sin);
    if (m_jcp.interleave) {
        // dst: 0-2 4-6 8-10 12-14 16-18 20-22 24-26 28-30 ->
        // lower 64bit/128 lane
        //      0-2        4-6        8-10       12-14
        // higher 64bit/128 lane
        //           16-18      20-22      24-26       28-30
        static const uint64_t mask_zmm[] = {0, 4, 1, 5, 2, 6, 3, 7};
        if (isa == cpu_isa_t::avx512_core) {
            mov(reg_tmp, reinterpret_cast<uintptr_t>(mask_zmm));
            uni_vmovups(vmm_idx, ptr[reg_tmp]);
        }
        auto half_rotary_ndims = m_jcp.rotary_ndims / 2;
        for (size_t i = 0; i < half_rotary_ndims / vec_size; i++) {
            rotary_interleave(vec_size);
        }
    } else {
        auto half_rotary_ndims = m_jcp.rotary_ndims / 2;
        size_t steps = 0;
        if (m_jcp.do_quantize) {
            float max = -std::numeric_limits<float>::max();
            float min = std::numeric_limits<float>::max();
            int3();
            uni_vmovups(vmm_max, table_val(0));
            uni_vmovups(vmm_min, table_val(1));
        }

        for (size_t i = 0; i < half_rotary_ndims / vec_size; i++) {
            rotary_half(vec_size, vmm_max, vmm_min);
            steps += vec_size;
        }

        if (half_rotary_ndims % vec_size != 0) {
            rotary_half(half_rotary_ndims % vec_size, vmm_max, vmm_min);
            steps += half_rotary_ndims % vec_size;
        }

        printf("can_inplace %d rotary_ndims %d head_size %d\n", m_jcp.can_inplace, m_jcp.rotary_ndims, m_jcp.head_size);
        if (!m_jcp.can_inplace && m_jcp.rotary_ndims != m_jcp.head_size) {
            for (size_t i = m_jcp.rotary_ndims; i < m_jcp.head_size / vec_size; i++) {
                data_copy(vec_size);
                steps += vec_size;
            }
            if (m_jcp.head_size % vec_size != 0) {
                data_copy(m_jcp.head_size % vec_size);
                steps += m_jcp.head_size % vec_size;
            }
        }

        if (isa == cpu_isa_t::avx512_core && m_jcp.do_quantize) {
            reduce_to_scalar(vmm_max, true);
            reduce_to_scalar(vmm_min, false);
            int3();
            auto xmm_max = Xbyak::Xmm(vmm_max.getIdx());
            auto xmm_min = Xbyak::Xmm(vmm_min.getIdx());
            auto xmm_aux = Xbyak::Xmm(vmm_aux.getIdx());
            //and with abs mask
            uni_vandps(xmm_max, xmm_max, table_val(2));
            uni_vandps(xmm_min, xmm_min, table_val(2));

            uni_vmaxps(xmm_max, xmm_max, xmm_min);
            uni_vdivps(xmm_max, xmm_max, table_val(3));
        }
    }
    this->postamble();
    for (const auto& emitter : emitters) {
        if (emitter.second) {
            emitter.second->emit_data();
        }
    }
    prepare_table();
}

template <cpu_isa_t isa>
void jit_rotary_kernel<isa>::rotary_half(size_t step, const Vmm& vmm_max, const Vmm& vmm_min) {
    // for (; i < half_rotary_dims; i++) {
    //     auto src0 = src[i];
    //     auto src1 = src[i + half_rotary_dims];
    //     dst[i] = cos[i] * src0 - sin[i] * src1;
    //     dst[i + half_rotary_dims] = cos[i + half_rotary_dims] * src1 + sin[i + half_rotary_dims] * src0;
    // }
    auto half_rotary_ndims = m_jcp.rotary_ndims / 2;
    // src0: src[i]
    load(vmm_src0, reg_src, m_jcp.src_prc, step, false);
    // src1: src[i + halfRotaryNdims]
    load(vmm_src1, reg_src, m_jcp.src_prc, step, false, half_rotary_ndims * m_jcp.src_prc.size());
    // cos[i]
    load(vmm_cos, reg_cos, ov::element::f32, step, false);
    // sin[i]
    load(vmm_sin, reg_sin, ov::element::f32, step, false);
    // sin[i] * src1
    uni_vmulps(vmm_dst0, vmm_sin, vmm_src1);
    // cos[i] * src0 - sin[i] * src1
    vfmsub231ps(vmm_dst0, vmm_cos, vmm_src0);
    if (m_jcp.do_quantize) {
        if (vec_size > step) {
            fill_tail(vmm_dst0, step, -std::numeric_limits<float>::max());
        }
        uni_vmaxps(vmm_max, vmm_max, vmm_dst0);
        if (vec_size > step) {
            fill_tail(vmm_dst0, step, std::numeric_limits<float>::max());
        }
        uni_vminps(vmm_min, vmm_min, vmm_dst0);
    }
    store(reg_dst, vmm_dst0, m_jcp.dst_prc, step);

    // cos[i + halfRotaryNdims]
    load(vmm_cos, reg_cos, ov::element::f32, step, false, half_rotary_ndims * sizeof(float));
    // sin[i + halfRotaryNdims]
    load(vmm_sin, reg_sin, ov::element::f32, step, false, half_rotary_ndims * sizeof(float));
    // cos[i + half_rotary_dims] * src1
    uni_vmulps(vmm_dst0, vmm_cos, vmm_src1);
    // cos[i + half_rotary_dims] * src1 + sin[i + half_rotary_dims] * src0
    vfmadd231ps(vmm_dst0, vmm_sin, vmm_src0);
    if (m_jcp.do_quantize) {
        if (vec_size > step) {
            fill_tail(vmm_dst0, step, -std::numeric_limits<float>::max());
        }
        uni_vmaxps(vmm_max, vmm_max, vmm_dst0);
        if (vec_size > step) {
            fill_tail(vmm_dst0, step, std::numeric_limits<float>::max());
        }
        uni_vminps(vmm_min, vmm_min, vmm_dst0);
    }
    store(reg_dst, vmm_dst0, m_jcp.dst_prc, step, half_rotary_ndims * m_jcp.dst_prc.size());

    add(reg_src, m_jcp.src_prc.size() * step);
    add(reg_dst, m_jcp.dst_prc.size() * step);
    add(reg_cos, sizeof(float) * step);
    add(reg_sin, sizeof(float) * step);
}

template <cpu_isa_t isa>
void jit_rotary_kernel<isa>::rotary_interleave(size_t step) {
    // for (size_t j = 0; i < rotary_dims; i += 2, j++) {
    //     dst[i] = cos[j] * x[i] - sin[j] * x[i + 1];
    //     dst[i + 1] = cos[j] * x[i + 1] + sin[j] * x[i];
    // }
    load(vmm_src0, reg_src, m_jcp.src_prc, step, false);
    load(vmm_src1, reg_src, m_jcp.src_prc, step, false, step * m_jcp.src_prc.size());
    auto deinterlace = [&](const Vmm& src0, const Vmm& src1, const Vmm& tmp0, const Vmm& tmp1) {
        if (isa == cpu_isa_t::avx2) {
            // src0: 0 1  2  3  4  5  6  7
            // src1: 8 9 10 11 12 13 14 15
            // 0 1 2 3  8  9 10 11
            vperm2i128(tmp0, src0, src1, 0x20);
            // 4 5 6 7 12 13 14 15
            vperm2i128(tmp1, src0, src1, 0x31);
            // src0 x[i]:     0 2 4 6 8 10 12 14
            vshufps(src0, tmp0, tmp1, 0x88);
            // src1 x[i + 1]: 1 3 5 7 9 11 13 15
            vshufps(src1, tmp0, tmp1, 0xdd);
        } else {
            // src0: 0   1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
            // src1: 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
            // 0 1 2 3  8  9 10 11 16 17 18 19 24 25 26 27
            vshuff32x4(tmp0, src0, src1, 0x88);
            // 4 5 6 7 12 13 14 15 20 21 22 23 28 29 30 31
            vshuff32x4(tmp1, src0, src1, 0xdd);
            // src0 x[i]:     0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30
            vshufps(src0, tmp0, tmp1, 0x88);
            // src1 x[i + 1]: 1 3 5 7 9 11 13 15 17 19 21 23 25 27 29 31
            vshufps(src1, tmp0, tmp1, 0xdd);
        }
    };
    deinterlace(vmm_src0, vmm_src1, vmm_dst0, vmm_dst1);
    // cos[j]
    load(vmm_cos, reg_cos, ov::element::f32, step, false);
    // sin[j]
    if (m_jcp.mix_cos_sin) {
        load(vmm_sin, reg_cos, ov::element::f32, step, false, step * sizeof(float));
        deinterlace(vmm_cos, vmm_sin, vmm_dst0, vmm_dst1);
    } else {
        load(vmm_sin, reg_sin, ov::element::f32, step, false);
    }
    // sin[j] * src1
    uni_vmulps(vmm_dst0, vmm_sin, vmm_src1);
    // cos[j] * src0 - sin[j] * src1
    vfmsub231ps(vmm_dst0, vmm_cos, vmm_src0);

    // cos[j] * src1
    uni_vmulps(vmm_dst1, vmm_cos, vmm_src1);
    // cos[j] * src1 + sin[j] * src0
    vfmadd231ps(vmm_dst1, vmm_sin, vmm_src0);
    if (isa == cpu_isa_t::avx2) {
        // dst0: 0 2 4 6 8 10 12 14
        // dst1: 1 3 5 7 9 11 13 15
        // 0 1 2 3  8  9 10 11
        vunpcklps(vmm_cos, vmm_dst0, vmm_dst1);
        // 4 5 6 7 12 13 14 15
        vunpckhps(vmm_sin, vmm_dst0, vmm_dst1);
        // 0 1  2  3  4  5  6  7
        vperm2i128(vmm_dst0, vmm_cos, vmm_sin, 0x20);
        // 8 9 10 11 12 13 14 15
        vperm2i128(vmm_dst1, vmm_cos, vmm_sin, 0x31);
    } else {
        // dst0: 0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30
        // dst1: 1 3 5 7 9 11 13 15 17 19 21 23 25 27 29 31
        // 0 2 16 18 4 6 20 22 8 10 24 26 12 14 28 30
        vpermq(vmm_cos, vmm_idx, vmm_dst0);
        // 1 3 17 19 5 7 21 23 9 11 25 27 13 15 29 31
        vpermq(vmm_sin, vmm_idx, vmm_dst1);
        // 0   1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
        vunpcklps(vmm_dst0, vmm_cos, vmm_sin);
        // 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
        vunpckhps(vmm_dst1, vmm_cos, vmm_sin);
    }
    store(reg_dst, vmm_dst0, m_jcp.dst_prc, step);
    store(reg_dst, vmm_dst1, m_jcp.dst_prc, step, step * m_jcp.dst_prc.size());
    add(reg_src, m_jcp.src_prc.size() * step * 2);
    add(reg_dst, m_jcp.dst_prc.size() * step * 2);
    if (m_jcp.mix_cos_sin) {
        add(reg_cos, 2 * sizeof(float) * step);
    } else {
        add(reg_cos, sizeof(float) * step);
        add(reg_sin, sizeof(float) * step);
    }
}

template <cpu_isa_t isa>
void jit_rotary_kernel<isa>::load(const Vmm& vmm_dst,
                                  const Xbyak::Reg64& reg_src,
                                  ov::element::Type src_prc,
                                  const int& elt_num,
                                  bool fill,
                                  size_t offset) {
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
void jit_rotary_kernel<isa>::store(const Xbyak::Reg64& reg_dst,
                                   const Vmm& vmm_src,
                                   ov::element::Type dst_prc,
                                   const int& elt_num,
                                   size_t offset) {
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
void jit_rotary_kernel<isa>::reduce_to_scalar(const Vmm& vmm_dst, bool is_max) {
    if (isa == cpu_isa_t::avx512_core) {
        printf("Reudce max/min\n");
        // int3();
        //4 group each has 4 float
        //0|1|2|3
        vshuff32x4(vmm_aux, vmm_dst, vmm_dst, 0x4E);
        //data in aux
        //2|3|1|0
        if (is_max)
            uni_vmaxps(vmm_dst, vmm_dst, vmm_aux);
        else
            uni_vminps(vmm_dst, vmm_dst, vmm_aux);
        //|max(0,2)|max(1,3)|max(2,1)|max(3,0)
        vshuff32x4(vmm_aux, vmm_dst, vmm_dst, 0xB1);
        //data in aux
        //1|0|3|2
        //|max(1,3)|max(0,2)|max(3,0)|max(2,1)
        if (is_max)
            uni_vmaxps(vmm_dst, vmm_dst, vmm_aux);
        else
            uni_vminps(vmm_dst, vmm_dst, vmm_aux);
        //|max(0-3)|max(0-3)|max(0-3)|max(0-3)
        auto xmm_max = Xbyak::Xmm(vmm_dst.getIdx());
        auto xmm_aux = Xbyak::Xmm(vmm_aux.getIdx());
        uni_vmovshdup(xmm_aux, xmm_max); // dst:1,2,3,4; aux3:2,2,4,4
        uni_vmaxps(xmm_max, xmm_max, xmm_aux); // dst:f(1,2),f(2,2),f(3,4),f(4,4)
        uni_vmovhlps(xmm_aux, xmm_aux, xmm_max); // aux3:f(3,4),f(4,4),4,4
        if (is_max)
            uni_vmaxps(xmm_max, xmm_max, xmm_aux); // dst:f(1,2,3,4)
        else
            uni_vminps(xmm_max, xmm_max, xmm_aux);
    }   
}

template <cpu_isa_t isa>
void jit_rotary_kernel<isa>::fill_tail(const Vmm& vmm, size_t step, float val) {
    if (isa == cpu_isa_t::avx512_core) {
        uint64_t tail_mask = 1;
        tail_mask = ~((tail_mask << step) - tail_mask);
        mov(reg_tmp, tail_mask);
        kmovq(k_mask, reg_tmp);
        // fill tail of max with lower bound of float
        init_vmm(vmm_aux, reg_tmp, val);
        vblendmps(vmm_max | k_mask, vmm_max, vmm_aux);
    } else {
        uint8_t imm = 1;
        imm = ~((imm << step) - imm);
        init_vmm(vmm_aux, reg_tmp, val);
        uni_vblendps(vmm, vmm, vmm_aux, imm);
    }
}

template <cpu_isa_t isa>
void jit_rotary_kernel<isa>::data_copy(size_t step) {
    load(vmm_src0, reg_src, m_jcp.src_prc, step, false);
    if (m_jcp.do_quantize) {
        if (vec_size > step) {
            fill_tail(vmm_src0, step, -std::numeric_limits<float>::max());
        }
        uni_vmaxps(vmm_max, vmm_max, vmm_src0);
        if (vec_size > step) {
            fill_tail(vmm_src0, step, std::numeric_limits<float>::max());
        }
        uni_vminps(vmm_min, vmm_min, vmm_src0);

    }
    store(reg_dst, vmm_dst0, m_jcp.dst_prc, step);
    add(reg_src, m_jcp.src_prc.size() * step);
    add(reg_dst, m_jcp.dst_prc.size() * step);
}

template struct jit_rotary_kernel<cpu_isa_t::avx512_core>;
template struct jit_rotary_kernel<cpu_isa_t::avx2>;

}  // namespace ov::intel_cpu::kernel
