// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <string>

#include "intel_gpu/runtime/device_info.hpp"
#include "utils/kernel_generator.hpp"

// Configuration shared by the chunked DPAS Gated DeltaNet kernels (gated_delta_net_opt.cl and
// paged_gated_delta_net_dpas.cl). The kernels derive all tile counts and strides from the JIT constants set here.
namespace ov::intel_gpu::ocl::gdn_dpas {

inline constexpr size_t chunk_size = 64;      // GDN_CHUNK: tokens per chunk
inline constexpr size_t v_block = 64;         // GDN_V_BLOCK: value columns per fwd_h workgroup (tuned: 64)
inline constexpr size_t subgroup_size = 16;   // SUBGROUP_SIZE
inline constexpr size_t wg_subgroups = 8;     // GDN_SUBGROUPS: subgroups per workgroup
inline constexpr size_t slm_pad = 4;          // GDN_SLM_PAD: padding (halves) of SLM tile rows
inline constexpr size_t prefetch_chunks = 1;  // GDN_PREFETCH_CHUNKS: fwd_h prefetch distance
inline constexpr size_t wg_lanes = subgroup_size * wg_subgroups;

inline constexpr size_t k_head_dim_supported = 128;
inline constexpr size_t min_local_mem_bytes = 56 * 1024;  // SLM needed by the largest stage

// Stage ids; they match GDN_STAGE in the kernels.
enum GdnStage : size_t { PREP = 0, KKT_SOLVE, WY, FWD_H, FWD_O, STAGE_COUNT };

inline void make_jit_constants(JitConstants& jit, size_t stage) {
    jit.make("GDN_STAGE", stage);
    jit.make("GDN_CHUNK", chunk_size);
    jit.make("GDN_V_BLOCK", v_block);
    jit.make("GDN_SUBGROUPS", wg_subgroups);
    jit.make("GDN_SLM_PAD", slm_pad);
    jit.make("GDN_PREFETCH_CHUNKS", prefetch_chunks);
}

// The GEMM-heavy stages keep many accumulator tiles in registers; they are built for the large register file.
inline std::string add_build_options(std::string options, size_t stage) {
    if (stage == WY || stage == FWD_H || stage == FWD_O) {
        options += " -cl-intel-256-GRF-per-thread";
    }
    return options;
}

// f16 DPAS with 16 lanes per subgroup (Xe-HPC and newer), enough SLM, and a discrete GPU.
// The chunked path trades arithmetic for memory traffic: it moves a lot of scratch data through DRAM, where the per-token
// kernels keep their state in registers and are limited by fp32 FMA throughput. That pays off with the memory bandwidth of a
// discrete card but not with the LPDDR shared with the CPU on an integrated GPU, where the traffic alone costs more than the
// per-token kernels take.
// TODO: replace the device type with a bandwidth / compute estimate once integrated parts are evaluated.
inline bool device_supported(const cldnn::device_info& info) {
    return info.dev_type == cldnn::device_type::discrete_gpu && info.supports_immad && info.arch >= cldnn::gpu_arch::xe_hpc &&
           info.max_local_mem_size >= min_local_mem_bytes;
}

inline bool head_layout_supported(int64_t k_heads, int64_t k_head_dim, int64_t v_heads, int64_t v_head_dim) {
    return k_head_dim == static_cast<int64_t>(k_head_dim_supported) && v_head_dim % static_cast<int64_t>(v_block) == 0 && k_heads > 0 && v_heads % k_heads == 0;
}

// Chunks needed for `tokens` packed tokens of `sequences` sequences (each sequence wastes at most one partial chunk).
inline size_t max_chunks(size_t tokens, size_t sequences) {
    return (tokens + chunk_size - 1) / chunk_size + sequences;
}

}  // namespace ov::intel_gpu::ocl::gdn_dpas
