// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <array>
#include <memory>
#include <utility>

#include "gated_delta_net_dpas_common.hpp"
#include "gated_delta_net_inst.h"
#include "paged_gated_delta_net_jit_utils.hpp"
#include "program_node.h"
#include "registry/implementation_manager.hpp"

using namespace cldnn;  // TODO: Remove once namespaces are aligned

namespace ov::intel_gpu::ocl {

// Per-token kernel on the dense layouts (gated_delta_net_tok.cl): the choice where the chunked DPAS path does not pay off (integrated GPUs).
namespace gdn_tok {
inline constexpr int64_t max_k_head_dim = 128;     // a lane holds a whole state column: K floats of the 256-register file's 256 per lane, half of it
inline constexpr int64_t k_head_dim_step = 16;     // K tiles over the 16 lanes
inline constexpr int64_t v_rows_per_subgroup = 16;  // = the subgroup size: one v column per lane

inline bool device_supported(const cldnn::device_info& info) {
    return paged_gated_delta_net_jit::get_subgroup_size(info.arch) == 16 && paged_gated_delta_net_jit::supports_large_grf(info.arch);
}

inline bool head_layout_supported(int64_t k_heads, int64_t k_head_dim, int64_t v_heads, int64_t v_head_dim) {
    return k_head_dim > 0 && k_head_dim <= max_k_head_dim && k_head_dim % k_head_dim_step == 0 && v_head_dim > 0 && v_head_dim % v_rows_per_subgroup == 0 &&
           k_heads > 0 && v_heads % k_heads == 0;
}
}  // namespace gdn_tok

// Chunked (DPAS/XMX) Gated DeltaNet for long sequences: prep -> kkt -> solve -> wy -> fwd_h -> fwd_o, or on devices where that does not pay off
// the per-token kernel. Shorter calls run the ref kernel inside the same impl; the choice is made per call, so dynamic shapes are supported.
struct GatedDeltaNetOpt : public ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("ocl::gated_delta_net::opt")
    explicit GatedDeltaNetOpt(shape_types shape_type, ValidateFunc vf = nullptr) : ImplementationManager(impl_types::ocl, shape_type, std::move(vf)) {}
    [[nodiscard]] std::unique_ptr<primitive_impl> create_impl(const program_node& node, const RuntimeParams& params) const override;

    [[nodiscard]] bool validate_impl(const program_node& node) const override {
        assert(node.is_type<gated_delta_net>());
        static constexpr std::array supported_fmts = {
            format::bfyx,
        };

        static constexpr std::array supported_types = {
            ov::element::f16,
            ov::element::f32,
        };

        if (node.get_dependencies().size() != 6) {
            return false;
        }

        for (size_t i = 0; i < node.get_dependencies().size(); i++) {
            const auto& in_layout = node.get_input_layout(i);
            if (!one_of(in_layout.format, supported_fmts) || !one_of(in_layout.data_type, supported_types)) {
                return false;
            }
        }

        const auto& out_layout = node.get_output_layout(0);
        if (!one_of(out_layout.format, supported_fmts) || !one_of(out_layout.data_type, supported_types)) {
            return false;
        }

        const auto& q_shape = node.get_input_layout(0).get_partial_shape();
        const auto& v_shape = node.get_input_layout(2).get_partial_shape();
        // Batch and sequence length may be dynamic; the head layout has to be known.
        if (q_shape.rank().is_dynamic() || v_shape.rank().is_dynamic() || q_shape.size() != 4 || v_shape.size() != 4 || q_shape[2].is_dynamic() ||
            q_shape[3].is_dynamic() || v_shape[2].is_dynamic() || v_shape[3].is_dynamic()) {
            return false;
        }

        const auto k_heads = q_shape[2].get_length();
        const auto k_dim = q_shape[3].get_length();
        const auto v_heads = v_shape[2].get_length();
        const auto v_dim = v_shape[3].get_length();
        const auto& info = node.get_program().get_engine().get_device_info();
        // The DPAS GEMMs run in f16, so an f32 execution keeps the full-precision kernels.
        const bool dpas = node.get_input_layout(0).data_type == ov::element::f16 && gdn_dpas::head_layout_supported(k_heads, k_dim, v_heads, v_dim) &&
                          gdn_dpas::device_supported(info);
        const bool tok = gdn_tok::head_layout_supported(k_heads, k_dim, v_heads, v_dim) && gdn_tok::device_supported(info);
        return dpas || tok;
    }
};

}  // namespace ov::intel_gpu::ocl
