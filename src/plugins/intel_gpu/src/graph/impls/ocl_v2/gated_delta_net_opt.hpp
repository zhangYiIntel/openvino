// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <array>
#include <memory>
#include <utility>

#include "gated_delta_net_dpas_common.hpp"
#include "gated_delta_net_inst.h"
#include "program_node.h"
#include "registry/implementation_manager.hpp"

using namespace cldnn;  // TODO: Remove once namespaces are aligned

namespace ov::intel_gpu::ocl {

// Chunked (DPAS/XMX) Gated DeltaNet for long sequences: prep -> kkt -> solve -> wy -> fwd_h -> fwd_o.
// Shorter calls run the ref kernel inside the same impl; the choice is made per call, so dynamic shapes are supported.
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

        // The GEMMs run in f16, so an f32 execution keeps the full-precision ref kernel.
        if (node.get_input_layout(0).data_type != ov::element::f16) {
            return false;
        }

        const auto& q_shape = node.get_input_layout(0).get_partial_shape();
        const auto& v_shape = node.get_input_layout(2).get_partial_shape();
        // Batch and sequence length may be dynamic; the head layout has to be known.
        if (q_shape.rank().is_dynamic() || v_shape.rank().is_dynamic() || q_shape.size() != 4 || v_shape.size() != 4 || q_shape[2].is_dynamic() ||
            q_shape[3].is_dynamic() || v_shape[2].is_dynamic() || v_shape[3].is_dynamic()) {
            return false;
        }

        if (!gdn_dpas::head_layout_supported(q_shape[2].get_length(), q_shape[3].get_length(), v_shape[2].get_length(), v_shape[3].get_length())) {
            return false;
        }
        return gdn_dpas::device_supported(node.get_program().get_engine().get_device_info());
    }
};

}  // namespace ov::intel_gpu::ocl
