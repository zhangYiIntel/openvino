// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gated_delta_net_ref.hpp"

#include "gated_delta_net_ref_generator.hpp"
#include "intel_gpu/primitives/gated_delta_net.hpp"
#include "primitive_ocl_base.hpp"

namespace ov::intel_gpu::ocl {
namespace {

class GatedDeltaNetRefImpl : public PrimitiveImplOCL {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::ocl::GatedDeltaNetRefImpl)

    Stage::Ptr gated_delta_net = make_stage<GatedDeltaNetRefGenerator>();

    GatedDeltaNetRefImpl() : PrimitiveImplOCL(GatedDeltaNetRef::get_type_info_static()) {}
    GatedDeltaNetRefImpl(const program_node& node, const RuntimeParams& params) : GatedDeltaNetRefImpl() {
        add_stage(gated_delta_net, params);
    }

    [[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<GatedDeltaNetRefImpl>(this);
    }
};

}  // namespace

std::unique_ptr<primitive_impl> GatedDeltaNetRef::create_impl(const program_node& node, const RuntimeParams& params) const {
    assert(node.is_type<gated_delta_net>());
    return std::make_unique<GatedDeltaNetRefImpl>(node, params);
}

}  // namespace ov::intel_gpu::ocl

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::gated_delta_net)
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::ocl::GatedDeltaNetRefImpl)
