// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/validation_util.hpp"
#include "primitive_base.hpp"
#include "dynamic_quantize/dynamic_quantize_kernel_ref.h"
#include "dynamic_quantize/dynamic_quantize_kernel_selector.h"
#include "dynamic_quantize_inst.h"

namespace cldnn {
namespace ocl {

struct dynamic_quantize_impl : typed_primitive_impl_ocl<dynamic_quantize> {
    using parent = typed_primitive_impl_ocl<dynamic_quantize>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::dynamic_quantize_kernel_selector;
    using kernel_params_t = kernel_selector::dynamic_quantize_params;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::dynamic_quantize_impl);

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<dynamic_quantize_impl>(*this);
    }

    void load(BinaryInputBuffer& ib) override {
        parent::load(ib);
        if (is_dynamic()) {
            auto& kernel_selector = kernel_selector_t::Instance();
            auto kernel_impl = kernel_selector.GetImplementation(_kernel_data.kernelName);
            kernel_impl->GetUpdateDispatchDataFunc(_kernel_data);
        }
    }

    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param, bool is_shape_agnostic = false) {
        /// TODO: handle group_size here
        auto params = get_default_params<kernel_selector::dynamic_quantize_params>(impl_param, is_shape_agnostic);
        params.outputs.push_back(convert_data_tensor(impl_param.get_output_layout(1)));

        return params;
    }

    void update_dispatch_data(const kernel_impl_params& impl_param) override {
        auto kernel_params = get_kernel_params(impl_param, true);
        (_kernel_data.update_dispatch_data_func)(kernel_params, _kernel_data);
    }
};

namespace detail {

attach_dynamic_quantize_impl::attach_dynamic_quantize_impl() {
    auto types = {
        data_types::f16,
        data_types::i8
    };

    auto formats = {
        format::bfyx,
    };

    implementation_map<dynamic_quantize>::add(impl_types::ocl,
                                    shape_types::any,
                                    typed_primitive_impl_ocl<dynamic_quantize>::create<dynamic_quantize_impl>,
                                    types,
                                    formats);
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::dynamic_quantize_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::dynamic_quantize)
