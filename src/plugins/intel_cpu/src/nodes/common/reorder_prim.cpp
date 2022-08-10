// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reorder_prim.h"

#include <memory>
#include <string>
#include <algorithm>
#include <dnnl_types.h>
#include <dnnl_extension_utils.h>
#include "utils/general_utils.h"
#include <cpu/x64/cpu_isa_traits.hpp>
#include <common/primitive_hashing_utils.hpp>

namespace ov {
namespace intel_cpu {

struct ReorderKey {
    dnnl::memory::desc src;
    dnnl::memory::desc dest;
    size_t hash() const;
    bool operator==(const ReorderKey& rhs) const;
};

size_t ReorderKey::hash() const {
    using namespace dnnl::impl;
    using namespace dnnl::impl::primitive_hashing;

    size_t seed = 0;
    seed = hash_combine(seed, get_md_hash(src.data));
    seed = hash_combine(seed, get_md_hash(dest.data));

    return seed;
}

bool ReorderKey::operator==(const ReorderKey& rhs) const {
    bool retVal = true;
    retVal = src == rhs.src && dest == rhs.dest;
    return retVal;
}

std::shared_ptr<dnnl::primitive> getReorderPrim(NodeRuntime& nodeRT,
                                              const dnnl::memory::desc& src,
                                              const dnnl::memory::desc& dest,
                                              impl_desc_type* p_impl_type) {
    auto builder = [&nodeRT, &p_impl_type](const ReorderKey& key) -> std::shared_ptr<dnnl::primitive> {
        dnnl::primitive_attr attr;
        dnnl::reorder::primitive_desc pd =
            dnnl::reorder::primitive_desc(nodeRT.engine, key.src, nodeRT.engine, key.dest, attr, true);
        if (!pd)
            return nullptr;
        auto info = pd.impl_info_str();
        if (p_impl_type)
            *p_impl_type = parse_impl_name(info);
        return std::make_shared<dnnl::reorder>(pd);
    };

    ReorderKey key = {src, dest};
    auto result = nodeRT.paramsCache.getOrCreate(key, builder);
    return result.first;
}

std::shared_ptr<dnnl::primitive> getReorderPrim(NodeRuntime& nodeRT,
                                              const dnnl::memory & src,
                                              const dnnl::memory & dest,
                                              impl_desc_type* p_impl_type) {
    return getReorderPrim(nodeRT, src.get_desc(), dest.get_desc(), p_impl_type);
}

}  // namespace intel_cpu
}  // namespace ov