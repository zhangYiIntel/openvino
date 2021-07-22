// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <algorithm>
#include <numeric>
#include <unordered_set>

#include "utils/general_utils.h"

#include <mkldnn_types.h>
#include <dnnl_types.h>
#include <common/memory_desc_wrapper.hpp>
#include "mkldnn_memory.h"
#include "mkldnn_extension_utils.h"
#include "nodes/common/cpu_memcpy.h"
#include "nodes/common/cpu_convert.h"
#include "mkldnn/ie_mkldnn.h"
#include "cpu_shape.h"
#include "cpu_memory_desc_utils.h"

using namespace InferenceEngine;
using namespace mkldnn;

namespace MKLDNNPlugin {
namespace {
    inline void setSubnormalsToZero(float *data, size_t size) {
        uint32_t *u32data = reinterpret_cast<uint32_t *>(data);
        for (size_t i = 0; i < size; ++i) {
            if ((u32data[i] & (0xFF << 23)) == 0) {
                u32data[i] = 0;
            }
        }
    }
}   // namespace

MKLDNNMemory::MKLDNNMemory(const mkldnn::engine& eng) : eng(eng) {}

size_t MKLDNNMemory::GetSize() const {
    uint8_t itemSize = MKLDNNExtensionUtils::sizeOfDataType(mkldnn::memory::data_type(GetDataType()));
    return GetElementsCount() * itemSize;
}

size_t MKLDNNMemory::GetElementsCount() const {
    auto desc = GetDescriptor();
    std::vector<int> dims(desc.data.padded_dims,
                          desc.data.padded_dims + desc.data.ndims);
    return std::accumulate(std::begin(dims), std::end(dims), (size_t) 1, std::multiplies<size_t>());
}

void MKLDNNMemory::Create(const memory::dims& dims, memory::data_type data_type, memory::format_tag format, const void* data) {
    if (format == memory::format_tag::undef) {
        format = memory::format_tag::any;
    }

    memory::desc desc = MKLDNNMemoryDesc({dims}, data_type, format);

    Create(desc, data);
}

void MKLDNNMemory::Create(const mkldnn::memory::desc& desc, const void *data, bool pads_zeroing) {
    if (data == nullptr) {
        prim.reset(new memory(desc, eng));

        size_t real_size = 0;
        if (desc.data.format_kind == dnnl_format_kind_wino)
            return;
        auto desc_loc = prim->get_desc().data;
        if (desc_loc.ndims > 0) {
            real_size = static_cast<size_t>(desc_loc.padded_dims[0]);
            for (int i = 1; i < desc_loc.ndims; i++) {
                real_size *= desc_loc.padded_dims[i];
            }
        }
    } else {
        // MKLDNN accepts not a const data, probably need to remove some level of consteness in a call stack

        // ========================
        // Equivalent of constructor memory(const primitive_desc &desc, void *hdl)
        // but with ability to skipp pads zeroing.
        prim.reset(new memory(desc, eng, DNNL_MEMORY_NONE));
        if (pads_zeroing)
            prim->set_data_handle(const_cast<void*>(data));
        else
            prim->set_data_handle_no_pads_proc(const_cast<void*>(data));
        //
        // ========================
    }
}

void MKLDNNMemory::Create(const MemoryDesc &desc, const void *data, bool pads_zeroing) {
    pMemDesc = desc.clone();
    Create(mkldnn::memory::desc(MemoryDescUtils::convertToMKLDNNMemoryDesc(desc)), data, pads_zeroing);
}


void MKLDNNMemory::reorderData(const MKLDNNMemory &input, const MKLDNNMemory &output, size_t size) {
    if (size != 0)
        IE_ASSERT(size <= output.GetDescriptor().get_size());
    if (input.GetMKLDNNDesc() == output.GetMKLDNNDesc()) {
        auto srcPtr = static_cast<uint8_t*>(input.GetPtr());
        auto dstPtr = static_cast<uint8_t*>(output.GetPtr());

        auto copySize = size == 0 ? output.GetSize() : size;
        cpu_memcpy(dstPtr, srcPtr, copySize);
    } else {
        std::unique_ptr<mkldnn::reorder> pReorder;
        std::shared_ptr<memory> srcMemoryPtr;
        std::vector<uint8_t> tmpBuff;

        try {
            pReorder = std::unique_ptr<mkldnn::reorder>(new mkldnn::reorder(input.GetPrimitive(), output.GetPrimitive()));
            srcMemoryPtr = input.prim;
        }
        catch (const mkldnn::error& err) {
            if (mkldnn_unimplemented == err.status && output.GetDataType() != input.GetDataType()) {
                //we probably could not make the reorder because there is no one supporting this precision conversion
                //lets try to convert data first using cpu_convert
                auto data = static_cast<const uint8_t *>(input.GetPtr());
                tmpBuff.resize(input.GetSize());

                cpu_convert(data, tmpBuff.data(), MKLDNNExtensionUtils::DataTypeToIEPrecision(input.GetDataType()),
                            MKLDNNExtensionUtils::DataTypeToIEPrecision(output.GetDataType()), input.GetElementsCount());

                MKLDNNMemory tmpMem(output.eng);
                tmpMem.Create(input.GetDims(), output.GetDataType(), input.GetMKLDNNDesc().getFormat(), tmpBuff.data());

                pReorder = std::unique_ptr<mkldnn::reorder>(new mkldnn::reorder(tmpMem.GetPrimitive(), output.GetPrimitive()));
                srcMemoryPtr = tmpMem.prim;
            } else {
                throw;
            }
        }
        if (pReorder) {
            mkldnn::stream loc_stream(output.eng, stream::flags::default_order);
            pReorder->execute(loc_stream, *srcMemoryPtr, *output.prim);
        } else {
            IE_THROW() << "Could not make mkldnn reorder.";
        }
    }
}

// TODO: It should be done via wrap into Memory;
void MKLDNNMemory::SetData(memory::data_type dataType, memory::format_tag format, const void* data, size_t size, bool ftz) const {
    IE_ASSERT(!one_of(format, memory::format_tag::undef, memory::format_tag::any));

    auto dst_desc = GetDescriptor();
    memory::desc src_desc{dst_desc.dims(), dataType, format};

    IE_ASSERT(size <= dst_desc.get_size());

    if (dst_desc == src_desc) {
        uint8_t itemSize = MKLDNNExtensionUtils::sizeOfDataType(mkldnn::memory::data_type(dataType));
        uint8_t* dataPtr = static_cast<uint8_t*>(GetData());
        // We cannot support strides for i/o blobs because it affects performance.
        dataPtr += itemSize * prim->get_desc().data.offset0;
        cpu_memcpy(dataPtr, data, size);
    } else {
        auto memData = this->GetDescriptor().data;
        memory::dims dims(memData.dims, memData.dims + memData.ndims);

        MKLDNNMemory src(this->eng);
        src.Create(dims, dataType, format, data);

        reorderData(src, *this);
    }
    if (ftz
        && dataType == memory::data_type::f32
        && prim->get_desc().data.format_kind != dnnl_format_kind_wino
        && GetDataType() != memory::data_type::bf16) {
        // Internal blobs haven't strides yet.
        auto *memData = static_cast<float *>(GetData());
        memData += prim->get_desc().data.offset0;
        setSubnormalsToZero(memData, GetSize() / sizeof(float));
    }
}

void MKLDNNMemory::SetData(const MKLDNNMemory& src, size_t size, bool ftz) const {
    reorderData(src, *this, size);

    if (ftz
        && src.GetDataType() == memory::data_type::f32
        && prim->get_desc().data.format_kind != dnnl_format_kind_wino
        && GetDataType() != memory::data_type::bf16) {
        // Internal blobs haven't strides yet.
        auto *memData = static_cast<float *>(GetData());
        memData += prim->get_desc().data.offset0;
        setSubnormalsToZero(memData, GetSize() / sizeof(float));
    }
}

void MKLDNNMemory::FillZero() {
    void* dataPtr = GetData();
    memset(dataPtr, 0, GetSize());
}

memory::format_tag MKLDNNMemory::GetPlainFormatByRank(size_t rank) {
    switch (rank) {
        case 0:
        case 1:
            return memory::format_tag::a;
        case 2:
            return memory::format_tag::ab;
        case 3:
            return memory::format_tag::abc;
        case 4:
            return memory::format_tag::abcd;
        case 5:
            return memory::format_tag::abcde;
        case 6:
            return memory::format_tag::abcdef;
        default:
            return memory::format_tag::undef;
    }
}

InferenceEngine::Layout MKLDNNMemory::GetPlainLayout(const memory::dims& dims) {
    switch (dims.size()) {
        case 0: return Layout::SCALAR;
        case 1: return Layout::C;
        case 2: return Layout::NC;
        case 3: return Layout::CHW;
        case 4: return Layout::NCHW;
        case 5: return Layout::NCDHW;
        default:
            return Layout::BLOCKED;
    }
}

Precision MKLDNNMemory::convertToIePrec(memory::data_type dataType) {
    return MKLDNNExtensionUtils::DataTypeToIEPrecision(dataType);
}

memory::data_type MKLDNNMemory::convertToDataType(const InferenceEngine::Precision &precision) {
    return MKLDNNExtensionUtils::IEPrecisionToDataType(precision);
}

memory::format_tag MKLDNNMemory::Convert(const InferenceEngine::Layout layout) {
    switch (layout) {
        case NCHW:
            return memory::format_tag::nchw;
        case NHWC:
            return memory::format_tag::nhwc;
        case NCDHW:
            return memory::format_tag::ncdhw;
        case NDHWC:
            return memory::format_tag::ndhwc;
        case CHW:
            return memory::format_tag::tnc;
        case NC:
            return memory::format_tag::nc;
        case C:
            return memory::format_tag::x;
        case SCALAR:
            return memory::format_tag::x;
        default:
            return memory::format_tag::undef;
    }
}

std::string MKLDNNMemory::formatToString(memory::format_tag fmt) {
    return mkldnn::utils::fmt2str(fmt);
}

void *MKLDNNMemory::GetPtr() const  {
    auto ptr = static_cast<uint8_t*>(GetData());
    auto md = GetDescriptor().data;
    mkldnn::impl::memory_desc_wrapper wrapper(md);
    ptr += wrapper.offset0() * wrapper.data_type_size();
    return ptr;
}

template<>
MKLDNNMemoryDesc MKLDNNMemory::GetDescWithType<MKLDNNMemoryDesc, 0, 0>() const {
    if (auto descPtr = dynamic_cast<const MKLDNNMemoryDesc*>(pMemDesc.get())) {
        return *descPtr;
    } else {
        switch (pMemDesc->getType()) {
            case (MemoryDescType::Blocked):
                return MemoryDescUtils::convertToMKLDNNMemoryDesc(*(pMemDesc->as<BlockedMemoryDesc>()));
            default:
                IE_THROW() << "Can not convert unsupported memory descriptor";
        }
    }
}

template<>
BlockedMemoryDesc MKLDNNMemory::GetDescWithType<BlockedMemoryDesc, 0, 0>() const {
    if (auto descPtr = dynamic_cast<const BlockedMemoryDesc*>(pMemDesc.get())) {
        return *descPtr;
    } else {
        switch (pMemDesc->getType()) {
            case (MemoryDescType::Mkldnn):
                return MemoryDescUtils::convertToBlockedDescriptor(*(pMemDesc->as<MKLDNNMemoryDesc>()));
            default:
                IE_THROW() << "Can not convert unsupported memory descriptor";
        }
    }
}

bool MKLDNNMemoryDesc::operator==(const MKLDNNMemoryDesc &rhs) const {
    return this->desc == rhs.desc;
}

bool MKLDNNMemoryDesc::operator!=(const MKLDNNMemoryDesc &rhs) const {
    return !(*this == rhs);
}

MKLDNNMemoryDesc::operator mkldnn::memory::desc() const {
    return desc;
}

MKLDNNMemoryDesc::MKLDNNMemoryDesc(const mkldnn::memory::desc& desc) :
    MemoryDesc(Shape(desc.dims()), Mkldnn), desc(desc) {
    if (desc.data.format_kind == dnnl::impl::format_kind::any)
        IE_THROW(Unexpected) << "Memory format any is prohibited!";
}

MKLDNNMemoryDesc::MKLDNNMemoryDesc(const mkldnn::memory::dims& dims, mkldnn::memory::data_type dataType, mkldnn::memory::format_tag format)
       : MemoryDesc(Shape(dims), Mkldnn) {
    if (format == memory::format_tag::any)
        IE_THROW(Unexpected) << "Memory format any is prohibited!";
    if (format != memory::format_tag::undef) {
        if (format == memory::format_tag::x && dims.size() == 0) {
            desc = mkldnn::memory::desc(mkldnn::memory::dims(1, 1), dataType, format);
        } else {
            desc = mkldnn::memory::desc(dims, dataType, format);
        }
    } else {
        // Trying to create plain descriptor
        // This WA is needed since memory::format_tag doesn't contain plain tag for tensors with rank > 6D
        mkldnn::memory::dims strides(dims.size(), 1);
        for (int d = dims.size() - 2; d >= 0; d--) {
            strides[d] = strides[d + 1] * dims[d + 1];
        }

        desc = mkldnn::memory::desc(dims, dataType, strides);
    }
}

MKLDNNMemoryDesc::MKLDNNMemoryDesc(const mkldnn::memory::dims& dims, mkldnn::memory::data_type dataType)
        : MemoryDesc(Shape(dims), Mkldnn), desc() {
    const auto ndims = dims.size();
    mkldnn::memory::dims plain_strides(ndims, 1);
    for (size_t i = 1; i < ndims; i++) {
        plain_strides[ndims - i -1] = plain_strides[ndims - i] * dims[ndims - i];
    }
    desc = {dims, dataType, plain_strides};
}

size_t MKLDNNMemoryDesc::GetElementSize() const {
    const auto type = desc.data_type();
    switch (type) {
        case memory::data_type::f16 :
        case memory::data_type::bf16 :
            return 2;
        case memory::data_type::f32 :
        case memory::data_type::s32 :
            return 4;
        case memory::data_type::s8 :
        case memory::data_type::u8 :
        case memory::data_type::bin :
            return 1;
        default:
            IE_THROW() << "Unknown data type";
    }
}

static const std::map<int, std::vector<mkldnn::memory::format_tag>> form_tags_by_ndims {
    {0, {
        mkldnn::memory::format_tag::a   // TODO :: really 1d layout for scalar??
     }}, {1, {
        mkldnn::memory::format_tag::a
     }}, {2, {
        mkldnn::memory::format_tag::ab,
        mkldnn::memory::format_tag::ba
     }}, {3, {
        mkldnn::memory::format_tag::abc,
        mkldnn::memory::format_tag::acb,
        mkldnn::memory::format_tag::bac,
        mkldnn::memory::format_tag::bca,
        mkldnn::memory::format_tag::cba,

        mkldnn::memory::format_tag::Abc16a,
        mkldnn::memory::format_tag::ABc16a16b,
        mkldnn::memory::format_tag::ABc4a4b,
        mkldnn::memory::format_tag::aBc16b,
        mkldnn::memory::format_tag::aBc32b,
        mkldnn::memory::format_tag::ABc16b16a,
        mkldnn::memory::format_tag::Abc4a,
        mkldnn::memory::format_tag::aBc4b,
        mkldnn::memory::format_tag::ABc4b16a4b,
        mkldnn::memory::format_tag::ABc2b8a4b,
        mkldnn::memory::format_tag::ABc16b16a4b,
        mkldnn::memory::format_tag::ABc16b16a2b,
        mkldnn::memory::format_tag::ABc4b4a,
        mkldnn::memory::format_tag::ABc8a16b2a,
        mkldnn::memory::format_tag::ABc8a8b,
        mkldnn::memory::format_tag::ABc8a4b,
        mkldnn::memory::format_tag::aBc8b,
        mkldnn::memory::format_tag::ABc8b16a2b,
        mkldnn::memory::format_tag::ABc8b8a,
        mkldnn::memory::format_tag::Acb16a,
        mkldnn::memory::format_tag::Acb4a,
        mkldnn::memory::format_tag::Acb8a,
        mkldnn::memory::format_tag::BAc16a16b,
        mkldnn::memory::format_tag::BAc16b16a,
     }}, {4, {                                 // Popular
        mkldnn::memory::format_tag::abcd,      // plain
        mkldnn::memory::format_tag::acdb,      // tail_c
        mkldnn::memory::format_tag::aBcd8b,    // blocked 8c
        mkldnn::memory::format_tag::aBcd16b,   // blocked 16c

        mkldnn::memory::format_tag::abdc,

        mkldnn::memory::format_tag::bacd,
        mkldnn::memory::format_tag::bcda,
        mkldnn::memory::format_tag::cdba,
        mkldnn::memory::format_tag::dcab,

        mkldnn::memory::format_tag::Abcd8a,
        mkldnn::memory::format_tag::Abcd16a,
        mkldnn::memory::format_tag::Abcd32a,
        mkldnn::memory::format_tag::ABcd16a16b,
        mkldnn::memory::format_tag::aBcd32b,
        mkldnn::memory::format_tag::ABcd16b16a,
        mkldnn::memory::format_tag::aBCd16b16c,
        mkldnn::memory::format_tag::aBCd16c16b,
        mkldnn::memory::format_tag::Abcd4a,
        mkldnn::memory::format_tag::aBcd4b,
        mkldnn::memory::format_tag::ABcd4b16a4b,
        mkldnn::memory::format_tag::ABcd2b8a4b,
        mkldnn::memory::format_tag::ABcd4b4a,
        mkldnn::memory::format_tag::ABcd4a4b,
        mkldnn::memory::format_tag::aBCd4c16b4c,
        mkldnn::memory::format_tag::aBCd2c8b4c,
        mkldnn::memory::format_tag::ABcd16b16a4b,
        mkldnn::memory::format_tag::ABcd16b16a2b,
        mkldnn::memory::format_tag::aBCd16c16b4c,
        mkldnn::memory::format_tag::aBCd16c16b2c,
        mkldnn::memory::format_tag::aBCd4c4b,
        mkldnn::memory::format_tag::aBCd4b4c,
        mkldnn::memory::format_tag::ABcd8a16b2a,
        mkldnn::memory::format_tag::ABcd8a8b,
        mkldnn::memory::format_tag::ABcd8a32b,
        mkldnn::memory::format_tag::ABcd32a32b,
        mkldnn::memory::format_tag::ABcd8a4b,

        mkldnn::memory::format_tag::ABcd8b16a2b,
        mkldnn::memory::format_tag::aBCd8b16c2b,
        mkldnn::memory::format_tag::ABcd8b8a,
        mkldnn::memory::format_tag::aBCd8b8c,
        mkldnn::memory::format_tag::aBCd8b4c,
        mkldnn::memory::format_tag::aBCd8c16b2c,
        mkldnn::memory::format_tag::aBCd8c8b,

        mkldnn::memory::format_tag::ABcd4a8b8a4b,
        mkldnn::memory::format_tag::ABcd2a8b8a2b,

        mkldnn::memory::format_tag::aBdc16b,
        mkldnn::memory::format_tag::aBdc4b,
        mkldnn::memory::format_tag::aBdc8b,
        mkldnn::memory::format_tag::aCBd16b16c,
        mkldnn::memory::format_tag::aCBd16c16b,
        mkldnn::memory::format_tag::Acdb16a,
        mkldnn::memory::format_tag::Acdb4a,
        mkldnn::memory::format_tag::Acdb8a,
        mkldnn::memory::format_tag::BAcd16a16b,
        mkldnn::memory::format_tag::BAcd16b16a,
        mkldnn::memory::format_tag::ABcd32a32b,
        mkldnn::memory::format_tag::Acdb32a,
        mkldnn::memory::format_tag::aBCd2b4c2b,
        mkldnn::memory::format_tag::aBCd2c4b2c,
        mkldnn::memory::format_tag::aBCd4b8c2b,
        mkldnn::memory::format_tag::aBCd4c8b2c,
    }}, {5, {                                   // Popular
        mkldnn::memory::format_tag::abcde,      // plain
        mkldnn::memory::format_tag::acdeb,      // tail_c
        mkldnn::memory::format_tag::aBcde8b,    // blocked 8c
        mkldnn::memory::format_tag::aBcde16b,   // blocked 16c

        mkldnn::memory::format_tag::abdec,
        mkldnn::memory::format_tag::acbde,
        mkldnn::memory::format_tag::bacde,
        mkldnn::memory::format_tag::bcdea,
        mkldnn::memory::format_tag::cdeba,
        mkldnn::memory::format_tag::decab,

        mkldnn::memory::format_tag::Abcde16a,
        mkldnn::memory::format_tag::Abcde32a,
        mkldnn::memory::format_tag::ABcde16a16b,
        mkldnn::memory::format_tag::aBcde32b,
        mkldnn::memory::format_tag::ABcde16b16a,
        mkldnn::memory::format_tag::aBCde16b16c,
        mkldnn::memory::format_tag::aBCde16c16b,
        mkldnn::memory::format_tag::aBCde2c8b4c,
        mkldnn::memory::format_tag::Abcde4a,
        mkldnn::memory::format_tag::aBcde4b,
        mkldnn::memory::format_tag::ABcde4b4a,
        mkldnn::memory::format_tag::ABcde4a4b,
        mkldnn::memory::format_tag::aBCde4b4c,
        mkldnn::memory::format_tag::aBCde4c16b4c,
        mkldnn::memory::format_tag::aBCde16c16b4c,
        mkldnn::memory::format_tag::aBCde16c16b2c,
        mkldnn::memory::format_tag::aBCde4c4b,
        mkldnn::memory::format_tag::Abcde8a,
        mkldnn::memory::format_tag::ABcde8a8b,
        mkldnn::memory::format_tag::ABcde8a4b,
        mkldnn::memory::format_tag::ABcde8b16a2b,
        mkldnn::memory::format_tag::ABcde4b16a4b,
        mkldnn::memory::format_tag::ABcde2b8a4b,
        mkldnn::memory::format_tag::aBCde8b16c2b,
        mkldnn::memory::format_tag::ABcde8b8a,
        mkldnn::memory::format_tag::aBCde8b8c,
        mkldnn::memory::format_tag::aBCde8b4c,
        mkldnn::memory::format_tag::aBCde4b8c8b4c,
        mkldnn::memory::format_tag::aBCde2b8c8b2c,
        mkldnn::memory::format_tag::aBCde8c16b2c,
        mkldnn::memory::format_tag::aBCde8c8b,
        mkldnn::memory::format_tag::aBdec16b,
        mkldnn::memory::format_tag::aBdec4b,
        mkldnn::memory::format_tag::aBdec8b,
        mkldnn::memory::format_tag::aCBde16b16c,
        mkldnn::memory::format_tag::aCBde16c16b,
        mkldnn::memory::format_tag::Acdeb16a,
        mkldnn::memory::format_tag::Acdeb4a,
        mkldnn::memory::format_tag::Acdeb8a,
        mkldnn::memory::format_tag::BAcde16b16a,
        mkldnn::memory::format_tag::BAcde16a16b,
        mkldnn::memory::format_tag::aBdec32b,
        mkldnn::memory::format_tag::aBCde2b4c2b,
        mkldnn::memory::format_tag::aBCde2c4b2c,
        mkldnn::memory::format_tag::aBCde4b8c2b,
        mkldnn::memory::format_tag::aBCde4c8b2c,
    }}, {6, {                                    // Popular
        mkldnn::memory::format_tag::abcdef,      // plain
        mkldnn::memory::format_tag::acbdef,      // permute
        mkldnn::memory::format_tag::defcab,      // permute
        mkldnn::memory::format_tag::aBcdef16b,   // blocked 16c

        mkldnn::memory::format_tag::aBCdef16b16c,
        mkldnn::memory::format_tag::aBCdef16c16b,
        mkldnn::memory::format_tag::aBcdef4b,
        mkldnn::memory::format_tag::aBCdef2c8b4c,
        mkldnn::memory::format_tag::aBCdef4c4b,
        mkldnn::memory::format_tag::aBCdef4b4c,
        mkldnn::memory::format_tag::aBCdef8b8c,
        mkldnn::memory::format_tag::aBCdef8b4c,
        mkldnn::memory::format_tag::aBCdef8c16b2c,
        mkldnn::memory::format_tag::aBCdef4c16b4c,
        mkldnn::memory::format_tag::aBCdef8c8b,

        mkldnn::memory::format_tag::aBdefc16b,
        mkldnn::memory::format_tag::aCBdef16c16b,
        mkldnn::memory::format_tag::aCBdef16b16c,
        mkldnn::memory::format_tag::aBdefc4b,
        mkldnn::memory::format_tag::aBdefc8b,

        mkldnn::memory::format_tag::Abcdef4a,
        mkldnn::memory::format_tag::Abcdef8a,
        mkldnn::memory::format_tag::Abcdef16a,
        mkldnn::memory::format_tag::Abcdef32a,
        mkldnn::memory::format_tag::aBCdef2b4c2b,
        mkldnn::memory::format_tag::aBCdef2c4b2c,
        mkldnn::memory::format_tag::aBCdef4b8c2b,
        mkldnn::memory::format_tag::aBCdef4c8b2c,
        }}
};

mkldnn::memory::format_tag MKLDNNMemoryDesc::getFormat() const {
    // TODO [OneDNN]: Previously it was a field of tdesc, but now the brute
    //                force search here. Please avoid of using this method.
    const auto ndims = desc.dims().size();

    // There are no suitable format_tag for this
    if (ndims == 0 || ndims > 6)
        return mkldnn::memory::format_tag::undef;

    for (const auto fmt : form_tags_by_ndims.at(ndims)) {
        if (this->isSame(fmt))
            return fmt;
    }

    return mkldnn::memory::format_tag::undef;
}

bool MKLDNNMemoryDesc::isSame(mkldnn::memory::format_tag fmt) const {
    memory::desc refDesc(desc.dims(), desc.data_type(), fmt);

    if (desc.data.ndims != refDesc.data.ndims)
        return false;

    if (desc.data.format_kind != dnnl_blocked || refDesc.data.format_kind != dnnl_blocked)
        IE_THROW() << "MKLDNNMemoryDesc::isSame is not implemented for non blocked memory format";

    auto actualBlkDesc = desc.data.format_desc.blocking;
    auto refBlkDesc = refDesc.data.format_desc.blocking;
    if (actualBlkDesc.inner_nblks != refBlkDesc.inner_nblks)
        return false;

    for (size_t i = 0; i < actualBlkDesc.inner_nblks; ++i)
        if (actualBlkDesc.inner_blks[i] != refBlkDesc.inner_blks[i])
            return false;

    for (size_t i = 0; i < actualBlkDesc.inner_nblks; ++i)
        if (actualBlkDesc.inner_idxs[i] != refBlkDesc.inner_idxs[i])
            return false;

    auto actualStrides = desc.data.format_desc.blocking.strides;
    auto refStrides = refDesc.data.format_desc.blocking.strides;

    std::vector<size_t> actualOrder(desc.data.ndims);
    {
        const auto dims = desc.dims();
        std::vector<size_t> total_block_per_dim(dims.size(), 1);
        const auto &blk_desc = desc.data.format_desc.blocking;
        for (int i = 0; i < blk_desc.inner_nblks; i++) {
            total_block_per_dim[blk_desc.inner_idxs[i]] *= blk_desc.inner_blks[i];
        }
        std::vector<size_t> outer_block_dims(std::begin(dims), std::begin(dims) + dims.size());
        for (size_t i = 0; i < outer_block_dims.size(); i++) {
            outer_block_dims[i] = div_up(outer_block_dims[i], total_block_per_dim[i]);
        }

        std::iota(actualOrder.begin(), actualOrder.end(), 0);
        std::sort(actualOrder.begin(), actualOrder.end(),
                  [&actualStrides, &outer_block_dims] (size_t ind_l, size_t ind_r) {
                      return (actualStrides[ind_l] > actualStrides[ind_r]) ||
                             (actualStrides[ind_l] == actualStrides[ind_r] && outer_block_dims[ind_l] > outer_block_dims[ind_r]);
                  });
    }

    std::vector<size_t> refOrder(refDesc.data.ndims);
    {
        const auto dims = refDesc.dims();
        std::vector<size_t> total_block_per_dim(dims.size(), 1);
        const auto &blk_desc = refDesc.data.format_desc.blocking;
        for (int i = 0; i < blk_desc.inner_nblks; i++) {
            total_block_per_dim[blk_desc.inner_idxs[i]] *= blk_desc.inner_blks[i];
        }
        std::vector<size_t> outer_block_dims(std::begin(dims), std::begin(dims) + dims.size());
        for (size_t i = 0; i < outer_block_dims.size(); i++) {
            outer_block_dims[i] = div_up(outer_block_dims[i], total_block_per_dim[i]);
        }

        std::iota(refOrder.begin(), refOrder.end(), 0);
        std::sort(refOrder.begin(), refOrder.end(),
                  [&refStrides, &outer_block_dims] (size_t ind_l, size_t ind_r) {
                      return (refStrides[ind_l] > refStrides[ind_r]) ||
                             (refStrides[ind_l] == refStrides[ind_r] && outer_block_dims[ind_l] > outer_block_dims[ind_r]);
                  });
    }

    if (actualOrder != refOrder) {
        return false;
    }

    return true;
}

bool MKLDNNMemoryDesc::isPlainFormat() const {
    if (desc.data.format_kind != dnnl_blocked ||
        desc.data.format_desc.blocking.inner_nblks != 0)
        return false;

    const auto ndims = desc.data.ndims;
    const auto dims = desc.data.dims;
    const auto &strides = desc.data.format_desc.blocking.strides;
    bool is_plain_strides = (strides[ndims-1] == 1);
    for (int i = 0; i < ndims - 1; i++) {
        is_plain_strides &= (strides[i] == strides[i+1] * dims[i+1]);
    }

    return is_plain_strides;
}

bool MKLDNNMemoryDesc::isBlockedCFormat(size_t blk_size) const {
    const auto &blocking = desc.data.format_desc.blocking;

    if (desc.data.format_kind != dnnl_blocked ||
        blocking.inner_nblks != 1 ||
        blocking.inner_idxs[0] != 1)
        return false;

    const auto &ndims = desc.data.ndims;
    const auto &strides = desc.data.format_desc.blocking.strides;
    const auto &dims = desc.data.padded_dims;

    if (blk_size == UNREACHABLE_DIM) {
        blk_size = blocking.inner_blks[0];
    } else {
        if (blk_size != blocking.inner_blks[0])
            return false;
    }

    bool is_direct_order = (strides[ndims-1] == blocking.inner_blks[0]);
    for (int i = 0; i < ndims - 1; i++) {
        auto dim = (i == 0) ? div_up(dims[i+1], blk_size) : dims[i+1];
        is_direct_order &= (strides[i] >= strides[i+1] * dim);
    }

    return is_direct_order;
}

bool MKLDNNMemoryDesc::isTailCFormat() const {
    const auto &blocking = desc.data.format_desc.blocking;

    if (desc.data.format_kind != dnnl_blocked ||
        blocking.inner_nblks != 0)
        return false;

    const auto &ndims = desc.data.ndims;
    const auto &strides = desc.data.format_desc.blocking.strides;
    const auto &dims = desc.data.padded_dims;

    // dense permutation of acd..b
    bool is_tailc_strides = (strides[1] == 1 && strides[ndims-1] == dims[1] && strides[0] == dims[2] * strides[2]);
    for (int i = 2; i < ndims - 1; i++) {
        is_tailc_strides &= (strides[i] == strides[i+1] * dims[i+1]);
    }

    return is_tailc_strides;
}

bool MKLDNNMemoryDesc::blocksExtended() const {
    for (int i = 0; i < desc.data.ndims; i++) {
        if (desc.data.dims[i] != desc.data.padded_dims[i])
            return true;
    }
    return false;
}

size_t MKLDNNMemoryDesc::getMemSizeImp() const {
    return desc.get_size();
}

size_t MKLDNNMemoryDesc::getOffset(size_t elemNumber) const {
    mkldnn::impl::memory_desc_wrapper wrapped(desc.data);
    return wrapped.off_l(elemNumber);
}

bool MKLDNNMemoryDesc::isCompatible(const MemoryDesc &rhs) const {
    if (MemoryDescType::Blocked == rhs.getType()) {
        return isCompatible(*(rhs.as<BlockedMemoryDesc>()));
    } else if (MemoryDescType::Mkldnn == rhs.getType()) {
        return isCompatible(*(rhs.as<MKLDNNMemoryDesc>()));
    } else {
        return false;
    }
}

bool MKLDNNMemoryDesc::isCompatible(const MKLDNNMemoryDesc &rhs) const {
    using namespace dnnl;
    using namespace impl;
    using namespace dnnl::impl::utils;
    if (this->desc == rhs.desc) {
        return true;
    }
    mkldnn::impl::memory_desc_wrapper wrappedThis(this->desc.data);
    mkldnn::impl::memory_desc_wrapper wrappedRhs(rhs.desc.data);
    if (one_of(wrappedThis.format_kind(), format_kind::undef, format_kind::any))
        return false;
    if (wrappedThis.is_wino_desc() || wrappedThis.is_rnn_packed_desc()) return false;

    const auto &blk = wrappedThis.blocking_desc();
    const auto &r_blk = wrappedRhs.blocking_desc();

    int stride_start = wrappedThis.ndims() >0 && wrappedThis.dims()[0] == 1 ? 1 : 0;  //ignore batch axis stride if batch size == 1

    // Here is a slightly modified version of mkldnn::impl::memory_desc_wrapper::similar_to() call able to skip specific strides check.
    return wrappedThis.ndims() == wrappedRhs.ndims()
           && wrappedThis.format_kind() == wrappedRhs.format_kind()
           && wrappedThis.data_type() == wrappedRhs.data_type()
           && array_cmp(wrappedThis.dims(), wrappedRhs.dims(), wrappedThis.ndims())
           && array_cmp(blk.strides + stride_start, r_blk.strides + stride_start, wrappedThis.ndims() - stride_start)
           && blk.inner_nblks == r_blk.inner_nblks
           && array_cmp(blk.inner_blks, r_blk.inner_blks, blk.inner_nblks)
           && array_cmp(blk.inner_idxs, r_blk.inner_idxs, blk.inner_nblks)
           && array_cmp(wrappedThis.padded_dims(), wrappedRhs.padded_dims(), wrappedRhs.ndims())
           && array_cmp(wrappedThis.padded_offsets(), wrappedRhs.padded_offsets(), wrappedThis.ndims())
           && dimsEqualWeak(wrappedThis.offset0(), wrappedRhs.offset0());
}


/**
 * Check compatibility with to BlockedMemoryDesc
 *
 * mkl:  IOhw_4i16o4i    dims {32, 64, 128, 128}
 *   strides               // the order of outer dims is encoded here
 *   inner_blks   4 16 4
 *   inner_idxs   1  0 1
 *
 * BlockedMemoryDesc desc has more expressive ability.
 * How to check compatibility with BlockedMemoryDesc representation:
 *    0. Detect a new_outer_order of outer_dims via descending strides.
 *    1. BlockedMemoryDesc strides :  concatenate strides in new_outer_order and inner strides.
 *    2. BlockedMemoryDesc dims    :  concatenate outer dims in new_outer_order with auto padding and inner blocks
 *    3. BlockedMemoryDesc order   :  concatenate new_outer_order and inner_idxs
 */

bool MKLDNNMemoryDesc::isCompatible(const BlockedMemoryDesc &rhs) const {
    if (this->getShape() != rhs.getShape() || this->getPrecision() != rhs.getPrecision()) {
        return false;
    }

    const auto dims = desc.dims();

    if (desc.data.format_kind != dnnl_blocked) {
        return false;
    }

    const auto &blk_desc = desc.data.format_desc.blocking;

    const size_t outer_ndims = dims.size();
    const size_t inner_ndims = blk_desc.inner_nblks;
    const size_t total_ndims = outer_ndims + inner_ndims;

    // strides of inner dims. In case of 4i16o4i will be {64, 4, 1}
    std::vector<size_t> inner_strides(inner_ndims, 1);
    for (size_t i = 1; i < blk_desc.inner_nblks; i++) {
        inner_strides[blk_desc.inner_nblks - 1 - i] = inner_strides[blk_desc.inner_nblks - i] * blk_desc.inner_blks[blk_desc.inner_nblks - i];
    }

    // total inner block size. in case of 4i16o4i will be {16, 16, 1, 1}
    std::vector<size_t> total_block_per_dim(outer_ndims, 1);
    for (int i = 0; i < inner_ndims; i++) {
        total_block_per_dim[blk_desc.inner_idxs[i]] *= blk_desc.inner_blks[i];
    }
    std::vector<size_t> outer_block_dims(std::begin(dims), std::begin(dims) + outer_ndims);
    for (size_t i = 0; i < outer_block_dims.size(); i++) {
        outer_block_dims[i] = div_up(outer_block_dims[i], total_block_per_dim[i]);
    }

    // order of outer dims. In case of IOhw_ will be {1, 0, 2, 3}
    std::vector<size_t> outer_order(outer_ndims);
    std::iota(outer_order.begin(), outer_order.end(), 0);
    std::sort(outer_order.begin(), outer_order.end(),
              [&blk_desc, &outer_block_dims] (size_t ind_l, size_t ind_r) {
                  return (blk_desc.strides[ind_l] > blk_desc.strides[ind_r]) ||
                         (blk_desc.strides[ind_l] == blk_desc.strides[ind_r] && outer_block_dims[ind_l] > outer_block_dims[ind_r]);
              });

    // blocked order
    // [new_outer_order] U [inner_idxs]
    SizeVector blk_order(total_ndims, 0);
    std::copy(outer_order.begin(), outer_order.end(), blk_order.begin());
    std::copy(blk_desc.inner_idxs, blk_desc.inner_idxs + blk_desc.inner_nblks, blk_order.begin() + dims.size());

    if (!isEqualOrUndefined(blk_order, rhs.getOrder())) {
        return false;
    }

    //TODO [DS]: undefined offset is also used now as an indicator of undefined strides
    if (desc.data.offset0 != Shape::UNDEFINED_DIM) {
        // blocked strides
        // [outer_strides via new_outer_order] U [inner_strides]
        SizeVector blk_strides(total_ndims, 0);
        std::copy(inner_strides.rbegin(), inner_strides.rend(), blk_strides.rbegin());
        std::transform(outer_order.begin(), outer_order.end(), blk_strides.begin(),
                       [&](size_t i) { return blk_desc.strides[i]; });

        size_t skipAxis = this->getShape().getRank() > 0 && this->getShape().getDims().front() == 1 ? 0 :
                Shape::UNDEFINED_DIM; //ignore batch axis if batch size == 1
        if (!isEqualOrUndefined(blk_strides, rhs.getStrides(), skipAxis)) {
            return false;
        }
    }

    // blocked dims
    // [dims via new_outer_order with auto pad] U [inner_blk_dims]
    SizeVector blk_dims(total_ndims, 0);
    std::copy(blk_desc.inner_blks, blk_desc.inner_blks + blk_desc.inner_nblks,
              blk_dims.end() - blk_desc.inner_nblks);
    std::transform(outer_order.begin(), outer_order.end(), blk_dims.begin(),
                   [&] (size_t i) { return outer_block_dims[i]; });

    if (!isEqualOrUndefined(blk_dims, rhs.getBlockDims())) {
        return false;
    }

    // offset padded to data. Same as for oneDNN
    SizeVector blk_offset_to_data {desc.data.padded_offsets, desc.data.padded_offsets + desc.data.ndims};
    // TODO: The BlockedMemoryDesc implementation allow to specify offset_to_data for inner blocked dims.
    //       Which is not obvious behavior. It required offset_to_data.size == total_ndims, so will
    //       fill it with zero.
    blk_offset_to_data.insert(blk_offset_to_data.end(), inner_ndims, 0);
    if (!isEqualOrUndefined(blk_offset_to_data, rhs.getOffsetPaddingToData())) {
        return false;
    }

    size_t blk_offset0 = desc.data.offset0;

    return dimsEqualWeak(blk_offset0, rhs.getOffsetPadding());
}

bool MKLDNNMemoryDesc::checkGeneralLayout(GeneralLayout layoutType) const {
    switch (layoutType) {
        case GeneralLayout::ncsp:
            return isPlainFormat();
        case GeneralLayout::nspc:
            return isTailCFormat();
        case GeneralLayout::nCsp8c:
            return isBlockedCFormat(8);
        case GeneralLayout::nCsp16c:
            return isBlockedCFormat(16);
        default:
            return false;
    }
}

std::string MKLDNNMemoryDesc::serializeFormat() const {
    if (desc.data.format_kind == dnnl_format_kind_wino) {
        switch (desc.data.format_desc.wino_desc.wino_format) {
            case dnnl_wino_memory_format_t::dnnl_wino_wei_aaOIoi: return "wino_aaOIoi";
            case dnnl_wino_memory_format_t::dnnl_wino_wei_aaOio: return "wino_aaOio";
            case dnnl_wino_memory_format_t::dnnl_wino_wei_aaOBiOo: return "wino_aaOBiOo";
            case dnnl_wino_memory_format_t::dnnl_wino_wei_OBaaIBOIio: return "wino_OBaaIBOIio";
            default: return "wino_undef";
        }
    }
    auto fmt = getFormat();
    return mkldnn::utils::fmt2str(fmt);
}

bool MKLDNNMemoryDesc::isDefined() const {
    return desc.data.offset0 != Shape::UNDEFINED_DIM;
}

InferenceEngine::Precision MKLDNNMemoryDesc::getPrecision() const {
    return MKLDNNExtensionUtils::DataTypeToIEPrecision(desc.data_type());
}

void MKLDNNMemoryDesc::setPrecision(InferenceEngine::Precision prc) {
    desc.data.data_type = static_cast<dnnl_data_type_t>(MKLDNNExtensionUtils::IEPrecisionToDataType(prc));
}
}  // namespace MKLDNNPlugin
