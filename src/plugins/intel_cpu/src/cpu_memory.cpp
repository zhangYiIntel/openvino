// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cpu_memory.h"
#include "memory_desc/cpu_memory_desc_utils.h"
#include <common/memory_desc_wrapper.hpp>
#include "nodes/reorder.h"
#include "utils/debug_capabilities.h"
#if defined(__linux__)
#    include <sys/syscall.h> /* Definition of SYS_* constants */
#    include <unistd.h>
#    include <cstring> /* strerror(errno) */
#endif

namespace ov {
namespace intel_cpu {
template <>
DnnlMemoryDescPtr IMemory::getDescWithType<DnnlMemoryDesc, 0, 0>() const {
    return MemoryDescUtils::convertToDnnlMemoryDesc(getDescPtr());
}

template <>
BlockedMemoryDescPtr IMemory::getDescWithType<BlockedMemoryDesc, 0, 0>() const {
    return MemoryDescUtils::convertToBlockedMemoryDesc(getDescPtr());
}

namespace {
    inline void setSubnormalsToZero(float *data, size_t size) {
        uint32_t *u32data = reinterpret_cast<uint32_t *>(data);
        for (size_t i = 0; i < size; ++i) {
            if ((u32data[i] & (0xFF << 23)) == 0) {
                u32data[i] = 0;
            }
        }
    }

    void transferData(const IMemory& src, const IMemory& dst, bool ftz) {
        node::Reorder::reorderData(src, dst);

        if (!ftz) {
            return;
        }
        if (src.getDesc().getPrecision() != ov::element::f32 || dst.getDesc().getPrecision() == ov::element::bf16) {
            return;
        }
        size_t offset = 0;
        if (dst.getDesc().getType() & MemoryDescType::Dnnl) {
            // here we can safely cast to DnnlMemoryDesc
            auto dnnl_desc = dst.getDescWithType<DnnlMemoryDesc>();
            auto desc = dnnl_desc->getDnnlDesc();
            dnnl::impl::memory_desc_wrapper wrapper(desc.get());
            offset = wrapper.offset0();
            if (wrapper.is_wino_desc() || wrapper.is_rnn_packed_desc()) {
                return;
            }
        }
        // actual FTZ
        auto* memData = static_cast<float*>(dst.getData());
        memData += offset;
        setSubnormalsToZero(memData, dst.getSize() / sizeof(float));
    }

}   // namespace

Memory::Memory(const dnnl::engine& eng, MemoryDescPtr desc, const void* data, bool pads_zeroing) :
    m_eng(eng),
    m_pMemDesc(desc),
    m_mgrHandle(std::make_shared<DnnlMemoryMngr>(make_unique<MemoryMngrWithReuse>()), this),
    dnnlMemHandle(this) {
        if (desc->getPrecision() == element::string) {
            OPENVINO_THROW("[CPU] Memory object cannot be created for string data.");
        }
        create(m_pMemDesc, data, pads_zeroing);
    }

Memory::Memory(const dnnl::engine& eng, const MemoryDesc& desc, const void* data, bool pads_zeroing) :
    Memory::Memory(eng, desc.clone(), data, pads_zeroing) {}

Memory::Memory(const dnnl::engine& eng, MemoryDescPtr desc, MemoryMngrPtr mngr) :
    m_eng(eng), m_pMemDesc(desc), m_mgrHandle(mngr, this), dnnlMemHandle(this) {
        if (desc->getPrecision() == element::string) {
            OPENVINO_THROW("[CPU] Memory object can't be created for string data.");
        }
        bool memAllocated = m_mgrHandle->getRawPtr();

        create(desc, nullptr, !memAllocated);
    }

Memory::Memory(const dnnl::engine& eng, const MemoryDesc& desc, MemoryMngrPtr mngr) :
    Memory::Memory(eng, desc.clone(), mngr) {}

size_t Memory::getSize() const {
    auto size = getDesc().getCurrentMemSize();
    if (size == MemoryDesc::UNDEFINED_SIZE) {
        OPENVINO_THROW("Can't get memory size for undefined shape");
    }
    return size;
}

void Memory::create(const MemoryDesc &desc, const void *data, bool pads_zeroing) {
    create(desc.clone(), data, pads_zeroing);
}

void Memory::create(MemoryDescPtr desc, const void* data, bool pads_zeroing) {
    m_pMemDesc = desc;
    m_padsZeroing = pads_zeroing;
    dnnlMemHandle.resetDnnlPrim();

    if (!m_pMemDesc->isDefined()) {
        return;
    }
    auto memSize = m_pMemDesc->getCurrentMemSize();
    if (nullptr != data) {
        m_mgrHandle->setExtBuff(const_cast<void*>(data), memSize);
    } else {
        m_mgrHandle->resize(memSize);
    }
}

void Memory::load(const IMemory& src, bool ftz) const {
    if (src.getDesc().getPrecision() == element::string) {
        OPENVINO_THROW("[CPU] Memory object cannot load string data.");
    }
    transferData(src, *this, ftz);
}

void Memory::nullify() {
    void* dataPtr = getData();
    if (dataPtr != nullptr)
        memset(dataPtr, 0, getDesc().getCurrentMemSize());
}

void Memory::redefineDesc(MemoryDescPtr desc) {
    if (desc->getPrecision() == element::string) {
        OPENVINO_THROW("[CPU] Memory object cannot accept a descriptor with a string type.");
    }
    if (!desc->hasDefinedMaxSize()) {
        OPENVINO_THROW("Can not reset descriptor, memory upper bound is unknown.");
    }

    this->create(desc, nullptr, false);
}

void Memory::update() {
    if (dnnlMemHandle.isInit()) {
        auto prim = dnnlMemHandle.getPrim();
        prim.set_data_handle(m_mgrHandle->getRawPtr());
    }
}

dnnl::memory Memory::getPrimitive() const {
    return dnnlMemHandle.getPrim();
}

void Memory::DnnlMemPrimHandle::resetDnnlPrim() {
    m_prim = dnnl::memory();
}

bool Memory::DnnlMemPrimHandle::isInit() const {
    std::lock_guard<std::mutex> guard(m_primCachingLock);
    return m_prim.get(true) != nullptr;
}

dnnl::memory Memory::DnnlMemPrimHandle::getPrim() const {
    std::lock_guard<std::mutex> guard(m_primCachingLock);
    if (!m_prim) {
        if (!m_memObjPtr->getDesc().isDefined()) {
            OPENVINO_THROW("Can not create oneDNN memory from undefined memory descriptor");
        }

        // ========================
        // Equivalent of constructor memory(const primitive_desc &desc, void *hdl)
        // but with ability to skip pads zeroing.
        auto desc = MemoryDescUtils::convertToDnnlMemoryDesc(m_memObjPtr->getDescPtr());
        m_prim = dnnl::memory(desc->getDnnlDesc(), m_memObjPtr->getEngine(), DNNL_MEMORY_NONE);
        //
        // ========================
        auto data = m_memObjPtr->getDataNoThrow();
        if (data != nullptr) {
            m_prim.set_data_handle(data);
        }
    }
    return m_prim;
}

bool Memory::isAllocated() const noexcept {
    if (m_mgrHandle->getRawPtr()) {
        return true;
    }
    if (!m_pMemDesc) {
        return false;
    }
    if (!(m_pMemDesc->isDefined())) {
        return true;
    }
    if (m_pMemDesc->getCurrentMemSize() == 0) {
        return true;
    }
    return false;
}

void* Memory::getData() const {
    void* data = getDataNoThrow();
    if (data == nullptr &&
        m_pMemDesc->getShape().isStatic() &&
        m_pMemDesc->getShape().getElementsCount() != 0)
        OPENVINO_THROW("Memory has not been allocated");
    return data;
}

void* MemoryMngrWithReuse::getRawPtr() const noexcept {
    return m_data.get();
}

void MemoryMngrWithReuse::setExtBuff(void *ptr, size_t size) {
    m_useExternalStorage = true;
    m_memUpperBound = size;
    m_data = decltype(m_data)(ptr, release);
}

bool MemoryMngrWithReuse::resize(size_t size) {
    constexpr int cacheLineSize = 64;
    bool sizeChanged = false;
    if (size > m_memUpperBound) {
        void *ptr = dnnl::impl::malloc(size, cacheLineSize);
        if (!ptr) {
            OPENVINO_THROW("Failed to allocate ", size, " bytes of memory");
        }
        m_memUpperBound = size;
        m_useExternalStorage = false;
        m_data = decltype(m_data)(ptr, destroy);
        sizeChanged = true;

        if (numa_node >= 0) {
            if (!mbind_move(ptr, size, numa_node)) {
                DEBUG_LOG("MemoryMngrWithReuse move_memory to node ", numa_node, " failed\n");
            }
        }
    }
    return sizeChanged;
}

bool MemoryMngrWithReuse::hasExtBuffer() const noexcept {
    return m_useExternalStorage;
}

void MemoryMngrWithReuse::release(void *ptr) {}

void MemoryMngrWithReuse::destroy(void *ptr) {
    dnnl::impl::free(ptr);
}

void* MemoryMngrRealloc::getRawPtr() const noexcept {
    return m_data.get();
}

void MemoryMngrRealloc::setExtBuff(void *ptr, size_t size) {
    m_useExternalStorage = true;
    m_memUpperBound = size;
    m_data = decltype(m_data)(ptr, release);
}

bool MemoryMngrRealloc::resize(size_t size) {
    constexpr int cacheLineSize = 64;
    constexpr size_t growFactor = 2;
    bool sizeChanged = false;
    if (size > m_memUpperBound) {
        size *= growFactor;
        void *ptr = dnnl::impl::malloc(size, cacheLineSize);
        if (!ptr) {
            OPENVINO_THROW("Failed to allocate ", size, " bytes of memory");
        }

        if (auto src = m_data.get()) {
            std::memcpy(ptr, src, m_memUpperBound);
        }

        m_memUpperBound = size;
        m_useExternalStorage = false;
        m_data = decltype(m_data)(ptr, destroy);
        sizeChanged = true;
    }
    return sizeChanged;
}

bool MemoryMngrRealloc::hasExtBuffer() const noexcept {
    return m_useExternalStorage;
}

void MemoryMngrRealloc::release(void *ptr) {}

void MemoryMngrRealloc::destroy(void *ptr) {
    dnnl::impl::free(ptr);
}

/////////////// StringMemory ///////////////

StringMemory::StringMemory(const dnnl::engine& engine, const MemoryDescPtr& desc, const void* data) : m_engine(engine), m_mem_desc(desc) {
    if (m_mem_desc->getPrecision() != element::string) {
        OPENVINO_THROW("[CPU] StringMemory supports String type only.");
    }

    m_manager = std::make_shared<StringMemoryMngr>();

    if (!m_mem_desc->isDefined()) {
        return;
    }

    const auto string_size = m_mem_desc->getShape().getElementsCount();

    if (data != nullptr) {
        auto not_const_data = const_cast<void *>(data);
        m_manager->setExtBuff(reinterpret_cast<OvString *>(not_const_data), string_size);
    } else {
        m_manager->resize(string_size);
    }
}

void StringMemory::load(const IMemory& src, bool ftz) const {
    if (src.getDesc().getPrecision() != element::string) {
        OPENVINO_THROW("[CPU] String memory cannot load a non-string object.");
    }

    transferData(src, *this, false);
}

void* StringMemory::getData() const  {
    return m_manager->getRawPtr();
}

void StringMemory::redefineDesc(MemoryDescPtr desc) {
    if (desc->getPrecision() != element::string) {
        OPENVINO_THROW("[CPU] StringMemory supports String type only.");
    }
    if (!desc->hasDefinedMaxSize()) {
        OPENVINO_THROW("[CPU] StringMemory cannot reset descriptor. Memory upper bound is unknown.");
    }

    m_mem_desc = desc;
    const auto string_size = m_mem_desc->getShape().getElementsCount();
    m_manager->resize(string_size);
}

void StringMemory::nullify() {
    auto data_ptr = m_manager->getStringPtr();
    if (data_ptr != nullptr) {
        std::fill(data_ptr, data_ptr + m_manager->getStrLen(), OvString());
    }
}

bool StringMemory::isAllocated() const noexcept {
    if (getData()) {
        return true;
    }
    if (!m_mem_desc) {
        return false;
    }
    if (!(m_mem_desc->isDefined())) {
        return true;
    }
    if (m_mem_desc->getCurrentMemSize() == 0) {
        return true;
    }
    return false;
}

size_t StringMemory::getSize() const { // In bytes
    auto size = getDesc().getCurrentMemSize();
    if (size == MemoryDesc::UNDEFINED_SIZE) {
        OPENVINO_THROW("Can't get memory size for undefined shape.");
    }
    return size;
}

MemoryMngrPtr StringMemory::getMemoryMngr() const {
    OPENVINO_THROW("Unexpected call of StringMemory::getMemoryMngr()");
}

dnnl::memory StringMemory::getPrimitive() const {
    OPENVINO_THROW("Unexpected call of StringMemory::getPrimitive()");
}

void StringMemory::StringMemoryMngr::setExtBuff(OvString* ptr, size_t size) {
    m_use_external_storage = true;
    m_str_upper_bound = size;
    m_data = decltype(m_data)(ptr, release);
}

StringMemory::OvString* StringMemory::StringMemoryMngr::getStringPtr() const noexcept {
    return m_data.get();
}

bool StringMemory::StringMemoryMngr::resize(size_t size) {
    bool sizeChanged = false;
    if (size > m_str_upper_bound) {
        if (size > PTRDIFF_MAX) {
            OPENVINO_THROW("Requested allocation size { ", size, " } exceeds PTRDIFF_MAX.");
        }
        auto ptr_size = static_cast<ptrdiff_t>(size); // WA for warning alloc-size-larger-than
        auto ptr = new OvString[ptr_size];
        if (!ptr) {
            OPENVINO_THROW("Failed to allocate ", size, " bytes of memory");
        }
        m_str_upper_bound = size;
        m_use_external_storage = false;
        m_data = decltype(m_data)(ptr, destroy);
        sizeChanged = true;
    }
    return sizeChanged;
}

bool StringMemory::StringMemoryMngr::hasExtBuffer() const noexcept {
    return m_use_external_storage;
}

size_t StringMemory::StringMemoryMngr::getStrLen() const noexcept {
    return m_str_upper_bound;
}

void StringMemory::StringMemoryMngr::destroy(OvString* ptr) {
    delete[] ptr;
}

void* StringMemory::StringMemoryMngr::getRawPtr() const noexcept {
    return reinterpret_cast<void *>(m_data.get());
}

/////////////// DnnlMemoryMngr ///////////////

void* DnnlMemoryMngr::getRawPtr() const noexcept {
    return m_pMemMngr->getRawPtr();
}

void DnnlMemoryMngr::setExtBuff(void *ptr, size_t size) {
    m_pMemMngr->setExtBuff(ptr, size);
    notifyUpdate();
}

bool DnnlMemoryMngr::resize(size_t size) {
    bool sizeChanged = m_pMemMngr->resize(size);
    if (sizeChanged) {
        notifyUpdate();
    }
    return sizeChanged;
}

bool DnnlMemoryMngr::hasExtBuffer() const noexcept {
    return m_pMemMngr->hasExtBuffer();
}

void DnnlMemoryMngr::registerMemory(Memory* memPtr) {
    if (memPtr) {
        m_setMemPtrs.insert(memPtr);
    }
}

void DnnlMemoryMngr::unregisterMemory(Memory* memPtr) {
    if (memPtr) {
        m_setMemPtrs.erase(memPtr);
    }
}

void DnnlMemoryMngr::notifyUpdate() {
    for (auto& item : m_setMemPtrs) {
        if (item) {
            item->update();
        }
    }
}

StaticMemory::StaticMemory(const dnnl::engine& eng, MemoryDescPtr desc, const void* data, bool pads_zeroing) :
    m_eng(eng), m_pMemDesc(desc) {
    if (desc->getPrecision() == element::string) {
        OPENVINO_THROW("[CPU] StaticMemory object cannot be created for string data.");
    }
    if (!m_pMemDesc->isDefined()) {
        OPENVINO_THROW("Can not create StaticMemory object. The memory desc is undefined");
    }

    m_size = m_pMemDesc->getCurrentMemSize();

    if (data) {
        m_pMemMngr = std::make_shared<StaticMemoryMngr>(const_cast<void*>(data), m_size);
    } else {
        m_pMemMngr = std::make_shared<StaticMemoryMngr>(m_size);
    }

    try {
        auto dnnl_desc = MemoryDescUtils::convertToDnnlMemoryDesc(m_pMemDesc);
        // ========================
        // Equivalent of constructor memory(const primitive_desc &desc, void *hdl)
        // but with ability to skip pads zeroing.
        m_prim = dnnl::memory(dnnl_desc->getDnnlDesc(), m_eng, DNNL_MEMORY_NONE);
        //
        // ========================
        m_prim.set_data_handle(m_pMemMngr->getRawPtr());
    }
    catch (const std::exception& exc) {
        dnnlErrorCtx = exc.what();
    }
}

StaticMemory::StaticMemory(const dnnl::engine& eng, const MemoryDesc& desc, const void* data, bool pads_zeroing) :
    StaticMemory::StaticMemory(eng, desc.clone(), data, pads_zeroing) {}

bool StaticMemory::isAllocated() const noexcept {
    return 0 == m_size || getData() != nullptr;
}

const MemoryDesc& StaticMemory::getDesc() const {
    return *m_pMemDesc;
}

MemoryDescPtr StaticMemory::getDescPtr() const {
    return m_pMemDesc;
}

void* StaticMemory::getData() const {
    return m_pMemMngr->getRawPtr();
}

size_t StaticMemory::getSize() const {
    return m_size;
}

const Shape& StaticMemory::getShape() const {
    return m_pMemDesc->getShape();
}

const VectorDims& StaticMemory::getStaticDims() const {
    return getShape().getStaticDims();
}

void StaticMemory::redefineDesc(MemoryDescPtr desc) {
    OPENVINO_THROW("Unexpected: Memory descriptor may not be modified in StaticMemory object");
}

void StaticMemory::load(const IMemory& src, bool ftz) const {
    if (src.getDesc().getPrecision() == element::string) {
        OPENVINO_THROW("[CPU] StaticMemory cannot load string data.");
    }
    transferData(src, *this, ftz);
}

MemoryMngrPtr StaticMemory::getMemoryMngr() const {
    return m_pMemMngr;
}

//oneDNN specifics for backward compatibility
dnnl::memory StaticMemory::getPrimitive() const {
    if (!m_prim) {
        OPENVINO_THROW("Couldn't create dnnl::memory object: ", dnnlErrorCtx);
    }
    return m_prim;
}

void StaticMemory::nullify() {
    void* dataPtr = getData();
    if (dataPtr != nullptr)
        memset(dataPtr, 0, getSize());
}

StaticMemory::StaticMemoryMngr::StaticMemoryMngr(size_t size) : m_size(size) {
    memMngrImpl.resize(m_size);
}

StaticMemory::StaticMemoryMngr::StaticMemoryMngr(void* data, size_t size) : m_size(size) {
    memMngrImpl.setExtBuff(data, m_size);
}

void* StaticMemory::StaticMemoryMngr::getRawPtr() const noexcept {
    return memMngrImpl.getRawPtr();
}

void StaticMemory::StaticMemoryMngr::setExtBuff(void* ptr, size_t size) {
    OPENVINO_THROW("Unexpected: StaticMemoryMngr may not be modified");
}

bool StaticMemory::StaticMemoryMngr::resize(size_t size) {
    if (size != m_size) {
        OPENVINO_THROW("Unexpected: StaticMemoryMngr may not resize the memory");
    }
    return false;
}

bool StaticMemory::StaticMemoryMngr::hasExtBuffer() const noexcept {
    return memMngrImpl.hasExtBuffer();
}

void StaticMemory::StaticMemoryMngr::registerMemory(Memory* memPtr) {
    //do nothing
}

void StaticMemory::StaticMemoryMngr::unregisterMemory(Memory* memPtr) {
    //do nothing
}

#if defined(__linux__)
#    define MPOL_DEFAULT   0
#    define MPOL_BIND      2
#    define MPOL_MF_STRICT (1 << 0)
#    define MPOL_MF_MOVE   (1 << 1)
#if !defined(__NR_mbind) && defined(__x86_64__)
#    define __NR_mbind 237
#endif
static long mbind(void* start,
                  unsigned long len,
                  int mode,
                  const unsigned long* nmask,
                  unsigned long maxnode,
                  unsigned flags) {
    return syscall(__NR_mbind, (long)start, len, mode, (long)nmask, maxnode, flags);
}
#endif

#if defined(__linux__)
bool mbind_move(void* data, size_t size, int targetNode) {
    int realNode = ov::get_org_numa_id(targetNode);
    auto pagesize = getpagesize();
    auto page_count = (size + pagesize - 1) / pagesize;
    char* pages = reinterpret_cast<char*>((((uintptr_t)data) & ~((uintptr_t)(pagesize - 1))));
    unsigned long mask = 0;
    unsigned flags = 0;
    if (realNode < 0) {
        // restore default policy
        mask = -1;
        flags = 0;
    } else {
        mask = 1ul << realNode;
        flags = MPOL_MF_MOVE | MPOL_MF_STRICT;
    }

    auto rc = mbind(pages, page_count * pagesize, MPOL_BIND, &mask, sizeof(mask) * 8, flags);
    if (rc < 0) {
        DEBUG_LOG("mbind failed: ", strerror(errno));
        return false;
    }
    return true;
}
#else
bool mbind_move(void* data, size_t size, int targetNode) {
    return false;
}
#endif

bool mbind_move(const MemoryCPtr mem, int numaNodeID) {
    void* data = mem->getData();
    auto size = mem->getSize();
    return mbind_move(data, size, numaNodeID);
}

bool mbind_move(const dnnl::memory mem, int numaNodeID) {
    void* data = mem.get_data_handle();
    auto desc = mem.get_desc();
    auto size = desc.get_size();
    return mbind_move(data, size, numaNodeID);
}

}   // namespace intel_cpu
}   // namespace ov
