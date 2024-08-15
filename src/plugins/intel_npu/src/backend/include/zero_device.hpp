// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ze_api.h>
#include <ze_graph_ext.h>

#include "intel_npu/al/icompiled_model.hpp"
#include "intel_npu/utils/logger/logger.hpp"
#include "npu.hpp"
#include "openvino/runtime/intel_npu/remote_properties.hpp"
#include "zero_init.hpp"
#include "zero_types.hpp"

namespace intel_npu {

class ZeroDevice : public IDevice {
public:
    ZeroDevice(const std::shared_ptr<ZeroInitStructsHolder>& initStructs);

    std::shared_ptr<IExecutor> createExecutor(const std::shared_ptr<const NetworkDescription>& networkDescription,
                                              const Config& config) override;

    std::string getName() const override;
    std::string getFullDeviceName() const override;
    Uuid getUuid() const override;
    uint32_t getSubDevId() const override;
    uint32_t getMaxNumSlices() const override;
    uint64_t getAllocMemSize() const override;
    uint64_t getTotalMemSize() const override;
    ov::device::PCIInfo getPciInfo() const override;
    std::map<ov::element::Type, float> getGops() const override;
    ov::device::Type getDeviceType() const override;

    std::shared_ptr<SyncInferRequest> createInferRequest(const std::shared_ptr<const ICompiledModel>& compiledModel,
                                                         const std::shared_ptr<IExecutor>& executor,
                                                         const Config& config) override;
    void updateInfo(const Config& config) override {
        log.setLevel(config.get<LOG_LEVEL>());
    }

    ov::SoPtr<ov::IRemoteTensor> createRemoteTensor(
        std::shared_ptr<ov::IRemoteContext> context,
        const ov::element::Type& element_type,
        const ov::Shape& shape,
        const Config& config,
        ov::intel_npu::TensorType tensor_type = ov::intel_npu::TensorType::BINDED,
        ov::intel_npu::MemType mem_type = ov::intel_npu::MemType::L0_INTERNAL_BUF,
        void* mem = nullptr) override;

    ov::SoPtr<ov::ITensor> createHostTensor(std::shared_ptr<ov::IRemoteContext> context,
                                            const ov::element::Type& element_type,
                                            const ov::Shape& shape,
                                            const Config& config) override;

    ZeroDevice& operator=(const ZeroDevice&) = delete;
    ZeroDevice(const ZeroDevice&) = delete;

private:
    const std::shared_ptr<ZeroInitStructsHolder> _initStructs;

    ze_graph_dditable_ext_curr_t* _graph_ddi_table_ext = nullptr;

    ze_device_properties_t device_properties = {};

    ze_pci_ext_properties_t pci_properties = {};

    std::map<ov::element::Type, float> device_gops = {{ov::element::f32, 0.f},
                                                      {ov::element::f16, 0.f},
                                                      {ov::element::u8, 0.f},
                                                      {ov::element::i8, 0.f}};

    uint32_t _group_ordinal;

    Logger log;
};
}  // namespace intel_npu
