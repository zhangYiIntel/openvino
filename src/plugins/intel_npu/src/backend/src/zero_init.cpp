// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "zero_init.hpp"

#include "intel_npu/al/itt.hpp"
#include "intel_npu/utils/zero/zero_api.hpp"
#include "ze_api.h"
#include "ze_command_queue_npu_ext.h"
#include "zero_utils.hpp"

#ifdef _WIN32
namespace {
constexpr uint32_t WIN_DRIVER_NO_MCL_SUPPORT = 2688;
}  // namespace
#endif

namespace intel_npu {

const ze_driver_uuid_t ZeroInitStructsHolder::uuid = ze_intel_npu_driver_uuid;

static std::tuple<uint32_t, std::string> queryDriverExtensionVersion(
    std::vector<ze_driver_extension_properties_t>& extProps,
    uint32_t count) {
    // query the extension properties
    const char* graphExtName = nullptr;
    uint32_t targetVersion = 0;
    for (uint32_t i = 0; i < count; ++i) {
        auto& property = extProps[i];

        if (strncmp(property.name, ZE_GRAPH_EXT_NAME, strlen(ZE_GRAPH_EXT_NAME)) != 0)
            continue;

        // If the driver version is latest, will just use its name.
        if (property.version == ZE_GRAPH_EXT_VERSION_CURRENT) {
            graphExtName = property.name;
            targetVersion = property.version;
            break;
        }

        // Use the latest version supported by the driver.
        if (property.version > targetVersion) {
            graphExtName = property.name;
            targetVersion = property.version;
        }
    }

    if (graphExtName == nullptr) {
        OPENVINO_THROW("queryDriverExtensionVersion: Failed to find Graph extension in NPU Driver");
    }

    const uint16_t supportedDriverExtMajorVersion = 1;
    const uint16_t driverExtMajorVersion = ZE_MAJOR_VERSION(targetVersion);
    if (supportedDriverExtMajorVersion != driverExtMajorVersion) {
        OPENVINO_THROW("Plugin supports only driver with extension major version ",
                       supportedDriverExtMajorVersion,
                       "; discovered driver extension has major version ",
                       driverExtMajorVersion);
    }

    return std::make_tuple(targetVersion, graphExtName);
}

static uint32_t queryMutableCommandListVersion(std::vector<ze_driver_extension_properties_t>& extProps,
                                               uint32_t count) {
    // query the mutable command list properties
    uint32_t targetVersion = 0;

    for (uint32_t i = 0; i < count; ++i) {
        auto& property = extProps[i];

        if (strncmp(property.name, ZE_MUTABLE_COMMAND_LIST_EXP_NAME, strlen(ZE_MUTABLE_COMMAND_LIST_EXP_NAME)) != 0) {
            continue;
        }

        if (property.version == ZE_MUTABLE_COMMAND_LIST_EXP_VERSION_CURRENT) {
            targetVersion = property.version;
            break;
        }

        // Use the latest version supported by the driver.
        if (property.version > targetVersion) {
            targetVersion = property.version;
        }
    }

    return targetVersion;
}

ZeroInitStructsHolder::ZeroInitStructsHolder() : log("NPUZeroInitStructsHolder", Logger::global().level()) {
    OV_ITT_SCOPED_TASK(itt::domains::LevelZeroBackend, "ZeroInitStructsHolder::ZeroInitStructsHolder");
    log.debug("ZeroInitStructsHolder - performing zeInit on VPU only");
    zeroUtils::throwOnFail("zeInit", zeInit(ZE_INIT_FLAG_VPU_ONLY));

    uint32_t drivers = 0;
    zeroUtils::throwOnFail("zeDriverGet", zeDriverGet(&drivers, nullptr));

    std::vector<ze_driver_handle_t> all_drivers(drivers);
    zeroUtils::throwOnFail("zeDriverGet", zeDriverGet(&drivers, all_drivers.data()));

    // Get our target driver
    driver_properties.stype = ZE_STRUCTURE_TYPE_DRIVER_PROPERTIES;
    log.debug("ZeroInitStructsHolder - setting driver properties to ZE_STRUCTURE_TYPE_DRIVER_PROPERTIES");
    for (uint32_t i = 0; i < drivers; ++i) {
        zeDriverGetProperties(all_drivers[i], &driver_properties);

        if (memcmp(&driver_properties.uuid, &uuid, sizeof(uuid)) == 0) {
            driver_handle = all_drivers[i];
            break;
        }
    }
    if (driver_handle == nullptr) {
        OPENVINO_THROW("zeDriverGet failed to return NPU driver");
    }

    // Check L0 API version
    ze_api_version_t ze_drv_api_version = {};
    zeroUtils::throwOnFail("zeDriverGetApiVersion", zeDriverGetApiVersion(driver_handle, &ze_drv_api_version));

    if (ZE_MAJOR_VERSION(ZE_API_VERSION_CURRENT) != ZE_MAJOR_VERSION(ze_drv_api_version)) {
        OPENVINO_THROW("Incompatibility between NPU plugin and driver! ",
                       "Plugin L0 API major version = ",
                       ZE_MAJOR_VERSION(ZE_API_VERSION_CURRENT),
                       ", ",
                       "Driver L0 API major version = ",
                       ZE_MAJOR_VERSION(ze_drv_api_version));
    }
    if (ZE_MINOR_VERSION(ZE_API_VERSION_CURRENT) != ZE_MINOR_VERSION(ze_drv_api_version)) {
        log.debug("Some features might not be available! "
                  "Plugin L0 API minor version = %d, Driver L0 API minor version = %d",
                  ZE_MINOR_VERSION(ZE_API_VERSION_CURRENT),
                  ZE_MINOR_VERSION(ze_drv_api_version));
    }

    uint32_t count = 0;
    zeroUtils::throwOnFail("zeDriverGetExtensionProperties",
                           zeDriverGetExtensionProperties(driver_handle, &count, nullptr));

    std::vector<ze_driver_extension_properties_t> extProps;
    extProps.resize(count);
    zeroUtils::throwOnFail("zeDriverGetExtensionProperties",
                           zeDriverGetExtensionProperties(driver_handle, &count, extProps.data()));

    // Query our graph extension version
    std::string graph_ext_name;
    log.debug("ZeroInitStructsHolder - tie output of queryDriverExtensionVersion");
    std::tie(driver_ext_version, graph_ext_name) = queryDriverExtensionVersion(extProps, count);

    log.debug("Found Driver Version %d.%d, Driver Extension Version %d.%d (%s)",
              ZE_MAJOR_VERSION(ze_drv_api_version),
              ZE_MINOR_VERSION(ze_drv_api_version),
              ZE_MAJOR_VERSION(driver_ext_version),
              ZE_MINOR_VERSION(driver_ext_version),
              graph_ext_name.c_str());

    // Load our command queue extension
    try {
        zeroUtils::throwOnFail(
            "zeDriverGetExtensionFunctionAddress " + std::string(ZE_COMMAND_QUEUE_NPU_EXT_NAME),
            zeDriverGetExtensionFunctionAddress(driver_handle,
                                                ZE_COMMAND_QUEUE_NPU_EXT_NAME,
                                                reinterpret_cast<void**>(&_command_queue_npu_dditable_ext)));
    } catch (const ov::Exception& error) {
        log.debug("Current Driver Version does not have the command queue extension: %s", error.what());
    }

    // Load our graph extension
    ze_graph_dditable_ext_last_t* graph_ddi_table_ext = nullptr;
    zeroUtils::throwOnFail("zeDriverGetExtensionFunctionAddress",
                           zeDriverGetExtensionFunctionAddress(driver_handle,
                                                               graph_ext_name.c_str(),
                                                               reinterpret_cast<void**>(&graph_ddi_table_ext)));
    graph_dditable_ext_decorator =
        std::make_unique<ze_graph_dditable_ext_decorator>(graph_ddi_table_ext, driver_ext_version);

    // Query the mutable command list version
#ifdef _WIN32
    // The 2688 Windows driver version doesn't support as expected the MutableCommandList feature
    if (driver_properties.driverVersion != WIN_DRIVER_NO_MCL_SUPPORT) {
#endif
        log.debug("ZeroInitStructsHolder - tie output of queryMutableCommandListVersion");
        mutable_command_list_version = queryMutableCommandListVersion(extProps, count);
#ifdef _WIN32
    }
#endif

    log.debug("Mutable command list version %d.%d",
              ZE_MAJOR_VERSION(mutable_command_list_version),
              ZE_MINOR_VERSION(mutable_command_list_version));

    // Load our profiling extension
    zeroUtils::throwOnFail(
        "zeDriverGetExtensionFunctionAddress",
        zeDriverGetExtensionFunctionAddress(driver_handle,
                                            "ZE_extension_profiling_data",
                                            reinterpret_cast<void**>(&_graph_profiling_ddi_table_ext)));

    uint32_t device_count = 1;
    // Get our target device
    zeroUtils::throwOnFail("zeDeviceGet", zeDeviceGet(driver_handle, &device_count, &device_handle));

    ze_context_desc_t context_desc = {ZE_STRUCTURE_TYPE_CONTEXT_DESC, 0, 0};
    zeroUtils::throwOnFail("zeContextCreate", zeContextCreate(driver_handle, &context_desc, &context));
    log.debug("ZeroInitStructsHolder initialize complete");
}

ZeroInitStructsHolder::~ZeroInitStructsHolder() {
    if (context) {
        log.debug("ZeroInitStructsHolder - performing zeContextDestroy");
        auto result = zeContextDestroy(context);
        if (ZE_RESULT_SUCCESS != result) {
            log.error("zeContextDestroy failed %#X", uint64_t(result));
        }
    }
}

}  // namespace intel_npu
