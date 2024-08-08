// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compiled_model.hpp"

#include <fstream>
#include <string_view>

#include "async_infer_request.hpp"
#include "intel_npu/al/config/common.hpp"
#include "intel_npu/al/config/compiler.hpp"
#include "intel_npu/al/config/config.hpp"
#include "intel_npu/al/config/runtime.hpp"
#include "intel_npu/al/icompiler.hpp"
#include "intel_npu/al/itt.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/system_conf.hpp"
#include "openvino/runtime/threading/executor_manager.hpp"
#include "transformations/utils/utils.hpp"

namespace {

constexpr std::string_view NO_EXECUTOR_FOR_INFERENCE =
    "Can't create infer request!\n"
    "Please make sure that the device is available. Only exports can be made.";

std::uint32_t hash(const std::vector<uint8_t>& data) {
    std::uint32_t result = 1171117u;
    for (const auto& c : data)
        result = ((result << 7) + result) + static_cast<uint32_t>(c);
    return result;
}

}  // namespace

namespace intel_npu {

using intel_npu::envVarStrToBool;

CompiledModel::CompiledModel(const std::shared_ptr<const ov::Model>& model,
                             const std::shared_ptr<const ov::IPlugin>& plugin,
                             const std::shared_ptr<IDevice>& device,
                             const ov::SoPtr<ICompiler>& compiler,
                             const bool profiling,
                             const Config& config)
    : ICompiledModel(model, plugin),
      _model(model),
      _config(config),
      _logger("CompiledModel", config.get<LOG_LEVEL>()),
      _device(device),
      _compiler(profiling ? std::optional(compiler) : std::nullopt) {
    OV_ITT_SCOPED_TASK(itt::domains::NPUPlugin, "CompiledModel::CompiledModel");
    OPENVINO_ASSERT(compiler != nullptr, "NPU CompiledModel: the pointer towards the compiler object is null");

    try {
        _networkPtr = std::make_shared<const NetworkDescription>(compiler->compile(model, config));
    } catch (const std::exception& ex) {
        OPENVINO_THROW(ex.what());
    } catch (...) {
        _logger.error("Unexpected exception");
        OPENVINO_THROW("NPU CompiledModel: got an unexpected exception from compiler");
    }

    OV_ITT_TASK_CHAIN(COMPILED_MODEL, itt::domains::NPUPlugin, "CompiledModel::CompiledModel", "initialize_properties");
    initialize_properties();
    configure_stream_executors();

    OV_ITT_TASK_NEXT(COMPILED_MODEL, "create_executor");
    create_executor();

    OV_ITT_TASK_SKIP(COMPILED_MODEL);
}

CompiledModel::CompiledModel(const std::shared_ptr<const ov::Model>& model,
                             const std::shared_ptr<const ov::IPlugin>& plugin,
                             const std::shared_ptr<const NetworkDescription>& networkDescription,
                             const std::shared_ptr<IDevice>& device,
                             const std::optional<ov::SoPtr<ICompiler>>& compiler,
                             const Config& config)
    : ICompiledModel(model, plugin),
      _networkPtr(networkDescription),
      _model(model),
      _config(config),
      _logger("CompiledModel", config.get<LOG_LEVEL>()),
      _device(device),
      _compiler(compiler) {
    OV_ITT_SCOPED_TASK(itt::domains::NPUPlugin, "CompiledModel::CompiledModel");
    OPENVINO_ASSERT(_networkPtr != nullptr,
                    "NPU CompiledModel: the pointer towards the NetworkDescription object is null");

    OV_ITT_TASK_CHAIN(COMPILED_MODEL, itt::domains::NPUPlugin, "CompiledModel::CompiledModel", "initialize_properties");
    initialize_properties();
    configure_stream_executors();

    OV_ITT_TASK_NEXT(COMPILED_MODEL, "create_executor");
    create_executor();

    OV_ITT_TASK_SKIP(COMPILED_MODEL);
}

std::shared_ptr<ov::IAsyncInferRequest> CompiledModel::create_infer_request() const {
    OV_ITT_SCOPED_TASK(itt::domains::NPUPlugin, "CompiledModel::create_infer_request");

    if (_executorPtr == nullptr && _device != nullptr) {
        _executorPtr = _device->createExecutor(_networkPtr, _config);
    }
    if (_executorPtr == nullptr) {
        OPENVINO_THROW(NO_EXECUTOR_FOR_INFERENCE);
    }

    const std::shared_ptr<SyncInferRequest>& syncInferRequest =
        _device->createInferRequest(shared_from_this(), _executorPtr, _config);
    syncInferRequest->initialize_states();

    return std::make_shared<AsyncInferRequest>(syncInferRequest,
                                               get_task_executor(),
                                               _resultExecutor,
                                               get_callback_executor());
}

std::shared_ptr<ov::ISyncInferRequest> CompiledModel::create_sync_infer_request() const {
    OPENVINO_THROW_NOT_IMPLEMENTED(
        "The synchronous inference request structure implemented by the NPU plugin does not inherit "
        "the \"ov::ISyncInferRequest\" class");
}

void CompiledModel::export_model(std::ostream& stream) const {
    const auto& blob = _networkPtr->compiledNetwork;
    stream.write(reinterpret_cast<const char*>(blob.data()), blob.size());

    std::stringstream str;
    str << "Blob size: " << blob.size() << ", hash: " << std::hex << hash(blob);
    _logger.info(str.str().c_str());
}

std::shared_ptr<const ov::Model> CompiledModel::get_runtime_model() const {
    return _model;
}

void CompiledModel::set_property(const ov::AnyMap& properties) {
    std::map<std::string, std::string> config;
    for (auto&& value : properties) {
        config.emplace(value.first, value.second.as<std::string>());
    }
    for (const auto& configEntry : config) {
        if (_properties.find(configEntry.first) == _properties.end()) {
            OPENVINO_THROW("Unsupported configuration key: ", configEntry.first);
        } else {
            if (std::get<1>(_properties[configEntry.first]) == ov::PropertyMutability::RO) {
                OPENVINO_THROW("READ-ONLY configuration key: ", configEntry.first);
            }
        }
    }

    _config.update(config);
    if (_executorPtr != nullptr && config.find(ov::workload_type.name()) != config.end()) {
        const auto workloadType = properties.at(ov::workload_type.name()).as<ov::WorkloadType>();
        _executorPtr->setWorkloadType(workloadType);
    }
}

ov::Any CompiledModel::get_property(const std::string& name) const {
    auto configIterator = _properties.find(name);
    if (configIterator != _properties.cend()) {
        return std::get<2>(configIterator->second)(_config);
    }

    OPENVINO_THROW("Unsupported property ", name);
}

const std::shared_ptr<const NetworkDescription>& CompiledModel::get_network_description() const {
    return _networkPtr;
}

const Config& CompiledModel::get_config() const {
    return _config;
}

const ov::SoPtr<ICompiler>& CompiledModel::get_compiler() const {
    if (_compiler.has_value()) {
        return _compiler.value();
    }
    OPENVINO_THROW("PERF_COUNT property is not set");
}

void CompiledModel::configure_stream_executors() {
    std::shared_ptr<ov::threading::ITaskExecutor> task_executor;
    if (get_plugin()->get_property(ov::internal::exclusive_async_requests.name(), {}).as<bool>()) {
        task_executor = ov::threading::executor_manager()->get_executor("NPU");
    } else if (get_property(ov::hint::enable_cpu_pinning.name()).as<bool>()) {
        auto executor_config = ov::threading::IStreamsExecutor::Config{
            "Intel NPU plugin executor",
            get_plugin()->get_property(ov::num_streams.name(), {}).as<ov::streams::Num>(),
            1,
            ov::hint::SchedulingCoreType::PCORE_ONLY,
            true};
        task_executor = std::make_shared<ov::threading::CPUStreamsExecutor>(executor_config);
    } else {
        task_executor = std::make_shared<ov::threading::CPUStreamsExecutor>(
            ov::threading::IStreamsExecutor::Config{"NPUPlugin executor"});
    }

    set_task_executor(std::move(task_executor));
    const auto executorId = _networkPtr->metadata.name + "_NPUResultExecutor";
    _resultExecutor = ov::threading::executor_manager()->get_executor(executorId);
}

void CompiledModel::initialize_properties() {
    const auto pluginSupportedProperties =
        get_plugin()->get_property(ov::supported_properties.name(), {}).as<std::vector<ov::PropertyName>>();
    const auto isPropertySupported = [&pluginSupportedProperties](const std::string& name) {
        return std::any_of(pluginSupportedProperties.begin(),
                           pluginSupportedProperties.end(),
                           [&name](const ov::PropertyName& property) {
                               return property == name;
                           });
    };
    _properties = {
        // OV Public
        // =========
        {ov::supported_properties.name(),
         {true,
          ov::PropertyMutability::RO,
          [&](const Config&) {
              return _supportedProperties;
          }}},
        {ov::device::id.name(),
         {true,
          ov::PropertyMutability::RO,
          [](const Config& config) {
              return config.get<DEVICE_ID>();
          }}},
        {ov::enable_profiling.name(),
         {true,
          ov::PropertyMutability::RO,
          [](const Config& config) {
              return config.get<PERF_COUNT>();
          }}},
        {ov::model_name.name(),
         {true,
          ov::PropertyMutability::RO,
          [&](const Config&) {
              OPENVINO_ASSERT(_networkPtr != nullptr, "Missing network descriptor");
              return _networkPtr->metadata.name;
          }}},
        {ov::optimal_number_of_infer_requests.name(),
         {true,
          ov::PropertyMutability::RO,
          [&](const Config& config) {
              // value is allowed to be queried prior the network is compiled
              return static_cast<uint32_t>(getOptimalNumberOfInferRequestsInParallel(config));
          }}},
        {ov::execution_devices.name(),
         {true,
          ov::PropertyMutability::RO,
          [&](const Config&) {
              return std::string("NPU");
          }}},
        {ov::loaded_from_cache.name(),
         {true,
          ov::PropertyMutability::RO,
          [](const Config& config) {
              return config.get<LOADED_FROM_CACHE>();
          }}},
        {ov::workload_type.name(),
         {isPropertySupported(ov::workload_type.name()),
          ov::PropertyMutability::RW,
          [](const Config& config) {
              return config.get<WORKLOAD_TYPE>();
          }}},
        // OV Public Hints
        // =========
        {ov::hint::performance_mode.name(),
         {true,
          ov::PropertyMutability::RO,
          [](const Config& config) {
              return config.get<PERFORMANCE_HINT>();
          }}},
        {ov::hint::execution_mode.name(),
         {true,
          ov::PropertyMutability::RO,
          [](const Config& config) {
              return config.get<EXECUTION_MODE_HINT>();
          }}},
        {ov::hint::num_requests.name(),
         {true,
          ov::PropertyMutability::RO,
          [](const Config& config) {
              return config.get<PERFORMANCE_HINT_NUM_REQUESTS>();
          }}},
        {ov::hint::inference_precision.name(),
         {true,
          ov::PropertyMutability::RO,
          [](const Config& config) {
              return config.get<INFERENCE_PRECISION_HINT>();
          }}},
        {ov::hint::enable_cpu_pinning.name(),
         {true,
          ov::PropertyMutability::RO,
          [](const Config& config) {
              return config.get<ENABLE_CPU_PINNING>();
          }}},
        {ov::hint::model_priority.name(),
         {true,
          ov::PropertyMutability::RO,
          [](const Config& config) {
              return config.get<MODEL_PRIORITY>();
          }}},
        // OV Internals
        // =========
        {ov::internal::supported_properties.name(),
         {false,
          ov::PropertyMutability::RO,
          [&](const Config&) {
              static const std::vector<ov::PropertyName> supportedProperty{
                  ov::PropertyName(ov::internal::caching_properties.name(), ov::PropertyMutability::RO),
              };
              return supportedProperty;
          }}},
        // NPU Public
        // =========
        {ov::intel_npu::compilation_mode_params.name(),
         {true,
          ov::PropertyMutability::RO,
          [](const Config& config) {
              return config.get<COMPILATION_MODE_PARAMS>();
          }}},
        {ov::intel_npu::turbo.name(),
         {isPropertySupported(ov::intel_npu::turbo.name()),
          ov::PropertyMutability::RO,
          [](const Config& config) {
              return config.get<TURBO>();
          }}},
        // NPU Private
        // =========
        {ov::intel_npu::tiles.name(),
         {false,
          ov::PropertyMutability::RO,
          [](const Config& config) {
              return config.get<TILES>();
          }}},
        {ov::intel_npu::profiling_type.name(),
         {false,
          ov::PropertyMutability::RO,
          [](const Config& config) {
              return config.getString<PROFILING_TYPE>();
          }}},
        {ov::intel_npu::platform.name(),
         {false,
          ov::PropertyMutability::RO,
          [](const Config& config) {
              return config.get<PLATFORM>();
          }}},
        {ov::intel_npu::dynamic_shape_to_static.name(),
         {false,
          ov::PropertyMutability::RO,
          [](const Config& config) {
              return config.getString<DYNAMIC_SHAPE_TO_STATIC>();
          }}},
        {ov::intel_npu::use_elf_compiler_backend.name(),
         {false,
          ov::PropertyMutability::RO,
          [](const Config& config) {
              return config.getString<USE_ELF_COMPILER_BACKEND>();
          }}},
        {ov::intel_npu::create_executor.name(),
         {false,
          ov::PropertyMutability::RO,
          [](const Config& config) {
              return config.get<CREATE_EXECUTOR>();
          }}},
        {ov::intel_npu::batch_mode.name(),
         {false,
          ov::PropertyMutability::RO,
          [](const Config& config) {
              return config.getString<BATCH_MODE>();
          }}},
    };

    for (auto& property : _properties) {
        if (std::get<0>(property.second)) {
            _supportedProperties.emplace_back(property.first, std::get<1>(property.second));
        }
    }
}

void CompiledModel::create_executor() {
    if (_config.get<CREATE_EXECUTOR>()) {
        _logger.info("Creating the executor inside the \"CompiledModel\" constructor");

        // If no device has been defined, the executor shall keep the default value of "nullptr". In this scenario,
        // only export operations will be allowed
        if (_device != nullptr) {
            _executorPtr = _device->createExecutor(_networkPtr, _config);
        }
    } else {
        _logger.info("Executor will not be created inside the \"CompiledModel\" constructor");
    }
}

}  // namespace intel_npu
