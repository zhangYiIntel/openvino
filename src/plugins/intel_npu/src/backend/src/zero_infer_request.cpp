// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "zero_infer_request.hpp"

#include "intel_npu/al/config/common.hpp"
#include "intel_npu/al/config/compiler.hpp"
#include "intel_npu/al/config/runtime.hpp"
#include "intel_npu/al/itt.hpp"
#include "intel_npu/al/prefix.hpp"
#include "intel_npu/utils/zero/zero_api.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/runtime/intel_npu/remote_properties.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "zero_memory.hpp"

using namespace intel_npu;

namespace {

constexpr std::size_t BATCH_AXIS = 0;

/**
 * @brief Checks that the metadata of the provided descriptor corresponds to the values registered in the Level Zero
 * structure.
 * @param nodeDescriptor The OpenVINO API specific I/O descriptor which shall be compared.
 * @param zeDescriptor The Level Zero specific structure used for comparison.
 * @param name Tensor identifier used for error logging.
 */
void checkLevelZeroAttributesMatch(const IONodeDescriptor& nodeDescriptor,
                                   const ZeroExecutor::ArgumentDescriptor& zeDescriptor,
                                   const std::string& name) {
    const ov::element::Type_t ovPrecision = nodeDescriptor.precision;
    const ze_graph_argument_precision_t zePrecision = zeDescriptor.info.devicePrecision;

    if (zeroUtils::getZePrecision(ovPrecision) != zePrecision) {
        OPENVINO_THROW("Precision mismatch for parameter " + name);
    }

    const std::vector<size_t>& ovDimensions = nodeDescriptor.transposedShape.get_max_shape();

    if (ovDimensions.size() > ZE_MAX_GRAPH_ARGUMENT_DIMENSIONS_SIZE) {
        OPENVINO_THROW(
            "Maximum number of dimensions supported: " + std::to_string(ZE_MAX_GRAPH_ARGUMENT_DIMENSIONS_SIZE) + '\n' +
            "Given: " + std::to_string(ovDimensions.size()));
    }

    for (size_t index = ovDimensions.size(); index < ZE_MAX_GRAPH_ARGUMENT_DIMENSIONS_SIZE; ++index) {
        if (zeDescriptor.info.dims[index] != 0 && zeDescriptor.info.dims[index] != 1) {
            OPENVINO_THROW("Shape mismatch for parameter " + name);
        }
    }

    for (size_t index = 1; index < ovDimensions.size(); ++index) {
        if (ovDimensions[index] != zeDescriptor.info.dims[index] && !nodeDescriptor.transposedShape.is_dynamic()) {
            OPENVINO_THROW("Shape mismatch for parameter " + name);
        }
    }
}

std::optional<size_t> getBatchSizeForNode(const IONodeDescriptor& nodeDescriptor,
                                          const ZeroExecutor::ArgumentDescriptor& zeDescriptor) {
    Logger logger("GetBatchSizeForNode", Logger::global().level());

    if (nodeDescriptor.originalShape.rank().get_length() == 0) {
        logger.info("Networks with empty shapes are not supported when batching is handled by the plugin");
        return std::nullopt;
    }

    if (nodeDescriptor.originalShape.is_dynamic()) {
        logger.info("Dynamic networks are not supported when batching is handled by the plugin");
        return std::nullopt;
    }

    const std::vector<size_t>& ovDimensions = nodeDescriptor.originalShape.get_shape();

    if (ovDimensions[BATCH_AXIS] == zeDescriptor.info.dims[BATCH_AXIS] &&
        ovDimensions[BATCH_AXIS] != DEFAULT_BATCH_SIZE) {
        logger.info("Batching on the plugin is not used, batching is handled by the compiler");
        return std::nullopt;
    }

    if (zeDescriptor.info.dims[BATCH_AXIS] == DEFAULT_BATCH_SIZE) {
        return ovDimensions[BATCH_AXIS];
    }

    return DEFAULT_BATCH_SIZE;
}

/**
 * @brief Get the batch size to be handled on the plugin.
 * @details Analyze the shape from the compiled model with the shape from the originalShape and get the originalShape if
 * it is different.
 * @param metadata A map to represent descriptions for inputs and outputs of a network.
 * @param executorInputDescriptors A map to represent Level zero inputs descriptors.
 * @param executorOutputDescriptors A map to represent Level zero outputs descriptors.
 */

std::optional<size_t> getBatchSize(
    const NetworkMetadata& metadata,
    const std::unordered_map<std::string, ZeroExecutor::ArgumentDescriptor>& executorInputDescriptors,
    const std::unordered_map<std::string, ZeroExecutor::ArgumentDescriptor>& executorOutputDescriptors) {
    std::set<size_t> batch_size;

    Logger logger("getBatchSize", Logger::global().level());

    for (const std::string& inputName : metadata.inputNames) {
        auto batchSizeForNode =
            getBatchSizeForNode(metadata.parameters.at(inputName), executorInputDescriptors.at(inputName));

        if (batchSizeForNode.has_value()) {
            batch_size.insert(*batchSizeForNode);
        } else {
            return std::nullopt;
        }
    }

    for (const std::string& outputName : metadata.outputNames) {
        if (!executorOutputDescriptors.count(outputName)) {
            OPENVINO_THROW("Invalid graph output descriptor key: " + outputName);
        }
        auto batchSizeForNode =
            getBatchSizeForNode(metadata.results.at(outputName), executorOutputDescriptors.at(outputName));

        if (batchSizeForNode.has_value()) {
            batch_size.insert(*batchSizeForNode);
        } else {
            return std::nullopt;
        }
    }

    if (batch_size.size() != 1) {
        logger.info("Batching works only when we have the same batch size for all tensors!");
        return std::nullopt;
    }

    auto it = batch_size.begin();
    if (*it) {
        return *it;
    }

    return std::nullopt;
}

template <typename Type>
Type extract_object(const ov::AnyMap& params, const ov::Property<Type>& p) {
    auto itrHandle = params.find(p.name());
    ov::Any res = nullptr;
    if (itrHandle == params.end()) {
        OPENVINO_THROW("No parameter ", p.name(), " found in parameters map");
    }
    res = itrHandle->second;
    return res.as<Type>();
}

}  // namespace

//------------------------------------------------------------------------------
ZeroInferRequest::ZeroInferRequest(const std::shared_ptr<ZeroInitStructsHolder>& initStructs,
                                   const std::shared_ptr<const ICompiledModel>& compiledModel,
                                   const std::shared_ptr<const IExecutor>& executor,
                                   const Config& config)
    : SyncInferRequest(compiledModel),
      _initStructs(initStructs),
      _executorPtr(executor),
      _executor(static_cast<const ZeroExecutor*>(_executorPtr.get())),
      _config(config),
      _logger("ZeroInferRequest", config.get<LOG_LEVEL>()),
      _profilingPool(_executor->graph(), zeroProfiling::POOL_SIZE, _executor->getInitStructs()->getProfilingDdiTable()),
      _profilingQuery(0,
                      _executor->getInitStructs()->getDevice(),
                      _executor->getInitStructs()->getProfilingDdiTable()) {
    _logger.debug("ZeroInferRequest::ZeroInferRequest - SyncInferRequest");
    const std::unordered_map<std::string, ZeroExecutor::ArgumentDescriptor>& executorInputDescriptors =
        _executor->inputs_desc_map();
    const std::unordered_map<std::string, ZeroExecutor::ArgumentDescriptor>& executorOutputDescriptors =
        _executor->outputs_desc_map();

    auto proftype = config.get<PROFILING_TYPE>();
    if (proftype == ov::intel_npu::ProfilingType::INFER) {
        _logger.debug("ZeroInferRequest::ZeroInferRequest - profiling type == ov::intel_npu::ProfilingType::INFER");
        _npuProfiling = std::make_shared<zeroProfiling::NpuInferProfiling>(_executor->getInitStructs()->getContext(),
                                                                           _executor->getInitStructs()->getDevice(),
                                                                           _config.get<LOG_LEVEL>());
    }

    _properties.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
    zeroUtils::throwOnFail("zeDeviceGetProperties",
                           zeDeviceGetProperties(_executor->getInitStructs()->getDevice(), &_properties));

    const auto contains = [](const auto& container, const auto& value) {
        return std::find(container.begin(), container.end(), value) != container.end();
    };

    _outputAllocator = std::make_shared<const zeroMemory::HostMemAllocator>(_initStructs);
    _inputAllocator =
        (_properties.flags & ZE_DEVICE_PROPERTY_FLAG_INTEGRATED)
            ? std::make_shared<const zeroMemory::HostMemAllocator>(_initStructs,
                                                                   ZE_HOST_MEM_ALLOC_FLAG_BIAS_WRITE_COMBINED)
            : _outputAllocator;

    _logger.debug("ZeroInferRequest::ZeroInferRequest - performing I/O buffer allocation using Level Zero API");
    for (const std::string& inputName : _metadata.inputNames) {
        if (!executorInputDescriptors.count(inputName)) {
            OPENVINO_THROW("Invalid graph input descriptor key: " + inputName);
        }
    }

    for (const std::string& outputName : _metadata.outputNames) {
        if (!executorOutputDescriptors.count(outputName)) {
            OPENVINO_THROW("Invalid graph output descriptor key: " + outputName);
        }
    }

    if (config.get<BATCH_MODE>() != ov::intel_npu::BatchMode::COMPILER) {
        auto batchSize = getBatchSize(_metadata, executorInputDescriptors, executorOutputDescriptors);

        if (batchSize.has_value()) {
            _batchSize = *batchSize;
        }
    }

    for (const std::string& inputName : _metadata.inputNames) {
        IONodeDescriptor& parameterDescriptor = _metadata.parameters.at(inputName);
        checkLevelZeroAttributesMatch(parameterDescriptor, executorInputDescriptors.at(inputName), inputName);

        // When batching is handled by the plugin we need to modify transposed shape with the original batch size since
        // it will be forced to 1 at the compilation time
        if (_batchSize > DEFAULT_BATCH_SIZE) {
            parameterDescriptor.transposedShape[BATCH_AXIS] = _batchSize;
        }

        if (contains(_metadata.shapeNames, inputName)) {
            const std::string shapeBufferName = SHAPE_TENSOR_PREFIX + inputName;
            const IONodeDescriptor& shapeDescriptor = _metadata.shapes.at(inputName);

            checkLevelZeroAttributesMatch(shapeDescriptor,
                                          executorInputDescriptors.at(shapeBufferName),
                                          shapeBufferName);

            allocate_tensor(inputName, shapeDescriptor, TensorType::Shape, *_inputAllocator);
            _tensorsData[shapeBufferName] = TensorData{_copyAllTensors.at(shapeBufferName)->data(),
                                                       _copyAllTensors.at(shapeBufferName)->get_byte_size()};
        }
    }

    for (const std::string& outputName : _metadata.outputNames) {
        IONodeDescriptor& resultDescriptor = _metadata.results.at(outputName);
        checkLevelZeroAttributesMatch(resultDescriptor, executorOutputDescriptors.at(outputName), outputName);

        // When batching is handled by the plugin we need to modify transposed shape with the original batch size since
        // it will be forced to 1 at the compilation time
        if (_batchSize > DEFAULT_BATCH_SIZE) {
            resultDescriptor.transposedShape[BATCH_AXIS] = _batchSize;
        }

        const auto& shapeNameMatch = _nodeNameToLegacyName.find(outputName);
        if (shapeNameMatch != _nodeNameToLegacyName.end()) {
            if (contains(_metadata.shapeNames, shapeNameMatch->second)) {
                const std::string shapeBufferName = SHAPE_TENSOR_PREFIX + shapeNameMatch->second;
                const IONodeDescriptor& shapeDescriptor = _metadata.shapes.at(shapeNameMatch->second);

                checkLevelZeroAttributesMatch(shapeDescriptor,
                                              executorOutputDescriptors.at(shapeBufferName),
                                              shapeBufferName);

                allocate_tensor(shapeNameMatch->second, shapeDescriptor, TensorType::Shape, *_outputAllocator);
                _tensorsData[shapeBufferName] = TensorData{_copyAllTensors.at(shapeBufferName)->data(),
                                                           _copyAllTensors.at(shapeBufferName)->get_byte_size()};
            }
        }
    }

    for (const std::string& stateName : _metadata.stateNames) {
        const std::string& stateInputBufferName = READVALUE_PREFIX + stateName;
        const std::string& stateOutputBufferName = ASSIGN_PREFIX + stateName;

        if (!executorInputDescriptors.count(stateInputBufferName)) {
            OPENVINO_THROW("Invalid graph input descriptor key: " + stateInputBufferName);
        }
        if (!executorOutputDescriptors.count(stateOutputBufferName)) {
            OPENVINO_THROW("Invalid graph output descriptor key: " + stateOutputBufferName);
        }

        const IONodeDescriptor& stateDescriptor = _metadata.states.at(stateName);
        checkLevelZeroAttributesMatch(stateDescriptor,
                                      executorInputDescriptors.at(stateInputBufferName),
                                      stateInputBufferName);
        checkLevelZeroAttributesMatch(stateDescriptor,
                                      executorOutputDescriptors.at(stateOutputBufferName),
                                      stateOutputBufferName);

        // Only one buffer per state variable is required, we'll use the "output" one since this one captures the latest
        // tensor value
        allocate_tensor(stateName, stateDescriptor, TensorType::State, *_outputAllocator);
        _tensorsData[stateInputBufferName] = TensorData{_copyAllTensors.at(stateInputBufferName)->data(),
                                                        _copyAllTensors.at(stateInputBufferName)->get_byte_size()};
        _tensorsData[stateOutputBufferName] = TensorData{_copyAllTensors.at(stateOutputBufferName)->data(),
                                                         _copyAllTensors.at(stateOutputBufferName)->get_byte_size()};
    }
}

void ZeroInferRequest::create_pipeline() {
    for (const std::string& inputName : _metadata.inputNames) {
        if (_copyAllTensors.find(inputName) != _copyAllTensors.end()) {
            _logger.debug("ZeroInferRequest::create_pipeline - tensor %s was already allocated", inputName.c_str());
            continue;
        }

        IONodeDescriptor& parameterDescriptor = _metadata.parameters.at(inputName);

        _logger.debug("ZeroInferRequest::create_pipeline - Allocate new tensor");
        allocate_tensor(inputName, parameterDescriptor, TensorType::InputOrOutput, *_inputAllocator);
        _tensorsData[inputName] =
            TensorData{_copyAllTensors.at(inputName)->data(), _copyAllTensors.at(inputName)->get_byte_size()};
    }

    for (const std::string& outputName : _metadata.outputNames) {
        if (_copyAllTensors.find(outputName) != _copyAllTensors.end()) {
            _logger.debug("ZeroInferRequest::create_pipeline - tensor %s was already allocated", outputName.c_str());
            continue;
        }

        IONodeDescriptor& resultDescriptor = _metadata.results.at(outputName);

        _logger.debug("ZeroInferRequest::create_pipeline - allocate new tensor");
        allocate_tensor(outputName, resultDescriptor, TensorType::InputOrOutput, *_outputAllocator);
        _tensorsData[outputName] =
            TensorData{_copyAllTensors.at(outputName)->data(), _copyAllTensors.at(outputName)->get_byte_size()};
    }

    _logger.debug("ZeroInferRequest::create_pipeline - constructing pipeline");
    // Construct pipeline
    _pipeline =
        makePipeline(_executorPtr, _config, _profilingPool, _profilingQuery, _npuProfiling, _tensorsData, _batchSize);
    _logger.debug("ZeroInferRequest::create_pipeline - SyncInferRequest completed");
}

void ZeroInferRequest::set_tensor_data(std::shared_ptr<ov::ITensor> tensor, const std::string& name, bool isParameter) {
    OV_ITT_TASK_CHAIN(ZERO_SET_TENSOR, itt::domains::LevelZeroBackend, "set_tensor", "set_tensor_data");
    bool setTensorData = false;
    bool levelZeroTensorCreatedLocally = true;

    OV_ITT_TASK_NEXT(ZERO_SET_TENSOR, "check_data_allocation");
    ze_memory_allocation_properties_t desc = {};
    desc.stype = ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES;
    auto res = zeMemGetAllocProperties(_initStructs->getContext(), tensor->data(), &desc, nullptr);
    if (res == ZE_RESULT_SUCCESS) {
        if (desc.id) {
            switch (desc.type) {
            case ZE_MEMORY_TYPE_HOST:
            case ZE_MEMORY_TYPE_DEVICE:
            case ZE_MEMORY_TYPE_SHARED:
                _logger.debug("ZeroInferRequest::set_tensor_data - tensor was created in the same L0 context");
                _copyAllTensors[name] = tensor;
                levelZeroTensorCreatedLocally = false;
                setTensorData = true;
                break;
            case ZE_MEMORY_TYPE_UNKNOWN:
            case ZE_MEMORY_TYPE_FORCE_UINT32:
                break;
            }
        }
    }

    if (!setTensorData) {
        // make sure that the L0 tensor was allocated locally and is not received from the user when receiving random
        // tensor
        if ((_tensorsData.find(name) != _tensorsData.end()) && !_tensorsData.at(name).levelZeroTensorCreatedLocally) {
            _logger.debug("ZeroInferRequest::set_tensor_data - create locally L0 tensor");
            OV_ITT_TASK_NEXT(ZERO_SET_TENSOR, "allocate tensor");

            allocate_tensor(name,
                            isParameter ? _metadata.parameters.at(name) : _metadata.results.at(name),
                            TensorType::InputOrOutput,
                            isParameter ? *_inputAllocator : *_outputAllocator);

            setTensorData = true;
            levelZeroTensorCreatedLocally = true;
        }
    }

    if (setTensorData) {
        _tensorsData[name] = TensorData{_copyAllTensors.at(name)->data(),
                                        _copyAllTensors.at(name)->get_byte_size(),
                                        levelZeroTensorCreatedLocally};

        if (_pipelineIsCreated) {
            _logger.debug("ZeroInferRequest::infer_async - update command list");

            intel_npu::ZeroExecutor::ArgumentDescriptor desc;
            if (isParameter) {
                desc = _executor->inputs_desc_map().at(name);
            } else {
                desc = _executor->outputs_desc_map().at(name);
            }

            OV_ITT_TASK_NEXT(ZERO_SET_TENSOR, "updateCommandList");
            _pipeline->updateCommandList(_tensorsData[name], desc.idx, _batchSize);
        }
    }
}

void ZeroInferRequest::set_remote_tensor_data(std::shared_ptr<ZeroRemoteTensor> tensor,
                                              const std::string& name,
                                              bool isParameter) {
    OV_ITT_TASK_CHAIN(ZERO_SET_REMOTE_TENSOR, itt::domains::LevelZeroBackend, "set_tensor", "set_remote_tensor_data");

    auto l0_context = reinterpret_cast<ze_context_handle_t>(
        extract_object(tensor->get_context()->get_property(), ov::intel_npu::l0_context));
    if (_initStructs->getContext() != l0_context) {
        OPENVINO_THROW("Using different context for creating the tensor is not supported");
    }

    auto data = extract_object(tensor->get_properties(), ov::intel_npu::mem_handle);
    if (data == nullptr) {
        OPENVINO_THROW("Empty buffer");
    }

    _copyAllTensors[name] = tensor;
    _tensorsData[name] = TensorData{data, tensor->get_byte_size(), false};

    if (_pipelineIsCreated) {
        _logger.debug("ZeroInferRequest::infer_async - update command list");

        intel_npu::ZeroExecutor::ArgumentDescriptor desc;
        if (isParameter) {
            desc = _executor->inputs_desc_map().at(name);
        } else {
            desc = _executor->outputs_desc_map().at(name);
        }

        OV_ITT_TASK_NEXT(ZERO_SET_REMOTE_TENSOR, "updateCommandList");
        _pipeline->updateCommandList(_tensorsData[name], desc.idx, _batchSize);
    }
}

void ZeroInferRequest::set_tensor(const ov::Output<const ov::Node>& port, const ov::SoPtr<ov::ITensor>& tensor) {
    OV_ITT_SCOPED_TASK(itt::domains::LevelZeroBackend, "set_tensor");
    try {
        check_tensor(port, tensor);
    } catch (const ov::Exception& ex) {
        OPENVINO_THROW("Failed to set tensor. ", ex.what());
    }

    _allTensors[port.get_node()->get_friendly_name()] = tensor._ptr;

    if (_initStructs->getMutableCommandListVersion()) {
        auto remoteTensor = std::dynamic_pointer_cast<ZeroRemoteTensor>(tensor._ptr);

        if (remoteTensor == nullptr) {
            _logger.debug("ZeroInferRequest::set_tensor - set new tensor");
            set_tensor_data(tensor._ptr,
                            port.get_node()->get_friendly_name(),
                            ov::op::util::is_parameter(port.get_node()));
        } else {
            _logger.debug("ZeroInferRequest::set_tensor - set new remote tensor");
            set_remote_tensor_data(remoteTensor,
                                   port.get_node()->get_friendly_name(),
                                   ov::op::util::is_parameter(port.get_node()));
        }
    }
}

ov::SoPtr<ov::ITensor> ZeroInferRequest::get_tensor(const ov::Output<const ov::Node>& port) const {
    OV_ITT_SCOPED_TASK(itt::domains::LevelZeroBackend, "get_tensor");
    const std::string& nodeFriendlyName = port.get_node()->get_friendly_name();

    if (_allTensors.find(nodeFriendlyName) != _allTensors.end()) {
        _logger.debug("ZeroInferRequest::get_tensor - tensor allocated, get the tensor");
        return _allTensors.at(nodeFriendlyName);
    }

    _logger.debug("ZeroInferRequest::get_tensor - tensor is not allocated, create the tensor");

    const bool isParameter = ov::op::util::is_parameter(port.get_node());
    allocate_tensor(nodeFriendlyName,
                    isParameter ? _metadata.parameters.at(nodeFriendlyName) : _metadata.results.at(nodeFriendlyName),
                    TensorType::InputOrOutput,
                    isParameter ? *_inputAllocator : *_outputAllocator);

    _tensorsData[nodeFriendlyName] =
        TensorData{_copyAllTensors.at(nodeFriendlyName)->data(), _copyAllTensors.at(nodeFriendlyName)->get_byte_size()};

    return _allTensors.at(nodeFriendlyName);
}

void ZeroInferRequest::infer() {
    infer_async();
    get_result();
}

void ZeroInferRequest::infer_async() {
    _logger.debug("InferRequest::infer_async started");
    OV_ITT_TASK_CHAIN(ZERO_INFER, itt::domains::LevelZeroBackend, "infer_async", "start");

    _executor->mutexLock();
    if (!_pipelineIsCreated) {
        OV_ITT_TASK_NEXT(ZERO_INFER, "create_pipeline");
        create_pipeline();

        _pipelineIsCreated = true;
    }
    _executor->mutexUnlock();

    for (const std::string& name : _inputAndStateInputNames) {
        auto& inputTensor = _allTensors.at(name);

        if (isShapeTensorName(name)) {
            const auto actualTensorName = name.substr(SHAPE_TENSOR_PREFIX.size());
            const auto& inputDims = _allTensors.at(actualTensorName)->get_shape();

            for (size_t i = 0; i < inputTensor->get_size(); ++i) {
                const auto reverseIdx = inputDims.size() - 1 - i;
                inputTensor->data<uint32_t>()[i] = static_cast<uint32_t>(inputDims[reverseIdx]);
            }
        }

        auto remoteTensor = std::dynamic_pointer_cast<ZeroRemoteTensor>(inputTensor);
        void* data = !remoteTensor ? inputTensor->data()
                                   : extract_object(remoteTensor->get_properties(), ov::intel_npu::mem_handle);

        const auto& copyInputTensor = _copyAllTensors.at(name);
        auto copyRemoteTensor = std::dynamic_pointer_cast<ZeroRemoteTensor>(copyInputTensor);
        if (copyRemoteTensor == nullptr) {
            void* copyData = copyInputTensor->data();

            if (data != copyData) {
                if (data == nullptr || copyData == nullptr) {
                    OPENVINO_THROW("Empty buffer");
                }

                _logger.info("Tensor is not allocated in the current Level Zero context");
                OV_ITT_TASK_NEXT(ZERO_INFER, "memcpy");
                std::memcpy(copyData, data, inputTensor->get_byte_size());
            }
        }
    }

    OV_ITT_TASK_NEXT(ZERO_INFER, "push");
    for (size_t i = 0; i < _batchSize; i++) {
        _pipeline->push(i);
    }
}

void ZeroInferRequest::get_result() {
    OV_ITT_TASK_CHAIN(ZERO_RESULT, itt::domains::LevelZeroBackend, "get_result", "pull");

    for (size_t i = 0; i < _batchSize; i++) {
        _pipeline->pull(i);
    }

    for (const auto& name : _outputAndStateOutputNames) {
        const auto& outputTensor = _allTensors.at(name);

        if (isShapeTensorName(name)) {
            const auto actualTensorName = name.substr(SHAPE_TENSOR_PREFIX.size());
            const auto& shapeNameMatch = _legacyNameToNodeName.find(actualTensorName);
            if (shapeNameMatch != _legacyNameToNodeName.end()) {
                ov::Shape actualDims;
                actualDims.reserve(outputTensor->get_size());

                for (size_t i = 0; i < outputTensor->get_size(); ++i) {
                    const auto reverseIdx = outputTensor->get_size() - 1 - i;
                    actualDims.push_back(outputTensor->data<uint32_t>()[reverseIdx]);
                }
                auto& tensorToBeReshaped = _allTensors.at(shapeNameMatch->second);
                tensorToBeReshaped->set_shape(actualDims);
            }
        }

        auto remoteTensor = std::dynamic_pointer_cast<ZeroRemoteTensor>(outputTensor);
        void* data = nullptr;
        if (remoteTensor == nullptr) {
            data = outputTensor->data();
        } else {
            data = extract_object(remoteTensor->get_properties(), ov::intel_npu::mem_handle);
        }

        const auto& copyOutputTensor = _copyAllTensors.at(name);
        auto copyRemoteTensor = std::dynamic_pointer_cast<ZeroRemoteTensor>(copyOutputTensor);
        if (copyRemoteTensor == nullptr) {
            void* copyData = copyOutputTensor->data();

            if (data != copyData) {
                if (data == nullptr || copyData == nullptr) {
                    OPENVINO_THROW("Empty buffer");
                }

                _logger.info("Tensor is not allocated in the current Level Zero context");
                OV_ITT_TASK_NEXT(ZERO_RESULT, "memcpy");
                std::memcpy(data, copyData, outputTensor->get_byte_size());
            }
        }
    }

    OV_ITT_TASK_NEXT(ZERO_RESULT, "reset");
    for (size_t i = 0; i < _batchSize; i++) {
        _pipeline->reset(i);
    }
    _logger.debug("InferRequest::get_result finished");
}

void ZeroInferRequest::check_network_precision(const ov::element::Type_t precision) const {
    switch (precision) {
    case ov::element::Type_t::f32:
        break;
    case ov::element::Type_t::f16:
        break;
    case ov::element::Type_t::u4:
        break;
    case ov::element::Type_t::i4:
        break;
    case ov::element::Type_t::u8:
        break;
    case ov::element::Type_t::i8:
        break;
    case ov::element::Type_t::u16:
        break;
    case ov::element::Type_t::i16:
        break;
    case ov::element::Type_t::u32:
        break;
    case ov::element::Type_t::i32:
        break;
    case ov::element::Type_t::u64:
        break;
    case ov::element::Type_t::i64:
        break;
    case ov::element::Type_t::f64:
        break;
    default:
        OPENVINO_THROW("Unsupported tensor precision: " + ov::element::Type(precision).get_type_name() +
                       "! Supported precisions: FP32, FP16, U4, I4, U8, I8, U16, I16, U32, I32, U64, I64, FP64");
    }
}

std::vector<ov::ProfilingInfo> ZeroInferRequest::get_profiling_info() const {
    _logger.debug("InferRequest::get_profiling_info started");
    const auto& compiledModel = *std::dynamic_pointer_cast<const ICompiledModel>(_compiledModel);
    const auto& compilerConfig = compiledModel.get_config();
    if (!compilerConfig.get<PERF_COUNT>() || !_config.get<PERF_COUNT>()) {
        _logger.debug("InferRequest::get_profiling_info complete with empty {}.");
        return {};
    }

    auto compilerType = compilerConfig.get<COMPILER_TYPE>();
    if (compilerType == ov::intel_npu::CompilerType::MLIR) {
        // For plugin compiler retreive raw profiling data from backend and delegate
        // processing to the compiler
        const auto& networkDesc = compiledModel.get_network_description();
        const auto& compiler = compiledModel.get_compiler();
        const auto& blob = networkDesc->compiledNetwork;
        auto profData = get_raw_profiling_data();
        _logger.debug("InferRequest::get_profiling_info complete with compiler->process_profiling_output().");
        return compiler->process_profiling_output(profData, blob, compilerConfig);
    } else {
        auto proftype = _config.get<PROFILING_TYPE>();
        if (proftype == ov::intel_npu::ProfilingType::INFER) {
            _logger.debug("InferRequest::get_profiling_info complete with _npuProfiling->getNpuInferStatistics().");
            return _npuProfiling->getNpuInferStatistics();
        } else {  /// proftype = MODEL or undefined = fallback to model profiling
            _logger.debug("InferRequest::get_profiling_info complete with _profilingQuery.getLayerStatistics().");
            return _profilingQuery.getLayerStatistics();
        }
    }
}

std::vector<uint8_t> ZeroInferRequest::get_raw_profiling_data() const {
    return _profilingQuery.getData<uint8_t>();
}
