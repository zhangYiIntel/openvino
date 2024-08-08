// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "zero_compiler_in_driver.hpp"

#include <fstream>
#include <regex>
#include <string_view>

#include "graph_transformations.hpp"
#include "intel_npu/al/config/common.hpp"
#include "intel_npu/al/config/compiler.hpp"
#include "intel_npu/al/config/runtime.hpp"
#include "intel_npu/al/itt.hpp"
#include "intel_npu/al/prefix.hpp"
#include "intel_npu/utils/zero/zero_result.hpp"
#include "openvino/core/model.hpp"

namespace {

constexpr std::string_view INPUTS_PRECISIONS_KEY = "--inputs_precisions";
constexpr std::string_view INPUTS_LAYOUTS_KEY = "--inputs_layouts";
constexpr std::string_view OUTPUTS_PRECISIONS_KEY = "--outputs_precisions";
constexpr std::string_view OUTPUTS_LAYOUTS_KEY = "--outputs_layouts";

// <option key>="<option value>"
constexpr std::string_view KEY_VALUE_SEPARATOR = "=";
constexpr std::string_view VALUE_DELIMITER = "\"";  // marks beginning and end of value

// Format inside "<option value>"
// <name1>:<value (precision / layout)> [<name2>:<value>]
constexpr std::string_view NAME_VALUE_SEPARATOR = ":";
constexpr std::string_view VALUES_SEPARATOR = " ";

// Constants indicating the order indices needed to be applied as to perform conversions between legacy layout values
const std::vector<size_t> NC_TO_CN_LAYOUT_DIMENSIONS_ORDER = {1, 0};
const std::vector<size_t> NCHW_TO_NHWC_LAYOUT_DIMENSIONS_ORDER = {0, 2, 3, 1};
const std::vector<size_t> NCDHW_TO_NDHWC_LAYOUT_DIMENSIONS_ORDER = {0, 2, 3, 4, 1};

/**
 * @brief A standard copy function concerning memory segments. Additional checks on the given arguments are performed
 * before copying.
 * @details This is meant as a replacement for the legacy "ie_memcpy" function coming from the OpenVINO API.
 */
void checkedMemcpy(void* destination, size_t destinationSize, void const* source, size_t numberOfBytes) {
    if (numberOfBytes == 0) {
        return;
    }

    OPENVINO_ASSERT(destination != nullptr, "Memcpy: received a null destination address");
    OPENVINO_ASSERT(source != nullptr, "Memcpy: received a null source address");
    OPENVINO_ASSERT(numberOfBytes <= destinationSize,
                    "Memcpy: the source buffer does not fit inside the destination one");
    OPENVINO_ASSERT(numberOfBytes <= (destination > source ? ((uintptr_t)destination - (uintptr_t)source)
                                                           : ((uintptr_t)source - (uintptr_t)destination)),
                    "Memcpy: the offset between the two buffers does not allow a safe execution of the operation");

    memcpy(destination, source, numberOfBytes);
}

ov::element::Type_t toOVElementType(const ze_graph_argument_precision_t zeElementType) {
    switch (zeElementType) {
    case ZE_GRAPH_ARGUMENT_PRECISION_UNKNOWN:
        return ov::element::Type_t::undefined;
    case ZE_GRAPH_ARGUMENT_PRECISION_DYNAMIC:
        return ov::element::Type_t::dynamic;
    case ZE_GRAPH_ARGUMENT_PRECISION_BOOLEAN:
        return ov::element::Type_t::boolean;
    case ZE_GRAPH_ARGUMENT_PRECISION_BF16:
        return ov::element::Type_t::bf16;
    case ZE_GRAPH_ARGUMENT_PRECISION_FP16:
        return ov::element::Type_t::f16;
    case ZE_GRAPH_ARGUMENT_PRECISION_FP32:
        return ov::element::Type_t::f32;
    case ZE_GRAPH_ARGUMENT_PRECISION_FP64:
        return ov::element::Type_t::f64;
    case ZE_GRAPH_ARGUMENT_PRECISION_INT4:
        return ov::element::Type_t::i4;
    case ZE_GRAPH_ARGUMENT_PRECISION_INT8:
        return ov::element::Type_t::i8;
    case ZE_GRAPH_ARGUMENT_PRECISION_INT16:
        return ov::element::Type_t::i16;
    case ZE_GRAPH_ARGUMENT_PRECISION_INT32:
        return ov::element::Type_t::i32;
    case ZE_GRAPH_ARGUMENT_PRECISION_INT64:
        return ov::element::Type_t::i64;
    case ZE_GRAPH_ARGUMENT_PRECISION_BIN:
        return ov::element::Type_t::u1;
    case ZE_GRAPH_ARGUMENT_PRECISION_UINT4:
        return ov::element::Type_t::u4;
    case ZE_GRAPH_ARGUMENT_PRECISION_UINT8:
        return ov::element::Type_t::u8;
    case ZE_GRAPH_ARGUMENT_PRECISION_UINT16:
        return ov::element::Type_t::u16;
    case ZE_GRAPH_ARGUMENT_PRECISION_UINT32:
        return ov::element::Type_t::u32;
    case ZE_GRAPH_ARGUMENT_PRECISION_UINT64:
        return ov::element::Type_t::u64;
    default:
        return ov::element::Type_t::undefined;
    }
}

/**
 * @brief For driver backward compatibility reasons, the given value shall be converted to a string corresponding to the
 * adequate legacy precision.
 */
std::string ovPrecisionToLegacyPrecisionString(const ov::element::Type& precision) {
    switch (precision) {
    case ov::element::Type_t::undefined:
        return "UNSPECIFIED";
    case ov::element::Type_t::f16:
        return "FP16";
    case ov::element::Type_t::f32:
        return "FP32";
    case ov::element::Type_t::f64:
        return "FP64";
    case ov::element::Type_t::bf16:
        return "BF16";
    case ov::element::Type_t::i4:
        return "I4";
    case ov::element::Type_t::i8:
        return "I8";
    case ov::element::Type_t::i16:
        return "I16";
    case ov::element::Type_t::i32:
        return "I32";
    case ov::element::Type_t::i64:
        return "I64";
    case ov::element::Type_t::u4:
        return "U4";
    case ov::element::Type_t::u8:
        return "U8";
    case ov::element::Type_t::u16:
        return "U16";
    case ov::element::Type_t::u32:
        return "U32";
    case ov::element::Type_t::u64:
        return "U64";
    case ov::element::Type_t::u1:
        return "BIN";
    case ov::element::Type_t::boolean:
        return "BOOL";
    case ov::element::Type_t::dynamic:
        return "DYNAMIC";
    default:
        OPENVINO_THROW("Incorrect precision: ", precision);
    }
}

/**
 * @brief Gives the string representation of the default legacy layout value corresponding to the given rank.
 * @details This is done in order to assure the backward compatibility with the driver. Giving a layout different from
 * the default one may lead either to error or to accuracy failures since unwanted transposition layers may be
 * introduced.
 */
std::string rankToLegacyLayoutString(const size_t rank) {
    switch (rank) {
    case 0:
        return "**SCALAR**";
    case 1:
        return "C";
    case 2:
        return "NC";
    case 3:
        return "CHW";
    case 4:
        return "NCHW";
    case 5:
        return "NCDHW";
    default:
        return "BLOCKED";
    }
}

size_t zeLayoutToRank(const ze_graph_argument_layout_t layout) {
    switch (layout) {
    case ZE_GRAPH_ARGUMENT_LAYOUT_C:
        return 1;
    case ZE_GRAPH_ARGUMENT_LAYOUT_CN:
        return 2;
    case ZE_GRAPH_ARGUMENT_LAYOUT_HW:
        return 2;
    case ZE_GRAPH_ARGUMENT_LAYOUT_NC:
        return 2;
    case ZE_GRAPH_ARGUMENT_LAYOUT_CHW:
        return 3;
    case ZE_GRAPH_ARGUMENT_LAYOUT_NCHW:
        return 4;
    case ZE_GRAPH_ARGUMENT_LAYOUT_NHWC:
        return 4;
    case ZE_GRAPH_ARGUMENT_LAYOUT_NCDHW:
        return 5;
    case ZE_GRAPH_ARGUMENT_LAYOUT_NDHWC:
        return 5;
    default:
        // TODO #-30200 Extend to support all cases
        return 0;
    }
}

/**
 * @brief Transposes the original shape value according to given layout.
 */
std::vector<size_t> reshapeByLayout(const std::vector<size_t>& originalDimensions,
                                    const ze_graph_argument_layout_t layout) {
    std::vector<size_t> order;
    std::vector<size_t> reshapedDimensions;

    switch (layout) {
    case ZE_GRAPH_ARGUMENT_LAYOUT_CN:
        order = NC_TO_CN_LAYOUT_DIMENSIONS_ORDER;
        break;
    case ZE_GRAPH_ARGUMENT_LAYOUT_NHWC:
        order = NCHW_TO_NHWC_LAYOUT_DIMENSIONS_ORDER;
        break;
    case ZE_GRAPH_ARGUMENT_LAYOUT_NDHWC:
        order = NCDHW_TO_NDHWC_LAYOUT_DIMENSIONS_ORDER;
        break;
    default:
        // TODO #-30200 Extend to support all cases
        return originalDimensions;
    }

    for (const size_t& orderElement : order) {
        reshapedDimensions.push_back(originalDimensions[orderElement]);
    }

    return reshapedDimensions;
}

}  // namespace

namespace intel_npu {
namespace driverCompilerAdapter {

template <typename TableExtension>
LevelZeroCompilerInDriver<TableExtension>::~LevelZeroCompilerInDriver() {
    if (_context) {
        auto result = zeContextDestroy(_context);
        if (ZE_RESULT_SUCCESS != result) {
            _logger.warning("zeContextDestroy failed %#X", uint64_t(result));
        }
    }
    _logger.debug("LevelZeroCompilerInDriver obj destroyed");
}

/**
 * @brief Place xml + weights in sequential memory
 * @details Format of the memory:
 */
template <typename TableExtension>
SerializedIR LevelZeroCompilerInDriver<TableExtension>::serializeIR(
    const std::shared_ptr<const ov::Model>& model,
    ze_graph_compiler_version_info_t compilerVersion) const {
    IRSerializer irSerializer(model, getSupportedOpsetVersion());

    // Contract between adapter and compiler in driver
    const uint32_t maxNumberOfElements = 10;
    const uint64_t maxSizeOfXML = std::numeric_limits<uint64_t>::max() / 3;
    const uint64_t maxSizeOfWeights = maxSizeOfXML * 2;

    const uint32_t numberOfInputData = 2;
    const uint64_t xmlSize = static_cast<uint64_t>(irSerializer.getXmlSize());
    const uint64_t weightsSize = static_cast<uint64_t>(irSerializer.getWeightsSize());

    OPENVINO_ASSERT(numberOfInputData < maxNumberOfElements);
    if (xmlSize >= maxSizeOfXML) {
        OPENVINO_THROW("LevelZeroCompilerInDriver: Xml file is too big to process. xmlSize: ",
                       xmlSize,
                       " >= maxSizeOfXML: ",
                       maxSizeOfXML);
    }
    if (weightsSize >= maxSizeOfWeights) {
        OPENVINO_THROW("LevelZeroCompilerInDriver: Bin file is too big to process. xmlSize: ",
                       weightsSize,
                       " >= maxSizeOfWeights: ",
                       maxSizeOfWeights);
    }

    const uint64_t sizeOfSerializedIR = sizeof(compilerVersion) + sizeof(numberOfInputData) + sizeof(xmlSize) +
                                        xmlSize + sizeof(weightsSize) + weightsSize;

    // use array to avoid vector's memory zeroing overhead
    std::shared_ptr<uint8_t> buffer(new uint8_t[sizeOfSerializedIR], std::default_delete<uint8_t[]>());
    uint8_t* serializedIR = buffer.get();

    uint64_t offset = 0;
    checkedMemcpy(serializedIR + offset, sizeOfSerializedIR - offset, &compilerVersion, sizeof(compilerVersion));
    offset += sizeof(compilerVersion);

    checkedMemcpy(serializedIR + offset, sizeOfSerializedIR - offset, &numberOfInputData, sizeof(numberOfInputData));
    offset += sizeof(numberOfInputData);
    checkedMemcpy(serializedIR + offset, sizeOfSerializedIR - offset, &xmlSize, sizeof(xmlSize));
    offset += sizeof(xmlSize);
    // xml data is filled in serializeModel()
    uint64_t xmlOffset = offset;
    offset += xmlSize;
    checkedMemcpy(serializedIR + offset, sizeOfSerializedIR - offset, &weightsSize, sizeof(weightsSize));
    offset += sizeof(weightsSize);
    // weights data is filled in serializeModel()
    uint64_t weightsOffset = offset;
    offset += weightsSize;

    irSerializer.serializeModelToBuffer(serializedIR + xmlOffset, serializedIR + weightsOffset);

    OPENVINO_ASSERT(offset == sizeOfSerializedIR);

    return std::make_pair(sizeOfSerializedIR, buffer);
}

template <typename TableExtension>
std::string LevelZeroCompilerInDriver<TableExtension>::serializeIOInfo(const std::shared_ptr<const ov::Model>& model) {
    const ov::ParameterVector& parameters = model->get_parameters();
    const ov::ResultVector& results = model->get_results();

    std::stringstream inputsPrecisionSS;
    std::stringstream inputsLayoutSS;
    std::stringstream outputsPrecisionSS;
    std::stringstream outputsLayoutSS;

    inputsPrecisionSS << INPUTS_PRECISIONS_KEY << KEY_VALUE_SEPARATOR << VALUE_DELIMITER;
    inputsLayoutSS << INPUTS_LAYOUTS_KEY << KEY_VALUE_SEPARATOR << VALUE_DELIMITER;

    if (!parameters.empty()) {
        const std::string& firstInputName = parameters.at(0)->get_friendly_name();

        for (const std::shared_ptr<ov::op::v0::Parameter>& parameter : parameters) {
            const std::string& name = parameter->get_friendly_name();
            const ov::element::Type& precision = parameter->get_element_type();
            const size_t rank = parameter->get_shape().size();

            if (name != firstInputName) {
                inputsPrecisionSS << VALUES_SEPARATOR;
                inputsLayoutSS << VALUES_SEPARATOR;
            }

            inputsPrecisionSS << name << NAME_VALUE_SEPARATOR << ovPrecisionToLegacyPrecisionString(precision);
            // Ticket: E-88902
            inputsLayoutSS << name << NAME_VALUE_SEPARATOR << rankToLegacyLayoutString(rank);
        }
    }

    inputsPrecisionSS << VALUE_DELIMITER;
    inputsLayoutSS << VALUE_DELIMITER;

    outputsPrecisionSS << OUTPUTS_PRECISIONS_KEY << KEY_VALUE_SEPARATOR << VALUE_DELIMITER;
    outputsLayoutSS << OUTPUTS_LAYOUTS_KEY << KEY_VALUE_SEPARATOR << VALUE_DELIMITER;

    const std::string& firstOutputName = results.at(0)->get_input_node_ptr(0)->get_friendly_name();

    for (const std::shared_ptr<ov::op::v0::Result>& result : results) {
        const std::string& name = result->get_input_node_ptr(0)->get_friendly_name();
        const ov::element::Type_t precision = result->get_element_type();
        const size_t rank = result->get_shape().size();

        if (name != firstOutputName) {
            outputsPrecisionSS << VALUES_SEPARATOR;
            outputsLayoutSS << VALUES_SEPARATOR;
        }

        outputsPrecisionSS << name << NAME_VALUE_SEPARATOR << ovPrecisionToLegacyPrecisionString(precision);
        outputsLayoutSS << name << NAME_VALUE_SEPARATOR << rankToLegacyLayoutString(rank);
    }

    outputsPrecisionSS << VALUE_DELIMITER;
    outputsLayoutSS << VALUE_DELIMITER;

    // One line without spaces to avoid parsing as config option inside CID
    return inputsPrecisionSS.str() + VALUES_SEPARATOR.data() + inputsLayoutSS.str() + VALUES_SEPARATOR.data() +
           outputsPrecisionSS.str() + VALUES_SEPARATOR.data() + outputsLayoutSS.str();
}

template <typename TableExtension>
std::string LevelZeroCompilerInDriver<TableExtension>::serializeConfig(
    const Config& config,
    ze_graph_compiler_version_info_t& compilerVersion) const {
    std::string content = config.toString();
    _logger.debug("Original content of config: %s", content.c_str());

    // Remove optimization-level and performance-hint-override for old driver which not support them
    if ((compilerVersion.major < 5) || (compilerVersion.major == 5 && compilerVersion.minor < 7)) {
        std::string valueOfParams = config.get<COMPILATION_MODE_PARAMS>();
        std::string keyOfOptL("optimization-level");
        std::string keyOfPerfHO("performance-hint-override");
        if (valueOfParams != "" && (valueOfParams.find(keyOfOptL) != std::string::npos ||
                                    valueOfParams.find(keyOfPerfHO) != std::string::npos)) {
            // Remove unsupported options from value
            std::ostringstream optLevelStr;
            optLevelStr << keyOfOptL << KEY_VALUE_SEPARATOR << "\\d+";
            std::ostringstream perfHintStr;
            perfHintStr << keyOfPerfHO << KEY_VALUE_SEPARATOR << "\\S+";
            _logger.warning("%s property is not suppored by this compiler version. Removing from parameters",
                            keyOfOptL.c_str());
            valueOfParams = std::regex_replace(valueOfParams, std::regex(optLevelStr.str()), "");
            _logger.warning("%s property is not suppored by this compiler version. Removing from parameters",
                            keyOfPerfHO.c_str());
            valueOfParams = std::regex_replace(valueOfParams, std::regex(perfHintStr.str()), "");

            // Trim space
            valueOfParams = std::regex_replace(valueOfParams, std::regex(R"(^\s+|\s+$)"), "");

            // Replace the value in content with new value
            std::ostringstream compilationParamsStr;
            compilationParamsStr << ov::intel_npu::compilation_mode_params.name() << KEY_VALUE_SEPARATOR
                                 << VALUE_DELIMITER << ".*" << VALUE_DELIMITER;
            if (valueOfParams == "") {
                _logger.warning("Clear empty NPU_COMPILATION_MODE_PARAMS. Removing from parameters");
                content = std::regex_replace(content, std::regex(compilationParamsStr.str()), "");
            } else {
                std::ostringstream newValue;
                newValue << ov::intel_npu::compilation_mode_params.name() << KEY_VALUE_SEPARATOR << VALUE_DELIMITER
                         << valueOfParams << VALUE_DELIMITER;
                _logger.warning("Replace value of NPU_COMPILATION_MODE_PARAMS with new value %s",
                                newValue.str().c_str());
                content = std::regex_replace(content, std::regex(compilationParamsStr.str()), newValue.str().c_str());
            }
        }
    }

    // As a consequence of complying to the conventions established in the 2.0 OV API, the set of values corresponding
    // to the "model priority" key has been modified
    // cpu_pinning property is not supported in compilers < v5.2 - need to remove it
    if ((compilerVersion.major < 5) || (compilerVersion.major == 5 && compilerVersion.minor < 2)) {
        const auto& getTargetRegex = [](const ov::hint::Priority& priorityValue) -> std::regex {
            std::ostringstream result;
            result << ov::hint::model_priority.name() << KEY_VALUE_SEPARATOR << VALUE_DELIMITER << priorityValue
                   << VALUE_DELIMITER;
            return std::regex(result.str());
        };
        const auto& getStringReplacement = [](const ov::intel_npu::LegacyPriority& priorityValue) -> std::string {
            std::ostringstream result;
            result << ov::intel_npu::legacy_model_priority.name() << KEY_VALUE_SEPARATOR << VALUE_DELIMITER
                   << priorityValue << VALUE_DELIMITER;
            return result.str();
        };

        // E.g. (valid as of writing this): MODEL_PRIORITY="MEDIUM" -> MODEL_PRIORITY="MODEL_PRIORITY_MED"
        content = std::regex_replace(content,
                                     getTargetRegex(ov::hint::Priority::LOW),
                                     getStringReplacement(ov::intel_npu::LegacyPriority::LOW));
        content = std::regex_replace(content,
                                     getTargetRegex(ov::hint::Priority::MEDIUM),
                                     getStringReplacement(ov::intel_npu::LegacyPriority::MEDIUM));
        content = std::regex_replace(content,
                                     getTargetRegex(ov::hint::Priority::HIGH),
                                     getStringReplacement(ov::intel_npu::LegacyPriority::HIGH));

        // Removing cpu_pinning from the command string
        std::ostringstream pinningstr;
        pinningstr << ov::hint::enable_cpu_pinning.name() << KEY_VALUE_SEPARATOR << VALUE_DELIMITER << "\\S+"
                   << VALUE_DELIMITER;
        _logger.warning(
            "ENABLE_CPU_PINNING property is not suppored by this compiler version. Removing from parameters");
        content = std::regex_replace(content, std::regex(pinningstr.str()), "");
    }

    /// Stepping and max_tiles are not supported in versions < 5.3 - need to remove it
    if ((compilerVersion.major < 5) || (compilerVersion.major == 5 && compilerVersion.minor < 3)) {
        std::ostringstream stepstr;
        stepstr << ov::intel_npu::stepping.name() << KEY_VALUE_SEPARATOR << VALUE_DELIMITER << "\\d+"
                << VALUE_DELIMITER;
        std::ostringstream maxtilestr;
        maxtilestr << ov::intel_npu::max_tiles.name() << KEY_VALUE_SEPARATOR << VALUE_DELIMITER << "\\d+"
                   << VALUE_DELIMITER;
        _logger.warning("NPU_STEPPING property is not suppored by this compiler version. Removing from parameters");
        content = std::regex_replace(content, std::regex(stepstr.str()), "");
        _logger.warning("NPU_MAX_TILES property is not suppored by this compiler version. Removing from parameters");
        content = std::regex_replace(content, std::regex(maxtilestr.str()), "");
    }

    /// Removing INFERENCE_PRECISION_HINT for older compilers
    if ((compilerVersion.major < 5) || (compilerVersion.major == 5 && compilerVersion.minor < 4)) {
        std::ostringstream precstr;
        precstr << ov::hint::inference_precision.name() << KEY_VALUE_SEPARATOR << VALUE_DELIMITER << "\\S+"
                << VALUE_DELIMITER;
        _logger.warning(
            "INFERENCE_PRECISION_HINT property is not suppored by this compiler version. Removing from parameters");
        content = std::regex_replace(content, std::regex(precstr.str()), "");
    }

    /// Replacing NPU_TILES (for all versions) with NPU_DPU_GROUPS for backwards compatibility
    if (std::regex_search(content, std::regex(ov::intel_npu::tiles.name()))) {
        _logger.warning("NPU_TILES property is not suppored by this compiler version. Swaping it to "
                        "NPU_DPU_GROUPS (obsolete)");
        content = std::regex_replace(content, std::regex(ov::intel_npu::tiles.name()), "NPU_DPU_GROUPS");
    }

    // Batch mode property is not supported in versions < 5.5 - need to remove it
    if ((compilerVersion.major < 5) || (compilerVersion.major == 5 && compilerVersion.minor < 5)) {
        std::ostringstream batchstr;
        batchstr << ov::intel_npu::batch_mode.name() << KEY_VALUE_SEPARATOR << VALUE_DELIMITER << "\\S+"
                 << VALUE_DELIMITER;

        _logger.warning("NPU_BATCH_MODE property is not suppored by this compiler version. Removing from parameters");
        content = std::regex_replace(content, std::regex(batchstr.str()), "");
    }

    // EXECUTION_MODE_HINT is not supported in versions < 5.6 - need to remove it
    if ((compilerVersion.major < 5) || (compilerVersion.major == 5 && compilerVersion.minor < 6)) {
        std::ostringstream batchstr;
        batchstr << ov::hint::execution_mode.name() << KEY_VALUE_SEPARATOR << VALUE_DELIMITER << "\\S+"
                 << VALUE_DELIMITER;
        _logger.warning(
            "EXECUTION_MODE_HINT property is not suppored by this compiler version. Removing from parameters");
        content = std::regex_replace(content, std::regex(batchstr.str()), "");
    }

    // Remove the properties that are not used by the compiler
    // WorkloadType is used only by compiled model
    std::ostringstream workloadtypestr;
    workloadtypestr << ov::workload_type.name() << KEY_VALUE_SEPARATOR << VALUE_DELIMITER << "\\S+" << VALUE_DELIMITER;
    content = std::regex_replace(content, std::regex(workloadtypestr.str()), "");
    // Remove turbo property as it is not used by compiler
    std::ostringstream turbostring;
    turbostring << ov::intel_npu::turbo.name() << KEY_VALUE_SEPARATOR << VALUE_DELIMITER << "\\S+" << VALUE_DELIMITER;
    content = std::regex_replace(content, std::regex(turbostring.str()), "");

    // FINAL step to convert prefixes of remaining params, to ensure backwards compatibility
    // From 5.0.0, driver compiler start to use NPU_ prefix, the old version uses VPU_ prefix
    if (compilerVersion.major < 5) {
        std::regex reg("NPU_");
        content = std::regex_replace(content, reg, "VPU_");
        // From 4.0.0, driver compiler start to use VPU_ prefix, the old version uses VPUX_ prefix
        if (compilerVersion.major < 4) {
            // Replace VPU_ with VPUX_ for old driver compiler
            std::regex reg("VPU_");
            content = std::regex_replace(content, reg, "VPUX_");
        }
    }

    return "--config " + content;
}

// Parse the result string of query from foramt <name_0><name_1><name_2> to unordered_set of string
static std::unordered_set<std::string> parseQueryResult(std::vector<char>& data) {
    std::string dataString(data.begin(), data.end());
    std::unordered_set<std::string> result;
    size_t i = 0, start = 0;
    while (i < dataString.length()) {
        if (dataString[i] == '<') {
            start = ++i;
        } else if (dataString[i] == '>') {
            std::string temp(dataString.begin() + start, dataString.begin() + i);
            result.insert(temp);
            i++;
        } else {
            i++;
        }
    }
    return result;
}

// For ext version < 1.3, query is unsupported, return empty result and add debug log here
template <typename TableExtension>
template <typename T, std::enable_if_t<NotSupportQuery(T), bool>>
std::unordered_set<std::string> LevelZeroCompilerInDriver<TableExtension>::queryImpl(
    const std::shared_ptr<const ov::Model>& /*model*/,
    const Config&) const {
    _logger.debug("queryImpl - Driver version is less than 1.3, queryNetwork is unsupported.");
    return std::unordered_set<std::string>();
}

// For ext version == 1.3 && == 1.4
template <typename TableExtension>
template <typename T, std::enable_if_t<SupportAPIGraphQueryNetworkV1(T), bool>>
ze_result_t LevelZeroCompilerInDriver<TableExtension>::seriazlideIRModelAndQueryNetworkCreateV1(
    const std::shared_ptr<const ov::Model>& model,
    const Config& config,
    ze_device_graph_properties_t deviceGraphProperties,
    const ze_device_handle_t& _deviceHandle,
    ze_graph_query_network_handle_t& hGraphQueryNetwork) const {
    ze_graph_compiler_version_info_t& compilerVersion = deviceGraphProperties.compilerVersion;

    auto serializedIR = serializeIR(model, compilerVersion);

    std::string buildFlags;
    buildFlags += serializeConfig(config, compilerVersion);
    _logger.debug("queryImpl build flags : %s", buildFlags.c_str());

    ze_graph_desc_t desc = {ZE_STRUCTURE_TYPE_GRAPH_DESC_PROPERTIES,
                            nullptr,
                            ZE_GRAPH_FORMAT_NGRAPH_LITE,
                            serializedIR.first,
                            serializedIR.second.get(),
                            buildFlags.c_str()};

    // Create querynetwork handle
    ze_result_t result = _graphDdiTableExt->pfnQueryNetworkCreate(_context, _deviceHandle, &desc, &hGraphQueryNetwork);

    return result;
}

// For ext version == 1.3 && == 1.4, query is supported, calling querynetwork api in _graphDdiTableExt
template <typename TableExtension>
template <typename T, std::enable_if_t<SupportAPIGraphQueryNetworkV1(T), bool>>
std::unordered_set<std::string> LevelZeroCompilerInDriver<TableExtension>::queryImpl(
    const std::shared_ptr<const ov::Model>& model,
    const Config& config) const {
    _logger.debug("queryImpl - Calling queryNetwork of 1.3 version.");

    ze_device_graph_properties_t deviceGraphProperties{};
    auto result = _graphDdiTableExt->pfnDeviceGetGraphProperties(_deviceHandle, &deviceGraphProperties);
    if (ZE_RESULT_SUCCESS != result) {
        OPENVINO_THROW("L0 pfnDeviceGetGraphProperties",
                       " result: ",
                       ze_result_to_string(result),
                       ", code 0x",
                       std::hex,
                       uint64_t(result));
    }

    ze_graph_query_network_handle_t hGraphQueryNetwork = nullptr;

    result = seriazlideIRModelAndQueryNetworkCreateV1(model,
                                                      config,
                                                      deviceGraphProperties,
                                                      _deviceHandle,
                                                      hGraphQueryNetwork);

    return getQueryResultFromSupportedLayers(result, hGraphQueryNetwork);
}

// For ext version >= 1.5
template <typename TableExtension>
template <typename T, std::enable_if_t<SupportAPIGraphQueryNetworkV2(T), bool>>
ze_result_t LevelZeroCompilerInDriver<TableExtension>::seriazlideIRModelAndQueryNetworkCreateV2(
    const std::shared_ptr<const ov::Model>& model,
    const Config& config,
    ze_device_graph_properties_t deviceGraphProperties,
    const ze_device_handle_t& _deviceHandle,
    ze_graph_query_network_handle_t& hGraphQueryNetwork) const {
    ze_graph_compiler_version_info_t& compilerVersion = deviceGraphProperties.compilerVersion;

    auto serializedIR = serializeIR(model, compilerVersion);

    std::string buildFlags;
    buildFlags += serializeConfig(config, compilerVersion);
    _logger.debug("queryImpl build flags : %s", buildFlags.c_str());

    ze_graph_desc_2_t desc = {ZE_STRUCTURE_TYPE_GRAPH_DESC_PROPERTIES,
                              nullptr,
                              ZE_GRAPH_FORMAT_NGRAPH_LITE,
                              serializedIR.first,
                              serializedIR.second.get(),
                              buildFlags.c_str(),
                              ZE_GRAPH_FLAG_NONE};

    // Create querynetwork handle
    ze_result_t result = _graphDdiTableExt->pfnQueryNetworkCreate2(_context, _deviceHandle, &desc, &hGraphQueryNetwork);

    return result;
}

// For ext version >= 1.5
template <typename TableExtension>
template <typename T, std::enable_if_t<SupportAPIGraphQueryNetworkV2(T), bool>>
std::unordered_set<std::string> LevelZeroCompilerInDriver<TableExtension>::queryImpl(
    const std::shared_ptr<const ov::Model>& model,
    const Config& config) const {
    _logger.debug("queryImpl - Calling queryNetwork of 1.5 version.");

    ze_device_graph_properties_t deviceGraphProperties{};
    auto result = _graphDdiTableExt->pfnDeviceGetGraphProperties(_deviceHandle, &deviceGraphProperties);
    if (ZE_RESULT_SUCCESS != result) {
        OPENVINO_THROW("L0 pfnDeviceGetGraphProperties",
                       " result: ",
                       ze_result_to_string(result),
                       ", code 0x",
                       std::hex,
                       uint64_t(result));
    }

    ze_graph_query_network_handle_t hGraphQueryNetwork = nullptr;

    result = seriazlideIRModelAndQueryNetworkCreateV2(model,
                                                      config,
                                                      deviceGraphProperties,
                                                      _deviceHandle,
                                                      hGraphQueryNetwork);

    return getQueryResultFromSupportedLayers(result, hGraphQueryNetwork);
}

template <typename TableExtension>
template <typename T, std::enable_if_t<!NotSupportQuery(T), bool>>
std::unordered_set<std::string> LevelZeroCompilerInDriver<TableExtension>::getQueryResultFromSupportedLayers(
    ze_result_t result,
    ze_graph_query_network_handle_t& hGraphQueryNetwork) const {
    if (ZE_RESULT_SUCCESS != result) {
        OPENVINO_THROW("L0 getQueryResultFromSupportedLayers",
                       " result: ",
                       ze_result_to_string(result),
                       ", code 0x",
                       std::hex,
                       uint64_t(result));
    }

    // Get the size of query result
    size_t size = 0;
    result = _graphDdiTableExt->pfnQueryNetworkGetSupportedLayers(hGraphQueryNetwork, &size, nullptr);
    if (ZE_RESULT_SUCCESS != result) {
        _graphDdiTableExt->pfnQueryNetworkDestroy(hGraphQueryNetwork);
        OPENVINO_THROW("L0 pfnQueryNetworkGetSupportedLayers get size of query result",
                       " result: ",
                       ze_result_to_string(result),
                       ", code 0x",
                       std::hex,
                       uint64_t(result));
    }

    // Get the result data of query
    std::vector<char> supportedLayers(size);
    result = _graphDdiTableExt->pfnQueryNetworkGetSupportedLayers(hGraphQueryNetwork, &size, supportedLayers.data());
    if (ZE_RESULT_SUCCESS != result) {
        _graphDdiTableExt->pfnQueryNetworkDestroy(hGraphQueryNetwork);
        OPENVINO_THROW("L0 pfnQueryNetworkGetSupportedLayers get result data of query",
                       " result: ",
                       ze_result_to_string(result),
                       ", code 0x",
                       std::hex,
                       uint64_t(result));
    }

    result = _graphDdiTableExt->pfnQueryNetworkDestroy(hGraphQueryNetwork);
    if (ZE_RESULT_SUCCESS != result) {
        OPENVINO_THROW("L0 pfnQueryNetworkDestroy",
                       " result: ",
                       ze_result_to_string(result),
                       ", code 0x",
                       std::hex,
                       uint64_t(result));
    }

    return parseQueryResult(supportedLayers);
}

template <typename TableExtension>
ov::SupportedOpsMap LevelZeroCompilerInDriver<TableExtension>::query(const std::shared_ptr<const ov::Model>& model,
                                                                     const Config& config) const {
    _logger.setLevel(config.get<LOG_LEVEL>());
    _logger.debug("query");

    ov::SupportedOpsMap result;
    const std::string deviceName = "NPU";

    try {
        const auto supportedLayers = queryImpl(model, config);
        ;
        for (auto&& layerName : supportedLayers) {
            result.emplace(layerName, deviceName);
        }
        _logger.info("For given model, there are %d supported layers", supportedLayers.size());
    } catch (std::exception& e) {
        OPENVINO_THROW("Fail in calling querynetwork : ", e.what());
    }

    _logger.debug("query end");
    return result;
}

// For ext version <1.5, calling pfnCreate api in _graphDdiTableExt
template <typename TableExtension>
template <typename T, std::enable_if_t<NotSupportGraph2(T), bool>>
ze_result_t LevelZeroCompilerInDriver<TableExtension>::createGraph(const ze_graph_format_t& format,
                                                                   const SerializedIR& serializedIR,
                                                                   const std::string& buildFlags,
                                                                   const uint32_t& /*flags*/,
                                                                   ze_graph_handle_t* graph) const {
    ze_graph_desc_t desc = {ZE_STRUCTURE_TYPE_GRAPH_DESC_PROPERTIES,
                            nullptr,
                            format,
                            serializedIR.first,
                            serializedIR.second.get(),
                            buildFlags.c_str()};

    // Create querynetwork handle
    return _graphDdiTableExt->pfnCreate(_context, _deviceHandle, &desc, graph);
}

// For ext version >= 1.5, calling pfnCreate2 api in _graphDdiTableExt
template <typename TableExtension>
template <typename T, std::enable_if_t<!NotSupportGraph2(T), bool>>
ze_result_t LevelZeroCompilerInDriver<TableExtension>::createGraph(const ze_graph_format_t& format,
                                                                   const SerializedIR& serializedIR,
                                                                   const std::string& buildFlags,
                                                                   const uint32_t& flags,
                                                                   ze_graph_handle_t* graph) const {
    ze_graph_desc_2_t desc = {ZE_STRUCTURE_TYPE_GRAPH_DESC_PROPERTIES,
                              nullptr,
                              format,
                              serializedIR.first,
                              serializedIR.second.get(),
                              buildFlags.c_str(),
                              flags};

    // Create querynetwork handle
    return _graphDdiTableExt->pfnCreate2(_context, _deviceHandle, &desc, graph);
}
template <typename TableExtension>
ze_result_t LevelZeroCompilerInDriver<TableExtension>::seriazlideIRModelAndCreateGraph(
    const std::shared_ptr<const ov::Model>& model,
    const Config& config,
    ze_device_graph_properties_t deviceGraphProperties,
    ze_graph_handle_t& graphHandle) const {
    const ze_graph_compiler_version_info_t& compilerVersion = deviceGraphProperties.compilerVersion;
    auto serializedIR = serializeIR(model, compilerVersion);

    ze_graph_format_t format = ZE_GRAPH_FORMAT_NGRAPH_LITE;

    std::string buildFlags;

    buildFlags += serializeIOInfo(model);
    buildFlags += " ";
    buildFlags += serializeConfig(config, const_cast<ze_graph_compiler_version_info_t&>(compilerVersion));

    _logger.debug("compileIR Build flags : %s", buildFlags.c_str());

    // If OV cache is enabled, disable driver caching
    uint32_t flags = ZE_GRAPH_FLAG_NONE;
    const auto set_cache_dir = config.get<CACHE_DIR>();
    if (!set_cache_dir.empty()) {
        flags = flags | ZE_GRAPH_FLAG_DISABLE_CACHING;
    }

    _logger.info("compileIR Using extension version: %s", typeid(TableExtension).name());
    ze_result_t result = createGraph(format, serializedIR, buildFlags, flags, &graphHandle);
    return result;
}

template <typename TableExtension>
NetworkDescription LevelZeroCompilerInDriver<TableExtension>::compile(const std::shared_ptr<const ov::Model>& model,
                                                                      const Config& config) const {
    _logger.setLevel(config.get<LOG_LEVEL>());
    _logger.debug("compile");

    ze_device_graph_properties_t deviceGraphProperties{};
    auto result = _graphDdiTableExt->pfnDeviceGetGraphProperties(_deviceHandle, &deviceGraphProperties);
    if (ZE_RESULT_SUCCESS != result) {
        OPENVINO_THROW("Failed to compile network. L0 pfnDeviceGetGraphProperties",
                       " result: ",
                       ze_result_to_string(result),
                       ", code 0x",
                       std::hex,
                       uint64_t(result));
    }

    // Graph handle should be used only in scope of compile / parse functions.
    ze_graph_handle_t graphHandle;

    result = seriazlideIRModelAndCreateGraph(model, config, deviceGraphProperties, graphHandle);

    OPENVINO_ASSERT(result == ZE_RESULT_SUCCESS,
                    "Failed to compile network. L0 createGraph",
                    " result: ",
                    ze_result_to_string(result),
                    ", code 0x",
                    std::hex,
                    uint64_t(result),
                    ". ",
                    getLatestBuildError());

    // Get blob size first
    size_t blobSize = -1;

    result = _graphDdiTableExt->pfnGetNativeBinary(graphHandle, &blobSize, nullptr);

    OPENVINO_ASSERT(result == ZE_RESULT_SUCCESS,
                    "Failed to compile network. L0 pfnGetNativeBinary get blob size",
                    " result: ",
                    ze_result_to_string(result),
                    ", code 0x",
                    std::hex,
                    uint64_t(result),
                    ". ",
                    getLatestBuildError());

    std::vector<uint8_t> blob(blobSize);
    // Get blob data
    result = _graphDdiTableExt->pfnGetNativeBinary(graphHandle, &blobSize, blob.data());

    OPENVINO_ASSERT(result == ZE_RESULT_SUCCESS,
                    "Failed to compile network. L0 pfnGetNativeBinary get blob data",
                    " result: ",
                    ze_result_to_string(result),
                    ", code 0x",
                    std::hex,
                    uint64_t(result),
                    ". ",
                    getLatestBuildError());

    auto networkMeta = getNetworkMeta(graphHandle);
    networkMeta.name = model->get_friendly_name();

    result = _graphDdiTableExt->pfnDestroy(graphHandle);

    if (ZE_RESULT_SUCCESS != result) {
        OPENVINO_THROW("Failed to compile network. L0 pfnDestroy",
                       " result: ",
                       ze_result_to_string(result),
                       ", code 0x",
                       std::hex,
                       uint64_t(result));
    }

    _logger.debug("compile end");
    return NetworkDescription(std::move(blob), std::move(networkMeta));
}

template <typename TableExtension>
NetworkMetadata LevelZeroCompilerInDriver<TableExtension>::parse(const std::vector<uint8_t>& network,
                                                                 const Config& config) const {
    OV_ITT_TASK_CHAIN(PARSE_BLOB, itt::domains::NPUPlugin, "LevelZeroCompilerInDriver::parse", "desc");
    _logger.setLevel(config.get<LOG_LEVEL>());
    _logger.debug("getNetworkMeta");
    ze_graph_handle_t graphHandle;

    if (!network.empty()) {
        _logger.debug("Import network case");
        ze_graph_format_t format = ZE_GRAPH_FORMAT_NATIVE;
        ze_graph_desc_t desc{ZE_STRUCTURE_TYPE_GRAPH_DESC_PROPERTIES,
                             nullptr,
                             format,
                             network.size(),
                             network.data(),
                             nullptr};

        auto result = _graphDdiTableExt->pfnCreate(_context, _deviceHandle, &desc, &graphHandle);
        OV_ITT_TASK_NEXT(PARSE_BLOB, "_graphDdiTableExt");

        if (ZE_RESULT_SUCCESS != result) {
            OPENVINO_THROW("L0 pfnCreate",
                           " result: ",
                           ze_result_to_string(result),
                           ", code 0x",
                           std::hex,
                           uint64_t(result));
        }
    } else {
        OPENVINO_THROW("Empty blob");
    }

    OV_ITT_TASK_NEXT(PARSE_BLOB, "getNetworkMeta");
    const auto networkMeta = getNetworkMeta(graphHandle);
    OV_ITT_TASK_NEXT(PARSE_BLOB, "NetworkDescription");

    auto result = _graphDdiTableExt->pfnDestroy(graphHandle);

    if (ZE_RESULT_SUCCESS != result) {
        OPENVINO_THROW("L0 pfnDestroy",
                       " result: ",
                       ze_result_to_string(result),
                       ", code 0x",
                       std::hex,
                       uint64_t(result));
    }

    return networkMeta;
}

template <typename TableExtension>
uint32_t LevelZeroCompilerInDriver<TableExtension>::getSupportedOpsetVersion() const {
    _logger.debug("getSupportedOpsetVersion");
    ze_device_graph_properties_t graphProperties;

    auto result = _graphDdiTableExt->pfnDeviceGetGraphProperties(_deviceHandle, &graphProperties);

    if (ZE_RESULT_SUCCESS != result) {
        OPENVINO_THROW("L0 pfnDeviceGetGraphProperties",
                       " result: ",
                       ze_result_to_string(result),
                       ", code 0x",
                       std::hex,
                       uint64_t(result));
    }
    const auto maxOpsetVersion = graphProperties.maxOVOpsetVersionSupported;
    _logger.info("getSupportedOpsetVersion Max supported version of opset in CiD: %d", maxOpsetVersion);
    return maxOpsetVersion;
}

template <typename TableExtension>
template <typename T>
void LevelZeroCompilerInDriver<TableExtension>::getLayoutOrStateDescriptor(IONodeDescriptorMap& parameters,
                                                                           IONodeDescriptorMap& results,
                                                                           IONodeDescriptorMap& states,
                                                                           std::vector<std::string>& stateNames,
                                                                           const T& arg) const {
    std::string legacyName = arg.name;

    // The layout may differ from the default one only when using significantly older drivers. In order to accommodate
    // this case, an extra attribute needs to be stored which holds the transposed shape.
    const std::vector<size_t> originalDimensions(arg.dims, arg.dims + zeLayoutToRank(arg.deviceLayout));
    const std::vector<size_t> reshapedDimensions = reshapeByLayout(originalDimensions, arg.deviceLayout);
    const ov::Shape shape = ov::Shape(reshapedDimensions);

    if (!isStateInputName(legacyName) && !isStateOutputName(legacyName)) {
        if (arg.type == ZE_GRAPH_ARGUMENT_TYPE_INPUT) {
            _logger.info("getLayoutOrStateDescriptor Found input \"%s\"", legacyName.c_str());

            parameters[legacyName].transposedShape = shape;
        }
        if (arg.type == ZE_GRAPH_ARGUMENT_TYPE_OUTPUT) {
            _logger.info("getLayoutOrStateDescriptor Found output \"%s\"", legacyName.c_str());

            results[legacyName].transposedShape = shape;
        }
    } else if (isStateInputName(legacyName)) {
        // The inputs and outputs of the state nodes share the same metadata, thus we'll consider only the the inputs
        // here
        legacyName = legacyName.substr(READVALUE_PREFIX.length());
        _logger.info("getLayoutOrStateDescriptor Found state variable \"%s\"", legacyName.c_str());

        const ov::element::Type_t precision = toOVElementType(arg.devicePrecision);

        stateNames.push_back(legacyName);
        states[legacyName] = {legacyName, "", {}, precision, shape, shape};
    }
}

/**
 * @brief Extracts the parameter/result (i.e. input/output) descriptors from Level Zero specific structures into
 * OpenVINO specific ones.
 * @param nodeDescriptors The map in which the result shall be stored.
 * @param names The I/O identifiers shall be stored here in the order found within the compiled model.
 * @param metadata The Level Zero structure fomr which the descriptors will be extracted.
 */
static void getNodeDescriptor(IONodeDescriptorMap& nodeDescriptors,
                              std::vector<std::string>& names,
                              ze_graph_argument_properties_3_t& arg) {
    ov::element::Type_t precision = toOVElementType(arg.devicePrecision);
    ov::Shape shape;
    std::unordered_set<std::string> outputTensorNames;

    for (uint32_t id = 0; id < arg.associated_tensor_names_count; id++) {
        outputTensorNames.insert(arg.associated_tensor_names[id]);
    }

    for (uint32_t id = 0; id < arg.dims_count; id++) {
        shape.push_back(arg.dims[id]);
    }

    const std::string& legacyName = arg.name;

    names.push_back(arg.debug_friendly_name);
    nodeDescriptors[arg.debug_friendly_name] =
        {legacyName, arg.debug_friendly_name, std::move(outputTensorNames), precision, shape, shape};
}

static void getNodeDescriptor(IONodeDescriptorMap& nodeDescriptors,
                              std::vector<std::string>& names,
                              ze_graph_argument_properties_3_t& arg,
                              ze_graph_argument_metadata_t& metadata) {
    ov::element::Type_t precision = toOVElementType(arg.devicePrecision);
    ov::Shape transposedShape, originalShape;
    std::unordered_set<std::string> outputTensorNames;

    for (uint32_t id = 0; id < arg.associated_tensor_names_count; id++) {
        outputTensorNames.insert(arg.associated_tensor_names[id]);
    }

    for (uint32_t id = 0; id < arg.dims_count; id++) {
        transposedShape.push_back(arg.dims[id]);
    }

    for (uint32_t id = 0; id < metadata.shape_size; id++) {
        originalShape.push_back(metadata.shape[id]);
    }

    const std::string& legacyName = arg.name;

    names.push_back(arg.debug_friendly_name);
    nodeDescriptors[arg.debug_friendly_name] =
        {legacyName, arg.debug_friendly_name, std::move(outputTensorNames), precision, originalShape, transposedShape};
}

template <typename TableExtension>
template <typename T, std::enable_if_t<NotSupportOriginalShape(T), bool>>
void LevelZeroCompilerInDriver<TableExtension>::getMetadata(TableExtension* graphDdiTableExt,
                                                            ze_graph_handle_t graphHandle,
                                                            uint32_t index,
                                                            std::vector<std::string>& inputNames,
                                                            std::vector<std::string>& outputNames,
                                                            std::vector<std::string>& stateNames,
                                                            IONodeDescriptorMap& parameters,
                                                            IONodeDescriptorMap& results,
                                                            IONodeDescriptorMap& states) const {
    ze_graph_argument_properties_3_t arg;
    auto result = graphDdiTableExt->pfnGetArgumentProperties3(graphHandle, index, &arg);
    if (ZE_RESULT_SUCCESS != result) {
        OPENVINO_THROW("L0 pfnGetArgumentProperties3",
                       " result: ",
                       ze_result_to_string(result),
                       ", code 0x",
                       std::hex,
                       uint64_t(result));
    }

    if (!isStateInputName(arg.name) && !isStateOutputName(arg.name)) {
        if (ZE_GRAPH_ARGUMENT_TYPE_INPUT == arg.type) {
            getNodeDescriptor(parameters, inputNames, arg);
        }

        if (ZE_GRAPH_ARGUMENT_TYPE_OUTPUT == arg.type) {
            getNodeDescriptor(results, outputNames, arg);
        }
    }

    getLayoutOrStateDescriptor(parameters, results, states, stateNames, arg);
}

template <typename TableExtension>
template <typename T, std::enable_if_t<!NotSupportOriginalShape(T), bool>>
void LevelZeroCompilerInDriver<TableExtension>::getMetadata(TableExtension* graphDdiTableExt,
                                                            ze_graph_handle_t graphHandle,
                                                            uint32_t index,
                                                            std::vector<std::string>& inputNames,
                                                            std::vector<std::string>& outputNames,
                                                            std::vector<std::string>& stateNames,
                                                            IONodeDescriptorMap& parameters,
                                                            IONodeDescriptorMap& results,
                                                            IONodeDescriptorMap& states) const {
    ze_graph_argument_properties_3_t arg;
    auto result = graphDdiTableExt->pfnGetArgumentProperties3(graphHandle, index, &arg);
    if (ZE_RESULT_SUCCESS != result) {
        OPENVINO_THROW("L0 pfnGetArgumentProperties3",
                       " result: ",
                       ze_result_to_string(result),
                       ", code 0x",
                       std::hex,
                       uint64_t(result));
    }

    if (!isStateInputName(arg.name) && !isStateOutputName(arg.name)) {
        ze_graph_argument_metadata_t metadata;
        result = graphDdiTableExt->pfnGraphGetArgumentMetadata(graphHandle, index, &metadata);
        if (ZE_RESULT_SUCCESS != result) {
            OPENVINO_THROW("L0 pfnGraphGetArgumentMetadata",
                           " result: ",
                           ze_result_to_string(result),
                           ", code 0x",
                           std::hex,
                           uint64_t(result));
        }

        if (ZE_GRAPH_ARGUMENT_TYPE_INPUT == arg.type) {
            getNodeDescriptor(parameters, inputNames, arg, metadata);
        }

        if (ZE_GRAPH_ARGUMENT_TYPE_OUTPUT == arg.type) {
            getNodeDescriptor(results, outputNames, arg, metadata);
        }
    }

    getLayoutOrStateDescriptor(parameters, results, states, stateNames, arg);
}

template <typename TableExtension>
NetworkMetadata LevelZeroCompilerInDriver<TableExtension>::getNetworkMeta(ze_graph_handle_t graphHandle) const {
    ze_graph_properties_t graphProperties{};

    auto result = _graphDdiTableExt->pfnGetProperties(graphHandle, &graphProperties);

    if (ZE_RESULT_SUCCESS != result) {
        OPENVINO_THROW("L0 pfnGetProperties",
                       " result: ",
                       ze_result_to_string(result),
                       ", code 0x",
                       std::hex,
                       uint64_t(result));
    }

    NetworkMetadata meta;

    for (uint32_t index = 0; index < graphProperties.numGraphArgs; ++index) {
        getMetadata(_graphDdiTableExt,
                    graphHandle,
                    index,
                    meta.inputNames,
                    meta.outputNames,
                    meta.stateNames,
                    meta.parameters,
                    meta.results,
                    meta.states);
    }
    // TODO: support this information in CiD [track: E#33479]
    meta.numStreams = 1;
    return meta;
}

template <typename TableExtension>
template <typename T, typename std::enable_if_t<!NotSupportLogHandle(T), bool>>
std::string LevelZeroCompilerInDriver<TableExtension>::getLatestBuildError() const {
    _logger.debug("getLatestBuildError()");

    // Get log size
    uint32_t size = 0;
    // Null graph handle to get erro log
    auto result = _graphDdiTableExt->pfnBuildLogGetString(nullptr, &size, nullptr);
    if (ZE_RESULT_SUCCESS != result) {
        // The failure will not break normal execution, only warning here
        _logger.warning("getLatestBuildError Failed to get size of latest error log!");
        return "";
    }

    if (size <= 0) {
        // The failure will not break normal execution, only warning here
        _logger.warning("getLatestBuildError No error log stored in driver when error "
                        "detected, may not be compiler issue!");
        return "";
    }

    // Get log content
    std::string logContent{};
    logContent.resize(size);
    result = _graphDdiTableExt->pfnBuildLogGetString(nullptr, &size, const_cast<char*>(logContent.data()));
    if (ZE_RESULT_SUCCESS != result) {
        // The failure will not break normal execution, only warning here
        _logger.warning("getLatestBuildError size of latest error log > 0, failed to get "
                        "content of latest error log!");
        return "";
    }
    return logContent;
}

template class LevelZeroCompilerInDriver<ze_graph_dditable_ext_1_2_t>;
template class LevelZeroCompilerInDriver<ze_graph_dditable_ext_1_3_t>;
template class LevelZeroCompilerInDriver<ze_graph_dditable_ext_1_4_t>;
template class LevelZeroCompilerInDriver<ze_graph_dditable_ext_1_5_t>;
template class LevelZeroCompilerInDriver<ze_graph_dditable_ext_1_6_t>;

}  // namespace driverCompilerAdapter
}  // namespace intel_npu
