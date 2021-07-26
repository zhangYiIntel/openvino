// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/matrix_nms.hpp"

namespace LayerTestsDefinitions {

using namespace ngraph;
using namespace InferenceEngine;
using namespace FuncTestUtils::PrecisionUtils;

std::string MatrixNmsLayerTest::getTestCaseName(testing::TestParamInfo<NmsParams> obj) {
    InputShapeParams inShapeParams;
    InputPrecisions inPrecisions;
    op::v8::MatrixNms::SortResultType sortResultType;
    element::Type outType;
    int backgroudClass;
    op::v8::MatrixNms::DecayFunction decayFunction;
    TopKParams topKParams;
    ThresholdParams thresholdParams;
    bool normalized;
    std::string targetDevice;
    std::tie(inShapeParams, inPrecisions, sortResultType, outType, topKParams, thresholdParams,
        backgroudClass, normalized, decayFunction, targetDevice) = obj.param;

    size_t numBatches, numBoxes, numClasses;
    std::tie(numBatches, numBoxes, numClasses) = inShapeParams;

    Precision paramsPrec, maxBoxPrec, thrPrec;
    std::tie(paramsPrec, maxBoxPrec, thrPrec) = inPrecisions;

    int nmsTopK, keepTopK;
    std::tie(nmsTopK, keepTopK) = topKParams;

    float score_threshold, gaussian_sigma, post_threshold;
    std::tie(score_threshold, gaussian_sigma, post_threshold) = thresholdParams;

    std::ostringstream result;
    result << "numBatches=" << numBatches << "_numBoxes=" << numBoxes << "_numClasses=" << numClasses << "_";
    result << "paramsPrec=" << paramsPrec << "_maxBoxPrec=" << maxBoxPrec << "_thrPrec=" << thrPrec << "_";
    result << "sortResultType=" << sortResultType << "_normalized=" << normalized << "_";
    result << "outType=" << outType << "_nmsTopK=" << nmsTopK << "_keepTopK=" << keepTopK << "_";
    result << "backgroudClass=" << backgroudClass << "_decayFunction=" << decayFunction << "_";
    result << "score_threshold=" << score_threshold << "_gaussian_sigma=" << gaussian_sigma << "_";
    result << "post_threshold=" << post_threshold << "_TargetDevice=" << targetDevice;
    return result.str();
}

void MatrixNmsLayerTest::GenerateInputs() {
    size_t it = 0;
    for (const auto &input : cnnNetwork.getInputsInfo()) {
        const auto &info = input.second;
        Blob::Ptr blob;

        if (it == 1) {
            blob = make_blob_with_precision(info->getTensorDesc());
            blob->allocate();
            CommonTestUtils::fill_data_random_float<Precision::FP32>(blob, 1, 0, 100000);
        } else {
            blob = GenerateInput(*info);
        }
        inputs.push_back(blob);
        it++;
    }
}

void MatrixNmsLayerTest::Compare(const std::vector<std::pair<ngraph::element::Type, std::vector<std::uint8_t>>> &expectedOutputs,
                                     const std::vector<Blob::Ptr> &actualOutputs) {
    for (int outputIndex = static_cast<int>(expectedOutputs.size()) - 1; outputIndex >= 0 ; outputIndex--) {
        const auto& expected = expectedOutputs[outputIndex];
        const auto& actual = actualOutputs[outputIndex];

        const auto &expectedBuffer = expected.second.data();
        auto memory = InferenceEngine::as<InferenceEngine::MemoryBlob>(actual);
        IE_ASSERT(memory);
        const auto lockedMemory = memory->wmap();
        const auto actualBuffer = lockedMemory.as<const uint8_t *>();

        auto k =  static_cast<float>(expected.first.size()) / actual->getTensorDesc().getPrecision().size();
        // W/A for int4, uint4
        if (expected.first == ngraph::element::Type_t::u4 || expected.first == ngraph::element::Type_t::i4) {
            k /= 2;
        }
        if (outputIndex == 2) {
            if (expected.second.size() != k * actual->byteSize())
                throw std::runtime_error("Expected and actual size 3rd output have different size");
        }

        const auto &precision = actual->getTensorDesc().getPrecision();
        size_t size = expected.second.size() / (k * actual->getTensorDesc().getPrecision().size());
        switch (precision) {
            case InferenceEngine::Precision::FP32: {
                switch (expected.first) {
                    case ngraph::element::Type_t::f32:
                        LayerTestsUtils::LayerTestsCommon::Compare(
                                reinterpret_cast<const float *>(expectedBuffer),
                                reinterpret_cast<const float *>(actualBuffer), size, 0);
                        break;
                    case ngraph::element::Type_t::f64:
                        LayerTestsUtils::LayerTestsCommon::Compare(
                                reinterpret_cast<const double *>(expectedBuffer),
                                reinterpret_cast<const float *>(actualBuffer), size, 0);
                        break;
                    default:
                        break;
                }

                break;
            }
            case InferenceEngine::Precision::I32: {
                switch (expected.first) {
                    case ngraph::element::Type_t::i32:
                        LayerTestsUtils::LayerTestsCommon::Compare(
                                reinterpret_cast<const int32_t *>(expectedBuffer),
                                reinterpret_cast<const int32_t *>(actualBuffer), size, 0);
                        break;
                    case ngraph::element::Type_t::i64:
                        LayerTestsUtils::LayerTestsCommon::Compare(
                                reinterpret_cast<const int64_t *>(expectedBuffer),
                                reinterpret_cast<const int32_t *>(actualBuffer), size, 0);
                        break;
                    default:
                        break;
                }
                break;
            }
            default:
                FAIL() << "Comparator for " << precision << " precision isn't supported";
        }
    }
}

void MatrixNmsLayerTest::SetUp() {
    InputShapeParams inShapeParams;
    InputPrecisions inPrecisions;
    op::v8::MatrixNms::Attributes attrs;
    TopKParams topKParams;
    ThresholdParams thresholdParams;

    std::tie(inShapeParams, inPrecisions, attrs.sort_result_type, attrs.output_type, topKParams, thresholdParams,
        attrs.background_class, attrs.normalized, attrs.decay_function, targetDevice) = this->GetParam();

    std::tie(attrs.nms_top_k, attrs.keep_top_k) = topKParams;
    std::tie(attrs.score_threshold, attrs.gaussian_sigma, attrs.post_threshold) = thresholdParams;
    std::tie(numBatches, numBoxes, numClasses) = inShapeParams;
    auto realClasses = numClasses;
    if (attrs.background_class >=0 && attrs.background_class <= numClasses) {
        realClasses = realClasses - 1;
    }

    maxOutputBoxesPerClass = 0;
    if (attrs.nms_top_k >= 0)
        maxOutputBoxesPerClass = std::min(numBoxes, static_cast<size_t>(attrs.nms_top_k));
    else
        maxOutputBoxesPerClass = numBoxes;

    maxOutputBoxesPerBatch  = maxOutputBoxesPerClass * realClasses;
    if (attrs.keep_top_k >= 0)
        maxOutputBoxesPerBatch =
                std::min(maxOutputBoxesPerBatch, static_cast<size_t>(attrs.keep_top_k));
    Precision paramsPrec, maxBoxPrec, thrPrec;
    std::tie(paramsPrec, maxBoxPrec, thrPrec) = inPrecisions;

    const std::vector<size_t> boxesShape{numBatches, numBoxes, 4}, scoresShape{numBatches, numClasses, numBoxes};
    auto ngPrc = convertIE2nGraphPrc(paramsPrec);
    auto params = builder::makeParams(ngPrc, {boxesShape, scoresShape});
    auto paramOuts = helpers::convert2OutputVector(helpers::castOps2Nodes<op::Parameter>(params));
    auto nms = std::make_shared<opset8::MatrixNms>(paramOuts[0], paramOuts[1], attrs);
    // auto nms_0_identity = std::make_shared<opset5::Multiply>(nms->output(0), opset5::Constant::create(element::f32, Shape{1}, {1}));
    // auto nms_1_identity = std::make_shared<opset5::Multiply>(nms->output(1), opset5::Constant::create(attrs.output_type, Shape{1}, {1}));
    // auto nms_2_identity = std::make_shared<opset5::Multiply>(nms->output(2), opset5::Constant::create(attrs.output_type, Shape{1}, {1}));
    function = std::make_shared<Function>(nms, params, "NMS");
}

}  // namespace LayerTestsDefinitions
