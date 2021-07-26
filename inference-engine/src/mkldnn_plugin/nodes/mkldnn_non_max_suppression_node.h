// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <mkldnn_node.h>
#include <string>
#include <memory>
#include <vector>

using namespace InferenceEngine;

namespace MKLDNNPlugin {

class MKLDNNNonMaxSuppressionNode : public MKLDNNNode {
public:
    MKLDNNNonMaxSuppressionNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);

    void getSupportedDescriptors() override {};
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override {};
    void execute(mkldnn::stream strm) override;
    bool created() const override;

    static bool isSupportedOperation(const std::shared_ptr<ngraph::Node>& op, std::string& errorMessage) noexcept;

    struct filteredBoxes {
        float score;
        int batch_index;
        int class_index;
        int box_index;
        filteredBoxes() = default;
        filteredBoxes(float _score, int _batch_index, int _class_index, int _box_index) :
                score(_score), batch_index(_batch_index), class_index(_class_index), box_index(_box_index) {}
    };

    struct boxInfo {
        float score;
        int idx;
        int suppress_begin_index;
    };

    float intersectionOverUnion(const float *boxesI, const float *boxesJ);

    void nmsWithSoftSigma(const float *boxes, const float *scores, const SizeVector &boxesStrides,
                          const SizeVector &scoresStrides, std::vector<filteredBoxes> &filtBoxes);

    void nmsWithoutSoftSigma(const float *boxes, const float *scores, const SizeVector &boxesStrides,
                             const SizeVector &scoresStrides, std::vector<filteredBoxes> &filtBoxes);

private:
    // input
    enum : size_t {
        NMS_BOXES,
        NMS_SCORES,
        NMS_MAXOUTPUTBOXESPERCLASS,
        NMS_IOUTHRESHOLD,
        NMS_SCORETHRESHOLD,
        NMS_SOFTNMSSIGMA,
    } InputNumber;

    // output
    enum : size_t {
        NMS_SELECTEDINDICES,
        NMS_SELECTEDSCORES,
        NMS_VALIDOUTPUTS
    } OutputNumber;


    enum class boxEncoding {
        CORNER,
        CENTER
    };
    boxEncoding boxEncodingType = boxEncoding::CORNER;
    bool sort_result_descending = true;

    size_t num_batches;
    size_t num_boxes;
    size_t num_classes;

    size_t max_output_boxes_per_class = 0lu;
    float iou_threshold = 0.0f;
    float score_threshold = 0.0f;
    float soft_nms_sigma = 0.0f;
    float scale = 1.f;

    Shape inputShape_MAXOUTPUTBOXESPERCLASS;
    Shape inputShape_IOUTHRESHOLD;
    Shape inputShape_SCORETHRESHOLD;
    Shape inputShape_SOFTNMSSIGMA;

    Shape outputShape_SELECTEDINDICES;
    Shape outputShape_SELECTEDSCORES;

    std::string errorPrefix;

    std::vector<std::vector<size_t>> numFiltBox;
    const std::string inType = "input", outType = "output";

    void checkPrecision(const Precision& prec, const std::vector<Precision>& precList, const std::string& name, const std::string& type);
    void check1DInput(const Shape& shape, const std::vector<Precision>& precList, const std::string& name, const size_t port);
    void checkOutput(const Shape& shape, const std::vector<Precision>& precList, const std::string& name, const size_t port);
};

}  // namespace MKLDNNPlugin
