// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/convert_fq_to_scale.hpp"
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include "ngraph_ops/type_relaxed.hpp"
#include <transformations/utils/utils.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include "ngraph_ops/type_relaxed.hpp"
#include "openvino/op/util/attr_types.hpp"

#include "low_precision/common/ie_lpt_exception.hpp"
#include "low_precision/network_helper.hpp"
#include "itt.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

ConvertFQToScale::ConvertFQToScale(const Params& params) : LayerTransformation(params) {
    MATCHER_SCOPE(ConvertFQToScale);

    auto data = ngraph::pattern::any_input(ngraph::pattern::has_static_rank());
    auto input_low = ngraph::pattern::wrap_type<ngraph::opset1::Constant>();
    auto input_high = ngraph::pattern::wrap_type<ngraph::opset1::Constant>();
    auto output_low = ngraph::pattern::wrap_type<ngraph::opset1::Constant>();
    auto output_high = ngraph::pattern::wrap_type<ngraph::opset1::Constant>();
    auto fq = ngraph::pattern::wrap_type<ngraph::opset1::FakeQuantize>({data,
    input_low, input_high, output_low, output_high}, ngraph::pattern::has_static_rank());

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        static int count = 0;
        auto fake_quantize = std::dynamic_pointer_cast<ngraph::opset1::FakeQuantize>(pattern_map.at(fq).get_node_shared_ptr());
        if (!fake_quantize || transformation_callback(fake_quantize)) {
            return false;
        }

        if (fake_quantize) {
            auto data_node = pattern_map.at(data).get_node_shared_ptr();
            bool concat_before = ov::is_type<ov::op::v0::Concat>(data_node);
            bool embeding_before = ov::is_type<ngraph::opset8::EmbeddingBagOffsetsSum>(data_node);
            printf("FQ name %s concat_before %d embeding_before %d\n", fake_quantize->get_friendly_name().c_str(), concat_before, embeding_before);
            if (!concat_before)
                return false;
        }
        auto data_input = pattern_map.at(data);
        auto scales = std::make_shared<ngraph::op::Constant>(element::f32, Shape{}, std::vector<float>{1});
        auto multiply = std::make_shared<op::TypeRelaxed<ov::op::v1::Multiply>>(
            std::vector<element::Type>{element::f32, element::f32},
            std::vector<element::Type>{element::i8},
            ngraph::op::TemporaryReplaceOutputType(fake_quantize->input_value(0), deqPrecision).get(),
            ngraph::op::TemporaryReplaceOutputType(scales->output(0), deqPrecision).get());
        multiply->set_friendly_name(fake_quantize->get_friendly_name());
        multiply->set_output_type(0, element::i8, multiply->get_output_partial_shape(0));
        ngraph::copy_runtime_info(fake_quantize, multiply);
        std::cout << "ConvertFQToScale  count " << count++ << std::endl;
        ngraph::replace_node(fake_quantize, multiply);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(fq, matcher_name);
    this->register_matcher(m, callback);
}

bool ConvertFQToScale::transform(TransformationContext& context, ngraph::pattern::Matcher &m) {
    return true;
}

bool ConvertFQToScale::isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept {
    return false;
}

} // namespace low_precision
} // namespace pass
} // namespace ngraph

