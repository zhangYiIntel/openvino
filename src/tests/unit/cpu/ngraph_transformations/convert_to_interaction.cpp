// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph_transformations/op/interaction.hpp>
#include <ngraph_transformations/convert_to_interaction.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>
#include <ngraph/pass/manager.hpp>
#include <ie_core.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;
using namespace ov::intel_cpu;

TEST(TransformationTests, ConvertToInteractionTest1) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        ov::Core core;
        core.read_model("");
        // f = std::make_shared<ngraph::Function>(ngraph::NodeVector{ matmul }, ngraph::ParameterVector{ input1 });
        // ngraph::pass::Manager m;
        // m.register_pass<ngraph::pass::InitNodeInfo>();
        // m.register_pass<ConvertToInteraction>();
        // m.run_passes(f);
        // ASSERT_NO_THROW(check_rt_info(f));
    }
}
