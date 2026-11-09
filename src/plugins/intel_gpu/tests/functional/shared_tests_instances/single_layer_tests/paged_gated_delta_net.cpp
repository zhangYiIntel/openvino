// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/paged_gated_delta_net.hpp"

namespace ov::test {

std::vector<PagedGatedDeltaNetLayerParams> paged_gdn_test_cases = {
    {2, 4, 8, 8, {3, 3}, {2, 3}, ov::element::f32, "GPU"},
    {2, 4, 8, 8, {3, 2}, {0, 0}, ov::element::f32, "GPU"},
    {1, 4, 16, 8, {2, 5, 1}, {1, 4, 2}, ov::element::f32, "GPU"},
    {2, 6, 32, 64, {4, 2, 3}, {3, 2, 5}, ov::element::f32, "GPU"},
    {4, 8, 32, 64, {15, 32, 33}, {16, 16, 16}, ov::element::f32, "GPU"},
    {8, 8, 128, 128, {15, 32, 33}, {16, 16, 16}, ov::element::f32, "GPU"},
    {2, 4, 8, 8, {3, 3}, {2, 3}, ov::element::f16, "GPU"},
    {2, 4, 8, 8, {3, 2}, {0, 0}, ov::element::f16, "GPU"},
    {1, 4, 16, 8, {2, 5, 1}, {1, 4, 2}, ov::element::f16, "GPU"},
    {2, 6, 32, 64, {4, 2, 3}, {3, 2, 5}, ov::element::f16, "GPU"},
    {4, 8, 32, 64, {15, 32, 33}, {16, 16, 16}, ov::element::f16, "GPU"},
    {8, 8, 128, 128, {15, 32, 33}, {16, 16, 16}, ov::element::f16, "GPU"},
    // Long prefill calls (K = 128, V multiple of 64, at most 2 sequences of >= 256 tokens) take the chunked DPAS path where
    // supported (f16). Intervals of 64 / 100 start mid-chunk (past length offsets the first snapshot), 0 keeps only the
    // final state. Three sequences and f32 use the per-token kernels.
    {2, 4, 128, 128, {600, 300}, {64, 100}, ov::element::f16, "GPU"},
    {2, 4, 128, 128, {600, 300, 400}, {64, 100, 0}, ov::element::f16, "GPU"},
    {2, 4, 128, 128, {600, 300}, {64, 100}, ov::element::f32, "GPU"},
    // A snapshot every 16 chunks (interval 1024) next to a sequence without snapshots.
    {2, 4, 128, 128, {1100, 300}, {1024, 0}, ov::element::f16, "GPU"},
    // Several snapshots inside one chunk, partial tail chunks.
    {2, 4, 128, 128, {257, 513}, {32, 16}, ov::element::f16, "GPU"},
    // GQA (16 key heads, 32 value heads), a snapshot interval of 128 and one sequence without snapshots.
    {16, 32, 128, 128, {300, 280}, {0, 128}, ov::element::f16, "GPU"},
    // Value size 256 (several v-blocks), a single sequence.
    {1, 2, 128, 256, {1000}, {128}, ov::element::f16, "GPU"},
};

INSTANTIATE_TEST_SUITE_P(smoke_PagedGatedDeltaNet,
                         PagedGatedDeltaNetLayerTest,
                         ::testing::ValuesIn(paged_gdn_test_cases),
                         PagedGatedDeltaNetLayerTest::getTestCaseName);

}  // namespace ov::test
