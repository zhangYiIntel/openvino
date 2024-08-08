// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"
#include "include/batch_headers/sub_group_block_read.cl"
#include "include/batch_headers/sub_group_block_write.cl"

#if IS_DYNAMIC
#define CALC_POWER(n) ({uint pos = 0; uint i = n; do { i >>= 1; ++pos; } while (i); --pos;})
#endif

// Check alignment restrictions for using block writes on output.
#define USE_BLOCK_WRITE ((OUTPUT_TYPE_SIZE * OUTPUT_FEATURE_PITCH) & 0xF == 0)

#if SUBGROUP_BLOCK_SIZE == 1
#define BLOCK_READ(ptr, offset) DT_INPUT_BLOCK_READ(ptr, offset)
#define BLOCK_WRITE(ptr, offset, val) DT_OUTPUT_BLOCK_WRITE(ptr, offset, val)
#define INPUT_VEC_TYPE INPUT0_TYPE
#define OUTPUT_VEC_TYPE OUTPUT_TYPE
#else
#define BLOCK_READ(ptr, offset) CAT(DT_INPUT_BLOCK_READ, SUBGROUP_BLOCK_SIZE)(ptr, offset)
#define BLOCK_WRITE(ptr, offset, val) CAT(DT_OUTPUT_BLOCK_WRITE, SUBGROUP_BLOCK_SIZE)(ptr, offset, val)
#define INPUT_VEC_TYPE MAKE_VECTOR_TYPE(INPUT0_TYPE, SUBGROUP_BLOCK_SIZE)
#define OUTPUT_VEC_TYPE MAKE_VECTOR_TYPE(OUTPUT_TYPE, SUBGROUP_BLOCK_SIZE)
#endif

REQD_SUB_GROUP_SIZE(SUB_GROUP_SIZE)
KERNEL(rms_gpu_bfyx_opt)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* input,
    const __global INPUT1_TYPE* gamma,
    __global OUTPUT_TYPE* output)
{
    const uint data_idx = get_global_id(1);
    const uint in_data_idx = get_global_id(0);
    const uint workers_per_data = LWS;
    const uint data_size = DATA_SIZE;
#if !IS_DYNAMIC
    const uint items_num = ITEMS_NUM; 
    const uint leftovers = LEFTOVERS;
#else
    const uint power = CALC_POWER(workers_per_data);
    const uint items_num = data_size >> power;
    const uint leftovers = data_size - (items_num << power);
#endif
    const uint data_offset = data_idx * data_size;
    const uint subgroup_offset = get_sub_group_id() * get_sub_group_size() * items_num;

    INPUT0_TYPE data[STACK_SIZE];
    ACCUMULATOR_TYPE rms = ACCUMULATOR_VAL_ZERO;

    __local ACCUMULATOR_TYPE slm_buf[SLM_SIZE];

    uint i = 0;
    if (workers_per_data > SUB_GROUP_SIZE)
    {
        for (; i < items_num - (items_num % SUBGROUP_BLOCK_SIZE); i += SUBGROUP_BLOCK_SIZE)
        {
            INPUT_VEC_TYPE vec_tmp = BLOCK_READ(input, data_offset + subgroup_offset + i * get_sub_group_size());
#if SUBGROUP_BLOCK_SIZE == 1
            rms += TO_ACCUMULATOR_TYPE(native_powr(vec_tmp, 2));
            data[i] = vec_tmp;
#else
            unroll_for (int j = 0; j < SUBGROUP_BLOCK_SIZE; j++)
            {
                INPUT0_TYPE tmp = vec_tmp[j];
                rms += TO_ACCUMULATOR_TYPE(native_powr(tmp, 2));
                data[i + j] = tmp;
            }
#endif
        }
    }

    for (; i < items_num; i++)
    {
        INPUT0_TYPE tmp = input[data_offset + subgroup_offset + get_sub_group_local_id() + i * get_sub_group_size()];
        rms += TO_ACCUMULATOR_TYPE(native_powr(tmp, 2));
        data[i] = tmp;
    }

    if (in_data_idx < leftovers)
    {
        INPUT0_TYPE tmp = input[data_offset + workers_per_data * items_num + in_data_idx];
        rms += TO_ACCUMULATOR_TYPE(native_powr(tmp, 2));
        data[items_num] = tmp;
    }

    rms = sub_group_reduce_add(rms);

    if (get_sub_group_local_id() == 0)
        slm_buf[get_sub_group_id()] = rms;

    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint offset = get_num_sub_groups() / 2; offset > 0; offset /= 2) {
        if (in_data_idx < offset) {
            slm_buf[in_data_idx] += slm_buf[in_data_idx + offset];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (in_data_idx == 0) {
        rms = slm_buf[0] / data_size;
        slm_buf[0] = native_powr(sqrt(rms + TO_ACCUMULATOR_TYPE(EPSILON)), -1);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    rms = slm_buf[0];

    i = 0;
    if ((workers_per_data > SUB_GROUP_SIZE) && USE_BLOCK_WRITE)
    {
        for (; i < items_num - (items_num % SUBGROUP_BLOCK_SIZE); i += SUBGROUP_BLOCK_SIZE)
        {
            INPUT_VEC_TYPE vec_gamma = BLOCK_READ(gamma, subgroup_offset + i * get_sub_group_size());
            OUTPUT_VEC_TYPE vec_tmp;
#if SUBGROUP_BLOCK_SIZE == 1
            vec_tmp = TO_OUTPUT_TYPE(rms * TO_ACCUMULATOR_TYPE(data[i]) * TO_ACCUMULATOR_TYPE(vec_gamma));
#else
            unroll_for (int j = 0; j < SUBGROUP_BLOCK_SIZE; j++)
                vec_tmp[j] = TO_OUTPUT_TYPE(rms * TO_ACCUMULATOR_TYPE(data[i + j]) * TO_ACCUMULATOR_TYPE(vec_gamma[j]));
#endif
            BLOCK_WRITE(output, data_offset + subgroup_offset + i * get_sub_group_size(), vec_tmp);
        }
    }

    for (; i < items_num; i++)
    {
        INPUT1_TYPE temp = gamma[subgroup_offset + get_sub_group_local_id() + i * get_sub_group_size()];
        output[data_offset + subgroup_offset + get_sub_group_local_id() + i * get_sub_group_size()] = TO_OUTPUT_TYPE(rms * TO_ACCUMULATOR_TYPE(data[i]) * TO_ACCUMULATOR_TYPE(temp));
    }

    if (in_data_idx < leftovers)
    {
        INPUT1_TYPE temp = gamma[workers_per_data * items_num + in_data_idx];
        output[data_offset + workers_per_data * items_num + in_data_idx] = TO_OUTPUT_TYPE(rms * TO_ACCUMULATOR_TYPE(data[items_num]) * TO_ACCUMULATOR_TYPE(temp));
    }
}
#ifdef CALC_POWER
#undef CALC_POWER
#endif
#undef USE_BLOCK_WRITE
#undef BLOCK_READ
#undef BLOCK_WRITE
#undef INPUT_VEC_TYPE
#undef OUTPUT_VEC_TYPE
