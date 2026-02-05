#include "include/batch_headers/fetch_data.cl"
#include "include/batch_headers/common.cl"
#include "include/batch_headers/sub_group_block_read.cl"
#include "include/batch_headers/sub_group_block_write.cl"
#include "include/batch_headers/sub_group_shuffle.cl"
#define V_BLOCK_SIZE 4
float sg_read_f(__global const float* p) {
    return as_float(intel_sub_group_block_read((__global const uint*)p));
}
float2 sg_read2_f(__global const float* p) {
    return as_float2(intel_sub_group_block_read2((__global const uint*)p));
}
float8 sg_read8_f(__global const float* p) {
    return as_float8(intel_sub_group_block_read8((__global const uint*)p));
}
void sg_write2_f(__global float* p, float2 v) {
    intel_sub_group_block_write2((__global uint*)p, as_uint2(v));
}
void sg_write8_f(__global float* p, float8 v) {
    intel_sub_group_block_write8((__global uint*)p, as_uint8(v));
}
void sg_write2_h(__global half* p, half2 v) {
    intel_sub_group_block_write((__global uint*)p, as_uint(v));
}
void sg_write8_h(__global half* p, half8 v) {
    intel_sub_group_block_write4((__global uint*)p, as_uint4(v));
}
float sum2(float2 v) {
    return v.s0 + v.s1;
}
float sum8(float8 v) {
    return v.s0 + v.s1 + v.s2 + v.s3 + v.s4 + v.s5 + v.s6 + v.s7;
}
void sg_write_f(__global float* p, float v) {
    intel_sub_group_block_write((__global uint*)p, as_uint(v));
}
inline float l2norm_scale(float sum, float extra_scale) {
    sum = sub_group_reduce_add(sum);
    sum = sub_group_broadcast(sum, 0);
    return rsqrt(sum + 0.000001f) * extra_scale;
}
#if IO_TYPE == 1
REQD_SUB_GROUP_SIZE(SUBGROUP_SIZE)
KERNEL(linear_attention_ref)(
    __global INPUT0_TYPE* q, 
    __global INPUT1_TYPE* k, 
    __global INPUT2_TYPE* v, 
    __global INPUT3_TYPE* g,
    __global INPUT4_TYPE* beta,
    __global INPUT5_TYPE* initial_state,
    __global OUTPUT_TYPE* output,
    int seq_len) {
    int b = get_global_id(0);
    int gid1 = get_global_id(1);
    int id_local = get_local_id(2);
    int BATCH_STRIDE = K_HEAD_NUMS * seq_len * K_HEAD_DIMS;
    int STEP_STRIDE = K_HEAD_NUMS * K_HEAD_DIMS;
    int v_blocks = (K_HEAD_DIMS + V_BLOCK_SIZE - 1) / V_BLOCK_SIZE;
    int h = gid1 / v_blocks;
    int v_block_id = gid1 - h * v_blocks;
    int i_v_base = v_block_id * V_BLOCK_SIZE;
    __global const INPUT0_TYPE* q_ptr = q + b * BATCH_STRIDE;
    __global const INPUT1_TYPE* k_ptr = k + b * BATCH_STRIDE;
    __global const INPUT2_TYPE* v_ptr = v + b * BATCH_STRIDE;
    __global const INPUT3_TYPE* g_ptr = g + b * K_HEAD_NUMS * seq_len;
    __global const INPUT4_TYPE* beta_ptr = beta + b * K_HEAD_NUMS * seq_len;
    int out_base = b * K_HEAD_NUMS * seq_len * K_HEAD_DIMS + h * K_HEAD_DIMS;
#if (K_HEAD_DIMS == 128)
#if (SUBGROUP_SIZE == 8)
    float8 init_state[V_BLOCK_SIZE][2];
    float8 b_k[2];
    float8 b_q[2];
#else
    float8 init_state[V_BLOCK_SIZE];
    float8 b_k;
    float8 b_q;
#endif
#elif (K_HEAD_DIMS % 32) == 0
    float2 init_state[V_BLOCK_SIZE][K_HEAD_DIMS / 32];
    float2 b_k[K_HEAD_DIMS / 32];
    float2 b_q[K_HEAD_DIMS / 32];
#else
    float init_state[V_BLOCK_SIZE][K_HEAD_DIMS / SUBGROUP_SIZE] = {0};
    float b_k[K_HEAD_DIMS / SUBGROUP_SIZE] = {0};
    float b_q[K_HEAD_DIMS / SUBGROUP_SIZE] = {0};
#endif
    int id_sg_local = get_sub_group_local_id();
    for (int iv = 0; iv < V_BLOCK_SIZE; iv++) {
        int i_v = i_v_base + iv;
        int init_base = b * K_HEAD_NUMS * K_HEAD_DIMS * K_HEAD_DIMS + h * K_HEAD_DIMS * K_HEAD_DIMS + i_v * K_HEAD_DIMS;
#if (K_HEAD_DIMS == 128)
#if (SUBGROUP_SIZE == 8)
    init_state[iv][0] = sg_read8_f(initial_state + init_base);
    init_state[iv][1] = sg_read8_f(initial_state + init_base + (SUBGROUP_SIZE * 8));
#else
    init_state[iv] = sg_read8_f(initial_state + init_base);
#endif
#elif (K_HEAD_DIMS % 32) == 0
#if (SUBGROUP_SIZE == 16)
    for (int j = id_sg_local; j < K_HEAD_DIMS; j += 32) {
        int idx = j >> 5;
        init_state[iv][idx] = sg_read2_f(initial_state + init_base + (j - id_sg_local));
    }
#else
    for (int j = id_sg_local; j < K_HEAD_DIMS; j += SUBGROUP_SIZE) {
        int idx = j / SUBGROUP_SIZE;
        float val = sg_read_f(initial_state + init_base + (j - id_sg_local));
        init_state[iv][idx] = val;
    }
#endif
#else
    for (int j = id_sg_local; j < K_HEAD_DIMS; j += SUBGROUP_SIZE) {
        int idx = j / SUBGROUP_SIZE;
        float val = sg_read_f(initial_state + init_base + (j - id_sg_local));
        init_state[iv][idx] = val;
    }
#endif
    }
    int kv_base = h * K_HEAD_DIMS;
    int out_i_base = out_base;
    for (int i = 0; i < seq_len; i++, kv_base += STEP_STRIDE, out_i_base += STEP_STRIDE) {
        float b_g = g_ptr[i * K_HEAD_NUMS + h];
        float b_beta = beta_ptr[i * K_HEAD_NUMS + h];
        b_g = exp(b_g);
    #if (K_HEAD_DIMS == 128)
        #if (SUBGROUP_SIZE == 8)
            b_k[0] = sg_read8_f(k_ptr + kv_base);
            b_k[1] = sg_read8_f(k_ptr + kv_base + (SUBGROUP_SIZE * 8));
            b_q[0] = sg_read8_f(q_ptr + kv_base);
            b_q[1] = sg_read8_f(q_ptr + kv_base + (SUBGROUP_SIZE * 8));
        #else
            b_k = sg_read8_f(k_ptr + kv_base);
            b_q = sg_read8_f(q_ptr + kv_base);
        #endif
    #elif (K_HEAD_DIMS % 32) == 0
        #if (SUBGROUP_SIZE == 16)
            #pragma unroll
            for (int j = id_sg_local; j < K_HEAD_DIMS; j += 32) {
                int idx = j >> 5;
                b_k[idx] = sg_read2_f(k_ptr + kv_base + (j - id_sg_local));
                b_q[idx] = sg_read2_f(q_ptr + kv_base + (j - id_sg_local));
            }
        #else
            for (int j = id_sg_local; j < K_HEAD_DIMS; j += SUBGROUP_SIZE) {
                int idx = j / SUBGROUP_SIZE;
                b_k[idx] = sg_read_f(k_ptr + kv_base + (j - id_sg_local));
                b_q[idx] = sg_read_f(q_ptr + kv_base + (j - id_sg_local));
            }
        #endif
    #else
        for (int j = id_sg_local; j < K_HEAD_DIMS; j += SUBGROUP_SIZE) {
            int idx = j / SUBGROUP_SIZE;
            b_k[idx] = sg_read_f(k_ptr + kv_base + (j - id_sg_local));
            b_q[idx] = sg_read_f(q_ptr + kv_base + (j - id_sg_local));
        }
    #endif

    #if (K_HEAD_DIMS == 128)
    #if (SUBGROUP_SIZE == 8)
            float k_sum = sum8(b_k[0] * b_k[0]) + sum8(b_k[1] * b_k[1]);
            float k_scale = l2norm_scale(k_sum, 1.0f);
            b_k[0] *= k_scale;
            b_k[1] *= k_scale;

            float q_sum = sum8(b_q[0] * b_q[0]) + sum8(b_q[1] * b_q[1]);
            float q_scale = l2norm_scale(q_sum, SCALE_FACTOR);
            b_q[0] = b_q[0] * q_scale;
            b_q[1] = b_q[1] * q_scale;
    #else
            float k_sum = sum8(b_k * b_k);
            float k_scale = l2norm_scale(k_sum, 1.0f);
            b_k *= k_scale;

            float q_sum = sum8(b_q * b_q);
            float q_scale = l2norm_scale(q_sum, SCALE_FACTOR);
            b_q = b_q * q_scale;
    #endif
    #elif (K_HEAD_DIMS % 32) == 0
    #if (SUBGROUP_SIZE == 16)
            float k_sum = 0.0f;
            for (int j = id_sg_local; j < K_HEAD_DIMS; j += 32) {
                int idx = j >> 5;
                k_sum += sum2(b_k[idx] * b_k[idx]);
            }
            float k_scale = l2norm_scale(k_sum, 1.0f);
            for (int j = id_sg_local; j < K_HEAD_DIMS; j += 32) {
                int idx = j >> 5;
                b_k[idx] *= (float2)(k_scale);
            }

            float q_sum = 0.0f;
            for (int j = id_sg_local; j < K_HEAD_DIMS; j += 32) {
                int idx = j >> 5;
                q_sum += sum2(b_q[idx] * b_q[idx]);
            }
            float q_scale = l2norm_scale(q_sum, SCALE_FACTOR);
            for (int j = id_sg_local; j < K_HEAD_DIMS; j += 32) {
                int idx = j >> 5;
                b_q[idx] *= (float2)(q_scale);
            }
    #else
            float k_sum = 0.0f;
            for (int j = id_sg_local; j < K_HEAD_DIMS; j += SUBGROUP_SIZE) {
                int idx = j / SUBGROUP_SIZE;
                k_sum = fma(b_k[idx], b_k[idx], k_sum);
            }
            float k_scale = l2norm_scale(k_sum, 1.0f);
            for (int j = id_sg_local; j < K_HEAD_DIMS; j += SUBGROUP_SIZE) {
                int idx = j / SUBGROUP_SIZE;
                b_k[idx] *= k_scale;
            }

            float q_sum = 0.0f;
            for (int j = id_sg_local; j < K_HEAD_DIMS; j += SUBGROUP_SIZE) {
                int idx = j / SUBGROUP_SIZE;
                q_sum = fma(b_q[idx], b_q[idx], q_sum);
            }
            float q_scale = l2norm_scale(q_sum, SCALE_FACTOR);
            for (int j = id_sg_local; j < K_HEAD_DIMS; j += SUBGROUP_SIZE) {
                int idx = j / SUBGROUP_SIZE;
                b_q[idx] *= (q_scale);
            }
    #endif
    #else
            float k_sum = 0.0f;
            for (int j = id_sg_local; j < K_HEAD_DIMS; j += SUBGROUP_SIZE) {
                int idx = j / SUBGROUP_SIZE;
                k_sum = fma(b_k[idx], b_k[idx], k_sum);
            }
            float k_scale = l2norm_scale(k_sum, 1.0f);
            for (int j = id_sg_local; j < K_HEAD_DIMS; j += SUBGROUP_SIZE) {
                int idx = j / SUBGROUP_SIZE;
                b_k[idx] *= k_scale;
            }

            float q_sum = 0.0f;
            for (int j = id_sg_local; j < K_HEAD_DIMS; j += SUBGROUP_SIZE) {
                int idx = j / SUBGROUP_SIZE;
                q_sum = fma(b_q[idx], b_q[idx], q_sum);
            }
            float q_scale = l2norm_scale(q_sum, SCALE_FACTOR);
            for (int j = id_sg_local; j < K_HEAD_DIMS; j += SUBGROUP_SIZE) {
                int idx = j / SUBGROUP_SIZE;
                b_q[idx] *= (q_scale);
            }
    #endif
        for (int iv = 0; iv < V_BLOCK_SIZE; iv++) {
            int i_v = i_v_base + iv;
#if (K_HEAD_DIMS == 128)
        #if (SUBGROUP_SIZE == 8)
            init_state[iv][0] *= b_g;
            init_state[iv][1] *= b_g;
            float hk_acc = sum8(init_state[iv][0] * b_k[0]) + sum8(init_state[iv][1] * b_k[1]);
            hk_acc = sub_group_reduce_add(hk_acc);
            hk_acc = sub_group_broadcast(hk_acc, 0);

            int v_base = kv_base + (i_v & ~(SUBGROUP_SIZE - 1));
            int v_lane = i_v & (SUBGROUP_SIZE - 1);
            float v_val = as_float(intel_sub_group_block_read((__global const uint*)(v_ptr + v_base)));
            float b_v = sub_group_broadcast(v_val, v_lane);
            b_v -= hk_acc;
            b_v *= b_beta;
            init_state[iv][0] = fma(b_k[0], (float8)(b_v), init_state[iv][0]);
            init_state[iv][1] = fma(b_k[1], (float8)(b_v), init_state[iv][1]);

            float out_acc = sum8(init_state[iv][0] * b_q[0]) + sum8(init_state[iv][1] * b_q[1]);
            out_acc = sub_group_reduce_add(out_acc);
            out_acc = sub_group_broadcast(out_acc, 0);
            if (id_sg_local == 0) {
                output[out_i_base + i_v] = out_acc;
            }
        #else
            init_state[iv] *= b_g;
            float hk_acc = sum8(init_state[iv] * b_k);
            hk_acc = sub_group_reduce_add(hk_acc);
            hk_acc = sub_group_broadcast(hk_acc, 0);

            int v_base = kv_base + (i_v & ~(SUBGROUP_SIZE - 1));
            int v_lane = i_v & (SUBGROUP_SIZE - 1);
            float v_val = as_float(intel_sub_group_block_read((__global const uint*)(v_ptr + v_base)));
            float b_v = sub_group_broadcast(v_val, v_lane);
            b_v -= hk_acc;
            b_v *= b_beta;
            init_state[iv] = fma(b_k, (float8)(b_v), init_state[iv]);

            float out_acc = sum8(init_state[iv] * b_q);
            out_acc = sub_group_reduce_add(out_acc);
            out_acc = sub_group_broadcast(out_acc, 0);
            if (id_sg_local == 0) {
                output[out_i_base + i_v] = out_acc;
            }
        #endif
#elif (K_HEAD_DIMS % 32) == 0
        #if (SUBGROUP_SIZE == 16)
            for (int j = id_sg_local; j < K_HEAD_DIMS; j += 32) {
                int idx = j >> 5;
                init_state[iv][idx] *= b_g;
            }
            float hk_acc = 0.0f;
            for (int j = id_sg_local; j < K_HEAD_DIMS; j += 32) {
                int idx = j >> 5;
                hk_acc += sum2(init_state[iv][idx] * b_k[idx]);
            }
            hk_acc = sub_group_reduce_add(hk_acc);
            hk_acc = sub_group_broadcast(hk_acc, 0);

            int v_base = kv_base + (i_v & ~(SUBGROUP_SIZE - 1));
            int v_lane = i_v & (SUBGROUP_SIZE - 1);
            float v_val = as_float(intel_sub_group_block_read((__global const uint*)(v_ptr + v_base)));
            float b_v = sub_group_broadcast(v_val, v_lane);
            b_v -= hk_acc;
            b_v *= b_beta;
            for (int j = id_sg_local; j < K_HEAD_DIMS; j += 32) {
                int idx = j >> 5;
                init_state[iv][idx] = fma(b_k[idx], (float2)(b_v), init_state[iv][idx]);
            }

            float out_acc = 0.0f;
            for (int j = id_sg_local; j < K_HEAD_DIMS; j += 32) {
                int idx = j >> 5;
                out_acc += sum2(init_state[iv][idx] * b_q[idx]);
            }
            out_acc = sub_group_reduce_add(out_acc);
            out_acc = sub_group_broadcast(out_acc, 0);
            if (id_sg_local == 0) {
                output[out_i_base + i_v] = out_acc;
            }
        #else
            for (int n = id_sg_local; n < K_HEAD_DIMS; n += SUBGROUP_SIZE) {
                int idx = n / SUBGROUP_SIZE;
                init_state[iv][idx] *= b_g;
            }
            float hk_acc = 0.0f;
            for (int n = id_sg_local; n < K_HEAD_DIMS; n += SUBGROUP_SIZE) {
                int idx = n / SUBGROUP_SIZE;
                hk_acc = fma(init_state[iv][idx], b_k[idx], hk_acc);
            }
            hk_acc = sub_group_reduce_add(hk_acc);
            hk_acc = sub_group_broadcast(hk_acc, 0);

            int v_base = kv_base + (i_v & ~(SUBGROUP_SIZE - 1));
            int v_lane = i_v & (SUBGROUP_SIZE - 1);
            float v_val = as_float(intel_sub_group_block_read((__global const uint*)(v_ptr + v_base)));
            float b_v = sub_group_broadcast(v_val, v_lane);
            b_v -= hk_acc;
            b_v *= b_beta;
            for (int n = id_sg_local; n < K_HEAD_DIMS; n += SUBGROUP_SIZE) {
                int idx = n / SUBGROUP_SIZE;
                init_state[iv][idx] = fma(b_k[idx], b_v, init_state[iv][idx]);
            }

            float out_acc = 0.0f;
            for (int n = id_sg_local; n < K_HEAD_DIMS; n += SUBGROUP_SIZE) {
                int idx = n / SUBGROUP_SIZE;
                out_acc = fma(init_state[iv][idx], b_q[idx], out_acc);
            }
            out_acc = sub_group_reduce_add(out_acc);
            out_acc = sub_group_broadcast(out_acc, 0);
            if (id_sg_local == 0) {
                output[out_i_base + i_v] = out_acc;
            }
        #endif
#else
        // h0 * g
        for (int n = id_sg_local; n < K_HEAD_DIMS; n += SUBGROUP_SIZE) {
            int idx = n / SUBGROUP_SIZE;
            init_state[iv][idx] *= b_g;
        }
        float hk_acc = 0.0f;
        for (int n = id_sg_local; n < K_HEAD_DIMS; n += SUBGROUP_SIZE) {
            int idx = n / SUBGROUP_SIZE;
            hk_acc = fma(init_state[iv][idx], b_k[idx], hk_acc);
        }
        hk_acc = sub_group_reduce_add(hk_acc);
        hk_acc = sub_group_broadcast(hk_acc, 0);

        int v_base = kv_base + (i_v & ~(SUBGROUP_SIZE - 1));
        int v_lane = i_v & (SUBGROUP_SIZE - 1);
        float v_val = as_float(intel_sub_group_block_read((__global const uint*)(v_ptr + v_base)));
        float b_v = sub_group_broadcast(v_val, v_lane);
        b_v -= hk_acc;
        // b_v * b_k
        b_v *= b_beta;
        // h0 = h0 + b_k * b_v;
        for (int n = id_sg_local; n < K_HEAD_DIMS; n += SUBGROUP_SIZE) {
            int idx = n / SUBGROUP_SIZE;
            init_state[iv][idx] = fma(b_k[idx], b_v, init_state[iv][idx]);
        };
        float out_acc = 0.0f;
        for (int n = id_sg_local; n < K_HEAD_DIMS; n += SUBGROUP_SIZE) {
            int idx = n / SUBGROUP_SIZE;
            out_acc = fma(init_state[iv][idx], b_q[idx], out_acc);
        }
        out_acc = sub_group_reduce_add(out_acc);
        out_acc = sub_group_broadcast(out_acc, 0);
        if (id_sg_local == 0) {
            output[out_i_base + i_v] = out_acc;
        }
#endif
        }
    }
    for (int iv = 0; iv < V_BLOCK_SIZE; iv++) {
        int i_v = i_v_base + iv;
        int init_base = b * K_HEAD_NUMS * K_HEAD_DIMS * K_HEAD_DIMS + h * K_HEAD_DIMS * K_HEAD_DIMS + i_v * K_HEAD_DIMS;
#if (K_HEAD_DIMS == 128)
#if (SUBGROUP_SIZE == 8)
        sg_write8_f(initial_state + init_base, init_state[iv][0]);
        sg_write8_f(initial_state + init_base + (SUBGROUP_SIZE * 8), init_state[iv][1]);
#else
        sg_write8_f(initial_state + init_base, init_state[iv]);
#endif
#elif (K_HEAD_DIMS % 32) == 0
#if (SUBGROUP_SIZE == 16)
        for (int j = id_sg_local; j < K_HEAD_DIMS; j += 32) {
            int idx = j >> 5;
            sg_write2_f(initial_state + init_base + (j - id_sg_local), init_state[iv][idx]);
        }
#else
        for (int j = id_sg_local; j < K_HEAD_DIMS; j += SUBGROUP_SIZE) {
            int idx = j / SUBGROUP_SIZE;
            sg_write_f(initial_state + init_base + (j - id_sg_local), init_state[iv][idx]);
        }
#endif
#else
        for (int j = id_sg_local; j < K_HEAD_DIMS; j += SUBGROUP_SIZE) {
            int idx = j / SUBGROUP_SIZE;
            sg_write_f(initial_state + init_base + (j - id_sg_local), init_state[iv][idx]);
        }
#endif
    }

/*
    printf("b %d h %d get_sub_group_local_id %d\n", b, h, get_sub_group_local_id());
    for (size_t i = 0; i < T; i++) {
        float partial = 0.0f;
        
        for (int j = lid; j < 16; j += lsize) {
            half a = as_half(intel_sub_group_block_read_us((__global ushort*)(q_ptr + j + i *16)));
            partial += a;
        }
        float sub_sum = sub_group_reduce_add(partial);

        if (get_sub_group_local_id() == 0) {
            printf("batch %d head %d q %zu value %f\n", b, h, i, sub_sum);
        }
    }
*/
}
#endif

#if IO_TYPE == 0
REQD_SUB_GROUP_SIZE(SUBGROUP_SIZE)
KERNEL(linear_attention_ref)(
    __global INPUT0_TYPE* q,
    __global INPUT1_TYPE* k,
    __global INPUT2_TYPE* v,
    __global INPUT3_TYPE* g,
    __global INPUT4_TYPE* beta,
    __global INPUT5_TYPE* initial_state,
    __global OUTPUT_TYPE* output,
    int seq_len) {
    int b = get_global_id(0);
    int gid1 = get_global_id(1);
    int id_local = get_local_id(2);
    int BATCH_STRIDE = K_HEAD_NUMS * seq_len * K_HEAD_DIMS;
    int STEP_STRIDE = K_HEAD_NUMS * K_HEAD_DIMS;
    int v_blocks = (K_HEAD_DIMS + V_BLOCK_SIZE - 1) / V_BLOCK_SIZE;
    int h = gid1 / v_blocks;
    int v_block_id = gid1 - h * v_blocks;
    int i_v_base = v_block_id * V_BLOCK_SIZE;
    __global const INPUT0_TYPE* q_ptr = q + b * BATCH_STRIDE;
    __global const INPUT1_TYPE* k_ptr = k + b * BATCH_STRIDE;
    __global const INPUT2_TYPE* v_ptr = v + b * BATCH_STRIDE;
    __global const INPUT3_TYPE* g_ptr = g + b * K_HEAD_NUMS * seq_len;
    __global const INPUT4_TYPE* beta_ptr = beta + b * K_HEAD_NUMS * seq_len;
    int out_base = b * K_HEAD_NUMS * seq_len * K_HEAD_DIMS + h * K_HEAD_DIMS;
#if (K_HEAD_DIMS == 128)
#if (SUBGROUP_SIZE == 8)
     float8 init_state[V_BLOCK_SIZE][2];
     float8 b_k[2];
     float8 b_q[2];
#else
     float8 init_state[V_BLOCK_SIZE];
     float8 b_k;
     float8 b_q;
#endif
 #elif (K_HEAD_DIMS % 32) == 0
     float2 init_state[V_BLOCK_SIZE][K_HEAD_DIMS / 32];
     float2 b_k[K_HEAD_DIMS / 32];
     float2 b_q[K_HEAD_DIMS / 32];
 #else
     float init_state[V_BLOCK_SIZE][(K_HEAD_DIMS + SUBGROUP_SIZE - 1) / SUBGROUP_SIZE] = {0};
     float b_k[(K_HEAD_DIMS + SUBGROUP_SIZE - 1) / SUBGROUP_SIZE] = {0};
     float b_q[(K_HEAD_DIMS + SUBGROUP_SIZE - 1) / SUBGROUP_SIZE] = {0};
 #endif
    int id_sg_local = get_sub_group_local_id();

    for (int iv = 0; iv < V_BLOCK_SIZE; iv++) {
        int i_v = i_v_base + iv;
        int init_base = b * K_HEAD_NUMS * K_HEAD_DIMS * K_HEAD_DIMS + h * K_HEAD_DIMS * K_HEAD_DIMS + i_v * K_HEAD_DIMS;
 #if (K_HEAD_DIMS == 128)
#if (SUBGROUP_SIZE == 8)
    #define DATA_VEC MAKE_VECTOR_TYPE(INPUT5_TYPE, 8)
    DATA_VEC h8_0 = BLOCK_READN(INPUT5_TYPE, 8, initial_state, init_base);
    DATA_VEC h8_1 = BLOCK_READN(INPUT5_TYPE, 8, initial_state, init_base + (SUBGROUP_SIZE * 8));
    #undef DATA_VEC
    init_state[iv][0] = convert_float8(h8_0);
    init_state[iv][1] = convert_float8(h8_1);
#else
    #define DATA_VEC MAKE_VECTOR_TYPE(INPUT5_TYPE, 8)
    DATA_VEC h8 = BLOCK_READN(INPUT5_TYPE, 8, initial_state, init_base);
    #undef DATA_VEC
    init_state[iv] = convert_float8(h8);
 #endif
 #elif (K_HEAD_DIMS % 32) == 0
#if (SUBGROUP_SIZE == 16)
     for (int j = id_sg_local; j < K_HEAD_DIMS; j += 32) {
         int idx = j >> 5;
         #define DATA_VEC MAKE_VECTOR_TYPE(INPUT5_TYPE, 2)
         DATA_VEC h2 = BLOCK_READN(INPUT5_TYPE, 2, initial_state, init_base + (j - id_sg_local));
         #undef DATA_VEC
         init_state[iv][idx] = convert_float2(h2);
     }
 #else
     for (int j = id_sg_local; j < K_HEAD_DIMS; j += SUBGROUP_SIZE) {
         int idx = j / SUBGROUP_SIZE;
         init_state[iv][idx] = convert_float(initial_state[init_base + j]);
     }
 #endif
 #else
     for (int j = id_sg_local; j < K_HEAD_DIMS; j += SUBGROUP_SIZE) {
         int idx = j / SUBGROUP_SIZE;
         init_state[iv][idx] = convert_float(initial_state[init_base + j]);
     }
 #endif
    }

        int kv_base = h * K_HEAD_DIMS;
        int out_i_base = out_base;
        for (int i = 0; i < seq_len; i++, kv_base += STEP_STRIDE, out_i_base += STEP_STRIDE) {
            float b_g = exp(convert_float(g_ptr[i * K_HEAD_NUMS + h]));
            float b_beta = convert_float(beta_ptr[i * K_HEAD_NUMS + h]);

     #if (K_HEAD_DIMS == 128)
            #if (SUBGROUP_SIZE == 8)
                #define DATA_VEC MAKE_VECTOR_TYPE(INPUT0_TYPE, 8)
                b_k[0] = convert_float8(BLOCK_READN(INPUT1_TYPE, 8, k_ptr, kv_base));
                b_k[1] = convert_float8(BLOCK_READN(INPUT1_TYPE, 8, k_ptr, kv_base + (SUBGROUP_SIZE * 8)));
                b_q[0] = convert_float8(BLOCK_READN(INPUT0_TYPE, 8, q_ptr, kv_base));
                b_q[1] = convert_float8(BLOCK_READN(INPUT0_TYPE, 8, q_ptr, kv_base + (SUBGROUP_SIZE * 8)));
                #undef DATA_VEC
            #else
                #define DATA_VEC MAKE_VECTOR_TYPE(INPUT0_TYPE, 8)
                b_k = convert_float8(BLOCK_READN(INPUT1_TYPE, 8, k_ptr, kv_base));
                b_q = convert_float8(BLOCK_READN(INPUT0_TYPE, 8, q_ptr, kv_base));
                #undef DATA_VEC
     #endif
     #elif (K_HEAD_DIMS % 32) == 0
    #if (SUBGROUP_SIZE == 16)
         #pragma unroll
         for (int j = id_sg_local; j < K_HEAD_DIMS; j += 32) {
             int idx = j >> 5;
             #define DATA_VEC MAKE_VECTOR_TYPE(INPUT0_TYPE, 2)
             b_k[idx] = convert_float2(BLOCK_READN(INPUT1_TYPE, 2, k_ptr, kv_base + (j - id_sg_local)));
             b_q[idx] = convert_float2(BLOCK_READN(INPUT0_TYPE, 2, q_ptr, kv_base + (j - id_sg_local)));
             #undef DATA_VEC
         }
     #else
         for (int j = id_sg_local; j < K_HEAD_DIMS; j += SUBGROUP_SIZE) {
             int idx = j / SUBGROUP_SIZE;
             b_k[idx] = convert_float(k_ptr[kv_base + j]);
             b_q[idx] = convert_float(q_ptr[kv_base + j]);
         }
     #endif
     #else
         for (int j = id_sg_local; j < K_HEAD_DIMS; j += SUBGROUP_SIZE) {
             int idx = j / SUBGROUP_SIZE;
             b_k[idx] = convert_float(k_ptr[kv_base + j]);
             b_q[idx] = convert_float(q_ptr[kv_base + j]);
         }
     #endif

    #if (K_HEAD_DIMS == 128)
    #if (SUBGROUP_SIZE == 8)
            float k_sum = sum8(b_k[0] * b_k[0]) + sum8(b_k[1] * b_k[1]);
            float k_scale = l2norm_scale(k_sum, 1.0f);
            b_k[0] *= k_scale;
            b_k[1] *= k_scale;

            float q_sum = sum8(b_q[0] * b_q[0]) + sum8(b_q[1] * b_q[1]);
            float q_scale = l2norm_scale(q_sum, SCALE_FACTOR);
            b_q[0] = b_q[0] * q_scale;
            b_q[1] = b_q[1] * q_scale;
    #else
            float k_sum = sum8(b_k * b_k);
            float k_scale = l2norm_scale(k_sum, 1.0f);
            b_k *= k_scale;

            float q_sum = sum8(b_q * b_q);
            float q_scale = l2norm_scale(q_sum, SCALE_FACTOR);
            b_q = b_q * q_scale;
    #endif
    #elif (K_HEAD_DIMS % 32) == 0
    #if (SUBGROUP_SIZE == 16)
            float k_sum = 0.0f;
            for (int j = id_sg_local; j < K_HEAD_DIMS; j += 32) {
                int idx = j >> 5;
                k_sum += sum2(b_k[idx] * b_k[idx]);
            }
            float k_scale = l2norm_scale(k_sum, 1.0f);
            for (int j = id_sg_local; j < K_HEAD_DIMS; j += 32) {
                int idx = j >> 5;
                b_k[idx] *= (float2)(k_scale);
            }

            float q_sum = 0.0f;
            for (int j = id_sg_local; j < K_HEAD_DIMS; j += 32) {
                int idx = j >> 5;
                q_sum += sum2(b_q[idx] * b_q[idx]);
            }
            float q_scale = l2norm_scale(q_sum, SCALE_FACTOR);
            for (int j = id_sg_local; j < K_HEAD_DIMS; j += 32) {
                int idx = j >> 5;
                b_q[idx] *= (float2)(q_scale);
            }
    #else
            float k_sum = 0.0f;
            for (int j = id_sg_local; j < K_HEAD_DIMS; j += SUBGROUP_SIZE) {
                int idx = j / SUBGROUP_SIZE;
                k_sum = fma(b_k[idx], b_k[idx], k_sum);
            }
            k_sum = sub_group_reduce_add(k_sum);
            k_sum = sub_group_broadcast(k_sum, 0);
            float k_scale = rsqrt(k_sum + 0.000001f);
            for (int j = id_sg_local; j < K_HEAD_DIMS; j += SUBGROUP_SIZE) {
                int idx = j / SUBGROUP_SIZE;
                b_k[idx] *= k_scale;
            }

            float q_sum = 0.0f;
            for (int j = id_sg_local; j < K_HEAD_DIMS; j += SUBGROUP_SIZE) {
                int idx = j / SUBGROUP_SIZE;
                q_sum = fma(b_q[idx], b_q[idx], q_sum);
            }
            float q_scale = l2norm_scale(q_sum, SCALE_FACTOR);
            for (int j = id_sg_local; j < K_HEAD_DIMS; j += SUBGROUP_SIZE) {
                int idx = j / SUBGROUP_SIZE;
                b_q[idx] *= (q_scale);
            }
    #endif
    #else
            float k_sum = 0.0f;
            for (int j = id_sg_local; j < K_HEAD_DIMS; j += SUBGROUP_SIZE) {
                int idx = j / SUBGROUP_SIZE;
                k_sum = fma(b_k[idx], b_k[idx], k_sum);
            }
            float k_scale = l2norm_scale(k_sum, 1.0f);
            for (int j = id_sg_local; j < K_HEAD_DIMS; j += SUBGROUP_SIZE) {
                int idx = j / SUBGROUP_SIZE;
                b_k[idx] *= k_scale;
            }

            float q_sum = 0.0f;
            for (int j = id_sg_local; j < K_HEAD_DIMS; j += SUBGROUP_SIZE) {
                int idx = j / SUBGROUP_SIZE;
                q_sum = fma(b_q[idx], b_q[idx], q_sum);
            }
            float q_scale = l2norm_scale(q_sum, SCALE_FACTOR);
            for (int j = id_sg_local; j < K_HEAD_DIMS; j += SUBGROUP_SIZE) {
                int idx = j / SUBGROUP_SIZE;
                b_q[idx] *= (q_scale);
            }
    #endif

        for (int iv = 0; iv < V_BLOCK_SIZE; iv++) {
            int i_v = i_v_base + iv;
 #if (K_HEAD_DIMS == 128)
 #if (SUBGROUP_SIZE == 8)
            init_state[iv][0] *= b_g;
            init_state[iv][1] *= b_g;
            float hk_acc = sum8(init_state[iv][0] * b_k[0]) + sum8(init_state[iv][1] * b_k[1]);
            hk_acc = sub_group_reduce_add(hk_acc);
            hk_acc = sub_group_broadcast(hk_acc, 0);

            int v_base = kv_base + (i_v & ~(SUBGROUP_SIZE - 1));
            int v_lane = i_v & (SUBGROUP_SIZE - 1);
            INPUT2_TYPE v_val_h = AS_INPUT0_TYPE(BLOCK_READN(INPUT2_TYPE, 1, v_ptr, v_base));
            float v_val = convert_float(v_val_h);
            float b_v = sub_group_broadcast(v_val, v_lane);
            b_v -= hk_acc;
            b_v *= b_beta;
            init_state[iv][0] = fma(b_k[0], (float8)(b_v), init_state[iv][0]);
            init_state[iv][1] = fma(b_k[1], (float8)(b_v), init_state[iv][1]);

            float out_acc = sum8(init_state[iv][0] * b_q[0]) + sum8(init_state[iv][1] * b_q[1]);
            out_acc = sub_group_reduce_add(out_acc);
            out_acc = sub_group_broadcast(out_acc, 0);
            if (id_sg_local == 0) {
                output[out_i_base + i_v] = convert_half_rte(out_acc);
            }
 #else
            init_state[iv] *= b_g;
            float hk_acc = sum8(init_state[iv] * b_k);
            hk_acc = sub_group_reduce_add(hk_acc);
            hk_acc = sub_group_broadcast(hk_acc, 0);

            int v_base = kv_base + (i_v & ~(SUBGROUP_SIZE - 1));
            int v_lane = i_v & (SUBGROUP_SIZE - 1);
            INPUT2_TYPE v_val_h = AS_INPUT0_TYPE(BLOCK_READN(INPUT2_TYPE, 1, v_ptr, v_base));
            float v_val = convert_float(v_val_h);
            float b_v = sub_group_broadcast(v_val, v_lane);
            b_v -= hk_acc;
            b_v *= b_beta;
            init_state[iv] = fma(b_k, (float8)(b_v), init_state[iv]);

            float out_acc = sum8(init_state[iv] * b_q);
            out_acc = sub_group_reduce_add(out_acc);
            out_acc = sub_group_broadcast(out_acc, 0);
            if (id_sg_local == 0) {
                output[out_i_base + i_v] = convert_half_rte(out_acc);
            }
 #endif
 #elif (K_HEAD_DIMS % 32) == 0
 #if (SUBGROUP_SIZE == 16)
            for (int j = id_sg_local; j < K_HEAD_DIMS; j += 32) {
                int idx = j >> 5;
                init_state[iv][idx] *= b_g;
            }
            float hk_acc = 0.0f;
            for (int j = id_sg_local; j < K_HEAD_DIMS; j += 32) {
                int idx = j >> 5;
                hk_acc += sum2(init_state[iv][idx] * b_k[idx]);
            }
            hk_acc = sub_group_reduce_add(hk_acc);
            hk_acc = sub_group_broadcast(hk_acc, 0);

            int v_base = kv_base + (i_v & ~(SUBGROUP_SIZE - 1));
            int v_lane = i_v & (SUBGROUP_SIZE - 1);
            INPUT2_TYPE v_val_h = AS_INPUT0_TYPE(BLOCK_READN(INPUT2_TYPE, 1, v_ptr, v_base));
            float v_val = convert_float(v_val_h);
            float b_v = sub_group_broadcast(v_val, v_lane);
            b_v -= hk_acc;
            b_v *= b_beta;
            for (int j = id_sg_local; j < K_HEAD_DIMS; j += 32) {
                int idx = j >> 5;
                init_state[iv][idx] = fma(b_k[idx], (float2)(b_v), init_state[iv][idx]);
            }

            float out_acc = 0.0f;
            for (int j = id_sg_local; j < K_HEAD_DIMS; j += 32) {
                int idx = j >> 5;
                out_acc += sum2(init_state[iv][idx] * b_q[idx]);
            }
            out_acc = sub_group_reduce_add(out_acc);
            out_acc = sub_group_broadcast(out_acc, 0);
            if (id_sg_local == 0) {
                output[out_i_base + i_v] = convert_half_rte(out_acc);
            }
 #else
            for (int n = id_sg_local; n < K_HEAD_DIMS; n += SUBGROUP_SIZE) {
                int idx = n / SUBGROUP_SIZE;
                init_state[iv][idx] *= b_g;
            }
            float hk_acc = 0.0f;
            for (int n = id_sg_local; n < K_HEAD_DIMS; n += SUBGROUP_SIZE) {
                int idx = n / SUBGROUP_SIZE;
                hk_acc = fma(init_state[iv][idx], b_k[idx], hk_acc);
            }
            hk_acc = sub_group_reduce_add(hk_acc);
            hk_acc = sub_group_broadcast(hk_acc, 0);

            int v_base = kv_base + (i_v & ~(SUBGROUP_SIZE - 1));
            int v_lane = i_v & (SUBGROUP_SIZE - 1);
            INPUT2_TYPE v_val_h = AS_INPUT0_TYPE(BLOCK_READN(INPUT2_TYPE, 1, v_ptr, v_base));
            float v_val = convert_float(v_val_h);
            float b_v = sub_group_broadcast(v_val, v_lane);
            b_v -= hk_acc;
            b_v *= b_beta;
            for (int n = id_sg_local; n < K_HEAD_DIMS; n += SUBGROUP_SIZE) {
                int idx = n / SUBGROUP_SIZE;
                init_state[iv][idx] = fma(b_k[idx], (float2)(b_v), init_state[iv][idx]);
            }

            float out_acc = 0.0f;
            for (int n = id_sg_local; n < K_HEAD_DIMS; n += SUBGROUP_SIZE) {
                int idx = n / SUBGROUP_SIZE;
                out_acc += sum2(init_state[iv][idx] * b_q[idx]);
            }
            out_acc = sub_group_reduce_add(out_acc);
            out_acc = sub_group_broadcast(out_acc, 0);
            if (id_sg_local == 0) {
                output[out_i_base + i_v] = convert_half_rte(out_acc);
            }
 #endif
 #else
            for (int n = id_sg_local; n < K_HEAD_DIMS; n += SUBGROUP_SIZE) {
                int idx = n / SUBGROUP_SIZE;
                init_state[iv][idx] *= b_g;
            }
            float hk_acc = 0.0f;
            for (int n = id_sg_local; n < K_HEAD_DIMS; n += SUBGROUP_SIZE) {
                int idx = n / SUBGROUP_SIZE;
                hk_acc = fma(init_state[iv][idx], b_k[idx], hk_acc);
            }
            hk_acc = sub_group_reduce_add(hk_acc);
            hk_acc = sub_group_broadcast(hk_acc, 0);

            float b_v = convert_float(v_ptr[kv_base + i_v]);
            b_v -= hk_acc;
            b_v *= b_beta;

            for (int n = id_sg_local; n < K_HEAD_DIMS; n += SUBGROUP_SIZE) {
                int idx = n / SUBGROUP_SIZE;
                init_state[iv][idx] = fma(b_k[idx], b_v, init_state[iv][idx]);
            }

            float out_acc = 0.0f;
            for (int n = id_sg_local; n < K_HEAD_DIMS; n += SUBGROUP_SIZE) {
                int idx = n / SUBGROUP_SIZE;
                out_acc = fma(init_state[iv][idx], b_q[idx], out_acc);
            }
            out_acc = sub_group_reduce_add(out_acc);
            out_acc = sub_group_broadcast(out_acc, 0);
            if (id_sg_local == 0) {
                output[out_i_base + i_v] = convert_half_rte(out_acc);
            }
 #endif
        }
    }

    for (int iv = 0; iv < V_BLOCK_SIZE; iv++) {
        int i_v = i_v_base + iv;
        int init_base = b * K_HEAD_NUMS * K_HEAD_DIMS * K_HEAD_DIMS + h * K_HEAD_DIMS * K_HEAD_DIMS + i_v * K_HEAD_DIMS;
 #if (K_HEAD_DIMS == 128)
 #if (SUBGROUP_SIZE == 8)
    #define DATA_VEC MAKE_VECTOR_TYPE(INPUT5_TYPE, 8)
    DATA_VEC h8_0 = convert_half8_rte(init_state[iv][0]);
    DATA_VEC h8_1 = convert_half8_rte(init_state[iv][1]);
    BLOCK_WRITEN(INPUT5_TYPE, 8, initial_state, init_base, h8_0);
    BLOCK_WRITEN(INPUT5_TYPE, 8, initial_state, init_base + (SUBGROUP_SIZE * 8), h8_1);
    #undef DATA_VEC
 #else
    #define DATA_VEC MAKE_VECTOR_TYPE(INPUT5_TYPE, 8)
    DATA_VEC h8 = convert_half8_rte(init_state[iv]);
    BLOCK_WRITEN(INPUT5_TYPE, 8, initial_state, init_base, h8);
    #undef DATA_VEC
 #endif
 #elif (K_HEAD_DIMS % 32) == 0
 #if (SUBGROUP_SIZE == 16)
     for (int j = id_sg_local; j < K_HEAD_DIMS; j += 32) {
         int idx = j >> 5;
        #define DATA_VEC MAKE_VECTOR_TYPE(INPUT5_TYPE, 2)
        DATA_VEC h2 = convert_half2_rte(init_state[iv][idx]);
        BLOCK_WRITEN(INPUT5_TYPE, 2, initial_state, init_base + (j - id_sg_local), h2);
        #undef DATA_VEC
     }
 #else
     for (int j = id_sg_local; j < K_HEAD_DIMS; j += SUBGROUP_SIZE) {
         int idx = j / SUBGROUP_SIZE;
         initial_state[init_base + j] = convert_half_rte(init_state[iv][idx]);
     }
 #endif
 #else
     for (int j = id_sg_local; j < K_HEAD_DIMS; j += SUBGROUP_SIZE) {
         int idx = j / SUBGROUP_SIZE;
         initial_state[init_base + j] = convert_half_rte(init_state[iv][idx]);
     }
 #endif
    }
}
#endif