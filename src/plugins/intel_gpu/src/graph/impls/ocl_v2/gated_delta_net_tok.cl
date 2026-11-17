// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "include/batch_headers/common.cl"
#include "include/batch_headers/sub_group_block_read.cl"
#include "include/batch_headers/sub_group_block_write.cl"
#include "include/batch_headers/sub_group_shuffle.cl"
#include "paged_gated_delta_net_type_utils.cl"

// Per-token Gated DeltaNet, S_t = a_t S_{t-1} (I - b_t k_t k_t^T) + b_t v_t k_t^T, o_t = S_t q_t, for the dense op: the tuned
// paged_gated_delta_net_opt.cl kernel (d_v-on-lanes tiling) on the dense layouts. It is the choice for devices where the chunked DPAS
// path does not pay off.
//
// One subgroup owns V_BLOCK_SIZE = SUBGROUP_SIZE columns of the state of one (batch, head): lane l holds column start_iv + l as
// K_HEAD_DIM floats in registers, so the state needs no reduction across lanes. q and k are read as a lane's interleaved share of the
// row (a subgroup block read), the dot products over the k axis run inside the lane, and q / k elements reach every lane by
// broadcast (KD). The operands of token t + 1 are issued before the state work of token t, and q / k / v of token t + PF_AHEAD are
// prefetched. The state is [B, HV, DK, V]: for a fixed k the 16 lanes read 16 consecutive v, one block read each.
//
// Defined by the host (gated_delta_net_opt.cpp): K_HEAD_NUM, V_HEAD_NUM, K_HEAD_DIM, V_HEAD_DIM, SUBGROUP_SIZE, V_BLOCK_SIZE (= SUBGROUP_SIZE),
// K_VEC_SIZE, ACC_WAYS, PF_AHEAD, PF_BYTES, FUSE_QK_L2NORM, Q_L2_NORM_EPS, K_L2_NORM_EPS, SCALE_FACTOR, OUTPUT_STATE.

#if (V_BLOCK_SIZE != SUBGROUP_SIZE)
#    error "each lane holds one v column, so V_BLOCK_SIZE must equal SUBGROUP_SIZE"
#endif
#if ((K_HEAD_DIM % (SUBGROUP_SIZE * K_VEC_SIZE)) != 0)
#    error "K_VEC_SIZE must divide the per-lane share of K_HEAD_DIM"
#endif
#if ((V_HEAD_DIM % V_BLOCK_SIZE) != 0)
#    error "V_BLOCK_SIZE must tile V_HEAD_DIM"
#endif
#if ((K_HEAD_DIM % ACC_WAYS) != 0)
#    error "ACC_WAYS must divide K_HEAD_DIM"
#endif

#define K_LANE_ELEMS (K_HEAD_DIM / SUBGROUP_SIZE)  // elements of a q / k row per lane
#define K_VEC_COUNT (K_LANE_ELEMS / K_VEC_SIZE)    // block reads per row


// Chunk c of a row: component j of lane l is element (c * K_VEC_SIZE + j) * SUBGROUP_SIZE + l, so array element e of a lane holds row element
// e * SUBGROUP_SIZE + l, the order the unrolled loops below index it.
#if (K_VEC_SIZE == 8)
#    define TOK_SPREAD(dst, f, c)                                                                          \
        dst[(c) * 8 + 0] = f.s0; dst[(c) * 8 + 1] = f.s1; dst[(c) * 8 + 2] = f.s2; dst[(c) * 8 + 3] = f.s3; \
        dst[(c) * 8 + 4] = f.s4; dst[(c) * 8 + 5] = f.s5; dst[(c) * 8 + 6] = f.s6; dst[(c) * 8 + 7] = f.s7
#    define TOK_CHUNK_DECV(KT, v) PGDN_DECV(KT, v, 8)
#elif (K_VEC_SIZE == 4)
#    define TOK_SPREAD(dst, f, c) dst[(c) * 4 + 0] = f.s0; dst[(c) * 4 + 1] = f.s1; dst[(c) * 4 + 2] = f.s2; dst[(c) * 4 + 3] = f.s3
#    define TOK_CHUNK_DECV(KT, v) PGDN_DECV(KT, v, 4)
#elif (K_VEC_SIZE == 2)
#    define TOK_SPREAD(dst, f, c) dst[(c) * 2 + 0] = f.s0; dst[(c) * 2 + 1] = f.s1
#    define TOK_CHUNK_DECV(KT, v) PGDN_DECV(KT, v, 2)
#else
#    define TOK_SPREAD(dst, f, c) dst[(c)] = f
#    define TOK_CHUNK_DECV(KT, v) PGDN_SDEC(KT, v)
#endif

#define TOK_READ_CHUNK(TYPE, KT, dst, p, c)                                                              \
    do {                                                                                                 \
        const MAKE_VECTOR_TYPE(float, K_VEC_SIZE) _f = TOK_CHUNK_DECV(KT, BLOCK_READN(TYPE, K_VEC_SIZE, (p), (c) * K_VEC_SIZE * SUBGROUP_SIZE)); \
        TOK_SPREAD(dst, _f, c);                                                                          \
    } while (0)

#define TOK_READ_ROW(TYPE, KT, dst, p)                                                                   \
    do {                                                                                                 \
        unroll_for(int _c = 0; _c < K_VEC_COUNT; ++_c) TOK_READ_CHUNK(TYPE, KT, dst, p, _c);             \
    } while (0)

// Element d of a q / k row is array element d / SUBGROUP_SIZE of lane d % SUBGROUP_SIZE.
#define KD(v, d) sub_group_broadcast((v)[(d) / SUBGROUP_SIZE], (d) % SUBGROUP_SIZE)

// One token's operands: q and k (normalized when FUSE_QK_L2NORM), this lane's v column, exp(gate), beta.
#define TOK_READ_TOKEN(q, k, v, g, b)                                         \
    do {                                                                      \
        TOK_READ_ROW(INPUT0_TYPE, Q_KT, q, qp);                               \
        TOK_READ_ROW(INPUT1_TYPE, K_KT, k, kp);                               \
        FUNC_CALL(l2_normalize_qk)(q, k, SCALE_FACTOR);                       \
        v = PGDN_SDEC(V_KT, BLOCK_READN(INPUT2_TYPE, 1, vp, 0));              \
        g = exp(PGDN_SDEC(G_KT, *gp));                                        \
        b = PGDN_SDEC(B_KT, *bp);                                             \
    } while (0)

inline void FUNC(l2_normalize_qk)(float* qq, float* kk, float scale_factor) {
#if FUSE_QK_L2NORM
    float qs = 0.0f, ks = 0.0f;
    unroll_for(int c = 0; c < K_LANE_ELEMS; ++c) {
        qs += qq[c] * qq[c];
        ks += kk[c] * kk[c];
    }
    const float q_scale = rsqrt(sub_group_reduce_add(qs) + Q_L2_NORM_EPS) * scale_factor;
    const float k_scale = rsqrt(sub_group_reduce_add(ks) + K_L2_NORM_EPS);
#else
    const float q_scale = scale_factor;
    const float k_scale = 1.0f;
#endif
    unroll_for(int c = 0; c < K_LANE_ELEMS; ++c) {
        qq[c] *= q_scale;
        kk[c] *= k_scale;
    }
}

REQD_SUB_GROUP_SIZE(SUBGROUP_SIZE)
KERNEL(gated_delta_net_tok)
(__global INPUT0_TYPE* query,   // [B, T, HK, DK]
 __global INPUT1_TYPE* key,     // [B, T, HK, DK]
 __global INPUT2_TYPE* value,   // [B, T, HV, V]
 __global INPUT3_TYPE* initial_state,  // [B, HV, DK, V]
 __global INPUT4_TYPE* gate,    // [B, T, HV]
 __global INPUT5_TYPE* beta,    // [B, T, HV]
 __global OUTPUT_TYPE* output,  // [B, T, HV, V]
#if OUTPUT_STATE
 __global OUTPUT1_TYPE* output_state,  // [B, HV, DK, V]
#endif
 int seq_len,
 int query_head_offset,
 int key_head_offset,
 int value_head_offset,
 int q_token_stride,
 int k_token_stride,
 int v_token_stride) {
    const int v_tile = (int)get_group_id(0);
    const int h = (int)get_global_id(1);
    const int b = (int)get_global_id(2);
    const int lane = (int)get_sub_group_local_id();
    const int start_iv = v_tile * V_BLOCK_SIZE;

    const int group_size = V_HEAD_NUM / K_HEAD_NUM;
    const int hk = h / group_size;
    const long state_base = (long)(b * V_HEAD_NUM + h) * K_HEAD_DIM * V_HEAD_DIM + start_iv;

    float state[K_HEAD_DIM];
    {
        const __global INPUT3_TYPE* sp = initial_state + state_base;
        unroll_for(int d = 0; d < K_HEAD_DIM; ++d) state[d] = PGDN_SDEC(S_KT, BLOCK_READN(INPUT3_TYPE, 1, sp, (long)d * V_HEAD_DIM));
    }

    const long token0 = (long)b * seq_len;
    const __global INPUT0_TYPE* qp = query + token0 * q_token_stride + (hk + query_head_offset) * K_HEAD_DIM;
    const __global INPUT1_TYPE* kp = key + token0 * k_token_stride + (hk + key_head_offset) * K_HEAD_DIM;
    const __global INPUT2_TYPE* vp = value + token0 * v_token_stride + (h + value_head_offset) * V_HEAD_DIM + start_iv;
    const __global INPUT4_TYPE* gp = gate + token0 * V_HEAD_NUM + h;
    const __global INPUT5_TYPE* bp = beta + token0 * V_HEAD_NUM + h;
    __global OUTPUT_TYPE* op = output + (token0 * V_HEAD_NUM + h) * V_HEAD_DIM + start_iv;

#if PF_AHEAD > 0
    // The prefetch runs PF_AHEAD tokens ahead, which is past the end for a short sequence: clamped.
    const int pf_first = min(PF_AHEAD, max(seq_len - 1, 0));
    const __global INPUT0_TYPE* pfq = qp + (long)pf_first * q_token_stride;
    const __global INPUT1_TYPE* pfk = kp + (long)pf_first * k_token_stride;
    const __global INPUT2_TYPE* pfv = vp + (long)pf_first * v_token_stride;
#endif

    // Prologue of the software pipeline: the first token's operands.
    float q_pre[K_LANE_ELEMS], k_pre[K_LANE_ELEMS];
    float v_pre, g_pre, b_pre;
    TOK_READ_TOKEN(q_pre, k_pre, v_pre, g_pre, b_pre);

    // The pipeline stages token + 1 and the prefetch reads PF_AHEAD further, so both would leave the buffer at the end of the sequence. Instead of
    // checking the bounds per token the range is split into three segments and the advance strides are zeroed where advancing would overrun:
    // segment 0 advances both, segment 1 freezes the prefetch, segment 2 freezes both. A frozen pointer re-reads the last valid token and the
    // staged value is discarded by the loop exit.
    int token = 0;
    __attribute__((opencl_unroll_hint)) for (int seg = 0; seg < 3; ++seg) {
        int seg_end;
        if (seg == 0)
            seg_end = min(seq_len, max(seq_len - PF_AHEAD - 1, token));
        else if (seg == 1)
            seg_end = min(seq_len, max(seq_len - 1, token));
        else
            seg_end = seq_len;

        const int qs = (seg < 2) ? q_token_stride : 0;
        const int ks = (seg < 2) ? k_token_stride : 0;
        const int vs = (seg < 2) ? v_token_stride : 0;
        const int gs = (seg < 2) ? V_HEAD_NUM : 0;
#if PF_AHEAD > 0
        const int pq = (seg < 1) ? q_token_stride : 0;
        const int pk = (seg < 1) ? k_token_stride : 0;
        const int pv = (seg < 1) ? v_token_stride : 0;
#endif

        for (; token < seg_end; ++token) {
            float qv[K_LANE_ELEMS], kv[K_LANE_ELEMS];
            unroll_for(int c = 0; c < K_LANE_ELEMS; ++c) {
                qv[c] = q_pre[c];
                kv[c] = k_pre[c];
            }
            const float v_val = v_pre;
            const float b_g = g_pre;
            const float b_beta = b_pre;

            qp += qs;
            kp += ks;
            vp += vs;
            gp += gs;
            bp += gs;
            TOK_READ_TOKEN(q_pre, k_pre, v_pre, g_pre, b_pre);

#if PF_AHEAD > 0
            // One request each for q, k and v per token, at the start of each row (PF_BYTES is the request size, not the row).
            pfq += pq;
            pfk += pk;
            pfv += pv;
            prefetch(pfq, PF_BYTES / sizeof(INPUT0_TYPE));
            prefetch(pfk, PF_BYTES / sizeof(INPUT1_TYPE));
            prefetch(pfv, PF_BYTES / sizeof(INPUT2_TYPE));
#endif

            // Pass 1 decays the state and takes h_k against the decayed state; pass 2 adds the update and takes the output dot product. Both
            // dot products run inside the lane, in ACC_WAYS independent chains.
            float hk_acc[ACC_WAYS];
            unroll_for(int a = 0; a < ACC_WAYS; ++a) {
                state[a] *= b_g;
                hk_acc[a] = state[a] * KD(kv, a);
            }
            unroll_for(int d = ACC_WAYS; d < K_HEAD_DIM; ++d) {
                state[d] *= b_g;
                hk_acc[d % ACC_WAYS] += state[d] * KD(kv, d);
            }
            float h_k = 0.0f;
            unroll_for(int a = 0; a < ACC_WAYS; ++a) h_k += hk_acc[a];
            const float update = (v_val - h_k) * b_beta;

            float out_acc[ACC_WAYS];
            unroll_for(int a = 0; a < ACC_WAYS; ++a) {
                state[a] = fma(KD(kv, a), update, state[a]);
                out_acc[a] = state[a] * KD(qv, a);
            }
            unroll_for(int d = ACC_WAYS; d < K_HEAD_DIM; ++d) {
                state[d] = fma(KD(kv, d), update, state[d]);
                out_acc[d % ACC_WAYS] += state[d] * KD(qv, d);
            }
            float o = 0.0f;
            unroll_for(int a = 0; a < ACC_WAYS; ++a) o += out_acc[a];

            BLOCK_WRITEN(OUTPUT_TYPE, 1, op, 0, PGDN_SENC(O_KT, o));
            // The output is contiguous [B, T, HV, V]; the value input may be a padded view, so op does not follow v_token_stride.
            op += V_HEAD_NUM * V_HEAD_DIM;
        }
    }

#if OUTPUT_STATE
    __global OUTPUT1_TYPE* sp_out = output_state + state_base;
#    define TOK_STATE_KT PGDN_KT(OUTPUT1_TYPE_SIZE, OUTPUT1_IS_FP)
#    define TOK_STATE_TYPE OUTPUT1_TYPE
#else
    __global INPUT3_TYPE* sp_out = initial_state + state_base;
#    define TOK_STATE_KT S_KT
#    define TOK_STATE_TYPE INPUT3_TYPE
#endif
    unroll_for(int d = 0; d < K_HEAD_DIM; ++d) BLOCK_WRITEN(TOK_STATE_TYPE, 1, sp_out, (long)d * V_HEAD_DIM, PGDN_SENC(TOK_STATE_KT, state[d]));
}
