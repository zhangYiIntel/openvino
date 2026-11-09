// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// Chunked paged Gated DeltaNet on DPAS/XMX, five stages selected by GDN_STAGE (same algorithm as
// gated_delta_net_opt.cl; see there for the math):
//   0 prep       l2-normalize q/k into f16 scratch qn/kn
//   1 kkt_solve  X = (I + A)^-1 per chunk
//   2 wy         w = X (kn beta e2g), u = X (v beta) per chunk
//   3 fwd_h      per (sequence, head, v-block): loop over the sequence's chunks, keep the state in registers,
//                read it from / snapshot it to the paged state table
//   4 fwd_o      o = (qn e2g) h + tril((qn kn^T) decay) vnew per chunk
//
// Tokens of all sequences are packed; chunks of up to GDN_CHUNK tokens never cross a sequence. A global chunk index c
// is mapped to (sequence, first token, length) from subsequence_begins on the device. Scratch buffers are indexed by
// (head, c) with c < max_chunks = ceil(tokens / GDN_CHUNK) + sequences.
//
// The state table is [block][v head][v][k]: a lane reads and writes 8 contiguous k values of its v row directly.
// Snapshots (see cache_interval) are taken after every interval-th token of the sequence and after its last token. A
// chunk is split at the snapshot positions inside it; each part applies the same decayed update, which is exact.

#include "gated_delta_net_dpas_common.cl"

#if INPUT3_TYPE_SIZE == 2
#    define GDN_LOAD_STATE8(p) convert_float8(vload8(0, (const __global half*)(p)))
#    define GDN_STORE_STATE8(v, p) vstore8(convert_half8(v), 0, (__global half*)(p))
#else
#    define GDN_LOAD_STATE8(p) vload8(0, (const __global float*)(p))
#    define GDN_STORE_STATE8(v, p) vstore8((v), 0, (__global float*)(p))
#endif

// Chunk c -> sequence, first token and token count. seq is -1 for chunks past the end of the batch.
inline void FUNC(gdn_locate_chunk)(__global const INPUT6_TYPE* begins, int num_sequences, int c, int* seq, int* tok0, int* len) {
    int rem = c;
    for (int s = 0; s < num_sequences; s++) {
        const int b = begins[s];
        const int n = begins[s + 1] - b;
        const int nc = (n + GDN_CHUNK - 1) / GDN_CHUNK;
        if (rem < nc) {
            *seq = s;
            *tok0 = b + rem * GDN_CHUNK;
            *len = min(GDN_CHUNK, b + n - *tok0);
            return;
        }
        rem -= nc;
    }
    *seq = -1;
    *tok0 = 0;
    *len = 0;
}

// ---------------------------------------------------------------------------
// Stage 0: prep. Grid (max_chunks, HK). Subgroup sg handles chunk tokens [sg * 8, sg * 8 + 8), one at a time; the lanes
// split the key dim, so the l2 norm is a subgroup reduction. Rows past the chunk length are written as zeros.
// ---------------------------------------------------------------------------
#if GDN_STAGE == 0
REQD_SUB_GROUP_SIZE(SUBGROUP_SIZE)
KERNEL(gdn_paged_dpas_prep)(__global const INPUT0_TYPE* q,
                            __global const INPUT1_TYPE* k,
                            __global half* qn,  // out: [HK, max_chunks * CHUNK, DK] f16
                            __global half* kn,  // out: [HK, max_chunks * CHUNK, DK] f16
                            __global const INPUT6_TYPE* subsequence_begins,
                            int num_sequences,
                            int max_chunks,
                            int q_head_offset,
                            int k_head_offset,
                            int q_token_stride,
                            int q_head_stride,
                            int k_token_stride,
                            int k_head_stride) {
    const int DK = K_HEAD_DIM;
    const int i_c = get_group_id(0);
    const int i_hk = get_global_id(1);
    const int lid = get_sub_group_local_id();
    const int sg = get_sub_group_id();

    int seq, tok0, len;
    FUNC(gdn_locate_chunk)(subsequence_begins, num_sequences, i_c, &seq, &tok0, &len);
    if (seq < 0)
        return;

    const size_t q_head = (size_t)(i_hk + q_head_offset) * q_head_stride;
    const size_t k_head = (size_t)(i_hk + k_head_offset) * k_head_stride;

    for (int tt = 0; tt < GDN_CHUNK / GDN_SUBGROUPS; tt++) {
        const int tok = sg * (GDN_CHUNK / GDN_SUBGROUPS) + tt;
        const bool tvalid = (tok < len);
        const size_t dst_row = ((size_t)i_hk * max_chunks + i_c) * GDN_CHUNK + tok;

        {
            float qs = 0.0f;
            float8 qv[GDN_PREP_VECS];
#pragma unroll
            for (int c = 0; c < GDN_PREP_VECS; c++) {
                const int d = (c * SUBGROUP_SIZE + lid) * GDN_VEC;
                qv[c] = tvalid ? FUNC(gdn_load8_q)(q + (size_t)(tok0 + tok) * q_token_stride + q_head + d) : (float8)(0.0f);
                qs += FUNC(gdn_dot8)(qv[c], qv[c]);
            }
#if FUSE_QK_L2NORM
            qs = sub_group_reduce_add(qs);
#endif
            float scale = SCALE_FACTOR;
#if FUSE_QK_L2NORM
            scale *= rsqrt(qs + Q_L2_NORM_EPS);
#endif
#pragma unroll
            for (int c = 0; c < GDN_PREP_VECS; c++)
                vstore8(convert_half8(qv[c] * scale), 0, qn + dst_row * DK + (c * SUBGROUP_SIZE + lid) * GDN_VEC);
        }
        {
            float ks = 0.0f;
            float8 kv[GDN_PREP_VECS];
#pragma unroll
            for (int c = 0; c < GDN_PREP_VECS; c++) {
                const int d = (c * SUBGROUP_SIZE + lid) * GDN_VEC;
                kv[c] = tvalid ? FUNC(gdn_load8_k)(k + (size_t)(tok0 + tok) * k_token_stride + k_head + d) : (float8)(0.0f);
                ks += FUNC(gdn_dot8)(kv[c], kv[c]);
            }
#if FUSE_QK_L2NORM
            ks = sub_group_reduce_add(ks);
#endif
            float scale = 1.0f;
#if FUSE_QK_L2NORM
            scale = rsqrt(ks + K_L2_NORM_EPS);
#endif
#pragma unroll
            for (int c = 0; c < GDN_PREP_VECS; c++)
                vstore8(convert_half8(kv[c] * scale), 0, kn + dst_row * DK + (c * SUBGROUP_SIZE + lid) * GDN_VEC);
        }
    }
}
#endif  // GDN_STAGE == 0

// ---------------------------------------------------------------------------
// Stage 1: kkt_solve. Grid (max_chunks, HV). A = strict_lower(beta exp2(gc_i - gc_j) kn_i . kn_j) stays in SLM and is
// inverted in place; X goes to xinv[head][chunk] in A-fragment-major order.
// ---------------------------------------------------------------------------
#if GDN_STAGE == 1
REQD_SUB_GROUP_SIZE(SUBGROUP_SIZE)
KERNEL(gdn_paged_dpas_kkt_solve)(__global const half* kn,  // [HK, max_chunks * CHUNK, DK]
                                 __global const INPUT4_TYPE* gate,  // [tokens, HV]
                                 __global const INPUT5_TYPE* beta,  // [tokens, HV]
                                 __global half* xinv,               // out: [HV, max_chunks, CHUNK, CHUNK]
                                 __global const INPUT6_TYPE* subsequence_begins,
                                 int num_sequences,
                                 int max_chunks) {
    const int DK = K_HEAD_DIM;
    const int HV = V_HEAD_NUM;
    const int GROUP = V_HEAD_NUM / K_HEAD_NUM;
    const int i_c = get_group_id(0);
    const int i_h = get_global_id(1);
    const int i_hk = i_h / GROUP;
    const int lid = get_sub_group_local_id();
    const int sg = get_sub_group_id();
    const int ltid = sg * SUBGROUP_SIZE + lid;

    int seq, tok0, len;
    FUNC(gdn_locate_chunk)(subsequence_begins, num_sequences, i_c, &seq, &tok0, &len);
    if (seq < 0)
        return;

    __local float x[GDN_CHUNK * GDN_SOLVE_LD] GDN_ALIGNED;  // also the kn tile while the GEMM reads it
    __local half* tile = (__local half*)x;
    __local float a[GDN_CHUNK * GDN_SOLVE_LD] GDN_ALIGNED;
    __local float tmp[GDN_NTMP * GDN_BLK * GDN_BLK_LD] GDN_ALIGNED;
    __local float gc2[GDN_GC_SIZE] GDN_ALIGNED;
    __local float sbeta[GDN_CHUNK] GDN_ALIGNED;
    const int LD = GDN_SOLVE_LD;

    FUNC(gdn_stage_tile_g2l)(kn + ((size_t)i_hk * max_chunks + i_c) * GDN_CHUNK * DK, DK, tile, GDN_DK_PAD, GDN_CHUNK, DK, ltid);
    FUNC(gdn_stage_gc)(gc2, gate, i_h, HV, tok0, tok0 + len, ltid);
    if (ltid < GDN_CHUNK)
        sbeta[ltid] = (ltid < len) ? convert_float(beta[(size_t)(tok0 + ltid) * HV + i_h]) : 0.0f;
    barrier(CLK_LOCAL_MEM_FENCE);

    float8 acc[GDN_TOK_TILES];
#pragma unroll
    for (int c = 0; c < GDN_TOK_TILES; c++)
        acc[c] = 0.0f;
    const int r0 = sg * GDN_TM;
#pragma unroll
    for (int ks = 0; ks < GDN_DK_TILES; ks++) {
        const short8 af = FUNC(gdn_a_frag_l)(tile, GDN_DK_PAD, r0, ks, lid);
#pragma unroll
        for (int c = 0; c < GDN_TOK_TILES; c++)
            acc[c] = FUNC(gdn_dpas)(af, FUNC(gdn_b_frag_l)(tile, GDN_DK_PAD, GDN_TN * c + lid, ks), acc[c]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);  // the tile (aliased with x) is dead

#pragma unroll
    for (int c = 0; c < GDN_TOK_TILES; c++) {
        const int jj = GDN_TN * c + lid;
        const float gj = gc2[jj];
#pragma unroll
        for (int j = 0; j < GDN_TM; j++) {
            const int i = r0 + j;
            a[(size_t)i * LD + jj] = (i > jj) ? acc[c][j] * exp2(gc2[i] - gj) * sbeta[i] : 0.0f;
        }
    }
    for (int e = ltid; e < GDN_CHUNK * GDN_CHUNK; e += GDN_LANES)
        x[(size_t)(e / GDN_CHUNK) * LD + (e % GDN_CHUNK)] = 0.0f;
    barrier(CLK_LOCAL_MEM_FENCE);

    FUNC(gdn_solve_slm)(a, x, tmp, sg, lid);
    FUNC(gdn_store_x_fragments)(x, xinv + ((size_t)i_h * max_chunks + i_c) * GDN_CHUNK * GDN_CHUNK, sg, lid);
}
#endif  // GDN_STAGE == 1

// ---------------------------------------------------------------------------
// Stage 2: wy. Grid (max_chunks, HV). w = X (kn beta e2g), stored fragment-major [sg][ks][lane][8 rows]; u = X (v beta)
// per v-block, stored transposed as uT[block][v col][token]. The n-major B operands are transposed through SLM.
// ---------------------------------------------------------------------------
#if GDN_STAGE == 2
REQD_SUB_GROUP_SIZE(SUBGROUP_SIZE)
KERNEL(gdn_paged_dpas_wy)(__global const half* kn,
                          __global const INPUT2_TYPE* value,  // [tokens, HV, DV], head offset / strides below
                          __global const INPUT4_TYPE* gate,
                          __global const INPUT5_TYPE* beta,
                          __global const half* xinv,
                          __global half* w,   // out: [HV, max_chunks, CHUNK, DK], fragment-major
                          __global half* uT,  // out: [HV, NV, max_chunks, V_BLOCK, CHUNK]
                          __global const INPUT6_TYPE* subsequence_begins,
                          int num_sequences,
                          int max_chunks,
                          int value_head_offset,
                          int v_token_stride,
                          int v_head_stride) {
    const int DK = K_HEAD_DIM;
    const int HV = V_HEAD_NUM;
    const int GROUP = V_HEAD_NUM / K_HEAD_NUM;
    const int i_c = get_group_id(0);
    const int i_h = get_global_id(1);
    const int i_hk = i_h / GROUP;
    const int lid = get_sub_group_local_id();
    const int sg = get_sub_group_id();
    const int ltid = sg * SUBGROUP_SIZE + lid;

    int seq, tok0, len;
    FUNC(gdn_locate_chunk)(subsequence_begins, num_sequences, i_c, &seq, &tok0, &len);
    if (seq < 0)
        return;

    __local half t_kbT[K_HEAD_DIM * GDN_CHUNK_PAD] GDN_ALIGNED;
    __local half t_v[GDN_V_BLOCK * GDN_CHUNK_PAD] GDN_ALIGNED;
    __local float gc2[GDN_GC_SIZE] GDN_ALIGNED;
    __local float sbeta[GDN_CHUNK] GDN_ALIGNED;

    FUNC(gdn_stage_gc)(gc2, gate, i_h, HV, tok0, tok0 + len, ltid);
    if (ltid < GDN_CHUNK)
        sbeta[ltid] = (ltid < len) ? convert_float(beta[(size_t)(tok0 + ltid) * HV + i_h]) : 0.0f;
    barrier(CLK_LOCAL_MEM_FENCE);  // sbeta

    // kbT is built straight from global memory (no SLM tile): this keeps the stage at ~27 KB of SLM, so 4 workgroups fit
    // on an Xe core instead of 2 (the stage is latency bound and gains from every extra resident workgroup).
    const __global half* kn_chunk = kn + ((size_t)i_hk * max_chunks + i_c) * GDN_CHUNK * DK;
    for (int e = ltid; e < GDN_CHUNK * DK / GDN_VEC; e += GDN_LANES) {
        const int tok = e % GDN_CHUNK;
        const int d0 = (e / GDN_CHUNK) * GDN_VEC;
        const float scale = sbeta[tok] * exp2(gc2[tok]);
        const half8 kk = vload8(0, kn_chunk + (size_t)tok * DK + d0);
#pragma unroll
        for (int i = 0; i < GDN_VEC; i++)
            t_kbT[(size_t)(d0 + i) * GDN_CHUNK_PAD + tok] = convert_half(convert_float(kk[i]) * scale);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const size_t i_hc = (size_t)i_h * max_chunks + i_c;
    const __global half* x = xinv + i_hc * GDN_CHUNK * GDN_CHUNK;
    const int r0 = sg * GDN_TM;

    {
        float8 accw[GDN_DK_TILES];
#pragma unroll
        for (int c = 0; c < GDN_DK_TILES; c++)
            accw[c] = 0.0f;
#pragma unroll
        for (int ks = 0; ks < GDN_TOK_TILES; ks++) {
            const short8 af = as_short8(vload8(0, x + (((size_t)sg * GDN_TOK_TILES + ks) * SUBGROUP_SIZE + lid) * GDN_TM));
#pragma unroll
            for (int c = 0; c < GDN_DK_TILES; c++)
                accw[c] = FUNC(gdn_dpas)(af, FUNC(gdn_b_frag_l)(t_kbT, GDN_CHUNK_PAD, GDN_TN * c + lid, ks), accw[c]);
        }
        __global half* wout = w + i_hc * GDN_CHUNK * DK;
#pragma unroll
        for (int c = 0; c < GDN_DK_TILES; c++)
            vstore8(convert_half8(accw[c]), 0, wout + (((size_t)sg * GDN_DK_TILES + c) * SUBGROUP_SIZE + lid) * GDN_TM);
    }

    for (int nv = 0; nv < GDN_NV; nv++) {
        const __global INPUT2_TYPE* vsrc = value + (size_t)tok0 * v_token_stride + (size_t)(i_h + value_head_offset) * v_head_stride + nv * GDN_V_BLOCK;
        for (int e = ltid; e < GDN_CHUNK * GDN_V_BLOCK / GDN_VEC; e += GDN_LANES) {
            const int tok = e % GDN_CHUNK;
            const int c0 = (e / GDN_CHUNK) * GDN_VEC;
            float8 vv = (float8)(0.0f);
            if (tok < len)
                vv = FUNC(gdn_load8_v)(vsrc + (size_t)tok * v_token_stride + c0);
            const float bt = sbeta[tok];
#pragma unroll
            for (int i = 0; i < GDN_VEC; i++)
                t_v[(size_t)(c0 + i) * GDN_CHUNK_PAD + tok] = convert_half(vv[i] * bt);
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        float8 accu[GDN_NJ];
#pragma unroll
        for (int c = 0; c < GDN_NJ; c++)
            accu[c] = 0.0f;
#pragma unroll
        for (int ks = 0; ks < GDN_TOK_TILES; ks++) {
            const short8 af = as_short8(vload8(0, x + (((size_t)sg * GDN_TOK_TILES + ks) * SUBGROUP_SIZE + lid) * GDN_TM));
#pragma unroll
            for (int c = 0; c < GDN_NJ; c++)
                accu[c] = FUNC(gdn_dpas)(af, FUNC(gdn_b_frag_l)(t_v, GDN_CHUNK_PAD, GDN_TN * c + lid, ks), accu[c]);
        }
        __global half* uout = uT + (((size_t)i_h * GDN_NV + nv) * max_chunks + i_c) * GDN_V_BLOCK * GDN_CHUNK;
#pragma unroll
        for (int c = 0; c < GDN_NJ; c++)
            vstore8(convert_half8(accu[c]), 0, uout + (size_t)(GDN_TN * c + lid) * GDN_CHUNK + r0);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}
#endif  // GDN_STAGE == 2

// ---------------------------------------------------------------------------
// Stage 3: fwd_h. Grid (sequences * HV * NV workgroups). A workgroup owns a (sequence, head, v-block) and loops over
// the sequence's chunks. hacc[tile][j] covers state rows sg * DK_PER_SG + 8 tile + e and v columns 16 j + lane.
// Per chunk: publish h (f16) to SLM and hbuf; vnew = u - w h; then, per part between snapshot positions,
// h = exp2(g_hi - g_lo) h + kn^T (vnew exp2(g_hi - gc)) and a snapshot write if the part ends at a snapshot position.
// ---------------------------------------------------------------------------
#if GDN_STAGE == 3
REQD_SUB_GROUP_SIZE(SUBGROUP_SIZE)
KERNEL(gdn_paged_dpas_fwd_h)(__global const half* kn,
                             __global const half* w,
                             __global const half* uT,
                             __global const INPUT4_TYPE* gate,
                             __global INPUT3_TYPE* state_table,  // [blocks, HV, DV, DK]
                             __global half* hbuf,                // out: [HV, NV, max_chunks, V_BLOCK, DK]
                             __global half* vnewT,               // out: [HV, NV, max_chunks, V_BLOCK, CHUNK]
                             __global const INPUT6_TYPE* subsequence_begins,
                             __global const INPUT7_TYPE* block_indices,
                             __global const INPUT8_TYPE* block_indices_begins,
                             __global const INPUT9_TYPE* past_lens,
                             __global const INPUT10_TYPE* cache_interval,
                             int num_sequences,
                             int max_chunks) {
    const int DK = K_HEAD_DIM;
    const int DV = V_HEAD_DIM;
    const int HV = V_HEAD_NUM;
    const int GROUP = V_HEAD_NUM / K_HEAD_NUM;
    const int gid = get_group_id(0);
    const int i_seq = gid / (HV * GDN_NV);
    const int i_h = (gid % (HV * GDN_NV)) / GDN_NV;
    const int i_nv = gid % GDN_NV;
    const int i_hk = i_h / GROUP;
    const int lid = get_sub_group_local_id();
    const int sg = get_sub_group_id();
    const int ltid = sg * SUBGROUP_SIZE + lid;

    const int token_begin = subsequence_begins[i_seq];
    const int seq_len = subsequence_begins[i_seq + 1] - token_begin;
    if (seq_len <= 0)
        return;
    const int n_chunks = (seq_len + GDN_CHUNK - 1) / GDN_CHUNK;
    int c_first = 0;  // global index of the sequence's first chunk
    for (int s = 0; s < i_seq; s++)
        c_first += (subsequence_begins[s + 1] - subsequence_begins[s] + GDN_CHUNK - 1) / GDN_CHUNK;

    const int interval = cache_interval[i_seq];
    const int prev_nums = interval > 0 ? past_lens[i_seq] % interval : 0;
    int next_snapshot = interval > 0 ? (prev_nums > 0 ? interval - prev_nums : interval) : 0x7fffffff;  // tokens into this call
    const int block_begin = block_indices_begins[i_seq];
    const int block_end = block_indices_begins[i_seq + 1];
    int slot = 1;

    __local half hT[GDN_V_BLOCK * GDN_DK_PAD] GDN_ALIGNED;
    __local half uadjT[GDN_V_BLOCK * GDN_CHUNK_PAD] GDN_ALIGNED;
    __local half vnT[GDN_V_BLOCK * GDN_CHUNK_PAD] GDN_ALIGNED;
    __local float gc2[GDN_GC_SIZE] GDN_ALIGNED;

    const int row0 = sg * GDN_DK_PER_SG;
    const size_t state_head = (size_t)i_h * DV * DK + (size_t)(i_nv * GDN_V_BLOCK) * DK;  // v-block of this head within a block
    float8 hacc[GDN_H_TILES][GDN_NJ];
    {
        const __global INPUT3_TYPE* s = state_table + (size_t)block_indices[block_begin] * HV * DV * DK + state_head;
#pragma unroll
        for (int b = 0; b < GDN_H_TILES; b++)
#pragma unroll
            for (int j = 0; j < GDN_NJ; j++)
                hacc[b][j] = GDN_LOAD_STATE8(s + (size_t)(lid + GDN_TN * j) * DK + row0 + b * GDN_TM);
    }

    for (int cc = 0; cc < n_chunks; cc++) {
        const int c = c_first + cc;
        const int done = cc * GDN_CHUNK;  // tokens of the sequence before this chunk
        const int len = min(GDN_CHUNK, seq_len - done);
        const int tok0 = token_begin + done;
        {
            const int cp = c_first + min(cc + GDN_PREFETCH_CHUNKS, n_chunks - 1);
            prefetch(w + ((size_t)i_h * max_chunks + cp) * GDN_CHUNK * DK + ltid * GDN_PF_W, GDN_PF_W);
            prefetch(kn + ((size_t)i_hk * max_chunks + cp) * GDN_CHUNK * DK + ltid * GDN_PF_W, GDN_PF_W);
            prefetch(uT + (((size_t)i_h * GDN_NV + i_nv) * max_chunks + cp) * GDN_V_BLOCK * GDN_CHUNK + ltid * GDN_PF_U, GDN_PF_U);
        }
        const __global half* kg = kn + ((size_t)i_hk * max_chunks + c) * GDN_CHUNK * DK;
        const __global half* wg = w + ((size_t)i_h * max_chunks + c) * GDN_CHUNK * DK;
        const __global half* ug = uT + (((size_t)i_h * GDN_NV + i_nv) * max_chunks + c) * GDN_V_BLOCK * GDN_CHUNK;

        // 1. publish h as f16 n-major, then copy it to hbuf
#pragma unroll
        for (int b = 0; b < GDN_H_TILES; b++)
#pragma unroll
            for (int j = 0; j < GDN_NJ; j++)
                vstore8(convert_half8(hacc[b][j]), 0, hT + (size_t)(lid + GDN_TN * j) * GDN_DK_PAD + row0 + b * GDN_TM);
        barrier(CLK_LOCAL_MEM_FENCE);
        {
            __global half* hb = hbuf + (((size_t)i_h * GDN_NV + i_nv) * max_chunks + c) * GDN_V_BLOCK * DK;
            for (int e = ltid * GDN_VEC; e < GDN_V_BLOCK * DK; e += GDN_LANES * GDN_VEC) {
                const int n = e / DK;
                const int d = e % DK;
                vstore4(vload4(0, (const __local uint*)(hT + (size_t)n * GDN_DK_PAD + d)), 0, (__global uint*)(hb + (size_t)n * DK + d));
            }
        }
        FUNC(gdn_stage_gc)(gc2, gate, i_h, HV, tok0, tok0 + len, ltid);

        // 2. accu = w h
        float8 accu[GDN_NJ];
#pragma unroll
        for (int j = 0; j < GDN_NJ; j++)
            accu[j] = 0.0f;
#pragma unroll
        for (int ks = 0; ks < GDN_DK_TILES; ks++) {
            const short8 af = as_short8(vload8(0, wg + (((size_t)sg * GDN_DK_TILES + ks) * SUBGROUP_SIZE + lid) * GDN_TM));
#pragma unroll
            for (int j = 0; j < GDN_NJ; j++)
                accu[j] = FUNC(gdn_dpas)(af, FUNC(gdn_b_frag_l)(hT, GDN_DK_PAD, lid + GDN_TN * j, ks), accu[j]);
        }

        // 3. vnew = u - accu: raw copy (tokens past the chunk are 0) for fwd_o, decayed copy for the first part
        int hi = len;  // end of the current part: the next snapshot position inside the chunk, else the chunk end
        if (done + len >= next_snapshot)
            hi = next_snapshot - done;
        {
            const float g_hi = gc2[hi - 1];
            float dec[GDN_TM];
#pragma unroll
            for (int e = 0; e < GDN_TM; e++)
                dec[e] = (sg * GDN_TM + e < hi) ? exp2(g_hi - gc2[sg * GDN_TM + e]) : 0.0f;
#pragma unroll
            for (int j = 0; j < GDN_NJ; j++) {
                const half8 uu = vload8(0, ug + (size_t)(lid + GDN_TN * j) * GDN_CHUNK + sg * GDN_TM);
                const float8 vnew = convert_float8(uu) - accu[j];
                float8 vmask, uadj;
#pragma unroll
                for (int e = 0; e < GDN_TM; e++) {
                    vmask[e] = (sg * GDN_TM + e < len) ? vnew[e] : 0.0f;
                    uadj[e] = vnew[e] * dec[e];
                }
                vstore8(convert_half8(vmask), 0, vnT + (size_t)(lid + GDN_TN * j) * GDN_CHUNK_PAD + sg * GDN_TM);
                vstore8(convert_half8(uadj), 0, uadjT + (size_t)(lid + GDN_TN * j) * GDN_CHUNK_PAD + sg * GDN_TM);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        {
            __global half* vb = vnewT + (((size_t)i_h * GDN_NV + i_nv) * max_chunks + c) * GDN_V_BLOCK * GDN_CHUNK;
            for (int e = ltid * GDN_VEC; e < GDN_V_BLOCK * GDN_CHUNK; e += GDN_LANES * GDN_VEC) {
                const int n = e / GDN_CHUNK;
                const int t = e % GDN_CHUNK;
                vstore4(vload4(0, (const __local uint*)(vnT + (size_t)n * GDN_CHUNK_PAD + t)), 0, (__global uint*)(vb + (size_t)n * GDN_CHUNK + t));
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // 4. state update per part [lo, hi) of the chunk
        int lo = 0;
        while (true) {
            const float g_hi = gc2[hi - 1];
            const float g_lo = (lo > 0) ? gc2[lo - 1] : 0.0f;
            if (lo > 0) {  // decayed vnew of a later part, rebuilt from the raw copy
                float dec[GDN_TM];
#pragma unroll
                for (int e = 0; e < GDN_TM; e++) {
                    const int tok = sg * GDN_TM + e;
                    dec[e] = (tok >= lo && tok < hi) ? exp2(g_hi - gc2[tok]) : 0.0f;
                }
#pragma unroll
                for (int j = 0; j < GDN_NJ; j++) {
                    float8 vv = convert_float8(vload8(0, vnT + (size_t)(lid + GDN_TN * j) * GDN_CHUNK_PAD + sg * GDN_TM));
#pragma unroll
                    for (int e = 0; e < GDN_TM; e++)
                        vv[e] *= dec[e];
                    vstore8(convert_half8(vv), 0, uadjT + (size_t)(lid + GDN_TN * j) * GDN_CHUNK_PAD + sg * GDN_TM);
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            const float decay = exp2(g_hi - g_lo);
#pragma unroll
            for (int b = 0; b < GDN_H_TILES; b++)
#pragma unroll
                for (int j = 0; j < GDN_NJ; j++)
                    hacc[b][j] *= decay;
            // Tokens outside [lo, hi) have zero weight: skip the K tiles that hold none of them (uniform branch).
            const int ks_lo = lo / GDN_TK;
            const int ks_hi = (hi + GDN_TK - 1) / GDN_TK;
#pragma unroll
            for (int ks = 0; ks < GDN_TOK_TILES; ks++) {
                if (ks < ks_lo || ks >= ks_hi)
                    continue;
                const __global half* kp = kg + (size_t)(ks * GDN_TK + lid) * DK + row0;
                short8 af[GDN_H_TILES];
#pragma unroll
                for (int b = 0; b < GDN_H_TILES; b++)
                    af[b] = as_short8(vload8(0, kp + b * GDN_TM));
#pragma unroll
                for (int j = 0; j < GDN_NJ; j++) {
                    const int8 bf = FUNC(gdn_b_frag_l)(uadjT, GDN_CHUNK_PAD, lid + GDN_TN * j, ks);
#pragma unroll
                    for (int b = 0; b < GDN_H_TILES; b++)
                        hacc[b][j] = FUNC(gdn_dpas)(af[b], bf, hacc[b][j]);
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            // snapshot: at every interval-th token of the sequence and after its last token
            const bool at_interval = (done + hi == next_snapshot);
            if (at_interval || done + hi == seq_len) {
                if (block_begin + slot < block_end) {
                    __global INPUT3_TYPE* s = state_table + (size_t)block_indices[block_begin + slot] * HV * DV * DK + state_head;
#if INPUT3_TYPE_SIZE == 2
                    // Stage through SLM (hT is free) and copy out with the lanes along k: each lane writes 16 contiguous
                    // bytes of one state row, so the stores coalesce (direct stores touch one cache line per lane).
#pragma unroll
                    for (int b = 0; b < GDN_H_TILES; b++)
#pragma unroll
                        for (int j = 0; j < GDN_NJ; j++)
                            vstore8(convert_half8(hacc[b][j]), 0, hT + (size_t)(lid + GDN_TN * j) * GDN_DK_PAD + row0 + b * GDN_TM);
                    barrier(CLK_LOCAL_MEM_FENCE);
                    for (int e = ltid * GDN_VEC; e < GDN_V_BLOCK * DK; e += GDN_LANES * GDN_VEC) {
                        const int n = e / DK;
                        const int d = e % DK;
                        vstore4(vload4(0, (const __local uint*)(hT + (size_t)n * GDN_DK_PAD + d)), 0, (__global uint*)(s + (size_t)n * DK + d));
                    }
                    barrier(CLK_LOCAL_MEM_FENCE);
#else
#pragma unroll
                    for (int b = 0; b < GDN_H_TILES; b++)
#pragma unroll
                        for (int j = 0; j < GDN_NJ; j++)
                            GDN_STORE_STATE8(hacc[b][j], s + (size_t)(lid + GDN_TN * j) * DK + row0 + b * GDN_TM);
#endif
                }
                slot++;
            }
            if (at_interval)
                next_snapshot += interval;

            lo = hi;
            if (lo >= len)
                break;
            hi = (done + len >= next_snapshot) ? next_snapshot - done : len;
        }
    }
}
#endif  // GDN_STAGE == 3

// ---------------------------------------------------------------------------
// Stage 4: fwd_o. Grid (max_chunks, HV); one workgroup covers all v-blocks. o = (qn e2g) h + tril((qn kn^T) decay)
// vnew; the masked qn kn^T accumulator is already in the A layout of the second product.
// ---------------------------------------------------------------------------
#if GDN_STAGE == 4
REQD_SUB_GROUP_SIZE(SUBGROUP_SIZE)
KERNEL(gdn_paged_dpas_fwd_o)(__global const half* qn,
                             __global const half* kn,
                             __global const INPUT4_TYPE* gate,
                             __global const half* hbuf,
                             __global const half* vnewT,
                             __global OUTPUT_TYPE* output,  // [tokens, HV, DV]
                             __global const INPUT6_TYPE* subsequence_begins,
                             int num_sequences,
                             int max_chunks) {
    const int DK = K_HEAD_DIM;
    const int HV = V_HEAD_NUM;
    const int DV = V_HEAD_DIM;
    const int GROUP = V_HEAD_NUM / K_HEAD_NUM;
    const int i_c = get_group_id(0);
    const int i_h = get_global_id(1);
    const int i_hk = i_h / GROUP;
    const int lid = get_sub_group_local_id();
    const int sg = get_sub_group_id();
    const int ltid = sg * SUBGROUP_SIZE + lid;

    int seq, tok0, len;
    FUNC(gdn_locate_chunk)(subsequence_begins, num_sequences, i_c, &seq, &tok0, &len);
    if (seq < 0)
        return;

    __local float gc2[GDN_GC_SIZE] GDN_ALIGNED;
    FUNC(gdn_stage_gc)(gc2, gate, i_h, HV, tok0, tok0 + len, ltid);

    const __global half* qg = qn + ((size_t)i_hk * max_chunks + i_c) * GDN_CHUNK * DK;
    const __global half* kg = kn + ((size_t)i_hk * max_chunks + i_c) * GDN_CHUNK * DK;

    const int r0 = sg * GDN_TM;
    float8 acco[GDN_NV][GDN_NJ], accA[GDN_TOK_TILES];
#pragma unroll
    for (int nv = 0; nv < GDN_NV; nv++)
#pragma unroll
        for (int j = 0; j < GDN_NJ; j++)
            acco[nv][j] = 0.0f;
#pragma unroll
    for (int j = 0; j < GDN_TOK_TILES; j++)
        accA[j] = 0.0f;

#pragma unroll
    for (int ks = 0; ks < GDN_DK_TILES; ks++) {
        const short8 af = FUNC(gdn_a_frag_g)(qg, DK, r0, ks, lid);
#pragma unroll
        for (int nv = 0; nv < GDN_NV; nv++) {
            const __global half* hg = hbuf + (((size_t)i_h * GDN_NV + nv) * max_chunks + i_c) * GDN_V_BLOCK * DK;
#pragma unroll
            for (int j = 0; j < GDN_NJ; j++)
                acco[nv][j] = FUNC(gdn_dpas)(af, FUNC(gdn_b_frag_g)(hg, DK, lid + GDN_TN * j, ks), acco[nv][j]);
        }
#pragma unroll
        for (int j = 0; j < GDN_TOK_TILES; j++)
            accA[j] = FUNC(gdn_dpas)(af, FUNC(gdn_b_frag_g)(kg, DK, lid + GDN_TN * j, ks), accA[j]);
    }

#pragma unroll
    for (int nv = 0; nv < GDN_NV; nv++)
#pragma unroll
        for (int j = 0; j < GDN_NJ; j++)
#pragma unroll
            for (int e = 0; e < GDN_TM; e++)
                acco[nv][j][e] = (r0 + e < len) ? acco[nv][j][e] * exp2(gc2[r0 + e]) : 0.0f;
#pragma unroll
    for (int j = 0; j < GDN_TOK_TILES; j++) {
        const int jj = GDN_TN * j + lid;
        const float gj = gc2[jj];
#pragma unroll
        for (int e = 0; e < GDN_TM; e++) {
            const bool keep = (r0 + e < len) && (jj <= r0 + e);
            accA[j][e] = keep ? accA[j][e] * exp2(gc2[r0 + e] - gj) : 0.0f;
        }
    }

#pragma unroll
    for (int ks = 0; ks < GDN_TOK_TILES; ks++) {
        short8 af;
#pragma unroll
        for (int e = 0; e < GDN_TM; e++)
            af[e] = as_short(convert_half(accA[ks][e]));
#pragma unroll
        for (int nv = 0; nv < GDN_NV; nv++) {
            const __global half* vg = vnewT + (((size_t)i_h * GDN_NV + nv) * max_chunks + i_c) * GDN_V_BLOCK * GDN_CHUNK;
#pragma unroll
            for (int j = 0; j < GDN_NJ; j++)
                acco[nv][j] = FUNC(gdn_dpas)(af, FUNC(gdn_b_frag_g)(vg, GDN_CHUNK, lid + GDN_TN * j, ks), acco[nv][j]);
        }
    }

    __global OUTPUT_TYPE* o = output + ((size_t)tok0 * HV + i_h) * DV;
#pragma unroll
    for (int nv = 0; nv < GDN_NV; nv++)
#pragma unroll
        for (int j = 0; j < GDN_NJ; j++) {
            const int col = nv * GDN_V_BLOCK + GDN_TN * j + lid;
#pragma unroll
            for (int e = 0; e < GDN_TM; e++)
                if (r0 + e < len)
                    o[(size_t)(r0 + e) * HV * DV + col] = (OUTPUT_TYPE)acco[nv][j][e];
        }
}
#endif  // GDN_STAGE == 4
