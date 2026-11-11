// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// Chunked Gated DeltaNet on DPAS/XMX (equivalent of fla chunk_gated_delta_rule), five stages selected by GDN_STAGE:
//   0 prep       l2-normalize q/k (q also scaled by SCALE_FACTOR) into f16 scratch qn/kn
//   1 kkt_solve  A = strict_lower(beta * decay * kn kn^T) and X = (I + A)^-1, per chunk
//   2 wy         w = X (kn beta e2g), u = X (v beta), per chunk
//   3 fwd_h      serial loop over chunks: state h, vnew = u - w h, h = decay h + kn^T vnew
//   4 fwd_o      o = (qn e2g) h + tril((qn kn^T) decay) vnew, per chunk

// Defined by the host (gated_delta_net_opt.cpp): GDN_STAGE, K_HEAD_NUM, V_HEAD_NUM, K_HEAD_DIM, V_HEAD_DIM,
// SUBGROUP_SIZE (16), GDN_CHUNK (tokens per chunk), GDN_V_BLOCK (value columns per fwd_h workgroup),
// GDN_SUBGROUPS (subgroups per workgroup), GDN_SLM_PAD (SLM row padding in halves), GDN_PREFETCH_CHUNKS,
// FUSE_QK_L2NORM, Q_L2_NORM_EPS, K_L2_NORM_EPS, SCALE_FACTOR, OUTPUT_STATE.

// Numerics: gate math in the log2 domain on fp32; GEMM operands f16, accumulators fp32. The gate must be <= 0
// (log decay) so every exp2 factor is <= 1.

#include "gated_delta_net_dpas_common.cl"

// ---------------------------------------------------------------------------
// Stage 0: prep. Grid (chunks, B*HK). Subgroup sg handles chunk tokens [sg * 8, sg * 8 + 8), one at a time;
// the lanes split the key dim, so the l2 norm is a subgroup reduction. Padded tail rows are written as zeros.
// ---------------------------------------------------------------------------
#if GDN_STAGE == 0
REQD_SUB_GROUP_SIZE(SUBGROUP_SIZE)
KERNEL(gdn_opt_prep)(__global INPUT0_TYPE* q,  // [B, T, HK, DK], head offset = hk + query_offset
                     __global INPUT1_TYPE* k,  // [B, T, HK, DK], head offset = hk + key_offset
                     __global half* qn,        // out: [B, HK, NT * CHUNK, DK] f16
                     __global half* kn,        // out: [B, HK, NT * CHUNK, DK] f16
                     int seq_len,
                     int query_offset,
                     int key_offset,
                     int q_t_stride,
                     int k_t_stride) {
    const int DK = K_HEAD_DIM;
    const int HK = K_HEAD_NUM;
    const int NT = (seq_len + GDN_CHUNK - 1) / GDN_CHUNK;
    const int i_t = get_group_id(0);
    const int i_bhk = get_global_id(1);
    const int i_b = i_bhk / HK;
    const int i_hk = i_bhk % HK;
    const int lid = get_sub_group_local_id();
    const int sg = get_sub_group_id();

    const size_t q_base = (size_t)i_b * seq_len * q_t_stride + (size_t)(i_hk + query_offset) * DK;
    const size_t k_base = (size_t)i_b * seq_len * k_t_stride + (size_t)(i_hk + key_offset) * DK;

    for (int tt = 0; tt < GDN_CHUNK / GDN_SUBGROUPS; tt++) {
        const int tok = sg * (GDN_CHUNK / GDN_SUBGROUPS) + tt;
        const int t = i_t * GDN_CHUNK + tok;
        const bool tvalid = (t < seq_len);
        const size_t dst_row = ((size_t)(i_b * HK + i_hk) * NT * GDN_CHUNK) + (size_t)(i_t * GDN_CHUNK + tok);

        // q: l2norm * SCALE_FACTOR and k: l2norm -> f16. Both rows are loaded before either norm is reduced, so the block reads
        // (and the two independent subgroup reductions) overlap.
        float8 qv[GDN_PREP_VECS], kv[GDN_PREP_VECS];
#pragma unroll
        for (int c = 0; c < GDN_PREP_VECS; c++) {
            qv[c] = tvalid ? FUNC(gdn_block_read_q)(q + q_base + (size_t)t * q_t_stride, c) : (float8)(0.0f);
            kv[c] = tvalid ? FUNC(gdn_block_read_k)(k + k_base + (size_t)t * k_t_stride, c) : (float8)(0.0f);
        }
        float qs = 0.0f, ks = 0.0f;
#pragma unroll
        for (int c = 0; c < GDN_PREP_VECS; c++) {
            qs += FUNC(gdn_dot8)(qv[c], qv[c]);
            ks += FUNC(gdn_dot8)(kv[c], kv[c]);
        }
#if FUSE_QK_L2NORM
        qs = sub_group_reduce_add(qs);
        ks = sub_group_reduce_add(ks);
#endif
        float q_scale = SCALE_FACTOR;
        float k_scale = 1.0f;
#if FUSE_QK_L2NORM
        q_scale *= rsqrt(qs + Q_L2_NORM_EPS);
        k_scale = rsqrt(ks + K_L2_NORM_EPS);
#endif
#pragma unroll
        for (int c = 0; c < GDN_PREP_VECS; c++) {
            FUNC(gdn_block_write_row)(qn + dst_row * DK, c, qv[c] * q_scale);
            FUNC(gdn_block_write_row)(kn + dst_row * DK, c, kv[c] * k_scale);
        }
    }
}
#endif  // GDN_STAGE == 0

// ---------------------------------------------------------------------------
// Stage 1: kkt_solve. Grid (chunks, B*HV). A[i][j] = (i > j) ? beta_i exp2(gc_i - gc_j) (kn_i . kn_j) : 0 stays in
// SLM; X = (I + A)^-1 comes from per-block forward substitution plus a block merge (fla chunk_fwd). X is written
// to xinv in A-fragment-major order: [sg][ks][lane][8 rows], element e of a lane = X[8 sg + e][16 ks + lane].
// ---------------------------------------------------------------------------
#if GDN_STAGE == 1
REQD_SUB_GROUP_SIZE(SUBGROUP_SIZE)
KERNEL(gdn_opt_kkt_solve)(__global const half* kn,          // [B, HK, NT * CHUNK, DK] f16
                          __global const INPUT4_TYPE* g,    // [B, T, HV]
                          __global const INPUT5_TYPE* beta,  // [B, T, HV]
                          __global half* xinv,              // out: [B, HV, NT, CHUNK, CHUNK] f16, fragment-major
                          int seq_len) {
    const int DK = K_HEAD_DIM;
    const int HV = V_HEAD_NUM;
    const int HK = K_HEAD_NUM;
    const int GROUP = HV / HK;
    const int NT = (seq_len + GDN_CHUNK - 1) / GDN_CHUNK;
    const int i_t = get_group_id(0);
    const int i_bhv = get_global_id(1);
    const int i_b = i_bhv / HV;
    const int i_h = i_bhv % HV;
    const int i_hk = i_h / GROUP;
    const int lid = get_sub_group_local_id();
    const int sg = get_sub_group_id();
    const int ltid = sg * SUBGROUP_SIZE + lid;

    // x doubles as the kn tile while the GEMM reads it.
    __local float x[GDN_CHUNK * GDN_SOLVE_LD] GDN_ALIGNED;
    __local half* tile = (__local half*)x;
    __local float a[GDN_CHUNK * GDN_SOLVE_LD] GDN_ALIGNED;
    __local float tmp[GDN_NTMP * GDN_BLK * GDN_BLK_LD] GDN_ALIGNED;
    __local float gc2[GDN_GC_SIZE] GDN_ALIGNED;
    __local float sbeta[GDN_CHUNK] GDN_ALIGNED;
    const int LD = GDN_SOLVE_LD;

    FUNC(gdn_stage_tile_g2l)(kn + ((size_t)(i_b * HK + i_hk) * NT + i_t) * GDN_CHUNK * DK, DK, tile, GDN_DK_PAD, GDN_CHUNK, DK, ltid);
    FUNC(gdn_stage_gc)(gc2, g, ((size_t)i_b * seq_len) * HV + i_h, HV, i_t * GDN_CHUNK, seq_len, ltid);
    if (ltid < GDN_CHUNK) {
        const int t = i_t * GDN_CHUNK + ltid;
        sbeta[ltid] = (t < seq_len) ? convert_float(beta[((size_t)i_b * seq_len + t) * HV + i_h]) : 0.0f;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // kn kn^T: subgroup sg owns rows [8 sg, 8 sg + 8); lane covers columns 16 c + lid.
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

    // beta scales row i; the column factor is 1 (fla: b_A * b_b[:, None]).
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

    FUNC(gdn_store_x_fragments)(x, xinv + ((size_t)(i_bhv * NT + i_t)) * GDN_CHUNK * GDN_CHUNK, sg, lid);
}
#endif  // GDN_STAGE == 1

// ---------------------------------------------------------------------------
// Stage 2: wy. Grid (chunks, B*HV). w = X (kn beta e2g) is stored fragment-major [sg][ks][lane][8 rows]; per v-block
// u = X (v beta) is stored transposed as uT[n][token]. The B operands (n-major) are transposed through SLM.
// ---------------------------------------------------------------------------
#if GDN_STAGE == 2
REQD_SUB_GROUP_SIZE(SUBGROUP_SIZE)
KERNEL(gdn_opt_wy)(__global const half* kn,            // [B, HK, NT * CHUNK, DK] f16
                   __global const INPUT2_TYPE* v,      // [B, T, HV, V]
                   __global const INPUT4_TYPE* g,
                   __global const INPUT5_TYPE* beta,
                   __global const half* xinv,          // fragment-major, see stage 1
                   __global half* w,                   // out: [B, HV, NT, CHUNK, DK], fragment-major
                   __global half* uT,                  // out: [B, HV, NV, NT, V_BLOCK, CHUNK]
                   int seq_len,
                   int value_offset,
                   int v_t_stride) {
    const int DK = K_HEAD_DIM;
    const int HV = V_HEAD_NUM;
    const int HK = K_HEAD_NUM;
    const int VLEN = V_HEAD_DIM;
    const int GROUP = HV / HK;
    const int NT = (seq_len + GDN_CHUNK - 1) / GDN_CHUNK;
    const int i_t = get_group_id(0);
    const int i_bhv = get_global_id(1);
    const int i_b = i_bhv / HV;
    const int i_h = i_bhv % HV;
    const int i_hk = i_h / GROUP;
    const int lid = get_sub_group_local_id();
    const int sg = get_sub_group_id();
    const int ltid = sg * SUBGROUP_SIZE + lid;
    const int t0 = i_t * GDN_CHUNK;

    __local half tile[GDN_CHUNK * GDN_DK_PAD] GDN_ALIGNED;    // kn rows
    __local half t_kbT[K_HEAD_DIM * GDN_CHUNK_PAD] GDN_ALIGNED;  // [dk][token], n-major
    __local half t_v[GDN_V_BLOCK * GDN_CHUNK_PAD] GDN_ALIGNED;   // [v col][token], n-major
    __local float gc2[GDN_GC_SIZE] GDN_ALIGNED;
    __local float sbeta[GDN_CHUNK] GDN_ALIGNED;

    FUNC(gdn_stage_gc)(gc2, g, ((size_t)i_b * seq_len) * HV + i_h, HV, t0, seq_len, ltid);
    if (ltid < GDN_CHUNK) {
        const int t = t0 + ltid;
        sbeta[ltid] = (t < seq_len) ? convert_float(beta[((size_t)i_b * seq_len + t) * HV + i_h]) : 0.0f;
    }
    FUNC(gdn_stage_tile_g2l)(kn + (((size_t)(i_b * HK + i_hk) * NT + i_t) * GDN_CHUNK) * DK, DK, tile, GDN_DK_PAD, GDN_CHUNK, DK, ltid);
    barrier(CLK_LOCAL_MEM_FENCE);

    // kbT[dim][tok] = f16(kn[tok][dim] beta[tok] exp2(gc[tok])); consecutive lanes take consecutive tokens.
    for (int e = ltid; e < GDN_CHUNK * DK / GDN_VEC; e += GDN_LANES) {
        const int tok = e % GDN_CHUNK;
        const int d0 = (e / GDN_CHUNK) * GDN_VEC;
        const float scale = sbeta[tok] * exp2(gc2[tok]);
        const half8 kk = vload8(0, tile + (size_t)tok * GDN_DK_PAD + d0);
#pragma unroll
        for (int i = 0; i < GDN_VEC; i++)
            t_kbT[(size_t)(d0 + i) * GDN_CHUNK_PAD + tok] = convert_half(convert_float(kk[i]) * scale);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const __global half* x = xinv + ((size_t)i_bhv * NT + i_t) * GDN_CHUNK * GDN_CHUNK;
    const int r0 = sg * GDN_TM;

    // w = X kbT^T: M = tokens (8 rows per subgroup), N = DK, K = tokens.
    {
        float8 accw[GDN_DK_TILES];
#pragma unroll
        for (int c = 0; c < GDN_DK_TILES; c++)
            accw[c] = 0.0f;
#pragma unroll
        for (int ks = 0; ks < GDN_TOK_TILES; ks++) {
            const short8 af = as_short8(BLOCK_READN(half, GDN_TM, x, ((size_t)sg * GDN_TOK_TILES + ks) * SUBGROUP_SIZE * GDN_TM));
#pragma unroll
            for (int c = 0; c < GDN_DK_TILES; c++)
                accw[c] = FUNC(gdn_dpas)(af, FUNC(gdn_b_frag_l)(t_kbT, GDN_CHUNK_PAD, GDN_TN * c + lid, ks), accw[c]);
        }
        __global half* wout = w + ((size_t)i_bhv * NT + i_t) * GDN_CHUNK * DK;
#pragma unroll
        for (int c = 0; c < GDN_DK_TILES; c++)
            BLOCK_WRITEN(half, GDN_TM, wout, ((size_t)sg * GDN_DK_TILES + c) * SUBGROUP_SIZE * GDN_TM, convert_half8(accw[c]));
    }

    // u = X vb per v-block, written as uT[n][token].
    for (int nv = 0; nv < GDN_NV; nv++) {
        const __global INPUT2_TYPE* vsrc = v + ((size_t)i_b * seq_len + t0) * v_t_stride + (size_t)(i_h + value_offset) * VLEN + nv * GDN_V_BLOCK;
        for (int e = ltid; e < GDN_CHUNK * GDN_V_BLOCK / GDN_VEC; e += GDN_LANES) {
            const int tok = e % GDN_CHUNK;
            const int c0 = (e / GDN_CHUNK) * GDN_VEC;
            float8 vv = (float8)(0.0f);
            if (t0 + tok < seq_len)
                vv = FUNC(gdn_load8_v)(vsrc + (size_t)tok * v_t_stride + c0);
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
            const short8 af = as_short8(BLOCK_READN(half, GDN_TM, x, ((size_t)sg * GDN_TOK_TILES + ks) * SUBGROUP_SIZE * GDN_TM));
#pragma unroll
            for (int c = 0; c < GDN_NJ; c++)
                accu[c] = FUNC(gdn_dpas)(af, FUNC(gdn_b_frag_l)(t_v, GDN_CHUNK_PAD, GDN_TN * c + lid, ks), accu[c]);
        }
        __global half* uout = uT + ((((size_t)i_bhv * GDN_NV + nv) * NT + i_t) * GDN_V_BLOCK) * GDN_CHUNK;
#pragma unroll
        for (int c = 0; c < GDN_NJ; c++)
            BLOCK_WRITEN(half, GDN_TM, uout, ((size_t)sg * GDN_NJ + c) * SUBGROUP_SIZE * GDN_TM, convert_half8(accu[c]));
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}
#endif  // GDN_STAGE == 2

// ---------------------------------------------------------------------------
// Stage 3: fwd_h. Grid (B * HV * NV workgroups). One workgroup owns a (b, head, v-block) and loops over the chunks.
// Per chunk: publish h (f16) to SLM and hbuf; vnew = u - w h; h = exp2(g_last) h + kn^T (vnew exp2(g_last - gc)).
// h lives in fp32 registers: hacc[tile][j] covers state rows sg * DK_PER_SG + 8 tile + e, value columns 16 j + lane.
// ---------------------------------------------------------------------------
#if GDN_STAGE == 3
REQD_SUB_GROUP_SIZE(SUBGROUP_SIZE)
KERNEL(gdn_opt_fwd_h)(__global const half* kn,               // [B, HK, NT * CHUNK, DK]
                      __global const half* w,                // fragment-major, see stage 2
                      __global const half* uT,               // [B, HV, NV, NT, V_BLOCK, CHUNK]
                      __global const INPUT4_TYPE* g,         // [B, T, HV]
                      __global INPUT3_TYPE* initial_state,   // [B, HV, DK, V]
                      __global half* hbuf,                   // out: [B, HV, NV, NT, V_BLOCK, DK]
                      __global half* vnewT,                  // out: [B, HV, NV, NT, V_BLOCK, CHUNK]
#if OUTPUT_STATE
                      __global OUTPUT1_TYPE* output_state,   // [B, HV, DK, V]
#endif
                      int seq_len) {
    const int DK = K_HEAD_DIM;
    const int HV = V_HEAD_NUM;
    const int HK = K_HEAD_NUM;
    const int VLEN = V_HEAD_DIM;
    const int GROUP = HV / HK;
    const int NT = (seq_len + GDN_CHUNK - 1) / GDN_CHUNK;
    const int gid = get_group_id(0);
    const int i_b = gid / (HV * GDN_NV);
    const int i_h = (gid % (HV * GDN_NV)) / GDN_NV;
    const int i_nv = gid % GDN_NV;
    const int i_hk = i_h / GROUP;
    const int lid = get_sub_group_local_id();
    const int sg = get_sub_group_id();
    const int ltid = sg * SUBGROUP_SIZE + lid;

    __local half hT[GDN_V_BLOCK * GDN_DK_PAD] GDN_ALIGNED;       // h^T [v col][dk], n-major
    __local half uadjT[GDN_V_BLOCK * GDN_CHUNK_PAD] GDN_ALIGNED;  // decayed vnew^T [v col][token]
    __local half vnT[GDN_V_BLOCK * GDN_CHUNK_PAD] GDN_ALIGNED;    // raw vnew^T [v col][token]
    __local float gc2[GDN_GC_SIZE] GDN_ALIGNED;

    const int row0 = sg * GDN_DK_PER_SG;  // first state row of this subgroup
    float8 hacc[GDN_H_TILES][GDN_NJ];
    {
        const __global INPUT3_TYPE* s = initial_state + ((size_t)(i_b * HV + i_h)) * DK * VLEN;
#pragma unroll
        for (int b = 0; b < GDN_H_TILES; b++)
#pragma unroll
            for (int j = 0; j < GDN_NJ; j++)
#pragma unroll
                for (int e = 0; e < GDN_TM; e++)
                    hacc[b][j][e] = convert_float(s[(size_t)(row0 + b * GDN_TM + e) * VLEN + (i_nv * GDN_V_BLOCK + lid + GDN_TN * j)]);
    }

    const size_t g_base0 = ((size_t)i_b * seq_len) * HV + i_h;
    const size_t w_chunk0 = (size_t)(i_b * HV + i_h) * NT;
    const size_t k_chunk0 = (size_t)(i_b * HK + i_hk) * NT;
    const size_t u_chunk0 = ((size_t)(i_b * HV + i_h) * GDN_NV + i_nv) * NT;

    for (int c = 0; c < NT; c++) {
        const int t0 = c * GDN_CHUNK;
        // Prefetch a later chunk into cache; holding it in registers would spill (SIMD16: a short8 is 4 GRFs).
        {
            const int cp = min(c + GDN_PREFETCH_CHUNKS, NT - 1);
            prefetch(w + ((w_chunk0 + cp) * GDN_CHUNK) * DK + ltid * GDN_PF_W, GDN_PF_W);
            prefetch(kn + ((k_chunk0 + cp) * GDN_CHUNK) * DK + ltid * GDN_PF_W, GDN_PF_W);
            prefetch(uT + ((u_chunk0 + cp) * GDN_V_BLOCK) * GDN_CHUNK + ltid * GDN_PF_U, GDN_PF_U);
        }
        const __global half* kg = kn + ((k_chunk0 + c) * GDN_CHUNK) * DK;
        const __global half* wg = w + ((w_chunk0 + c) * GDN_CHUNK) * DK;
        const __global half* ug = uT + ((u_chunk0 + c) * GDN_V_BLOCK) * GDN_CHUNK;

        // 1. publish h as f16 n-major, then copy it to hbuf
#pragma unroll
        for (int b = 0; b < GDN_H_TILES; b++)
#pragma unroll
            for (int j = 0; j < GDN_NJ; j++)
                vstore8(convert_half8(hacc[b][j]), 0, hT + (size_t)(lid + GDN_TN * j) * GDN_DK_PAD + row0 + b * GDN_TM);
        barrier(CLK_LOCAL_MEM_FENCE);
        {
            __global half* hb = hbuf + ((((size_t)(i_b * HV + i_h)) * GDN_NV + i_nv) * NT + c) * GDN_V_BLOCK * DK;
            for (int e = ltid * GDN_VEC; e < GDN_V_BLOCK * DK; e += GDN_LANES * GDN_VEC) {
                const int n = e / DK;
                const int d = e % DK;
                vstore4(vload4(0, (const __local uint*)(hT + (size_t)n * GDN_DK_PAD + d)), 0, (__global uint*)(hb + (size_t)n * DK + d));
            }
        }
        FUNC(gdn_stage_gc)(gc2, g, g_base0, HV, t0, seq_len, ltid);

        // 2. accu = w h: M = tokens (8 rows per subgroup), N = v-block, K = DK
        float8 accu[GDN_NJ];
#pragma unroll
        for (int j = 0; j < GDN_NJ; j++)
            accu[j] = 0.0f;
#pragma unroll
        for (int ks = 0; ks < GDN_DK_TILES; ks++) {
            const short8 af = as_short8(BLOCK_READN(half, GDN_TM, wg, ((size_t)sg * GDN_DK_TILES + ks) * SUBGROUP_SIZE * GDN_TM));
#pragma unroll
            for (int j = 0; j < GDN_NJ; j++)
                accu[j] = FUNC(gdn_dpas)(af, FUNC(gdn_b_frag_l)(hT, GDN_DK_PAD, lid + GDN_TN * j, ks), accu[j]);
        }

        // 3. vnew = u - accu (raw copy for fwd_o) and the decayed copy for the state update; padded tokens are 0
        const float g_last = gc2[GDN_CHUNK - 1];
        float dec[GDN_TM];
#pragma unroll
        for (int e = 0; e < GDN_TM; e++)
            dec[e] = ((t0 + sg * GDN_TM + e) < seq_len) ? exp2(g_last - gc2[sg * GDN_TM + e]) : 0.0f;
#pragma unroll
        for (int j = 0; j < GDN_NJ; j++) {
            const half8 uu = BLOCK_READN(half, GDN_TM, ug, ((size_t)sg * GDN_NJ + j) * SUBGROUP_SIZE * GDN_TM);
            const float8 vnew = convert_float8(uu) - accu[j];
            float8 vmask, uadj;
#pragma unroll
            for (int e = 0; e < GDN_TM; e++) {
                vmask[e] = ((t0 + sg * GDN_TM + e) < seq_len) ? vnew[e] : 0.0f;
                uadj[e] = vnew[e] * dec[e];
            }
            vstore8(convert_half8(vmask), 0, vnT + (size_t)(lid + GDN_TN * j) * GDN_CHUNK_PAD + sg * GDN_TM);
            vstore8(convert_half8(uadj), 0, uadjT + (size_t)(lid + GDN_TN * j) * GDN_CHUNK_PAD + sg * GDN_TM);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        {
            __global half* vb = vnewT + ((((size_t)(i_b * HV + i_h)) * GDN_NV + i_nv) * NT + c) * GDN_V_BLOCK * GDN_CHUNK;
            for (int e = ltid * GDN_VEC; e < GDN_V_BLOCK * GDN_CHUNK; e += GDN_LANES * GDN_VEC) {
                const int n = e / GDN_CHUNK;
                const int t = e % GDN_CHUNK;
                vstore4(vload4(0, (const __local uint*)(vnT + (size_t)n * GDN_CHUNK_PAD + t)), 0, (__global uint*)(vb + (size_t)n * GDN_CHUNK + t));
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // 4. h = exp2(g_last) h + kn^T uadj: A = kn^T (element = dk row, lane = token, one 16-byte load per lane)
        const float decay = exp2(g_last);
#pragma unroll
        for (int b = 0; b < GDN_H_TILES; b++)
#pragma unroll
            for (int j = 0; j < GDN_NJ; j++)
                hacc[b][j] *= decay;
#pragma unroll
        for (int ks = 0; ks < GDN_TOK_TILES; ks++) {
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
    }

#if OUTPUT_STATE
    __global OUTPUT1_TYPE* s_out = output_state + ((size_t)(i_b * HV + i_h)) * DK * VLEN;
#else
    __global INPUT3_TYPE* s_inp = initial_state + ((size_t)(i_b * HV + i_h)) * DK * VLEN;
#endif
#pragma unroll
    for (int b = 0; b < GDN_H_TILES; b++)
#pragma unroll
        for (int j = 0; j < GDN_NJ; j++)
#pragma unroll
            for (int e = 0; e < GDN_TM; e++) {
                const size_t idx = (size_t)(row0 + b * GDN_TM + e) * VLEN + (i_nv * GDN_V_BLOCK + lid + GDN_TN * j);
#if OUTPUT_STATE
                s_out[idx] = (OUTPUT1_TYPE)hacc[b][j][e];
#else
                s_inp[idx] = (INPUT3_TYPE)hacc[b][j][e];
#endif
            }
}
#endif  // GDN_STAGE == 3

// ---------------------------------------------------------------------------
// Stage 4: fwd_o. Grid (chunks, B*HV); one workgroup covers all v-blocks, so qn and kn are read once.
// o = (qn exp2(gc)) h + tril((qn kn^T) exp2(gc_i - gc_j)) vnew. The masked qn kn^T accumulator is already in the A
// layout of the second product (element = token row, lane = token column).
// ---------------------------------------------------------------------------
#if GDN_STAGE == 4
REQD_SUB_GROUP_SIZE(SUBGROUP_SIZE)
KERNEL(gdn_opt_fwd_o)(__global const half* qn,           // [B, HK, NT * CHUNK, DK]
                      __global const half* kn,           // [B, HK, NT * CHUNK, DK]
                      __global const INPUT4_TYPE* g,
                      __global const half* hbuf,         // [B, HV, NV, NT, V_BLOCK, DK]
                      __global const half* vnewT,        // [B, HV, NV, NT, V_BLOCK, CHUNK]
                      __global OUTPUT_TYPE* output,      // [B, T, HV, V]
                      int seq_len,
                      int out_row_stride) {              // HV * V
    const int DK = K_HEAD_DIM;
    const int HV = V_HEAD_NUM;
    const int HK = K_HEAD_NUM;
    const int VLEN = V_HEAD_DIM;
    const int GROUP = HV / HK;
    const int NT = (seq_len + GDN_CHUNK - 1) / GDN_CHUNK;
    const int i_t = get_group_id(0);
    const int i_bh = get_global_id(1);
    const int i_b = i_bh / HV;
    const int i_h = i_bh % HV;
    const int i_hk = i_h / GROUP;
    const int lid = get_sub_group_local_id();
    const int sg = get_sub_group_id();
    const int ltid = sg * SUBGROUP_SIZE + lid;
    const int t0 = i_t * GDN_CHUNK;

    __local float gc2[GDN_GC_SIZE] GDN_ALIGNED;
    FUNC(gdn_stage_gc)(gc2, g, ((size_t)i_b * seq_len) * HV + i_h, HV, t0, seq_len, ltid);

    const size_t qk_row_base = ((size_t)(i_b * HK + i_hk) * NT + i_t) * GDN_CHUNK;
    const __global half* qg = qn + qk_row_base * DK;
    const __global half* kg = kn + qk_row_base * DK;
    const size_t bh_nv0 = (size_t)(i_b * HV + i_h) * GDN_NV;  // v-block 0 of this head in hbuf / vnewT

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

    // qn h and qn kn^T share the A operand (K = DK)
#pragma unroll
    for (int ks = 0; ks < GDN_DK_TILES; ks++) {
        const short8 af = FUNC(gdn_a_frag_g)(qg, DK, r0, ks, lid);
#pragma unroll
        for (int nv = 0; nv < GDN_NV; nv++) {
            const __global half* hg = hbuf + ((bh_nv0 + nv) * NT + i_t) * GDN_V_BLOCK * DK;
#pragma unroll
            for (int j = 0; j < GDN_NJ; j++)
                acco[nv][j] = FUNC(gdn_dpas)(af, FUNC(gdn_b_frag_g)(hg, DK, lid + GDN_TN * j, ks), acco[nv][j]);
        }
#pragma unroll
        for (int j = 0; j < GDN_TOK_TILES; j++)
            accA[j] = FUNC(gdn_dpas)(af, FUNC(gdn_b_frag_g)(kg, DK, lid + GDN_TN * j, ks), accA[j]);
    }

    // inter-chunk term decays with exp2(gc_i); intra-chunk term is masked to j <= i and decays with exp2(gc_i - gc_j)
#pragma unroll
    for (int nv = 0; nv < GDN_NV; nv++)
#pragma unroll
        for (int j = 0; j < GDN_NJ; j++)
#pragma unroll
            for (int e = 0; e < GDN_TM; e++)
                acco[nv][j][e] = ((t0 + r0 + e) < seq_len) ? acco[nv][j][e] * exp2(gc2[r0 + e]) : 0.0f;
#pragma unroll
    for (int j = 0; j < GDN_TOK_TILES; j++) {
        const int jj = GDN_TN * j + lid;
        const float gj = gc2[jj];
#pragma unroll
        for (int e = 0; e < GDN_TM; e++) {
            const bool keep = ((t0 + r0 + e) < seq_len) && (jj <= r0 + e);
            accA[j][e] = keep ? accA[j][e] * exp2(gc2[r0 + e] - gj) : 0.0f;
        }
    }

    // o += A_masked vnew (K = tokens)
#pragma unroll
    for (int ks = 0; ks < GDN_TOK_TILES; ks++) {
        short8 af;
#pragma unroll
        for (int e = 0; e < GDN_TM; e++)
            af[e] = as_short(convert_half(accA[ks][e]));
#pragma unroll
        for (int nv = 0; nv < GDN_NV; nv++) {
            const __global half* vg = vnewT + ((bh_nv0 + nv) * NT + i_t) * GDN_V_BLOCK * GDN_CHUNK;
#pragma unroll
            for (int j = 0; j < GDN_NJ; j++)
                acco[nv][j] = FUNC(gdn_dpas)(af, FUNC(gdn_b_frag_g)(vg, GDN_CHUNK, lid + GDN_TN * j, ks), acco[nv][j]);
        }
    }

    __global OUTPUT_TYPE* o = output + ((size_t)i_b * seq_len + t0) * out_row_stride + (size_t)i_h * VLEN;
#pragma unroll
    for (int nv = 0; nv < GDN_NV; nv++)
#pragma unroll
        for (int j = 0; j < GDN_NJ; j++) {
            const int col = nv * GDN_V_BLOCK + GDN_TN * j + lid;
#pragma unroll
            for (int e = 0; e < GDN_TM; e++)
                if (t0 + r0 + e < seq_len)  // uniform per subgroup: 16 consecutive columns of a row are one block write
                    BLOCK_WRITEN(OUTPUT_TYPE, 1, o + (size_t)(r0 + e) * out_row_stride, nv * GDN_V_BLOCK + GDN_TN * j, (OUTPUT_TYPE)acco[nv][j][e]);
        }
}
#endif  // GDN_STAGE == 4

// ---------------------------------------------------------------------------
// Stage 5: kkt_wy = kkt_solve + wy in one workgroup. X stays in SLM (f16 A-operand tile) instead of going through xinv, and
// the SLM regions are reused: x holds the kn tile and then X (fp32), a holds A and then kbT, rv holds the solve temporaries
// and then the v tile.
// ---------------------------------------------------------------------------
#if GDN_STAGE == 5
#if (K_HEAD_DIM * GDN_CHUNK_PAD * 2 > GDN_CHUNK * GDN_SOLVE_LD * 4)
#    error "kbT must fit in the A buffer it aliases"
#endif
#if (GDN_NTMP * GDN_BLK * GDN_BLK_LD * 4 > GDN_V_BLOCK * GDN_CHUNK_PAD * 2)
#    error "the solve temporaries must fit in the v tile they alias"
#endif
REQD_SUB_GROUP_SIZE(SUBGROUP_SIZE)
KERNEL(gdn_opt_kkt_wy)(__global const half* kn,            // [B, HK, NT * CHUNK, DK] f16
                       __global const INPUT2_TYPE* v,      // [B, T, HV, V]
                       __global const INPUT4_TYPE* g,
                       __global const INPUT5_TYPE* beta,
                       __global half* w,                   // out: [B, HV, NT, CHUNK, DK], fragment-major
                       __global half* uT,                  // out: [B, HV, NV, NT, V_BLOCK, CHUNK]
                       int seq_len,
                       int value_offset,
                       int v_t_stride) {
    const int DK = K_HEAD_DIM;
    const int HV = V_HEAD_NUM;
    const int HK = K_HEAD_NUM;
    const int VLEN = V_HEAD_DIM;
    const int GROUP = HV / HK;
    const int NT = (seq_len + GDN_CHUNK - 1) / GDN_CHUNK;
    const int i_t = get_group_id(0);
    const int i_bhv = get_global_id(1);
    const int i_b = i_bhv / HV;
    const int i_h = i_bhv % HV;
    const int i_hk = i_h / GROUP;
    const int lid = get_sub_group_local_id();
    const int sg = get_sub_group_id();
    const int ltid = sg * SUBGROUP_SIZE + lid;
    const int t0 = i_t * GDN_CHUNK;
    const int LD = GDN_SOLVE_LD;

    __local float x[GDN_CHUNK * GDN_SOLVE_LD] GDN_ALIGNED;  // kn tile, then X
    __local half* tile = (__local half*)x;
    __local float a[GDN_CHUNK * GDN_SOLVE_LD] GDN_ALIGNED;  // A, then kbT[dk][token]
    __local half* t_kbT = (__local half*)a;
    __local half rv[GDN_V_BLOCK * GDN_CHUNK_PAD] GDN_ALIGNED;  // solve temporaries, then v tile [v col][token]
    __local float* tmp = (__local float*)rv;
    __local half xh[GDN_CHUNK * GDN_CHUNK_PAD] GDN_ALIGNED;  // X as f16 A operand
    __local float gc2[GDN_GC_SIZE] GDN_ALIGNED;
    __local float sbeta[GDN_CHUNK] GDN_ALIGNED;

    FUNC(gdn_stage_tile_g2l)(kn + ((size_t)(i_b * HK + i_hk) * NT + i_t) * GDN_CHUNK * DK, DK, tile, GDN_DK_PAD, GDN_CHUNK, DK, ltid);
    FUNC(gdn_stage_gc)(gc2, g, ((size_t)i_b * seq_len) * HV + i_h, HV, t0, seq_len, ltid);
    if (ltid < GDN_CHUNK) {
        const int t = t0 + ltid;
        sbeta[ltid] = (t < seq_len) ? convert_float(beta[((size_t)i_b * seq_len + t) * HV + i_h]) : 0.0f;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // A = strict_lower(beta decay kn kn^T)
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

    FUNC(gdn_solve_slm)(a, x, tmp, sg, lid);  // ends with a barrier: a and tmp are dead

    // X -> f16 tile; kbT[dim][tok] = f16(kn[tok][dim] beta[tok] exp2(gc[tok])) straight from global kn (just read, cached).
    for (int e = ltid; e < GDN_CHUNK * GDN_CHUNK; e += GDN_LANES) {
        const int r = e / GDN_CHUNK;
        const int c = e % GDN_CHUNK;
        xh[(size_t)r * GDN_CHUNK_PAD + c] = convert_half(x[(size_t)r * LD + c]);
    }
    const __global half* kn_chunk = kn + ((size_t)(i_b * HK + i_hk) * NT + i_t) * GDN_CHUNK * DK;
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

    // w = X kbT^T: M = tokens (8 rows per subgroup), N = DK, K = tokens.
    {
        float8 accw[GDN_DK_TILES];
#pragma unroll
        for (int c = 0; c < GDN_DK_TILES; c++)
            accw[c] = 0.0f;
#pragma unroll
        for (int ks = 0; ks < GDN_TOK_TILES; ks++) {
            const short8 af = FUNC(gdn_a_frag_l)(xh, GDN_CHUNK_PAD, r0, ks, lid);
#pragma unroll
            for (int c = 0; c < GDN_DK_TILES; c++)
                accw[c] = FUNC(gdn_dpas)(af, FUNC(gdn_b_frag_l)(t_kbT, GDN_CHUNK_PAD, GDN_TN * c + lid, ks), accw[c]);
        }
        __global half* wout = w + ((size_t)i_bhv * NT + i_t) * GDN_CHUNK * DK;
#pragma unroll
        for (int c = 0; c < GDN_DK_TILES; c++)
            BLOCK_WRITEN(half, GDN_TM, wout, ((size_t)sg * GDN_DK_TILES + c) * SUBGROUP_SIZE * GDN_TM, convert_half8(accw[c]));
    }

    // u = X vb per v-block, written as uT[n][token].
    for (int nv = 0; nv < GDN_NV; nv++) {
        const __global INPUT2_TYPE* vsrc = v + ((size_t)i_b * seq_len + t0) * v_t_stride + (size_t)(i_h + value_offset) * VLEN + nv * GDN_V_BLOCK;
        for (int e = ltid; e < GDN_CHUNK * GDN_V_BLOCK / GDN_VEC; e += GDN_LANES) {
            const int tok = e % GDN_CHUNK;
            const int c0 = (e / GDN_CHUNK) * GDN_VEC;
            float8 vv = (float8)(0.0f);
            if (t0 + tok < seq_len)
                vv = FUNC(gdn_load8_v)(vsrc + (size_t)tok * v_t_stride + c0);
            const float bt = sbeta[tok];
#pragma unroll
            for (int i = 0; i < GDN_VEC; i++)
                rv[(size_t)(c0 + i) * GDN_CHUNK_PAD + tok] = convert_half(vv[i] * bt);
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        float8 accu[GDN_NJ];
#pragma unroll
        for (int c = 0; c < GDN_NJ; c++)
            accu[c] = 0.0f;
#pragma unroll
        for (int ks = 0; ks < GDN_TOK_TILES; ks++) {
            const short8 af = FUNC(gdn_a_frag_l)(xh, GDN_CHUNK_PAD, r0, ks, lid);
#pragma unroll
            for (int c = 0; c < GDN_NJ; c++)
                accu[c] = FUNC(gdn_dpas)(af, FUNC(gdn_b_frag_l)(rv, GDN_CHUNK_PAD, GDN_TN * c + lid, ks), accu[c]);
        }
        __global half* uout = uT + ((((size_t)i_bhv * GDN_NV + nv) * NT + i_t) * GDN_V_BLOCK) * GDN_CHUNK;
#pragma unroll
        for (int c = 0; c < GDN_NJ; c++)
            BLOCK_WRITEN(half, GDN_TM, uout, ((size_t)sg * GDN_NJ + c) * SUBGROUP_SIZE * GDN_TM, convert_half8(accu[c]));
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}
#endif  // GDN_STAGE == 5
