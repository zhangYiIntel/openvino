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

#include "include/batch_headers/common.cl"
#include "include/batch_headers/fetch_data.cl"
#include "include/batch_headers/sub_group_shuffle.cl"

#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable

// ---------------------------------------------------------------------------
// Derived constants and layout checks
// ---------------------------------------------------------------------------
// DPAS tile: D[TM x TN] += A[TM x TK] * B[TK x TN], f16 in, fp32 accumulate, TN = subgroup size.
#define GDN_TM 8
#define GDN_TK 16
#define GDN_TN SUBGROUP_SIZE
#define GDN_VEC 8  // halves per 16-byte vector access
#define GDN_ALIGNED __attribute__((aligned(16)))  // SLM arrays accessed with vector loads / stores
#define GDN_NTMP 4  // temporary blocks of the solve merge
#define GDN_LANES (SUBGROUP_SIZE * GDN_SUBGROUPS)
#define GDN_LOG2E 1.4426950408889634f

#define GDN_TOK_TILES (GDN_CHUNK / GDN_TN)              // 16-token column tiles of a chunk (= K steps over tokens)
#define GDN_DK_TILES (K_HEAD_DIM / GDN_TK)              // K steps over the key dim
#define GDN_NJ (GDN_V_BLOCK / GDN_TN)                   // value column tiles per v-block
#define GDN_NV (V_HEAD_DIM / GDN_V_BLOCK)               // v-blocks per head
#define GDN_DK_PER_SG (K_HEAD_DIM / GDN_SUBGROUPS)      // state rows owned by a subgroup in fwd_h
#define GDN_H_TILES (GDN_DK_PER_SG / GDN_TM)            // row tiles of the state per subgroup
#define GDN_DK_PAD (K_HEAD_DIM + GDN_SLM_PAD)           // SLM row stride (halves) of [*][DK] tiles
#define GDN_CHUNK_PAD (GDN_CHUNK + GDN_SLM_PAD)         // SLM row stride (halves) of [*][CHUNK] tiles
#define GDN_SOLVE_LD (GDN_CHUNK + GDN_SLM_PAD)          // SLM row stride (floats) of the CHUNK x CHUNK matrices
#define GDN_BLK SUBGROUP_SIZE                           // solve block size
#define GDN_NBLK (GDN_CHUNK / GDN_BLK)                  // diagonal blocks in the solve
#define GDN_BLK_LD (GDN_BLK + 1)                        // padded stride of a solve temporary block
#define GDN_GC_SIZE (GDN_CHUNK + GDN_CHUNK / SUBGROUP_SIZE)  // gate prefix sums + one total per subgroup
#define GDN_PREP_VECS (K_HEAD_DIM / (SUBGROUP_SIZE * GDN_VEC))  // 8-element q/k vectors per lane in prep
#define GDN_PF_W (GDN_CHUNK * K_HEAD_DIM / GDN_LANES)   // elements of w / kn prefetched per lane
#define GDN_PF_U (GDN_V_BLOCK * GDN_CHUNK / GDN_LANES)  // elements of uT prefetched per lane

#if (SUBGROUP_SIZE != 16)
#    error "gated_delta_net_opt requires subgroup size 16"
#endif
#if (GDN_CHUNK / GDN_SUBGROUPS != GDN_TM)
#    error "each subgroup must own exactly one TM-row tile of a chunk"
#endif
#if (GDN_CHUNK != 64 || GDN_NBLK != 4)
#    error "the solve is written for 4 diagonal blocks of 16 (chunk 64)"
#endif
#if (K_HEAD_DIM % (SUBGROUP_SIZE * GDN_VEC) != 0 || K_HEAD_DIM % GDN_SUBGROUPS != 0 || GDN_DK_PER_SG % GDN_TM != 0)
#    error "K_HEAD_DIM must tile over lanes and subgroups"
#endif
#if (GDN_V_BLOCK % GDN_TN != 0 || V_HEAD_DIM % GDN_V_BLOCK != 0)
#    error "V_HEAD_DIM must be a multiple of GDN_V_BLOCK, itself a multiple of 16"
#endif
#if (GDN_CHUNK * GDN_DK_PAD * 2 > GDN_CHUNK * GDN_SOLVE_LD * 4)
#    error "the kn tile must fit in the X buffer it aliases"
#endif

// ---------------------------------------------------------------------------
// DPAS helpers
// ---------------------------------------------------------------------------
inline float8 FUNC(gdn_dpas)(short8 a, int8 b, float8 acc) {
    return intel_sub_group_f16_f16_matrix_mad_k16(a, b, acc);
}

// A fragment from a row-major tile: element j = row r0 + j, lane = K index ks * TK + lid.
inline short8 FUNC(gdn_a_frag_l)(const __local half* A, int lda, int r0, int ks, int lid) {
    short8 a;
    const __local short* p = (const __local short*)(A + (size_t)r0 * lda + ks * GDN_TK + lid);
#pragma unroll
    for (int j = 0; j < GDN_TM; j++)
        a[j] = p[(size_t)j * lda];
    return a;
}

inline short8 FUNC(gdn_a_frag_g)(const __global half* A, size_t lda, int r0, int ks, int lid) {
    short8 a;
    const __global short* p = (const __global short*)(A + (size_t)r0 * lda + ks * GDN_TK + lid);
#pragma unroll
    for (int j = 0; j < GDN_TM; j++)
        a[j] = p[(size_t)j * lda];
    return a;
}

// B fragment from an n-major tile B[n][k]: lane = column n, TK contiguous halves (4-byte aligned loads).
inline int8 FUNC(gdn_b_frag_l)(const __local half* B, int ldb, int n, int ks) {
    return as_int8(vload8(0, (const __local int*)(B + (size_t)n * ldb + ks * GDN_TK)));
}

inline int8 FUNC(gdn_b_frag_g)(const __global half* B, size_t ldb, int n, int ks) {
    return as_int8(vload8(0, (const __global int*)(B + (size_t)n * ldb + ks * GDN_TK)));
}

// Copy a [rows][cols] f16 global tile (row stride lds) into a padded SLM tile, one 16-byte vector per access.
inline void FUNC(gdn_stage_tile_g2l)(const __global half* src, size_t lds, __local half* dst, int ldd, int rows, int cols, int ltid) {
    for (int e = ltid * GDN_VEC; e < rows * cols; e += GDN_LANES * GDN_VEC) {
        const int r = e / cols;
        const int c = e % cols;
        vstore4(vload4(0, (const __global uint*)(src + (size_t)r * lds + c)), 0, (__local uint*)(dst + (size_t)r * ldd + c));
    }
}

inline float FUNC(gdn_dot8)(float8 a, float8 b) {
    float s = a.s0 * b.s0;
    s = fma(a.s1, b.s1, s);
    s = fma(a.s2, b.s2, s);
    s = fma(a.s3, b.s3, s);
    s = fma(a.s4, b.s4, s);
    s = fma(a.s5, b.s5, s);
    s = fma(a.s6, b.s6, s);
    s = fma(a.s7, b.s7, s);
    return s;
}

inline float8 FUNC(gdn_load8_q)(const __global INPUT0_TYPE* p) {
#if INPUT0_TYPE_SIZE == 2
    return convert_float8(vload8(0, (const __global half*)p));
#else
    float8 v;
#pragma unroll
    for (int i = 0; i < GDN_VEC; i++)
        v[i] = convert_float(p[i]);
    return v;
#endif
}

inline float8 FUNC(gdn_load8_k)(const __global INPUT1_TYPE* p) {
#if INPUT1_TYPE_SIZE == 2
    return convert_float8(vload8(0, (const __global half*)p));
#else
    float8 v;
#pragma unroll
    for (int i = 0; i < GDN_VEC; i++)
        v[i] = convert_float(p[i]);
    return v;
#endif
}

inline float8 FUNC(gdn_load8_v)(const __global INPUT2_TYPE* p) {
#if INPUT2_TYPE_SIZE == 2
    return convert_float8(vload8(0, (const __global half*)p));
#else
    float8 v;
#pragma unroll
    for (int i = 0; i < GDN_VEC; i++)
        v[i] = convert_float(p[i]);
    return v;
#endif
}

// ---------------------------------------------------------------------------
// Chunk gate prefix sums: gc[t] = sum_{j<=t} g[j] * log2(e). Tail tokens contribute 0.
// ---------------------------------------------------------------------------
// Gate value of token ltid (0 for lanes >= GDN_CHUNK and for tail tokens), in the log2 domain.
inline float FUNC(gdn_load_gate)(const __global INPUT4_TYPE* g, size_t g_base, int hv_stride, int t0, int seq_len, int ltid) {
    float gv = 0.0f;
    if (ltid < GDN_CHUNK && t0 + ltid < seq_len)
        gv = convert_float(g[g_base + (size_t)(t0 + ltid) * hv_stride]) * GDN_LOG2E;
    return gv;
}

// Inclusive scan: every subgroup scans its tokens, then adds the totals of the earlier subgroups.
// slm_gc holds GDN_GC_SIZE floats; it is readable by all lanes on return.
inline void FUNC(gdn_scan_gc)(__local float* slm_gc, float gv, int ltid) {
    const float sc = sub_group_scan_inclusive_add(gv);
    if (ltid < GDN_CHUNK)
        slm_gc[ltid] = sc;
    if (ltid < GDN_CHUNK && (ltid % SUBGROUP_SIZE) == SUBGROUP_SIZE - 1)
        slm_gc[GDN_CHUNK + ltid / SUBGROUP_SIZE] = sc;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (ltid < GDN_CHUNK) {
        float off = 0.0f;
        for (int k = 0; k < ltid / SUBGROUP_SIZE; k++)
            off += slm_gc[GDN_CHUNK + k];
        slm_gc[ltid] = sc + off;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}

inline void FUNC(gdn_stage_gc)(__local float* slm_gc, const __global INPUT4_TYPE* g, size_t g_base, int hv_stride, int t0, int seq_len, int ltid) {
    FUNC(gdn_scan_gc)(slm_gc, FUNC(gdn_load_gate)(g, g_base, hv_stride, t0, seq_len, ltid), ltid);
}

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

        // q: l2norm * SCALE_FACTOR -> f16
        {
            float qs = 0.0f;
            float8 qv[GDN_PREP_VECS];
#pragma unroll
            for (int c = 0; c < GDN_PREP_VECS; c++) {
                const int d = (c * SUBGROUP_SIZE + lid) * GDN_VEC;
                qv[c] = tvalid ? FUNC(gdn_load8_q)(q + q_base + (size_t)t * q_t_stride + d) : (float8)(0.0f);
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
        // k: l2norm -> f16
        {
            float ks = 0.0f;
            float8 kv[GDN_PREP_VECS];
#pragma unroll
            for (int c = 0; c < GDN_PREP_VECS; c++) {
                const int d = (c * SUBGROUP_SIZE + lid) * GDN_VEC;
                kv[c] = tvalid ? FUNC(gdn_load8_k)(k + k_base + (size_t)t * k_t_stride + d) : (float8)(0.0f);
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
// Stage 1 helpers: 16x16 fp32 products in SLM, one lane per output row.
// ---------------------------------------------------------------------------
inline void FUNC(gdn_mma_blk)(const __local float* A, int lda, const __local float* B, int ldb, const __local float* D, int ldd,
                              __local float* C, int ldc, int lane, float sign) {
    float c[GDN_BLK];
#pragma unroll
    for (int j = 0; j < GDN_BLK; j++)
        c[j] = (D != 0) ? D[(size_t)lane * ldd + j] : 0.0f;
    for (int k = 0; k < GDN_BLK; k++) {
        const float av = A[(size_t)lane * lda + k];
#pragma unroll
        for (int j = 0; j < GDN_BLK; j++)
            c[j] = fma(av, B[(size_t)k * ldb + j], c[j]);
    }
#pragma unroll
    for (int j = 0; j < GDN_BLK; j++)
        C[(size_t)lane * ldc + j] = sign * c[j];
}

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
    const int LDT = GDN_BLK_LD;

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

    // Phase A: invert the diagonal blocks, one subgroup each, lane j holds column j of X_mm.
    // xcol[i] = -a[i][j] (i > j); for i >= 2: xcol[i] += sum_{k<i} (-a[i][k]) xcol[k]; finally xcol[j] += 1.
    if (sg < GDN_NBLK) {
        const int m = sg;
        const int j = lid;
        float xcol[GDN_BLK];
#pragma unroll
        for (int i = 0; i < GDN_BLK; i++)
            xcol[i] = (i > j) ? -a[(size_t)(m * GDN_BLK + i) * LD + (m * GDN_BLK + j)] : 0.0f;
        for (int i = 2; i < GDN_BLK; i++) {
            const float a_orig = xcol[i];  // lane k holds -A[i][k]
            float a_i = a_orig;
            for (int k = 1; k < i; k++)
                a_i = fma(sub_group_broadcast(a_orig, k), xcol[k], a_i);
            xcol[i] = a_i;
        }
        xcol[j] += 1.0f;
#pragma unroll
        for (int i = 0; i < GDN_BLK; i++)
            x[(size_t)(m * GDN_BLK + i) * LD + (m * GDN_BLK + j)] = xcol[i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Phase B: off-diagonal blocks, X_mn = -X_mm (A_mn + sum_{n<q<m} A_mq X_qn) X_nn, as 16x16 FMA products with
    // barriers outside the subgroup branches:
    //   X10 = -(X11 A10) X00, X21 = -(X22 A21) X11, X32 = -(X33 A32) X22
    //   X20 = -X22 (A20 X00 + A21 X10), X31 = -X33 (A31 X11 + A32 X21), X30 = -X33 (A30 X00 + A31 X10 + A32 X20)
#define GDN_A_BLK(mm, nn) (a + ((size_t)(mm)*GDN_BLK) * LD + (nn)*GDN_BLK)
#define GDN_X_BLK(mm, nn) (x + ((size_t)(mm)*GDN_BLK) * LD + (nn)*GDN_BLK)
#define GDN_T_BLK(id) (tmp + ((size_t)(id)*GDN_BLK) * LDT)
#define GDN_NO_ADD ((const __local float*)0)
    if (lid < GDN_BLK) {
        // step 1: three independent products plus the first term of X20
        if (sg == 0)
            FUNC(gdn_mma_blk)(GDN_X_BLK(1, 1), LD, GDN_A_BLK(1, 0), LD, GDN_NO_ADD, 0, GDN_T_BLK(0), LDT, lid, 1.0f);
        else if (sg == 1)
            FUNC(gdn_mma_blk)(GDN_X_BLK(2, 2), LD, GDN_A_BLK(2, 1), LD, GDN_NO_ADD, 0, GDN_T_BLK(1), LDT, lid, 1.0f);
        else if (sg == 2)
            FUNC(gdn_mma_blk)(GDN_X_BLK(3, 3), LD, GDN_A_BLK(3, 2), LD, GDN_NO_ADD, 0, GDN_T_BLK(2), LDT, lid, 1.0f);
        else if (sg == 3)
            FUNC(gdn_mma_blk)(GDN_A_BLK(2, 0), LD, GDN_X_BLK(0, 0), LD, GDN_NO_ADD, 0, GDN_T_BLK(3), LDT, lid, 1.0f);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid < GDN_BLK) {  // step 2: X10, X21, X32
        if (sg == 0)
            FUNC(gdn_mma_blk)(GDN_T_BLK(0), LDT, GDN_X_BLK(0, 0), LD, GDN_NO_ADD, 0, GDN_X_BLK(1, 0), LD, lid, -1.0f);
        else if (sg == 1)
            FUNC(gdn_mma_blk)(GDN_T_BLK(1), LDT, GDN_X_BLK(1, 1), LD, GDN_NO_ADD, 0, GDN_X_BLK(2, 1), LD, lid, -1.0f);
        else if (sg == 2)
            FUNC(gdn_mma_blk)(GDN_T_BLK(2), LDT, GDN_X_BLK(2, 2), LD, GDN_NO_ADD, 0, GDN_X_BLK(3, 2), LD, lid, -1.0f);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid < GDN_BLK) {  // step 3: second term of X20, start X31 / X30
        if (sg == 3)
            FUNC(gdn_mma_blk)(GDN_A_BLK(2, 1), LD, GDN_X_BLK(1, 0), LD, GDN_T_BLK(3), LDT, GDN_T_BLK(3), LDT, lid, 1.0f);
        else if (sg == 0)
            FUNC(gdn_mma_blk)(GDN_A_BLK(3, 1), LD, GDN_X_BLK(1, 1), LD, GDN_NO_ADD, 0, GDN_T_BLK(0), LDT, lid, 1.0f);
        else if (sg == 1)
            FUNC(gdn_mma_blk)(GDN_A_BLK(3, 0), LD, GDN_X_BLK(0, 0), LD, GDN_NO_ADD, 0, GDN_T_BLK(1), LDT, lid, 1.0f);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid < GDN_BLK) {  // step 4: X20; accumulate X31 / X30
        if (sg == 3)
            FUNC(gdn_mma_blk)(GDN_X_BLK(2, 2), LD, GDN_T_BLK(3), LDT, GDN_NO_ADD, 0, GDN_X_BLK(2, 0), LD, lid, -1.0f);
        else if (sg == 0)
            FUNC(gdn_mma_blk)(GDN_A_BLK(3, 2), LD, GDN_X_BLK(2, 1), LD, GDN_T_BLK(0), LDT, GDN_T_BLK(0), LDT, lid, 1.0f);
        else if (sg == 1)
            FUNC(gdn_mma_blk)(GDN_A_BLK(3, 1), LD, GDN_X_BLK(1, 0), LD, GDN_T_BLK(1), LDT, GDN_T_BLK(1), LDT, lid, 1.0f);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid < GDN_BLK) {  // step 5: X31; last term of X30 (needs X20)
        if (sg == 0)
            FUNC(gdn_mma_blk)(GDN_X_BLK(3, 3), LD, GDN_T_BLK(0), LDT, GDN_NO_ADD, 0, GDN_X_BLK(3, 1), LD, lid, -1.0f);
        else if (sg == 1)
            FUNC(gdn_mma_blk)(GDN_A_BLK(3, 2), LD, GDN_X_BLK(2, 0), LD, GDN_T_BLK(1), LDT, GDN_T_BLK(1), LDT, lid, 1.0f);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid < GDN_BLK && sg == 1)  // step 6: X30
        FUNC(gdn_mma_blk)(GDN_X_BLK(3, 3), LD, GDN_T_BLK(1), LDT, GDN_NO_ADD, 0, GDN_X_BLK(3, 0), LD, lid, -1.0f);
    barrier(CLK_LOCAL_MEM_FENCE);
#undef GDN_A_BLK
#undef GDN_X_BLK
#undef GDN_T_BLK
#undef GDN_NO_ADD

    __global half* out = xinv + ((size_t)(i_bhv * NT + i_t)) * GDN_CHUNK * GDN_CHUNK;
#pragma unroll
    for (int ks = 0; ks < GDN_TOK_TILES; ks++) {
        half8 f;
#pragma unroll
        for (int e = 0; e < GDN_TM; e++)
            f[e] = convert_half(x[(size_t)(sg * GDN_TM + e) * LD + ks * GDN_TK + lid]);
        vstore8(f, 0, out + (((size_t)sg * GDN_TOK_TILES + ks) * SUBGROUP_SIZE + lid) * GDN_TM);
    }
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
            const short8 af = as_short8(vload8(0, x + (((size_t)sg * GDN_TOK_TILES + ks) * SUBGROUP_SIZE + lid) * GDN_TM));
#pragma unroll
            for (int c = 0; c < GDN_DK_TILES; c++)
                accw[c] = FUNC(gdn_dpas)(af, FUNC(gdn_b_frag_l)(t_kbT, GDN_CHUNK_PAD, GDN_TN * c + lid, ks), accw[c]);
        }
        __global half* wout = w + ((size_t)i_bhv * NT + i_t) * GDN_CHUNK * DK;
#pragma unroll
        for (int c = 0; c < GDN_DK_TILES; c++)
            vstore8(convert_half8(accw[c]), 0, wout + (((size_t)sg * GDN_DK_TILES + c) * SUBGROUP_SIZE + lid) * GDN_TM);
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
            const short8 af = as_short8(vload8(0, x + (((size_t)sg * GDN_TOK_TILES + ks) * SUBGROUP_SIZE + lid) * GDN_TM));
#pragma unroll
            for (int c = 0; c < GDN_NJ; c++)
                accu[c] = FUNC(gdn_dpas)(af, FUNC(gdn_b_frag_l)(t_v, GDN_CHUNK_PAD, GDN_TN * c + lid, ks), accu[c]);
        }
        __global half* uout = uT + ((((size_t)i_bhv * GDN_NV + nv) * NT + i_t) * GDN_V_BLOCK) * GDN_CHUNK;
#pragma unroll
        for (int c = 0; c < GDN_NJ; c++)
            vstore8(convert_half8(accu[c]), 0, uout + (size_t)(GDN_TN * c + lid) * GDN_CHUNK + r0);
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
            const short8 af = as_short8(vload8(0, wg + (((size_t)sg * GDN_DK_TILES + ks) * SUBGROUP_SIZE + lid) * GDN_TM));
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
            const half8 uu = vload8(0, ug + (size_t)(lid + GDN_TN * j) * GDN_CHUNK + sg * GDN_TM);
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
                if (t0 + r0 + e < seq_len)
                    o[(size_t)(r0 + e) * out_row_stride + col] = (OUTPUT_TYPE)acco[nv][j][e];
        }
}
#endif  // GDN_STAGE == 4
