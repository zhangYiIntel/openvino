// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// Shared building blocks of the chunked Gated DeltaNet DPAS kernels (gated_delta_net_opt.cl and
// paged_gated_delta_net_dpas.cl): derived constants, DPAS fragment helpers, chunk gate prefix sums and the
// 64x64 triangular solve. All sizes derive from the JIT constants defined by the host:
// K_HEAD_NUM, V_HEAD_NUM, K_HEAD_DIM, V_HEAD_DIM, SUBGROUP_SIZE (16), GDN_CHUNK (tokens per chunk),
// GDN_V_BLOCK (value columns per fwd_h workgroup), GDN_SUBGROUPS (subgroups per workgroup), GDN_SLM_PAD
// (SLM row padding in halves), GDN_PREFETCH_CHUNKS.

#include "include/batch_headers/common.cl"
#include "include/batch_headers/fetch_data.cl"
#include "include/batch_headers/sub_group_block_read.cl"
#include "include/batch_headers/sub_group_block_write.cl"
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

// q / k row access with sub group block reads and writes (a block message moves 16 lanes x GDN_VEC elements at once, which
// streams better than per-lane 16-byte vector accesses). Chunk c of a row is read as one block: lane l holds elements
// (c * GDN_VEC + j) * SUBGROUP_SIZE + l, j = 0..7. Reading and writing a row with the same mapping keeps it row-major, and
// the mapping is irrelevant for the norm. The row pointer must be 16-byte aligned and the subgroup uniform.
inline float8 FUNC(gdn_block_read_q)(const __global INPUT0_TYPE* row, int c) {
    return convert_float8(BLOCK_READN(INPUT0_TYPE, GDN_VEC, row, c * SUBGROUP_SIZE * GDN_VEC));
}
inline float8 FUNC(gdn_block_read_k)(const __global INPUT1_TYPE* row, int c) {
    return convert_float8(BLOCK_READN(INPUT1_TYPE, GDN_VEC, row, c * SUBGROUP_SIZE * GDN_VEC));
}
inline void FUNC(gdn_block_write_row)(__global half* row, int c, float8 v) {
    BLOCK_WRITEN(half, GDN_VEC, row, c * SUBGROUP_SIZE * GDN_VEC, convert_half8(v));
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
// Triangular solve. On entry a holds A (strictly lower, CHUNK x CHUNK floats, stride GDN_SOLVE_LD) and x is
// zeroed; on return x holds X = (I + A)^-1 (fla chunk_fwd: per-block forward substitution, then block merge).
// ---------------------------------------------------------------------------
inline void FUNC(gdn_solve_slm)(__local float* a, __local float* x, __local float* tmp, int sg, int lid) {
    const int LD = GDN_SOLVE_LD;
    const int LDT = GDN_BLK_LD;
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
}

// Writes X as f16 in A-fragment-major block order: [sg][ks][element e][lane], element e of lane l = X[8 sg + e][16 ks + l]. A
// sub group block write of a fragment (8 elements per lane) produces exactly this order and a block read of 8 per lane returns the
// DPAS A fragment (element = row, lane = k) directly. out points at the chunk's X block.
inline void FUNC(gdn_store_x_fragments)(const __local float* x, __global half* out, int sg, int lid) {
    const int LD = GDN_SOLVE_LD;
#pragma unroll
    for (int ks = 0; ks < GDN_TOK_TILES; ks++) {
        half8 f;
#pragma unroll
        for (int e = 0; e < GDN_TM; e++)
            f[e] = convert_half(x[(size_t)(sg * GDN_TM + e) * LD + ks * GDN_TK + lid]);
        BLOCK_WRITEN(half, GDN_TM, out, ((size_t)sg * GDN_TOK_TILES + ks) * SUBGROUP_SIZE * GDN_TM, f);
    }
}
