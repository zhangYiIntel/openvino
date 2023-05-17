
#pragma once

//#include "tensor2D.hpp"
#include <memory>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <functional>

#ifdef _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

#define FORCE_INLINE

#define rndup(x, n) (((x + n - 1)/n)*n)

// https://stackoverflow.com/questions/570669/checking-if-a-double-or-float-is-nan-in-c/57770634#57770634
static inline uint32_t load_ieee754_rep(float a) {
    uint32_t r;
    static_assert(sizeof r == sizeof a, "Unexpected sizes.");
    std::memcpy(&r, &a, sizeof a); // Generates movd instruction.
    return r;
}
constexpr uint32_t inf_float_shl1 = UINT32_C(0xff000000);
// The shift left removes the sign bit. The exponent moves into the topmost bits,
// so that plain unsigned comparison is enough.
static inline bool isnan2(float a)     { return load_ieee754_rep(a) << 1  > inf_float_shl1; }
static inline bool isinf2(float a)     { return load_ieee754_rep(a) << 1 == inf_float_shl1; }
static inline bool isfinite2(float a)  { return load_ieee754_rep(a) << 1  < inf_float_shl1; }

template<typename T>
struct tensor2D {
    int dims[2] = {0};
    std::shared_ptr<T> data;
    int64_t capacity = 0;
    int stride = 0;
    bool force_compact = false;
    int padded_dim1 = 0;

    tensor2D() = default;

    operator bool() {
        return dims[0] * dims[1] > 0;
    }

    tensor2D(int d0, int d1, bool _force_compact = false) {
        capacity = 0;
        resize(d0, d1, _force_compact);
    }

    tensor2D(int d0, int d1, T * ext, int _stride) {
        capacity = 1;
        data = std::shared_ptr<T>(ext, [](void *) {});
        dims[0] = d0;
        dims[1] = d1;
        stride = _stride;
        padded_dim1 = stride / sizeof(T);
    }

    tensor2D<T> Tr() {
        tensor2D<T> ret(dims[1], dims[0]);
        for(int c0=0; c0 < dims[0]; ++c0) {
            for(int c1=0; c1 < dims[1]; ++c1) {
                ret(c1, c0) = (*this)(c0, c1);
            }
        }
        return ret;
    }
    tensor2D<T> clone() {
        tensor2D<T> ret;
        ret.resize(dims[0], dims[1], force_compact);
        if (ret.stride == stride) {
            memcpy(ret.data.get(), data.get(), dims[0] * stride);
        }else{
            for(int i=0;i<dims[0];i++) {
                memcpy(&ret(i,0), &(*this)(i,0), ret.stride);
            }
        }
        return ret;
    }
    void resize(int d0, int d1, bool _force_compact = false) {
        force_compact = _force_compact;
        dims[0] = d0;
        dims[1] = d1;
        stride = d1 * sizeof(T);
        if ((stride % 64) && (!force_compact)) {
            auto stride_fix = rndup(stride, 64);
            //logger() << "\tWarnning: stride " << stride << " is not aligned to cache line, will increase to " << stride_fix
            //          << " (" << stride_fix/64 << " cache lines)\n";
            stride = stride_fix;
        }
        padded_dim1 = stride / sizeof(T);

        // resize method never shrink capacity, and extra T is added to put nan as test
        auto need_capacity = dims[0] * stride + sizeof(T);
        if (capacity < need_capacity) {
            capacity = need_capacity;
            // align begin address to cache line is vital, so tile load can
            // use all bandwidth (L1D/L2 only deliver data in unit of 64-byte aligned cache-line)

#ifdef ENABLE_NUMA
            if (USE_NUMA) {
                data = std::shared_ptr<T>(
                            reinterpret_cast<T*>(numa_alloc_local(capacity)),
                            [need_capacity](void * p){ numa_free(p, need_capacity); });
            } else {
#else
            {
#endif

#ifdef _WIN32
                data = std::shared_ptr<T>(
                            reinterpret_cast<T*>(_aligned_malloc(capacity, 64)),
                            [](void * p) { _aligned_free(p); });
#else
                data = std::shared_ptr<T>(
                            reinterpret_cast<T*>(aligned_alloc(64, capacity)),
                            [](void * p) { ::free(p); });
#endif
            }
            if (reinterpret_cast<uintptr_t>(data.get()) % 64)
                std::cout << "WARNING: resize(), data is not cache-line aligned!" << std::endl;
        }
        // put a NaN at the end to test over-read
        // https://en.wikipedia.org/wiki/Bfloat16_floating-point_format
        #define INF 0xff80 
        #define NAN1 (INF + 1)
        if (sizeof(T) == 2) {
            *reinterpret_cast<uint16_t*>(data.get() + dims[0] * padded_dim1) = NAN1;
        }
        if (sizeof(T) == 4) {
            *reinterpret_cast<uint32_t*>(data.get() + dims[0] * padded_dim1) = (INF << 16) + 1;
        }
    }

    T & operator[](int i) {
        return data.get()[i];
    }

    const T & operator[](int i) const {
        return data.get()[i];
    }

    //https://stackoverflow.com/questions/1936399/c-array-operator-with-multiple-arguments
    T & operator()(int i0, int i1) {
        return (*this)[i0 * padded_dim1 + i1];
    }

    const T & operator()(int i0, int i1) const {
        return (*this)[i0 * padded_dim1 + i1];
    }

    void operator=(const T & v) {
        for(int k = 0; k<dims[0]*padded_dim1; k++)
            (*this)[k] = v;
    }

    tensor2D<T>& operator=(const tensor2D<T> & t2) {
        assert(dims[0]*dims[1] == t2.dims[0] * t2.dims[1]);
        for(int c0 = 0; c0 < dims[0]; c0++)
        for(int c1 = 0; c1 < dims[1]; c1++) {
            int k = c0*dims[1] + c1;
            auto c2 = k / t2.dims[1];
            auto c3 = k % t2.dims[1];
            (*this)(c0, c1) = t2(c2, c3);
        }
        return *this;
    }

    // move semantics
    tensor2D(tensor2D<T> && t2) {
        dims[0] = t2.dims[0];
        dims[1] = t2.dims[1];
        data = t2.data;
        capacity = t2.capacity;
        stride = t2.stride;
        padded_dim1 = t2.padded_dim1;
        force_compact = t2.force_compact;
        t2.capacity = 0;
        t2.data.reset();
    }

    tensor2D<T>&  operator=(tensor2D<T> && t2) {
        dims[0] = t2.dims[0];
        dims[1] = t2.dims[1];
        data = t2.data;
        capacity = t2.capacity;
        stride = t2.stride;
        padded_dim1 = t2.padded_dim1;
        force_compact = t2.force_compact;
        t2.capacity = 0;
        t2.data.reset();
        return *this;
    }

    bool operator==(const tensor2D<T> & rhs) const {
        if (dims[0] != rhs.dims[0] || dims[1] != rhs.dims[1])
            return false;
        for(int i0=0; i0<dims[0]; i0++)
        for(int i1=0; i1<dims[1]; i1++) {
            // with -ffast-math,  std::isnan, std::isinf,  x != x  always return false
            // so we need special logic to test nan here
            if (std::is_same<T, ov::bfloat16>::value ||
                std::is_same<T, float>::value) {
                float f0 = (*this)(i0,i1);
                float f1 = rhs(i0,i1);
                if (isnan2(f1) || isnan2(f0)) {
                    std::cout << " nan is found: f0=" << f0 << ",  f1=" << f1 << std::endl;
                    return false;
                }
            }

            if ((*this)(i0,i1) == rhs(i0,i1))
                continue;
            std::cout << " operator== failed at (" << i0 << ", " << i1 << ")  value "
                        << (*this)(i0,i1) << "!=" << rhs(i0,i1) << std::endl;
            return false;
        }
        return true;
    }

    bool is_normal() {
        for(int i0=0; i0<dims[0]; i0++)
        for(int i1=0; i1<dims[1]; i1++) {
            float f0 = (*this)(i0,i1);
            if (isnan2(f0)) {
                std::cout << " found nan at (" << i0 << "," << i1 << ")" << std::endl;
                return false;
            }
            if (isinf2(f0)) {
                std::cout << " found inf at (" << i0 << "," << i1 << ")" << std::endl;
                return false;
            }
        }
        return true;
    }

    bool compare(const tensor2D<T> & rhs, float tolerance) {
        float max_abs_diff = 0;
        float max_rel_diff = 0;
        if (dims[0] != rhs.dims[0] || dims[1] != rhs.dims[1])
            return false;
        for(int i0=0; i0<dims[0]; i0++)
        for(int i1=0; i1<dims[1]; i1++) {
            auto diff = std::fabs((*this)(i0,i1) - rhs(i0,i1));
            auto rel_diff = diff/std::fabs((*this)(i0,i1));
            max_abs_diff = std::max(max_abs_diff, diff);
            if (std::fabs((*this)(i0,i1) > 0) && diff > 0)
                max_rel_diff = std::max(max_rel_diff, rel_diff);
        }
        std::cout << "max_abs_diff=" << max_abs_diff << " max_rel_diff=" << max_rel_diff;
        return tolerance > max_abs_diff;
    }
    friend std::ostream& operator<<(std::ostream& out, const tensor2D<T>& obj) {
        int i0;
        auto showline = [&](int i) {
            out << "[" << i << "," << 0 << "]: ";
            int i1;
            for(i1=0; i1<obj.dims[1] && i1 < 8; i1++) {
                out << +obj(i0,i1) << ",";
            }
            if (i1 < obj.dims[1]) out << "...";
            out << std::endl;
        };
        for(i0=0; i0 < obj.dims[0] && i0 < 32; i0++) {
            showline(i0);
        }
        if (i0 < obj.dims[0]) {
            out << "... ... ... ..." << std::endl;
            showline(obj.dims[0] - 1);
        }
        return out;
    }
};

//https://stackoverflow.com/questions/68340319/differences-between-avx-and-avx2
//
//  The number of architectural YMM registers is 16 for 64-bit AVX2
//  thus 
// 
namespace avx2 {
/*
 numactl -m 0 -C 12 ./benchdnn --ip --reset --mode=p --allow-enum-tags-only=0 --engine=cpu --dir=FWD_B \
 --cfg=f32 --stag=ab --wtag=AB16b64a --dtag=ab mb12ic4864oc256

perf,cpu,x64:gemm:jit,,--mode=P --ip --allow-enum-tags-only=false --dir=FWD_I --stag=ab --wtag=ab --dtag=ab --attr-scratchpad=user mb128ic384oc51864,5.09844,9.29395,548.576,9.44089,540.038
tests:1 passed:1 skipped:0 mistrusted:0 unimplemented:0 invalid_arguments:0 failed:0 listed:0
total perf: min(ms):9.29395 avg(ms):9.44089

mb128ic384oc51864

*/

namespace functional {
    // https://stackoverflow.com/questions/25622745/transpose-an-8x8-float-using-avx-avx2
    inline void transpose8_ps(__m256 &row0, __m256 &row1, __m256 &row2, __m256 &row3, __m256 &row4, __m256 &row5, __m256 &row6, __m256 &row7) {
        __m256 __t0, __t1, __t2, __t3, __t4, __t5, __t6, __t7;
        __m256 __tt0, __tt1, __tt2, __tt3, __tt4, __tt5, __tt6, __tt7;
        __t0 = _mm256_unpacklo_ps(row0, row1);
        __t1 = _mm256_unpackhi_ps(row0, row1);
        __t2 = _mm256_unpacklo_ps(row2, row3);
        __t3 = _mm256_unpackhi_ps(row2, row3);
        __t4 = _mm256_unpacklo_ps(row4, row5);
        __t5 = _mm256_unpackhi_ps(row4, row5);
        __t6 = _mm256_unpacklo_ps(row6, row7);
        __t7 = _mm256_unpackhi_ps(row6, row7);
        __tt0 = _mm256_shuffle_ps(__t0,__t2,_MM_SHUFFLE(1,0,1,0));
        __tt1 = _mm256_shuffle_ps(__t0,__t2,_MM_SHUFFLE(3,2,3,2));
        __tt2 = _mm256_shuffle_ps(__t1,__t3,_MM_SHUFFLE(1,0,1,0));
        __tt3 = _mm256_shuffle_ps(__t1,__t3,_MM_SHUFFLE(3,2,3,2));
        __tt4 = _mm256_shuffle_ps(__t4,__t6,_MM_SHUFFLE(1,0,1,0));
        __tt5 = _mm256_shuffle_ps(__t4,__t6,_MM_SHUFFLE(3,2,3,2));
        __tt6 = _mm256_shuffle_ps(__t5,__t7,_MM_SHUFFLE(1,0,1,0));
        __tt7 = _mm256_shuffle_ps(__t5,__t7,_MM_SHUFFLE(3,2,3,2));
        row0 = _mm256_permute2f128_ps(__tt0, __tt4, 0x20);
        row1 = _mm256_permute2f128_ps(__tt1, __tt5, 0x20);
        row2 = _mm256_permute2f128_ps(__tt2, __tt6, 0x20);
        row3 = _mm256_permute2f128_ps(__tt3, __tt7, 0x20);
        row4 = _mm256_permute2f128_ps(__tt0, __tt4, 0x31);
        row5 = _mm256_permute2f128_ps(__tt1, __tt5, 0x31);
        row6 = _mm256_permute2f128_ps(__tt2, __tt6, 0x31);
        row7 = _mm256_permute2f128_ps(__tt3, __tt7, 0x31);
    }

    template<bool skipB1>
    inline void transpose_16xK_ps(float * pBdst, float *pBsrc, int strideB, int K) {
        for(int k = 0; k < K; k+=8, pBsrc+=8) {
            {
                auto b0 = _mm256_loadu_ps(pBsrc);
                auto b1 = _mm256_loadu_ps(pBsrc + strideB);
                auto b2 = _mm256_loadu_ps(pBsrc + strideB*2);
                auto b3 = _mm256_loadu_ps(pBsrc + strideB*3);
                auto b4 = _mm256_loadu_ps(pBsrc + strideB*4);
                auto b5 = _mm256_loadu_ps(pBsrc + strideB*5);
                auto b6 = _mm256_loadu_ps(pBsrc + strideB*6);
                auto b7 = _mm256_loadu_ps(pBsrc + strideB*7);
                functional::transpose8_ps(b0, b1, b2, b3, b4, b5, b6, b7);
                _mm256_storeu_ps(pBdst, b0);
                _mm256_storeu_ps(pBdst + 8*2, b1);
                _mm256_storeu_ps(pBdst + 8*4, b2);
                _mm256_storeu_ps(pBdst + 8*6, b3);
                _mm256_storeu_ps(pBdst + 8*8, b4);
                _mm256_storeu_ps(pBdst + 8*10, b5);
                _mm256_storeu_ps(pBdst + 8*12, b6);
                _mm256_storeu_ps(pBdst + 8*14, b7);
            }
            if (!skipB1) {
                auto b0 = _mm256_loadu_ps(pBsrc + strideB*8);
                auto b1 = _mm256_loadu_ps(pBsrc + strideB*9);
                auto b2 = _mm256_loadu_ps(pBsrc + strideB*10);
                auto b3 = _mm256_loadu_ps(pBsrc + strideB*11);
                auto b4 = _mm256_loadu_ps(pBsrc + strideB*12);
                auto b5 = _mm256_loadu_ps(pBsrc + strideB*13);
                auto b6 = _mm256_loadu_ps(pBsrc + strideB*14);
                auto b7 = _mm256_loadu_ps(pBsrc + strideB*15);
                functional::transpose8_ps(b0, b1, b2, b3, b4, b5, b6, b7);
                _mm256_storeu_ps(pBdst + 8, b0);
                _mm256_storeu_ps(pBdst + 8*3, b1);
                _mm256_storeu_ps(pBdst + 8*5, b2);
                _mm256_storeu_ps(pBdst + 8*7, b3);
                _mm256_storeu_ps(pBdst + 8*9, b4);
                _mm256_storeu_ps(pBdst + 8*11, b5);
                _mm256_storeu_ps(pBdst + 8*13, b6);
                _mm256_storeu_ps(pBdst + 8*15, b7);
            }
            pBdst += 8*16;
        }
    }

    inline void hmax(__m256 & x) {
        __m256 y;                       //x:  0 1 2 3   4 5 6 7
        y = _mm256_permute_ps(x, 0x39); //y:  1 2 3 0   5 6 7 4
        x = _mm256_max_ps(x, y);        //X:  01 12 23 30  45 56 67 74
        y = _mm256_permute_ps(x, 0x4e); //y:  23 30 01 12  67 74 45 56
        x = _mm256_max_ps(x, y);             //x: 0123 x x x   4567 x x x 
        y = _mm256_permute2f128_ps(x, x, 1); //y: 4567 x x x  0123 x x x
        x = _mm256_max_ps(x, y);             //x: 01234567 x x x x x x x
    }
    inline void hsum(__m256 & x) {
        __m256 y;                       //x:  0 1 2 3   4 5 6 7
        y = _mm256_permute_ps(x, 0x39); //y:  1 2 3 0   5 6 7 4
        x = _mm256_add_ps(x, y);        //X:  01 12 23 30  45 56 67 74
        y = _mm256_permute_ps(x, 0x4e); //y:  23 30 01 12  67 74 45 56
        x = _mm256_add_ps(x, y);             //x: 0123 x x x   4567 x x x 
        y = _mm256_permute2f128_ps(x, x, 1); //y: 4567 x x x  0123 x x x
        x = _mm256_add_ps(x, y);             //x: 01234567 x x x x x x x
    }
    inline void exp_ps(__m256 & src) {
        static __m256 exp_ln_flt_min_f = _mm256_castsi256_ps(_mm256_set1_epi32(0xc2aeac50));    // log(FLT_MIN)
        static __m256 exp_ln_flt_max_f = _mm256_castsi256_ps(_mm256_set1_epi32(0x42b17218));    // log(FLT_MAX)
        static __m256 exp_log2ef = _mm256_castsi256_ps(_mm256_set1_epi32(0x3fb8aa3b));          // log2(e)
        static __m256 half = _mm256_castsi256_ps(_mm256_set1_epi32(0x3f000000));                // 0.5f
        static __m256 ln2f = _mm256_castsi256_ps(_mm256_set1_epi32(0x3f317218));                // ln(2)
        static __m256 one = _mm256_castsi256_ps(_mm256_set1_epi32(0x3f800000));                 // 1.0f
        static __m256i exponent_bias = _mm256_set1_epi32(0x0000007f);                           // 127
        static constexpr int n_mantissa_bits = 23;
        static __m256 exp_pol1 = _mm256_castsi256_ps(_mm256_set1_epi32(0x3f7ffffb));            // p1 = 0.999999701f
        static __m256 exp_pol2 = _mm256_castsi256_ps(_mm256_set1_epi32(0x3efffee3));            // p2 = 0.499991506f
        static __m256 exp_pol3 = _mm256_castsi256_ps(_mm256_set1_epi32(0x3e2aad40));            // p3 = 0.166676521f
        static __m256 exp_pol4 = _mm256_castsi256_ps(_mm256_set1_epi32(0x3d2b9d0d));            // p4 = 0.0418978221f
        static __m256 exp_pol5 = _mm256_castsi256_ps(_mm256_set1_epi32(0x3c07cfce));            // p5 = 0.00828929059f
        static __m256 two = _mm256_castsi256_ps(_mm256_set1_epi32(0x40000000));                 // 2
        // exp(x) =
        // = exp(n * ln(2) + r) // divide x by ln(2) and get quot and rem
        // = 2^n * exp(r)       // simplify the exp(n*ln(2)) expression

        // get mask of values lower than log(FLT_MIN) to zero them in the output
        auto zero_mask = _mm256_cmp_ps(src, exp_ln_flt_min_f, _CMP_LT_OS);

        // clip src
        src = _mm256_min_ps(src, exp_ln_flt_max_f);
        src = _mm256_max_ps(src, exp_ln_flt_min_f);

        // aux1 : r
        auto aux1 = src;

        // calculate exp(x)
        // fx = x * log2(e) + 0.5
        src = _mm256_mul_ps(src, exp_log2ef);
        src = _mm256_add_ps(src, half);

        // tmp = floorf(fx)
        src = _mm256_floor_ps(src);

        // aux1 = x - fx * ln2
        aux1 = _mm256_fnmadd_ps(src, ln2f, aux1);
        
        // We do not count 2^n here, because n can reach 128 and 2^128 is not
        // representable by fp32, so to get around this problem, instead of computing
        // 2^n * exp(r) will be counted 2*2^(n-1)*exp(r), because 2^127
        // and 2 are numbers representable in fp32.

        // compute 2^(n-1)
        src = _mm256_sub_ps(src, one);
        auto aux2_i = _mm256_cvtps_epi32(src);
        aux2_i = _mm256_add_epi32(aux2_i, exponent_bias);
        aux2_i = _mm256_slli_epi32 (aux2_i, n_mantissa_bits);

        // set zeroes at those points which were < log(FLT_MIN)
        auto zero = _mm256_setzero_ps();
        auto aux2 = _mm256_blendv_ps(_mm256_castsi256_ps(aux2_i), zero, zero_mask);

        // compute polynomial
        src = exp_pol5;
        src = _mm256_fmadd_ps(src, aux1, exp_pol4);
        src = _mm256_fmadd_ps(src, aux1, exp_pol3);
        src = _mm256_fmadd_ps(src, aux1, exp_pol2);
        src = _mm256_fmadd_ps(src, aux1, exp_pol1);
        src = _mm256_fmadd_ps(src, aux1, one);

        // y = y * 2^n
        src = _mm256_mul_ps(src, aux2);
        src = _mm256_mul_ps(src, two);
    }

    template<typename T=float>
    void softmax(T * v, int N) {
        static __m256 one = _mm256_castsi256_ps(_mm256_set1_epi32(0x3f800000));                 // 1.0f
        static __m256 lower_32 = _mm256_castsi256_ps(_mm256_set_epi32(0,0,0,0,0,0,0,-1));
        auto x_max = _mm256_set1_ps(std::numeric_limits<float>::lowest());
        int i;
        for(i = 0; (i+8) <= N; i+=8) {
            auto x = _mm256_loadu_ps(v + i);
            x_max = _mm256_max_ps(x_max, x);
        }
        // tails
        for(;i<N;i++) {
            auto x = _mm256_broadcast_ss(v + i);
            x_max = _mm256_max_ps(x_max, x);
        }
        avx2::functional::hmax(x_max);

        // softmax
        auto sum_exp = _mm256_setzero_ps();
        for(i = 0; (i+8) <= N; i+=8) {
            auto x = _mm256_loadu_ps(v + i);
            x = _mm256_sub_ps(x, x_max);
            avx2::functional::exp_ps(x);         // exp(x-x_max)
            sum_exp = _mm256_add_ps(sum_exp, x); // sum(exp(x-x_max))
            _mm256_storeu_ps(v + i, x);          // save exp(x-x_max)
        }

        if (i < N) {
            auto sum_exp_tail = _mm256_setzero_ps();
            for(;i<N;i++) {
                auto x = _mm256_broadcast_ss(v + i);
                x = _mm256_sub_ps(x, x_max);
                avx2::functional::exp_ps(x);
                sum_exp_tail = _mm256_add_ps(sum_exp_tail, x);
                v[i] = _mm256_cvtss_f32(x);
            }
            sum_exp_tail = _mm256_and_ps(sum_exp_tail, lower_32);// only lowest f32 is valid sum
            sum_exp = _mm256_add_ps(sum_exp, sum_exp_tail); // add tail
        }
        avx2::functional::hsum(sum_exp);
        auto reciprocal_sum_exp = _mm256_div_ps(one, sum_exp);     // 1/sum_exp

        // divide
        for(i = 0; (i+8) <= N; i+=8) {
            auto x = _mm256_loadu_ps(v + i);
            x = _mm256_mul_ps(x, reciprocal_sum_exp);
            _mm256_storeu_ps(v + i, x);
        }
        for(;i<N;i++) {
            auto x = _mm256_broadcast_ss(v + i);
            x = _mm256_mul_ps(x, reciprocal_sum_exp);
            v[i] = _mm256_cvtss_f32(x);
        }
    }

    // gelu_erf_minimax_approx_compute_vector_fwd in oneDNN
    //   x*0.5*(1+erf(x/sqrt(2))) = x*0.5*(1 + x*Polynomial(x^2))
    inline __m256 gelu_erf_minmax_approx(__m256 & x) {
        auto x2 = _mm256_mul_ps(x, x); // x^2
        
        auto x_positive = _mm256_castsi256_ps(_mm256_and_si256(_mm256_castps_si256(x), _mm256_set1_epi32(0x7FFFFFFF)));    // clear sign mask
        auto x_half = _mm256_mul_ps(x, _mm256_set1_ps(0.5f));

        auto poly = _mm256_castsi256_ps(_mm256_set1_epi32(0x1f1c83fd));
        poly = _mm256_fmadd_ps(poly, x2, _mm256_castsi256_ps(_mm256_set1_epi32(0xa3198977))); // poly * x^2 + xxx
        poly = _mm256_fmadd_ps(poly, x2, _mm256_castsi256_ps(_mm256_set1_epi32(0x268a7927)));
        poly = _mm256_fmadd_ps(poly, x2, _mm256_castsi256_ps(_mm256_set1_epi32(0xa998c963)));
        poly = _mm256_fmadd_ps(poly, x2, _mm256_castsi256_ps(_mm256_set1_epi32(0x2c67ddb2)));
        poly = _mm256_fmadd_ps(poly, x2, _mm256_castsi256_ps(_mm256_set1_epi32(0xaf013b2c)));
        poly = _mm256_fmadd_ps(poly, x2, _mm256_castsi256_ps(_mm256_set1_epi32(0x315d4a4f)));
        poly = _mm256_fmadd_ps(poly, x2, _mm256_castsi256_ps(_mm256_set1_epi32(0xb3969b11)));
        poly = _mm256_fmadd_ps(poly, x2, _mm256_castsi256_ps(_mm256_set1_epi32(0x35a776e9)));
        poly = _mm256_fmadd_ps(poly, x2, _mm256_castsi256_ps(_mm256_set1_epi32(0xb79b0914)));
        poly = _mm256_fmadd_ps(poly, x2, _mm256_castsi256_ps(_mm256_set1_epi32(0x3970b255)));
        poly = _mm256_fmadd_ps(poly, x2, _mm256_castsi256_ps(_mm256_set1_epi32(0xbb1b7399)));
        poly = _mm256_fmadd_ps(poly, x2, _mm256_castsi256_ps(_mm256_set1_epi32(0x3ca3621f)));
        poly = _mm256_fmadd_ps(poly, x2, _mm256_castsi256_ps(_mm256_set1_epi32(0xbe082bc7)));
        poly = _mm256_fmadd_ps(poly, x2, _mm256_castsi256_ps(_mm256_set1_epi32(0x3f4c4228)));

        // 1.0f + erf(x * inv_sqrt2) = 1.0f + x * P(x^2)
        poly = _mm256_fmadd_ps(poly, x, _mm256_set1_ps(1.0f));
        // x*0.5*(1 + x*Polynomial(x^2))
        poly = _mm256_mul_ps(poly, x_half);

        // combine:
        // zone_id
        //  1 -inf; -saturation_lbound           : 0.0f
        //  2 -saturation_lbound; -linear_ubound : x*0.5*(1 + x*Polynomial(x^2))
        //  3 -linear_ubound, linear_ubound         : x*0.5
        //  4 linear_ubound : saturation_lbound     : x*0.5*(1 + x*Polynomial(x^2))
        //  5 saturation_lbound: +inf               : x
        constexpr int neg_saturation_lbound = 0xc0a00000;
        constexpr int linear_ubound = 0x33800000;
        constexpr int saturation_lbound = 0x40a00000;

        auto mask_x_not_zone1 = _mm256_cmp_ps(x, _mm256_castsi256_ps(_mm256_set1_epi32(neg_saturation_lbound)), _CMP_NLT_UQ);
        x = _mm256_blendv_ps(_mm256_setzero_ps(), x, mask_x_not_zone1); // not_zone1 => keep x

        auto mask_x_in_zone5 = _mm256_cmp_ps(x_positive, _mm256_castsi256_ps(_mm256_set1_epi32(saturation_lbound)), _CMP_NLE_UQ);
        poly = _mm256_blendv_ps(poly, x, mask_x_in_zone5);

        auto mask_x_in_zone3 = _mm256_cmp_ps(x_positive, _mm256_castsi256_ps(_mm256_set1_epi32(linear_ubound)), _CMP_LE_OQ);
        poly = _mm256_blendv_ps(poly, x_half, mask_x_in_zone3);
        return poly;
    }
}

namespace PP {
    struct None {
        None() { };

        template<int start=0, typename ... INT>
        void prepare(int n0, INT ... ns) {}

        template<typename ... M256>
        FORCE_INLINE void exec(M256& ... vs) {}
    };

    enum Act {
        Act_NONE = 0,
        Act_RELU = 1,
        Act_GELU = 2,
    };

    template<Act act>
    struct AddbiasAct {
        float * bias;
        AddbiasAct(float * bias) : bias(bias) {
        };

        __m256 bias0;
        __m256 bias1;
        __m256 bias2;
        __m256 bias3;
        __m256 zero;

        template<int start>
        void prepare() {
            // prepare zero
            zero = _mm256_setzero_ps();
        }

        template<int start=0, typename ... INT>
        void prepare(int n0, INT ... ns) {
            // prepare biases
            if (start == 0) bias0 = _mm256_loadu_ps(bias + n0);
            if (start == 1) bias1 = _mm256_loadu_ps(bias + n0);
            if (start == 2) bias2 = _mm256_loadu_ps(bias + n0);
            if (start == 3) bias3 = _mm256_loadu_ps(bias + n0);
            prepare<start+1>(ns...);
        }

        void relu(){}

        template<typename ... M256>
        void relu(__m256 & vfirst, M256& ... vs) {
            vfirst = _mm256_max_ps(vfirst, zero);
            relu(vs...);
        }

        void gelu(){}

        template<typename ... M256>
        void gelu(__m256 & vfirst, M256& ... vs) {
            vfirst = functional::gelu_erf_minmax_approx(vfirst);
            gelu(vs...);
        }

        // terminator
        template<int start>
        void add_bias() {}

        template<int start=0, typename ... M256>
        void add_bias(__m256 & vfirst, M256& ... vs) {
            if (start == 0) vfirst = _mm256_add_ps(vfirst, bias0);
            if (start == 1) vfirst = _mm256_add_ps(vfirst, bias1);
            if (start == 2) vfirst = _mm256_add_ps(vfirst, bias2);
            if (start == 3) vfirst = _mm256_add_ps(vfirst, bias3);
            add_bias<start + 1>(vs...);
        }

        template<typename ... M256>
        FORCE_INLINE void exec(M256& ... vs) {
            add_bias(vs...);
            if (act == Act_RELU)
                relu(vs...);
            if (act == Act_GELU)
                gelu(vs...);
        }
    };
}

template<int bN, class F>
FORCE_INLINE void loop2D_no_bM(int M, int N, F f) {
    for(int n=0; n<N; n += bN) {
        int valid_n = std::min(N - n, bN);
        f(0, n, M, valid_n);
    }
    return;
}

template<int bM, int bN, class F>
FORCE_INLINE void loop2D(int M, int N, int mc, F f) {
    for(int m0=0; m0<M; m0 += mc*bM) {
        for(int n=0; n<N; n += bN) {
            int valid_n = std::min(N - n, bN);
            int mcnt = std::min(mc, ((M - m0) + bM - 1)/bM);
            for(int m1=0; m1<mcnt; m1++) {
                int m = m0 + m1*bM;
                int valid_m = std::min(M - m, bM);
                f(m, n, valid_m, valid_n);
            }
        }
    }
}

/**************************************
 * loop order: column by column, in case
 * where B needs dynamic transpose, this
 * loop order can keep B slice hot in cache
 */
template<int bM, int bN, class F>
FORCE_INLINE void loop2D_ColumnMajor(int M, int N, F f) {
    for(int n=0; n<N; n += bN) {
        int valid_n = std::min(N - n, bN);
        for(int m=0; m<M; m += bM) {
            int valid_m = std::min(M - m, bM);
            f(m, n, valid_m, valid_n);
        }
    }
}

#if 0

// A: 14xK   B:Kx8 (no-transpose)  C: 14x8
template<int valid_m, int valid_n, typename PP>
void kernel_14x8(float * pA, int strideA,
                 float * pB, int strideB,
                 float * pC, int strideC,
                 int K, int n,
                 PP pp) {
    static_assert(valid_n == 8);
    __m256 c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13;
    __m256 b0;

    c0 = _mm256_setzero_ps();
    c1 = _mm256_setzero_ps();
    c2 = _mm256_setzero_ps();
    c3 = _mm256_setzero_ps();
    c4 = _mm256_setzero_ps();
    c5 = _mm256_setzero_ps();
    c6 = _mm256_setzero_ps();
    c7 = _mm256_setzero_ps();
    c8 = _mm256_setzero_ps();
    c9 = _mm256_setzero_ps();
    c10 = _mm256_setzero_ps();
    c11 = _mm256_setzero_ps();
    c12 = _mm256_setzero_ps();
    c13 = _mm256_setzero_ps();

    #define FMADD(n) \
        if (valid_m > n) { \
            auto a = _mm256_set1_ps(pA[n*strideA]); \
            c##n = _mm256_fmadd_ps(a, b0, c##n); \
        }

    for(int k = 0; k < K; k++, pB += strideB, pA++) {
        b0 = _mm256_loadu_ps(pB);
        //_mm_prefetch(pB + 64, _MM_HINT_T0);

        FMADD(0);
        FMADD(1);
        FMADD(2);
        FMADD(3);
        FMADD(4);
        FMADD(5);
        FMADD(6);
        FMADD(7);
        FMADD(8);
        FMADD(9);
        FMADD(10);
        FMADD(11);
        FMADD(12);
        FMADD(13);
    }

    pp.prepare(n);

    #define STORE(n) \
        if (valid_m > n) { \
            pp.exec(c##n);  \
            _mm256_storeu_ps(pC, c##n);  \
            pC += strideC; \
        }

    STORE(0);
    STORE(1);
    STORE(2);
    STORE(3);
    STORE(4);
    STORE(5);
    STORE(6);
    STORE(7);
    STORE(8);
    STORE(9);
    STORE(10);
    STORE(11);
    STORE(12);
    STORE(13);

    #undef FMADD
    #undef STORE
};

// A: 4xK   B:Kx24 (no-transpose)  C: 4x24
template<int valid_m, int valid_n, typename PP>
void kernel_4x24(float * pA, int strideA,
                 float * pB, int strideB,
                 float * pC, int strideC,
                 int K, int n,
                 PP pp) {
    __m256 c00, c01, c02;
    __m256 c10, c11, c12;
    __m256 c20, c21, c22;
    __m256 c30, c31, c32;
    __m256 b0, b1, b2;

    #define SETZERO(c0, c1, c2) \
        c0 = _mm256_setzero_ps();  \
        if (valid_n > 8) c1 = _mm256_setzero_ps(); \
        if (valid_n > 16) c2 = _mm256_setzero_ps();

    SETZERO(c00, c01, c02);
    if (valid_m > 1) { SETZERO(c10, c11, c12); }
    if (valid_m > 2) { SETZERO(c20, c21, c22); }
    if (valid_m > 3) { SETZERO(c30, c31, c32); }

    #define FMADD(a, b0, b1, b2, c0, c1, c2) \
        c0 = _mm256_fmadd_ps(a, b0, c0); \
        if (valid_n > 8) c1 = _mm256_fmadd_ps(a, b1, c1); \
        if (valid_n > 16) c2 = _mm256_fmadd_ps(a, b2, c2);

    for(int k = 0; k < K; k++, pB += strideB, pA++) {
        b0 = _mm256_loadu_ps(pB);
        if (valid_n > 8) b1 = _mm256_loadu_ps(pB + 8);
        if (valid_n > 16) b2 = _mm256_loadu_ps(pB + 16);

        //_mm_prefetch(pB + 64, _MM_HINT_T0);

        if (valid_m > 0) {
            auto a0 = _mm256_set1_ps(pA[0]);
            FMADD(a0, b0, b1, b2, c00, c01, c02);
        }
        if (valid_m > 1) {
            auto a1 = _mm256_set1_ps(pA[1*strideA]);
            FMADD(a1, b0, b1, b2, c10, c11, c12);
        }
        if (valid_m > 2) {
            auto a2 = _mm256_set1_ps(pA[2*strideA]);
            FMADD(a2, b0, b1, b2, c20, c21, c22);
        }
        if (valid_m > 3) {
            auto a3 = _mm256_set1_ps(pA[3*strideA]);
            FMADD(a3, b0, b1, b2, c30, c31, c32);
        }
    }

    if (valid_n > 16)
        pp.prepare(n, n+8, n+16);
    else if (valid_n > 8)
        pp.prepare(n, n+8);
    else
        pp.prepare(n);

    #define STORE(c0, c1, c2) \
        pp.exec(c0, c1, c2);  \
        _mm256_storeu_ps(pC, c0);  \
        if (valid_n > 8) _mm256_storeu_ps(pC + 8, c1);  \
        if (valid_n > 16) _mm256_storeu_ps(pC + 16, c2);  \
        pC += strideC;

    STORE(c00, c01, c02);
    if (valid_m > 1) { STORE(c10, c11, c12); }
    if (valid_m > 2) { STORE(c20, c21, c22); }
    if (valid_m > 3) { STORE(c30, c31, c32); }

    #undef SETZERO
    #undef FMADD
    #undef STORE
};
#endif

struct Matmul {
    tensor2D<float> internalB;

    bool constB;
    bool transposeB;
    Matmul(bool constB = false, bool transposeB = false) : constB(constB), transposeB(transposeB) {};

    // A: 6xK   B:Kx16 (no-transpose)  C: 6x16
    // when (valid_n < 8)||(valid_n > 8 && valid_n < 16)
    // _mm256_maskstore_ps() is used to avoid write beyond
    // the begin of C buffer
    template<int valid_m, bool valid_n_gt8, typename PP>
    static void kernel_6x16(float * pA, int strideA,
                            float * pB, int strideB,
                            float * pC, int strideC,
                            int K, int n, __m256i n_mask,
                            PP pp) {
        static_assert(valid_m > 0 && valid_m < 7);
        __m256 c00, c01;
        __m256 c10, c11;
        __m256 c20, c21;
        __m256 c30, c31;
        __m256 c40, c41;
        __m256 c50, c51;
        __m256 b0, b1;

        #define SETZERO(c0, c1) \
            c0 = _mm256_setzero_ps();  \
            if (valid_n_gt8) c1 = _mm256_setzero_ps();

        SETZERO(c00, c01);
        if (valid_m > 1) { SETZERO(c10, c11); }
        if (valid_m > 2) { SETZERO(c20, c21); }
        if (valid_m > 3) { SETZERO(c30, c31); }
        if (valid_m > 4) { SETZERO(c40, c41); }
        if (valid_m > 5) { SETZERO(c50, c51); }

        #define FMADD(a, b0, b1, c0, c1) \
            c0 = _mm256_fmadd_ps(a, b0, c0); \
            if (valid_n_gt8) c1 = _mm256_fmadd_ps(a, b1, c1);

        for(int k = 0; k < K; k++, pB += strideB, pA++) {
            b0 = _mm256_loadu_ps(pB);
            if (valid_n_gt8) b1 = _mm256_loadu_ps(pB + 8);
            //if (pB < pBEnd) pB += strideB;
            //_mm_prefetch(pB + 512, _MM_HINT_T0);

            if (valid_m > 0) {
                auto a0 = _mm256_set1_ps(pA[0]);
                FMADD(a0, b0, b1, c00, c01);
            }
            if (valid_m > 1) {
                auto a1 = _mm256_set1_ps(pA[1*strideA]);
                FMADD(a1, b0, b1, c10, c11);
            }
            if (valid_m > 2) {
                auto a2 = _mm256_set1_ps(pA[2*strideA]);
                FMADD(a2, b0, b1, c20, c21);
            }
            if (valid_m > 3) {
                auto a3 = _mm256_set1_ps(pA[3*strideA]);
                FMADD(a3, b0, b1, c30, c31);
            }
            if (valid_m > 4) {
                auto a4 = _mm256_set1_ps(pA[4*strideA]);
                FMADD(a4, b0, b1, c40, c41);
            }
            if (valid_m > 5) {
                auto a5 = _mm256_set1_ps(pA[5*strideA]);
                FMADD(a5, b0, b1, c50, c51);
            }
        }

        if (valid_n_gt8)
            pp.prepare(n, n+8);
        else
            pp.prepare(n);

        if (n < 0) {
            // use `vmaskmovps` in first store to prevent access beyond the begin
            #define STORE(c0, c1) \
                pp.exec(c0, c1);  \
                _mm256_maskstore_ps(pC, n_mask, c0); \
                if (valid_n_gt8) _mm256_storeu_ps(pC + 8, c1);  \
                pC += strideC;

            STORE(c00, c01);
            if (valid_m > 1) { STORE(c10, c11); }
            if (valid_m > 2) { STORE(c20, c21); }
            if (valid_m > 3) { STORE(c30, c31); }
            if (valid_m > 4) { STORE(c40, c41); }
            if (valid_m > 5) { STORE(c50, c51); }
            #undef STORE
        } else {
            #define STORE(c0, c1) \
                pp.exec(c0, c1);  \
                _mm256_storeu_ps(pC, c0);  \
                if (valid_n_gt8) _mm256_storeu_ps(pC + 8, c1);  \
                pC += strideC;

            STORE(c00, c01);
            if (valid_m > 1) { STORE(c10, c11); }
            if (valid_m > 2) { STORE(c20, c21); }
            if (valid_m > 3) { STORE(c30, c31); }
            if (valid_m > 4) { STORE(c40, c41); }
            if (valid_m > 5) { STORE(c50, c51); }
            #undef STORE
        }

        #undef SETZERO
        #undef FMADD
    };

    void reorderB(tensor2D<float> & matB, int n0, int n1) {
        // transposeB : B_NxK
        //
        int K = matB.dims[transposeB ? 1 : 0];
        int N = n1 - n0;
        auto strideB = matB.stride/sizeof(float);
        if (!transposeB) {
            // N tails in internalB matrix is aligned to right border of B
            internalB.resize((N + 15)/16, K*16);
            loop2D_no_bM<16>(1, N, [&](int m, int n, int valid_m, int valid_n) {
                // align to right border of B at N tails
                int nsrc = (valid_n <= 8) ? (n1 - 8) : ((valid_n < 16) ? (n1 - 16) : (n0 + n));
                auto * pBdst = &internalB(n/16, 0);
                auto * pBsrc = &matB(0, nsrc);
                for(int k = 0; k < K; k++) {
                    auto b0 = _mm256_loadu_ps(pBsrc);
                    auto b1 = _mm256_loadu_ps(pBsrc + 8);
                    _mm256_storeu_ps(pBdst, b0);
                    _mm256_storeu_ps(pBdst + 8, b1);
                    pBsrc += strideB;
                    pBdst += 16;
                }
            });
        } else {
            // transpose B(NxK) is costly for non-constB:
            //   - it consumes lots of instructions
            //   - it takes more than 8 HW registers (possibly 9 is enough),
            //     so no more register for storing C
            // thus we only want to do it once, due to limited register resource,
            // we cannot archieve that with on-the-fly transpose. so we transpose it
            // into a temp buffer at once
            internalB.resize((N + 15)/16, rndup(K, 8) *16);
            loop2D_no_bM<16>(1, N, [&](int m, int n, int valid_m, int valid_n) {
                // align to right border of B at N tails
                int nsrc = (valid_n <= 8) ? (n1 - 8) : ((valid_n < 16) ? (n1 - 16) : (n0 + n));
                auto * pBdst = &internalB(n/16, 0);
                auto * pBsrc = &matB(nsrc, 0);
                if (valid_n <= 8)
                    functional::transpose_16xK_ps<true>(pBdst, pBsrc, strideB, K);
                else
                    functional::transpose_16xK_ps<false>(pBdst, pBsrc, strideB, K);
            });
        }
    }

    template<typename P>
    void operator()(tensor2D<float> & matA,
                    tensor2D<float> & matB,
                    tensor2D<float> & matC,
                    int n0, int n1,
                    P pp) {
        int M = matA.dims[0];
        int K = matA.dims[1];
        int N = n1 - n0;
        
        assert(K == matB.dims[transposeB ? 1:0]);
        assert(N <= matB.dims[transposeB ? 0:1]);
        assert(M == matC.dims[0]);
        assert(N <= matC.dims[1]);

        auto strideA = matA.stride/sizeof(float);
        auto strideB = matB.stride/sizeof(float);
        auto strideC = matC.stride/sizeof(float);

        //std::cout << "Matmul:  transposeB=" << transposeB << "  M,K,N=" << M << "," << K << "," << N << "  strideA,B,C=" << strideA << "," << strideB << "," << strideC << std::endl;

        auto use_dynTransB = false;
        auto use_dynReorderB = false;

        if (constB) {
            if (internalB.capacity == 0)
                reorderB(matB, n0, n1);
        } else {
            // dynamically transpose/reorder 16 columns of matB into internalB
            internalB.resize(1, rndup(K, 8) * 16);
            internalB = 0;
            use_dynTransB = transposeB;
            use_dynReorderB = !transposeB;
        }

        constexpr int strideBi = 16;

        //
        __m256i n_mask;
        int32_t i32_n_mask[8];
        if (N & 7) {
            auto n_invalid = 8 - (N&7);
            memset(i32_n_mask, 0xFF, sizeof(i32_n_mask));
            for(int i = 0; i<n_invalid; i++) i32_n_mask[i] = 0;
            n_mask = _mm256_loadu_si256((__m256i const *)i32_n_mask);
        }

        // do a 6x16 result, use 6x(2x8)=12 256bits register
        auto lambda_kernel_6x16 = [&](int m, int n, int valid_m, int valid_n) {
            auto * pA = &matA(m, 0);
            int ndst = (valid_n <= 8) ? (n1 - 8) : ((valid_n < 16) ? (n1 - 16) : (n0 + n));
            auto * pB = (use_dynTransB || use_dynReorderB) ? &internalB[0] : &internalB(n >> 4, 0);
            auto * pC = &matC(m, ndst);
            if (use_dynTransB && m == 0) {
                // dynamically transpose 16 rows of matB into internalB
                if (valid_n <= 8)
                    functional::transpose_16xK_ps<true>(&internalB[0], &matB(ndst, 0), strideB, K);
                else
                    functional::transpose_16xK_ps<false>(&internalB[0], &matB(ndst, 0), strideB, K);
            }
            if (use_dynReorderB && m == 0) {
                // dynamically reorder B matrix into continous internalB
                auto * pBdst = &internalB[0];
                auto * pBsrc = &matB(0, ndst);
                for(int k = 0; k < K; k++) {
                    auto b0 = _mm256_loadu_ps(pBsrc);
                    auto b1 = _mm256_loadu_ps(pBsrc + 8);
                    _mm_prefetch(pBsrc + 32*strideB, _MM_HINT_T1);
                    _mm256_storeu_ps(pBdst, b0);
                    _mm256_storeu_ps(pBdst + 8, b1);
                    pBsrc += strideB;
                    pBdst += strideBi;
                }
            }
            // when valid_n < 8, the store of row if c0 is shifted left so it won't access beyond the end of row
            // but if at the same time n1 < 8 (ndst < 0), it may store beyond the head of the row, so _mm256_maskstore_ps/vmaskmovps
            // has to be used to prevent it (this instruction is slower than _mm256_storeu_ps/vmovups in terms of
            // both throughput & latency)
            if (valid_n <= 8) {
                switch (valid_m)
                {
                    case 6: kernel_6x16<6, false>(pA, strideA, pB, strideBi, pC, strideC, K, ndst, n_mask, pp); break;
                    case 5: kernel_6x16<5, false>(pA, strideA, pB, strideBi, pC, strideC, K, ndst, n_mask, pp); break;
                    case 4: kernel_6x16<4, false>(pA, strideA, pB, strideBi, pC, strideC, K, ndst, n_mask, pp); break;
                    case 3: kernel_6x16<3, false>(pA, strideA, pB, strideBi, pC, strideC, K, ndst, n_mask, pp); break;
                    case 2: kernel_6x16<2, false>(pA, strideA, pB, strideBi, pC, strideC, K, ndst, n_mask, pp); break;
                    case 1: kernel_6x16<1, false>(pA, strideA, pB, strideBi, pC, strideC, K, ndst, n_mask, pp); break;
                }
            } else {
                switch (valid_m)
                {
                    case 6: kernel_6x16<6, true>(pA, strideA, pB, strideBi, pC, strideC, K, ndst, n_mask, pp); break;
                    case 5: kernel_6x16<5, true>(pA, strideA, pB, strideBi, pC, strideC, K, ndst, n_mask, pp); break;
                    case 4: kernel_6x16<4, true>(pA, strideA, pB, strideBi, pC, strideC, K, ndst, n_mask, pp); break;
                    case 3: kernel_6x16<3, true>(pA, strideA, pB, strideBi, pC, strideC, K, ndst, n_mask, pp); break;
                    case 2: kernel_6x16<2, true>(pA, strideA, pB, strideBi, pC, strideC, K, ndst, n_mask, pp); break;
                    case 1: kernel_6x16<1, true>(pA, strideA, pB, strideBi, pC, strideC, K, ndst, n_mask, pp); break;
                }
            }
        };

        /*
        // determine blocking scheme
        int elesz = sizeof(uint16_t);
        int L2 = (256+32)*1024; // Coffee Lake 256K L2/core
        int slice_size = 6 * K *elesz;
        int mc = L2/slice_size - 1;

        // if 1 32xK slice cannot fit L2, use 1 slice at least
        if (mc == 0)
            mc = 1;

        //std::cout << "mc=" << mc << std::endl;
        //loop2D<6, 16>(M, N, mc, lambda_kernel_6x16);
        */
        loop2D_ColumnMajor<6, 16>(M, N, lambda_kernel_6x16);
    }
};

}
