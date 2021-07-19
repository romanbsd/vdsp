#include <string.h>
#include <math.h>
#include <assert.h>
#include "vdsp.h"

#ifdef __ARM_NEON

#include <arm_neon.h>

#else
#warning "NEON is not used"

#endif

#ifndef M_PI
# define M_PI 3.14159265358979323846
#endif

#ifdef __ARM_NEON

static inline float AccumulateNeonLane(const float32x4_t lane) {
#ifdef __aarch64__
    return vaddvq_f32(lane);
#else
    return vgetq_lane_f32(lane, 0) + vgetq_lane_f32(lane, 1) +
         vgetq_lane_f32(lane, 2) + vgetq_lane_f32(lane, 3);
#endif
}

// From:
// https://github.com/Tencent/ncnn/blob/master/src/layer/arm/neon_mathfun.h

#define c_inv_mant_mask ~0x7f800000u
#define c_cephes_SQRTHF 0.707106781186547524f
#define c_cephes_log_p0 7.0376836292E-2f
#define c_cephes_log_p1 - 1.1514610310E-1f
#define c_cephes_log_p2 1.1676998740E-1f
#define c_cephes_log_p3 - 1.2420140846E-1f
#define c_cephes_log_p4 + 1.4249322787E-1f
#define c_cephes_log_p5 - 1.6668057665E-1f
#define c_cephes_log_p6 + 2.0000714765E-1f
#define c_cephes_log_p7 - 2.4999993993E-1f
#define c_cephes_log_p8 + 3.3333331174E-1f
#define c_cephes_log_q1 -2.12194440e-4f
#define c_cephes_log_q2 0.693359375f

/* natural logarithm computed for 4 simultaneous float
   return NaN for x <= 0
*/
static inline float32x4_t log_ps(float32x4_t x) {
    float32x4_t one = vdupq_n_f32(1);

    x = vmaxq_f32(x, vdupq_n_f32(0)); /* force flush to zero on denormal values */
    uint32x4_t invalid_mask = vcleq_f32(x, vdupq_n_f32(0));

    int32x4_t ux = vreinterpretq_s32_f32(x);

    int32x4_t emm0 = vshrq_n_s32(ux, 23);

    /* keep only the fractional part */
    ux = vandq_s32(ux, vdupq_n_s32(c_inv_mant_mask));
    ux = vorrq_s32(ux, vreinterpretq_s32_f32(vdupq_n_f32(0.5f)));
    x = vreinterpretq_f32_s32(ux);

    emm0 = vsubq_s32(emm0, vdupq_n_s32(0x7f));
    float32x4_t e = vcvtq_f32_s32(emm0);

    e = vaddq_f32(e, one);

    /* part2:
       if( x < SQRTHF ) {
         e -= 1;
         x = x + x - 1.0;
       } else { x = x - 1.0; }
    */
    uint32x4_t mask = vcltq_f32(x, vdupq_n_f32(c_cephes_SQRTHF));
    float32x4_t tmp = vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(x), mask));
    x = vsubq_f32(x, one);
    e = vsubq_f32(e, vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(one), mask)));
    x = vaddq_f32(x, tmp);

    float32x4_t z = vmulq_f32(x, x);

    float32x4_t y = vdupq_n_f32(c_cephes_log_p0);
    y = vmulq_f32(y, x);
    y = vaddq_f32(y, vdupq_n_f32(c_cephes_log_p1));
    y = vmulq_f32(y, x);
    y = vaddq_f32(y, vdupq_n_f32(c_cephes_log_p2));
    y = vmulq_f32(y, x);
    y = vaddq_f32(y, vdupq_n_f32(c_cephes_log_p3));
    y = vmulq_f32(y, x);
    y = vaddq_f32(y, vdupq_n_f32(c_cephes_log_p4));
    y = vmulq_f32(y, x);
    y = vaddq_f32(y, vdupq_n_f32(c_cephes_log_p5));
    y = vmulq_f32(y, x);
    y = vaddq_f32(y, vdupq_n_f32(c_cephes_log_p6));
    y = vmulq_f32(y, x);
    y = vaddq_f32(y, vdupq_n_f32(c_cephes_log_p7));
    y = vmulq_f32(y, x);
    y = vaddq_f32(y, vdupq_n_f32(c_cephes_log_p8));
    y = vmulq_f32(y, x);

    y = vmulq_f32(y, z);

    tmp = vmulq_f32(e, vdupq_n_f32(c_cephes_log_q1));
    y = vaddq_f32(y, tmp);

    tmp = vmulq_f32(z, vdupq_n_f32(0.5f));
    y = vsubq_f32(y, tmp);

    tmp = vmulq_f32(e, vdupq_n_f32(c_cephes_log_q2));
    x = vaddq_f32(x, y);
    x = vaddq_f32(x, tmp);
    x = vreinterpretq_f32_u32(
            vorrq_u32(vreinterpretq_u32_f32(x), invalid_mask)); // negative arg will be NAN
    return x;
}

#define c_exp_hi 88.3762626647949f
#define c_exp_lo -88.3762626647949f

#define c_cephes_LOG2EF 1.44269504088896341f
#define c_cephes_exp_C1 0.693359375f
#define c_cephes_exp_C2 -2.12194440e-4f

#define c_cephes_exp_p0 1.9875691500E-4f
#define c_cephes_exp_p1 1.3981999507E-3f
#define c_cephes_exp_p2 8.3334519073E-3f
#define c_cephes_exp_p3 4.1665795894E-2f
#define c_cephes_exp_p4 1.6666665459E-1f
#define c_cephes_exp_p5 5.0000001201E-1f

/* exp() computed for 4 float at once */
static inline float32x4_t exp_ps(float32x4_t x) {
    float32x4_t tmp, fx;

    float32x4_t one = vdupq_n_f32(1);
    x = vminq_f32(x, vdupq_n_f32(c_exp_hi));
    x = vmaxq_f32(x, vdupq_n_f32(c_exp_lo));

    /* express exp(x) as exp(g + n*log(2)) */
    fx = vmlaq_f32(vdupq_n_f32(0.5f), x, vdupq_n_f32(c_cephes_LOG2EF));

    /* perform a floorf */
    tmp = vcvtq_f32_s32(vcvtq_s32_f32(fx));

    /* if greater, substract 1 */
    uint32x4_t mask = vcgtq_f32(tmp, fx);
    mask = vandq_u32(mask, vreinterpretq_u32_f32(one));

    fx = vsubq_f32(tmp, vreinterpretq_f32_u32(mask));

    tmp = vmulq_f32(fx, vdupq_n_f32(c_cephes_exp_C1));
    float32x4_t z = vmulq_f32(fx, vdupq_n_f32(c_cephes_exp_C2));
    x = vsubq_f32(x, tmp);
    x = vsubq_f32(x, z);

    static const float cephes_exp_p[6] = {c_cephes_exp_p0, c_cephes_exp_p1, c_cephes_exp_p2,
                                          c_cephes_exp_p3, c_cephes_exp_p4, c_cephes_exp_p5};
    float32x4_t y = vld1q_dup_f32(cephes_exp_p + 0);
    float32x4_t c1 = vld1q_dup_f32(cephes_exp_p + 1);
    float32x4_t c2 = vld1q_dup_f32(cephes_exp_p + 2);
    float32x4_t c3 = vld1q_dup_f32(cephes_exp_p + 3);
    float32x4_t c4 = vld1q_dup_f32(cephes_exp_p + 4);
    float32x4_t c5 = vld1q_dup_f32(cephes_exp_p + 5);

    y = vmulq_f32(y, x);
    z = vmulq_f32(x, x);

    y = vaddq_f32(y, c1);
    y = vmulq_f32(y, x);
    y = vaddq_f32(y, c2);
    y = vmulq_f32(y, x);
    y = vaddq_f32(y, c3);
    y = vmulq_f32(y, x);
    y = vaddq_f32(y, c4);
    y = vmulq_f32(y, x);
    y = vaddq_f32(y, c5);

    y = vmulq_f32(y, z);
    y = vaddq_f32(y, x);
    y = vaddq_f32(y, one);

    /* build 2^n */
    int32x4_t mm;
    mm = vcvtq_s32_f32(fx);
    mm = vaddq_s32(mm, vdupq_n_s32(0x7f));
    mm = vshlq_n_s32(mm, 23);
    float32x4_t pow2n = vreinterpretq_f32_s32(mm);

    y = vmulq_f32(y, pow2n);
    return y;
}

static inline float32x4_t pow_ps(float32x4_t a, float32x4_t b) {
    // pow(x, m) = exp(m * log(x))
    return exp_ps(vmulq_f32(b, log_ps(a)));
}

#endif

#define HAMM_A 0.54f
#define HAMM_B 0.46f

void vDSP_hamm_window(float *__C, vDSP_Length __N, int __Flag) {
    vDSP_Length n = __Flag ? (__N + 1) / 2 : __N;
    for (vDSP_Length i = 0; i < n; ++i) {
        __C[i] = HAMM_A - HAMM_B * cosf(2 * (float) M_PI * i / __N);
    }
}

void vDSP_hann_window(float *__C, vDSP_Length __N, int __Flag) {
    vDSP_Length n = (__Flag & vDSP_HALF_WINDOW) ? (__N + 1) / 2 : __N;
    const float W = (__Flag & vDSP_HANN_NORM) ? 0.8165f : 0.5f;
    for (vDSP_Length i = 0; i < n; ++i) {
        __C[i] = W * (1.0f - cosf(2 * (float) M_PI * i / __N));
    }
}

void vDSP_vfill(const float *__A, float *__C, vDSP_Stride __IC, vDSP_Length __N) {
    for (vDSP_Length n = 0; n < __N; n += __IC) {
        __C[n] = *__A;
    }
}

void vDSP_dotpr(const float *__A, vDSP_Stride __IA, const float *__B, vDSP_Stride __IB, float *__C,
                vDSP_Length __N) {
    vDSP_Length n = 0;
    *__C = 0;
#ifdef __ARM_NEON
    vDSP_Length postamble_start = __N & ~3;
    float32x4_t c = vdupq_n_f32(0);
    for (; n < postamble_start; n += 4) {
        float32x4_t a = vld1q_f32(__A + n);
        float32x4_t b = vld1q_f32(__B + n);
        c = vmlaq_f32(c, a, b);
    }
    *__C = AccumulateNeonLane(c);
#endif
    for (; n < __N; n++) {
        *__C += __A[n] * __B[n];
    }
}

void vDSP_vadd(const float *__A, vDSP_Stride __IA, const float *__B, vDSP_Stride __IB, float *__C,
               vDSP_Stride __IC, vDSP_Length __N) {
    vDSP_Length n = 0;
#ifdef __ARM_NEON
    vDSP_Length postamble_start = __N & ~3;
    for (; n < postamble_start; n += 4) {
        float32x4_t a = vld1q_f32(__A + n);
        float32x4_t b = vld1q_f32(__B + n);
        float32x4_t c = vaddq_f32(a, b);
        vst1q_f32(__C + n, c);
    }
#endif
    for (; n < __N; ++n) {
        __C[n] = __A[n] + __B[n];
    }
}

void vDSP_vsub(const float *__B, vDSP_Stride __IB, const float *__A, vDSP_Stride __IA, float *__C,
               vDSP_Stride __IC, vDSP_Length __N) {
    for (vDSP_Length n = 0; n < __N; ++n) {
        __C[n] = __A[n] - __B[n];
    }
}

void vDSP_vmul(const float *__A, vDSP_Stride __IA, const float *__B, vDSP_Stride __IB, float *__C,
               vDSP_Stride __IC, vDSP_Length __N) {
#ifdef __ARM_NEON
    vDSP_Length postamble_start = __N & ~3;
    vDSP_Length n = 0;
    for (; n < postamble_start; n += 4) {
        float32x4_t a = vld1q_f32(__A + n);
        float32x4_t b = vld1q_f32(__B + n);
        float32x4_t c = vmulq_f32(a, b);
        vst1q_f32(__C + n, c);
    }
    for (; n < __N; n++) {
        __C[n] = __A[n] * __B[n];
    }
#else
    for (vDSP_Length i = 0; i < __N; ++i) {
        __C[i] = __A[i] * __B[i];
    }
#endif
}

void vDSP_vdiv(const float *__B, vDSP_Stride __IB, const float *__A, vDSP_Stride __IA, float *__C,
               vDSP_Stride __IC, vDSP_Length __N) {
    for (vDSP_Length i = 0; i < __N; ++i) {
        __C[i] = __A[i] / __B[i];
    }
}


void vDSP_zvmul(const DSPSplitComplex *__A, vDSP_Stride __IA, const DSPSplitComplex *__B,
                vDSP_Stride __IB,
                const DSPSplitComplex *__C, vDSP_Stride __IC, vDSP_Length __N, int __Conjugate) {
    assert(__Conjugate == 1 || __Conjugate == -1);
#ifdef __ARM_NEON
    vDSP_Length postamble_start = __N & ~3;
    vDSP_Length n = 0;
    for (; n < postamble_start; n += 4) {
        float32x4_t Ar = vld1q_f32(__A->realp + n);
        float32x4_t Br = vld1q_f32(__B->realp + n);
        float32x4_t Ai = vld1q_f32(__A->imagp + n);
        float32x4_t Bi = vld1q_f32(__B->imagp + n);
        float32x4_t Cr = vmulq_f32(Ar, Br);
        float32x4_t Ci = vmulq_f32(Ar, Bi);
        if (__Conjugate == 1) {
            Cr = vmlsq_f32(Cr, Ai, Bi);
            Ci = vmlaq_f32(Ci, Ai, Br);
        } else {
            Cr = vmlaq_f32(Cr, Ai, Bi);
            Ci = vmlsq_f32(Ci, Ai, Br);
        }
        vst1q_f32(__C->realp + n, Cr);
        vst1q_f32(__C->imagp + n, Ci);
    }
    for (; n < __N; n++) {
        __C->realp[n] =
                __A->realp[n] * __B->realp[n] - (float) __Conjugate * __A->imagp[n] * __B->imagp[n];
        __C->imagp[n] =
                __A->realp[n] * __B->imagp[n] + (float) __Conjugate * __A->imagp[n] * __B->realp[n];
    }
#else
    for (vDSP_Length n = 0; n < __N; ++n) {
        __C->realp[n] =
                __A->realp[n] * __B->realp[n] - (float) __Conjugate * __A->imagp[n] * __B->imagp[n];
        __C->imagp[n] =
                __A->realp[n] * __B->imagp[n] + (float) __Conjugate * __A->imagp[n] * __B->realp[n];
    }
#endif
}

void vDSP_zrvmul(const DSPSplitComplex *__A, vDSP_Stride __IA, const float *__B, vDSP_Stride __IB,
                 const DSPSplitComplex *__C, vDSP_Stride __IC, vDSP_Length __N) {
    vDSP_Length n = 0;
#ifdef __ARM_NEON
    vDSP_Length postamble_start = __N & ~3;
    for (; n < postamble_start; n += 4) {
        float32x4_t Ar = vld1q_f32(__A->realp + n);
        float32x4_t Ai = vld1q_f32(__A->imagp + n);
        float32x4_t B = vld1q_f32(__B + n);
        float32x4_t Cr = vmulq_f32(Ar, B);
        float32x4_t Ci = vmulq_f32(Ai, B);
        vst1q_f32(__C->realp + n, Cr);
        vst1q_f32(__C->imagp + n, Ci);
    }
#endif
    for (; n < __N; ++n) {
        __C->realp[n] = __A->realp[n] * __B[n];
        __C->imagp[n] = __A->imagp[n] * __B[n];
    }
}

void vDSP_zrvdiv(const DSPSplitComplex *__A, vDSP_Stride __IA, const float *__B, vDSP_Stride __IB,
                 const DSPSplitComplex *__C, vDSP_Stride __IC, vDSP_Length __N) {
    for (vDSP_Length n = 0; n < __N; ++n) {
        __C->realp[n] = __A->realp[n] / __B[n];
        __C->imagp[n] = __A->imagp[n] / __B[n];
    }
}

void vDSP_zvadd(const DSPSplitComplex *__A, vDSP_Stride __IA, const DSPSplitComplex *__B,
                vDSP_Stride __IB, const DSPSplitComplex *__C, vDSP_Stride __IC, vDSP_Length __N) {
#ifdef __ARM_NEON
    vDSP_Length postamble_start = __N & ~3;
    vDSP_Length n = 0;
    for (; n < postamble_start; n += 4) {
        float32x4_t Ar = vld1q_f32(__A->realp + n);
        float32x4_t Br = vld1q_f32(__B->realp + n);
        float32x4_t Ai = vld1q_f32(__A->imagp + n);
        float32x4_t Bi = vld1q_f32(__B->imagp + n);
        float32x4_t Cr = vaddq_f32(Ar, Br);
        float32x4_t Ci = vaddq_f32(Ai, Bi);;
        vst1q_f32(__C->realp + n, Cr);
        vst1q_f32(__C->imagp + n, Ci);
    }
    for (; n < __N; n++) {
        __C->realp[n] = __A->realp[n] + __B->realp[n];
        __C->imagp[n] = __A->imagp[n] + __B->imagp[n];
    }
#else
    for (vDSP_Length n = 0; n < __N; ++n) {
        __C->realp[n] = __A->realp[n] + __B->realp[n];
        __C->imagp[n] = __A->imagp[n] + __B->imagp[n];
    }
#endif
}

void vDSP_zvsub(const DSPSplitComplex *__A, vDSP_Stride __IA, const DSPSplitComplex *__B,
                vDSP_Stride __IB, const DSPSplitComplex *__C, vDSP_Stride __IC, vDSP_Length __N) {
    vDSP_Length n = 0;
#ifdef __ARM_NEON
    vDSP_Length postamble_start = __N & ~3;
    for (; n < postamble_start; n += 4) {
        float32x4_t Ar = vld1q_f32(__A->realp + n);
        float32x4_t Br = vld1q_f32(__B->realp + n);
        float32x4_t Ai = vld1q_f32(__A->imagp + n);
        float32x4_t Bi = vld1q_f32(__B->imagp + n);
        float32x4_t Cr = vsubq_f32(Ar, Br);
        float32x4_t Ci = vsubq_f32(Ai, Bi);;
        vst1q_f32(__C->realp + n, Cr);
        vst1q_f32(__C->imagp + n, Ci);
    }
#endif
    for (; n < __N; n++) {
        __C->realp[n] = __A->realp[n] - __B->realp[n];
        __C->imagp[n] = __A->imagp[n] - __B->imagp[n];
    }
}

void vDSP_zvma(const DSPSplitComplex *__A, vDSP_Stride __IA, const DSPSplitComplex *__B,
               vDSP_Stride __IB, const DSPSplitComplex *__C, vDSP_Stride __IC,
               const DSPSplitComplex *__D, vDSP_Stride __ID, vDSP_Length __N) {
    vDSP_Length n = 0;
#ifdef __ARM_NEON
    vDSP_Length postamble_start = __N & ~3;
    for (; n < postamble_start; n += 4) {
        float32x4_t Ar = vld1q_f32(__A->realp + n);
        float32x4_t Br = vld1q_f32(__B->realp + n);
        float32x4_t Cr = vld1q_f32(__C->realp + n);
        float32x4_t Ai = vld1q_f32(__A->imagp + n);
        float32x4_t Bi = vld1q_f32(__B->imagp + n);
        float32x4_t Ci = vld1q_f32(__C->imagp + n);

        float32x4_t Dr = vmlaq_f32(Cr, Ar, Br);
        Dr = vmlsq_f32(Dr, Ai, Bi);
        vst1q_f32(__D->realp + n, Dr);

        float32x4_t Di = vmlaq_f32(Ci, Ar, Bi);
        Di = vmlaq_f32(Di, Ai, Br);
        vst1q_f32(__D->imagp + n, Di);
    }
#endif
    for (; n < __N; n++) {
        __D->realp[n] =
                __C->realp[n] + __A->realp[n] * __B->realp[n] - __A->imagp[n] * __B->imagp[n];
        __D->imagp[n] =
                __C->imagp[n] + __A->realp[n] * __B->imagp[n] + __A->imagp[n] * __B->realp[n];
    }
}

void vDSP_zvabs(const DSPSplitComplex *__A, vDSP_Stride __IA, float *__C, vDSP_Stride __IC,
                vDSP_Length __N) {
#ifdef __ARM_NEON
    vDSP_Length postamble_start = __N & ~3;
    vDSP_Length n = 0;
    for (; n < postamble_start; n += 4) {
        float32x4_t a = vld1q_f32(__A->realp + n);
        float32x4_t b = vld1q_f32(__A->imagp + n);
        a = vmulq_f32(a, a);
        a = vmlaq_f32(a, b, b);

        float32x4_t a1 = vmaxq_f32(a, vdupq_n_f32(FLT_MIN));
        float32x4_t e = vrsqrteq_f32(a1);
        e = vmulq_f32(vrsqrtsq_f32(vmulq_f32(a1, e), e), e);
        e = vmulq_f32(vrsqrtsq_f32(vmulq_f32(a1, e), e), e);
        a = vmulq_f32(a, e);

        vst1q_f32(__C + n, a);
    }
    for (; n < __N; n++) {
        __C[n] = sqrtf(__A->realp[n] * __A->realp[n] + __A->imagp[n] * __A->imagp[n]);
    }
#else
    for (vDSP_Length n = 0; n < __N; ++n) {
        __C[n] = sqrtf(__A->realp[n] * __A->realp[n] + __A->imagp[n] * __A->imagp[n]);
    }
#endif
}

void vDSP_vmsa(const float *__A, vDSP_Stride __IA, const float *__B, vDSP_Stride __IB,
               const float *__C, float *__D, vDSP_Stride __ID, vDSP_Length __N) {
    for (vDSP_Length n = 0; n < __N; ++n) {
        __D[n] = __A[n] * __B[n] + *__C;
    }
}

void
vDSP_vsma(const float *__A, vDSP_Stride __IA, const float *__B, const float *__C, vDSP_Stride __IC,
          float *__D, vDSP_Stride __ID, vDSP_Length __N) {
    vDSP_Length n = 0;
#ifdef __ARM_NEON
    vDSP_Length postamble_start = __N & ~3;
    const float32x4_t b = vdupq_n_f32(*__B);
    for (; n < postamble_start; n += 4) {
        float32x4_t a = vld1q_f32(__A + n);
        float32x4_t c = vld1q_f32(__C + n);
        float32x4_t d = vmlaq_f32(c, a, b);
        vst1q_f32(__D + n, d);
    }
#endif
    for (; n < __N; n++) {
        __D[n] = __A[n] * *__B + __C[n];
    }
}

void vDSP_vsmsa(const float *__A, vDSP_Stride __IA, const float *__B, const float *__C, float *__D,
                vDSP_Stride __ID, vDSP_Length __N) {
    vDSP_Length n = 0;
#ifdef __ARM_NEON
    vDSP_Length postamble_start = __N & ~3;
    const float32x4_t b = vdupq_n_f32(*__B);
    const float32x4_t c = vdupq_n_f32(*__C);
    for (; n < postamble_start; n += 4) {
        float32x4_t a = vld1q_f32(__A + n);
        float32x4_t d = vmlaq_f32(c, a, b);
        vst1q_f32(__D + n, d);
    }
#endif
    for (; n < __N; n++) {
        __D[n] = __A[n] * *__B + *__C;
    }
}

void vDSP_vsmsma(const float *__A, vDSP_Stride __IA, const float *__B, const float *__C,
                 vDSP_Stride __IC, const float *__D, float *__E, vDSP_Stride __IE,
                 vDSP_Length __N) {
    vDSP_Length n = 0;
#ifdef __ARM_NEON
    vDSP_Length postamble_start = __N & ~3;
    float32x4_t b = vdupq_n_f32(*__B);
    float32x4_t d = vdupq_n_f32(*__D);
    for (; n < postamble_start; n += 4) {
        float32x4_t a = vld1q_f32(__A + n);
        float32x4_t c = vld1q_f32(__C + n);
        float32x4_t e = vmlaq_f32(vmulq_f32(a, b), c, d);
        vst1q_f32(__E + n, e);
    }
#endif
    for (; n < __N; n++) {
        __E[n] = __A[n] * *__B + __C[n] * *__D;
    }
}

void vDSP_maxvi(const float *__A, vDSP_Stride __IA, float *__C, vDSP_Length *__I, vDSP_Length __N) {
    *__C = -INFINITY;
    for (vDSP_Length n = 0; n < __N; ++n) {
        if (__A[n] > *__C) {
            *__C = __A[n];
            *__I = n;
        }
    }
}

void vDSP_rmsqv(const float *__A, vDSP_Stride __IA, float *__C, vDSP_Length __N) {
    float sum = 0;
    for (vDSP_Length n = 0; n < __N; ++n) {
        sum += __A[n] * __A[n];
    }
    *__C = sqrtf(sum / __N);
}

void vDSP_vdbcon(const float *__A, vDSP_Stride __IA, const float *__B, float *__C, vDSP_Stride __IC,
                 vDSP_Length __N, unsigned int __F) {
#ifdef __ARM_NEON
    assert(__N % 4 == 0);
    const float32x4_t alpha = vmulq_f32(vdupq_n_f32(__F == 1 ? 20 : 10),
                                        vdupq_n_f32(0.43429448190325176f)); // 1/ln(10);
    const float32x4_t b = vdupq_n_f32(*__B);
    float32x4_t reciprocal = vrecpeq_f32(b);
    // use a couple Newton-Raphson steps to refine the estimate
    reciprocal = vmulq_f32(vrecpsq_f32(b, reciprocal), reciprocal);
    reciprocal = vmulq_f32(vrecpsq_f32(b, reciprocal), reciprocal);
    for (vDSP_Length i = 0; i < __N; i += 4) {
        float32x4_t a = vld1q_f32(__A + i);
        float32x4_t c = log_ps(vmulq_f32(a, reciprocal));
        c = vmulq_f32(c, alpha);
        vst1q_f32(__C + i, c);
    }
#else
    const int alpha = __F == 1 ? 20 : 10;
    for (vDSP_Length n = 0; n < __N; ++n) {
        __C[n] = alpha * log10f(__A[n] / *__B);
    }
#endif
}

void vDSP_vclip(const float *__A, vDSP_Stride __IA, const float *__B, const float *__C, float *__D,
                vDSP_Stride __ID, vDSP_Length __N) {
    vDSP_Length n = 0;
#ifdef __ARM_NEON
    vDSP_Length postamble_start = __N & ~3;
    const float32x4_t b = vdupq_n_f32(*__B);
    const float32x4_t c = vdupq_n_f32(*__C);
    for (; n < postamble_start; n += 4) {
        float32x4_t a = vld1q_f32(__A + n);
        a = vminq_f32(a, c);
        a = vmaxq_f32(a, b);
        vst1q_f32(__D + n, a);
    }
#endif
    for (; n < __N; n++) {
        if (__A[n * __IA] < *__B) {
            __D[n * __ID] = *__B;
        } else if (__A[n * __IA] > *__C) {
            __D[n * __ID] = *__C;
        } else {
            __D[n * __ID] = __A[n * __IA];
        }
    }
}

void vDSP_vthr(const float *__A, vDSP_Stride __IA, const float *__B, float *__C, vDSP_Stride __IC,
               vDSP_Length __N) {
    vDSP_Length n = 0;
#ifdef __ARM_NEON
    vDSP_Length postamble_start = __N & ~3;
    const float32x4_t b = vdupq_n_f32(*__B);
    for (; n < postamble_start; n += 4) {
        float32x4_t a = vld1q_f32(__A + n);
        a = vmaxq_f32(a, b);
        vst1q_f32(__C + n, a);
    }
#endif
    for (; n < __N; n++) {
        if (__A[n * __IA] < *__B) {
            __C[n * __IC] = *__B;
        } else {
            __C[n * __IC] = __A[n * __IA];
        }
    }
}

#ifndef __ARM_NEON

// https://developer.apple.com/reference/accelerate/1449771-vdsp_vgenp?language=objc
void vDSP_vgenp(const float *__A, vDSP_Stride __IA, const float *__B, vDSP_Stride __IB, float *__C,
                vDSP_Stride __IC, vDSP_Length __N, vDSP_Length __M) {
    vDSP_Length n, m = 0;
    vDSP_Length B[__M];
    for (n = 0; n < __M; ++n) {
        B[n] = (vDSP_Length) __B[n]; // truncate
    }
    for (n = 0; n < __N; ++n) {
        if (n <= B[0]) {
            __C[n] = __A[0];
        } else if (n > B[__M - 1]) {
            __C[n] = __A[__M - 1];
        } else {
            for (; m < __M && B[m] <= n; ++m);
            m--;
            __C[n] = __A[m] + (__A[m + 1] - __A[m]) * (n - __B[m]) / (__B[m + 1] - __B[m]);
        }
    }
}

#endif

void vDSP_ctoz(const DSPComplex *__C, vDSP_Stride __IC, const DSPSplitComplex *__Z,
               vDSP_Stride __IZ, vDSP_Length __N) {
    assert(__IC > 1);
    vDSP_Length n = 0;
#ifdef __ARM_NEON
    if (__IC == 2 && __IZ == 1) {
        vDSP_Length postamble_start = __N & ~3;
        for (; n < postamble_start; n += 4) {
            float32x4x2_t C = vld2q_f32((float *) __C + 2 * n);
            vst1q_f32(__Z->realp + n, C.val[0]);
            vst1q_f32(__Z->imagp + n, C.val[1]);
        }
    }
#endif
    for (; n < __N; n++) {
        __Z->realp[n * __IZ] = __C[n * __IC / 2].real;
        __Z->imagp[n * __IZ] = __C[n * __IC / 2].imag;
    }
}

void vDSP_ztoc(const DSPSplitComplex *__Z, vDSP_Stride __IZ, DSPComplex *__C, vDSP_Stride __IC,
               vDSP_Length __N) {
    assert(__IC > 1);
    vDSP_Length n = 0;
#ifdef __ARM_NEON
    if (__IC == 2 && __IZ == 1) {
        vDSP_Length postamble_start = __N & ~3;
        float32x4x2_t Z;
        for (; n < postamble_start; n += 4) {
            Z.val[0] = vld1q_f32(__Z->realp + n);
            Z.val[1] = vld1q_f32(__Z->imagp + n);
            vst2q_f32((float *) __C + 2 * n, Z);
        }
    }
#endif
    for (; n < __N; n++) {
        __C[n * __IC / 2].real = __Z->realp[n * __IZ];
        __C[n * __IC / 2].imag = __Z->imagp[n * __IZ];
    }
}

void vDSP_sve(const float *__A, vDSP_Stride __I, float *__C, vDSP_Length __N) {
    vDSP_Length n = 0;
    *__C = 0;
#ifdef __ARM_NEON
    vDSP_Length postamble_start = __N & ~3;
    float32x4_t c = vdupq_n_f32(0);
    for (; n < postamble_start; n += 4) {
        float32x4_t a = vld1q_f32(__A + n);
        c = vaddq_f32(a, c);
    }
    *__C = AccumulateNeonLane(c);
#endif
    for (; n < __N; n++) {
        *__C += __A[n];
    }
}

void vDSP_svesq(const float *__A, vDSP_Stride __IA, float *__C, vDSP_Length __N) {
    *__C = 0;
    for (vDSP_Length n = 0; n < __N; ++n) {
        *__C += __A[n] * __A[n];
    }
}

void vDSP_vsmul(const float *__A, vDSP_Stride __IA, const float *__B, float *__C, vDSP_Stride __IC,
                vDSP_Length __N) {
#ifdef __ARM_NEON
    vDSP_Length postamble_start = __N & ~3;
    vDSP_Length n = 0;
    float32_t b = *__B;
    for (; n < postamble_start; n += 4) {
        float32x4_t a = vld1q_f32(__A + n);
        float32x4_t c = vmulq_n_f32(a, b);
        vst1q_f32(__C + n, c);
    }
    for (; n < __N; n++) {
        __C[n] = __A[n] * b;
    }
#else
    for (vDSP_Length i = 0; i < __N; i++) {
        __C[i] = __A[i] * *__B;
    }
#endif
}

void vDSP_vsadd(const float *__A, vDSP_Stride __IA, const float *__B, float *__C, vDSP_Stride __IC,
                vDSP_Length __N) {
#ifdef __ARM_NEON
    float32x4_t b = vdupq_n_f32(*__B);
    vDSP_Length postamble_start = __N & ~3;
    vDSP_Length n = 0;
    for (; n < postamble_start; n += 4) {
        float32x4_t a = vld1q_f32(__A + n);
        float32x4_t c = vaddq_f32(a, b);
        vst1q_f32(__C + n, c);
    }
    for (; n < __N; n++) {
        __C[n] = __A[n] + *__B;
    }
#else
    for (vDSP_Length i = 0; i < __N; i++) {
        __C[i] = __A[i] + *__B;
    }
#endif
}

void vDSP_vsdiv(const float *__A, vDSP_Stride __IA, const float *__B, float *__C, vDSP_Stride __IC,
                vDSP_Length __N) {
    const float divider = 1.0f / *__B;
    vDSP_vsmul(__A, __IA, &divider, __C, __IC, __N);
}

void
vDSP_vflt16(const short *__A, vDSP_Stride __IA, float *__C, vDSP_Stride __IC, vDSP_Length __N) {
#ifdef __ARM_NEON
    for (vDSP_Length i = 0; i < __N; i += 4) {
        // Load, convert to unsigned int, then to float
        float32x4_t c = vcvtq_f32_s32(vmovl_s16(vld1_s16(__A + i)));
        vst1q_f32(__C + i, c);
    }
#else
    for (vDSP_Length i = 0; i < __N; i++) {
        __C[i] = (float) __A[i];
    }
#endif
}

void
vDSP_vfix16(const float *__A, vDSP_Stride __IA, short *__C, vDSP_Stride __IC, vDSP_Length __N) {
// #ifdef __ARM_NEON
//   for (vDSP_Length i = 0; i < __N; i+=4) {
//     float32x4_t a = vld1q_f32(__A + i);
//     int16x4_t c = vmovn_s32(vcvtq_s32_f32(a));
//     vst1_s16(__C + i, c);
//   }
// #else
    for (vDSP_Length i = 0; i < __N; i++) {
        __C[i] = (short) roundf(__A[i]);
    }
// #endif
}

void vDSP_mmul(const float *__A, vDSP_Stride __IA, const float *__B, vDSP_Stride __IB, float *__C,
               vDSP_Stride __IC, vDSP_Length __M, vDSP_Length __N, vDSP_Length __P) {
    if (__N == 1) { // Matrix x Vector aka cblas_sgemv
        for (unsigned m = 0; m < __M; m++) {
            vDSP_dotpr(__A + m * __P, __IA, __B, __IB, __C + m, __P);
        }
    } else {
        for (int m = 0; m < __M; m++) {
            for (int n = 0; n < __N; n++) {
                float cell = 0;
                for (int p = 0; p < __P; p++) {
                    cell += __A[(m * __P + p) * __IA] * __B[(p * __N + n) * __IB];
                }
                __C[(m * __N + n) * __IC] = cell;
            }
        }
    }
}

static const float minus_one = -1.0f;

void vDSP_zvconj(const DSPSplitComplex *__A, vDSP_Stride __IA, const DSPSplitComplex *__C,
                 vDSP_Stride __IC, vDSP_Length __N) {
    /*
    for (vDSP_Length n = 0; n < __N; n++) {
        __C->realp[n] = __A->realp[n];
        __C->imagp[n] = -1.0f * __A->imagp[n];
    }
    */
    if (__A->realp != __C->realp) {
        memcpy(__C->realp, __A->realp, __N * sizeof(float));
    }
    vDSP_vsmul(__A->imagp, 1, &minus_one, __C->imagp, 1, __N);
}

void vDSP_zmmul(const DSPSplitComplex *__A, vDSP_Stride __IA, const DSPSplitComplex *__B,
                vDSP_Stride __IB, const DSPSplitComplex *__C, vDSP_Stride __IC, vDSP_Length __M,
                vDSP_Length __N, vDSP_Length __P) {
    for (int m = 0; m < __M; m++) {
        for (int n = 0; n < __N; n++) {
            float realp = 0, imagp = 0;
            // TODO: optimize
            for (int p = 0; p < __P; p++) {
                realp += __A->realp[(m * __P + p) * __IA] * __B->realp[(p * __N + n) * __IB] -
                         __A->imagp[(m * __P + p) * __IA] * __B->imagp[(p * __N + n) * __IB];
                imagp += __A->realp[(m * __P + p) * __IA] * __B->imagp[(p * __N + n) * __IB] +
                         __A->imagp[(m * __P + p) * __IA] * __B->realp[(p * __N + n) * __IB];
            }
            __C->realp[(m * __N + n) * __IC] = realp;
            __C->imagp[(m * __N + n) * __IC] = imagp;
        }
    }
}

void vDSP_zvmags(const DSPSplitComplex *__A, vDSP_Stride __IA, float *__C, vDSP_Stride __IC,
                 vDSP_Length __N) {
#ifdef __ARM_NEON
    vDSP_Length postamble_start = __N & ~3;
    vDSP_Length n = 0;
    for (; n < postamble_start; n += 4) {
        float32x4_t Ar = vld1q_f32(__A->realp + n);
        float32x4_t Ai = vld1q_f32(__A->imagp + n);
        float32x4_t C = vmlaq_f32(vmulq_f32(Ar, Ar), Ai, Ai);
        vst1q_f32(__C + n, C);
    }
    for (; n < __N; n++) {
        __C[n] = __A->realp[n] * __A->realp[n] + __A->imagp[n] * __A->imagp[n];
    }
#else
    for (int n = 0; n < __N; ++n) {
        __C[n] = __A->realp[n] * __A->realp[n] + __A->imagp[n] * __A->imagp[n];
    }
#endif
}

void vDSP_mtrans(const float *__A, vDSP_Stride __IA, float *__C, vDSP_Stride __IC, vDSP_Length __M,
                 vDSP_Length __N) {
    for (int m = 0; m < __M; m++) {
        for (int n = 0; n < __N; n++) {
            __C[(m * __N + n) * __IC] = __A[(n * __M + m) * __IA];
        }
    }
}

void vDSP_deq22(const float *__A, vDSP_Stride __IA, const float *__B, float *__C, vDSP_Stride __IC,
                vDSP_Length __N) {
    for (int n = 2; n < __N + 2; ++n) {
        __C[n] = __A[n] * __B[0] + __A[n - 1] * __B[1] + __A[n - 2] * __B[2] -
                 __C[n - 1] * __B[3] - __C[n - 2] * __B[4];
    }
}

/* Reasonably fast for small sizes, use FFT implementation for large sizes instead */
void vDSP_conv(const float *__A, vDSP_Stride __IA, const float *__F, vDSP_Stride __IF,
               float *__C, vDSP_Stride __IC, vDSP_Length __N, vDSP_Length __P) {
    for (vDSP_Length n = 0; n < __N; ++n) {
        float sum = 0;
        for (vDSP_Length p = 0; p < __P; p++) {
            sum += __A[n + p] * __F[p * __IF];
        }
        __C[n] = sum;
    }
}

void vvcosf(float *out, const float *in, const int *size) {
    for (int i = 0; i < *size; i++) {
        out[i] = cosf(in[i]);
    }
}

void vvsinf(float *out, const float *in, const int *size) {
    for (int i = 0; i < *size; i++) {
        out[i] = sinf(in[i]);
    }
}

void vvlog10f(float *out, const float *in, const int *size) {
    for (int i = 0; i < *size; i++) {
        out[i] = log10f(in[i]);
    }
}

void vvsqrtf(float *out, const float *in, const int *size) {
    for (int i = 0; i < *size; i++) {
        out[i] = sqrtf(in[i]);
    }
}

void vvpowf(float *z, const float *y, const float *x, const int *n) {
    int i = 0;
#ifdef __ARM_NEON
    int postamble_start = *n & ~3;
    for (; i < postamble_start; i += 4) {
        float32x4_t X = vld1q_f32(x + i);
        float32x4_t Y = vld1q_f32(y + i);
        vst1q_f32(z + i, pow_ps(X, Y));
    }
#endif
    for (; i < *n; i++) {
        z[i] = powf(x[i], y[i]);
    }
}

void vvexpf(float *y, const float *x, const int *n) {
    int i = 0;
#ifdef __ARM_NEON
    int postamble_start = *n & ~3;
    for (; i < postamble_start; i += 4) {
        float32x4_t X = vld1q_f32(x + i);
        vst1q_f32(y + i, exp_ps(X));
    }
#endif
    for (; i < *n; i++) {
        y[i] = expf(x[i]);
    }
}
