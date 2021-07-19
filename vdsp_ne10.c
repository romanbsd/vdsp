#include "vdsp.h"
#include "fft/NE10_types.h"
#include "fft/NE10_dsp.h"

#ifdef __ARM_NEON

ne10_fft_cfg_float32_t
(*ne10_fft_alloc_c2c_float32)(ne10_int32_t nfft) = ne10_fft_alloc_c2c_float32_neon;

void (*ne10_fft_c2c_1d_float32)(ne10_fft_cpx_float32_t *fout,
                                ne10_fft_cpx_float32_t *fin,
                                ne10_fft_cfg_float32_t cfg,
                                ne10_int32_t inverse_fft) = ne10_fft_c2c_1d_float32_neon;

void (*ne10_fft_r2c_1d_float32)(ne10_fft_cpx_float32_t *fout,
                                ne10_float32_t *fin,
                                ne10_fft_r2c_cfg_float32_t cfg) = ne10_fft_r2c_1d_float32_neon;

void (*ne10_fft_c2r_1d_float32)(ne10_float32_t *fout,
                                ne10_fft_cpx_float32_t *fin,
                                ne10_fft_r2c_cfg_float32_t cfg) = ne10_fft_c2r_1d_float32_neon;

#else
#warning "NEON is not used"

ne10_fft_cfg_float32_t
(*ne10_fft_alloc_c2c_float32)(ne10_int32_t nfft) = ne10_fft_alloc_c2c_float32_c;

void (*ne10_fft_c2c_1d_float32)(ne10_fft_cpx_float32_t *fout,
                                ne10_fft_cpx_float32_t *fin,
                                ne10_fft_cfg_float32_t cfg,
                                ne10_int32_t inverse_fft) = ne10_fft_c2c_1d_float32_c;

void (*ne10_fft_r2c_1d_float32)(ne10_fft_cpx_float32_t *fout,
                                ne10_float32_t *fin,
                                ne10_fft_r2c_cfg_float32_t cfg) = ne10_fft_r2c_1d_float32_c;

void (*ne10_fft_c2r_1d_float32)(ne10_float32_t *fout,
                                ne10_fft_cpx_float32_t *fin,
                                ne10_fft_r2c_cfg_float32_t cfg) = ne10_fft_c2r_1d_float32_c;

#endif

FFTSetup vDSP_create_fftsetup(vDSP_Length __Log2n, FFTRadix __Radix) {
    ne10_fft_r2c_cfg_float32_t cfg;
    ne10_int32_t len = 1U << __Log2n;
    cfg = ne10_fft_alloc_r2c_float32(len);
    return (FFTSetup) cfg;
}

void vDSP_destroy_fftsetup(FFTSetup __setup) {
    ne10_fft_destroy_r2c_float32((ne10_fft_r2c_cfg_float32_t) __setup);
}

void vDSP_fft_zrip(FFTSetup __Setup, const DSPSplitComplex *__C, vDSP_Stride __IC,
                   vDSP_Length __Log2N,
                   FFTDirection __Direction) {
    unsigned long N = 1U << __Log2N;
    ne10_fft_cpx_float32_t input[N];
    ne10_fft_cpx_float32_t output[N];
    unsigned long half = N >> 1U;

    vDSP_ztoc(__C, 1, (DSPComplex *) input, 2, half);

    float scale = 2.F;
    if (__Direction == kFFTDirection_Forward) {
        ne10_fft_r2c_1d_float32(output, (float *) input, (ne10_fft_r2c_cfg_float32_t) __Setup);
        output[0].i = output[half].r;
    } else {
        scale = N;
        input[half].r = input[0].i;
        ne10_fft_c2r_1d_float32((float *) output, input, (ne10_fft_r2c_cfg_float32_t) __Setup);
    }

    vDSP_vsmul((float *) output, 1, &scale, (float *) output, 1, N);
    vDSP_ctoz((DSPComplex *) output, 2, __C, 1, half);
}
