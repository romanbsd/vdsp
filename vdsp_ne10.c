#include <assert.h>
#include <stdlib.h>

#include "vdsp.h"
#include "fft/NE10_types.h"
#include "fft/NE10_dsp.h"

typedef struct {
    ne10_fft_r2c_cfg_float32_t r2c;
    ne10_fft_cfg_float32_t c2c;
    vDSP_Length log2n;
    /* Packed workspace for vDSP_fft_zip: N fin samples + N fout (NE10 c2c is out-of-place). */
    ne10_fft_cpx_float32_t *zip_io;
} vdsp_ne10_fft_setup_t;

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
    (void) __Radix;
    ne10_int32_t len = 1U << __Log2n;
    vdsp_ne10_fft_setup_t *s = (vdsp_ne10_fft_setup_t *) calloc(1, sizeof(*s));
    if (!s) {
        return NULL;
    }
    s->log2n = __Log2n;
    s->r2c = ne10_fft_alloc_r2c_float32(len);
    s->c2c = ne10_fft_alloc_c2c_float32(len);
    s->zip_io = (ne10_fft_cpx_float32_t *) malloc((size_t) (2 * len) * sizeof(ne10_fft_cpx_float32_t));
    if (!s->r2c || !s->c2c || !s->zip_io) {
        if (s->r2c) {
            ne10_fft_destroy_r2c_float32(s->r2c);
        }
        if (s->c2c) {
            ne10_fft_destroy_c2c_float32(s->c2c);
        }
        free(s->zip_io);
        free(s);
        return NULL;
    }
    return (FFTSetup) s;
}

void vDSP_destroy_fftsetup(FFTSetup __setup) {
    vdsp_ne10_fft_setup_t *s = (vdsp_ne10_fft_setup_t *) __setup;
    if (!s) {
        return;
    }
    if (s->r2c) {
        ne10_fft_destroy_r2c_float32(s->r2c);
    }
    if (s->c2c) {
        ne10_fft_destroy_c2c_float32(s->c2c);
    }
    free(s->zip_io);
    free(s);
}

void vDSP_fft_zrip(FFTSetup __Setup, const DSPSplitComplex *__C, vDSP_Stride __IC,
                   vDSP_Length __Log2N,
                   FFTDirection __Direction) {
    const vdsp_ne10_fft_setup_t *setup = (const vdsp_ne10_fft_setup_t *) __Setup;
    unsigned long N = 1U << __Log2N;
    ne10_fft_cpx_float32_t input[N];
    ne10_fft_cpx_float32_t output[N];
    unsigned long half = N >> 1U;

    vDSP_ztoc(__C, __IC, (DSPComplex *) input, 2, half);

    float scale = 2.F;
    if (__Direction == kFFTDirection_Forward) {
        ne10_fft_r2c_1d_float32(output, (float *) input, setup->r2c);
        output[0].i = output[half].r;
    } else {
        scale = N;
        input[half].r = input[0].i;
        ne10_fft_c2r_1d_float32((float *) output, input, setup->r2c);
    }

    vDSP_vsmul((float *) output, 1, &scale, (float *) output, 1, N);
    vDSP_ctoz((DSPComplex *) output, 2, (DSPSplitComplex *) __C, __IC, half);
}

void vDSP_fft_zip(FFTSetup __Setup, const DSPSplitComplex *__C, vDSP_Stride __IC, vDSP_Length __Log2N,
                  FFTDirection __Direction) {
    const vdsp_ne10_fft_setup_t *setup = (const vdsp_ne10_fft_setup_t *) __Setup;
    assert(__Log2N == setup->log2n);
    assert(setup->zip_io != NULL);
    unsigned long N = 1U << __Log2N;
    ne10_fft_cpx_float32_t *in = setup->zip_io;
    ne10_fft_cpx_float32_t *out = setup->zip_io + N;

    vDSP_ztoc(__C, __IC, (DSPComplex *) in, 2, N);

    const ne10_int32_t inverse_fft = (__Direction == kFFTDirection_Inverse) ? 1 : 0;
    ne10_fft_c2c_1d_float32(out, in, setup->c2c, inverse_fft);
    /* NE10 divides by N on inverse; Apple vDSP_fft_zip inverse is unnormalized (omits 1/N). */
    if (inverse_fft) {
        float scale = (float) N;
        vDSP_vsmul((float *) out, 1, &scale, (float *) out, 1, 2 * (vDSP_Length) N);
    }

    vDSP_ctoz((DSPComplex *) out, 2, (DSPSplitComplex *) __C, __IC, N);
}
