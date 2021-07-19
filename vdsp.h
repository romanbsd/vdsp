#ifndef _H_VDSP
#define _H_VDSP
#ifdef __cplusplus
extern "C"
{
#endif

typedef struct {
    float *realp;
    float *imagp;
} DSPSplitComplex;

typedef struct {
    float real;
    float imag;
} DSPComplex;

typedef long vDSP_Stride;
typedef unsigned long vDSP_Length;
typedef void *FFTSetup;
typedef int FFTDirection;
typedef int FFTRadix;

#define FFT_RADIX2 0

enum {
    kFFTDirection_Forward = +1,
    kFFTDirection_Inverse = -1
};

enum {
    FFT_FORWARD = kFFTDirection_Forward,
    FFT_INVERSE = kFFTDirection_Inverse
};

enum {
    kFFTRadix2 = 0,
    kFFTRadix3 = 1,
    kFFTRadix5 = 2
};

enum {
    vDSP_HALF_WINDOW = 1,
    vDSP_HANN_DENORM = 0,
    vDSP_HANN_NORM = 2
};

/* Creates a single-precision Hamming window. */
void vDSP_hamm_window(float *__C, vDSP_Length __N, int __Flag);

/* Creates a single-precision Hanning window. */
void vDSP_hann_window(float *__C, vDSP_Length __N, int __Flag);

/* Populate a single-precision vector with a specified scalar value. */
void vDSP_vfill(const float *__A, float *__C, vDSP_Stride __IC, vDSP_Length __N);

/* Calculates the dot product of a single-precision vector. */
void vDSP_dotpr(const float *__A, vDSP_Stride __IA, const float *__B, vDSP_Stride __IB, float *__C,
                vDSP_Length __N);

/* Vector multiplication; single precision. */
void vDSP_vmul(const float *__A, vDSP_Stride __IA, const float *__B, vDSP_Stride __IB, float *__C,
               vDSP_Stride __IC, vDSP_Length __N);

/* Vector divide; single precision. */
void vDSP_vdiv(const float *__B, vDSP_Stride __IB, const float *__A, vDSP_Stride __IA, float *__C,
               vDSP_Stride __IC, vDSP_Length __N);

/* Multiplies two complex vectors, optionally conjugating one of them; single precision. */
void vDSP_zvmul(const DSPSplitComplex *__A, vDSP_Stride __IA, const DSPSplitComplex *__B,
                vDSP_Stride __IB, const DSPSplitComplex *__C, vDSP_Stride __IC, vDSP_Length __N,
                int __Conjugate);

/* Complex vector absolute values; single precision. */
void vDSP_zvabs(const DSPSplitComplex *__A, vDSP_Stride __IA, float *__C, vDSP_Stride __IC,
                vDSP_Length __N);

/* Vector generate by extrapolation and interpolation; single precision. */
void vDSP_vgenp(const float *__A, vDSP_Stride __IA, const float *__B, vDSP_Stride __IB, float *__C,
                vDSP_Stride __IC, vDSP_Length __N, vDSP_Length __M);

/* Vector scalar multiply */
void vDSP_vsmul(const float *__A, vDSP_Stride __IA, const float *__B, float *__C, vDSP_Stride __IC,
                vDSP_Length __N);

/* Vector scalar divide; single precision. */
void vDSP_vsdiv(const float *__A, vDSP_Stride __IA, const float *__B, float *__C, vDSP_Stride __IC,
                vDSP_Length __N);

/* Vector-scalar add; single precision. */
void vDSP_vsadd(const float *__A, vDSP_Stride __IA, const float *__B, float *__C, vDSP_Stride __IC,
                vDSP_Length __N);

/* Vector multiply and scalar add; single precision. */
void vDSP_vmsa(const float *__A, vDSP_Stride __IA, const float *__B, vDSP_Stride __IB,
               const float *__C, float *__D, vDSP_Stride __ID, vDSP_Length __N);

/* Adds a single-precision vector to the product of a single-precision scalar value and a single-precision vector. */
void
vDSP_vsma(const float *__A, vDSP_Stride __IA, const float *__B, const float *__C, vDSP_Stride __IC,
          float *__D, vDSP_Stride __ID, vDSP_Length __N);

/* Single-precision real vector-scalar multiply and scalar add. */
void vDSP_vsmsa(const float *__A, vDSP_Stride __IA, const float *__B, const float *__C, float *__D,
                vDSP_Stride __ID, vDSP_Length __N);

/* Single-precision real vector-scalar multiply, vector-scalar multiply, and vector add. */
void vDSP_vsmsma(const float *__A, vDSP_Stride __IA, const float *__B, const float *__C,
                 vDSP_Stride __IC, const float *__D, float *__E, vDSP_Stride __IE, vDSP_Length __N);

/* Adds two vectors; single precision. */
void vDSP_vadd(const float *__A, vDSP_Stride __IA, const float *__B, vDSP_Stride __IB, float *__C,
               vDSP_Stride __IC, vDSP_Length __N);

/* Vector subtract; single precision. */
void vDSP_vsub(const float *__B, vDSP_Stride __IB, const float *__A, vDSP_Stride __IA, float *__C,
               vDSP_Stride __IC, vDSP_Length __N);

/* Vector convert power or amplitude to decibels; single precision. */
void vDSP_vdbcon(const float *__A, vDSP_Stride __IA, const float *__B, float *__C, vDSP_Stride __IC,
                 vDSP_Length __N, unsigned int __F);

/* Vector clip; single precision. */
void vDSP_vclip(const float *__A, vDSP_Stride __IA, const float *__B, const float *__C, float *__D,
                vDSP_Stride __ID, vDSP_Length __N);

/* Calculates single-precision vector threshold to the specified range. */
void vDSP_vthr(const float *__A, vDSP_Stride __IA, const float *__B, float *__C, vDSP_Stride __IC,
               vDSP_Length __N);

/* Vector maximum value with index; single precision. */
void vDSP_maxvi(const float *__A, vDSP_Stride __IA, float *__C, vDSP_Length *__I, vDSP_Length __N);

/* Vector root-mean-square; single precision. */
void vDSP_rmsqv(const float *__A, vDSP_Stride __IA, float *__C, vDSP_Length __N);

/* Copies the contents of an interleaved complex vector C to a split complex vector Z; single precision. */
void vDSP_ctoz(const DSPComplex *__C, vDSP_Stride __IC, const DSPSplitComplex *__Z,
               vDSP_Stride __IZ, vDSP_Length __N);

/* Copies the contents of a split complex vector Z to an interleaved complex vector C; single precision. */
void vDSP_ztoc(const DSPSplitComplex *__Z, vDSP_Stride __IZ, DSPComplex *__C, vDSP_Stride __IC,
               vDSP_Length __N);

/* Vector sum; single precision. */
void vDSP_sve(const float *__A, vDSP_Stride __I, float *__C, vDSP_Length __N);

/* Vector sum of squares; single precision. */
void vDSP_svesq(const float *__A, vDSP_Stride __IA, float *__C, vDSP_Length __N);

/* Converts an array of signed 16-bit integers to single-precision floating-point values. */
void vDSP_vflt16(const short *__A, vDSP_Stride __IA, float *__C, vDSP_Stride __IC, vDSP_Length __N);

/* Converts an array of single-precision floating-point values to signed 16-bit integer values, rounding towards zero. */
void vDSP_vfix16(const float *__A, vDSP_Stride __IA, short *__C, vDSP_Stride __IC, vDSP_Length __N);

/* Builds a data structure that contains precalculated data for use by single-precision FFT functions. */
FFTSetup vDSP_create_fftsetup(vDSP_Length __Log2n, FFTRadix __Radix);

/* Frees an existing single-precision FFT data structure. */
void vDSP_destroy_fftsetup(FFTSetup __setup);

/* Computes an in-place single-precision real discrete Fourier transform */
void
vDSP_fft_zrip(FFTSetup __Setup, const DSPSplitComplex *__C, vDSP_Stride __IC, vDSP_Length __Log2N,
              FFTDirection __Direction);

/* Computes an in-place single-precision complex discrete Fourier transform */
void
vDSP_fft_zip(FFTSetup __Setup, const DSPSplitComplex *__C, vDSP_Stride __IC, vDSP_Length __Log2N,
             FFTDirection __Direction);

/* Performs an out-of-place multiplication of two matrices; single precision. */
void vDSP_mmul(const float *__A, vDSP_Stride __IA, const float *__B, vDSP_Stride __IB, float *__C,
               vDSP_Stride __IC, vDSP_Length __M, vDSP_Length __N, vDSP_Length __P);


/* Complex vector conjugate; single precision. */
void vDSP_zvconj(const DSPSplitComplex *__A, vDSP_Stride __IA, const DSPSplitComplex *__C,
                 vDSP_Stride __IC, vDSP_Length __N);

/* Multiplies two matrices of complex numbers; out-of-place; single precision. */
void vDSP_zmmul(const DSPSplitComplex *__A, vDSP_Stride __IA, const DSPSplitComplex *__B,
                vDSP_Stride __IB, const DSPSplitComplex *__C, vDSP_Stride __IC, vDSP_Length __M,
                vDSP_Length __N, vDSP_Length __P);

/* Multiplies a complex vector by a real vector ; single precision. */
void vDSP_zrvmul(const DSPSplitComplex *__A, vDSP_Stride __IA, const float *__B, vDSP_Stride __IB,
                 const DSPSplitComplex *__C, vDSP_Stride __IC, vDSP_Length __N);

/* Divides a single-precision complex vector by a single-precision real vector. */
void vDSP_zrvdiv(const DSPSplitComplex *__A, vDSP_Stride __IA, const float *__B, vDSP_Stride __IB,
                 const DSPSplitComplex *__C, vDSP_Stride __IC, vDSP_Length __N);

/* Adds two complex vectors; single precision. */
void vDSP_zvadd(const DSPSplitComplex *__A, vDSP_Stride __IA, const DSPSplitComplex *__B,
                vDSP_Stride __IB, const DSPSplitComplex *__C, vDSP_Stride __IC, vDSP_Length __N);

/* Subtracts two single-precision complex vectors. */
void vDSP_zvsub(const DSPSplitComplex *__A, vDSP_Stride __IA, const DSPSplitComplex *__B,
                vDSP_Stride __IB, const DSPSplitComplex *__C, vDSP_Stride __IC, vDSP_Length __N);

/* Complex vector multiply and add; single precision. */
void vDSP_zvma(const DSPSplitComplex *__A, vDSP_Stride __IA, const DSPSplitComplex *__B,
               vDSP_Stride __IB, const DSPSplitComplex *__C, vDSP_Stride __IC,
               const DSPSplitComplex *__D, vDSP_Stride __ID, vDSP_Length __N);

/* Complex vector magnitudes squared; single precision. */
void vDSP_zvmags(const DSPSplitComplex *__A, vDSP_Stride __IA, float *__C, vDSP_Stride __IC,
                 vDSP_Length __N);

/* Creates a transposed matrix C from a source matrix A; single precision. */
void vDSP_mtrans(const float *__A, vDSP_Stride __IA, float *__C, vDSP_Stride __IC, vDSP_Length __M,
                 vDSP_Length __N);

/* Difference equation, 2 poles, 2 zeros; single precision. */
void vDSP_deq22(const float *__A, vDSP_Stride __IA, const float *__B, float *__C, vDSP_Stride __IC,
                vDSP_Length __N);

/* Performs either correlation or convolution on two real vectors; single-precision. */
/* Note: better use an FFT implementation for large __N or __P */
void vDSP_conv(const float *__A, vDSP_Stride __IA, const float *__F, vDSP_Stride __IF,
               float *__C, vDSP_Stride __IC, vDSP_Length __N, vDSP_Length __P);

/* Calculates the cosine of each element in an array of single-precision values. */
void vvcosf(float *, const float *, const int *);

/* Calculates the sine of each element in an array of single-precision values. */
void vvsinf(float *, const float *, const int *);

/* Calculates the base 10 logarithm of each element in an array of single-precision values. */
void vvlog10f(float *, const float *, const int *);

/* Calculates the square root of each element in an array of single-precision values. */
void vvsqrtf(float *, const float *, const int *);

/* Raises each element in an array to the power of the corresponding element in a second array of single-precision values. */
void vvpowf(float *, const float *, const float *, const int *);

/* Calculates e raised to the power of each element in an array of single-precision values. */
void vvexpf(float *y, const float *x, const int *n);

#ifdef __cplusplus
}
#endif
#endif
