#pragma once

#include <stdint.h>
#include <stdbool.h>
#include <math.h>

/* define a bunch of kernels useful for writing higher level functions.
  Isolating simd-size loops inside tiny functions, in conjuction with
  pragma omp simd, makes it easy for gcc to emit simple simd instructions.
  the forced inlining then allows the compiler to see that the pointers
  aren't aliased, etc, removing further overhead.
*/
#ifndef _MSC_VER

#ifdef NOINLINE
#define INLINE __attribute__((noinline))
#else
#define INLINE __attribute__((always_inline)) inline
#endif

#define kernel(name, width, expr, args...) \
INLINE static void name ## width (args) \
{ \
  _Pragma("omp simd") \
  for (int i=0; i < width; i++) \
    expr;\
}
#else
#define INLINE

#define kernel(name, width, expr, ...) \
INLINE static void name ## width (__VA_ARGS__) \
{ \
  _Pragma("loop(hint_parallel(8))") \
  _Pragma("loop(ivdep)") \
  for (int i=0; i < width; i++) \
    expr;\
}

#endif

/* simple stuff */
kernel(inc,      8, x[i]                    += w*y[i], float *x, float *y, float w)
kernel(adds,     8, x[i]                         += a, float *x, float a)
kernel(load,     8, x[i]                       = y[i], float *x, float *y)
kernel(zero,     8, x[i]                        = 0.f, float *x)

/* Heun stages */
kernel(heunpred, 8, xi[i]           = x[i] + dt*dx[i], float *x, float *xi, float *dx, float dt)
kernel(heuncorr, 8, x[i] += dt*0.5f*(dx1[i] + dx2[i]), float *x, float *dx1, float *dx2, float dt)
kernel(sheunpred, 8, xi[i]           = x[i] + dt*dx[i] + z[i], float *x, float *xi, float *dx, float *z, float dt)
kernel(sheuncorr, 8, x[i] += dt*0.5f*(dx1[i] + dx2[i]) + z[i], float *x, float *dx1, float *dx2, float *z, float dt)

/* activation functions */
kernel(sigm,     8, x[i] = 1.0f/(1.0f + expf(y[i])), float *x, float *y)
kernel(heavi,    8, x[i] = y[i] >= 0.0f ? 1.0f : 0.0f, float *x, float *y)
kernel(relu,     8, x[i] = y[i] >= 0.0f ? y[i] : 0.0f, float *x, float *y)
kernel(lrelu,    8, x[i] = y[i] >= 0.0f ? y[i] : 0.01f*y[i], float *x, float *y)

/* transcendentals; vectorized by gcc w/ libmvec;
   need sleef or similar elsewhere */
kernel(fabsf, 8, x[i] = fabsf(y[i]), float *x, float *y)
kernel(logf,  8, x[i] = logf(y[i]), float *x, float *y)
kernel(powfp,  8, x[i] = powf(y[i], z[i]), float *x, float *y, float *z)
kernel(powf,  8, x[i] = powf(y[i], z), float *x, float *y, float z)
kernel(expf,  8, x[i] = expf(y[i]), float *x, float *y)
kernel(exp2f, 8, x[i] = exp2f(y[i]), float *x, float *y)
kernel(sqrtf, 8, x[i] = sqrtf(y[i]), float *x, float *y)
kernel(sinf,  8, x[i] = sinf(y[i]), float *x, float *y)
kernel(cosf,  8, x[i] = cosf(y[i]), float *x, float *y)
kernel(tanf,  8, x[i] = tanf(y[i]), float *x, float *y)
kernel(erff,  8, x[i] = erff(y[i]), float *x, float *y)

/* short length dot product accumulates, so doesn't fit into 
   macro above */
INLINE static void dot8(float *dst, float *x, float *y)
{
    float acc=0.f;
    #pragma omp simd reduction(+:acc)
    for (int i=0; i<8; i++) acc+=x[i]*y[i];
    *dst = acc;
}
