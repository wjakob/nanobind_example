#pragma once
#include <stdint.h>
#include <math.h>

// at -O3 -fopen-simd, these kernels result in compact asm
#ifdef _MSC_VER
#include <intrin.h>
#define INLINE __forceinline
#define kernel(name, expr, ...) \
template <int width> INLINE static void name (__VA_ARGS__) \
{ \
  _Pragma("loop(hint_parallel(8))") \
  _Pragma("loop(ivdep)") \
  for (int i=0; i < width; i++) \
    expr;\
}
#else
#define INLINE __attribute__((always_inline)) inline
#define kernel(name, expr, args...) \
template <int width> INLINE static void name (args) \
{ \
  _Pragma("omp simd") \
  for (int i=0; i < width; i++) \
    expr;\
}
#endif

namespace tvbk {

struct cx {
  /* values for 1st and 2nd Heun stage respectively.
     each shaped (num_node, ) */
  float *cx1;
  float *cx2;
  /* delay buffer (num_node, num_time)*/
  float *buf;
  const uint32_t num_node;
  const uint32_t num_time; // horizon, power of 2
  cx(const uint32_t num_node, const uint32_t num_time)
      : num_node(num_node), num_time(num_time), cx1(new float[num_node]),
        cx2(new float[num_node]), buf(new float[num_node * num_time]) {}
};

struct conn {
  const uint32_t num_node;
  const uint32_t num_nonzero;
  const float *weights;    // (num_nonzero,)
  const uint32_t *indices; // (num_nonzero,)
  const uint32_t *indptr;  // (num_nodes+1,)
  const uint32_t *idelays; // (num_nonzero,)
  conn(const uint32_t num_node, const uint32_t num_nonzero)
      : num_node(num_node), num_nonzero(num_nonzero),
        weights(new float[num_nonzero]), indices(new uint32_t[num_nonzero]),
        indptr(new uint32_t[num_node + 1]), idelays(new uint32_t[num_nonzero]) {
  }
};

INLINE void cx_all_j(const cx &cx, const conn &c, uint32_t t, uint32_t j) {
  uint32_t wrap_mask = cx.num_time - 1; // assume num_time is power of 2
  float *const buf = cx.buf + j * cx.num_time;
  uint32_t th = t + cx.num_time;
#pragma omp simd
  for (uint32_t l = c.indptr[j]; l < c.indptr[j + 1]; l++) {
    uint32_t i = c.indices[l];
    float w = c.weights[l];
    uint32_t d = c.idelays[l];
    uint32_t p1 = (th - d) & wrap_mask;
    uint32_t p2 = (th - d + 1) & wrap_mask;
    cx.cx1[i] += w * buf[p1];
    cx.cx2[i] += w * buf[p2];
  }
}

INLINE void cx_j(const cx &cx, const conn &c, uint32_t t) {
#pragma omp simd
  for (int i = 0; i < c.num_node; i++)
    cx.cx1[i] = cx.cx2[i] = 0.0f;
  for (int j = 0; j < c.num_node; j++)
    cx_all_j(cx, c, t, j);
}

INLINE void cx_i(const cx &cx, const conn &c, uint32_t t) {
  uint32_t wrap_mask = cx.num_time - 1; // assume num_time is power of 2
  uint32_t th = t + cx.num_time;
#pragma omp simd
  for (int i = 0; i < c.num_node; i++) {
    float cx1 = 0.f, cx2 = 0.f;
    for (uint32_t l = c.indptr[i]; l < c.indptr[i + 1]; l++) {
      uint32_t j = c.indices[l];
      float w = c.weights[l];
      uint32_t d = c.idelays[l];
      uint32_t p1 = (th - d) & wrap_mask;
      uint32_t p2 = (th - d + 1) & wrap_mask;
      cx1 += w * cx.buf[j * cx.num_time + p1];
      cx2 += w * cx.buf[j * cx.num_time + p2];
    }
    cx.cx1[i] = cx1;
    cx.cx2[i] = cx2;
  }
}

INLINE uint64_t sfc64(uint64_t s[4]) {
  uint64_t r = s[0] + s[1] + s[3]++;
  s[0] = (s[1] >> 11) ^ s[1];
  s[1] = (s[2] << 3) + s[2];
  s[2] = r + (s[2] << 24 | s[2] >> 40);
  return r;
}

INLINE float randn1(uint64_t s[4]) {
  uint64_t u = sfc64(s);
#ifdef _MSC_VER
  // TODO check, cf https://stackoverflow.com/a/42913358
  double x = __popcnt64(u >> 32);
#else
  double x = __builtin_popcount(u >> 32);
#endif
  x += (uint32_t)u * (1 / 4294967296.);
  x -= 16.5;
  x *= 0.3517262290563295;
  return (float)x;
}

INLINE void randn(uint64_t *seed, int n, float *out) {
  #pragma omp simd
  for (int i = 0; i < n; i++)
    out[i] = randn1(seed);
}

/* simple stuff */
kernel(inc,      x[i]                    += w*y[i], float *x, float *y, float w)
kernel(adds,     x[i]                         += a, float *x, float a)
kernel(load,     x[i]                       = y[i], float *x, float *y)
kernel(zero,     x[i]                        = 0.f, float *x)

/* Heun stages */
kernel(heunpred, xi[i]           = x[i] + dt*dx[i], float *x, float *xi, float *dx, float dt)
kernel(heuncorr, x[i] += dt*0.5f*(dx1[i] + dx2[i]), float *x, float *dx1, float *dx2, float dt)
kernel(sheunpred, xi[i]           = x[i] + dt*dx[i] + z[i], float *x, float *xi, float *dx, float *z, float dt)
kernel(sheuncorr, x[i] += dt*0.5f*(dx1[i] + dx2[i]) + z[i], float *x, float *dx1, float *dx2, float *z, float dt)

/* activation functions */
kernel(sigm,     x[i] = 1.0f/(1.0f + expf(y[i])), float *x, float *y)
kernel(heavi,    x[i] = y[i] >= 0.0f ? 1.0f : 0.0f, float *x, float *y)
kernel(relu,     x[i] = y[i] >= 0.0f ? y[i] : 0.0f, float *x, float *y)
kernel(lrelu,    x[i] = y[i] >= 0.0f ? y[i] : 0.01f*y[i], float *x, float *y)

/* transcendentals; vectorized by gcc w/ libmvec;
   need sleef or similar elsewhere */
kernel(kfabsf, x[i] = fabsf(y[i]), float *x, float *y)
kernel(klogf,  x[i] = logf(y[i]), float *x, float *y)
kernel(kpowfp,  x[i] = powf(y[i], z[i]), float *x, float *y, float *z)
kernel(kpowf,  x[i] = powf(y[i], z), float *x, float *y, float z)
kernel(kexpf,  x[i] = expf(y[i]), float *x, float *y)
kernel(kexp2f, x[i] = exp2f(y[i]), float *x, float *y)
kernel(ksqrtf, x[i] = sqrtf(y[i]), float *x, float *y)
kernel(ksinf,  x[i] = sinf(y[i]), float *x, float *y)
kernel(kcosf,  x[i] = cosf(y[i]), float *x, float *y)
kernel(ktanf,  x[i] = tanf(y[i]), float *x, float *y)
kernel(kerff,  x[i] = erff(y[i]), float *x, float *y)

/* short length dot product accumulates, so doesn't fit into
   macro above */
template <int width>
INLINE static void dot(float *dst, float *x, float *y)
{
    float acc=0.f;
    #pragma omp simd reduction(+:acc)
    for (int i=0; i<width; i++) acc+=x[i]*y[i];
    *dst = acc;
}

/* WIP

template <uint8_t nsvar, uint8_t width, typename dfun>
struct model
{
  const uint32_t num_node, horizon, horizon_minus_1;
  float *states, *params, dt;
  const cx &cx;
  
  void step(const uint32_t i_node, const uint32_t i_time,
            const float* cx1, const float* cx2)
  {
    float x[nsvar*width], xi[nsvar*width], dx1[nsvar*width], dx2[nsvar*width], z[nsvar*width];
    for (int svar=0; svar < nsvar; svar++)
    {
        load<width>(x+svar*width, states+width*(i_node + num_node*svar));
        zero<width>(xi+svar*width);
        zero<width>(dx1+svar*width);
        zero<width>(dx2+svar*width);
        zero<width>(z+svar*width);
    }
    
    dfun(x, x+width, cx1, params, params+width, params+2*width, dx1, dx1+width);

    for (int svar=0; svar < nsvar; svar++)
        heunpred<width>(x+svar*width, xi+svar*width, dx1+svar*width, dt);

    dfun(xi, xi + width, cx2, params, params + width, params + 2 * width, dx2,
         dx2 + width);
    for (int svar=0; svar < nsvar; svar++)
    {
        heuncorr<width>(x+svar*width, dx1+svar*width, dx2+svar*width, dt);
        load<width>(states+width*(i_node + num_node*svar), x+svar*width);
    }
    int write_time = i_time & horizon_minus_1;
    load<width>(cx.buf + width * (i_node * horizon + write_time),
                states + width * (i_node + num_node * 0));
  }
};

template <uint8_t width>
struct dfun_linear {
  void operator()(const float *x, const float *y, const float *cx,
                  const float *a, const float *tau, const float *k, float *dx,
                  float *dy) {
#pragma omp simd
    for (int i=0; i<width; i++) {
      dx[i] = -x[i] + cx[i];
      dy[i] = -y[i];
    }
  }
};

template <uint8_t width> struct linear : model<2,width,dfun_linear<width>> { };

void foobar() {
    linear<8> l {4, 5, 6};
    uint32_t m=2,n=5;
    float cx1[8], cx2[8];
    l.step(m, n, cx1, cx2);
}
*/

} // namespace tvbk
