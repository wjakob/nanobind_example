#pragma once
#include <stdint.h>

#ifdef _MSC_VER
#define INLINE __forceinline
#else
#define INLINE __attribute__((always_inline)) inline
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

} // namespace tvbk
