#include "tvbk.h"
#include "util.h"

// setup pointers b1 & b2 to delay_buffer to read from
static void INLINE prep_ij(the_sim *s, const int i_time, const int nz,
                           float **b1, float **b2, float *w) {
  w[0] = s->weights[nz];
  float *b0 = s->delay_buffer + s->horizon * s->indices[nz] * 8;
  int t0 = s->horizon + i_time - s->idelays[nz];
  b1[0] = b0 + ((t0 + 0) & s->horizon_minus_1) * 8;
  b2[0] = b0 + ((t0 + 1) & s->horizon_minus_1) * 8;
}

// do one non-zero increment for cx1 & cx2
static void INLINE csr_ij(the_sim *s, const int nz, const int i_time,
                          float *cx1, float *cx2) {
  float *b1, *b2, w;
  prep_ij(s, i_time, nz, &b1, &b2, &w);
  inc8(cx1, b1, w);
  inc8(cx2, b2, w);
}

// vectorized prep_ij but poitners are 8 bytes, so only do 4 at time
static void INLINE prep4_ij(the_sim *s, const int i_time, const int nz,
                            float **b1, float **b2, float *w) {
  const float *weights = s->weights + nz;
  const int *indices = s->indices + nz;
  const int *idelays = s->idelays + nz;
  float *buf = s->delay_buffer;
#pragma omp simd
  for (int i = 0; i < 4; i++) {
    w[i] = weights[i];
    int t0 = s->horizon + i_time - idelays[i];
    b1[i] =
        buf + s->horizon * indices[i] * 8 + ((t0 + 0) & s->horizon_minus_1) * 8;
    b2[i] =
        buf + s->horizon * indices[i] * 8 + ((t0 + 1) & s->horizon_minus_1) * 8;
  }
}

static void INLINE csr4_ij(the_sim *s, const int nz, const int i_time,
                           float *cx1, float *cx2) {
  float *b1[4], *b2[4], w[4];
  prep4_ij(s, i_time, nz, b1, b2, w);
  inc8(cx1, b1[0], w[0]);
  inc8(cx2, b2[0], w[0]);
  inc8(cx1, b1[1], w[1]);
  inc8(cx2, b2[1], w[1]);
  inc8(cx1, b1[2], w[2]);
  inc8(cx2, b2[2], w[2]);
  inc8(cx1, b1[3], w[3]);
  inc8(cx2, b2[3], w[3]);
}

static void INLINE csr_node(const tvbk_cx *cx, const tvbk_conn *conn,
                            int i_time, int i_node, float *cx1, float *cx2) {
  zero8(cx1);
  zero8(cx2);

  int i0 = s->indptr[i_node];
  int i1 = s->indptr[i_node + 1];
  int nnz = i1 - i0;
  int n4 = nnz / 4;
  for (int i_n4 = 0; i_n4 < n4; i_n4++)
    csr4_ij(s, i0 + i_n4 * 4, i_time, cx1, cx2);
  nnz -= (n4 * 4);
  for (int nz = i1 - nnz; nz < i1; nz++)
    csr_ij(s, nz, i_time, cx1, cx2);
}

void tvbk_cx_i_b8(const tvbk_cx *cx, const tvbk_conn *c, uint32_t t) {
  float cx1[8], cx2[8];
  for (uint32_t i = 0; i < cx->num_node; i++) {
    csr_node(cx, c, t, i, cx1, cx2);
    load8(cx->cx1 + i * 8, cx1);
    load8(cx->cx2 + i * 8, cx2);
  }
}
