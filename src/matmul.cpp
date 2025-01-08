#include <stdio.h>

#include "util.h"

#define N 8
extern "C" {
void tvbk_mm8_ref(float A[N * N], float B[N * N], float C[N * N], float b[N]) {
  for (int i = 0; i < N; i++) {
    /* load ith row of A for reuse */
    float ai[N];
    for (int k = 0; k < N; k++)
      ai[k] = A[i * N + k];
    /* loop over columns */
    for (int j = 0; j < N; j++) {
      float acc = 0.f;
      for (int k = 0; k < N; k++)
        acc += ai[k] * B[k * N + j];
      C[i * N + j] = acc; // + b[i];
    }
  }
}

/* TODO might want INLINE on this */
void tvbk_mm8_fast(float A[N * N], float B[N * N], float C[N * N], float b[N]) {
#ifdef _MSC_VER
#pragma loop(unroll)
#else
#pragma GCC unroll(8)
#endif
  for (int i = 0; i < N; i++) {
    float acc[N], arow[N], brow[N];
    zero8(acc);
    load8(arow, A + i * N);
    for (int k = 0; k < N; k++) {
      // load8(brow, B+k*N);
      inc8(acc, B + k * N, arow[k]);
    }
    // printf("acc %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f\n",
    // acc[0], acc[1], acc[2], acc[3], acc[4], acc[5], acc[6], acc[7]);
    load8(C + i * N, acc);
  }
}
}

// TODO tiled mm for neural ode based on mm8