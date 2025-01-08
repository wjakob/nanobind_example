#include <stdint.h>
#include <math.h>


#include "Random123/philox.h"
/* doesn't have sincosf ? */
#ifndef _MSC_VER
#include "Random123/boxmuller.hpp"
#endif

#define M_PI_2 ((float) (2.0f * 3.141592653589f))

extern "C" {
void tvbk_randn(uint32_t seed1, uint32_t count, float *out) {
  philox4x32_ukey_t uk = {{}};
  uk.v[0] = seed1;
  philox4x32_key_t k = philox4x32keyinit(uk);
  typedef struct { float x, y; } float2;
  /* loop generates 4-way SIMD x 4 numbers / lane */
  int n16 = count / 16;
  for (int j = 0; j < n16; j++) {
    float *oj = out + j * 16;
#pragma omp simd
    for (int i = 0; i < 4; i++) {
      philox4x32_ctr_t c = {{}};
      c.v[0] = i; /* some loop-dependent application variable */
      c.v[1] = j; /* another loop-dependent application variable */
      philox4x32_ctr_t r = philox4x32(c, k);

#ifndef _MSC_VER
      r123::float2 o1, o2;
      o1 = r123::boxmuller(r.v[0], r.v[1]);
      o2 = r123::boxmuller(r.v[2], r.v[3]);
#else
      float2 o1, o2;
      float s1 = sqrtf(-2.0f * r.v[0]);
      float s2 = sqrtf(-2.0f * r.v[2]);
      o1.x = s1 * cosf(M_PI_2 * r.v[1]);
      o1.y = s1 * sinf(M_PI_2 * r.v[1]);
      o2.x = s2 * cosf(M_PI_2 * r.v[3]);
      o2.y = s2 * sinf(M_PI_2 * r.v[3]);
#endif

      oj[i] = o1.x;
      oj[4 + i] = o1.y;
      oj[8 + i] = o2.x;
      oj[12 + i] = o2.y;
    }
  }
}
}
