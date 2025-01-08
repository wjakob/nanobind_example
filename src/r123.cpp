#include <stdint.h>

#ifdef _MSC_VER
#define INLINE __forceinline
#else
#define INLINE __attribute__((always_inline)) inline
#endif

static uint64_t INLINE sfc64(uint64_t s[4])
{
    uint64_t r = s[0] + s[1] + s[3]++;
    s[0] = (s[1] >> 11) ^ s[1];
    s[1] = (s[2] <<  3) + s[2];
    s[2] = r + (s[2]<<24 | s[2] >>40);
    return r;
}

static float INLINE randn1(uint64_t s[4])
{
    uint64_t u = sfc64(s);
    double x = __builtin_popcount(u>>32);
    x += (uint32_t)u * (1 / 4294967296.);
    x -= 16.5;
    x *= 0.3517262290563295;
    return (float) x;
}

extern "C" {
void tvbk_randn(uint64_t *seed, int n, float *out)
{
    for (int i=0; i<n; i++) out[i] = randn1(seed);
}
}
