#include <stdint.h>

#include "util.h"
#include "tvbk.h"


#define WIDTH 4

/* these macros provide the boilerplate for a ODE Heun step
   implementation of functions liek tvbk_bistable, etc.
   They evaluate the ODEs for WIDTH nodes at a time. */

/* clang doesn't like these pragmas, wtf
    _Pragma("GCC unroll _nsvar") \
    _Pragma("GCC ivdep") \
    */



#define make_ode_step(model, nsvar, width) \
tvbk_declare_ode_step(model) \
{ \
    const int _nsvar = nsvar; \
    float x[nsvar*width], xi[nsvar*width], dx1[nsvar*width], dx2[nsvar*width], z[nsvar*width]; \
    for (int svar=0; svar < nsvar; svar++) \
    { \
        load ## width(x+svar*width, s->states+width*(i_node + s->num_node*svar)); \
        zero ## width(xi+svar*width); \
        zero ## width(dx1+svar*width); \
        zero ## width(dx2+svar*width); \
        zero ## width(z+svar*width); \
    } \
     \
    dfun_ ## model ## width(x, x+width, cx1, s->params, s->params+width, s->params+2*width, dx1, dx1+width); \
\
    for (int svar=0; svar < nsvar; svar++) \
        heunpred8(x+svar*width, xi+svar*width, dx1+svar*width, s->dt); \
       \
    dfun_ ## model ## width(xi, xi+width, cx2, s->params, s->params+width, s->params+2*width, dx2, dx2+width); \
\
    for (int svar=0; svar < nsvar; svar++) \
    { \
        heuncorr ## width(x+svar*width, dx1+svar*width, dx2+svar*width, s->dt); \
        load ## width(s->states+width*(i_node + s->num_node*svar), x+svar*width); \
    } \
\
    int write_time = i_time & s->horizon_minus_1; \
    load ## width(s->delay_buffer + width*(i_node*s->horizon + write_time),  \
          s->states+width*(i_node + s->num_node*0)); \
}

kernel(
  dfun_bistable, 8, {
    dx[i] = tau[i] * (x[i] - x[i]*x[i]*x[i]/3 + y[i]);
    dy[i] = (1/tau[i]) * (a[i] + k[i]*cx[i] - x[i]);
  },
  const float *x, const float *y, const float *cx, const float *a, const float *tau, const float *k,
  float *dx, float *dy)

make_ode_step(bistable, 2, 8)

kernel(
  dfun_linear, 8, {
    dx[i] = -x[i] + cx[i];
    dy[i] = -y[i];
},
  const float *x, const float *y, const float *cx, const float *a, const float *tau, const float *k,
  float *dx, float *dy
)

make_ode_step(linear, 2, 8)
