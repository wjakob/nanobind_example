#pragma once

#include <stdint.h>

void tvbk_mm8_ref(float *A, float *B, float *C, float *b);
void tvbk_mm8_fast(float *A, float *B, float *C, float *b);

typedef struct tvbk_params tvbk_params;

struct tvbk_params {
  const uint32_t count;
  const float *const values;
};

/* a afferent coupling buffer into which the cx functions
   accumulate their results */
typedef struct tvbk_cx tvbk_cx;

struct tvbk_cx {
  /* values for 1st and 2nd Heun stage respectively.
     each shaped (num_node, ) */
  float *const cx1;
  float *const cx2;
  /* delay buffer (num_node, num_time)*/
  float *const buf;
  const uint32_t num_node;
  const uint32_t num_time; // horizon, power of 2
};

typedef struct tvbk_conn tvbk_conn;

struct tvbk_conn {
  const int num_node;
  const int num_nonzero;
  const int num_cvar;
  const float *const weights;    // (num_nonzero,)
  const uint32_t *const indices; // (num_nonzero,)
  const uint32_t *const indptr;  // (num_nodes+1,)
  const uint32_t *const idelays; // (num_nonzero,)
};

/* not currently used */
// typedef struct tvbk_sim tvbk_sim;
// struct tvbk_sim {
//   // keep invariant stuff at the top, per sim stuff below
//   const int rng_seed;
//   const int num_node;
//   const int num_svar;
//   const int num_time;
//   const int num_params;
//   const int num_spatial_params;
//   const float dt;
//   const int oversample; // TODO "oversample" for stability,
//   const int num_skip;   // skip per output sample
//   float *z_scale;       // (num_svar), sigma*sqrt(dt)

//   // parameters
//   const tvbk_params global_params;
//   const tvbk_params spatial_params;

//   float *state_trace; // (num_time//num_skip, num_svar, num_nodes)
//   float *states;      // full states (num_svar, num_nodes)

//   const tvbk_conn conn;
// };

void tvbk_cx_j(const tvbk_cx *cx, const tvbk_conn *conn, uint32_t t);
void tvbk_cx_i(const tvbk_cx *cx, const tvbk_conn *conn, uint32_t t);
void tvbk_cx_nop(const tvbk_cx *cx, const tvbk_conn *conn, uint32_t t);

void tvbk_cx_i_b8(const tvbk_cx *cx, const tvbk_conn *conn, uint32_t t);

void tvbk_randn(uint32_t seed1, uint32_t count, float *out);

/* draft towards ode.c */
typedef struct tvbk_sim tvbk_sim;
struct tvbk_sim {
  uint32_t horizon, num_node, horizon_minus_1;
  float dt, *states, *delay_buffer, *params;
};

#define tvbk_declare_ode_step(model)                                           \
  void tvbk_##model(const tvbk_sim *s, const int i_node, const int i_time,     \
                    float *cx1, float *cx2)

tvbk_declare_ode_step(linear);
tvbk_declare_ode_step(bistable);

/* draft towards a multithreaded work queue */
/* TODO probably generate this based on the function declarations? */
typedef struct tvbk_op tvbk_op;

struct tvbk_op {
  enum {
    TVBK_OP_NOP,
    TVBK_OP_TICK,
    TVBK_OP_CX_J,
    TVBK_OP_CX_I,
    TVBK_OP_RANDN,
    TVBK_OP_SEQ,
    TVBK_OP_LOOP
  } tag;
  union {
    /* tick */
    struct {
      uint32_t t;
    } tick;
    /* cx_j, cx_i */
    struct {
      const tvbk_cx *cx;
      const tvbk_conn *conn;
      const uint32_t *t; /* points to some tick value */
    } cx;
    /* randn */
    struct {
      uint32_t seed1, count;
      const uint32_t *t;
      float *out;
    } randn;
    /* ops run & loop*/
    struct {
      uint32_t len, loops;
      tvbk_op *ops;
    } seq;
  };
};

void tvbk_ops_run(uint32_t len, tvbk_op *seq);
#define tvbk_ops_run1(op) tvbk_ops_run(1, &op);
