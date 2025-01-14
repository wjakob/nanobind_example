#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "tvbk.hpp"

namespace nb = nanobind;
using namespace nb::literals;
typedef nb::ndarray<float, nb::numpy, nb::device::cpu, nb::shape<-1>, nb::c_contig> fvec;
typedef nb::ndarray<float, nb::numpy, nb::device::cpu, nb::shape<-1,-1>, nb::c_contig> fmat;
typedef nb::ndarray<float, nb::numpy, nb::device::cpu, nb::shape<-1,-1,-1>, nb::c_contig> farr3;
typedef nb::ndarray<float, nb::numpy, nb::device::cpu, nb::shape<-1,-1,-1,-1>, nb::c_contig> farr4;
typedef nb::ndarray<uint32_t, nb::numpy, nb::device::cpu, nb::shape<-1>, nb::c_contig> uvec;

template<typename shape> 
using farr = nb::ndarray<float, nb::numpy, nb::device::cpu, shape, nb::c_contig>;

NB_MODULE(tvbk_ext, m) {

  m.def(
      "randn",
      [](const uint32_t seed, fvec z) {
        uint64_t s[4] = {seed, seed, seed, seed};
        tvbk::randn(s, z.shape(0), (float *)z.data());
      },
      "seed"_a, "z"_a,
      "This functions generates normally distributed random numbers using a "
      "popcount trick.");

  nb::class_<tvbk::cx>(m, "Cx")
      .def(nb::init<uint32_t, uint32_t>(), "num_node"_a, "num_time"_a)
      .def_ro("num_node", &tvbk::cx::num_node)
      .def_ro("num_time", &tvbk::cx::num_time)
      .def_prop_ro("buf",
                   [](tvbk::cx &cx) {
                     return fmat(cx.buf, {cx.num_node, cx.num_time});
                   })
      .def_prop_ro("cx1",
                   [](tvbk::cx &cx) { return fvec(cx.cx1, {cx.num_node}); })
      .def_prop_ro("cx2",
                   [](tvbk::cx &cx) { return fvec(cx.cx2, {cx.num_node}); });

  nb::class_<tvbk::cx8>(m, "Cx8")
      .def(nb::init<uint32_t, uint32_t>(), "num_node"_a, "num_time"_a)
      .def_ro("num_node", &tvbk::cx8::num_node)
      .def_ro("num_time", &tvbk::cx8::num_time)
      .def_ro("num_item", &tvbk::cx8::num_item)
      .def_prop_ro("buf",
                   [](tvbk::cx8 &cx) {
                     return fmat(cx.buf, {cx.num_node, cx.num_time, cx.num_item});
                   })
      .def_prop_ro("cx1",
                   [](tvbk::cx8 &cx) { return fmat(cx.cx1, {cx.num_node, cx.num_item}); })
      .def_prop_ro("cx2",
                   [](tvbk::cx8 &cx) { return fmat(cx.cx2, {cx.num_node, cx.num_item}); });

  nb::class_<tvbk::cx8s>(m, "Cx8s")
      .def(nb::init<uint32_t, uint32_t, uint32_t>(), "num_node"_a, "num_time"_a, "num_batches"_a)
      .def_ro("num_node", &tvbk::cx8s::num_node)
      .def_ro("num_time", &tvbk::cx8s::num_time)
      .def_ro("num_item", &tvbk::cx8s::num_item)
      .def_ro("num_batch", &tvbk::cx8s::num_batch)
      .def_prop_ro("buf",
                   [](tvbk::cx8s &cx) {
                     return farr4(cx.buf, {cx.num_batch, cx.num_node, cx.num_time, cx.num_item});
                   })
      .def_prop_ro("cx1",
                   [](tvbk::cx8s &cx) { return farr3(cx.cx1, {cx.num_batch, cx.num_node, cx.num_item}); })
      .def_prop_ro("cx2",
                   [](tvbk::cx8s &cx) { return farr3(cx.cx2, {cx.num_batch, cx.num_node, cx.num_item}); })
    ;

  nb::class_<tvbk::conn>(m, "Conn")
      .def(nb::init<uint32_t, uint32_t>(), "num_node"_a, "num_nonzero"_a)
      .def_ro("num_node", &tvbk::conn::num_node)
      .def_ro("num_nonzero", &tvbk::conn::num_nonzero)
      .def_prop_ro("weights",
                   [](tvbk::conn &c) {
                     return fvec(const_cast<float *>(c.weights),
                                 {c.num_nonzero});
                   })
      .def_prop_ro("indices",
                   [](tvbk::conn &c) {
                     return uvec(const_cast<uint32_t *>(c.indices),
                                 {c.num_nonzero});
                   })
      .def_prop_ro("indptr",
                   [](tvbk::conn &c) {
                     return uvec(const_cast<uint32_t *>(c.indptr),
                                 {c.num_node + 1});
                   })
      .def_prop_ro("idelays", [](tvbk::conn &c) {
        return uvec(const_cast<uint32_t *>(c.idelays), {c.num_nonzero});
      });

  m.def(
      "cx_j",
      [](const tvbk::cx &cx, const tvbk::conn &conn, uint32_t t) {
        tvbk::cx_j(cx, conn, t);
      },
      "cx"_a, "conn"_a, "t"_a,
      "This function calculates the afferent coupling buffer.");

  m.def(
      "cx_j8",
      [](const tvbk::cx8 &cx8, const tvbk::conn &conn, uint32_t t) {
        tvbk::cx_j_b<8>(cx8, conn, t);
      },
      "cx8"_a, "conn"_a, "t"_a,
      "This function calculates batched afferent coupling buffer.");

  m.def("cxs8_j",
        [](const tvbk::cx8s &cxs8, const tvbk::conn &conn, uint32_t t) {
          tvbk::cx_j_bs<8>(cxs8, conn, t);
        },
        "cxs8"_a, "conn"_a, "t"_a,
        "This function calculates many batched afferent coupling buffer.");

  m.def("dfun_jr8",
    [](farr<nb::shape<tvbk::jr::num_svar,8>> &dx,
       farr<nb::shape<tvbk::jr::num_svar,8>> &x,
       farr<nb::shape<tvbk::jr::num_cvar,8>> &c,
       farr<nb::shape<tvbk::jr::num_parm,8>> &p)
    {  
      tvbk::jr::dfun<8>((float *)dx.data(), (float *)x.data(), (float *)c.data(), (float *)p.data());
    }, "dx"_a, "x"_a, "c"_a, "p"_a);

  m.def("dfun_mpr8",
    [](farr<nb::shape<tvbk::mpr::num_svar,8>> &dx,
       farr<nb::shape<tvbk::mpr::num_svar,8>> &x,
       farr<nb::shape<tvbk::mpr::num_cvar,8>> &c,
       farr<nb::shape<tvbk::mpr::num_parm,8>> &p)
    {  
      tvbk::mpr::dfun<8>((float *)dx.data(), (float *)x.data(), (float *)c.data(), (float *)p.data());
    }, "dx"_a, "x"_a, "c"_a, "p"_a);

  m.def("step_jr8",
    [](const tvbk::cxbs<8> &cx, const tvbk::conn &c,
    farr<nb::shape<-1,tvbk::jr::num_svar,-1,8>> &x,
    farr<nb::shape<-1,-1,tvbk::jr::num_parm,8>> &p,
    uint32_t t0, uint32_t nt, float dt)
    {
      // check x.shape[0] == p.shape[0] == cx.num_batch
      if (!(x.shape(0) == p.shape(0) && x.shape(0) == cx.num_batch))
        throw std::runtime_error("batch shapes don't match: check that x.shape[0] == p.shape[0] == cx.num_batch");
      // check x.shape[2] == cx.num_node == p.shape[1]
      if (!(x.shape(2) == cx.num_node && x.shape(2) == p.shape(1)))
        throw std::runtime_error("node shapes don't match: check that x.shape[2] == cx.num_node == p.shape[1]");
      tvbk::step_batches<tvbk::jr,8>(cx, c, (float *)x.data(), (float *)p.data(), t0, t1, dt);
    }, "cx"_a, "c"_a, "x"_a, "p"_a, "t0"_a, "nt"_a, "dt"_a);

  m.def("step_mpr8",
    [](const tvbk::cxbs<8> &cx, const tvbk::conn &c,
    farr<nb::shape<-1,tvbk::mpr::num_svar,-1,8>> &x,
    farr<nb::shape<-1,-1,tvbk::mpr::num_parm,8>> &p,
    uint32_t t0, uint32_t nt, float dt)
    {
      // check x.shape[0] == p.shape[0] == cx.num_batch
      if (!(x.shape(0) == p.shape(0) && x.shape(0) == cx.num_batch))
        throw std::runtime_error("batch shapes don't match: check that x.shape[0] == p.shape[0] == cx.num_batch");
      // check x.shape[2] == cx.num_node == p.shape[1]
      if (!(x.shape(2) == cx.num_node && x.shape(2) == p.shape(1)))
        throw std::runtime_error("node shapes don't match: check that x.shape[2] == cx.num_node == p.shape[1]");
      tvbk::step_batches<tvbk::mpr,8>(cx, c, (float *)x.data(), (float *)p.data(), t0, t1, dt);
    }, "cx"_a, "c"_a, "x"_a, "p"_a, "t0"_a, "nt"_a, "dt"_a);

}
