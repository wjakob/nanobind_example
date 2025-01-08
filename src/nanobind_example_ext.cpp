#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "tvbk.hpp"

namespace nb = nanobind;
using namespace nb::literals;
typedef nb::ndarray<float, nb::numpy, nb::device::cpu, nb::shape<-1>, nb::c_contig> fvec;
typedef nb::ndarray<float, nb::numpy, nb::device::cpu, nb::shape<-1,-1>, nb::c_contig> fmat;
typedef nb::ndarray<uint32_t, nb::numpy, nb::device::cpu, nb::shape<-1>, nb::c_contig> uvec;

NB_MODULE(nanobind_example_ext, m) {

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
}
