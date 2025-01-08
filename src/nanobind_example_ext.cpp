#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

extern "C" {
#include "tvbk.h"
}

namespace nb = nanobind;
using namespace nb::literals;
typedef nb::ndarray<float, nb::device::cpu, nb::shape<-1>> fvec;
typedef nb::ndarray<uint32_t, nb::device::cpu, nb::shape<-1>> uvec;

namespace Tvbk {
    struct Cx : tvbk_cx {
        Cx(uint32_t num_node, uint32_t num_time)
            : tvbk_cx{
                new float[num_node],
                new float[num_node],
                new float[num_node * num_time],
                num_node,
                num_time
            }
        {}
    };

    enum CxMode {
        CxModeJ,
        CxModeI
    };

    struct Conn : tvbk_conn {
        Conn(int num_node, int num_nonzero, 
             const fvec weights, const uvec indices,
             const uvec indptr, const uvec idelays)
            : tvbk_conn{
                num_node,
                num_nonzero,
                1,
                (float*) weights.data(),
                (uint32_t*) indices.data(),
                (uint32_t*) indptr.data(),
                (uint32_t*) idelays.data()
            }
        {}
    };
}

NB_MODULE(nanobind_example_ext, m) {
    m.doc() = "This is a \"hello world\" example with nanobind";

    m.def(
        "randn",
        [](const uint32_t seed,
           nb::ndarray<float, nb::device::cpu, nb::shape<-1>> z)
        {
            uint64_t s[4] = { seed, seed, seed, seed };
            tvbk_randn(s, z.shape(0), (float *)z.data());
        },
        "seed"_a, "z"_a,
        "This functions generates normally distributed random numbers using a popcount trick.");

    typedef nb::ndarray<float, nb::device::cpu, nb::shape<8, 8>> m8;

    m.def(
        "mm8",
        [](m8 A, m8 B, m8 C)
        {
            float b[8] = {0};
            tvbk_mm8_fast((float *)A.data(), (float *)B.data(), (float *)C.data(), b);
        },
        "A"_a, "B"_a, "C"_a,
        "This function performs 8x8 matrix multiplication.");

    nb::class_<Tvbk::Cx>(m, "Cx")
        .def(nb::init<uint32_t, uint32_t>(), "num_node"_a, "num_time"_a)
        .def_ro("num_node", &Tvbk::Cx::num_node)
        .def_ro("num_time", &Tvbk::Cx::num_time)
        .def_prop_ro("buf", [](Tvbk::Cx &cx) {
            return nb::ndarray<float, nb::numpy, nb::shape<-1,-1>, nb::c_contig>(cx.buf, {cx.num_node, cx.num_time});
        })
        ;


    nb::class_<Tvbk::Conn>(m, "Conn")
        .def(nb::init<int, int, const fvec, const uvec, const uvec, const uvec>(),
             "num_node"_a, "num_nonzero"_a, "weights"_a, "indices"_a, "indptr"_a, "idelays"_a);

    m.def(
        "cx_j",
        [](Tvbk::Cx &cx, Tvbk::Conn &conn, uint32_t t)
        {
            auto cx_ = static_cast<tvbk_cx*>(&cx);
            auto conn_ = static_cast<tvbk_conn*>(&conn);
            printf("cx_ = %p, conn_ = %p, t = %d\n", cx_, conn_, t);
            tvbk_cx_j(cx_, conn_, t);
        },
        "cx"_a, "conn"_a, "t"_a,
        "This function calculates the afferent coupling buffer.");

    // m.def("cx_i", [](tvbk_cx *cx, tvbk_conn *conn, uint32_t t) {
    //     tvbk_cx_i(cx, conn, t);
    // });
}
