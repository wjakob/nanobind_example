#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

extern "C" {
#include "tvbk.h"
}

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(nanobind_example_ext, m) {
    m.doc() = "This is a \"hello world\" example with nanobind";
    m.def("add", [](int a, int b) { return a + b; }, "a"_a, "b"_a);
    m.def("mul", [](int a, int b) { return a * b; }, "a"_a, "b"_a);

    // nanobind higher order function example
    m.def("partial_add", [](const int a) {
        auto adder = [a](int b) { return a + b; };
        return nb::cpp_function(adder, "b"_a);
    }, "a"_a);

    m.def(
        "randn",
        [](const uint32_t seed,
           nb::ndarray<float, nb::device::cpu, nb::shape<-1>> z)
        {
            tvbk_randn(seed, z.shape(0), (float *)z.data());
        },
        "seed"_a, "z"_a,
        "This functions generates normally distributed random numbers.");

    typedef nb::ndarray<float, nb::device::cpu, nb::shape<8, 8>> m8;

    m.def(
        "mm8_ref",
        [](m8 A, m8 B, m8 C)
        {
            float b[8] = {0};
            tvbk_mm8_ref((float *)A.data(), (float *)B.data(), (float *)C.data(), b);
        },
        "A"_a, "B"_a, "C"_a,
        "This function performs 8x8 matrix multiplication.");

    m.def(
        "mm8_fast",
        [](m8 A, m8 B, m8 C)
        {
            float b[8] = {0};
            tvbk_mm8_fast((float *)A.data(), (float *)B.data(), (float *)C.data(), b);
        },
        "A"_a, "B"_a, "C"_a,
        "This function performs 8x8 matrix multiplication.");
}
