#include <nanobind/nanobind.h>

namespace nb = nanobind;

using namespace nb::literals;

NB_MODULE(nanobind_example_ext, m) {
    m.doc() = "This is a \"hello world\" example with nanobind";
    m.def("add", [](int a, int b) { return a + b; }, "a"_a, "b"_a);

    nb::module_ sm = m.def_submodule("sub_ext", "A submodule of 'nanobind_example_ext'");

    sm.def("sub", [](int a, int b) { return a - b; }, "a"_a, "b"_a);
}
