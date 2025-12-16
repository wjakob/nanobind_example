#include <nanobind/nanobind.h>

namespace nb = nanobind;

using namespace nb::literals;

static int add(int a, int b) { return a + b; }

NB_MODULE(nanobind_example_ext, m) {
  m.doc() = "This is a \"hello world\" example with nanobind";
  m.def("add", &add, "a"_a, "b"_a, R"(Add two integers together.)");
}
