load(
    "@nanobind_bazel//:build_defs.bzl",
    "nanobind_extension",
    "nanobind_stubgen",
)

py_library(
    name = "nanobind_example",
    srcs = ["nanobind_example/__init__.py", "nanobind_example/sub_ext/__init__.py"],
    data = [":nanobind_example_ext"],
    visibility = ["//visibility:public"],
)

nanobind_extension(
    name = "nanobind_example_ext",
    srcs = ["nanobind_example_ext.cpp"],
)

nanobind_stubgen(
    name = "nanobind_example_ext_stubgen",
    module = ":nanobind_example_ext",
    marker_file = "src/py.typed",
    output_directory = "src",
    recursive = True,
)
