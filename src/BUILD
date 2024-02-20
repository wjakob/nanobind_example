load("@nanobind_bazel//:build_defs.bzl", "nanobind_extension")

py_library(
    name = "nanobind_example",
    srcs = ["nanobind_example/__init__.py"],
    data = [":nanobind_example_ext"],
    visibility = ["//visibility:public"],
)

nanobind_extension(
    name = "nanobind_example_ext",
    srcs = ["nanobind_example_ext.cpp"],
)
