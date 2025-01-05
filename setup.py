import os
import platform
import sys
import sysconfig

from pathlib import Path

import nanobind
import setuptools


NB_INCLUDE = nanobind.include_dir()
NB_SRCDIR = Path(NB_INCLUDE).with_name("src")
PY_INCLUDE = sysconfig.get_paths()["include"]
ROBIN_MAP_INCLUDE = NB_SRCDIR.with_name("ext") / "robin_map" / "include"

# Setting these envvars prevents setuptools' compiler internals
# from picking up sysconfig's own Python flags.
# This allows us to use chained fixups and linker response files on MacOS.
# TODO: Change the shared linking flags below to your linker of choice.
os.environ["CFLAGS"] = os.environ["CXXFLAGS"] = ""
os.environ["LDSHARED"] = os.environ["LDCXXSHARED"] = "clang++"

# free-threaded build option, requires Python 3.13+.
# Source: https://docs.python.org/3/howto/free-threading-python.html#identifying-free-threaded-python
free_threaded = "experimental free-threading build" in sys.version
# Options for building against the stable ABI (ABI3).
# Cannot be used together with free-threading.
py_limited_api = sys.version_info >= (3, 12) and not free_threaded
options = {"bdist_wheel": {"py_limited_api": "cp312"}} if py_limited_api else {}


NB_MACROS = [("NB_COMPACT_ASSERTIONS", None)]
if py_limited_api:
    NB_MACROS.append(("Py_LIMITED_API", "0x030C0000"))  # CPython 3.12+
if free_threaded:
    NB_MACROS.append(("NB_FREE_THREADED", None))


platform_system = platform.system()
if platform_system == "Windows":
    NB_CFLAGS, NB_LINKOPTS = [], []
elif platform_system in ("Darwin", "Linux"):
    NB_CFLAGS = [
        "-std=c++17",
        "-fvisibility=hidden",
        "-fPIC",
        "-fno-strict-aliasing",
        "-ffunction-sections",
        "-fdata-sections",
    ]
    if platform_system == "Darwin":
        NB_LINKOPTS = [
            "-shared",
            f"-Wl,@{Path(nanobind.cmake_dir()) / 'darwin-ld-cpython.sym'}",
            "-Wl,-x",
            "-Wl,-S",
            "-Wl,-dead_strip",
        ]
    else:
        NB_LINKOPTS = ["-shared", "-Wl,-s", "-Wl,--gc-sections"]
else:
    raise ValueError(f"unsupported platform {platform_system!r}")


setuptools.setup(
    package_data={'nanobind_example': ["py.typed", "*.pyi", "**/*.pyi"]},
    libraries=[
        (
            "nanobind",
            {
                "sources": [str(NB_SRCDIR / "nb_combined.cpp")],
                "cflags": NB_CFLAGS, "macros": NB_MACROS,
                "include_dirs": [NB_INCLUDE, ROBIN_MAP_INCLUDE, PY_INCLUDE]
            }
        ),
    ],
    ext_modules=[
        setuptools.Extension(
            name="nanobind_example.nanobind_example_ext",
            sources=["src/nanobind_example_ext.cpp"],
            libraries=["nanobind"],
            extra_compile_args=NB_CFLAGS + ["-Os"],
            extra_link_args=NB_LINKOPTS,
            include_dirs=[NB_INCLUDE],
            py_limited_api=py_limited_api,
        )
    ],
    options=options,
)
