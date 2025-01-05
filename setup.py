import os
import platform
import shutil
import sys
from pathlib import Path

import setuptools
from setuptools.command import build_ext

# free-threaded build option, requires Python 3.13+.
# Source: https://docs.python.org/3/howto/free-threading-python.html#identifying-free-threaded-python
free_threaded = "experimental free-threading build" in sys.version
# SABI-related options. Requires that each Python interpreter
# (hermetic or not) participating is of the same major-minor version.
# Cannot be used together with free-threading.
py_limited_api = sys.version_info >= (3, 12) and not free_threaded
options = {"bdist_wheel": {"py_limited_api": "cp312"}} if py_limited_api else {}


class BazelExtension(setuptools.Extension):
    """A C/C++ extension that is defined as a Bazel BUILD target."""

    def __init__(self, name: str, bazel_target: str, **kwargs):
        super().__init__(name=name, sources=[], **kwargs)

        self.free_threaded = free_threaded
        self.bazel_target = bazel_target
        stripped_target = bazel_target.split("//")[-1]
        self.relpath, self.target_name = stripped_target.split(":")


class BuildBazelExtension(build_ext.build_ext):
    """A command that runs Bazel to build a C/C++ extension."""

    def run(self):
        for ext in self.extensions:
            self.bazel_build(ext)
        # explicitly call `bazel shutdown` for graceful exit
        self.spawn(["bazel", "shutdown"])

    def copy_extensions_to_source(self):
        """
        Copy generated extensions into the source tree.
        This is done in the ``bazel_build`` method, so it's not necessary to
        do again in the `build_ext` base class.
        """
        pass

    def bazel_build(self, ext: BazelExtension) -> None:
        """Runs the bazel build to create a nanobind extension."""
        temp_path = Path(self.build_temp)

        # Specifying only MAJOR.MINOR makes rules_python do an internal
        # lookup selecting the newest patch version.
        python_version = "{0}.{1}".format(*sys.version_info[:2])

        bazel_argv = [
            "bazel",
            "run",
            ext.bazel_target,
            f"--symlink_prefix={temp_path / 'bazel-'}",
            f"--compilation_mode={'dbg' if self.debug else 'opt'}",
            f"--target_python_version={python_version}",
        ]

        if ext.py_limited_api:
            bazel_argv += ["--py_limited_api=cp312"]
        if ext.free_threaded:
            bazel_argv += ["--free_threaded=yes"]

        self.spawn(bazel_argv)

        if platform.system() == "Windows":
            suffix = ".pyd"
        else:
            suffix = ".abi3.so" if ext.py_limited_api else ".so"

        # copy the Bazel build artifacts into setuptools' libdir,
        # from where the wheel is built.
        srcdir = temp_path / "bazel-bin" / "src"
        libdir = Path(self.build_lib) / "nanobind_example"
        for root, dirs, files in os.walk(srcdir, topdown=True):
            # exclude runfiles directories and children.
            dirs[:] = [d for d in dirs if "runfiles" not in d]

            for f in files:
                fp = Path(f)
                should_copy = False
                # we do not want the bare .so file included
                # when building for ABI3, so we require a
                # full and exact match on the file extension.
                if "".join(fp.suffixes) == suffix:
                    should_copy = True
                elif fp.suffix == ".pyi":
                    should_copy = True
                elif Path(root) == srcdir and f == "py.typed":
                    # copy py.typed, but only at the package root.
                    should_copy = True

                if should_copy:
                    dstdir = libdir / os.path.relpath(root, srcdir)
                    if not os.path.exists(dstdir):
                        os.mkdir(dstdir)
                    shutil.copyfile(root / fp, dstdir / fp)


setuptools.setup(
    cmdclass=dict(build_ext=BuildBazelExtension),
    package_data={'nanobind_example': ["py.typed", "*.pyi", "**/*.pyi"]},
    ext_modules=[
        BazelExtension(
            name="nanobind_example.nanobind_example_ext",
            bazel_target="//src:nanobind_example_ext_stubgen",
            free_threaded=free_threaded,
            py_limited_api=py_limited_api,
        )
    ],
    options=options,
)
