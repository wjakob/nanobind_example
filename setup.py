import sys

try:
    from skbuild import setup
    import nanobind
except ImportError:
    print("The preferred way to invoke 'setup.py' is via pip, as in 'pip "
          "install .'. If you wish to run the setup script directly, you must "
          "first install the build dependencies listed in pyproject.toml!",
          file=sys.stderr)
    raise

setup(
    name="nanobind_example",
    version="0.0.1",
    author="Wenzel Jakob",
    author_email="wenzel.jakob@epfl.ch",
    description="An example minimal project that compiles bindings using nanobind and scikit-build",
    url="https://github.com/wjakob/nanobind_example",
    license="BSD",
    packages=['nanobind_example'],
    package_dir={'': 'src'},
    cmake_install_dir="src/nanobind_example",
    include_package_data=True,
    python_requires=">=3.8"
)
