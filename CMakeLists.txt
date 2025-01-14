cmake_minimum_required(VERSION 3.15...3.26)

project(tvbk LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT SKBUILD)
  message(WARNING "\
  This CMake file is meant to be executed using 'scikit-build'. Running
  it directly will almost certainly not produce the desired result. If
  you are a user trying to install this package, please use the command
  below, which will install all necessary build dependencies, compile
  the package in an isolated environment, and then install it.
  =====================================================================
   $ pip install .
  =====================================================================
  If you are a software developer, and this is your own package, then
  it is usually much more efficient to install the build dependencies
  in your environment once and use the following command that avoids
  a costly creation of a new virtual environment at every compilation:
  =====================================================================
   $ pip install nanobind scikit-build-core[pyproject]
   $ pip install --no-build-isolation -ve .
  =====================================================================
  You may optionally add -Ceditable.rebuild=true to auto-rebuild when
  the package is imported. Otherwise, you need to re-run the above
  after editing C++ files.")
endif()

# Try to import all Python components potentially needed by nanobind
find_package(Python 3.8
  REQUIRED COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule)

# Import nanobind through CMake's find_package mechanism
find_package(nanobind CONFIG REQUIRED)

# setup binding module
nanobind_add_module(
  tvbk_ext
  STABLE_ABI NB_STATIC LTO NOMINSIZE
  src/tvbk_ext.cpp
)

# if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
#   find_package(OpenMP REQUIRED)
#   target_compile_options(nanobind_example_ext PRIVATE -march=native -ffast-math -fopenmp -funroll-loops -mavx2)
#   target_link_libraries(nanobind_example_ext PUBLIC OpenMP::OpenMP_CXX)
# endif()
# clang: -fno-math-errno -fveclib=libmvec, but no omp on macos

# Install directive for scikit-build-core
install(TARGETS tvbk_ext LIBRARY DESTINATION tvbk)
