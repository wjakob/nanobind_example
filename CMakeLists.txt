project(nanobind_example)
cmake_minimum_required(VERSION 3.17...3.22)

if (NOT SKBUILD)
  message(WARNING "This CMake file should be executed via scikit-build. "
                      "Please run\n$ pip install .")
endif()

if (SKBUILD)
  # Constrain FindPython to find the Python version used by scikit-build
  set(Python_VERSION "${PYTHON_VERSION_STRING}")
  set(Python_EXECUTABLE "${PYTHON_EXECUTABLE}")
  set(Python_INCLUDE_DIR "${PYTHON_INCLUDE_DIR}")
  set(Python_LIBRARIES "${PYTHON_LIBRARY}")
elseif (MSVC)
  # MSVC needs a little extra help finding the Python library
  find_package(PythonInterp)
  find_package(Python)
endif()

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

# Run `nanobind.cmake_dir()` from Python to detect install location
execute_process(
  COMMAND
  "${PYTHON_EXECUTABLE}" -c "import nanobind; print(nanobind.cmake_dir())"
  OUTPUT_VARIABLE _tmp_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ECHO STDOUT)
list(APPEND CMAKE_PREFIX_PATH "${_tmp_dir}")

# Now import nanobind from CMake
find_package(nanobind CONFIG REQUIRED)

# Build the actual extension module
nanobind_add_module(
  nanobind_example_ext
  NB_STATIC # Build static libnanobind (the extension module itself remains a shared library)
  src/nanobind_example_ext.cpp
)

# Install directive for scikit-build
install(TARGETS nanobind_example_ext LIBRARY DESTINATION .)
