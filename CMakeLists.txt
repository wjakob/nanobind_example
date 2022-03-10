project(nanobind_example)
cmake_minimum_required(VERSION 3.17...3.22)

if (NOT SKBUILD)
  message(FATAL_ERROR "This CMake file should be executed via scikit-build. "
                      "Please run\n$ pip install .")
endif()

# Constrain FindPython to find the Python version used by scikit-build
if (SKBUILD)
  message(STATUS "Python_VERSION ${PYTHON_VERSION_STRING}")
  message(STATUS "Python_EXECUTABLE ${PYTHON_EXECUTABLE}")
  message(STATUS "Python_INCLUDE_DIR ${PYTHON_INCLUDE_DIR}")
  message(STATUS "Python_LIBRARIES ${PYTHON_LIBRARY}")
  set(Python_VERSION "${PYTHON_VERSION_STRING}")
  set(Python_EXECUTABLE "${PYTHON_EXECUTABLE}")
  set(Python_INCLUDE_DIR "${PYTHON_INCLUDE_DIR}")
  set(Python_LIBRARIES "${PYTHON_LIBRARY}")
endif()
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

# Detect nanobind and import it
execute_process(
  COMMAND
  "${PYTHON_EXECUTABLE}" -c "import nanobind; print(nanobind.cmake_dir())"
  OUTPUT_VARIABLE _tmp_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ECHO STDOUT)
  list(APPEND CMAKE_PREFIX_PATH "${_tmp_dir}")

find_package(nanobind CONFIG REQUIRED)

message(STATUS ${NB_DIR})

nanobind_add_module(
  nanobind_example_ext
  NB_STATIC # Build static libnanobind (the extension module itself remains a shared library)
  src/nanobind_example_ext.cpp
)

install(TARGETS nanobind_example_ext LIBRARY DESTINATION .)
