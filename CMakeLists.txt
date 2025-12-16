cmake_minimum_required(VERSION 3.15...3.26)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
  message(FATAL_ERROR "In-tree builds are not supported. Run CMake from a separate directory: cmake -B build")
endif()

project(nanobind_example LANGUAGES CXX)

# Try to import all Python components potentially needed by nanobind
find_package(Python 3.12
  REQUIRED COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule
)

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
   $ uv sync
  =====================================================================
  When using uv run, the package is already rebuilt automatically if
  it detects sources listed in cache-keys are updated.")

  # NOTE: We allow fetching nanobind directly here for development this allows
  # `cmake -B build -G Ninja -DCMAKE_EXPORT_COMPILE_COMMANDS=ON` to work as
  # expected and get proper language server support in editors like VSCode.

  # Extract the minimum nanobind version from pyproject.toml
  file(READ "${CMAKE_CURRENT_SOURCE_DIR}/pyproject.toml" PYPROJECT_TOML_CONTENT)
  if(PYPROJECT_TOML_CONTENT MATCHES "\"nanobind *[>=]= *([^\"]+)\"")
    set(NANOBIND_TAG "v${CMAKE_MATCH_1}")
  else()
    message(FATAL_ERROR "Could not find nanobind version in pyproject.toml")
  endif()
  
  # Fix warnings about DOWNLOAD_EXTRACT_TIMESTAMP
  if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
  endif()
  include(FetchContent)
  message(STATUS "Fetching nanobind (${NANOBIND_TAG})...")
  FetchContent_Declare(nanobind
    GIT_REPOSITORY
      "https://github.com/wjakob/nanobind"
    GIT_TAG
      ${NANOBIND_TAG}
  )
  FetchContent_MakeAvailable(nanobind)
else()
  # Use the nanobind provided by scikit-build-core
  find_package(nanobind REQUIRED)
endif()

set(PYTHON_MODULE nanobind_example)
set(EXTENSION_MODULE nanobind_example_ext)
set(EXTENSION_SOURCES
  src/nanobind_example_ext.cpp
)

# We are now ready to compile the actual extension module
nanobind_add_module(
  # Name of the extension
  ${EXTENSION_MODULE}

  # Target the stable ABI for Python 3.12+, which reduces
  # the number of binary wheels that must be built. This
  # does nothing on older Python versions
  STABLE_ABI

  # Build libnanobind statically and merge it into the
  # extension (which itself remains a shared library)
  #
  # If your project builds multiple extensions, you can
  # replace this flag by NB_SHARED to conserve space by
  # reusing a shared libnanobind across libraries
  NB_STATIC

  # Source code goes here
  ${EXTENSION_SOURCES}
)

# Generate type stubs for the extension module
nanobind_add_stub(
  ${EXTENSION_MODULE}_stub
  MODULE ${EXTENSION_MODULE}
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${EXTENSION_MODULE}.pyi
  PYTHON_PATH $<TARGET_FILE_DIR:${EXTENSION_MODULE}>
  DEPENDS ${EXTENSION_MODULE}
)

# Install the extension module and type stubs
install(TARGETS ${EXTENSION_MODULE} LIBRARY DESTINATION ${PYTHON_MODULE})
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/py.typed" "")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/py.typed" DESTINATION ${PYTHON_MODULE})
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${EXTENSION_MODULE}.pyi DESTINATION ${PYTHON_MODULE})

# Install the Python package files
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/${PYTHON_MODULE}/__init__.py DESTINATION ${PYTHON_MODULE})
