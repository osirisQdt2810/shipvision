# The compiled half of shipvision: C++17 with CUDA or HIP kernels, exposed as
# `shipvision._C` through pybind11.
#
# One source tree, two vendors. Every device call goes through the `gpu*` aliases in
# core/platform.h, so the ROCm build is the same code — a raw `cudaMalloc` anywhere else
# silently breaks it, which is why there is a guard test for exactly that.
#
# The tree under `csrc/shipvision/` mirrors the Python package directory for directory, and a
# component's header sits next to its translation unit. `csrc/` itself is the include root, so
# `#include "shipvision/imgproc/image_ops.h"` reads the same from a kernel, from the bindings
# and from a consumer — while `image_ops.h` and `image_ops.cu` stay one line apart in `ls`.
#
# Deliberately narrow. Anything a well-optimised library already does — allocation, streams,
# graph capture, a plain resize, gemm — is not reimplemented here. What lives here is what
# no library expresses: several memory-bound passes fused into one, and the association
# inner loops where a Python call per track is the whole frame budget.
#
# Build:
#   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
#   cmake --build build -j
# or simply:
#   python build.py

cmake_minimum_required(VERSION 3.22 FATAL_ERROR)

project(shipvision LANGUAGES CXX VERSION 0.1.0)

option(SHIPVISION_WITH_CUDA "Build the CUDA kernels" ON)
option(SHIPVISION_WITH_HIP  "Build the HIP/ROCm kernels instead of CUDA" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  # A debug build of a preprocessing kernel is ~10x slower and would make every benchmark
  # meaningless, so Release is the default rather than something to remember.
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

if(SHIPVISION_WITH_HIP)
  set(SHIPVISION_WITH_CUDA OFF)
endif()

# ---------------------------------------------------------------------------------------
# Python + pybind11
# ---------------------------------------------------------------------------------------
find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)

# Ask the active interpreter where pybind11 lives rather than requiring a system package:
# the extension must build against the same interpreter that will import it.
execute_process(
  COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
  OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE PYBIND11_LOOKUP)
if(NOT PYBIND11_LOOKUP EQUAL 0)
  message(FATAL_ERROR "pybind11 is not installed for ${Python3_EXECUTABLE}. "
                      "Run: ${Python3_EXECUTABLE} -m pip install pybind11")
endif()
list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
find_package(pybind11 CONFIG REQUIRED)

# ---------------------------------------------------------------------------------------
# Accelerator
# ---------------------------------------------------------------------------------------
# Globbed rather than listed, with CONFIGURE_DEPENDS so adding a kernel re-runs CMake
# instead of silently not building. The usual objection to globbing — that a new file is
# missed until someone reconfigures — is exactly what CONFIGURE_DEPENDS removes, and a
# hand-maintained list across five algorithm domains would be forgotten within a month.
#
# GLOB_RECURSE rather than GLOB: sources now live one directory per family under
# `csrc/shipvision/`, so a flat glob would compile nothing and the FATAL_ERROR below is the
# only thing that would say so.
file(GLOB_RECURSE SHIPVISION_CUDA_SOURCES CONFIGURE_DEPENDS
     ${CMAKE_CURRENT_SOURCE_DIR}/csrc/shipvision/*.cu)
file(GLOB_RECURSE SHIPVISION_CXX_SOURCES CONFIGURE_DEPENDS
     ${CMAKE_CURRENT_SOURCE_DIR}/csrc/shipvision/*.cpp)

# The bindings are the one thing that is not a mirror of the Python package: they are the
# Python edge itself, so they sit beside the library rather than inside it.
file(GLOB SHIPVISION_BINDING_SOURCES CONFIGURE_DEPENDS
     ${CMAKE_CURRENT_SOURCE_DIR}/csrc/bindings/*.cpp)
list(APPEND SHIPVISION_CXX_SOURCES ${SHIPVISION_BINDING_SOURCES})

if(NOT SHIPVISION_CUDA_SOURCES)
  message(FATAL_ERROR
    "No kernels found under csrc/shipvision/ — is the source tree complete?")
endif()

set(SHIPVISION_SOURCES ${SHIPVISION_CXX_SOURCES})

if(SHIPVISION_WITH_HIP)
  enable_language(HIP)
  add_compile_definitions(SHIPVISION_WITH_HIP)
  # The .cu files ARE the HIP sources — one tree, compiled by whichever vendor is present.
  # This is what the gpu* aliases in core/platform.h exist to make possible.
  set_source_files_properties(${SHIPVISION_CUDA_SOURCES} PROPERTIES LANGUAGE HIP)
  list(APPEND SHIPVISION_SOURCES ${SHIPVISION_CUDA_SOURCES})
  message(STATUS "shipvision native: HIP/ROCm backend")

elseif(SHIPVISION_WITH_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  set(CMAKE_CUDA_STANDARD 17)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # Pascal through Ada. `native` would be faster to build but produces a binary that
    # silently fails on any other node in the fleet.
    set(CMAKE_CUDA_ARCHITECTURES 61 70 75 80 86 89)
  endif()
  list(APPEND SHIPVISION_SOURCES ${SHIPVISION_CUDA_SOURCES})
  message(STATUS "shipvision native: CUDA ${CUDAToolkit_VERSION}, arch ${CMAKE_CUDA_ARCHITECTURES}")

else()
  message(FATAL_ERROR "Enable exactly one of SHIPVISION_WITH_CUDA / SHIPVISION_WITH_HIP")
endif()

# ---------------------------------------------------------------------------------------
# Target
# ---------------------------------------------------------------------------------------
pybind11_add_module(_C MODULE ${SHIPVISION_SOURCES})

# `csrc/` is the include root, which is what keeps every `#include "shipvision/..."` path
# unchanged now that the headers live next to their translation units.
target_include_directories(_C PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/csrc)
target_compile_options(_C PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wno-unused-parameter>
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

if(SHIPVISION_WITH_CUDA)
  target_link_libraries(_C PRIVATE CUDA::cudart)
  # Separable compilation is deliberately OFF. There are no cross-translation-unit device
  # calls here, so relocatable device code would buy nothing — and combined with the
  # link-time optimisation pybind11 enables, the device-link step's fatbin symbol gets
  # stripped and the module fails to import with "undefined symbol: fatbinData".
  set_target_properties(_C PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
endif()

# Install straight into the package tree so an editable install picks it up with no
# PYTHONPATH games: `import shipvision` must work immediately after a build.
#
# The package sits at the repository ROOT (`shipvision/`), not under a `python/` prefix.
# Asserted rather than assumed: when the layout moved, this path was left behind, the
# POST_BUILD copy failed at 100% — and because a failed POST_BUILD makes the target dirty,
# make then DELETED the freshly-linked module. A clean checkout could not produce a working
# `.so` at all, and the error scrolled past under a "[100%] Built target" line.
set(SHIPVISION_PACKAGE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/shipvision")
if(NOT EXISTS "${SHIPVISION_PACKAGE_DIR}/__init__.py")
  message(FATAL_ERROR
    "Expected the Python package at ${SHIPVISION_PACKAGE_DIR} but found no __init__.py. "
    "If the layout moved, update SHIPVISION_PACKAGE_DIR — do not let the POST_BUILD copy "
    "fail silently.")
endif()
add_custom_command(TARGET _C POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:_C> "${SHIPVISION_PACKAGE_DIR}/"
  COMMENT "Installing $<TARGET_FILE_NAME:_C> into shipvision/")
