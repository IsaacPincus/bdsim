cmake_minimum_required(VERSION 3.18)
project(bdsim LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(BDSIM_NATIVE "Build with -march=native" OFF)
add_compile_options($<$<CONFIG:Release>:-O3>)
if(BDSIM_NATIVE AND NOT MSVC)
  add_compile_options(-march=native)
endif()

# --- core library (grows each phase) ---
add_library(bdcore STATIC
  src/rng.cpp
  src/numerics.cpp
  src/config.cpp
  src/spring.cpp
  src/bending.cpp
  src/excluded_volume.cpp
  src/external_force.cpp
  src/flow.cpp
  src/mobility.cpp
  src/linalg.cpp
  src/hydrodynamics.cpp
  src/integrator.cpp
)
target_include_directories(bdcore PUBLIC src)

# Optional BLAS/LAPACK backend for the HI dense linear algebra.
option(BDSIM_LAPACK "Use BLAS/LAPACK for the HI dense kernels" OFF)
if(BDSIM_LAPACK)
  find_package(LAPACK REQUIRED)
  target_compile_definitions(bdcore PUBLIC BDSIM_USE_LAPACK)
  target_link_libraries(bdcore PUBLIC LAPACK::LAPACK)
endif()
set_target_properties(bdcore PROPERTIES POSITION_INDEPENDENT_CODE ON)

# --- tests ---
enable_testing()
foreach(t rng spring config bending ev flow spring_force
          mobility hi hi_stability external_force direction integrator_rouse integrator_smoke)
  add_executable(test_${t} tests/test_${t}.cpp)
  target_link_libraries(test_${t} PRIVATE bdcore)
  add_test(NAME ${t} COMMAND test_${t})
endforeach()

add_executable(bench bench/bench.cpp)
target_link_libraries(bench PRIVATE bdcore)

add_executable(sens_equiv bench/sens_equiv.cpp)
target_link_libraries(sens_equiv PRIVATE bdcore)

# --- Fortran reference core (SingleChainBD), opt-in --------------------------
#
# Builds the original Fortran physics from fortran/ so the two codes can be
# compared without leaving this repository. Only LAPACK is needed: the five
# vendored sources contain no MPI and no NetCDF (those live in the drivers,
# sensemble.f90 and DataOutput.f90, which are deliberately not vendored).
#
#   cmake -S . -B build -DBDSIM_FORTRAN=ON
#   ./build/fortran_oracle 1.0d-12
#
# CMake scans Fortran sources for module dependencies, so the order below does
# not matter, but the real dependency chain is
#   modules -> utils -> {properties, InitialPos} -> gsipc -> oracle
option(BDSIM_FORTRAN "Build the Fortran reference core and oracle driver" OFF)
if(BDSIM_FORTRAN)
  enable_language(Fortran)
  find_package(LAPACK REQUIRED)

  add_library(sens_core STATIC
    fortran/modules.f90
    fortran/utils.f90
    fortran/properties.f90
    fortran/InitialPos.f90
    fortran/gsipc.f90
  )
  # Keep the .mod files out of the source tree.
  set_target_properties(sens_core PROPERTIES
    Fortran_MODULE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/fortran_modules")
  target_link_libraries(sens_core PUBLIC LAPACK::LAPACK)

  add_executable(fortran_oracle fortran/oracle.f90)
  target_link_libraries(fortran_oracle PRIVATE sens_core)
  set_target_properties(fortran_oracle PROPERTIES
    Fortran_MODULE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/fortran_modules")

  # The cross-check against the C++ is registered at the end of this file: it
  # runs through the Python bindings, so it needs Python_EXECUTABLE, which the
  # BDSIM_PYTHON block below is what defines.
endif()

# --- Python bindings (nanobind), opt-in ---
#
# Enable with -DBDSIM_PYTHON=ON. nanobind ships its CMake config inside the
# installed Python package rather than anywhere CMake searches by default, so we
# ask the interpreter where it is. That means an active venv with `pip install
# nanobind` is all that is needed -- no -Dnanobind_DIR=... on the command line.
option(BDSIM_PYTHON "Build the nanobind Python module" OFF)
if(BDSIM_PYTHON)
  find_package(Python 3.9 REQUIRED COMPONENTS Interpreter Development.Module)
  if(NOT nanobind_ROOT AND NOT nanobind_DIR)
    execute_process(
      COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
      OUTPUT_VARIABLE _nanobind_cmake_dir
      OUTPUT_STRIP_TRAILING_WHITESPACE
      RESULT_VARIABLE _nanobind_status
      ERROR_VARIABLE  _nanobind_error)
    if(NOT _nanobind_status EQUAL 0)
      message(FATAL_ERROR
        "BDSIM_PYTHON=ON but nanobind was not found for ${Python_EXECUTABLE}.\n"
        "Install it into the environment you are building against:\n"
        "    pip install nanobind\n"
        "(or pass -Dnanobind_DIR=<path> yourself).\n"
        "The interpreter reported: ${_nanobind_error}")
    endif()
    set(nanobind_ROOT "${_nanobind_cmake_dir}")
    message(STATUS "Found nanobind via ${Python_EXECUTABLE}: ${nanobind_ROOT}")
  endif()
  find_package(nanobind CONFIG REQUIRED)
  nanobind_add_module(_bdsim bindings/module.cpp)
  target_link_libraries(_bdsim PRIVATE bdcore)

  # Put the extension straight where the Python package expects it, so a build
  # is immediately importable and there is no copy step to forget.
  set_target_properties(_bdsim PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/python/bdsim")
endif()

# --- Fortran <-> C++ cross-check -------------------------------------------
#
# Registered last because it needs both fortran_oracle and Python_EXECUTABLE.
# `setup.sh --fortran` therefore runs it as part of ctest.
#
# The threshold is 1e-11 at a corrector tolerance of 1e-12. Measured agreement
# was 4.7e-14 after two blocks, so this leaves three orders of headroom for a
# different compiler, BLAS or optimisation level while still failing loudly if
# the two codes genuinely diverge.
if(BDSIM_FORTRAN)
  if(BDSIM_PYTHON)
    add_test(NAME fortran_cross_check
             COMMAND ${Python_EXECUTABLE}
                     ${CMAKE_CURRENT_SOURCE_DIR}/validation/compare_fortran_oracle.py
                     --oracle $<TARGET_FILE:fortran_oracle>
                     --case full --tol 1e-12 --blocks 2 --check 1e-11)
    set_tests_properties(fortran_cross_check PROPERTIES
                         WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
  else()
    message(STATUS "BDSIM_FORTRAN is on but BDSIM_PYTHON is off: the Fortran "
                   "reference will build, but the cross-check test compares "
                   "through the Python bindings and is not registered")
  endif()
endif()
