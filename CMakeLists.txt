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
