cmake_minimum_required(VERSION 3.16)
project(local_utils_cpp LANGUAGES CXX)

find_package(pybind11 CONFIG REQUIRED)
find_package(Eigen3   CONFIG REQUIRED)   # ★ 新增

pybind11_add_module(local_utils_cpp util.cpp)

# 输出到源码目录（local_planner）
set_target_properties(local_utils_cpp PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

target_compile_features(local_utils_cpp PRIVATE cxx_std_17)
target_compile_options(local_utils_cpp PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang>:-O3 -march=native -ffast-math -DNDEBUG>
  $<$<CXX_COMPILER_ID:MSVC>:/O2 /DNDEBUG>
)

target_link_libraries(local_utils_cpp PRIVATE Eigen3::Eigen)
set_property(TARGET local_utils_cpp PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
