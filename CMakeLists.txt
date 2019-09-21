cmake_minimum_required(VERSION 3.14)
project(ivf)

set(CMAKE_CXX_STANDARD 17)

aux_source_directory(${CMAKE_SOURCE_DIR} srcs)

set(CPU_FLAGS "-mavx2 -mpopcnt -msse4")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC ${CPU_FLAGS}")

add_executable(ivf ${srcs})

target_link_libraries(ivf openblas)