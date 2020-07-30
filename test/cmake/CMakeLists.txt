cmake_minimum_required(VERSION 3.11 FATAL_ERROR)

project(Likwid-Test LANGUAGES C)

find_package(likwid REQUIRED COMPONENTS marker OPTIONAL_COMPONENTS nvmarker)

add_executable(test ${CMAKE_SOURCE_DIR}/test.c)
target_link_libraries(test PRIVATE likwid::likwid)
