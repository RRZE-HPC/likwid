cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(Likwid-Test LANGUAGES C)

find_package(likwid REQUIRED COMPONENTS marker OPTIONAL_COMPONENTS nvmarker)

message(STATUS "")
message(STATUS "likwid_FOUND: ${likwid_FOUND}")

foreach(_VAR LIKWID_INCLUDES LIKWID_LIBRARIES LIKWID_ROOT_DIR
    LIKWID_NVIDIA_INTERFACE LIKWID_FORTRAN_INTERFACE
    LIKWID_INCLUDE_DIR LIKWID_LIBRARY LIKWID_INCLUDE_DIRS
    LIKWID_LIBRARIES LIKWID_LIBRARY_DIRS)
    message(STATUS "${_VAR}: ${${_VAR}}")
endforeach()

foreach(_VAR ${LIKWID_EXECUTABLE_OPTIONS})
    message(STATUS "LIKWID_${_VAR}_EXECUTABLE: ${LIKWID_${_VAR}_EXECUTABLE}")
endforeach()

foreach(_VAR ${LIKWID_COMPONENT_OPTIONS})
    message(STATUS "likwid_${_VAR}_FOUND: ${likwid_${_VAR}_FOUND}")
endforeach()
message(STATUS "")

add_executable(test_install_tree ${PROJECT_SOURCE_DIR}/test.c)
target_link_libraries(test_install_tree PRIVATE likwid::likwid)
