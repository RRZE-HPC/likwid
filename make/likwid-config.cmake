# legacy variables
set(LIKWID_INCLUDES @PREFIX@/include)
set(LIKWID_LIBRARIES @PREFIX@/lib/liblikwid.so -llikwid)

# uses IN_LIST so return if less than 3.3.2
if(CMAKE_VERSION VERSION_LESS 3.3.2)
    return()
endif()

# Support new if() IN_LIST operator
cmake_policy(PUSH)
cmake_policy(SET CMP0057 NEW)

# updates
include(FindPackageHandleStandardArgs)

# variables configured during install
set(LIKWID_ROOT_DIR @PREFIX@ CACHE PATH "Path to LIKWID installation")
set(LIKWID_NVIDIA_INTERFACE @NVIDIA_INTERFACE@)
set(LIKWID_FORTRAN_INTERFACE @FORTRAN_INTERFACE@)

set(_LIKWID_PATH_HINTS @PREFIX@ @LIBPREFIX@ @BINPREFIX@)
set(_LIKWID_LIB_SUFFIXES lib lib64)

#------------------------------------------------------------------------------#
#
#       Interface library
#
#------------------------------------------------------------------------------#

# create an imported target library that CMake projects can just "link" to.
# e.g. target_link_libraries(myexe PUBLIC likwid)
add_library(likwid-library INTERFACE)
add_library(likwid::likwid ALIAS likwid-library)
# add_library(likwid::Likwid ALIAS likwid)

#------------------------------------------------------------------------------#
#
#       Include directory and library
#
#------------------------------------------------------------------------------#

# this is the cached variable to the include path
# if not found, will be LIKWID_INCLUDE_DIR-NOTFOUND
find_path(LIKWID_INCLUDE_DIR
    NAMES likwid.h
    HINTS ${LIKWID_ROOT_DIR} ${_LIKWID_PATH_HINTS}
    PATHS ${LIKWID_ROOT_DIR} ${_LIKWID_PATH_HINTS}
    PATH_SUFFIXES include
    DOC "LIKWID profiler include directory"
    NO_DEFAULT_PATH)

find_library(LIKWID_LIBRARY
    NAMES likwid
    PATH_SUFFIXES ${_LIKWID_LIB_SUFFIXES}
    HINTS ${LIKWID_ROOT_DIR} ${_LIKWID_PATH_HINTS}
    PATHS ${LIKWID_ROOT_DIR} ${_LIKWID_PATH_HINTS}
    DOC "LIKWID library")

#------------------------------------------------------------------------------#
#
#       Local (non-cached) variables
#
#------------------------------------------------------------------------------#

# this is local (directory-scoped) variable to the include path
# if cached variable is LIKWID_INCLUDE_DIR-NOTFOUND, then
# LIKWID_INCLUDE_DIRS will be empty
if(LIKWID_INCLUDE_DIR)
    list(APPEND LIKWID_INCLUDE_DIRS ${LIKWID_INCLUDE_DIR})
endif()

# this is local (directory-scoped) variable to the likwid library
# if cached variable is LIKWID_LIBRARY-NOTFOUND, then
# LIKWID_LIBRARIES will be empty
if(LIKWID_LIBRARY)
    get_filename_component(_LIB_DIR "${LIKWID_LIBRARY}" PATH)
    list(APPEND LIKWID_LIBRARIES ${LIKWID_LIBRARY})
    list(APPEND LIKWID_LIBRARY_DIRS ${_LIB_DIR})
    target_link_libraries(likwid-library INTERFACE ${LIKWID_LIBRARY})
endif()

#------------------------------------------------------------------------------#
#
#       Executables
#
#------------------------------------------------------------------------------#

set(LIKWID_EXECUTABLE_OPTIONS
    bench
    features
    genTopoCfg
    lua
    memsweeper
    mpirun
    perfctr
    perfscope
    pin
    powermeter
    setFrequencies
    topology)

foreach(_EXE ${LIKWID_EXECUTABLE_OPTIONS})
    find_program(LIKWID_${_EXE}_EXECUTABLE
        NAMES likwid-${_EXE}
        HINTS ${LIKWID_ROOT_DIR} ${_LIKWID_PATH_HINTS}
        PATHS ${LIKWID_ROOT_DIR} ${_LIKWID_PATH_HINTS}
        PATH_SUFFIXES bin
        DOC "LIKWID ${_EXE} executable"
        NO_DEFAULT_PATH)
endforeach()

#------------------------------------------------------------------------------#
#
#       Components
#
#------------------------------------------------------------------------------#

set(LIKWID_COMPONENT_OPTIONS
    gotcha
    hwloc
    lua
    pin
    appDaemon
    marker
    nvmarker)

message(STATUS "COMPONENTS: ${likwid_FIND_COMPONENTS}")
if("marker" IN_LIST likwid_FIND_COMPONENTS)
    target_compile_definitions(likwid-library INTERFACE LIKWID_PERFMON)
    set(LIKWID_COMPILE_DEFINITIONS "-DLIKWID_PERFMON")
    set(likwid_marker_FOUND ON)
    list(APPEND likwid_FOUND_COMPONENTS marker)
endif()

if("nvmarker" IN_LIST likwid_FIND_COMPONENTS)
    if(LIKWID_NVIDIA_INTERFACE)
        target_compile_definitions(likwid-library INTERFACE LIKWID_NVMON)
        set(LIKWID_COMPILE_DEFINITIONS "${LIKWID_COMPILE_DEFINITIONS} -DLIKWID_NVMON")
        set(likwid_nvmarker_FOUND ON)
    elseif(likwid_FIND_REQUIRED_nvmarker)
        list(APPEND _LIKWID_MISSING_COMPONENTS nvmarker)
    endif()
    list(APPEND likwid_FOUND_COMPONENTS nvmarker)
endif()

# loop over the remaining components which have libraries, e.g. liblikwid-${NAME}.so
foreach(_COMP ${likwid_FIND_COMPONENTS})
    if("${_COMP}" MATCHES "marker")
        continue()
    endif()
    # find the library, e.g. likwid-lua, likwid-gotcha, likwidpin, etc.
    find_library(LIKWID_${_COMP}_LIBRARY
        NAMES   likwid-${_COMP} likwid${_COMP}
        HINTS   ${LIKWID_ROOT_DIR} ${_LIKWID_PATH_HINTS}
        PATHS   ${LIKWID_ROOT_DIR} ${_LIKWID_PATH_HINTS}
        DOC     "LIKWID ${_COMP} library"
        PATH_SUFFIXES ${_LIKWID_LIB_SUFFIXES})

    # doesn't show up in GUI
    mark_as_advanced(LIKWID_${_COMP}_LIBRARY)

    if(LIKWID_${_COMP}_LIBRARY)
        get_filename_component(_LIB_DIR "${LIKWID_${_COMP}_LIBRARY}" PATH)
        list(APPEND LIKWID_LIBRARIES ${LIKWID_${_COMP}_LIBRARY})
        list(APPEND LIKWID_LIBRARY_DIRS ${_LIB_DIR})
        list(APPEND likwid_FOUND_COMPONENTS ${_COMP})
        target_link_libraries(likwid-library INTERFACE ${LIKWID_${_COMP}_LIBRARY})
    elseif(likwid_FIND_REQUIRED_${_COMP})
        # only append to missing if required
        list(APPEND _LIKWID_MISSING_COMPONENTS LIKWID_${_COMP}_LIBRARY)
    endif()
endforeach()

#------------------------------------------------------------------------------#
# find package variables, e.g. LIKWID_FOUND
# if all required components were found _LIKWID_MISSING_COMPONENTS will
# be empty
find_package_handle_standard_args(
    likwid
    REQUIRED_VARS
        LIKWID_ROOT_DIR LIKWID_INCLUDE_DIR LIKWID_LIBRARY ${_LIKWID_MISSING_COMPONENTS}
    HANDLE_COMPONENTS)

#------------------------------------------------------------------------------#

# cleanup temporary variables
unset(_LIB_DIR)
unset(_LIKWID_PATH_HINTS)
unset(_LIKWID_LIB_SUFFIXES)

cmake_policy(POP)