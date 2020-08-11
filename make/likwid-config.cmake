# legacy variables
set(LIKWID_INCLUDES @PREFIX@/include)

# uses IN_LIST so return if less than 3.3.2
if(CMAKE_VERSION VERSION_LESS 3.3.2)
    set(LIKWID_LIBRARIES @PREFIX@/lib/liblikwid.so -llikwid)
    return()
endif()

# Support new if() IN_LIST operator
cmake_policy(PUSH)
cmake_policy(SET CMP0057 NEW)

# updates
include(FindPackageHandleStandardArgs)

# variables configured during install
set(_DEFAULT_PREFIXES @PREFIX@ @LIBPREFIX@ @BINPREFIX@)

# do not change MOVE_LIKWID_INSTALL line, used when
# moving installation (override above defaults)
# @MOVE_LIKWID_INSTALL@

list(GET _DEFAULT_PREFIXES 0 _PREFIX)
list(GET _DEFAULT_PREFIXES 1 _LIBPREFIX)
list(GET _DEFAULT_PREFIXES 2 _BINPREFIX)

# cache this variable so the user can change it if desired
set(LIKWID_ROOT_DIR ${_PREFIX} CACHE PATH "Path to LIKWID installation")

# put the cached variable first to give it priority
# and then append the paths from the install
set(_LIKWID_PATH_HINTS ${LIKWID_ROOT_DIR} ${_PREFIX} ${_LIBPREFIX} ${_BINPREFIX})
set(_LIKWID_LIB_SUFFIXES lib lib64)

# relevant options
set(LIKWID_NVIDIA_INTERFACE @NVIDIA_INTERFACE@)
set(LIKWID_FORTRAN_INTERFACE @FORTRAN_INTERFACE@)

#------------------------------------------------------------------------------#
#
#       Interface library
#
#------------------------------------------------------------------------------#

# create an imported target library that CMake projects can just "link" to.
# e.g. target_link_libraries(myexe PUBLIC likwid::likwid)
add_library(likwid-library INTERFACE)
add_library(likwid::likwid ALIAS likwid-library)

#------------------------------------------------------------------------------#
#
#       Include directory and library
#
#------------------------------------------------------------------------------#

# this is the cached variable to the include path
# if not found, will be LIKWID_INCLUDE_DIR-NOTFOUND
find_path(LIKWID_INCLUDE_DIR
    NAMES likwid.h
    HINTS ${_LIKWID_PATH_HINTS}
    PATHS ${_LIKWID_PATH_HINTS}
    PATH_SUFFIXES include
    DOC "LIKWID profiler include directory"
    NO_DEFAULT_PATH)

find_library(LIKWID_LIBRARY
    NAMES likwid
    PATH_SUFFIXES ${_LIKWID_LIB_SUFFIXES}
    HINTS ${_LIKWID_PATH_HINTS}
    PATHS ${_LIKWID_PATH_HINTS}
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
        HINTS ${_LIKWID_PATH_HINTS}
        PATHS ${_LIKWID_PATH_HINTS}
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

if("marker" IN_LIST likwid_FIND_COMPONENTS)
    target_compile_definitions(likwid-library INTERFACE LIKWID_PERFMON)
    set(LIKWID_COMPILE_DEFINITIONS "-DLIKWID_PERFMON")
    set(likwid_marker_FOUND ON)
endif()

if("nvmarker" IN_LIST likwid_FIND_COMPONENTS)
    if(LIKWID_NVIDIA_INTERFACE)
        target_compile_definitions(likwid-library INTERFACE LIKWID_NVMON)
        set(LIKWID_COMPILE_DEFINITIONS "${LIKWID_COMPILE_DEFINITIONS} -DLIKWID_NVMON")
        set(likwid_nvmarker_FOUND ON)
    elseif(likwid_FIND_REQUIRED_nvmarker)
        set(likwid_nvmarker_FOUND OFF)
    endif()
endif()

# loop over the remaining components which have libraries, e.g. liblikwid-${NAME}.so
foreach(_COMP ${likwid_FIND_COMPONENTS})
    if("${_COMP}" MATCHES "marker")
        continue()
    endif()
    # find the library, e.g. likwid-lua, likwid-gotcha, likwidpin, etc.
    find_library(LIKWID_${_COMP}_LIBRARY
        NAMES   likwid-${_COMP} likwid${_COMP}
        HINTS   ${_LIKWID_PATH_HINTS}
        PATHS   ${_LIKWID_PATH_HINTS}
        DOC     "LIKWID ${_COMP} library"
        PATH_SUFFIXES ${_LIKWID_LIB_SUFFIXES})

    # doesn't show up in GUI
    mark_as_advanced(LIKWID_${_COMP}_LIBRARY)

    if(LIKWID_${_COMP}_LIBRARY)
        get_filename_component(_LIB_DIR "${LIKWID_${_COMP}_LIBRARY}" PATH)
        list(APPEND LIKWID_LIBRARIES ${LIKWID_${_COMP}_LIBRARY})
        list(APPEND LIKWID_LIBRARY_DIRS ${_LIB_DIR})
        list(APPEND likwid_FOUND_COMPONENTS ${_COMP})
        set(likwid_${_COMP}_FOUND ON)
        target_link_libraries(likwid-library INTERFACE ${LIKWID_${_COMP}_LIBRARY})
    elseif(likwid_FIND_REQUIRED_${_COMP})
        # only append to missing if required
        set(likwid_${_COMP}_FOUND OFF)
    endif()
endforeach()

#------------------------------------------------------------------------------#
# find package variables, e.g. LIKWID_FOUND
find_package_handle_standard_args(
    likwid
    REQUIRED_VARS
        LIKWID_ROOT_DIR LIKWID_INCLUDE_DIR LIKWID_LIBRARY
    HANDLE_COMPONENTS)

#------------------------------------------------------------------------------#

# cleanup temporary variables
unset(_PREFIX)
unset(_LIB_DIR)
unset(_LIBPREFIX)
unset(_BINPREFIX)
unset(_DEFAULT_PREFIXES)
unset(_LIKWID_PATH_HINTS)
unset(_LIKWID_LIB_SUFFIXES)

cmake_policy(POP)
