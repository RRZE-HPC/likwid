include(GNUInstallDirs)
set(GOTCHA_SOURCES
  gotcha_utils.c 
  gotcha.c 
  gotcha_auxv.c 
  libc_wrappers.c 
  elf_ops.c 
  hash.c 
  tool.c 
  library_filters.c
  gotcha_dl.c
  translations.c
)

add_library(gotcha SHARED ${GOTCHA_SOURCES})


set_target_properties(gotcha PROPERTIES SOVERSION ${LIBTOOL_INTERFACE})
set_target_properties(gotcha PROPERTIES VERSION "${LIBTOOL_INTERFACE}.${LIBTOOL_REVISION}.${LIBTOOL_AGE}")
set_target_properties(gotcha PROPERTIES COMPILE_FLAGS "-fvisibility=${DEFAULT_SYMBOL_VISIBILITY}")
set_target_properties(gotcha PROPERTIES LINK_FLAGS "-fvisibility=${DEFAULT_SYMBOL_VISIBILITY}")
if(GOTCHA_ENABLE_TESTS)
add_library(gotcha_no_libc SHARED ${GOTCHA_SOURCES})
set_target_properties(gotcha_no_libc PROPERTIES SOVERSION ${LIBTOOL_INTERFACE})
set_target_properties(gotcha_no_libc PROPERTIES VERSION "${LIBTOOL_INTERFACE}.${LIBTOOL_REVISION}.${LIBTOOL_AGE}")
set_target_properties(gotcha_no_libc PROPERTIES COMPILE_DEFINITIONS "FORCE_NO_LIBC")
set_target_properties(gotcha_no_libc PROPERTIES COMPILE_FLAGS "-fvisibility=${DEFAULT_SYMBOL_VISIBILITY}")
set_target_properties(gotcha_no_libc PROPERTIES LINK_FLAGS "-fvisibility=${DEFAULT_SYMBOL_VISIBILITY}")
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

install(TARGETS gotcha EXPORT gotcha-targets DESTINATION ${CMAKE_INSTALL_LIBDIR}) 
install(EXPORT gotcha-targets DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/gotcha)

add_subdirectory(example)
