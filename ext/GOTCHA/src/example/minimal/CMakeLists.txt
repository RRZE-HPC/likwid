add_library(wrap_me SHARED sampleLib.c)
add_executable(symb_look symbolLookup.c)
target_link_libraries(wrap_me gotcha)
target_link_libraries(symb_look wrap_me gotcha)

