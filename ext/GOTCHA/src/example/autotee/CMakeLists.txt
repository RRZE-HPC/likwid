add_library(autotee SHARED autotee.c)
add_executable(autotee_test test_autotee.c)
target_link_libraries(autotee gotcha)
target_link_libraries(autotee_test autotee)


