include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/release-1.12.1.zip
)

set(INSTALL_GTEST OFF)
set(BUILD_GMOCK OFF)
set(BUILD_GTEST ON)
FetchContent_MakeAvailable(googletest)

add_executable(matrix_test matrix_test.cu)
target_link_libraries(matrix_test matrix_lib utils_lib gtest gtest_main)

add_executable(memory_test memory_test.cu)
target_link_libraries(memory_test memory_lib utils_lib gtest gtest_main)

add_test(matrix_test matrix_test)
add_test(memory_test memory_test)
