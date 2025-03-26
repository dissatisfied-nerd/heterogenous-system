file(GLOB MATRIX_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.c*)

add_library(matrix_lib STATIC ${MATRIX_SRC})

target_include_directories(matrix_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
