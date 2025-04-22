file(GLOB UTILS_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.c*)

add_library(utils_lib STATIC ${UTILS_SRC})

target_include_directories(utils_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
