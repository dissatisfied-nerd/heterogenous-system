file(GLOB MEMORY_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.c*)

add_library(memory_lib STATIC ${MEMORY_SRC})

target_include_directories(memory_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
