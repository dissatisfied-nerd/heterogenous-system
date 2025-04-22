file(GLOB BALANCING_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.c*)

add_library(balancing_lib STATIC ${BALANCING_SRC})

target_include_directories(balancing_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
