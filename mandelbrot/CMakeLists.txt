file(GLOB MANDELBROT_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.c*)

add_library(mandelbrot_lib STATIC ${MANDELBROT_SRC})

target_include_directories(mandelbrot_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
