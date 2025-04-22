cmake_minimum_required(VERSION 3.18)
project(HeterogenousSystem LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(nlohmann_json REQUIRED)

enable_testing()

include_directories(${CMAKE_SOURCE_DIR}/include)

add_subdirectory(matrix)
add_subdirectory(mandelbrot)
add_subdirectory(utils)
add_subdirectory(tests)
add_subdirectory(memory)
add_subdirectory(balancing)

add_executable(${PROJECT_NAME} main.cu)

target_link_libraries(${PROJECT_NAME} PRIVATE matrix_lib)
target_link_libraries(${PROJECT_NAME} PRIVATE mandelbrot_lib)
target_link_libraries(${PROJECT_NAME} PRIVATE utils_lib)
target_link_libraries(${PROJECT_NAME} PRIVATE memory_lib)
target_link_libraries(${PROJECT_NAME} PRIVATE balancing_lib)
