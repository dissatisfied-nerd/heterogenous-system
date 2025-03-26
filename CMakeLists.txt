cmake_minimum_required(VERSION 3.18)
project(HeterogenousSystem LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
endif()

include_directories(${CMAKE_SOURCE_DIR}/include)

add_subdirectory(matrix)

add_executable(${PROJECT_NAME} main.cu)

target_link_libraries(${PROJECT_NAME} PRIVATE matrix_lib)
target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX)
