cmake_minimum_required(VERSION 3.20)

project(swizzle LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_ARCHITECTURES 86)

# disable cuda compiler optimization
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O0 -Xcicc -O0 -Xptxas -O0")

include_directories(${PROJECT_SOURCE_DIR}/src/utils)

add_subdirectory(src)

add_subdirectory(examples)