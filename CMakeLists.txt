cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

# Project Name
project(CudaHomework LANGUAGES CXX CUDA)

# Set the path to the CUDA toolkit if necessary
# Find CUDA toolkit
find_package(CUDA REQUIRED)

# Set the source files
set(SOURCES
    src/wavInterpolator.cu
    src/waver.cpp
)

# Create the executable
add_executable(wavInterpolator ${SOURCES})

# Specify the CUDA flags
set_target_properties(wavInterpolator PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_STANDARD 11
    CUDA_ARCHITECTURES native
)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")

# Include directories
target_include_directories(wavInterpolator PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)