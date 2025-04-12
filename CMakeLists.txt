cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(CudaHomework LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)

set(SOURCES
    src/wavInterpolator.cu
    src/waver.cpp
)

add_executable(wavInterpolator ${SOURCES})

set_target_properties(wavInterpolator PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_STANDARD 11
    CUDA_ARCHITECTURES native
)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")

target_include_directories(wavInterpolator PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

add_executable(runTest test/run_test.cpp src/waver.cpp)

add_dependencies(runTest wavInterpolator)

target_include_directories(runTest PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

set_target_properties(runTest PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/test
)

file(COPY test/test_data DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/test)