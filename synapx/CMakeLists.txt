cmake_minimum_required(VERSION 3.12)
project(synapx_project)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set optimization flags based on the compiler
# Set optimization flags based on the compiler
if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    # Use /O2 for MSVC and enable AVX
    set(CMAKE_CXX_FLAGS_RELEASE "/O2")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    # Use -O3 and enable AVX/AVX2 for GCC/Clang
    set(CMAKE_CXX_FLAGS_RELEASE "-O3")
endif()

set(CMAKE_BUILD_TYPE Release)

include_directories(${PROJECT_SOURCE_DIR}/_C)

# Conda environment path
set(CONDA_ENV_PATH "C:/Users/pablo/anaconda3/envs/base-3.9")

# OpenBLAS
set(OpenBLAS_HOME "${CONDA_ENV_PATH}/Library")
include_directories("${OpenBLAS_HOME}/include")
link_directories("${OpenBLAS_HOME}/lib")
link_directories("${OpenBLAS_HOME}/include")

# pybind11
set(pybind11_DIR "${CONDA_ENV_PATH}/Lib/site-packages/pybind11/share/cmake/pybind11/")
find_package(pybind11 REQUIRED)

# Add all necessary source files from the _C directory
set(TENSOR_SOURCES
    ${PROJECT_SOURCE_DIR}/_C/tensor.cpp
    # ${PROJECT_SOURCE_DIR}/_C/dtype.cpp
    # ${PROJECT_SOURCE_DIR}/_C/utils.cpp
)

# Build the cpp_lib static library
add_library(synapx_cpp_lib STATIC ${TENSOR_SOURCES})

# Link OpenBLAS to the cpp_lib
target_link_libraries(synapx_cpp_lib PRIVATE openblas)

# Python bindings
pybind11_add_module(synapx_python tensor_bindings.cpp)

# Link cpp_lib to the Python bindings
target_link_libraries(synapx_python PRIVATE synapx_cpp_lib openblas)

set_target_properties(synapx_python PROPERTIES
    OUTPUT_NAME synapx_c
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
