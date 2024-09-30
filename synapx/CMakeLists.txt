cmake_minimum_required(VERSION 3.12)
project(synapx_project)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${PROJECT_SOURCE_DIR}/_C)

set(pybind11_DIR "C:/Users/pablo/anaconda3/envs/base-3.9/Lib/site-packages/pybind11/share/cmake/pybind11/")
find_package(pybind11 REQUIRED)

# Add all necessary source files from the _C directory
set(TENSOR_SOURCES
    ${PROJECT_SOURCE_DIR}/_C/tensor.cpp
    ${PROJECT_SOURCE_DIR}/_C/dtype.cpp
    ${PROJECT_SOURCE_DIR}/_C/utils.cpp
)

# Build the cpp_lib static library
add_library(synapx_cpp_lib STATIC ${TENSOR_SOURCES})

# Python bindings
pybind11_add_module(synapx_python tensor_bindings.cpp)

# Link cpp_lib to the Python bindings
target_link_libraries(synapx_python PRIVATE synapx_cpp_lib)

set_target_properties(synapx_python PROPERTIES
    OUTPUT_NAME synapx_c
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)