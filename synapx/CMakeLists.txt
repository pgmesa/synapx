cmake_minimum_required(VERSION 3.12)
project(tensor_cpp_project)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(pybind11_DIR "C:/Users/pablo/anaconda3/envs/base-3.9/Lib/site-packages/pybind11/share/cmake/pybind11/")
find_package(pybind11 REQUIRED)

include_directories(${PROJECT_SOURCE_DIR})

add_library(tensor_cpp_lib STATIC tensor.cpp)

pybind11_add_module(tensor_python tensor_bindings.cpp)

target_link_libraries(tensor_python PRIVATE tensor_cpp_lib)

set_target_properties(tensor_python PROPERTIES
    OUTPUT_NAME cpptensor
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)