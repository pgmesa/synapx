cmake_minimum_required(VERSION 3.12)
project(synapx_project)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set optimization flags based on the compiler
if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /Ob2 /DNDEBUG /arch:AVX2")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -mavx2")
endif()

# Verify Conda environment is active
if(NOT DEFINED ENV{CONDA_PREFIX})
    message(FATAL_ERROR "No active Conda environment found! Please activate your Conda environment first.")
endif()

# Add conda environment to CMAKE_PREFIX_PATH
list(APPEND CMAKE_PREFIX_PATH $ENV{CONDA_PREFIX})
if(WIN32)
    list(APPEND CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}/Library")
endif()

# Find dependencies
# set(XTENSOR_USE_XSIMD 1)
# set(XTENSOR_USE_TBB 1)
# -- xsimd and TBB dependencies are automatically
# -- searched when the following is executed
find_package(xtensor REQUIRED)   # Will find conda-installed xtensor
find_package(pybind11 REQUIRED)  # Will find conda-installed pybind11
find_package(BLAS REQUIRED)      # Will find conda-installed OpenBLAS

# Define project structure
set(SYNAPX_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include/synapx)
set(SYNAPX_SRC_DIR ${PROJECT_SOURCE_DIR}/src)
set(SYNAPX_BINDINGS_DIR ${PROJECT_SOURCE_DIR}/bindings)

# Collect source files (excluding bindings)
file(GLOB_RECURSE SYNAPX_SOURCES 
    "${SYNAPX_SRC_DIR}/*.cpp"
)

# Collect binding sources
file(GLOB_RECURSE BINDING_SOURCES 
    "${SYNAPX_BINDINGS_DIR}/*.cpp"
)

# Build the cpp_lib static library
add_library(synapx_cpp_lib STATIC ${SYNAPX_SOURCES})
target_include_directories(synapx_cpp_lib 
    PUBLIC 
        ${PROJECT_SOURCE_DIR}/include
    PRIVATE 
        ${SYNAPX_SRC_DIR}
)

# Link libraries to cpp_lib
target_link_libraries(synapx_cpp_lib 
    PUBLIC 
        xtensor
        ${BLAS_LIBRARIES}
)

# Python bindings
pybind11_add_module(synapx_python ${BINDING_SOURCES})
target_include_directories(synapx_python 
    PRIVATE 
        ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(synapx_python PRIVATE synapx_cpp_lib)

# Set output properties for Python module
set_target_properties(synapx_python PROPERTIES
    OUTPUT_NAME "synapx_c"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/synapx"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/synapx"
)