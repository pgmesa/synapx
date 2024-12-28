cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(synapx VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
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
list(APPEND CMAKE_PREFIX_PATH ${PROJECT_SOURCE_DIR}/external/libtorch)
list(APPEND CMAKE_PREFIX_PATH $ENV{CONDA_PREFIX})
if(WIN32)
    list(APPEND CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}\\Lib\\site-packages")
endif()

message(STATUS "${CMAKE_PREFIX_PATH}")

# Find dependencies
find_package(Torch REQUIRED)

# Define project structure
set(SYNAPX_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include/synapx)
set(SYNAPX_SRC_DIR ${PROJECT_SOURCE_DIR}/src)
set(SYNAPX_BINDINGS_DIR ${PROJECT_SOURCE_DIR}/bindings)

# Collect source files (excluding bindings)
file(GLOB_RECURSE SYNAPX_SOURCES 
    "${SYNAPX_SRC_DIR}/*.cpp"
)

# Create the core library
add_library(synapx SHARED ${SYNAPX_SOURCES})

# Set include directories
target_include_directories(synapx
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${TORCH_INCLUDE_DIRS}
)

# Link dependencies
target_link_libraries(synapx
    PUBLIC
        ${TORCH_LIBRARIES}
)

# Option to build tests
option(BUILD_CPP_TESTS "Build C++ Tests" OFF)
if(BUILD_CPP_TESTS)
    add_subdirectory(tests)
endif()

# Option to build Python bindings
option(BUILD_PYTHON_BINDINGS "Build Python bindings" ON)
if(BUILD_PYTHON_BINDINGS)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/bindings 
                     ${CMAKE_CURRENT_BINARY_DIR}/bindings)
endif()

# Windows-specific DLL handling for libtorch
# if(MSVC)
#     file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
#     add_custom_command(TARGET synapx POST_BUILD
#         COMMAND ${CMAKE_COMMAND} -E copy_if_different
#         ${TORCH_DLLS}
#         $<TARGET_FILE_DIR:synapx>)
# endif()