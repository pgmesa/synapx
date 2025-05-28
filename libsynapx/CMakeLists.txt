cmake_minimum_required(VERSION 3.22 FATAL_ERROR)
project(synapx VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
option(CMAKE_INCREMENTAL_BUILD "Enable incremental builds" ON)

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Runtime output directory: ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
message(STATUS "Debug flags:   ${CMAKE_CXX_FLAGS_DEBUG}")
message(STATUS "Release flags: ${CMAKE_CXX_FLAGS_RELEASE}")

# ========================= Python and site-packages =========================
# Allow to provide a custom path
set(PYTHON_EXECUTABLE "" STRING "Path to the Python executable to use")

if(PYTHON_EXECUTABLE)
    set(Python_EXECUTABLE ${PYTHON_EXECUTABLE})
endif()

# If not defined, find the current active Python environment 
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# Print the result for verification
message(STATUS "Python Interpreter: ${Python_EXECUTABLE}")
message(STATUS "Python Version: ${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}")

# Get the parent directory of Python_EXECUTABLE
get_filename_component(PYTHON_EXECUTABLE_DIR ${Python_EXECUTABLE} DIRECTORY)
message(STATUS "Python executable directory: ${PYTHON_EXECUTABLE_DIR}")

if(UNIX AND NOT WIN32)
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())"
        OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    message(STATUS "Python site-packages directory: ${PYTHON_SITE_PACKAGES}")
    list(APPEND CMAKE_PREFIX_PATH "${PYTHON_SITE_PACKAGES}")
else()
    # Path for Windows or when using a venv
    if(WIN32)
        list(APPEND CMAKE_PREFIX_PATH "${PYTHON_EXECUTABLE_DIR}\\Lib\\site-packages")
    else()
        list(APPEND CMAKE_PREFIX_PATH 
            "${PYTHON_EXECUTABLE_DIR}/lib/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}/site-packages")
    endif()
endif()
# ============================================================================

message(STATUS "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")

find_package(Torch CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)

# Define project structure
set(SYNAPX_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
set(SYNAPX_SRC_DIR ${PROJECT_SOURCE_DIR}/src)
set(SYNAPX_BINDINGS_DIR ${PROJECT_SOURCE_DIR}/bindings)
set(SYNAPX_EXAMPLES_DIR ${PROJECT_SOURCE_DIR}/examples)

# Collect source files (excluding bindings)
file(GLOB_RECURSE SYNAPX_SOURCES "${SYNAPX_SRC_DIR}/*.cpp")

# Create the core library
add_library(synapx SHARED ${SYNAPX_SOURCES})

# Set include directories
target_include_directories(synapx
    PUBLIC
        ${SYNAPX_INCLUDE_DIR}
        ${TORCH_INCLUDE_DIRS}
)

# Link dependencies
target_link_libraries(synapx
    PUBLIC
        ${TORCH_LIBRARIES}
    PRIVATE
        spdlog::spdlog
)

add_library(synapx::synapx ALIAS synapx)

# Option to build examples
option(BUILD_EXAMPLES "Build C++ Examples" ON)
if(BUILD_EXAMPLES)
    add_subdirectory("${SYNAPX_EXAMPLES_DIR}")
endif()

# Option to build Python bindings
option(BUILD_PYTHON_BINDINGS "Build Python bindings" ON)
if(BUILD_PYTHON_BINDINGS)
    add_subdirectory("${SYNAPX_BINDINGS_DIR}")
endif()