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

# -------------------------------- Provide Custom Paths -------------------------------
set(PYTHON_EXECUTABLE "" CACHE STRING "Path to the Python executable to use")
set(TORCH_PATH "" CACHE PATH "Path to Torch installation")
set(PYBIND11_PATH "" CACHE PATH "Path to Pybind11 installation")

if(PYTHON_EXECUTABLE)
    set(Python_EXECUTABLE ${PYTHON_EXECUTABLE})
endif()

if(TORCH_PATH)
    list(APPEND CMAKE_PREFIX_PATH ${TORCH_PATH})
endif()

if(PYBIND11_PATH)
    list(APPEND CMAKE_PREFIX_PATH ${PYBIND11_PATH})
endif()
# --------------------------------------------------------------------------------------

# ------------------------------ Python and site-packages ------------------------------
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
# --------------------------------------------------------------------------------------

message(STATUS "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")

find_package(Torch REQUIRED)

# Define project structure
set(SYNAPX_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include/synapx)
set(SYNAPX_SRC_DIR ${PROJECT_SOURCE_DIR}/src)
set(SYNAPX_BINDINGS_DIR ${PROJECT_SOURCE_DIR}/bindings)

# Collect source files (excluding bindings)
file(GLOB_RECURSE SYNAPX_SOURCES "${SYNAPX_SRC_DIR}/*.cpp")

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