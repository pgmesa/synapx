cmake_minimum_required(VERSION 3.22 FATAL_ERROR)
project(synapx VERSION 0.1.0 LANGUAGES CXX)

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

# ========================= Python and site-packages ==========================
# ─────────────────────────────────────────────────────────────────────────────
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
# =============================================================================

# ============================== SynapX Library ===============================
# ─────────────────────────────────────────────────────────────────────────────
message(STATUS "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")

find_package(Torch CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)

# Parse major and minor version from Torch_VERSION
string(REGEX MATCH "^[0-9]+" TORCH_VERSION_MAJOR "${Torch_VERSION}")
string(REGEX MATCH "^[0-9]+\\.([0-9]+)" _match "${Torch_VERSION}")
string(REGEX REPLACE "^[0-9]+\\.([0-9]+).*" "\\1" TORCH_VERSION_MINOR "${Torch_VERSION}")

message(STATUS "Torch version: ${Torch_VERSION}")
message(STATUS "Torch major version: ${TORCH_VERSION_MAJOR}")
message(STATUS "Torch minor version: ${TORCH_VERSION_MINOR}")

# Define project structure
set(SYNAPX_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
set(SYNAPX_SRC_DIR ${PROJECT_SOURCE_DIR}/src)
set(SYNAPX_BINDINGS_DIR ${PROJECT_SOURCE_DIR}/bindings)
set(SYNAPX_EXAMPLES_DIR ${PROJECT_SOURCE_DIR}/examples)

# Collect source files (excluding bindings)
file(GLOB_RECURSE SYNAPX_SOURCES "${SYNAPX_SRC_DIR}/*.cpp")

# Create the core library
add_library(synapx SHARED ${SYNAPX_SOURCES})

target_compile_definitions(synapx
    PRIVATE
        SYNAPX_EXPORTS
    PUBLIC
        SPDLOG_HEADER_ONLY
        SPDLOG_FMT_EXTERNAL  # use external fmt headers
        FMT_HEADER_ONLY      # fmt header-only
)

# Set symbol visibility to hidden by default
if(NOT WIN32)
    target_compile_options(synapx PRIVATE -fvisibility=hidden)
endif()

# Set include directories
target_include_directories(synapx
    PUBLIC
        $<BUILD_INTERFACE:${SYNAPX_INCLUDE_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        ${TORCH_INCLUDE_DIRS}
)

# Link dependencies
target_link_libraries(synapx
    PUBLIC
        ${TORCH_LIBRARIES}
        spdlog::spdlog_header_only
)

add_library(synapx::synapx ALIAS synapx)
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Option to build examples
option(BUILD_EXAMPLES "Build C++ Examples" ON)
if(BUILD_EXAMPLES)
    add_subdirectory("${SYNAPX_EXAMPLES_DIR}")
endif()
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Option to build Python bindings
option(BUILD_PYTHON_BINDINGS "Build Python bindings" ON)
if(BUILD_PYTHON_BINDINGS)
    add_subdirectory("${SYNAPX_BINDINGS_DIR}")

    # =============================== Installation ================================
    # ─────────────────────────────────────────────────────────────────────────────
    # 1) standard install dirs and package‐config helpers
    include(GNUInstallDirs)
    include(CMakePackageConfigHelpers)
    # ─────────────────────────────────────────────────────────────────────────────

    # ─────────────────────────────────────────────────────────────────────────────
    # 2) install libraries and headers
    set(LIBTORCH_VERSIONED_DIR "libtorch-${TORCH_VERSION_MAJOR}.${TORCH_VERSION_MINOR}.X")
    set(SYNAPX_INSTALL_LIBDIR "${CMAKE_INSTALL_LIBDIR}/${LIBTORCH_VERSIONED_DIR}")
    install(TARGETS synapx synapx_bindings
        EXPORT      synapxTargets
        RUNTIME     DESTINATION ${SYNAPX_INSTALL_LIBDIR}   # .dll / .pyd on Windows
        LIBRARY     DESTINATION ${SYNAPX_INSTALL_LIBDIR}   # .so on UNIX
        ARCHIVE     DESTINATION ${SYNAPX_INSTALL_LIBDIR}   # .lib on Windows, .a on UNIX
    )

    install(DIRECTORY ${SYNAPX_INCLUDE_DIR}/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
    )
    # ─────────────────────────────────────────────────────────────────────────────

    # ─────────────────────────────────────────────────────────────────────────────
    # 3) write out version-compat file
    write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/SynapxConfigVersion.cmake"
        VERSION       ${PROJECT_VERSION}
        COMPATIBILITY AnyNewerVersion
    )

    # 4) configure main Config file from template
    configure_package_config_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/SynapxConfig.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/SynapxConfig.cmake"
        INSTALL_DESTINATION "${CMAKE_INSTALL_DATADIR}/cmake/synapx"
    )

    # 5) install the generated Config files
    install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/SynapxConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/SynapxConfigVersion.cmake"
    DESTINATION "${CMAKE_INSTALL_DATADIR}/cmake/synapx"
    )
    # ─────────────────────────────────────────────────────────────────────────────
endif()
# ─────────────────────────────────────────────────────────────────────────────