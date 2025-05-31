# Allow the user to specify a pybind11 installation, if not, the one installed
# in the current virtual environment will be used
set(PYBIND11_PATH "" CACHE PATH "Path to Pybind11 installation")

if(PYBIND11_PATH)
    list(APPEND CMAKE_PREFIX_PATH ${PYBIND11_PATH})
endif()

find_package(pybind11 REQUIRED)

# Find torch_python library
find_library(TORCH_PYTHON_LIBRARY torch_python
    PATHS ${TORCH_INSTALL_PREFIX}/lib
    PATH_SUFFIXES lib
    NO_DEFAULT_PATH
)

if(TORCH_PYTHON_LIBRARY)
    message(STATUS "Found torch_python: ${TORCH_PYTHON_LIBRARY}")
else()
    message(WARNING "torch_python library not found - PyTorch tensor conversion may not work")
endif()

# Create Python module
pybind11_add_module(synapx_bindings synapx_bindings.cpp)
add_library(synapx::bindings ALIAS synapx_bindings)

# Set C++ standard to match PyTorch
set_target_properties(synapx_bindings PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)

# Link against core library and PyTorch libraries
target_link_libraries(synapx_bindings PRIVATE synapx::synapx)

# Link torch_python if found (essential for PyTorch tensor conversion)
if(TORCH_PYTHON_LIBRARY)
    target_link_libraries(synapx_bindings PRIVATE ${TORCH_PYTHON_LIBRARY})
endif()

# Set output properties
set_target_properties(synapx_bindings 
    PROPERTIES
        PREFIX ""                             # no "lib" prefix
        OUTPUT_NAME "_C"
        SUFFIX "${PYTHON_MODULE_EXTENSION}"   # .so or .pyd
        CXX_VISIBILITY_PRESET "hidden"
)