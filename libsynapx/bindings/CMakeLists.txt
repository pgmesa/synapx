
# Allow the user to specify a pybind11 installation, if not, the one installed
# in the current virtual environment will be used
set(PYBIND11_PATH "" PATH "Path to Pybind11 installation")

if(PYBIND11_PATH)
    list(APPEND CMAKE_PREFIX_PATH ${PYBIND11_PATH})
endif()

find_package(pybind11 REQUIRED)

# Create Python module
pybind11_add_module(synapx_bindings python_bindings.cpp)

# Link against core library
target_link_libraries(synapx_bindings PRIVATE synapx::synapx)

# Set output properties
set_target_properties(synapx_bindings 
    PROPERTIES
        OUTPUT_NAME "_C"
)