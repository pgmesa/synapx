
find_package(Python COMPONENTS Interpreter Development REQUIRED)
message(STATUS "Python Interpreter: ${Python_EXECUTABLE}")

find_package(pybind11 REQUIRED)

# Create Python module
pybind11_add_module(synapx_module
    python_bindings.cpp
)

# Link against core library
target_link_libraries(synapx_module PRIVATE synapx)

# Set output properties
set_target_properties(synapx_module PROPERTIES
    OUTPUT_NAME "_C"  # Will create _C.pyd
    # LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../../synapx"
    # RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../synapx"
)