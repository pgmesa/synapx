
find_package(argparse CONFIG REQUIRED)

# On Windows/MSVC, grab all the Torch DLLs so we can bundle them per‐example
if(MSVC)
    file(GLOB TORCH_DLLS
        "${TORCH_INSTALL_PREFIX}/lib/*.dll"
    )
endif()

# Helper function to build an example and copy its dependencies
function(add_synapx_example EXAMPLE_NAME SOURCE_FILE)
    add_executable(${EXAMPLE_NAME} ${SOURCE_FILE})

    set_target_properties(${EXAMPLE_NAME} PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)

    target_link_libraries(${EXAMPLE_NAME}
        PRIVATE
            synapx::synapx
            spdlog::spdlog
            argparse::argparse
    )

    if(MSVC)
        # After building the EXE, copy each Torch DLL next to it
        foreach(_dll ${TORCH_DLLS})
            add_custom_command(TARGET ${EXAMPLE_NAME} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${_dll}"
                        $<TARGET_FILE_DIR:${EXAMPLE_NAME}>
            )
        endforeach()

        # Also copy the synapx DLL itself
        add_custom_command(TARGET ${EXAMPLE_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    $<TARGET_FILE:synapx>
                    $<TARGET_FILE_DIR:${EXAMPLE_NAME}>
        )
    endif()
endfunction()

# Build each example
add_synapx_example(example_add    example_add.cpp)
add_synapx_example(example_matmul example_matmul.cpp)
# add_synapx_example(example_torch_matmul example_torch_matmul.cpp)