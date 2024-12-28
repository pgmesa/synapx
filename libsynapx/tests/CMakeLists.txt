# Find dependencies
find_package(Torch REQUIRED)

# Function to create test executable with proper settings
function(add_synapx_test TEST_NAME SOURCE_FILES)
    add_executable(${TEST_NAME} ${SOURCE_FILES})
    
    target_link_libraries(${TEST_NAME}
        PRIVATE
            synapx
            ${TORCH_LIBRARIES}
    )
    
    # Windows DLL handling
    if(MSVC)
        file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
        add_custom_command(TARGET ${TEST_NAME} 
                          POST_BUILD
                          COMMAND ${CMAKE_COMMAND} -E copy_if_different
                          ${TORCH_DLLS}
                          $<TARGET_FILE_DIR:${TEST_NAME}>)
    endif()
    
    # Add to CTest
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endfunction()

# Add different test executables
add_synapx_test(tensor_test "tensor_test.cpp")
add_synapx_test(matmul_test "matmul_test.cpp")

# Set working directory for tests if needed
set_tests_properties(tensor_test PROPERTIES
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)