# Find libtorch
find_package(Torch REQUIRED)


# Specify the executable
set(SOURCE_FILE tensordual_test.cpp)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
add_executable(dualtest ${SOURCE_FILE})

# Include GTest
find_package(GTest REQUIRED)

# Include CUDA Toolkit (if needed)
find_package(CUDAToolkit)
if(CUDAToolkit_FOUND)
    target_link_libraries(dualtest PRIVATE CUDA::cudart)
    target_include_directories(dualtest PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
else()
    message(WARNING "CUDA Toolkit not found. Building without CUDA support.")
endif()

# Include directories
target_include_directories(dualtest
    PRIVATE
        ${TORCH_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/src/cpp
)

# Link TACO, GTest, and libtorch
# Specify the Taco library
set(TACO_LIB ${CMAKE_BINARY_DIR}/lib/libtaco.so)

target_link_libraries(dualtest
    PRIVATE
        ${TORCH_LIBRARIES}
        ${TACO_LIB}
        ${GTEST_LIBRARIES}
)

# Set C++ standard for the target
set_property(TARGET dualtest PROPERTY CXX_STANDARD 17)

# Add the test executable to CTest
add_test(NAME DualTest COMMAND dualtest)

# Ensure the test executable is built and tests are run
add_custom_target(run_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    DEPENDS dualtest
)
