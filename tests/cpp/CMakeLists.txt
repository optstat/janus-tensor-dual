# Find libtorch
find_package(Torch REQUIRED)


# Specify the executable
set(SOURCE_FILE1 tensordual_test.cpp)
set(SOURCE_FILE2 tensordualcuda_test.cu)
set(SOURCE_FILE3 tensordualcuda_sparse_test.cu)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
add_executable(dualtest ${SOURCE_FILE1})
add_executable(dualtest_cuda ${SOURCE_FILE2})
add_executable(dualtest_cuda_sparse ${SOURCE_FILE3})

# Include GTest
find_package(GTest REQUIRED)

# Include CUDA Toolkit (if needed)
find_package(CUDAToolkit)

# Set Thrust device system to CUDA explicitly
target_compile_definitions(dualtest_cuda PRIVATE THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA)

# Link CUDA, cuBLAS, and cuDNN (if applicable)
if(CUDAToolkit_FOUND)
    target_link_libraries(dualtest_cuda PRIVATE 
        CUDA::cudart         # CUDA Runtime
        CUDA::cublas         # cuBLAS library
        CUDA::cublasLt       # cuBLAS Lt (optional, for advanced operations)
    )
    target_include_directories(dualtest_cuda PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
else()
    message(WARNING "CUDA Toolkit not found. Ensure CUDA libraries are available.")
endif()

# Include directories
target_include_directories(dualtest
    PRIVATE
        ${TORCH_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/src/cpp
)

# Include directories for CUDA
target_include_directories(dualtest_cuda
    PRIVATE
        ${TORCH_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/src/cpp
)

target_include_directories(dualtest_cuda_sparse
    PRIVATE
        ${TORCH_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/src/cpp
)

# Link TACO, GTest, and libtorch
# Specify the Taco library
#set(TACO_LIB ${CMAKE_BINARY_DIR}/lib/libtaco.so)

target_link_libraries(dualtest
    PRIVATE
        ${TORCH_LIBRARIES}
        ${TACO_LIB}
        ${GTEST_LIBRARIES}
)

# Link GTest, and libtorch for CUDA
target_link_libraries(dualtest_cuda
    PRIVATE
        ${TORCH_LIBRARIES}
        ${GTEST_LIBRARIES}
)

target_link_libraries(dualtest_cuda_sparse
    PRIVATE
        ${TORCH_LIBRARIES}
        ${GTEST_LIBRARIES}
)

# Set C++ standard for the target
set_property(TARGET dualtest PROPERTY CXX_STANDARD 17)
set_property(TARGET dualtest_cuda PROPERTY CXX_STANDARD 17)

# Add the test executable to CTest
add_test(NAME DualTest COMMAND dualtest)
add_test(NAME DualTestCuda COMMAND dualtest_cuda)
add_test(NAME DualTestCudaSparse COMMAND dualtest_cuda_sparse)

