#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cassert>
#include "../../src/cpp/cudatensordense.cu"
// Include your VectorBool implementation here
using namespace janus;

// CUDA kernel for testing boolIndexGet
__global__ void testBoolIndexGet(bool* input, int start, int end, bool* result) {
    boolIndexGet(input, start, end, result);
}

__global__ void testIndexPut(bool* input, int start, int end, bool* subvector) {
    indexPut(input, start, end, subvector);
}


class VectorBoolTest : public ::testing::Test {
public:
    VectorBool input, output, subvector;

    void SetUp() override {
        // Initialize vectors
        int size = 10;
        cudaError_t err;

        // Allocate and initialize `input`
        input.size_ = size;
        err = cudaMalloc(&input.data_, size * sizeof(bool));
        ASSERT_EQ(err, cudaSuccess) << "cudaMalloc failed for input: " << cudaGetErrorString(err);

        bool host_data[10] = {true, false, true, false, true, false, true, false, true, false};
        err = cudaMemcpy(input.data_, host_data, size * sizeof(bool), cudaMemcpyHostToDevice);
        ASSERT_EQ(err, cudaSuccess) << "cudaMemcpy failed for input: " << cudaGetErrorString(err);

        // Allocate `output`
        output.size_ = size;
        err = cudaMalloc(&output.data_, size * sizeof(bool));
        ASSERT_EQ(err, cudaSuccess) << "cudaMalloc failed for output: " << cudaGetErrorString(err);

        // Allocate and initialize `subvector`
        subvector.size_ = size;
        err = cudaMalloc(&subvector.data_, size * sizeof(bool));
        ASSERT_EQ(err, cudaSuccess) << "cudaMalloc failed for subvector: " << cudaGetErrorString(err);
    }

    void TearDown() override {
        if (input.data_ != nullptr) cudaFree(input.data_);
        if (output.data_ != nullptr) cudaFree(output.data_);
        if (subvector.data_ != nullptr) cudaFree(subvector.data_);
    }
};

// Test case for boolIndexGet
TEST_F(VectorBoolTest, BoolIndexGetTest) {
    int start = 2, end = 5;
    int range = end - start;

    // Launch kernel
    int threads_per_block = 256; // Adjust based on your hardware
    int num_blocks = (range + threads_per_block - 1) / threads_per_block;

    testBoolIndexGet<<<1, 10>>>(input.data_, start, end, output.data_);
    cudaDeviceSynchronize();

    // Check for kernel errors
    cudaError_t err = cudaGetLastError();
    EXPECT_EQ(err, cudaSuccess) << "Kernel launch failed: " << cudaGetErrorString(err);

    // Copy result back to host
    bool result[range];
    cudaMemcpy(result, output.data_, range * sizeof(bool), cudaMemcpyDeviceToHost);

    // Expected values
    bool expected[] = {true, false, true};
    for (int i = 0; i < range; i++) {
        EXPECT_EQ(result[i], expected[i]) << "Mismatch at index " << i;
    }
}

// Test case for indexPut
TEST_F(VectorBoolTest, IndexPutTest) {
    int start = 2, end = 5;
    int range = end - start;

    // Prepare subvector on the host
    bool host_subvector[] = {false, true, false};
    cudaMemcpy(subvector.data_, host_subvector, range * sizeof(bool), cudaMemcpyHostToDevice);

    // Launch kernel
    testIndexPut<<<1, range>>>(input.data_, start, end, subvector.data_);
    cudaDeviceSynchronize();

    // Copy result back to host
    bool result[10];
    cudaMemcpy(result, input.data_, 10 * sizeof(bool), cudaMemcpyDeviceToHost);

    // Expected values
    bool expected[] = {true, false, false, true, false, false, true, false, true, false};
    for (int i = 0; i < 10; i++) {
        EXPECT_EQ(result[i], expected[i]) << "Mismatch at index " << i;
    }
}

// Main entry point for Google Test
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}