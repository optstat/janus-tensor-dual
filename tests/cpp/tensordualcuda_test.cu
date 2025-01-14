#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cassert>
#include "../../src/cpp/cudatensordense.cu"
// Include your VectorBool implementation here
using namespace janus;



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

    boolIndexGetKernel<<<1, 10>>>(input.data_, start, end, output.data_);
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
    boolIndexPutKernel<<<1, range>>>(input.data_, start, end, subvector.data_);
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


// Test fixture class
class VectorAddTest : public ::testing::Test {
protected:
    const int size = 5;
    thrust::complex<float> h_a[5] = {
        {1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}, {7.0f, 8.0f}, {9.0f, 10.0f}};
    thrust::complex<float> h_b[5] = {
        {10.0f, 20.0f}, {30.0f, 40.0f}, {50.0f, 60.0f}, {70.0f, 80.0f}, {90.0f, 100.0f}};
    thrust::complex<float> h_result[5];

    thrust::complex<float> *d_a, *d_b, *d_result;

    void SetUp() override {
        cudaMalloc(&d_a, size * sizeof(thrust::complex<float>));
        cudaMalloc(&d_b, size * sizeof(thrust::complex<float>));
        cudaMalloc(&d_result, size * sizeof(thrust::complex<float>));

        cudaMemcpy(d_a, h_a, size * sizeof(thrust::complex<float>), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b, size * sizeof(thrust::complex<float>), cudaMemcpyHostToDevice);
    }

    void TearDown() override {
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_result);
    }
};

// Test case
TEST_F(VectorAddTest, ElementwiseAddition) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    VectorElementwiseAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, size, d_result);
    cudaMemcpy(h_result, d_result, size * sizeof(thrust::complex<float>), cudaMemcpyDeviceToHost);

    for (int i = 0; i < size; ++i) {
        thrust::complex<float> expected = h_a[i] + h_b[i];
        ASSERT_EQ(h_result[i], expected) << "Mismatch at index " << i;
    }
}




// Test fixture class
class VectorTest : public ::testing::Test {
protected:
    const int size = 5;
    thrust::complex<float> h_a[5] = {
        {1.0f, 2.0f}, {-3.0f, 4.0f}, {5.0f, -6.0f}, {-7.0f, -8.0f}, {0.0f, 10.0f}};
    thrust::complex<float> h_b[5] = {
        {10.0f, 20.0f}, {-30.0f, 40.0f}, {50.0f, -60.0f}, {-70.0f, 80.0f}, {90.0f, -100.0f}};
    thrust::complex<float> h_result[5];

    thrust::complex<float> *d_a, *d_b, *d_result;

    void SetUp() override {
        cudaMalloc(&d_a, size * sizeof(thrust::complex<float>));
        cudaMalloc(&d_b, size * sizeof(thrust::complex<float>));
        cudaMalloc(&d_result, size * sizeof(thrust::complex<float>));

        cudaMemcpy(d_a, h_a, size * sizeof(thrust::complex<float>), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b, size * sizeof(thrust::complex<float>), cudaMemcpyHostToDevice);
    }

    void TearDown() override {
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_result);
    }
};

// Test case for VectorElementwiseAdd
TEST_F(VectorTest, ElementwiseAddition) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    VectorElementwiseAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, size, d_result);
    cudaMemcpy(h_result, d_result, size * sizeof(thrust::complex<float>), cudaMemcpyDeviceToHost);

    for (int i = 0; i < size; ++i) {
        thrust::complex<float> expected = h_a[i] + h_b[i];
        ASSERT_EQ(h_result[i], expected) << "Mismatch at index " << i;
    }
}

// Test case for VectorSigncond
TEST_F(VectorTest, SignCondition) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    VectorSigncondKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, size, d_result, 1.0e-6);
    cudaMemcpy(h_result, d_result, size * sizeof(thrust::complex<float>), cudaMemcpyDeviceToHost);

    for (int i = 0; i < size; ++i) {
        thrust::complex<float> expected;
        int a_sign = (fabs(h_a[i].real()) >= 1.0e-6) ? (h_a[i].real() >= 0 ? 1 : -1) : 1;
        int b_sign = (fabs(h_b[i].real()) >= 1.0e-6) ? (h_b[i].real() >= 0 ? 1 : -1) : 1;

        if (b_sign >= 0) {
            expected = (a_sign >= 0) ? h_a[i] : -h_a[i];
        } else {
            expected = (a_sign >= 0) ? -h_a[i] : h_a[i];
        }

        ASSERT_EQ(h_result[i], expected) << "Mismatch at index " << i;
    }
}


// Test case for VectorSquare
TEST_F(VectorTest, Square) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    VectorSquareKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, size, d_result);
    cudaMemcpy(h_result, d_result, size * sizeof(thrust::complex<float>), cudaMemcpyDeviceToHost);

    for (int i = 0; i < size; ++i) {
        thrust::complex<float> expected = h_a[i] * h_a[i];
        ASSERT_NEAR(h_result[i].real(), expected.real(), 1.0e-12f) << "Mismatch at index " << i;
        ASSERT_NEAR(h_result[i].imag(), expected.imag(), 1.0e-12f) << "Mismatch at index " << i;
    }
}



// Test case for VectorScalarMultiply
TEST_F(VectorTest, ScalarMultiply) {
    float scalar = 2.0f;
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    VectorScalarMultiplyKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, scalar, size, d_result);
    cudaMemcpy(h_result, d_result, size * sizeof(thrust::complex<float>), cudaMemcpyDeviceToHost);

    for (int i = 0; i < size; ++i) {
        thrust::complex<float> expected = h_a[i] * scalar;
        ASSERT_EQ(h_result[i], expected) << "Mismatch at index " << i;
    }
}


// Test case for VectorReciprocal
TEST_F(VectorTest, Reciprocal) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    VectorReciprocalKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, size, d_result);
    cudaMemcpy(h_result, d_result, size * sizeof(thrust::complex<float>), cudaMemcpyDeviceToHost);

    for (int i = 0; i < size; ++i) {
        thrust::complex<float> expected = 1.0f / h_a[i];
        ASSERT_NEAR(h_result[i].real(), expected.real(), 1.0e-6) << "Mismatch at index " << i;
        ASSERT_NEAR(h_result[i].imag(), expected.imag(), 1.0e-6) << "Mismatch at index " << i;
    }
}


// Test case for VectorElementwiseMultiply
TEST_F(VectorTest, ElementwiseMultiply) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    VectorElementwiseMultiplyKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, size, d_result);
    cudaMemcpy(h_result, d_result, size * sizeof(thrust::complex<float>), cudaMemcpyDeviceToHost);

    for (int i = 0; i < size; ++i) {
        thrust::complex<float> expected = h_a[i] * h_b[i];
        ASSERT_EQ(h_result[i], expected) << "Mismatch at index " << i;
    }
}


// Test case for VectorSqrt
TEST_F(VectorTest, ElementwiseSqrt) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    VectorSqrtKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, size, d_result);
    cudaMemcpy(h_result, d_result, size * sizeof(thrust::complex<float>), cudaMemcpyDeviceToHost);

    for (int i = 0; i < size; ++i) {
        thrust::complex<float> expected = thrust::sqrt(h_a[i]);
        ASSERT_EQ(h_result[i], expected) << "Mismatch at index " << i;
    }
}

// Test case for VectorPow
TEST_F(VectorTest, ElementwisePow) {
    float power = 2.0f;
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    VectorPowKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, power, size, d_result);
    cudaMemcpy(h_result, d_result, size * sizeof(thrust::complex<float>), cudaMemcpyDeviceToHost);

    for (int i = 0; i < size; ++i) {
        thrust::complex<float> expected = pow(h_a[i], power);
        ASSERT_NEAR(h_result[i].real(), expected.real(), 1.0e-5) << "Mismatch at index " << i;
        ASSERT_NEAR(h_result[i].imag(), expected.imag(), 1.0e-5) << "Mismatch at index " << i;
    }
}


// Test case for VectorReduce
TEST_F(VectorTest, Reduce) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    // Allocate device memory for block-level results
    thrust::complex<float>* d_block_results;
    cudaMalloc(&d_block_results, blocksPerGrid * sizeof(thrust::complex<float>));

    // Launch the kernel
    VectorReduceKernel<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(thrust::complex<float>)>>>(d_a, size, d_block_results);

    // Copy the block-level results to host
    thrust::complex<float> h_block_results[blocksPerGrid];
    cudaMemcpy(h_block_results, d_block_results, blocksPerGrid * sizeof(thrust::complex<float>), cudaMemcpyDeviceToHost);

    // Final reduction on the host
    thrust::complex<float> final_result(0, 0);
    for (int i = 0; i < blocksPerGrid; ++i) {
        final_result += h_block_results[i];
    }

    // Expected result
    thrust::complex<float> expected(0, 0);
    for (int i = 0; i < size; ++i) {
        expected += h_a[i];
    }

    ASSERT_EQ(final_result, expected) << "Mismatch in reduction result.";

    // Free device memory
    cudaFree(d_block_results);
}


// Test case for VectorIndexGet
TEST_F(VectorTest, IndexGet) {
    const int start = 1;
    const int end = 4;
    const int result_size = end - start;

    // Allocate device memory for the result
    thrust::complex<float>* d_index_result;
    cudaMalloc(&d_index_result, result_size * sizeof(thrust::complex<float>));

    // Launch the kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    VectorIndexGetKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, start, end, size, d_index_result);

    // Copy the result to host
    thrust::complex<float> h_index_result[result_size];
    cudaMemcpy(h_index_result, d_index_result, result_size * sizeof(thrust::complex<float>), cudaMemcpyDeviceToHost);

    // Expected result
    for (int i = 0; i < result_size; ++i) {
        ASSERT_EQ(h_index_result[i], h_a[start + i]) << "Mismatch at index " << i;
    }

    // Free device memory
    cudaFree(d_index_result);
}

// Test case for VectorIndexPut
TEST_F(VectorTest, IndexPut) {
    const int start = 1;
    const int end = 4;
    const int result_size = size;

    // Allocate device memory for the result
    thrust::complex<float>* d_index_result;
    cudaMalloc(&d_index_result, result_size * sizeof(thrust::complex<float>));

    // Initialize the result array on the device to zeros
    thrust::complex<float> h_zeros[result_size] = {thrust::complex<float>(0, 0)};
    cudaMemcpy(d_index_result, h_zeros, result_size * sizeof(thrust::complex<float>), cudaMemcpyHostToDevice);

    // Allocate input subarray on the device
    const int input_size = end - start;
    thrust::complex<float> h_input[input_size] = {
        h_a[start], h_a[start + 1], h_a[start + 2]
    };
    thrust::complex<float>* d_input;
    cudaMalloc(&d_input, input_size * sizeof(thrust::complex<float>));
    cudaMemcpy(d_input, h_input, input_size * sizeof(thrust::complex<float>), cudaMemcpyHostToDevice);

    // Launch the kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (result_size + threadsPerBlock - 1) / threadsPerBlock;
    VectorIndexPutKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, start, end, result_size, d_index_result);

    // Copy the result to host
    cudaMemcpy(h_result, d_index_result, result_size * sizeof(thrust::complex<float>), cudaMemcpyDeviceToHost);

    // Verify the results
    for (int i = 0; i < result_size; ++i) {
        if (i >= start && i < end) {
            ASSERT_EQ(h_result[i], h_a[i]) << "Mismatch at index " << i;
        } else {
            ASSERT_EQ(h_result[i], thrust::complex<float>(0, 0)) << "Mismatch at index " << i;
        }
    }

    // Free device memory
    cudaFree(d_index_result);
    cudaFree(d_input);
}

// Main entry point for Google Test
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}