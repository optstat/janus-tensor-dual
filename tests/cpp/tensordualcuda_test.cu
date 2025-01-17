#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cassert>
#include "../../src/cpp/cudatensordense.cu"
// Include your VectorBool implementation here
using namespace janus;


template <typename T>
thrust::complex<T> generate_random() {
    return thrust::complex<T>(rand() % 100, rand() % 100);
}

//Generate N random complex nunbers
template <typename T>
std::vector<thrust::complex<T>> generate_random_vector(int N) {
    std::vector<thrust::complex<T>> vec(N);
    for (int i = 0; i < N; i++) {
        vec[i] = generate_random<T>();
    }
    return vec;
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


template <typename T>
void AllocateAndCopy(const std::vector<thrust::complex<T>>& host_data, thrust::complex<T>** device_data) {
    size_t size = host_data.size() * sizeof(thrust::complex<T>);
    cudaMalloc(device_data, size);
    cudaMemcpy(*device_data, host_data.data(), size, cudaMemcpyHostToDevice);
}

template <typename T>
std::vector<thrust::complex<T>> CopyToHost(const thrust::complex<T>* device_data, size_t size) {
    std::vector<thrust::complex<T>> host_data(size);
    cudaMemcpy(host_data.data(), device_data, size * sizeof(thrust::complex<T>), cudaMemcpyDeviceToHost);
    return host_data;
}

TEST(VectorDualTest, RealDualProduct) {
    using T = float;

    // Input size
    int real_size = 3;
    int dual_size = 2;

    // Host input data
    std::vector<thrust::complex<T>> a_real = {{1, 2}, {3, 4}, {5, 6}};
    std::vector<thrust::complex<T>> a_dual = {{0.1, 0.2}, {0.3, 0.4}, {0.5, 0.6}};
    std::vector<thrust::complex<T>> b_real = {{7, 8}, {9, 10}, {11, 12}};
    std::vector<thrust::complex<T>> b_dual = {{0.7, 0.8}, {0.9, 1.0}, {1.1, 1.2}};

    // Allocate device memory
    thrust::complex<T> *d_a_real, *d_a_dual, *d_b_real, *d_b_dual;
    AllocateAndCopy(a_real, &d_a_real);
    AllocateAndCopy(a_dual, &d_a_dual);
    AllocateAndCopy(b_real, &d_b_real);
    AllocateAndCopy(b_dual, &d_b_dual);

    // Output memory
    thrust::complex<T> *d_result_real, *d_result_dual;
    cudaMalloc(&d_result_real, real_size * sizeof(thrust::complex<T>));
    cudaMalloc(&d_result_dual, dual_size * sizeof(thrust::complex<T>));

    // Launch kernel
    VectorRealDualProductKernel<T><<<1, real_size * dual_size>>>(
        d_a_real, d_a_dual, d_b_real, d_b_dual, real_size, dual_size, d_result_real, d_result_dual);

    // Copy results to host
    auto result_real = CopyToHost(d_result_real, real_size);
    auto result_dual = CopyToHost(d_result_dual, dual_size);

    // Free device memory
    cudaFree(d_a_real);
    cudaFree(d_a_dual);
    cudaFree(d_b_real);
    cudaFree(d_b_dual);
    cudaFree(d_result_real);
    cudaFree(d_result_dual);

    // Validate results
    EXPECT_EQ(result_real[0], a_real[0] * b_real[0]); // Example validation
    EXPECT_EQ(result_dual[0], a_real[0] * b_dual[0] + b_real[0] * a_dual[0]);
}

TEST(VectorDualTest, ElementwiseAdd) {
    using T = float;

    int real_size = 3;
    int dual_size = 3;

    std::vector<thrust::complex<T>> a_real = {{1, 1}, {2, 2}, {3, 3}};
    std::vector<thrust::complex<T>> a_dual = {{0.1, 0.1}, {0.2, 0.2}, {0.3, 0.3}};
    std::vector<thrust::complex<T>> b_real = {{4, 4}, {5, 5}, {6, 6}};
    std::vector<thrust::complex<T>> b_dual = {{0.4, 0.4}, {0.5, 0.5}, {0.6, 0.6}};

    thrust::complex<T> *d_a_real, *d_a_dual, *d_b_real, *d_b_dual;
    AllocateAndCopy(a_real, &d_a_real);
    AllocateAndCopy(a_dual, &d_a_dual);
    AllocateAndCopy(b_real, &d_b_real);
    AllocateAndCopy(b_dual, &d_b_dual);

    thrust::complex<T> *d_result_real, *d_result_dual;
    cudaMalloc(&d_result_real, real_size * sizeof(thrust::complex<T>));
    cudaMalloc(&d_result_dual, dual_size * sizeof(thrust::complex<T>));

    VectorDualElementwiseAddKernel<T><<<1, real_size>>>(
        d_a_real, d_a_dual, d_b_real, d_b_dual, real_size, dual_size, d_result_real, d_result_dual);

    auto result_real = CopyToHost(d_result_real, real_size);
    auto result_dual = CopyToHost(d_result_dual, dual_size);

    cudaFree(d_a_real);
    cudaFree(d_a_dual);
    cudaFree(d_b_real);
    cudaFree(d_b_dual);
    cudaFree(d_result_real);
    cudaFree(d_result_dual);

    for (int i = 0; i < real_size; ++i) {
        EXPECT_EQ(result_real[i], a_real[i] + b_real[i]);
    }
    for (int i = 0; i < dual_size; ++i) {
        EXPECT_EQ(result_dual[i], a_dual[i] + b_dual[i]);
    }
}

TEST(VectorDualTest, ElementwiseMultiply) {
    using T = float;

    int real_size = 3;
    int dual_size = 2;  //There are dual_size dual numbers for each real number

    std::vector<thrust::complex<T>> a_real = {{1, 1}, {2, 2}, {3, 3}};
    std::vector<thrust::complex<T>> a_dual = generate_random_vector<T>(real_size*dual_size);
    std::vector<thrust::complex<T>> b_real = {{4, 4}, {5, 5}, {6, 6}};
    std::vector<thrust::complex<T>> b_dual = generate_random_vector<T>(real_size*dual_size);

    thrust::complex<T> *d_a_real, *d_a_dual, *d_b_real, *d_b_dual;
    AllocateAndCopy(a_real, &d_a_real);
    AllocateAndCopy(a_dual, &d_a_dual);
    AllocateAndCopy(b_real, &d_b_real);
    AllocateAndCopy(b_dual, &d_b_dual);

    thrust::complex<T> *d_result_real, *d_result_dual;
    cudaMalloc(&d_result_real, real_size * sizeof(thrust::complex<T>));
    cudaMalloc(&d_result_dual, real_size*dual_size * sizeof(thrust::complex<T>));

    VectorDualElementwiseMultiplyKernel<T><<<1, real_size*dual_size>>>(
        d_a_real, d_a_dual, d_b_real, d_b_dual, real_size, dual_size, d_result_real, d_result_dual);

    auto result_real = CopyToHost(d_result_real, real_size);
    auto result_dual = CopyToHost(d_result_dual, real_size*dual_size);

    cudaFree(d_a_real);
    cudaFree(d_a_dual);
    cudaFree(d_b_real);
    cudaFree(d_b_dual);
    cudaFree(d_result_real);
    cudaFree(d_result_dual);

    for (int i = 0; i < real_size; ++i) {
        EXPECT_EQ(result_real[i], a_real[i] * b_real[i]);
    }
    std::vector<thrust::complex<T>> check_dual(real_size*dual_size);
    for (int i = 0; i < real_size; ++i) {
        for (int j = 0; j < dual_size; ++j) {
            int off = i*dual_size + j;
            check_dual[off] = a_real[i] * b_dual[off] + 
                                           a_dual[off] * b_real[i];
        }
    }
    for (int i = 0; i < real_size*dual_size; ++i) {
        EXPECT_EQ(result_dual[i], check_dual[i]);
    }
}


TEST(VectorDualTest, IndexGet) {
    using T = float;

    // Input sizes and ranges
    int real_size = 6;
    int dual_size = 4;
    int start = 2, end = 5;

    // Host input data
    std::vector<thrust::complex<T>> a_real = generate_random_vector<T>(real_size);
    std::vector<thrust::complex<T>> a_dual = generate_random_vector<T>(real_size*dual_size);

    // Allocate device memory
    thrust::complex<T> *d_a_real, *d_a_dual;
    AllocateAndCopy(a_real, &d_a_real);
    AllocateAndCopy(a_dual, &d_a_dual);

    // Output memory
    thrust::complex<T> *d_result_real, *d_result_dual;
    cudaMalloc(&d_result_real, (end - start) * sizeof(thrust::complex<T>));
    cudaMalloc(&d_result_dual, (end - start) * dual_size * sizeof(thrust::complex<T>));

    // Launch kernel
    VectorDualIndexGetKernel<T><<<1, (end - start) * dual_size>>>(
        d_a_real, d_a_dual, real_size, dual_size, start, end, d_result_real, d_result_dual);

    // Copy results to host
    auto result_real = CopyToHost(d_result_real, end - start);
    auto result_dual = CopyToHost(d_result_dual, (end - start)*dual_size);

    // Free device memory
    cudaFree(d_a_real);
    cudaFree(d_a_dual);
    cudaFree(d_result_real);
    cudaFree(d_result_dual);

    // Validate results
    std::vector<thrust::complex<T>> expected_real = {a_real[2], a_real[3], a_real[4]};
    std::vector<thrust::complex<T>> expected_dual((end-start)*dual_size);
    int count=0;
    for (int i = start; i < end; ++i) {
        for ( int j=0; j<dual_size; ++j) {
            int off = i*dual_size + j;
            expected_dual[count] = a_dual[off];
            count++;
        }
    }

    EXPECT_EQ(result_real.size(), expected_real.size());
    EXPECT_EQ(result_dual.size(), expected_dual.size());

    for (size_t i = 0; i < result_real.size(); ++i) {
        EXPECT_EQ(result_real[i], expected_real[i]) << "Mismatch at index " << i << " in real part.";
    }
    for (size_t i = 0; i < result_dual.size(); ++i) {
        EXPECT_EQ(result_dual[i], expected_dual[i]) << "Mismatch at index " << i << " in dual part.";
    }
}

TEST(VectorDualTest, IndexPut) {
    using T = float;

    // Input sizes and ranges
    int real_size = 6;
    int dual_size = 6;
    int start = 2, end = 5;

    // Host input data (values to insert)
    std::vector<thrust::complex<T>> input_real = generate_random_vector<T>(end - start);
    std::vector<thrust::complex<T>> input_dual = generate_random_vector<T>(dual_size*(end - start));
    // Host output data (initial values of result arrays)
    std::vector<thrust::complex<T>> result_real= generate_random_vector<T>(real_size);
    std::vector<thrust::complex<T>> result_dual= generate_random_vector<T>(real_size*dual_size);

    // Allocate device memory
    thrust::complex<T> *d_input_real, *d_input_dual, *d_result_real, *d_result_dual;
    AllocateAndCopy(input_real, &d_input_real);
    AllocateAndCopy(input_dual, &d_input_dual);
    AllocateAndCopy(result_real, &d_result_real);
    AllocateAndCopy(result_dual, &d_result_dual);

    // Launch kernel
    VectorDualIndexPutKernel<T><<<1, (end - start)*dual_size>>>(
        d_input_real, d_input_dual, start, end, d_result_real, d_result_dual, real_size, dual_size);

    // Copy results to host
    auto result_real_host = CopyToHost(d_result_real, real_size);
    auto result_dual_host = CopyToHost(d_result_dual, real_size*dual_size);

    // Free device memory
    cudaFree(d_input_real);
    cudaFree(d_input_dual);
    cudaFree(d_result_real);
    cudaFree(d_result_dual);

    // Validate results
    std::vector<thrust::complex<T>> expected_real(result_real);
    //Substitute the values in the range [start, end) with the input values
    for (int i = start; i < end; ++i) {
        expected_real[i] = input_real[i-start];
    }

    std::vector<thrust::complex<T>> expected_dual(result_dual);

    //Substitute the values in the range [start, end) with the input values
    int count=0;
    for (int i = start; i < end; ++i) {
        for (int j = 0; j < dual_size; ++j) {
            int off = i*dual_size + j;
            expected_dual[off] = input_dual[count];
            count++;
        }
    }

    EXPECT_EQ(result_real_host.size(), expected_real.size());

    EXPECT_EQ(result_dual_host.size(), expected_dual.size());

    for (size_t i = 0; i < result_real.size(); ++i) {
        EXPECT_EQ(result_real_host[i], expected_real[i]) << "Mismatch at index " << i << " in real part.";
    }
    for (size_t i = 0; i < result_dual_host.size(); ++i) {
        EXPECT_EQ(result_dual_host[i], expected_dual[i]) << "Mismatch at index " << i << " in dual part.";
    }
}

TEST(VectorDualTest, Square) {
    using T = float;

    // Input size
    int real_size = 3;
    int dual_size = 6; //There are two dual numbers for each real number

    // Host input data
    std::vector<thrust::complex<T>> a_real = generate_random_vector<T>(real_size);
    std::vector<thrust::complex<T>> a_dual = generate_random_vector<T>(real_size*dual_size);

    // Allocate device memory
    thrust::complex<T> *d_a_real, *d_a_dual, *d_result_real, *d_result_dual;
    AllocateAndCopy(a_real, &d_a_real);
    AllocateAndCopy(a_dual, &d_a_dual);
    cudaMalloc(&d_result_real, real_size * sizeof(thrust::complex<T>));
    cudaMalloc(&d_result_dual, real_size*dual_size * sizeof(thrust::complex<T>));

    // Launch kernel
    VectorDualSquareKernel<T><<<1, real_size*dual_size>>>(
        d_a_real, d_a_dual, real_size, dual_size, d_result_real, d_result_dual);

    // Copy results to host
    auto result_real = CopyToHost(d_result_real, real_size);
    auto result_dual = CopyToHost(d_result_dual, real_size*dual_size);

    // Free device memory
    cudaFree(d_a_real);
    cudaFree(d_a_dual);
    cudaFree(d_result_real);
    cudaFree(d_result_dual);

    // Validate results
    std::vector<thrust::complex<T>> expected_real(real_size);
    for (int i = 0; i < real_size; ++i) {
        expected_real[i] = a_real[i] * a_real[i];
    }

    std::vector<thrust::complex<T>> expected_dual(real_size*dual_size);
    for (int i = 0; i < real_size; ++i) {
        for (int j = 0; j < dual_size; ++j) {
            int off = i*dual_size + j;
            expected_dual[off] = a_real[i] * a_dual[off] + a_dual[off] * a_real[i];
        }
    }

    EXPECT_EQ(result_real.size(), expected_real.size());
    EXPECT_EQ(result_dual.size(), expected_dual.size());

    for (size_t i = 0; i < result_real.size(); ++i) {
        EXPECT_EQ(result_real[i], expected_real[i]) << "Mismatch at index " << i << " in real part.";
    }
    for (size_t i = 0; i < result_dual.size(); ++i) {
        EXPECT_EQ(result_dual[i], expected_dual[i]) << "Mismatch at index " << i << " in dual part.";
    }
}


TEST(VectorDualTest, Pow) {
    using T = float;

    // Input size and power
    int real_size = 3;
    int dual_size = 3;
    T power = 1.5;  // Example power

    // Host input data
    std::vector<thrust::complex<T>> a_real = {{1, 2}, {3, 4}, {5, 6}};
    std::vector<thrust::complex<T>> a_dual = generate_random_vector<T>(real_size*dual_size);

    // Allocate device memory
    thrust::complex<T> *d_a_real, *d_a_dual, *d_result_real, *d_result_dual;
    AllocateAndCopy(a_real, &d_a_real);
    AllocateAndCopy(a_dual, &d_a_dual);
    cudaMalloc(&d_result_real, real_size * sizeof(thrust::complex<T>));
    cudaMalloc(&d_result_dual, real_size * dual_size * sizeof(thrust::complex<T>));

    // Launch kernel
    VectorDualPowKernel<T><<<1, real_size * dual_size>>>(
        d_a_real, d_a_dual, power, real_size, dual_size, d_result_real, d_result_dual);

    // Copy results to host
    auto result_real = CopyToHost(d_result_real, real_size);
    auto result_dual = CopyToHost(d_result_dual, real_size * dual_size);

    // Free device memory
    cudaFree(d_a_real);
    cudaFree(d_a_dual);
    cudaFree(d_result_real);
    cudaFree(d_result_dual);

    // Validate results
    std::vector<thrust::complex<T>> expected_real = {
        pow(a_real[0], power), pow(a_real[1], power), pow(a_real[2], power)};
    std::vector<thrust::complex<T>> expected_dual(real_size * dual_size);

    for (int i = 0; i < real_size; ++i) {
        for (int j = 0; j < dual_size; ++j) {
            int idx = i * dual_size + j;
            expected_dual[idx] = power * pow(a_real[i], power - 1) * a_dual[idx];
        }
    }

    EXPECT_EQ(result_real.size(), expected_real.size());
    EXPECT_EQ(result_dual.size(), expected_dual.size());

    for (size_t i = 0; i < result_real.size(); ++i) {
        EXPECT_NEAR(result_real[i].real(), expected_real[i].real(), 1.0e-4) << "Mismatch at index " << i << " in real part.";
        EXPECT_NEAR(result_real[i].imag(), expected_real[i].imag(), 1.0e-4) << "Mismatch at index " << i << " in real part.";
    }
    for (size_t i = 0; i < result_dual.size(); ++i) {
        EXPECT_NEAR(result_dual[i].real(), expected_dual[i].real(), 1.0e-4) << "Mismatch at index " << i << " in dual part.";
        EXPECT_NEAR(result_dual[i].imag(), expected_dual[i].imag(), 1.0e-4) << "Mismatch at index " << i << " in dual part.";
    }
}

TEST(VectorDualTest, Sqrt) {
    using T = float;

    // Input size
    int real_size = 3;
    int dual_size = 3;

    // Host input data
    std::vector<thrust::complex<T>> a_real = {{4, 0}, {9, 0}, {16, 0}};
    std::vector<thrust::complex<T>> a_dual = generate_random_vector<T>(real_size*dual_size);

    // Allocate device memory
    thrust::complex<T> *d_a_real, *d_a_dual, *d_result_real, *d_result_dual;
    AllocateAndCopy(a_real, &d_a_real);
    AllocateAndCopy(a_dual, &d_a_dual);
    cudaMalloc(&d_result_real, real_size * sizeof(thrust::complex<T>));
    cudaMalloc(&d_result_dual, real_size * dual_size * sizeof(thrust::complex<T>));

    // Launch kernel
    VectorDualSqrtKernel<T><<<1, real_size * dual_size>>>(
        d_a_real, d_a_dual, real_size, dual_size, d_result_real, d_result_dual);

    // Copy results to host
    auto result_real = CopyToHost(d_result_real, real_size);
    auto result_dual = CopyToHost(d_result_dual, real_size * dual_size);

    // Free device memory
    cudaFree(d_a_real);
    cudaFree(d_a_dual);
    cudaFree(d_result_real);
    cudaFree(d_result_dual);

    // Validate results
    std::vector<thrust::complex<T>> expected_real = {
        sqrt(a_real[0]), sqrt(a_real[1]), sqrt(a_real[2])};
    std::vector<thrust::complex<T>> expected_dual(real_size * dual_size);

    for (int i = 0; i < real_size; ++i) {
        for (int j = 0; j < dual_size; ++j) {
            int idx = i * dual_size + j;
            expected_dual[idx] = 0.5 * pow(a_real[i], -0.5) * a_dual[idx];
        }
    }

    EXPECT_EQ(result_real.size(), expected_real.size());
    EXPECT_EQ(result_dual.size(), expected_dual.size());

    for (size_t i = 0; i < result_real.size(); ++i) {
        EXPECT_NEAR(result_real[i].real(), expected_real[i].real(), 1.0e-4) << "Mismatch at index " << i << " in real part.";
        EXPECT_NEAR(result_real[i].imag(), expected_real[i].imag(), 1.0e-4) << "Mismatch at index " << i << " in real part.";

    }
    for (size_t i = 0; i < result_dual.size(); ++i) {
        EXPECT_NEAR(result_dual[i].real(), expected_dual[i].real(), 1.0e-4) << "Mismatch at index " << i << " in dual part.";
        EXPECT_NEAR(result_dual[i].imag(), expected_dual[i].imag(), 1.0e-4) << "Mismatch at index " << i << " in dual part.";

    }
}



// Test case for the get_hyperdual_vector_offsets_kernel
TEST(GetHyperdualVectorOffsetsKernel, ComputesOffsetsCorrectly) {
    // Input parameters
    int i = 2;
    int k = 1;
    int l = 3;
    int rows = 5; // Not used in the kernel but retained for completeness
    int dual = 4;

    // Device pointers for outputs
    int *d_off_i, *d_off_k, *d_off_l;

    // Host pointers for validation
    int h_off_i, h_off_k, h_off_l;

    // Allocate device memory
    ASSERT_EQ(cudaMalloc((void **)&d_off_i, sizeof(int)), cudaSuccess);
    ASSERT_EQ(cudaMalloc((void **)&d_off_k, sizeof(int)), cudaSuccess);
    ASSERT_EQ(cudaMalloc((void **)&d_off_l, sizeof(int)), cudaSuccess);

    // Launch the kernel (1 thread)
    get_hyperdual_vector_offsets_kernel<<<1, 1>>>(i, k, l, rows, dual, d_off_i, d_off_k, d_off_l);

    // Copy results back to host
    ASSERT_EQ(cudaMemcpy(&h_off_i, d_off_i, sizeof(int), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(&h_off_k, d_off_k, sizeof(int), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(&h_off_l, d_off_l, sizeof(int), cudaMemcpyDeviceToHost), cudaSuccess);

    // Free device memory
    cudaFree(d_off_i);
    cudaFree(d_off_k);
    cudaFree(d_off_l);

    // Expected results
    int expected_off_i = i;
    int expected_off_k = i * dual + k;
    int expected_off_l = i * dual * dual + k * dual + l;

    // Validate results
    EXPECT_EQ(h_off_i, expected_off_i);
    EXPECT_EQ(h_off_k, expected_off_k);
    EXPECT_EQ(h_off_l, expected_off_l);
}

TEST(VectorHyperDualTest, IndexGet) {
    using T = float;
    // Input sizes and ranges
    int real_size = 6;
    int dual_size = 6;

    int start_real = 1, end_real = 4;

    // Host input data
    std::vector<thrust::complex<T>> a_real = generate_random_vector<T>(real_size);
    std::vector<thrust::complex<T>> a_dual = generate_random_vector<T>(real_size*dual_size);
    std::vector<thrust::complex<T>> a_hyper = generate_random_vector<T>(real_size*dual_size*dual_size);

    // Allocate device memory
    thrust::complex<T> *d_a_real, *d_a_dual, *d_a_hyper;
    AllocateAndCopy(a_real,  &d_a_real);
    AllocateAndCopy(a_dual,  &d_a_dual);
    AllocateAndCopy(a_hyper, &d_a_hyper);

    // Output memory
    thrust::complex<T> *d_result_real, *d_result_dual, *d_result_hyper;
    cudaMalloc(&d_result_real, (end_real - start_real) * sizeof(thrust::complex<T>));
    cudaMalloc(&d_result_dual, dual_size*(end_real - start_real) * sizeof(thrust::complex<T>));
    cudaMalloc(&d_result_hyper, dual_size*dual_size*(end_real - start_real) * sizeof(thrust::complex<T>));
    int result_real_size = end_real - start_real;
    int result_dual_size = (end_real - start_real)*dual_size;
    int result_hyper_size = (end_real - start_real)*dual_size*dual_size;
    // Launch kernel
    //We need to launch at least (end_real - start_real)*(end_dual - start_dual)*(end_dual - start_dual) threads
    VectorHyperDualIndexGetKernel<T><<<1, result_hyper_size>>>(
        d_a_real, d_a_dual, d_a_hyper, real_size, dual_size, start_real, end_real,  
        d_result_real, d_result_dual, d_result_hyper);

    // Copy results to host
    auto result_real = CopyToHost(d_result_real, end_real - start_real);
    auto result_dual = CopyToHost(d_result_dual, (end_real - start_real)*dual_size);
    auto result_hyper = CopyToHost(d_result_hyper, (end_real - start_real)*dual_size*dual_size);

    // Free device memory
    cudaFree(d_a_real);
    cudaFree(d_a_dual);
    cudaFree(d_a_hyper);
    cudaFree(d_result_real);
    cudaFree(d_result_dual);
    cudaFree(d_result_hyper);

    // Validate results
    std::vector<thrust::complex<T>> expected_real(result_real_size);
    std::vector<thrust::complex<T>> expected_dual(result_dual_size);
    std::vector<thrust::complex<T>> expected_hyper(result_hyper_size);

    EXPECT_EQ(result_real.size(), expected_real.size());
    EXPECT_EQ(result_dual.size(), expected_dual.size());
    EXPECT_EQ(result_hyper.size(), expected_hyper.size());
    for ( int i=0; i<result_real_size; ++i) {
        expected_real[i] = a_real[start_real + i];
    }
    int count=0;
    for (int i = start_real; i < end_real; ++i) {
        for ( int j=0; j<dual_size; ++j) {
            int off = i*dual_size + j;
            expected_dual[count] = a_dual[off];
            count++;
        }
    }
    count=0;
    for (int i = start_real; i < end_real; ++i) {
        for ( int j=0; j< dual_size; ++j) {
            for ( int k=0; k<dual_size; ++k) {
                int off = i*dual_size*dual_size + j*dual_size + k;
                expected_hyper[count] = a_hyper[off];
                count++;
            }
        }
    }

    for (size_t i = 0; i < result_real.size(); ++i) {
        EXPECT_EQ(result_real[i], expected_real[i]) << "Mismatch at index " << i << " in real part.";
    }
    for (size_t i = 0; i < result_dual.size(); ++i) {
        EXPECT_EQ(result_dual[i], expected_dual[i]) << "Mismatch at index " << i << " in dual part.";
    }
    for (size_t i = 0; i < result_hyper.size(); ++i) {
        EXPECT_EQ(result_hyper[i], expected_hyper[i]) << "Mismatch at index " << i << " in hyper part.";
    }
}

TEST(VectorHyperDualTest, IndexPut) {
    using T = float;
    // Input sizes and ranges
    int real_size = 6;
    int dual_size = 6;

    int start_real = 1, end_real = 4;

    // Host output data
    std::vector<thrust::complex<T>> dest_real = generate_random_vector<T>(real_size);
    std::vector<thrust::complex<T>> dest_dual = generate_random_vector<T>(real_size*dual_size);
    std::vector<thrust::complex<T>> dest_hyper = generate_random_vector<T>(real_size*dual_size*dual_size);

    // Host input data (values to insert)
    std::vector<thrust::complex<T>> input_real = generate_random_vector<T>(end_real - start_real);
    std::vector<thrust::complex<T>> input_dual = generate_random_vector<T>(dual_size*(end_real - start_real));
    std::vector<thrust::complex<T>> input_hyper = generate_random_vector<T>(dual_size*dual_size*(end_real - start_real));


    // Copy the original values to the expected values
    std::vector<thrust::complex<T>> expected_real(dest_real);
    std::vector<thrust::complex<T>> expected_dual(dest_dual);
    std::vector<thrust::complex<T>> expected_hyper(dest_hyper);

    // Allocate device memory
    thrust::complex<T> *d_dest_real, *d_dest_dual, *d_dest_hyper, *d_input_real, *d_input_dual, *d_input_hyper;
    AllocateAndCopy(dest_real,  &d_dest_real);
    AllocateAndCopy(dest_dual,  &d_dest_dual);
    AllocateAndCopy(dest_hyper, &d_dest_hyper);
    AllocateAndCopy(input_real,  &d_input_real);
    AllocateAndCopy(input_dual,  &d_input_dual);
    AllocateAndCopy(input_hyper, &d_input_hyper);

    // Launch kernel
    //We need to launch at least (end_real - start_real)*(end_dual - start_dual)*(end_dual - start_dual) threads
    VectorHyperDualIndexPutKernel<T><<<1, input_hyper.size() >>>(
         d_input_real, d_input_dual, d_input_hyper,
         end_real-start_real, dual_size, start_real, end_real,  
         d_dest_real, d_dest_dual, d_dest_hyper);

    // Copy results to host
    auto result_real = CopyToHost(d_dest_real, real_size);
    auto result_dual = CopyToHost(d_dest_dual, real_size*dual_size);
    auto result_hyper = CopyToHost(d_dest_hyper, real_size*dual_size*dual_size);

    // Free device memory
    cudaFree(d_dest_real);
    cudaFree(d_dest_dual);
    cudaFree(d_dest_hyper);
    cudaFree(d_input_real);
    cudaFree(d_input_dual);
    cudaFree(d_input_hyper);


    EXPECT_EQ(result_real.size(), expected_real.size());
    EXPECT_EQ(result_dual.size(), expected_dual.size());
    EXPECT_EQ(result_hyper.size(), expected_hyper.size());
    for ( int i=start_real; i<end_real; ++i) {
        expected_real[i] = input_real[i-start_real];
    }
    for (int i = start_real; i < end_real; ++i) {
        for ( int j=0; j<dual_size; ++j) {
            int off = i*dual_size + j;
            expected_dual[off] = input_dual[(i-start_real)*dual_size + j];
        }
    }
    for (int i = start_real; i < end_real; ++i) {
        for ( int j=0; j< dual_size; ++j) {
            for ( int k=0; k<dual_size; ++k) {
                int off = i*dual_size*dual_size + j*dual_size + k;
                expected_hyper[off] = input_hyper[(i-start_real)*dual_size*dual_size + j*dual_size + k];
            }
        }
    }

    for (size_t i = 0; i < result_real.size(); ++i) {
        EXPECT_EQ(result_real[i], expected_real[i]) << "Mismatch at index " << i << " in real part.";
    }
    for (size_t i = 0; i < result_dual.size(); ++i) {
        EXPECT_EQ(result_dual[i], expected_dual[i]) << "Mismatch at index " << i << " in dual part.";
    }
    for (size_t i = 0; i < result_hyper.size(); ++i) {
        EXPECT_EQ(result_hyper[i], expected_hyper[i]) << "Mismatch at index " << i << " in hyper part.";
    }
}


// Add more tests for IndexGet, IndexPut, ElementwiseMultiply, Square, Pow, and Sqrt similarly.
// Main entry point for Google Test
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}