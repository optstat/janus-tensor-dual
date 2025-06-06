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


// A helper macro to check for CUDA errors
#define CUDA_CHECK(call)                                               \
    do {                                                               \
        cudaError_t err = call;                                        \
        if (err != cudaSuccess) {                                      \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << ": " << cudaGetErrorString(err) << std::endl; \
            std::exit(EXIT_FAILURE);                                   \
        }                                                              \
    } while (0)


// Google Test for MatrixElementwiseAddKernel
TEST(MatrixElementwiseAddTest, BasicAddition) 
{
    using Complex = thrust::complex<double>;

    // Define the dimensions of the matrix
    const int rows = 2;
    const int cols = 3;
    const int totalSize = rows * cols;

    // Prepare host data for matrix A and B
    // We'll just pick arbitrary values
    std::vector<Complex> h_a = {
        Complex(1.0, 2.0), Complex(3.0, 4.0), Complex(5.0,  6.0), 
        Complex(7.0, 8.0), Complex(9.0, 0.0), Complex(-1.0, 2.5)
    };
    std::vector<Complex> h_b = {
        Complex(0.5,  0.5),  Complex(1.0,  1.0), Complex(2.0,  2.0),
        Complex(-3.0, 1.0), Complex(10.0, 10.0), Complex( 0.0, 0.0)
    };

    // Compute expected result on the host
    std::vector<Complex> h_expected(totalSize);
    for (int i = 0; i < totalSize; ++i) {
        h_expected[i] = h_a[i] + h_b[i];
    }

    // Allocate device memory
    Complex* d_a = nullptr;
    Complex* d_b = nullptr;
    Complex* d_result = nullptr;

    CUDA_CHECK(cudaMalloc(&d_a,      totalSize * sizeof(Complex)));
    CUDA_CHECK(cudaMalloc(&d_b,      totalSize * sizeof(Complex)));
    CUDA_CHECK(cudaMalloc(&d_result, totalSize * sizeof(Complex)));

    // Copy host data to device
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), totalSize * sizeof(Complex), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), totalSize * sizeof(Complex), cudaMemcpyHostToDevice));

    // Launch the kernel
    int blockSize = 256; 
    int gridSize  = (totalSize + blockSize - 1) / blockSize; 
    MatrixElementwiseAddKernel<<<gridSize, blockSize>>>(d_a, d_b, rows, cols, d_result);

    // Check for kernel errors
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy the result back to the host
    std::vector<Complex> h_result(totalSize);
    CUDA_CHECK(cudaMemcpy(h_result.data(), d_result, totalSize * sizeof(Complex), cudaMemcpyDeviceToHost));

    // Verify the result against the expected values
    for (int i = 0; i < totalSize; ++i) {
        // We can use EXPECT_NEAR on real and imag parts, 
        // because Complex is a struct of double. 
        // Alternatively, we can define a tolerance for floating comparisons:
        double tol = 1e-9;
        EXPECT_NEAR(h_expected[i].real(), h_result[i].real(), tol) 
            << "Mismatch at index " << i << " (real part)";
        EXPECT_NEAR(h_expected[i].imag(), h_result[i].imag(), tol)
            << "Mismatch at index " << i << " (imag part)";
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_result));
}


// Test for elementwise matrix multiplication
TEST(MatrixElementwiseMultiplyTest, BasicMultiply)
{
    using Complex = thrust::complex<double>;

    // Define matrix dimensions
    const int rows = 2;
    const int cols = 3;
    const int totalSize = rows * cols;

    // Prepare host data for matrices A and B
    // (Just some example values)
    std::vector<Complex> h_a = {
        Complex(1.0, 2.0),  Complex(3.0,  -1.0), Complex(-1.0, 0.5),
        Complex(2.0, 0.0),  Complex(1.5,  2.5),  Complex( 0.0, -3.0)
    };
    std::vector<Complex> h_b = {
        Complex(2.0,  1.0), Complex( 4.0, 0.5),  Complex(1.0, 2.0),
        Complex(-2.0, 0.5), Complex( 1.0, 1.0),  Complex(2.0,  2.0)
    };

    // Compute the expected result on the host
    // result[i] = a[i] * b[i]
    std::vector<Complex> h_expected(totalSize);
    for (int i = 0; i < totalSize; ++i) {
        h_expected[i] = h_a[i] * h_b[i];
    }

    // Allocate device memory
    Complex* d_a = nullptr;
    Complex* d_b = nullptr;
    Complex* d_result = nullptr;

    CUDA_CHECK(cudaMalloc(&d_a,      totalSize * sizeof(Complex)));
    CUDA_CHECK(cudaMalloc(&d_b,      totalSize * sizeof(Complex)));
    CUDA_CHECK(cudaMalloc(&d_result, totalSize * sizeof(Complex)));

    // Copy host memory to device
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), totalSize * sizeof(Complex), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), totalSize * sizeof(Complex), cudaMemcpyHostToDevice));

    // Launch the kernel
    int blockSize = 128;
    int gridSize  = (totalSize + blockSize - 1) / blockSize;
    MatrixElementwiseMultiplyKernel<<<gridSize, blockSize>>>(d_a, d_b, rows, cols, d_result);

    // Check for kernel errors and synchronize
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy the result back to the host
    std::vector<Complex> h_result(totalSize);
    CUDA_CHECK(cudaMemcpy(h_result.data(), d_result, totalSize * sizeof(Complex), cudaMemcpyDeviceToHost));

    // Compare the GPU result to the expected result
    // We use EXPECT_NEAR for both real and imag parts because these are double values.
    double tol = 1e-9;
    for (int i = 0; i < totalSize; ++i) {
        EXPECT_NEAR(h_expected[i].real(), h_result[i].real(), tol)
            << "Mismatch at index " << i << " (real part)";
        EXPECT_NEAR(h_expected[i].imag(), h_result[i].imag(), tol)
            << "Mismatch at index " << i << " (imag part)";
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_result));
}


// Google Test for MatrixSquareKernel
TEST(MatrixSquareTest, BasicSquare) 
{
    using Complex = thrust::complex<double>;

    // Matrix dimensions
    const int rows = 2;
    const int cols = 3;
    const int totalSize = rows * cols;

    // Initialize a small matrix on the host
    std::vector<Complex> h_a = {
        Complex(1.0, 2.0),  Complex(2.0, -1.0), Complex(-1.0,  3.0),
        Complex(2.0, 0.0),  Complex(0.5,  0.5), Complex(-3.0, -3.0)
    };

    // Compute the expected result (elementwise square) on the host
    std::vector<Complex> h_expected(totalSize);
    for (int i = 0; i < totalSize; ++i) {
        // Square is simply a[i] * a[i]
        h_expected[i] = h_a[i] * h_a[i];
    }

    // Device pointers
    Complex* d_a = nullptr;
    Complex* d_result = nullptr;

    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_a,      totalSize * sizeof(Complex)));
    CUDA_CHECK(cudaMalloc(&d_result, totalSize * sizeof(Complex)));

    // Copy host data to device
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), totalSize * sizeof(Complex), cudaMemcpyHostToDevice));

    // Launch the kernel
    int blockSize = 128;
    int gridSize = (totalSize + blockSize - 1) / blockSize;
    MatrixSquareKernel<<<gridSize, blockSize>>>(d_a, rows, cols, d_result);

    // Check for kernel errors, synchronize
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back to host
    std::vector<Complex> h_result(totalSize);
    CUDA_CHECK(cudaMemcpy(h_result.data(), d_result, totalSize * sizeof(Complex), cudaMemcpyDeviceToHost));

    // Compare with the expected results
    double tol = 1e-9;
    for (int i = 0; i < totalSize; ++i) {
        EXPECT_NEAR(h_expected[i].real(), h_result[i].real(), tol)
            << "Mismatch at index " << i << " (real part)";
        EXPECT_NEAR(h_expected[i].imag(), h_result[i].imag(), tol)
            << "Mismatch at index " << i << " (imag part)";
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_result));
}


// Google Test for MatrixPowKernel
TEST(MatrixPowTest, ElementwisePower)
{
    using Complex = thrust::complex<double>;

    // Define dimensions of the matrix
    const int rows = 2;
    const int cols = 3;
    const int totalSize = rows * cols;

    // Define a matrix with arbitrary complex values
    std::vector<Complex> h_a = {
        Complex( 1.0,  2.0), Complex(-2.0,  3.0), Complex( 3.0, -1.0),
        Complex(-1.0, -1.0), Complex( 2.0,  2.0), Complex( 0.5,  1.5)
    };

    // The exponent we want to apply
    double power = 2.5; // for example

    // Compute the expected result on the host
    // We can use std::pow(...) or thrust::pow(...). Since it's host code,
    // std::pow is typically fine, but we'll cast to thrust::complex<double>.
    std::vector<Complex> h_expected(totalSize);
    for (int i = 0; i < totalSize; ++i) {
        // std::pow(...) can be used for complex numbers if <complex> is included,
        // or we can do something like thrust::pow(h_a[i], power).
        // For demonstration, let's use thrust::pow:
        h_expected[i] = thrust::pow(h_a[i], power);
    }

    // Allocate device memory
    Complex* d_a      = nullptr;
    Complex* d_result = nullptr;

    CUDA_CHECK(cudaMalloc(&d_a,      totalSize * sizeof(Complex)));
    CUDA_CHECK(cudaMalloc(&d_result, totalSize * sizeof(Complex)));

    // Copy host data to device
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), totalSize * sizeof(Complex), cudaMemcpyHostToDevice));

    // Launch the kernel
    int blockSize = 128;
    int gridSize  = (totalSize + blockSize - 1) / blockSize;
    MatrixPowKernel<<<gridSize, blockSize>>>(d_a, power, rows, cols, d_result);

    // Check for launch errors and synchronize
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy the result back to host
    std::vector<Complex> h_result(totalSize);
    CUDA_CHECK(cudaMemcpy(h_result.data(), d_result, totalSize * sizeof(Complex), cudaMemcpyDeviceToHost));

    // Validate the results
    double tol = 1e-9;
    for (int i = 0; i < totalSize; ++i) {
        // Compare real and imaginary parts
        EXPECT_NEAR(h_expected[i].real(), h_result[i].real(), tol)
            << "Mismatch at index " << i << " (real part)";
        EXPECT_NEAR(h_expected[i].imag(), h_result[i].imag(), tol)
            << "Mismatch at index " << i << " (imag part)";
    }

    // Free device memory
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_result));
}

//--------------------------------------------------
// Google Test for MatrixReduceKernel
//--------------------------------------------------
TEST(MatrixReduceTest, BasicReduce)
{
    using Complex = thrust::complex<double>;

    // Define matrix dimensions
    const int rows = 4;
    const int cols = 4;
    const int totalSize = rows * cols;

    // Create a small matrix with known complex values
    // 16 elements in total
    std::vector<Complex> h_a = {
        Complex(1.0,  2.0), Complex(2.0,  1.0), Complex(-1.0,  3.0), Complex(0.5,  -1.5),
        Complex(2.5,  2.5), Complex(-2.0, 1.0), Complex( 1.0, -2.0), Complex(-1.0, -1.0),
        Complex(3.0, -1.0), Complex(0.0,  0.5), Complex( 1.0,  1.0), Complex(-0.5,  2.0),
        Complex(2.0,  0.0), Complex(-3.0, 2.0), Complex( 1.5, -1.5), Complex(0.0,  0.75)
    };

    // Compute the expected sum (entire matrix) on the host
    Complex expectedSum(0.0, 0.0);
    for (int i = 0; i < totalSize; ++i) {
        expectedSum += h_a[i];
    }

    // Allocate device memory
    Complex* d_a = nullptr;
    // Each block writes one partial sum, so we need "gridSize" output elements.
    // We'll allocate enough space for the maximum expected gridSize.
    // We'll compute gridSize after we define blockSize.
    Complex* d_partialSums = nullptr;

    // Copy host vector to device
    CUDA_CHECK(cudaMalloc(&d_a, totalSize * sizeof(Complex)));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), totalSize * sizeof(Complex), cudaMemcpyHostToDevice));

    // Decide on block size and grid size
    int blockSize = 8; // e.g., 8 threads per block
    int gridSize  = (totalSize + blockSize - 1) / blockSize; 
    // We will store each block's result in d_partialSums
    CUDA_CHECK(cudaMalloc(&d_partialSums, gridSize * sizeof(Complex)));

    // Launch the kernel
    // IMPORTANT: we need dynamic shared memory size = blockSize * sizeof(Complex)
    size_t sharedMemSize = blockSize * sizeof(Complex);
    MatrixReduceKernel<<<gridSize, blockSize, sharedMemSize>>>(d_a, rows, cols, d_partialSums);

    // Check for launch errors and synchronize
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy partial sums back to host
    std::vector<Complex> h_partialSums(gridSize);
    CUDA_CHECK(cudaMemcpy(h_partialSums.data(),
                          d_partialSums,
                          gridSize * sizeof(Complex),
                          cudaMemcpyDeviceToHost));

    // Now reduce the partial sums on the host
    Complex finalSum(0.0, 0.0);
    for (int i = 0; i < gridSize; ++i) {
        finalSum += h_partialSums[i];
    }

    // Compare with the reference sum
    double tol = 1e-9;
    EXPECT_NEAR(expectedSum.real(), finalSum.real(), tol)
        << "Mismatch in real part of reduction result";
    EXPECT_NEAR(expectedSum.imag(), finalSum.imag(), tol)
        << "Mismatch in imaginary part of reduction result";

    // Cleanup
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_partialSums));
}


TEST(MatrixSumTest, SumColumns)
{
    using Complex = thrust::complex<double>;

    // Small 2x3 matrix
    const int rows = 2;
    const int cols = 3;
    const int totalSize = rows * cols;

    // Host data
    // Matrix layout (row-major):
    // Row 0: (1.0,2.0), (2.0,1.0), (3.0,-1.0)
    // Row 1: (4.0,1.5), (0.0,-2.0), (1.0,2.0)
    std::vector<Complex> h_a = {
        Complex(1.0, 2.0),  Complex(2.0, 1.0),  Complex(3.0, -1.0),
        Complex(4.0, 1.5),  Complex(0.0, -2.0), Complex(1.0, 2.0)
    };

    // We want column sums => result length = cols (3)
    // Summation along dim=0 means summing "down" each column.
    // Let's compute the expected sums on the host:
    // Column 0: (1+4, 2+1.5)   = (5.0, 3.5)
    // Column 1: (2+0, 1-2)     = (2.0, -1.0)
    // Column 2: (3+1, -1+2)    = (4.0, 1.0)
    std::vector<Complex> h_expected(cols);
    h_expected[0] = Complex(5.0, 3.5);
    h_expected[1] = Complex(2.0, -1.0);
    h_expected[2] = Complex(4.0, 1.0);

    // Allocate device memory
    Complex* d_a = nullptr;
    Complex* d_result = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a,      totalSize * sizeof(Complex)));
    CUDA_CHECK(cudaMalloc(&d_result, cols       * sizeof(Complex)));

    // Copy host data to device
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), totalSize * sizeof(Complex), cudaMemcpyHostToDevice));

    // Initialize result array to zero
    std::vector<Complex> h_init(cols, Complex(0.0, 0.0));
    CUDA_CHECK(cudaMemcpy(d_result, h_init.data(), cols * sizeof(Complex), cudaMemcpyHostToDevice));

    // Launch kernel with a single block (grid=1) so each element is processed by exactly one thread
    int blockSize = totalSize; // i.e., 6 threads
    int gridSize  = 1;
    MatrixSumKernel<<<gridSize, blockSize>>>(d_a, rows, cols, /*dim=*/0, d_result);

    // Check for launch errors, then synchronize
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back to host
    std::vector<Complex> h_result(cols);
    CUDA_CHECK(cudaMemcpy(h_result.data(), d_result, cols * sizeof(Complex), cudaMemcpyDeviceToHost));

    // Check correctness
    double tol = 1e-9;
    for (int c = 0; c < cols; ++c) {
        EXPECT_NEAR(h_expected[c].real(), h_result[c].real(), tol)
            << "Mismatch in real part of column " << c;
        EXPECT_NEAR(h_expected[c].imag(), h_result[c].imag(), tol)
            << "Mismatch in imaginary part of column " << c;
    }

    // Clean up
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_result));
}

TEST(MatrixSumTest, SumRows)
{
    using Complex = thrust::complex<double>;

    // Same 2x3 matrix as above
    const int rows = 2;
    const int cols = 3;
    const int totalSize = rows * cols;

    std::vector<Complex> h_a = {
        Complex(1.0, 2.0),  Complex(2.0, 1.0),  Complex(3.0, -1.0),
        Complex(4.0, 1.5),  Complex(0.0, -2.0), Complex(1.0, 2.0)
    };

    // Summation along dim=1 => sum across columns, producing a result of length = rows (2).
    // Let's compute the expected sums on the host:
    // Row 0: (1.0+2.0+3.0, 2.0+1.0-1.0) = (6.0, 2.0)
    // Row 1: (4.0+0.0+1.0, 1.5-2.0+2.0) = (5.0, 1.5)
    std::vector<Complex> h_expected(rows);
    h_expected[0] = Complex(6.0, 2.0);
    h_expected[1] = Complex(5.0, 1.5);

    // Allocate device memory
    Complex* d_a = nullptr;
    Complex* d_result = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a,      totalSize * sizeof(Complex)));
    CUDA_CHECK(cudaMalloc(&d_result, rows       * sizeof(Complex)));

    // Copy matrix to device
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), totalSize * sizeof(Complex), cudaMemcpyHostToDevice));

    // Initialize result array to zero
    std::vector<Complex> h_init(rows, Complex(0.0, 0.0));
    CUDA_CHECK(cudaMemcpy(d_result, h_init.data(), rows * sizeof(Complex), cudaMemcpyHostToDevice));

    // Launch the kernel
    int blockSize = totalSize; // 6 threads, 1 block
    int gridSize  = 1;
    MatrixSumKernel<<<gridSize, blockSize>>>(d_a, rows, cols, /*dim=*/1, d_result);

    // Check for errors and sync
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back to host
    std::vector<Complex> h_result(rows);
    CUDA_CHECK(cudaMemcpy(h_result.data(), d_result, rows * sizeof(Complex), cudaMemcpyDeviceToHost));

    // Verify correctness
    double tol = 1e-9;
    for (int r = 0; r < rows; ++r) {
        EXPECT_NEAR(h_expected[r].real(), h_result[r].real(), tol)
            << "Mismatch in real part of row " << r;
        EXPECT_NEAR(h_expected[r].imag(), h_result[r].imag(), tol)
            << "Mismatch in imaginary part of row " << r;
    }

    // Clean up
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_result));
}


TEST(MatrixIndexGetTest, SliceAllElements)
{
    using Complex = thrust::complex<double>;

    // Let's define a small 2x3 matrix => totalSize = 6
    // Flattened in row-major order:
    // idx:  0        1        2        3         4         5
    // val: (1,2), (2,1), (3,3), (4,0.5), (5,-2), (6,6)
    const int rows = 2;
    const int cols = 3;
    const int totalSize = rows * cols;

    // Build a small host array
    std::vector<Complex> h_a = {
        {1.0, 2.0}, {2.0, 1.0}, {3.0, 3.0}, 
        {4.0, 0.5}, {5.0, -2.0}, {6.0, 6.0}
    };

    // We'll slice from start=0 to end=6 => entire array
    int start = 0;
    int end   = totalSize;

    // Expected result is the same as the input
    std::vector<Complex> h_expected(h_a);

    // Allocate device buffers
    Complex* d_a = nullptr;
    Complex* d_result = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, totalSize * sizeof(Complex)));
    CUDA_CHECK(cudaMalloc(&d_result, (end - start) * sizeof(Complex)));

    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), totalSize * sizeof(Complex), cudaMemcpyHostToDevice));

    // Launch kernel: we need enough threads to cover [0, totalSize).
    int blockSize = 128;
    int gridSize  = (totalSize + blockSize - 1) / blockSize;
    MatrixIndexGetKernel<<<gridSize, blockSize>>>(d_a, start, end, rows, cols, d_result);

    // Check for kernel errors and synchronize
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy slice result back
    std::vector<Complex> h_result(end - start);
    CUDA_CHECK(cudaMemcpy(h_result.data(), d_result, (end - start) * sizeof(Complex), cudaMemcpyDeviceToHost));

    // Compare results
    double tol = 1e-9;
    for (int i = 0; i < (end - start); ++i) {
        EXPECT_NEAR(h_expected[i].real(), h_result[i].real(), tol)
            << "Mismatch at index " << i << " in real part";
        EXPECT_NEAR(h_expected[i].imag(), h_result[i].imag(), tol)
            << "Mismatch at index " << i << " in imag part";
    }

    // Clean up
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_result));
}

TEST(MatrixIndexGetTest, PartialSlice)
{
    using Complex = thrust::complex<double>;

    // Same 2x3 matrix => totalSize = 6
    // Indices: 0..5
    // Values: (1.0,2.0), (2.0,1.0), (3.0,3.0), (4.0,0.5), (5.0,-2.0), (6.0,6.0)
    const int rows = 2;
    const int cols = 3;
    const int totalSize = rows * cols;

    std::vector<Complex> h_a = {
        {1.0,  2.0}, {2.0, 1.0}, {3.0, 3.0},
        {4.0,  0.5}, {5.0, -2.0}, {6.0, 6.0}
    };

    // Let's take a slice from idx=1 to idx=4 => includes elements at [1,2,3]
    // That should produce 3 elements
    int start = 1;
    int end   = 4;

    // Expected: a[1], a[2], a[3]
    // => (2.0,1.0), (3.0,3.0), (4.0,0.5)
    std::vector<Complex> h_expected = {
        {2.0, 1.0}, {3.0, 3.0}, {4.0, 0.5}
    };

    // Allocate device memory
    Complex* d_a = nullptr;
    Complex* d_result = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, totalSize * sizeof(Complex)));
    CUDA_CHECK(cudaMalloc(&d_result, (end - start) * sizeof(Complex)));

    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), totalSize * sizeof(Complex), cudaMemcpyHostToDevice));

    // Launch kernel
    int blockSize = 128;
    int gridSize  = (totalSize + blockSize - 1) / blockSize;
    MatrixIndexGetKernel<<<gridSize, blockSize>>>(d_a, start, end, rows, cols, d_result);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy slice result back
    std::vector<Complex> h_result(end - start);
    CUDA_CHECK(cudaMemcpy(h_result.data(), d_result, (end - start) * sizeof(Complex), cudaMemcpyDeviceToHost));

    // Compare results
    double tol = 1e-9;
    ASSERT_EQ(h_expected.size(), h_result.size());
    for (int i = 0; i < (end - start); ++i) {
        EXPECT_NEAR(h_expected[i].real(), h_result[i].real(), tol)
            << "Mismatch at slice index " << i << " in real part";
        EXPECT_NEAR(h_expected[i].imag(), h_result[i].imag(), tol)
            << "Mismatch at slice index " << i << " in imag part";
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_result));
}

TEST(MatrixIndexGetTest, EmptySlice)
{
    using Complex = thrust::complex<double>;

    // 2x3 matrix => totalSize = 6
    const int rows = 2;
    const int cols = 3;
    const int totalSize = rows * cols;

    std::vector<Complex> h_a = {
        {1.0,  2.0}, {2.0, 1.0}, {3.0, 3.0},
        {4.0,  0.5}, {5.0, -2.0}, {6.0, 6.0}
    };

    // If we choose start=end=3, it yields zero elements in [3,3).
    int start = 3;
    int end   = 3; // empty

    // We expect an empty result
    std::vector<Complex> h_expected;  // size=0

    // Allocate device memory
    Complex* d_a = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, totalSize * sizeof(Complex)));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), totalSize * sizeof(Complex), cudaMemcpyHostToDevice));

    // For an empty slice, we can skip allocating d_result 
    // or we can allocate zero bytes for it. Let's do a non-null pointer for safety:
    Complex* d_result = nullptr;
    CUDA_CHECK(cudaMalloc(&d_result, 0)); // 0 bytes

    // Launch kernel
    int blockSize = 128;
    int gridSize  = (totalSize + blockSize - 1) / blockSize;
    MatrixIndexGetKernel<<<gridSize, blockSize>>>(d_a, start, end, rows, cols, d_result);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy results: none to copy
    std::vector<Complex> h_result; 
    // We expect it to remain empty.

    // Compare sizes (both should be 0)
    EXPECT_EQ(h_expected.size(), h_result.size());

    // Cleanup
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_result));
}


//------------------------------------------------------------------------------
// TEST 1: Put the entire input array into the result from index 0 to totalSize
//------------------------------------------------------------------------------
TEST(MatrixIndexPutTest, FullRangePut)
{
    using Complex = thrust::complex<double>;

    // For a 2x3 matrix, totalSize = 6
    const int rows = 2;
    const int cols = 3;
    const int totalSize = rows * cols;

    // Define input array of length totalSize
    std::vector<Complex> h_input = {
        {1.0,  2.0}, {2.0,  1.0}, {3.0, -1.0}, 
        {4.0,  4.5}, {5.0,  2.0}, {6.0,  6.0}
    };

    // We'll put into an initially empty (or zero) result array of length totalSize
    std::vector<Complex> h_resultInit(totalSize, {0.0, 0.0});

    // Expected final result = exactly h_input
    std::vector<Complex> h_expected(h_input);

    // Copy data to device
    Complex* d_input  = nullptr;
    Complex* d_result = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input,  totalSize * sizeof(Complex)));
    CUDA_CHECK(cudaMalloc(&d_result, totalSize * sizeof(Complex)));

    CUDA_CHECK(cudaMemcpy(d_input,  h_input.data(),   totalSize * sizeof(Complex), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_result, h_resultInit.data(), totalSize * sizeof(Complex), cudaMemcpyHostToDevice));

    // We'll put [start=0..end=6)
    int start = 0;
    int end   = totalSize;

    // Launch kernel
    int blockSize = 128;
    int gridSize  = (totalSize + blockSize - 1) / blockSize;
    MatrixIndexPutKernel<<<gridSize, blockSize>>>(d_input, start, end, rows, cols, d_result);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back to host
    std::vector<Complex> h_out(totalSize);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_result, totalSize * sizeof(Complex), cudaMemcpyDeviceToHost));

    // Check correctness
    double tol = 1e-9;
    for (int i = 0; i < totalSize; ++i) {
        EXPECT_NEAR(h_expected[i].real(), h_out[i].real(), tol)
            << "Mismatch at index " << i << " in real part";
        EXPECT_NEAR(h_expected[i].imag(), h_out[i].imag(), tol)
            << "Mismatch at index " << i << " in imaginary part";
    }

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_result));
}

//------------------------------------------------------------------------------
// TEST 2: Put a partial slice of input into the middle of the result
//------------------------------------------------------------------------------
TEST(MatrixIndexPutTest, PartialRangePut)
{
    using Complex = thrust::complex<double>;

    // For a 2x3 matrix, totalSize = 6
    const int rows = 2;
    const int cols = 3;
    const int totalSize = rows * cols;

    // The input we want to copy: length = 3
    std::vector<Complex> h_input = {
        {10.0, -1.0}, {20.0, 1.0}, {30.0,  5.0}
    };

    // The result array initially (some placeholder values)
    // We'll define 6 elements so we can put data into [start=2..5)
    // Note: result array has totalSize = 6
    std::vector<Complex> h_resultInit = {
        {1.0, 1.0},  {2.0, 2.0},  // Indices 0,1
        {0.0, 0.0},  {0.0, 0.0},  // Indices 2,3 (to be overwritten with input[0..1])
        {0.0, 0.0},  {3.0, 4.0}   // Indices 4,5 (Index 4 will be overwritten with input[2], 5 remains)
    };

    // We'll put the input array of length 3 into the subrange [2..5) of the result
    int start = 2;
    int end   = 5;  // This covers indices 2,3,4 in 'result'

    // So the expected final result:
    //   Index: 0: {1.0,1.0}    (unchanged)
    //          1: {2.0,2.0}    (unchanged)
    //          2: {10.0,-1.0}  (from input[0])
    //          3: {20.0, 1.0}  (from input[1])
    //          4: {30.0, 5.0}  (from input[2])
    //          5: {3.0,4.0}    (unchanged)
    std::vector<Complex> h_expected = {
        {1.0, 1.0}, {2.0, 2.0}, 
        {10.0, -1.0}, {20.0, 1.0}, 
        {30.0, 5.0}, {3.0, 4.0}
    };

    // Allocate device memory
    Complex* d_input  = nullptr;
    Complex* d_result = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input,  h_input.size() * sizeof(Complex)));
    CUDA_CHECK(cudaMalloc(&d_result, totalSize * sizeof(Complex)));

    CUDA_CHECK(cudaMemcpy(d_input,  h_input.data(),   h_input.size() * sizeof(Complex), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_result, h_resultInit.data(), totalSize * sizeof(Complex), cudaMemcpyHostToDevice));

    // Launch kernel
    int blockSize = 128;
    int gridSize  = (totalSize + blockSize - 1) / blockSize;
    MatrixIndexPutKernel<<<gridSize, blockSize>>>(d_input, start, end, rows, cols, d_result);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back to host
    std::vector<Complex> h_out(totalSize);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_result, totalSize * sizeof(Complex), cudaMemcpyDeviceToHost));

    // Check correctness
    double tol = 1e-9;
    ASSERT_EQ(h_out.size(), h_expected.size());
    for (int i = 0; i < totalSize; ++i) {
        EXPECT_NEAR(h_expected[i].real(), h_out[i].real(), tol)
            << "Mismatch at index " << i << " in real part";
        EXPECT_NEAR(h_expected[i].imag(), h_out[i].imag(), tol)
            << "Mismatch at index " << i << " in imaginary part";
    }

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_result));
}

//------------------------------------------------------------------------------
// TEST 3: Put an empty range => no changes
//------------------------------------------------------------------------------
TEST(MatrixIndexPutTest, EmptyRangePut)
{
    using Complex = thrust::complex<double>;

    // For a 2x3 matrix, totalSize=6
    const int rows = 2;
    const int cols = 3;
    const int totalSize = rows * cols;

    // The input array (3 elements) but we won't actually use them
    std::vector<Complex> h_input = {
        {1.0,1.0}, {2.0,2.0}, {3.0,3.0}
    };

    // The result array
    std::vector<Complex> h_resultInit = {
        {1.0, 1.0}, {2.0, 2.0}, {3.0, 3.0},
        {4.0, 4.0}, {5.0, 5.0}, {6.0, 6.0}
    };

    // start = end => empty slice
    int start = 3;
    int end   = 3;

    // We expect no changes to h_resultInit
    std::vector<Complex> h_expected = h_resultInit;

    // Allocate device memory
    Complex* d_input  = nullptr;
    Complex* d_result = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input,  h_input.size() * sizeof(Complex)));
    CUDA_CHECK(cudaMalloc(&d_result, totalSize * sizeof(Complex)));

    CUDA_CHECK(cudaMemcpy(d_input,  h_input.data(),   h_input.size() * sizeof(Complex), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_result, h_resultInit.data(), totalSize * sizeof(Complex), cudaMemcpyHostToDevice));

    // Launch kernel
    int blockSize = 128;
    int gridSize  = (totalSize + blockSize - 1) / blockSize;
    MatrixIndexPutKernel<<<gridSize, blockSize>>>(d_input, start, end, rows, cols, d_result);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back
    std::vector<Complex> h_out(totalSize);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_result, totalSize * sizeof(Complex), cudaMemcpyDeviceToHost));

    // Validate
    double tol = 1e-9;
    for (int i = 0; i < totalSize; ++i) {
        EXPECT_NEAR(h_expected[i].real(), h_out[i].real(), tol)
            << "Mismatch at index " << i << " in real part";
        EXPECT_NEAR(h_expected[i].imag(), h_out[i].imag(), tol)
            << "Mismatch at index " << i << " in imaginary part";
    }

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_result));
}


//------------------------------------------------------------------------------
// Test 1: (1 x N), dim=0 => should copy the row [0..N-1] into a 1D array of length N
//------------------------------------------------------------------------------
TEST(MatrixSqueezeTest, OneByN_Dim0)
{
    using Complex = thrust::complex<double>;

    // 1 x 4 example
    int rows = 1;
    int cols = 4;

    // Flattened matrix: (1 row, 4 columns)
    // Indices: a[0], a[1], a[2], a[3]
    std::vector<Complex> h_a = {
        {1.0, 1.0}, {2.0, 2.0}, {3.0, 3.0}, {4.0, 4.0}
    };

    // We expect the output to be exactly these 4 elements
    std::vector<Complex> h_expected = h_a;

    // Allocate device memory
    Complex* d_a = nullptr;
    Complex* d_result = nullptr;
    int totalSize = rows * cols; // = 4
    CUDA_CHECK(cudaMalloc(&d_a,      totalSize * sizeof(Complex)));
    CUDA_CHECK(cudaMalloc(&d_result, cols * sizeof(Complex))); // length=4

    // Copy host -> device
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), totalSize * sizeof(Complex), cudaMemcpyHostToDevice));

    // Launch the kernel
    // We only need up to 'cols' threads if (rows=1, dim=0).
    int blockSize = 128;
    int gridSize  = (cols + blockSize - 1) / blockSize;
    MatrixSqueezeKernel<<<gridSize, blockSize>>>(d_a, rows, cols, /*dim=*/0, d_result);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy back the result
    std::vector<Complex> h_result(cols);
    CUDA_CHECK(cudaMemcpy(h_result.data(), d_result, cols*sizeof(Complex), cudaMemcpyDeviceToHost));

    // Verify
    double tol = 1e-9;
    for (int i = 0; i < cols; ++i) {
        EXPECT_NEAR(h_expected[i].real(), h_result[i].real(), tol)
            << "Mismatch in real part at index " << i;
        EXPECT_NEAR(h_expected[i].imag(), h_result[i].imag(), tol)
            << "Mismatch in imag part at index " << i;
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_result));
}

//------------------------------------------------------------------------------
// Test 2: (N x 1), dim=1 => should copy the column [0..N-1] into a 1D array of length N
//------------------------------------------------------------------------------
TEST(MatrixSqueezeTest, NxOne_Dim1)
{
    using Complex = thrust::complex<double>;

    // 3 x 1
    int rows = 3;
    int cols = 1;

    // Flattened matrix: 3 elements, all in column 0
    // a[0] => (1, 1), a[1] => (2, 2), a[2] => (3, 3)
    std::vector<Complex> h_a = {
        {1.0,1.0}, {2.0,2.0}, {3.0,3.0}
    };
    int totalSize = rows * cols;  // = 3
    // We expect the output to be these 3 elements in a 1D array
    std::vector<Complex> h_expected = h_a;

    // Allocate
    Complex* d_a = nullptr;
    Complex* d_result = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a,      totalSize*sizeof(Complex)));
    CUDA_CHECK(cudaMalloc(&d_result, rows*sizeof(Complex))); // =3

    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), totalSize*sizeof(Complex), cudaMemcpyHostToDevice));

    // Launch the kernel (rows threads if dim=1 and cols=1)
    int blockSize = 128;
    int gridSize  = (rows + blockSize - 1) / blockSize;
    MatrixSqueezeKernel<<<gridSize, blockSize>>>(d_a, rows, cols, /*dim=*/1, d_result);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result
    std::vector<Complex> h_result(rows);
    CUDA_CHECK(cudaMemcpy(h_result.data(), d_result, rows*sizeof(Complex), cudaMemcpyDeviceToHost));

    // Check
    double tol = 1e-9;
    for (int i = 0; i < rows; ++i) {
        EXPECT_NEAR(h_expected[i].real(), h_result[i].real(), tol)
            << "Mismatch in real part at index " << i;
        EXPECT_NEAR(h_expected[i].imag(), h_result[i].imag(), tol)
            << "Mismatch in imag part at index " << i;
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_result));
}

//------------------------------------------------------------------------------
// Test 3: If dimension isn't actually size=1, it does nothing
//         -> result remains unchanged
//------------------------------------------------------------------------------
TEST(MatrixSqueezeTest, NoOpIfNotSizeOne)
{
    using Complex = thrust::complex<double>;

    // 2 x 3 => neither dim=0 nor dim=1 is size=1
    int rows = 2;
    int cols = 3;
    int totalSize = rows * cols;

    // Input data
    std::vector<Complex> h_a = {
        {1.0,1.0}, {2.0,2.0}, {3.0,3.0},
        {4.0,4.0}, {5.0,5.0}, {6.0,6.0}
    };

    // The result array (initialized with some distinct values to see if overwritten)
    // We'll just make it the same size to simplify; but the code won't fill it anyway.
    std::vector<Complex> h_resultInit = {
        {10.0,10.0}, {20.0,20.0}, {30.0,30.0},
        {40.0,40.0}, {50.0,50.0}, {60.0,60.0}
    };

    // Expected is the same as the init, because we do "nothing"
    std::vector<Complex> h_expected = h_resultInit;

    // Device memory
    Complex* d_a      = nullptr;
    Complex* d_result = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a,      totalSize*sizeof(Complex)));
    CUDA_CHECK(cudaMalloc(&d_result, totalSize*sizeof(Complex)));

    // Copy
    CUDA_CHECK(cudaMemcpy(d_a,      h_a.data(),      totalSize*sizeof(Complex), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_result, h_resultInit.data(), totalSize*sizeof(Complex), cudaMemcpyHostToDevice));

    // Launch kernel with dim=0 but rows=2 => do nothing
    int blockSize = 128;
    int gridSize  = (totalSize + blockSize - 1) / blockSize;
    MatrixSqueezeKernel<<<gridSize, blockSize>>>(d_a, rows, cols, /*dim=*/0, d_result);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result
    std::vector<Complex> h_out(totalSize);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_result, totalSize*sizeof(Complex), cudaMemcpyDeviceToHost));

    // Check => no changes
    double tol = 1e-9;
    for (int i = 0; i < totalSize; ++i) {
        EXPECT_NEAR(h_expected[i].real(), h_out[i].real(), tol)
            << "Mismatch in real part at index " << i;
        EXPECT_NEAR(h_expected[i].imag(), h_out[i].imag(), tol)
            << "Mismatch in imag part at index " << i;
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_result));
}

//------------------------------------------------------------------------------
// Test 4: (1 x 1) matrix for both dim=0 and dim=1
//         Both rows==1 and cols==1 => either dimension is size=1
//         Usually you'd pick dim=0 or dim=1. In typical libraries, you'd only
//         remove each dimension once, but let's see the code behavior
//------------------------------------------------------------------------------
TEST(MatrixSqueezeTest, OneByOne)
{
    using Complex = thrust::complex<double>;

    int rows = 1, cols = 1;
    // single element
    std::vector<Complex> h_a = { {42.0, -3.5} };

    // We'll try dim=0
    {
        // result should have length=cols=1
        // which is just the same single element
        std::vector<Complex> h_expected = h_a;

        Complex *d_a = nullptr, *d_result = nullptr;
        CUDA_CHECK(cudaMalloc(&d_a,      1*sizeof(Complex)));
        CUDA_CHECK(cudaMalloc(&d_result, 1*sizeof(Complex)));

        CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), 1*sizeof(Complex), cudaMemcpyHostToDevice));

        // We only need 1 thread
        MatrixSqueezeKernel<<<1,1>>>(d_a, rows, cols, /*dim=*/0, d_result);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Check
        std::vector<Complex> h_out(1);
        CUDA_CHECK(cudaMemcpy(h_out.data(), d_result, 1*sizeof(Complex), cudaMemcpyDeviceToHost));

        double tol = 1e-9;
        EXPECT_NEAR(h_expected[0].real(), h_out[0].real(), tol) << "dim=0, single (1x1)";
        EXPECT_NEAR(h_expected[0].imag(), h_out[0].imag(), tol) << "dim=0, single (1x1)";

        CUDA_CHECK(cudaFree(d_a));
        CUDA_CHECK(cudaFree(d_result));
    }

    // Now dim=1
    {
        // Similarly, the result should have length=rows=1, the same single element
        std::vector<Complex> h_expected = h_a;

        Complex *d_a = nullptr, *d_result = nullptr;
        CUDA_CHECK(cudaMalloc(&d_a,      1*sizeof(Complex)));
        CUDA_CHECK(cudaMalloc(&d_result, 1*sizeof(Complex)));

        CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), 1*sizeof(Complex), cudaMemcpyHostToDevice));

        MatrixSqueezeKernel<<<1,1>>>(d_a, rows, cols, /*dim=*/1, d_result);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Check
        std::vector<Complex> h_out(1);
        CUDA_CHECK(cudaMemcpy(h_out.data(), d_result, 1*sizeof(Complex), cudaMemcpyDeviceToHost));

        double tol = 1e-9;
        EXPECT_NEAR(h_expected[0].real(), h_out[0].real(), tol) << "dim=1, single (1x1)";
        EXPECT_NEAR(h_expected[0].imag(), h_out[0].imag(), tol) << "dim=1, single (1x1)";

        CUDA_CHECK(cudaFree(d_a));
        CUDA_CHECK(cudaFree(d_result));
    }
}


//------------------------------------------------------------------------
// Google Test for MatrixDualElementwiseAdd
//------------------------------------------------------------------------
TEST(MatrixDualElementwiseAddTest, BasicAddition)
{
    using Complex = thrust::complex<double>;

    // Example shape
    int rows = 2;
    int cols = 3;
    int dual_size = 2; // let's say we store the real part in dual[0], derivative in dual[1]

    int realSize = rows * cols;                 // size for real_ arrays
    int dualSize = rows * cols * dual_size;     // size for dual_ arrays

    // 1) Prepare host arrays for A and B (real + dual)
    // Let's create A and B with small test data

    // A: real part = [ (1,1), (2,2), (3,3),  (4,4), (5,5), (6,6) ]
    // B: real part = [ (0.5,0.5), (1.0,1.0), (2.0,2.0),  ... ]

    std::vector<Complex> h_a_real = {
        {1.0,1.0}, {2.0,2.0}, {3.0,3.0},
        {4.0,4.0}, {5.0,5.0}, {6.0,6.0}
    };
    std::vector<Complex> h_b_real = {
        {0.5, 0.5}, {1.0,1.0}, {2.0,2.0},
        {3.0,3.0},  {4.0,4.0}, {5.0,5.0}
    };

    // For dual parts, each has shape [rows * cols * dual_size].
    // We'll store some small example values. We'll treat each (i,j,k) as:
    //   A_dual[idx] = (real= i+ j/10. + k, imag=0), just to have distinct values.
    // We do something similar for B_dual.

    std::vector<Complex> h_a_dual(dualSize);
    std::vector<Complex> h_b_dual(dualSize);

    for (int i = 0; i < dualSize; ++i) {
        // Let’s do something arbitrary but consistent
        double reA = 10.0 + i;     // each index i
        double reB = 100.0 + i; 
        h_a_dual[i] = Complex(reA, -1.0 * i);  // imaginary part depends on i
        h_b_dual[i] = Complex(reB,  2.0 * i);
    }

    // 2) Compute the expected result on the host
    //   - result_real[i*cols + j] = a_real + b_real (for the row/col)
    //   - result_dual[idx]       = a_dual[idx] + b_dual[idx]

    std::vector<Complex> h_expected_real(realSize);
    for (int off = 0; off < realSize; ++off) {
        h_expected_real[off] = h_a_real[off] + h_b_real[off];
    }

    std::vector<Complex> h_expected_dual(dualSize);
    for (int idx = 0; idx < dualSize; ++idx) {
        h_expected_dual[idx] = h_a_dual[idx] + h_b_dual[idx];
    }

    // 3) Allocate device memory
    Complex *d_a_real = nullptr, *d_b_real = nullptr;
    Complex *d_a_dual = nullptr, *d_b_dual = nullptr;
    Complex *d_result_real = nullptr, *d_result_dual = nullptr;

    CUDA_CHECK(cudaMalloc(&d_a_real,    realSize*sizeof(Complex)));
    CUDA_CHECK(cudaMalloc(&d_b_real,    realSize*sizeof(Complex)));
    CUDA_CHECK(cudaMalloc(&d_a_dual,    dualSize*sizeof(Complex)));
    CUDA_CHECK(cudaMalloc(&d_b_dual,    dualSize*sizeof(Complex)));
    CUDA_CHECK(cudaMalloc(&d_result_real, realSize*sizeof(Complex)));
    CUDA_CHECK(cudaMalloc(&d_result_dual, dualSize*sizeof(Complex)));

    // 4) Copy host -> device
    CUDA_CHECK(cudaMemcpy(d_a_real,  h_a_real.data(),  realSize*sizeof(Complex), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b_real,  h_b_real.data(),  realSize*sizeof(Complex), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_a_dual,  h_a_dual.data(),  dualSize*sizeof(Complex), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b_dual,  h_b_dual.data(),  dualSize*sizeof(Complex), cudaMemcpyHostToDevice));

    // 5) Launch the kernel
    // total threads = rows * cols * dual_size
    int totalThreads = rows * cols * dual_size;
    int blockSize = 128;
    int gridSize  = (totalThreads + blockSize - 1) / blockSize;

    MatrixDualElementwiseAddKernel<<<gridSize, blockSize>>>(
        d_a_real, d_a_dual,
        d_b_real, d_b_dual,
        rows, cols, dual_size,
        d_result_real, d_result_dual
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 6) Copy results back to host
    std::vector<Complex> h_result_real(realSize);
    std::vector<Complex> h_result_dual(dualSize);

    CUDA_CHECK(cudaMemcpy(h_result_real.data(), d_result_real, realSize*sizeof(Complex), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_result_dual.data(), d_result_dual, dualSize*sizeof(Complex), cudaMemcpyDeviceToHost));

    // 7) Compare with expected
    double tol = 1e-9;

    // Compare real part
    for (int i = 0; i < realSize; ++i) {
        EXPECT_NEAR(h_expected_real[i].real(), h_result_real[i].real(), tol)
            << "Mismatch in real-part (REAL array) at index " << i << " (real component)";
        EXPECT_NEAR(h_expected_real[i].imag(), h_result_real[i].imag(), tol)
            << "Mismatch in real-part (REAL array) at index " << i << " (imag component)";
    }

    // Compare dual part
    for (int i = 0; i < dualSize; ++i) {
        EXPECT_NEAR(h_expected_dual[i].real(), h_result_dual[i].real(), tol)
            << "Mismatch in DUAL array at index " << i << " (real component)";
        EXPECT_NEAR(h_expected_dual[i].imag(), h_result_dual[i].imag(), tol)
            << "Mismatch in DUAL array at index " << i << " (imag component)";
    }

    // 8) Free device memory
    CUDA_CHECK(cudaFree(d_a_real));
    CUDA_CHECK(cudaFree(d_b_real));
    CUDA_CHECK(cudaFree(d_a_dual));
    CUDA_CHECK(cudaFree(d_b_dual));
    CUDA_CHECK(cudaFree(d_result_real));
    CUDA_CHECK(cudaFree(d_result_dual));
}


//--------------------------------------------------
// Google Test: Elementwise Multiply for Dual Tensors
//--------------------------------------------------
TEST(MatrixDualElementwiseMultiplyTest, BasicMultiply)
{
    using Complex = thrust::complex<double>;

    // Define small matrix dimensions
    int rows = 2;
    int cols = 3;
    int dual_size = 2;  // For example, storing 1 real + 1 derivative dimension

    // Size for real parts
    int realSize = rows * cols;       // 6
    // Size for dual parts
    int dualSize = rows * cols * dual_size;  // 12

    // 1) Prepare host data for matrix A and B
    //    We'll fill them with some distinct values.

    // A_real: e.g. [(1,1), (2,2), (3,3), (4,4), (5,5), (6,6)]
    std::vector<Complex> h_a_real = {
        {1.0,1.0}, {2.0,2.0}, {3.0,3.0},
        {4.0,4.0}, {5.0,5.0}, {6.0,6.0}
    };

    // B_real: e.g. [(0.5, 0.5), (1.0,1.0), (2.0, 2.0), ...]
    std::vector<Complex> h_b_real = {
        {0.5,0.5}, {1.0,1.0}, {2.0,2.0},
        {3.0,3.0}, {4.0,4.0}, {5.0,5.0}
    };

    // A_dual and B_dual: each of size dualSize=12
    // We'll create a small pattern so we can verify cross terms
    std::vector<Complex> h_a_dual(dualSize), h_b_dual(dualSize);
    for (int idx = 0; idx < dualSize; ++idx) {
        // Something like real= 10+idx, imag= -idx, etc.
        h_a_dual[idx] = Complex(10.0 + idx, -1.0 * idx);
        h_b_dual[idx] = Complex(20.0 + idx,  2.0 * idx);
    }

    // 2) Compute expected result on the host
    // real part: rA * rB
    // dual part: rA * dB + rB * dA
    std::vector<Complex> h_expected_real(realSize);
    std::vector<Complex> h_expected_dual(dualSize);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            int off = i * cols + j;
            Complex rA = h_a_real[off];
            Complex rB = h_b_real[off];

            // Real part
            Complex realProd = rA * rB; 
            h_expected_real[off] = realProd;

            // For each partial dimension k
            for (int k = 0; k < dual_size; ++k) {
                int idx = (i*(cols*dual_size)) + (j*dual_size) + k;
                Complex dA = h_a_dual[idx];
                Complex dB = h_b_dual[idx];

                Complex dualVal = rA * dB + rB * dA;
                h_expected_dual[idx] = dualVal;
            }
        }
    }

    // 3) Allocate device memory
    Complex *d_a_real = nullptr, *d_b_real = nullptr;
    Complex *d_a_dual = nullptr, *d_b_dual = nullptr;
    Complex *d_result_real = nullptr, *d_result_dual = nullptr;

    CUDA_CHECK(cudaMalloc(&d_a_real,    realSize * sizeof(Complex)));
    CUDA_CHECK(cudaMalloc(&d_b_real,    realSize * sizeof(Complex)));
    CUDA_CHECK(cudaMalloc(&d_a_dual,    dualSize * sizeof(Complex)));
    CUDA_CHECK(cudaMalloc(&d_b_dual,    dualSize * sizeof(Complex)));
    CUDA_CHECK(cudaMalloc(&d_result_real, realSize * sizeof(Complex)));
    CUDA_CHECK(cudaMalloc(&d_result_dual, dualSize * sizeof(Complex)));

    // 4) Copy data to device
    CUDA_CHECK(cudaMemcpy(d_a_real, h_a_real.data(), 
                          realSize*sizeof(Complex), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b_real, h_b_real.data(), 
                          realSize*sizeof(Complex), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(d_a_dual, h_a_dual.data(), 
                          dualSize*sizeof(Complex), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b_dual, h_b_dual.data(), 
                          dualSize*sizeof(Complex), cudaMemcpyHostToDevice));

    // 5) Launch the kernel
    int totalThreads = realSize * dual_size;  // = rows*cols*dual_size
    int blockSize = 128;
    int gridSize  = (totalThreads + blockSize - 1) / blockSize;

    MatrixDualElementwiseMultiplyKernel<<<gridSize, blockSize>>>(
        d_a_real, d_a_dual, d_b_real, d_b_dual, 
        rows, cols, dual_size, 
        d_result_real, d_result_dual
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 6) Copy results back
    std::vector<Complex> h_result_real(realSize);
    std::vector<Complex> h_result_dual(dualSize);

    CUDA_CHECK(cudaMemcpy(h_result_real.data(), d_result_real, 
                          realSize*sizeof(Complex), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_result_dual.data(), d_result_dual, 
                          dualSize*sizeof(Complex), cudaMemcpyDeviceToHost));

    // 7) Compare with expected
    double tol = 1e-9;

    // Check real part
    for (int i = 0; i < realSize; ++i) {
        EXPECT_NEAR(h_expected_real[i].real(), h_result_real[i].real(), tol)
            << "Mismatch in real-part (REAL array) at index " << i << " (Re)";
        EXPECT_NEAR(h_expected_real[i].imag(), h_result_real[i].imag(), tol)
            << "Mismatch in real-part (REAL array) at index " << i << " (Im)";
    }

    // Check dual part
    for (int i = 0; i < dualSize; ++i) {
        EXPECT_NEAR(h_expected_dual[i].real(), h_result_dual[i].real(), tol)
            << "Mismatch in (DUAL array) at index " << i << " (Re)";
        EXPECT_NEAR(h_expected_dual[i].imag(), h_result_dual[i].imag(), tol)
            << "Mismatch in (DUAL array) at index " << i << " (Im)";
    }

    // 8) Cleanup
    CUDA_CHECK(cudaFree(d_a_real));
    CUDA_CHECK(cudaFree(d_b_real));
    CUDA_CHECK(cudaFree(d_a_dual));
    CUDA_CHECK(cudaFree(d_b_dual));
    CUDA_CHECK(cudaFree(d_result_real));
    CUDA_CHECK(cudaFree(d_result_dual));
}


// ---------------------------------------------------------------------------
// Google Test
// ---------------------------------------------------------------------------
TEST(MatrixDualVectorTest, Multiply2x3_DualSize2)
{
    // We'll define:
    //   rows = 2, cols = 3, dual_size = 2.
    // Then:
    //   a_real size = rows*cols = 6
    //   a_dual size = rows*cols*dual_size = 12
    //   b_real size = cols = 3
    //   b_dual size = cols*dual_size = 6
    //   result_real size = rows = 2
    //   result_dual size = rows*dual_size = 4

    int rows = 2;
    int cols = 3;
    int dual_size = 2;

    // We'll fill each array with simple numeric patterns
    // so we can easily compute a reference solution.

    // 1) Host vectors
    std::vector<thrust::complex<double>> a_real(rows*cols), a_dual(rows*cols*dual_size);
    std::vector<thrust::complex<double>> b_real(cols),       b_dual(cols*dual_size);

    // (a) A_real: 
    //     Let's do: row=0 => (1,2,3), row=1 => (4,5,6), all with no imaginary part.
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            a_real[i*cols + j] = thrust::complex<double>(double(i*cols + j + 1), 0.0);
        }
    }

    // (b) A_dual: 
    //     We'll fill with a pattern so each element differs:
    //     a_dual( (i*cols+j)*dual_size + k ) = (real=(i+j+0.1+k), imag=0)
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            for (int k = 0; k < dual_size; ++k) {
                int idx = (i*cols + j)*dual_size + k;
                double val = double(i + j) + 0.1 + double(k);
                a_dual[idx] = thrust::complex<double>(val, 0.0);
            }
        }
    }

    // (c) B_real: 
    //     3 elements => (10, 20, 30)
    for (int j = 0; j < cols; ++j) {
        b_real[j] = thrust::complex<double>(double(10*(j+1)), 0.0);
    }

    // (d) B_dual:
    //     size=6 => b_dual[j*dual_size + k], fill with pattern
    //     e.g. b_dual( j,k ) =  ( real=(j*10 + k + 0.5), imag=0 )
    for (int j = 0; j < cols; ++j) {
        for (int k = 0; k < dual_size; ++k) {
            int idx = j*dual_size + k;
            double val = double(j*10) + double(k) + 0.5;
            b_dual[idx] = thrust::complex<double>(val, 0.0);
        }
    }

    // 2) Device memory
    thrust::complex<double> *d_a_real=nullptr, *d_a_dual=nullptr;
    thrust::complex<double> *d_b_real=nullptr, *d_b_dual=nullptr;
    thrust::complex<double> *d_result_real=nullptr, *d_result_dual=nullptr;

    // Copy inputs to device
    AllocateAndCopy(a_real, &d_a_real);
    AllocateAndCopy(a_dual, &d_a_dual);
    AllocateAndCopy(b_real, &d_b_real);
    AllocateAndCopy(b_dual, &d_b_dual);

    // Allocate outputs (rows for real, rows*dual_size for dual)
    cudaMalloc(&d_result_real, rows*sizeof(thrust::complex<double>));
    cudaMalloc(&d_result_dual, rows*dual_size*sizeof(thrust::complex<double>));

    // Initialize them to zero
    cudaMemset(d_result_real, 0, rows*sizeof(thrust::complex<double>));
    cudaMemset(d_result_dual, 0, rows*dual_size*sizeof(thrust::complex<double>));

    // 3) Launch the kernel
    // We have one thread per (i,k) => total = rows*dual_size = 4 threads
    int totalThreads = rows * dual_size; // 2*2=4
    MatrixDualVectorDualMultiplyKernel<<<1, totalThreads>>>(
        d_a_real, d_a_dual, 
        d_b_real, d_b_dual,
        rows, cols, dual_size,
        d_result_real, d_result_dual
    );
    cudaDeviceSynchronize();

    // 4) Copy results back
    auto h_result_real = CopyToHost(d_result_real, rows);  // 2
    auto h_result_dual = CopyToHost(d_result_dual, rows*dual_size); // 4

    // Free device memory
    cudaFree(d_a_real);
    cudaFree(d_a_dual);
    cudaFree(d_b_real);
    cudaFree(d_b_dual);
    cudaFree(d_result_real);
    cudaFree(d_result_dual);

    // 5) Compute a reference result on the host
    //    result_real[i]  = sum_{j=0..2} A_real(i,j) * B_real(j)
    //    result_dual[i,k] = sum_{j=0..2} [ A_real(i,j)*B_dual(j,k) + A_dual(i,j,k)*B_real(j) ]
    std::vector<thrust::complex<double>> ref_real(rows, thrust::complex<double>(0,0));
    std::vector<thrust::complex<double>> ref_dual(rows*dual_size, thrust::complex<double>(0,0));

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            auto Ar = a_real[i*cols + j];
            auto Br = b_real[j];
            ref_real[i] += (Ar * Br);

            for (int k = 0; k < dual_size; ++k) {
                auto Ad = a_dual[(i*cols + j)*dual_size + k];
                auto Bd = b_dual[j*dual_size + k];
                ref_dual[i*dual_size + k] += (Ar * Bd + Ad * Br);
            }
        }
    }

    // 6) Compare
    // Real part => 2 elements
    for (int i = 0; i < rows; ++i) {
        EXPECT_EQ(h_result_real[i], ref_real[i]) << "Mismatch in real part at i=" << i;
    }

    // Dual part => 4 elements
    for (int i = 0; i < rows*dual_size; ++i) {
        EXPECT_EQ(h_result_dual[i], ref_dual[i]) << "Mismatch in dual part at i=" << i;
    }
}


//--------------------------------------------------
// Google Test: MatrixDualSquare
//--------------------------------------------------
TEST(MatrixDualSquareTest, BasicSquare)
{
    using Complex = thrust::complex<double>;

    // Dimensions
    int rows = 2;
    int cols = 3;
    int dual_size = 2; // e.g., storing real + 1 partial derivative

    int realSize = rows * cols;          // 6
    int dualSize = rows * cols * dual_size; // 12

    // 1) Prepare a small matrix A on the host
    // For example, A_real = [ (1,1), (2,2), (3,3), (4,4), (5,5), (6,6) ]
    std::vector<Complex> h_a_real = {
        {1.0,1.0}, {2.0,2.0}, {3.0,3.0},
        {4.0,4.0}, {5.0,5.0}, {6.0,6.0}
    };

    // A_dual: size=12
    // We'll fill it with a distinct pattern so we can see how the derivative is handled.
    // For instance, let a_dual[i] = (10+i, -i).
    std::vector<Complex> h_a_dual(dualSize);
    for (int i = 0; i < dualSize; ++i) {
        h_a_dual[i] = Complex(10.0 + i, -1.0 * i);
    }

    // 2) Compute the expected "square" on the host
    //    For each element (r + eps*d): (r^2 + eps(2*r*d)).
    std::vector<Complex> h_expected_real(realSize);
    std::vector<Complex> h_expected_dual(dualSize);

    // We'll loop over (rows,cols, dual_size)
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            int off = i * cols + j;
            Complex r = h_a_real[off];

            // r^2 => real part
            h_expected_real[off] = r * r;

            // dual part => 2 * r * d
            for (int k = 0; k < dual_size; ++k) {
                int idx = (i*(cols*dual_size)) + (j*dual_size) + k;
                Complex d = h_a_dual[idx];
                // (r + eps*d)^2 => r^2 + eps(2*r*d)
                h_expected_dual[idx] = (Complex)(2.0) * r * d;
            }
        }
    }

    // 3) Allocate device memory
    Complex* d_a_real = nullptr;
    Complex* d_a_dual = nullptr;
    Complex* d_result_real = nullptr;
    Complex* d_result_dual = nullptr;

    CUDA_CHECK(cudaMalloc(&d_a_real,      realSize * sizeof(Complex)));
    CUDA_CHECK(cudaMalloc(&d_a_dual,      dualSize * sizeof(Complex)));
    CUDA_CHECK(cudaMalloc(&d_result_real, realSize * sizeof(Complex)));
    CUDA_CHECK(cudaMalloc(&d_result_dual, dualSize * sizeof(Complex)));

    // 4) Copy to device
    CUDA_CHECK(cudaMemcpy(d_a_real, h_a_real.data(), realSize*sizeof(Complex), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_a_dual, h_a_dual.data(), dualSize*sizeof(Complex), cudaMemcpyHostToDevice));

    // 5) Launch the kernel
    // total threads = rows*cols*dual_size
    int totalThreads = realSize * dual_size; 
    int blockSize = 128;
    int gridSize  = (totalThreads + blockSize - 1) / blockSize;
    MatrixDualSquareKernel<<<gridSize, blockSize>>>(d_a_real, d_a_dual, rows, cols, dual_size, 
                                                    d_result_real, d_result_dual);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 6) Copy results back
    std::vector<Complex> h_result_real(realSize);
    std::vector<Complex> h_result_dual(dualSize);

    CUDA_CHECK(cudaMemcpy(h_result_real.data(), d_result_real, realSize*sizeof(Complex), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_result_dual.data(), d_result_dual, dualSize*sizeof(Complex), cudaMemcpyDeviceToHost));

    // 7) Validate
    double tol = 1e-9;
    // Real part
    for (int i = 0; i < realSize; ++i) {
        EXPECT_NEAR(h_expected_real[i].real(), h_result_real[i].real(), tol)
            << "Mismatch in real-part (REAL array) at index " << i << " (Re)";
        EXPECT_NEAR(h_expected_real[i].imag(), h_result_real[i].imag(), tol)
            << "Mismatch in real-part (REAL array) at index " << i << " (Im)";
    }
    // Dual part
    for (int i = 0; i < dualSize; ++i) {
        EXPECT_NEAR(h_expected_dual[i].real(), h_result_dual[i].real(), tol)
            << "Mismatch in (DUAL array) at index " << i << " (Re)";
        EXPECT_NEAR(h_expected_dual[i].imag(), h_result_dual[i].imag(), tol)
            << "Mismatch in (DUAL array) at index " << i << " (Im)";
    }

    // 8) Cleanup
    CUDA_CHECK(cudaFree(d_a_real));
    CUDA_CHECK(cudaFree(d_a_dual));
    CUDA_CHECK(cudaFree(d_result_real));
    CUDA_CHECK(cudaFree(d_result_dual));
}


//----------------------------------------------------------------------
// Google Test for MatrixDualPowKernel
//----------------------------------------------------------------------
TEST(MatrixDualPowTest, BasicPow)
{
    using Complex = thrust::complex<double>;

    // Example dimensions
    int rows = 2;
    int cols = 3;
    int dual_size = 2;  // e.g., store real + 1 partial derivative per element

    int realSize = rows * cols;            // 6
    int totalSize = realSize * dual_size;  // 12

    // Chosen exponent
    double power = 2.5;  // for example

    // 1) Prepare host data: a_real, a_dual
    // Let's do a_real with 6 distinct values
    std::vector<Complex> h_a_real = {
        {1.0,  0.0}, {2.0,  -1.0}, {3.0,  0.5},
        {0.5,  1.5}, {2.0,   2.0}, {4.0,  -2.0}
    };

    // a_dual = 12 values
    // We'll fill them with a simple pattern
    std::vector<Complex> h_a_dual(totalSize);
    for (int i = 0; i < totalSize; ++i) {
        double re = 10.0 + i;
        double im = -i;
        h_a_dual[i] = Complex(re, im);
    }

    // 2) Compute the expected results on the host
    //    result_real[i] = pow(a_real[i], power)
    //    result_dual[idx] = power * pow(a_real[off], power - 1) * a_dual[idx]
    std::vector<Complex> h_expected_real(realSize);
    std::vector<Complex> h_expected_dual(totalSize);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            int off = i * cols + j;
            // base for this element
            Complex base = h_a_real[off];
            // real part
            h_expected_real[off] = thrust::pow(base, power);

            // fill derivative parts
            for (int k = 0; k < dual_size; ++k) {
                int idx = off * dual_size + k;
                Complex d = h_a_dual[idx];
                // chain rule derivative
                h_expected_dual[idx] = power * thrust::pow(base, power - 1) * d;
            }
        }
    }

    // 3) Device allocations
    Complex* d_a_real      = nullptr;
    Complex* d_a_dual      = nullptr;
    Complex* d_result_real = nullptr;
    Complex* d_result_dual = nullptr;

    CUDA_CHECK(cudaMalloc(&d_a_real,      realSize  * sizeof(Complex)));
    CUDA_CHECK(cudaMalloc(&d_a_dual,      totalSize * sizeof(Complex)));
    CUDA_CHECK(cudaMalloc(&d_result_real, realSize  * sizeof(Complex)));
    CUDA_CHECK(cudaMalloc(&d_result_dual, totalSize * sizeof(Complex)));

    // 4) Copy host -> device
    CUDA_CHECK(cudaMemcpy(d_a_real,  h_a_real.data(), 
                          realSize*sizeof(Complex), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_a_dual,  h_a_dual.data(), 
                          totalSize*sizeof(Complex), cudaMemcpyHostToDevice));

    // 5) Launch kernel
    int blockSize = 128;
    int gridSize  = (totalSize + blockSize - 1) / blockSize;
    MatrixDualPowKernel<<<gridSize, blockSize>>>(
        d_a_real, d_a_dual, power, rows, cols, dual_size,
        d_result_real, d_result_dual
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 6) Copy results back
    std::vector<Complex> h_result_real(realSize);
    std::vector<Complex> h_result_dual(totalSize);

    CUDA_CHECK(cudaMemcpy(h_result_real.data(), d_result_real, 
                          realSize*sizeof(Complex), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_result_dual.data(), d_result_dual, 
                          totalSize*sizeof(Complex), cudaMemcpyDeviceToHost));

    // 7) Compare with expected
    double tol = 1e-9;

    // Compare the real part
    for (int i = 0; i < realSize; ++i) {
        EXPECT_NEAR(h_expected_real[i].real(), h_result_real[i].real(), tol)
            << "Mismatch in real-part (REAL array) at index " << i << " (Re)";
        EXPECT_NEAR(h_expected_real[i].imag(), h_result_real[i].imag(), tol)
            << "Mismatch in real-part (REAL array) at index " << i << " (Im)";
    }

    // Compare the dual part
    for (int i = 0; i < totalSize; ++i) {
        EXPECT_NEAR(h_expected_dual[i].real(), h_result_dual[i].real(), tol)
            << "Mismatch in (DUAL array) at index " << i << " (Re)";
        EXPECT_NEAR(h_expected_dual[i].imag(), h_result_dual[i].imag(), tol)
            << "Mismatch in (DUAL array) at index " << i << " (Im)";
    }

    // 8) Cleanup
    CUDA_CHECK(cudaFree(d_a_real));
    CUDA_CHECK(cudaFree(d_a_dual));
    CUDA_CHECK(cudaFree(d_result_real));
    CUDA_CHECK(cudaFree(d_result_dual));
}

// Test for slicing a dual matrix
TEST(MatrixDualIndexGet2DTest, Basic2DSlice)
{
    using Complex = thrust::complex<double>;

    // Full matrix dimension: rows=3, cols=4
    // So the real part has 3*4=12 elements
    int rows = 3;
    int cols = 4;
    // Let's say dual_size=2
    int dual_size = 2;
    int realSize  = rows * cols;           // 12
    int dualSize  = realSize * dual_size;  // 24

    // 1) Create host data for the real part
    // We'll store something easily trackable
    // row=0 => indices:0..3, row=1 => 4..7, row=2 => 8..11
    // for example:
    //   h_a_real[i] = ( (double)i, -(double)i ), to see them clearly
    std::vector<Complex> h_a_real(realSize);
    for (int i = 0; i < realSize; ++i) {
        double re = (double)(100 + i);
        double im = -(double)i;
        h_a_real[i] = Complex(re, im);
    }

    // 2) Create host data for the dual part (24 elements)
    // We'll make each partial distinct as well
    std::vector<Complex> h_a_dual(dualSize);
    for (int i = 0; i < dualSize; ++i) {
        double re = 200.0 + i;
        double im = 10.0 + i;
        h_a_dual[i] = Complex(re, im);
    }

    // 3) Define the slice parameters:
    //   rowStart=1, rowEnd=3 => that means row indices {1,2}
    //   colStart=1, colEnd=3 => that means col indices {1,2}
    // So the submatrix is shape (2 x 2)
    int rowStart = 1, rowEnd = 3;
    int colStart = 1, colEnd = 3;
    int outRows = rowEnd - rowStart; // 2
    int outCols = colEnd - colStart; // 2
    int outRealSize = outRows * outCols;        // 4
    int outDualSize = outRealSize * dual_size;  // 8

    // 4) Compute the expected submatrix on the host
    // For each row in [1..2], col in [1..2]
    // we read the real array => originalOff = row*cols + col
    // we read the dual array => originalOff*dual_size + k
    std::vector<Complex> h_expectedReal(outRealSize);
    std::vector<Complex> h_expectedDual(outDualSize);

    int idxSub = 0;  // index in submatrix
    for (int r = rowStart; r < rowEnd; ++r) {
        for (int c = colStart; c < colEnd; ++c) {
            int originalOff = r * cols + c;
            // copy real
            h_expectedReal[idxSub] = h_a_real[originalOff];
            // copy dual for partial dimensions
            for (int k = 0; k < dual_size; ++k) {
                int origDualOff = originalOff * dual_size + k;
                int subDualOff  = idxSub * dual_size + k;
                h_expectedDual[subDualOff] = h_a_dual[origDualOff];
            }
            idxSub++;
        }
    }

    // 5) Device allocations for input and output
    Complex *d_a_real = nullptr, *d_a_dual = nullptr;
    Complex *d_out_real = nullptr, *d_out_dual = nullptr;

    CUDA_CHECK(cudaMalloc(&d_a_real, realSize*sizeof(Complex)));
    CUDA_CHECK(cudaMalloc(&d_a_dual, dualSize*sizeof(Complex)));
    CUDA_CHECK(cudaMalloc(&d_out_real, outRealSize*sizeof(Complex)));
    CUDA_CHECK(cudaMalloc(&d_out_dual, outDualSize*sizeof(Complex)));

    // Copy host -> device
    CUDA_CHECK(cudaMemcpy(d_a_real, h_a_real.data(), 
                          realSize*sizeof(Complex), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_a_dual, h_a_dual.data(), 
                          dualSize*sizeof(Complex), cudaMemcpyHostToDevice));

    // 6) Launch the kernel
    // We'll need outRealSize*dual_size threads => (2*2)*2=8 in this example
    int totalThreads = outRealSize * dual_size;  
    int blockSize = 128;
    int gridSize  = (totalThreads + blockSize - 1) / blockSize;

    MatrixDualIndexGet2DKernel<<<gridSize, blockSize>>>(
        d_a_real, d_a_dual,
        rows, cols, dual_size,
        rowStart, rowEnd, colStart, colEnd,
        d_out_real, d_out_dual
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 7) Copy results back
    std::vector<Complex> h_resultReal(outRealSize);
    std::vector<Complex> h_resultDual(outDualSize);

    CUDA_CHECK(cudaMemcpy(h_resultReal.data(), d_out_real, 
                          outRealSize*sizeof(Complex), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_resultDual.data(), d_out_dual, 
                          outDualSize*sizeof(Complex), cudaMemcpyDeviceToHost));

    // 8) Compare with expected
    double tol = 1e-9;

    // Check real
    for (int i = 0; i < outRealSize; ++i) {
        EXPECT_NEAR(h_expectedReal[i].real(), h_resultReal[i].real(), tol)
            << "Mismatch in real part of submatrix index=" << i << " (Re)";
        EXPECT_NEAR(h_expectedReal[i].imag(), h_resultReal[i].imag(), tol)
            << "Mismatch in real part of submatrix index=" << i << " (Im)";
    }

    // Check dual
    for (int i = 0; i < outDualSize; ++i) {
        EXPECT_NEAR(h_expectedDual[i].real(), h_resultDual[i].real(), tol)
            << "Mismatch in dual part of submatrix index=" << i << " (Re)";
        EXPECT_NEAR(h_expectedDual[i].imag(), h_resultDual[i].imag(), tol)
            << "Mismatch in dual part of submatrix index=" << i << " (Im)";
    }

    // 9) Cleanup
    CUDA_CHECK(cudaFree(d_a_real));
    CUDA_CHECK(cudaFree(d_a_dual));
    CUDA_CHECK(cudaFree(d_out_real));
    CUDA_CHECK(cudaFree(d_out_dual));
}


//--------------------------------------------------
// Test
//--------------------------------------------------
TEST(MatrixDualIndexPut2DTest, BasicSubSlicePut)
{
    using Complex = thrust::complex<double>;

    // Source dual matrix: shape = (4 x 5), dual_size=2
    // => src_real has 20 elements, src_dual has 40
    int srcRows=4, srcCols=5, dual_size=2;
    int srcRealSize= srcRows * srcCols;      //=20
    int srcDualSize= srcRealSize * dual_size;//=40

    // We'll define a sub-slice:
    //   rowStartSrc=1, rowEndSrc=3 => rows in [1..2], 2 total
    //   colStartSrc=2, colEndSrc=4 => cols in [2..3], 2 total
    // So the sub-slice is shape (2 x 2).
    int rowStartSrc=1, rowEndSrc=3;
    int colStartSrc=2, colEndSrc=4;

    // Destination dual matrix: shape = (5 x 6), dual_size=2
    // => dst_real has 30 elements, dst_dual has 60
    int dstRows=5, dstCols=6;
    int dstRealSize= dstRows * dstCols;      //=30
    int dstDualSize= dstRealSize * dual_size;//=60

    // We'll place the sub-slice at (rowStartDst=2, colStartDst=1) in the destination
    int rowStartDst=2, colStartDst=1;

    // 1) Build host data for src_real, src_dual
    //    Fill them with recognizable patterns
    std::vector<Complex> h_srcReal(srcRealSize);
    for (int i=0; i< srcRealSize; ++i) {
        double re= 10.0 + i;
        double im= -1.0 * i;
        h_srcReal[i] = Complex(re, im);
    }
    std::vector<Complex> h_srcDual(srcDualSize);
    for (int i=0; i< srcDualSize; ++i) {
        double re= 100.0 + i;
        double im= 5.0 + i;
        h_srcDual[i] = Complex(re, im);
    }

    // 2) Build host data for dst_real, dst_dual
    //    We'll fill with placeholders
    std::vector<Complex> h_dstReal(dstRealSize);
    for (int i=0; i< dstRealSize; ++i) {
        double re= 1000.0 + i;
        double im= 10.0 + i;
        h_dstReal[i] = Complex(re, im);
    }
    std::vector<Complex> h_dstDual(dstDualSize);
    for (int i=0; i< dstDualSize; ++i) {
        double re= 2000.0 + i;
        double im= -(50.0 + i);
        h_dstDual[i] = Complex(re, im);
    }

    // 3) Build the expected final destination on the host
    //    We'll copy the sub-slice from src into this subregion of the original dst
    std::vector<Complex> h_expectedReal = h_dstReal; // start from original
    std::vector<Complex> h_expectedDual = h_dstDual;

    int subRows = rowEndSrc - rowStartSrc;  //=2
    int subCols = colEndSrc - colStartSrc;  //=2
    for(int rr=0; rr< subRows; ++rr) {
        for(int cc=0; cc< subCols; ++cc) {
            // global row/col in src
            int srcRow= rowStartSrc + rr;
            int srcCol= colStartSrc + cc;
            int srcOff = srcRow* srcCols + srcCol;

            // global row/col in dst
            int dstRow= rowStartDst + rr;
            int dstCol= colStartDst + cc;
            int dstOff= dstRow* dstCols + dstCol;

            // real part
            h_expectedReal[dstOff] = h_srcReal[srcOff];

            // dual part for each partial index
            for(int k=0; k< dual_size; ++k) {
                int srcDualOff= srcOff* dual_size + k;
                int dstDualOff= dstOff* dual_size + k;
                h_expectedDual[dstDualOff] = h_srcDual[srcDualOff];
            }
        }
    }

    // 4) Device allocations
    Complex *d_srcReal=nullptr, *d_srcDual=nullptr;
    Complex *d_dstReal=nullptr, *d_dstDual=nullptr;

    CUDA_CHECK(cudaMalloc(&d_srcReal, srcRealSize*sizeof(Complex)));
    CUDA_CHECK(cudaMalloc(&d_srcDual, srcDualSize*sizeof(Complex)));
    CUDA_CHECK(cudaMalloc(&d_dstReal, dstRealSize*sizeof(Complex)));
    CUDA_CHECK(cudaMalloc(&d_dstDual, dstDualSize*sizeof(Complex)));

    // Copy host -> device
    CUDA_CHECK(cudaMemcpy(d_srcReal, h_srcReal.data(),
                          srcRealSize*sizeof(Complex), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_srcDual, h_srcDual.data(),
                          srcDualSize*sizeof(Complex), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dstReal, h_dstReal.data(),
                          dstRealSize*sizeof(Complex), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dstDual, h_dstDual.data(),
                          dstDualSize*sizeof(Complex), cudaMemcpyHostToDevice));

    // 5) Launch kernel for the sub-slice
    int subSize = subRows * subCols * dual_size; // 2*2*2=8
    int blockSize=128;
    int gridSize= (subSize + blockSize - 1)/ blockSize;

    MatrixDualIndexPut2DKernel<<<gridSize, blockSize>>>(
        d_srcReal, d_srcDual,
        srcRows, srcCols, dual_size,
        rowStartSrc, rowEndSrc,
        colStartSrc, colEndSrc,
        d_dstReal, d_dstDual,
        dstRows, dstCols,
        rowStartDst, colStartDst
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 6) Copy results back
    std::vector<Complex> h_resultReal(dstRealSize);
    std::vector<Complex> h_resultDual(dstDualSize);

    CUDA_CHECK(cudaMemcpy(h_resultReal.data(), d_dstReal,
                          dstRealSize*sizeof(Complex), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_resultDual.data(), d_dstDual,
                          dstDualSize*sizeof(Complex), cudaMemcpyDeviceToHost));

    // 7) Compare
    double tol = 1e-9;
    // Real
    for(int i=0; i< dstRealSize; ++i) {
        EXPECT_NEAR(h_expectedReal[i].real(), h_resultReal[i].real(), tol)
            << "Mismatch in real part, index=" << i;
        EXPECT_NEAR(h_expectedReal[i].imag(), h_resultReal[i].imag(), tol)
            << "Mismatch in real part, index=" << i;
    }
    // Dual
    for(int i=0; i< dstDualSize; ++i) {
        EXPECT_NEAR(h_expectedDual[i].real(), h_resultDual[i].real(), tol)
            << "Mismatch in dual part, index=" << i << " (Re)";
        EXPECT_NEAR(h_expectedDual[i].imag(), h_resultDual[i].imag(), tol)
            << "Mismatch in dual part, index=" << i << " (Im)";
    }

    // 8) Cleanup
    CUDA_CHECK(cudaFree(d_srcReal));
    CUDA_CHECK(cudaFree(d_srcDual));
    CUDA_CHECK(cudaFree(d_dstReal));
    CUDA_CHECK(cudaFree(d_dstDual));
}


TEST(MatrixDualSigncondTest, BasicTest) {
    const int rows = 2;
    const int cols = 2;
    const int dual_size = 2;

    // Input matrices (real and dual parts)
    std::vector<thrust::complex<double>> a_real = {
        {1.0, 0.0}, {-2.0, 0.0},
        {3.0, 0.0}, {-4.0, 0.0}
    };
    std::vector<thrust::complex<double>> a_dual = {
        {0.1, 0.0}, {0.2, 0.0},
        {0.3, 0.0}, {0.4, 0.0},
        {0.5, 0.0}, {0.6, 0.0},
        {0.7, 0.0}, {0.8, 0.0}
    };

    std::vector<thrust::complex<double>> b_real = {
        {1.0, 0.0}, {1.0, 0.0},
        {-1.0, 0.0}, {-1.0, 0.0}
    };
    std::vector<thrust::complex<double>> b_dual = {
        {0.0, 0.0}, {0.0, 0.0},
        {0.0, 0.0}, {0.0, 0.0},
        {0.0, 0.0}, {0.0, 0.0},
        {0.0, 0.0}, {0.0, 0.0}
    };

    // Allocate device memory
    thrust::complex<double> *d_a_real, *d_a_dual, *d_b_real, *d_b_dual;
    thrust::complex<double> *d_result_real, *d_result_dual;

    size_t real_size = rows * cols * sizeof(thrust::complex<double>);
    size_t dual_size_total = rows * cols * dual_size * sizeof(thrust::complex<double>);

    cudaMalloc(&d_a_real, real_size);
    cudaMalloc(&d_a_dual, dual_size_total);
    cudaMalloc(&d_b_real, real_size);
    cudaMalloc(&d_b_dual, dual_size_total);
    cudaMalloc(&d_result_real, real_size);
    cudaMalloc(&d_result_dual, dual_size_total);

    // Copy data to device
    cudaMemcpy(d_a_real, a_real.data(), real_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_dual, a_dual.data(), dual_size_total, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_real, b_real.data(), real_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_dual, b_dual.data(), dual_size_total, cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 blockDim(256);
    dim3 gridDim((rows * cols * dual_size + blockDim.x - 1) / blockDim.x);
    MatrixDualSigncondKernel<<<1, blockDim>>>(d_a_real, d_a_dual, d_b_real, d_b_dual,
                                                   rows, cols, dual_size,
                                                   d_result_real, d_result_dual, 1.0e-6);

    // Copy results back to host
    std::vector<thrust::complex<double>> result_real(rows * cols);
    std::vector<thrust::complex<double>> result_dual(rows * cols * dual_size);

    cudaMemcpy(result_real.data(), d_result_real, real_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(result_dual.data(), d_result_dual, dual_size_total, cudaMemcpyDeviceToHost);

    // Expected results
    std::vector<thrust::complex<double>> expected_real = {
        {1.0, 0.0}, {2.0, 0.0},
        {-3.0, 0.0}, {-4.0, 0.0}
    };
    std::vector<thrust::complex<double>> expected_dual = {
        {0.1, 0.0}, {0.2, 0.0},
        {-0.3, 0.0}, {-0.4, 0.0},
        {-0.5, 0.0}, {-0.6, 0.0},
        {0.7, 0.0}, {0.8, 0.0}
    };

    // Validate results
    for (int i = 0; i < rows * cols; ++i) {
        EXPECT_EQ(result_real[i], expected_real[i]) << "Mismatch in real part at index " << i;
    }

    for (int i = 0; i < rows * cols * dual_size; ++i) {
        EXPECT_EQ(result_dual[i], expected_dual[i]) << "Mismatch in dual part at index " << i;
    }

    // Free device memory
    cudaFree(d_a_real);
    cudaFree(d_a_dual);
    cudaFree(d_b_real);
    cudaFree(d_b_dual);
    cudaFree(d_result_real);
    cudaFree(d_result_dual);
}


// ---------------------------------------------------------------------------
// Google Test: MatrixHyperDualTest
// ---------------------------------------------------------------------------
TEST(MatrixHyperDualTest, ElementwiseAdd_2x2_dualsize2)
{
    using T = double;

    // Matrix dimensions
    int rows = 2;
    int cols = 2;
    int dual_size = 2;

    // total_real = number of matrix entries
    int total_real  = rows * cols;             // = 4
    // total_dual = number of first-order partials
    int total_dual  = rows * cols * dual_size; // = 8
    // total_hyper = number of second-order partials
    int total_hyper = rows * cols * dual_size * dual_size; // = 16

    // ---------------------------------------------------------------------
    // 1) Prepare host data
    // ---------------------------------------------------------------------
    std::vector<thrust::complex<T>> a_real(total_real), b_real(total_real);
    for (int i = 0; i < total_real; ++i) {
        a_real[i] = thrust::complex<T>(static_cast<T>(i+1), static_cast<T>(i+1));
        b_real[i] = thrust::complex<T>(static_cast<T>(i+5), static_cast<T>(i+5));
    }

    std::vector<thrust::complex<T>> a_dual(total_dual), b_dual(total_dual);
    for (int i = 0; i < total_dual; ++i) {
        a_dual[i] = thrust::complex<T>(i + 0.1, i + 0.2);
        b_dual[i] = thrust::complex<T>(10*(i+1), 10*(i+1));
    }

    std::vector<thrust::complex<T>> a_hyper(total_hyper), b_hyper(total_hyper);
    for (int i = 0; i < total_hyper; ++i) {
        a_hyper[i] = thrust::complex<T>(2.0*i, 2.0*i);
        b_hyper[i] = thrust::complex<T>(3.0*i, -1.0*i);
    }

    // ---------------------------------------------------------------------
    // 2) Allocate device arrays & copy
    // ---------------------------------------------------------------------
    thrust::complex<T> *d_a_real, *d_b_real,
                       *d_a_dual, *d_b_dual,
                       *d_a_hyper, *d_b_hyper;
    thrust::complex<T> *d_result_real, *d_result_dual, *d_result_hyper;

    AllocateAndCopy(a_real,   &d_a_real);
    AllocateAndCopy(b_real,   &d_b_real);
    AllocateAndCopy(a_dual,   &d_a_dual);
    AllocateAndCopy(b_dual,   &d_b_dual);
    AllocateAndCopy(a_hyper,  &d_a_hyper);
    AllocateAndCopy(b_hyper,  &d_b_hyper);

    cudaMalloc(&d_result_real,  total_real  * sizeof(thrust::complex<T>));
    // We allocate "total_hyper" for result_dual to avoid any boundary issues,
    // though only the first "total_dual" entries matter.
    cudaMalloc(&d_result_dual,  total_hyper * sizeof(thrust::complex<T>));
    cudaMalloc(&d_result_hyper, total_hyper * sizeof(thrust::complex<T>));

    // ---------------------------------------------------------------------
    // 3) Launch the kernel
    // ---------------------------------------------------------------------
    int blockSize = 128;
    int gridSize  = (total_hyper + blockSize - 1) / blockSize;
    MatrixHyperDualElementwiseAddKernel<T><<<gridSize, blockSize>>>(
        d_a_real, d_a_dual, d_a_hyper,
        d_b_real, d_b_dual, d_b_hyper,
        rows, cols, dual_size,
        d_result_real, d_result_dual, d_result_hyper
    );
    cudaDeviceSynchronize();

    // ---------------------------------------------------------------------
    // 4) Copy results back to host
    // ---------------------------------------------------------------------
    auto res_real  = CopyToHost(d_result_real,  total_real);
    auto res_dual  = CopyToHost(d_result_dual,  total_hyper);  // read back 16
    auto res_hyper = CopyToHost(d_result_hyper, total_hyper);

    // Clean up device memory
    cudaFree(d_a_real);
    cudaFree(d_b_real);
    cudaFree(d_a_dual);
    cudaFree(d_b_dual);
    cudaFree(d_a_hyper);
    cudaFree(d_b_hyper);
    cudaFree(d_result_real);
    cudaFree(d_result_dual);
    cudaFree(d_result_hyper);

    // ---------------------------------------------------------------------
    // 5) Verify correctness
    // ---------------------------------------------------------------------
    // The kernel logic does:
    //   if (k=0 && l=0) => result_real[off] = a_real[off] + b_real[off]
    //   if (l=0) => result_dual[idx/dual_size] = a_dual[idx/dual_size] + b_dual[idx/dual_size]
    //   result_hyper[idx] = a_hyper[idx] + b_hyper[idx]
    //
    // So the result_dual array is assigned for exactly "dual_size * rows*cols" threads 
    // (the ones where l=0). We'll check only the first total_dual entries.

    // -- Real part (4 elements)
    for (int i = 0; i < total_real; ++i) {
        thrust::complex<T> expected = a_real[i] + b_real[i];
        EXPECT_EQ(res_real[i], expected) << "Real mismatch at i=" << i;
    }

    // -- Dual part (8 elements) 
    //   The kernel uses half as many threads (l=0) to store these. 
    //   We only verify the first 8 entries in res_dual, which should match a_dual + b_dual.
    for (int i = 0; i < total_dual; ++i) {
        thrust::complex<T> expected = a_dual[i] + b_dual[i];
        EXPECT_EQ(res_dual[i], expected) << "Dual mismatch at i=" << i;
    }

    // -- Hyper part (16 elements)
    //   Always a_hyper[idx] + b_hyper[idx], for idx in [0..15].
    for (int i = 0; i < total_hyper; ++i) {
        thrust::complex<T> expected = a_hyper[i] + b_hyper[i];
        EXPECT_EQ(res_hyper[i], expected) << "Hyper mismatch at i=" << i;
    }
}


// ---------------------------------------------------------------------------
// Test: Multiply two 2x2 hyper-dual matrices (dual_size=2) elementwise
// ---------------------------------------------------------------------------
TEST(MatrixHyperDualTest, ElementwiseMul_2x2_dualsize2)
{
    using T = double;

    // Dimensions
    int rows = 2;
    int cols = 2;
    int dual_size = 2;

    // Sizes
    int total_real  = rows * cols;             // 4
    int total_dual  = rows * cols * dual_size; // 8
    int total_hyper = rows * cols * dual_size * dual_size; // 16

    // 1) Prepare host data for A and B
    //    a_real, a_dual, a_hyper; b_real, b_dual, b_hyper
    std::vector<thrust::complex<T>> a_real(total_real), b_real(total_real);
    std::vector<thrust::complex<T>> a_dual(total_dual), b_dual(total_dual);
    std::vector<thrust::complex<T>> a_hyper(total_hyper), b_hyper(total_hyper);

    // Fill with some patterns
    // real: (1,1), (2,2), (3,3), (4,4) for "A"
    //       (5,5), (6,6), (7,7), (8,8) for "B"
    for (int i = 0; i < total_real; ++i) {
        a_real[i] = thrust::complex<T>(i+1, i+1);
        b_real[i] = thrust::complex<T>(i+5, i+5);
    }

    // dual: for A => a_dual[i] = (i+0.1, i+0.2)
    //       for B => b_dual[i] = (10(i+1), 10(i+1))
    for (int i = 0; i < total_dual; ++i) {
        a_dual[i] = thrust::complex<T>(i + 0.1, i + 0.2);
        b_dual[i] = thrust::complex<T>(10.0 * (i+1), 10.0 * (i+1));
    }

    // hyper: for A => (2i, 2i)
    //         for B => (3i, -i)
    for (int i = 0; i < total_hyper; ++i) {
        a_hyper[i] = thrust::complex<T>(2.0*i, 2.0*i);
        b_hyper[i] = thrust::complex<T>(3.0*i, -1.0*i);
    }

    // 2) Allocate device memory & copy
    thrust::complex<T>* d_a_real;    thrust::complex<T>* d_b_real;
    thrust::complex<T>* d_a_dual;    thrust::complex<T>* d_b_dual;
    thrust::complex<T>* d_a_hyper;   thrust::complex<T>* d_b_hyper;
    thrust::complex<T>* d_c_real;    thrust::complex<T>* d_c_dual; 
    thrust::complex<T>* d_c_hyper;

    AllocateAndCopy(a_real,   &d_a_real);
    AllocateAndCopy(b_real,   &d_b_real);
    AllocateAndCopy(a_dual,   &d_a_dual);
    AllocateAndCopy(b_dual,   &d_b_dual);
    AllocateAndCopy(a_hyper,  &d_a_hyper);
    AllocateAndCopy(b_hyper,  &d_b_hyper);

    cudaMalloc(&d_c_real,  total_real * sizeof(thrust::complex<T>));
    cudaMalloc(&d_c_dual,  total_hyper * sizeof(thrust::complex<T>));
    cudaMalloc(&d_c_hyper, total_hyper * sizeof(thrust::complex<T>));

    // 3) Launch the kernel
    int blockSize = 128;
    int gridSize  = (total_hyper + blockSize - 1) / blockSize;
    MatrixHyperDualElementwiseMulKernel<T><<<gridSize, blockSize>>>(
        d_a_real, d_a_dual, d_a_hyper,
        d_b_real, d_b_dual, d_b_hyper,
        rows, cols, dual_size,
        d_c_real, d_c_dual, d_c_hyper
    );
    cudaDeviceSynchronize();

    // 4) Copy results back to host
    auto c_real  = CopyToHost(d_c_real,  total_real);
    auto c_dual  = CopyToHost(d_c_dual,  total_hyper);
    auto c_hyper = CopyToHost(d_c_hyper, total_hyper);

    // Free device memory
    cudaFree(d_a_real);
    cudaFree(d_b_real);
    cudaFree(d_a_dual);
    cudaFree(d_b_dual);
    cudaFree(d_a_hyper);
    cudaFree(d_b_hyper);
    cudaFree(d_c_real);
    cudaFree(d_c_dual);
    cudaFree(d_c_hyper);

    // 5) Compute reference on host & compare
    //    We'll do a small nested loop:
    //    c_real(off) = a_real(off)*b_real(off)
    //    c_dual(off,k) = a_real(off)*b_dual(off,k) + b_real(off)*a_dual(off,k)
    //    c_hyper(off,k,l) = ...
    // 
    //    Where:
    //      off = i*cols + j
    //      dual_off(i,j,k)  = i*cols*dual_size + j*dual_size + k
    //      hyper_off(i,j,k,l)= i*cols*dual_size*dual_size + j*dual_size*dual_size + k*dual_size + l

    std::vector<thrust::complex<T>> ref_real(total_real);
    std::vector<thrust::complex<T>> ref_dual(total_dual);
    std::vector<thrust::complex<T>> ref_hyper(total_hyper);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            int off = i*cols + j;
            // real
            auto ar = a_real[off];
            auto br = b_real[off];
            ref_real[off] = ar * br;

            // dual
            for (int k = 0; k < dual_size; ++k) {
                int dual_off = i*cols*dual_size + j*dual_size + k;
                auto ad = a_dual[dual_off];
                auto bd = b_dual[dual_off];
                ref_dual[dual_off] = ar * bd + br * ad;
            }

            // hyper
            for (int k = 0; k < dual_size; ++k) {
                for (int l = 0; l < dual_size; ++l) {
                    int hyper_off = i*cols*dual_size*dual_size + j*dual_size*dual_size + k*dual_size + l;
                    auto ah = a_hyper[hyper_off];
                    auto bh = b_hyper[hyper_off];
                    auto adk = a_dual[i*cols*dual_size + j*dual_size + k];
                    auto bdl = b_dual[i*cols*dual_size + j*dual_size + l];

                    // c_hyper = a_real*b_hyper + b_real*a_hyper + a_dual[k]*b_dual[l]
                    ref_hyper[hyper_off] = (ar * bh) + (br * ah) + (adk * bdl);
                }
            }
        }
    }

    // 6) Compare c_* vs. ref_*
    // real
    for (int i = 0; i < total_real; ++i) {
        EXPECT_NEAR(c_real[i].real(), ref_real[i].real(), 1.0e-6) << "Mismatch in real part at i=" << i;
        EXPECT_NEAR(c_real[i].imag(), ref_real[i].imag(), 1.0e-6) << "Mismatch in real part at i=" << i;
    }
    // dual
    // We only wrote "dual_size" threads per matrix entry, but we stored them in c_dual 
    // across "total_hyper" allocated space. In the kernel code, we used 
    //   result_dual[off_dual_k], 
    // so only the first "total_dual" elements matter.
    for (int i = 0; i < total_dual; ++i) {
        EXPECT_NEAR(c_dual[i].real(), ref_dual[i].real(), 1.0e-6) << "Mismatch in dual part at i=" << i;
        EXPECT_NEAR(c_dual[i].imag(), ref_dual[i].imag(), 1.0e-6) << "Mismatch in dual part at i=" << i;
    }
    // hyper
    for (int i = 0; i < total_hyper; ++i) {
        EXPECT_NEAR(c_hyper[i].real(), ref_hyper[i].real(), 1.0e-6) << "Mismatch in hyper part at i=" << i;
        EXPECT_NEAR(c_hyper[i].imag(), ref_hyper[i].imag(), 1.0e-6) << "Mismatch in hyper part at i=" << i;
    }
}


// ---------------------------------------------------------------------------
// Test: Multiply a 2x3 hyper-dual matrix by a 3x1 hyper-dual vector
// ---------------------------------------------------------------------------
TEST(MatrixHyperDualVectorHyperDualTest, Multiply_2x3_DualSize2)
{
    int rows = 2;
    int cols = 3;
    int dual_size = 2;

    // A has:
    //   A_real   => rows*cols = 6
    //   A_dual   => rows*cols*dual_size = 12
    //   A_hyper  => rows*cols*dual_size*dual_size = 24
    int totalA_real  = rows*cols;
    int totalA_dual  = rows*cols*dual_size;
    int totalA_hyper = rows*cols*dual_size*dual_size;

    // B has:
    //   B_real   => cols = 3
    //   B_dual   => cols*dual_size = 6
    //   B_hyper  => cols*dual_size*dual_size = 12
    int totalB_real  = cols;
    int totalB_dual  = cols*dual_size;
    int totalB_hyper = cols*dual_size*dual_size;

    // R has:
    //   R_real   => rows = 2
    //   R_dual   => rows*dual_size = 4
    //   R_hyper  => rows*dual_size*dual_size = 8
    int totalR_real  = rows;
    int totalR_dual  = rows*dual_size;
    int totalR_hyper = rows*dual_size*dual_size;

    // 1) Host vectors
    std::vector<thrust::complex<double>> A_real(totalA_real),
                                         A_dual(totalA_dual),
                                         A_hyper(totalA_hyper);

    std::vector<thrust::complex<double>> B_real(totalB_real),
                                         B_dual(totalB_dual),
                                         B_hyper(totalB_hyper);

    // Fill them with simple patterns:

    // (a) A_real: row=0 => (1,2,3); row=1 => (4,5,6)
    for (int i = 0; i < totalA_real; ++i) {
        double val = double(i+1);
        A_real[i] = thrust::complex<double>(val, 0.0);
    }

    // (b) A_dual => just a pattern
    for (int i = 0; i < totalA_dual; ++i) {
        double val = 0.1 + double(i);
        A_dual[i] = thrust::complex<double>(val, 0.0);
    }

    // (c) A_hyper => another pattern
    for (int i = 0; i < totalA_hyper; ++i) {
        double val = 0.01 * double(i);
        A_hyper[i] = thrust::complex<double>(val, -val); // give it some imag part
    }

    // (d) B_real => e.g. (10, 20, 30)
    for (int j = 0; j < totalB_real; ++j) {
        double val = 10.0 * (j+1);
        B_real[j] = thrust::complex<double>(val, 0.0);
    }

    // (e) B_dual => pattern
    for (int i = 0; i < totalB_dual; ++i) {
        double val = 0.5 + double(i);
        B_dual[i] = thrust::complex<double>(val, val*0.1);
    }

    // (f) B_hyper => pattern
    for (int i = 0; i < totalB_hyper; ++i) {
        double val = 0.01 * double(i + 5);
        B_hyper[i] = thrust::complex<double>(val, val);
    }

    // 2) Device memory
    thrust::complex<double> *dA_real, *dA_dual, *dA_hyper;
    thrust::complex<double> *dB_real, *dB_dual, *dB_hyper;
    thrust::complex<double> *dR_real, *dR_dual, *dR_hyper;

    AllocateAndCopy(A_real,  &dA_real);
    AllocateAndCopy(A_dual,  &dA_dual);
    AllocateAndCopy(A_hyper, &dA_hyper);

    AllocateAndCopy(B_real,  &dB_real);
    AllocateAndCopy(B_dual,  &dB_dual);
    AllocateAndCopy(B_hyper, &dB_hyper);

    cudaMalloc(&dR_real,  totalR_real * sizeof(thrust::complex<double>));
    cudaMalloc(&dR_dual,  totalR_dual * sizeof(thrust::complex<double>));
    cudaMalloc(&dR_hyper, totalR_hyper * sizeof(thrust::complex<double>));

    // Zero-initialize the result
    cudaMemset(dR_real,  0, totalR_real*sizeof(thrust::complex<double>));
    cudaMemset(dR_dual,  0, totalR_dual*sizeof(thrust::complex<double>));
    cudaMemset(dR_hyper, 0, totalR_hyper*sizeof(thrust::complex<double>));

    // 3) Launch kernel
    int totalThreads = rows * dual_size * dual_size; // 2*2*2 = 8
    int blockSize = 128;
    int gridSize  = (totalThreads + blockSize - 1)/blockSize;

    MatrixHyperDualVectorHyperDualMultiplyKernel<<<gridSize, blockSize>>>(
        dA_real, dA_dual, dA_hyper,
        dB_real, dB_dual, dB_hyper,
        rows, cols, dual_size,
        dR_real, dR_dual, dR_hyper
    );
    cudaDeviceSynchronize();

    // 4) Copy results back
    auto hR_real  = CopyToHost(dR_real,  totalR_real);
    auto hR_dual  = CopyToHost(dR_dual,  totalR_dual);
    auto hR_hyper = CopyToHost(dR_hyper, totalR_hyper);

    // Free device memory
    cudaFree(dA_real);
    cudaFree(dA_dual);
    cudaFree(dA_hyper);
    cudaFree(dB_real);
    cudaFree(dB_dual);
    cudaFree(dB_hyper);
    cudaFree(dR_real);
    cudaFree(dR_dual);
    cudaFree(dR_hyper);

    // 5) Compute reference on CPU
    // Reference formula:
    //   R_real[i] = sum_j (A_real(i,j) * B_real[j])
    //   R_dual(i,k) = sum_j [ A_real(i,j)*B_dual(j,k) + A_dual(i,j,k)*B_real[j] ]
    //   R_hyper(i,k,l) = sum_j [ 
    //       A_real(i,j)*B_hyper(j,k,l)
    //     + A_hyper(i,j,k,l)*B_real[j]
    //     + A_dual(i,j,k)*B_dual(j,l)
    //   ]

    std::vector<thrust::complex<double>> ref_real(totalR_real,  thrust::complex<double>(0,0));
    std::vector<thrust::complex<double>> ref_dual(totalR_dual,  thrust::complex<double>(0,0));
    std::vector<thrust::complex<double>> ref_hyper(totalR_hyper,thrust::complex<double>(0,0));

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            // Offsets in A
            auto aR = A_real[i*cols + j];
            for (int k = 0; k < dual_size; ++k) {
                // A_dual offset
                auto aD = A_dual[(i*cols + j)*dual_size + k];
                for (int l = 0; l < dual_size; ++l) {
                    auto aH = A_hyper[(i*cols + j)*dual_size*dual_size + k*dual_size + l];
                    // Offsets in B
                    auto bR   = B_real[j];
                    auto bDkl = B_dual[j*dual_size + l]; // careful with indexing: j,l for the dual cross
                    auto bDk  = B_dual[j*dual_size + k]; // for the partial with aD
                    auto bHkl = B_hyper[j*dual_size*dual_size + k*dual_size + l];

                    // c_hyper(i,k,l) += aR*bH + aH*bR + aD*bD
                    ref_hyper[i*dual_size*dual_size + k*dual_size + l] +=
                        (aR * bHkl) + (aH * bR) + (aD * bDkl);
                }
                // c_dual(i,k) => we must accumulate aR*bD + aD*bR over j
                auto bR_ = B_real[j];
                auto bD_ = B_dual[j*dual_size + k];
                ref_dual[i*dual_size + k] += (aR*bD_ + aD*bR_);
            }
            // c_real(i) => sum_j (aR*bR)
            ref_real[i] += (aR * B_real[j]);
        }
    }

    // 6) Compare
    for (int i = 0; i < totalR_real; ++i) {
        EXPECT_EQ(hR_real[i], ref_real[i]) 
            << "Mismatch in R_real at i=" << i;
    }
    for (int i = 0; i < totalR_dual; ++i) {
        EXPECT_EQ(hR_dual[i], ref_dual[i])
            << "Mismatch in R_dual at i=" << i;
    }
    for (int i = 0; i < totalR_hyper; ++i) {
        EXPECT_EQ(hR_hyper[i], ref_hyper[i])
            << "Mismatch in R_hyper at i=" << i;
    }
}



// ---------------------------------------------------------------------------
// Google Test: MatrixHyperDualPowTest
// ---------------------------------------------------------------------------
TEST(MatrixHyperDualPowTest, Power2_2x2_dual2)
{
    // We'll do a small matrix: rows=2, cols=2, dual_size=2,
    // and set power=2.0 for a quick check.

    int rows = 2;
    int cols = 2;
    int dual_size = 2;
    double power = 2.0;

    // Matrix shape => 4 entries for real
    // dual shape => 4 * 2 = 8
    // hyper shape => 4 * 2 * 2 = 16

    int totalReal   = rows * cols;             // 4
    int totalDual   = rows * cols * dual_size; // 8
    int totalHyper  = rows * cols * dual_size * dual_size; //16

    // 1) Host arrays
    std::vector<thrust::complex<double>> a_real(totalReal),
                                         a_dual(totalDual),
                                         a_hyper(totalHyper);

    // (a) Fill a_real => e.g.: (1,0), (2,0), (3,0), (4,0)
    for (int i = 0; i < totalReal; ++i) {
        a_real[i] = thrust::complex<double>(double(i+1), 0.0);
    }

    // (b) a_dual => small pattern
    //     e.g., a_dual[i] = (0.1 + i, -0.2 - i)
    for (int i = 0; i < totalDual; ++i) {
        double re = 0.1 + double(i);
        double im = -0.2 - double(i);
        a_dual[i] = thrust::complex<double>(re, im);
    }

    // (c) a_hyper => fill with e.g. (0.01*i, 0.02*i)
    for (int i = 0; i < totalHyper; ++i) {
        double re = 0.01 * i;
        double im = 0.02 * i;
        a_hyper[i] = thrust::complex<double>(re, im);
    }

    // 2) Allocate device arrays
    thrust::complex<double>* d_a_real   = nullptr;
    thrust::complex<double>* d_a_dual   = nullptr;
    thrust::complex<double>* d_a_hyper  = nullptr;
    thrust::complex<double>* d_r_real   = nullptr;
    thrust::complex<double>* d_r_dual   = nullptr;
    thrust::complex<double>* d_r_hyper  = nullptr;

    AllocateAndCopy(a_real,  &d_a_real);
    AllocateAndCopy(a_dual,  &d_a_dual);
    AllocateAndCopy(a_hyper, &d_a_hyper);

    cudaMalloc(&d_r_real,  totalReal  * sizeof(thrust::complex<double>));
    cudaMalloc(&d_r_dual,  totalDual  * sizeof(thrust::complex<double>));
    cudaMalloc(&d_r_hyper, totalHyper * sizeof(thrust::complex<double>));

    // Zero out the result
    cudaMemset(d_r_real,  0, totalReal*sizeof(thrust::complex<double>));
    cudaMemset(d_r_dual,  0, totalDual*sizeof(thrust::complex<double>));
    cudaMemset(d_r_hyper, 0, totalHyper*sizeof(thrust::complex<double>));

    // 3) Launch the kernel
    int totalThreads = totalHyper; // 16
    int blockSize = 128;
    int gridSize = (totalThreads + blockSize - 1)/blockSize;
    MatrixHyperDualPowKernel<<<gridSize, blockSize>>>(
        d_a_real, d_a_dual, d_a_hyper,
        power,
        rows, cols, dual_size,
        d_r_real, d_r_dual, d_r_hyper
    );
    cudaDeviceSynchronize();

    // 4) Copy results back
    auto h_r_real  = CopyToHost(d_r_real,  totalReal);
    auto h_r_dual  = CopyToHost(d_r_dual,  totalDual);
    auto h_r_hyper = CopyToHost(d_r_hyper, totalHyper);

    // Free device memory
    cudaFree(d_a_real);
    cudaFree(d_a_dual);
    cudaFree(d_a_hyper);
    cudaFree(d_r_real);
    cudaFree(d_r_dual);
    cudaFree(d_r_hyper);

    // 5) Compute reference on CPU
    //    For each entry x in [a_real, a_dual, a_hyper],
    //    we do:
    //       real = (a_real)^2
    //       dual(k) = 2 * a_real^(2-1) * a_dual(k)
    //       hyper(k,l) = 2*a_real^(1)* a_hyper(k,l)
    //                   + 2*(2-1)*a_real^(0)* a_dual(k)*a_dual(l)
    //                 = 2*a_real* a_hyper(k,l) + 2*1*a_dual(k)*a_dual(l)

    std::vector<thrust::complex<double>> ref_real(totalReal);
    std::vector<thrust::complex<double>> ref_dual(totalDual);
    std::vector<thrust::complex<double>> ref_hyper(totalHyper);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            int off = i*cols + j;
            thrust::complex<double> Ar = a_real[off];

            // real => (Ar)^2
            auto realVal = Ar * Ar;

            // fill it
            ref_real[off] = realVal;

            // dual => for k in [0..1]
            for (int k = 0; k < dual_size; ++k) {
                int dualOff = (i*cols + j)*dual_size + k;
                thrust::complex<double> Ad = a_dual[dualOff];
                // result = 2 * Ar^(2-1) * Ad = 2 * Ar * Ad
                ref_dual[dualOff] = thrust::complex<double>(2.0,0.0) * (Ar * Ad);
            }

            // hyper => for k,l in [0..1]
            for (int k = 0; k < dual_size; ++k) {
                for (int l = 0; l < dual_size; ++l) {
                    int hyperOff = (i*cols + j)*dual_size*dual_size + k*dual_size + l;
                    thrust::complex<double> Ah = a_hyper[hyperOff];
                    // formula:
                    //   2*Ar^1 * Ah + 2*(2-1)* Ar^(0)* a_dual(k)*a_dual(l)
                    // = 2*Ar*Ah + 2* a_dual(k)*a_dual(l)
                    auto Ad_k = a_dual[(i*cols + j)*dual_size + k];
                    auto Ad_l = a_dual[(i*cols + j)*dual_size + l];
                    thrust::complex<double> part1 = 2.0 * Ar * Ah; 
                    thrust::complex<double> part2 = 2.0 * (Ad_k * Ad_l);
                    ref_hyper[hyperOff] = part1 + part2;
                }
            }
        }
    }

    // 6) Compare
    // Real part
    for (int i = 0; i < totalReal; ++i) {
        EXPECT_NEAR(h_r_real[i].real(), ref_real[i].real(), 1.0e-6) << "Mismatch real at index i=" << i;
        EXPECT_NEAR(h_r_real[i].imag(), ref_real[i].imag(), 1.0e-6) << "Mismatch real at index i=" << i;
    }
    // Dual part
    for (int i = 0; i < totalDual; ++i) {
        EXPECT_NEAR(h_r_dual[i].real(), ref_dual[i].real(), 1.0e-6) << "Mismatch dual at index i=" << i;
        EXPECT_NEAR(h_r_dual[i].imag(), ref_dual[i].imag(), 1.0e-6) << "Mismatch dual at index i=" << i;
    }
    // Hyper part
    for (int i = 0; i < totalHyper; ++i) {
        EXPECT_NEAR(h_r_hyper[i].real(), ref_hyper[i].real(), 1.0e-6) << "Mismatch hyper at index i=" << i;
        EXPECT_NEAR(h_r_hyper[i].imag(), ref_hyper[i].imag(), 1.0e-6) << "Mismatch hyper at index i=" << i;
    }
}


// ---------------------------------------------------------------------------
// Test: Slicing a 3x4 matrix with row range [1,3) and col range [1,3)
// ---------------------------------------------------------------------------
TEST(MatrixHyperDualIndexGetTest, Slice3x4_DualSize2)
{
    // 1) Setup
    int rows = 3;
    int cols = 4;
    int dual_size = 2;

    int rowStart = 1;
    int rowEnd   = 3; // => outRows = 2
    int colStart = 1;
    int colEnd   = 3; // => outCols = 2

    // Full matrix sizes
    //   real => rows*cols = 3*4=12
    //   dual => rows*cols*dual_size = 12*2=24
    //   hyper=> rows*cols*dual_size*dual_size =12*2*2=48
    int total_real  = rows*cols; 
    int total_dual  = rows*cols*dual_size;
    int total_hyper = rows*cols*dual_size*dual_size;

    // Output submatrix sizes
    //   outRows=2, outCols=2
    //   out_real => 4
    //   out_dual => 4*2=8
    //   out_hyper => 4*2*2=16
    int outRows = rowEnd - rowStart; // 2
    int outCols = colEnd - colStart; // 2
    int out_real_size  = outRows*outCols;          // 4
    int out_dual_size  = outRows*outCols*dual_size; // 8
    int out_hyper_size = outRows*outCols*dual_size*dual_size; // 16

    // 2) Create host input arrays
    std::vector<thrust::complex<double>> a_real(total_real);
    std::vector<thrust::complex<double>> a_dual(total_dual);
    std::vector<thrust::complex<double>> a_hyper(total_hyper);

    // Fill them with some simple pattern:
    // (a) a_real => e.g. real(i,j)= (i+1)*10 + j => so we can see differences easily
    // For i in [0..2], j in [0..3]
    // e.g. row0 => (10,11,12,13), row1 => (20,21,22,23), row2 => (30,31,32,33)
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            a_real[i*cols + j] = thrust::complex<double>(double((i+1)*10 + j), 0.0);
        }
    }

    // (b) a_dual => length=24, fill with pattern a_dual[idx] = (idx +0.1, -idx*0.1)
    for (int idx = 0; idx < total_dual; ++idx) {
        double re = double(idx) + 0.1;
        double im = -0.1*idx;
        a_dual[idx] = thrust::complex<double>(re, im);
    }

    // (c) a_hyper => length=48 => fill with (0.01*i, 0.02*i)
    for (int i = 0; i < total_hyper; ++i) {
        double re = 0.01 * i;
        double im = 0.02 * i;
        a_hyper[i] = thrust::complex<double>(re, im);
    }

    // 3) Allocate device memory and copy
    thrust::complex<double>* d_a_real=nullptr;
    thrust::complex<double>* d_a_dual=nullptr;
    thrust::complex<double>* d_a_hyper=nullptr;

    AllocateAndCopy(a_real,  &d_a_real);
    AllocateAndCopy(a_dual,  &d_a_dual);
    AllocateAndCopy(a_hyper, &d_a_hyper);

    // Output arrays
    thrust::complex<double>* d_out_real=nullptr;
    thrust::complex<double>* d_out_dual=nullptr;
    thrust::complex<double>* d_out_hyper=nullptr;

    cudaMalloc(&d_out_real,  out_real_size*sizeof(thrust::complex<double>));
    cudaMalloc(&d_out_dual,  out_dual_size*sizeof(thrust::complex<double>));
    cudaMalloc(&d_out_hyper, out_hyper_size*sizeof(thrust::complex<double>));

    // 4) Launch kernel
    int totalThreads = outRows*outCols*dual_size*dual_size; // 2*2*2*2=8
    int blockSize=128;
    int gridSize=(totalThreads + blockSize-1)/blockSize;

    MatrixHyperDualIndexGetKernel<<<gridSize, blockSize>>>(
        d_a_real, d_a_dual, d_a_hyper,
        rows, cols, dual_size,
        rowStart, rowEnd, colStart, colEnd,
        d_out_real, d_out_dual, d_out_hyper
    );
    cudaDeviceSynchronize();

    // 5) Copy results back
    auto h_out_real  = CopyToHost(d_out_real,  out_real_size);
    auto h_out_dual  = CopyToHost(d_out_dual,  out_dual_size);
    auto h_out_hyper = CopyToHost(d_out_hyper, out_hyper_size);

    // free device mem
    cudaFree(d_a_real);
    cudaFree(d_a_dual);
    cudaFree(d_a_hyper);
    cudaFree(d_out_real);
    cudaFree(d_out_dual);
    cudaFree(d_out_hyper);

    // 6) Compute the reference on host
    // We'll do nested loops:
    //   i in [rowStart..rowEnd-1]
    //   j in [colStart..colEnd-1]
    //   localRow = i - rowStart
    //   localCol = j - colStart
    // Then copy:
    //   out_real[localOff] = a_real[globalOff], etc.
    //   out_dual[localOff*dual_size + k] = a_dual[globalOff*dual_size + k]
    //   out_hyper[localOff*dual_size^2 + k*dual_size + l] = ...
    std::vector<thrust::complex<double>> ref_real(out_real_size);
    std::vector<thrust::complex<double>> ref_dual(out_dual_size);
    std::vector<thrust::complex<double>> ref_hyper(out_hyper_size);

    for (int i = rowStart; i < rowEnd; ++i) {
        for (int j = colStart; j < colEnd; ++j) {
            int localRow = i - rowStart; 
            int localCol = j - colStart;
            int outOff   = localRow*outCols + localCol; // for real

            // copy real
            ref_real[outOff] = a_real[i*cols + j];

            // copy dual
            for (int k = 0; k < dual_size; ++k) {
                int outDualOff = outOff*dual_size + k;
                ref_dual[outDualOff] = a_dual[(i*cols + j)*dual_size + k];
            }

            // copy hyper
            for (int k = 0; k < dual_size; ++k) {
                for (int l = 0; l < dual_size; ++l) {
                    int outHyperOff = outOff*dual_size*dual_size + k*dual_size + l;
                    ref_hyper[outHyperOff] = a_hyper[(i*cols + j)*dual_size*dual_size + k*dual_size + l];
                }
            }
        }
    }

    // 7) Compare
    // real part
    for (int i = 0; i < out_real_size; ++i) {
        EXPECT_EQ(h_out_real[i], ref_real[i]) 
            << "Mismatch in real slice at i=" << i;
    }

    // dual part
    for (int i = 0; i < out_dual_size; ++i) {
        EXPECT_EQ(h_out_dual[i], ref_dual[i]) 
            << "Mismatch in dual slice at i=" << i;
    }

    // hyper part
    for (int i = 0; i < out_hyper_size; ++i) {
        EXPECT_EQ(h_out_hyper[i], ref_hyper[i]) 
            << "Mismatch in hyper slice at i=" << i;
    }
}



// ---------------------------------------------------------------------------
// Test: Copy a 2x2 sub-slice from a 3x4 source (rows=3,cols=4) into a 5x5 dest
// ---------------------------------------------------------------------------
TEST(MatrixHyperDualIndexPutTest, SubSlice3x4_Into5x5)
{
    using T = double;

    // 1) Setup source dimensions
    int srcRows = 3;
    int srcCols = 4;
    int dual_size = 2;

    // We define a sub-slice in the source: row range [1..3), col range [1..3)
    int rowStartSrc = 1, rowEndSrc = 3;  // => subRows=2
    int colStartSrc = 1, colEndSrc = 3;  // => subCols=2

    // => sub-slice shape = 2x2

    // 2) Setup destination dimensions
    int dstRows = 5;
    int dstCols = 5;

    // 3) Compute array sizes
    // source real => srcRows*srcCols=3*4=12
    // source dual => 12*dual_size=24
    // source hyper => 12*(dual_size^2)=48
    int srcTotalReal  = srcRows*srcCols; 
    int srcTotalDual  = srcRows*srcCols*dual_size;
    int srcTotalHyper = srcRows*srcCols*dual_size*dual_size;

    // destination real => dstRows*dstCols=5*5=25
    // destination dual => 25*dual_size=50
    // destination hyper => 25*(dual_size^2)=100
    int dstTotalReal  = dstRows*dstCols; 
    int dstTotalDual  = dstTotalReal*dual_size;  // 25*2=50
    int dstTotalHyper = dstTotalReal*dual_size*dual_size; //25*2*2=100

    // 4) Create host arrays for source
    std::vector<thrust::complex<T>> src_real(srcTotalReal);
    std::vector<thrust::complex<T>> src_dual(srcTotalDual);
    std::vector<thrust::complex<T>> src_hyper(srcTotalHyper);

    // Fill them with some pattern:
    // a) src_real => row i => ( (i+1)*10 + j ) as real
    for (int i = 0; i < srcRows; ++i) {
        for (int j = 0; j < srcCols; ++j) {
            int off = i*srcCols + j;
            double val = double((i+1)*10 + j); 
            src_real[off] = thrust::complex<T>(val, 0.0);
        }
    }
    // b) src_dual => length=24 => fill with (0.1+ idx, -0.2-idx)
    for (int i = 0; i < srcTotalDual; ++i) {
        double re = 0.1 + double(i);
        double im = -0.2 - double(i);
        src_dual[i] = thrust::complex<T>(re, im);
    }
    // c) src_hyper => length=48 => fill with (0.01*i, 0.02*i)
    for (int i = 0; i < srcTotalHyper; ++i) {
        double re = 0.01* i;
        double im = 0.02* i;
        src_hyper[i] = thrust::complex<T>(re, im);
    }

    // 5) Create host arrays for destination
    // We'll fill them with a different pattern so we can confirm that only 
    // the sub-slice region is changed.
    std::vector<thrust::complex<T>> dst_real(dstTotalReal),
                                    dst_dual(dstTotalDual),
                                    dst_hyper(dstTotalHyper);
    // e.g. set everything to (999,999) so we can detect overwrites
    for (int i = 0; i < dstTotalReal; ++i) {
        dst_real[i] = thrust::complex<T>(999.0, 999.0);
    }
    for (int i = 0; i < dstTotalDual; ++i) {
        dst_dual[i] = thrust::complex<T>(999.0, 999.0);
    }
    for (int i = 0; i < dstTotalHyper; ++i) {
        dst_hyper[i] = thrust::complex<T>(999.0, 999.0);
    }

    // 6) Allocate device memory & copy
    thrust::complex<T>* d_src_real=nullptr;
    thrust::complex<T>* d_src_dual=nullptr;
    thrust::complex<T>* d_src_hyper=nullptr;

    thrust::complex<T>* d_dst_real=nullptr;
    thrust::complex<T>* d_dst_dual=nullptr;
    thrust::complex<T>* d_dst_hyper=nullptr;

    AllocateAndCopy(src_real,  &d_src_real);
    AllocateAndCopy(src_dual,  &d_src_dual);
    AllocateAndCopy(src_hyper, &d_src_hyper);

    // copy the "dst_" arrays to device
    size_t dstRealBytes  = dstTotalReal*sizeof(thrust::complex<T>);
    size_t dstDualBytes  = dstTotalDual*sizeof(thrust::complex<T>);
    size_t dstHyperBytes = dstTotalHyper*sizeof(thrust::complex<T>);

    cudaMalloc(&d_dst_real,  dstRealBytes);
    cudaMalloc(&d_dst_dual,  dstDualBytes);
    cudaMalloc(&d_dst_hyper, dstHyperBytes);

    cudaMemcpy(d_dst_real,  dst_real.data(),  dstRealBytes,  cudaMemcpyHostToDevice);
    cudaMemcpy(d_dst_dual,  dst_dual.data(),  dstDualBytes,  cudaMemcpyHostToDevice);
    cudaMemcpy(d_dst_hyper, dst_hyper.data(), dstHyperBytes, cudaMemcpyHostToDevice);

    // 7) Launch the kernel
    int subRows = rowEndSrc - rowStartSrc; // 2
    int subCols = colEndSrc - colStartSrc; // 2
    int totalHyper = subRows * subCols * dual_size * dual_size; // 2*2*2*2=8

    int blockSize = 128;
    int gridSize  = (totalHyper + blockSize - 1)/blockSize;

    int rowStartDst = 1; 
    int colStartDst = 2;  // we place the 2x2 sub-slice in the dest at (1,2)

    MatrixHyperDualIndexPutKernel<<<gridSize, blockSize>>>(
        // source
        d_src_real, d_src_dual, d_src_hyper,
        srcRows, srcCols, dual_size,
        rowStartSrc, rowEndSrc,
        colStartSrc, colEndSrc,
        // destination
        d_dst_real, d_dst_dual, d_dst_hyper,
        dstRows, dstCols,
        rowStartDst, colStartDst
    );
    cudaDeviceSynchronize();

    // 8) Copy the destination back to host
    cudaMemcpy(dst_real.data(),  d_dst_real,  dstRealBytes,  cudaMemcpyDeviceToHost);
    cudaMemcpy(dst_dual.data(),  d_dst_dual,  dstDualBytes,  cudaMemcpyDeviceToHost);
    cudaMemcpy(dst_hyper.data(), d_dst_hyper, dstHyperBytes, cudaMemcpyDeviceToHost);

    // free device memory
    cudaFree(d_src_real);
    cudaFree(d_src_dual);
    cudaFree(d_src_hyper);
    cudaFree(d_dst_real);
    cudaFree(d_dst_dual);
    cudaFree(d_dst_hyper);

    // 9) Build a host reference
    // We'll replicate the logic: for each localRow in [0..subRows-1],
    // localCol in [0..subCols-1], copy the real, dual, hyper from src to dst
    //  at (rowStartDst+localRow, colStartDst+localCol).
    // We'll create reference arrays same size as the dest,
    // then fill with 999,999, and override sub-slice region from source.

    std::vector<thrust::complex<T>> ref_real(dstTotalReal,  thrust::complex<T>(999,999));
    std::vector<thrust::complex<T>> ref_dual(dstTotalDual,  thrust::complex<T>(999,999));
    std::vector<thrust::complex<T>> ref_hyper(dstTotalHyper,thrust::complex<T>(999,999));

    // copy the sub-slice from source
    for (int localRow=0; localRow < subRows; ++localRow) {
        for (int localCol=0; localCol < subCols; ++localCol) {
            int srcRow = rowStartSrc + localRow; 
            int srcCol = colStartSrc + localCol; 
            int srcOff = srcRow*srcCols + srcCol;

            int dstRow = rowStartDst + localRow; 
            int dstCol = colStartDst + localCol; 
            int dstOff = dstRow*dstCols + dstCol;

            // copy real
            ref_real[dstOff] = src_real[srcOff];

            // copy dual
            for (int k = 0; k < dual_size; ++k) {
                int srcDualOff = srcOff*dual_size + k;
                int dstDualOff = dstOff*dual_size + k;
                ref_dual[dstDualOff] = src_dual[srcDualOff];
            }

            // copy hyper
            for (int k = 0; k < dual_size; ++k) {
                for (int l=0; l< dual_size; ++l) {
                    int srcHyperOff = srcOff*dual_size*dual_size + k*dual_size + l;
                    int dstHyperOff = dstOff*dual_size*dual_size + k*dual_size + l;
                    ref_hyper[dstHyperOff] = src_hyper[srcHyperOff];
                }
            }
        }
    }

    // 10) Compare
    // Compare ref_real vs. dst_real, etc.  Everything outside sub-slice 
    // should remain (999,999).  Inside sub-slice should match the src sub-slice.
    for (int i = 0; i < dstTotalReal; ++i) {
        EXPECT_EQ(dst_real[i], ref_real[i]) << "Mismatch real at i=" << i;
    }
    for (int i = 0; i < dstTotalDual; ++i) {
        EXPECT_EQ(dst_dual[i], ref_dual[i]) << "Mismatch dual at i=" << i;
    }
    for (int i = 0; i < dstTotalHyper; ++i) {
        EXPECT_EQ(dst_hyper[i], ref_hyper[i]) << "Mismatch hyper at i=" << i;
    }
}





// ---------------------------------------------------------------------------
// Test A: (rows=1, cols=3, dim=0) => we remove the row dimension
//         so result shape => real=3, dual=3*dual_size, hyper=3*dual_size^2
// ---------------------------------------------------------------------------
TEST(MatrixHyperDualSqueezeTest, Rows1_Cols3_Dim0)
{
    int rows=1, cols=3, dual_size=2, dim=0; 
    // => total real=1*3=3, dual=3*2=6, hyper=3*2*2=12

    // Fill host arrays
    std::vector<thrust::complex<double>> a_real(rows*cols), 
                                         a_dual(rows*cols*dual_size),
                                         a_hyper(rows*cols*dual_size*dual_size);
    for (int i=0; i<(int)a_real.size(); ++i) {
        a_real[i] =thrust::complex<double>(double(i+1), 0.0 );
    }
    for (int i=0; i<(int)a_dual.size(); ++i) {
        a_dual[i] = thrust::complex<double>( double(i+0.1), double(i+0.2) );
    }
    for (int i=0; i<(int)a_hyper.size(); ++i) {
        a_hyper[i] = thrust::complex<double>( 0.01* i, 0.02*i );
    }

    // Device copies
    thrust::complex<double>* d_a_real=nullptr,* d_a_dual=nullptr,* d_a_hyper=nullptr;
    AllocateAndCopy(a_real,  &d_a_real);
    AllocateAndCopy(a_dual,  &d_a_dual);
    AllocateAndCopy(a_hyper, &d_a_hyper);

    // We'll store the result in shape => real=3, dual=3*2=6, hyper=3*2*2=12
    std::vector<thrust::complex<double>> hostR_real(cols), 
                                         hostR_dual(cols*dual_size),
                                         hostR_hyper(cols*dual_size*dual_size);
    // device
    thrust::complex<double>* d_r_real=nullptr,* d_r_dual=nullptr,* d_r_hyper=nullptr;
    cudaMalloc(&d_r_real,   cols*sizeof(thrust::complex<double>));
    cudaMalloc(&d_r_dual,   cols*dual_size*sizeof(thrust::complex<double>));
    cudaMalloc(&d_r_hyper,  cols*dual_size*dual_size*sizeof(thrust::complex<double>));

    // Launch kernel
    int totalHyper = rows*cols*dual_size*dual_size; // 1*3*(2*2)=12
    int blockSize=128;
    int gridSize=(totalHyper+blockSize-1)/blockSize;
    MatrixHyperDualSqueezeKernel<<<gridSize, blockSize>>>(
        d_a_real, d_a_dual, d_a_hyper,
        rows, cols, dual_size,
        dim,
        d_r_real, d_r_dual, d_r_hyper
    );
    cudaDeviceSynchronize();

    // Copy back
    auto r_real  = CopyToHost(d_r_real,  cols);
    auto r_dual  = CopyToHost(d_r_dual,  cols*dual_size);
    auto r_hyper = CopyToHost(d_r_hyper, cols*dual_size*dual_size);

    // Build reference
    // Because rows=1 => i=0 is the only row
    // => result index = j
    // real => if (k,l)==(0,0), result_real[j] = a_real[j]
    // dual => if l==0 => result_dual[j*dual_size+k] = a_dual[j*dual_size+k]
    // hyper=> result_hyper[ j*(dsize^2) + k*dsize + l ] = ...
    std::vector<thrust::complex<double>> ref_real(cols),
                                         ref_dual(cols*dual_size),
                                         ref_hyper(cols*dual_size*dual_size);
    // We'll just do direct copy because i=0
    for (int j0=0; j0<cols; ++j0) {
        // real
        ref_real[j0] = a_real[j0]; // only (k,l)==(0,0) but effectively storing last
        // dual
        for (int k0=0; k0<dual_size; ++k0) {
            ref_dual[j0*dual_size + k0] = a_dual[j0*dual_size + k0];
        }
        // hyper
        for (int k0=0; k0<dual_size; ++k0){
          for (int l0=0; l0<dual_size; ++l0){
             ref_hyper[j0*dual_size*dual_size + k0*dual_size + l0] 
                = a_hyper[j0*dual_size*dual_size + k0*dual_size + l0];
          }
        }
    }

    // Compare
    for (int i=0; i<cols; ++i){
       EXPECT_EQ(r_real[i], ref_real[i]) << "Mismatch real at col="<<i<<" rows=1 case";
    }
    for (int i=0; i< (int)ref_dual.size(); ++i){
       EXPECT_EQ(r_dual[i], ref_dual[i]) << "Mismatch dual at i="<<i<<" rows=1 case";
    }
    for (int i=0; i< (int)ref_hyper.size(); ++i){
       EXPECT_EQ(r_hyper[i], ref_hyper[i]) << "Mismatch hyper at i="<<i<<" rows=1 case";
    }

    cudaFree(d_a_real);  cudaFree(d_a_dual);  cudaFree(d_a_hyper);
    cudaFree(d_r_real);  cudaFree(d_r_dual);  cudaFree(d_r_hyper);
}

// ---------------------------------------------------------------------------
// Test B: (rows=4, cols=1, dim=1) => we remove the column dimension
//         result shape => real=4, dual=4*2, hyper=4*(2*2)
// ---------------------------------------------------------------------------
TEST(MatrixHyperDualSqueezeTest, Rows4_Cols1_Dim1)
{
    int rows=4, cols=1, dual_size=2, dim=1;
    // total real=4*1=4, dual=4*1*2=8, hyper=4*1*(2*2)=16

    std::vector<thrust::complex<double>> a_real(rows*cols),
                                         a_dual(rows*cols*dual_size),
                                         a_hyper(rows*cols*dual_size*dual_size);
    for (int i=0; i<(int)a_real.size(); ++i){
        a_real[i] = thrust::complex<double>(double(i+10), 0.0 );
    }
    for (int i=0; i<(int)a_dual.size(); ++i){
        double re= i+0.1, im= i+0.2;
        a_dual[i] = thrust::complex<double>(re, im);
    }
    for (int i=0; i<(int)a_hyper.size(); ++i){
        double re=0.01*i, im=0.02*i;
        a_hyper[i] = thrust::complex<double>(re, im);
    }

    thrust::complex<double>* d_a_real=nullptr,* d_a_dual=nullptr,* d_a_hyper=nullptr;
    AllocateAndCopy(a_real,  &d_a_real);
    AllocateAndCopy(a_dual,  &d_a_dual);
    AllocateAndCopy(a_hyper, &d_a_hyper);

    // result => real=rows=4, dual=4*2=8, hyper=4*(2*2)=16
    thrust::complex<double>* d_r_real=nullptr, *d_r_dual=nullptr, *d_r_hyper=nullptr;
    cudaMalloc(&d_r_real,  rows*sizeof(thrust::complex<double>));
    cudaMalloc(&d_r_dual,  rows*dual_size*sizeof(thrust::complex<double>));
    cudaMalloc(&d_r_hyper, rows*dual_size*dual_size*sizeof(thrust::complex<double>));
    cudaMemset(d_r_real,  0, rows*sizeof(thrust::complex<double>));
    cudaMemset(d_r_dual,  0, rows*dual_size*sizeof(thrust::complex<double>));
    cudaMemset(d_r_hyper, 0, rows*dual_size*dual_size*sizeof(thrust::complex<double>));

    int totalHyper=rows*cols*dual_size*dual_size; //4*1*(2*2)=16
    int blockSize=128;
    int gridSize=(totalHyper+blockSize-1)/blockSize;

    MatrixHyperDualSqueezeKernel<<<gridSize,blockSize>>>(
        d_a_real, d_a_dual, d_a_hyper,
        rows, cols, dual_size,
        dim,
        d_r_real, d_r_dual, d_r_hyper
    );
    cudaDeviceSynchronize();

    auto r_real  = CopyToHost(d_r_real,  rows);
    auto r_dual  = CopyToHost(d_r_dual,  rows*dual_size);
    auto r_hyper = CopyToHost(d_r_hyper, rows*dual_size*dual_size);

    // Build ref
    // Because cols=1 => j=0 is the only column
    // => result index is i
    std::vector<thrust::complex<double>> ref_real(rows),
                                         ref_dual(rows*dual_size),
                                         ref_hyper(rows*dual_size*dual_size);
    for (int i0=0; i0<rows; ++i0){
       ref_real[i0] = a_real[i0]; // j=0
       for (int k0=0; k0<dual_size; ++k0){
         ref_dual[i0*dual_size + k0] = a_dual[i0*dual_size + k0];
       }
       for (int k0=0; k0<dual_size; ++k0){
         for (int l0=0; l0<dual_size; ++l0){
           int off= (i0*cols + 0)*dual_size*dual_size + k0*dual_size + l0;
           ref_hyper[i0*dual_size*dual_size + k0*dual_size + l0] = a_hyper[off];
         }
       }
    }

    for (int i=0; i<rows; ++i){
        EXPECT_EQ(r_real[i], ref_real[i])
            << "Mismatch real i="<<i<<" for (rows=4,cols=1,dim=1)";
    }
    for (int i=0; i<(int)ref_dual.size(); ++i){
        EXPECT_EQ(r_dual[i], ref_dual[i])
            << "Mismatch dual i="<<i;
    }
    for (int i=0; i<(int)ref_hyper.size(); ++i){
        EXPECT_EQ(r_hyper[i], ref_hyper[i])
            << "Mismatch hyper i="<<i;
    }

    cudaFree(d_a_real); cudaFree(d_a_dual); cudaFree(d_a_hyper);
    cudaFree(d_r_real); cudaFree(d_r_dual); cudaFree(d_r_hyper);
}

// ---------------------------------------------------------------------------
// Test C: If dimension isn't 1 in the chosen 'dim', we do "no effect" => 
// no shape change. We'll just do one example: rows=2, cols=3, dim=1 but cols=3
// so we expect no writing. We'll store the result in separate arrays 
// but verify they're unchanged from an initial "garbage" pattern.
// ---------------------------------------------------------------------------
TEST(MatrixHyperDualSqueezeTest, NoEffectDim1WhenColsNot1)
{
    int rows=2, cols=3, dual_size=2, dim=1;
    // Because cols !=1, we do nothing.

    // Build a small input
    int totalReal= rows*cols;  //2*3=6
    int totalDual= totalReal*dual_size; //6*2=12
    int totalHyper= totalReal*dual_size*dual_size; //6*(2*2)=24
    std::vector<thrust::complex<double>> a_real(totalReal), 
                                         a_dual(totalDual),
                                         a_hyper(totalHyper);
    for (int i=0; i<totalReal; ++i){
        a_real[i]=thrust::complex<double>(double(i+1),0);
    }
    for (int i=0; i<totalDual; ++i){
        double re=i+0.1, im=i+0.2;
        a_dual[i]=thrust::complex<double>(re,im);
    }
    for (int i=0; i<totalHyper; ++i){
        double re=0.01*i, im=0.02*i;
        a_hyper[i]=thrust::complex<double>(re,im);
    }

    // We'll allocate some result arrays but fill them with "garbage" 
    // to detect if anything changes.
    // We'll guess the "squeezed" shape might be something else, but we actually 
    // don't expect the code to do anything at all.
    std::vector<thrust::complex<double>> r_realH(5, {999,999}),
                                         r_dualH(10,{999,999}),
                                         r_hyperH(20,{999,999});
    auto r_realC= r_realH, r_dualC=r_dualH, r_hyperC=r_hyperH; 
      // copy for compare after kernel

    // Device
    thrust::complex<double>* d_a_real=nullptr,* d_a_dual=nullptr,* d_a_hyper=nullptr;
    AllocateAndCopy(a_real, &d_a_real);
    AllocateAndCopy(a_dual, &d_a_dual);
    AllocateAndCopy(a_hyper,&d_a_hyper);

    // For results, let's allocate the same shape we put in host 
    // but the code won't do anything if the dimension isn't 1
    thrust::complex<double>* d_r_real=nullptr,* d_r_dual=nullptr,* d_r_hyper=nullptr;
    cudaMalloc(&d_r_real,   r_realH.size()*sizeof(thrust::complex<double>));
    cudaMalloc(&d_r_dual,   r_dualH.size()*sizeof(thrust::complex<double>));
    cudaMalloc(&d_r_hyper,  r_hyperH.size()*sizeof(thrust::complex<double>));
    cudaMemcpy(d_r_real,   r_realH.data(),  r_realH.size()*sizeof(thrust::complex<double>), cudaMemcpyHostToDevice);
    cudaMemcpy(d_r_dual,   r_dualH.data(),  r_dualH.size()*sizeof(thrust::complex<double>), cudaMemcpyHostToDevice);
    cudaMemcpy(d_r_hyper,  r_hyperH.data(), r_hyperH.size()*sizeof(thrust::complex<double>),cudaMemcpyHostToDevice);

    // Launch 
    totalHyper= rows*cols*dual_size*dual_size; //24
    int blockSize=128;
    int gridSize=(totalHyper+blockSize-1)/blockSize;
    MatrixHyperDualSqueezeKernel<<<gridSize, blockSize>>>(
        d_a_real, d_a_dual, d_a_hyper,
        rows, cols, dual_size,
        dim, 
        d_r_real, d_r_dual, d_r_hyper
    );
    cudaDeviceSynchronize();

    // get results
    auto r_realOut= CopyToHost(d_r_real,  r_realH.size());
    auto r_dualOut= CopyToHost(d_r_dual,  r_dualH.size());
    auto r_hyperOut=CopyToHost(d_r_hyper, r_hyperH.size());

    // Compare => expect no changes
    for (size_t i=0; i<r_realH.size(); ++i){
       EXPECT_EQ(r_realOut[i], r_realC[i]) 
           << "Expected no change in real i="<<i<<" because squeeze not triggered.";
    }
    for (size_t i=0; i<r_dualH.size(); ++i){
       EXPECT_EQ(r_dualOut[i], r_dualC[i])
           << "Expected no change in dual i="<<i<<" because squeeze not triggered.";
    }
    for (size_t i=0; i<r_hyperH.size(); ++i){
       EXPECT_EQ(r_hyperOut[i], r_hyperC[i])
           << "Expected no change in hyper i="<<i<<" because squeeze not triggered.";
    }

    cudaFree(d_a_real); cudaFree(d_a_dual); cudaFree(d_a_hyper);
    cudaFree(d_r_real); cudaFree(d_r_dual); cudaFree(d_r_hyper);
}



// ---------------------------------------------------------------------------
// Google Test for MatrixHyperDualSigncondKernel
// ---------------------------------------------------------------------------
TEST(MatrixHyperDualSigncondTest, Basic_2x2_dual2)
{
    // We pick: rows=2, cols=2, dual_size=2
    // => a_real, b_real => 4 elements each
    // => a_dual, b_dual => 4 * 2 = 8 elements each, 
    // => a_hyper, b_hyper => 4 * 2 * 2 = 16 elements each
    // We'll only use b_real for sign, ignoring b_dual/b_hyper if we want.

    // 1) Setup
    int rows=2, cols=2, dual_size=2;
    int totalReal  = rows*cols;             // 4
    int totalDual  = totalReal*dual_size;   // 8
    int totalHyper = totalReal*dual_size*dual_size; // 16
    double tol=1.0e-6;

    // 2) Host arrays: a_real, a_dual, a_hyper, b_real
    std::vector<thrust::complex<double>> a_real(totalReal),
                                         a_dual(totalDual),
                                         a_hyper(totalHyper),
                                         b_real(totalReal);
    // Fill a with patterns
    // a_real => e.g. [ -2, -0.0001, 1, 3 ] to test sign logic 
    // (some negative, some small near zero, some positive)
    a_real[0] = thrust::complex<double>(-2.0,  0.0);
    a_real[1] = thrust::complex<double>(-0.0001, 0.0); // close to zero
    a_real[2] = thrust::complex<double>(1.0,   0.0);
    a_real[3] = thrust::complex<double>(3.0,   0.0);

    // a_dual => fill with arbitrary patterns
    for (int i=0; i<totalDual; ++i) {
        double re = i+0.1, im = i+0.2;
        a_dual[i] = thrust::complex<double>(re, im);
    }
    // a_hyper => fill with distinct pattern
    for (int i=0; i<totalHyper; ++i) {
        double re=0.01*(i), im=0.02*(i);
        a_hyper[i] = thrust::complex<double>(re, im);
    }

    // b_real => e.g. [ 0.00005, -4, 10, -1 ]
    // We'll see how that toggles sign logic
    b_real[0] = thrust::complex<double>(0.00005, 0.0); 
    b_real[1] = thrust::complex<double>(-4.0,    0.0);
    b_real[2] = thrust::complex<double>(10.0,    0.0);
    b_real[3] = thrust::complex<double>(-1.0,    0.0);

    // 3) Device memory
    thrust::complex<double>* d_a_real=nullptr,* d_a_dual=nullptr,* d_a_hyper=nullptr;
    thrust::complex<double>* d_b_real=nullptr;
    AllocateAndCopy(a_real,   &d_a_real);
    AllocateAndCopy(a_dual,   &d_a_dual);
    AllocateAndCopy(a_hyper,  &d_a_hyper);
    AllocateAndCopy(b_real,   &d_b_real);

    // 4) Allocate device memory for result
    std::vector<thrust::complex<double>> dummy_real(totalReal), 
                                         dummy_dual(totalDual), 
                                         dummy_hyper(totalHyper);
    thrust::complex<double>* d_result_real=nullptr,
                            * d_result_dual=nullptr,
                            * d_result_hyper=nullptr;
    cudaMalloc(&d_result_real,  totalReal*sizeof(thrust::complex<double>));
    cudaMalloc(&d_result_dual,  totalDual*sizeof(thrust::complex<double>));
    cudaMalloc(&d_result_hyper, totalHyper*sizeof(thrust::complex<double>));

    // 5) Launch kernel
    int totalThreads= rows*cols * dual_size*dual_size; // 2*2*(2*2)=8
    int blockSize=128;
    int gridSize=(totalThreads+blockSize-1)/blockSize;

    MatrixHyperDualSigncondKernel<<<gridSize, blockSize>>>(
        d_a_real, d_a_dual, d_a_hyper,
        d_b_real,
        rows, cols, dual_size,
        d_result_real, d_result_dual, d_result_hyper,
        tol
    );
    cudaDeviceSynchronize();

    // 6) Copy result back
    auto res_real  = CopyToHost(d_result_real,  totalReal);
    auto res_dual  = CopyToHost(d_result_dual,  totalDual);
    auto res_hyper = CopyToHost(d_result_hyper, totalHyper);

    // free device
    cudaFree(d_a_real);    cudaFree(d_a_dual);    cudaFree(d_a_hyper);
    cudaFree(d_b_real);
    cudaFree(d_result_real);
    cudaFree(d_result_dual);
    cudaFree(d_result_hyper);

    // 7) Build host reference
    // For each (i,j):
    //   a_sign = sign( a_real(i,j).real() ) with tol
    //   b_sign = sign( b_real(i,j).real() ) with tol
    //   Then real => only once if (k,l)=(0,0)
    //   dual => if (l=0)
    //   hyper => for all (k,l)
    // sign logic => if (b_sign>=0), result= (a_sign>=0)? +a : -a
    //               else,           result= (a_sign>=0)? -a : +a

    auto ref_real  = std::vector<thrust::complex<double>>(totalReal);
    auto ref_dual  = std::vector<thrust::complex<double>>(totalDual);
    auto ref_hyper = std::vector<thrust::complex<double>>(totalHyper);

    for (int i0=0; i0<rows; ++i0) {
      for (int j0=0; j0<cols; ++j0) {
        int offReal = i0*cols + j0;
        double aVal= a_real[offReal].real();
        double bVal= b_real[offReal].real();
        int a_sign= (fabs(aVal)>=tol)? ((aVal>=0)? +1 : -1) : +1;
        int b_sign= (fabs(bVal)>=tol)? ((bVal>=0)? +1 : -1) : +1;

        // real => if (k,l)==(0,0)
        // let's do it once
        thrust::complex<double> A_r= a_real[offReal];
        thrust::complex<double> r_r;
        if (b_sign>=0) {
          r_r= (a_sign>=0)? A_r : -A_r;
        } else {
          r_r= (a_sign>=0)? -A_r : A_r;
        }
        // store
        ref_real[offReal]= r_r;

        // dual => for k in [0..1], l=0
        for (int k0=0; k0<dual_size; ++k0){
          int offDual= offReal*dual_size + k0;
          auto A_du= a_dual[offDual];
          if (b_sign>=0) {
            ref_dual[offDual]= (a_sign>=0)? A_du : -A_du;
          } else {
            ref_dual[offDual]= (a_sign>=0)? -A_du : A_du;
          }
        }

        // hyper => for k0,l0 in [0..1]
        for (int k0=0; k0<dual_size; ++k0){
          for (int l0=0; l0<dual_size; ++l0){
            int offHyp= offReal*dual_size*dual_size + k0*dual_size + l0;
            auto A_hy= a_hyper[offHyp];
            if (b_sign>=0) {
              ref_hyper[offHyp]= (a_sign>=0)? A_hy : -A_hy;
            } else {
              ref_hyper[offHyp]= (a_sign>=0)? -A_hy : A_hy;
            }
          }
        }
      }
    }

    // 8) Compare
    // real => 4
    for (int i=0; i< totalReal; ++i) {
      EXPECT_EQ(res_real[i], ref_real[i])
          << "Mismatch real at i="<<i;
    }
    // dual => 8
    for (int i=0; i< totalDual; ++i) {
      EXPECT_EQ(res_dual[i], ref_dual[i])
          << "Mismatch dual at i="<<i;
    }
    // hyper =>16
    for (int i=0; i< totalHyper; ++i) {
      EXPECT_EQ(res_hyper[i], ref_hyper[i])
          << "Mismatch hyper at i="<<i;
    }
}


// Add more tests for IndexGet, IndexPut, ElementwiseMultiply, Square, Pow, and Sqrt similarly.
// Main entry point for Google Test
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}