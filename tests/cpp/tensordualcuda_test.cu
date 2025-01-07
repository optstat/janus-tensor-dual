#include <gtest/gtest.h>
#include <torch/torch.h>
#include <random>
#include "../../src/cpp/cudatensordense.cu"
#include "../../src/cpp/cudatensorsparse.cu"
#include <thrust/host_vector.h>
using namespace janus;

// Test internal memory allocation constructor
TEST(VectorDenseComplexTest, ConstructorInternalMemory) {
    int batch_size = 2;
    int vector_size = 3;
    VectorDenseCuda<thrust::complex<float>> vec(batch_size, vector_size);

    ASSERT_EQ(vec.batchSize(), batch_size);
    ASSERT_EQ(vec.size(), vector_size);
}

// Test external memory constructor
TEST(VectorDenseComplexTest, ConstructorExternalMemory) {
    int batch_size = 2;
    int vector_size = 3;
    thrust::complex<float>* device_data;
    cudaMalloc(&device_data, batch_size * vector_size * sizeof(thrust::complex<float>));

    VectorDenseCuda<float> vec(batch_size, vector_size, device_data);

    ASSERT_EQ(vec.batchSize(), batch_size);
    ASSERT_EQ(vec.size(), vector_size);

    cudaFree(device_data);
}

// Test initialization with host data
TEST(VectorDenseComplexTest, InitializeData) {
    int batch_size = 2;
    int vector_size = 3;
    std::vector<thrust::complex<float>> host_data = {
        {1.0, 1.0}, {2.0, 2.0}, {3.0, 3.0}, {4.0, 4.0}, {5.0, 5.0}, {6.0, 6.0}};

    VectorDenseCuda<float> vec(batch_size, vector_size);

    vec.initialize(host_data.data(), host_data.size());

    thrust::host_vector<thrust::complex<float>> device_to_host(host_data.size());
    cudaMemcpy(device_to_host.data(), vec.data(),
               host_data.size() * sizeof(thrust::complex<float>), cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < host_data.size(); ++i) {
        EXPECT_FLOAT_EQ(device_to_host[i].real(), host_data[i].real());
        EXPECT_FLOAT_EQ(device_to_host[i].imag(), host_data[i].imag());
    }
}

// Test elementwise addition
TEST(VectorDenseComplexTest, ElementwiseAdd) {
    int batch_size = 2;
    int vector_size = 3;
    std::vector<thrust::complex<float>> host_data = {
        {1.0, 1.0}, {2.0, 2.0}, {3.0, 3.0}, {4.0, 4.0}, {5.0, 5.0}, {6.0, 6.0}};

    VectorDenseCuda<float> vec1(batch_size, vector_size);
    VectorDenseCuda<float> vec2(batch_size, vector_size);

    vec1.initialize(host_data.data(), host_data.size());
    vec2.initialize(host_data.data(), host_data.size());

    VectorDenseCuda<float> result = vec1.elementwiseAdd(vec2);

    thrust::host_vector<thrust::complex<float>> result_host(batch_size * vector_size);
    cudaMemcpy(result_host.data(), result.data(),
               batch_size * vector_size * sizeof(thrust::complex<float>), cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < host_data.size(); ++i) {
        EXPECT_FLOAT_EQ(result_host[i].real(), host_data[i].real() + host_data[i].real());
        EXPECT_FLOAT_EQ(result_host[i].imag(), host_data[i].imag() + host_data[i].imag());
    }
}

// Test elementwise multiplication
TEST(VectorDenseComplexTest, ElementwiseMultiply) {
    int batch_size = 2;
    int vector_size = 3;
    std::vector<thrust::complex<float>> host_data = {
        {1.0, 1.0}, {2.0, 2.0}, {3.0, 3.0}, {4.0, 4.0}, {5.0, 5.0}, {6.0, 6.0}};

    VectorDenseCuda<float> vec1(batch_size, vector_size);
    VectorDenseCuda<float> vec2(batch_size, vector_size);

    vec1.initialize(host_data.data(), host_data.size());
    vec2.initialize(host_data.data(), host_data.size());

    auto result = vec1.elementwiseMultiply(vec2);

    thrust::host_vector<thrust::complex<float>> result_host(batch_size * vector_size);
    cudaMemcpy(result_host.data(), result.data(),
               batch_size * vector_size * sizeof(thrust::complex<float>), cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < host_data.size(); ++i) {
        auto expected = host_data[i] * host_data[i];
        EXPECT_FLOAT_EQ(result_host[i].real(), expected.real());
        EXPECT_FLOAT_EQ(result_host[i].imag(), expected.imag());
    }
}



__global__ void validateDataKernel(
    const thrust::complex<float>* real, const thrust::complex<float>* dual,
    thrust::complex<float>* real_out, thrust::complex<float>* dual_out,
    int real_size, int dual_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < real_size) {
        real_out[idx] = real[idx];
    }
    if (idx < dual_size) {
        dual_out[idx] = dual[idx];
    }
}

TEST(VectorDualDenseCudaTest, Initialize) {
    int batch_size = 2;
    int size = 5;
    int dual_dim = 3;

    // Allocate GPU memory for real and dual data
    size_t real_size = batch_size * size * sizeof(thrust::complex<float>);
    size_t dual_size = batch_size * size * dual_dim * sizeof(thrust::complex<float>);

    thrust::complex<float>* d_real;
    thrust::complex<float>* d_dual;
    cudaMalloc(&d_real, real_size);
    cudaMalloc(&d_dual, dual_size);

    // Initialize data on host
    std::vector<thrust::complex<float>> primal_data(batch_size * size, thrust::complex<float>(1.0f, 2.0f));
    std::vector<thrust::complex<float>> dual_data(batch_size * size * dual_dim, thrust::complex<float>(0.5f, 0.5f));

    // Copy data to GPU
    cudaMemcpy(d_real, primal_data.data(), real_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dual, dual_data.data(), dual_size, cudaMemcpyHostToDevice);

    // Create GPU object
    janus::VectorDualDenseCuda<float> vec(batch_size, size, dual_dim, d_real, d_dual);

    // Allocate temporary buffers to copy GPU data back to host
    thrust::complex<float>* d_real_out;
    thrust::complex<float>* d_dual_out;
    cudaMalloc(&d_real_out, real_size);
    cudaMalloc(&d_dual_out, dual_size);

    // Launch a kernel to verify that the data is accessible on the GPU
    int threadsPerBlock = 256;
    int blocksPerGrid = (batch_size * size + threadsPerBlock - 1) / threadsPerBlock;

    validateDataKernel<<<blocksPerGrid, threadsPerBlock>>>(
        vec.real(), vec.dual(), d_real_out, d_dual_out, batch_size * size, batch_size * size * dual_dim);
    cudaDeviceSynchronize();

    // Copy real data back to host for validation
    std::vector<thrust::complex<float>> host_primal_data(batch_size * size);
    cudaMemcpy(host_primal_data.data(), d_real_out, real_size, cudaMemcpyDeviceToHost);

    // Validate real data
    for (size_t i = 0; i < primal_data.size(); ++i) {
        EXPECT_FLOAT_EQ(host_primal_data[i].real(), primal_data[i].real());
        EXPECT_FLOAT_EQ(host_primal_data[i].imag(), primal_data[i].imag());
    }

    // Copy dual data back to host for validation
    std::vector<thrust::complex<float>> host_dual_data(batch_size * size * dual_dim);
    cudaMemcpy(host_dual_data.data(), d_dual_out, dual_size, cudaMemcpyDeviceToHost);

    // Validate dual data
    for (size_t i = 0; i < dual_data.size(); ++i) {
        EXPECT_FLOAT_EQ(host_dual_data[i].real(), dual_data[i].real());
        EXPECT_FLOAT_EQ(host_dual_data[i].imag(), dual_data[i].imag());
    }

    // Free GPU memory
    cudaFree(d_real);
    cudaFree(d_dual);
    cudaFree(d_real_out);
    cudaFree(d_dual_out);
}







// Function to generate a vector of random complex numbers
std::vector<thrust::complex<float>> generateRandomComplexVector(size_t size, float real_min, float real_max, float imag_min, float imag_max) {
    // Random number generator and distributions
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> real_dist(real_min, real_max);
    std::uniform_real_distribution<float> imag_dist(imag_min, imag_max);

    // Create the vector
    std::vector<thrust::complex<float>> result(size);

    // Populate the vector with random complex numbers
    for (size_t i = 0; i < size; ++i) {
        float real_part = real_dist(gen);
        float imag_part = imag_dist(gen);
        result[i] = thrust::complex<float>(real_part, imag_part);
    }

    return result;
}


TEST(VectorDualDenseCudaTest, ElementwiseAddRandomComplex) {
    const int batch_size = 2;
    const int size = 3;
    const int dual_dim = 2;

    // Generate random complex numbers for primal and dual data
    std::vector<thrust::complex<float>> primal_data1 = generateRandomComplexVector(batch_size * size, -1.0f, 1.0f, -1.0f, 1.0f);
    std::vector<thrust::complex<float>> primal_data2 = generateRandomComplexVector(batch_size * size, -1.0f, 1.0f, -1.0f, 1.0f);
    std::vector<thrust::complex<float>> dual_data1 = generateRandomComplexVector(batch_size * size * dual_dim, -1.0f, 1.0f, -1.0f, 1.0f);
    std::vector<thrust::complex<float>> dual_data2 = generateRandomComplexVector(batch_size * size * dual_dim, -1.0f, 1.0f, -1.0f, 1.0f);

    // Allocate GPU memory
    size_t real_size = batch_size * size * sizeof(thrust::complex<float>);
    size_t dual_size = batch_size * size * dual_dim * sizeof(thrust::complex<float>);
    thrust::complex<float>* d_real1;
    thrust::complex<float>* d_real2;
    thrust::complex<float>* d_dual1;
    thrust::complex<float>* d_dual2;
    thrust::complex<float>* d_real_result;
    thrust::complex<float>* d_dual_result;

    cudaMalloc(&d_real1, real_size);
    cudaMalloc(&d_real2, real_size);
    cudaMalloc(&d_dual1, dual_size);
    cudaMalloc(&d_dual2, dual_size);
    cudaMalloc(&d_real_result, real_size);
    cudaMalloc(&d_dual_result, dual_size);

    // Copy data to GPU
    cudaMemcpy(d_real1, primal_data1.data(), real_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_real2, primal_data2.data(), real_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dual1, dual_data1.data(), dual_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dual2, dual_data2.data(), dual_size, cudaMemcpyHostToDevice);

    // Create GPU objects
    janus::VectorDualDenseCuda<float> vec1(batch_size, size, dual_dim, d_real1, d_dual1);
    janus::VectorDualDenseCuda<float> vec2(batch_size, size, dual_dim, d_real2, d_dual2);
    janus::VectorDualDenseCuda<float> result(batch_size, size, dual_dim, d_real_result, d_dual_result);

    // Configure and launch the kernel for elementwise addition
    int threadsPerBlock = 256;
    int blocksPerGrid = (batch_size * size + threadsPerBlock - 1) / threadsPerBlock;

    elementwiseAddKernel<<<blocksPerGrid, threadsPerBlock>>>(vec1.real(), vec2.real(), 
                                                             vec1.dual(), vec2.dual(),
                                                             result.real(), result.dual(),
                                                             batch_size * size, batch_size * size * dual_dim);
    cudaDeviceSynchronize();

    // Copy result back to host for verification
    std::vector<thrust::complex<float>> host_primal_result(batch_size * size);
    std::vector<thrust::complex<float>> host_dual_result(batch_size * size * dual_dim);
    cudaMemcpy(host_primal_result.data(), d_real_result, real_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_dual_result.data(), d_dual_result, dual_size, cudaMemcpyDeviceToHost);

    // Verify the result
    for (size_t i = 0; i < primal_data1.size(); ++i) {
        EXPECT_EQ(host_primal_result[i], primal_data1[i] + primal_data2[i]);
    }
    for (size_t i = 0; i < dual_data1.size(); ++i) {
        EXPECT_EQ(host_dual_result[i], dual_data1[i] + dual_data2[i]);
    }

    // Free GPU memory
    cudaFree(d_real1);
    cudaFree(d_real2);
    cudaFree(d_dual1);
    cudaFree(d_dual2);
    cudaFree(d_real_result);
    cudaFree(d_dual_result);
}




template <typename T>
std::vector<thrust::complex<T>> generateRandomComplexVector(int size, T real_min, T real_max, T imag_min, T imag_max) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<T> real_dist(real_min, real_max);
    std::uniform_real_distribution<T> imag_dist(imag_min, imag_max);

    std::vector<thrust::complex<T>> vec(size);
    for (int i = 0; i < size; ++i) {
        vec[i] = thrust::complex<T>(real_dist(gen), imag_dist(gen));
    }
    return vec;
}

__global__ void validateHyperDualDataKernel(
    const thrust::complex<float>* real, const thrust::complex<float>* dual, const thrust::complex<float>* hyperdual,
    thrust::complex<float>* real_out, thrust::complex<float>* dual_out, thrust::complex<float>* hyperdual_out,
    int total_real_elements, int total_dual_elements, int total_hyperdual_elements) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < total_real_elements) {
        real_out[idx] = real[idx];
    }

    if (idx < total_dual_elements) {
        dual_out[idx] = dual[idx];
    }

    if (idx < total_hyperdual_elements) {
        hyperdual_out[idx] = hyperdual[idx];
    }
}

TEST(VectorHyperDualDenseCudaTest, Initialize) {
    int batch_size = 2;
    int real_size = 5;
    int dual_size = 3;

    // Allocate GPU memory for real, dual, and hyperdual data
    size_t real_mem_size = batch_size * real_size * sizeof(thrust::complex<float>);
    size_t dual_mem_size = batch_size * real_size * dual_size * sizeof(thrust::complex<float>);
    size_t hyperdual_mem_size = batch_size * real_size * dual_size * dual_size * sizeof(thrust::complex<float>);

    thrust::complex<float>* d_real;
    thrust::complex<float>* d_dual;
    thrust::complex<float>* d_hyperdual;
    cudaMalloc(&d_real, real_mem_size);
    cudaMalloc(&d_dual, dual_mem_size);
    cudaMalloc(&d_hyperdual, hyperdual_mem_size);

    // Initialize data on host
    std::vector<thrust::complex<float>> real_data(batch_size * real_size, thrust::complex<float>(1.0f, 2.0f));
    std::vector<thrust::complex<float>> dual_data(batch_size * real_size * dual_size, thrust::complex<float>(0.5f, 0.5f));
    std::vector<thrust::complex<float>> hyperdual_data(batch_size * real_size * dual_size * dual_size,
                                                       thrust::complex<float>(0.25f, 0.25f));

    // Copy data to GPU
    cudaMemcpy(d_real, real_data.data(), real_mem_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dual, dual_data.data(), dual_mem_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_hyperdual, hyperdual_data.data(), hyperdual_mem_size, cudaMemcpyHostToDevice);

    // Create GPU object
    janus::VectorHyperDualDenseCuda<float> vec(batch_size, real_size, dual_size, d_real, d_dual, d_hyperdual);

    // Allocate temporary buffers to copy GPU data back to host
    thrust::complex<float>* d_real_out;
    thrust::complex<float>* d_dual_out;
    thrust::complex<float>* d_hyperdual_out;
    cudaMalloc(&d_real_out, real_mem_size);
    cudaMalloc(&d_dual_out, dual_mem_size);
    cudaMalloc(&d_hyperdual_out, hyperdual_mem_size);

    // Launch a kernel to verify that the data is accessible on the GPU
    int threadsPerBlock = 256;
    int blocksPerGrid = (batch_size * real_size * dual_size * dual_size + threadsPerBlock - 1) / threadsPerBlock;

    validateHyperDualDataKernel<<<blocksPerGrid, threadsPerBlock>>>(
        vec.real(), vec.dual(), vec.hyperdual(),
        d_real_out, d_dual_out, d_hyperdual_out,
        batch_size * real_size, batch_size * real_size * dual_size, batch_size * real_size * dual_size * dual_size);
    cudaDeviceSynchronize();

    // Copy real data back to host for validation
    std::vector<thrust::complex<float>> host_real_data(batch_size * real_size);
    cudaMemcpy(host_real_data.data(), d_real_out, real_mem_size, cudaMemcpyDeviceToHost);

    // Validate real data
    for (size_t i = 0; i < real_data.size(); ++i) {
        EXPECT_FLOAT_EQ(host_real_data[i].real(), real_data[i].real());
        EXPECT_FLOAT_EQ(host_real_data[i].imag(), real_data[i].imag());
    }

    // Copy dual data back to host for validation
    std::vector<thrust::complex<float>> host_dual_data(batch_size * real_size * dual_size);
    cudaMemcpy(host_dual_data.data(), d_dual_out, dual_mem_size, cudaMemcpyDeviceToHost);

    // Validate dual data
    for (size_t i = 0; i < dual_data.size(); ++i) {
        EXPECT_FLOAT_EQ(host_dual_data[i].real(), dual_data[i].real());
        EXPECT_FLOAT_EQ(host_dual_data[i].imag(), dual_data[i].imag());
    }

    // Copy hyperdual data back to host for validation
    std::vector<thrust::complex<float>> host_hyperdual_data(batch_size * real_size * dual_size * dual_size);
    cudaMemcpy(host_hyperdual_data.data(), d_hyperdual_out, hyperdual_mem_size, cudaMemcpyDeviceToHost);

    // Validate hyperdual data
    for (size_t i = 0; i < hyperdual_data.size(); ++i) {
        EXPECT_FLOAT_EQ(host_hyperdual_data[i].real(), hyperdual_data[i].real());
        EXPECT_FLOAT_EQ(host_hyperdual_data[i].imag(), hyperdual_data[i].imag());
    }

    // Free GPU memory
    cudaFree(d_real);
    cudaFree(d_dual);
    cudaFree(d_hyperdual);
    cudaFree(d_real_out);
    cudaFree(d_dual_out);
    cudaFree(d_hyperdual_out);
}


__global__ void elementwiseAddKernel(
    const thrust::complex<float>* real1, const thrust::complex<float>* real2,
    const thrust::complex<float>* dual1, const thrust::complex<float>* dual2,
    const thrust::complex<float>* hyperdual1, const thrust::complex<float>* hyperdual2,
    thrust::complex<float>* real_result, thrust::complex<float>* dual_result, thrust::complex<float>* hyperdual_result,
    int total_real_elements, int total_dual_elements, int total_hyperdual_elements) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Add real part
    if (idx < total_real_elements) {
        real_result[idx] = real1[idx] + real2[idx];
    }

    // Add dual part
    if (idx < total_dual_elements) {
        dual_result[idx] = dual1[idx] + dual2[idx];
    }

    // Add hyperdual part
    if (idx < total_hyperdual_elements) {
        hyperdual_result[idx] = hyperdual1[idx] + hyperdual2[idx];
    }
}



TEST(VectorHyperDualDenseCudaTest, ElementwiseAddRandomComplex) {
    const int batch_size = 2;
    const int real_size = 3;
    const int dual_size = 2;

    // Generate random complex numbers for real, dual, and hyperdual data
    std::vector<thrust::complex<float>> real_data1 = generateRandomComplexVector(batch_size * real_size, -1.0f, 1.0f, -1.0f, 1.0f);
    std::vector<thrust::complex<float>> real_data2 = generateRandomComplexVector(batch_size * real_size, -1.0f, 1.0f, -1.0f, 1.0f);
    std::vector<thrust::complex<float>> dual_data1 = generateRandomComplexVector(batch_size * real_size * dual_size, -1.0f, 1.0f, -1.0f, 1.0f);
    std::vector<thrust::complex<float>> dual_data2 = generateRandomComplexVector(batch_size * real_size * dual_size, -1.0f, 1.0f, -1.0f, 1.0f);
    std::vector<thrust::complex<float>> hyperdual_data1 = generateRandomComplexVector(batch_size * real_size * dual_size * dual_size, -1.0f, 1.0f, -1.0f, 1.0f);
    std::vector<thrust::complex<float>> hyperdual_data2 = generateRandomComplexVector(batch_size * real_size * dual_size * dual_size, -1.0f, 1.0f, -1.0f, 1.0f);

    // Allocate GPU memory
    size_t real_mem_size = batch_size * real_size * sizeof(thrust::complex<float>);
    size_t dual_mem_size = batch_size * real_size * dual_size * sizeof(thrust::complex<float>);
    size_t hyperdual_mem_size = batch_size * real_size * dual_size * dual_size * sizeof(thrust::complex<float>);
    thrust::complex<float>* d_real1;
    thrust::complex<float>* d_real2;
    thrust::complex<float>* d_dual1;
    thrust::complex<float>* d_dual2;
    thrust::complex<float>* d_hyperdual1;
    thrust::complex<float>* d_hyperdual2;
    thrust::complex<float>* d_real_result;
    thrust::complex<float>* d_dual_result;
    thrust::complex<float>* d_hyperdual_result;

    cudaMalloc(&d_real1, real_mem_size);
    cudaMalloc(&d_real2, real_mem_size);
    cudaMalloc(&d_dual1, dual_mem_size);
    cudaMalloc(&d_dual2, dual_mem_size);
    cudaMalloc(&d_hyperdual1, hyperdual_mem_size);
    cudaMalloc(&d_hyperdual2, hyperdual_mem_size);
    cudaMalloc(&d_real_result, real_mem_size);
    cudaMalloc(&d_dual_result, dual_mem_size);
    cudaMalloc(&d_hyperdual_result, hyperdual_mem_size);

    // Copy data to GPU
    cudaMemcpy(d_real1, real_data1.data(), real_mem_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_real2, real_data2.data(), real_mem_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dual1, dual_data1.data(), dual_mem_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dual2, dual_data2.data(), dual_mem_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_hyperdual1, hyperdual_data1.data(), hyperdual_mem_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_hyperdual2, hyperdual_data2.data(), hyperdual_mem_size, cudaMemcpyHostToDevice);

    // Create GPU objects
    janus::VectorHyperDualDenseCuda<float> vec1(batch_size, real_size, dual_size, d_real1, d_dual1, d_hyperdual1);
    janus::VectorHyperDualDenseCuda<float> vec2(batch_size, real_size, dual_size, d_real2, d_dual2, d_hyperdual2);
    janus::VectorHyperDualDenseCuda<float> result(batch_size, real_size, dual_size, d_real_result, d_dual_result, d_hyperdual_result);

    // Configure and launch the kernel for elementwise addition
    int threadsPerBlock = 256;
    int blocksPerGrid = (batch_size * real_size * dual_size * dual_size + threadsPerBlock - 1) / threadsPerBlock;

    elementwiseAddKernel<<<blocksPerGrid, threadsPerBlock>>>(
        vec1.real(), vec2.real(), vec1.dual(), vec2.dual(), vec1.hyperdual(), vec2.hyperdual(),
        result.real(), result.dual(), result.hyperdual(),
        batch_size * real_size, batch_size * real_size * dual_size, batch_size * real_size * dual_size * dual_size);
    cudaDeviceSynchronize();

    // Copy result back to host for verification
    std::vector<thrust::complex<float>> host_real_result(batch_size * real_size);
    std::vector<thrust::complex<float>> host_dual_result(batch_size * real_size * dual_size);
    std::vector<thrust::complex<float>> host_hyperdual_result(batch_size * real_size * dual_size * dual_size);
    cudaMemcpy(host_real_result.data(), d_real_result, real_mem_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_dual_result.data(), d_dual_result, dual_mem_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_hyperdual_result.data(), d_hyperdual_result, hyperdual_mem_size, cudaMemcpyDeviceToHost);

    // Verify the result
    for (size_t i = 0; i < real_data1.size(); ++i) {
        EXPECT_EQ(host_real_result[i], real_data1[i] + real_data2[i]);
    }
    for (size_t i = 0; i < dual_data1.size(); ++i) {
        EXPECT_EQ(host_dual_result[i], dual_data1[i] + dual_data2[i]);
    }
    for (size_t i = 0; i < hyperdual_data1.size(); ++i) {
        EXPECT_EQ(host_hyperdual_result[i], hyperdual_data1[i] + hyperdual_data2[i]);
    }

    // Free GPU memory
    cudaFree(d_real1);
    cudaFree(d_real2);
    cudaFree(d_dual1);
    cudaFree(d_dual2);
    cudaFree(d_hyperdual1);
    cudaFree(d_hyperdual2);
    cudaFree(d_real_result);
    cudaFree(d_dual_result);
    cudaFree(d_hyperdual_result);
}

__global__ void elementwiseMultiplyKernel(
    const janus::VectorHyperDualDenseCuda<float> vec1,
    const janus::VectorHyperDualDenseCuda<float> vec2,
    janus::VectorHyperDualDenseCuda<float> result) {

    // Use the member function on the device
    vec1.elementwiseMultiply(vec2, result);
}


TEST(VectorHyperDualDenseCudaTest, ElementwiseMultiply) {
    int batch_size = 2;
    int real_size = 3;
    int dual_size = 2;

    // Initialize host data [2,3]
    std::vector<thrust::complex<float>> real_data = {
        {1.0, 1.0}, {2.0, 2.0}, {3.0, 3.0},
        {4.0, 4.0}, {5.0, 5.0}, {6.0, 6.0}};
    //Dual part is [2, 3, 2]
    std::vector<thrust::complex<float>> dual_data = {
        {0.1, 0.1}, {0.2, 0.2}, {0.3, 0.3},
        {0.4, 0.4}, {0.5, 0.5}, {0.6, 0.6},
        {0.7, 0.7}, {0.8, 0.8}, {0.9, 0.9},
        {1.0, 1.0}, {1.1, 1.1}, {1.2, 1.2}};
    //Hyperdual part is [2, 3, 2, 2]
    std::vector<thrust::complex<float>> hyperdual_data = {
        {0.01, 0.01}, {0.02, 0.02}, {0.03, 0.03},
        {0.04, 0.04}, {0.05, 0.05}, {0.06, 0.06},
        {0.07, 0.07}, {0.08, 0.08}, {0.09, 0.09},
        {0.10, 0.10}, {0.11, 0.11}, {0.12, 0.12},
        {0.13, 0.13}, {0.14, 0.14}, {0.15, 0.15},
        {0.16, 0.16}, {0.17, 0.17}, {0.18, 0.18}};

    // Allocate GPU memory
    thrust::complex<float>* d_real1;
    thrust::complex<float>* d_real2;
    thrust::complex<float>* d_dual1;
    thrust::complex<float>* d_dual2;
    thrust::complex<float>* d_hyperdual1;
    thrust::complex<float>* d_hyperdual2;
    thrust::complex<float>* d_result_real;
    thrust::complex<float>* d_result_dual;
    thrust::complex<float>* d_result_hyperdual;

    size_t real_mem_size = batch_size * real_size * sizeof(thrust::complex<float>);
    size_t dual_mem_size = batch_size * real_size * dual_size * sizeof(thrust::complex<float>);
    size_t hyperdual_mem_size = batch_size * real_size * dual_size * dual_size * sizeof(thrust::complex<float>);

    cudaMalloc(&d_real1, real_mem_size);
    cudaMalloc(&d_real2, real_mem_size);
    cudaMalloc(&d_dual1, dual_mem_size);
    cudaMalloc(&d_dual2, dual_mem_size);
    cudaMalloc(&d_hyperdual1, hyperdual_mem_size);
    cudaMalloc(&d_hyperdual2, hyperdual_mem_size);
    cudaMalloc(&d_result_real, real_mem_size);
    cudaMalloc(&d_result_dual, dual_mem_size);
    cudaMalloc(&d_result_hyperdual, hyperdual_mem_size);

    // Copy data to GPU
    cudaMemcpy(d_real1, real_data.data(), real_mem_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_real2, real_data.data(), real_mem_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dual1, dual_data.data(), dual_mem_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dual2, dual_data.data(), dual_mem_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_hyperdual1, hyperdual_data.data(), hyperdual_mem_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_hyperdual2, hyperdual_data.data(), hyperdual_mem_size, cudaMemcpyHostToDevice);

    // Create GPU objects
    janus::VectorHyperDualDenseCuda<float> vec1(batch_size, real_size, dual_size, d_real1, d_dual1, d_hyperdual1);
    janus::VectorHyperDualDenseCuda<float> vec2(batch_size, real_size, dual_size, d_real2, d_dual2, d_hyperdual2);
    janus::VectorHyperDualDenseCuda<float> result(batch_size, real_size, dual_size, d_result_real, d_result_dual, d_result_hyperdual);

    // Configure and launch the kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = 1;
    elementwiseMultiplyKernel<<<blocksPerGrid, threadsPerBlock>>>(vec1, vec2, result);
    cudaDeviceSynchronize();

    // Check for kernel errors
    cudaError_t err = cudaGetLastError();
    EXPECT_EQ(err, cudaSuccess) << "CUDA kernel failed: " << cudaGetErrorString(err);

    // Copy result back to host for verification
    std::vector<thrust::complex<float>> result_real(batch_size * real_size);
    std::vector<thrust::complex<float>> result_dual(batch_size * real_size * dual_size);
    std::vector<thrust::complex<float>> result_hyperdual(batch_size * real_size * dual_size * dual_size);

    cudaMemcpy(result_real.data(), d_result_real, real_mem_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(result_dual.data(), d_result_dual, dual_mem_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(result_hyperdual.data(), d_result_hyperdual, hyperdual_mem_size, cudaMemcpyDeviceToHost);

    float tolerance = 1e-6;

    // Verify real part
    for (size_t m = 0; m < batch_size; ++m) {
        for ( size_t i=0; i < real_size; i++) {
            size_t idx = m * real_size + i;
            auto expected = real_data[idx] * real_data[idx];
            EXPECT_NEAR(result_real[idx].real(), expected.real(), tolerance);
            EXPECT_NEAR(result_real[idx].imag(), expected.imag(), tolerance);
        }
    }

    // Verify dual part
    for ( int m=0; m < batch_size; m++) {
        for (int i = 0; i < real_size; ++i) {
            auto realval = real_data[m * real_size + i];
            for (int j = 0; j < dual_size; ++j) {
                int dual_idx = m * real_size * dual_size + i * dual_size + j;
                auto dualval = dual_data[dual_idx];
                auto expected = 2*realval * dualval;
                EXPECT_NEAR(result_dual[dual_idx].real(), expected.real(), tolerance);
                EXPECT_NEAR(result_dual[dual_idx].imag(), expected.imag(), tolerance);
                
            }
        }
    }


    // Verify hyperdual part
    for (int m = 0; m < batch_size; m++) {
        for (int i = 0; i < real_size; ++i) {
            auto realval = real_data[m * real_size + i];
            for (int j = 0; j < dual_size; ++j) {
                auto dualval = dual_data[m * real_size * dual_size + i * dual_size + j];
                for (int k = 0; k < dual_size; ++k) {
                    int hyperdual_idx = m * real_size * dual_size * dual_size + i * dual_size * dual_size + j * dual_size + k;
                    auto hyperdualval = hyperdual_data[hyperdual_idx];
                    auto expected = 2*realval * hyperdualval + 2*dualval * dualval;
                    EXPECT_NEAR(result_hyperdual[hyperdual_idx].real(), expected.real(), tolerance);
                    //EXPECT_NEAR(result_hyperdual[hyperdual_idx].imag(), expected.imag(), tolerance);
                }
            }
        }
    }

    // Free GPU memory
    cudaFree(d_real1);
    cudaFree(d_real2);
    cudaFree(d_dual1);
    cudaFree(d_dual2);
    cudaFree(d_hyperdual1);
    cudaFree(d_hyperdual2);
    cudaFree(d_result_real);
    cudaFree(d_result_dual);
    cudaFree(d_result_hyperdual);
}






// Main function for running all tests
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);  // Initialize GTest
    return RUN_ALL_TESTS();                  // Run all defined tests
}