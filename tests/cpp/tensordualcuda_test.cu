#include <gtest/gtest.h>
#include <torch/torch.h>
#include <random>
#include "../../src/cpp/cudatensordense.cu"
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

    auto result = vec1.elementwiseAdd(vec2);

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

// Additional tests like `sum`, `sign`, etc., would follow a similar pattern.

TEST(VectorDualDenseCudaTest, ConstructorWithInternalMemory) {
    int batch_size = 2;
    int size = 5;
    int dual_dim = 3;

    janus::VectorDualDenseCuda<float> vec(batch_size, size, dual_dim);

    EXPECT_EQ(vec.batchSize(), batch_size);
    EXPECT_EQ(vec.size(), size);
    EXPECT_EQ(vec.dualSize(), dual_dim);
}


TEST(VectorDualDenseCudaTest, Initialize) {
    int batch_size = 2;
    int size = 5;
    int dual_dim = 3;

    janus::VectorDualDenseCuda<float> vec(batch_size, size, dual_dim);

    // Initialize data
    std::vector<thrust::complex<float>> primal_data(batch_size * size, thrust::complex<float>(1.0f, 2.0f));
    std::vector<thrust::complex<float>> dual_data(batch_size * size * dual_dim, thrust::complex<float>(0.5f, 0.5f));

    vec.initialize(primal_data.data(), dual_data.data());

    // Copy primal data back to host
    std::vector<thrust::complex<float>> host_primal_data(batch_size * size);
    cudaMemcpy(host_primal_data.data(), vec.real(),
               batch_size * size * sizeof(thrust::complex<float>), cudaMemcpyDeviceToHost);

    // Validate primal data
    for (size_t i = 0; i < primal_data.size(); ++i) {
        EXPECT_FLOAT_EQ(host_primal_data[i].real(), primal_data[i].real());
        EXPECT_FLOAT_EQ(host_primal_data[i].imag(), primal_data[i].imag());
    }

    // Copy dual data back to host
    std::vector<thrust::complex<float>> host_dual_data(batch_size * size * dual_dim);
    cudaMemcpy(host_dual_data.data(), vec.dual(),
               batch_size * size * dual_dim * sizeof(thrust::complex<float>), cudaMemcpyDeviceToHost);

    // Validate dual data
    for (size_t i = 0; i < dual_data.size(); ++i) {
        EXPECT_FLOAT_EQ(host_dual_data[i].real(), dual_data[i].real());
        EXPECT_FLOAT_EQ(host_dual_data[i].imag(), dual_data[i].imag());
    }
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

// Test for elementwise addition
TEST(VectorDualDenseCudaTest, ElementwiseAddRandomComplex) {
    const int batch_size = 2;
    const int size = 3;
    const int dual_dim = 2;

    // Generate random complex numbers for primal and dual data
    std::vector<thrust::complex<float>> primal_data1 = generateRandomComplexVector(batch_size * size, -1.0f, 1.0f, -1.0f, 1.0f);
    std::vector<thrust::complex<float>> primal_data2 = generateRandomComplexVector(batch_size * size, -1.0f, 1.0f, -1.0f, 1.0f);
    std::vector<thrust::complex<float>> dual_data1 = generateRandomComplexVector(batch_size * size * dual_dim, -1.0f, 1.0f, -1.0f, 1.0f);
    std::vector<thrust::complex<float>> dual_data2 = generateRandomComplexVector(batch_size * size * dual_dim, -1.0f, 1.0f, -1.0f, 1.0f);

    // Initialize CUDA vectors
    VectorDualDenseCuda<float> vec1(batch_size, size, dual_dim);
    VectorDualDenseCuda<float> vec2(batch_size, size, dual_dim);
    

    //Copy data to device
    vec1.initialize(primal_data1.data(), dual_data1.data());

    vec2.initialize(primal_data2.data(), dual_data2.data());

    // Perform elementwise addition
    VectorDualDenseCuda<float> result = vec1.elementwiseAdd(vec2);

    // Copy result back to the host for verification
    std::vector<thrust::complex<float>> host_primal_result(batch_size * size);
    std::vector<thrust::complex<float>> host_dual_result(batch_size * size * dual_dim);
    cudaMemcpy(host_primal_result.data(), result.real(), batch_size*size * sizeof(thrust::complex<float>), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_dual_result.data(), result.dual(), batch_size*size*dual_dim * sizeof(thrust::complex<float>), cudaMemcpyDeviceToHost);

    // Verify the result
    for (size_t i = 0; i < primal_data1.size(); ++i) {
        EXPECT_EQ(host_primal_result[i], primal_data1[i] + primal_data2[i]);
    }
    for (size_t i = 0; i < dual_data1.size(); ++i) {
        EXPECT_EQ(host_dual_result[i], dual_data1[i] + dual_data2[i]);
    }
}

TEST(VectorDualDenseCudaTest, ElementwiseMultiplyRandomComplex) {
    const int batch_size = 2;
    const int size = 3;
    const int dual_dim = 2;

    // Generate random complex numbers for primal and dual data
    std::vector<thrust::complex<float>> primal_data1 = generateRandomComplexVector(batch_size * size, -1.0f, 1.0f, -1.0f, 1.0f);
    std::vector<thrust::complex<float>> primal_data2 = generateRandomComplexVector(batch_size * size, -1.0f, 1.0f, -1.0f, 1.0f);
    std::vector<thrust::complex<float>> dual_data1 = generateRandomComplexVector(batch_size * size * dual_dim, -1.0f, 1.0f, -1.0f, 1.0f);
    std::vector<thrust::complex<float>> dual_data2 = generateRandomComplexVector(batch_size * size * dual_dim, -1.0f, 1.0f, -1.0f, 1.0f);

    // Initialize CUDA vectors
    VectorDualDenseCuda<float> vec1(batch_size, size, dual_dim);
    VectorDualDenseCuda<float> vec2(batch_size, size, dual_dim);
    
    // Copy data to device
    vec1.initialize(primal_data1.data(), dual_data1.data());
    vec2.initialize(primal_data2.data(), dual_data2.data());

    // Perform elementwise multiplication
    VectorDualDenseCuda<float> result = vec1.elementwiseMultiply(vec2);

    // Copy result back to the host for verification
    std::vector<thrust::complex<float>> host_primal_result(batch_size * size);
    std::vector<thrust::complex<float>> host_dual_result(batch_size * size * dual_dim);
    cudaMemcpy(host_primal_result.data(), result.real(), batch_size * size * sizeof(thrust::complex<float>), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_dual_result.data(), result.dual(), batch_size * size * dual_dim * sizeof(thrust::complex<float>), cudaMemcpyDeviceToHost);
    // Tolerance for floating-point comparison
    const float tolerance = 1e-6;

    //The dual part is real1*dual2 + real2*dual1


    // Verify the result
    for (size_t i = 0; i < primal_data1.size(); ++i) {
        EXPECT_NEAR(host_primal_result[i].real(), (primal_data1[i] * primal_data2[i]).real(), tolerance);
        EXPECT_NEAR(host_primal_result[i].imag(), (primal_data1[i] * primal_data2[i]).imag(), tolerance);
    }

    std::vector<thrust::complex<float>> dual_result_local(batch_size * size * dual_dim);

    for ( int m=0; m < batch_size; m++) {
        for ( int i=0; i < size; i++) {
            int realidx = m*size + i;
            for ( int j=0; j < dual_dim; j++) {
                int dualidx = m*size*dual_dim + i*dual_dim + j;
                dual_result_local[dualidx] = primal_data1[realidx]*dual_data2[dualidx]+
                                            primal_data2[realidx]*dual_data1[dualidx];  
            }
        }
    }
    for (size_t i = 0; i < batch_size*size*dual_dim; ++i) {
        EXPECT_NEAR(host_dual_result[i].real(), dual_result_local[i].real(), tolerance);
        EXPECT_NEAR(host_dual_result[i].imag(), dual_result_local[i].imag(), tolerance);
    }

}




// Main function for running all tests
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);  // Initialize GTest
    return RUN_ALL_TESTS();                  // Run all defined tests
}