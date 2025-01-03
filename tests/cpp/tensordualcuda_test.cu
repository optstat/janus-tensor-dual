#include <gtest/gtest.h>
#include <torch/torch.h>
#include <random>
#include "../../src/cpp/cudatensordense.cu"
#include <thrust/host_vector.h>
using namespace janus;
// Test internal memory allocation constructor
TEST(VectorDenseFloatTest, ConstructorInternalMemory) {
    int batch_size = 2;
    int    vector_size = 3;
    std::vector<float>  host_data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

    VectorDense<float> vec(batch_size, vector_size);

    ASSERT_EQ(vec.batchSize(), batch_size);
    ASSERT_EQ(vec.size(), vector_size);
}

// Test external memory constructor
TEST(VectorDenseFloatTest, ConstructorExternalMemory) {
    int batch_size = 2;
    int    vector_size = 3;
    std::vector<float>  host_data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

    float* device_data;
    cudaMalloc(&device_data, batch_size * vector_size * sizeof(float));

    VectorDense<float> vec(batch_size, vector_size, device_data);

    ASSERT_EQ(vec.batchSize(), batch_size);
    ASSERT_EQ(vec.size(), vector_size);

    cudaFree(device_data);
}


// Test initialization with host data
TEST(VectorDenseFloatTest, InitializeData) {
    int batch_size = 2;
    int    vector_size = 3;
    std::vector<float>  host_data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

    VectorDense<float> vec(batch_size, vector_size);

    vec.initialize(host_data.data(), host_data.size());

    thrust::host_vector<float> device_to_host(host_data.size());
    cudaMemcpy(device_to_host.data(), vec.data(),
               host_data.size() * sizeof(float), cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < host_data.size(); ++i) {
        EXPECT_FLOAT_EQ(device_to_host[i], host_data[i]);
    }
}

// Test elementwise addition
TEST(VectorDenseFloatTest, ElementwiseAdd) {
    int batch_size = 2;
    int    vector_size = 3;
    std::vector<float>  host_data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

    VectorDense<float> vec1(batch_size, vector_size);
    VectorDense<float> vec2(batch_size, vector_size);

    vec1.initialize(host_data.data(), host_data.size());
    vec2.initialize(host_data.data(), host_data.size());

    auto result = vec1.elementwiseAdd(vec2);

    thrust::host_vector<float> result_host(batch_size * vector_size);
    cudaMemcpy(result_host.data(), result.data(),
               batch_size * vector_size * sizeof(float), cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < host_data.size(); ++i) {
        EXPECT_FLOAT_EQ(result_host[i], host_data[i] + host_data[i]);
    }
}


// Main function for running all tests
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);  // Initialize GTest
    return RUN_ALL_TESTS();                  // Run all defined tests
}