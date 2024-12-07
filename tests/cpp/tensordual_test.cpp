#include <gtest/gtest.h>
#include <torch/torch.h>
#include <random>
#include "../../src/cpp/tensordual.hpp"
#include "../../src/cpp/janus_util.hpp"

using namespace janus;
// Test case for zeros method
/**
 *  g++ -g -std=c++17 tensordual_test.cpp -o dualtest   -I /home/panos/Applications/libtorch/include/torch/csrc/api/include/   -I /home/panos/Applications/libtorch/include/torch/csrc/api/include/torch/   -I /home/panos/Applications/libtorch/include/   -I /usr/local/include/gtest  -L /usr/local/lib   -lgtest  -lgtest_main -L /home/panos/Applications/libtorch/lib/ -ltorch   -ltorch_cpu   -ltorch_cuda   -lc10 -lpthread -Wl,-rpath,/home/panos/Applications/libtorch/lib
 */
// Helper function to check if a tensor is filled with zeros
bool isZeroTensor(const torch::Tensor& tensor) {
    return tensor.sum().item<double>() == 0.0;
}

// Test: Basic functionality of zeros_like
TEST(TensorDualTest, ZerosLikeBasic) {
    // Create a TensorDual object
    torch::Tensor r = torch::rand({3, 3}, torch::dtype(torch::kFloat32));
    torch::Tensor d = torch::rand({3, 3, 6}, torch::dtype(torch::kFloat32));
    TensorDual tensorDual(r, d);

    // Call zeros_like
    TensorDual result = tensorDual.zeros_like();

    // Validate that the new tensors are zeros
    EXPECT_TRUE(isZeroTensor(result.r)) << "Real part is not zero.";
    EXPECT_TRUE(isZeroTensor(result.d)) << "Dual part is not zero.";

    // Validate that the shapes match
    EXPECT_EQ(result.r.sizes(), r.sizes()) << "Real part shape mismatch.";
    EXPECT_EQ(result.d.sizes(), d.sizes()) << "Dual part shape mismatch.";

    // Validate that the dtypes match
    EXPECT_EQ(result.r.dtype(), r.dtype()) << "Real part dtype mismatch.";
    EXPECT_EQ(result.d.dtype(), d.dtype()) << "Dual part dtype mismatch.";

    // Validate that the devices match
    EXPECT_EQ(result.r.device(), r.device()) << "Real part device mismatch.";
    EXPECT_EQ(result.d.device(), d.device()) << "Dual part device mismatch.";
}



// Test: Edge case - Tensors on GPU
TEST(TensorDualTest, ZerosLikeTensorsOnGPU) {
    if (torch::cuda::is_available()) {
        // Create a TensorDual object with tensors on GPU
        torch::Tensor r = torch::rand({3, 3}, torch::device(torch::kCUDA));
        torch::Tensor d = torch::rand({3, 3, 6}, torch::device(torch::kCUDA));
        TensorDual tensorDual(r, d);

        // Call zeros_like
        TensorDual result = tensorDual.zeros_like();

        // Validate that the new tensors are zeros and on the GPU
        EXPECT_TRUE(isZeroTensor(result.r)) << "Real part is not zero on GPU.";
        EXPECT_TRUE(isZeroTensor(result.d)) << "Dual part is not zero on GPU.";
        EXPECT_EQ(result.r.device().type(), torch::kCUDA) << "Real part is not on GPU.";
        EXPECT_EQ(result.d.device().type(), torch::kCUDA) << "Dual part is not on GPU.";
    }
}
// Test: Basic functionality of ones_like
TEST(TensorDualTest, OnesLikeBasic) {
    // Input TensorDual object
    auto r = torch::randn({3, 3});
    auto d = torch::randn({3, 3, 2});
    TensorDual input(r, d);

    // Call ones_like
    TensorDual result = TensorDual::ones_like(input);

    // Validate the real part is filled with ones
    EXPECT_TRUE(torch::allclose(result.r, torch::ones_like(r)));

    // Validate the dual part is filled with zeros
    EXPECT_TRUE(torch::allclose(result.d, torch::zeros_like(d)));

    // Validate that the result tensors have the same shape, dtype, and device as the input
    EXPECT_EQ(result.r.sizes(), r.sizes());
    EXPECT_EQ(result.d.sizes(), d.sizes());
    EXPECT_EQ(result.r.device(), r.device());
    EXPECT_EQ(result.d.device(), d.device());
    EXPECT_EQ(result.r.dtype(), r.dtype());
}



// Test: Device-specific tensors (e.g., GPU)
TEST(TensorDualTest, OnesLikeGpuTensors) {
    if (torch::cuda::is_available()) {
        // Input TensorDual on GPU
        auto r = torch::randn({3, 3}, torch::device(torch::kCUDA));
        auto d = torch::randn({3, 3, 2}, torch::device(torch::kCUDA));
        TensorDual input(r, d);

        // Call ones_like
        TensorDual result = TensorDual::ones_like(input);

        // Validate the real part is ones
        EXPECT_TRUE(torch::allclose(result.r, torch::ones_like(r)));

        // Validate the dual part is zeros
        EXPECT_TRUE(torch::allclose(result.d, torch::zeros_like(d)));

        // Validate device
        EXPECT_EQ(result.r.device(), r.device());
        EXPECT_EQ(result.d.device(), d.device());
    }
}


// Test: Basic functionality of bool_like
TEST(TensorDualTest, BoolLikeBasic) {
    // Input TensorDual with a defined real tensor
    auto r = torch::randn({4, 4});
    auto d = torch::randn({4, 4, 3});
    TensorDual input(r, d);

    // Call bool_like
    torch::Tensor result = TensorDual::bool_like(input);

    // Validate the result is a boolean tensor of the same shape and device as r
    EXPECT_EQ(result.sizes(), r.sizes());
    EXPECT_EQ(result.device(), r.device());
    EXPECT_EQ(result.dtype(), torch::kBool);

    // Validate the values are all false (zeros)
    EXPECT_TRUE(torch::allclose(result, torch::zeros_like(r, r.options().dtype(torch::kBool))));
}
// Test: Basic functionality of createZero
TEST(TensorDualTest, CreateZeroBasic) {
    // Input tensor
    auto r = torch::randn({4, 4});

    // Call createZero
    int ddim = 3;
    TensorDual result = TensorDual::createZero(r, ddim);

    // Validate the real part matches the input tensor
    EXPECT_TRUE(torch::allclose(result.r, r));

    // Validate the dual part is a zero tensor with the additional dimension
    auto expected_dshape = r.sizes().vec();
    expected_dshape.push_back(ddim);
    EXPECT_EQ(result.d.sizes(), torch::IntArrayRef(expected_dshape));
    EXPECT_TRUE(torch::allclose(result.d, torch::zeros_like(result.d)));

    // Validate dtype and device
    EXPECT_EQ(result.d.dtype(), r.dtype());
    EXPECT_EQ(result.d.device(), r.device());
}

// Test: Device-specific tensors (e.g., GPU)
TEST(TensorDualTest, CreateZeroGpuTensor) {
    if (torch::cuda::is_available()) {
        // Input tensor on GPU
        auto r = torch::randn({4, 4}, torch::device(torch::kCUDA));

        // Call createZero
        int ddim = 2;
        TensorDual result = TensorDual::createZero(r, ddim);

        // Validate the real part matches the input tensor
        EXPECT_TRUE(torch::allclose(result.r, r));

        // Validate the dual part is a zero tensor with the additional dimension
        auto expected_dshape = r.sizes().vec();
        expected_dshape.push_back(ddim);
        EXPECT_EQ(result.d.sizes(), torch::IntArrayRef(expected_dshape));
        EXPECT_TRUE(torch::allclose(result.d, torch::zeros_like(result.d)));

        // Validate dtype and device
        EXPECT_EQ(result.d.dtype(), r.dtype());
        EXPECT_EQ(result.d.device(), r.device());
    }
}

// Test: Device-specific tensors (e.g., GPU)
TEST(TensorDualTest, BoolLikeGpuTensor) {
    if (torch::cuda::is_available()) {
        // Input TensorDual with tensors on GPU
        auto r = torch::randn({5, 5}, torch::device(torch::kCUDA));
        auto d = torch::randn({5, 5, 3}, torch::device(torch::kCUDA));
        TensorDual input(r, d);

        // Call bool_like
        torch::Tensor result = TensorDual::bool_like(input);

        // Validate the result is a boolean tensor on GPU
        EXPECT_EQ(result.sizes(), r.sizes());
        EXPECT_EQ(result.device(), r.device());
        EXPECT_EQ(result.dtype(), torch::kBool);

        // Validate the values are all false (zeros)
        EXPECT_TRUE(torch::allclose(result, torch::zeros_like(r, r.options().dtype(torch::kBool))));
    }
}


// Test: Basic functionality of empty_like
TEST(TensorDualTest, EmptyLikeBasic) {
    // Input TensorDual with defined tensors
    auto r = torch::randn({4, 4});
    auto d = torch::randn({4, 4, 3});
    TensorDual input(r, d);

    // Call empty_like
    TensorDual result = TensorDual::empty_like(input);

    // Validate the real part has the same shape, dtype, and device as input
    EXPECT_EQ(result.r.sizes(), r.sizes());
    EXPECT_EQ(result.r.dtype(), r.dtype());
    EXPECT_EQ(result.r.device(), r.device());

    // Validate the dual part has the same shape, dtype, and device as input
    EXPECT_EQ(result.d.sizes(), d.sizes());
    EXPECT_EQ(result.d.dtype(), d.dtype());
    EXPECT_EQ(result.d.device(), d.device());
}


// Test: Device-specific tensors (e.g., GPU)
TEST(TensorDualTest, EmptyLikeGpuTensors) {
    if (torch::cuda::is_available()) {
        // Input TensorDual with tensors on GPU
        auto r = torch::randn({4, 4}, torch::device(torch::kCUDA));
        auto d = torch::randn({4, 4, 3}, torch::device(torch::kCUDA));
        TensorDual input(r, d);

        // Call empty_like
        TensorDual result = TensorDual::empty_like(input);

        // Validate the real part matches the input shape, dtype, and device
        EXPECT_EQ(result.r.sizes(), r.sizes());
        EXPECT_EQ(result.r.dtype(), r.dtype());
        EXPECT_EQ(result.r.device(), r.device());

        // Validate the dual part matches the input shape, dtype, and device
        EXPECT_EQ(result.d.sizes(), d.sizes());
        EXPECT_EQ(result.d.dtype(), d.dtype());
        EXPECT_EQ(result.d.device(), d.device());
    }
}

// Test: Large tensors
TEST(TensorDualTest, EmptyLikeLargeTensors) {
    // Input TensorDual with large tensors
    auto r = torch::randn({1024, 1024});
    auto d = torch::randn({1024, 1024, 2});
    TensorDual input(r, d);

    // Call empty_like
    TensorDual result = TensorDual::empty_like(input);

    // Validate the real part matches the input shape, dtype, and device
    EXPECT_EQ(result.r.sizes(), r.sizes());
    EXPECT_EQ(result.r.dtype(), r.dtype());
    EXPECT_EQ(result.r.device(), r.device());

    // Validate the dual part matches the input shape, dtype, and device
    EXPECT_EQ(result.d.sizes(), d.sizes());
    EXPECT_EQ(result.d.dtype(), d.dtype());
    EXPECT_EQ(result.d.device(), d.device());
}


// Test: Basic functionality of cat
TEST(TensorDualTest, CatBasic) {
    // Input TensorDual objects
    TensorDual td1(torch::randn({2, 3}), torch::randn({2, 3, 4}));
    TensorDual td2(torch::randn({2, 3}), torch::randn({2, 3, 4}));
    std::vector<TensorDual> args = {td1, td2};

    // Call cat
    TensorDual result = TensorDual::cat(args, 0);

    // Validate the real part concatenation
    auto expected_r = torch::cat({td1.r, td2.r}, 0);
    EXPECT_TRUE(torch::allclose(result.r, expected_r));

    // Validate the dual part concatenation
    auto expected_d = torch::cat({td1.d, td2.d}, 0);
    EXPECT_TRUE(torch::allclose(result.d, expected_d));

    // Validate shapes
    EXPECT_EQ(result.r.sizes(), expected_r.sizes());
    EXPECT_EQ(result.d.sizes(), expected_d.sizes());
}

// Test: Concatenation along a non-zero dimension
TEST(TensorDualTest, CatNonZeroDimension) {
    // Input TensorDual objects
    TensorDual td1(torch::randn({2, 3}), torch::randn({2, 3, 4}));
    TensorDual td2(torch::randn({2, 3}), torch::randn({2, 3, 4}));
    std::vector<TensorDual> args = {td1, td2};

    // Call cat along dimension 1
    TensorDual result = TensorDual::cat(args, 1);

    // Validate the real part concatenation
    auto expected_r = torch::cat({td1.r, td2.r}, 1);
    EXPECT_TRUE(torch::allclose(result.r, expected_r));

    // Validate the dual part concatenation
    auto expected_d = torch::cat({td1.d, td2.d}, 1);
    EXPECT_TRUE(torch::allclose(result.d, expected_d));

    // Validate shapes
    EXPECT_EQ(result.r.sizes(), expected_r.sizes());
    EXPECT_EQ(result.d.sizes(), expected_d.sizes());
}

// Test: Edge case - Single TensorDual in input
TEST(TensorDualTest, CatSingleInput) {
    // Input vector with a single TensorDual
    TensorDual td1(torch::randn({2, 3}), torch::randn({2, 3, 4}));
    std::vector<TensorDual> args = {td1};

    // Call cat
    TensorDual result = TensorDual::cat(args, 0);

    // Validate the real and dual parts are unchanged
    EXPECT_TRUE(torch::allclose(result.r, td1.r));
    EXPECT_TRUE(torch::allclose(result.d, td1.d));
}

// Test: Device-specific tensors (e.g., GPU)
TEST(TensorDualTest, CatGpuTensors) {
    if (torch::cuda::is_available()) {
        // Input TensorDual objects on GPU
        TensorDual td1(torch::randn({2, 3}, torch::device(torch::kCUDA)), torch::randn({2, 3, 4}, torch::device(torch::kCUDA)));
        TensorDual td2(torch::randn({2, 3}, torch::device(torch::kCUDA)), torch::randn({2, 3, 4}, torch::device(torch::kCUDA)));
        std::vector<TensorDual> args = {td1, td2};

        // Call cat
        TensorDual result = TensorDual::cat(args, 0);

        // Validate the real part concatenation
        auto expected_r = torch::cat({td1.r, td2.r}, 0);
        EXPECT_TRUE(torch::allclose(result.r, expected_r));

        // Validate the dual part concatenation
        auto expected_d = torch::cat({td1.d, td2.d}, 0);
        EXPECT_TRUE(torch::allclose(result.d, expected_d));

        // Validate device
        EXPECT_EQ(result.r.device(), td1.r.device());
        EXPECT_EQ(result.d.device(), td1.d.device());
    }
}

// Test: Basic functionality of einsum
TEST(TensorDualTest, EinsumBasic) {
    // Input TensorDual objects
    TensorDual td1(torch::randn({2, 3}), torch::randn({2, 3, 6}));
    TensorDual td2(torch::randn({3, 4}), torch::randn({3, 4, 6}));

    // Einsum operation
    std::string einsum_str = "ij,jk->ik";
    TensorDual result = TensorDual::einsum(einsum_str, td1, td2);

    // Expected real part
    auto expected_r = torch::einsum(einsum_str, {td1.r, td2.r});
    EXPECT_TRUE(torch::allclose(result.r, expected_r));

    // Expected dual part
    auto d1 = torch::einsum("ij,jkz->ikz", {td1.r, td2.d});
    auto d2 = torch::einsum("ijz,jk->ikz", {td1.d, td2.r});
    auto expected_d = d1 + d2;

    EXPECT_TRUE(torch::allclose(result.d, expected_d));
}

// Test: Invalid einsum string
TEST(TensorDualTest, EinsumInvalidString) {
    // Input TensorDual objects
    TensorDual td1(torch::randn({2, 3}), torch::randn({2, 3, 4}));
    TensorDual td2(torch::randn({3, 4}), torch::randn({3, 4, 4}));

    // Invalid einsum string (missing '->')
    std::string einsum_str = "ij,jk";

    EXPECT_THROW(
        {
            try {
                TensorDual::einsum(einsum_str, td1, td2);
            } catch (const std::invalid_argument& e) {
                EXPECT_STREQ("Einsum string must contain '->'.", e.what());
                throw;
            }
        },
        std::invalid_argument
    );
}

// Test: Edge case - Einsum operation with identity matrix
TEST(TensorDualTest, EinsumIdentityMatrix) {
    // Input TensorDual objects
    TensorDual td1(torch::randn({3, 3}), torch::randn({3, 3, 2}));
    TensorDual td2(torch::eye(3), torch::zeros({3, 3, 2}));

    // Einsum operation (identity operation)
    std::string einsum_str = "ij,jk->ik";
    TensorDual result = TensorDual::einsum(einsum_str, td1, td2);

    // Expected real part
    auto expected_r = torch::einsum(einsum_str, {td1.r, td2.r});
    EXPECT_TRUE(torch::allclose(result.r, expected_r));

    // Expected dual part
    auto darg = "ij,jkz->ikz"; // Modify einsum string for dual computation
    auto d1 = torch::einsum(darg, {td1.r, td2.d});
    auto d2 = torch::einsum(darg, {td2.r, td1.d});
    auto expected_d = d1 + d2;

    EXPECT_TRUE(torch::allclose(result.d, expected_d));
}

// Test: Device-specific tensors (e.g., GPU)
TEST(TensorDualTest, EinsumGpuTensors) {
    if (torch::cuda::is_available()) {
        // Input TensorDual objects on GPU
        TensorDual td1(torch::randn({2, 3}, torch::device(torch::kCUDA)), torch::randn({2, 3, 10}, torch::device(torch::kCUDA)));
        TensorDual td2(torch::randn({3, 4}, torch::device(torch::kCUDA)), torch::randn({3, 4, 10}, torch::device(torch::kCUDA)));

        // Einsum operation
        std::string einsum_str = "ij,jk->ik";
        TensorDual result = TensorDual::einsum(einsum_str, td1, td2);

        // Expected real part
        auto expected_r = torch::einsum(einsum_str, {td1.r, td2.r});
        EXPECT_TRUE(torch::allclose(result.r, expected_r));

        // Expected dual part
        auto d1 = torch::einsum("ij,jkz->ikz", {td1.r, td2.d});
        auto d2 = torch::einsum("ijz,jk->ikz", {td1.d, td2.r});
        auto expected_d = d1 + d2;

        EXPECT_TRUE(torch::allclose(result.d, expected_d));
    }
}

// Test: Incompatible shapes for einsum
TEST(TensorDualTest, EinsumIncompatibleShapes) {
    // Input TensorDual objects with incompatible shapes
    TensorDual td1(torch::randn({2, 3}), torch::randn({2, 3, 4}));
    TensorDual td2(torch::randn({4, 5}), torch::randn({4, 5, 2}));

    // Valid einsum string
    std::string einsum_str = "ij,jk->ik";

    EXPECT_THROW(
        {
            try {
                TensorDual::einsum(einsum_str, td1, td2);
            } catch (const std::invalid_argument& e) {
                EXPECT_STREQ("The dual part of the TensorDual objects must have the same number of leaves.", e.what());
                throw;
            }

        },
        std::invalid_argument
    );
}

// Test: Basic functionality of einsum with Tensor and TensorDual
TEST(TensorDualTest, EinsumTensorAndTensorDualBasic) {
    // Input Tensor and TensorDual
    torch::Tensor first = torch::randn({2, 3});
    TensorDual second(torch::randn({3, 4}), torch::randn({3, 4, 5}));

    // Einsum operation
    std::string einsum_str = "ij,jk->ik";
    TensorDual result = TensorDual::einsum(einsum_str, first, second);

    // Expected real part
    auto expected_r = torch::einsum(einsum_str, {first, second.r});
    EXPECT_TRUE(torch::allclose(result.r, expected_r));

    // Expected dual part
    auto darg = "ij,jkz->ikz"; // Modify einsum string for dual computation
    auto expected_d = torch::einsum(darg, {first, second.d});
    EXPECT_TRUE(torch::allclose(result.d, expected_d));
}

// Test: Invalid einsum string
TEST(TensorDualTest, EinsumTensorAndTensorDualInvalidString) {
    // Input Tensor and TensorDual
    torch::Tensor first = torch::randn({2, 3});
    TensorDual second(torch::randn({3, 4}), torch::randn({3, 4, 5}));

    // Invalid einsum string (missing '->')
    std::string einsum_str = "ij,jk";

    EXPECT_THROW(
        {
            try {
                TensorDual::einsum(einsum_str, first, second);
            } catch (const std::invalid_argument& e) {
                EXPECT_STREQ("Einsum string must contain '->'.", e.what());
                throw;
            }
        },
        std::invalid_argument
    );
}


// Test: Edge case - Einsum operation with identity matrix
TEST(TensorDualTest, EinsumTensorAndTensorDualIdentityMatrix) {
    // Input Tensor and TensorDual
    torch::Tensor first = torch::eye(3); // Identity matrix
    TensorDual second(torch::randn({3, 3}), torch::randn({3, 3, 2}));

    // Einsum operation
    std::string einsum_str = "ij,jk->ik";
    TensorDual result = TensorDual::einsum(einsum_str, first, second);

    // Expected real part
    auto expected_r = torch::einsum(einsum_str, {first, second.r});
    EXPECT_TRUE(torch::allclose(result.r, expected_r));

    // Expected dual part
    auto darg = "ij,jkz->ikz"; // Modify einsum string for dual computation
    auto expected_d = torch::einsum(darg, {first, second.d});
    EXPECT_TRUE(torch::allclose(result.d, expected_d));
}

// Test: Device-specific tensors (e.g., GPU)
TEST(TensorDualTest, EinsumTensorAndTensorDualGpuTensors) {
    if (torch::cuda::is_available()) {
        // Input Tensor and TensorDual on GPU
        torch::Tensor first = torch::randn({2, 3}, torch::device(torch::kCUDA));
        TensorDual second(
            torch::randn({3, 4}, torch::device(torch::kCUDA)),
            torch::randn({3, 4, 5}, torch::device(torch::kCUDA))
        );

        // Einsum operation
        std::string einsum_str = "ij,jk->ik";
        TensorDual result = TensorDual::einsum(einsum_str, first, second);

        // Expected real part
        auto expected_r = torch::einsum(einsum_str, {first, second.r});
        EXPECT_TRUE(torch::allclose(result.r, expected_r));

        // Expected dual part
        auto darg = "ij,jkz->ikz"; // Modify einsum string for dual computation
        auto expected_d = torch::einsum(darg, {first, second.d});
        EXPECT_TRUE(torch::allclose(result.d, expected_d));
    }
}

// Test: Incompatible shapes for einsum
TEST(TensorDualTest, EinsumTensorAndTensorDualIncompatibleShapes) {
    // Input Tensor and TensorDual with incompatible shapes
    torch::Tensor first = torch::randn({2, 3});
    TensorDual second(torch::randn({4, 5}), torch::randn({4, 5, 6}));

    // Valid einsum string
    std::string einsum_str = "ij,jk->ik";

    EXPECT_THROW(
        {
            try {
                TensorDual::einsum(einsum_str, first, second);
            } catch (const c10::Error& e) {
                // Torch throws a c10::Error for incompatible shapes
                SUCCEED();
                throw;
            }
        },
        c10::Error
    );
}


// Test: Basic functionality of einsum with TensorDual and Tensor
TEST(TensorDualTest, EinsumTensorDualAndTensorBasic) {
    // Input TensorDual and Tensor
    TensorDual first(torch::randn({2, 3}), torch::randn({2, 3, 4}));
    torch::Tensor second = torch::randn({3, 4});

    // Einsum operation
    std::string einsum_str = "ij,jk->ik";
    TensorDual result = TensorDual::einsum(einsum_str, first, second);

    // Expected real part
    auto expected_r = torch::einsum(einsum_str, {first.r, second});
    EXPECT_TRUE(torch::allclose(result.r, expected_r));

    // Expected dual part
    auto darg = "ijz,jk->ikz"; // Modify einsum string for dual computation
    auto expected_d = torch::einsum(darg, {first.d, second});
    EXPECT_TRUE(torch::allclose(result.d, expected_d));
}

// Test: Invalid einsum string
TEST(TensorDualTest, EinsumTensorDualAndTensorInvalidString) {
    // Input TensorDual and Tensor
    TensorDual first(torch::randn({2, 3}), torch::randn({2, 3, 4}));
    torch::Tensor second = torch::randn({3, 4});

    // Invalid einsum string (missing ',' or '->')
    std::string einsum_str = "ij jk";

    EXPECT_THROW(
        {
            try {
                TensorDual::einsum(einsum_str, first, second);
            } catch (const std::invalid_argument& e) {
                EXPECT_STREQ("Einsum string must contain ',' and '->' in the correct order.", e.what());
                throw;
            }
        },
        std::invalid_argument
    );
}



// Test: Edge case - Einsum operation with identity matrix
TEST(TensorDualTest, EinsumTensorDualAndTensorIdentityMatrix) {
    // Input TensorDual and Tensor
    TensorDual first(torch::randn({3, 3}), torch::randn({3, 3, 2}));
    torch::Tensor second = torch::eye(3); // Identity matrix

    // Einsum operation
    std::string einsum_str = "ij,jk->ik";
    TensorDual result = TensorDual::einsum(einsum_str, first, second);

    // Expected real part
    auto expected_r = torch::einsum(einsum_str, {first.r, second});
    EXPECT_TRUE(torch::allclose(result.r, expected_r));

    // Expected dual part
    auto darg = "ijz,jk->ikz"; // Modify einsum string for dual computation
    auto expected_d = torch::einsum(darg, {first.d, second});
    EXPECT_TRUE(torch::allclose(result.d, expected_d));
}

// Test: Device-specific tensors (e.g., GPU)
TEST(TensorDualTest, EinsumTensorDualAndTensorGpuTensors) {
    if (torch::cuda::is_available()) {
        // Input TensorDual and Tensor on GPU
        TensorDual first(
            torch::randn({2, 3}, torch::device(torch::kCUDA)),
            torch::randn({2, 3, 4}, torch::device(torch::kCUDA))
        );
        torch::Tensor second = torch::randn({3, 4}, torch::device(torch::kCUDA));

        // Einsum operation
        std::string einsum_str = "ij,jk->ik";
        TensorDual result = TensorDual::einsum(einsum_str, first, second);

        // Expected real part
        auto expected_r = torch::einsum(einsum_str, {first.r, second});
        EXPECT_TRUE(torch::allclose(result.r, expected_r));

        // Expected dual part
        auto darg = "ijz,jk->ikz"; // Modify einsum string for dual computation
        auto expected_d = torch::einsum(darg, {first.d, second});
        EXPECT_TRUE(torch::allclose(result.d, expected_d));
    }
}

// Test: Incompatible shapes for einsum
TEST(TensorDualTest, EinsumTensorDualAndTensorIncompatibleShapes) {
    // Input TensorDual and Tensor with incompatible shapes
    TensorDual first(torch::randn({2, 3}), torch::randn({2, 3, 4}));
    torch::Tensor second = torch::randn({5, 6});

    // Valid einsum string
    std::string einsum_str = "ij,jk->ik";

    EXPECT_THROW(
        {
            try {
                TensorDual::einsum(einsum_str, first, second);
            } catch (const c10::Error& e) {
                // Torch throws a c10::Error for incompatible shapes
                SUCCEED();
                throw;
            }
        },
        c10::Error
    );
}


// Test: Basic functionality of generalized einsum
TEST(TensorDualTest, GeneralizedEinsumBasic) {
    // Input TensorDual objects
    //They have to be the same shape
    TensorDual td1(torch::randn({2, 3}), torch::randn({2, 3, 10}));
    TensorDual td2(torch::randn({2, 3}), torch::randn({2, 3, 10}));
    TensorDual td3(torch::randn({2, 3}), torch::randn({2, 3, 10}));
    std::vector<TensorDual> tensors = {td1, td2, td3};

    // Einsum operation
    std::string einsum_str = "mi,mi,mi->mi";
    TensorDual result = TensorDual::einsum(einsum_str, tensors);

    // Validate the real part
    auto expected_r = torch::einsum(einsum_str, {td1.r, td2.r, td3.r});
    EXPECT_TRUE(torch::allclose(result.r, expected_r));

    // Validate the dual part
    auto darg1 = "miz,mi,mi->miz";
    auto d1 = torch::einsum(darg1, {td1.d, td2.r, td3.r});

    auto darg2 = "mi,miz,mi->miz";
    auto d2 = torch::einsum(darg2, {td1.r, td2.d, td3.r});

    auto darg3 = "mi,mi,miz->miz";
    auto d3 = torch::einsum(darg3, {td1.r, td2.r, td3.d});

    auto expected_d = d1 + d2 + d3;
    EXPECT_TRUE(torch::allclose(result.d, expected_d));
}

// Test: Invalid einsum string - Missing '->'
TEST(TensorDualTest, GeneralizedEinsumInvalidStringMissingArrow) {
    // Input TensorDual objects
    TensorDual td1(torch::randn({2, 3}), torch::randn({2, 3, 4}));
    TensorDual td2(torch::randn({3, 4}), torch::randn({3, 4, 5}));
    std::vector<TensorDual> tensors = {td1, td2};

    // Invalid einsum string
    std::string einsum_str = "ij,jk";

    EXPECT_THROW(
        {
            try {
                TensorDual::einsum(einsum_str, tensors);
            } catch (const std::invalid_argument& e) {
                EXPECT_STREQ("Einsum string must contain '->'.", e.what());
                throw;
            }
        },
        std::invalid_argument
    );
}

// Test: Invalid einsum string - Contains 'z'
TEST(TensorDualTest, GeneralizedEinsumInvalidStringContainsZ) {
    // Input TensorDual objects
    TensorDual td1(torch::randn({2, 3}), torch::randn({2, 3, 4}));
    TensorDual td2(torch::randn({3, 4}), torch::randn({3, 4, 5}));
    std::vector<TensorDual> tensors = {td1, td2};

    // Invalid einsum string
    std::string einsum_str = "ijz,jk->ik";

    EXPECT_THROW(
        {
            try {
                TensorDual::einsum(einsum_str, tensors);
            } catch (const std::invalid_argument& e) {
                EXPECT_STREQ("Character 'z' is reserved for dual dimensions and cannot appear in the einsum string.", e.what());
                throw;
            }
        },
        std::invalid_argument
    );
}

// Test: Empty input vector
TEST(TensorDualTest, GeneralizedEinsumEmptyInput) {
    // Empty input tensor vector
    std::vector<TensorDual> tensors;

    // Valid einsum string
    std::string einsum_str = "ij,jk->ik";

    EXPECT_THROW(
        {
            try {
                TensorDual::einsum(einsum_str, tensors);
            } catch (const std::invalid_argument& e) {
                EXPECT_STREQ("Input vector `tensors` must not be empty.", e.what());
                throw;
            }
        },
        std::invalid_argument
    );
}


// Test: Device-specific tensors (e.g., GPU)
TEST(TensorDualTest, GeneralizedEinsumGpuTensors) {
    if (torch::cuda::is_available()) {
        // Input TensorDual objects on GPU
        TensorDual td1(
            torch::randn({2, 3}, torch::device(torch::kCUDA)),
            torch::randn({2, 3, 4}, torch::device(torch::kCUDA))
        );
        TensorDual td2(
            torch::randn({2, 3}, torch::device(torch::kCUDA)),
            torch::randn({2, 3, 4}, torch::device(torch::kCUDA))
        );
        std::vector<TensorDual> tensors = {td1, td2};

        // Einsum operation
        std::string einsum_str = "mi,mi->mi";
        TensorDual result = TensorDual::einsum(einsum_str, tensors);

        // Validate the real part
        auto expected_r = torch::einsum(einsum_str, {td1.r, td2.r});
        EXPECT_TRUE(torch::allclose(result.r, expected_r));

        // Validate the dual part
        auto darg1 = "miz,mi->miz";
        auto d1 = torch::einsum(darg1, {td1.d, td2.r});

        auto darg2 = "mi,miz->miz";
        auto d2 = torch::einsum(darg2, {td1.r, td2.d});

        auto expected_d = d1 + d2;
        EXPECT_TRUE(torch::allclose(result.d, expected_d));
    }
}


// Test: Basic functionality of where
TEST(TensorDualTest, WhereBasic) {
    // Input condition tensor
    auto cond = torch::tensor({{1, 0}, {0, 1}}, torch::kBool);

    // Input TensorDual objects
    TensorDual x(torch::tensor({{1.0, 2.0}, {3.0, 4.0}}),
                 torch::tensor({{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}}));
    TensorDual y(torch::tensor({{10.0, 20.0}, {30.0, 40.0}}),
                 torch::tensor({{{10.0, 20.0}, {30.0, 40.0}}, {{50.0, 60.0}, {70.0, 80.0}}}));

    // Perform where operation
    TensorDual result = TensorDual::where(cond, x, y);

    // Expected real and dual parts
    auto expected_r = torch::where(cond, x.r, y.r);
    auto expected_d = torch::where(cond.unsqueeze(2).expand_as(x.d), x.d, y.d);

    // Validate the result
    EXPECT_TRUE(torch::allclose(result.r, expected_r));
    EXPECT_TRUE(torch::allclose(result.d, expected_d));
}

// Test: Condition tensor with broadcasting
TEST(TensorDualTest, WhereConditionBroadcasting) {
    // Input condition tensor (1D)
    auto cond = torch::tensor({1, 0}, torch::kBool);

    // Input TensorDual objects
    TensorDual x(torch::randn({2, 3}), torch::randn({2, 3, 4}));
    TensorDual y(torch::randn({2, 3}), torch::randn({2, 3, 4}));

    // Perform where operation
    TensorDual result = TensorDual::where(cond, x, y);

    // Broadcasted condition tensors
    auto condr = cond.unsqueeze(1).expand_as(x.r);
    auto condd = cond.unsqueeze(1).unsqueeze(2).expand_as(x.d);

    // Expected real and dual parts
    auto expected_r = torch::where(condr, x.r, y.r);
    auto expected_d = torch::where(condd, x.d, y.d);

    // Validate the result
    EXPECT_TRUE(torch::allclose(result.r, expected_r));
    EXPECT_TRUE(torch::allclose(result.d, expected_d));
}

// Test: Condition tensor with too many dimensions
TEST(TensorDualTest, WhereConditionTooManyDimensions) {
    // Input condition tensor with excessive dimensions
    auto cond = torch::randn({1, 1, 0}, torch::kBool);

    // Input TensorDual objects
    TensorDual x(torch::randn({2, 2}), torch::randn({2, 2, 3}));
    TensorDual y(torch::randn({2, 2}), torch::randn({2, 2, 3}));

    EXPECT_THROW(
        {
            try {
                TensorDual::where(cond, x, y);
            } catch (const std::invalid_argument& e) {
                EXPECT_STREQ("Condition tensor has too many dimensions.", e.what());
                throw;
            }
        },
        std::invalid_argument
    );
}


// Test: GPU tensors
TEST(TensorDualTest, WhereGpuTensors) {
    if (torch::cuda::is_available()) {
        // Input condition tensor
        auto cond = torch::tensor({{1, 0}, {0, 1}}, torch::kBool).to(torch::kCUDA);

        // Input TensorDual objects on GPU
        TensorDual x(torch::randn({2, 2}, torch::kCUDA), torch::randn({2, 2, 3}, torch::kCUDA));
        TensorDual y(torch::randn({2, 2}, torch::kCUDA), torch::randn({2, 2, 3}, torch::kCUDA));

        // Perform where operation
        TensorDual result = TensorDual::where(cond, x, y);

        // Expected real and dual parts
        auto expected_r = torch::where(cond, x.r, y.r);
        auto expected_d = torch::where(cond.unsqueeze(2).expand_as(x.d), x.d, y.d);

        // Validate the result
        EXPECT_TRUE(torch::allclose(result.r, expected_r));
        EXPECT_TRUE(torch::allclose(result.d, expected_d));
    }
}

// Test: Mismatched shapes
TEST(TensorDualTest, WhereShapeMismatch) {
    // Input condition tensor
    auto cond = torch::tensor({{1, 0}}, torch::kBool);

    // Input TensorDual objects with mismatched shapes
    TensorDual x(torch::randn({2, 2}), torch::randn({2, 2, 3}));
    TensorDual y(torch::randn({3, 2}), torch::randn({3, 2, 3}));

    EXPECT_THROW(
        {
            try {
                TensorDual::where(cond, x, y);
            } catch (const c10::Error& e) {
                SUCCEED();
                throw;
            }
        },
        c10::Error
    );
}

// Test: Basic functionality of sum
TEST(TensorDualTest, SumBasic) {
    // Input TensorDual object
    TensorDual x(torch::tensor({{1.0, 2.0}, {3.0, 4.0}}),
                 torch::tensor({{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}}));

    // Perform sum along dimension 1
    TensorDual result = TensorDual::sum(x);

    // Expected real and dual parts
    auto expected_r = torch::sum(x.r, /*dim=*/1, /*keepdim=*/true);
    auto expected_d = torch::sum(x.d, /*dim=*/1, /*keepdim=*/true);

    // Validate the result
    EXPECT_TRUE(torch::allclose(result.r, expected_r));
    EXPECT_TRUE(torch::allclose(result.d, expected_d));
}

// Test: Sum along default dimension
TEST(TensorDualTest, SumDefaultDimension) {
    // Input TensorDual object
    TensorDual x(torch::randn({3, 4}), torch::randn({3, 4, 5}));

    // Perform sum along the default dimension (1)
    TensorDual result = TensorDual::sum(x);

    // Expected real and dual parts
    auto expected_r = torch::sum(x.r, /*dim=*/1, /*keepdim=*/true);
    auto expected_d = torch::sum(x.d, /*dim=*/1, /*keepdim=*/true);

    // Validate the result
    EXPECT_TRUE(torch::allclose(result.r, expected_r));
    EXPECT_TRUE(torch::allclose(result.d, expected_d));
}



// Test: GPU tensors
TEST(TensorDualTest, SumGpuTensors) {
    if (torch::cuda::is_available()) {
        // Input TensorDual object on GPU
        TensorDual x(torch::randn({2, 3}, torch::kCUDA), torch::randn({2, 3, 4}, torch::kCUDA));

        // Perform sum along dimension 1
        TensorDual result = TensorDual::sum(x);

        // Expected real and dual parts
        auto expected_r = torch::sum(x.r, /*dim=*/1, /*keepdim=*/true);
        auto expected_d = torch::sum(x.d, /*dim=*/1, /*keepdim=*/true);

        // Validate the result
        EXPECT_TRUE(torch::allclose(result.r, expected_r));
        EXPECT_TRUE(torch::allclose(result.d, expected_d));
    }
}

// Test: Sum with high-dimensional tensors
TEST(TensorDualTest, SumHighDimensionalTensors) {
    // Input TensorDual object with high-dimensional tensors
    TensorDual x(torch::randn({100, 10000}), torch::randn({100, 10000, 10}));

    // Perform sum along dimension 3
    TensorDual result = TensorDual::sum(x);

    // Expected real and dual parts
    auto expected_r = torch::sum(x.r, /*dim=*/1, /*keepdim=*/true);
    auto expected_d = torch::sum(x.d, /*dim=*/1, /*keepdim=*/true);

    // Validate the result
    EXPECT_TRUE(torch::allclose(result.r, expected_r));
    EXPECT_TRUE(torch::allclose(result.d, expected_d));
}

// Test: Basic functionality of normL2
TEST(TensorDualTest, NormL2Basic) {
    // Input TensorDual object
    TensorDual x(torch::tensor({{3.0, 4.0}, {6.0, 8.0}}), 
                 torch::tensor({{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}}));

    // Perform normL2 operation
    TensorDual result = x.normL2();

    // Expected real part (L2 norm along dimension 1)
    auto expected_r = torch::norm(x.r, 2, /*dim=*/1, /*keepdim=*/true);

    // Avoid division by zero
    auto expected_r_clamped = expected_r.clamp_min(1e-12);

    // Compute gradient with respect to the real part
    auto grad_r = x.r / expected_r_clamped;

    // Expected dual part
    auto expected_d = torch::einsum("mi,mij->mj", {grad_r, x.d}).unsqueeze(1);

    // Validate the result
    EXPECT_TRUE(torch::allclose(result.r, expected_r));
    EXPECT_TRUE(torch::allclose(result.d, expected_d));
}

// Test: L2 norm with zero tensor
TEST(TensorDualTest, NormL2ZeroTensor) {
    // Input TensorDual object with zero tensors
    TensorDual x(torch::zeros({2, 2}), torch::zeros({2, 2, 3}));

    // Perform normL2 operation
    TensorDual result = x.normL2();

    // Expected real part (L2 norm is clamped to 1e-12)
    auto expected_r = torch::zeros({2, 1}).clamp_min(1e-12);

    // Expected dual part (all zeros since input dual part is zero)
    auto expected_d = torch::zeros({2, 1, 3});

    // Validate the result
    EXPECT_TRUE(torch::allclose(result.r, expected_r));
    EXPECT_TRUE(torch::allclose(result.d, expected_d));
}


// Test: High-dimensional tensors
TEST(TensorDualTest, NormL2HighDimensionalTensors) {
    // Input TensorDual object with high-dimensional tensors
    TensorDual x(torch::randn({3, 5}), torch::randn({3, 5, 4}));

    // Perform normL2 operation
    TensorDual result = x.normL2();

    // Expected real part (L2 norm along dimension 1)
    auto expected_r = torch::norm(x.r, 2, /*dim=*/1, /*keepdim=*/true);

    // Avoid division by zero
    auto expected_r_clamped = expected_r.clamp_min(1e-12);

    // Compute gradient with respect to the real part
    auto grad_r = x.r / expected_r_clamped;

    // Expected dual part
    auto expected_d = torch::einsum("mi,mij->mj", {grad_r, x.d}).unsqueeze(1);

    // Validate the result
    EXPECT_TRUE(torch::allclose(result.r, expected_r));
    EXPECT_TRUE(torch::allclose(result.d, expected_d));
}

// Test: GPU tensors
TEST(TensorDualTest, NormL2GpuTensors) {
    if (torch::cuda::is_available()) {
        // Input TensorDual object on GPU
        TensorDual x(torch::randn({2, 3}, torch::kCUDA), torch::randn({2, 3, 4}, torch::kCUDA));

        // Perform normL2 operation
        TensorDual result = x.normL2();

        // Expected real part (L2 norm along dimension 1)
        auto expected_r = torch::norm(x.r, 2, /*dim=*/1, /*keepdim=*/true);

        // Avoid division by zero
        auto expected_r_clamped = expected_r.clamp_min(1e-12);

        // Compute gradient with respect to the real part
        auto grad_r = x.r / expected_r_clamped;

        // Expected dual part
        auto expected_d = torch::einsum("mi,mij->mj", {grad_r, x.d}).unsqueeze(1);

        // Validate the result
        EXPECT_TRUE(torch::allclose(result.r, expected_r));
        EXPECT_TRUE(torch::allclose(result.d, expected_d));
    }
}


// Test: Basic functionality of unary negation operator
TEST(TensorDualTest, UnaryNegationBasic) {
    // Input TensorDual object
    TensorDual x(torch::tensor({{1.0, -2.0}, {3.0, -4.0}}), 
                 torch::tensor({{{1.0, -2.0}, {3.0, -4.0}}, {{5.0, -6.0}, {7.0, -8.0}}}));

    // Perform unary negation
    TensorDual result = -x;

    // Expected negated real and dual parts
    auto expected_r = -x.r;
    auto expected_d = -x.d;

    // Validate the result
    EXPECT_TRUE(torch::allclose(result.r, expected_r));
    EXPECT_TRUE(torch::allclose(result.d, expected_d));
}

// Test: Unary negation on zero tensors
TEST(TensorDualTest, UnaryNegationZeroTensor) {
    // Input TensorDual object with zero tensors
    TensorDual x(torch::zeros({2, 2}), torch::zeros({2, 2, 3}));

    // Perform unary negation
    TensorDual result = -x;

    // Expected negated real and dual parts (still zeros)
    auto expected_r = torch::zeros({2, 2});
    auto expected_d = torch::zeros({2, 2, 3});

    // Validate the result
    EXPECT_TRUE(torch::allclose(result.r, expected_r));
    EXPECT_TRUE(torch::allclose(result.d, expected_d));
}


// Test: Unary negation on high-dimensional tensors
TEST(TensorDualTest, UnaryNegationHighDimensionalTensors) {
    // Input TensorDual object with high-dimensional tensors
    TensorDual x(torch::randn({3, 4}), torch::randn({3, 4, 5}));

    // Perform unary negation
    TensorDual result = -x;

    // Expected negated real and dual parts
    auto expected_r = -x.r;
    auto expected_d = -x.d;

    // Validate the result
    EXPECT_TRUE(torch::allclose(result.r, expected_r));
    EXPECT_TRUE(torch::allclose(result.d, expected_d));
}

// Test: GPU tensors
TEST(TensorDualTest, UnaryNegationGpuTensors) {
    if (torch::cuda::is_available()) {
        // Input TensorDual object on GPU
        TensorDual x(torch::randn({2, 3}, torch::kCUDA), torch::randn({2, 3, 4}, torch::kCUDA));

        // Perform unary negation
        TensorDual result = -x;

        // Expected negated real and dual parts
        auto expected_r = -x.r;
        auto expected_d = -x.d;

        // Validate the result
        EXPECT_TRUE(torch::allclose(result.r, expected_r));
        EXPECT_TRUE(torch::allclose(result.d, expected_d));
    }
}


// Test: Basic functionality of clone
TEST(TensorDualTest, CloneBasic) {
    // Input TensorDual object
    TensorDual x(torch::tensor({{1.0, 2.0}, {3.0, 4.0}}), 
                 torch::tensor({{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}}));

    // Clone the TensorDual
    TensorDual cloned = x.clone();

    // Validate that the cloned object is equal to the original
    EXPECT_TRUE(torch::allclose(cloned.r, x.r));
    EXPECT_TRUE(torch::allclose(cloned.d, x.d));

    // Validate that the cloned object is independent of the original
    cloned.r.add_(1.0);
    cloned.d.add_(1.0);

    EXPECT_FALSE(torch::allclose(cloned.r, x.r));
    EXPECT_FALSE(torch::allclose(cloned.d, x.d));
}

// Test: Clone zero TensorDual
TEST(TensorDualTest, CloneZeroTensor) {
    // Input TensorDual object with zero tensors
    TensorDual x(torch::zeros({2, 2}), torch::zeros({2, 2, 3}));

    // Clone the TensorDual
    TensorDual cloned = x.clone();

    // Validate that the cloned object is equal to the original
    EXPECT_TRUE(torch::allclose(cloned.r, x.r));
    EXPECT_TRUE(torch::allclose(cloned.d, x.d));

    // Validate that the cloned object is independent of the original
    cloned.r.add_(1.0);
    cloned.d.add_(1.0);

    EXPECT_FALSE(torch::allclose(cloned.r, x.r));
    EXPECT_FALSE(torch::allclose(cloned.d, x.d));
}


// Test: Clone high-dimensional TensorDual
TEST(TensorDualTest, CloneHighDimensionalTensors) {
    // Input TensorDual object with high-dimensional tensors
    TensorDual x(torch::randn({100, 100}), torch::randn({100,100,1000}));

    // Clone the TensorDual
    TensorDual cloned = x.clone();

    // Validate that the cloned object is equal to the original
    EXPECT_TRUE(torch::allclose(cloned.r, x.r));
    EXPECT_TRUE(torch::allclose(cloned.d, x.d));

    // Validate that the cloned object is independent of the original
    cloned.r.add_(1.0);
    cloned.d.add_(1.0);

    EXPECT_FALSE(torch::allclose(cloned.r, x.r));
    EXPECT_FALSE(torch::allclose(cloned.d, x.d));
}

// Test: GPU tensors
TEST(TensorDualTest, CloneGpuTensors) {
    if (torch::cuda::is_available()) {
        // Input TensorDual object on GPU
        TensorDual x(torch::randn({2, 3}, torch::kCUDA), torch::randn({2, 3, 4}, torch::kCUDA));

        // Clone the TensorDual
        TensorDual cloned = x.clone();

        // Validate that the cloned object is equal to the original
        EXPECT_TRUE(torch::allclose(cloned.r, x.r));
        EXPECT_TRUE(torch::allclose(cloned.d, x.d));

        // Validate that the cloned object is independent of the original
        cloned.r.add_(1.0);
        cloned.d.add_(1.0);

        EXPECT_FALSE(torch::allclose(cloned.r, x.r));
        EXPECT_FALSE(torch::allclose(cloned.d, x.d));
    }
}


// Test: Basic functionality of addition operator
TEST(TensorDualTest, AdditionBasic) {
    // Input TensorDual objects
    TensorDual x(torch::tensor({{1.0, 2.0}, {3.0, 4.0}}), 
                 torch::tensor({{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}}));
    TensorDual y(torch::tensor({{10.0, 20.0}, {30.0, 40.0}}), 
                 torch::tensor({{{10.0, 20.0}, {30.0, 40.0}}, {{50.0, 60.0}, {70.0, 80.0}}}));

    // Perform addition
    TensorDual result = x + y;

    // Expected real and dual parts
    auto expected_r = x.r + y.r;
    auto expected_d = x.d + y.d;

    // Validate the result
    EXPECT_TRUE(torch::allclose(result.r, expected_r));
    EXPECT_TRUE(torch::allclose(result.d, expected_d));
}

// Test: Addition with zero TensorDual
TEST(TensorDualTest, AdditionZeroTensor) {
    // Input TensorDual objects
    TensorDual x(torch::tensor({{1.0, 2.0}, {3.0, 4.0}}), 
                 torch::tensor({{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}}));
    TensorDual zero(torch::zeros({2, 2}), torch::zeros({2, 2, 2}));

    // Perform addition
    TensorDual result = x + zero;

    // Expected real and dual parts
    auto expected_r = x.r;
    auto expected_d = x.d;

    // Validate the result
    EXPECT_TRUE(torch::allclose(result.r, expected_r));
    EXPECT_TRUE(torch::allclose(result.d, expected_d));
}


// Test: Dimension mismatch
TEST(TensorDualTest, AdditionDimensionMismatch) {
    // Input TensorDual objects with mismatched dimensions
    TensorDual x(torch::randn({2, 3}), torch::randn({2, 3, 4}));
    TensorDual y(torch::randn({3, 2}), torch::randn({3, 2, 4}));

    EXPECT_THROW(
        {
            try {
                TensorDual result = x + y;
            } catch (const std::invalid_argument& e) {
                EXPECT_STREQ("Dimension mismatch: Real and dual tensors of both TensorDual objects must have the same shape.", e.what());
                throw;
            }
        },
        std::invalid_argument
    );
}

// Test: High-dimensional tensors
TEST(TensorDualTest, AdditionHighDimensionalTensors) {
    // Input TensorDual objects with high-dimensional tensors
    TensorDual x(torch::randn({3, 4000}), torch::randn({3, 4000, 10}));
    TensorDual y(torch::randn({3, 4000}), torch::randn({3, 4000, 10}));

    // Perform addition
    TensorDual result = x + y;

    // Expected real and dual parts
    auto expected_r = x.r + y.r;
    auto expected_d = x.d + y.d;

    // Validate the result
    EXPECT_TRUE(torch::allclose(result.r, expected_r));
    EXPECT_TRUE(torch::allclose(result.d, expected_d));
}

// Test: GPU tensors
TEST(TensorDualTest, AdditionGpuTensors) {
    if (torch::cuda::is_available()) {
        // Input TensorDual objects on GPU
        TensorDual x(torch::randn({2, 3}, torch::kCUDA), torch::randn({2, 3, 4}, torch::kCUDA));
        TensorDual y(torch::randn({2, 3}, torch::kCUDA), torch::randn({2, 3, 4}, torch::kCUDA));

        // Perform addition
        TensorDual result = x + y;

        // Expected real and dual parts
        auto expected_r = x.r + y.r;
        auto expected_d = x.d + y.d;

        // Validate the result
        EXPECT_TRUE(torch::allclose(result.r, expected_r));
        EXPECT_TRUE(torch::allclose(result.d, expected_d));
    }
}



// Test: Basic functionality of addition with torch::Tensor
TEST(TensorDualTest, AdditionWithTensorBasic) {
    // Input TensorDual object and torch::Tensor
    TensorDual x(torch::tensor({{1.0, 2.0}, {3.0, 4.0}}), 
                 torch::tensor({{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}}));
    torch::Tensor other = torch::tensor({{10.0, 20.0}, {30.0, 40.0}});

    // Perform addition
    TensorDual result = x + other;

    // Expected real and dual parts
    auto expected_r = x.r + other;
    auto expected_d = x.d.clone();

    // Validate the result
    EXPECT_TRUE(torch::allclose(result.r, expected_r));
    EXPECT_TRUE(torch::allclose(result.d, expected_d));
}

// Test: Addition with zero tensor
TEST(TensorDualTest, AdditionWithZeroTensor) {
    // Input TensorDual object and torch::Tensor
    TensorDual x(torch::tensor({{1.0, 2.0}, {3.0, 4.0}}), 
                 torch::tensor({{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}}));
    torch::Tensor other = torch::zeros({2, 2});

    // Perform addition
    TensorDual result = x + other;

    // Expected real and dual parts
    auto expected_r = x.r;
    auto expected_d = x.d.clone();

    // Validate the result
    EXPECT_TRUE(torch::allclose(result.r, expected_r));
    EXPECT_TRUE(torch::allclose(result.d, expected_d));
}

// Test: Undefined torch::Tensor
TEST(TensorDualTest, AdditionWithUndefinedTensor) {
    // Input TensorDual object and undefined torch::Tensor
    TensorDual x(torch::randn({2, 2}), torch::randn({2, 2, 3}));
    torch::Tensor other;

    EXPECT_THROW(
        {
            try {
                TensorDual result = x + other;
            } catch (const std::invalid_argument& e) {
                EXPECT_STREQ("Cannot add TensorDual with an undefined tensor.", e.what());
                throw;
            }
        },
        std::invalid_argument
    );
}

// Test: Dimension mismatch
TEST(TensorDualTest, AdditionWithTensorDimensionMismatch) {
    // Input TensorDual object and mismatched torch::Tensor
    TensorDual x(torch::randn({2, 3}), torch::randn({2, 3, 4}));
    torch::Tensor other = torch::randn({3, 2});

    EXPECT_THROW(
        {
            try {
                TensorDual result = x + other;
            } catch (const std::invalid_argument& e) {
                EXPECT_STREQ("Dimension mismatch: The other tensor must have the same shape as the real part of the TensorDual.", e.what());
                throw;
            }
        },
        std::invalid_argument
    );
}

// Test: High-dimensional tensors
TEST(TensorDualTest, AdditionWithTensorHighDimensional) {
    // Input TensorDual object and torch::Tensor
    TensorDual x(torch::randn({3, 4000}), torch::randn({3, 4000, 6}));
    torch::Tensor other = torch::randn({3, 4000});

    // Perform addition
    TensorDual result = x + other;

    // Expected real and dual parts
    auto expected_r = x.r + other;
    auto expected_d = x.d.clone();

    // Validate the result
    EXPECT_TRUE(torch::allclose(result.r, expected_r));
    EXPECT_TRUE(torch::allclose(result.d, expected_d));
}

// Test: GPU tensors
TEST(TensorDualTest, AdditionWithTensorGpuTensors) {
    if (torch::cuda::is_available()) {
        // Input TensorDual object and torch::Tensor on GPU
        TensorDual x(torch::randn({2, 3}, torch::kCUDA), torch::randn({2, 3, 4}, torch::kCUDA));
        torch::Tensor other = torch::randn({2, 3}, torch::kCUDA);

        // Perform addition
        TensorDual result = x + other;

        // Expected real and dual parts
        auto expected_r = x.r + other;
        auto expected_d = x.d.clone();

        // Validate the result
        EXPECT_TRUE(torch::allclose(result.r, expected_r));
        EXPECT_TRUE(torch::allclose(result.d, expected_d));
    }
}


// Test: Basic functionality of addition with scalar
TEST(TensorDualTest, AdditionWithScalarBasic) {
    // Input TensorDual object and scalar
    TensorDual x(torch::tensor({{1.0, 2.0}, {3.0, 4.0}}), 
                 torch::tensor({{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}}));
    double scalar = 10.0;

    // Perform addition
    TensorDual result = x + scalar;

    // Expected real and dual parts
    auto expected_r = x.r + scalar;
    auto expected_d = x.d.clone();

    // Validate the result
    EXPECT_TRUE(torch::allclose(result.r, expected_r));
    EXPECT_TRUE(torch::allclose(result.d, expected_d));
}

// Test: Addition with zero scalar
TEST(TensorDualTest, AdditionWithZeroScalar) {
    // Input TensorDual object and zero scalar
    TensorDual x(torch::tensor({{1.0, 2.0}, {3.0, 4.0}}), 
                 torch::tensor({{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}}));
    double scalar = 0.0;

    // Perform addition
    TensorDual result = x + scalar;

    // Expected real and dual parts
    auto expected_r = x.r; // Unchanged real part
    auto expected_d = x.d.clone(); // Unchanged dual part

    // Validate the result
    EXPECT_TRUE(torch::allclose(result.r, expected_r));
    EXPECT_TRUE(torch::allclose(result.d, expected_d));
}

// Test: Addition with a negative scalar
TEST(TensorDualTest, AdditionWithNegativeScalar) {
    // Input TensorDual object and negative scalar
    TensorDual x(torch::tensor({{1.0, 2.0}, {3.0, 4.0}}), 
                 torch::tensor({{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}}));
    double scalar = -5.0;

    // Perform addition
    TensorDual result = x + scalar;

    // Expected real and dual parts
    auto expected_r = x.r + scalar;
    auto expected_d = x.d.clone();

    // Validate the result
    EXPECT_TRUE(torch::allclose(result.r, expected_r));
    EXPECT_TRUE(torch::allclose(result.d, expected_d));
}

// Test: Addition with a high-dimensional TensorDual
TEST(TensorDualTest, AdditionWithScalarHighDimensional) {
    // Input TensorDual object and scalar
    TensorDual x(torch::randn({3, 4000}), torch::randn({3, 4000, 5}));
    double scalar = 3.5;

    // Perform addition
    TensorDual result = x + scalar;

    // Expected real and dual parts
    auto expected_r = x.r + scalar;
    auto expected_d = x.d.clone();

    // Validate the result
    EXPECT_TRUE(torch::allclose(result.r, expected_r));
    EXPECT_TRUE(torch::allclose(result.d, expected_d));
}

// Test: GPU tensors
TEST(TensorDualTest, AdditionWithScalarGpuTensors) {
    if (torch::cuda::is_available()) {
        // Input TensorDual object on GPU and scalar
        TensorDual x(torch::randn({2, 3}, torch::kCUDA), torch::randn({2, 3, 4}, torch::kCUDA));
        double scalar = 7.0;

        // Perform addition
        TensorDual result = x + scalar;

        // Expected real and dual parts
        auto expected_r = x.r + scalar;
        auto expected_d = x.d.clone();

        // Validate the result
        EXPECT_TRUE(torch::allclose(result.r, expected_r));
        EXPECT_TRUE(torch::allclose(result.d, expected_d));
    }
}


// Test: Basic functionality of subtraction with torch::Tensor
TEST(TensorDualTest, SubtractionWithTensorBasic) {
    // Input TensorDual object and torch::Tensor
    TensorDual x(torch::tensor({{10.0, 20.0}, {30.0, 40.0}}), 
                 torch::tensor({{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}}));
    torch::Tensor other = torch::tensor({{1.0, 2.0}, {3.0, 4.0}});

    // Perform subtraction
    TensorDual result = x - other;

    // Expected real and dual parts
    auto expected_r = x.r - other;
    auto expected_d = x.d.clone();

    // Validate the result
    EXPECT_TRUE(torch::allclose(result.r, expected_r));
    EXPECT_TRUE(torch::allclose(result.d, expected_d));
}

// Test: Subtraction with zero tensor
TEST(TensorDualTest, SubtractionWithZeroTensor) {
    // Input TensorDual object and zero torch::Tensor
    TensorDual x(torch::tensor({{10.0, 20.0}, {30.0, 40.0}}), 
                 torch::tensor({{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}}));
    torch::Tensor other = torch::zeros({2, 2});

    // Perform subtraction
    TensorDual result = x - other;

    // Expected real and dual parts
    auto expected_r = x.r;
    auto expected_d = x.d.clone();

    // Validate the result
    EXPECT_TRUE(torch::allclose(result.r, expected_r));
    EXPECT_TRUE(torch::allclose(result.d, expected_d));
}

// Test: Undefined torch::Tensor
TEST(TensorDualTest, SubtractionWithUndefinedTensor) {
    // Input TensorDual object and undefined torch::Tensor
    TensorDual x(torch::randn({2, 2}), torch::randn({2, 2, 3}));
    torch::Tensor other;

    EXPECT_THROW(
        {
            try {
                TensorDual result = x - other;
            } catch (const std::invalid_argument& e) {
                EXPECT_STREQ("Cannot subtract an undefined tensor from TensorDual.", e.what());
                throw;
            }
        },
        std::invalid_argument
    );
}

// Test: Broadcasting compatibility
TEST(TensorDualTest, SubtractionWithBroadcasting) {
    // Input TensorDual object and a broadcastable torch::Tensor
    TensorDual x(torch::tensor({{10.0, 20.0}, {30.0, 40.0}}), 
                 torch::tensor({{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}}));
    torch::Tensor other = torch::tensor({10.0});

    // Perform subtraction
    TensorDual result = x - other;

    // Expected real and dual parts
    auto expected_r = x.r - other;
    auto expected_d = x.d.clone();

    // Validate the result
    EXPECT_TRUE(torch::allclose(result.r, expected_r));
    EXPECT_TRUE(torch::allclose(result.d, expected_d));
}

// Test: High-dimensional tensors
TEST(TensorDualTest, SubtractionWithTensorHighDimensional) {
    // Input TensorDual object and torch::Tensor
    TensorDual x(torch::randn({3, 4000}), torch::randn({3, 4000, 6}));
    torch::Tensor other = torch::randn({3, 4000});

    // Perform subtraction
    TensorDual result = x - other;

    // Expected real and dual parts
    auto expected_r = x.r - other;
    auto expected_d = x.d.clone();

    // Validate the result
    EXPECT_TRUE(torch::allclose(result.r, expected_r));
    EXPECT_TRUE(torch::allclose(result.d, expected_d));
}


// Test: Basic functionality of subtraction with a scalar
TEST(TensorDualTest, SubtractionWithScalarBasic) {
    // Input TensorDual object and scalar
    TensorDual x(torch::tensor({{10.0, 20.0}, {30.0, 40.0}}), 
                 torch::tensor({{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}}));
    double scalar = 5.0;

    // Perform subtraction
    TensorDual result = x - scalar;

    // Expected real and dual parts
    auto expected_r = x.r - scalar;
    auto expected_d = x.d.clone();

    // Validate the result
    EXPECT_TRUE(torch::allclose(result.r, expected_r));
    EXPECT_TRUE(torch::allclose(result.d, expected_d));
}

// Test: Subtraction with zero scalar
TEST(TensorDualTest, SubtractionWithZeroScalar) {
    // Input TensorDual object and zero scalar
    TensorDual x(torch::tensor({{10.0, 20.0}, {30.0, 40.0}}), 
                 torch::tensor({{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}}));
    double scalar = 0.0;

    // Perform subtraction
    TensorDual result = x - scalar;

    // Expected real and dual parts
    auto expected_r = x.r;
    auto expected_d = x.d.clone();

    // Validate the result
    EXPECT_TRUE(torch::allclose(result.r, expected_r));
    EXPECT_TRUE(torch::allclose(result.d, expected_d));
}

// Test: Subtraction with a negative scalar
TEST(TensorDualTest, SubtractionWithNegativeScalar) {
    // Input TensorDual object and negative scalar
    TensorDual x(torch::tensor({{10.0, 20.0}, {30.0, 40.0}}), 
                 torch::tensor({{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}}));
    double scalar = -5.0;

    // Perform subtraction
    TensorDual result = x - scalar;

    // Expected real and dual parts
    auto expected_r = x.r - scalar;
    auto expected_d = x.d.clone();

    // Validate the result
    EXPECT_TRUE(torch::allclose(result.r, expected_r));
    EXPECT_TRUE(torch::allclose(result.d, expected_d));
}

// Test: Subtraction with high-dimensional tensors
TEST(TensorDualTest, SubtractionWithScalarHighDimensional) {
    // Input TensorDual object and scalar
    TensorDual x(torch::randn({3, 10000}), torch::randn({3, 10000, 6}));
    double scalar = 2.5;

    // Perform subtraction
    TensorDual result = x - scalar;

    // Expected real and dual parts
    auto expected_r = x.r - scalar;
    auto expected_d = x.d.clone();

    // Validate the result
    EXPECT_TRUE(torch::allclose(result.r, expected_r));
    EXPECT_TRUE(torch::allclose(result.d, expected_d));
}

// Test: Subtraction with GPU tensors
TEST(TensorDualTest, SubtractionWithScalarGpuTensors) {
    if (torch::cuda::is_available()) {
        // Input TensorDual object on GPU and scalar
        TensorDual x(torch::randn({2, 3}, torch::kCUDA), torch::randn({2, 3, 4}, torch::kCUDA));
        double scalar = 7.0;

        // Perform subtraction
        TensorDual result = x - scalar;

        // Expected real and dual parts
        auto expected_r = x.r - scalar;
        auto expected_d = x.d.clone();

        // Validate the result
        EXPECT_TRUE(torch::allclose(result.r, expected_r));
        EXPECT_TRUE(torch::allclose(result.d, expected_d));
    }
}

// Test: GPU tensors
TEST(TensorDualTest, SubtractionWithTensorGpuTensors) {
    if (torch::cuda::is_available()) {
        // Input TensorDual object and torch::Tensor on GPU
        TensorDual x(torch::randn({2, 3}, torch::kCUDA), torch::randn({2, 3, 4}, torch::kCUDA));
        torch::Tensor other = torch::randn({2, 3}, torch::kCUDA);

        // Perform subtraction
        TensorDual result = x - other;

        // Expected real and dual parts
        auto expected_r = x.r - other;
        auto expected_d = x.d.clone();

        // Validate the result
        EXPECT_TRUE(torch::allclose(result.r, expected_r));
        EXPECT_TRUE(torch::allclose(result.d, expected_d));
    }
}


// Test: Basic functionality of contiguous
TEST(TensorDualTest, ContiguousBasic) {
    // Input TensorDual object
    TensorDual x(torch::tensor({{1.0, 2.0}, {3.0, 4.0}}).transpose(0, 1), 
                 torch::randn({2, 2, 3}).permute({1, 0, 2}));

    // Validate that the input tensors are not contiguous
    EXPECT_FALSE(x.r.is_contiguous());
    EXPECT_FALSE(x.d.is_contiguous());

    // Perform contiguous operation
    TensorDual result = x.contiguous();

    // Validate that the resulting tensors are contiguous
    EXPECT_TRUE(result.r.is_contiguous());
    EXPECT_TRUE(result.d.is_contiguous());

    // Validate that the data is unchanged
    EXPECT_TRUE(torch::allclose(result.r, x.r));
    EXPECT_TRUE(torch::allclose(result.d, x.d));
}

// Test: Already contiguous tensors
TEST(TensorDualTest, ContiguousAlreadyContiguous) {
    // Input TensorDual object with already contiguous tensors
    TensorDual x(torch::randn({2, 2}), torch::randn({2, 2, 3}));

    // Validate that the input tensors are already contiguous
    EXPECT_TRUE(x.r.is_contiguous());
    EXPECT_TRUE(x.d.is_contiguous());

    // Perform contiguous operation
    TensorDual result = x.contiguous();

    // Validate that the resulting tensors are still contiguous
    EXPECT_TRUE(result.r.is_contiguous());
    EXPECT_TRUE(result.d.is_contiguous());

    // Validate that the data is unchanged
    EXPECT_TRUE(torch::allclose(result.r, x.r));
    EXPECT_TRUE(torch::allclose(result.d, x.d));
}


// Test: High-dimensional tensors
TEST(TensorDualTest, ContiguousHighDimensionalTensors) {
    // Input TensorDual object with high-dimensional tensors
    TensorDual x(torch::randn({10000,2}).permute({1,0}), 
                 torch::randn({10000,2,5}).permute({1,0,2}));

    // Validate that the input tensors are not contiguous
    EXPECT_FALSE(x.r.is_contiguous());
    EXPECT_FALSE(x.d.is_contiguous());

    // Perform contiguous operation
    TensorDual result = x.contiguous();

    // Validate that the resulting tensors are contiguous
    EXPECT_TRUE(result.r.is_contiguous());
    EXPECT_TRUE(result.d.is_contiguous());

    // Validate that the data is unchanged
    EXPECT_TRUE(torch::allclose(result.r, x.r));
    EXPECT_TRUE(torch::allclose(result.d, x.d));
}

// Test: GPU tensors
TEST(TensorDualTest, ContiguousGpuTensors) {
    if (torch::cuda::is_available()) {
        // Input TensorDual object on GPU with non-contiguous tensors
        TensorDual x(torch::randn({3, 2}, torch::kCUDA).transpose(0, 1), 
                     torch::randn({4, 3, 2}, torch::kCUDA).permute({2, 1, 0}));

        // Validate that the input tensors are not contiguous
        EXPECT_FALSE(x.r.is_contiguous());
        EXPECT_FALSE(x.d.is_contiguous());

        // Perform contiguous operation
        TensorDual result = x.contiguous();

        // Validate that the resulting tensors are contiguous
        EXPECT_TRUE(result.r.is_contiguous());
        EXPECT_TRUE(result.d.is_contiguous());

        // Validate that the data is unchanged
        EXPECT_TRUE(torch::allclose(result.r, x.r));
        EXPECT_TRUE(torch::allclose(result.d, x.d));
    }
}


// Test: Basic functionality of multiplication operator
TEST(TensorDualTest, MultiplicationTensorDualBasic) {
    // Input TensorDual objects
    TensorDual x(torch::tensor({{1.0, 2.0}, {3.0, 4.0}}), 
                 torch::tensor({{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}}));
    TensorDual y(torch::tensor({{10.0, 20.0}, {30.0, 40.0}}), 
                 torch::tensor({{{10.0, 20.0}, {30.0, 40.0}}, {{50.0, 60.0}, {70.0, 80.0}}}));

    // Perform multiplication
    TensorDual result = x * y;

    // Expected real and dual parts
    auto expected_r = x.r * y.r;
    auto expected_d = y.r.unsqueeze(-1) * x.d + x.r.unsqueeze(-1) * y.d;

    // Validate the result
    EXPECT_TRUE(torch::allclose(result.r, expected_r));
    EXPECT_TRUE(torch::allclose(result.d, expected_d));
}

// Test: Multiplication with zero TensorDual
TEST(TensorDualTest, MultiplicationTensorDualWithZeroTensorDual) {
    // Input TensorDual objects
    TensorDual x(torch::tensor({{1.0, 2.0}, {3.0, 4.0}}), 
                 torch::tensor({{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}}));
    TensorDual zero(torch::zeros({2, 2}), torch::zeros({2, 2, 2}));

    // Perform multiplication
    TensorDual result = x * zero;

    // Expected real and dual parts
    auto expected_r = x.r * zero.r;  // Should be zeros
    auto expected_d = zero.r.unsqueeze(-1) * x.d + x.r.unsqueeze(-1) * zero.d; // Should be zeros

    // Validate the result
    EXPECT_TRUE(torch::allclose(result.r, expected_r));
    EXPECT_TRUE(torch::allclose(result.d, expected_d));
}


// Test: Dimension mismatch
TEST(TensorDualTest, MultiplicationTensorDualDimensionMismatch) {
    // Input TensorDual objects with mismatched dimensions
    TensorDual x(torch::randn({2, 3}), torch::randn({2, 3, 4}));
    TensorDual y(torch::randn({3, 2}), torch::randn({3, 2, 4}));

    EXPECT_THROW(
        {
            try {
                TensorDual result = x * y;
            } catch (const std::invalid_argument& e) {
                EXPECT_STREQ("Dimension mismatch: Real and dual tensors must have the same shape for multiplication.", e.what());
                throw;
            }
        },
        std::invalid_argument
    );
}

// Test: High-dimensional tensors
TEST(TensorDualTest, MultiplicationTensorDualHighDimensionalTensors) {
    // Input TensorDual objects with high-dimensional tensors
    TensorDual x(torch::randn({3, 10000}), torch::randn({3, 10000, 5}));
    TensorDual y(torch::randn({3, 10000}), torch::randn({3, 10000, 5}));

    // Perform multiplication
    TensorDual result = x * y;

    // Expected real and dual parts
    auto expected_r = x.r * y.r;
    auto expected_d = y.r.unsqueeze(-1) * x.d + x.r.unsqueeze(-1) * y.d;

    // Validate the result
    EXPECT_TRUE(torch::allclose(result.r, expected_r));
    EXPECT_TRUE(torch::allclose(result.d, expected_d));
}

// Test: GPU tensors
TEST(TensorDualTest, MultiplicationTensorDualGpuTensors) {
    if (torch::cuda::is_available()) {
        // Input TensorDual objects on GPU
        TensorDual x(torch::randn({2, 3}, torch::kCUDA), torch::randn({2, 3, 4}, torch::kCUDA));
        TensorDual y(torch::randn({2, 3}, torch::kCUDA), torch::randn({2, 3, 4}, torch::kCUDA));

        // Perform multiplication
        TensorDual result = x * y;

        // Expected real and dual parts
        auto expected_r = x.r * y.r;
        auto expected_d = y.r.unsqueeze(-1) * x.d + x.r.unsqueeze(-1) * y.d;

        // Validate the result
        EXPECT_TRUE(torch::allclose(result.r, expected_r));
        EXPECT_TRUE(torch::allclose(result.d, expected_d));
    }
}


// Test: Basic functionality of multiplication with torch::Tensor
TEST(TensorDualTest, MultiplicationWithTensorBasic) {
    // Input TensorDual object and torch::Tensor
    TensorDual x(torch::tensor({{1.0, 2.0}, {3.0, 4.0}}), 
                 torch::tensor({{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}}));
    torch::Tensor other = torch::tensor({{10.0, 20.0}, {30.0, 40.0}});

    // Perform multiplication
    TensorDual result = x * other;

    // Expected real and dual parts
    auto expected_r = other * x.r;
    auto scaled_other = other.unsqueeze(-1);
    auto expected_d = scaled_other * x.d;

    // Validate the result
    EXPECT_TRUE(torch::allclose(result.r, expected_r));
    EXPECT_TRUE(torch::allclose(result.d, expected_d));
}

// Test: Multiplication with zero tensor
TEST(TensorDualTest, MultiplicationWithZeroTensor) {
    // Input TensorDual object and zero torch::Tensor
    TensorDual x(torch::tensor({{1.0, 2.0}, {3.0, 4.0}}), 
                 torch::tensor({{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}}));
    torch::Tensor other = torch::zeros({2, 2});

    // Perform multiplication
    TensorDual result = x * other;

    // Expected real and dual parts
    auto expected_r = other * x.r;  // Should be zeros
    auto scaled_other = other.unsqueeze(-1);  // Should be zeros
    auto expected_d = scaled_other * x.d;  // Should be zeros

    // Validate the result
    EXPECT_TRUE(torch::allclose(result.r, expected_r));
    EXPECT_TRUE(torch::allclose(result.d, expected_d));
}

// Test: Undefined torch::Tensor
TEST(TensorDualTest, MultiplicationWithUndefinedTensor) {
    // Input TensorDual object and undefined torch::Tensor
    TensorDual x(torch::randn({2, 2}), torch::randn({2, 2, 3}));
    torch::Tensor other;

    EXPECT_THROW(
        {
            try {
                TensorDual result = x * other;
            } catch (const std::invalid_argument& e) {
                EXPECT_STREQ("Cannot multiply: Input tensor is undefined.", e.what());
                throw;
            }
        },
        std::invalid_argument
    );
}

// Test: Dimension mismatch
TEST(TensorDualTest, MultiplicationWithTensorDimensionMismatch) {
    // Input TensorDual object and mismatched torch::Tensor
    TensorDual x(torch::randn({2, 3}), torch::randn({2, 3, 4}));
    torch::Tensor other = torch::randn({3, 2});

    EXPECT_THROW(
        {
            try {
                TensorDual result = x * other;
            } catch (const std::invalid_argument& e) {
                EXPECT_STREQ("Dimension mismatch: Input tensor must have the same shape as the real part of the TensorDual.", e.what());
                throw;
            }
        },
        std::invalid_argument
    );
}

// Test: High-dimensional tensors
TEST(TensorDualTest, MultiplicationWithTensorHighDimensional) {
    // Input TensorDual object and torch::Tensor
    TensorDual x(torch::randn({3, 40000}), torch::randn({3, 40000, 5}));
    torch::Tensor other = torch::randn({3, 40000});

    // Perform multiplication
    TensorDual result = x * other;

    // Expected real and dual parts
    auto expected_r = other * x.r;
    auto scaled_other = other.unsqueeze(-1);
    auto expected_d = scaled_other * x.d;

    // Validate the result
    EXPECT_TRUE(torch::allclose(result.r, expected_r));
    EXPECT_TRUE(torch::allclose(result.d, expected_d));
}

// Test: GPU tensors
TEST(TensorDualTest, MultiplicationWithTensorGpuTensors) {
    if (torch::cuda::is_available()) {
        // Input TensorDual object on GPU and torch::Tensor
        TensorDual x(torch::randn({2, 3}, torch::kCUDA), torch::randn({2, 3, 4}, torch::kCUDA));
        torch::Tensor other = torch::randn({2, 3}, torch::kCUDA);

        // Perform multiplication
        TensorDual result = x * other;

        // Expected real and dual parts
        auto expected_r = other * x.r;
        auto scaled_other = other.unsqueeze(-1);
        auto expected_d = scaled_other * x.d;

        // Validate the result
        EXPECT_TRUE(torch::allclose(result.r, expected_r));
        EXPECT_TRUE(torch::allclose(result.d, expected_d));
    }
}


// Test: Basic functionality of scalar multiplication
TEST(TensorDualTest, ScalarMultiplicationBasic) {
    // Input TensorDual object and scalar
    TensorDual x(torch::tensor({{1.0, 2.0}, {3.0, 4.0}}), 
                 torch::tensor({{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}}));
    double scalar = 2.0;

    // Perform scalar multiplication
    TensorDual result = x * scalar;

    // Expected real and dual parts
    auto expected_r = x.r * scalar;
    auto expected_d = x.d * scalar;

    // Validate the result
    EXPECT_TRUE(torch::allclose(result.r, expected_r));
    EXPECT_TRUE(torch::allclose(result.d, expected_d));
}

// Test: Multiplication with zero scalar
TEST(TensorDualTest, ScalarMultiplicationZero) {
    // Input TensorDual object and zero scalar
    TensorDual x(torch::tensor({{1.0, 2.0}, {3.0, 4.0}}), 
                 torch::tensor({{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}}));
    double scalar = 0.0;

    // Perform scalar multiplication
    TensorDual result = x * scalar;

    // Expected real and dual parts (both zero)
    auto expected_r = x.r * scalar; // Should be zeros
    auto expected_d = x.d * scalar; // Should be zeros

    // Validate the result
    EXPECT_TRUE(torch::allclose(result.r, expected_r));
    EXPECT_TRUE(torch::allclose(result.d, expected_d));
}

// Test: Multiplication with a negative scalar
TEST(TensorDualTest, ScalarMultiplicationNegative) {
    // Input TensorDual object and negative scalar
    TensorDual x(torch::tensor({{1.0, 2.0}, {3.0, 4.0}}), 
                 torch::tensor({{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}}));
    double scalar = -3.0;

    // Perform scalar multiplication
    TensorDual result = x * scalar;

    // Expected real and dual parts
    auto expected_r = x.r * scalar;
    auto expected_d = x.d * scalar;

    // Validate the result
    EXPECT_TRUE(torch::allclose(result.r, expected_r));
    EXPECT_TRUE(torch::allclose(result.d, expected_d));
}

// Test: Multiplication with high-dimensional tensors
TEST(TensorDualTest, ScalarMultiplicationHighDimensional) {
    // Input TensorDual object and scalar
    TensorDual x(torch::randn({3, 50000}), torch::randn({3, 50000, 50}));
    double scalar = 2.5;

    // Perform scalar multiplication
    TensorDual result = x * scalar;

    // Expected real and dual parts
    auto expected_r = x.r * scalar;
    auto expected_d = x.d * scalar;

    // Validate the result
    EXPECT_TRUE(torch::allclose(result.r, expected_r));
    EXPECT_TRUE(torch::allclose(result.d, expected_d));
}

// Test: GPU tensors
TEST(TensorDualTest, ScalarMultiplicationGpuTensors) {
    if (torch::cuda::is_available()) {
        // Input TensorDual object on GPU and scalar
        TensorDual x(torch::randn({2, 3}, torch::kCUDA), torch::randn({2, 3, 4}, torch::kCUDA));
        double scalar = 4.0;

        // Perform scalar multiplication
        TensorDual result = x * scalar;

        // Expected real and dual parts
        auto expected_r = x.r * scalar;
        auto expected_d = x.d * scalar;

        // Validate the result
        EXPECT_TRUE(torch::allclose(result.r, expected_r));
        EXPECT_TRUE(torch::allclose(result.d, expected_d));
    }
}


// Test: Basic functionality of less-than-or-equal-to operator
TEST(TensorDualTest, LessThanOrEqualBasic) {
    // Input TensorDual objects
    TensorDual x(torch::tensor({{1.0, 3.0}, {5.0, 7.0}}), 
                 torch::tensor({{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}}));
    TensorDual y(torch::tensor({{2.0, 3.0}, {4.0, 8.0}}), 
                 torch::tensor({{{2.0, 3.0}, {4.0, 5.0}}, {{6.0, 7.0}, {8.0, 9.0}}}));

    // Perform comparison
    torch::Tensor result = x <= y;

    // Expected boolean tensor
    torch::Tensor expected = torch::tensor({{true, true}, {false, true}});

    // Validate the result
    EXPECT_TRUE(torch::equal(result, expected));
}

// Test: All elements less-than-or-equal
TEST(TensorDualTest, LessThanOrEqualAllTrue) {
    // Input TensorDual objects
    TensorDual x(torch::tensor({{1.0, 2.0}, {3.0, 4.0}}), 
                 torch::tensor({{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}}));
    TensorDual y(torch::tensor({{5.0, 6.0}, {7.0, 8.0}}), 
                 torch::tensor({{{5.0, 6.0}, {7.0, 8.0}}, {{9.0, 10.0}, {11.0, 12.0}}}));

    // Perform comparison
    torch::Tensor result = x <= y;

    // Expected boolean tensor
    torch::Tensor expected = torch::ones_like(x.r, torch::kBool);

    // Validate the result
    EXPECT_TRUE(torch::equal(result, expected));
}

// Test: All elements greater-than
TEST(TensorDualTest, LessThanOrEqualAllFalse) {
    // Input TensorDual objects
    TensorDual x(torch::tensor({{9.0, 10.0}, {11.0, 12.0}}), 
                 torch::tensor({{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}}));
    TensorDual y(torch::tensor({{1.0, 2.0}, {3.0, 4.0}}), 
                 torch::tensor({{{5.0, 6.0}, {7.0, 8.0}}, {{9.0, 10.0}, {11.0, 12.0}}}));

    // Perform comparison
    torch::Tensor result = x <= y;

    // Expected boolean tensor
    torch::Tensor expected = torch::zeros_like(x.r, torch::kBool);

    // Validate the result
    EXPECT_TRUE(torch::equal(result, expected));
}

// Test: Dimension mismatch
TEST(TensorDualTest, LessThanOrEqualDimensionMismatch) {
    // Input TensorDual objects with mismatched dimensions
    TensorDual x(torch::randn({2, 3}), torch::randn({2, 3, 4}));
    TensorDual y(torch::randn({3, 2}), torch::randn({3, 2, 4}));

    EXPECT_THROW(
        {
            try {
                torch::Tensor result = x <= y;
            } catch (const std::invalid_argument& e) {
                EXPECT_STREQ("Dimension mismatch: Real tensors must have the same shape for comparison.", e.what());
                throw;
            }
        },
        std::invalid_argument
    );
}

// Test: High-dimensional tensors
TEST(TensorDualTest, LessThanOrEqualHighDimensional) {
    // Input TensorDual objects with high-dimensional tensors
    TensorDual x(torch::randn({3, 4000}), torch::randn({3, 4000, 5}));
    TensorDual y(torch::randn({3, 4000}), torch::randn({3, 4000, 5}));

    // Perform comparison
    torch::Tensor result = x <= y;

    // Validate the result
    EXPECT_TRUE(torch::all(result == (x.r <= y.r)).item<bool>());
}

// Test: GPU tensors
TEST(TensorDualTest, LessThanOrEqualGpuTensors) {
    if (torch::cuda::is_available()) {
        // Input TensorDual objects on GPU
        TensorDual x(torch::randn({2, 3}, torch::kCUDA), torch::randn({2, 3, 4}, torch::kCUDA));
        TensorDual y(torch::randn({2, 3}, torch::kCUDA), torch::randn({2, 3, 4}, torch::kCUDA));

        // Perform comparison
        torch::Tensor result = x <= y;

        // Validate the result
        EXPECT_TRUE(torch::all(result == (x.r <= y.r)).item<bool>());
    }
}


// Test: Basic functionality of equality operator
TEST(TensorDualTest, EqualityBasic) {
    // Input TensorDual objects
    TensorDual x(torch::tensor({{1.0, 2.0}, {3.0, 4.0}}), 
                 torch::tensor({{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}}));
    TensorDual y(torch::tensor({{1.0, 0.0}, {3.0, 5.0}}), 
                 torch::tensor({{{2.0, 3.0}, {4.0, 5.0}}, {{6.0, 7.0}, {8.0, 9.0}}}));

    // Perform comparison
    torch::Tensor result = x == y;

    // Expected boolean tensor
    torch::Tensor expected = torch::tensor({{true, false}, {true, false}}, torch::kBool);

    // Validate the result
    EXPECT_TRUE(torch::equal(result, expected));
}

// Test: All elements equal
TEST(TensorDualTest, EqualityAllTrue) {
    // Input TensorDual objects
    TensorDual x(torch::tensor({{1.0, 2.0}, {3.0, 4.0}}), 
                 torch::tensor({{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}}));
    TensorDual y(torch::tensor({{1.0, 2.0}, {3.0, 4.0}}), 
                 torch::tensor({{{9.0, 8.0}, {7.0, 6.0}}, {{5.0, 4.0}, {3.0, 2.0}}}));

    // Perform comparison
    torch::Tensor result = x == y;

    // Expected boolean tensor (all true)
    torch::Tensor expected = torch::ones_like(x.r, torch::kBool);

    // Validate the result
    EXPECT_TRUE(torch::equal(result, expected));
}

// Test: All elements not equal
TEST(TensorDualTest, EqualityAllFalse) {
    // Input TensorDual objects
    TensorDual x(torch::tensor({{1.0, 2.0}, {3.0, 4.0}}), 
                 torch::tensor({{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}}));
    TensorDual y(torch::tensor({{5.0, 6.0}, {7.0, 8.0}}), 
                 torch::tensor({{{2.0, 3.0}, {4.0, 5.0}}, {{6.0, 7.0}, {8.0, 9.0}}}));

    // Perform comparison
    torch::Tensor result = x == y;

    // Expected boolean tensor (all false)
    torch::Tensor expected = torch::zeros_like(x.r, torch::kBool);

    // Validate the result
    EXPECT_TRUE(torch::equal(result, expected));
}

// Test: Dimension mismatch
TEST(TensorDualTest, EqualityDimensionMismatch) {
    // Input TensorDual objects with mismatched dimensions
    TensorDual x(torch::randn({2, 3}), torch::randn({2, 3, 4}));
    TensorDual y(torch::randn({3, 2}), torch::randn({3, 2, 4}));

    EXPECT_THROW(
        {
            try {
                torch::Tensor result = x == y;
            } catch (const std::invalid_argument& e) {
                EXPECT_STREQ("Dimension mismatch: Real tensors must have the same shape for comparison.", e.what());
                throw;
            }
        },
        std::invalid_argument
    );
}

// Test: High-dimensional tensors
TEST(TensorDualTest, EqualityHighDimensional) {
    // Input TensorDual objects with high-dimensional tensors
    TensorDual x(torch::randn({3, 4000}), torch::randn({3, 4000, 5}));
    TensorDual y(torch::randn({3, 4000}), torch::randn({3, 4000, 5}));

    // Perform comparison
    torch::Tensor result = x == y;

    // Validate the result
    EXPECT_TRUE(torch::all(result == (x.r == y.r)).item<bool>());
}

// Test: GPU tensors
TEST(TensorDualTest, EqualityGpuTensors) {
    if (torch::cuda::is_available()) {
        // Input TensorDual objects on GPU
        TensorDual x(torch::randn({2, 3}, torch::kCUDA), torch::randn({2, 3, 4}, torch::kCUDA));
        TensorDual y(torch::randn({2, 3}, torch::kCUDA), torch::randn({2, 3, 4}, torch::kCUDA));

        // Perform comparison
        torch::Tensor result = x == y;

        // Validate the result
        EXPECT_TRUE(torch::all(result == (x.r == y.r)).item<bool>());
    }
}



// Test: Basic functionality of less-than-or-equal-to operator with torch::Tensor
TEST(TensorDualTest, LessThanOrEqualWithTensorBasic) {
    // Input TensorDual object and torch::Tensor
    TensorDual x(torch::tensor({{1.0, 3.0}, {5.0, 7.0}}), 
                 torch::tensor({{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}}));
    torch::Tensor other = torch::tensor({{2.0, 3.0}, {6.0, 8.0}});

    // Perform comparison
    torch::Tensor result = x <= other;

    // Expected boolean tensor
    torch::Tensor expected = torch::tensor({{true, true}, {true, true}}, torch::kBool);

    // Validate the result
    EXPECT_TRUE(torch::equal(result, expected));
}

// Test: All elements less-than-or-equal
TEST(TensorDualTest, LessThanOrEqualWithTensorAllTrue) {
    // Input TensorDual object and torch::Tensor
    TensorDual x(torch::tensor({{1.0, 2.0}, {3.0, 4.0}}), 
                 torch::tensor({{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}}));
    torch::Tensor other = torch::tensor({{5.0, 6.0}, {7.0, 8.0}});

    // Perform comparison
    torch::Tensor result = x <= other;

    // Expected boolean tensor
    torch::Tensor expected = torch::ones_like(x.r, torch::kBool);

    // Validate the result
    EXPECT_TRUE(torch::equal(result, expected));
}

// Test: All elements greater-than
TEST(TensorDualTest, LessThanOrEqualWithTensorAllFalse) {
    // Input TensorDual object and torch::Tensor
    TensorDual x(torch::tensor({{9.0, 10.0}, {11.0, 12.0}}), 
                 torch::tensor({{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}}));
    torch::Tensor other = torch::tensor({{5.0, 6.0}, {7.0, 8.0}});

    // Perform comparison
    torch::Tensor result = x <= other;

    // Expected boolean tensor
    torch::Tensor expected = torch::zeros_like(x.r, torch::kBool);

    // Validate the result
    EXPECT_TRUE(torch::equal(result, expected));
}

// Test: Broadcasting compatibility
TEST(TensorDualTest, LessThanOrEqualWithTensorBroadcasting) {
    // Input TensorDual object and a broadcastable torch::Tensor
    TensorDual x(torch::tensor({{1.0, 3.0}, {5.0, 7.0}}), 
                 torch::tensor({{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}}));
    torch::Tensor other = torch::tensor({5.0});

    // Perform comparison
    torch::Tensor result = x <= other;

    // Expected boolean tensor
    torch::Tensor expected = torch::tensor({{true, true}, {true, false}}, torch::kBool);

    // Validate the result
    EXPECT_TRUE(torch::equal(result, expected));
}

// Test: Undefined torch::Tensor
TEST(TensorDualTest, LessThanOrEqualWithTensorUndefined) {
    // Input TensorDual object and undefined torch::Tensor
    TensorDual x(torch::randn({2, 2}), torch::randn({2, 2, 3}));
    torch::Tensor other;

    EXPECT_THROW(
        {
            try {
                torch::Tensor result = x <= other;
            } catch (const std::invalid_argument& e) {
                EXPECT_STREQ("Cannot compare: Input tensor is undefined.", e.what());
                throw;
            }
        },
        std::invalid_argument
    );
}

// Test: High-dimensional tensors
TEST(TensorDualTest, LessThanOrEqualWithTensorHighDimensional) {
    // Input TensorDual object and torch::Tensor
    TensorDual x(torch::randn({3, 10000}), torch::randn({3, 10000, 5}));
    torch::Tensor other = torch::randn({3, 10000});

    // Perform comparison
    torch::Tensor result = x <= other;

    // Validate the result
    EXPECT_TRUE(torch::all(result == (x.r <= other)).item<bool>());
}

// Test: GPU tensors
TEST(TensorDualTest, LessThanOrEqualWithTensorGpuTensors) {
    if (torch::cuda::is_available()) {
        // Input TensorDual object on GPU and torch::Tensor
        TensorDual x(torch::randn({2, 3}, torch::kCUDA), torch::randn({2, 3, 4}, torch::kCUDA));
        torch::Tensor other = torch::randn({2, 3}, torch::kCUDA);

        // Perform comparison
        torch::Tensor result = x <= other;

        // Validate the result
        EXPECT_TRUE(torch::all(result == (x.r <= other)).item<bool>());
    }
}


// Test: Basic functionality of less-than-or-equal-to operator with a scalar
TEST(TensorDualTest, LessThanOrEqualWithScalarBasic) {
    // Input TensorDual object and scalar
    TensorDual x(torch::tensor({{1.0, 3.0}, {5.0, 7.0}}), 
                 torch::tensor({{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}}));
    double scalar = 5.0;

    // Perform comparison
    torch::Tensor result = x <= scalar;

    // Expected boolean tensor
    torch::Tensor expected = torch::tensor({{true, true}, {true, false}}, torch::kBool);

    // Validate the result
    EXPECT_TRUE(torch::equal(result, expected));
}

// Test: All elements less-than-or-equal
TEST(TensorDualTest, LessThanOrEqualWithScalarAllTrue) {
    // Input TensorDual object and scalar
    TensorDual x(torch::tensor({{1.0, 2.0}, {3.0, 4.0}}), 
                 torch::tensor({{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}}));
    double scalar = 10.0;

    // Perform comparison
    torch::Tensor result = x <= scalar;

    // Expected boolean tensor (all true)
    torch::Tensor expected = torch::ones_like(x.r, torch::kBool);

    // Validate the result
    EXPECT_TRUE(torch::equal(result, expected));
}

// Test: All elements greater-than
TEST(TensorDualTest, LessThanOrEqualWithScalarAllFalse) {
    // Input TensorDual object and scalar
    TensorDual x(torch::tensor({{11.0, 12.0}, {13.0, 14.0}}), 
                 torch::tensor({{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}}));
    double scalar = 10.0;

    // Perform comparison
    torch::Tensor result = x <= scalar;

    // Expected boolean tensor (all false)
    torch::Tensor expected = torch::zeros_like(x.r, torch::kBool);

    // Validate the result
    EXPECT_TRUE(torch::equal(result, expected));
}

// Test: High-dimensional tensors
TEST(TensorDualTest, LessThanOrEqualWithScalarHighDimensional) {
    // Input TensorDual object and scalar
    TensorDual x(torch::randn({3, 100000}), torch::randn({3, 100000, 6}));
    float scalar = 0.5;

    // Perform comparison
    torch::Tensor result = x <= scalar;

    // Validate the result
    EXPECT_TRUE(torch::all(result == (x.r <= scalar)).item<bool>());
}

// Test: GPU tensors
TEST(TensorDualTest, LessThanOrEqualWithScalarGpuTensors) {
    if (torch::cuda::is_available()) {
        // Input TensorDual object on GPU and scalar
        TensorDual x(torch::randn({2, 3}, torch::kCUDA), torch::randn({2, 3, 4}, torch::kCUDA));
        double scalar = 1.0;

        // Perform comparison
        torch::Tensor result = x <= scalar;

        // Validate the result
        EXPECT_TRUE(torch::all(result == (x.r <= scalar)).item<bool>());
    }
}

// Test: Integer scalar comparison
TEST(TensorDualTest, LessThanOrEqualWithScalarInteger) {
    // Input TensorDual object and integer scalar
    TensorDual x(torch::tensor({{1.0, 3.0}, {5.0, 7.0}}), 
                 torch::tensor({{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}}));
    int scalar = 5;

    // Perform comparison
    torch::Tensor result = x <= scalar;

    // Expected boolean tensor
    torch::Tensor expected = torch::tensor({{true, true}, {true, false}}, torch::kBool);

    // Validate the result
    EXPECT_TRUE(torch::equal(result, expected));
}


// Test: Basic functionality of greater-than operator
TEST(TensorDualTest, GreaterThanBasic) {
    // Input TensorDual objects
    TensorDual x(torch::tensor({{3.0, 7.0}, {5.0, 2.0}}), 
                 torch::tensor({{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}}));
    TensorDual y(torch::tensor({{1.0, 5.0}, {5.0, 3.0}}), 
                 torch::tensor({{{2.0, 3.0}, {4.0, 5.0}}, {{6.0, 7.0}, {8.0, 9.0}}}));

    // Perform comparison
    torch::Tensor result = x > y;

    // Expected boolean tensor
    torch::Tensor expected = torch::tensor({{true, true}, {false, false}}, torch::kBool);

    // Validate the result
    EXPECT_TRUE(torch::equal(result, expected));
}

// Test: All elements greater
TEST(TensorDualTest, GreaterThanAllTrue) {
    // Input TensorDual objects
    TensorDual x(torch::tensor({{5.0, 6.0}, {7.0, 8.0}}), 
                 torch::tensor({{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}}));
    TensorDual y(torch::tensor({{1.0, 2.0}, {3.0, 4.0}}), 
                 torch::tensor({{{2.0, 3.0}, {4.0, 5.0}}, {{6.0, 7.0}, {8.0, 9.0}}}));

    // Perform comparison
    torch::Tensor result = x > y;

    // Expected boolean tensor (all true)
    torch::Tensor expected = torch::ones_like(x.r, torch::kBool);

    // Validate the result
    EXPECT_TRUE(torch::equal(result, expected));
}

// Test: No elements greater
TEST(TensorDualTest, GreaterThanAllFalse) {
    // Input TensorDual objects
    TensorDual x(torch::tensor({{1.0, 2.0}, {3.0, 4.0}}), 
                 torch::tensor({{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}}));
    TensorDual y(torch::tensor({{5.0, 6.0}, {7.0, 8.0}}), 
                 torch::tensor({{{2.0, 3.0}, {4.0, 5.0}}, {{6.0, 7.0}, {8.0, 9.0}}}));

    // Perform comparison
    torch::Tensor result = x > y;

    // Expected boolean tensor (all false)
    torch::Tensor expected = torch::zeros_like(x.r, torch::kBool);

    // Validate the result
    EXPECT_TRUE(torch::equal(result, expected));
}

// Test: Dimension mismatch
TEST(TensorDualTest, GreaterThanDimensionMismatch) {
    // Input TensorDual objects with mismatched dimensions
    TensorDual x(torch::randn({2, 3}), torch::randn({2, 3, 4}));
    TensorDual y(torch::randn({3, 2}), torch::randn({3, 2, 4}));

    EXPECT_THROW(
        {
            try {
                torch::Tensor result = x > y;
            } catch (const std::invalid_argument& e) {
                EXPECT_STREQ("Dimension mismatch: Real tensors must have the same shape for comparison.", e.what());
                throw;
            }
        },
        std::invalid_argument
    );
}

// Test: High-dimensional tensors
TEST(TensorDualTest, GreaterThanHighDimensional) {
    // Input TensorDual objects with high-dimensional tensors
    TensorDual x(torch::randn({3, 40000}), torch::randn({3, 40000, 5}));
    TensorDual y(torch::randn({3, 40000}), torch::randn({3, 40000, 5}));

    // Perform comparison
    torch::Tensor result = x > y;

    // Validate the result
    EXPECT_TRUE(torch::all(result == (x.r > y.r)).item<bool>());
}

// Test: GPU tensors
TEST(TensorDualTest, GreaterThanGpuTensors) {
    if (torch::cuda::is_available()) {
        // Input TensorDual objects on GPU
        TensorDual x(torch::randn({2, 3}, torch::kCUDA), torch::randn({2, 3, 4}, torch::kCUDA));
        TensorDual y(torch::randn({2, 3}, torch::kCUDA), torch::randn({2, 3, 4}, torch::kCUDA));

        // Perform comparison
        torch::Tensor result = x > y;

        // Validate the result
        EXPECT_TRUE(torch::all(result == (x.r > y.r)).item<bool>());
    }
}


// Test: Basic functionality of greater-than operator with torch::Tensor
TEST(TensorDualTest, GreaterThanWithTensorBasic) {
    // Input TensorDual object and torch::Tensor
    TensorDual x(torch::tensor({{3.0, 7.0}, {5.0, 2.0}}), 
                 torch::tensor({{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}}));
    torch::Tensor other = torch::tensor({{1.0, 5.0}, {5.0, 3.0}});

    // Perform comparison
    torch::Tensor result = x > other;

    // Expected boolean tensor
    torch::Tensor expected = torch::tensor({{true, true}, {false, false}}, torch::kBool);

    // Validate the result
    EXPECT_TRUE(torch::equal(result, expected));
}

// Test: All elements greater
TEST(TensorDualTest, GreaterThanWithTensorAllTrue) {
    // Input TensorDual object and torch::Tensor
    TensorDual x(torch::tensor({{5.0, 6.0}, {7.0, 8.0}}), 
                 torch::tensor({{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}}));
    torch::Tensor other = torch::tensor({{1.0, 2.0}, {3.0, 4.0}});

    // Perform comparison
    torch::Tensor result = x > other;

    // Expected boolean tensor (all true)
    torch::Tensor expected = torch::ones_like(x.r, torch::kBool);

    // Validate the result
    EXPECT_TRUE(torch::equal(result, expected));
}

// Test: No elements greater
TEST(TensorDualTest, GreaterThanWithTensorAllFalse) {
    // Input TensorDual object and torch::Tensor
    TensorDual x(torch::tensor({{1.0, 2.0}, {3.0, 4.0}}), 
                 torch::tensor({{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}}));
    torch::Tensor other = torch::tensor({{5.0, 6.0}, {7.0, 8.0}});

    // Perform comparison
    torch::Tensor result = x > other;

    // Expected boolean tensor (all false)
    torch::Tensor expected = torch::zeros_like(x.r, torch::kBool);

    // Validate the result
    EXPECT_TRUE(torch::equal(result, expected));
}

// Test: Undefined torch::Tensor
TEST(TensorDualTest, GreaterThanWithTensorUndefined) {
    // Input TensorDual object and undefined torch::Tensor
    TensorDual x(torch::randn({2, 2}), torch::randn({2, 2, 3}));
    torch::Tensor other;

    EXPECT_THROW(
        {
            try {
                torch::Tensor result = x > other;
            } catch (const std::invalid_argument& e) {
                EXPECT_STREQ("Cannot compare: Input tensor is undefined.", e.what());
                throw;
            }
        },
        std::invalid_argument
    );
}

// Test: Dimension mismatch
TEST(TensorDualTest, GreaterThanWithTensorDimensionMismatch) {
    // Input TensorDual object and torch::Tensor with mismatched dimensions
    TensorDual x(torch::randn({2, 3}), torch::randn({2, 3, 4}));
    torch::Tensor other = torch::randn({3, 2});

    EXPECT_THROW(
        {
            try {
                torch::Tensor result = x > other;
            } catch (const std::invalid_argument& e) {
                EXPECT_STREQ("Dimension mismatch: Input tensor must have the same shape as the real part of the TensorDual.", e.what());
                throw;
            }
        },
        std::invalid_argument
    );
}

// Test: High-dimensional tensors
TEST(TensorDualTest, GreaterThanWithTensorHighDimensional) {
    // Input TensorDual object and torch::Tensor
    TensorDual x(torch::randn({3, 400}), torch::randn({3, 400, 5}));
    torch::Tensor other = torch::randn({3, 400});

    // Perform comparison
    torch::Tensor result = x > other;

    // Validate the result
    EXPECT_TRUE(torch::all(result == (x.r > other)).item<bool>());
}

// Test: GPU tensors
TEST(TensorDualTest, GreaterThanWithTensorGpuTensors) {
    if (torch::cuda::is_available()) {
        // Input TensorDual object on GPU and torch::Tensor
        TensorDual x(torch::randn({2, 3}, torch::kCUDA), torch::randn({2, 3, 4}, torch::kCUDA));
        torch::Tensor other = torch::randn({2, 3}, torch::kCUDA);

        // Perform comparison
        torch::Tensor result = x > other;

        // Validate the result
        EXPECT_TRUE(torch::all(result == (x.r > other)).item<bool>());
    }
}



// Test: Basic functionality of greater-than operator with a scalar
TEST(TensorDualTest, GreaterThanWithScalarBasic) {
    // Input TensorDual object and scalar
    TensorDual x(torch::tensor({{1.0, 5.0}, {3.0, 7.0}}), 
                 torch::tensor({{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}}));
    double scalar = 4.0;

    // Perform comparison
    torch::Tensor result = x > scalar;

    // Expected boolean tensor
    torch::Tensor expected = torch::tensor({{false, true}, {false, true}}, torch::kBool);

    // Validate the result
    EXPECT_TRUE(torch::equal(result, expected));
}

// Test: All elements greater
TEST(TensorDualTest, GreaterThanWithScalarAllTrue) {
    // Input TensorDual object and scalar
    TensorDual x(torch::tensor({{6.0, 7.0}, {8.0, 9.0}}), 
                 torch::tensor({{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}}));
    double scalar = 5.0;

    // Perform comparison
    torch::Tensor result = x > scalar;

    // Expected boolean tensor (all true)
    torch::Tensor expected = torch::ones_like(x.r, torch::kBool);

    // Validate the result
    EXPECT_TRUE(torch::equal(result, expected));
}

// Test: No elements greater
TEST(TensorDualTest, GreaterThanWithScalarAllFalse) {
    // Input TensorDual object and scalar
    TensorDual x(torch::tensor({{1.0, 2.0}, {3.0, 4.0}}), 
                 torch::tensor({{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}}));
    double scalar = 5.0;

    // Perform comparison
    torch::Tensor result = x > scalar;

    // Expected boolean tensor (all false)
    torch::Tensor expected = torch::zeros_like(x.r, torch::kBool);

    // Validate the result
    EXPECT_TRUE(torch::equal(result, expected));
}

// Test: High-dimensional tensors
TEST(TensorDualTest, GreaterThanWithScalarHighDimensional) {
    // Input TensorDual object and scalar
    TensorDual x(torch::randn({3, 4000}), torch::randn({3, 4000, 5}));
    float scalar = 0.0;

    // Perform comparison
    torch::Tensor result = x > scalar;

    // Validate the result
    EXPECT_TRUE(torch::all(result == (x.r > scalar)).item<bool>());
}

// Test: GPU tensors
TEST(TensorDualTest, GreaterThanWithScalarGpuTensors) {
    if (torch::cuda::is_available()) {
        // Input TensorDual object on GPU and scalar
        TensorDual x(torch::randn({2, 3}, torch::kCUDA), torch::randn({2, 3, 4}, torch::kCUDA));
        double scalar = 0.5;

        // Perform comparison
        torch::Tensor result = x > scalar;

        // Validate the result
        EXPECT_TRUE(torch::all(result == (x.r > scalar)).item<bool>());
    }
}

// Test: Integer scalar comparison
TEST(TensorDualTest, GreaterThanWithScalarInteger) {
    // Input TensorDual object and integer scalar
    TensorDual x(torch::tensor({{2.0, 4.0}, {6.0, 8.0}}), 
                 torch::tensor({{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}}));
    int scalar = 5;

    // Perform comparison
    torch::Tensor result = x > scalar;

    // Expected boolean tensor
    torch::Tensor expected = torch::tensor({{false, false}, {true, true}}, torch::kBool);

    // Validate the result
    EXPECT_TRUE(torch::equal(result, expected));
}



// Test: Basic functionality of less-than operator
TEST(TensorDualTest, LessThanBasic) {
    // Input TensorDual objects
    TensorDual x(torch::tensor({{1.0, 5.0}, {3.0, 7.0}}), 
                 torch::tensor({{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}}));
    TensorDual y(torch::tensor({{3.0, 5.0}, {5.0, 6.0}}), 
                 torch::tensor({{{2.0, 3.0}, {4.0, 5.0}}, {{6.0, 7.0}, {8.0, 9.0}}}));

    // Perform comparison
    torch::Tensor result = x < y;

    // Expected boolean tensor
    torch::Tensor expected = torch::tensor({{true, false}, {true, false}}, torch::kBool);

    // Validate the result
    EXPECT_TRUE(torch::equal(result, expected));
}

// Test: All elements less
TEST(TensorDualTest, LessThanAllTrue) {
    // Input TensorDual objects
    TensorDual x(torch::tensor({{1.0, 2.0}, {3.0, 4.0}}), 
                 torch::tensor({{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}}));
    TensorDual y(torch::tensor({{5.0, 6.0}, {7.0, 8.0}}), 
                 torch::tensor({{{2.0, 3.0}, {4.0, 5.0}}, {{6.0, 7.0}, {8.0, 9.0}}}));

    // Perform comparison
    torch::Tensor result = x < y;

    // Expected boolean tensor (all true)
    torch::Tensor expected = torch::ones_like(x.r, torch::kBool);

    // Validate the result
    EXPECT_TRUE(torch::equal(result, expected));
}

// Test: No elements less
TEST(TensorDualTest, LessThanAllFalse) {
    // Input TensorDual objects
    TensorDual x(torch::tensor({{5.0, 6.0}, {7.0, 8.0}}), 
                 torch::tensor({{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}}));
    TensorDual y(torch::tensor({{1.0, 2.0}, {3.0, 4.0}}), 
                 torch::tensor({{{2.0, 3.0}, {4.0, 5.0}}, {{6.0, 7.0}, {8.0, 9.0}}}));

    // Perform comparison
    torch::Tensor result = x < y;

    // Expected boolean tensor (all false)
    torch::Tensor expected = torch::zeros_like(x.r, torch::kBool);

    // Validate the result
    EXPECT_TRUE(torch::equal(result, expected));
}

// Test: Dimension mismatch
TEST(TensorDualTest, LessThanDimensionMismatch) {
    // Input TensorDual objects with mismatched dimensions
    TensorDual x(torch::randn({2, 3}), torch::randn({2, 3, 4}));
    TensorDual y(torch::randn({3, 2}), torch::randn({3, 2, 4}));

    EXPECT_THROW(
        {
            try {
                torch::Tensor result = x < y;
            } catch (const std::invalid_argument& e) {
                EXPECT_STREQ("Dimension mismatch: Real tensors must have the same shape for comparison.", e.what());
                throw;
            }
        },
        std::invalid_argument
    );
}

// Test: High-dimensional tensors
TEST(TensorDualTest, LessThanHighDimensional) {
    // Input TensorDual objects with high-dimensional tensors
    TensorDual x(torch::randn({3, 4000}), torch::randn({3, 4000, 6}));
    TensorDual y(torch::randn({3, 4000}), torch::randn({3, 4000, 6}));

    // Perform comparison
    torch::Tensor result = x < y;

    // Validate the result
    EXPECT_TRUE(torch::all(result == (x.r < y.r)).item<bool>());
}

// Test: GPU tensors
TEST(TensorDualTest, LessThanGpuTensors) {
    if (torch::cuda::is_available()) {
        // Input TensorDual objects on GPU
        TensorDual x(torch::randn({2, 3}, torch::kCUDA), torch::randn({2, 3, 4}, torch::kCUDA));
        TensorDual y(torch::randn({2, 3}, torch::kCUDA), torch::randn({2, 3, 4}, torch::kCUDA));

        // Perform comparison
        torch::Tensor result = x < y;

        // Validate the result
        EXPECT_TRUE(torch::all(result == (x.r < y.r)).item<bool>());
    }
}


// Test: Basic functionality of less-than operator with torch::Tensor
TEST(TensorDualTest, LessThanWithTensorBasic) {
    // Input TensorDual object and torch::Tensor
    TensorDual x(torch::tensor({{3.0, 7.0}, {5.0, 2.0}}), 
                 torch::tensor({{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}}));
    torch::Tensor other = torch::tensor({{5.0, 5.0}, {6.0, 6.0}});

    // Perform comparison
    torch::Tensor result = x < other;

    // Expected boolean tensor
    torch::Tensor expected = torch::tensor({{true, false}, {true, true}}, torch::kBool);

    // Validate the result
    EXPECT_TRUE(torch::equal(result, expected));
}

// Test: All elements less
TEST(TensorDualTest, LessThanWithTensorAllTrue) {
    // Input TensorDual object and torch::Tensor
    TensorDual x(torch::tensor({{1.0, 2.0}, {3.0, 4.0}}), 
                 torch::tensor({{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}}));
    torch::Tensor other = torch::tensor({{5.0, 5.0}, {5.0, 5.0}});

    // Perform comparison
    torch::Tensor result = x < other;

    // Expected boolean tensor (all true)
    torch::Tensor expected = torch::ones_like(x.r, torch::kBool);

    // Validate the result
    EXPECT_TRUE(torch::equal(result, expected));
}

// Test: No elements less
TEST(TensorDualTest, LessThanWithTensorAllFalse) {
    // Input TensorDual object and torch::Tensor
    TensorDual x(torch::tensor({{5.0, 6.0}, {7.0, 8.0}}), 
                 torch::tensor({{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}}));
    torch::Tensor other = torch::tensor({{1.0, 2.0}, {3.0, 4.0}});

    // Perform comparison
    torch::Tensor result = x < other;

    // Expected boolean tensor (all false)
    torch::Tensor expected = torch::zeros_like(x.r, torch::kBool);

    // Validate the result
    EXPECT_TRUE(torch::equal(result, expected));
}

// Test: Undefined torch::Tensor
TEST(TensorDualTest, LessThanWithTensorUndefined) {
    // Input TensorDual object and undefined torch::Tensor
    TensorDual x(torch::randn({2, 2}), torch::randn({2, 2, 3}));
    torch::Tensor other;

    EXPECT_THROW(
        {
            try {
                torch::Tensor result = x < other;
            } catch (const std::invalid_argument& e) {
                EXPECT_STREQ("Cannot compare: Input tensor is undefined.", e.what());
                throw;
            }
        },
        std::invalid_argument
    );
}

// Test: Dimension mismatch
TEST(TensorDualTest, LessThanWithTensorDimensionMismatch) {
    // Input TensorDual object and torch::Tensor with mismatched dimensions
    TensorDual x(torch::randn({2, 3}), torch::randn({2, 3, 4}));
    torch::Tensor other = torch::randn({3, 2});

    EXPECT_THROW(
        {
            try {
                torch::Tensor result = x < other;
            } catch (const std::invalid_argument& e) {
                EXPECT_STREQ("Dimension mismatch: Input tensor must have the same shape as the real part of the TensorDual.", e.what());
                throw;
            }
        },
        std::invalid_argument
    );
}

// Test: High-dimensional tensors
TEST(TensorDualTest, LessThanWithTensorHighDimensional) {
    // Input TensorDual object and torch::Tensor
    TensorDual x(torch::randn({3, 500}), torch::randn({3, 500, 6}));
    torch::Tensor other = torch::randn({3, 500});

    // Perform comparison
    torch::Tensor result = x < other;

    // Validate the result
    EXPECT_TRUE(torch::all(result == (x.r < other)).item<bool>());
}

// Test: GPU tensors
TEST(TensorDualTest, LessThanWithTensorGpuTensors) {
    if (torch::cuda::is_available()) {
        // Input TensorDual object on GPU and torch::Tensor
        TensorDual x(torch::randn({2, 3}, torch::kCUDA), torch::randn({2, 3, 4}, torch::kCUDA));
        torch::Tensor other = torch::randn({2, 3}, torch::kCUDA);

        // Perform comparison
        torch::Tensor result = x < other;

        // Validate the result
        EXPECT_TRUE(torch::all(result == (x.r < other)).item<bool>());
    }
}


// Test: Basic functionality of less-than operator with a scalar
TEST(TensorDualTest, LessThanWithScalarBasic) {
    // Input TensorDual object and scalar
    TensorDual x(torch::tensor({{1.0, 5.0}, {3.0, 7.0}}), 
                 torch::tensor({{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}}));
    double scalar = 4.0;

    // Perform comparison
    torch::Tensor result =  x < scalar;

    // Expected boolean tensor
    torch::Tensor expected = torch::tensor({{true, false}, {true, false}}, torch::kBool);

    // Validate the result
    EXPECT_TRUE(torch::equal(result, expected));
}

// Test: All elements less
TEST(TensorDualTest, LessThanWithScalarAllTrue) {
    // Input TensorDual object and scalar
    TensorDual x(torch::tensor({{5.0, 6.0}, {7.0, 8.0}}), 
                 torch::tensor({{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}}));
    double scalar = 10.0;

    // Perform comparison
    torch::Tensor result = x < scalar;

    // Expected boolean tensor (all true)
    torch::Tensor expected = torch::ones_like(x.r, torch::kBool);

    // Validate the result
    EXPECT_TRUE(torch::equal(result, expected));
}

// Test: No elements less
TEST(TensorDualTest, LessThanWithScalarAllFalse) {
    // Input TensorDual object and scalar
    TensorDual x(torch::tensor({{11.0, 12.0}, {13.0, 14.0}}), 
                 torch::tensor({{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}}));
    double scalar = 10.0;

    // Perform comparison
    torch::Tensor result = x < scalar;

    // Expected boolean tensor (all false)
    torch::Tensor expected = torch::zeros_like(x.r, torch::kBool);

    // Validate the result
    EXPECT_TRUE(torch::equal(result, expected));
}

// Test: High-dimensional tensors
TEST(TensorDualTest, LessThanWithScalarHighDimensional) {
    // Input TensorDual object and scalar
    TensorDual x(torch::randn({3, 400}), torch::randn({3, 400, 6}));
    float scalar = 0.0;

    // Perform comparison
    torch::Tensor result = x < scalar;

    // Validate the result
    EXPECT_TRUE(torch::all(result == (x.r < scalar)).item<bool>());
}

// Test: GPU tensors
TEST(TensorDualTest, LessThanWithScalarGpuTensors) {
    if (torch::cuda::is_available()) {
        // Input TensorDual object on GPU and scalar
        TensorDual x(torch::randn({2, 3}, torch::kCUDA), torch::randn({2, 3, 4}, torch::kCUDA));
        double scalar = 0.5;

        // Perform comparison
        torch::Tensor result = x < scalar;

        // Validate the result
        EXPECT_TRUE(torch::all(result == (x.r < scalar)).item<bool>());
    }
}

// Test: Integer scalar comparison
TEST(TensorDualTest, LessThanWithScalarInteger) {
    // Input TensorDual object and integer scalar
    TensorDual x(torch::tensor({{2.0, 4.0}, {6.0, 8.0}}), 
                 torch::tensor({{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}}));
    int scalar = 5;

    // Perform comparison
    torch::Tensor result = x < scalar;

    // Expected boolean tensor
    torch::Tensor expected = torch::tensor({{true, true}, {false, false}}, torch::kBool);

    // Validate the result
    EXPECT_TRUE(torch::equal(result, expected));
}


// Test: Basic functionality of greater-than-or-equal-to operator
TEST(TensorDualTest, GreaterThanOrEqualBasic) {
    // Input TensorDual objects
    TensorDual x(torch::tensor({{3.0, 5.0}, {7.0, 2.0}}), 
                 torch::tensor({{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}}));
    TensorDual y(torch::tensor({{3.0, 4.0}, {7.0, 5.0}}), 
                 torch::tensor({{{2.0, 3.0}, {4.0, 5.0}}, {{6.0, 7.0}, {8.0, 9.0}}}));

    // Perform comparison
    torch::Tensor result = x >= y;

    // Expected boolean tensor
    torch::Tensor expected = torch::tensor({{true, true}, {true, false}}, torch::kBool);

    // Validate the result
    EXPECT_TRUE(torch::equal(result, expected));
}

// Test: All elements greater or equal
TEST(TensorDualTest, GreaterThanOrEqualAllTrue) {
    // Input TensorDual objects
    TensorDual x(torch::tensor({{6.0, 7.0}, {8.0, 9.0}}), 
                 torch::tensor({{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}}));
    TensorDual y(torch::tensor({{6.0, 6.0}, {7.0, 8.0}}), 
                 torch::tensor({{{2.0, 3.0}, {4.0, 5.0}}, {{6.0, 7.0}, {8.0, 9.0}}}));

    // Perform comparison
    torch::Tensor result = x >= y;

    // Expected boolean tensor (all true)
    torch::Tensor expected = torch::ones_like(x.r, torch::kBool);

    // Validate the result
    EXPECT_TRUE(torch::equal(result, expected));
}

// Test: No elements greater or equal
TEST(TensorDualTest, GreaterThanOrEqualAllFalse) {
    // Input TensorDual objects
    TensorDual x(torch::tensor({{1.0, 2.0}, {3.0, 4.0}}), 
                 torch::tensor({{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}}));
    TensorDual y(torch::tensor({{5.0, 6.0}, {7.0, 8.0}}), 
                 torch::tensor({{{2.0, 3.0}, {4.0, 5.0}}, {{6.0, 7.0}, {8.0, 9.0}}}));

    // Perform comparison
    torch::Tensor result = x >= y;

    // Expected boolean tensor (all false)
    torch::Tensor expected = torch::zeros_like(x.r, torch::kBool);

    // Validate the result
    EXPECT_TRUE(torch::equal(result, expected));
}

// Test: Dimension mismatch
TEST(TensorDualTest, GreaterThanOrEqualDimensionMismatch) {
    // Input TensorDual objects with mismatched dimensions
    TensorDual x(torch::randn({2, 3}), torch::randn({2, 3, 4}));
    TensorDual y(torch::randn({3, 2}), torch::randn({3, 2, 4}));

    EXPECT_THROW(
        {
            try {
                torch::Tensor result = x >= y;
            } catch (const std::invalid_argument& e) {
                EXPECT_STREQ("Dimension mismatch: Real tensors must have the same shape for comparison.", e.what());
                throw;
            }
        },
        std::invalid_argument
    );
}

// Test: High-dimensional tensors
TEST(TensorDualTest, GreaterThanOrEqualHighDimensional) {
    // Input TensorDual objects with high-dimensional tensors
    TensorDual x(torch::randn({3, 5000}), torch::randn({3, 5000, 6}));
    TensorDual y(torch::randn({3, 5000}), torch::randn({3, 5000, 6}));

    // Perform comparison
    torch::Tensor result = x >= y;

    // Validate the result
    EXPECT_TRUE(torch::all(result == (x.r >= y.r)).item<bool>());
}

// Test: GPU tensors
TEST(TensorDualTest, GreaterThanOrEqualGpuTensors) {
    if (torch::cuda::is_available()) {
        // Input TensorDual objects on GPU
        TensorDual x(torch::randn({2, 3}, torch::kCUDA), torch::randn({2, 3, 4}, torch::kCUDA));
        TensorDual y(torch::randn({2, 3}, torch::kCUDA), torch::randn({2, 3, 4}, torch::kCUDA));

        // Perform comparison
        torch::Tensor result = x >= y;

        // Validate the result
        EXPECT_TRUE(torch::all(result == (x.r >= y.r)).item<bool>());
    }
}


// Test: Basic functionality of greater-than-or-equal-to operator with torch::Tensor
TEST(TensorDualTest, GreaterThanOrEqualWithTensorBasic) {
    // Input TensorDual object and torch::Tensor
    TensorDual x(torch::tensor({{3.0, 7.0}, {5.0, 2.0}}), 
                 torch::tensor({{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}}));
    torch::Tensor other = torch::tensor({{3.0, 5.0}, {6.0, 2.0}});

    // Perform comparison
    torch::Tensor result = x >= other;

    // Expected boolean tensor
    torch::Tensor expected = torch::tensor({{true, true}, {false, true}}, torch::kBool);

    // Validate the result
    EXPECT_TRUE(torch::equal(result, expected));
}

// Test: All elements greater or equal
TEST(TensorDualTest, GreaterThanOrEqualWithTensorAllTrue) {
    // Input TensorDual object and torch::Tensor
    TensorDual x(torch::tensor({{6.0, 7.0}, {8.0, 9.0}}), 
                 torch::tensor({{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}}));
    torch::Tensor other = torch::tensor({{6.0, 6.0}, {7.0, 8.0}});

    // Perform comparison
    torch::Tensor result = x >= other;

    // Expected boolean tensor (all true)
    torch::Tensor expected = torch::ones_like(x.r, torch::kBool);

    // Validate the result
    EXPECT_TRUE(torch::equal(result, expected));
}

// Test: No elements greater or equal
TEST(TensorDualTest, GreaterThanOrEqualWithTensorAllFalse) {
    // Input TensorDual object and torch::Tensor
    TensorDual x(torch::tensor({{1.0, 2.0}, {3.0, 4.0}}), 
                 torch::tensor({{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}}));
    torch::Tensor other = torch::tensor({{5.0, 6.0}, {7.0, 8.0}});

    // Perform comparison
    torch::Tensor result = x >= other;

    // Expected boolean tensor (all false)
    torch::Tensor expected = torch::zeros_like(x.r, torch::kBool);

    // Validate the result
    EXPECT_TRUE(torch::equal(result, expected));
}

// Test: Undefined torch::Tensor
TEST(TensorDualTest, GreaterThanOrEqualWithTensorUndefined) {
    // Input TensorDual object and undefined torch::Tensor
    TensorDual x(torch::randn({2, 2}), torch::randn({2, 2, 3}));
    torch::Tensor other;

    EXPECT_THROW(
        {
            try {
                torch::Tensor result = x >= other;
            } catch (const std::invalid_argument& e) {
                EXPECT_STREQ("Cannot compare: Input tensor is undefined.", e.what());
                throw;
            }
        },
        std::invalid_argument
    );
}

// Test: Dimension mismatch
TEST(TensorDualTest, GreaterThanOrEqualWithTensorDimensionMismatch) {
    // Input TensorDual object and torch::Tensor with mismatched dimensions
    TensorDual x(torch::randn({2, 3}), torch::randn({2, 3, 4}));
    torch::Tensor other = torch::randn({3, 2});

    EXPECT_THROW(
        {
            try {
                torch::Tensor result = x >= other;
            } catch (const std::invalid_argument& e) {
                EXPECT_STREQ("Dimension mismatch: Input tensor must have the same shape as the real part of the TensorDual.", e.what());
                throw;
            }
        },
        std::invalid_argument
    );
}


// Test: GPU tensors
TEST(TensorDualTest, GreaterThanOrEqualWithTensorGpuTensors) {
    if (torch::cuda::is_available()) {
        // Input TensorDual object on GPU and torch::Tensor
        TensorDual x(torch::randn({2, 3}, torch::kCUDA), torch::randn({2, 3, 4}, torch::kCUDA));
        torch::Tensor other = torch::randn({2, 3}, torch::kCUDA);

        // Perform comparison
        torch::Tensor result = x >= other;

        // Validate the result
        EXPECT_TRUE(torch::all(result == (x.r >= other)).item<bool>());
    }
}

// Test: Basic functionality of greater-than-or-equal-to operator with a scalar
TEST(TensorDualTest, GreaterThanOrEqualWithScalarBasic) {
    // Input TensorDual object and scalar
    TensorDual x(torch::tensor({{3.0, 7.0}, {5.0, 2.0}}), 
                 torch::tensor({{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}}));
    double scalar = 4.0;

    // Perform comparison
    torch::Tensor result = x >= scalar;

    // Expected boolean tensor
    torch::Tensor expected = torch::tensor({{false, true}, {true, false}}, torch::kBool);

    // Validate the result
    EXPECT_TRUE(torch::equal(result, expected));
}

// Test: All elements greater or equal
TEST(TensorDualTest, GreaterThanOrEqualWithScalarAllTrue) {
    // Input TensorDual object and scalar
    TensorDual x(torch::tensor({{5.0, 6.0}, {7.0, 8.0}}), 
                 torch::tensor({{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}}));
    double scalar = 4.0;

    // Perform comparison
    torch::Tensor result = x >= scalar;

    // Expected boolean tensor (all true)
    torch::Tensor expected = torch::ones_like(x.r, torch::kBool);

    // Validate the result
    EXPECT_TRUE(torch::equal(result, expected));
}

// Test: No elements greater or equal
TEST(TensorDualTest, GreaterThanOrEqualWithScalarAllFalse) {
    // Input TensorDual object and scalar
    TensorDual x(torch::tensor({{1.0, 2.0}, {3.0, 4.0}}), 
                 torch::tensor({{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}}));
    double scalar = 5.0;

    // Perform comparison
    torch::Tensor result = x >= scalar;

    // Expected boolean tensor (all false)
    torch::Tensor expected = torch::zeros_like(x.r, torch::kBool);

    // Validate the result
    EXPECT_TRUE(torch::equal(result, expected));
}


// Test: GPU tensors
TEST(TensorDualTest, GreaterThanOrEqualWithScalarGpuTensors) {
    if (torch::cuda::is_available()) {
        // Input TensorDual object on GPU and scalar
        TensorDual x(torch::randn({2, 3}, torch::kCUDA), torch::randn({2, 3, 4}, torch::kCUDA));
        double scalar = 0.5;

        // Perform comparison
        torch::Tensor result = x >= scalar;

        // Validate the result
        EXPECT_TRUE(torch::all(result == (x.r >= scalar)).item<bool>());
    }
}

// Test: Integer scalar comparison
TEST(TensorDualTest, GreaterThanOrEqualWithScalarInteger) {
    // Input TensorDual object and integer scalar
    TensorDual x(torch::tensor({{2.0, 4.0}, {6.0, 8.0}}), 
                 torch::tensor({{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}}));
    int scalar = 4;

    // Perform comparison
    torch::Tensor result = x >= scalar;

    // Expected boolean tensor
    torch::Tensor expected = torch::tensor({{false, true}, {true, true}}, torch::kBool);

    // Validate the result
    EXPECT_TRUE(torch::equal(result, expected));
}

// Test: Basic functionality of equality operator with torch::Tensor
TEST(TensorDualTest, EqualityWithTensorBasic) {
    // Input TensorDual object and torch::Tensor
    TensorDual x(torch::tensor({{1.0, 2.0}, {3.0, 4.0}}), 
                 torch::tensor({{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}}));
    torch::Tensor other = torch::tensor({{1.0, 3.0}, {3.0, 4.0}});

    // Perform comparison
    torch::Tensor result = x == other;

    // Expected boolean tensor
    torch::Tensor expected = torch::tensor({{true, false}, {true, true}}, torch::kBool);

    // Validate the result
    EXPECT_TRUE(torch::equal(result, expected));
}

// Test: All elements equal
TEST(TensorDualTest, EqualityWithTensorAllTrue) {
    // Input TensorDual object and torch::Tensor
    TensorDual x(torch::tensor({{1.0, 1.0}, {1.0, 1.0}}), 
                 torch::tensor({{{0.0, 0.0}, {0.0, 0.0}}, {{0.0, 0.0}, {0.0, 0.0}}}));
    torch::Tensor other = torch::tensor({{1.0, 1.0}, {1.0, 1.0}});

    // Perform comparison
    torch::Tensor result = x == other;

    // Expected boolean tensor (all true)
    torch::Tensor expected = torch::ones_like(x.r, torch::kBool);

    // Validate the result
    EXPECT_TRUE(torch::equal(result, expected));
}

// Test: No elements equal
TEST(TensorDualTest, EqualityWithTensorAllFalse) {
    // Input TensorDual object and torch::Tensor
    TensorDual x(torch::tensor({{1.0, 2.0}, {3.0, 4.0}}), 
                 torch::tensor({{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}}));
    torch::Tensor other = torch::tensor({{5.0, 6.0}, {7.0, 8.0}});

    // Perform comparison
    torch::Tensor result = x == other;

    // Expected boolean tensor (all false)
    torch::Tensor expected = torch::zeros_like(x.r, torch::kBool);

    // Validate the result
    EXPECT_TRUE(torch::equal(result, expected));
}

// Test: Undefined torch::Tensor
TEST(TensorDualTest, EqualityWithTensorUndefined) {
    // Input TensorDual object and undefined torch::Tensor
    TensorDual x(torch::randn({2, 2}), torch::randn({2, 2, 3}));
    torch::Tensor other;

    EXPECT_THROW(
        {
            try {
                torch::Tensor result = x == other;
            } catch (const std::invalid_argument& e) {
                EXPECT_STREQ("Cannot compare: Input tensor is undefined.", e.what());
                throw;
            }
        },
        std::invalid_argument
    );
}

// Test: Dimension mismatch
TEST(TensorDualTest, EqualityWithTensorDimensionMismatch) {
    // Input TensorDual object and torch::Tensor with mismatched dimensions
    TensorDual x(torch::randn({2, 3}), torch::randn({2, 3, 4}));
    torch::Tensor other = torch::randn({3, 2});

    EXPECT_THROW(
        {
            try {
                torch::Tensor result = x == other;
            } catch (const std::invalid_argument& e) {
                EXPECT_STREQ("Dimension mismatch: Input tensor must have the same shape as the real part of the TensorDual.", e.what());
                throw;
            }
        },
        std::invalid_argument
    );
}


// Test: GPU tensors
TEST(TensorDualTest, EqualityWithTensorGpuTensors) {
    if (torch::cuda::is_available()) {
        // Input TensorDual object on GPU and torch::Tensor
        TensorDual x(torch::randn({2, 3}, torch::kCUDA), torch::randn({2, 3, 4}, torch::kCUDA));
        torch::Tensor other = torch::randn({2, 3}, torch::kCUDA);

        // Perform comparison
        torch::Tensor result = x == other;

        // Validate the result
        EXPECT_TRUE(torch::all(result == (x.r == other)).item<bool>());
    }
}



// Test: Basic functionality of equality operator with a scalar
TEST(TensorDualTest, EqualityWithScalarBasic) {
    // Input TensorDual object and scalar
    TensorDual x(torch::tensor({{1.0, 2.0}, {3.0, 4.0}}), 
                 torch::tensor({{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}}));
    double scalar = 3.0;

    // Perform comparison
    torch::Tensor result = x == scalar;

    // Expected boolean tensor
    torch::Tensor expected = torch::tensor({{false, false}, {true, false}}, torch::kBool);

    // Validate the result
    EXPECT_TRUE(torch::equal(result, expected));
}

// Test: All elements equal to scalar
TEST(TensorDualTest, EqualityWithScalarAllTrue) {
    // Input TensorDual object and scalar
    TensorDual x(torch::tensor({{2.0, 2.0}, {2.0, 2.0}}), 
                 torch::tensor({{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}}));
    double scalar = 2.0;

    // Perform comparison
    torch::Tensor result = x == scalar;

    // Expected boolean tensor (all true)
    torch::Tensor expected = torch::ones_like(x.r, torch::kBool);

    // Validate the result
    EXPECT_TRUE(torch::equal(result, expected));
}

// Test: No elements equal to scalar
TEST(TensorDualTest, EqualityWithScalarAllFalse) {
    // Input TensorDual object and scalar
    TensorDual x(torch::tensor({{1.0, 2.0}, {3.0, 4.0}}), 
                 torch::tensor({{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}}));
    double scalar = 5.0;

    // Perform comparison
    torch::Tensor result = x == scalar;

    // Expected boolean tensor (all false)
    torch::Tensor expected = torch::zeros_like(x.r, torch::kBool);

    // Validate the result
    EXPECT_TRUE(torch::equal(result, expected));
}


// Test: GPU tensors
TEST(TensorDualTest, EqualityWithScalarGpuTensors) {
    if (torch::cuda::is_available()) {
        // Input TensorDual object on GPU and scalar
        TensorDual x(torch::randn({2, 3}, torch::kCUDA), torch::randn({2, 3, 4}, torch::kCUDA));
        double scalar = 0.0;

        // Perform comparison
        torch::Tensor result = x == scalar;

        // Validate the result
        EXPECT_TRUE(torch::all(result == (x.r == scalar)).item<bool>());
    }
}

// Test: Integer scalar comparison
TEST(TensorDualTest, EqualityWithScalarInteger) {
    // Input TensorDual object and integer scalar
    TensorDual x(torch::tensor({{1, 2}, {3, 4}}, torch::kInt32), 
                 torch::tensor({{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}}, torch::kInt32));
    int scalar = 3;

    // Perform comparison
    torch::Tensor result = x == scalar;

    // Expected boolean tensor
    torch::Tensor expected = torch::tensor({{false, false}, {true, false}}, torch::kBool);

    // Validate the result
    EXPECT_TRUE(torch::equal(result, expected));
}


// Test: Basic functionality of inequality operator
TEST(TensorDualTest, InequalityBasic) {
    // Input TensorDual objects
    TensorDual x(torch::tensor({{1.0, 2.0}, {3.0, 4.0}}), 
                 torch::tensor({{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}}));
    TensorDual y(torch::tensor({{1.0, 3.0}, {3.0, 5.0}}), 
                 torch::tensor({{{2.0, 3.0}, {4.0, 5.0}}, {{6.0, 7.0}, {8.0, 9.0}}}));

    // Perform comparison
    torch::Tensor result = x != y;

    // Expected boolean tensor
    torch::Tensor expected = torch::tensor({{false, true}, {false, true}}, torch::kBool);

    // Validate the result
    EXPECT_TRUE(torch::equal(result, expected));
}

// Test: All elements different
TEST(TensorDualTest, InequalityAllTrue) {
    // Input TensorDual objects
    TensorDual x(torch::tensor({{1.0, 2.0}, {3.0, 4.0}}), 
                 torch::tensor({{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}}));
    TensorDual y(torch::tensor({{5.0, 6.0}, {7.0, 8.0}}), 
                 torch::tensor({{{2.0, 3.0}, {4.0, 5.0}}, {{6.0, 7.0}, {8.0, 9.0}}}));

    // Perform comparison
    torch::Tensor result = x != y;

    // Expected boolean tensor (all true)
    torch::Tensor expected = torch::ones_like(x.r, torch::kBool);

    // Validate the result
    EXPECT_TRUE(torch::equal(result, expected));
}

// Test: No elements different
TEST(TensorDualTest, InequalityAllFalse) {
    // Input TensorDual objects
    TensorDual x(torch::tensor({{1.0, 2.0}, {3.0, 4.0}}), 
                 torch::tensor({{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}}));
    TensorDual y(torch::tensor({{1.0, 2.0}, {3.0, 4.0}}), 
                 torch::tensor({{{2.0, 3.0}, {4.0, 5.0}}, {{6.0, 7.0}, {8.0, 9.0}}}));

    // Perform comparison
    torch::Tensor result = x != y;

    // Expected boolean tensor (all false)
    torch::Tensor expected = torch::zeros_like(x.r, torch::kBool);

    // Validate the result
    EXPECT_TRUE(torch::equal(result, expected));
}

// Test: Dimension mismatch
TEST(TensorDualTest, InequalityDimensionMismatch) {
    // Input TensorDual objects with mismatched dimensions
    TensorDual x(torch::randn({2, 3}), torch::randn({2, 3, 4}));
    TensorDual y(torch::randn({3, 2}), torch::randn({3, 2, 4}));

    EXPECT_THROW(
        {
            try {
                torch::Tensor result = x != y;
            } catch (const std::invalid_argument& e) {
                EXPECT_STREQ("Dimension mismatch: Real tensors must have the same shape for comparison.", e.what());
                throw;
            }
        },
        std::invalid_argument
    );
}


// Test: GPU tensors
TEST(TensorDualTest, InequalityGpuTensors) {
    if (torch::cuda::is_available()) {
        // Input TensorDual objects on GPU
        TensorDual x(torch::randn({2, 3}, torch::kCUDA), torch::randn({2, 3, 4}, torch::kCUDA));
        TensorDual y(torch::randn({2, 3}, torch::kCUDA), torch::randn({2, 3, 4}, torch::kCUDA));

        // Perform comparison
        torch::Tensor result = x != y;

        // Validate the result
        EXPECT_TRUE(torch::all(result == (x.r != y.r)).item<bool>());
    }
}



// Test: Basic functionality of inequality operator with torch::Tensor
TEST(TensorDualTest, InequalityWithTensorBasic) {
    // Input TensorDual object and torch::Tensor
    TensorDual x(torch::tensor({{1.0, 2.0}, {3.0, 4.0}}), 
                 torch::tensor({{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}}));
    torch::Tensor other = torch::tensor({{1.0, 3.0}, {3.0, 5.0}});

    // Perform comparison
    torch::Tensor result = x != other;

    // Expected boolean tensor
    torch::Tensor expected = torch::tensor({{false, true}, {false, true}}, torch::kBool);

    // Validate the result
    EXPECT_TRUE(torch::equal(result, expected));
}

// Test: All elements different
TEST(TensorDualTest, InequalityWithTensorAllTrue) {
    // Input TensorDual object and torch::Tensor
    TensorDual x(torch::tensor({{1.0, 2.0}, {3.0, 4.0}}), 
                 torch::tensor({{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}}));
    torch::Tensor other = torch::tensor({{5.0, 6.0}, {7.0, 8.0}});

    // Perform comparison
    torch::Tensor result = x != other;

    // Expected boolean tensor (all true)
    torch::Tensor expected = torch::ones_like(x.r, torch::kBool);

    // Validate the result
    EXPECT_TRUE(torch::equal(result, expected));
}

// Test: No elements different
TEST(TensorDualTest, InequalityWithTensorAllFalse) {
    // Input TensorDual object and torch::Tensor
    TensorDual x(torch::tensor({{1.0, 2.0}, {3.0, 4.0}}), 
                 torch::tensor({{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}}));
    torch::Tensor other = torch::tensor({{1.0, 2.0}, {3.0, 4.0}});

    // Perform comparison
    torch::Tensor result = x != other;

    // Expected boolean tensor (all false)
    torch::Tensor expected = torch::zeros_like(x.r, torch::kBool);

    // Validate the result
    EXPECT_TRUE(torch::equal(result, expected));
}

// Test: Undefined torch::Tensor
TEST(TensorDualTest, InequalityWithTensorUndefined) {
    // Input TensorDual object and undefined torch::Tensor
    TensorDual x(torch::randn({2, 2}), torch::randn({2, 2, 3}));
    torch::Tensor other;

    EXPECT_THROW(
        {
            try {
                torch::Tensor result = x != other;
            } catch (const std::invalid_argument& e) {
                EXPECT_STREQ("Cannot compare: Input tensor is undefined.", e.what());
                throw;
            }
        },
        std::invalid_argument
    );
}

// Test: Dimension mismatch
TEST(TensorDualTest, InequalityWithTensorDimensionMismatch) {
    // Input TensorDual object and torch::Tensor with mismatched dimensions
    TensorDual x(torch::randn({2, 3}), torch::randn({2, 3, 4}));
    torch::Tensor other = torch::randn({3, 2});

    EXPECT_THROW(
        {
            try {
                torch::Tensor result = x != other;
            } catch (const std::invalid_argument& e) {
                EXPECT_STREQ("Dimension mismatch: Input tensor must have the same shape as the real part of the TensorDual.", e.what());
                throw;
            }
        },
        std::invalid_argument
    );
}


// Test: GPU tensors
TEST(TensorDualTest, InequalityWithTensorGpuTensors) {
    if (torch::cuda::is_available()) {
        // Input TensorDual object on GPU and torch::Tensor
        TensorDual x(torch::randn({2, 3}, torch::kCUDA), torch::randn({2, 3, 4}, torch::kCUDA));
        torch::Tensor other = torch::randn({2, 3}, torch::kCUDA);

        // Perform comparison
        torch::Tensor result = x != other;

        // Validate the result
        EXPECT_TRUE(torch::all(result == (x.r != other)).item<bool>());
    }
}


// Test: Basic functionality of inequality operator with a scalar
TEST(TensorDualTest, InequalityWithScalarBasic) {
    // Input TensorDual object and scalar
    TensorDual x(torch::tensor({{1.0, 2.0}, {3.0, 4.0}}), 
                 torch::tensor({{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}}));
    double scalar = 3.0;

    // Perform comparison
    torch::Tensor result = x != scalar;

    // Expected boolean tensor
    torch::Tensor expected = torch::tensor({{true, true}, {false, true}}, torch::kBool);

    // Validate the result
    EXPECT_TRUE(torch::equal(result, expected));
}

// Test: All elements different from scalar
TEST(TensorDualTest, InequalityWithScalarAllTrue) {
    // Input TensorDual object and scalar
    TensorDual x(torch::tensor({{1.0, 2.0}, {3.0, 4.0}}), 
                 torch::tensor({{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}}));
    double scalar = 5.0;

    // Perform comparison
    torch::Tensor result = x != scalar;

    // Expected boolean tensor (all true)
    torch::Tensor expected = torch::ones_like(x.r, torch::kBool);

    // Validate the result
    EXPECT_TRUE(torch::equal(result, expected));
}

// Test: No elements different from scalar
TEST(TensorDualTest, InequalityWithScalarAllFalse) {
    // Input TensorDual object and scalar
    TensorDual x(torch::tensor({{3.0, 3.0}, {3.0, 3.0}}), 
                 torch::tensor({{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}}));
    double scalar = 3.0;

    // Perform comparison
    torch::Tensor result = x != scalar;

    // Expected boolean tensor (all false)
    torch::Tensor expected = torch::zeros_like(x.r, torch::kBool);

    // Validate the result
    EXPECT_TRUE(torch::equal(result, expected));
}


// Test: GPU tensors
TEST(TensorDualTest, InequalityWithScalarGpuTensors) {
    if (torch::cuda::is_available()) {
        // Input TensorDual object on GPU and scalar
        TensorDual x(torch::randn({2, 3}, torch::kCUDA), torch::randn({2, 3, 4}, torch::kCUDA));
        double scalar = 0.0;

        // Perform comparison
        torch::Tensor result = x != scalar;

        // Validate the result
        EXPECT_TRUE(torch::all(result == (x.r != scalar)).item<bool>());
    }
}

// Test: Integer scalar comparison
TEST(TensorDualTest, InequalityWithScalarInteger) {
    // Input TensorDual object and integer scalar
    TensorDual x(torch::tensor({{1, 2}, {3, 4}}, torch::kInt32), 
                 torch::tensor({{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}}, torch::kInt32));
    int scalar = 3;

    // Perform comparison
    torch::Tensor result = x != scalar;

    // Expected boolean tensor
    torch::Tensor expected = torch::tensor({{true, true}, {false, true}}, torch::kBool);

    // Validate the result
    EXPECT_TRUE(torch::equal(result, expected));
}


// Test: Basic functionality of division operator
TEST(TensorDualTest, DivisionBasic) {
    // Input TensorDual objects
    TensorDual x(torch::randn({2, 2}).to(torch::kFloat64), 
                 torch::randn({2, 2, 3}).to(torch::kFloat64)); 
    TensorDual y(torch::randn({2, 2}).to(torch::kFloat64), 
                 torch::randn({2, 2, 3}).to(torch::kFloat64));

    // Perform division
    TensorDual result = x / y;

    // Expected results
    auto expected_r = x.r/y.r;
    auto expected_d = torch::einsum("mij,mi->mij", {x.d, 1.0 / y.r}) + 
                      torch::einsum("mi,mij->mij", {-x.r / (y.r * y.r), y.d });

    // Validate the real part
    EXPECT_TRUE(torch::allclose(result.r, expected_r, 1e-8));

    // Validate the dual part
    std::cerr << result.d << std::endl;
    std::cerr << expected_d << std::endl;
    EXPECT_TRUE(torch::allclose(result.d, expected_d, 1e-8));
}

// Test: Division with zeros in the denominator
TEST(TensorDualTest, DivisionWithZeros) {
    // Input TensorDual objects
    TensorDual x(torch::tensor({{4.0, 6.0}, {8.0, 10.0}}), 
                 torch::tensor({{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}}));
    TensorDual y(torch::tensor({{0.0, 3.0}, {4.0, 5.0}}), 
                 torch::tensor({{{0.5, 1.0}, {1.5, 2.0}}, {{2.5, 3.0}, {3.5, 4.0}}}));

    // Perform division
    TensorDual result = x / y;

    // Ensure clamped behavior (safe division by zero is handled correctly)
    auto safe_r_y = torch::sign(y.r) * y.r.abs().clamp_min(1e-12);
    auto expected_r = x.r / safe_r_y;

    // Validate the real part
    EXPECT_TRUE(torch::allclose(result.r, expected_r, 1e-5));
}

// Test: Dimension mismatch
TEST(TensorDualTest, DivisionDimensionMismatch) {
    // Input TensorDual objects with mismatched dimensions
    TensorDual x(torch::randn({2, 3}), torch::randn({2, 3, 4}));
    TensorDual y(torch::randn({3, 2}), torch::randn({3, 2, 4}));

    EXPECT_THROW(
        {
            try {
                TensorDual result = x / y;
            } catch (const std::invalid_argument& e) {
                EXPECT_STREQ("Dimension mismatch: Tensors in TensorDual must have the same shape for division.", e.what());
                throw;
            }
        },
        std::invalid_argument
    );
}


// Test: GPU tensors
TEST(TensorDualTest, DivisionGpuTensors) {
    if (torch::cuda::is_available()) {
        // Input TensorDual objects on GPU
        TensorDual x(torch::randn({2, 3}, torch::kCUDA), torch::randn({2, 3, 4}, torch::kCUDA));
        TensorDual y(torch::randn({2, 3}, torch::kCUDA), torch::randn({2, 3, 4}, torch::kCUDA));

        // Perform division
        TensorDual result = x / y;

        // Compute expected real part
        auto safe_r_y = torch::sign(y.r) * y.r.abs().clamp_min(1e-12);
        auto expected_r = x.r / safe_r_y;

        // Validate the real part
        EXPECT_TRUE(torch::allclose(result.r, expected_r, 1e-5));
    }
}



// Test: Basic functionality of division operator with torch::Tensor
TEST(TensorDualTest, DivisionWithTensorBasic) {
    // Input TensorDual object and torch::Tensor
    TensorDual x(torch::tensor({{4.0, 6.0}, {8.0, 10.0}}), 
                 torch::tensor({{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}}));
    torch::Tensor other = torch::tensor({{2.0, 5.0}, {3.0, 2.0}});

    // Perform division
    TensorDual result = x / other;

    // Expected results
    auto expected_r = x.r / other;
    auto expected_d = x.d / other.unsqueeze(-1);

    // Validate the real part
    EXPECT_TRUE(torch::allclose(result.r, expected_r, 1e-5));

    // Validate the dual part
    EXPECT_TRUE(torch::allclose(result.d, expected_d, 1e-5));
}

// Test: Division with zeros in the denominator
TEST(TensorDualTest, DivisionWithTensorZeros) {
    // Input TensorDual object and torch::Tensor with zeros
    TensorDual x(torch::tensor({{4.0, 6.0}, {8.0, 10.0}}), 
                 torch::tensor({{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}}));
    torch::Tensor other = torch::tensor({{0.0, 5.0}, {3.0, 2.0}});

    // Perform division
    TensorDual result = x / other;

    // Ensure safe division behavior
    auto safe_other = torch::sign(other) * other.abs().clamp_min(1e-12);
    auto expected_r = x.r / safe_other;

    // Validate the real part
    EXPECT_TRUE(torch::allclose(result.r, expected_r, 1e-5));
}


// Test: Dimension mismatch
TEST(TensorDualTest, DivisionWithTensorDimensionMismatch) {
    // Input TensorDual object and torch::Tensor with incompatible dimensions
    TensorDual x(torch::randn({2, 3}), torch::randn({2, 3, 4}));
    torch::Tensor other = torch::randn({3, 2});

    EXPECT_THROW(
        {
            try {
                TensorDual result = x / other;
            } catch (const std::invalid_argument& e) {
                EXPECT_STREQ("Dimension mismatch: The input tensor must have the same shape as the real part of the TensorDual.", e.what());
                throw;
            }
        },
        std::invalid_argument
    );
}


// Test: GPU tensors
TEST(TensorDualTest, DivisionWithTensorGpuTensors) {
    if (torch::cuda::is_available()) {
        // Input TensorDual object on GPU and torch::Tensor
        TensorDual x(torch::randn({2, 3}, torch::kCUDA), torch::randn({2, 3, 4}, torch::kCUDA));
        torch::Tensor other = torch::randn({2, 3}, torch::kCUDA);

        // Perform division
        TensorDual result = x / other;

        // Compute expected real part
        auto safe_other = torch::sign(other) * other.abs().clamp_min(1e-12);
        auto expected_r = x.r / safe_other;

        // Validate the real part
        EXPECT_TRUE(torch::allclose(result.r, expected_r, 1e-5));
    }
}


// Test: Basic functionality of division operator with a scalar
TEST(TensorDualTest, DivisionWithScalarBasic) {
    // Input TensorDual object and scalar
    TensorDual x(torch::tensor({{4.0, 6.0}, {8.0, 10.0}}), 
                 torch::tensor({{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}}));
    double scalar = 2.0;

    // Perform division
    TensorDual result = x / scalar;

    // Expected results
    auto expected_r = torch::tensor({{2.0, 3.0}, {4.0, 5.0}});
    auto expected_d = torch::tensor({
        {{0.5, 1.0}, {1.5, 2.0}},
        {{2.5, 3.0}, {3.5, 4.0}}
    });

    // Validate the real part
    EXPECT_TRUE(torch::allclose(result.r, expected_r, 1e-5));

    // Validate the dual part
    EXPECT_TRUE(torch::allclose(result.d, expected_d, 1e-5));
}

// Test: Division by a negative scalar
TEST(TensorDualTest, DivisionWithScalarNegative) {
    // Input TensorDual object and negative scalar
    TensorDual x(torch::tensor({{4.0, -6.0}, {-8.0, 10.0}}), 
                 torch::tensor({{{1.0, -2.0}, {-3.0, 4.0}}, {{5.0, -6.0}, {-7.0, 8.0}}}));
    double scalar = -2.0;

    // Perform division
    TensorDual result = x / scalar;

    // Expected results
    auto expected_r = torch::tensor({{-2.0, 3.0}, {4.0, -5.0}});
    auto expected_d = torch::tensor({
        {{-0.5, 1.0}, {1.5, -2.0}},
        {{-2.5, 3.0}, {3.5, -4.0}}
    });

    // Validate the real part
    EXPECT_TRUE(torch::allclose(result.r, expected_r, 1e-5));

    // Validate the dual part
    EXPECT_TRUE(torch::allclose(result.d, expected_d, 1e-5));
}


// Test: Division by a very small scalar
TEST(TensorDualTest, DivisionWithScalarSmallValue) {
    // Input TensorDual object and small scalar
    TensorDual x(torch::tensor({{4.0, 6.0}, {8.0, 10.0}}).to(torch::kFloat64), 
                 torch::tensor({{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}}).to(torch::kFloat64));
    double scalar = 1e-14;

    // Perform division
    TensorDual result = x / scalar;

    // Ensure safe scalar handling
    auto safe_scalar = torch::sign(torch::tensor(scalar)) * torch::tensor(scalar).abs().clamp_min(1e-12);
    auto expected_r = x.r / safe_scalar;
    auto expected_d = x.d / safe_scalar;

    // Validate the real part
    std::cerr << result.r << std::endl;
    std::cerr << expected_r << std::endl;
    EXPECT_TRUE(torch::allclose(result.r, expected_r, 1e-4));

    // Validate the dual part
    std::cerr << result.d << std::endl;
    std::cerr << expected_d << std::endl;
    EXPECT_TRUE(torch::allclose(result.d, expected_d, 1e-4));
}


// Test: GPU tensors
TEST(TensorDualTest, DivisionWithScalarGpuTensors) {
    if (torch::cuda::is_available()) {
        // Input TensorDual object on GPU and scalar
        TensorDual x(torch::randn({2, 3}, torch::kCUDA), torch::randn({2, 3, 4}, torch::kCUDA));
        double scalar = 3.0;

        // Perform division
        TensorDual result = x / scalar;

        // Expected results
        auto expected_r = x.r / scalar;
        auto expected_d = x.d / scalar;

        // Validate the real part
        EXPECT_TRUE(torch::allclose(result.r, expected_r, 1e-5));

        // Validate the dual part
        EXPECT_TRUE(torch::allclose(result.d, expected_d, 1e-5));
    }
}


// Test: Basic functionality of gather
TEST(TensorDualTest, GatherBasic) {
    // Input TensorDual object
    TensorDual x(torch::tensor({{1.0, 2.0}, {3.0, 4.0}}), 
                 torch::tensor({{{1.0, 1.1}, {2.0, 2.1}}, {{3.0, 3.1}, {4.0, 4.1}}}));

    // Index tensor
    torch::Tensor index = torch::tensor({{0, 1}, {1, 0}});

    // Perform gather operation along dimension 1
    TensorDual result = x.gather(1, index);

    // Expected results
    auto expected_r = torch::tensor({{1.0, 2.0}, {4.0, 3.0}});
    auto expected_d = torch::tensor({{{1.0, 1.1}, {2.0, 2.1}}, {{4.0, 4.1}, {3.0, 3.1}}});

    // Validate the real part
    EXPECT_TRUE(torch::allclose(result.r, expected_r, 1e-5));

    // Validate the dual part
    EXPECT_TRUE(torch::allclose(result.d, expected_d, 1e-5));
}

// Test: Gather along dimension 0
TEST(TensorDualTest, GatherAlongDim0) {
    // Input TensorDual object
    TensorDual x(torch::tensor({{1.0, 2.0}, {3.0, 4.0}}), 
                 torch::tensor({{{1.0, 1.1}, {2.0, 2.1}}, {{3.0, 3.1}, {4.0, 4.1}}}));

    // Index tensor
    torch::Tensor index = torch::tensor({{1, 0}, {0, 1}});

    // Perform gather operation along dimension 0
    TensorDual result = x.gather(0, index);

    // Expected results
    auto expected_r = torch::tensor({{3.0, 2.0}, {1.0, 4.0}});
    auto expected_d = torch::tensor({{{3.0, 3.1}, {2.0, 2.1}}, {{1.0, 1.1}, {4.0, 4.1}}});

    // Validate the real part
    EXPECT_TRUE(torch::allclose(result.r, expected_r, 1e-5));

    // Validate the dual part
    EXPECT_TRUE(torch::allclose(result.d, expected_d, 1e-5));
}

// Test: Invalid dimension
TEST(TensorDualTest, GatherInvalidDimension) {
    // Input TensorDual object
    TensorDual x(torch::randn({2, 2}), torch::randn({2, 2, 3}));

    // Invalid dimension
    int64_t invalid_dim = 3;

    // Index tensor
    torch::Tensor index = torch::tensor({{0, 1}, {1, 0}});

    EXPECT_THROW(
        {
            try {
                TensorDual result = x.gather(invalid_dim, index);
            } catch (const std::invalid_argument& e) {
                EXPECT_STREQ("Invalid dimension: 'dim' must be within the range of the tensor dimensions.", e.what());
                throw;
            }
        },
        std::invalid_argument
    );
}

// Test: Invalid index tensor dimensions
TEST(TensorDualTest, GatherInvalidIndexDimensions) {
    // Input TensorDual object
    TensorDual x(torch::randn({2, 2}), torch::randn({2, 2, 3}));

    // Invalid index tensor dimensions
    torch::Tensor index = torch::randn({3, 2, 1});

    EXPECT_THROW(
        {
            try {
                TensorDual result = x.gather(1, index);
            } catch (const std::invalid_argument& e) {
                EXPECT_STREQ("Index tensor dimensions are incompatible with the target tensor.", e.what());
                throw;
            }
        },
        std::invalid_argument
    );
}


// Test: GPU tensors
TEST(TensorDualTest, GatherGpuTensors) {
    if (torch::cuda::is_available()) {
        // Input TensorDual object on GPU
        TensorDual x(torch::randn({2, 3}, torch::kCUDA), torch::randn({2, 3, 4}, torch::kCUDA));

        // Index tensor
        torch::Tensor index = torch::tensor({{0, 1}, {1, 2}}, torch::kCUDA);

        // Perform gather operation along dimension 1
        TensorDual result = x.gather(1, index);

        // Validate shapes
        EXPECT_EQ(result.r.sizes(), index.sizes());
        EXPECT_EQ(result.d.sizes(), torch::IntArrayRef({2, 2, 4}));
    }
}



// Test: Basic functionality of scatter
TEST(TensorDualTest, ScatterBasic) {
    // Input TensorDual object
    TensorDual x(torch::tensor({{1.0, 2.0}, {3.0, 4.0}}), 
                 torch::tensor({{{1.0, 1.1}, {2.0, 2.1}}, {{3.0, 3.1}, {4.0, 4.1}}}));

    // Source TensorDual
    TensorDual src(torch::tensor({{10.0, 20.0}, {30.0, 40.0}}), 
                   torch::tensor({{{10.0, 11.0}, {20.0, 21.0}}, {{30.0, 31.0}, {40.0, 41.0}}}));

    // Index tensor
    torch::Tensor index = torch::tensor({{0, 1}, {1, 0}});

    // Perform scatter operation along dimension 1
    TensorDual result = x.scatter(1, index, src);

    // Expected results
    auto expected_r = torch::tensor({{10.0, 20.0}, {40.0, 30.0}});
    auto expected_d = torch::tensor({{{10.0, 11.0}, {20.0, 21.0}}, {{40.0, 41.0}, {30.0, 31.0}}});
    std::cerr << "result.r" << result.r << std::endl;
    std::cerr << "expected_r" << expected_r << std::endl;
    std::cerr << "result.d" << result.d << std::endl;
    std::cerr << "expected_d" << expected_d << std::endl;

    // Validate the real part
    EXPECT_TRUE(torch::allclose(result.r, expected_r, 1e-5));

    // Validate the dual part
    EXPECT_TRUE(torch::allclose(result.d, expected_d, 1e-5));
}

// Test: Scatter along dimension 0
TEST(TensorDualTest, ScatterAlongDim0) {
    // Input TensorDual object
    TensorDual x(torch::zeros({2, 2}), torch::zeros({2, 2, 3}));

    // Source TensorDual
    TensorDual src(torch::tensor({{10.0, 20.0}, {30.0, 40.0}}), 
                   torch::tensor({{{10.0, 11.0, 12.0}, {20.0, 21.0, 22.0}}, {{30.0, 31.0, 32.0}, {40.0, 41.0, 42.0}}}));

    // Index tensor
    torch::Tensor index = torch::tensor({{1, 0}, {0, 1}});

    // Perform scatter operation along dimension 0
    TensorDual result = x.scatter(0, index, src);

    // Expected results
    auto expected_r = torch::tensor({{30.0, 20.0}, {10.0, 40.0}});
    auto expected_d = torch::tensor({{{30.0, 31.0, 32.0}, {20.0, 21.0, 22.0}}, 
                                     {{10.0, 11.0, 12.0}, {40.0, 41.0, 42.0}}});

    // Validate the real part
    EXPECT_TRUE(torch::allclose(result.r, expected_r, 1e-5));

    // Validate the dual part
    EXPECT_TRUE(torch::allclose(result.d, expected_d, 1e-5));
}

// Test: Invalid dimension
TEST(TensorDualTest, ScatterInvalidDimension) {
    // Input TensorDual object
    TensorDual x(torch::randn({2, 2}), torch::randn({2, 2, 3}));

    // Source TensorDual
    TensorDual src(torch::randn({2, 2}), torch::randn({2, 2, 3}));

    // Invalid dimension
    int64_t invalid_dim = 3;

    // Index tensor
    torch::Tensor index = torch::randn({2, 2}).to(torch::kLong);

    EXPECT_THROW(
        {
            try {
                TensorDual result = x.scatter(invalid_dim, index, src);
            } catch (const std::invalid_argument& e) {
                EXPECT_STREQ("Invalid dimension: 'dim' must be within the range of the tensor dimensions.", e.what());
                throw;
            }
        },
        std::invalid_argument
    );
}

// Test: Invalid source TensorDual dimensions
TEST(TensorDualTest, ScatterInvalidSourceDimensions) {
    // Input TensorDual object
    TensorDual x(torch::randn({2, 2}), torch::randn({2, 2, 3}));

    // Source TensorDual with mismatched dimensions
    TensorDual src(torch::randn({3, 2}), torch::randn({3, 2, 3}));

    // Index tensor
    torch::Tensor index = torch::randn({2, 2}).to(torch::kLong);

    EXPECT_THROW(
        {
            try {
                TensorDual result = x.scatter(1, index, src);
            } catch (const std::invalid_argument& e) {
                EXPECT_STREQ("Source TensorDual must have the same shape as the target TensorDual.", e.what());
                throw;
            }
        },
        std::invalid_argument
    );
}

// Test: GPU tensors
TEST(TensorDualTest, ScatterGpuTensors) {
    if (torch::cuda::is_available()) {
        // Input TensorDual object on GPU
        TensorDual x(torch::zeros({2, 3}, torch::kCUDA), torch::zeros({2, 3, 4}, torch::kCUDA));

        // Source TensorDual
        TensorDual src(torch::tensor({{10.0, 20.0, 30.0}, {40.0, 50.0, 60.0}}, torch::kCUDA), 
                       torch::tensor({{{10.0, 11.0, 12.0, 13.0}, {20.0, 21.0, 22.0, 23.0}, {30.0, 31.0, 32.0, 33.0}}, 
                                      {{40.0, 41.0, 42.0, 43.0}, {50.0, 51.0, 52.0, 53.0}, {60.0, 61.0, 62.0, 63.0}}}, torch::kCUDA));

        // Index tensor
        torch::Tensor index = torch::tensor({{0, 2, 1}, {2, 1, 0}}, torch::kCUDA);

        // Perform scatter operation along dimension 1
        TensorDual result = x.scatter(1, index, src);

        // Validate shapes
        EXPECT_EQ(result.r.sizes(), x.r.sizes());
        EXPECT_EQ(result.d.sizes(), x.d.sizes());
    }
}


// Test: Basic functionality of reciprocal
TEST(TensorDualTest, ReciprocalBasic) {
    // Input TensorDual object
    TensorDual x(torch::tensor({{2.0, 4.0}, {1.0, 0.5}}), 
                 torch::tensor({{{0.1, 0.2}, {0.3, 0.4}}, {{0.5, 0.6}, {0.7, 0.8}}}));

    // Perform reciprocal operation
    TensorDual result = x.reciprocal();

    // Expected results
    auto expected_r = torch::tensor({{0.5, 0.25}, {1.0, 2.0}});
    auto rrec_squared = expected_r.unsqueeze(-1) * expected_r.unsqueeze(-1);
    auto expected_d = -rrec_squared * x.d;

    // Validate the real part
    EXPECT_TRUE(torch::allclose(result.r, expected_r, 1e-5));

    // Validate the dual part
    EXPECT_TRUE(torch::allclose(result.d, expected_d, 1e-5));
}

// Test: Reciprocal with negative values
TEST(TensorDualTest, ReciprocalNegativeValues) {
    // Input TensorDual object
    TensorDual x(torch::tensor({{-2.0, -4.0}, {-1.0, -0.5}}), 
                 torch::tensor({{{0.1, -0.2}, {0.3, -0.4}}, {{-0.5, 0.6}, {-0.7, 0.8}}}));

    // Perform reciprocal operation
    TensorDual result = x.reciprocal();

    // Expected results
    auto expected_r = torch::tensor({{-0.5, -0.25}, {-1.0, -2.0}});
    auto rrec_squared = expected_r.unsqueeze(-1) * expected_r.unsqueeze(-1);
    auto expected_d = -rrec_squared * x.d;

    // Validate the real part
    EXPECT_TRUE(torch::allclose(result.r, expected_r, 1e-5));

    // Validate the dual part
    EXPECT_TRUE(torch::allclose(result.d, expected_d, 1e-5));
}

// Test: Reciprocal with small values
TEST(TensorDualTest, ReciprocalSmallValues) {
    // Input TensorDual object
    TensorDual x(torch::tensor({{1e-10, 1e-8}, {1e-6, 1e-4}}), 
                 torch::randn({2, 2, 3}));

    // Perform reciprocal operation
    TensorDual result = x.reciprocal();

    // Ensure small values are clamped
    auto r_safe = torch::sign(x.r) * x.r.abs().clamp_min(1e-12);
    auto expected_r = r_safe.reciprocal();
    auto rrec_squared = expected_r.unsqueeze(-1) * expected_r.unsqueeze(-1);
    auto expected_d = -rrec_squared * x.d;

    // Validate the real part
    EXPECT_TRUE(torch::allclose(result.r, expected_r, 1e-5));

    // Validate the dual part
    EXPECT_TRUE(torch::allclose(result.d, expected_d, 1e-5));
}



// Test: GPU tensors
TEST(TensorDualTest, ReciprocalGpuTensors) {
    if (torch::cuda::is_available()) {
        // Input TensorDual object on GPU
        TensorDual x(torch::randn({2, 3}, torch::kCUDA), torch::randn({2, 3, 4}, torch::kCUDA));

        // Perform reciprocal operation
        TensorDual result = x.reciprocal();

        // Expected results
        auto r_safe = torch::sign(x.r) * x.r.abs().clamp_min(1e-12);
        auto expected_r = r_safe.reciprocal();
        auto rrec_squared = expected_r.unsqueeze(-1) * expected_r.unsqueeze(-1);
        auto expected_d = -rrec_squared * x.d;

        // Validate the real part
        EXPECT_TRUE(torch::allclose(result.r, expected_r, 1e-5));

        // Validate the dual part
        EXPECT_TRUE(torch::allclose(result.d, expected_d, 1e-5));
    }
}



// Test: Basic functionality of square
TEST(TensorDualTest, SquareBasic) {
    // Input TensorDual object
    TensorDual x(torch::tensor({{1.0, 2.0}, {3.0, 4.0}}), 
                 torch::tensor({{{0.1, 0.2}, {0.3, 0.4}}, {{0.5, 0.6}, {0.7, 0.8}}}));

    // Perform square operation
    TensorDual result = x.square();

    // Expected results
    auto expected_r = torch::tensor({{1.0, 4.0}, {9.0, 16.0}});
    auto expected_d = 2 * x.r.unsqueeze(-1) * x.d;

    // Validate the real part
    EXPECT_TRUE(torch::allclose(result.r, expected_r, 1e-5));

    // Validate the dual part
    EXPECT_TRUE(torch::allclose(result.d, expected_d, 1e-5));
}

// Test: Square with negative values
TEST(TensorDualTest, SquareNegativeValues) {
    // Input TensorDual object
    TensorDual x(torch::tensor({{-1.0, -2.0}, {-3.0, -4.0}}), 
                 torch::tensor({{{0.1, -0.2}, {0.3, -0.4}}, {{-0.5, 0.6}, {-0.7, 0.8}}}));

    // Perform square operation
    TensorDual result = x.square();

    // Expected results
    auto expected_r = torch::tensor({{1.0, 4.0}, {9.0, 16.0}});
    auto expected_d = 2 * x.r.unsqueeze(-1) * x.d;

    // Validate the real part
    EXPECT_TRUE(torch::allclose(result.r, expected_r, 1e-5));

    // Validate the dual part
    EXPECT_TRUE(torch::allclose(result.d, expected_d, 1e-5));
}


// Test: Incompatible dual dimensions
TEST(TensorDualTest, SquareIncompatibleDimensions) {
    // Input TensorDual object with incompatible dual dimensions
    TensorDual x(torch::randn({2, 2}), torch::randn({2, 3, 2}));

    EXPECT_THROW(
        {
            try {
                TensorDual result = x.square();
            } catch (const std::invalid_argument& e) {
                EXPECT_STREQ("Dual part dimensions are incompatible with the real part.", e.what());
                throw;
            }
        },
        std::invalid_argument
    );
}

// Test: GPU tensors
TEST(TensorDualTest, SquareGpuTensors) {
    if (torch::cuda::is_available()) {
        // Input TensorDual object on GPU
        TensorDual x(torch::randn({2, 3}, torch::kCUDA), torch::randn({2, 3, 4}, torch::kCUDA));

        // Perform square operation
        TensorDual result = x.square();

        // Expected results
        auto expected_r = x.r.square();
        auto expected_d = 2 * x.r.unsqueeze(-1) * x.d;

        // Validate the real part
        EXPECT_TRUE(torch::allclose(result.r, expected_r, 1e-5));

        // Validate the dual part
        EXPECT_TRUE(torch::allclose(result.d, expected_d, 1e-5));
    }
}


// Test: Basic functionality of sin
TEST(TensorDualTest, SinBasic) {
    // Input TensorDual object
    TensorDual x(torch::tensor({{0.0, M_PI / 2}, {M_PI, 3 * M_PI / 2}}), 
                 torch::tensor({{{0.1, 0.2}, {0.3, 0.4}}, {{0.5, 0.6}, {0.7, 0.8}}}));

    // Perform sine operation
    TensorDual result = x.sin();

    // Expected results
    auto expected_r = torch::sin(x.r);
    auto expected_d = torch::cos(x.r).unsqueeze(-1) * x.d;

    // Validate the real part
    EXPECT_TRUE(torch::allclose(result.r, expected_r, 1e-5));

    // Validate the dual part
    EXPECT_TRUE(torch::allclose(result.d, expected_d, 1e-5));
}


// Test: Zero values
TEST(TensorDualTest, SinZeroValues) {
    // Input TensorDual object with zero real part
    TensorDual x(torch::zeros({2, 2}), torch::randn({2, 2, 3}));

    // Perform sine operation
    TensorDual result = x.sin();

    // Expected results
    auto expected_r = torch::zeros_like(x.r); // sin(0) = 0
    auto expected_d = torch::ones_like(x.d) * x.d; // cos(0) = 1

    // Validate the real part
    EXPECT_TRUE(torch::allclose(result.r, expected_r, 1e-5));

    // Validate the dual part
    EXPECT_TRUE(torch::allclose(result.d, expected_d, 1e-5));
}

// Test: Negative values
TEST(TensorDualTest, SinNegativeValues) {
    // Input TensorDual object with negative real values
    TensorDual x(torch::tensor({{-M_PI / 6, -M_PI / 4}, {-M_PI / 3, -M_PI / 2}}), 
                 torch::randn({2, 2, 3}));

    // Perform sine operation
    TensorDual result = x.sin();

    // Expected results
    auto expected_r = torch::sin(x.r);
    auto expected_d = torch::cos(x.r).unsqueeze(-1) * x.d;

    // Validate the real part
    EXPECT_TRUE(torch::allclose(result.r, expected_r, 1e-5));

    // Validate the dual part
    EXPECT_TRUE(torch::allclose(result.d, expected_d, 1e-5));
}

// Test: Incompatible dual dimensions
TEST(TensorDualTest, SinIncompatibleDimensions) {
    // Input TensorDual object with mismatched real and dual dimensions
    TensorDual x(torch::randn({2, 2}), torch::randn({2, 3, 2}));

    // Expect invalid_argument exception
    EXPECT_THROW(
        {
            try {
                TensorDual result = x.sin();
            } catch (const std::invalid_argument& e) {
                EXPECT_STREQ("Dual part dimensions are incompatible with the real part.", e.what());
                throw;
            }
        },
        std::invalid_argument
    );
}

// Test: GPU tensors
TEST(TensorDualTest, SinGpuTensors) {
    if (torch::cuda::is_available()) {
        // Input TensorDual object on GPU
        TensorDual x(torch::randn({2, 3}, torch::kCUDA), torch::randn({2, 3, 4}, torch::kCUDA));

        // Perform sine operation
        TensorDual result = x.sin();

        // Expected results
        auto expected_r = torch::sin(x.r);
        auto expected_d = torch::cos(x.r).unsqueeze(-1) * x.d;

        // Validate the real part
        EXPECT_TRUE(torch::allclose(result.r, expected_r, 1e-5));

        // Validate the dual part
        EXPECT_TRUE(torch::allclose(result.d, expected_d, 1e-5));
    }
}


// Test: Basic functionality of cos
TEST(TensorDualTest, CosBasic) {
    // Input TensorDual object
    TensorDual x(torch::tensor({{0.0, M_PI / 2}, {M_PI, 3 * M_PI / 2}}), 
                 torch::tensor({{{0.1, 0.2}, {0.3, 0.4}}, {{0.5, 0.6}, {0.7, 0.8}}}));

    // Perform cosine operation
    TensorDual result = x.cos();

    // Expected results
    auto expected_r = torch::cos(x.r);
    auto expected_d = -torch::sin(x.r).unsqueeze(-1) * x.d;

    // Validate the real part
    EXPECT_TRUE(torch::allclose(result.r, expected_r, 1e-5));

    // Validate the dual part
    EXPECT_TRUE(torch::allclose(result.d, expected_d, 1e-5));
}


// Test: Zero values
TEST(TensorDualTest, CosZeroValues) {
    // Input TensorDual object with zero real part
    TensorDual x(torch::zeros({2, 2}), torch::randn({2, 2, 3}));

    // Perform cosine operation
    TensorDual result = x.cos();

    // Expected results
    auto expected_r = torch::ones_like(x.r); // cos(0) = 1
    auto expected_d = -torch::zeros_like(x.d); // -sin(0) = 0

    // Validate the real part
    EXPECT_TRUE(torch::allclose(result.r, expected_r, 1e-5));

    // Validate the dual part
    EXPECT_TRUE(torch::allclose(result.d, expected_d, 1e-5));
}

// Test: Negative values
TEST(TensorDualTest, CosNegativeValues) {
    // Input TensorDual object with negative real values
    TensorDual x(torch::tensor({{-M_PI / 6, -M_PI / 4}, {-M_PI / 3, -M_PI / 2}}), 
                 torch::randn({2, 2, 3}));

    // Perform cosine operation
    TensorDual result = x.cos();

    // Expected results
    auto expected_r = torch::cos(x.r);
    auto expected_d = -torch::sin(x.r).unsqueeze(-1) * x.d;

    // Validate the real part
    EXPECT_TRUE(torch::allclose(result.r, expected_r, 1e-5));

    // Validate the dual part
    EXPECT_TRUE(torch::allclose(result.d, expected_d, 1e-5));
}

// Test: Incompatible dual dimensions
TEST(TensorDualTest, CosIncompatibleDimensions) {
    // Input TensorDual object with mismatched real and dual dimensions
    TensorDual x(torch::randn({2, 2}), torch::randn({2, 3, 2}));

    // Expect invalid_argument exception
    EXPECT_THROW(
        {
            try {
                TensorDual result = x.cos();
            } catch (const std::invalid_argument& e) {
                EXPECT_STREQ("Dual part dimensions are incompatible with the real part.", e.what());
                throw;
            }
        },
        std::invalid_argument
    );
}

// Test: GPU tensors
TEST(TensorDualTest, CosGpuTensors) {
    if (torch::cuda::is_available()) {
        // Input TensorDual object on GPU
        TensorDual x(torch::randn({2, 3}, torch::kCUDA), torch::randn({2, 3, 4}, torch::kCUDA));

        // Perform cosine operation
        TensorDual result = x.cos();

        // Expected results
        auto expected_r = torch::cos(x.r);
        auto expected_d = -torch::sin(x.r).unsqueeze(-1) * x.d;

        // Validate the real part
        EXPECT_TRUE(torch::allclose(result.r, expected_r, 1e-5));

        // Validate the dual part
        EXPECT_TRUE(torch::allclose(result.d, expected_d, 1e-5));
    }
}


// Test: Basic functionality of tan
TEST(TensorDualTest, TanBasic) {
    // Input TensorDual object
    TensorDual x(torch::tensor({{0.0, M_PI / 4}, {M_PI / 6, -M_PI / 4}}), 
                 torch::tensor({{{0.1, 0.2}, {0.3, 0.4}}, {{0.5, 0.6}, {0.7, 0.8}}}));

    // Perform tangent operation
    TensorDual result = x.tan();

    // Expected results
    auto expected_r = torch::tan(x.r);
    auto expected_d = torch::pow(torch::cos(x.r), -2).unsqueeze(-1) * x.d;

    // Validate the real part
    EXPECT_TRUE(torch::allclose(result.r, expected_r, 1e-5));

    // Validate the dual part
    EXPECT_TRUE(torch::allclose(result.d, expected_d, 1e-5));
}


// Test: Zero values
TEST(TensorDualTest, TanZeroValues) {
    // Input TensorDual object with zero real part
    TensorDual x(torch::zeros({2, 2}), torch::zeros({2, 2, 3}));
    //The dual part must be the unit matrix
    x.d = torch::eye(2).unsqueeze(2).expand({2, 2, 3});

    // Perform tangent operation
    TensorDual result = x.tan();

    // Expected results
    auto expected_r = torch::zeros_like(x.r); // tan(0) = 0
    auto expected_d = torch::eye(2).unsqueeze(2).expand({2, 2, 3}); // sec^2(0) = 1

    // Validate the real part
    EXPECT_TRUE(torch::allclose(result.r, expected_r, 1e-5));

    // Validate the dual part
    EXPECT_TRUE(torch::allclose(result.d, expected_d, 1e-5));
}

// Test: Negative values
TEST(TensorDualTest, TanNegativeValues) {
    // Input TensorDual object with negative real values
    TensorDual x(torch::tensor({{-M_PI / 6, -M_PI / 4}, {-M_PI / 3, -M_PI / 2}}), 
                 torch::randn({2, 2, 3}));

    // Perform tangent operation
    TensorDual result = x.tan();

    // Expected results
    auto expected_r = torch::tan(x.r);
    auto expected_d = torch::pow(torch::cos(x.r), -2).unsqueeze(-1) * x.d;

    // Validate the real part
    EXPECT_TRUE(torch::allclose(result.r, expected_r, 1e-5));

    // Validate the dual part
    EXPECT_TRUE(torch::allclose(result.d, expected_d, 1e-5));
}

// Test: Incompatible dual dimensions
TEST(TensorDualTest, TanIncompatibleDimensions) {
    // Input TensorDual object with mismatched real and dual dimensions
    TensorDual x(torch::randn({2, 2}), torch::randn({2, 3, 2}));

    // Expect invalid_argument exception
    EXPECT_THROW(
        {
            try {
                TensorDual result = x.tan();
            } catch (const std::invalid_argument& e) {
                EXPECT_STREQ("Dual part dimensions are incompatible with the real part.", e.what());
                throw;
            }
        },
        std::invalid_argument
    );
}

// Test: Near singularities
TEST(TensorDualTest, TanSingularities) {
    // Input TensorDual object near singularity (pi/2 + n*pi)
    TensorDual x(torch::tensor({{M_PI / 2 - 1e-6, -M_PI / 2 + 1e-6}}), 
                 torch::ones({1, 2, 3}));
    std::cerr << "x.d" << x.d << std::endl;

    // Perform tangent operation
    TensorDual result = x.tan();

    // Expected results
    auto expected_r = torch::tan(x.r);
    auto expected_d = torch::pow(torch::cos(x.r), -2).unsqueeze(-1) * x.d;

    // Validate the real part
    EXPECT_TRUE(torch::allclose(result.r, expected_r, 1e-5));

    // Validate the dual part
    EXPECT_TRUE(torch::allclose(result.d, expected_d, 1e-5));
}

// Test: GPU tensors
TEST(TensorDualTest, TanGpuTensors) {
    if (torch::cuda::is_available()) {
        // Input TensorDual object on GPU
        TensorDual x(torch::randn({2, 3}, torch::kCUDA), torch::randn({2, 3, 4}, torch::kCUDA));

        // Perform tangent operation
        TensorDual result = x.tan();

        // Expected results
        auto expected_r = torch::tan(x.r);
        auto expected_d = torch::pow(torch::cos(x.r), -2).unsqueeze(-1) * x.d;

        // Validate the real part
        EXPECT_TRUE(torch::allclose(result.r, expected_r, 1e-5));

        // Validate the dual part
        EXPECT_TRUE(torch::allclose(result.d, expected_d, 1e-5));
    }
}


// Test: Basic functionality of asin
TEST(TensorDualTest, AsinBasic) {
    // Input TensorDual object
    TensorDual x(torch::tensor({{-1.0, -0.5, 0.0, 0.5, 1.0}}), 
                 torch::tensor({{{0.1}, {0.2}, {0.3}, {0.4}, {0.5}}}));

    // Perform arcsine operation
    TensorDual result = x.asin();

    // Expected results
    auto expected_r = torch::asin(x.r);
    auto expected_d = (1 / torch::sqrt(1 - torch::pow(x.r, 2))).unsqueeze(-1) * x.d;

    // Validate the real part
    EXPECT_TRUE(torch::allclose(result.r, expected_r, 1e-5));

    // Validate the dual part
    EXPECT_TRUE(torch::allclose(result.d, expected_d, 1e-5));
}


// Test: Edge values [-1, 1]
TEST(TensorDualTest, AsinEdgeValues) {
    // Input TensorDual object with edge values
    TensorDual x(torch::tensor({{-1.0, 1.0}}), torch::tensor({{{0.1}, {0.2}}}));

    // Perform arcsine operation
    TensorDual result = x.asin();

    // Expected results
    auto expected_r = torch::asin(x.r);
    auto expected_d = (1 / torch::sqrt(1 - torch::pow(x.r, 2))).unsqueeze(-1) * x.d;

    // Validate the real part
    EXPECT_TRUE(torch::allclose(result.r, expected_r, 1e-5));

    // Validate the dual part
    EXPECT_TRUE(torch::allclose(result.d, expected_d, 1e-5));
}

// Test: Values outside domain
TEST(TensorDualTest, AsinOutOfDomain) {
    // Input TensorDual object with values outside domain [-1, 1]
    TensorDual x(torch::tensor({{-1.5, 2.0}}), torch::randn({1, 2, 3}));

    // Expect domain_error exception
    EXPECT_THROW(
        {
            try {
                TensorDual result = x.asin();
            } catch (const std::domain_error& e) {
                EXPECT_STREQ("Real part out of domain: arcsine is only defined for values in the range [-1, 1].", e.what());
                throw;
            }
        },
        std::domain_error
    );
}

// Test: Incompatible dual dimensions
TEST(TensorDualTest, AsinIncompatibleDimensions) {
    // Input TensorDual object with mismatched real and dual dimensions
    TensorDual x(torch::rand({2, 2}), torch::rand({2, 3, 2}));

    // Expect invalid_argument exception
    EXPECT_THROW(
        {
            try {
                TensorDual result = x.asin();
            } catch (const std::invalid_argument& e) {
                EXPECT_STREQ("Dual part dimensions are incompatible with the real part.", e.what());
                throw;
            }
        },
        std::invalid_argument
    );
}

// Test: GPU tensors
TEST(TensorDualTest, AsinGpuTensors) {
    if (torch::cuda::is_available()) {
        // Input TensorDual object on GPU
        TensorDual x(torch::rand({2, 3}, torch::kCUDA) * 2 - 1, torch::randn({2, 3, 4}, torch::kCUDA));

        // Perform arcsine operation
        TensorDual result = x.asin();

        // Expected results
        auto expected_r = torch::asin(x.r);
        auto expected_d = (1 / torch::sqrt(1 - torch::pow(x.r, 2))).unsqueeze(-1) * x.d;

        // Validate the real part
        EXPECT_TRUE(torch::allclose(result.r, expected_r, 1e-5));

        // Validate the dual part
        EXPECT_TRUE(torch::allclose(result.d, expected_d, 1e-5));
    }
}


// Test: Basic functionality of acos
TEST(TensorDualTest, AcosBasic) {
    // Input TensorDual object
    TensorDual x(torch::tensor({{-1.0, -0.5, 0.0, 0.5, 1.0}}), 
                 torch::tensor({{{0.1}, {0.2}, {0.3}, {0.4}, {0.5}}}));

    // Perform arccosine operation
    TensorDual result = x.acos();

    // Expected results
    auto expected_r = torch::acos(x.r);
    auto expected_d = (-1 / torch::sqrt(1 - torch::pow(x.r, 2))).unsqueeze(-1) * x.d;

    // Validate the real part
    EXPECT_TRUE(torch::allclose(result.r, expected_r, 1e-5));

    // Validate the dual part
    EXPECT_TRUE(torch::allclose(result.d, expected_d, 1e-5));
}


// Test: Edge values [-1, 1]
TEST(TensorDualTest, AcosEdgeValues) {
    // Input TensorDual object with edge values
    TensorDual x(torch::tensor({{-1.0, 1.0}}), torch::tensor({{{0.1}, {0.2}}}));

    // Perform arccosine operation
    TensorDual result = x.acos();

    // Expected results
    auto expected_r = torch::acos(x.r);
    auto expected_d = (-1 / torch::sqrt(1 - torch::pow(x.r, 2))).unsqueeze(-1) * x.d;

    // Validate the real part
    EXPECT_TRUE(torch::allclose(result.r, expected_r, 1e-5));

    // Validate the dual part
    EXPECT_TRUE(torch::allclose(result.d, expected_d, 1e-5));
}

// Test: Values outside domain
TEST(TensorDualTest, AcosOutOfDomain) {
    // Input TensorDual object with values outside domain [-1, 1]
    TensorDual x(torch::tensor({{-1.5, 2.0}}), torch::randn({1, 2, 3}));

    // Expect domain_error exception
    EXPECT_THROW(
        {
            try {
                TensorDual result = x.acos();
            } catch (const std::domain_error& e) {
                EXPECT_STREQ("Real part out of domain: arccosine is only defined for values in the range [-1, 1].", e.what());
                throw;
            }
        },
        std::domain_error
    );
}

// Test: Incompatible dual dimensions
TEST(TensorDualTest, AcosIncompatibleDimensions) {
    // Input TensorDual object with mismatched real and dual dimensions
    TensorDual x(torch::rand({2, 2}), torch::rand({2, 3, 2}));

    // Expect invalid_argument exception
    EXPECT_THROW(
        {
            try {
                TensorDual result = x.acos();
            } catch (const std::invalid_argument& e) {
                EXPECT_STREQ("Dual part dimensions are incompatible with the real part.", e.what());
                throw;
            }
        },
        std::invalid_argument
    );
}

// Test: GPU tensors
TEST(TensorDualTest, AcosGpuTensors) {
    if (torch::cuda::is_available()) {
        // Input TensorDual object on GPU
        TensorDual x(torch::rand({2, 3}, torch::kCUDA) * 2 - 1, torch::randn({2, 3, 4}, torch::kCUDA));

        // Perform arccosine operation
        TensorDual result = x.acos();

        // Expected results
        auto expected_r = torch::acos(x.r);
        auto expected_d = (-1 / torch::sqrt(1 - torch::pow(x.r, 2))).unsqueeze(-1) * x.d;

        // Validate the real part
        EXPECT_TRUE(torch::allclose(result.r, expected_r, 1e-5));

        // Validate the dual part
        EXPECT_TRUE(torch::allclose(result.d, expected_d, 1e-5));
    }
}


// Test: Basic functionality of atan
TEST(TensorDualTest, AtanBasic) {
    // Input TensorDual object
    TensorDual x(torch::tensor({{-1.0, 0.0, 1.0}}), 
                 torch::tensor({{{0.1}, {0.2}, {0.3}}}));

    // Perform arctangent operation
    TensorDual result = x.atan();

    // Expected results
    auto expected_r = torch::atan(x.r);
    auto expected_d = (1 / (1 + torch::pow(x.r, 2))).unsqueeze(-1) * x.d;

    // Validate the real part
    EXPECT_TRUE(torch::allclose(result.r, expected_r, 1e-5));

    // Validate the dual part
    EXPECT_TRUE(torch::allclose(result.d, expected_d, 1e-5));
}


// Test: Zero values
TEST(TensorDualTest, AtanZeroValues) {
    // Input TensorDual object with zero real part
    TensorDual x(torch::zeros({2, 2}), torch::randn({2, 2, 3}));

    // Perform arctangent operation
    TensorDual result = x.atan();

    // Expected results
    auto expected_r = torch::atan(x.r); // atan(0) = 0
    auto expected_d = (1 / (1 + torch::pow(x.r, 2))).unsqueeze(-1) * x.d; // 1 / (1 + 0^2) = 1

    // Validate the real part
    EXPECT_TRUE(torch::allclose(result.r, expected_r, 1e-5));

    // Validate the dual part
    EXPECT_TRUE(torch::allclose(result.d, expected_d, 1e-5));
}

// Test: Negative values
TEST(TensorDualTest, AtanNegativeValues) {
    // Input TensorDual object with negative real values
    TensorDual x(torch::tensor({{-2.0, -1.0}}), torch::randn({1, 2, 3}));

    // Perform arctangent operation
    TensorDual result = x.atan();

    // Expected results
    auto expected_r = torch::atan(x.r);
    auto expected_d = (1 / (1 + torch::pow(x.r, 2))).unsqueeze(-1) * x.d;

    // Validate the real part
    EXPECT_TRUE(torch::allclose(result.r, expected_r, 1e-5));

    // Validate the dual part
    EXPECT_TRUE(torch::allclose(result.d, expected_d, 1e-5));
}

// Test: Large values
TEST(TensorDualTest, AtanLargeValues) {
    // Input TensorDual object with large real values
    TensorDual x(torch::tensor({{100.0, 1000.0}}), torch::randn({1, 2, 3}));

    // Perform arctangent operation
    TensorDual result = x.atan();

    // Expected results
    auto expected_r = torch::atan(x.r);
    auto expected_d = (1 / (1 + torch::pow(x.r, 2))).unsqueeze(-1) * x.d;

    // Validate the real part
    EXPECT_TRUE(torch::allclose(result.r, expected_r, 1e-5));

    // Validate the dual part
    EXPECT_TRUE(torch::allclose(result.d, expected_d, 1e-5));
}

// Test: Incompatible dual dimensions
TEST(TensorDualTest, AtanIncompatibleDimensions) {
    // Input TensorDual object with mismatched real and dual dimensions
    TensorDual x(torch::rand({2, 2}), torch::rand({2, 3, 2}));

    // Expect invalid_argument exception
    EXPECT_THROW(
        {
            try {
                TensorDual result = x.atan();
            } catch (const std::invalid_argument& e) {
                EXPECT_STREQ("Dual part dimensions are incompatible with the real part.", e.what());
                throw;
            }
        },
        std::invalid_argument
    );
}

// Test: GPU tensors
TEST(TensorDualTest, AtanGpuTensors) {
    if (torch::cuda::is_available()) {
        // Input TensorDual object on GPU
        TensorDual x(torch::rand({2, 3}, torch::kCUDA), torch::randn({2, 3, 4}, torch::kCUDA));

        // Perform arctangent operation
        TensorDual result = x.atan();

        // Expected results
        auto expected_r = torch::atan(x.r);
        auto expected_d = (1 / (1 + torch::pow(x.r, 2))).unsqueeze(-1) * x.d;

        // Validate the real part
        EXPECT_TRUE(torch::allclose(result.r, expected_r, 1e-5));

        // Validate the dual part
        EXPECT_TRUE(torch::allclose(result.d, expected_d, 1e-5));
    }
}


// Test: Basic functionality of sinh
TEST(TensorDualTest, SinhBasic) {
    // Input TensorDual object
    TensorDual x(torch::tensor({{-1.0, 0.0, 1.0}}), 
                 torch::tensor({{{0.1}, {0.2}, {0.3}}}));

    // Perform hyperbolic sine operation
    TensorDual result = x.sinh();

    // Expected results
    auto expected_r = torch::sinh(x.r);
    auto expected_d = torch::cosh(x.r).unsqueeze(-1) * x.d;

    // Validate the real part
    EXPECT_TRUE(torch::allclose(result.r, expected_r, 1e-5));

    // Validate the dual part
    EXPECT_TRUE(torch::allclose(result.d, expected_d, 1e-5));
}


// Test: Zero values
TEST(TensorDualTest, SinhZeroValues) {
    // Input TensorDual object with zero real part
    TensorDual x(torch::zeros({2, 2}), torch::randn({2, 2, 3}));

    // Perform hyperbolic sine operation
    TensorDual result = x.sinh();

    // Expected results
    auto expected_r = torch::sinh(x.r); // sinh(0) = 0
    auto expected_d = torch::cosh(x.r).unsqueeze(-1) * x.d; // cosh(0) = 1

    // Validate the real part
    EXPECT_TRUE(torch::allclose(result.r, expected_r, 1e-5));

    // Validate the dual part
    EXPECT_TRUE(torch::allclose(result.d, expected_d, 1e-5));
}

// Test: Negative values
TEST(TensorDualTest, SinhNegativeValues) {
    // Input TensorDual object with negative real values
    TensorDual x(torch::tensor({{-2.0, -1.0}}), torch::randn({1, 2, 3}));

    // Perform hyperbolic sine operation
    TensorDual result = x.sinh();

    // Expected results
    auto expected_r = torch::sinh(x.r);
    auto expected_d = torch::cosh(x.r).unsqueeze(-1) * x.d;

    // Validate the real part
    EXPECT_TRUE(torch::allclose(result.r, expected_r, 1e-5));

    // Validate the dual part
    EXPECT_TRUE(torch::allclose(result.d, expected_d, 1e-5));
}

// Test: Large values
TEST(TensorDualTest, SinhLargeValues) {
    // Input TensorDual object with large real values
    TensorDual x(torch::tensor({{100.0, -100.0}}), torch::randn({1, 2, 3}));

    // Perform hyperbolic sine operation
    TensorDual result = x.sinh();

    // Expected results
    auto expected_r = torch::sinh(x.r);
    auto expected_d = torch::cosh(x.r).unsqueeze(-1) * x.d;

    // Validate the real part
    EXPECT_TRUE(torch::allclose(result.r, expected_r, 1e-5));

    // Validate the dual part
    EXPECT_TRUE(torch::allclose(result.d, expected_d, 1e-5));
}

// Test: Incompatible dual dimensions
TEST(TensorDualTest, SinhIncompatibleDimensions) {
    // Input TensorDual object with mismatched real and dual dimensions
    TensorDual x(torch::rand({2, 2}), torch::rand({2, 3, 2}));

    // Expect invalid_argument exception
    EXPECT_THROW(
        {
            try {
                TensorDual result = x.sinh();
            } catch (const std::invalid_argument& e) {
                EXPECT_STREQ("Dual part dimensions are incompatible with the real part.", e.what());
                throw;
            }
        },
        std::invalid_argument
    );
}

// Test: GPU tensors
TEST(TensorDualTest, SinhGpuTensors) {
    if (torch::cuda::is_available()) {
        // Input TensorDual object on GPU
        TensorDual x(torch::randn({2, 3}, torch::kCUDA), torch::randn({2, 3, 4}, torch::kCUDA));

        // Perform hyperbolic sine operation
        TensorDual result = x.sinh();

        // Expected results
        auto expected_r = torch::sinh(x.r);
        auto expected_d = torch::cosh(x.r).unsqueeze(-1) * x.d;

        // Validate the real part
        EXPECT_TRUE(torch::allclose(result.r, expected_r, 1e-5));

        // Validate the dual part
        EXPECT_TRUE(torch::allclose(result.d, expected_d, 1e-5));
    }
}


// Test: Basic functionality of cosh
TEST(TensorDualTest, CoshBasic) {
    // Input TensorDual object
    TensorDual x(torch::tensor({{-1.0, 0.0, 1.0}}), 
                 torch::tensor({{{0.1}, {0.2}, {0.3}}}));

    // Perform hyperbolic cosine operation
    TensorDual result = x.cosh();

    // Expected results
    auto expected_r = torch::cosh(x.r);
    auto expected_d = torch::sinh(x.r).unsqueeze(-1) * x.d;

    // Validate the real part
    EXPECT_TRUE(torch::allclose(result.r, expected_r, 1e-5));

    // Validate the dual part
    EXPECT_TRUE(torch::allclose(result.d, expected_d, 1e-5));
}


// Test: Zero values
TEST(TensorDualTest, CoshZeroValues) {
    // Input TensorDual object with zero real part
    TensorDual x(torch::zeros({2, 2}), torch::randn({2, 2, 3}));

    // Perform hyperbolic cosine operation
    TensorDual result = x.cosh();

    // Expected results
    auto expected_r = torch::cosh(x.r); // cosh(0) = 1
    auto expected_d = torch::sinh(x.r).unsqueeze(-1) * x.d; // sinh(0) = 0

    // Validate the real part
    EXPECT_TRUE(torch::allclose(result.r, expected_r, 1e-5));

    // Validate the dual part
    EXPECT_TRUE(torch::allclose(result.d, expected_d, 1e-5));
}

// Test: Negative values
TEST(TensorDualTest, CoshNegativeValues) {
    // Input TensorDual object with negative real values
    TensorDual x(torch::tensor({{-2.0, -1.0}}), torch::randn({1, 2, 3}));

    // Perform hyperbolic cosine operation
    TensorDual result = x.cosh();

    // Expected results
    auto expected_r = torch::cosh(x.r);
    auto expected_d = torch::sinh(x.r).unsqueeze(-1) * x.d;

    // Validate the real part
    EXPECT_TRUE(torch::allclose(result.r, expected_r, 1e-5));

    // Validate the dual part
    EXPECT_TRUE(torch::allclose(result.d, expected_d, 1e-5));
}

// Test: Large values
TEST(TensorDualTest, CoshLargeValues) {
    // Input TensorDual object with large real values
    TensorDual x(torch::tensor({{100.0, -100.0}}), torch::randn({1, 2, 3}));

    // Perform hyperbolic cosine operation
    TensorDual result = x.cosh();

    // Expected results
    auto expected_r = torch::cosh(x.r);
    auto expected_d = torch::sinh(x.r).unsqueeze(-1) * x.d;

    // Validate the real part
    EXPECT_TRUE(torch::allclose(result.r, expected_r, 1e-5));

    // Validate the dual part
    EXPECT_TRUE(torch::allclose(result.d, expected_d, 1e-5));
}

// Test: Incompatible dual dimensions
TEST(TensorDualTest, CoshIncompatibleDimensions) {
    // Input TensorDual object with mismatched real and dual dimensions
    TensorDual x(torch::rand({2, 2}), torch::rand({2, 3, 2}));

    // Expect invalid_argument exception
    EXPECT_THROW(
        {
            try {
                TensorDual result = x.cosh();
            } catch (const std::invalid_argument& e) {
                EXPECT_STREQ("Dual part dimensions are incompatible with the real part.", e.what());
                throw;
            }
        },
        std::invalid_argument
    );
}

// Test: GPU tensors
TEST(TensorDualTest, CoshGpuTensors) {
    if (torch::cuda::is_available()) {
        // Input TensorDual object on GPU
        TensorDual x(torch::randn({2, 3}, torch::kCUDA), torch::randn({2, 3, 4}, torch::kCUDA));

        // Perform hyperbolic cosine operation
        TensorDual result = x.cosh();

        // Expected results
        auto expected_r = torch::cosh(x.r);
        auto expected_d = torch::sinh(x.r).unsqueeze(-1) * x.d;

        // Validate the real part
        EXPECT_TRUE(torch::allclose(result.r, expected_r, 1e-5));

        // Validate the dual part
        EXPECT_TRUE(torch::allclose(result.d, expected_d, 1e-5));
    }
}


// Test: Basic functionality of tanh
TEST(TensorDualTest, TanhBasic) {
    // Input TensorDual object
    TensorDual x(torch::tensor({{-1.0, 0.0, 1.0}}), 
                 torch::tensor({{{0.1}, {0.2}, {0.3}}}));

    // Perform hyperbolic tangent operation
    TensorDual result = x.tanh();

    // Expected results
    auto expected_r = torch::tanh(x.r);
    auto expected_d = torch::pow(torch::cosh(x.r), -2).unsqueeze(-1) * x.d;

    // Validate the real part
    EXPECT_TRUE(torch::allclose(result.r, expected_r, 1e-5));

    // Validate the dual part
    EXPECT_TRUE(torch::allclose(result.d, expected_d, 1e-5));
}


// Test: Zero values
TEST(TensorDualTest, TanhZeroValues) {
    // Input TensorDual object with zero real part
    TensorDual x(torch::zeros({2, 2}), torch::randn({2, 2, 3}));

    // Perform hyperbolic tangent operation
    TensorDual result = x.tanh();

    // Expected results
    auto expected_r = torch::tanh(x.r); // tanh(0) = 0
    auto expected_d = torch::pow(torch::cosh(x.r), -2).unsqueeze(-1) * x.d; // sech^2(0) = 1

    // Validate the real part
    EXPECT_TRUE(torch::allclose(result.r, expected_r, 1e-5));

    // Validate the dual part
    EXPECT_TRUE(torch::allclose(result.d, expected_d, 1e-5));
}

// Test: Negative values
TEST(TensorDualTest, TanhNegativeValues) {
    // Input TensorDual object with negative real values
    TensorDual x(torch::tensor({{-2.0, -1.0}}), torch::randn({1, 2, 3}));

    // Perform hyperbolic tangent operation
    TensorDual result = x.tanh();

    // Expected results
    auto expected_r = torch::tanh(x.r);
    auto expected_d = torch::pow(torch::cosh(x.r), -2).unsqueeze(-1) * x.d;

    // Validate the real part
    EXPECT_TRUE(torch::allclose(result.r, expected_r, 1e-5));

    // Validate the dual part
    EXPECT_TRUE(torch::allclose(result.d, expected_d, 1e-5));
}

// Test: Large values
TEST(TensorDualTest, TanhLargeValues) {
    // Input TensorDual object with large real values
    TensorDual x(torch::tensor({{10.0, -10.0}}), torch::randn({1, 2, 3}));

    // Perform hyperbolic tangent operation
    TensorDual result = x.tanh();

    // Expected results
    auto expected_r = torch::tanh(x.r);
    auto expected_d = torch::pow(torch::cosh(x.r), -2).unsqueeze(-1) * x.d;

    // Validate the real part
    EXPECT_TRUE(torch::allclose(result.r, expected_r, 1e-5));

    // Validate the dual part
    EXPECT_TRUE(torch::allclose(result.d, expected_d, 1e-5));
}

// Test: Incompatible dual dimensions
TEST(TensorDualTest, TanhIncompatibleDimensions) {
    // Input TensorDual object with mismatched real and dual dimensions
    TensorDual x(torch::rand({2, 2}), torch::rand({2, 3, 2}));

    // Expect invalid_argument exception
    EXPECT_THROW(
        {
            try {
                TensorDual result = x.tanh();
            } catch (const std::invalid_argument& e) {
                EXPECT_STREQ("Dual part dimensions are incompatible with the real part.", e.what());
                throw;
            }
        },
        std::invalid_argument
    );
}

// Test: GPU tensors
TEST(TensorDualTest, TanhGpuTensors) {
    if (torch::cuda::is_available()) {
        // Input TensorDual object on GPU
        TensorDual x(torch::randn({2, 3}, torch::kCUDA), torch::randn({2, 3, 4}, torch::kCUDA));

        // Perform hyperbolic tangent operation
        TensorDual result = x.tanh();

        // Expected results
        auto expected_r = torch::tanh(x.r);
        auto expected_d = torch::pow(torch::cosh(x.r), -2).unsqueeze(-1) * x.d;

        // Validate the real part
        EXPECT_TRUE(torch::allclose(result.r, expected_r, 1e-5));

        // Validate the dual part
        EXPECT_TRUE(torch::allclose(result.d, expected_d, 1e-5));
    }
}

// Test: Basic functionality of asinh
TEST(TensorDualTest, AsinhBasic) {
    // Input TensorDual object
    TensorDual x(torch::tensor({{-1.0, 0.0, 1.0}}), 
                 torch::tensor({{{0.1}, {0.2}, {0.3}}}));

    // Perform hyperbolic arcsine operation
    TensorDual result = x.asinh();

    // Expected results
    auto expected_r = torch::asinh(x.r);
    auto expected_d = torch::pow(1 + torch::pow(x.r, 2), -0.5).unsqueeze(-1) * x.d;

    // Validate the real part
    EXPECT_TRUE(torch::allclose(result.r, expected_r, 1e-5));

    // Validate the dual part
    EXPECT_TRUE(torch::allclose(result.d, expected_d, 1e-5));
}


// Test: Zero values
TEST(TensorDualTest, AsinhZeroValues) {
    // Input TensorDual object with zero real part
    TensorDual x(torch::zeros({2, 2}), torch::randn({2, 2, 3}));

    // Perform hyperbolic arcsine operation
    TensorDual result = x.asinh();

    // Expected results
    auto expected_r = torch::asinh(x.r); // asinh(0) = 0
    auto expected_d = torch::pow(1 + torch::pow(x.r, 2), -0.5).unsqueeze(-1) * x.d; // f'(0) = 1

    // Validate the real part
    EXPECT_TRUE(torch::allclose(result.r, expected_r, 1e-5));

    // Validate the dual part
    EXPECT_TRUE(torch::allclose(result.d, expected_d, 1e-5));
}

// Test: Negative values
TEST(TensorDualTest, AsinhNegativeValues) {
    // Input TensorDual object with negative real values
    TensorDual x(torch::tensor({{-2.0, -1.0}}), torch::randn({1, 2, 3}));

    // Perform hyperbolic arcsine operation
    TensorDual result = x.asinh();

    // Expected results
    auto expected_r = torch::asinh(x.r);
    auto expected_d = torch::pow(1 + torch::pow(x.r, 2), -0.5).unsqueeze(-1) * x.d;

    // Validate the real part
    EXPECT_TRUE(torch::allclose(result.r, expected_r, 1e-5));

    // Validate the dual part
    EXPECT_TRUE(torch::allclose(result.d, expected_d, 1e-5));
}

// Test: Large values
TEST(TensorDualTest, AsinhLargeValues) {
    // Input TensorDual object with large real values
    TensorDual x(torch::tensor({{100.0, -100.0}}), torch::randn({1, 2, 3}));

    // Perform hyperbolic arcsine operation
    TensorDual result = x.asinh();

    // Expected results
    auto expected_r = torch::asinh(x.r);
    auto expected_d = torch::pow(1 + torch::pow(x.r, 2), -0.5).unsqueeze(-1) * x.d;

    // Validate the real part
    EXPECT_TRUE(torch::allclose(result.r, expected_r, 1e-5));

    // Validate the dual part
    EXPECT_TRUE(torch::allclose(result.d, expected_d, 1e-5));
}

// Test: Incompatible dual dimensions
TEST(TensorDualTest, AsinhIncompatibleDimensions) {
    // Input TensorDual object with mismatched real and dual dimensions
    TensorDual x(torch::rand({2, 2}), torch::rand({2, 3, 2}));

    // Expect invalid_argument exception
    EXPECT_THROW(
        {
            try {
                TensorDual result = x.asinh();
            } catch (const std::invalid_argument& e) {
                EXPECT_STREQ("Dual part dimensions are incompatible with the real part.", e.what());
                throw;
            }
        },
        std::invalid_argument
    );
}

// Test: GPU tensors
TEST(TensorDualTest, AsinhGpuTensors) {
    if (torch::cuda::is_available()) {
        // Input TensorDual object on GPU
        TensorDual x(torch::randn({2, 3}, torch::kCUDA), torch::randn({2, 3, 4}, torch::kCUDA));

        // Perform hyperbolic arcsine operation
        TensorDual result = x.asinh();

        // Expected results
        auto expected_r = torch::asinh(x.r);
        auto expected_d = torch::pow(1 + torch::pow(x.r, 2), -0.5).unsqueeze(-1) * x.d;

        // Validate the real part
        EXPECT_TRUE(torch::allclose(result.r, expected_r, 1e-5));

        // Validate the dual part
        EXPECT_TRUE(torch::allclose(result.d, expected_d, 1e-5));
    }
}


// Test: Basic functionality of acosh
TEST(TensorDualTest, AcoshBasic) {
    // Input TensorDual object
    TensorDual x(torch::tensor({{1.0, 2.0, 3.0}}), 
                 torch::tensor({{{0.1}, {0.2}, {0.3}}}));

    // Perform hyperbolic arccosine operation
    TensorDual result = x.acosh();

    // Expected results
    auto expected_r = torch::acosh(x.r);
    auto expected_d = torch::pow(torch::pow(x.r, 2.0) - 1.0, -0.5).unsqueeze(-1) * x.d;

    // Validate the real part
    EXPECT_TRUE(torch::allclose(result.r, expected_r, 1e-5));

    // Validate the dual part
    EXPECT_TRUE(torch::allclose(result.d, expected_d, 1e-5));
}


// Test: Edge case with r = 1.0
TEST(TensorDualTest, AcoshEdgeCase) {
    // Input TensorDual object with r = 1.0
    TensorDual x(torch::ones({2, 2}), torch::randn({2, 2, 3}));

    // Perform hyperbolic arccosine operation
    TensorDual result = x.acosh();

    // Expected results
    auto expected_r = torch::acosh(x.r);
    auto expected_d = torch::pow(torch::pow(x.r, 2.0) - 1.0, -0.5).unsqueeze(-1) * x.d;

    // Validate the real part
    EXPECT_TRUE(torch::allclose(result.r, expected_r, 1e-5));

    // Validate the dual part
    EXPECT_TRUE(torch::allclose(result.d, expected_d, 1e-5));
}

// Test: Negative values (invalid domain)
TEST(TensorDualTest, AcoshNegativeValues) {
    // Input TensorDual object with invalid real part
    TensorDual x(torch::tensor({{0.5, -1.0}}), torch::randn({1, 2, 3}));

    // Expect domain_error exception
    EXPECT_THROW(
        {
            try {
                TensorDual result = x.acosh();
            } catch (const std::domain_error& e) {
                EXPECT_STREQ("All real elements passed to acosh must be >= 1.0.", e.what());
                throw;
            }
        },
        std::domain_error
    );
}

// Test: Incompatible dual dimensions
TEST(TensorDualTest, AcoshIncompatibleDimensions) {
    // Input TensorDual object with mismatched real and dual dimensions
    TensorDual x(torch::rand({2, 2})+1.0, torch::rand({2, 3, 2}));

    // Expect invalid_argument exception
    EXPECT_THROW(
        {
            try {
                TensorDual result = x.acosh();
            } catch (const std::invalid_argument& e) {
                EXPECT_STREQ("Dual part dimensions are incompatible with the real part.", e.what());
                throw;
            }
        },
        std::invalid_argument
    );
}

// Test: Large values
TEST(TensorDualTest, AcoshLargeValues) {
    // Input TensorDual object with large real values
    TensorDual x(torch::tensor({{100.0, 200.0}}), torch::randn({1, 2, 3}));

    // Perform hyperbolic arccosine operation
    TensorDual result = x.acosh();

    // Expected results
    auto expected_r = torch::acosh(x.r);
    auto expected_d = torch::pow(torch::pow(x.r, 2.0) - 1.0, -0.5).unsqueeze(-1) * x.d;

    // Validate the real part
    EXPECT_TRUE(torch::allclose(result.r, expected_r, 1e-5));

    // Validate the dual part
    EXPECT_TRUE(torch::allclose(result.d, expected_d, 1e-5));
}

// Test: GPU tensors
TEST(TensorDualTest, AcoshGpuTensors) {
    if (torch::cuda::is_available()) {
        // Input TensorDual object on GPU
        TensorDual x(torch::rand({2, 3}, torch::kCUDA) + 2.0, torch::randn({2, 3, 4}, torch::kCUDA));

        // Perform hyperbolic arccosine operation
        TensorDual result = x.acosh();

        // Expected results
        auto expected_r = torch::acosh(x.r);
        auto expected_d = torch::pow(torch::pow(x.r, 2.0) - 1.0, -0.5).unsqueeze(-1) * x.d;

        // Validate the real part
        EXPECT_TRUE(torch::allclose(result.r, expected_r, 1e-5));

        // Validate the dual part
        EXPECT_TRUE(torch::allclose(result.d, expected_d, 1e-5));
    }
}

// Test: Basic functionality of atanh
TEST(TensorDualTest, AtanhBasic) {
    // Input TensorDual object
    TensorDual x(torch::tensor({{-0.5, 0.0, 0.5}}), 
                 torch::tensor({{{0.1}, {0.2}, {0.3}}}));

    // Perform hyperbolic arctangent operation
    TensorDual result = x.atanh();

    // Expected results
    auto expected_r = torch::atanh(x.r);
    auto expected_d = torch::pow(1 - torch::pow(x.r, 2), -1).unsqueeze(-1) * x.d;

    // Validate the real part
    EXPECT_TRUE(torch::allclose(result.r, expected_r, 1e-5));

    // Validate the dual part
    EXPECT_TRUE(torch::allclose(result.d, expected_d, 1e-5));
}


// Test: Edge case with r = 0
TEST(TensorDualTest, AtanhEdgeCaseZero) {
    // Input TensorDual object with r = 0
    TensorDual x(torch::zeros({2, 2}), torch::randn({2, 2, 3}));

    // Perform hyperbolic arctangent operation
    TensorDual result = x.atanh();

    // Expected results
    auto expected_r = torch::atanh(x.r); // atanh(0) = 0
    auto expected_d = torch::pow(1 - torch::pow(x.r, 2), -1).unsqueeze(-1) * x.d; // f'(0) = 1

    // Validate the real part
    EXPECT_TRUE(torch::allclose(result.r, expected_r, 1e-5));

    // Validate the dual part
    EXPECT_TRUE(torch::allclose(result.d, expected_d, 1e-5));
}

// Test: Invalid domain (r > 1.0 or r < -1.0)
TEST(TensorDualTest, AtanhInvalidDomain) {
    // Input TensorDual object with invalid real values
    TensorDual x(torch::tensor({{1.5, -1.5}}), torch::randn({1, 2, 3}));

    // Expect domain_error exception
    EXPECT_THROW(
        {
            try {
                TensorDual result = x.atanh();
            } catch (const std::domain_error& e) {
                EXPECT_STREQ("All real values must be between -1.0 and 1.0 for atanh.", e.what());
                throw;
            }
        },
        std::domain_error
    );
}

// Test: Incompatible dual dimensions
TEST(TensorDualTest, AtanhIncompatibleDimensions) {
    // Input TensorDual object with mismatched real and dual dimensions
    TensorDual x(torch::rand({2, 2}), torch::rand({2, 3, 2}));

    // Expect invalid_argument exception
    EXPECT_THROW(
        {
            try {
                TensorDual result = x.atanh();
            } catch (const std::invalid_argument& e) {
                EXPECT_STREQ("Dual part dimensions are incompatible with the real part.", e.what());
                throw;
            }
        },
        std::invalid_argument
    );
}

// Test: Large negative values within the domain
TEST(TensorDualTest, AtanhLargeNegativeValues) {
    // Input TensorDual object with large negative values close to -1
    TensorDual x(torch::tensor({{-0.99, -0.999}}), torch::randn({1, 2, 3}));

    // Perform hyperbolic arctangent operation
    TensorDual result = x.atanh();

    // Expected results
    auto expected_r = torch::atanh(x.r);
    auto expected_d = torch::pow(1 - torch::pow(x.r, 2), -1).unsqueeze(-1) * x.d;

    // Validate the real part
    EXPECT_TRUE(torch::allclose(result.r, expected_r, 1e-5));

    // Validate the dual part
    EXPECT_TRUE(torch::allclose(result.d, expected_d, 1e-5));
}

// Test: GPU tensors
TEST(TensorDualTest, AtanhGpuTensors) {
    if (torch::cuda::is_available()) {
        // Input TensorDual object on GPU
        TensorDual x(torch::rand({2, 3}, torch::kCUDA) * 0.9, torch::randn({2, 3, 4}, torch::kCUDA));

        // Perform hyperbolic arctangent operation
        TensorDual result = x.atanh();

        // Expected results
        auto expected_r = torch::atanh(x.r);
        auto expected_d = torch::pow(1 - torch::pow(x.r, 2), -1).unsqueeze(-1) * x.d;

        // Validate the real part
        EXPECT_TRUE(torch::allclose(result.r, expected_r, 1e-5));

        // Validate the dual part
        EXPECT_TRUE(torch::allclose(result.d, expected_d, 1e-5));
    }
}


// Test: Basic functionality of exp
TEST(TensorDualTest, ExpBasic) {
    // Input TensorDual object
    TensorDual x(torch::tensor({{-1.0, 0.0, 1.0}}), 
                 torch::tensor({{{0.1}, {0.2}, {0.3}}}));

    // Perform exponential operation
    TensorDual result = x.exp();

    // Expected results
    auto expected_r = torch::exp(x.r);
    auto expected_d = expected_r.unsqueeze(-1) * x.d;

    // Validate the real part
    EXPECT_TRUE(torch::allclose(result.r, expected_r, 1e-5));

    // Validate the dual part
    EXPECT_TRUE(torch::allclose(result.d, expected_d, 1e-5));
}


// Test: Edge case with r = 0
TEST(TensorDualTest, ExpEdgeCaseZero) {
    // Input TensorDual object with r = 0
    TensorDual x(torch::zeros({2, 2}), torch::randn({2, 2, 3}));

    // Perform exponential operation
    TensorDual result = x.exp();

    // Expected results
    auto expected_r = torch::exp(x.r); // exp(0) = 1
    auto expected_d = expected_r.unsqueeze(-1) * x.d; // f'(0) = exp(0) * d = d

    // Validate the real part
    EXPECT_TRUE(torch::allclose(result.r, expected_r, 1e-5));

    // Validate the dual part
    EXPECT_TRUE(torch::allclose(result.d, expected_d, 1e-5));
}

// Test: Large values
TEST(TensorDualTest, ExpLargeValues) {
    // Input TensorDual object with large real values
    TensorDual x(torch::tensor({{10.0, 20.0}}), torch::randn({1, 2, 3}));

    // Perform exponential operation
    TensorDual result = x.exp();

    // Expected results
    auto expected_r = torch::exp(x.r);
    auto expected_d = expected_r.unsqueeze(-1) * x.d;

    // Validate the real part
    EXPECT_TRUE(torch::allclose(result.r, expected_r, 1e-5));

    // Validate the dual part
    EXPECT_TRUE(torch::allclose(result.d, expected_d, 1e-5));
}

// Test: Incompatible dual dimensions
TEST(TensorDualTest, ExpIncompatibleDimensions) {
    // Input TensorDual object with mismatched real and dual dimensions
    TensorDual x(torch::rand({2, 2}), torch::rand({1, 3, 2}));

    // Expect invalid_argument exception
    EXPECT_THROW(
        {
            try {
                TensorDual result = x.exp();
            } catch (const std::invalid_argument& e) {
                EXPECT_STREQ("Dual part dimensions are incompatible with the real part.", e.what());
                throw;
            }
        },
        std::invalid_argument
    );
}

// Test: GPU tensors
TEST(TensorDualTest, ExpGpuTensors) {
    if (torch::cuda::is_available()) {
        // Input TensorDual object on GPU
        TensorDual x(torch::randn({2, 3}, torch::kCUDA), torch::randn({2, 3, 4}, torch::kCUDA));

        // Perform exponential operation
        TensorDual result = x.exp();

        // Expected results
        auto expected_r = torch::exp(x.r);
        auto expected_d = expected_r.unsqueeze(-1) * x.d;

        // Validate the real part
        EXPECT_TRUE(torch::allclose(result.r, expected_r, 1e-5));

        // Validate the dual part
        EXPECT_TRUE(torch::allclose(result.d, expected_d, 1e-5));
    }
}



// Test: Basic functionality of log
TEST(TensorDualTest, LogBasic) {
    // Input TensorDual object
    TensorDual x(torch::tensor({{1.0, 2.0, 3.0}}), 
                 torch::tensor({{{0.1}, {0.2}, {0.3}}}));

    // Perform natural logarithm operation
    TensorDual result = x.log();

    // Expected results
    auto expected_r = torch::log(x.r);
    auto expected_d = (1 / x.r).unsqueeze(-1) * x.d;

    // Validate the real part
    EXPECT_TRUE(torch::allclose(result.r, expected_r, 1e-5));

    // Validate the dual part
    EXPECT_TRUE(torch::allclose(result.d, expected_d, 1e-5));
}


// Test: Edge case with r = 1
TEST(TensorDualTest, LogEdgeCaseOne) {
    // Input TensorDual object with r = 1
    TensorDual x(torch::ones({2, 2}), torch::randn({2, 2, 3}));

    // Perform natural logarithm operation
    TensorDual result = x.log();

    // Expected results
    auto expected_r = torch::log(x.r); // log(1) = 0
    auto expected_d = (1 / x.r).unsqueeze(-1) * x.d; // f'(1) = 1

    // Validate the real part
    EXPECT_TRUE(torch::allclose(result.r, expected_r, 1e-5));

    // Validate the dual part
    EXPECT_TRUE(torch::allclose(result.d, expected_d, 1e-5));
}

// Test: Invalid domain (r <= 0)
TEST(TensorDualTest, LogInvalidDomain) {
    // Input TensorDual object with invalid real values
    TensorDual x(torch::tensor({{0.0, -1.0}}), torch::randn({2, 2, 3}));

    // Expect domain_error exception
    EXPECT_THROW(
        {
            try {
                TensorDual result = x.log();
            } catch (const std::domain_error& e) {
                EXPECT_STREQ("All real values must be greater than 0 for the natural logarithm.", e.what());
                throw;
            }
        },
        std::domain_error
    );
}

// Test: Incompatible dual dimensions
TEST(TensorDualTest, LogIncompatibleDimensions) {
    // Input TensorDual object with mismatched real and dual dimensions
    TensorDual x(torch::rand({2, 2}), torch::rand({2, 3, 2}));

    // Expect invalid_argument exception
    EXPECT_THROW(
        {
            try {
                TensorDual result = x.log();
            } catch (const std::invalid_argument& e) {
                EXPECT_STREQ("Dual part dimensions are incompatible with the real part.", e.what());
                throw;
            }
        },
        std::invalid_argument
    );
}

// Test: Large positive values
TEST(TensorDualTest, LogLargeValues) {
    // Input TensorDual object with large real values
    TensorDual x(torch::tensor({{100.0, 1000.0}}), torch::randn({1, 2, 3}));

    // Perform natural logarithm operation
    TensorDual result = x.log();

    // Expected results
    auto expected_r = torch::log(x.r);
    auto expected_d = (1 / x.r).unsqueeze(-1) * x.d;

    // Validate the real part
    EXPECT_TRUE(torch::allclose(result.r, expected_r, 1e-5));

    // Validate the dual part
    EXPECT_TRUE(torch::allclose(result.d, expected_d, 1e-5));
}

// Test: GPU tensors
TEST(TensorDualTest, LogGpuTensors) {
    if (torch::cuda::is_available()) {
        // Input TensorDual object on GPU
        TensorDual x(torch::randn({2, 3}, torch::kCUDA).abs() + 1.0, torch::randn({2, 3, 4}, torch::kCUDA));

        // Perform natural logarithm operation
        TensorDual result = x.log();

        // Expected results
        auto expected_r = torch::log(x.r);
        auto expected_d = (1 / x.r).unsqueeze(-1) * x.d;

        // Validate the real part
        EXPECT_TRUE(torch::allclose(result.r, expected_r, 1e-5));

        // Validate the dual part
        EXPECT_TRUE(torch::allclose(result.d, expected_d, 1e-5));
    }
}



// Test: Basic functionality of sqrt
TEST(TensorDualTest, SqrtBasic) {
    // Input TensorDual object
    TensorDual x(torch::tensor({{4.0, 9.0, 16.0}}), 
                 torch::tensor({{{0.1}, {0.2}, {0.3}}}));

    // Perform square root operation
    TensorDual result = x.sqrt();

    // Expected results
    auto expected_r = torch::sqrt(x.r);
    auto expected_d = (0.5 / expected_r).unsqueeze(-1) * x.d;

    // Validate the real part
    EXPECT_TRUE(torch::allclose(result.r, expected_r, 1e-5));

    // Validate the dual part
    EXPECT_TRUE(torch::allclose(result.d, expected_d, 1e-5));
}


// Test: Edge case with r = 1
TEST(TensorDualTest, SqrtEdgeCaseOne) {
    // Input TensorDual object with r = 1
    TensorDual x(torch::ones({2, 2}), torch::randn({2, 2, 3}));

    // Perform square root operation
    TensorDual result = x.sqrt();

    // Expected results
    auto expected_r = torch::sqrt(x.r); // sqrt(1) = 1
    auto expected_d = (0.5 / expected_r).unsqueeze(-1) * x.d;

    // Validate the real part
    EXPECT_TRUE(torch::allclose(result.r, expected_r, 1e-5));

    // Validate the dual part
    EXPECT_TRUE(torch::allclose(result.d, expected_d, 1e-5));
}

// Test: Negative values in real part
TEST(TensorDualTest, SqrtNegativeValues) {
    // Input TensorDual object with negative real values
    TensorDual x(torch::tensor({{-4.0, 9.0}}), torch::randn({1, 2, 3}));

    // Expect that the `complex` conversion is invoked
    EXPECT_NO_THROW({
        TensorDual result = x.sqrt();
    });
}

// Test: Incompatible dual dimensions
TEST(TensorDualTest, SqrtIncompatibleDimensions) {
    // Input TensorDual object with mismatched real and dual dimensions
    TensorDual x(torch::rand({2, 2}), torch::rand({2, 3, 2}));

    // Expect invalid_argument exception
    EXPECT_THROW(
        {
            try {
                TensorDual result = x.sqrt();
            } catch (const std::invalid_argument& e) {
                EXPECT_STREQ("Dual part dimensions are incompatible with the real part.", e.what());
                throw;
            }
        },
        std::invalid_argument
    );
}

// Test: GPU tensors
TEST(TensorDualTest, SqrtGpuTensors) {
    if (torch::cuda::is_available()) {
        // Input TensorDual object on GPU
        TensorDual x(torch::randn({2, 3}, torch::kCUDA).abs() + 1.0, torch::randn({2, 3, 4}, torch::kCUDA));

        // Perform square root operation
        TensorDual result = x.sqrt();

        // Expected results
        auto expected_r = torch::sqrt(x.r);
        auto expected_d = (0.5 / expected_r).unsqueeze(-1) * x.d;

        // Validate the real part
        EXPECT_TRUE(torch::allclose(result.r, expected_r, 1e-5));

        // Validate the dual part
        EXPECT_TRUE(torch::allclose(result.d, expected_d, 1e-5));
    }
}


// Test: Basic functionality of abs
TEST(TensorDualTest, AbsBasic) {
    // Input TensorDual object
    TensorDual x(torch::tensor({{-4.0, 0.0, 5.0}}), 
                 torch::tensor({{{0.1}, {0.2}, {0.3}}}));

    // Perform absolute value operation
    TensorDual result = x.abs();

    // Expected results
    auto expected_r = torch::abs(x.r);
    auto expected_d = torch::sign(x.r).unsqueeze(-1) * x.d;

    // Validate the real part
    EXPECT_TRUE(torch::allclose(result.r, expected_r, 1e-5));

    // Validate the dual part
    EXPECT_TRUE(torch::allclose(result.d, expected_d, 1e-5));
}


// Test: Edge case with r = 0
TEST(TensorDualTest, AbsEdgeCaseZero) {
    // Input TensorDual object with r = 0
    TensorDual x(torch::zeros({2, 2}), torch::randn({2, 2, 3}));

    // Perform absolute value operation
    TensorDual result = x.abs();

    // Expected results
    auto expected_r = torch::abs(x.r); // abs(0) = 0
    auto expected_d = torch::sign(x.r).unsqueeze(-1) * x.d; // sign(0) = 0

    // Validate the real part
    EXPECT_TRUE(torch::allclose(result.r, expected_r, 1e-5));

    // Validate the dual part
    EXPECT_TRUE(torch::allclose(result.d, expected_d, 1e-5));
}

// Test: Negative values in real part
TEST(TensorDualTest, AbsNegativeValues) {
    // Input TensorDual object with negative real values
    TensorDual x(torch::tensor({{-10.0, -20.0}}), torch::randn({1, 2, 3}));

    // Perform absolute value operation
    TensorDual result = x.abs();

    // Expected results
    auto expected_r = torch::abs(x.r);
    auto expected_d = torch::sign(x.r).unsqueeze(-1) * x.d;

    // Validate the real part
    EXPECT_TRUE(torch::allclose(result.r, expected_r, 1e-5));

    // Validate the dual part
    EXPECT_TRUE(torch::allclose(result.d, expected_d, 1e-5));
}

// Test: Incompatible dual dimensions
TEST(TensorDualTest, AbsIncompatibleDimensions) {
    // Input TensorDual object with mismatched real and dual dimensions
    TensorDual x(torch::rand({2, 2}), torch::rand({2, 3, 2}));

    // Expect invalid_argument exception
    EXPECT_THROW(
        {
            try {
                TensorDual result = x.abs();
            } catch (const std::invalid_argument& e) {
                EXPECT_STREQ("Dual part dimensions are incompatible with the real part.", e.what());
                throw;
            }
        },
        std::invalid_argument
    );
}

// Test: GPU tensors
TEST(TensorDualTest, AbsGpuTensors) {
    if (torch::cuda::is_available()) {
        // Input TensorDual object on GPU
        TensorDual x(torch::randn({2, 3}, torch::kCUDA) * 2 - 1.0, torch::randn({2, 3, 4}, torch::kCUDA));

        // Perform absolute value operation
        TensorDual result = x.abs();

        // Expected results
        auto expected_r = torch::abs(x.r);
        auto expected_d = torch::sign(x.r).unsqueeze(-1) * x.d;

        // Validate the real part
        EXPECT_TRUE(torch::allclose(result.r, expected_r, 1e-5));

        // Validate the dual part
        EXPECT_TRUE(torch::allclose(result.d, expected_d, 1e-5));
    }
}



// Test: Basic functionality of sign
TEST(TensorDualTest, SignBasic) {
    // Input TensorDual object
    TensorDual x(torch::tensor({{-4.0, 0.0, 5.0}}), torch::randn({1,3, 1}));

    // Perform sign operation
    TensorDual result = x.sign();

    // Expected results
    auto expected_r = torch::sign(x.r);
    auto expected_d = torch::zeros_like(x.d);

    // Validate the real part
    EXPECT_TRUE(torch::allclose(result.r, expected_r, 1e-5));

    // Validate the dual part
    EXPECT_TRUE(torch::allclose(result.d, expected_d, 1e-5));
}


// Test: Edge case with r = 0
TEST(TensorDualTest, SignEdgeCaseZero) {
    // Input TensorDual object with r = 0
    TensorDual x(torch::zeros({2, 2}), torch::randn({2, 2, 3}));

    // Perform sign operation
    TensorDual result = x.sign();

    // Expected results
    auto expected_r = torch::sign(x.r); // sign(0) = 0
    auto expected_d = torch::zeros_like(x.d); // Dual part remains zero

    // Validate the real part
    EXPECT_TRUE(torch::allclose(result.r, expected_r, 1e-5));

    // Validate the dual part
    EXPECT_TRUE(torch::allclose(result.d, expected_d, 1e-5));
}

// Test: Negative values in real part
TEST(TensorDualTest, SignNegativeValues) {
    // Input TensorDual object with negative real values
    TensorDual x(torch::tensor({{-10.0, -20.0}}), torch::randn({1, 2, 3}));

    // Perform sign operation
    TensorDual result = x.sign();

    // Expected results
    auto expected_r = torch::sign(x.r);
    auto expected_d = torch::zeros_like(x.d);

    // Validate the real part
    EXPECT_TRUE(torch::allclose(result.r, expected_r, 1e-5));

    // Validate the dual part
    EXPECT_TRUE(torch::allclose(result.d, expected_d, 1e-5));
}

// Test: Incompatible dual dimensions
TEST(TensorDualTest, SignIncompatibleDimensions) {
    // Input TensorDual object with mismatched real and dual dimensions
    TensorDual x(torch::rand({2, 2}), torch::rand({2, 3, 2}));

    // Expect invalid_argument exception
    EXPECT_THROW(
        {
            try {
                TensorDual result = x.sign();
            } catch (const std::invalid_argument& e) {
                EXPECT_STREQ("Dual part dimensions are incompatible with the real part.", e.what());
                throw;
            }
        },
        std::invalid_argument
    );
}

// Test: GPU tensors
TEST(TensorDualTest, SignGpuTensors) {
    if (torch::cuda::is_available()) {
        // Input TensorDual object on GPU
        TensorDual x(torch::randn({2, 3}, torch::kCUDA) * 2 - 1.0, torch::randn({2, 3, 4}, torch::kCUDA));

        // Perform sign operation
        TensorDual result = x.sign();

        // Expected results
        auto expected_r = torch::sign(x.r);
        auto expected_d = torch::zeros_like(x.d);

        // Validate the real part
        EXPECT_TRUE(torch::allclose(result.r, expected_r, 1e-5));

        // Validate the dual part
        EXPECT_TRUE(torch::allclose(result.d, expected_d, 1e-5));
    }
}



// Test: Basic functionality of slog
TEST(TensorDualTest, SlogBasic) {
    // Input TensorDual object
    TensorDual x(torch::tensor({{-2.0, 1.0, 3.0}}), torch::randn({1, 3, 1}));

    // Perform slog operation
    TensorDual result = x.slog();

    // Expected results
    auto expected_r = torch::sign(x.r) * torch::log(torch::abs(x.r) + 1.0);
    auto expected_d = x.d / (torch::abs(x.r) + 1.0).unsqueeze(-1);

    // Validate the real part
    EXPECT_TRUE(torch::allclose(result.r, expected_r, 1e-5));

    // Validate the dual part
    EXPECT_TRUE(torch::allclose(result.d, expected_d, 1e-5));
}

// Test: Edge case with positive values
TEST(TensorDualTest, SlogPositiveValues) {
    // Input TensorDual object with positive real values
    TensorDual x(torch::tensor({{2.0, 4.0}}), torch::randn({1,2, 1}));

    // Perform slog operation
    TensorDual result = x.slog();

    // Expected results
    auto expected_r = torch::sign(x.r) * torch::log(torch::abs(x.r) + 1.0);
    auto expected_d = x.d / (torch::abs(x.r) + 1.0).unsqueeze(-1);

    // Validate the real part
    EXPECT_TRUE(torch::allclose(result.r, expected_r, 1e-5));

    // Validate the dual part
    EXPECT_TRUE(torch::allclose(result.d, expected_d, 1e-5));
}

// Test: Negative values in real part
TEST(TensorDualTest, SlogNegativeValues) {
    // Input TensorDual object with negative real values
    TensorDual x(torch::tensor({{-3.0, -5.0}}), torch::randn({1, 2, 1}));

    // Perform slog operation
    TensorDual result = x.slog();

    // Expected results
    auto expected_r = torch::sign(x.r) * torch::log(torch::abs(x.r) + 1.0);
    auto expected_d = x.d / (torch::abs(x.r) + 1.0).unsqueeze(-1);

    // Validate the real part
    EXPECT_TRUE(torch::allclose(result.r, expected_r, 1e-5));

    // Validate the dual part
    EXPECT_TRUE(torch::allclose(result.d, expected_d, 1e-5));
}

// Test: Real part with values near zero
TEST(TensorDualTest, SlogNearZero) {
    // Input TensorDual object with real values near zero
    TensorDual x(torch::tensor({{0.1, -0.2}}), torch::randn({1, 2, 1}));

    // Perform slog operation
    TensorDual result = x.slog();

    // Expected results
    auto expected_r = torch::sign(x.r) * torch::log(torch::abs(x.r) + 1.0);
    auto expected_d = x.d / (torch::abs(x.r) + 1.0).unsqueeze(-1);

    // Validate the real part
    EXPECT_TRUE(torch::allclose(result.r, expected_r, 1e-5));

    // Validate the dual part
    EXPECT_TRUE(torch::allclose(result.d, expected_d, 1e-5));
}

// Test: Real part with zero (throws exception)
TEST(TensorDualTest, SlogZeroRealPart) {
    // Input TensorDual object with zero in the real part
    TensorDual x(torch::tensor({{0.0, 2.0}}), torch::randn({1, 2, 1}));

    // Expect std::domain_error
    EXPECT_THROW(
        {
            try {
                TensorDual result = x.slog();
            } catch (const std::domain_error& e) {
                EXPECT_STREQ("slog is undefined for r = 0.", e.what());
                throw;
            }
        },
        std::domain_error
    );
}

// Test: Incompatible dual dimensions
TEST(TensorDualTest, SlogIncompatibleDualDimensions) {
    // Input TensorDual object with incompatible dual dimensions
    TensorDual x(torch::rand({3, 4}), torch::rand({3, 5, 6}));

    // Expect std::invalid_argument
    EXPECT_THROW(
        {
            try {
                TensorDual result = x.slog();
            } catch (const std::invalid_argument& e) {
                EXPECT_STREQ("Dual part dimensions are incompatible with the real part.", e.what());
                throw;
            }
        },
        std::invalid_argument
    );
}

// Test: GPU tensors
TEST(TensorDualTest, SlogGpuTensors) {
    if (torch::cuda::is_available()) {
        // Input TensorDual object on GPU
        TensorDual x(torch::randn({2, 3}, torch::kCUDA) * 2 - 1.0, torch::randn({2, 3, 4}, torch::kCUDA));

        // Perform slog operation
        TensorDual result = x.slog();

        // Expected results
        auto expected_r = torch::sign(x.r) * torch::log(torch::abs(x.r) + 1.0);
        auto expected_d = x.d / (torch::abs(x.r) + 1.0).unsqueeze(-1);

        // Validate the real part
        EXPECT_TRUE(torch::allclose(result.r, expected_r, 1e-5));

        // Validate the dual part
        EXPECT_TRUE(torch::allclose(result.d, expected_d, 1e-5));
    }
}


// Test: Basic functionality of softsigninv
TEST(TensorDualTest, SoftsigninvBasic) {
    // Input TensorDual object
    TensorDual x(torch::tensor({{-0.5, 0.0, 0.5}}), torch::randn({1, 3, 1}));

    // Perform softsigninv operation
    TensorDual result = x.softsigninv();

    // Expected results
    auto softsigninv_r = x.r / (1.0 - torch::abs(x.r));
    auto scaling_factor = (1.0 - torch::abs(x.r)).pow(-2);
    auto expected_d = scaling_factor.unsqueeze(-1) * x.d;

    // Validate the real part
    EXPECT_TRUE(torch::allclose(result.r, softsigninv_r, 1e-5));

    // Validate the dual part
    EXPECT_TRUE(torch::allclose(result.d, expected_d, 1e-5));
}


// Test: Edge case near boundaries
TEST(TensorDualTest, SoftsigninvBoundaryCase) {
    // Input TensorDual object with values near boundaries
    TensorDual x(torch::tensor({{-0.999, 0.999}}), torch::randn({1, 2, 1}));

    // Perform softsigninv operation
    TensorDual result = x.softsigninv();

    // Expected results
    auto softsigninv_r = x.r / (1.0 - torch::abs(x.r));
    auto scaling_factor = (1.0 - torch::abs(x.r)).pow(-2);
    auto expected_d = scaling_factor.unsqueeze(-1) * x.d;

    // Validate the real part
    EXPECT_TRUE(torch::allclose(result.r, softsigninv_r, 1e-5));

    // Validate the dual part
    EXPECT_TRUE(torch::allclose(result.d, expected_d, 1e-5));
}

// Test: Incompatible dual dimensions
TEST(TensorDualTest, SoftsigninvIncompatibleDualDimensions) {
    // Input TensorDual object with incompatible dual dimensions
    TensorDual x(torch::rand({3, 4}), torch::rand({3, 5, 6}));

    // Expect std::invalid_argument
    EXPECT_THROW(
        {
            try {
                TensorDual result = x.softsigninv();
            } catch (const std::invalid_argument& e) {
                EXPECT_STREQ("Dual part dimensions are incompatible with the real part.", e.what());
                throw;
            }
        },
        std::invalid_argument
    );
}

// Test: Values exceeding |r| >= 1
TEST(TensorDualTest, SoftsigninvOutOfDomain) {
    // Input TensorDual object with |r| >= 1
    TensorDual x(torch::tensor({{1.0, -1.1, 2.0}}), torch::randn({1, 3, 1}));

    // Expect std::domain_error
    EXPECT_THROW(
        {
            try {
                TensorDual result = x.softsigninv();
            } catch (const std::domain_error& e) {
                EXPECT_STREQ("softsigninv is only defined for |r| < 1.0.", e.what());
                throw;
            }
        },
        std::domain_error
    );
}

// Test: GPU tensors
TEST(TensorDualTest, SoftsigninvGpuTensors) {
    if (torch::cuda::is_available()) {
        // Input TensorDual object on GPU
        TensorDual x(torch::randn({2, 3}, torch::kCUDA).clamp(-0.8, 0.8), torch::randn({2, 3, 4}, torch::kCUDA));

        // Perform softsigninv operation
        TensorDual result = x.softsigninv();

        // Expected results
        auto softsigninv_r = x.r / (1.0 - torch::abs(x.r));
        auto scaling_factor = (1.0 - torch::abs(x.r)).pow(-2);
        auto expected_d = scaling_factor.unsqueeze(-1) * x.d;

        // Validate the real part
        EXPECT_TRUE(torch::allclose(result.r, softsigninv_r, 1e-5));

        // Validate the dual part
        EXPECT_TRUE(torch::allclose(result.d, expected_d, 1e-5));
    }
}



// Test: Basic functionality of max
TEST(TensorDualTest, MaxBasic) {
    // Input TensorDual object
    TensorDual x(
        torch::tensor({{1.0, 3.0, 2.0}, {4.0, 2.0, 5.0}}),
        torch::tensor({{{0.1}, {0.3}, {0.2}}, {{0.4}, {0.2}, {0.5}}})
    );

    // Perform max operation
    TensorDual result = x.max();

    // Expected results
    auto max_r = torch::tensor({{3.0}, {5.0}});
    auto max_d = torch::tensor({{{0.3}}, {{0.5}}});

    // Validate the real part
    EXPECT_TRUE(torch::allclose(result.r, max_r, 1e-5));

    // Validate the dual part
    EXPECT_TRUE(torch::allclose(result.d, max_d, 1e-5));
}


// Test: Edge case with identical values
TEST(TensorDualTest, MaxIdenticalValues) {
    // Input TensorDual object
    TensorDual x(
        torch::tensor({{2.0, 2.0, 2.0}, {5.0, 5.0, 5.0}}),
        torch::tensor({{{0.1}, {0.1}, {0.1}}, {{0.5}, {0.5}, {0.5}}})
    );

    // Perform max operation
    TensorDual result = x.max();

    // Expected results
    auto max_r = torch::tensor({{2.0}, {5.0}});
    auto max_d = torch::tensor({{{0.1}}, {{0.5}}});

    // Validate the real part
    EXPECT_TRUE(torch::allclose(result.r, max_r, 1e-5));

    // Validate the dual part
    EXPECT_TRUE(torch::allclose(result.d, max_d, 1e-5));
}

// Test: Incompatible dual dimensions
TEST(TensorDualTest, MaxIncompatibleDualDimensions) {
    // Input TensorDual object with incompatible dimensions
    TensorDual x(
        torch::randn({3, 4}),
        torch::randn({3, 5, 6})
    );

    // Expect std::invalid_argument
    EXPECT_THROW(
        {
            try {
                TensorDual result = x.max();
            } catch (const std::invalid_argument& e) {
                EXPECT_STREQ("Dual part dimensions are incompatible with the real part.", e.what());
                throw;
            }
        },
        std::invalid_argument
    );
}

// Test: GPU tensors
TEST(TensorDualTest, MaxGpuTensors) {
    if (torch::cuda::is_available()) {
        // Input TensorDual object on GPU
        TensorDual x(
            torch::randn({2, 3}, torch::kCUDA),
            torch::randn({2, 3, 4}, torch::kCUDA)
        );

        // Perform max operation
        TensorDual result = x.max();

        // Expected results
        auto max_result = torch::max(x.r, /*dim=*/1, /*keepdim=*/true);
        auto expected_r = std::get<0>(max_result);
        auto max_indices = std::get<1>(max_result);
        auto gather_indices = max_indices.unsqueeze(-1).expand({-1, -1, x.d.size(-1)});
        auto expected_d = torch::gather(x.d, /*dim=*/1, gather_indices);

        // Validate the real part
        EXPECT_TRUE(torch::allclose(result.r, expected_r, 1e-5));

        // Validate the dual part
        EXPECT_TRUE(torch::allclose(result.d, expected_d, 1e-5));
    }
}


// Test: Basic functionality of min
TEST(TensorDualTest, MinBasic) {
    // Input TensorDual object
    TensorDual x(
        torch::tensor({{1.0, 3.0, 2.0}, {4.0, 2.0, 5.0}}),
        torch::tensor({{{0.1}, {0.3}, {0.2}}, {{0.4}, {0.2}, {0.5}}})
    );

    // Perform min operation
    TensorDual result = x.min();

    // Expected results
    auto min_r = torch::tensor({{1.0}, {2.0}});
    auto min_d = torch::tensor({{{0.1}}, {{0.2}}});

    // Validate the real part
    EXPECT_TRUE(torch::allclose(result.r, min_r, 1e-5));

    // Validate the dual part
    EXPECT_TRUE(torch::allclose(result.d, min_d, 1e-5));
}


// Test: Edge case with identical values
TEST(TensorDualTest, MinIdenticalValues) {
    // Input TensorDual object
    TensorDual x(
        torch::tensor({{2.0, 2.0, 2.0}, {5.0, 5.0, 5.0}}),
        torch::tensor({{{0.1}, {0.1}, {0.1}}, {{0.5}, {0.5}, {0.5}}})
    );

    // Perform min operation
    TensorDual result = x.min();

    // Expected results
    auto min_r = torch::tensor({{2.0}, {5.0}});
    auto min_d = torch::tensor({{{0.1}}, {{0.5}}});

    // Validate the real part
    EXPECT_TRUE(torch::allclose(result.r, min_r, 1e-5));

    // Validate the dual part
    EXPECT_TRUE(torch::allclose(result.d, min_d, 1e-5));
}

// Test: Incompatible dual dimensions
TEST(TensorDualTest, MinIncompatibleDualDimensions) {
    // Input TensorDual object with incompatible dimensions
    TensorDual x(
        torch::randn({3, 4}),
        torch::randn({3, 5, 6})
    );

    // Expect std::invalid_argument
    EXPECT_THROW(
        {
            try {
                TensorDual result = x.min();
            } catch (const std::invalid_argument& e) {
                EXPECT_STREQ("Dual part dimensions are incompatible with the real part.", e.what());
                throw;
            }
        },
        std::invalid_argument
    );
}

// Test: GPU tensors
TEST(TensorDualTest, MinGpuTensors) {
    if (torch::cuda::is_available()) {
        // Input TensorDual object on GPU
        TensorDual x(
            torch::randn({2, 3}, torch::kCUDA),
            torch::randn({2, 3, 4}, torch::kCUDA)
        );

        // Perform min operation
        TensorDual result = x.min();

        // Expected results
        auto min_result = torch::min(x.r, /*dim=*/1, /*keepdim=*/true);
        auto expected_r = std::get<0>(min_result);
        auto min_indices = std::get<1>(min_result);
        auto gather_indices = min_indices.unsqueeze(-1).expand({-1, -1, x.d.size(-1)});
        auto expected_d = torch::gather(x.d, /*dim=*/1, gather_indices);

        // Validate the real part
        EXPECT_TRUE(torch::allclose(result.r, expected_r, 1e-5));

        // Validate the dual part
        EXPECT_TRUE(torch::allclose(result.d, expected_d, 1e-5));
    }
}



// Test cases
TEST(TensorDualTest, ComplexAlreadyComplexInputs) {
    auto r = torch::complex(torch::tensor({{1.0, 2.0}}), torch::tensor({{3.0, 4.0}}));
    auto d = torch::complex(torch::tensor({{{5.0, 6.0}, {4.0, 6.3}}}), torch::tensor({{{7.0, 8.0}, {2.0, 6.6}}}));
    TensorDual td(r, d);
    auto result = td.complex();

    EXPECT_TRUE(result.r.is_complex());
    EXPECT_TRUE(result.d.is_complex());
    EXPECT_TRUE(torch::equal(result.r, r));
    EXPECT_TRUE(torch::equal(result.d, d));
}

TEST(TensorDualTest, ComplexRealInputs) {
    auto r = torch::tensor({{1.0, 2.0}});
    auto d = torch::randn({1, 2, 2});
    TensorDual td(r, d);
    auto result = td.complex();

    EXPECT_TRUE(result.r.is_complex());
    EXPECT_TRUE(result.d.is_complex());
    auto expected_r = torch::complex(r, torch::zeros_like(r));
    auto expected_d = torch::complex(d, torch::zeros_like(d));
    EXPECT_TRUE(torch::equal(result.r, expected_r));
    EXPECT_TRUE(torch::equal(result.d, expected_d));
}

TEST(TensorDualTest, ComplexMixedInputs) {
    auto r = torch::complex(torch::tensor({{1.0, 2.0}}), torch::tensor({{3.0, 4.0}})).to(torch::kFloat64);
    auto d = torch::tensor({{{5.0, 6.0}, {7.0, 8.0}}}).to(torch::kFloat64);
    TensorDual td(r, d);
    auto result = td.complex();

    EXPECT_TRUE(result.r.is_complex());
    EXPECT_TRUE(result.d.is_complex());
    auto expected_d = torch::complex(d, torch::zeros_like(d));
    EXPECT_TRUE(torch::equal(result.r, r));
    EXPECT_TRUE(torch::equal(result.d, expected_d));
}


TEST(TensorDualTest, ComplexDeviceCompatibility) {
    if (torch::cuda::is_available()) {
        auto r = torch::tensor({{1.0, 2.0}}, torch::device(torch::kCUDA));
        auto d = torch::tensor({{{3.0, 4.0}, {2.0, 6.6}}}, torch::device(torch::kCUDA));
        TensorDual td(r, d);
        auto result = td.complex();

        EXPECT_TRUE(result.r.is_complex());
        EXPECT_TRUE(result.d.is_complex());
        EXPECT_EQ(result.r.device(), r.device());
        EXPECT_EQ(result.d.device(), d.device());
    }
}

TEST(TensorDualTest, ComplexDtypeCompatibility) {
    auto r = torch::tensor({{1.0, 2.0}}, torch::dtype(torch::kFloat64));
    auto d = torch::tensor({{{3.0, 4.0}, {3.0, 6.0}}}, torch::dtype(torch::kFloat64));
    TensorDual td(r, d);
    auto result = td.complex();

    EXPECT_TRUE(result.r.is_complex());
    EXPECT_TRUE(result.d.is_complex());
    EXPECT_EQ(result.r.dtype(), torch::kComplexDouble);
    EXPECT_EQ(result.d.dtype(), torch::kComplexDouble);
}



// Test cases
TEST(TensorDualTest, RealAlreadyRealInputs) {
    auto r = torch::tensor({{1.0, 2.0}});
    auto d = torch::tensor({{{3.0, 4.0}, {5.0, 6.0}}});
    TensorDual td(r, d);
    auto result = td.real();

    EXPECT_TRUE(torch::equal(result.r, r));
    EXPECT_TRUE(torch::equal(result.d, d));
}

TEST(TensorDualTest, RealComplexInputs) {
    auto r = torch::complex(torch::tensor({{1.0, 2.0}}), torch::tensor({{3.0, 4.0}}));
    auto d = torch::complex(torch::tensor({{{5.0, 6.0}, {2.0, 4.0}}}), torch::tensor({{{7.0, 8.0}, {3.0, 5.0}}}));
    TensorDual td(r, d);
    auto result = td.real();

    auto expected_r = torch::tensor({{1.0, 2.0}});
    auto expected_d = torch::tensor({{{5.0, 6.0}, {2.0, 4.0}}});
    EXPECT_TRUE(torch::equal(result.r, expected_r));
    EXPECT_TRUE(torch::equal(result.d, expected_d));
}

TEST(TensorDualTest, RealMixedInputs) {
    auto r = torch::complex(torch::tensor({{1.0, 2.0}}), torch::tensor({{3.0, 4.0}})).to(torch::kFloat64);
    auto d = torch::tensor({{{5.0, 6.0}, {7.0, 8.0}}}).to(torch::kFloat64);
    TensorDual td(r, d);
    auto result = td.real();

    auto expected_r = torch::tensor({{1.0, 2.0}});
    EXPECT_TRUE(torch::equal(result.r, expected_r));
    std::cerr << "result.d=" << result.d << std::endl;
    std::cerr << "d=" << d << std::endl;
    EXPECT_TRUE(torch::equal(result.d, d));
}


TEST(TensorDualTest, RealDeviceCompatibility) {
    if (torch::cuda::is_available()) {
        auto r = torch::complex(torch::tensor({{1.0, 2.0}}, torch::device(torch::kCUDA)),
                                torch::tensor({{3.0, 4.0}}, torch::device(torch::kCUDA)));
        auto d = torch::complex(torch::tensor({{{5.0, 6.0}, {3.5, 6.9}}}, torch::device(torch::kCUDA)),
                                torch::tensor({{{7.0, 8.0}, {3.7, 4.8}}}, torch::device(torch::kCUDA)));
        TensorDual td(r, d);
        auto result = td.real();

        EXPECT_EQ(result.r.device(), r.device());
        EXPECT_EQ(result.d.device(), d.device());
    }
}

TEST(TensorDualTest, RealDtypeCompatibility) {
    auto r = torch::complex(torch::tensor({{1.0, 2.0}}, torch::dtype(torch::kFloat64)),
                            torch::tensor({{3.0, 4.0}}, torch::dtype(torch::kFloat64)));
    auto d = torch::complex(torch::tensor({{{5.0, 6.0}, {4.7, 5.9}}}, torch::dtype(torch::kFloat64)),
                            torch::tensor({{{7.0, 8.0}, {3.0, 6.5}}}, torch::dtype(torch::kFloat64)));
    TensorDual td(r, d);
    auto result = td.real();

    EXPECT_EQ(result.r.dtype(), torch::kFloat64);
    EXPECT_EQ(result.d.dtype(), torch::kFloat64);
}





// Test cases
TEST(TensorDualTest, ImagAlreadyComplexInputs) {
    auto r = torch::complex(torch::tensor({{1.0, 2.0}}), torch::tensor({{3.0, 4.0}}));
    auto d = torch::complex(torch::tensor({{{5.0, 6.0}, {3.5, 2.9}}}), torch::tensor({{{7.0, 8.0}, {3.7, 4.8}}}));
    TensorDual td(r, d);
    auto result = td.imag();

    auto expected_r = torch::tensor({{3.0, 4.0}});
    auto expected_d = torch::tensor({{{7.0, 8.0}, {3.7, 4.8}}});

    EXPECT_TRUE(torch::equal(result.r, expected_r));
    EXPECT_TRUE(torch::equal(result.d, expected_d));
}

TEST(TensorDualTest, ImagRealInputs) {
    auto r = torch::tensor({{1.0, 2.0}});
    auto d = torch::tensor({{{3.0, 4.0}, {5.0, 6.0}}});
    TensorDual td(r, d);
    auto result = td.imag();

    auto expected_r = torch::zeros_like(r);
    auto expected_d = torch::zeros_like(d);

    EXPECT_TRUE(torch::equal(result.r, expected_r));
    EXPECT_TRUE(torch::equal(result.d, expected_d));
}

TEST(TensorDualTest, ImagMixedInputs) {
    auto r = torch::complex(torch::tensor({{1.0, 2.0}}).to(torch::kFloat64), 
                            torch::tensor({{3.0, 4.0}}).to(torch::kFloat64));
    auto d = torch::tensor({{{5.0, 6.0}, {7.0, 8.0}}}).to(torch::kFloat64);
    TensorDual td(r, d);
    auto result = td.imag();

    auto expected_r = torch::tensor({{3.0, 4.0}}).to(torch::kFloat64);
    auto expected_d = torch::zeros_like(d);
    
    EXPECT_TRUE(torch::equal(result.r, expected_r));
    EXPECT_TRUE(torch::equal(result.d, expected_d));
}


TEST(TensorDualTest, ImagDeviceCompatibility) {
    if (torch::cuda::is_available()) {
        auto r = torch::complex(torch::tensor({{1.0, 2.0}}, torch::device(torch::kCUDA)),
                                torch::tensor({{3.0, 4.0}}, torch::device(torch::kCUDA)));
        auto d = torch::complex(torch::tensor({{{5.0, 6.0}, {3.6, 7.4}}}, torch::device(torch::kCUDA)),
                                torch::tensor({{{7.0, 8.0}, {2.6, 8.3}}}, torch::device(torch::kCUDA)));
        TensorDual td(r, d);
        auto result = td.imag();

        EXPECT_EQ(result.r.device(), r.device());
        EXPECT_EQ(result.d.device(), d.device());
    }
}

TEST(TensorDualTest, ImagDtypeCompatibility) {
    auto r = torch::complex(torch::tensor({{1.0, 2.0}}, torch::dtype(torch::kFloat64)),
                            torch::tensor({{3.0, 4.0}}, torch::dtype(torch::kFloat64)));
    auto d = torch::complex(torch::tensor({{{5.0, 6.0}, {5.3, 5.1}}}, torch::dtype(torch::kFloat64)),
                            torch::tensor({{{7.0, 8.0}, {1.5, 8.7}}}, torch::dtype(torch::kFloat64)));
    TensorDual td(r, d);
    auto result = td.imag();

    EXPECT_EQ(result.r.dtype(), torch::kFloat64);
    EXPECT_EQ(result.d.dtype(), torch::kFloat64);
}



// Test cases
TEST(TensorDualTest, IndexValidIndices) {
    auto r = torch::tensor({{1.0, 2.0}, {3.0, 4.0}});
    auto d = torch::tensor({{{5.0, 6.0}, {7.0, 8.0}}, {{9.0, 10.0}, {11.0, 12.0}}});
    TensorDual td(r, d);

    auto result = td.index({torch::indexing::Slice(0, 1), torch::indexing::Slice(0, 2)});
    auto expected_r = torch::tensor({{1.0, 2.0}});
    auto expected_d = torch::tensor({{{5.0, 6.0}, {7.0, 8.0}}});

    EXPECT_TRUE(torch::equal(result.r, expected_r));
    EXPECT_TRUE(torch::equal(result.d, expected_d));
}


TEST(TensorDualTest, IndexOutOfBounds) {
    auto r = torch::tensor({{1.0, 2.0}, {3.0, 4.0}});
    auto d = torch::tensor({{{5.0, 6.0}, {7.0, 8.0}}, {{9.0, 10.0}, {11.0, 12.0}}});
    TensorDual td(r, d);

    EXPECT_THROW(td.index({torch::indexing::TensorIndex(2)}), c10::Error);
}

TEST(TensorDualTest, IndexInvalidDimensions) {
    auto r = torch::tensor({{1.0, 2.0}, {3.0, 4.0}});
    auto d = torch::tensor({{{5.0, 6.0}, {7.0, 8.0}}, {{9.0, 10.0}, {11.0, 12.0}}});
    TensorDual td(r, d);

    EXPECT_THROW(td.index({torch::indexing::Slice(0, 1), torch::indexing::Slice(0, 2), torch::indexing::TensorIndex(1)}),
                 std::invalid_argument);
}


TEST(TensorDualTest, IndexDeviceCompatibility) {
    if (torch::cuda::is_available()) {
        auto r = torch::tensor({{1.0, 2.0}, {3.0, 4.0}}, torch::device(torch::kCUDA));
        auto d = torch::tensor({{{5.0, 6.0}, {7.0, 8.0}}, {{9.0, 10.0}, {11.0, 12.0}}}, torch::device(torch::kCUDA));
        TensorDual td(r, d);
        std::vector<torch::indexing::TensorIndex> indices{torch::indexing::Slice(0, 1)};
        auto result = td.index(indices);
        EXPECT_EQ(result.r.device(), r.device());
        EXPECT_EQ(result.d.device(), d.device());
    }
}




// Test cases
TEST(TensorDualTest, IndexPutValidInput) {
    auto r = torch::tensor({{1.0, 2.0}, {3.0, 4.0}});
    auto d = torch::tensor({{{5.0, 6.0}, {7.0, 8.0}}, {{9.0, 10.0}, {11.0, 12.0}}});
    TensorDual td(r, d);

    auto mask = torch::tensor({true, false});
    auto value_r = torch::tensor({{10.0, 20.0}, {30.0, 40.0}});
    auto value_d = torch::tensor({{{15.0, 16.0}, {17.0, 18.0}}, {{19.0, 20.0}, {21.0, 22.0}}});
    TensorDual value(value_r, value_d);

    td.index_put_(mask, value.index(mask));

    auto expected_r = torch::tensor({{10.0, 20.0}, {3.0, 4.0}});
    auto expected_d = torch::tensor({{{15.0, 16.0}, {17.0, 18.0}}, {{9.0, 10.0}, {11.0, 12.0}}});

    EXPECT_TRUE(torch::equal(td.r, expected_r));
    EXPECT_TRUE(torch::equal(td.d, expected_d));
}

TEST(TensorDualTest, IndexPutInvalidMaskType) {
    auto r = torch::tensor({{1.0, 2.0}, {3.0, 4.0}});
    auto d = torch::tensor({{{5.0, 6.0}, {7.0, 8.0}}, {{9.0, 10.0}, {11.0, 12.0}}});
    TensorDual td(r, d);

    auto mask = torch::tensor({{1, 0}, {0, 1}}, torch::dtype(torch::kInt));
    auto value_r = torch::tensor({{10.0, 20.0}, {30.0, 40.0}});
    auto value_d = torch::tensor({{{15.0, 16.0}, {17.0, 18.0}}, {{19.0, 20.0}, {21.0, 22.0}}});
    TensorDual value(value_r, value_d);

    EXPECT_THROW(td.index_put_(mask, value), std::invalid_argument);
}





TEST(TensorDualTest, einsumTest4)
{
    auto r1 = torch::randn({2, 2});
    auto d1 = torch::randn({2, 2, 3});
    auto r2 = torch::randn({2, 2});
    auto d2 = torch::randn({2, 2, 3});
    auto r3 = torch::randn({2, 2});
    auto d3 = torch::randn({2, 2, 3});

    TensorDual td1(r1, d1);
    TensorDual td2(r2, d2);
    TensorDual td3(r3, d3);
    std::vector<TensorDual> tensors = {td1, td2, td3};
    TensorDual result = TensorDual::einsum("mi,mi,mi->mi", tensors);
    auto r = torch::einsum("mi,mi,mi->mi", {r1, r2, r3});
    auto d = torch::einsum("miz,mi,mi->miz", {d1, r2, r3}) +
             torch::einsum("mi,miz,mi->miz", {r1, d2, r3}) +
             torch::einsum("mi,mi,miz->miz", {r1, r2, d3});
    EXPECT_TRUE(torch::allclose(result.r,r));
    EXPECT_TRUE(torch::allclose(result.d,d));           
}

TEST(TensorDualTest, TensorDualTensorDualmult)
{
    auto r1 = torch::randn({2, 2}).to(torch::kFloat64).requires_grad_(false);
    auto d1 = torch::zeros({2, 2, 8}).to(torch::kFloat64).requires_grad_(false);
    auto r2 = torch::randn({2, 2}).to(torch::kFloat64).requires_grad_(false);
    auto d2 = torch::zeros({2, 2, 8}).to(torch::kFloat64).requires_grad_(false);
    //Set the dual parts to 1.0 so there is exactly one per element
    d1.index_put_({0,0,0}, 1.0);
    d1.index_put_({1,0,1}, 1.0);
    d1.index_put_({0,1,2}, 1.0);
    d1.index_put_({1,1,3}, 1.0);
    d2.index_put_({0,0,4}, 1.0);
    d2.index_put_({1,0,5}, 1.0);
    d2.index_put_({0,1,6}, 1.0);
    d2.index_put_({1,1,7}, 1.0);

    TensorDual td1(r1, d1);
    TensorDual td2(r2, d2);
    TensorDual result = td1 * td2;
    //Compare to the result of the derivative expected analytically
    
    EXPECT_TRUE(result.d.index({0,0,0}).item<double>() == r2.index({0,0}).item<double>());
    EXPECT_TRUE(result.d.index({1,0,1}).item<double>() == r2.index({1,0}).item<double>());
    EXPECT_TRUE(result.d.index({0,1,2}).item<double>() == r2.index({0,1}).item<double>());
    EXPECT_TRUE(result.d.index({1,1,3}).item<double>() == r2.index({1,1}).item<double>());
    EXPECT_TRUE(result.d.index({0,0,4}).item<double>() == r1.index({0,0}).item<double>());
    EXPECT_TRUE(result.d.index({1,0,5}).item<double>() == r1.index({1,0}).item<double>());
    EXPECT_TRUE(result.d.index({0,1,6}).item<double>() == r1.index({0,1}).item<double>());
    EXPECT_TRUE(result.d.index({1,1,7}).item<double>() == r1.index({1,1}).item<double>());

}

TEST(TensorDualTest, TensorDualSum)
{
    auto r1 = torch::randn({2, 2}).to(torch::kFloat64).requires_grad_(false);
    auto d1 = torch::zeros({2, 2, 4}).to(torch::kFloat64).requires_grad_(false);
    //Set the dual parts to 1.0 so there is exactly one per element
    d1.index_put_({0,0,0}, 1.0);
    d1.index_put_({1,0,1}, 1.0);
    d1.index_put_({0,1,2}, 1.0);
    d1.index_put_({1,1,3}, 1.0);

    TensorDual td1(r1, d1);
    //Sum over the first dimension
    TensorDual result = td1.sum();
    //calculate the result using back propagation
    torch::Tensor r = r1.clone().requires_grad_(true);
    auto J = r.sum(1);
    J.backward(torch::ones_like(J));
    
    EXPECT_TRUE(result.d.index({0,0,0}).item<double>() == r.grad().index({0,0}).item<double>());
    EXPECT_TRUE(result.d.index({1,0,1}).item<double>() == r.grad().index({1,0}).item<double>());
    EXPECT_TRUE(result.d.index({0,0,2}).item<double>() == r.grad().index({0,1}).item<double>());
    EXPECT_TRUE(result.d.index({1,0,3}).item<double>() == r.grad().index({1,1}).item<double>());
}

TEST(TensorDualTest, TensorDualnormL2)
{
    auto r1 = torch::randn({2, 2}).to(torch::kFloat64).requires_grad_(false);
    auto d1 = torch::zeros({2, 2, 4}).to(torch::kFloat64).requires_grad_(false);
    //Set the dual parts to 1.0 so there is exactly one per element
    d1.index_put_({0,0,0}, 1.0);
    d1.index_put_({1,0,1}, 1.0);
    d1.index_put_({0,1,2}, 1.0);
    d1.index_put_({1,1,3}, 1.0);

    TensorDual td1(r1, d1);
    //Sum over the first dimension
    TensorDual result = td1.normL2();
    //calculate the result using back propagation
    torch::Tensor r = r1.clone().requires_grad_(true);
    auto J = torch::norm(r, 2, 1);
    J.backward(torch::ones_like(J));
    
    EXPECT_TRUE(result.d.index({0,0,0}).item<double>() == r.grad().index({0,0}).item<double>());
    EXPECT_TRUE(result.d.index({1,0,1}).item<double>() == r.grad().index({1,0}).item<double>());
    EXPECT_TRUE(result.d.index({0,0,2}).item<double>() == r.grad().index({0,1}).item<double>());
    EXPECT_TRUE(result.d.index({1,0,3}).item<double>() == r.grad().index({1,1}).item<double>());
}

TEST(TensorDualTest, TensorDualSumStatic)
{
    auto r1 = torch::randn({2, 2}).to(torch::kFloat64).requires_grad_(false);
    auto d1 = torch::zeros({2, 2, 4}).to(torch::kFloat64).requires_grad_(false);
    //Set the dual parts to 1.0 so there is exactly one per element
    d1.index_put_({0,0,0}, 1.0);
    d1.index_put_({1,0,1}, 1.0);
    d1.index_put_({0,1,2}, 1.0);
    d1.index_put_({1,1,3}, 1.0);

    TensorDual td1(r1, d1);
    //Sum over the first dimension
    TensorDual result = TensorDual::sum(td1);
    //calculate the result using back propagation
    torch::Tensor r = r1.clone().requires_grad_(true);
    auto J = r.sum(1);
    J.backward(torch::ones_like(J));
    
    EXPECT_TRUE(result.d.index({0,0,0}).item<double>() == r.grad().index({0,0}).item<double>());
    EXPECT_TRUE(result.d.index({1,0,1}).item<double>() == r.grad().index({1,0}).item<double>());
    EXPECT_TRUE(result.d.index({0,0,2}).item<double>() == r.grad().index({0,1}).item<double>());
    EXPECT_TRUE(result.d.index({1,0,3}).item<double>() == r.grad().index({1,1}).item<double>());
}

TEST(TensorDualTest, TensorDualUnitMinus )
{
    auto r1 = torch::randn({2, 2}).to(torch::kFloat64).requires_grad_(false);
    auto d1 = torch::rand({2, 2, 2}).to(torch::kFloat64).requires_grad_(false);
    //Set the dual parts to 1.0 so there is exactly one per element
    //d1.index_put_({Slice(),0,0}, 1.0);
    //d1.index_put_({Slice(),1,1}, 1.0);

    TensorDual td1(r1.clone(), d1.clone());
    //Sum over the first dimension
    TensorDual result = -td1;
    //calculate the result using back propagation
    torch::Tensor r = r1.clone().requires_grad_(true);
    auto J = -r;
    auto jac = janus::compute_batch_jacobian(J, r);
    std::cerr << jac << std::endl;
    std::cerr << torch::einsum("mij, mjk->mik",{jac, d1}) << std::endl;
    std::cerr << result.d << std::endl;
    EXPECT_TRUE(torch::allclose(result.d, torch::einsum("mij, mjk->mik",{jac, d1})));
    //EXPECT_TRUE(result.d.index({1,0,1}).item<double>() == jac.index({1,0,1}).item<double>());
}

TEST(TensorDualTest, TensorDualPlusOther)
{
    auto r1 = torch::randn({2, 2}).to(torch::kFloat64).requires_grad_(false);
    auto d1 = torch::rand({2, 2, 2}).to(torch::kFloat64).requires_grad_(false);
    auto r2 = torch::randn({2, 2}).to(torch::kFloat64).requires_grad_(false);
    auto d2 = torch::rand({2, 2, 2}).to(torch::kFloat64).requires_grad_(false);
    //Set the dual parts to 1.0 so there is exactly one per element
    //d1.index_put_({Slice(),0,0}, 1.0);
    //d1.index_put_({Slice(),1,1}, 1.0);

    TensorDual td1(r1.clone(), d1.clone());
    TensorDual td2(r2.clone(), d2.clone());
    //Sum over the first dimension
    TensorDual result = td1 + td2;
    //calculate the result using back propagation
    torch::Tensor r = r1.clone().requires_grad_(true);
    torch::Tensor r3 = r2.clone().requires_grad_(true);
    auto J = r + r3;
    auto jac = janus::compute_batch_jacobian(J, r);
    std::cerr << jac << std::endl;
    std::cerr << torch::einsum("mij, mjk->mik",{jac, d1+d2}) << std::endl;
    std::cerr << result.d << std::endl;
    EXPECT_TRUE(torch::allclose(result.d, torch::einsum("mij, mjk->mik",{jac, d1+d2})));
}

TEST(TensorDualTest, TensorDualPlusOtherTensor)
{
    auto r1 = torch::randn({2, 2}).to(torch::kFloat64).requires_grad_(false);
    auto d1 = torch::rand({2, 2, 2}).to(torch::kFloat64).requires_grad_(false);
    auto r2 = torch::randn({2, 2}).to(torch::kFloat64).requires_grad_(false);
    //Set the dual parts to 1.0 so there is exactly one per element
    //d1.index_put_({Slice(),0,0}, 1.0);
    //d1.index_put_({Slice(),1,1}, 1.0);

    TensorDual td1(r1.clone(), d1.clone());
    //Sum over the first dimension
    TensorDual result = td1 + r2;
    //calculate the result using back propagation
    torch::Tensor r = r1.clone().requires_grad_(true);
    torch::Tensor r3 = r2.clone().requires_grad_(true);
    auto J = r + r3;
    auto jac = janus::compute_batch_jacobian(J, r);
    std::cerr << jac << std::endl;
    std::cerr << torch::einsum("mij, mjk->mik",{jac, d1}) << std::endl;
    std::cerr << result.d << std::endl;
    EXPECT_TRUE(torch::allclose(result.d, torch::einsum("mij, mjk->mik",{jac, d1})));
}

TEST(TensorDualTest, TensorDualPlusOtherDouble)
{
    auto r1 = torch::randn({2, 2}).to(torch::kFloat64).requires_grad_(false);
    auto d1 = torch::rand({2, 2, 2}).to(torch::kFloat64).requires_grad_(false);
    auto r2 = 1.3;
    //Set the dual parts to 1.0 so there is exactly one per element
    //d1.index_put_({Slice(),0,0}, 1.0);
    //d1.index_put_({Slice(),1,1}, 1.0);

    TensorDual td1(r1.clone(), d1.clone());
    //Sum over the first dimension
    TensorDual result = td1 + r2;
    //calculate the result using back propagation
    torch::Tensor r = r1.clone().requires_grad_(true);
    auto J = r + r2;
    auto jac = janus::compute_batch_jacobian(J, r);
    std::cerr << jac << std::endl;
    std::cerr << torch::einsum("mij, mjk->mik",{jac, d1}) << std::endl;
    std::cerr << result.d << std::endl;
    EXPECT_TRUE(torch::allclose(result.d, torch::einsum("mij, mjk->mik",{jac, d1})));
}

TEST(TensorDualTest, TensorDualMinusOther)
{
    auto r1 = torch::randn({2, 2}).to(torch::kFloat64).requires_grad_(false);
    auto d1 = torch::rand({2, 2, 2}).to(torch::kFloat64).requires_grad_(false);
    auto r2 = torch::randn({2, 2}).to(torch::kFloat64).requires_grad_(false);
    auto d2 = torch::rand({2, 2, 2}).to(torch::kFloat64).requires_grad_(false);
    //Set the dual parts to 1.0 so there is exactly one per element
    //d1.index_put_({Slice(),0,0}, 1.0);
    //d1.index_put_({Slice(),1,1}, 1.0);

    TensorDual td1(r1.clone(), d1.clone());
    TensorDual td2(r2.clone(), d2.clone());
    //Sum over the first dimension
    TensorDual result = td1 - td2;
    //calculate the result using back propagation
    torch::Tensor r = r1.clone().requires_grad_(true);
    torch::Tensor r3 = r2.clone().requires_grad_(true);
    auto J = r - r3;
    auto jac = janus::compute_batch_jacobian(J, r);
    std::cerr << jac << std::endl;
    std::cerr << torch::einsum("mij, mjk->mik",{jac, d1-d2}) << std::endl;
    std::cerr << result.d << std::endl;
    EXPECT_TRUE(torch::allclose(result.d, torch::einsum("mij, mjk->mik",{jac, d1-d2})));
}

TEST(TensorDualTest, TensorDualMinusOtherTensor)
{
    auto r1 = torch::randn({2, 2}).to(torch::kFloat64).requires_grad_(false);
    auto d1 = torch::rand({2, 2, 2}).to(torch::kFloat64).requires_grad_(false);
    auto r2 = torch::randn({2, 2}).to(torch::kFloat64).requires_grad_(false);

    TensorDual td1(r1.clone(), d1.clone());
    //Sum over the first dimension
    TensorDual result = td1 - r2;
    //calculate the result using back propagation
    torch::Tensor r = r1.clone().requires_grad_(true);
    torch::Tensor r3 = r2.clone().requires_grad_(true);
    auto J = r - r3;
    auto jac = janus::compute_batch_jacobian(J, r);
    std::cerr << jac << std::endl;
    std::cerr << torch::einsum("mij, mjk->mik",{jac, d1}) << std::endl;
    std::cerr << result.d << std::endl;
    EXPECT_TRUE(torch::allclose(result.d, torch::einsum("mij, mjk->mik",{jac, d1})));
}


TEST(TensorDualTest, TensorDualMinusOtherDouble)
{
    auto r1 = torch::randn({2, 2}).to(torch::kFloat64).requires_grad_(false);
    auto d1 = torch::rand({2, 2, 2}).to(torch::kFloat64).requires_grad_(false);
    auto r2 = 1.8;

    TensorDual td1(r1.clone(), d1.clone());
    //Sum over the first dimension
    TensorDual result = td1 - r2;
    //calculate the result using back propagation
    torch::Tensor r = r1.clone().requires_grad_(true);
    auto J = r - r2;
    auto jac = janus::compute_batch_jacobian(J, r);
    std::cerr << jac << std::endl;
    std::cerr << torch::einsum("mij, mjk->mik",{jac, d1}) << std::endl;
    std::cerr << result.d << std::endl;
    EXPECT_TRUE(torch::allclose(result.d, torch::einsum("mij, mjk->mik",{jac, d1})));
}

TEST(TensorDualTest, TensorDualMultiplyOther)
{
    auto r1 = torch::randn({2, 2}).to(torch::kFloat64).requires_grad_(false);
    auto d1 = torch::rand({2, 2, 2}).to(torch::kFloat64).requires_grad_(false);
    auto r2 = torch::randn({2, 2}).to(torch::kFloat64).requires_grad_(false);
    auto d2 = torch::rand({2, 2, 2}).to(torch::kFloat64).requires_grad_(false);
    //Set the dual parts to 1.0 so there is exactly one per element

    TensorDual td1(r1.clone(), d1.clone());
    TensorDual td2(r2.clone(), d2.clone());
    //Sum over the first dimension
    TensorDual result = td1*td2;
    //calculate the result using back propagation
    torch::Tensor r3 = r1.clone().requires_grad_(true);
    torch::Tensor r4 = r2.clone().requires_grad_(false);
    auto J1 = r3*r4;
    //The output is on the row side and the input on the column side
    auto jac1 = janus::compute_batch_jacobian(J1, r3);
    torch::Tensor r5 = r1.clone().requires_grad_(false);
    torch::Tensor r6 = r2.clone().requires_grad_(true);
    auto J2 = r5*r6;
    auto jac2 = janus::compute_batch_jacobian(J2, r6);
    //concatenate the two jacobians row wise
    auto jac = torch::bmm(jac1,d1)+ torch::bmm(jac2,d2); //Use the chain rule
    //std::cerr << torch::einsum("mij, mjk->mik",{jac, d1}) << std::endl;
    //std::cerr << result.d << std::endl;
    EXPECT_TRUE(torch::allclose(result.d, jac));
}

TEST(TensorDualTest, TensorDualMultiplyOtherTensor)
{
    auto r1 = torch::randn({2, 2}).to(torch::kFloat64).requires_grad_(false);
    auto d1 = torch::rand({2, 2, 2}).to(torch::kFloat64).requires_grad_(false);
    auto r2 = torch::randn({2, 2}).to(torch::kFloat64).requires_grad_(false);
    //Set the dual parts to 1.0 so there is exactly one per element

    TensorDual td1(r1.clone(), d1.clone());
    //Sum over the first dimension
    TensorDual result = td1*r2;
    //calculate the result using back propagation
    torch::Tensor r3 = r1.clone().requires_grad_(true);
    torch::Tensor r4 = r2.clone().requires_grad_(false);
    auto J1 = r3*r4;
    //The output is on the row side and the input on the column side
    auto jac1 = janus::compute_batch_jacobian(J1, r3);
    torch::Tensor r5 = r1.clone().requires_grad_(false);
    torch::Tensor r6 = r2.clone().requires_grad_(true);
    auto J2 = r5*r6;
    auto jac2 = janus::compute_batch_jacobian(J2, r6);
    //concatenate the two jacobians row wise
    auto jac = torch::bmm(jac1,d1); //Use the chain rule
    //std::cerr << torch::einsum("mij, mjk->mik",{jac, d1}) << std::endl;
    //std::cerr << result.d << std::endl;
    EXPECT_TRUE(torch::allclose(result.d, jac));
}


TEST(TensorDualTest, TensorDualMultiplyOtherDouble)
{
    auto r1 = torch::randn({2, 2}).to(torch::kFloat64).requires_grad_(false);
    auto d1 = torch::rand({2, 2, 2}).to(torch::kFloat64).requires_grad_(false);
    auto r2 = 1.8;
    //Set the dual parts to 1.0 so there is exactly one per element

    TensorDual td1(r1.clone(), d1.clone());
    //Sum over the first dimension
    TensorDual result = td1*r2;
    //calculate the result using back propagation
    torch::Tensor r3 = r1.clone().requires_grad_(true);
    auto J1 = r3*r2;
    //The output is on the row side and the input on the column side
    auto jac1 = janus::compute_batch_jacobian(J1, r3);
    auto jac = torch::bmm(jac1,d1); //Use the chain rule
    //std::cerr << torch::einsum("mij, mjk->mik",{jac, d1}) << std::endl;
    //std::cerr << result.d << std::endl;
    EXPECT_TRUE(torch::allclose(result.d, jac));
}


TEST(TensorDualTest, TensorDualDivideOther)
{
    auto r1 = torch::randn({2, 2}).to(torch::kFloat64).requires_grad_(false);
    auto d1 = torch::rand({2, 2, 2}).to(torch::kFloat64).requires_grad_(false);
    auto r2 = torch::randn({2, 2}).to(torch::kFloat64).requires_grad_(false);
    auto d2 = torch::rand({2, 2, 2}).to(torch::kFloat64).requires_grad_(false);
    //Set the dual parts to 1.0 so there is exactly one per element

    TensorDual td1(r1.clone(), d1.clone());
    TensorDual td2(r2.clone(), d2.clone());
    
    TensorDual result = td1/td2;
    //calculate the result using back propagation
    torch::Tensor r3 = r1.clone().requires_grad_(true);
    torch::Tensor r4 = r2.clone().requires_grad_(false);
    auto J1 = r3/r4;
    //The output is on the row side and the input on the column side
    auto jac1 = janus::compute_batch_jacobian(J1, r3);
    torch::Tensor r5 = r1.clone().requires_grad_(false);
    torch::Tensor r6 = r2.clone().requires_grad_(true);
    auto J2 = r5/r6;
    auto jac2 = janus::compute_batch_jacobian(J2, r6);
    //concatenate the two jacobians row wise
    auto jac = torch::bmm(jac1,d1)+ torch::bmm(jac2,d2); //Use the chain rule
    std::cerr << jac << std::endl;
    //std::cerr << torch::einsum("mij, mjk->mik",{jac, d1}) << std::endl;
    std::cerr << result.d << std::endl;
    EXPECT_TRUE(torch::allclose(result.d, jac));
}

TEST(TensorDualTest, TensorDualDivideOtherTensor)
{
    auto r1 = torch::randn({2, 2}).to(torch::kFloat64).requires_grad_(false);
    auto d1 = torch::rand({2, 2, 2}).to(torch::kFloat64).requires_grad_(false);
    auto r2 = torch::randn({2, 2}).to(torch::kFloat64).requires_grad_(false);
    //Set the dual parts to 1.0 so there is exactly one per element

    TensorDual td1(r1.clone(), d1.clone());

    
    TensorDual result = td1/r2;
    //calculate the result using back propagation
    torch::Tensor r3 = r1.clone().requires_grad_(true);
    torch::Tensor r4 = r2.clone().requires_grad_(false);
    auto J1 = r3/r4;
    //The output is on the row side and the input on the column side
    auto jac1 = janus::compute_batch_jacobian(J1, r3);
    auto jac = torch::bmm(jac1,d1); //Use the chain rule
    //std::cerr << torch::einsum("mij, mjk->mik",{jac, d1}) << std::endl;
    //std::cerr << result.d << std::endl;
    EXPECT_TRUE(torch::allclose(result.d, jac));
}

TEST(TensorDualTest, TensorDualDivideOtherDouble)
{
    auto r1 = torch::randn({2, 2}).to(torch::kFloat64).requires_grad_(false);
    auto d1 = torch::rand({2, 2, 2}).to(torch::kFloat64).requires_grad_(false);
    std::random_device rd;  // Seed
    std::mt19937 gen(rd()); // Mersenne Twister RNG

    // Define the range
    double lower_bound = 0.0;
    double upper_bound = 1.0;

    // Create a distribution
    std::uniform_real_distribution<> dis(lower_bound, upper_bound);

    // Generate a random double
    double r2 = dis(gen);

    //Set the dual parts to 1.0 so there is exactly one per element

    TensorDual td1(r1.clone(), d1.clone());

    
    TensorDual result = td1/r2;
    //calculate the result using back propagation
    torch::Tensor r3 = r1.clone().requires_grad_(true);
    auto J1 = r3/r2;
    //The output is on the row side and the input on the column side
    auto jac1 = janus::compute_batch_jacobian(J1, r3);
    auto jac = torch::bmm(jac1,d1); //Use the chain rule
    //std::cerr << torch::einsum("mij, mjk->mik",{jac, d1}) << std::endl;
    //std::cerr << result.d << std::endl;
    EXPECT_TRUE(torch::allclose(result.d, jac));
}

TEST(TensorDualTest, TensorDualReciprocal)
{
    auto r1 = torch::randn({2, 2}).to(torch::kFloat64).requires_grad_(false);
    auto d1 = torch::rand({2, 2, 2}).to(torch::kFloat64).requires_grad_(false);

    //Set the dual parts to 1.0 so there is exactly one per element

    TensorDual td1(r1.clone(), d1.clone());

    
    TensorDual result = td1.reciprocal();
    //calculate the result using back propagation
    torch::Tensor r3 = r1.clone().requires_grad_(true);
    auto J1 = r3.reciprocal();
    //The output is on the row side and the input on the column side
    auto jac1 = janus::compute_batch_jacobian(J1, r3);
    auto jac = torch::bmm(jac1,d1); //Use the chain rule
    //std::cerr << torch::einsum("mij, mjk->mik",{jac, d1}) << std::endl;
    //std::cerr << result.d << std::endl;
    EXPECT_TRUE(torch::allclose(result.d, jac));
}

TEST(TensorDualTest, TensorDualSquare)
{
    auto r1 = torch::randn({2, 2}).to(torch::kFloat64).requires_grad_(false);
    auto d1 = torch::rand({2, 2, 2}).to(torch::kFloat64).requires_grad_(false);

    //Set the dual parts to 1.0 so there is exactly one per element

    TensorDual td1(r1.clone(), d1.clone());

    
    TensorDual result = td1.square();
    //calculate the result using back propagation
    torch::Tensor r3 = r1.clone().requires_grad_(true);
    auto J1 = r3.square();
    //The output is on the row side and the input on the column side
    auto jac1 = janus::compute_batch_jacobian(J1, r3);
    auto jac = torch::bmm(jac1,d1); //Use the chain rule
    //std::cerr << torch::einsum("mij, mjk->mik",{jac, d1}) << std::endl;
    //std::cerr << result.d << std::endl;
    EXPECT_TRUE(torch::allclose(result.d, jac));
}


TEST(TensorDualTest, TensorDualSin)
{
    auto r1 = torch::randn({2, 2}).to(torch::kFloat64).requires_grad_(false);
    auto d1 = torch::rand({2, 2, 2}).to(torch::kFloat64).requires_grad_(false);

    //Set the dual parts to 1.0 so there is exactly one per element

    TensorDual td1(r1.clone(), d1.clone());

    
    TensorDual result = td1.sin();
    //calculate the result using back propagation
    torch::Tensor r3 = r1.clone().requires_grad_(true);
    auto J1 = r3.sin();
    //The output is on the row side and the input on the column side
    auto jac1 = janus::compute_batch_jacobian(J1, r3);
    auto jac = torch::bmm(jac1,d1); //Use the chain rule
    //std::cerr << torch::einsum("mij, mjk->mik",{jac, d1}) << std::endl;
    //std::cerr << result.d << std::endl;
    EXPECT_TRUE(torch::allclose(result.d, jac));
}

TEST(TensorDualTest, TensorDualCos)
{
    auto r1 = torch::randn({2, 2}).to(torch::kFloat64).requires_grad_(false);
    auto d1 = torch::rand({2, 2, 2}).to(torch::kFloat64).requires_grad_(false);

    //Set the dual parts to 1.0 so there is exactly one per element

    TensorDual td1(r1.clone(), d1.clone());

    
    TensorDual result = td1.cos();
    //calculate the result using back propagation
    torch::Tensor r3 = r1.clone().requires_grad_(true);
    auto J1 = r3.cos();
    //The output is on the row side and the input on the column side
    auto jac1 = janus::compute_batch_jacobian(J1, r3);
    auto jac = torch::bmm(jac1,d1); //Use the chain rule
    //std::cerr << torch::einsum("mij, mjk->mik",{jac, d1}) << std::endl;
    //std::cerr << result.d << std::endl;
    EXPECT_TRUE(torch::allclose(result.d, jac));
}

TEST(TensorDualTest, TensorDualTan)
{
    auto r1 = torch::randn({2, 2}).to(torch::kFloat64).requires_grad_(false);
    auto d1 = torch::rand({2, 2, 2}).to(torch::kFloat64).requires_grad_(false);

    //Set the dual parts to 1.0 so there is exactly one per element

    TensorDual td1(r1.clone(), d1.clone());

    
    TensorDual result = td1.tan();
    //calculate the result using back propagation
    torch::Tensor r3 = r1.clone().requires_grad_(true);
    auto J1 = r3.tan();
    //The output is on the row side and the input on the column side
    auto jac1 = janus::compute_batch_jacobian(J1, r3);
    auto jac = torch::bmm(jac1,d1); //Use the chain rule
    //std::cerr << torch::einsum("mij, mjk->mik",{jac, d1}) << std::endl;
    //std::cerr << result.d << std::endl;
    EXPECT_TRUE(torch::allclose(result.d, jac));
}


TEST(TensorDualTest, TensorDualASin)
{
    auto r1 = torch::rand({2, 2}).to(torch::kFloat64).requires_grad_(false);
    auto d1 = torch::rand({2, 2, 2}).to(torch::kFloat64).requires_grad_(false);

    //Set the dual parts to 1.0 so there is exactly one per element

    TensorDual td1(r1.clone(), d1.clone());

    
    TensorDual result = td1.asin();
    //calculate the result using back propagation
    torch::Tensor r3 = r1.clone().requires_grad_(true);
    auto J1 = r3.asin();
    //The output is on the row side and the input on the column side
    auto jac1 = janus::compute_batch_jacobian(J1, r3);
    auto jac = torch::bmm(jac1,d1); //Use the chain rule
    //std::cerr << torch::einsum("mij, mjk->mik",{jac, d1}) << std::endl;
    //std::cerr << result.d << std::endl;
    EXPECT_TRUE(torch::allclose(result.d, jac));
}

TEST(TensorDualTest, TensorDualACos)
{
    auto r1 = torch::rand({2, 2}).to(torch::kFloat64).requires_grad_(false);
    auto d1 = torch::rand({2, 2, 2}).to(torch::kFloat64).requires_grad_(false);

    //Set the dual parts to 1.0 so there is exactly one per element

    TensorDual td1(r1.clone(), d1.clone());

    
    TensorDual result = td1.acos();
    //calculate the result using back propagation
    torch::Tensor r3 = r1.clone().requires_grad_(true);
    auto J1 = r3.acos();
    //The output is on the row side and the input on the column side
    auto jac1 = janus::compute_batch_jacobian(J1, r3);
    auto jac = torch::bmm(jac1,d1); //Use the chain rule
    //std::cerr << torch::einsum("mij, mjk->mik",{jac, d1}) << std::endl;
    //std::cerr << result.d << std::endl;
    EXPECT_TRUE(torch::allclose(result.d, jac));
}

TEST(TensorDualTest, TensorDualATan)
{
    auto r1 = torch::rand({2, 2}).to(torch::kFloat64).requires_grad_(false);
    auto d1 = torch::rand({2, 2, 2}).to(torch::kFloat64).requires_grad_(false);

    //Set the dual parts to 1.0 so there is exactly one per element

    TensorDual td1(r1.clone(), d1.clone());

    
    TensorDual result = td1.atan();
    //calculate the result using back propagation
    torch::Tensor r3 = r1.clone().requires_grad_(true);
    auto J1 = r3.atan();
    //The output is on the row side and the input on the column side
    auto jac1 = janus::compute_batch_jacobian(J1, r3);
    auto jac = torch::bmm(jac1,d1); //Use the chain rule
    //std::cerr << torch::einsum("mij, mjk->mik",{jac, d1}) << std::endl;
    //std::cerr << result.d << std::endl;
    EXPECT_TRUE(torch::allclose(result.d, jac));
}

TEST(TensorDualTest, TensorDualSinh)
{
    auto r1 = torch::rand({2, 2}).to(torch::kFloat64).requires_grad_(false);
    auto d1 = torch::rand({2, 2, 2}).to(torch::kFloat64).requires_grad_(false);

    //Set the dual parts to 1.0 so there is exactly one per element

    TensorDual td1(r1.clone(), d1.clone());

    
    TensorDual result = td1.sinh();
    //calculate the result using back propagation
    torch::Tensor r3 = r1.clone().requires_grad_(true);
    auto J1 = r3.sinh();
    //The output is on the row side and the input on the column side
    auto jac1 = janus::compute_batch_jacobian(J1, r3);
    auto jac = torch::bmm(jac1,d1); //Use the chain rule
    //std::cerr << torch::einsum("mij, mjk->mik",{jac, d1}) << std::endl;
    //std::cerr << result.d << std::endl;
    EXPECT_TRUE(torch::allclose(result.d, jac));
}

TEST(TensorDualTest, TensorDualCosh)
{
    auto r1 = torch::rand({2, 2}).to(torch::kFloat64).requires_grad_(false);
    auto d1 = torch::rand({2, 2, 2}).to(torch::kFloat64).requires_grad_(false);

    //Set the dual parts to 1.0 so there is exactly one per element

    TensorDual td1(r1.clone(), d1.clone());

    
    TensorDual result = td1.cosh();
    //calculate the result using back propagation
    torch::Tensor r3 = r1.clone().requires_grad_(true);
    auto J1 = r3.cosh();
    //The output is on the row side and the input on the column side
    auto jac1 = janus::compute_batch_jacobian(J1, r3);
    auto jac = torch::bmm(jac1,d1); //Use the chain rule
    //std::cerr << torch::einsum("mij, mjk->mik",{jac, d1}) << std::endl;
    //std::cerr << result.d << std::endl;
    EXPECT_TRUE(torch::allclose(result.d, jac));
}

TEST(TensorDualTest, TensorDualTanh)
{
    auto r1 = torch::rand({2, 2}).to(torch::kFloat64).requires_grad_(false);
    auto d1 = torch::rand({2, 2, 2}).to(torch::kFloat64).requires_grad_(false);

    //Set the dual parts to 1.0 so there is exactly one per element

    TensorDual td1(r1.clone(), d1.clone());

    
    TensorDual result = td1.tanh();
    //calculate the result using back propagation
    torch::Tensor r3 = r1.clone().requires_grad_(true);
    auto J1 = r3.tanh();
    //The output is on the row side and the input on the column side
    auto jac1 = janus::compute_batch_jacobian(J1, r3);
    auto jac = torch::bmm(jac1,d1); //Use the chain rule
    //std::cerr << torch::einsum("mij, mjk->mik",{jac, d1}) << std::endl;
    //std::cerr << result.d << std::endl;
    EXPECT_TRUE(torch::allclose(result.d, jac));
}

TEST(TensorDualTest, TensorDualASinh)
{
    auto r1 = torch::rand({2, 2}).to(torch::kFloat64).requires_grad_(false);
    auto d1 = torch::rand({2, 2, 2}).to(torch::kFloat64).requires_grad_(false);

    //Set the dual parts to 1.0 so there is exactly one per element

    TensorDual td1(r1.clone(), d1.clone());

    
    TensorDual result = td1.asinh();
    //calculate the result using back propagation
    torch::Tensor r3 = r1.clone().requires_grad_(true);
    auto J1 = r3.asinh();
    //The output is on the row side and the input on the column side
    auto jac1 = janus::compute_batch_jacobian(J1, r3);
    auto jac = torch::bmm(jac1,d1); //Use the chain rule
    //std::cerr << torch::einsum("mij, mjk->mik",{jac, d1}) << std::endl;
    //std::cerr << result.d << std::endl;
    EXPECT_TRUE(torch::allclose(result.d, jac));
}

TEST(TensorDualTest, TensorDualACosh)
{
    auto r1 = torch::rand({2, 2}).to(torch::kFloat64).abs().requires_grad_(false)+1.0; //This is only correct for x>=1
    auto d1 = torch::rand({2, 2, 2}).to(torch::kFloat64).requires_grad_(false);

    //Set the dual parts to 1.0 so there is exactly one per element

    TensorDual td1(r1.clone(), d1.clone());

    
    TensorDual result = td1.acosh();
    //calculate the result using back propagation
    torch::Tensor r3 = r1.clone().requires_grad_(true);
    auto J1 = r3.acosh();
    //The output is on the row side and the input on the column side
    auto jac1 = janus::compute_batch_jacobian(J1, r3);
    auto jac = torch::bmm(jac1,d1); //Use the chain rule
    //std::cerr << torch::einsum("mij, mjk->mik",{jac, d1}) << std::endl;
    //std::cerr << result.d << std::endl;
    EXPECT_TRUE(torch::allclose(result.d, jac));
}


TEST(TensorDualTest, TensorDualATanh)
{
    auto r1 = torch::rand({2, 2}).to(torch::kFloat64).requires_grad_(false); //This is only correct for x>=1
    auto d1 = torch::rand({2, 2, 2}).to(torch::kFloat64).requires_grad_(false);

    //Set the dual parts to 1.0 so there is exactly one per element

    TensorDual td1(r1.clone(), d1.clone());

    
    TensorDual result = td1.atanh();
    //calculate the result using back propagation
    torch::Tensor r3 = r1.clone().requires_grad_(true);
    auto J1 = r3.atanh();
    //The output is on the row side and the input on the column side
    auto jac1 = janus::compute_batch_jacobian(J1, r3);
    auto jac = torch::bmm(jac1,d1); //Use the chain rule
    //std::cerr << torch::einsum("mij, mjk->mik",{jac, d1}) << std::endl;
    //std::cerr << result.d << std::endl;
    EXPECT_TRUE(torch::allclose(result.d, jac));
}


TEST(TensorDualTest, TensorDualExp)
{
    auto r1 = torch::rand({2, 2}).to(torch::kFloat64).requires_grad_(false); //This is only correct for x>=1
    auto d1 = torch::rand({2, 2, 2}).to(torch::kFloat64).requires_grad_(false);

    //Set the dual parts to 1.0 so there is exactly one per element

    TensorDual td1(r1.clone(), d1.clone());

    
    TensorDual result = td1.exp();
    //calculate the result using back propagation
    torch::Tensor r3 = r1.clone().requires_grad_(true);
    auto J1 = r3.exp();
    //The output is on the row side and the input on the column side
    auto jac1 = janus::compute_batch_jacobian(J1, r3);
    auto jac = torch::bmm(jac1,d1); //Use the chain rule
    //std::cerr << torch::einsum("mij, mjk->mik",{jac, d1}) << std::endl;
    //std::cerr << result.d << std::endl;
    EXPECT_TRUE(torch::allclose(result.d, jac));
}


TEST(TensorDualTest, TensorDualLog)
{
    auto r1 = torch::rand({2, 2}).to(torch::kFloat64).abs().requires_grad_(false); //This is only correct for x>=1
    auto d1 = torch::rand({2, 2, 2}).to(torch::kFloat64).requires_grad_(false);

    //Set the dual parts to 1.0 so there is exactly one per element

    TensorDual td1(r1.clone(), d1.clone());

    
    TensorDual result = td1.log();
    //calculate the result using back propagation
    torch::Tensor r3 = r1.clone().requires_grad_(true);
    auto J1 = r3.log();
    //The output is on the row side and the input on the column side
    auto jac1 = janus::compute_batch_jacobian(J1, r3);
    auto jac = torch::bmm(jac1,d1); //Use the chain rule
    //std::cerr << torch::einsum("mij, mjk->mik",{jac, d1}) << std::endl;
    //std::cerr << result.d << std::endl;
    EXPECT_TRUE(torch::allclose(result.d, jac));
}



TEST(TensorDualTest, TensorDualSoftSign)
{
    auto r1 = torch::rand({2, 2}).to(torch::kFloat64).abs().requires_grad_(false); //This is only correct for x>=1
    auto d1 = torch::rand({2, 2, 2}).to(torch::kFloat64).requires_grad_(false);

    //Set the dual parts to 1.0 so there is exactly one per element

    TensorDual td1(r1.clone(), d1.clone());

    
    TensorDual result = td1.softsign();
    //calculate the result using back propagation
    torch::Tensor r3 = r1.clone().requires_grad_(true);
    auto J1 = torch::nn::functional::softsign(r3);
    //The output is on the row side and the input on the column side
    auto jac1 = janus::compute_batch_jacobian(J1, r3);
    auto jac = torch::bmm(jac1,d1); //Use the chain rule
    //std::cerr << torch::einsum("mij, mjk->mik",{jac, d1}) << std::endl;
    //std::cerr << result.d << std::endl;
    EXPECT_TRUE(torch::allclose(result.d, jac));
}

// Define the inverse softsign function
torch::Tensor inverse_softsign(const torch::Tensor& y) {
    return y / (1 - y.abs());
}

TEST(TensorDualTest, TensorDualSoftSignInv)
{
    auto r1 = torch::rand({2, 2}).to(torch::kFloat64).abs().requires_grad_(false); //This is only correct for x>=1
    auto d1 = torch::rand({2, 2, 2}).to(torch::kFloat64).requires_grad_(false);

    //Set the dual parts to 1.0 so there is exactly one per element

    TensorDual td1(r1.clone(), d1.clone());

    
    TensorDual result = td1.softsigninv();
    //calculate the result using back propagation
    torch::Tensor r3 = r1.clone().requires_grad_(true);
    auto J1 = inverse_softsign(r3);
    //The output is on the row side and the input on the column side
    auto jac1 = janus::compute_batch_jacobian(J1, r3);
    auto jac = torch::bmm(jac1,d1); //Use the chain rule
    //std::cerr << torch::einsum("mij, mjk->mik",{jac, d1}) << std::endl;
    //std::cerr << result.d << std::endl;
    EXPECT_TRUE(torch::allclose(result.d, jac));
}



TEST(TensorDualTest, TensorDualSqrt)
{
    auto r1 = torch::rand({2, 2}).to(torch::kFloat64).abs().requires_grad_(false); //This is only correct for x>=1
    auto d1 = torch::rand({2, 2, 2}).to(torch::kFloat64).requires_grad_(false);

    //Set the dual parts to 1.0 so there is exactly one per element

    TensorDual td1(r1.clone(), d1.clone());

    
    TensorDual result = td1.sqrt();
    //calculate the result using back propagation
    torch::Tensor r3 = r1.clone().requires_grad_(true);
    auto J1 = r3.sqrt();
    //The output is on the row side and the input on the column side
    auto jac1 = janus::compute_batch_jacobian(J1, r3);
    auto jac = torch::bmm(jac1,d1); //Use the chain rule
    //std::cerr << torch::einsum("mij, mjk->mik",{jac, d1}) << std::endl;
    //std::cerr << result.d << std::endl;
    EXPECT_TRUE(torch::allclose(result.d, jac));
}

TEST(TensorDualTest, TensorDualAbs)
{
    auto r1 = torch::rand({2, 2}).to(torch::kFloat64).requires_grad_(false); //This is only correct for x>=1
    auto d1 = torch::rand({2, 2, 2}).to(torch::kFloat64).requires_grad_(false);

    //Set the dual parts to 1.0 so there is exactly one per element

    TensorDual td1(r1.clone(), d1.clone());

    
    TensorDual result = td1.abs();
    //calculate the result using back propagation
    torch::Tensor r3 = r1.clone().requires_grad_(true);
    auto J1 = r3.abs();
    //The output is on the row side and the input on the column side
    auto jac1 = janus::compute_batch_jacobian(J1, r3);
    auto jac = torch::bmm(jac1,d1); //Use the chain rule
    //std::cerr << torch::einsum("mij, mjk->mik",{jac, d1}) << std::endl;
    //std::cerr << result.d << std::endl;
    EXPECT_TRUE(torch::allclose(result.d, jac));
}

TEST(TensorDualTest, TensorDualSign)
{
    auto r1 = torch::rand({2, 2}).to(torch::kFloat64).requires_grad_(false); //This is only correct for x>=1
    auto d1 = torch::rand({2, 2, 2}).to(torch::kFloat64).requires_grad_(false);

    //Set the dual parts to 1.0 so there is exactly one per element

    TensorDual td1(r1.clone(), d1.clone());

    
    TensorDual result = td1.sign();
    //calculate the result using back propagation
    torch::Tensor r3 = r1.clone().requires_grad_(true);
    auto J1 = r3.sign();
    //The output is on the row side and the input on the column side
    auto jac1 = janus::compute_batch_jacobian(J1, r3);
    auto jac = torch::bmm(jac1,d1); //Use the chain rule
    //std::cerr << torch::einsum("mij, mjk->mik",{jac, d1}) << std::endl;
    //std::cerr << result.d << std::endl;
    EXPECT_TRUE(torch::allclose(result.d, jac));
}

TEST(TensorDualTest, TensorSLog)
{
    auto r1 = torch::rand({2, 2}).to(torch::kFloat64).requires_grad_(false); //This is only correct for x>=1
    auto d1 = torch::rand({2, 2, 2}).to(torch::kFloat64).requires_grad_(false);

    //Set the dual parts to 1.0 so there is exactly one per element

    TensorDual td1(r1.clone(), d1.clone());

    
    TensorDual result = td1.slog();
    //calculate the result using back propagation
    torch::Tensor r3 = r1.clone().requires_grad_(true);
    auto J1 = r3.sign()*(r3.abs()+1).log();
    //The output is on the row side and the input on the column side
    auto jac1 = janus::compute_batch_jacobian(J1, r3);
    auto jac = torch::bmm(jac1,d1); //Use the chain rule
    //std::cerr << torch::einsum("mij, mjk->mik",{jac, d1}) << std::endl;
    //std::cerr << result.d << std::endl;
    EXPECT_TRUE(torch::allclose(result.d, jac));
}

TEST(TensorDualTest, TensorSLogInv)
{
    auto r1 = torch::rand({2, 2}).to(torch::kFloat64).requires_grad_(false); //This is only correct for x>=1
    auto d1 = torch::rand({2, 2, 2}).to(torch::kFloat64).requires_grad_(false);

    //Set the dual parts to 1.0 so there is exactly one per element

    TensorDual td1(r1.clone(), d1.clone());

    
    TensorDual result = td1.sloginv();
    //calculate the result using back propagation
    torch::Tensor r3 = r1.clone().requires_grad_(true);
    auto J1 = ((r3.abs()).exp()-1);
    //The output is on the row side and the input on the column side
    auto jac1 = janus::compute_batch_jacobian(J1, r3);
    auto jac = torch::bmm(jac1,d1); //Use the chain rule
    //std::cerr << torch::einsum("mij, mjk->mik",{jac, d1}) << std::endl;
    //std::cerr << result.d << std::endl;
    EXPECT_TRUE(torch::allclose(result.d, jac));
}


TEST(TensorDualTest, TensorMax)
{
    auto r1 = torch::rand({2, 2}).to(torch::kFloat64).requires_grad_(false); //This is only correct for x>=1
    auto d1 = torch::rand({2, 2, 2}).to(torch::kFloat64).requires_grad_(false);

    //Set the dual parts to 1.0 so there is exactly one per element

    TensorDual td1(r1.clone(), d1.clone());

    
    TensorDual result = td1.max();
    //calculate the result using back propagation
    torch::Tensor r3 = r1.clone().requires_grad_(true);
    auto J1 = r3.amax(1, true);
    //The output is on the row side and the input on the column side
    auto jac1 = janus::compute_batch_jacobian(J1, r3);
    auto jac = torch::bmm(jac1,d1); //Use the chain rule
    EXPECT_TRUE(torch::allclose(result.d, jac));
}

TEST(TensorDualTest, TensorMin)
{
    auto r1 = torch::rand({2, 2}).to(torch::kFloat64).requires_grad_(false); //This is only correct for x>=1
    auto d1 = torch::rand({2, 2, 2}).to(torch::kFloat64).requires_grad_(false);

    //Set the dual parts to 1.0 so there is exactly one per element

    TensorDual td1(r1.clone(), d1.clone());

    
    TensorDual result = td1.min();
    //calculate the result using back propagation
    torch::Tensor r3 = r1.clone().requires_grad_(true);
    auto J1 = r3.amin(1, true);
    //The output is on the row side and the input on the column side
    auto jac1 = janus::compute_batch_jacobian(J1, r3);
    auto jac = torch::bmm(jac1,d1); //Use the chain rule
    EXPECT_TRUE(torch::allclose(result.d, jac));
}

TEST(TensorDualTest, TensorDualComplex)
{
    auto r1 = torch::rand({2, 2}).to(torch::kFloat64).requires_grad_(false); //This is only correct for x>=1
    auto d1 = torch::rand({2, 2, 2}).to(torch::kFloat64).requires_grad_(false);

    //Set the dual parts to 1.0 so there is exactly one per element

    TensorDual td1(r1.clone(), d1.clone());

    
    TensorDual result = td1.complex();
    EXPECT_TRUE(torch::is_complex(result.r));
    EXPECT_TRUE(torch::is_complex(result.d));
    EXPECT_TRUE(torch::allclose(torch::real(result.r), r1));
    EXPECT_TRUE(torch::allclose(torch::real(result.d), d1));
}

TEST(TensorDualTest, TensorDualPow)
{
    auto r1 = torch::rand({2, 2}).to(torch::kFloat64).requires_grad_(false); //This is only correct for x>=1
    auto d1 = torch::rand({2, 2, 2}).to(torch::kFloat64).requires_grad_(false);
    //generate a random double number
    std::random_device rd;  // Seed
    std::mt19937 gen(rd()); // Mersenne Twister RNG

    // Define the range
    double lower_bound = -100.0;
    double upper_bound = 100.0;

    // Create a distribution
    std::uniform_real_distribution<> dis(lower_bound, upper_bound);
    double exponent = dis(gen);

    //Set the dual parts to 1.0 so there is exactly one per element

    TensorDual td1(r1.clone(), d1.clone());

    
    TensorDual result = td1.pow(exponent);
    torch::Tensor r3 = r1.clone().requires_grad_(true);
    auto J1 = r3.pow(exponent);
    //The output is on the row side and the input on the column side
    auto jac1 = janus::compute_batch_jacobian(J1, r3);
    auto jac = torch::bmm(jac1,d1); //Use the chain rule
    //std::cerr << torch::einsum("mij, mjk->mik",{jac, d1}) << std::endl;
    //std::cerr << result.d << std::endl;
    EXPECT_TRUE(torch::allclose(result.d, jac));

}


TEST(TensorMatDualTest, TensorMatDualComplex)
{
    auto r1 = torch::rand({2, 2, 2}).to(torch::kFloat64).requires_grad_(false); //This is only correct for x>=1
    auto d1 = torch::rand({2, 2, 2, 4}).to(torch::kFloat64).requires_grad_(false);

    //Set the dual parts to 1.0 so there is exactly one per element

    TensorMatDual td1(r1.clone(), d1.clone());

    
    TensorMatDual result = td1.complex();
    EXPECT_TRUE(torch::is_complex(result.r));
    EXPECT_TRUE(torch::is_complex(result.d));
    EXPECT_TRUE(torch::allclose(torch::real(result.r), r1));
    EXPECT_TRUE(torch::allclose(torch::real(result.d), d1));
}

TEST(TensorMatDualTest, TensorMatDualSum1)
{
    auto r1 = torch::rand({2, 2, 3}).to(torch::kFloat64).requires_grad_(false);
    auto d1 = torch::rand({2, 2, 3, 4}).to(torch::kFloat64).requires_grad_(false);

    TensorMatDual td1(r1, d1);
    //Sum over the first dimension
    TensorMatDual result = td1.sum(1);
    //calculate the result using back propagation
    torch::Tensor r = r1.clone().requires_grad_(true);
    auto J = r.sum(1);
    auto jac = janus::compute_batch_jacobian2d(J, r);
    auto jacres = torch::einsum("bijk, bjkd->bid", {jac, d1}).unsqueeze(1);
    EXPECT_TRUE(torch::allclose(result.d, jacres));
}

TEST(TensorMatDualTest, TensorMatDualSum2)
{
    auto r1 = torch::rand({2, 2, 3}).to(torch::kFloat64).requires_grad_(false);
    auto d1 = torch::rand({2, 2, 3, 4}).to(torch::kFloat64).requires_grad_(false);

    TensorMatDual td1(r1, d1);
    //Sum over the first dimension
    TensorMatDual result = td1.sum(2);
    //calculate the result using back propagation
    torch::Tensor r = r1.clone().requires_grad_(true);
    auto J = r.sum(2);
    auto jac = janus::compute_batch_jacobian2d(J, r);
    auto jacres = torch::einsum("bijk, bjkd->bid", {jac, d1}).unsqueeze(2);
    EXPECT_TRUE(torch::allclose(result.d, jacres));
}

TEST(TensorMatDualTest, TensorMatDualSquare)
{
    auto r1 = torch::rand({2, 2, 3}).to(torch::kFloat64).requires_grad_(false);
    auto d1 = torch::rand({2, 2, 3, 4}).to(torch::kFloat64).requires_grad_(false);

    TensorMatDual td1(r1, d1);
    //Sum over the first dimension
    TensorMatDual result = td1.square();
    //calculate the result using back propagation
    torch::Tensor r = r1.clone().requires_grad_(true);
    auto J = r.square();
    auto jac = janus::compute_batch_jacobian3d(J, r);//This will have dimension 2,2,3,2,3
    auto jacres = torch::einsum("bijkl, bkld->bijd", {jac, d1});
    EXPECT_TRUE(torch::allclose(result.d, jacres));
}

TEST(TensorMatDualTest, TensorMatDualSqrt)
{
    auto r1 = torch::rand({2, 2, 3}).to(torch::kFloat64).requires_grad_(false);
    auto d1 = torch::rand({2, 2, 3, 4}).to(torch::kFloat64).requires_grad_(false);

    TensorMatDual td1(r1, d1);
    //Sum over the first dimension
    TensorMatDual result = td1.sqrt();
    //calculate the result using back propagation
    torch::Tensor r = r1.clone().requires_grad_(true);
    auto J = r.sqrt();
    auto jac = janus::compute_batch_jacobian3d(J, r);//This will have dimension 2,2,3,2,3
    auto jacres = torch::einsum("bijkl, bkld->bijd", {jac, d1});
    std::cerr << result.d << std::endl;
    std::cerr << jacres << std::endl;
    EXPECT_TRUE(torch::allclose(result.d, jacres));
}


TEST(TensorMatDualTest, TensorMatDualNormL2)
{
    auto r1 = torch::rand({2, 2, 3}).to(torch::kFloat64).requires_grad_(false);
    auto d1 = torch::rand({2, 2, 3, 4}).to(torch::kFloat64).requires_grad_(false);

    TensorMatDual td1(r1, d1);
    //Sum over the first dimension
    TensorMatDual result = td1.normL2();
    //calculate the result using back propagation
    torch::Tensor r = r1.clone().requires_grad_(true);
    auto J = r.norm(2, 2); //This will have dimension 2,2
    auto jac = janus::compute_batch_jacobian2d(J, r);//This will have dimension 2,2,2,3
    auto jacres = torch::einsum("bikl, bkld->bid", {jac, d1}).unsqueeze(2);
    EXPECT_TRUE(torch::allclose(result.d, jacres));
}




TEST(TensorMatDualTest, TensorMatDualAddTensorMatDual)
{
    auto r1 = torch::rand({2, 2, 3}).to(torch::kFloat64).requires_grad_(false);
    auto d1 = torch::rand({2, 2, 3, 4}).to(torch::kFloat64).requires_grad_(false);
    auto r2 = torch::rand({2, 2, 3}).to(torch::kFloat64).requires_grad_(false);
    auto d2 = torch::rand({2, 2, 3, 4}).to(torch::kFloat64).requires_grad_(false);

    TensorMatDual td1(r1, d1);
    TensorMatDual td2(r2, d2);
    //Sum over the first dimension
    TensorMatDual result = td1+td2;
    //calculate the result using back propagation
    torch::Tensor r3 = r1.clone().requires_grad_(true);
    torch::Tensor r4 = r2.clone().requires_grad_(false);
    auto J1 = r3+r4; //This will have dimension 2,2
    auto jac1 = janus::compute_batch_jacobian3d(J1, r3);//This will have dimension 2,2,3,2,3
    torch::Tensor r5 = r1.clone().requires_grad_(false);
    torch::Tensor r6 = r2.clone().requires_grad_(true);
    auto J2 = r5+r6; //This will have dimension 2,2
    auto jac2 = janus::compute_batch_jacobian3d(J2, r6);//This will have dimension 2,2,3,2,3
    //This is a combination of the two partial derivatives
    auto jacres = torch::einsum("bijkl, bkld->bijd", {jac1, d1})+
                  torch::einsum("bijkl, bkld->bijd", {jac2, d2});
    EXPECT_TRUE(torch::allclose(result.d, jacres));
}

TEST(TensorMatDualTest, TensorMatDualSubTensorMatDual)
{
    auto r1 = torch::rand({2, 2, 3}).to(torch::kFloat64).requires_grad_(false);
    auto d1 = torch::rand({2, 2, 3, 4}).to(torch::kFloat64).requires_grad_(false);
    auto r2 = torch::rand({2, 2, 3}).to(torch::kFloat64).requires_grad_(false);
    auto d2 = torch::rand({2, 2, 3, 4}).to(torch::kFloat64).requires_grad_(false);

    TensorMatDual td1(r1, d1);
    TensorMatDual td2(r2, d2);
    //Sum over the first dimension
    TensorMatDual result = td1-td2;
    //calculate the result using back propagation
    torch::Tensor r3 = r1.clone().requires_grad_(true);
    torch::Tensor r4 = r2.clone().requires_grad_(false);
    auto J1 = r3-r4; //This will have dimension 2,2
    auto jac1 = janus::compute_batch_jacobian3d(J1, r3);//This will have dimension 2,2,3,2,3
    torch::Tensor r5 = r1.clone().requires_grad_(false);
    torch::Tensor r6 = r2.clone().requires_grad_(true);
    auto J2 = r5-r6; //This will have dimension 2,2
    auto jac2 = janus::compute_batch_jacobian3d(J2, r6);//This will have dimension 2,2,3,2,3
    //This is a combination of the two partial derivatives
    auto jacres = torch::einsum("bijkl, bkld->bijd", {jac1, d1})+
                  torch::einsum("bijkl, bkld->bijd", {jac2, d2});
    EXPECT_TRUE(torch::allclose(result.d, jacres));
}

TEST(TensorMatDualTest, TensorMatDualDivTensorMatDual)
{
    auto r1 = torch::rand({2, 2, 3}).to(torch::kFloat64).requires_grad_(false);
    auto d1 = torch::rand({2, 2, 3, 4}).to(torch::kFloat64).requires_grad_(false);
    auto r2 = torch::rand({2, 2, 3}).to(torch::kFloat64).requires_grad_(false);
    auto d2 = torch::rand({2, 2, 3, 4}).to(torch::kFloat64).requires_grad_(false);

    TensorMatDual td1(r1, d1);
    TensorMatDual td2(r2, d2);
    //Sum over the first dimension
    TensorMatDual result = td1/td2;
    //calculate the result using back propagation
    torch::Tensor r3 = r1.clone().requires_grad_(true);
    torch::Tensor r4 = r2.clone().requires_grad_(false);
    auto J1 = r3/r4; //This will have dimension 2,2
    auto jac1 = janus::compute_batch_jacobian3d(J1, r3);//This will have dimension 2,2,3,2,3
    torch::Tensor r5 = r1.clone().requires_grad_(false);
    torch::Tensor r6 = r2.clone().requires_grad_(true);
    auto J2 = r5/r6; //This will have dimension 2,2
    auto jac2 = janus::compute_batch_jacobian3d(J2, r6);//This will have dimension 2,2,3,2,3
    //This is a combination of the two partial derivatives
    auto jacres = torch::einsum("bijkl, bkld->bijd", {jac1, d1})+
                  torch::einsum("bijkl, bkld->bijd", {jac2, d2});
    EXPECT_TRUE(torch::allclose(result.d, jacres));
}



TEST(TensorMatDualTest, TensorMatDualDivTensor)
{
    auto r1 = torch::rand({2, 2, 3}).to(torch::kFloat64).requires_grad_(false);
    auto d1 = torch::rand({2, 2, 3, 4}).to(torch::kFloat64).requires_grad_(false);
    auto r2 = torch::rand({2, 2, 3}).to(torch::kFloat64).requires_grad_(false);

    TensorMatDual td1(r1, d1);
    //Sum over the first dimension
    TensorMatDual result = td1/r2;
    //calculate the result using back propagation
    torch::Tensor r3 = r1.clone().requires_grad_(true);
    torch::Tensor r4 = r2.clone().requires_grad_(false);
    auto J1 = r3/r4; //This will have dimension 2,2
    auto jac1 = janus::compute_batch_jacobian3d(J1, r3);//This will have dimension 2,2,3,2,3
    torch::Tensor r5 = r1.clone().requires_grad_(false);
    torch::Tensor r6 = r2.clone().requires_grad_(true);
    auto J2 = r5/r6; //This will have dimension 2,2
    auto jac2 = janus::compute_batch_jacobian3d(J2, r6);//This will have dimension 2,2,3,2,3
    //This is a combination of the two partial derivatives
    auto jacres = torch::einsum("bijkl, bkld->bijd", {jac1, d1});
    EXPECT_TRUE(torch::allclose(result.d, jacres));
}


TEST(TensorMatDualTest, TensorMatDualAbs)
{
    auto r1 = torch::rand({2, 2, 3}).to(torch::kFloat64).requires_grad_(false);
    auto d1 = torch::rand({2, 2, 3, 4}).to(torch::kFloat64).requires_grad_(false);

    TensorMatDual td1(r1, d1);
    //Sum over the first dimension
    TensorMatDual result = td1.abs();
    //calculate the result using back propagation
    torch::Tensor r = r1.clone().requires_grad_(true);
    auto J = r.abs();
    auto jac = janus::compute_batch_jacobian3d(J, r);//This will have dimension 2,2,3,2,3
    auto jacres = torch::einsum("bijkl, bkld->bijd", {jac, d1});
    EXPECT_TRUE(torch::allclose(result.d, jacres));
}

TEST (TensorMatDualTest, einsumTest)
{
    auto r1 = torch::randn({2, 2, 3});
    auto d1 = torch::randn({2, 2, 3, 4});
    auto r2 = torch::randn({2, 2, 3});
    auto d2 = torch::randn({2, 2, 3, 4});
    TensorMatDual td1(r1, d1);
    TensorMatDual td2(r2, d2);
    TensorMatDual result = TensorMatDual::einsum("mij,mij->mij", td1, td2);
    auto r = torch::einsum("mij,mij->mij", {r1, r2});
    auto d = torch::einsum("mij,mijk->mijk", {r1, d2}) + torch::einsum("mij,mijk->mijk", {r2, d1});
    EXPECT_TRUE(result.r.equal(r));
    EXPECT_TRUE(result.d.equal(d));
}

TEST (TensorMatDualTest, einsumTest2)
{
    auto r1 = torch::randn({2, 2, 3});
    auto d1 = torch::randn({2, 2, 3, 4});
    auto r2 = torch::randn({2, 3});
    auto d2 = torch::randn({2, 3, 4});
    TensorMatDual td1(r1, d1);
    TensorDual td2(r2, d2);
    TensorDual result = TensorMatDual::einsum("mij,mj->mi", td1, td2);
    auto r = torch::einsum("mij,mj->mi", {r1, r2});
    auto d = torch::einsum("mij,mjk->mik", {r1, d2}) + torch::einsum("mijk,mj->mik", {d1, r2});
    EXPECT_TRUE(result.r.equal(r));
    EXPECT_TRUE(result.d.equal(d));
}

TEST (TensorMatDualTest, einsumTest3)
{
    auto r1 = torch::randn({2, 2, 3});
    auto d1 = torch::randn({2, 2, 3, 4});
    auto r2 = torch::randn({2, 2});
    TensorMatDual td1(r1, d1);
    TensorMatDual result = TensorMatDual::einsum("mij,mi->mij", td1, r2);
    auto r = torch::einsum("mij,mi->mij", {r1, r2});
    auto d = torch::einsum("mi,mijk->mijk", {r2, d1});
    EXPECT_TRUE(result.r.equal(r));
    EXPECT_TRUE(result.d.equal(d));
}

TEST (TensorMatDualTest, einsumTest4)
{
    auto r1 = torch::randn({2, 2, 3});
    auto d1 = torch::randn({2, 2, 3, 4});
    auto r2 = torch::randn({2, 2});
    auto d2 = torch::randn({2, 2, 4});
    TensorMatDual td1(r1, d1);
    TensorDual td2(r2, d2);
    TensorMatDual result = TensorMatDual::einsum("mi,mij->mij", td2, td1);
    auto r = torch::einsum("mij,mi->mij", {r1, r2});
    auto d = torch::einsum("mij,mik->mijk", {r1, d2}) + torch::einsum("mi,mijk->mijk", {r2, d1});
    EXPECT_TRUE(result.r.equal(r));
    EXPECT_TRUE(result.d.equal(d));
}


TEST (TensorMatDualTest, einsumTest5)
{
    auto r1 = torch::randn({2, 2, 3});
    auto d1 = torch::randn({2, 2, 3, 4});
    auto r2 = torch::randn({2, 2});
    TensorMatDual td1(r1, d1);
    TensorMatDual result = TensorMatDual::einsum("mi,mij->mij", r2, td1);
    auto r = torch::einsum("mij,mi->mij", {r1, r2});
    auto d = torch::einsum("mi,mijk->mijk", {r2, d1});
    EXPECT_TRUE(result.r.equal(r));
    EXPECT_TRUE(result.d.equal(d));
}


TEST(TensorMatDualTest, TensorMatDualMax)
{
    auto r1 = torch::rand({2, 2, 3}).to(torch::kFloat64).requires_grad_(false);
    auto d1 = torch::rand({2, 2, 3, 4}).to(torch::kFloat64).requires_grad_(false);

    TensorMatDual td1(r1, d1);
    //Sum over the first dimension
    TensorMatDual result = td1.max();
    //calculate the result using back propagation
    torch::Tensor r = r1.clone().requires_grad_(true);
    auto Jres = r.max(1);
    auto J = std::get<0>(Jres);
    auto jac = janus::compute_batch_jacobian2d(J, r);//This will have dimension 2,3,2,3
    auto jacres = torch::einsum("bjkl, bkld->bjd", {jac, d1}).unsqueeze(1);
    EXPECT_TRUE(torch::allclose(result.d, jacres));
}

TEST(TensorMatDualTest, TensorMatDualMin)
{
    auto r1 = torch::rand({2, 2, 3}).to(torch::kFloat64).requires_grad_(false);
    auto d1 = torch::rand({2, 2, 3, 4}).to(torch::kFloat64).requires_grad_(false);

    TensorMatDual td1(r1, d1);
    //Sum over the first dimension
    TensorMatDual result = td1.min();
    //calculate the result using back propagation
    torch::Tensor r = r1.clone().requires_grad_(true);
    auto Jres = r.min(1);
    auto J = std::get<0>(Jres);
    auto jac = janus::compute_batch_jacobian2d(J, r);//This will have dimension 2,3,2,3
    auto jacres = torch::einsum("bjkl, bkld->bjd", {jac, d1}).unsqueeze(1);
    EXPECT_TRUE(torch::allclose(result.d, jacres));
}

TEST(TensorDualTest, TensorAddTensorDual)
{
    auto r1 = torch::randn({2, 2}).to(torch::kFloat64).requires_grad_(false);
    auto d1 = torch::rand({2, 2, 2}).to(torch::kFloat64).requires_grad_(false);
    auto r2 = torch::randn({2, 2}).to(torch::kFloat64).requires_grad_(false);
    //Set the dual parts to 1.0 so there is exactly one per element
    //d1.index_put_({Slice(),0,0}, 1.0);
    //d1.index_put_({Slice(),1,1}, 1.0);

    TensorDual td1(r1.clone(), d1.clone());
    //Sum over the first dimension
    TensorDual result = r2+td1;
    //calculate the result using back propagation
    torch::Tensor r = r1.clone().requires_grad_(true);
    torch::Tensor r3 = r2.clone().requires_grad_(true);
    auto J = r + r3;
    auto jac = janus::compute_batch_jacobian(J, r);
    EXPECT_TRUE(torch::allclose(result.d, torch::einsum("mij, mjk->mik",{jac, d1})));

}

TEST(TensorDualTest, DoubleAddTensorDual)
{
    auto r1 = torch::randn({2, 2}).to(torch::kFloat64).requires_grad_(false);
    auto d1 = torch::rand({2, 2, 2}).to(torch::kFloat64).requires_grad_(false);
    
    auto r2 = 1.8;
    //Set the dual parts to 1.0 so there is exactly one per element
    //d1.index_put_({Slice(),0,0}, 1.0);
    //d1.index_put_({Slice(),1,1}, 1.0);

    TensorDual td1(r1.clone(), d1.clone());
    //Sum over the first dimension
    TensorDual result = r2+td1;
    //calculate the result using back propagation
    torch::Tensor r = r1.clone().requires_grad_(true);
    auto J = r2+r;
    auto jac = janus::compute_batch_jacobian(J, r);
    EXPECT_TRUE(torch::allclose(result.d, torch::einsum("mij, mjk->mik",{jac, d1})));

}

TEST(TensorDualTest, TensorSubTensorDual)
{
    auto r1 = torch::randn({2, 2}).to(torch::kFloat64).requires_grad_(false);
    auto d1 = torch::rand({2, 2, 2}).to(torch::kFloat64).requires_grad_(false);
    auto r2 = torch::randn({2, 2}).to(torch::kFloat64).requires_grad_(false);
    //Set the dual parts to 1.0 so there is exactly one per element
    //d1.index_put_({Slice(),0,0}, 1.0);
    //d1.index_put_({Slice(),1,1}, 1.0);

    TensorDual td1(r1.clone(), d1.clone());
    //Sum over the first dimension
    TensorDual result = r2-td1;
    //calculate the result using back propagation
    torch::Tensor r = r1.clone().requires_grad_(true);
    torch::Tensor r3 = r2.clone().requires_grad_(true);
    auto J = r3-r;
    auto jac = janus::compute_batch_jacobian(J, r);
    EXPECT_TRUE(torch::allclose(result.d, torch::einsum("mij, mjk->mik",{jac, d1})));

}

TEST(TensorDualTest, TensorDualMulTensorMatDualTest)
{
    auto r1 = torch::randn({2, 2}).to(torch::kFloat64).requires_grad_(false);
    auto d1 = torch::randn({2, 2, 4}).to(torch::kFloat64).requires_grad_(false);
    auto r2 = torch::randn({2, 2, 3}).to(torch::kFloat64).requires_grad_(false);
    auto d2 = torch::randn({2, 2, 3, 4}).to(torch::kFloat64).requires_grad_(false);
    TensorDual td1(r1.clone(), d1.clone());
    TensorMatDual td2(r2.clone(), d2.clone());
    TensorDual result = td1*td2;
    auto r3 = r1.clone().requires_grad_(true);
    auto r4 = r2.clone().requires_grad_(false);
    auto J1 = torch::einsum("bi,bij->bj", {r3,r4}); //This will have dimension 2,3
    auto jac1 = janus::compute_batch_jacobian2d2d(J1, r3); //This will have dimension 2,3,2
    auto r5 = r1.clone().requires_grad_(false);
    auto r6 = r2.clone().requires_grad_(true);
    auto J2 = torch::einsum("bi,bij->bj", {r5,r6}); //This will have dimension 2,3
    auto jac2 = janus::compute_batch_jacobian2d(J2, r6); //This will have dimension 2,3,2,3

    auto jacres = torch::einsum("bjk, bkd->bjd", {jac1, d1})+
                  torch::einsum("bjlk, blkd->bjd", {jac2, d2});
    EXPECT_TRUE(result.d.equal(jacres));

}

TEST(TensorMatDualTest, TensorMatDualMulTensorDualTest)
{
    auto r1 = torch::randn({2, 2, 3}).to(torch::kFloat64).requires_grad_(false);
    auto d1 = torch::randn({2, 2, 3, 4}).to(torch::kFloat64).requires_grad_(false);
    auto r2 = torch::randn({2, 2}).to(torch::kFloat64).requires_grad_(false);
    auto d2 = torch::randn({2, 2, 4}).to(torch::kFloat64).requires_grad_(false);
    TensorMatDual td1(r1.clone(), d1.clone());
    TensorDual td2(r2.clone(), d2.clone());
    TensorDual result = td1*td2;
    auto r3 = r1.clone().requires_grad_(true);
    auto r4 = r2.clone().requires_grad_(false);
    auto J1 = torch::einsum("bij,bi->bj", {r3,r4}); //This will have dimension 2,3
    auto jac1 = janus::compute_batch_jacobian2d(J1, r3); //This will have dimension 2,3,2
    auto r5 = r1.clone().requires_grad_(false);
    auto r6 = r2.clone().requires_grad_(true);
    auto J2 = torch::einsum("bij,bi->bj", {r5,r6}); //This will have dimension 2,3
    auto jac2 = janus::compute_batch_jacobian2d2d(J2, r6); //This will have dimension 2,3,2,3

    auto jacres = torch::einsum("bjlk, blkd->bjd", {jac1, d1})+
                  torch::einsum("bjk, bkd->bjd", {jac2, d2});
    EXPECT_TRUE(result.d.equal(jacres));

}


TEST(TensorMatDualTest, TensorMatDualMulTensorMatDualTest)
{
    //This is an element wise simple multiplication
    auto r1 = torch::randn({2, 2, 3}).to(torch::kFloat64).requires_grad_(false);
    auto d1 = torch::randn({2, 2, 3, 4}).to(torch::kFloat64).requires_grad_(false);
    auto r2 = torch::randn({2, 2, 3}).to(torch::kFloat64).requires_grad_(false);
    auto d2 = torch::randn({2, 2, 3, 4}).to(torch::kFloat64).requires_grad_(false);
    TensorMatDual td1(r1.clone(), d1.clone());
    TensorMatDual td2(r2.clone(), d2.clone());
    TensorMatDual result = td1*td2;
    auto r3 = r1.clone().requires_grad_(true);
    auto r4 = r2.clone().requires_grad_(false);
    auto J1 = torch::einsum("bij,bij->bij", {r3,r4}); //This will have dimension 2,2,3
    //std::cerr << "J1=" << J1 << std::endl;
    auto jac1 = janus::compute_batch_jacobian3d(J1, r3); //This will have dimension 2,2,3,2,3
    //std::cerr << "jac1" << jac1 << std::endl;
    auto r5 = r1.clone().requires_grad_(false);
    auto r6 = r2.clone().requires_grad_(true);
    auto J2 = torch::einsum("bij,bij->bij", {r5,r6}); //This will have dimension 2,2,3
    //std::cerr << "J2=" << J2 << std::endl;
    auto jac2 = janus::compute_batch_jacobian3d(J2, r6); //This will have dimension 2,2,3,2,3
    //std::cerr << "jac2" << jac2 << std::endl;
    auto jacres = torch::einsum("bijkl, bkld->bijd", {jac1, d1})+
                  torch::einsum("bijkl, bkld->bijd", {jac2, d2});
    EXPECT_TRUE(result.d.equal(jacres));

}


TEST(TensorDualTest, DoubleSubTensorDual)
{
    auto r1 = torch::randn({2, 2}).to(torch::kFloat64).requires_grad_(false);
    auto d1 = torch::rand({2, 2, 2}).to(torch::kFloat64).requires_grad_(false);
    auto r2 = 1.8;
    //Set the dual parts to 1.0 so there is exactly one per element
    //d1.index_put_({Slice(),0,0}, 1.0);
    //d1.index_put_({Slice(),1,1}, 1.0);

    TensorDual td1(r1.clone(), d1.clone());
    //Sum over the first dimension
    TensorDual result = r2-td1;
    //calculate the result using back propagation
    torch::Tensor r = r1.clone().requires_grad_(true);
    auto J = r2-r;
    auto jac = janus::compute_batch_jacobian(J, r);
    EXPECT_TRUE(torch::allclose(result.d, torch::einsum("mij, mjk->mik",{jac, d1})));

}

TEST(TensorDualTest, DoubleDivTensorDual)
{
    auto r1 = torch::randn({2, 2}).to(torch::kFloat64).requires_grad_(false);
    auto d1 = torch::rand({2, 2, 2}).to(torch::kFloat64).requires_grad_(false);
    auto r2 = 1.8;
    //Set the dual parts to 1.0 so there is exactly one per element
    //d1.index_put_({Slice(),0,0}, 1.0);
    //d1.index_put_({Slice(),1,1}, 1.0);

    TensorDual td1(r1.clone(), d1.clone());
    //Sum over the first dimension
    TensorDual result = r2/td1;
    //calculate the result using back propagation
    torch::Tensor r = r1.clone().requires_grad_(true);
    auto J = r2/r;
    auto jac = janus::compute_batch_jacobian(J, r);
    EXPECT_TRUE(torch::allclose(result.d, torch::einsum("mij, mjk->mik",{jac, d1})));

}

TEST(TensorDualTest, DoubleMulTensorDual)
{
    auto r1 = torch::randn({2, 2}).to(torch::kFloat64).requires_grad_(false);
    auto d1 = torch::rand({2, 2, 2}).to(torch::kFloat64).requires_grad_(false);
    auto r2 = 1.8;
    //Set the dual parts to 1.0 so there is exactly one per element
    //d1.index_put_({Slice(),0,0}, 1.0);
    //d1.index_put_({Slice(),1,1}, 1.0);

    TensorDual td1(r1.clone(), d1.clone());
    //Sum over the first dimension
    TensorDual result = r2*td1;
    //calculate the result using back propagation
    torch::Tensor r = r1.clone().requires_grad_(true);
    auto J = r2*r;
    auto jac = janus::compute_batch_jacobian(J, r);
    EXPECT_TRUE(torch::allclose(result.d, torch::einsum("mij, mjk->mik",{jac, d1})));
}

TEST(TensorDualTest, TensorDualPowTensorDual)
{
    auto r1 = torch::rand({2, 2}).to(torch::kFloat64).requires_grad_(false);
    auto d1 = torch::rand({2, 2, 3}).to(torch::kFloat64).requires_grad_(false);
    auto r2 = torch::rand({2, 2}).to(torch::kFloat64).requires_grad_(false);
    auto d2 = torch::rand({2, 2, 3}).to(torch::kFloat64).requires_grad_(false);
    //Set the dual parts to 1.0 so there is exactly one per element
    //d1.index_put_({Slice(),0,0}, 1.0);
    //d1.index_put_({Slice(),1,1}, 1.0);

    TensorDual td1(r1.clone(), d1.clone());
    TensorDual td2(r2.clone(), d2.clone());
    //Sum over the first dimension
    TensorDual result = pow(td1, td2);
    //calculate the result using back propagation
    torch::Tensor r3 = r1.clone().requires_grad_(true);
    torch::Tensor r4 = r2.clone().requires_grad_(false);
    auto J1 = torch::pow(r3, r4);
    auto jac1 = janus::compute_batch_jacobian(J1, r3);
    torch::Tensor r5 = r1.clone().requires_grad_(false);
    torch::Tensor r6 = r2.clone().requires_grad_(true);
    auto J2 = torch::pow(r5, r6);
    auto jac2 = janus::compute_batch_jacobian(J2, r6);
    auto jacres = torch::einsum("bij, bjk->bik", {jac1, d1})+
                  torch::einsum("bij, bjk->bik", {jac2, d2});
    EXPECT_TRUE(torch::allclose(result.d, jacres));
}

TEST(TensorDualTest, TensorDualPowDouble)
{
    auto r1 = torch::rand({2, 2}).to(torch::kFloat64).requires_grad_(false);
    auto d1 = torch::rand({2, 2, 3}).to(torch::kFloat64).requires_grad_(false);
    auto r2 = 2.5;
    //Set the dual parts to 1.0 so there is exactly one per element
    //d1.index_put_({Slice(),0,0}, 1.0);
    //d1.index_put_({Slice(),1,1}, 1.0);

    TensorDual td1(r1.clone(), d1.clone());
    //Sum over the first dimension
    TensorDual result = pow(td1, r2);
    //calculate the result using back propagation
    torch::Tensor r3 = r1.clone().requires_grad_(true);
    auto r4 = r2;
    auto J1 = torch::pow(r3, r4);
    auto jac1 = janus::compute_batch_jacobian(J1, r3);
    auto jacres = torch::einsum("bij, bjk->bik", {jac1, d1});
    EXPECT_TRUE(torch::allclose(result.d, jacres));
}



TEST(TensorDualTest, TensorDualMaxTensorDual)
{
    auto r1 = torch::rand({2, 2}).to(torch::kFloat64).requires_grad_(false);
    auto d1 = torch::rand({2, 2, 3}).to(torch::kFloat64).requires_grad_(false);
    auto r2 = torch::rand({2, 2}).to(torch::kFloat64).requires_grad_(false);
    auto d2 = torch::rand({2, 2, 3}).to(torch::kFloat64).requires_grad_(false);
    //Set the dual parts to 1.0 so there is exactly one per element
    //d1.index_put_({Slice(),0,0}, 1.0);
    //d1.index_put_({Slice(),1,1}, 1.0);

    TensorDual td1(r1.clone(), d1.clone());
    TensorDual td2(r2.clone(), d2.clone());
    //Sum over the first dimension
    TensorDual result = max(td1, td2);
    //calculate the result using back propagation
    torch::Tensor r3 = r1.clone().requires_grad_(true);
    torch::Tensor r4 = r2.clone().requires_grad_(false);
    auto J1 = torch::max(r3, r4);
    auto jac1 = janus::compute_batch_jacobian(J1, r3);
    torch::Tensor r5 = r1.clone().requires_grad_(false);
    torch::Tensor r6 = r2.clone().requires_grad_(true);
    auto J2 = torch::max(r5, r6);
    auto jac2 = janus::compute_batch_jacobian(J2, r6);
    auto jacres = torch::einsum("bij, bjk->bik", {jac1, d1})+
                  torch::einsum("bij, bjk->bik", {jac2, d2});
    EXPECT_TRUE(torch::allclose(result.d, jacres));
}

TEST(TensorDualTest, TensorDualMaxTensor)
{
    auto r1 = torch::rand({2, 2}).to(torch::kFloat64).requires_grad_(false);
    auto d1 = torch::rand({2, 2, 3}).to(torch::kFloat64).requires_grad_(false);
    auto r2 = torch::rand({2, 2}).to(torch::kFloat64).requires_grad_(false);

    TensorDual td1(r1.clone(), d1.clone());
    
    //Sum over the first dimension
    TensorDual result = max(td1, r2);
    //calculate the result using back propagation
    torch::Tensor r3 = r1.clone().requires_grad_(true);
    torch::Tensor r4 = r2.clone().requires_grad_(false);
    auto J1 = torch::max(r3, r4);
    auto jac1 = janus::compute_batch_jacobian(J1, r3);
    auto jacres = torch::einsum("bij, bjk->bik", {jac1, d1});
    EXPECT_TRUE(torch::allclose(result.d, jacres));
}


TEST(TensorDualTest, TensorDualMinTensorDual)
{
    auto r1 = torch::rand({2, 2}).to(torch::kFloat64).requires_grad_(false);
    auto d1 = torch::rand({2, 2, 3}).to(torch::kFloat64).requires_grad_(false);
    auto r2 = torch::rand({2, 2}).to(torch::kFloat64).requires_grad_(false);
    auto d2 = torch::rand({2, 2, 3}).to(torch::kFloat64).requires_grad_(false);
    //Set the dual parts to 1.0 so there is exactly one per element
    //d1.index_put_({Slice(),0,0}, 1.0);
    //d1.index_put_({Slice(),1,1}, 1.0);

    TensorDual td1(r1.clone(), d1.clone());
    TensorDual td2(r2.clone(), d2.clone());
    //Sum over the first dimension
    TensorDual result = min(td1, td2);
    //calculate the result using back propagation
    torch::Tensor r3 = r1.clone().requires_grad_(true);
    torch::Tensor r4 = r2.clone().requires_grad_(false);
    auto J1 = torch::min(r3, r4);
    auto jac1 = janus::compute_batch_jacobian(J1, r3);
    torch::Tensor r5 = r1.clone().requires_grad_(false);
    torch::Tensor r6 = r2.clone().requires_grad_(true);
    auto J2 = torch::min(r5, r6);
    auto jac2 = janus::compute_batch_jacobian(J2, r6);
    auto jacres = torch::einsum("bij, bjk->bik", {jac1, d1})+
                  torch::einsum("bij, bjk->bik", {jac2, d2});
    EXPECT_TRUE(torch::allclose(result.d, jacres));
}

TEST(TensorDualTest, TensorDualMinTensor)
{
    auto r1 = torch::rand({2, 2}).to(torch::kFloat64).requires_grad_(false);
    auto d1 = torch::rand({2, 2, 3}).to(torch::kFloat64).requires_grad_(false);
    auto r2 = torch::rand({2, 2}).to(torch::kFloat64).requires_grad_(false);

    TensorDual td1(r1.clone(), d1.clone());
    
    //Sum over the first dimension
    TensorDual result = min(td1, r2);
    //calculate the result using back propagation
    torch::Tensor r3 = r1.clone().requires_grad_(true);
    torch::Tensor r4 = r2.clone().requires_grad_(false);
    auto J1 = torch::min(r3, r4);
    auto jac1 = janus::compute_batch_jacobian(J1, r3);
    auto jacres = torch::einsum("bij, bjk->bik", {jac1, d1});
    EXPECT_TRUE(torch::allclose(result.d, jacres));
}




int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
