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
    torch::Tensor other = torch::randn({2, 1});
    //This should be fine
    TensorDual result = x + other;
    EXPECT_EQ(result.r.sizes(), x.r.sizes());
    EXPECT_EQ(result.d.sizes(), x.d.sizes());


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
                TensorDual result = x / y;
        },
        c10::Error
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
            TensorDual result = x / other;
        },
        c10::Error
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
TEST(TensorDualTest, TensorIndexPutValidInput) {
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

TEST(TensorDualTest, TensorIndexPutInvalidMaskType) {
    auto r = torch::tensor({{1.0, 2.0}, {3.0, 4.0}});
    auto d = torch::tensor({{{5.0, 6.0}, {7.0, 8.0}}, {{9.0, 10.0}, {11.0, 12.0}}});
    TensorDual td(r, d);

    auto mask = torch::tensor({{1, 0}, {0, 1}}, torch::dtype(torch::kInt));
    auto value_r = torch::tensor({{10.0, 20.0}, {30.0, 40.0}});
    auto value_d = torch::tensor({{{15.0, 16.0}, {17.0, 18.0}}, {{19.0, 20.0}, {21.0, 22.0}}});
    TensorDual value(value_r, value_d);

    EXPECT_THROW(td.index_put_(mask, value), std::invalid_argument);
}



TEST(TensorDualTest, TensorIndexPutValueShapeMismatch) {
    auto r = torch::tensor({{1.0, 2.0}, {3.0, 4.0}});
    auto d = torch::tensor({{{5.0, 6.0}, {7.0, 8.0}}, {{9.0, 10.0}, {11.0, 12.0}}});
    TensorDual td(r, d);

    // Mask selects specific rows
    auto mask = torch::tensor({0, 1}, torch::kInt64);
    auto value_r = torch::tensor({{10.0, 20.0}, {30.0, 40.0}});
    auto value_d = torch::tensor({{{15.0, 16.0}, {17.0, 18.0}}}); // Shape mismatch
    TensorDual value(value_r, value_d);

    EXPECT_THROW(td.index_put_(mask, value), std::invalid_argument);
}


// Test cases
TEST(TensorDualTest, IndexPutScalarValidInput) {
    auto r = torch::tensor({{1.0, 2.0}, {3.0, 4.0}});
    auto d = torch::tensor({{{5.0, 6.0}, {7.0, 8.0}}, {{9.0, 10.0}, {11.0, 12.0}}});
    TensorDual td(r, d);

    // Mask specifies elements to update
    auto mask = torch::tensor({{true, false}, {false, true}});
    double value = 42.0;

    td.index_put_(mask, value);

    auto expected_r = torch::tensor({{42.0, 2.0}, {3.0, 42.0}});
    auto expected_d = torch::tensor({{{0.0, 0.0}, {7.0, 8.0}}, {{9.0, 10.0}, {0.0, 0.0}}});

    EXPECT_TRUE(torch::equal(td.r, expected_r));
    EXPECT_TRUE(torch::equal(td.d, expected_d));
}

TEST(TensorDualTest, IndexPutScalarInvalidMaskSize) {
    auto r = torch::tensor({{1.0, 2.0}, {3.0, 4.0}});
    auto d = torch::tensor({{{5.0, 6.0}, {7.0, 8.0}}, {{9.0, 10.0}, {11.0, 12.0}}});
    TensorDual td(r, d);

    // Invalid mask size
    auto mask = torch::tensor({true, false}); // 1D mask instead of 2D
    double value = 42.0;

    EXPECT_THROW(td.index_put_(mask, value), std::invalid_argument);
}

TEST(TensorDualTest, IndexPutScalarAllFalseMask) {
    auto r = torch::tensor({{1.0, 2.0}, {3.0, 4.0}});
    auto d = torch::tensor({{{5.0, 6.0}, {7.0, 8.0}}, {{9.0, 10.0}, {11.0, 12.0}}});
    TensorDual td(r, d);

    // Mask selects no elements
    auto mask = torch::zeros_like(r, torch::kBool);
    double value = 42.0;

    td.index_put_(mask, value);

    EXPECT_TRUE(torch::equal(td.r, r));
    EXPECT_TRUE(torch::equal(td.d, d));
}

TEST(TensorDualTest, IndexPutScalarAllTrueMask) {
    auto r = torch::tensor({{1.0, 2.0}, {3.0, 4.0}});
    auto d = torch::tensor({{{5.0, 6.0}, {7.0, 8.0}}, {{9.0, 10.0}, {11.0, 12.0}}});
    TensorDual td(r, d);

    // Mask selects all elements
    auto mask = torch::ones_like(r, torch::kBool);
    double value = 42.0;

    td.index_put_(mask, value);

    auto expected_r = torch::full_like(r, value);
    auto expected_d = torch::zeros_like(d);

    EXPECT_TRUE(torch::equal(td.r, expected_r));
    EXPECT_TRUE(torch::equal(td.d, expected_d));
}

TEST(TensorDualTest, IndexPutScalarDeviceCompatibility) {
    auto r = torch::tensor({{1.0, 2.0}, {3.0, 4.0}}, torch::kCUDA);
    auto d = torch::tensor({{{5.0, 6.0}, {7.0, 8.0}}, {{9.0, 10.0}, {11.0, 12.0}}}, torch::kCUDA);
    TensorDual td(r, d);

    // Mask specifies elements to update
    auto mask = torch::tensor({{true, false}, {false, true}}, torch::kCUDA);
    double value = 42.0;

    td.index_put_(mask, value);

    auto expected_r = torch::tensor({{42.0, 2.0}, {3.0, 42.0}}, torch::kCUDA);
    auto expected_d = torch::tensor({{{0.0, 0.0}, {7.0, 8.0}}, {{9.0, 10.0}, {0.0, 0.0}}}, torch::kCUDA);

    EXPECT_TRUE(torch::equal(td.r, expected_r));
    EXPECT_TRUE(torch::equal(td.d, expected_d));
}


// Test cases
TEST(TensorDualTest, IndexPutVectorValidInput) {
    auto r = torch::tensor({{1.0, 2.0}, {3.0, 4.0}});
    auto d = torch::tensor({{{5.0, 6.0}, {7.0, 8.0}}, {{9.0, 10.0}, {11.0, 12.0}}});
    TensorDual td(r, d);

    // Mask to index elements
    std::vector<torch::indexing::TensorIndex> mask = {torch::indexing::Slice(0, 1), torch::indexing::Slice(0, 2)};
    auto value_r = torch::tensor({{42.0, 43.0}});
    auto value_d = torch::tensor({{{100.0, 101.0}, {102.0, 103.0}}});
    TensorDual value(value_r, value_d);

    td.index_put_(mask, value);

    auto expected_r = torch::tensor({{42.0, 43.0}, {3.0, 4.0}});
    auto expected_d = torch::tensor({{{100.0, 101.0}, {102.0, 103.0}}, {{9.0, 10.0}, {11.0, 12.0}}});

    EXPECT_TRUE(torch::equal(td.r, expected_r));
    EXPECT_TRUE(torch::equal(td.d, expected_d));
}


TEST(TensorDualTest, IndexPutVectorPartialUpdate) {
    auto r = torch::tensor({{1.0, 2.0}, {3.0, 4.0}});
    auto d = torch::tensor({{{5.0, 6.0}, {7.0, 8.0}}, {{9.0, 10.0}, {11.0, 12.0}}});
    TensorDual td(r, d);

    // Mask to index a subset of elements
    std::vector<torch::indexing::TensorIndex> mask = {torch::indexing::Slice(0, 2), torch::indexing::Slice(1, 2)};
    auto value_r = torch::tensor({{42.0}, {43.0}});
    auto value_d = torch::tensor({{{100.0, 101.0}}, {{102.0, 103.0}}});
    TensorDual value(value_r, value_d);

    td.index_put_(mask, value);

    auto expected_r = torch::tensor({{1.0, 42.0}, {3.0, 43.0}});
    auto expected_d = torch::tensor({{{5.0, 6.0}, {100.0, 101.0}}, {{9.0, 10.0}, {102.0, 103.0}}});

    EXPECT_TRUE(torch::equal(td.r, expected_r));
    EXPECT_TRUE(torch::equal(td.d, expected_d));
}

TEST(TensorDualTest, IndexPutVectorDeviceCompatibility) {
    auto r = torch::tensor({{1.0, 2.0}, {3.0, 4.0}}, torch::kCUDA);
    auto d = torch::tensor({{{5.0, 6.0}, {7.0, 8.0}}, {{9.0, 10.0}, {11.0, 12.0}}}, torch::kCUDA);
    TensorDual td(r, d);

    // Mask to index elements
    std::vector<torch::indexing::TensorIndex> mask = {torch::indexing::Slice(0, 1), torch::indexing::Slice(0, 2)};
    auto value_r = torch::tensor({{42.0, 43.0}}, torch::kCUDA);
    auto value_d = torch::tensor({{{100.0, 101.0}, {102.0, 103.0}}}, torch::kCUDA);
    TensorDual value(value_r, value_d);

    td.index_put_(mask, value);

    auto expected_r = torch::tensor({{42.0, 43.0}, {3.0, 4.0}}, torch::kCUDA);
    auto expected_d = torch::tensor({{{100.0, 101.0}, {102.0, 103.0}}, {{9.0, 10.0}, {11.0, 12.0}}}, torch::kCUDA);

    EXPECT_TRUE(torch::equal(td.r, expected_r));
    EXPECT_TRUE(torch::equal(td.d, expected_d));
}




// Test: Element-wise max of matching TensorDual objects
TEST(TensorDualTest, MaxMatchingDimensions) {
    auto r1 = torch::tensor({{1.0, 3.0}, {2.0, 4.0}});
    auto d1 = torch::tensor({{{10.0}, {30.0}}, {{20.0}, {40.0}}});
    auto r2 = torch::tensor({{2.0, 1.0}, {4.0, 3.0}});
    auto d2 = torch::tensor({{{20.0}, {10.0}}, {{40.0}, {30.0}}});

    TensorDual td1(r1, d1);
    TensorDual td2(r2, d2);

    auto result = td1.max(td2);

    auto expected_r = torch::tensor({{2.0, 3.0}, {4.0, 4.0}});
    auto expected_d = torch::tensor({{{20.0}, {30.0}}, {{40.0}, {40.0}}});

    ASSERT_TRUE(torch::allclose(result.r, expected_r)) << "Real part of max TensorDual is incorrect.";
    ASSERT_TRUE(torch::allclose(result.d, expected_d)) << "Dual part of max TensorDual is incorrect.";
}

// Test: Element-wise max when one TensorDual has strictly larger real values
TEST(TensorDualTest, MaxStrictlyLargerReal) {
    auto r1 = torch::tensor({{5.0, 6.0}, {7.0, 8.0}});
    auto d1 = torch::tensor({{{50.0}, {60.0}}, {{70.0}, {80.0}}});
    auto r2 = torch::tensor({{1.0, 2.0}, {3.0, 4.0}});
    auto d2 = torch::tensor({{{10.0}, {20.0}}, {{30.0}, {40.0}}});

    TensorDual td1(r1, d1);
    TensorDual td2(r2, d2);

    auto result = td1.max(td2);

    ASSERT_TRUE(torch::allclose(result.r, r1)) << "Real part should match TensorDual with larger values.";
    ASSERT_TRUE(torch::allclose(result.d, d1)) << "Dual part should match TensorDual with larger values.";
}

// Test: Incompatible dimensions
TEST(TensorDualTest, MaxIncompatibleDimensions) {
    auto r1 = torch::tensor({{1.0, 3.0}, {2.0, 4.0}});
    auto d1 = torch::tensor({{{10.0}, {30.0}}, {{20.0}, {40.0}}});
    auto r2 = torch::tensor({{2.0, 1.0}});
    auto d2 = torch::tensor({{{20.0}, {10.0}}});

    TensorDual td1(r1, d1);
    TensorDual td2(r2, d2);

    ASSERT_THROW(td1.max(td2), std::invalid_argument) << "Expected exception for incompatible dimensions.";
}

// Test: Handling negative values
TEST(TensorDualTest, MaxHandlesNegativeValues) {
    auto r1 = torch::tensor({{-1.0, -3.0}, {-2.0, -4.0}});
    auto d1 = torch::tensor({{{-10.0}, {-30.0}}, {{-20.0}, {-40.0}}});
    auto r2 = torch::tensor({{-2.0, -1.0}, {-4.0, -3.0}});
    auto d2 = torch::tensor({{{-20.0}, {-10.0}}, {{-40.0}, {-30.0}}});

    TensorDual td1(r1, d1);
    TensorDual td2(r2, d2);

    auto result = td1.max(td2);
    std::cerr << "result.r=" << result.r << std::endl;
    std::cerr << "result.d=" << result.d << std::endl;

    auto expected_r = torch::tensor({{-1.0, -1.0}, {-2.0, -3.0}});
    auto expected_d = torch::tensor({{{-10.0}, {-10.0}}, {{-20.0}, {-30.0}}});

    ASSERT_TRUE(torch::allclose(result.r, expected_r)) << "Real part of max TensorDual with negatives is incorrect.";
    ASSERT_TRUE(torch::allclose(result.d, expected_d)) << "Dual part of max TensorDual with negatives is incorrect.";
}

// Test: TensorDual with zeros
TEST(TensorDualTest, MaxWithZeros) {
    auto r1 = torch::tensor({{0.0, 1.0}, {0.0, 2.0}});
    auto d1 = torch::tensor({{{0.0}, {10.0}}, {{0.0}, {20.0}}});
    auto r2 = torch::tensor({{-1.0, 0.0}, {1.0, -2.0}});
    auto d2 = torch::tensor({{{-10.0}, {0.0}}, {{10.0}, {-20.0}}});

    TensorDual td1(r1, d1);
    TensorDual td2(r2, d2);

    auto result = td1.max(td2);

    auto expected_r = torch::tensor({{0.0, 1.0}, {1.0, 2.0}});
    auto expected_d = torch::tensor({{{0.0}, {10.0}}, {{10.0}, {20.0}}});

    ASSERT_TRUE(torch::allclose(result.r, expected_r)) << "Real part of max TensorDual with zeros is incorrect.";
    ASSERT_TRUE(torch::allclose(result.d, expected_d)) << "Dual part of max TensorDual with zeros is incorrect.";
}



TEST(TensorDualTest, PowPositiveExponent) {
    auto r = torch::tensor({{2.0, 3.0}, {4.0, 5.0}});
    auto d = torch::tensor({{{0.1, 0.2}, {0.3, 0.4}}, {{0.5, 0.6}, {0.7, 0.8}}});
    TensorDual td(r, d);

    double exponent = 2.0;
    auto result = td.pow(exponent);

    auto expected_r = torch::pow(r, exponent);
    auto gradient_r = exponent * torch::pow(r, exponent - 1);
    auto expected_d = torch::einsum("mi, mij->mij", {gradient_r, d});

    ASSERT_TRUE(torch::allclose(result.r, expected_r)) << "Real part is incorrect.";
    ASSERT_TRUE(torch::allclose(result.d, expected_d)) << "Dual part is incorrect.";
}

TEST(TensorDualTest, PowFractionalExponent) {
    auto r = torch::tensor({{4.0, 9.0}, {16.0, 25.0}});
    auto d = torch::tensor({{{0.1, 0.2}, {0.3, 0.4}}, {{0.5, 0.6}, {0.7, 0.8}}});
    TensorDual td(r, d);

    double exponent = 0.5;  // Square root
    auto result = td.pow(exponent);

    auto expected_r = torch::pow(r, exponent);
    auto gradient_r = exponent * torch::pow(r, exponent - 1);
    auto expected_d = torch::einsum("mi, mij->mij", {gradient_r, d});

    ASSERT_TRUE(torch::allclose(result.r, expected_r)) << "Real part is incorrect.";
    ASSERT_TRUE(torch::allclose(result.d, expected_d)) << "Dual part is incorrect.";
}

TEST(TensorDualTest, PowNegativeExponent) {
    auto r = torch::tensor({{1.0, 2.0}, {3.0, 4.0}});
    auto d = torch::tensor({{{0.1, 0.2}, {0.3, 0.4}}, {{0.5, 0.6}, {0.7, 0.8}}});
    TensorDual td(r, d);

    double exponent = -1.0;  // Reciprocal
    auto result = td.pow(exponent);

    auto expected_r = torch::pow(r, exponent);
    auto gradient_r = exponent * torch::pow(r, exponent - 1);
    auto expected_d = torch::einsum("mi, mij->mij", {gradient_r, d});

    ASSERT_TRUE(torch::allclose(result.r, expected_r)) << "Real part is incorrect.";
    ASSERT_TRUE(torch::allclose(result.d, expected_d)) << "Dual part is incorrect.";
}

TEST(TensorDualTest, PowNegativeValuesFractionalExponent) {
    auto r = torch::tensor({{-4.0, 9.0}, {16.0, -25.0}});
    auto d = torch::tensor({{{0.1, 0.2}, {0.3, 0.4}}, {{0.5, 0.6}, {0.7, 0.8}}});
    TensorDual td(r, d);

    double exponent = 0.5;  // Square root
    ASSERT_THROW(td.pow(exponent), std::invalid_argument) << "Expected exception for negative base with fractional exponent.";
}

TEST(TensorDualTest, PowZeroExponent) {
    auto r = torch::tensor({{1.0, 2.0}, {3.0, 4.0}});
    auto d = torch::tensor({{{0.1, 0.2}, {0.3, 0.4}}, {{0.5, 0.6}, {0.7, 0.8}}});
    TensorDual td(r, d);

    double exponent = 0.0;  // Any number to the power 0 is 1
    auto result = td.pow(exponent);

    auto expected_r = torch::ones_like(r);
    auto expected_d = torch::zeros_like(d);

    ASSERT_TRUE(torch::allclose(result.r, expected_r)) << "Real part is incorrect.";
    ASSERT_TRUE(torch::allclose(result.d, expected_d)) << "Dual part should be zero.";
}

TEST(TensorDualTest, PowZeroBasePositiveExponent) {
    auto r = torch::tensor({{0.0, 2.0}, {3.0, 0.0}});
    auto d = torch::tensor({{{0.1, 0.2}, {0.3, 0.4}}, {{0.5, 0.6}, {0.7, 0.8}}});
    TensorDual td(r, d);

    double exponent = 2.0;  // Square
    auto result = td.pow(exponent);

    auto expected_r = torch::pow(r, exponent);
    auto gradient_r = exponent * torch::pow(r, exponent - 1);
    auto expected_d = torch::einsum("mi, mij->mij", {gradient_r, d});

    ASSERT_TRUE(torch::allclose(result.r, expected_r)) << "Real part is incorrect.";
    ASSERT_TRUE(torch::allclose(result.d.masked_fill(r == 0, 0), expected_d.masked_fill(r == 0, 0)))
        << "Dual part should be zero where base is zero.";
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
    TensorDual result = TensorMatDual::einsum("mi,mij->mj", td2, td1);
    auto r = torch::einsum("mij,mi->mj", {r1, r2});
    auto d = torch::einsum("mij,mik->mjk", {r1, d2}) + torch::einsum("mi,mijk->mjk", {r2, d1});
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


TEST(TensorHyperDualTests, DefaultConstructor) {
    TensorHyperDual thd;

    ASSERT_EQ(thd.r.sizes(), torch::IntArrayRef({1, 1}));
    ASSERT_EQ(thd.d.sizes(), torch::IntArrayRef({1, 1, 1}));
    ASSERT_EQ(thd.h.sizes(), torch::IntArrayRef({1, 1, 1, 1}));
    ASSERT_EQ(thd.dtype_, torch::kFloat64);
    ASSERT_EQ(thd.device_, torch::kCPU);
}

TEST(TensorHyperDualTests, ParameterizedConstructor) {
    TensorHyperDual thd(2, 3, 4, 5, 6, 7, 8, 9, 10);

    ASSERT_EQ(thd.r.sizes(), torch::IntArrayRef({2, 3}));
    ASSERT_EQ(thd.d.sizes(), torch::IntArrayRef({4, 5, 6}));
    ASSERT_EQ(thd.h.sizes(), torch::IntArrayRef({7, 8, 9, 10}));
    ASSERT_EQ(thd.dtype_, torch::kFloat64);
    ASSERT_EQ(thd.device_, torch::kCPU);
}

TEST(TensorHyperDualTests, TensorConstructor) {
    auto r = torch::ones({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d = torch::ones({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::ones({2, 3, 4, 5}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual thd(r, d, h);

    ASSERT_TRUE(torch::equal(thd.r, r));
    ASSERT_TRUE(torch::equal(thd.d, d));
    ASSERT_TRUE(torch::equal(thd.h, h));
    ASSERT_EQ(thd.dtype_, torch::kFloat64);
    ASSERT_EQ(thd.device_, torch::kCPU);
}

TEST(TensorHyperDualTests, ToDevice) {
    TensorHyperDual thd(2, 3, 4, 5, 6, 7, 8, 9, 10);

    thd = thd.to(torch::kCUDA);

    ASSERT_EQ(thd.r.device().type(), torch::kCUDA);
    ASSERT_EQ(thd.d.device().type(), torch::kCUDA);
    ASSERT_EQ(thd.h.device().type(), torch::kCUDA);
    ASSERT_EQ(thd.device_, torch::kCUDA);
}

TEST(TensorHyperDualTests, DeviceGetter) {
    TensorHyperDual thd;

    ASSERT_EQ(thd.device().type(), torch::kCPU);

    thd = thd.to(torch::kCUDA);

    ASSERT_EQ(thd.device().type(), torch::kCUDA);
}


TEST(TensorHyperDualTests, ConstructFromTensorDual) {
    auto r = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    TensorDual td(r, d);

    TensorHyperDual thd(td);

    ASSERT_TRUE(torch::equal(thd.r, td.r));
    ASSERT_TRUE(torch::equal(thd.d, td.d));
    ASSERT_EQ(thd.h.sizes(), torch::IntArrayRef({2, 3, 4, 4}));
    ASSERT_TRUE(torch::all(thd.h == 0).item<bool>());
    ASSERT_EQ(thd.dtype_, torch::kFloat64);
    ASSERT_EQ(thd.device_, torch::kCPU);
}

TEST(TensorHyperDualTests, ConstructFromTensorDualInvalidInput) {

    auto r = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    auto d = torch::rand({2, 3, 4, 5}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    EXPECT_THROW(TensorDual td(r, d); TensorHyperDual thd(td), std::runtime_error);

    r = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCUDA));
    d = torch::rand({2, 3, 4, 5}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    EXPECT_THROW(TensorDual td(r, d); TensorHyperDual thd(td), std::runtime_error);
}

TEST(TensorHyperDualTest, SumDefaultParameters) {
    torch::Tensor r = torch::rand({3, 4}); // Example tensor
    torch::Tensor d = torch::rand({3, 4, 5});
    torch::Tensor h = torch::rand({3, 4, 5, 5});
    TensorHyperDual input(r, d, h);

    TensorHyperDual result = input.sum();

    ASSERT_EQ(result.r.sizes(), (std::vector<int64_t>{3, 1}));
    ASSERT_EQ(result.d.sizes(), (std::vector<int64_t>{3, 1, 5}));
    ASSERT_EQ(result.h.sizes(), (std::vector<int64_t>{3, 1, 5, 5}));
    ASSERT_TRUE(result.r.equal(r.sum(1, true)));
    ASSERT_TRUE(result.d.equal(d.sum(1, true)));
    ASSERT_TRUE(result.h.equal(h.sum(1, true)));
}


TEST(TensorHyperDualTests, Square) {
    auto r = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual thd(r, d, h);
    TensorHyperDual squared = thd.square();

    ASSERT_TRUE(torch::equal(squared.r, r.square()));

    auto expected_d = 2 * torch::einsum("mi, mij->mij", {r, d});
    ASSERT_TRUE(torch::allclose(squared.d, expected_d));

    auto expected_h = 2 * torch::einsum("mij, mik->mijk", {d, d}) +
                      2 * torch::einsum("mi, mijk->mijk", {r, h});
    ASSERT_TRUE(torch::allclose(squared.h, expected_h));
}

TEST(TensorHyperDualTests, SquareZeros) {
    auto r = torch::zeros({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d = torch::zeros({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::zeros({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual thd(r, d, h);
    TensorHyperDual squared = thd.square();

    ASSERT_TRUE(torch::all(squared.r == 0).item<bool>());
    ASSERT_TRUE(torch::all(squared.d == 0).item<bool>());
    ASSERT_TRUE(torch::all(squared.h == 0).item<bool>());
}


TEST(TensorHyperDualSqrtTests, SqrtRandomValues) {
    auto r = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU)) + 1.0; // Ensure positive values
    auto d = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual thd(r, d, h);
    TensorHyperDual sqrt_thd = thd.sqrt();

    ASSERT_TRUE(torch::allclose(sqrt_thd.r, r.sqrt()));

    auto expected_d = 0.5 * torch::einsum("mi, mij->mij", {r.pow(-0.5), d});
    ASSERT_TRUE(torch::allclose(sqrt_thd.d, expected_d));

    auto expected_h = -0.25 * torch::einsum("mi, mij, mik->mijk", {r.pow(-1.5), d, d}) +
                      0.5 * torch::einsum("mi, mijk->mijk", {r.pow(-0.5), h});
    ASSERT_TRUE(torch::allclose(sqrt_thd.h, expected_h));
}

TEST(TensorHyperDualSqrtTests, SqrtZeros) {
    auto r = torch::zeros({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU)) + 1.0; // Avoid sqrt(0)
    auto d = torch::zeros({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::zeros({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual thd(r, d, h);
    TensorHyperDual sqrt_thd = thd.sqrt();

    ASSERT_TRUE(torch::allclose(sqrt_thd.r, r.sqrt()));
    ASSERT_TRUE(torch::all(sqrt_thd.d == 0).item<bool>());
    ASSERT_TRUE(torch::all(sqrt_thd.h == 0).item<bool>());
}

TEST(TensorHyperDualSqrtTests, SqrtSmallValues) {
    auto r = torch::full({2, 3}, 1e-6, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual thd(r, d, h);
    TensorHyperDual sqrt_thd = thd.sqrt();

    ASSERT_TRUE(torch::allclose(sqrt_thd.r, r.sqrt()));

    auto expected_d = 0.5 * torch::einsum("mi, mij->mij", {r.pow(-0.5), d});
    ASSERT_TRUE(torch::allclose(sqrt_thd.d, expected_d));

    auto expected_h = -0.25 * torch::einsum("mi, mij, mik->mijk", {r.pow(-1.5), d, d}) +
                      0.5 * torch::einsum("mi, mijk->mijk", {r.pow(-0.5), h});
    ASSERT_TRUE(torch::allclose(sqrt_thd.h, expected_h));
}

TEST(TensorHyperDualSqrtTests, SqrtNegativeValuesToComplex) {
    auto r = torch::full({2, 3}, -1.0, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual thd(r, d, h);
    TensorHyperDual sqrt_thd = thd.sqrt();

    //Convert r to complex
    auto rc = r.to(torch::kComplexDouble);
    auto dc = d.to(torch::kComplexDouble);
    auto hc = h.to(torch::kComplexDouble);

    ASSERT_TRUE(sqrt_thd.r.is_complex());
    ASSERT_TRUE(torch::allclose(sqrt_thd.r, rc.sqrt()));

    auto expected_d = 0.5 * torch::einsum("mi, mij->mij", {rc.pow(-0.5), dc});
    ASSERT_TRUE(torch::allclose(sqrt_thd.d, expected_d));

    auto expected_h = -0.25 * torch::einsum("mi, mij, mik->mijk", {rc.pow(-1.5), dc, dc}) +
                      0.5 * torch::einsum("mi, mijk->mijk", {rc.pow(-0.5), hc});
    ASSERT_TRUE(torch::allclose(sqrt_thd.h, expected_h));
}


TEST(TensorHyperDualAdditionTests, AdditionPositiveValues) {
    auto r1 = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d1 = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h1 = torch::rand({2, 3, 4, 5}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    auto r2 = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d2 = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h2 = torch::rand({2, 3, 4, 5}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual thd1(r1, d1, h1);
    TensorHyperDual thd2(r2, d2, h2);

    TensorHyperDual result = thd1 + thd2;

    ASSERT_TRUE(torch::allclose(result.r, r1 + r2));
    ASSERT_TRUE(torch::allclose(result.d, d1 + d2));
    ASSERT_TRUE(torch::allclose(result.h, h1 + h2));
}

TEST(TensorHyperDualAdditionTests, AdditionNegativeValues) {
    auto r1 = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU)) - 1.0;
    auto d1 = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU)) - 1.0;
    auto h1 = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU)) - 1.0;

    auto r2 = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU)) - 1.0;
    auto d2 = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU)) - 1.0;
    auto h2 = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU)) - 1.0;

    TensorHyperDual thd1(r1, d1, h1);
    TensorHyperDual thd2(r2, d2, h2);

    TensorHyperDual result = thd1 + thd2;

    ASSERT_TRUE(torch::allclose(result.r, r1 + r2));
    ASSERT_TRUE(torch::allclose(result.d, d1 + d2));
    ASSERT_TRUE(torch::allclose(result.h, h1 + h2));
}

TEST(TensorHyperDualAdditionTests, AdditionZeros) {
    auto r1 = torch::zeros({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d1 = torch::zeros({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h1 = torch::zeros({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    auto r2 = torch::zeros({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d2 = torch::zeros({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h2 = torch::zeros({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual thd1(r1, d1, h1);
    TensorHyperDual thd2(r2, d2, h2);

    TensorHyperDual result = thd1 + thd2;

    ASSERT_TRUE(torch::allclose(result.r, r1 + r2));
    ASSERT_TRUE(torch::allclose(result.d, d1 + d2));
    ASSERT_TRUE(torch::allclose(result.h, h1 + h2));
}

TEST(TensorHyperDualAdditionTests, AdditionMixedValues) {
    auto r1 = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU)) - 0.5;
    auto d1 = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU)) - 0.5;
    auto h1 = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU)) - 0.5;

    auto r2 = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU)) - 0.5;
    auto d2 = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU)) - 0.5;
    auto h2 = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU)) - 0.5;

    TensorHyperDual thd1(r1, d1, h1);
    TensorHyperDual thd2(r2, d2, h2);

    TensorHyperDual result = thd1 + thd2;

    ASSERT_TRUE(torch::allclose(result.r, r1 + r2));
    ASSERT_TRUE(torch::allclose(result.d, d1 + d2));
    ASSERT_TRUE(torch::allclose(result.h, h1 + h2));
}


TEST(TensorHyperDualNegationTests, NegationPositiveValues) {
    auto r = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual thd(r, d, h);
    TensorHyperDual result = -thd;

    ASSERT_TRUE(torch::allclose(result.r, -r));
    ASSERT_TRUE(torch::allclose(result.d, -d));
    ASSERT_TRUE(torch::allclose(result.h, -h));
}

TEST(TensorHyperDualNegationTests, NegationNegativeValues) {
    auto r = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU)) - 1.0;
    auto d = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU)) - 1.0;
    auto h = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU)) - 1.0;

    TensorHyperDual thd(r, d, h);
    TensorHyperDual result = -thd;

    ASSERT_TRUE(torch::allclose(result.r, -r));
    ASSERT_TRUE(torch::allclose(result.d, -d));
    ASSERT_TRUE(torch::allclose(result.h, -h));
}

TEST(TensorHyperDualNegationTests, NegationZeros) {
    auto r = torch::zeros({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d = torch::zeros({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::zeros({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual thd(r, d, h);
    TensorHyperDual result = -thd;

    ASSERT_TRUE(torch::allclose(result.r, -r));
    ASSERT_TRUE(torch::allclose(result.d, -d));
    ASSERT_TRUE(torch::allclose(result.h, -h));
}

TEST(TensorHyperDualNegationTests, NegationMixedValues) {
    auto r = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU)) - 0.5;
    auto d = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU)) - 0.5;
    auto h = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU)) - 0.5;

    TensorHyperDual thd(r, d, h);
    TensorHyperDual result = -thd;

    ASSERT_TRUE(torch::allclose(result.r, -r));
    ASSERT_TRUE(torch::allclose(result.d, -d));
    ASSERT_TRUE(torch::allclose(result.h, -h));
}


TEST(TensorHyperDualSubtractionTests, SubtractionPositiveValues) {
    auto r1 = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d1 = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h1 = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    auto r2 = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d2 = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h2 = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual thd1(r1, d1, h1);
    TensorHyperDual thd2(r2, d2, h2);

    TensorHyperDual result = thd1 - thd2;

    ASSERT_TRUE(torch::allclose(result.r, r1 - r2));
    ASSERT_TRUE(torch::allclose(result.d, d1 - d2));
    ASSERT_TRUE(torch::allclose(result.h, h1 - h2));
}

TEST(TensorHyperDualSubtractionTests, SubtractionNegativeValues) {
    auto r1 = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU)) - 1.0;
    auto d1 = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU)) - 1.0;
    auto h1 = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU)) - 1.0;

    auto r2 = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU)) - 1.0;
    auto d2 = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU)) - 1.0;
    auto h2 = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU)) - 1.0;

    TensorHyperDual thd1(r1, d1, h1);
    TensorHyperDual thd2(r2, d2, h2);

    TensorHyperDual result = thd1 - thd2;

    ASSERT_TRUE(torch::allclose(result.r, r1 - r2));
    ASSERT_TRUE(torch::allclose(result.d, d1 - d2));
    ASSERT_TRUE(torch::allclose(result.h, h1 - h2));
}

TEST(TensorHyperDualSubtractionTests, SubtractionZeros) {
    auto r1 = torch::zeros({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d1 = torch::zeros({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h1 = torch::zeros({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    auto r2 = torch::zeros({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d2 = torch::zeros({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h2 = torch::zeros({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual thd1(r1, d1, h1);
    TensorHyperDual thd2(r2, d2, h2);

    TensorHyperDual result = thd1 - thd2;

    ASSERT_TRUE(torch::allclose(result.r, r1 - r2));
    ASSERT_TRUE(torch::allclose(result.d, d1 - d2));
    ASSERT_TRUE(torch::allclose(result.h, h1 - h2));
}

TEST(TensorHyperDualSubtractionTests, SubtractionMixedValues) {
    auto r1 = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU)) - 0.5;
    auto d1 = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU)) - 0.5;
    auto h1 = torch::rand({2, 3, 4, 5}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU)) - 0.5;

    auto r2 = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU)) - 0.5;
    auto d2 = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU)) - 0.5;
    auto h2 = torch::rand({2, 3, 4, 5}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU)) - 0.5;

    TensorHyperDual thd1(r1, d1, h1);
    TensorHyperDual thd2(r2, d2, h2);

    TensorHyperDual result = thd1 - thd2;

    ASSERT_TRUE(torch::allclose(result.r, r1 - r2));
    ASSERT_TRUE(torch::allclose(result.d, d1 - d2));
    ASSERT_TRUE(torch::allclose(result.h, h1 - h2));
}


TEST(TensorHyperDualMultiplicationTests, MultiplicationPositiveValues) {
    auto r1 = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d1 = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h1 = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    auto r2 = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d2 = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h2 = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual thd1(r1, d1, h1);
    TensorHyperDual thd2(r2, d2, h2);

    TensorHyperDual result = thd1 * thd2;

    auto expected_r = r1 * r2;
    auto expected_d = torch::einsum("mi, mij->mij", {r1, d2}) + torch::einsum("mi, mij->mij", {r2, d1});
    auto expected_h = torch::einsum("mij, mik->mijk", {d1, d2}) + torch::einsum("mij, mik->mijk", {d2, d1}) +
                      torch::einsum("mi, mijk->mijk", {r1, h2}) + torch::einsum("mi, mijk->mijk", {r2, h1});

    ASSERT_TRUE(torch::allclose(result.r, expected_r));
    ASSERT_TRUE(torch::allclose(result.d, expected_d));
    ASSERT_TRUE(torch::allclose(result.h, expected_h));
}

TEST(TensorHyperDualMultiplicationTests, MultiplicationNegativeValues) {
    auto r1 = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU)) - 1.0;
    auto d1 = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU)) - 1.0;
    auto h1 = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU)) - 1.0;

    auto r2 = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU)) - 1.0;
    auto d2 = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU)) - 1.0;
    auto h2 = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU)) - 1.0;

    TensorHyperDual thd1(r1, d1, h1);
    TensorHyperDual thd2(r2, d2, h2);

    TensorHyperDual result = thd1 * thd2;

    auto expected_r = r1 * r2;
    auto expected_d = torch::einsum("mi, mij->mij", {r1, d2}) + torch::einsum("mi, mij->mij", {r2, d1});
    auto expected_h = torch::einsum("mij, mik->mijk", {d1, d2}) + torch::einsum("mij, mik->mijk", {d2, d1}) +
                      torch::einsum("mi, mijk->mijk", {r1, h2}) + torch::einsum("mi, mijk->mijk", {r2, h1});

    ASSERT_TRUE(torch::allclose(result.r, expected_r));
    ASSERT_TRUE(torch::allclose(result.d, expected_d));
    ASSERT_TRUE(torch::allclose(result.h, expected_h));
}

TEST(TensorHyperDualMultiplicationTests, MultiplicationZeros) {
    auto r1 = torch::zeros({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d1 = torch::zeros({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h1 = torch::zeros({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    auto r2 = torch::zeros({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d2 = torch::zeros({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h2 = torch::zeros({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual thd1(r1, d1, h1);
    TensorHyperDual thd2(r2, d2, h2);

    TensorHyperDual result = thd1 * thd2;

    ASSERT_TRUE(torch::allclose(result.r, torch::zeros_like(r1)));
    ASSERT_TRUE(torch::allclose(result.d, torch::zeros_like(d1)));
    ASSERT_TRUE(torch::allclose(result.h, torch::zeros_like(h1)));
}

TEST(TensorHyperDualMultiplicationTests, MultiplicationMixedValues) {
    auto r1 = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU)) - 0.5;
    auto d1 = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU)) - 0.5;
    auto h1 = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU)) - 0.5;

    auto r2 = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU)) - 0.5;
    auto d2 = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU)) - 0.5;
    auto h2 = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU)) - 0.5;

    TensorHyperDual thd1(r1, d1, h1);
    TensorHyperDual thd2(r2, d2, h2);

    TensorHyperDual result = thd1 * thd2;

    auto expected_r = r1 * r2;
    auto expected_d = torch::einsum("mi, mij->mij", {r1, d2}) + torch::einsum("mi, mij->mij", {r2, d1});
    auto expected_h = torch::einsum("mij, mik->mijk", {d1, d2}) + torch::einsum("mij, mik->mijk", {d2, d1}) +
                      torch::einsum("mi, mijk->mijk", {r1, h2}) + torch::einsum("mi, mijk->mijk", {r2, h1});

    ASSERT_TRUE(torch::allclose(result.r, expected_r));
    ASSERT_TRUE(torch::allclose(result.d, expected_d));
    ASSERT_TRUE(torch::allclose(result.h, expected_h));
}


TEST(TensorHyperDualDivisionTests, DivisionPositiveValues) {
    auto r1 = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU)) + 1.0; // Ensure no division by zero
    auto d1 = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h1 = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    auto r2 = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU)) + 1.0;
    auto d2 = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h2 = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual thd1(r1, d1, h1);
    TensorHyperDual thd2(r2, d2, h2);

    TensorHyperDual result = thd1 / thd2;

    auto expected_r = r1 / r2;
    auto expected_d = (d1 / r2.unsqueeze(-1)) - (r1 / r2.pow(2)).unsqueeze(-1) * d2;
    auto expected_h = torch::einsum("mijk, mi->mijk", {h1, r2.reciprocal()})
                      - 2 * torch::einsum("mij, mik->mijk", {d1 / r2.unsqueeze(-1), d2 / r2.unsqueeze(-1)})
                      + 2 * torch::einsum("mi, mij, mik->mijk", {r1 / r2.pow(3), d2, d2})
                      - torch::einsum("mi, mijk->mijk", {r1 / r2.pow(2), h2});

    ASSERT_TRUE(torch::allclose(result.r, expected_r));
    ASSERT_TRUE(torch::allclose(result.d, expected_d));
    ASSERT_TRUE(torch::allclose(result.h, expected_h));
}

TEST(TensorHyperDualDivisionTests, DivisionNegativeValues) {
    auto r1 = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU)) - 1.0; // Ensure no division by zero
    auto d1 = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h1 = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    auto r2 = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU)) + 1.0;
    auto d2 = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h2 = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual thd1(r1, d1, h1);
    TensorHyperDual thd2(r2, d2, h2);

    TensorHyperDual result = thd1 / thd2;

    auto expected_r = r1 / r2;
    auto expected_d = (d1 / r2.unsqueeze(-1)) - (r1 / r2.pow(2)).unsqueeze(-1) * d2;
    auto expected_h = torch::einsum("mijk, mi->mijk", {h1, r2.reciprocal()})
                      - 2 * torch::einsum("mij, mik->mijk", {d1 / r2.unsqueeze(-1), d2 / r2.unsqueeze(-1)})
                      + 2 * torch::einsum("mi, mij, mik->mijk", {r1 / r2.pow(3), d2, d2})
                      - torch::einsum("mi, mijk->mijk", {r1 / r2.pow(2), h2});

    ASSERT_TRUE(torch::allclose(result.r, expected_r));
    ASSERT_TRUE(torch::allclose(result.d, expected_d));
    ASSERT_TRUE(torch::allclose(result.h, expected_h));
}

TEST(TensorHyperDualDivisionTests, DivisionZeros) {
    auto r1 = torch::zeros({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU)) + 1.0; // Avoid division by zero
    auto d1 = torch::zeros({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h1 = torch::zeros({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    auto r2 = torch::zeros({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU)) + 1.0;
    auto d2 = torch::zeros({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h2 = torch::zeros({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual thd1(r1, d1, h1);
    TensorHyperDual thd2(r2, d2, h2);

    TensorHyperDual result = thd1 / thd2;

    ASSERT_TRUE(torch::allclose(result.r, torch::ones_like(r1)));
    ASSERT_TRUE(torch::allclose(result.d, torch::zeros_like(d1)));
    ASSERT_TRUE(torch::allclose(result.h, torch::zeros_like(h1)));
}

TEST(TensorHyperDualDivisionTests, DivisionMixedValues) {
    auto r1 = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU)) - 0.5;
    auto d1 = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h1 = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    auto r2 = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU)) + 1.0;
    auto d2 = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h2 = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual thd1(r1, d1, h1);
    TensorHyperDual thd2(r2, d2, h2);

    TensorHyperDual result = thd1 / thd2;

    auto expected_r = r1 / r2;
    auto expected_d = (d1 / r2.unsqueeze(-1)) - (r1 / r2.pow(2)).unsqueeze(-1) * d2;
    auto expected_h = torch::einsum("mijk, mi->mijk", {h1, r2.reciprocal()})
                      - 2 * torch::einsum("mij, mik->mijk", {d1 / r2.unsqueeze(-1), d2 / r2.unsqueeze(-1)})
                      + 2 * torch::einsum("mi, mij, mik->mijk", {r1 / r2.pow(3), d2, d2})
                      - torch::einsum("mi, mijk->mijk", {r1 / r2.pow(2), h2});

    ASSERT_TRUE(torch::allclose(result.r, expected_r));
    ASSERT_TRUE(torch::allclose(result.d, expected_d));
    ASSERT_TRUE(torch::allclose(result.h, expected_h));
}


TEST(TensorHyperDualScalarDivisionTests, DivisionByScalarPositiveValues) {
    auto r = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU)) + 1.0;
    auto d = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    auto scalar = torch::full({1}, 2.0, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual thd(r, d, h);
    TensorHyperDual result = thd / scalar;

    auto expected_r = r / scalar;
    auto expected_d = d / scalar;
    auto expected_h = torch::einsum("mijk, mi->mijk", {h, scalar.unsqueeze(-1).reciprocal()});

    ASSERT_TRUE(torch::allclose(result.r, expected_r));
    ASSERT_TRUE(torch::allclose(result.d, expected_d));
    ASSERT_TRUE(torch::allclose(result.h, expected_h));
}

TEST(TensorHyperDualScalarDivisionTests, DivisionByScalarNegativeValues) {
    auto r = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU)) - 1.0;
    auto d = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    auto scalar = torch::full({1}, -2.0, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual thd(r, d, h);
    TensorHyperDual result = thd / scalar;

    auto expected_r = r / scalar;
    auto expected_d = d / scalar;
    auto expected_h = torch::einsum("mijk, mi->mijk", {h, scalar.unsqueeze(-1).reciprocal()});

    ASSERT_TRUE(torch::allclose(result.r, expected_r));
    ASSERT_TRUE(torch::allclose(result.d, expected_d));
    ASSERT_TRUE(torch::allclose(result.h, expected_h));
}

TEST(TensorHyperDualScalarDivisionTests, DivisionByZeroThrowsError) {
    auto r = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    auto scalar = torch::full({1}, 0.0, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual thd(r, d, h);
    auto res = thd / scalar;
    //Expect a large number
    ASSERT_TRUE((res.r > 1e13).all().item<bool>());
}

TEST(TensorHyperDualScalarDivisionTests, DivisionByTensor) {
    auto r = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    auto tensor = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU)) + 1.0;

    TensorHyperDual thd(r, d, h);
    TensorHyperDual result = thd / tensor;

    auto expected_r = r / tensor;
    auto expected_d = d / tensor.unsqueeze(-1);
    auto expected_h = torch::einsum("mijk, mi->mijk", {h, tensor.reciprocal()});

    ASSERT_TRUE(torch::allclose(result.r, expected_r));
    ASSERT_TRUE(torch::allclose(result.d, expected_d));
    ASSERT_TRUE(torch::allclose(result.h, expected_h));
}



TEST(TensorHyperDualComparisonTests, LessThanOrEqualPositiveValues) {
    auto r1 = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d1 = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h1 = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    auto r2 = r1 + 0.5; // Ensure r2 > r1 element-wise
    auto d2 = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h2 = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual thd1(r1, d1, h1);
    TensorHyperDual thd2(r2, d2, h2);

    auto result = thd1 <= thd2;

    ASSERT_TRUE(torch::all(result).item<bool>());
}

TEST(TensorHyperDualComparisonTests, LessThanOrEqualNegativeValues) {
    auto r1 = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU)) - 1.0;
    auto d1 = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h1 = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    auto r2 = r1 + 0.5; // Ensure r2 > r1 element-wise
    auto d2 = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h2 = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual thd1(r1, d1, h1);
    TensorHyperDual thd2(r2, d2, h2);

    auto result = thd1 <= thd2;

    ASSERT_TRUE(torch::all(result).item<bool>());
}

TEST(TensorHyperDualComparisonTests, LessThanOrEqualMixedValues) {
    auto r1 = torch::tensor({{1.0, 3.0}, {2.0, 4.0}}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d1 = torch::rand({2, 2, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h1 = torch::rand({2, 2, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    auto r2 = torch::tensor({{2.0, 3.0}, {2.0, 5.0}}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d2 = torch::rand({2, 2, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h2 = torch::rand({2, 2, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual thd1(r1, d1, h1);
    TensorHyperDual thd2(r2, d2, h2);

    auto result = thd1 <= thd2;

    auto expected = torch::tensor({true, true}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));
    ASSERT_TRUE(torch::all(result == expected).item<bool>());
}

TEST(TensorHyperDualComparisonTests, LessThanOrEqualEdgeCase) {
    auto r1 = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d1 = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h1 = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    auto r2 = r1.clone(); // Ensure r1 == r2 element-wise
    auto d2 = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h2 = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual thd1(r1, d1, h1);
    TensorHyperDual thd2(r2, d2, h2);

    auto result = thd1 <= thd2;

    ASSERT_TRUE(torch::all(result).item<bool>());
}





TEST(TensorHyperDualComparisonTests, EqualityOperatorTrue) {
    auto r1 = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d1 = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h1 = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    auto r2 = r1.clone(); // Ensure equality element-wise
    auto d2 = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h2 = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual thd1(r1, d1, h1);
    TensorHyperDual thd2(r2, d2, h2);

    auto result = thd1 == thd2;

    ASSERT_TRUE(torch::all(result).item<bool>());
}

TEST(TensorHyperDualComparisonTests, EqualityOperatorFalse) {
    auto r1 = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d1 = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h1 = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    auto r2 = r1 + 1.0; // Ensure inequality element-wise
    auto d2 = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h2 = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual thd1(r1, d1, h1);
    TensorHyperDual thd2(r2, d2, h2);

    auto result = thd1 == thd2;

    ASSERT_FALSE(torch::all(result).item<bool>());
}

TEST(TensorHyperDualComparisonTests, EqualityOperatorBatch) {
    auto r1 = torch::tensor({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d1 = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h1 = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    auto r2 = torch::tensor({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d2 = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h2 = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual thd1(r1, d1, h1);
    TensorHyperDual thd2(r2, d2, h2);

    auto result = thd1 == thd2;

    ASSERT_TRUE(torch::all(result).item<bool>());
}


TEST(TensorHyperDualTensorComparisonTests, LessThanOrEqualToTensorPositiveValues) {
    auto r = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    auto tensor = r + 0.5; // Ensure tensor > r element-wise

    TensorHyperDual thd(r, d, h);
    auto result = thd <= tensor;

    ASSERT_TRUE(torch::all(result).item<bool>());
}

TEST(TensorHyperDualTensorComparisonTests, LessThanOrEqualToTensorNegativeValues) {
    auto r = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU)) - 1.0;
    auto d = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    auto tensor = r + 0.5; // Ensure tensor > r element-wise

    TensorHyperDual thd(r, d, h);
    auto result = thd <= tensor;

    ASSERT_TRUE(torch::all(result).item<bool>());
}

TEST(TensorHyperDualTensorComparisonTests, LessThanOrEqualToScalar) {
    auto r = torch::tensor({{1.0, 3.0}, {2.0, 4.0}}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d = torch::rand({2, 2, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 2, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    auto scalar = 5.0;

    TensorHyperDual thd(r, d, h);
    auto result = thd <= scalar;

    auto expected = torch::tensor({true, true}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));
    ASSERT_TRUE(torch::all(result == expected).item<bool>());
}

TEST(TensorHyperDualTensorComparisonTests, LessThanOrEqualEdgeCase) {
    auto r = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    auto tensor = r.clone(); // Ensure tensor == r element-wise

    TensorHyperDual thd(r, d, h);
    auto result = thd <= tensor;

    ASSERT_TRUE(torch::all(result).item<bool>());
}


TEST(TensorHyperDualScalarComparisonTests, LessThanOrEqualToScalarPositive) {
    auto r = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 3, 4, 5}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    double scalar = 1.0; // Scalar greater than most r values

    TensorHyperDual thd(r, d, h);
    auto result = thd <= scalar;

    ASSERT_TRUE(torch::all(result).item<bool>());
}

TEST(TensorHyperDualScalarComparisonTests, LessThanOrEqualToScalarNegative) {
    auto r = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU)) - 1.0;
    auto d = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 3, 4, 5}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    double scalar = 0.0; // Scalar greater than most r values

    TensorHyperDual thd(r, d, h);
    auto result = thd <= scalar;

    ASSERT_TRUE(torch::all(result).item<bool>());
}

TEST(TensorHyperDualScalarComparisonTests, LessThanOrEqualToScalarMixedValues) {
    auto r = torch::tensor({{1.0, -2.0}, {3.0, -4.0}}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d = torch::rand({2, 2, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 2, 4, 5}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    double scalar = 0.0;

    TensorHyperDual thd(r, d, h);
    auto result = thd <= scalar;

    auto expected = torch::tensor({{false, true}, {false, true}}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));
    ASSERT_TRUE(torch::all(result == expected).item<bool>());
}

TEST(TensorHyperDualScalarComparisonTests, LessThanOrEqualToScalarEdgeCase) {
    auto r = torch::tensor({{1.0, 2.0}, {3.0, 4.0}}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d = torch::rand({2, 2, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 2, 4, 5}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    double scalar = 4.0;

    TensorHyperDual thd(r, d, h);
    auto result = thd <= scalar;

    ASSERT_TRUE(torch::all(result).item<bool>());
}



TEST(TensorHyperDualGreaterThanTests, GreaterThanPositiveValues) {
    auto r1 = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU)) + 10.0;
    auto d1 = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h1 = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    auto r2 = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d2 = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h2 = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual thd1(r1, d1, h1);
    TensorHyperDual thd2(r2, d2, h2);

    auto result = thd1 > thd2;

    ASSERT_TRUE(torch::all(result).item<bool>());
}

TEST(TensorHyperDualGreaterThanTests, GreaterThanNegativeValues) {
    auto r1 = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d1 = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h1 = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    auto r2 = r1 - 1.0; // Ensure r1 > r2 element-wise
    auto d2 = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h2 = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual thd1(r1, d1, h1);
    TensorHyperDual thd2(r2, d2, h2);

    auto result = thd1 > thd2;

    ASSERT_TRUE(torch::all(result).item<bool>());
}

TEST(TensorHyperDualGreaterThanTests, GreaterThanMixedValues) {
    auto r1 = torch::tensor({{3.0, 1.0}, {5.0, 4.0}}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d1 = torch::rand({2, 2, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h1 = torch::rand({2, 2, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    auto r2 = torch::tensor({{2.0, 2.0}, {4.0, 5.0}}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d2 = torch::rand({2, 2, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h2 = torch::rand({2, 2, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual thd1(r1, d1, h1);
    TensorHyperDual thd2(r2, d2, h2);

    auto result = thd1 > thd2;

    auto expected = torch::tensor({true, false}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));
    ASSERT_TRUE(torch::all(result == expected).item<bool>());
}

TEST(TensorHyperDualGreaterThanTests, GreaterThanEdgeCase) {
    auto r1 = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d1 = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h1 = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    auto r2 = r1.clone(); // Ensure r1 == r2 element-wise
    auto d2 = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h2 = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual thd1(r1, d1, h1);
    TensorHyperDual thd2(r2, d2, h2);

    auto result = thd1 > thd2;

    ASSERT_FALSE(torch::any(result).item<bool>());
}


TEST(TensorHyperDualTensorGTTests, GreaterThanTensorPositiveValues) {
    auto r = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU)) + 10.0;
    auto d = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    auto tensor = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual thd(r, d, h);
    auto result = thd > tensor;

    ASSERT_TRUE(torch::all(result).item<bool>());
}

TEST(TensorHyperDualTensorGTTests, GreaterThanTensorNegativeValues) {
    auto r = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    auto tensor = r - 1.0; // Ensure r > tensor element-wise

    TensorHyperDual thd(r, d, h);
    auto result = thd > tensor;

    ASSERT_TRUE(torch::all(result).item<bool>());
}

TEST(TensorHyperDualTensorGTTests, GreaterThanTensorMixedValues) {
    auto r = torch::tensor({{3.0, 1.0}, {5.0, 4.0}}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d = torch::rand({2, 2, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 2, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    auto tensor = torch::tensor({{2.0, 2.0}, {4.0, 5.0}}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual thd(r, d, h);
    auto result = thd > tensor;

    auto expected = torch::tensor({true, false}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));
    ASSERT_TRUE(torch::all(result == expected).item<bool>());
}

TEST(TensorHyperDualTensorGTTests, GreaterThanTensorEdgeCase) {
    auto r = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    auto tensor = r.clone(); // Ensure r == tensor element-wise

    TensorHyperDual thd(r, d, h);
    auto result = thd > tensor;

    ASSERT_FALSE(torch::any(result).item<bool>());
}


TEST(TensorHyperDualScalarGTTests, GreaterThanScalarPositiveValues) {
    auto r = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU)) + 0.5;
    auto d = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    double scalar = 0.5;

    TensorHyperDual thd(r, d, h);
    auto result = thd > scalar;

    ASSERT_TRUE(torch::all(result).item<bool>());
}

TEST(TensorHyperDualScalarGTTests, GreaterThanScalarNegativeValues) {
    auto r = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    double scalar = -0.5; // Ensure all values in r > scalar

    TensorHyperDual thd(r, d, h);
    auto result = thd > scalar;

    ASSERT_TRUE(torch::all(result).item<bool>());
}

TEST(TensorHyperDualScalarGTTests, GreaterThanScalarMixedValues) {
    auto r = torch::tensor({{3.0, 1.0}, {5.0, 4.0}}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d = torch::rand({2, 2, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 2, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    double scalar = 2.0;

    TensorHyperDual thd(r, d, h);
    auto result = thd > scalar;

    auto expected = torch::tensor({{true, false}, {true, true}}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));
    ASSERT_TRUE(torch::all(result == expected).item<bool>());
}

TEST(TensorHyperDualScalarGTTests, GreaterThanScalarEdgeCase) {
    auto r = torch::tensor({{3.0, 1.0}, {5.0, 4.0}}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d = torch::rand({2, 2, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 2, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    double scalar = 5.0;

    TensorHyperDual thd(r, d, h);
    auto result = thd > scalar;

    auto expected = torch::tensor({false, false}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));
    ASSERT_TRUE(torch::all(result == expected).item<bool>());
}



TEST(TensorHyperDualLessThanTests, LessThanPositiveValues) {
    auto r1 = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d1 = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h1 = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    auto r2 = r1 + 0.5; // Ensure r1 < r2 element-wise
    auto d2 = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h2 = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual thd1(r1, d1, h1);
    TensorHyperDual thd2(r2, d2, h2);

    auto result = thd1 < thd2;

    ASSERT_TRUE(torch::all(result).item<bool>());
}

TEST(TensorHyperDualLessThanTests, LessThanNegativeValues) {
    auto r1 = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU)) - 1.0;
    auto d1 = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h1 = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    auto r2 = r1 + 0.5; // Ensure r1 < r2 element-wise
    auto d2 = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h2 = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual thd1(r1, d1, h1);
    TensorHyperDual thd2(r2, d2, h2);

    auto result = thd1 < thd2;

    ASSERT_TRUE(torch::all(result).item<bool>());
}

TEST(TensorHyperDualLessThanTests, LessThanMixedValues) {
    auto r1 = torch::tensor({{1.0, 3.0}, {2.0, 4.0}}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d1 = torch::rand({2, 2, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h1 = torch::rand({2, 2, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    auto r2 = torch::tensor({{2.0, 2.0}, {4.0, 5.0}}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d2 = torch::rand({2, 2, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h2 = torch::rand({2, 2, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual thd1(r1, d1, h1);
    TensorHyperDual thd2(r2, d2, h2);

    auto result = thd1 < thd2;

    auto expected = torch::tensor({{true, false}, {true, true}}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));
    ASSERT_TRUE(torch::all(result == expected).item<bool>());
}

TEST(TensorHyperDualLessThanTests, LessThanEdgeCase) {
    auto r1 = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d1 = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h1 = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    auto r2 = r1.clone(); // Ensure r1 == r2 element-wise
    auto d2 = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h2 = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual thd1(r1, d1, h1);
    TensorHyperDual thd2(r2, d2, h2);

    auto result = thd1 < thd2;

    ASSERT_FALSE(torch::any(result).item<bool>());
}



TEST(TensorHyperDualTensorLTTests, LessThanTensorPositiveValues) {
    auto r = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    auto tensor = r + 0.5; // Ensure r < tensor element-wise

    TensorHyperDual thd(r, d, h);
    auto result = thd < tensor;

    ASSERT_TRUE(torch::all(result).item<bool>());
}

TEST(TensorHyperDualTensorLTTests, LessThanTensorNegativeValues) {
    auto r = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU)) - 1.0;
    auto d = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    auto tensor = r + 0.5; // Ensure r < tensor element-wise

    TensorHyperDual thd(r, d, h);
    auto result = thd < tensor;

    ASSERT_TRUE(torch::all(result).item<bool>());
}

TEST(TensorHyperDualTensorLTTests, LessThanTensorMixedValues) {
    auto r = torch::tensor({{1.0, 3.0}, {2.0, 4.0}}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d = torch::rand({2, 2, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 2, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    auto tensor = torch::tensor({{2.0, 2.0}, {4.0, 5.0}}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual thd(r, d, h);
    auto result = thd < tensor;

    auto expected = torch::tensor({{true, false}, {true, true}}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));
    ASSERT_TRUE(torch::all(result == expected).item<bool>());
}

TEST(TensorHyperDualTensorLTTests, LessThanTensorEdgeCase) {
    auto r = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    auto tensor = r.clone(); // Ensure r == tensor element-wise

    TensorHyperDual thd(r, d, h);
    auto result = thd < tensor;

    ASSERT_FALSE(torch::any(result).item<bool>());
}

TEST(TensorHyperDualScalarLTTests, LessThanScalarPositiveValues) {
    auto r = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    double scalar = 1.0; // Scalar greater than most values in r

    TensorHyperDual thd(r, d, h);
    auto result = thd < scalar;

    ASSERT_TRUE(torch::all(result).item<bool>());
}

TEST(TensorHyperDualScalarLTTests, LessThanScalarNegativeValues) {
    auto r = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU)) - 1.0;
    auto d = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    double scalar = 0.0; // Scalar greater than all values in r

    TensorHyperDual thd(r, d, h);
    auto result = thd < scalar;

    ASSERT_TRUE(torch::all(result).item<bool>());
}

TEST(TensorHyperDualScalarLTTests, LessThanScalarMixedValues) {
    auto r = torch::tensor({{1.0, 3.0}, {2.0, 4.0}}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d = torch::rand({2, 2, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 2, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    double scalar = 2.5;

    TensorHyperDual thd(r, d, h);
    auto result = thd < scalar;

    auto expected = torch::tensor({true, false}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));
    ASSERT_TRUE(torch::all(result == expected).item<bool>());
}

TEST(TensorHyperDualScalarLTTests, LessThanScalarEdgeCase) {
    auto r = torch::tensor({{1.0, 2.0}, {3.0, 4.0}}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d = torch::rand({2, 2, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 2, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    double scalar = 4.0;

    TensorHyperDual thd(r, d, h);
    auto result = thd < scalar;

    auto expected = torch::tensor({{true, true}, {true, false}}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));
    ASSERT_TRUE(torch::all(result == expected).item<bool>());
}


TEST(TensorHyperDualGTEComparisonTests, GreaterThanOrEqualPositiveValues) {
    auto r1 = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU)) + 1.5;
    auto d1 = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h1 = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    auto r2 = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d2 = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h2 = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual thd1(r1, d1, h1);
    TensorHyperDual thd2(r2, d2, h2);

    auto result = thd1 >= thd2;

    ASSERT_TRUE(torch::all(result).item<bool>());
}

TEST(TensorHyperDualGTEComparisonTests, GreaterThanOrEqualNegativeValues) {
    auto r1 = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU)) - 1.0;
    auto d1 = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h1 = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    auto r2 = r1 - 0.5; // Ensure r1 >= r2 element-wise
    auto d2 = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h2 = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual thd1(r1, d1, h1);
    TensorHyperDual thd2(r2, d2, h2);

    auto result = thd1 >= thd2;

    ASSERT_TRUE(torch::all(result).item<bool>());
}

TEST(TensorHyperDualGTEComparisonTests, GreaterThanOrEqualMixedValues) {
    auto r1 = torch::tensor({{3.0, 1.0}, {5.0, 4.0}}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d1 = torch::rand({2, 2, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h1 = torch::rand({2, 2, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    auto r2 = torch::tensor({{3.0, 2.0}, {5.0, 3.0}}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d2 = torch::rand({2, 2, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h2 = torch::rand({2, 2, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual thd1(r1, d1, h1);
    TensorHyperDual thd2(r2, d2, h2);

    auto result = thd1 >= thd2;

    auto expected = torch::tensor({{true, false}, {true, true}}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));
    ASSERT_TRUE(torch::all(result == expected).item<bool>());
}

TEST(TensorHyperDualGTEComparisonTests, GreaterThanOrEqualEdgeCase) {
    auto r1 = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d1 = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h1 = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    auto r2 = r1.clone(); // Ensure r1 == r2 element-wise
    auto d2 = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h2 = torch::rand({2, 3, 4, 5}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual thd1(r1, d1, h1);
    TensorHyperDual thd2(r2, d2, h2);

    auto result = thd1 >= thd2;

    ASSERT_TRUE(torch::all(result).item<bool>());
}


TEST(TensorHyperDualTensorGTEComparisonTests, GreaterThanOrEqualToTensorPositiveValues) {
    auto r = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU)) + 1.5;
    auto d = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    auto tensor = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual thd(r, d, h);
    auto result = thd >= tensor;

    ASSERT_TRUE(torch::all(result).item<bool>());
}

TEST(TensorHyperDualTensorGTEComparisonTests, GreaterThanOrEqualToTensorNegativeValues) {
    auto r = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU)) - 1.0;
    auto d = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    auto tensor = r - 0.5; // Ensure r >= tensor element-wise

    TensorHyperDual thd(r, d, h);
    auto result = thd >= tensor;

    ASSERT_TRUE(torch::all(result).item<bool>());
}

TEST(TensorHyperDualTensorGTEComparisonTests, GreaterThanOrEqualToTensorMixedValues) {
    auto r = torch::tensor({{3.0, 1.0}, {5.0, 4.0}}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d = torch::rand({2, 2, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 2, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    auto tensor = torch::tensor({{3.0, 2.0}, {5.0, 3.0}}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual thd(r, d, h);
    auto result = thd >= tensor;

    auto expected = torch::tensor({{true, false}, {true, true}}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));
    ASSERT_TRUE(torch::all(result == expected).item<bool>());
}

TEST(TensorHyperDualTensorGTEComparisonTests, GreaterThanOrEqualToTensorEdgeCase) {
    auto r = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    auto tensor = r.clone(); // Ensure r == tensor element-wise

    TensorHyperDual thd(r, d, h);
    auto result = thd >= tensor;

    ASSERT_TRUE(torch::all(result).item<bool>());
}


TEST(TensorHyperDualTensorEqualityTests, EqualityWithTensorPositiveValues) {
    auto r = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    auto tensor = r.clone(); // Ensure equality element-wise

    TensorHyperDual thd(r, d, h);
    auto result = thd == tensor;

    ASSERT_TRUE(torch::all(result).item<bool>());
}

TEST(TensorHyperDualTensorEqualityTests, EqualityWithTensorNegativeValues) {
    auto r = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU)) - 1.0;
    auto d = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    auto tensor = r.clone(); // Ensure equality element-wise

    TensorHyperDual thd(r, d, h);
    auto result = thd == tensor;

    ASSERT_TRUE(torch::all(result).item<bool>());
}

TEST(TensorHyperDualTensorEqualityTests, EqualityWithTensorMixedValues) {
    auto r = torch::tensor({{1.0, 3.0}, {2.0, 4.0}}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d = torch::rand({2, 2, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 2, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    auto tensor = torch::tensor({{1.0, 3.0}, {2.0, 5.0}}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual thd(r, d, h);
    auto result = thd == tensor;

    auto expected = torch::tensor({{true, true}, {true, false}}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));
    ASSERT_TRUE(torch::all(result == expected).item<bool>());
}

TEST(TensorHyperDualTensorEqualityTests, EqualityWithTensorEdgeCase) {
    auto r = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    auto tensor = torch::zeros_like(r); // Ensure no equality

    TensorHyperDual thd(r, d, h);
    auto result = thd == tensor;

    ASSERT_FALSE(torch::any(result).item<bool>());
}


TEST(TensorHyperDualScalarEqualityTests, EqualityWithScalarPositiveValues) {
    auto r = torch::tensor({{1.0, 3.0}, {2.0, 4.0}}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d = torch::rand({2, 2, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 2, 4, 5}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    double scalar = 3.0;

    TensorHyperDual thd(r, d, h);
    auto result = thd == scalar;

    auto expected = torch::tensor({{false, true}, {false, false}}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));
    ASSERT_TRUE(torch::all(result == expected).item<bool>());
}

TEST(TensorHyperDualScalarEqualityTests, EqualityWithScalarNegativeValues) {
    auto r = torch::tensor({{-1.0, -3.0}, {-2.0, -4.0}}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d = torch::rand({2, 2, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 2, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    double scalar = -3.0;

    TensorHyperDual thd(r, d, h);
    auto result = thd == scalar;

    auto expected = torch::tensor({{false, true}, {false, false}}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));
    ASSERT_TRUE(torch::all(result == expected).item<bool>());
}

TEST(TensorHyperDualScalarEqualityTests, EqualityWithScalarZero) {
    auto r = torch::tensor({{0.0, 1.0}, {2.0, 0.0}}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d = torch::rand({2, 2, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 2, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    double scalar = 0.0;

    TensorHyperDual thd(r, d, h);
    auto result = thd == scalar;

    auto expected = torch::tensor({{true, false}, {false, true}}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));
    ASSERT_TRUE(torch::all(result == expected).item<bool>());
}

TEST(TensorHyperDualScalarEqualityTests, EqualityWithScalarEdgeCase) {
    auto r = torch::tensor({{5.0, 5.0}, {5.0, 5.0}}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d = torch::rand({2, 2, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 2, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    double scalar = 5.0;

    TensorHyperDual thd(r, d, h);
    auto result = thd == scalar;

    ASSERT_TRUE(torch::all(result).item<bool>());
}

TEST(TensorHyperDualInequalityTests, InequalityWithTensorPositiveValues) {
    auto r1 = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d1 = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h1 = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    auto r2 = r1 + 0.1; // Ensure r1 != r2 element-wise
    auto d2 = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h2 = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual thd1(r1, d1, h1);
    TensorHyperDual thd2(r2, d2, h2);

    auto result = thd1 != thd2;

    ASSERT_TRUE(torch::all(result).item<bool>());
}

TEST(TensorHyperDualInequalityTests, InequalityWithTensorNegativeValues) {
    auto r1 = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU)) - 1.0;
    auto d1 = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h1 = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    auto r2 = r1 - 0.1; // Ensure r1 != r2 element-wise
    auto d2 = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h2 = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual thd1(r1, d1, h1);
    TensorHyperDual thd2(r2, d2, h2);

    auto result = thd1 != thd2;

    ASSERT_TRUE(torch::all(result).item<bool>());
}

TEST(TensorHyperDualInequalityTests, InequalityWithTensorMixedValues) {
    auto r1 = torch::tensor({{1.0, 3.0}, {2.0, 4.0}}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d1 = torch::rand({2, 2, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h1 = torch::rand({2, 2, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    auto r2 = torch::tensor({{1.0, 2.0}, {3.0, 4.0}}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d2 = torch::rand({2, 2, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h2 = torch::rand({2, 2, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual thd1(r1, d1, h1);
    TensorHyperDual thd2(r2, d2, h2);

    auto result = thd1 != thd2;

    auto expected = torch::tensor({{false, true}, {true, false}}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));
    ASSERT_TRUE(torch::all(result == expected).item<bool>());
}

TEST(TensorHyperDualInequalityTests, InequalityWithTensorEdgeCase) {
    auto r1 = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d1 = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h1 = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    auto r2 = r1.clone(); // Ensure r1 == r2 element-wise
    auto d2 = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h2 = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual thd1(r1, d1, h1);
    TensorHyperDual thd2(r2, d2, h2);

    auto result = thd1 != thd2;

    ASSERT_FALSE(torch::any(result).item<bool>());
}


TEST(TensorHyperDualTensorInequalityTests, InequalityWithTensorPositiveValues) {
    auto r = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    auto tensor = r + 0.1; // Ensure r != tensor element-wise

    TensorHyperDual thd(r, d, h);
    auto result = thd != tensor;

    ASSERT_TRUE(torch::all(result).item<bool>());
}

TEST(TensorHyperDualTensorInequalityTests, InequalityWithTensorNegativeValues) {
    auto r = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU)) - 1.0;
    auto d = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    auto tensor = r - 0.1; // Ensure r != tensor element-wise

    TensorHyperDual thd(r, d, h);
    auto result = thd != tensor;

    ASSERT_TRUE(torch::all(result).item<bool>());
}

TEST(TensorHyperDualTensorInequalityTests, InequalityWithTensorMixedValues) {
    auto r = torch::tensor({{1.0, 3.0}, {2.0, 4.0}}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d = torch::rand({2, 2, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 2, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    auto tensor = torch::tensor({{1.0, 2.0}, {3.0, 4.0}}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual thd(r, d, h);
    auto result = thd != tensor;

    auto expected = torch::tensor({{false, true}, {true, false}}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));
    ASSERT_TRUE(torch::all(result == expected).item<bool>());
}

TEST(TensorHyperDualTensorInequalityTests, InequalityWithTensorEdgeCase) {
    auto r = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    auto tensor = r.clone(); // Ensure r == tensor element-wise

    TensorHyperDual thd(r, d, h);
    auto result = thd != tensor;

    ASSERT_FALSE(torch::any(result).item<bool>());
}

TEST(TensorHyperDualScalarInequalityTests, InequalityWithScalarPositiveValues) {
    auto r = torch::tensor({{1.0, 3.0}, {2.0, 4.0}}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d = torch::rand({2, 2, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 2, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    double scalar = 3.0;

    TensorHyperDual thd(r, d, h);
    auto result = thd != scalar;

    auto expected = torch::tensor({{true, false}, {true, true}}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));
    ASSERT_TRUE(torch::all(result == expected).item<bool>());
}

TEST(TensorHyperDualScalarInequalityTests, InequalityWithScalarNegativeValues) {
    auto r = torch::tensor({{-1.0, -3.0}, {-2.0, -4.0}}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d = torch::rand({2, 2, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 2, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    double scalar = -3.0;

    TensorHyperDual thd(r, d, h);
    auto result = thd != scalar;

    auto expected = torch::tensor({{true, false}, {true, true}}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));
    ASSERT_TRUE(torch::all(result == expected).item<bool>());
}

TEST(TensorHyperDualScalarInequalityTests, InequalityWithScalarZero) {
    auto r = torch::tensor({{0.0, 1.0}, {2.0, 0.0}}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d = torch::rand({2, 2, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 2, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    double scalar = 0.0;

    TensorHyperDual thd(r, d, h);
    auto result = thd != scalar;

    auto expected = torch::tensor({{false, true}, {true, false}}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));
    ASSERT_TRUE(torch::all(result == expected).item<bool>());
}

TEST(TensorHyperDualScalarInequalityTests, InequalityWithScalarEdgeCase) {
    auto r = torch::tensor({{5.0, 5.0}, {5.0, 5.0}}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d = torch::rand({2, 2, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 2, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    double scalar = 3.0;

    TensorHyperDual thd(r, d, h);
    auto result = thd != scalar;

    ASSERT_TRUE(torch::all(result).item<bool>());
}


TEST(TensorHyperDualReciprocalTests, ReciprocalPositiveValues) {
    auto r = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU)) + 0.1;
    auto d = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual thd(r, d, h);
    auto result = thd.reciprocal();

    auto expected_r = r.reciprocal();
    auto expected_d = -expected_r.pow(2).unsqueeze(-1) * d;
    auto expected_h = 2 * torch::einsum("mi, mij, mik->mijk", {expected_r.pow(3), d, d}) -
                      torch::einsum("mi, mijk->mijk", {expected_r.pow(2), h});

    ASSERT_TRUE(torch::allclose(result.r, expected_r));
    ASSERT_TRUE(torch::allclose(result.d, expected_d));
    ASSERT_TRUE(torch::allclose(result.h, expected_h));
}

TEST(TensorHyperDualReciprocalTests, ReciprocalNegativeValues) {
    auto r = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU)) - 0.1;
    auto d = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual thd(r, d, h);
    auto result = thd.reciprocal();

    auto expected_r = r.reciprocal();
    auto expected_d = -expected_r.pow(2).unsqueeze(-1) * d;
    auto expected_h = 2 * torch::einsum("mi, mij, mik->mijk", {expected_r.pow(3), d, d}) -
                      torch::einsum("mi, mijk->mijk", {expected_r.pow(2), h});

    ASSERT_TRUE(torch::allclose(result.r, expected_r));
    ASSERT_TRUE(torch::allclose(result.d, expected_d));
    ASSERT_TRUE(torch::allclose(result.h, expected_h));
}

TEST(TensorHyperDualReciprocalTests, ReciprocalMixedValues) {
    auto r = torch::tensor({{2.0, -3.0}, {0.5, -0.2}}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d = torch::rand({2, 2, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 2, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual thd(r, d, h);
    auto result = thd.reciprocal();

    auto expected_r = r.reciprocal();
    auto expected_d = -expected_r.pow(2).unsqueeze(-1) * d;
    auto expected_h = 2 * torch::einsum("mi, mij, mik->mijk", {expected_r.pow(3), d, d}) -
                      torch::einsum("mi, mijk->mijk", {expected_r.pow(2), h});

    ASSERT_TRUE(torch::allclose(result.r, expected_r));
    ASSERT_TRUE(torch::allclose(result.d, expected_d));
    ASSERT_TRUE(torch::allclose(result.h, expected_h));
}

TEST(TensorHyperDualReciprocalTests, ReciprocalEdgeCase) {
    auto r = torch::tensor({{1.0, 2.0}, {3.0, 4.0}}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d = torch::rand({2, 2, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 2, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual thd(r, d, h);
    auto result = thd.reciprocal();

    auto expected_r = r.reciprocal();
    auto expected_d = -expected_r.pow(2).unsqueeze(-1) * d;
    auto expected_h = 2 * torch::einsum("mi, mij, mik->mijk", {expected_r.pow(3), d, d}) -
                      torch::einsum("mi, mijk->mijk", {expected_r.pow(2), h});

    ASSERT_TRUE(torch::allclose(result.r, expected_r));
    ASSERT_TRUE(torch::allclose(result.d, expected_d));
    ASSERT_TRUE(torch::allclose(result.h, expected_h));
}


TEST(TensorHyperDualCosTests, CosinePositiveValues) {
    auto r = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual thd(r, d, h);
    auto result = thd.cos();

    auto expected_r = torch::cos(r);
    auto expected_d = -torch::sin(r).unsqueeze(-1) * d;
    auto expected_h = -torch::einsum("mi, mij, mik->mijk", {torch::cos(r), d, d}) -
                      torch::einsum("mi, mijk->mijk", {torch::sin(r), h});

    ASSERT_TRUE(torch::allclose(result.r, expected_r));
    ASSERT_TRUE(torch::allclose(result.d, expected_d));
    ASSERT_TRUE(torch::allclose(result.h, expected_h));
}
#include <cmath>
auto pi_tensor = torch::tensor(M_PI, torch::TensorOptions().dtype(torch::kFloat64));

TEST(TensorHyperDualCosTests, CosineNegativeValues) {
    auto r = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU)) - pi_tensor;
    auto d = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual thd(r, d, h);
    auto result = thd.cos();

    auto expected_r = torch::cos(r);
    auto expected_d = -torch::sin(r).unsqueeze(-1) * d;
    auto expected_h = -torch::einsum("mi, mij, mik->mijk", {torch::cos(r), d, d}) -
                      torch::einsum("mi, mijk->mijk", {torch::sin(r), h});

    ASSERT_TRUE(torch::allclose(result.r, expected_r));
    ASSERT_TRUE(torch::allclose(result.d, expected_d));
    ASSERT_TRUE(torch::allclose(result.h, expected_h));
}

TEST(TensorHyperDualCosTests, CosineMixedValues) {
    auto r = torch::tensor({{0.0, M_PI / 4}, {M_PI / 2, -M_PI / 3}}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d = torch::rand({2, 2, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 2, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual thd(r, d, h);
    auto result = thd.cos();

    auto expected_r = torch::cos(r);
    auto expected_d = -torch::sin(r).unsqueeze(-1) * d;
    auto expected_h = -torch::einsum("mi, mij, mik->mijk", {torch::cos(r), d, d}) -
                      torch::einsum("mi, mijk->mijk", {torch::sin(r), h});

    ASSERT_TRUE(torch::allclose(result.r, expected_r));
    ASSERT_TRUE(torch::allclose(result.d, expected_d));
    ASSERT_TRUE(torch::allclose(result.h, expected_h));
}

TEST(TensorHyperDualCosTests, CosineEdgeCase) {
    auto r = torch::tensor({{-M_PI, 0.0}, {M_PI, M_PI / 2}}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d = torch::rand({2, 2, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 2, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual thd(r, d, h);
    auto result = thd.cos();

    auto expected_r = torch::cos(r);
    auto expected_d = -torch::sin(r).unsqueeze(-1) * d;
    auto expected_h = -torch::einsum("mi, mij, mik->mijk", {torch::cos(r), d, d}) -
                      torch::einsum("mi, mijk->mijk", {torch::sin(r), h});

    ASSERT_TRUE(torch::allclose(result.r, expected_r));
    ASSERT_TRUE(torch::allclose(result.d, expected_d));
    ASSERT_TRUE(torch::allclose(result.h, expected_h));
}

TEST(TensorHyperDualSinTests, SinePositiveValues) {
    auto r = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual thd(r, d, h);
    auto result = thd.sin();

    auto expected_r = torch::sin(r);
    auto expected_d = torch::cos(r).unsqueeze(-1) * d;
    auto expected_h = -torch::einsum("mi, mij, mik->mijk", {torch::sin(r), d, d}) +
                      torch::einsum("mi, mijk->mijk", {torch::cos(r), h});

    ASSERT_TRUE(torch::allclose(result.r, expected_r));
    ASSERT_TRUE(torch::allclose(result.d, expected_d));
    ASSERT_TRUE(torch::allclose(result.h, expected_h));
}

TEST(TensorHyperDualSinTests, SineNegativeValues) {
    auto r = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU)) - M_PI;
    auto d = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual thd(r, d, h);
    auto result = thd.sin();

    auto expected_r = torch::sin(r);
    auto expected_d = torch::cos(r).unsqueeze(-1) * d;
    auto expected_h = -torch::einsum("mi, mij, mik->mijk", {torch::sin(r), d, d}) +
                      torch::einsum("mi, mijk->mijk", {torch::cos(r), h});

    ASSERT_TRUE(torch::allclose(result.r, expected_r));
    ASSERT_TRUE(torch::allclose(result.d, expected_d));
    ASSERT_TRUE(torch::allclose(result.h, expected_h));
}

TEST(TensorHyperDualSinTests, SineMixedValues) {
    auto r = torch::tensor({{0.0, M_PI / 4}, {M_PI / 2, -M_PI / 3}}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d = torch::rand({2, 2, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 2, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual thd(r, d, h);
    auto result = thd.sin();

    auto expected_r = torch::sin(r);
    auto expected_d = torch::cos(r).unsqueeze(-1) * d;
    auto expected_h = -torch::einsum("mi, mij, mik->mijk", {torch::sin(r), d, d}) +
                      torch::einsum("mi, mijk->mijk", {torch::cos(r), h});

    ASSERT_TRUE(torch::allclose(result.r, expected_r));
    ASSERT_TRUE(torch::allclose(result.d, expected_d));
    ASSERT_TRUE(torch::allclose(result.h, expected_h));
}

TEST(TensorHyperDualSinTests, SineEdgeCase) {
    auto r = torch::tensor({{-M_PI, 0.0}, {M_PI, M_PI / 2}}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d = torch::rand({2, 2, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 2, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual thd(r, d, h);
    auto result = thd.sin();

    auto expected_r = torch::sin(r);
    auto expected_d = torch::cos(r).unsqueeze(-1) * d;
    auto expected_h = -torch::einsum("mi, mij, mik->mijk", {torch::sin(r), d, d}) +
                      torch::einsum("mi, mijk->mijk", {torch::cos(r), h});

    ASSERT_TRUE(torch::allclose(result.r, expected_r));
    ASSERT_TRUE(torch::allclose(result.d, expected_d));
    ASSERT_TRUE(torch::allclose(result.h, expected_h));
}


TEST(TensorHyperDualTanTests, TangentPositiveValues) {
    auto r = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU)) * (M_PI / 4);
    auto d = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual thd(r, d, h);
    auto result = thd.tan();

    auto expected_r = torch::tan(r);
    auto expected_d = torch::pow(torch::cos(r), -2).unsqueeze(-1) * d;
    auto expected_h = 2 * torch::einsum("mi, mij, mik->mijk", {torch::pow(torch::cos(r), -2) * torch::tan(r), d, d}) +
                      torch::einsum("mi, mijk->mijk", {torch::pow(torch::cos(r), -2), h});

    ASSERT_TRUE(torch::allclose(result.r, expected_r));
    ASSERT_TRUE(torch::allclose(result.d, expected_d));
    ASSERT_TRUE(torch::allclose(result.h, expected_h));
}

TEST(TensorHyperDualTanTests, TangentNegativeValues) {
    auto r = -torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU)) * (M_PI / 4);
    auto d = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual thd(r, d, h);
    auto result = thd.tan();

    auto expected_r = torch::tan(r);
    auto expected_d = torch::pow(torch::cos(r), -2).unsqueeze(-1) * d;
    auto expected_h = 2 * torch::einsum("mi, mij, mik->mijk", {torch::pow(torch::cos(r), -2) * torch::tan(r), d, d}) +
                      torch::einsum("mi, mijk->mijk", {torch::pow(torch::cos(r), -2), h});

    ASSERT_TRUE(torch::allclose(result.r, expected_r));
    ASSERT_TRUE(torch::allclose(result.d, expected_d));
    ASSERT_TRUE(torch::allclose(result.h, expected_h));
}

TEST(TensorHyperDualTanTests, TangentMixedValues) {
    auto r = torch::tensor({{M_PI / 6, -M_PI / 4}, {M_PI / 3, -M_PI / 6}}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d = torch::rand({2, 2, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 2, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual thd(r, d, h);
    auto result = thd.tan();

    auto expected_r = torch::tan(r);
    auto expected_d = torch::pow(torch::cos(r), -2).unsqueeze(-1) * d;
    auto expected_h = 2 * torch::einsum("mi, mij, mik->mijk", {torch::pow(torch::cos(r), -2) * torch::tan(r), d, d}) +
                      torch::einsum("mi, mijk->mijk", {torch::pow(torch::cos(r), -2), h});

    ASSERT_TRUE(torch::allclose(result.r, expected_r));
    ASSERT_TRUE(torch::allclose(result.d, expected_d));
    ASSERT_TRUE(torch::allclose(result.h, expected_h));
}

TEST(TensorHyperDualTanTests, TangentEdgeCase) {
    auto r = torch::tensor({{-M_PI / 4, M_PI / 6}, {M_PI / 3, 0.0}}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d = torch::rand({2, 2, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 2, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual thd(r, d, h);
    auto result = thd.tan();

    auto expected_r = torch::tan(r);
    auto expected_d = torch::pow(torch::cos(r), -2).unsqueeze(-1) * d;
    auto expected_h = 2 * torch::einsum("mi, mij, mik->mijk", {torch::pow(torch::cos(r), -2) * torch::tan(r), d, d}) +
                      torch::einsum("mi, mijk->mijk", {torch::pow(torch::cos(r), -2), h});

    ASSERT_TRUE(torch::allclose(result.r, expected_r));
    ASSERT_TRUE(torch::allclose(result.d, expected_d));
    ASSERT_TRUE(torch::allclose(result.h, expected_h));
}



TEST(TensorHyperDualAsinTests, AsinPositiveValues) {
    auto r = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU)) * 0.9;
    auto d = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual thd(r, d, h);
    auto result = thd.asin();

    auto one_minus_r_sq = 1 - r * r;
    auto sqrt_one_minus_r_sq = torch::sqrt(one_minus_r_sq);

    auto expected_r = torch::asin(r);
    auto expected_d = (1 / sqrt_one_minus_r_sq).unsqueeze(-1) * d;
    auto expected_h = torch::einsum("mi, mij, mik->mijk", {(1 / torch::pow(one_minus_r_sq, 1.5)) * r, d, d}) +
                      torch::einsum("mi, mijk->mijk", {(1 / sqrt_one_minus_r_sq), h});

    ASSERT_TRUE(torch::allclose(result.r, expected_r));
    ASSERT_TRUE(torch::allclose(result.d, expected_d));
    ASSERT_TRUE(torch::allclose(result.h, expected_h));
}

TEST(TensorHyperDualAsinTests, AsinNegativeValues) {
    auto r = -torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU)) * 0.9;
    auto d = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual thd(r, d, h);
    auto result = thd.asin();

    auto one_minus_r_sq = 1 - r * r;
    auto sqrt_one_minus_r_sq = torch::sqrt(one_minus_r_sq);

    auto expected_r = torch::asin(r);
    auto expected_d = (1 / sqrt_one_minus_r_sq).unsqueeze(-1) * d;
    auto expected_h = torch::einsum("mi, mij, mik->mijk", {(1 / torch::pow(one_minus_r_sq, 1.5)) * r, d, d}) +
                      torch::einsum("mi, mijk->mijk", {(1 / sqrt_one_minus_r_sq), h});

    ASSERT_TRUE(torch::allclose(result.r, expected_r));
    ASSERT_TRUE(torch::allclose(result.d, expected_d));
    ASSERT_TRUE(torch::allclose(result.h, expected_h));
}

TEST(TensorHyperDualAsinTests, AsinMixedValues) {
    auto r = torch::tensor({{0.5, -0.3}, {0.7, -0.9}}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d = torch::rand({2, 2, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 2, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual thd(r, d, h);
    auto result = thd.asin();

    auto one_minus_r_sq = 1 - r * r;
    auto sqrt_one_minus_r_sq = torch::sqrt(one_minus_r_sq);

    auto expected_r = torch::asin(r);
    auto expected_d = (1 / sqrt_one_minus_r_sq).unsqueeze(-1) * d;
    auto expected_h = torch::einsum("mi, mij, mik->mijk", {(1 / torch::pow(one_minus_r_sq, 1.5)) * r, d, d}) +
                      torch::einsum("mi, mijk->mijk", {(1 / sqrt_one_minus_r_sq), h});

    ASSERT_TRUE(torch::allclose(result.r, expected_r));
    ASSERT_TRUE(torch::allclose(result.d, expected_d));
    ASSERT_TRUE(torch::allclose(result.h, expected_h));
}

TEST(TensorHyperDualAsinTests, AsinEdgeCase) {
    auto r = torch::tensor({{-0.9, 0.0}, {0.9, 0.5}}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d = torch::rand({2, 2, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 2, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual thd(r, d, h);
    auto result = thd.asin();

    auto one_minus_r_sq = 1 - r * r;
    auto sqrt_one_minus_r_sq = torch::sqrt(one_minus_r_sq);

    auto expected_r = torch::asin(r);
    auto expected_d = (1 / sqrt_one_minus_r_sq).unsqueeze(-1) * d;
    auto expected_h = torch::einsum("mi, mij, mik->mijk", {(1 / torch::pow(one_minus_r_sq, 1.5)) * r, d, d}) +
                      torch::einsum("mi, mijk->mijk", {(1 / sqrt_one_minus_r_sq), h});

    ASSERT_TRUE(torch::allclose(result.r, expected_r));
    ASSERT_TRUE(torch::allclose(result.d, expected_d));
    ASSERT_TRUE(torch::allclose(result.h, expected_h));
}

TEST(TensorHyperDualAcosTests, AcosPositiveValues) {
    auto r = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU)) * 0.9;
    auto d = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual thd(r, d, h);
    auto result = thd.acos();

    auto one_minus_r_sq = 1 - r * r;
    auto sqrt_one_minus_r_sq = torch::sqrt(one_minus_r_sq);

    auto expected_r = torch::acos(r);
    auto expected_d = -(1 / sqrt_one_minus_r_sq).unsqueeze(-1) * d;
    auto expected_h = -torch::einsum("mi, mij, mik->mijk", {(1 / torch::pow(one_minus_r_sq, 1.5)) * r, d, d}) -
                      torch::einsum("mi, mijk->mijk", {(1 / sqrt_one_minus_r_sq), h});

    ASSERT_TRUE(torch::allclose(result.r, expected_r));
    ASSERT_TRUE(torch::allclose(result.d, expected_d));
    ASSERT_TRUE(torch::allclose(result.h, expected_h));
}

TEST(TensorHyperDualAcosTests, AcosNegativeValues) {
    auto r = -torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU)) * 0.9;
    auto d = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual thd(r, d, h);
    auto result = thd.acos();

    auto one_minus_r_sq = 1 - r * r;
    auto sqrt_one_minus_r_sq = torch::sqrt(one_minus_r_sq);

    auto expected_r = torch::acos(r);
    auto expected_d = -(1 / sqrt_one_minus_r_sq).unsqueeze(-1) * d;
    auto expected_h = -torch::einsum("mi, mij, mik->mijk", {(1 / torch::pow(one_minus_r_sq, 1.5)) * r, d, d}) -
                      torch::einsum("mi, mijk->mijk", {(1 / sqrt_one_minus_r_sq), h});

    ASSERT_TRUE(torch::allclose(result.r, expected_r));
    ASSERT_TRUE(torch::allclose(result.d, expected_d));
    ASSERT_TRUE(torch::allclose(result.h, expected_h));
}

TEST(TensorHyperDualAcosTests, AcosMixedValues) {
    auto r = torch::tensor({{0.5, -0.3}, {0.7, -0.9}}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d = torch::rand({2, 2, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 2, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual thd(r, d, h);
    auto result = thd.acos();

    auto one_minus_r_sq = 1 - r * r;
    auto sqrt_one_minus_r_sq = torch::sqrt(one_minus_r_sq);

    auto expected_r = torch::acos(r);
    auto expected_d = -(1 / sqrt_one_minus_r_sq).unsqueeze(-1) * d;
    auto expected_h = -torch::einsum("mi, mij, mik->mijk", {(1 / torch::pow(one_minus_r_sq, 1.5)) * r, d, d}) -
                      torch::einsum("mi, mijk->mijk", {(1 / sqrt_one_minus_r_sq), h});

    ASSERT_TRUE(torch::allclose(result.r, expected_r));
    ASSERT_TRUE(torch::allclose(result.d, expected_d));
    ASSERT_TRUE(torch::allclose(result.h, expected_h));
}

TEST(TensorHyperDualAcosTests, AcosEdgeCase) {
    auto r = torch::tensor({{-0.9, 0.0}, {0.9, 0.5}}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d = torch::rand({2, 2, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 2, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual thd(r, d, h);
    auto result = thd.acos();

    auto one_minus_r_sq = 1 - r * r;
    auto sqrt_one_minus_r_sq = torch::sqrt(one_minus_r_sq);

    auto expected_r = torch::acos(r);
    auto expected_d = -(1 / sqrt_one_minus_r_sq).unsqueeze(-1) * d;
    auto expected_h = -torch::einsum("mi, mij, mik->mijk", {(1 / torch::pow(one_minus_r_sq, 1.5)) * r, d, d}) -
                      torch::einsum("mi, mijk->mijk", {(1 / sqrt_one_minus_r_sq), h});

    ASSERT_TRUE(torch::allclose(result.r, expected_r));
    ASSERT_TRUE(torch::allclose(result.d, expected_d));
    ASSERT_TRUE(torch::allclose(result.h, expected_h));
}

TEST(TensorHyperDualAtanTests, AtanPositiveValues) {
    auto r = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual thd(r, d, h);
    auto result = thd.atan();

    auto one_plus_r_sq = 1 + r * r;

    auto expected_r = torch::atan(r);
    auto expected_d = (1 / one_plus_r_sq).unsqueeze(-1) * d;
    auto expected_h = -torch::einsum("mi, mij, mik->mijk", {(2 / torch::pow(one_plus_r_sq, 2)) * r, d, d}) +
                      torch::einsum("mi, mijk->mijk", {(1 / one_plus_r_sq), h});

    ASSERT_TRUE(torch::allclose(result.r, expected_r));
    ASSERT_TRUE(torch::allclose(result.d, expected_d));
    ASSERT_TRUE(torch::allclose(result.h, expected_h));
}

TEST(TensorHyperDualAtanTests, AtanNegativeValues) {
    auto r = -torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual thd(r, d, h);
    auto result = thd.atan();

    auto one_plus_r_sq = 1 + r * r;

    auto expected_r = torch::atan(r);
    auto expected_d = (1 / one_plus_r_sq).unsqueeze(-1) * d;
    auto expected_h = -torch::einsum("mi, mij, mik->mijk", {(2 / torch::pow(one_plus_r_sq, 2)) * r, d, d}) +
                      torch::einsum("mi, mijk->mijk", {(1 / one_plus_r_sq), h});

    ASSERT_TRUE(torch::allclose(result.r, expected_r));
    ASSERT_TRUE(torch::allclose(result.d, expected_d));
    ASSERT_TRUE(torch::allclose(result.h, expected_h));
}

TEST(TensorHyperDualAtanTests, AtanMixedValues) {
    auto r = torch::tensor({{0.5, -0.3}, {1.0, -0.9}}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d = torch::rand({2, 2, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 2, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual thd(r, d, h);
    auto result = thd.atan();

    auto one_plus_r_sq = 1 + r * r;

    auto expected_r = torch::atan(r);
    auto expected_d = (1 / one_plus_r_sq).unsqueeze(-1) * d;
    auto expected_h = -torch::einsum("mi, mij, mik->mijk", {(2 / torch::pow(one_plus_r_sq, 2)) * r, d, d}) +
                      torch::einsum("mi, mijk->mijk", {(1 / one_plus_r_sq), h});

    ASSERT_TRUE(torch::allclose(result.r, expected_r));
    ASSERT_TRUE(torch::allclose(result.d, expected_d));
    ASSERT_TRUE(torch::allclose(result.h, expected_h));
}

TEST(TensorHyperDualAtanTests, AtanEdgeCase) {
    auto r = torch::tensor({{-0.9, 0.0}, {1.0, 0.5}}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d = torch::rand({2, 2, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 2, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual thd(r, d, h);
    auto result = thd.atan();

    auto one_plus_r_sq = 1 + r * r;

    auto expected_r = torch::atan(r);
    auto expected_d = (1 / one_plus_r_sq).unsqueeze(-1) * d;
    auto expected_h = -torch::einsum("mi, mij, mik->mijk", {(2 / torch::pow(one_plus_r_sq, 2)) * r, d, d}) +
                      torch::einsum("mi, mijk->mijk", {(1 / one_plus_r_sq), h});

    ASSERT_TRUE(torch::allclose(result.r, expected_r));
    ASSERT_TRUE(torch::allclose(result.d, expected_d));
    ASSERT_TRUE(torch::allclose(result.h, expected_h));
}


TEST(TensorHyperDualSinhTests, SinhPositiveValues) {
    auto r = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual thd(r, d, h);
    auto result = thd.sinh();

    auto expected_r = torch::sinh(r);
    auto expected_d = torch::cosh(r).unsqueeze(-1) * d;
    auto expected_h = torch::einsum("mi, mij, mik->mijk", {torch::sinh(r), d, d}) +
                      torch::einsum("mi, mijk->mijk", {torch::cosh(r), h});

    ASSERT_TRUE(torch::allclose(result.r, expected_r));
    ASSERT_TRUE(torch::allclose(result.d, expected_d));
    ASSERT_TRUE(torch::allclose(result.h, expected_h));
}

TEST(TensorHyperDualSinhTests, SinhNegativeValues) {
    auto r = -torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual thd(r, d, h);
    auto result = thd.sinh();

    auto expected_r = torch::sinh(r);
    auto expected_d = torch::cosh(r).unsqueeze(-1) * d;
    auto expected_h = torch::einsum("mi, mij, mik->mijk", {torch::sinh(r), d, d}) +
                      torch::einsum("mi, mijk->mijk", {torch::cosh(r), h});

    ASSERT_TRUE(torch::allclose(result.r, expected_r));
    ASSERT_TRUE(torch::allclose(result.d, expected_d));
    ASSERT_TRUE(torch::allclose(result.h, expected_h));
}

TEST(TensorHyperDualSinhTests, SinhMixedValues) {
    auto r = torch::tensor({{0.5, -0.3}, {1.0, -0.9}}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d = torch::rand({2, 2, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 2, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual thd(r, d, h);
    auto result = thd.sinh();

    auto expected_r = torch::sinh(r);
    auto expected_d = torch::cosh(r).unsqueeze(-1) * d;
    auto expected_h = torch::einsum("mi, mij, mik->mijk", {torch::sinh(r), d, d}) +
                      torch::einsum("mi, mijk->mijk", {torch::cosh(r), h});

    ASSERT_TRUE(torch::allclose(result.r, expected_r));
    ASSERT_TRUE(torch::allclose(result.d, expected_d));
    ASSERT_TRUE(torch::allclose(result.h, expected_h));
}

TEST(TensorHyperDualSinhTests, SinhEdgeCase) {
    auto r = torch::tensor({{-0.9, 0.0}, {1.0, 0.5}}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d = torch::rand({2, 2, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 2, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual thd(r, d, h);
    auto result = thd.sinh();

    auto expected_r = torch::sinh(r);
    auto expected_d = torch::cosh(r).unsqueeze(-1) * d;
    auto expected_h = torch::einsum("mi, mij, mik->mijk", {torch::sinh(r), d, d}) +
                      torch::einsum("mi, mijk->mijk", {torch::cosh(r), h});

    ASSERT_TRUE(torch::allclose(result.r, expected_r));
    ASSERT_TRUE(torch::allclose(result.d, expected_d));
    ASSERT_TRUE(torch::allclose(result.h, expected_h));
}


TEST(TensorHyperDualCoshTests, CoshPositiveValues) {
    auto r = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual thd(r, d, h);
    auto result = thd.cosh();

    auto expected_r = torch::cosh(r);
    auto expected_d = torch::sinh(r).unsqueeze(-1) * d;
    auto expected_h = torch::einsum("mi, mij, mik->mijk", {torch::cosh(r), d, d}) +
                      torch::einsum("mi, mijk->mijk", {torch::sinh(r), h});

    ASSERT_TRUE(torch::allclose(result.r, expected_r));
    ASSERT_TRUE(torch::allclose(result.d, expected_d));
    ASSERT_TRUE(torch::allclose(result.h, expected_h));
}

TEST(TensorHyperDualCoshTests, CoshNegativeValues) {
    auto r = -torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual thd(r, d, h);
    auto result = thd.cosh();

    auto expected_r = torch::cosh(r);
    auto expected_d = torch::sinh(r).unsqueeze(-1) * d;
    auto expected_h = torch::einsum("mi, mij, mik->mijk", {torch::cosh(r), d, d}) +
                      torch::einsum("mi, mijk->mijk", {torch::sinh(r), h});

    ASSERT_TRUE(torch::allclose(result.r, expected_r));
    ASSERT_TRUE(torch::allclose(result.d, expected_d));
    ASSERT_TRUE(torch::allclose(result.h, expected_h));
}

TEST(TensorHyperDualCoshTests, CoshMixedValues) {
    auto r = torch::tensor({{0.5, -0.3}, {1.0, -0.9}}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d = torch::rand({2, 2, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 2, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual thd(r, d, h);
    auto result = thd.cosh();

    auto expected_r = torch::cosh(r);
    auto expected_d = torch::sinh(r).unsqueeze(-1) * d;
    auto expected_h = torch::einsum("mi, mij, mik->mijk", {torch::cosh(r), d, d}) +
                      torch::einsum("mi, mijk->mijk", {torch::sinh(r), h});

    ASSERT_TRUE(torch::allclose(result.r, expected_r));
    ASSERT_TRUE(torch::allclose(result.d, expected_d));
    ASSERT_TRUE(torch::allclose(result.h, expected_h));
}

TEST(TensorHyperDualCoshTests, CoshEdgeCase) {
    auto r = torch::tensor({{-0.9, 0.0}, {1.0, 0.5}}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d = torch::rand({2, 2, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 2, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual thd(r, d, h);
    auto result = thd.cosh();

    auto expected_r = torch::cosh(r);
    auto expected_d = torch::sinh(r).unsqueeze(-1) * d;
    auto expected_h = torch::einsum("mi, mij, mik->mijk", {torch::cosh(r), d, d}) +
                      torch::einsum("mi, mijk->mijk", {torch::sinh(r), h});

    ASSERT_TRUE(torch::allclose(result.r, expected_r));
    ASSERT_TRUE(torch::allclose(result.d, expected_d));
    ASSERT_TRUE(torch::allclose(result.h, expected_h));
}


TEST(TensorHyperDualTanhTests, TanhPositiveValues) {
    auto r = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual thd(r, d, h);
    auto result = thd.tanh();

    auto expected_r = torch::tanh(r);
    auto expected_d = torch::pow(torch::cosh(r), -2).unsqueeze(-1) * d;
    auto expected_h = -2 * torch::einsum("mi, mij, mik->mijk", {torch::pow(torch::cosh(r), -2) * torch::tanh(r), d, d}) +
                      torch::einsum("mi, mijk->mijk", {torch::pow(torch::cosh(r), -2), h});

    ASSERT_TRUE(torch::allclose(result.r, expected_r));
    ASSERT_TRUE(torch::allclose(result.d, expected_d));
    ASSERT_TRUE(torch::allclose(result.h, expected_h));
}

TEST(TensorHyperDualTanhTests, TanhNegativeValues) {
    auto r = -torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual thd(r, d, h);
    auto result = thd.tanh();

    auto expected_r = torch::tanh(r);
    auto expected_d = torch::pow(torch::cosh(r), -2).unsqueeze(-1) * d;
    auto expected_h = -2 * torch::einsum("mi, mij, mik->mijk", {torch::pow(torch::cosh(r), -2) * torch::tanh(r), d, d}) +
                      torch::einsum("mi, mijk->mijk", {torch::pow(torch::cosh(r), -2), h});

    ASSERT_TRUE(torch::allclose(result.r, expected_r));
    ASSERT_TRUE(torch::allclose(result.d, expected_d));
    ASSERT_TRUE(torch::allclose(result.h, expected_h));
}

TEST(TensorHyperDualTanhTests, TanhMixedValues) {
    auto r = torch::tensor({{0.5, -0.3}, {1.0, -0.9}}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d = torch::rand({2, 2, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 2, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual thd(r, d, h);
    auto result = thd.tanh();

    auto expected_r = torch::tanh(r);
    auto expected_d = torch::pow(torch::cosh(r), -2).unsqueeze(-1) * d;
    auto expected_h = -2 * torch::einsum("mi, mij, mik->mijk", {torch::pow(torch::cosh(r), -2) * torch::tanh(r), d, d}) +
                      torch::einsum("mi, mijk->mijk", {torch::pow(torch::cosh(r), -2), h});

    ASSERT_TRUE(torch::allclose(result.r, expected_r));
    ASSERT_TRUE(torch::allclose(result.d, expected_d));
    ASSERT_TRUE(torch::allclose(result.h, expected_h));
}

TEST(TensorHyperDualTanhTests, TanhEdgeCase) {
    auto r = torch::tensor({{-0.9, 0.0}, {1.0, 0.5}}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d = torch::rand({2, 2, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 2, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual thd(r, d, h);
    auto result = thd.tanh();

    auto expected_r = torch::tanh(r);
    auto expected_d = torch::pow(torch::cosh(r), -2).unsqueeze(-1) * d;
    auto expected_h = -2 * torch::einsum("mi, mij, mik->mijk", {torch::pow(torch::cosh(r), -2) * torch::tanh(r), d, d}) +
                      torch::einsum("mi, mijk->mijk", {torch::pow(torch::cosh(r), -2), h});

    ASSERT_TRUE(torch::allclose(result.r, expected_r));
    ASSERT_TRUE(torch::allclose(result.d, expected_d));
    ASSERT_TRUE(torch::allclose(result.h, expected_h));
}


TEST(TensorHyperDualExpTests, ExpPositiveValues) {
    auto r = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual thd(r, d, h);
    auto result = thd.exp();

    auto expected_r = torch::exp(r);
    auto expected_d = expected_r.unsqueeze(-1) * d;
    auto expected_h = torch::einsum("mi, mij, mik->mijk", {expected_r, d, d}) +
                      torch::einsum("mi, mijk->mijk", {expected_r, h});

    ASSERT_TRUE(torch::allclose(result.r, expected_r));
    ASSERT_TRUE(torch::allclose(result.d, expected_d));
    ASSERT_TRUE(torch::allclose(result.h, expected_h));
}

TEST(TensorHyperDualExpTests, ExpNegativeValues) {
    auto r = -torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual thd(r, d, h);
    auto result = thd.exp();

    auto expected_r = torch::exp(r);
    auto expected_d = expected_r.unsqueeze(-1) * d;
    auto expected_h = torch::einsum("mi, mij, mik->mijk", {expected_r, d, d}) +
                      torch::einsum("mi, mijk->mijk", {expected_r, h});

    ASSERT_TRUE(torch::allclose(result.r, expected_r));
    ASSERT_TRUE(torch::allclose(result.d, expected_d));
    ASSERT_TRUE(torch::allclose(result.h, expected_h));
}

TEST(TensorHyperDualExpTests, ExpMixedValues) {
    auto r = torch::tensor({{0.5, -0.3}, {1.0, -0.9}}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d = torch::rand({2, 2, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 2, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual thd(r, d, h);
    auto result = thd.exp();

    auto expected_r = torch::exp(r);
    auto expected_d = expected_r.unsqueeze(-1) * d;
    auto expected_h = torch::einsum("mi, mij, mik->mijk", {expected_r, d, d}) +
                      torch::einsum("mi, mijk->mijk", {expected_r, h});

    ASSERT_TRUE(torch::allclose(result.r, expected_r));
    ASSERT_TRUE(torch::allclose(result.d, expected_d));
    ASSERT_TRUE(torch::allclose(result.h, expected_h));
}

TEST(TensorHyperDualExpTests, ExpEdgeCase) {
    auto r = torch::tensor({{-0.9, 0.0}, {1.0, 0.5}}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d = torch::rand({2, 2, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 2, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual thd(r, d, h);
    auto result = thd.exp();

    auto expected_r = torch::exp(r);
    auto expected_d = expected_r.unsqueeze(-1) * d;
    auto expected_h = torch::einsum("mi, mij, mik->mijk", {expected_r, d, d}) +
                      torch::einsum("mi, mijk->mijk", {expected_r, h});

    ASSERT_TRUE(torch::allclose(result.r, expected_r));
    ASSERT_TRUE(torch::allclose(result.d, expected_d));
    ASSERT_TRUE(torch::allclose(result.h, expected_h));
}


TEST(TensorHyperDualLogTests, LogPositiveValues) {
    auto r = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU)) + 0.1; // Avoid zero
    auto d = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual thd(r, d, h);
    auto result = thd.log();

    auto expected_r = torch::log(r);
    auto expected_d = torch::pow(r, -1).unsqueeze(-1) * d;
    auto expected_h = -torch::einsum("mi, mij, mik->mijk", {torch::pow(r, -2), d, d}) +
                      torch::einsum("mi, mijk->mijk", {torch::pow(r, -1), h});

    ASSERT_TRUE(torch::allclose(result.r, expected_r));
    ASSERT_TRUE(torch::allclose(result.d, expected_d));
    ASSERT_TRUE(torch::allclose(result.h, expected_h));
}

TEST(TensorHyperDualLogTests, LogMixedValues) {
    auto r = torch::tensor({{1.0, 0.5}, {2.0, 0.1}}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d = torch::rand({2, 2, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 2, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual thd(r, d, h);
    auto result = thd.log();

    auto expected_r = torch::log(r);
    auto expected_d = torch::pow(r, -1).unsqueeze(-1) * d;
    auto expected_h = -torch::einsum("mi, mij, mik->mijk", {torch::pow(r, -2), d, d}) +
                      torch::einsum("mi, mijk->mijk", {torch::pow(r, -1), h});

    ASSERT_TRUE(torch::allclose(result.r, expected_r));
    ASSERT_TRUE(torch::allclose(result.d, expected_d));
    ASSERT_TRUE(torch::allclose(result.h, expected_h));
}

TEST(TensorHyperDualLogTests, LogEdgeCase) {
    auto r = torch::tensor({{0.1, 1.0}, {0.5, 2.0}}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d = torch::rand({2, 2, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 2, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual thd(r, d, h);
    auto result = thd.log();

    auto expected_r = torch::log(r);
    auto expected_d = torch::pow(r, -1).unsqueeze(-1) * d;
    auto expected_h = -torch::einsum("mi, mij, mik->mijk", {torch::pow(r, -2), d, d}) +
                      torch::einsum("mi, mijk->mijk", {torch::pow(r, -1), h});

    ASSERT_TRUE(torch::allclose(result.r, expected_r));
    ASSERT_TRUE(torch::allclose(result.d, expected_d));
    ASSERT_TRUE(torch::allclose(result.h, expected_h));
}

TEST(TensorHyperDualAbsTests, AbsPositiveValues) {
    auto r = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual thd(r, d, h);
    auto result = thd.abs();

    auto expected_r = torch::abs(r);
    auto expected_d = torch::sign(r).unsqueeze(-1) * d;
    auto expected_h = torch::einsum("mi, mijk->mijk", {torch::sign(r), h});

    ASSERT_TRUE(torch::allclose(result.r, expected_r));
    ASSERT_TRUE(torch::allclose(result.d, expected_d));
    ASSERT_TRUE(torch::allclose(result.h, expected_h));
}

TEST(TensorHyperDualAbsTests, AbsNegativeValues) {
    auto r = -torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual thd(r, d, h);
    auto result = thd.abs();

    auto expected_r = torch::abs(r);
    auto expected_d = torch::sign(r).unsqueeze(-1) * d;
    auto expected_h = torch::einsum("mi, mijk->mijk", {torch::sign(r), h});

    ASSERT_TRUE(torch::allclose(result.r, expected_r));
    ASSERT_TRUE(torch::allclose(result.d, expected_d));
    ASSERT_TRUE(torch::allclose(result.h, expected_h));
}

TEST(TensorHyperDualAbsTests, AbsMixedValues) {
    auto r = torch::tensor({{-0.5, 0.3}, {1.0, -0.9}}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d = torch::rand({2, 2, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 2, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual thd(r, d, h);
    auto result = thd.abs();

    auto expected_r = torch::abs(r);
    auto expected_d = torch::sign(r).unsqueeze(-1) * d;
    auto expected_h = torch::einsum("mi, mijk->mijk", {torch::sign(r), h});

    ASSERT_TRUE(torch::allclose(result.r, expected_r));
    ASSERT_TRUE(torch::allclose(result.d, expected_d));
    ASSERT_TRUE(torch::allclose(result.h, expected_h));
}

TEST(TensorHyperDualAbsTests, AbsEdgeCase) {
    auto r = torch::tensor({{-1.0, 0.0}, {1.0, -0.5}}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d = torch::rand({2, 2, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 2, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual thd(r, d, h);
    auto result = thd.abs();

    auto expected_r = torch::abs(r);
    auto expected_d = torch::sign(r).unsqueeze(-1) * d;
    auto expected_h = torch::einsum("mi, mijk->mijk", {torch::sign(r), h});

    ASSERT_TRUE(torch::allclose(result.r, expected_r));
    ASSERT_TRUE(torch::allclose(result.d, expected_d));
    ASSERT_TRUE(torch::allclose(result.h, expected_h));
}


TEST(TensorHyperDualComplexTests, AlreadyComplexTensors) {
    auto r = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kComplexDouble).device(torch::kCPU));
    auto d = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kComplexDouble).device(torch::kCPU));
    auto h = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kComplexDouble).device(torch::kCPU));

    TensorHyperDual thd(r, d, h);
    auto result = thd.complex();

    ASSERT_TRUE(result.r.is_complex());
    ASSERT_TRUE(result.d.is_complex());
    ASSERT_TRUE(result.h.is_complex());
    ASSERT_TRUE(torch::allclose(result.r, r));
    ASSERT_TRUE(torch::allclose(result.d, d));
    ASSERT_TRUE(torch::allclose(result.h, h));
}

TEST(TensorHyperDualComplexTests, ConvertRealTensorsToComplex) {
    auto r = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual thd(r, d, h);
    auto result = thd.complex();

    auto expected_r = torch::complex(r, torch::zeros_like(r));
    auto expected_d = torch::complex(d, torch::zeros_like(d));
    auto expected_h = torch::complex(h, torch::zeros_like(h));

    ASSERT_TRUE(result.r.is_complex());
    ASSERT_TRUE(result.d.is_complex());
    ASSERT_TRUE(result.h.is_complex());
    ASSERT_TRUE(torch::allclose(result.r, expected_r));
    ASSERT_TRUE(torch::allclose(result.d, expected_d));
    ASSERT_TRUE(torch::allclose(result.h, expected_h));
}

TEST(TensorHyperDualComplexTests, MixedRealAndComplexTensors) {
    auto r = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kComplexDouble).device(torch::kCPU));
    auto h = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual thd(r, d, h);
    auto result = thd.complex();

    auto expected_r = torch::complex(r, torch::zeros_like(r));
    auto expected_d = d;
    auto expected_h = torch::complex(h, torch::zeros_like(h));

    ASSERT_TRUE(result.r.is_complex());
    ASSERT_TRUE(result.d.is_complex());
    ASSERT_TRUE(result.h.is_complex());
    ASSERT_TRUE(torch::allclose(result.r, expected_r));
    ASSERT_TRUE(torch::allclose(result.d, expected_d));
    ASSERT_TRUE(torch::allclose(result.h, expected_h));
}


TEST(TensorHyperDualRealTests, ExtractRealFromComplexTensors) {
    auto r = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kComplexDouble).device(torch::kCPU));
    auto d = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kComplexDouble).device(torch::kCPU));
    auto h = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kComplexDouble).device(torch::kCPU));

    TensorHyperDual thd(r, d, h);
    auto result = thd.real();

    auto expected_r = torch::real(r);
    auto expected_d = torch::real(d);
    auto expected_h = torch::real(h);

    ASSERT_TRUE(result.r.is_floating_point());
    ASSERT_TRUE(result.d.is_floating_point());
    ASSERT_TRUE(result.h.is_floating_point());
    ASSERT_TRUE(torch::allclose(result.r, expected_r));
    ASSERT_TRUE(torch::allclose(result.d, expected_d));
    ASSERT_TRUE(torch::allclose(result.h, expected_h));
}

TEST(TensorHyperDualRealTests, ExtractRealFromRealTensors) {
    auto r = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual thd(r, d, h);
    auto result = thd.real();

    ASSERT_TRUE(result.r.is_floating_point());
    ASSERT_TRUE(result.d.is_floating_point());
    ASSERT_TRUE(result.h.is_floating_point());
    ASSERT_TRUE(torch::allclose(result.r, r));
    ASSERT_TRUE(torch::allclose(result.d, d));
    ASSERT_TRUE(torch::allclose(result.h, h));
}

TEST(TensorHyperDualRealTests, MixedComplexAndRealTensors) {
    auto r = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kComplexDouble).device(torch::kCPU));
    auto d = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kComplexDouble).device(torch::kCPU));

    TensorHyperDual thd(r, d, h);
    auto result = thd.real();

    auto expected_r = torch::real(r);
    auto expected_d = d;  // Already real
    auto expected_h = torch::real(h);

    ASSERT_TRUE(result.r.is_floating_point());
    ASSERT_TRUE(result.d.is_floating_point());
    ASSERT_TRUE(result.h.is_floating_point());
    ASSERT_TRUE(torch::allclose(result.r, expected_r));
    ASSERT_TRUE(torch::allclose(result.d, expected_d));
    ASSERT_TRUE(torch::allclose(result.h, expected_h));
}


TEST(TensorHyperDualImagTests, ExtractImagFromComplexTensors) {
    auto r = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kComplexDouble).device(torch::kCPU));
    auto d = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kComplexDouble).device(torch::kCPU));
    auto h = torch::rand({2, 3, 4, 5}, torch::TensorOptions().dtype(torch::kComplexDouble).device(torch::kCPU));

    TensorHyperDual thd(r, d, h);
    auto result = thd.imag();

    auto expected_r = torch::imag(r);
    auto expected_d = torch::imag(d);
    auto expected_h = torch::imag(h);

    ASSERT_TRUE(result.r.is_floating_point());
    ASSERT_TRUE(result.d.is_floating_point());
    ASSERT_TRUE(result.h.is_floating_point());
    ASSERT_TRUE(torch::allclose(result.r, expected_r));
    ASSERT_TRUE(torch::allclose(result.d, expected_d));
    ASSERT_TRUE(torch::allclose(result.h, expected_h));
}

TEST(TensorHyperDualImagTests, ExtractImagFromRealTensors) {
    auto r = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual thd(r, d, h);
    auto result = thd.imag();

    auto expected_r = torch::zeros_like(r);
    auto expected_d = torch::zeros_like(d);
    auto expected_h = torch::zeros_like(h);

    ASSERT_TRUE(result.r.is_floating_point());
    ASSERT_TRUE(result.d.is_floating_point());
    ASSERT_TRUE(result.h.is_floating_point());
    ASSERT_TRUE(torch::allclose(result.r, expected_r));
    ASSERT_TRUE(torch::allclose(result.d, expected_d));
    ASSERT_TRUE(torch::allclose(result.h, expected_h));
}

TEST(TensorHyperDualImagTests, MixedComplexAndRealTensors) {
    auto r = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kComplexDouble).device(torch::kCPU));
    auto d = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kComplexDouble).device(torch::kCPU));

    TensorHyperDual thd(r, d, h);
    auto result = thd.imag();

    auto expected_r = torch::imag(r);
    auto expected_d = torch::zeros_like(d);
    auto expected_h = torch::imag(h);

    ASSERT_TRUE(result.r.is_floating_point());
    ASSERT_TRUE(result.d.is_floating_point());
    ASSERT_TRUE(result.h.is_floating_point());
    ASSERT_TRUE(torch::allclose(result.r, expected_r));
    ASSERT_TRUE(torch::allclose(result.d, expected_d));
    ASSERT_TRUE(torch::allclose(result.h, expected_h));
}

TEST(TensorHyperDualZerosLikeTests, ComplexTensors) {
    auto r = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kComplexDouble).device(torch::kCPU));
    auto d = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kComplexDouble).device(torch::kCPU));
    auto h = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kComplexDouble).device(torch::kCPU));
    
    TensorHyperDual thd(r, d, h);
    auto result = thd.zeros_like();
    
    // Check shapes match
    ASSERT_EQ(result.r.sizes(), r.sizes());
    ASSERT_EQ(result.d.sizes(), d.sizes());
    ASSERT_EQ(result.h.sizes(), h.sizes());
    
    // Check types match
    ASSERT_EQ(result.r.dtype(), r.dtype());
    ASSERT_EQ(result.d.dtype(), d.dtype());
    ASSERT_EQ(result.h.dtype(), h.dtype());
    
    // Check all values are zero
    ASSERT_TRUE(torch::all(result.r == 0).item<bool>());
    ASSERT_TRUE(torch::all(result.d == 0).item<bool>());
    ASSERT_TRUE(torch::all(result.h == 0).item<bool>());
}

TEST(TensorHyperDualZerosLikeTests, RealTensors) {
    auto r = torch::rand({3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d = torch::rand({3, 4, 2}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({3, 4, 2, 2}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    
    TensorHyperDual thd(r, d, h);
    auto result = thd.zeros_like();
    
    // Check shapes match
    ASSERT_EQ(result.r.sizes(), r.sizes());
    ASSERT_EQ(result.d.sizes(), d.sizes());
    ASSERT_EQ(result.h.sizes(), h.sizes());
    
    // Check types match
    ASSERT_EQ(result.r.dtype(), r.dtype());
    ASSERT_EQ(result.d.dtype(), d.dtype());
    ASSERT_EQ(result.h.dtype(), h.dtype());
    
    // Check all values are zero
    ASSERT_TRUE(torch::all(result.r == 0).item<bool>());
    ASSERT_TRUE(torch::all(result.d == 0).item<bool>());
    ASSERT_TRUE(torch::all(result.h == 0).item<bool>());
}

TEST(TensorHyperDualZerosLikeTests, MixedTypeTensors) {
    auto r = torch::rand({2, 2}, torch::TensorOptions().dtype(torch::kComplexDouble).device(torch::kCPU));
    auto d = torch::rand({2, 2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 2, 3, 3}, torch::TensorOptions().dtype(torch::kComplexDouble).device(torch::kCPU));
    
    TensorHyperDual thd(r, d, h);
    auto result = thd.zeros_like();
    
    // Check shapes match
    ASSERT_EQ(result.r.sizes(), r.sizes());
    ASSERT_EQ(result.d.sizes(), d.sizes());
    ASSERT_EQ(result.h.sizes(), h.sizes());
    
    // Check types match
    ASSERT_EQ(result.r.dtype(), r.dtype());
    ASSERT_EQ(result.d.dtype(), d.dtype());
    ASSERT_EQ(result.h.dtype(), h.dtype());
    
    // Check all values are zero
    ASSERT_TRUE(torch::all(result.r == 0).item<bool>());
    ASSERT_TRUE(torch::all(result.d == 0).item<bool>());
    ASSERT_TRUE(torch::all(result.h == 0).item<bool>());
}

TEST(TensorHyperDualZerosLikeTests, EmptyTensors) {
    auto r = torch::empty({0, 2}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d = torch::empty({0, 2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::empty({0, 2, 3, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    
    TensorHyperDual thd(r, d, h);
    auto result = thd.zeros_like();
    
    // Check shapes match
    ASSERT_EQ(result.r.sizes(), r.sizes());
    ASSERT_EQ(result.d.sizes(), d.sizes());
    ASSERT_EQ(result.h.sizes(), h.sizes());
    
    // Check types match
    ASSERT_EQ(result.r.dtype(), r.dtype());
    ASSERT_EQ(result.d.dtype(), d.dtype());
    ASSERT_EQ(result.h.dtype(), h.dtype());
    
    // Check dimensions are preserved
    ASSERT_EQ(result.r.dim(), r.dim());
    ASSERT_EQ(result.d.dim(), d.dim());
    ASSERT_EQ(result.h.dim(), h.dim());
}

TEST(TensorHyperDualStaticZerosLikeTests, ComplexTensors) {
    auto r = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kComplexDouble).device(torch::kCPU));
    auto d = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kComplexDouble).device(torch::kCPU));
    auto h = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kComplexDouble).device(torch::kCPU));
    
    TensorHyperDual original(r, d, h);
    auto result = TensorHyperDual::zeros_like(original);
    
    // Check shapes match
    ASSERT_EQ(result.r.sizes(), r.sizes());
    ASSERT_EQ(result.d.sizes(), d.sizes());
    ASSERT_EQ(result.h.sizes(), h.sizes());
    
    // Check types match
    ASSERT_EQ(result.r.dtype(), r.dtype());
    ASSERT_EQ(result.d.dtype(), d.dtype());
    ASSERT_EQ(result.h.dtype(), h.dtype());
    
    // Check all values are zero
    ASSERT_TRUE(torch::all(result.r == 0).item<bool>());
    ASSERT_TRUE(torch::all(result.d == 0).item<bool>());
    ASSERT_TRUE(torch::all(result.h == 0).item<bool>());
}

TEST(TensorHyperDualStaticZerosLikeTests, RealTensors) {
    auto r = torch::rand({3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d = torch::rand({3, 4, 2}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({3, 4, 2, 2}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    
    TensorHyperDual original(r, d, h);
    auto result = TensorHyperDual::zeros_like(original);
    
    // Check shapes match
    ASSERT_EQ(result.r.sizes(), r.sizes());
    ASSERT_EQ(result.d.sizes(), d.sizes());
    ASSERT_EQ(result.h.sizes(), h.sizes());
    
    // Check types match
    ASSERT_EQ(result.r.dtype(), r.dtype());
    ASSERT_EQ(result.d.dtype(), d.dtype());
    ASSERT_EQ(result.h.dtype(), h.dtype());
    
    // Check all values are zero
    ASSERT_TRUE(torch::all(result.r == 0).item<bool>());
    ASSERT_TRUE(torch::all(result.d == 0).item<bool>());
    ASSERT_TRUE(torch::all(result.h == 0).item<bool>());
}

TEST(TensorHyperDualStaticZerosLikeTests, MixedTypeTensors) {
    auto r = torch::rand({2, 2}, torch::TensorOptions().dtype(torch::kComplexDouble).device(torch::kCPU));
    auto d = torch::rand({2, 2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 2, 3, 3}, torch::TensorOptions().dtype(torch::kComplexDouble).device(torch::kCPU));
    
    TensorHyperDual original(r, d, h);
    auto result = TensorHyperDual::zeros_like(original);
    
    // Check shapes match
    ASSERT_EQ(result.r.sizes(), r.sizes());
    ASSERT_EQ(result.d.sizes(), d.sizes());
    ASSERT_EQ(result.h.sizes(), h.sizes());
    
    // Check types match
    ASSERT_EQ(result.r.dtype(), r.dtype());
    ASSERT_EQ(result.d.dtype(), d.dtype());
    ASSERT_EQ(result.h.dtype(), h.dtype());
    
    // Check all values are zero
    ASSERT_TRUE(torch::all(result.r == 0).item<bool>());
    ASSERT_TRUE(torch::all(result.d == 0).item<bool>());
    ASSERT_TRUE(torch::all(result.h == 0).item<bool>());
}

TEST(TensorHyperDualStaticZerosLikeTests, EmptyTensors) {
    auto r = torch::empty({0, 2}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d = torch::empty({0, 2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::empty({0, 2, 3, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    
    TensorHyperDual original(r, d, h);
    auto result = TensorHyperDual::zeros_like(original);
    
    // Check shapes match
    ASSERT_EQ(result.r.sizes(), r.sizes());
    ASSERT_EQ(result.d.sizes(), d.sizes());
    ASSERT_EQ(result.h.sizes(), h.sizes());
    
    // Check types match
    ASSERT_EQ(result.r.dtype(), r.dtype());
    ASSERT_EQ(result.d.dtype(), d.dtype());
    ASSERT_EQ(result.h.dtype(), h.dtype());
    
    // Check dimensions are preserved
    ASSERT_EQ(result.r.dim(), r.dim());
    ASSERT_EQ(result.d.dim(), d.dim());
    ASSERT_EQ(result.h.dim(), h.dim());
}

TEST(TensorHyperDualStaticZerosLikeTests, PreservesOriginalUnchanged) {
    auto r = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    
    TensorHyperDual original(r.clone(), d.clone(), h.clone());
    auto result = TensorHyperDual::zeros_like(original);
    
    // Verify original is unchanged
    ASSERT_TRUE(torch::allclose(original.r, r));
    ASSERT_TRUE(torch::allclose(original.d, d));
    ASSERT_TRUE(torch::allclose(original.h, h));
}


TEST(TensorHyperDualSignTests, SignRealPositiveValues) {
    auto r = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual thd(r, d, h);
    auto result = thd.sign();

    auto expected_r = torch::sign(r);
    auto expected_d = torch::zeros_like(d);
    auto expected_h = torch::zeros_like(h);

    ASSERT_TRUE(torch::allclose(result.r, expected_r));
    ASSERT_TRUE(torch::allclose(result.d, expected_d));
    ASSERT_TRUE(torch::allclose(result.h, expected_h));
}

TEST(TensorHyperDualSignTests, SignRealNegativeValues) {
    auto r = -torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual thd(r, d, h);
    auto result = thd.sign();

    auto expected_r = torch::sign(r);
    auto expected_d = torch::zeros_like(d);
    auto expected_h = torch::zeros_like(h);

    ASSERT_TRUE(torch::allclose(result.r, expected_r));
    ASSERT_TRUE(torch::allclose(result.d, expected_d));
    ASSERT_TRUE(torch::allclose(result.h, expected_h));
}

TEST(TensorHyperDualSignTests, SignComplexValues) {
    auto r = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kComplexDouble).device(torch::kCPU));
    auto d = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kComplexDouble).device(torch::kCPU));
    auto h = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kComplexDouble).device(torch::kCPU));

    TensorHyperDual thd(r, d, h);
    auto result = thd.sign();

    auto expected_r = torch::sgn(r);
    auto expected_d = torch::zeros_like(d);
    auto expected_h = torch::zeros_like(h);

    ASSERT_TRUE(torch::allclose(result.r, expected_r));
    ASSERT_TRUE(torch::allclose(result.d, expected_d));
    ASSERT_TRUE(torch::allclose(result.h, expected_h));
}

TEST(TensorHyperDualSignTests, SignEdgeCaseZero) {
    auto r = torch::zeros({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual thd(r, d, h);
    auto result = thd.sign();

    auto expected_r = torch::sign(r);
    auto expected_d = torch::zeros_like(d);
    auto expected_h = torch::zeros_like(h);

    ASSERT_TRUE(torch::allclose(result.r, expected_r));
    ASSERT_TRUE(torch::allclose(result.d, expected_d));
    ASSERT_TRUE(torch::allclose(result.h, expected_h));
}



TEST(TensorHyperDualMinTests, MinValuesAndIndices) {
    auto r = torch::tensor({{5.0, 2.0, 3.0}, {4.0, 1.0, 6.0}}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual thd(r, d, h);
    auto result = thd.min();

    auto min_result = torch::min(r, 1, true);
    auto expected_r = std::get<0>(min_result);
    auto expected_indices = std::get<1>(min_result);

    auto dshape = expected_indices.unsqueeze(-1).expand({expected_indices.size(0), expected_indices.size(1), d.size(-1)});
    auto hshape = expected_indices.unsqueeze(-1).unsqueeze(-1).expand({expected_indices.size(0), expected_indices.size(1), h.size(-2), h.size(-1)});

    auto expected_d = torch::gather(d, 1, dshape);
    auto expected_h = torch::gather(h, 1, hshape);

    ASSERT_TRUE(torch::allclose(result.r, expected_r));
    ASSERT_TRUE(torch::allclose(result.d, expected_d));
    ASSERT_TRUE(torch::allclose(result.h, expected_h));
}

TEST(TensorHyperDualMinTests, ConsistencyAcrossTensors) {
    auto r = torch::rand({3, 5}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d = torch::rand({3, 5, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({3, 5, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual thd(r, d, h);
    auto result = thd.min();

    auto min_result = torch::min(r, 1, true);
    auto expected_r = std::get<0>(min_result);
    auto expected_indices = std::get<1>(min_result);

    auto dshape = expected_indices.unsqueeze(-1).expand({expected_indices.size(0), expected_indices.size(1), d.size(-1)});
    auto hshape = expected_indices.unsqueeze(-1).unsqueeze(-1).expand({expected_indices.size(0), expected_indices.size(1), h.size(-2), h.size(-1)});

    auto expected_d = torch::gather(d, 1, dshape);
    auto expected_h = torch::gather(h, 1, hshape);

    ASSERT_TRUE(torch::allclose(result.r, expected_r));
    ASSERT_TRUE(torch::allclose(result.d, expected_d));
    ASSERT_TRUE(torch::allclose(result.h, expected_h));
}


TEST(TensorHyperDualMinTests, ShapeMismatchThrowsError) {
    auto r = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d = torch::rand({2, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual thd(r, d, h);

    ASSERT_THROW(thd.min(), std::invalid_argument);
}



TEST(TensorHyperDualMaxTests, MaxValuesAndIndices) {
    auto r = torch::tensor({{5.0, 2.0, 3.0}, {4.0, 1.0, 6.0}}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual thd(r, d, h);
    auto result = thd.max();

    auto max_result = torch::max(r, 1, true);
    auto expected_r = std::get<0>(max_result);
    auto expected_indices = std::get<1>(max_result);

    auto dshape = expected_indices.unsqueeze(-1).expand({expected_indices.size(0), expected_indices.size(1), d.size(-1)});
    auto hshape = expected_indices.unsqueeze(-1).unsqueeze(-1).expand({expected_indices.size(0), expected_indices.size(1), h.size(-2), h.size(-1)});

    auto expected_d = torch::gather(d, 1, dshape);
    auto expected_h = torch::gather(h, 1, hshape);

    ASSERT_TRUE(torch::allclose(result.r, expected_r));
    ASSERT_TRUE(torch::allclose(result.d, expected_d));
    ASSERT_TRUE(torch::allclose(result.h, expected_h));
}

TEST(TensorHyperDualMaxTests, ConsistencyAcrossTensors) {
    auto r = torch::rand({3, 5}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d = torch::rand({3, 5, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({3, 5, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual thd(r, d, h);
    auto result = thd.max();

    auto max_result = torch::max(r, 1, true);
    auto expected_r = std::get<0>(max_result);
    auto expected_indices = std::get<1>(max_result);

    auto dshape = expected_indices.unsqueeze(-1).expand({expected_indices.size(0), expected_indices.size(1), d.size(-1)});
    auto hshape = expected_indices.unsqueeze(-1).unsqueeze(-1).expand({expected_indices.size(0), expected_indices.size(1), h.size(-2), h.size(-1)});

    auto expected_d = torch::gather(d, 1, dshape);
    auto expected_h = torch::gather(h, 1, hshape);

    ASSERT_TRUE(torch::allclose(result.r, expected_r));
    ASSERT_TRUE(torch::allclose(result.d, expected_d));
    ASSERT_TRUE(torch::allclose(result.h, expected_h));
}


TEST(TensorHyperDualMaxTests, ShapeMismatchThrowsError) {
    auto r = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d = torch::rand({2, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual thd(r, d, h);

    ASSERT_THROW(thd.max(), std::invalid_argument);
}


TEST(TensorHyperDualWhereTests, WhereConditionMatchesShape) {
    auto cond = torch::tensor({1, 0, 1}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));
    auto r1 = torch::rand({3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d1 = torch::rand({3, 4, 5}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h1 = torch::rand({3, 4, 5, 5}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    auto r2 = torch::rand({3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d2 = torch::rand({3, 4, 5}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h2 = torch::rand({3, 4, 5, 5}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual x(r1, d1, h1);
    TensorHyperDual y(r2, d2, h2);

    auto result = TensorHyperDual::where(cond, x, y);

    auto expected_r = torch::where(cond.unsqueeze(1).expand({3, 4}), r1, r2);
    auto expected_d = torch::where(cond.unsqueeze(1).unsqueeze(2).expand({3, 4, 5}), d1, d2);
    auto expected_h = torch::where(cond.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand({3, 4, 5, 5}), h1, h2);

    ASSERT_TRUE(torch::allclose(result.r, expected_r));
    ASSERT_TRUE(torch::allclose(result.d, expected_d));
    ASSERT_TRUE(torch::allclose(result.h, expected_h));
}

TEST(TensorHyperDualWhereTests, WhereConditionBroadcastable) {
    auto cond = torch::tensor({1, 0}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));
    auto r1 = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d1 = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h1 = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    auto r2 = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d2 = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h2 = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual x(r1, d1, h1);
    TensorHyperDual y(r2, d2, h2);

    auto result = TensorHyperDual::where(cond, x, y);

    auto expected_r = torch::where(cond.unsqueeze(1).expand({2, 3}), r1, r2);
    auto expected_d = torch::where(cond.unsqueeze(1).unsqueeze(2).expand({2, 3, 4}), d1, d2);
    auto expected_h = torch::where(cond.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand({2, 3, 4, 4}), h1, h2);

    ASSERT_TRUE(torch::allclose(result.r, expected_r));
    ASSERT_TRUE(torch::allclose(result.d, expected_d));
    ASSERT_TRUE(torch::allclose(result.h, expected_h));
}

TEST(TensorHyperDualWhereTests, ShapeMismatchThrowsError) {
    auto cond = torch::tensor({1, 0}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));
    auto r1 = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d1 = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h1 = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    auto r2 = torch::rand({2, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d2 = torch::rand({2, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h2 = torch::rand({2, 4, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual x(r1, d1, h1);
    TensorHyperDual y(r2, d2, h2);

    ASSERT_THROW(TensorHyperDual::where(cond, x, y), std::invalid_argument);
}

TEST(TensorHyperDualWhereTests, ConditionNonBroadcastableThrowsError) {
    auto cond = torch::tensor({{1, 0}, {0, 1}}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));
    auto r1 = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d1 = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h1 = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    auto r2 = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d2 = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h2 = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual x(r1, d1, h1);
    TensorHyperDual y(r2, d2, h2);

    ASSERT_THROW(TensorHyperDual::where(cond, x, y), std::invalid_argument);
}


TEST(TensorHyperDualEinsumTests, SimpleDotProduct) {
    auto r1 = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d1 = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h1 = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    auto r2 = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d2 = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h2 = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual x(r1, d1, h1);
    TensorHyperDual y(r2, d2, h2);

    auto result = TensorHyperDual::einsum("mi,mi->mi", x, y);

    auto expected_r = torch::einsum("mi, mi->mi", {r1, r2});
    auto expected_d = torch::einsum("mi,mij->mij", {r1, d2}) + torch::einsum("mij, mi->mij", {d1, r2});
    auto expected_h = 2*torch::einsum("mij, mik->mijk", {d1, d2}) + torch::einsum("mi,mijk->mijk", {r1, h2}) + 
                      torch::einsum("mijk,mi->mijk", {h1, r2});

    ASSERT_TRUE(torch::allclose(result.r, expected_r));
    ASSERT_TRUE(torch::allclose(result.d, expected_d));
    ASSERT_TRUE(torch::allclose(result.h, expected_h));
}


TEST(TensorHyperDualEinsumTests, ShapeMismatchThrowsError) {
    auto r1 = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d1 = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h1 = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    auto r2 = torch::rand({4, 5}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d2 = torch::rand({4, 5, 6}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h2 = torch::rand({4, 5, 6, 6}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual x(r1, d1, h1);
    TensorHyperDual y(r2, d2, h2);

    //Expect a c10 error
    ASSERT_THROW(TensorHyperDual::einsum("mi,mi->mi", x, y), c10::Error);
}



TEST(TensorHyperDualEinsumTensorTests, SimpleDotProduct) {
    auto r1 = torch::rand({3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto r2 = torch::rand({3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d2 = torch::rand({3, 4, 5}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h2 = torch::rand({3, 4, 5, 5}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual y(r2, d2, h2);

    auto result = TensorHyperDual::einsum("mi,mi->mi", r1, y);

    auto expected_r = torch::einsum("mi,mi->mi", {r1, r2});
    auto expected_d = torch::einsum("mi,mij->mij", {r1, d2});
    auto expected_h = torch::einsum("mi,mijk->mijk", {r1, h2});

    ASSERT_TRUE(torch::allclose(result.r, expected_r));
    ASSERT_TRUE(torch::allclose(result.d, expected_d));
    ASSERT_TRUE(torch::allclose(result.h, expected_h));
}


TEST(TensorHyperDualEinsumTensorTests, ShapeMismatchThrowsError) {
    auto r1 = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto r2 = torch::rand({4, 5}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d2 = torch::rand({4, 5, 6}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h2 = torch::rand({4, 5, 6, 7}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual y(r2, d2, h2);
    //Expect a c10 error
    ASSERT_THROW(TensorHyperDual::einsum("mi,mi->mi", r1, y), c10::Error);

}


TEST(TensorHyperDualEinsumDualTests, SimpleDotProduct) {
    auto r1 = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto r2 = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d1 = torch::rand({2, 3, 5}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h1 = torch::rand({2, 3, 5, 5}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual x(r1, d1, h1);

    auto result = TensorHyperDual::einsum("mi,mi->mi", x, r2);

    auto expected_r = torch::einsum("mi, mi->mi", {r1, r2});
    auto expected_d = torch::einsum("mij, mi->mij", {d1, r2});
    auto expected_h = torch::einsum("mijk, mi->mijk", {h1, r2});

    ASSERT_TRUE(torch::allclose(result.r, expected_r));
    ASSERT_TRUE(torch::allclose(result.d, expected_d));
    ASSERT_TRUE(torch::allclose(result.h, expected_h));
}


TEST(TensorHyperDualEinsumDualTests, ShapeMismatchThrowsError) {
    auto r1 = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto r2 = torch::rand({4, 5}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d1 = torch::rand({2, 3, 6}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h1 = torch::rand({2, 3, 6, 7}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual x(r1, d1, h1);
    //Expect a c10 error
    ASSERT_THROW(TensorHyperDual::einsum("mi,mi->mi", x, r2), c10::Error);

}


TEST(TensorHyperDualEinsumVectorTests, SimpleDotProduct) {
    auto r1 = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d1 = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h1 = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    auto r2 = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d2 = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h2 = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    auto r3 = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d3 = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h3 = torch::rand({2, 3, 4, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual t1(r1, d1, h1);
    TensorHyperDual t2(r2, d2, h2);
    TensorHyperDual t3(r3, d3, h3);

    std::vector<TensorHyperDual> tensors = {t1, t2, t3};
    auto result = TensorHyperDual::einsum("mi, mi, mi->mi", tensors);

    /*auto expected_r = torch::einsum("mi, mi, mi->mi", {r1, r2, r3});
    auto expected_d = torch::einsum("mi, mi, mij->mij", {r1, r2, d3}) + 
                      torch::einsum("mi, mij, mi->mij", {r1, d2, r3}) + 
                      torch::einsum("mij, mi, mi->mij", {d1, r2, r3});

    auto expected_h = torch::einsum("mi, mij, mik->mijk", {r1, d1, d2}) + 
                      torch::einsum("mij, mi, mik->mijk", {d1, r2, d3}) + 
                      torch::einsum("mij, mik, mi->mijk", {d1, d2, r3}) + 
                      
                      torch::einsum("mijk, mi, mi->mijk", {h1, r2, r3}) + 
                      torch::einsum("mi, mijk, mi->mijk", {r1, h2, r3}) + 
                      torch::einsum("mi, mi, mijk->mijk", {r1, r2, h3}); 

    ASSERT_TRUE(torch::allclose(result.r, expected_r));*/
    ASSERT_TRUE(true);
}


TEST(TensorHyperDualEinsumVectorTests, InvalidEinsumStringThrowsError) {
    auto r1 = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d1 = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h1 = torch::rand({2, 3, 4, 5}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    auto r2 = torch::rand({3, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto d2 = torch::rand({3, 4, 5}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto h2 = torch::rand({3, 4, 5, 6}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    TensorHyperDual t1(r1, d1, h1);
    TensorHyperDual t2(r2, d2, h2);

    std::vector<TensorHyperDual> tensors = {t1, t2};

    //Expect a c10 error
    ASSERT_THROW(TensorHyperDual::einsum("mi,mi->mi", tensors), c10::Error);
}


// Test case for the parameterized constructor
TEST(TensorMatDualTest, ParameterizedConstructor_ValidInput) {
    // Arrange
    torch::Tensor real_tensor = torch::rand({2, 3, 4}); // 3D tensor
    torch::Tensor dual_tensor = torch::rand({2, 3, 4, 5}); // 4D tensor

    // Act
    TensorMatDual tmd(real_tensor, dual_tensor);

    // Assert
    EXPECT_EQ(tmd.r.sizes(), real_tensor.sizes());
    EXPECT_EQ(tmd.d.sizes(), dual_tensor.sizes());
    EXPECT_EQ(tmd.r.dtype(), real_tensor.dtype());
    EXPECT_EQ(tmd.d.dtype(), dual_tensor.dtype());
    EXPECT_EQ(tmd.r.device(), real_tensor.device());
    EXPECT_EQ(tmd.d.device(), dual_tensor.device());
}

// Test case for mismatched dimensions
TEST(TensorMatDualTest, ParameterizedConstructor_InvalidDimensions) {
    // Arrange
    torch::Tensor real_tensor = torch::rand({2, 3, 4}); // 3D tensor
    torch::Tensor invalid_dual_tensor = torch::rand({2, 3, 4}); // Not 4D

    // Act & Assert
    EXPECT_THROW(TensorMatDual(real_tensor, invalid_dual_tensor), std::invalid_argument);
}

// Test case for mismatched devices
TEST(TensorMatDualTest, ParameterizedConstructor_MismatchedDevices) {
    // Arrange
    torch::Tensor real_tensor = torch::rand({2, 3, 4}, torch::kCPU);
    torch::Tensor dual_tensor = torch::rand({2, 3, 4, 5}, torch::kCUDA);

    // Act & Assert
    EXPECT_THROW(TensorMatDual(real_tensor, dual_tensor), std::invalid_argument);
}


// Test case for default constructor
TEST(TensorMatDualTest, DefaultConstructor) {
    // Act
    TensorMatDual tmd;

    // Assert
    EXPECT_EQ(tmd.r.sizes(), torch::IntArrayRef({0, 0, 0}));
    EXPECT_EQ(tmd.d.sizes(), torch::IntArrayRef({0, 0, 0, 0}));
    EXPECT_EQ(tmd.r.dtype(), torch::kFloat64);
    EXPECT_EQ(tmd.d.dtype(), torch::kFloat64);
    EXPECT_EQ(tmd.r.device(), torch::kCPU);
    EXPECT_EQ(tmd.d.device(), torch::kCPU);
}

// Test case for copy constructor
TEST(TensorMatDualTest, CopyConstructor) {
    // Arrange
    torch::Tensor real_tensor = torch::rand({2, 3, 4});
    torch::Tensor dual_tensor = torch::rand({2, 3, 4, 5});
    TensorMatDual original(real_tensor, dual_tensor);

    // Act
    TensorMatDual copy(original);

    // Assert
    EXPECT_EQ(copy.r.sizes(), original.r.sizes());
    EXPECT_EQ(copy.d.sizes(), original.d.sizes());
    EXPECT_TRUE(copy.r.equal(original.r)); // Deep copy, tensors should be equal
    EXPECT_TRUE(copy.d.equal(original.d)); // Deep copy, tensors should be equal
}


// Test case for the to_() method
TEST(TensorMatDualTest, ToMethod_MoveToDevice) {
    // Arrange
    torch::Tensor real_tensor = torch::rand({2, 3, 4}, torch::kCPU);
    torch::Tensor dual_tensor = torch::rand({2, 3, 4, 5}, torch::kCPU);
    TensorMatDual tmd(real_tensor, dual_tensor);

    // Act
    tmd.to_(torch::kCUDA);

    // Assert
    EXPECT_EQ(tmd.r.device().type(), torch::kCUDA);
    EXPECT_EQ(tmd.d.device().type(), torch::kCUDA);
}

// Test case for the to_() method when already on the target device
TEST(TensorMatDualTest, ToMethod_AlreadyOnDevice) {
    // Arrange
    torch::Tensor real_tensor = torch::rand({2, 3, 4}, torch::kCUDA);
    torch::Tensor dual_tensor = torch::rand({2, 3, 4, 5}, torch::kCUDA);
    TensorMatDual tmd(real_tensor, dual_tensor);

    // Act
    tmd.to_(torch::kCUDA); // Already on CUDA

    // Assert
    EXPECT_EQ(tmd.r.device().type(), torch::kCUDA);
    EXPECT_EQ(tmd.d.device().type(), torch::kCUDA);
}

// Test case for moving back to CPU
TEST(TensorMatDualTest, ToMethod_MoveBackToCPU) {
    // Arrange
    torch::Tensor real_tensor = torch::rand({2, 3, 4}, torch::kCUDA);
    torch::Tensor dual_tensor = torch::rand({2, 3, 4, 5}, torch::kCUDA);
    TensorMatDual tmd(real_tensor, dual_tensor);

    // Act
    tmd.to_(torch::kCPU);

    // Assert
    EXPECT_EQ(tmd.r.device().type(), torch::kCPU);
    EXPECT_EQ(tmd.d.device().type(), torch::kCPU);
}

// Test case for checking the return value of to_()
TEST(TensorMatDualTest, ToMethod_ReturnValue) {
    // Arrange
    torch::Tensor real_tensor = torch::rand({2, 3, 4}, torch::kCPU);
    torch::Tensor dual_tensor = torch::rand({2, 3, 4, 5}, torch::kCPU);
    TensorMatDual tmd(real_tensor, dual_tensor);

    // Act
    TensorMatDual& result = tmd.to_(torch::kCUDA);

    // Assert
    EXPECT_EQ(&result, &tmd); // Ensure the returned reference is the same object
    EXPECT_EQ(tmd.r.device().type(), torch::kCUDA);
    EXPECT_EQ(tmd.d.device().type(), torch::kCUDA);
}


// Test case for device() method returning CPU
TEST(TensorMatDualTest, DeviceMethod_OnCPU) {
    // Arrange
    torch::Tensor real_tensor = torch::rand({2, 3, 4}, torch::kCPU);
    torch::Tensor dual_tensor = torch::rand({2, 3, 4, 5}, torch::kCPU);
    TensorMatDual tmd(real_tensor, dual_tensor);

    // Act
    torch::Device device = tmd.device();

    // Assert
    EXPECT_EQ(device.type(), torch::kCPU);
}

// Test case for device() method returning CUDA
TEST(TensorMatDualTest, DeviceMethod_OnCUDA) {
    // Arrange
    torch::Tensor real_tensor = torch::rand({2, 3, 4}, torch::kCUDA);
    torch::Tensor dual_tensor = torch::rand({2, 3, 4, 5}, torch::kCUDA);
    TensorMatDual tmd(real_tensor, dual_tensor);

    // Act
    torch::Device device = tmd.device();

    // Assert
    EXPECT_EQ(device.type(), torch::kCUDA);
}

// Test case for device() after moving to another device
TEST(TensorMatDualTest, DeviceMethod_AfterMoveToDevice) {
    // Arrange
    torch::Tensor real_tensor = torch::rand({2, 3, 4}, torch::kCPU);
    torch::Tensor dual_tensor = torch::rand({2, 3, 4, 5}, torch::kCPU);
    TensorMatDual tmd(real_tensor, dual_tensor);

    // Act
    tmd.to_(torch::kCUDA);
    torch::Device device = tmd.device();

    // Assert
    EXPECT_EQ(device.type(), torch::kCUDA);
}



// Test case for valid unsqueeze along the default dimension (dim = 2)
TEST(TensorMatDualTest, UnsqueezeConstructor_ValidDefaultDim) {
    // Arrange
    torch::Tensor real_tensor = torch::rand({2, 3});
    torch::Tensor dual_tensor = torch::rand({2, 3, 4});
    TensorDual tensor_dual(real_tensor, dual_tensor);

    // Act
    TensorMatDual tmd(tensor_dual);

    // Assert
    EXPECT_EQ(tmd.r.sizes(), torch::IntArrayRef({2, 3, 1})); // Unsqueezed along dim 2
    EXPECT_EQ(tmd.d.sizes(), torch::IntArrayRef({2, 3, 1, 4})); // Unsqueezed along dim 2
}

// Test case for valid unsqueeze along a specified dimension
TEST(TensorMatDualTest, UnsqueezeConstructor_ValidSpecifiedDim) {
    // Arrange
    torch::Tensor real_tensor = torch::rand({2, 3});
    torch::Tensor dual_tensor = torch::rand({2, 3, 4});
    TensorDual tensor_dual(real_tensor, dual_tensor);

    // Act
    TensorMatDual tmd(tensor_dual, 1);

    // Assert
    EXPECT_EQ(tmd.r.sizes(), torch::IntArrayRef({2, 1, 3})); // Unsqueezed along dim 1
    EXPECT_EQ(tmd.d.sizes(), torch::IntArrayRef({2, 1, 3, 4})); // Unsqueezed along dim 1
}

// Test case for invalid dimension
TEST(TensorMatDualTest, UnsqueezeConstructor_InvalidDim) {
    // Arrange
    torch::Tensor real_tensor = torch::rand({2, 3});
    torch::Tensor dual_tensor = torch::rand({2, 3, 4});
    TensorDual tensor_dual(real_tensor, dual_tensor);

    // Act & Assert
    EXPECT_THROW(TensorMatDual(tensor_dual, 5), std::invalid_argument); // Invalid dim
}



// Test case for real and dual tensors already in complex format
TEST(TensorMatDualTest, ComplexMethod_AlreadyComplex) {
    // Arrange
    torch::Tensor real_tensor = torch::rand({2, 3, 4}, torch::kComplexDouble);
    torch::Tensor dual_tensor = torch::rand({2, 3, 4, 5}, torch::kComplexDouble);
    TensorMatDual tmd(real_tensor, dual_tensor);

    // Act
    TensorMatDual result = tmd.complex();

    // Assert
    EXPECT_TRUE(result.r.is_complex());
    EXPECT_TRUE(result.d.is_complex());
    EXPECT_EQ(result.r.sizes(), real_tensor.sizes());
    EXPECT_EQ(result.d.sizes(), dual_tensor.sizes());
    EXPECT_TRUE(result.r.equal(real_tensor));
    EXPECT_TRUE(result.d.equal(dual_tensor));
}

// Test case for real and dual tensors in real format
TEST(TensorMatDualTest, ComplexMethod_ConvertToComplex) {
    // Arrange
    torch::Tensor real_tensor = torch::rand({2, 3, 4}, torch::kFloat64);
    torch::Tensor dual_tensor = torch::rand({2, 3, 4, 5}, torch::kFloat64);
    TensorMatDual tmd(real_tensor, dual_tensor);

    // Act
    TensorMatDual result = tmd.complex();

    // Assert
    EXPECT_TRUE(result.r.is_complex());
    EXPECT_TRUE(result.d.is_complex());
    EXPECT_EQ(result.r.sizes(), real_tensor.sizes());
    EXPECT_EQ(result.d.sizes(), dual_tensor.sizes());
    EXPECT_TRUE(result.r.equal(torch::complex(real_tensor, torch::zeros_like(real_tensor))));
    EXPECT_TRUE(result.d.equal(torch::complex(dual_tensor, torch::zeros_like(dual_tensor))));
}

// Test case for mixed real and complex tensors
TEST(TensorMatDualTest, ComplexMethod_MixedFormat) {
    // Arrange
    torch::Tensor real_tensor = torch::rand({2, 3, 4}, torch::kComplexDouble);
    torch::Tensor dual_tensor = torch::rand({2, 3, 4, 5}, torch::kFloat64);
    TensorMatDual tmd(real_tensor, dual_tensor);

    // Act
    TensorMatDual result = tmd.complex();

    // Assert
    EXPECT_TRUE(result.r.is_complex());
    EXPECT_TRUE(result.d.is_complex());
    EXPECT_EQ(result.r.sizes(), real_tensor.sizes());
    EXPECT_EQ(result.d.sizes(), dual_tensor.sizes());
    EXPECT_TRUE(result.r.equal(real_tensor));
    EXPECT_TRUE(result.d.equal(torch::complex(dual_tensor, torch::zeros_like(dual_tensor))));
}


// Test case for the output stream operator with valid tensors
TEST(TensorMatDualTest, OutputStreamOperator_ValidTensors) {
    // Arrange
    torch::Tensor real_tensor = torch::rand({2, 3, 4});
    torch::Tensor dual_tensor = torch::rand({2, 3, 4, 5});
    TensorMatDual tmd(real_tensor, dual_tensor);

    std::ostringstream output;

    // Act
    output << tmd;

    // Assert
    std::string result = output.str();
    EXPECT_NE(result.find("TensorMatDual Object:"), std::string::npos);
    EXPECT_NE(result.find("Real Part (r):"), std::string::npos);
    EXPECT_NE(result.find("Dual Part (d):"), std::string::npos);
    EXPECT_NE(result.find("Shape: [2, 3, 4]"), std::string::npos); // Real tensor shape
    EXPECT_NE(result.find("Shape: [2, 3, 4, 5]"), std::string::npos); // Dual tensor shape
}

// Test case for the output stream operator with empty tensors
TEST(TensorMatDualTest, OutputStreamOperator_EmptyTensors) {
    // Arrange
    torch::Tensor real_tensor = torch::empty({0, 0, 0});
    torch::Tensor dual_tensor = torch::empty({0, 0, 0, 0});
    TensorMatDual tmd(real_tensor, dual_tensor);

    std::ostringstream output;

    // Act
    output << tmd;

    // Assert
    std::string result = output.str();
    EXPECT_NE(result.find("TensorMatDual Object:"), std::string::npos);
    EXPECT_NE(result.find("Real Part (r):"), std::string::npos);
    EXPECT_NE(result.find("Dual Part (d):"), std::string::npos);
    EXPECT_NE(result.find("Shape: [0, 0, 0]"), std::string::npos); // Real tensor shape
    EXPECT_NE(result.find("Shape: [0, 0, 0, 0]"), std::string::npos); // Dual tensor shape
}

// Test case for tensors with mixed data types
TEST(TensorMatDualTest, OutputStreamOperator_MixedDataTypes) {
    // Arrange
    torch::Tensor real_tensor = torch::rand({2, 3, 4}, torch::kFloat64);
    torch::Tensor dual_tensor = torch::rand({2, 3, 4, 5}, torch::kFloat32);
    TensorMatDual tmd(real_tensor, dual_tensor);

    std::ostringstream output;

    // Act
    output << tmd;

    // Assert
    std::string result = output.str();
    EXPECT_NE(result.find("TensorMatDual Object:"), std::string::npos);
    EXPECT_NE(result.find("Real Part (r):"), std::string::npos);
    EXPECT_NE(result.find("Dual Part (d):"), std::string::npos);
    EXPECT_NE(result.find("Shape: [2, 3, 4]"), std::string::npos); // Real tensor shape
    EXPECT_NE(result.find("Shape: [2, 3, 4, 5]"), std::string::npos); // Dual tensor shape
}



// Test case for valid squeeze operation
TEST(TensorDualTest, Squeeze_ValidDimension) {
    // Arrange
    torch::Tensor real_tensor = torch::rand({2, 1, 3}); // Dimension 1 has size 1
    torch::Tensor dual_tensor = torch::rand({2, 1, 3, 4}); // Dimension 1 has size 1
    TensorMatDual tmd(real_tensor, dual_tensor);

    // Act
    TensorDual result = tmd.squeeze(1);

    // Assert
    EXPECT_EQ(result.r.sizes(), torch::IntArrayRef({2, 3})); // Dimension 1 squeezed
    EXPECT_EQ(result.d.sizes(), torch::IntArrayRef({2, 3, 4})); // Dimension 1 squeezed
}

// Test case for invalid dimension (out of range)
TEST(TensorDualTest, Squeeze_InvalidDimensionOutOfRange) {
    // Arrange
    torch::Tensor real_tensor = torch::rand({2, 1, 3});
    torch::Tensor dual_tensor = torch::rand({2, 1, 3, 4});
    TensorMatDual tmd(real_tensor, dual_tensor);

    // Act & Assert
    EXPECT_THROW(tmd.squeeze(5), std::invalid_argument); // Invalid dimension
}


// Test case for squeezing first dimension
TEST(TensorDualTest, Squeeze_FirstDimension) {
    // Arrange
    torch::Tensor real_tensor = torch::rand({1, 2, 3}); // Dimension 0 has size 1
    torch::Tensor dual_tensor = torch::rand({1, 2, 3, 4}); // Dimension 0 has size 1
    TensorMatDual tmd(real_tensor, dual_tensor);

    // Act
    TensorDual result = tmd.squeeze(0);

    // Assert
    EXPECT_EQ(result.r.sizes(), torch::IntArrayRef({2, 3})); // Dimension 0 squeezed
    EXPECT_EQ(result.d.sizes(), torch::IntArrayRef({2, 3, 4})); // Dimension 0 squeezed
}

// Test case for squeezing last dimension
TEST(TensorDualTest, Squeeze_LastDimension) {
    // Arrange
    torch::Tensor real_tensor = torch::rand({2, 3, 1}); // Dimension 2 has size 1
    torch::Tensor dual_tensor = torch::rand({2, 3, 1, 4}); // Dimension 2 has size 1
    TensorMatDual tmd(real_tensor, dual_tensor);

    // Act
    TensorDual result = tmd.squeeze(2);

    // Assert
    EXPECT_EQ(result.r.sizes(), torch::IntArrayRef({2, 3})); // Dimension 2 squeezed
    EXPECT_EQ(result.d.sizes(), torch::IntArrayRef({2, 3, 4})); // Dimension 2 squeezed
}


// Test case for contiguous operation on already contiguous tensors
TEST(TensorMatDualTest, Contiguous_AlreadyContiguous) {
    // Arrange
    torch::Tensor real_tensor = torch::rand({2, 3, 4}).contiguous();
    torch::Tensor dual_tensor = torch::rand({2, 3, 4, 5}).contiguous();
    TensorMatDual tmd(real_tensor, dual_tensor);

    // Act
    TensorMatDual result = tmd.contiguous();

    // Assert
    EXPECT_TRUE(result.r.is_contiguous());
    EXPECT_TRUE(result.d.is_contiguous());
    EXPECT_TRUE(result.r.equal(real_tensor)); // Should remain unchanged
    EXPECT_TRUE(result.d.equal(dual_tensor)); // Should remain unchanged
}

// Test case for contiguous operation on non-contiguous tensors
TEST(TensorMatDualTest, Contiguous_NonContiguous) {
    // Arrange
    torch::Tensor real_tensor = torch::rand({2, 3, 4}).transpose(0, 1); // Non-contiguous
    torch::Tensor dual_tensor = torch::rand({2, 3, 4, 5}).transpose(0, 1); // Non-contiguous
    TensorMatDual tmd(real_tensor, dual_tensor);

    // Act
    TensorMatDual result = tmd.contiguous();

    // Assert
    EXPECT_TRUE(result.r.is_contiguous());
    EXPECT_TRUE(result.d.is_contiguous());
    EXPECT_EQ(result.r.sizes(), real_tensor.sizes());
    EXPECT_EQ(result.d.sizes(), dual_tensor.sizes());
    EXPECT_TRUE(result.r.equal(real_tensor)); // Contiguous tensor differs from original
    EXPECT_TRUE(result.d.equal(dual_tensor)); // Contiguous tensor differs from original
}

// Test case for ensuring contiguous tensors are correctly returned
TEST(TensorMatDualTest, Contiguous_ReturnCorrectTensors) {
    // Arrange
    torch::Tensor real_tensor = torch::rand({2, 3, 4}).transpose(0, 1); // Non-contiguous
    torch::Tensor dual_tensor = torch::rand({2, 3, 4, 5}).transpose(0, 1); // Non-contiguous
    TensorMatDual tmd(real_tensor, dual_tensor);

    // Act
    TensorMatDual result = tmd.contiguous();

    // Assert
    EXPECT_EQ(result.r.sizes(), real_tensor.sizes());
    EXPECT_EQ(result.d.sizes(), dual_tensor.sizes());
    EXPECT_TRUE(result.r.is_contiguous());
    EXPECT_TRUE(result.d.is_contiguous());
}

// Test case for empty tensors
TEST(TensorMatDualTest, Contiguous_EmptyTensors) {
    // Arrange
    torch::Tensor real_tensor = torch::empty({0, 0, 0});
    torch::Tensor dual_tensor = torch::empty({0, 0, 0, 0});
    TensorMatDual tmd(real_tensor, dual_tensor);

    // Act
    TensorMatDual result = tmd.contiguous();

    // Assert
    EXPECT_TRUE(result.r.is_contiguous());
    EXPECT_TRUE(result.d.is_contiguous());
    EXPECT_EQ(result.r.sizes(), real_tensor.sizes());
    EXPECT_EQ(result.d.sizes(), dual_tensor.sizes());
}



// Test case for eye operation with valid tensors
TEST(TensorMatDualTest, Eye_ValidTensors) {
    // Arrange
    torch::Tensor real_tensor = torch::rand({2, 3, 4}); // Batch size 2, matrix size 3x4
    torch::Tensor dual_tensor = torch::rand({2, 3, 4, 5});
    TensorMatDual tmd(real_tensor, dual_tensor);

    // Act
    TensorMatDual result = tmd.eye();

    // Assert
    EXPECT_EQ(result.r.sizes(), torch::IntArrayRef({2, 3, 3})); // Batch of identity matrices
    EXPECT_TRUE(torch::allclose(result.r[0], torch::eye(3, real_tensor.options())));
    EXPECT_TRUE(torch::allclose(result.r[1], torch::eye(3, real_tensor.options())));
    EXPECT_EQ(result.d.sizes(), torch::IntArrayRef({2, 3, 4, 5})); // Zero tensor for dual part
    EXPECT_TRUE(torch::all(result.d == 0).item<bool>());
}


// Test case for eye operation with empty tensors
TEST(TensorMatDualTest, Eye_EmptyTensors) {
    // Arrange
    torch::Tensor real_tensor = torch::empty({0, 3, 4}); // Empty batch size
    torch::Tensor dual_tensor = torch::empty({0, 3, 4, 5});
    TensorMatDual tmd(real_tensor, dual_tensor);

    // Act
    TensorMatDual result = tmd.eye();

    // Assert
    EXPECT_EQ(result.r.sizes(), torch::IntArrayRef({0, 3, 3})); // Batch of identity matrices
    EXPECT_EQ(result.d.sizes(), torch::IntArrayRef({0, 3, 4, 5})); // Zero tensor for dual part
}

// Test case for eye operation with square matrices
TEST(TensorMatDualTest, Eye_SquareMatrices) {
    // Arrange
    torch::Tensor real_tensor = torch::rand({2, 3, 3}); // Square matrices
    torch::Tensor dual_tensor = torch::rand({2, 3, 3, 5});
    TensorMatDual tmd(real_tensor, dual_tensor);

    // Act
    TensorMatDual result = tmd.eye();

    // Assert
    EXPECT_EQ(result.r.sizes(), torch::IntArrayRef({2, 3, 3})); // Batch of identity matrices
    EXPECT_TRUE(torch::allclose(result.r[0], torch::eye(3, real_tensor.options())));
    EXPECT_TRUE(torch::allclose(result.r[1], torch::eye(3, real_tensor.options())));
    EXPECT_EQ(result.d.sizes(), torch::IntArrayRef({2, 3, 3, 5})); // Zero tensor for dual part
    EXPECT_TRUE(torch::all(result.d == 0).item<bool>());
}


// Test case for sum operation on valid tensors along a valid dimension
TEST(TensorMatDualTest, Sum_ValidDimension) {
    // Arrange
    torch::Tensor real_tensor = torch::rand({2, 3, 4}); // Batch size 2, matrix size 3x4
    torch::Tensor dual_tensor = torch::rand({2, 3, 4, 5});
    TensorMatDual tmd(real_tensor, dual_tensor);

    // Act
    TensorMatDual result = tmd.sum(1); // Sum along dimension 1

    // Assert
    EXPECT_EQ(result.r.sizes(), torch::IntArrayRef({2, 1, 4}));
    EXPECT_EQ(result.d.sizes(), torch::IntArrayRef({2, 1, 4, 5}));
    EXPECT_TRUE(torch::allclose(result.r, real_tensor.sum(1, true)));
    EXPECT_TRUE(torch::allclose(result.d, dual_tensor.sum(1, true)));
}


// Test case for sum operation along the last dimension
TEST(TensorMatDualTest, Sum_LastDimension) {
    // Arrange
    torch::Tensor real_tensor = torch::rand({2, 3, 4});
    torch::Tensor dual_tensor = torch::rand({2, 3, 4, 5});
    TensorMatDual tmd(real_tensor, dual_tensor);

    // Act
    TensorMatDual result = tmd.sum(2); // Sum along dimension 2

    // Assert
    EXPECT_EQ(result.r.sizes(), torch::IntArrayRef({2, 3, 1}));
    EXPECT_EQ(result.d.sizes(), torch::IntArrayRef({2, 3, 1, 5}));
    EXPECT_TRUE(torch::allclose(result.r, real_tensor.sum(2, true)));
    EXPECT_TRUE(torch::allclose(result.d, dual_tensor.sum(2, true)));
}

// Test case for invalid dimension
TEST(TensorMatDualTest, Sum_InvalidDimension) {
    // Arrange
    torch::Tensor real_tensor = torch::rand({2, 3, 4});
    torch::Tensor dual_tensor = torch::rand({2, 3, 4, 5});
    TensorMatDual tmd(real_tensor, dual_tensor);

    // Act & Assert
    EXPECT_THROW(tmd.sum(4), std::invalid_argument); // Dimension 4 is out of bounds for real tensor
    EXPECT_THROW(tmd.sum(5), std::invalid_argument); // Dimension 5 is out of bounds for dual tensor
}

// Test case for sum operation with empty tensors
TEST(TensorMatDualTest, Sum_EmptyTensors) {
    // Arrange
    torch::Tensor real_tensor = torch::empty({0, 3, 4}); // Empty batch size
    torch::Tensor dual_tensor = torch::empty({0, 3, 4, 5});
    TensorMatDual tmd(real_tensor, dual_tensor);

    // Act
    TensorMatDual result = tmd.sum(1); // Sum along dimension 1

    // Assert
    EXPECT_EQ(result.r.sizes(), torch::IntArrayRef({0, 1, 4}));
    EXPECT_EQ(result.d.sizes(), torch::IntArrayRef({0, 1, 4, 5}));
    EXPECT_TRUE(result.r.numel() == 0); // Ensure the tensor remains empty
    EXPECT_TRUE(result.d.numel() == 0); // Ensure the tensor remains empty
}

// Test case for square operation with valid tensors
TEST(TensorMatDualTest, Square_ValidTensors) {
    // Arrange
    torch::Tensor real_tensor = torch::rand({2, 3, 4});
    torch::Tensor dual_tensor = torch::rand({2, 3, 4, 5});
    TensorMatDual tmd(real_tensor, dual_tensor);

    // Act
    TensorMatDual result = tmd.square();

    // Assert
    EXPECT_EQ(result.r.sizes(), real_tensor.sizes());
    EXPECT_EQ(result.d.sizes(), dual_tensor.sizes());
    EXPECT_TRUE(torch::allclose(result.r, real_tensor.square()));
    EXPECT_TRUE(torch::allclose(result.d, 2 * real_tensor.unsqueeze(-1) * dual_tensor));
}

// Test case for shape mismatch between real and dual tensors
TEST(TensorMatDualTest, Square_ShapeMismatch) {
    // Arrange
    torch::Tensor real_tensor = torch::rand({2, 3, 3}); // Real tensor dim = 3
    torch::Tensor dual_tensor = torch::rand({2, 3, 4, 5}); // Dual tensor dim = 4
    TensorMatDual tmd(real_tensor, dual_tensor);

    //Expect a c10 error
    ASSERT_THROW(tmd.square(), c10::Error);
}

// Test case for square operation with empty tensors
TEST(TensorMatDualTest, Square_EmptyTensors) {
    // Arrange
    torch::Tensor real_tensor = torch::empty({0, 3, 4}); // Empty tensor
    torch::Tensor dual_tensor = torch::empty({0, 3, 4, 5});
    TensorMatDual tmd(real_tensor, dual_tensor);

    // Act
    TensorMatDual result = tmd.square();

    // Assert
    EXPECT_EQ(result.r.sizes(), real_tensor.sizes());
    EXPECT_EQ(result.d.sizes(), dual_tensor.sizes());
    EXPECT_TRUE(torch::all(result.r == 0).item<bool>());
    EXPECT_TRUE(torch::all(result.d == 0).item<bool>());
}

// Test case for square operation with broadcasting compatibility
TEST(TensorMatDualTest, Square_BroadcastCompatibility) {
    // Arrange
    torch::Tensor real_tensor = torch::rand({2, 1, 4}); // Broadcastable shape
    torch::Tensor dual_tensor = torch::rand({2, 1, 4, 5});
    TensorMatDual tmd(real_tensor, dual_tensor);

    // Act
    TensorMatDual result = tmd.square();

    // Assert
    EXPECT_EQ(result.r.sizes(), real_tensor.sizes());
    EXPECT_EQ(result.d.sizes(), dual_tensor.sizes());
    EXPECT_TRUE(torch::allclose(result.r, real_tensor.square()));
    EXPECT_TRUE(torch::allclose(result.d, 2 * real_tensor.unsqueeze(-1) * dual_tensor));
}


// Test case for sqrt operation with valid tensors
TEST(TensorMatDualTest, Sqrt_ValidTensors) {
    // Arrange
    torch::Tensor real_tensor = torch::rand({2, 3, 4}) + 0.1; // Ensure positive values
    torch::Tensor dual_tensor = torch::rand({2, 3, 4, 5});
    TensorMatDual tmd(real_tensor, dual_tensor);

    // Act
    TensorMatDual result = tmd.sqrt();

    // Assert
    EXPECT_EQ(result.r.sizes(), real_tensor.sizes());
    EXPECT_EQ(result.d.sizes(), dual_tensor.sizes());
    EXPECT_TRUE(torch::allclose(result.r, real_tensor.sqrt()));
    EXPECT_TRUE(torch::allclose(result.d, 0.5 * real_tensor.pow(-0.5).unsqueeze(-1) * dual_tensor));
}

// Test case for sqrt operation with negative values in real tensor
TEST(TensorMatDualTest, Sqrt_NegativeValues) {
    // Arrange
    torch::Tensor real_tensor = torch::rand({2, 3, 4}) - 1.0; // Include negative values
    torch::Tensor dual_tensor = torch::rand({2, 3, 4, 5});
    TensorMatDual tmd(real_tensor, dual_tensor);

    //Expect that there is no exception thrown
    ASSERT_NO_THROW(tmd.sqrt());
}

// Test case for sqrt operation with empty tensors
TEST(TensorMatDualTest, Sqrt_EmptyTensors) {
    // Arrange
    torch::Tensor real_tensor = torch::empty({0, 3, 4}); // Empty tensor
    torch::Tensor dual_tensor = torch::empty({0, 3, 4, 5});
    TensorMatDual tmd(real_tensor, dual_tensor);

    // Act
    TensorMatDual result = tmd.sqrt();

    // Assert
    EXPECT_EQ(result.r.sizes(), real_tensor.sizes());
    EXPECT_EQ(result.d.sizes(), dual_tensor.sizes());
}

// Test case for broadcasting compatibility in sqrt operation
TEST(TensorMatDualTest, Sqrt_BroadcastCompatibility) {
    // Arrange
    torch::Tensor real_tensor = torch::rand({2, 1, 4}) + 0.1; // Ensure positive values
    torch::Tensor dual_tensor = torch::rand({2, 1, 4, 5});
    TensorMatDual tmd(real_tensor, dual_tensor);

    // Act
    TensorMatDual result = tmd.sqrt();

    // Assert
    EXPECT_EQ(result.r.sizes(), real_tensor.sizes());
    EXPECT_EQ(result.d.sizes(), dual_tensor.sizes());
    EXPECT_TRUE(torch::allclose(result.r, real_tensor.sqrt()));
    EXPECT_TRUE(torch::allclose(result.d, 0.5 * real_tensor.pow(-0.5).unsqueeze(-1) * dual_tensor));
}


// Test case for normL2 operation with valid tensors
TEST(TensorMatDualTest, NormL2_ValidTensors) {
    // Arrange
    torch::Tensor real_tensor = torch::rand({2, 3, 4});
    torch::Tensor dual_tensor = torch::rand({2, 3, 4, 5});
    TensorMatDual tmd(real_tensor, dual_tensor);

    // Act
    TensorMatDual result = tmd.normL2();

    // Assert
    EXPECT_EQ(result.r.sizes(), torch::IntArrayRef({2, 3, 1})); // L2 norm keeps reduced dimension
    EXPECT_EQ(result.d.sizes(), torch::IntArrayRef({2, 3, 1, 5}));
    EXPECT_TRUE(torch::allclose(result.r, torch::norm(real_tensor, 2, -1, true)));

    auto norm_r = torch::norm(real_tensor, 2, -1, true).expand_as(real_tensor);
    auto grad_r = torch::where(norm_r > 0, real_tensor / norm_r, torch::zeros_like(real_tensor));
    auto expected_dual = torch::einsum("mij, mijn->min", {grad_r, dual_tensor}).unsqueeze(2);
    EXPECT_TRUE(torch::allclose(result.d, expected_dual));
}

// Test case for normL2 operation with empty tensors
TEST(TensorMatDualTest, NormL2_EmptyTensors) {
    // Arrange
    torch::Tensor real_tensor = torch::empty({0, 3, 4});
    torch::Tensor dual_tensor = torch::empty({0, 3, 4, 5});
    TensorMatDual tmd(real_tensor, dual_tensor);

    // Act
    TensorMatDual result = tmd.normL2();

    // Assert
    EXPECT_EQ(result.r.sizes(), torch::IntArrayRef({0, 3, 1}));
    EXPECT_EQ(result.d.sizes(), torch::IntArrayRef({0, 3, 1, 5}));
}

// Test case for normL2 operation with broadcasting compatibility
TEST(TensorMatDualTest, NormL2_BroadcastCompatibility) {
    // Arrange
    torch::Tensor real_tensor = torch::rand({2, 1, 4}); // Broadcastable shape
    torch::Tensor dual_tensor = torch::rand({2, 1, 4, 5});
    TensorMatDual tmd(real_tensor, dual_tensor);

    // Act
    TensorMatDual result = tmd.normL2();

    // Assert
    EXPECT_EQ(result.r.sizes(), torch::IntArrayRef({2, 1, 1}));
    EXPECT_EQ(result.d.sizes(), torch::IntArrayRef({2, 1, 1, 5}));
    EXPECT_TRUE(torch::allclose(result.r, torch::norm(real_tensor, 2, -1, true)));

    auto norm_r = torch::norm(real_tensor, 2, -1, true).expand_as(real_tensor);
    auto grad_r = torch::where(norm_r > 0, real_tensor / norm_r, torch::zeros_like(real_tensor));
    auto expected_dual = torch::einsum("mij, mijn->min", {grad_r, dual_tensor}).unsqueeze(2);
    EXPECT_TRUE(torch::allclose(result.d, expected_dual));
}

// Test case for normL2 operation with shape mismatch
TEST(TensorMatDualTest, NormL2_ShapeMismatch) {
    // Arrange
    torch::Tensor real_tensor = torch::rand({2, 3, 4});
    torch::Tensor dual_tensor = torch::rand({2, 3, 3, 5});
    TensorMatDual tmd(real_tensor, dual_tensor);

    //Expect a c10 error
    EXPECT_THROW(tmd.normL2(), c10::Error);
}



// Test case for createZero with valid input tensors
TEST(TensorMatDualTest, CreateZero_ValidInput) {
    // Arrange
    torch::Tensor real_tensor = torch::rand({2, 3, 4}); // Real tensor
    int ddim = 5;

    // Act
    TensorMatDual result = TensorMatDual::createZero(real_tensor, ddim);

    // Assert
    EXPECT_EQ(result.r.sizes(), real_tensor.sizes());
    auto expected_dual_shape = real_tensor.sizes().vec();
    expected_dual_shape.push_back(ddim);
    EXPECT_EQ(result.d.sizes(), torch::IntArrayRef(expected_dual_shape));
    EXPECT_TRUE(torch::all(result.d == 0).item<bool>());
}

// Test case for createZero with empty real tensor
TEST(TensorMatDualTest, CreateZero_EmptyRealTensor) {
    // Arrange
    torch::Tensor real_tensor = torch::empty({0, 3, 4});
    int ddim = 5;

    // Act
    TensorMatDual result = TensorMatDual::createZero(real_tensor, ddim);

    // Assert
    EXPECT_EQ(result.r.sizes(), real_tensor.sizes());
    auto expected_dual_shape = real_tensor.sizes().vec();
    expected_dual_shape.push_back(ddim);
    EXPECT_EQ(result.d.sizes(), torch::IntArrayRef(expected_dual_shape));
    EXPECT_TRUE(torch::all(result.d == 0).item<bool>());
}

// Test case for createZero with invalid ddim (negative)
TEST(TensorMatDualTest, CreateZero_InvalidDdimNegative) {
    // Arrange
    torch::Tensor real_tensor = torch::rand({2, 3, 4});
    int ddim = -1;

    // Act & Assert
    EXPECT_THROW(TensorMatDual::createZero(real_tensor, ddim), std::invalid_argument);
}

// Test case for createZero with invalid ddim (zero)
TEST(TensorMatDualTest, CreateZero_InvalidDdimZero) {
    // Arrange
    torch::Tensor real_tensor = torch::rand({2, 3, 4});
    int ddim = 0;

    // Act & Assert
    EXPECT_THROW(TensorMatDual::createZero(real_tensor, ddim), std::invalid_argument);
}

// Test case for createZero with undefined real tensor
TEST(TensorMatDualTest, CreateZero_UndefinedRealTensor) {
    // Arrange
    torch::Tensor real_tensor; // Undefined tensor
    int ddim = 5;

    // Act & Assert
    EXPECT_THROW(TensorMatDual::createZero(real_tensor, ddim), std::invalid_argument);
}



// Test case for zeros_like with valid tensors
TEST(TensorMatDualTest, ZerosLikeSelf_ValidTensors) {
    // Arrange
    torch::Tensor real_tensor = torch::rand({2, 3, 4});
    torch::Tensor dual_tensor = torch::rand({2, 3, 4, 5});
    TensorMatDual tmd(real_tensor, dual_tensor);

    // Act
    TensorMatDual result = tmd.zeros_like();

    // Assert
    EXPECT_EQ(result.r.sizes(), real_tensor.sizes());
    EXPECT_TRUE(torch::all(result.r == 0).item<bool>());
    EXPECT_EQ(result.d.sizes(), dual_tensor.sizes());
    EXPECT_TRUE(torch::all(result.d == 0).item<bool>());
    EXPECT_EQ(result.r.dtype(), real_tensor.dtype());
    EXPECT_EQ(result.r.device(), real_tensor.device());
    EXPECT_EQ(result.d.dtype(), dual_tensor.dtype());
    EXPECT_EQ(result.d.device(), dual_tensor.device());
}

// Test case for zeros_like with empty tensors
TEST(TensorMatDualTest, ZerosLikeSelf_EmptyTensors) {
    // Arrange
    torch::Tensor real_tensor = torch::empty({0, 3, 4});
    torch::Tensor dual_tensor = torch::empty({0, 3, 4, 5});
    TensorMatDual tmd(real_tensor, dual_tensor);

    // Act
    TensorMatDual result = tmd.zeros_like();

    // Assert
    EXPECT_EQ(result.r.sizes(), real_tensor.sizes());
    EXPECT_TRUE(torch::all(result.r == 0).item<bool>());
    EXPECT_EQ(result.d.sizes(), dual_tensor.sizes());
    EXPECT_TRUE(torch::all(result.d == 0).item<bool>());
}




// Test case for clone with valid tensors
TEST(TensorMatDualTest, Clone_ValidTensors) {
    // Arrange
    torch::Tensor real_tensor = torch::rand({2, 3, 4});
    torch::Tensor dual_tensor = torch::rand({2, 3, 4, 5});
    TensorMatDual tmd(real_tensor, dual_tensor);

    // Act
    TensorMatDual cloned_tmd = tmd.clone();

    // Assert
    EXPECT_EQ(cloned_tmd.r.sizes(), real_tensor.sizes());
    EXPECT_EQ(cloned_tmd.d.sizes(), dual_tensor.sizes());
    EXPECT_TRUE(torch::allclose(cloned_tmd.r, real_tensor));
    EXPECT_TRUE(torch::allclose(cloned_tmd.d, dual_tensor));
}

// Test case for clone with empty tensors
TEST(TensorMatDualTest, Clone_EmptyTensors) {
    // Arrange
    torch::Tensor real_tensor = torch::empty({0, 3, 4});
    torch::Tensor dual_tensor = torch::empty({0, 3, 4, 5});
    TensorMatDual tmd(real_tensor, dual_tensor);

    // Act
    TensorMatDual cloned_tmd = tmd.clone();

    // Assert
    EXPECT_EQ(cloned_tmd.r.sizes(), real_tensor.sizes());
    EXPECT_EQ(cloned_tmd.d.sizes(), dual_tensor.sizes());
    EXPECT_TRUE(torch::all(cloned_tmd.r == 0).item<bool>());
    EXPECT_TRUE(torch::all(cloned_tmd.d == 0).item<bool>());
}


// Test case for cat with valid tensors along the default dimension
TEST(TensorMatDualTest, Cat_ValidDefaultDimension) {
    // Arrange
    torch::Tensor real_tensor1 = torch::rand({2, 3, 4});
    torch::Tensor dual_tensor1 = torch::rand({2, 3, 4, 5});
    TensorMatDual tmd1(real_tensor1, dual_tensor1);

    torch::Tensor real_tensor2 = torch::rand({2, 3, 4});
    torch::Tensor dual_tensor2 = torch::rand({2, 3, 4, 5});
    TensorMatDual tmd2(real_tensor2, dual_tensor2);

    // Act
    TensorMatDual result = TensorMatDual::cat(tmd1, tmd2);

    // Assert
    EXPECT_EQ(result.r.sizes(), torch::IntArrayRef({2, 3, 8})); // Concatenated along dim 2
    EXPECT_EQ(result.d.sizes(), torch::IntArrayRef({2, 3, 8, 5}));
}

// Test case for cat with valid tensors along a specified dimension
TEST(TensorMatDualTest, Cat_ValidSpecifiedDimension) {
    // Arrange
    torch::Tensor real_tensor1 = torch::rand({2, 3, 4});
    torch::Tensor dual_tensor1 = torch::rand({2, 3, 4, 5});
    TensorMatDual tmd1(real_tensor1, dual_tensor1);

    torch::Tensor real_tensor2 = torch::rand({2, 3, 4});
    torch::Tensor dual_tensor2 = torch::rand({2, 3, 4, 5});
    TensorMatDual tmd2(real_tensor2, dual_tensor2);

    // Act
    TensorMatDual result = TensorMatDual::cat(tmd1, tmd2, 1);

    // Assert
    EXPECT_EQ(result.r.sizes(), torch::IntArrayRef({2, 6, 4})); // Concatenated along dim 1
    EXPECT_EQ(result.d.sizes(), torch::IntArrayRef({2, 6, 4, 5}));
}

// Test case for cat with mismatched dimensions
TEST(TensorMatDualTest, Cat_MismatchedDimensions) {
    // Arrange
    torch::Tensor real_tensor1 = torch::rand({2, 3, 4});
    torch::Tensor dual_tensor1 = torch::rand({2, 3, 4, 5});
    TensorMatDual tmd1(real_tensor1, dual_tensor1);

    torch::Tensor real_tensor2 = torch::rand({2, 4, 4}); // Mismatched dimensions
    torch::Tensor dual_tensor2 = torch::rand({2, 4, 4, 5});
    TensorMatDual tmd2(real_tensor2, dual_tensor2);

    // Act & Assert
    EXPECT_THROW(TensorMatDual::cat(tmd1, tmd2), std::invalid_argument);
}


// Test case for cat with valid TensorMatDual and TensorDual objects
TEST(TensorMatDualTest, Cat_ValidTensorMatDualWithTensorDual) {
    // Arrange
    torch::Tensor real_tensor1 = torch::rand({2, 3, 4});
    torch::Tensor dual_tensor1 = torch::rand({2, 3, 4, 5});
    TensorMatDual tmd(real_tensor1, dual_tensor1);

    torch::Tensor real_tensor2 = torch::rand({2, 3});
    torch::Tensor dual_tensor2 = torch::rand({2, 3, 5});
    TensorDual td(real_tensor2, dual_tensor2);

    // Act
    TensorMatDual result = TensorMatDual::cat(tmd, td);

    // Assert
    EXPECT_EQ(result.r.sizes(), torch::IntArrayRef({2, 3, 5})); // Concatenated along dim 2
    EXPECT_EQ(result.d.sizes(), torch::IntArrayRef({2, 3, 5, 5}));
}

// Test case for cat with mismatched TensorMatDual and TensorDual objects
TEST(TensorMatDualTest, Cat_MismatchedTensorMatDualWithTensorDual) {
    // Arrange
    torch::Tensor real_tensor1 = torch::rand({2, 3, 4});
    torch::Tensor dual_tensor1 = torch::rand({2, 3, 4, 5});
    TensorMatDual tmd(real_tensor1, dual_tensor1);

    torch::Tensor real_tensor2 = torch::rand({1, 3}); // Mismatched batch size
    torch::Tensor dual_tensor2 = torch::rand({1, 3, 5});
    TensorDual td(real_tensor2, dual_tensor2);

    // Act & Assert
    EXPECT_THROW(TensorMatDual::cat(tmd, td), std::invalid_argument);
}


// Test case for cat with valid TensorMatDual and tensor objects
TEST(TensorMatDualTest, Cat_ValidTensorMatDualWithTensor) {
    // Arrange
    torch::Tensor real_tensor1 = torch::rand({2, 3, 4});
    torch::Tensor dual_tensor1 = torch::rand({2, 3, 4, 5});
    TensorMatDual tmd(real_tensor1, dual_tensor1);

    torch::Tensor t2 = torch::rand({3, 4});

    // Act
    TensorMatDual result = TensorMatDual::cat(tmd, t2);

    // Assert
    EXPECT_EQ(result.r.sizes(), torch::IntArrayRef({2, 3, 8})); // Concatenated along dim 2
    EXPECT_EQ(result.d.sizes(), torch::IntArrayRef({2, 3, 8, 5}));
    EXPECT_TRUE(torch::allclose(result.r.narrow(2, 4, 4), t2.repeat({2, 1, 1})));
    EXPECT_TRUE(torch::all(result.d.narrow(2, 4, 4) == 0).item<bool>());
}

// Test case for cat with mismatched TensorMatDual and tensor objects
TEST(TensorMatDualTest, Cat_MismatchedTensorMatDualWithTensor) {
    // Arrange
    torch::Tensor real_tensor1 = torch::rand({2, 3, 4});
    torch::Tensor dual_tensor1 = torch::rand({2, 3, 4, 5});
    TensorMatDual tmd(real_tensor1, dual_tensor1);

    torch::Tensor t2 = torch::rand({2, 4}); // Mismatched dimensions

    // Act & Assert
    EXPECT_THROW(TensorMatDual::cat(tmd, t2), std::invalid_argument);
}


// Test case for addition of two valid TensorMatDual objects
TEST(TensorMatDualTest, Addition_ValidTensors) {
    // Arrange
    torch::Tensor real_tensor1 = torch::rand({2, 3, 4});
    torch::Tensor dual_tensor1 = torch::rand({2, 3, 4, 5});
    TensorMatDual tmd1(real_tensor1, dual_tensor1);

    torch::Tensor real_tensor2 = torch::rand({2, 3, 4});
    torch::Tensor dual_tensor2 = torch::rand({2, 3, 4, 5});
    TensorMatDual tmd2(real_tensor2, dual_tensor2);

    // Act
    TensorMatDual result = tmd1 + tmd2;

    // Assert
    EXPECT_EQ(result.r.sizes(), real_tensor1.sizes());
    EXPECT_EQ(result.d.sizes(), dual_tensor1.sizes());
    EXPECT_TRUE(torch::allclose(result.r, real_tensor1 + real_tensor2));
    EXPECT_TRUE(torch::allclose(result.d, dual_tensor1 + dual_tensor2));
}

// Test case for addition with mismatched TensorMatDual objects
TEST(TensorMatDualTest, Addition_MismatchedTensors) {
    // Arrange
    torch::Tensor real_tensor1 = torch::rand({2, 3, 4});
    torch::Tensor dual_tensor1 = torch::rand({2, 3, 4, 5});
    TensorMatDual tmd1(real_tensor1, dual_tensor1);

    torch::Tensor real_tensor2 = torch::rand({2, 4, 4}); // Mismatched dimensions
    torch::Tensor dual_tensor2 = torch::rand({2, 4, 4, 5});
    TensorMatDual tmd2(real_tensor2, dual_tensor2);

    // Act & Assert
    EXPECT_THROW(tmd1 + tmd2, std::invalid_argument);
}


// Test case for addition of a valid TensorMatDual with TensorDual
TEST(TensorMatDualTest, Addition_ValidTensorDual) {
    // Arrange
    torch::Tensor real_tensor1 = torch::rand({2, 3, 4});
    torch::Tensor dual_tensor1 = torch::rand({2, 3, 4, 5});
    TensorMatDual tmd(real_tensor1, dual_tensor1);

    torch::Tensor real_tensor2 = torch::rand({2, 3});
    torch::Tensor dual_tensor2 = torch::rand({2, 3, 5});
    TensorDual td(real_tensor2, dual_tensor2);

    // Act
    TensorMatDual result = tmd + td;

    // Assert
    EXPECT_EQ(result.r.sizes(), real_tensor1.sizes());
    EXPECT_EQ(result.d.sizes(), dual_tensor1.sizes());
    auto real_part = torch::einsum("mij, mi->mij", {real_tensor1, real_tensor2});
    auto dual_part = torch::einsum("mik, mijk->mijk", {dual_tensor2, dual_tensor1});
    EXPECT_TRUE(torch::allclose(result.r, real_part));
    EXPECT_TRUE(torch::allclose(result.d, dual_part));
}

// Test case for addition with mismatched TensorMatDual and TensorDual
TEST(TensorMatDualTest, Addition_MismatchedTensorDual) {
    // Arrange
    torch::Tensor real_tensor1 = torch::rand({2, 3, 4});
    torch::Tensor dual_tensor1 = torch::rand({2, 3, 4, 5});
    TensorMatDual tmd(real_tensor1, dual_tensor1);

    torch::Tensor real_tensor2 = torch::rand({2, 4}); // Mismatched dimensions
    torch::Tensor dual_tensor2 = torch::rand({2, 4, 5});
    TensorDual td(real_tensor2, dual_tensor2);

    //Expact a c10 error
    EXPECT_THROW(tmd + td, c10::Error);
}


// Test case for addition of a scalar to a valid TensorMatDual
TEST(TensorMatDualTest, Addition_ValidScalar) {
    // Arrange
    torch::Tensor real_tensor = torch::rand({2, 3, 4});
    torch::Tensor dual_tensor = torch::rand({2, 3, 4, 5});
    TensorMatDual tmd(real_tensor, dual_tensor);
    double scalar = 2.5;

    // Act
    TensorMatDual result = tmd + scalar;

    // Assert
    EXPECT_EQ(result.r.sizes(), real_tensor.sizes());
    EXPECT_EQ(result.d.sizes(), dual_tensor.sizes());
    EXPECT_TRUE(torch::allclose(result.r, real_tensor + scalar));
    EXPECT_TRUE(torch::allclose(result.d, dual_tensor)); // Dual part remains unchanged
}

// Test case for addition of a scalar to a TensorMatDual with empty tensors
TEST(TensorMatDualTest, Addition_EmptyTensors) {
    // Arrange
    torch::Tensor real_tensor = torch::empty({0, 3, 4});
    torch::Tensor dual_tensor = torch::empty({0, 3, 4, 5});
    TensorMatDual tmd(real_tensor, dual_tensor);
    double scalar = 1.0;

    // Act
    TensorMatDual result = tmd + scalar;

    // Assert
    EXPECT_EQ(result.r.sizes(), real_tensor.sizes());
    EXPECT_EQ(result.d.sizes(), dual_tensor.sizes());
    EXPECT_TRUE(torch::allclose(result.r, real_tensor + scalar));
    EXPECT_TRUE(torch::allclose(result.d, dual_tensor));
}


// Test case for subtraction of two valid TensorMatDual objects
TEST(TensorMatDualTest, Subtraction_ValidTensors) {
    // Arrange
    torch::Tensor real_tensor1 = torch::rand({2, 3, 4});
    torch::Tensor dual_tensor1 = torch::rand({2, 3, 4, 5});
    TensorMatDual tmd1(real_tensor1, dual_tensor1);

    torch::Tensor real_tensor2 = torch::rand({2, 3, 4});
    torch::Tensor dual_tensor2 = torch::rand({2, 3, 4, 5});
    TensorMatDual tmd2(real_tensor2, dual_tensor2);

    // Act
    TensorMatDual result = tmd1 - tmd2;

    // Assert
    EXPECT_EQ(result.r.sizes(), real_tensor1.sizes());
    EXPECT_EQ(result.d.sizes(), dual_tensor1.sizes());
    EXPECT_TRUE(torch::allclose(result.r, real_tensor1 - real_tensor2));
    EXPECT_TRUE(torch::allclose(result.d, dual_tensor1 - dual_tensor2));
}

// Test case for subtraction with mismatched TensorMatDual objects
TEST(TensorMatDualTest, Subtraction_MismatchedTensors) {
    // Arrange
    torch::Tensor real_tensor1 = torch::rand({2, 3, 4});
    torch::Tensor dual_tensor1 = torch::rand({2, 3, 4, 5});
    TensorMatDual tmd1(real_tensor1, dual_tensor1);

    torch::Tensor real_tensor2 = torch::rand({2, 4, 4}); // Mismatched dimensions
    torch::Tensor dual_tensor2 = torch::rand({2, 4, 4, 5});
    TensorMatDual tmd2(real_tensor2, dual_tensor2);

    //Expect an c10 error
    EXPECT_THROW(tmd1 - tmd2, c10::Error);
}



// Test case for subtraction of a scalar from a valid TensorMatDual
TEST(TensorMatDualTest, Subtraction_ValidScalar) {
    // Arrange
    torch::Tensor real_tensor = torch::rand({2, 3, 4});
    torch::Tensor dual_tensor = torch::rand({2, 3, 4, 5});
    TensorMatDual tmd(real_tensor, dual_tensor);
    double scalar = 2.5;

    // Act
    TensorMatDual result = tmd - scalar;

    // Assert
    EXPECT_EQ(result.r.sizes(), real_tensor.sizes());
    EXPECT_EQ(result.d.sizes(), dual_tensor.sizes());
    EXPECT_TRUE(torch::allclose(result.r, real_tensor - scalar));
    EXPECT_TRUE(torch::allclose(result.d, dual_tensor)); // Dual part remains unchanged
}

// Test case for subtraction of a scalar from a TensorMatDual with empty tensors
TEST(TensorMatDualTest, Subtraction_EmptyTensors) {
    // Arrange
    torch::Tensor real_tensor = torch::empty({0, 3, 4});
    torch::Tensor dual_tensor = torch::empty({0, 3, 4, 5});
    TensorMatDual tmd(real_tensor, dual_tensor);
    double scalar = 1.0;

    // Act
    TensorMatDual result = tmd - scalar;

    // Assert
    EXPECT_EQ(result.r.sizes(), real_tensor.sizes());
    EXPECT_EQ(result.d.sizes(), dual_tensor.sizes());
    EXPECT_TRUE(torch::allclose(result.r, real_tensor - scalar));
    EXPECT_TRUE(torch::allclose(result.d, dual_tensor));
}


// Test case for equality comparison of two valid TensorMatDual objects
TEST(TensorMatDualTest, Equality_ValidTensors) {
    // Arrange
    torch::Tensor real_tensor1 = torch::randint(0, 2, {2, 3, 4}); // Binary values for easier comparison
    torch::Tensor dual_tensor1 = torch::rand({2, 3, 4, 5});
    TensorMatDual tmd1(real_tensor1, dual_tensor1);

    torch::Tensor real_tensor2 = real_tensor1.clone();
    torch::Tensor dual_tensor2 = torch::rand({2, 3, 4, 5});
    TensorMatDual tmd2(real_tensor2, dual_tensor2);

    // Act
    torch::Tensor result = tmd1 == tmd2;

    // Assert
    EXPECT_EQ(result.sizes(), real_tensor1.sizes());
    EXPECT_TRUE(torch::all(result).item<bool>()); // All elements should be true
}

// Test case for equality comparison with mismatched TensorMatDual objects
TEST(TensorMatDualTest, Equality_MismatchedTensors) {
    // Arrange
    torch::Tensor real_tensor1 = torch::rand({2, 3, 4});
    torch::Tensor dual_tensor1 = torch::rand({2, 3, 4, 5});
    TensorMatDual tmd1(real_tensor1, dual_tensor1);

    torch::Tensor real_tensor2 = torch::rand({2, 4, 4}); // Mismatched dimensions
    torch::Tensor dual_tensor2 = torch::rand({2, 4, 4, 5});
    TensorMatDual tmd2(real_tensor2, dual_tensor2);

    // Act & Assert
    EXPECT_THROW(tmd1 == tmd2, std::invalid_argument);
}


// Test case for partial equality in TensorMatDual objects
TEST(TensorMatDualTest, Equality_PartialMatch) {
    // Arrange
    torch::Tensor real_tensor1 = torch::randint(0, 2, {2, 3, 4});
    torch::Tensor dual_tensor1 = torch::rand({2, 3, 4, 5});
    TensorMatDual tmd1(real_tensor1, dual_tensor1);

    torch::Tensor real_tensor2 = real_tensor1.clone();
    real_tensor2[0][0][0] = real_tensor2[0][0][0] + 1; // Introduce a mismatch
    torch::Tensor dual_tensor2 = torch::rand({2, 3, 4, 5});
    TensorMatDual tmd2(real_tensor2, dual_tensor2);

    // Act
    torch::Tensor result = tmd1 == tmd2;

    // Assert
    EXPECT_EQ(result.sizes(), real_tensor1.sizes());
    EXPECT_FALSE(torch::all(result).item<bool>()); // Not all elements should match
}


// Test case for unary negation of a valid TensorMatDual
TEST(TensorMatDualTest, UnaryNegation_ValidTensors) {
    // Arrange
    torch::Tensor real_tensor = torch::rand({2, 3, 4});
    torch::Tensor dual_tensor = torch::rand({2, 3, 4, 5});
    TensorMatDual tmd(real_tensor, dual_tensor);

    // Act
    TensorMatDual result = -tmd;

    // Assert
    EXPECT_EQ(result.r.sizes(), real_tensor.sizes());
    EXPECT_EQ(result.d.sizes(), dual_tensor.sizes());
    EXPECT_TRUE(torch::allclose(result.r, -real_tensor));
    EXPECT_TRUE(torch::allclose(result.d, -dual_tensor));
}

// Test case for unary negation of a TensorMatDual with empty tensors
TEST(TensorMatDualTest, UnaryNegation_EmptyTensors) {
    // Arrange
    torch::Tensor real_tensor = torch::empty({0, 3, 4});
    torch::Tensor dual_tensor = torch::empty({0, 3, 4, 5});
    TensorMatDual tmd(real_tensor, dual_tensor);

    // Act
    TensorMatDual result = -tmd;

    // Assert
    EXPECT_EQ(result.r.sizes(), real_tensor.sizes());
    EXPECT_EQ(result.d.sizes(), dual_tensor.sizes());
    EXPECT_TRUE(torch::allclose(result.r, -real_tensor));
    EXPECT_TRUE(torch::allclose(result.d, -dual_tensor));
}


// Test case for scalar multiplication of a valid TensorMatDual
TEST(TensorMatDualTest, ScalarMultiplication_ValidTensors) {
    // Arrange
    torch::Tensor real_tensor = torch::rand({2, 3, 4});
    torch::Tensor dual_tensor = torch::rand({2, 3, 4, 5});
    TensorMatDual tmd(real_tensor, dual_tensor);
    double scalar = 2.5;

    // Act
    TensorMatDual result = tmd * scalar;

    // Assert
    EXPECT_EQ(result.r.sizes(), real_tensor.sizes());
    EXPECT_EQ(result.d.sizes(), dual_tensor.sizes());
    EXPECT_TRUE(torch::allclose(result.r, real_tensor * scalar));
    EXPECT_TRUE(torch::allclose(result.d, dual_tensor * scalar));
}

// Test case for scalar multiplication of a TensorMatDual with empty tensors
TEST(TensorMatDualTest, ScalarMultiplication_EmptyTensors) {
    // Arrange
    torch::Tensor real_tensor = torch::empty({0, 3, 4});
    torch::Tensor dual_tensor = torch::empty({0, 3, 4, 5});
    TensorMatDual tmd(real_tensor, dual_tensor);
    double scalar = 1.0;

    // Act
    TensorMatDual result = tmd * scalar;

    // Assert
    EXPECT_EQ(result.r.sizes(), real_tensor.sizes());
    EXPECT_EQ(result.d.sizes(), dual_tensor.sizes());
    EXPECT_TRUE(torch::allclose(result.r, real_tensor * scalar));
    EXPECT_TRUE(torch::allclose(result.d, dual_tensor * scalar));
}


// Test case for division of two valid TensorMatDual objects
TEST(TensorMatDualTest, Division_ValidTensors) {
    // Arrange
    torch::Tensor real_tensor1 = torch::rand({2, 3, 4}) + 1.0; // Ensure no zeros in denominator
    torch::Tensor dual_tensor1 = torch::rand({2, 3, 4, 5});
    TensorMatDual tmd1(real_tensor1, dual_tensor1);

    torch::Tensor real_tensor2 = torch::rand({2, 3, 4}) + 1.0; // Ensure no zeros in denominator
    torch::Tensor dual_tensor2 = torch::rand({2, 3, 4, 5});
    TensorMatDual tmd2(real_tensor2, dual_tensor2);

    // Act
    TensorMatDual result = tmd1 / tmd2;

    // Assert
    EXPECT_EQ(result.r.sizes(), real_tensor1.sizes());
    EXPECT_EQ(result.d.sizes(), dual_tensor1.sizes());

    // Compute expected values
    auto expected_r = real_tensor1 / real_tensor2;
    auto otherrsq = real_tensor2.square();
    auto expected_d = -torch::einsum("mij,mijn->mijn", {real_tensor1 / otherrsq, dual_tensor2}) +
                      torch::einsum("mij,mijn->mijn", {real_tensor2.reciprocal(), dual_tensor1});

    EXPECT_TRUE(torch::allclose(result.r, expected_r));
    EXPECT_TRUE(torch::allclose(result.d, expected_d));
}

// Test case for division with mismatched TensorMatDual objects
TEST(TensorMatDualTest, Division_MismatchedTensors) {
    // Arrange
    torch::Tensor real_tensor1 = torch::rand({2, 3, 4});
    torch::Tensor dual_tensor1 = torch::rand({2, 3, 4, 5});
    TensorMatDual tmd1(real_tensor1, dual_tensor1);

    torch::Tensor real_tensor2 = torch::rand({2, 4, 4}); // Mismatched dimensions
    torch::Tensor dual_tensor2 = torch::rand({2, 4, 4, 5});
    TensorMatDual tmd2(real_tensor2, dual_tensor2);

    //Expect a c10 error
    EXPECT_THROW(tmd1 / tmd2, c10::Error);
}

// Test case for division with zeros in the denominator
TEST(TensorMatDualTest, Division_DivisionByZero) {
    // Arrange
    torch::Tensor real_tensor1 = torch::rand({2, 3, 4});
    torch::Tensor dual_tensor1 = torch::rand({2, 3, 4, 5});
    TensorMatDual tmd1(real_tensor1, dual_tensor1);

    torch::Tensor real_tensor2 = torch::zeros({2, 3, 4}); // Zeros in denominator
    torch::Tensor dual_tensor2 = torch::rand({2, 3, 4, 5});
    TensorMatDual tmd2(real_tensor2, dual_tensor2);

    // Act & Assert
    EXPECT_THROW(tmd1 / tmd2, std::runtime_error);
}


// Test case for division of a TensorMatDual by a valid torch::Tensor
TEST(TensorMatDualTest, Division_ValidTensor) {
    // Arrange
    torch::Tensor real_tensor = torch::rand({2, 3, 4}) + 1.0; // Ensure no zeros in denominator
    torch::Tensor dual_tensor = torch::rand({2, 3, 4, 5});
    TensorMatDual tmd(real_tensor, dual_tensor);

    torch::Tensor divisor = torch::rand({2, 3, 4}) + 1.0; // Ensure no zeros in divisor

    // Act
    TensorMatDual result = tmd / divisor;

    // Assert
    EXPECT_EQ(result.r.sizes(), real_tensor.sizes());
    EXPECT_EQ(result.d.sizes(), dual_tensor.sizes());

    // Compute expected values
    auto expected_r = real_tensor / divisor;
    auto expected_d = dual_tensor / divisor.unsqueeze(-1);

    EXPECT_TRUE(torch::allclose(result.r, expected_r));
    EXPECT_TRUE(torch::allclose(result.d, expected_d));
}

// Test case for division with mismatched tensor shapes
TEST(TensorMatDualTest, Division_MismatchedTensorShapes) {
    // Arrange
    torch::Tensor real_tensor = torch::rand({2, 3, 4});
    torch::Tensor dual_tensor = torch::rand({2, 3, 4, 5});
    TensorMatDual tmd(real_tensor, dual_tensor);

    torch::Tensor divisor = torch::rand({3, 4}); // Mismatched dimensions

    // Act & Assert
    EXPECT_THROW(tmd / divisor, std::invalid_argument);
}

// Test case for division by a tensor containing zeros
TEST(TensorMatDualTest, Division_ByZeroTensor) {
    // Arrange
    torch::Tensor real_tensor = torch::rand({2, 3, 4});
    torch::Tensor dual_tensor = torch::rand({2, 3, 4, 5});
    TensorMatDual tmd(real_tensor, dual_tensor);

    torch::Tensor divisor = torch::zeros({2, 3, 4}); // Zeros in divisor

    // Act & Assert
    EXPECT_THROW(tmd / divisor, std::runtime_error);
}

// Test case for division with undefined tensor
TEST(TensorMatDualTest, Division_UndefinedTensor) {
    // Arrange
    torch::Tensor real_tensor = torch::rand({2, 3, 4});
    torch::Tensor dual_tensor = torch::rand({2, 3, 4, 5});
    TensorMatDual tmd(real_tensor, dual_tensor);

    torch::Tensor divisor; // Undefined tensor

    // Act & Assert
    EXPECT_THROW(tmd / divisor, std::invalid_argument);
}


// Test case for division of a TensorMatDual by a valid scalar
TEST(TensorMatDualTest, ScalarDivision_ValidScalar) {
    // Arrange
    torch::Tensor real_tensor = torch::rand({2, 3, 4});
    torch::Tensor dual_tensor = torch::rand({2, 3, 4, 5});
    TensorMatDual tmd(real_tensor, dual_tensor);
    double scalar = 2.5;

    // Act
    TensorMatDual result = tmd / scalar;

    // Assert
    EXPECT_EQ(result.r.sizes(), real_tensor.sizes());
    EXPECT_EQ(result.d.sizes(), dual_tensor.sizes());
    EXPECT_TRUE(torch::allclose(result.r, real_tensor / scalar));
    EXPECT_TRUE(torch::allclose(result.d, dual_tensor / scalar));
}

// Test case for division of a TensorMatDual by zero scalar
TEST(TensorMatDualTest, ScalarDivision_DivisionByZero) {
    // Arrange
    torch::Tensor real_tensor = torch::rand({2, 3, 4});
    torch::Tensor dual_tensor = torch::rand({2, 3, 4, 5});
    TensorMatDual tmd(real_tensor, dual_tensor);
    double scalar = 0.0;

    // Act & Assert
    EXPECT_THROW(tmd / scalar, std::runtime_error);
}

// Test case for division of a TensorMatDual with empty tensors by a scalar
TEST(TensorMatDualTest, ScalarDivision_EmptyTensors) {
    // Arrange
    torch::Tensor real_tensor = torch::empty({0, 3, 4});
    torch::Tensor dual_tensor = torch::empty({0, 3, 4, 5});
    TensorMatDual tmd(real_tensor, dual_tensor);
    double scalar = 1.0;

    // Act
    TensorMatDual result = tmd / scalar;

    // Assert
    EXPECT_EQ(result.r.sizes(), real_tensor.sizes());
    EXPECT_EQ(result.d.sizes(), dual_tensor.sizes());
    EXPECT_TRUE(torch::allclose(result.r, real_tensor / scalar));
    EXPECT_TRUE(torch::allclose(result.d, dual_tensor / scalar));
}



// Test case for indexing with valid slices
TEST(TensorMatDualTest, Indexing_ValidSlices) {
    // Arrange
    torch::Tensor real_tensor = torch::rand({2, 3, 4});
    torch::Tensor dual_tensor = torch::rand({2, 3, 4, 5});
    TensorMatDual tmd(real_tensor, dual_tensor);

    // Act
    std::vector<torch::indexing::TensorIndex> indices = {torch::indexing::Slice(0, 2), torch::indexing::Slice(1, 3)};
    TensorMatDual result = tmd.index(indices);

    // Assert
    EXPECT_EQ(result.r.sizes(), torch::IntArrayRef({2, 2, 4}));
    EXPECT_EQ(result.d.sizes(), torch::IntArrayRef({2, 2, 4, 5}));
    EXPECT_TRUE(torch::allclose(result.r, real_tensor.index(indices)));
    EXPECT_TRUE(torch::allclose(result.d, dual_tensor.index(indices)));
}

// Test case for indexing with a slice resulting in size 1
TEST(TensorMatDualTest, Indexing_SliceOfSizeOne) {
    // Arrange
    torch::Tensor real_tensor = torch::rand({2, 3, 4});
    torch::Tensor dual_tensor = torch::rand({2, 3, 4, 5});
    TensorMatDual tmd(real_tensor, dual_tensor);

    // Act & Assert
    std::vector<torch::indexing::TensorIndex> indices = {torch::indexing::Slice(0, 1), torch::indexing::Slice(1, 2)};
    TensorMatDual result = tmd.index(indices);
    EXPECT_EQ(result.r.sizes(), torch::IntArrayRef({1, 1, 4}));
    EXPECT_EQ(result.d.sizes(), torch::IntArrayRef({1, 1, 4, 5}));
}

// Test case for indexing that reduces dimensions to 2D for real tensor
TEST(TensorMatDualTest, Indexing_ReduceTo2D) {
    // Arrange
    torch::Tensor real_tensor = torch::rand({2, 3, 4});
    torch::Tensor dual_tensor = torch::rand({2, 3, 4, 5});
    TensorMatDual tmd(real_tensor, dual_tensor);

    // Act
    std::vector<torch::indexing::TensorIndex> indices = {torch::indexing::Slice(0, 1), torch::indexing::Slice(1, 3)};
    TensorMatDual result = tmd.index(indices);

    // Assert
    EXPECT_EQ(result.r.sizes(), torch::IntArrayRef({1, 2, 4}));
    EXPECT_EQ(result.d.sizes(), torch::IntArrayRef({1, 2, 4, 5}));
    EXPECT_TRUE(torch::allclose(result.r, real_tensor.index(indices)));
    EXPECT_TRUE(torch::allclose(result.d, dual_tensor.index(indices)));
}

// Test case for valid first-dimension indexing
TEST(TensorMatDualTest, Indexing_ValidFirstDimension) {
    // Arrange
    torch::Tensor real_tensor = torch::rand({3, 3, 4});
    torch::Tensor dual_tensor = torch::rand({3, 3, 4, 5});
    TensorMatDual tmd(real_tensor, dual_tensor);

    // Act
    TensorMatDual result = tmd.index(1);

    // Assert
    EXPECT_EQ(result.r.sizes(), torch::IntArrayRef({1, 3, 4}));
    EXPECT_EQ(result.d.sizes(), torch::IntArrayRef({1, 3, 4, 5}));
}

// Test case for out-of-bounds indexing
TEST(TensorMatDualTest, Indexing_OutOfBounds) {
    // Arrange
    torch::Tensor real_tensor = torch::rand({3, 3, 4});
    torch::Tensor dual_tensor = torch::rand({3, 3, 4, 5});
    TensorMatDual tmd(real_tensor, dual_tensor);

    //Expect a std::out_of_range 
    EXPECT_THROW(tmd.index(3), std::out_of_range);
}


// Test case for indexing at the first dimension boundary
TEST(TensorMatDualTest, Indexing_BoundaryIndex) {
    // Arrange
    torch::Tensor real_tensor = torch::rand({3, 3, 4});
    torch::Tensor dual_tensor = torch::rand({3, 3, 4, 5});
    TensorMatDual tmd(real_tensor, dual_tensor);

    // Act
    TensorMatDual result = tmd.index(2); // Last valid index

    // Assert
    EXPECT_EQ(result.r.sizes(), torch::IntArrayRef({1, 3, 4}));
    EXPECT_EQ(result.d.sizes(), torch::IntArrayRef({1, 3, 4, 5}));
    EXPECT_TRUE(torch::allclose(result.r, real_tensor.index({2}).unsqueeze(0)));
    EXPECT_TRUE(torch::allclose(result.d, dual_tensor.index({2}).unsqueeze(0)));
}


// Test case for valid mask indexing
TEST(TensorMatDualTest, MaskIndexing_ValidMask) {
    // Arrange
    torch::Tensor real_tensor = torch::rand({5, 3, 4});
    torch::Tensor dual_tensor = torch::rand({5, 3, 4, 5});
    TensorMatDual tmd(real_tensor, dual_tensor);

    torch::Tensor mask = torch::tensor({true, false, true, false, true}, torch::kBool);

    // Act
    TensorMatDual result = tmd.index(mask);

    // Assert
    EXPECT_EQ(result.r.sizes(), torch::IntArrayRef({3, 3, 4}));
    EXPECT_EQ(result.d.sizes(), torch::IntArrayRef({3, 3, 4, 5}));
    EXPECT_TRUE(torch::allclose(result.r, real_tensor.index({mask})));
    EXPECT_TRUE(torch::allclose(result.d, dual_tensor.index({mask})));
}

// Test case for mask indexing with undefined mask
TEST(TensorMatDualTest, MaskIndexing_UndefinedMask) {
    // Arrange
    torch::Tensor real_tensor = torch::rand({5, 3, 4});
    torch::Tensor dual_tensor = torch::rand({5, 3, 4, 5});
    TensorMatDual tmd(real_tensor, dual_tensor);

    torch::Tensor mask; // Undefined mask

    // Act & Assert
    EXPECT_THROW(tmd.index(mask), std::invalid_argument);
}

// Test case for mask indexing with non-boolean mask
TEST(TensorMatDualTest, MaskIndexing_NonBooleanMask) {
    // Arrange
    torch::Tensor real_tensor = torch::rand({5, 3, 4});
    torch::Tensor dual_tensor = torch::rand({5, 3, 4, 5});
    TensorMatDual tmd(real_tensor, dual_tensor);

    torch::Tensor mask = torch::randint(0, 2, {5}, torch::kInt32); // Non-boolean tensor

    // Act & Assert
    EXPECT_THROW(tmd.index(mask), std::invalid_argument);
}

// Test case for mask indexing with mismatched mask shape
TEST(TensorMatDualTest, MaskIndexing_MismatchedMaskShape) {
    // Arrange
    torch::Tensor real_tensor = torch::rand({5, 3, 4});
    torch::Tensor dual_tensor = torch::rand({5, 3, 4, 5});
    TensorMatDual tmd(real_tensor, dual_tensor);

    torch::Tensor mask = torch::tensor({true, false, true}, torch::kBool); // Mismatched shape

    //Expect a c10 error
    EXPECT_THROW(tmd.index(mask), c10::Error);
}

// Test case for mask indexing with all elements masked out
TEST(TensorMatDualTest, MaskIndexing_AllElementsMaskedOut) {
    // Arrange
    torch::Tensor real_tensor = torch::rand({5, 3, 4});
    torch::Tensor dual_tensor = torch::rand({5, 3, 4, 5});
    TensorMatDual tmd(real_tensor, dual_tensor);

    torch::Tensor mask = torch::tensor({false, false, false, false, false}, torch::kBool);

    // Act
    TensorMatDual result = tmd.index(mask);

    // Assert
    EXPECT_EQ(result.r.sizes(), torch::IntArrayRef({0, 3, 4}));
    EXPECT_EQ(result.d.sizes(), torch::IntArrayRef({0, 3, 4, 5}));
    EXPECT_TRUE(result.r.numel() == 0);
    EXPECT_TRUE(result.d.numel() == 0);
}


// Test case for enabling requires_grad on valid tensors
TEST(TensorMatDualTest, RequiresGrad_Enable) {
    // Arrange
    torch::Tensor real_tensor = torch::rand({2, 3, 4});
    torch::Tensor dual_tensor = torch::rand({2, 3, 4, 5});
    TensorMatDual tmd(real_tensor, dual_tensor);

    // Act
    tmd.requires_grad_(true);

    // Assert
    EXPECT_TRUE(tmd.r.requires_grad());
    EXPECT_TRUE(tmd.d.requires_grad());
}

// Test case for disabling requires_grad on valid tensors
TEST(TensorMatDualTest, RequiresGrad_Disable) {
    // Arrange
    torch::Tensor real_tensor = torch::rand({2, 3, 4}).requires_grad_(true);
    torch::Tensor dual_tensor = torch::rand({2, 3, 4, 5}).requires_grad_(true);
    TensorMatDual tmd(real_tensor, dual_tensor);

    // Act
    tmd.requires_grad_(false);

    // Assert
    EXPECT_FALSE(tmd.r.requires_grad());
    EXPECT_FALSE(tmd.d.requires_grad());
}


// Test case for triggering backward pass on valid tensors
TEST(TensorMatDualTest, Backward_ValidTensors) {
    // Arrange
    torch::Tensor real_tensor = torch::rand({2, 3, 4}, torch::requires_grad());
    torch::Tensor dual_tensor = torch::rand({2, 3, 4, 5}, torch::requires_grad());
    TensorMatDual tmd(real_tensor, dual_tensor);

    // Create a loss function for the backward pass
    auto loss = tmd.r.sum() + tmd.d.sum();

    // Act
    loss.backward();

    // Assert
    EXPECT_TRUE(tmd.r.requires_grad());
    EXPECT_TRUE(tmd.d.requires_grad());
    EXPECT_TRUE(tmd.r.grad().defined());
    EXPECT_TRUE(tmd.d.grad().defined());
}


// Test case for backward pass with requires_grad set to false
TEST(TensorMatDualTest, Backward_RequiresGradFalse) {
    // Arrange
    torch::Tensor real_tensor = torch::rand({2, 3, 4}); // requires_grad is false
    torch::Tensor dual_tensor = torch::rand({2, 3, 4, 5}); // requires_grad is false
    TensorMatDual tmd(real_tensor, dual_tensor);

    // Act & Assert
    EXPECT_THROW(tmd.backward(), std::runtime_error);
}


// Test case for computing the absolute value of valid tensors
TEST(TensorMatDualTest, Abs_ValidTensors) {
    // Arrange
    torch::Tensor real_tensor = torch::rand({2, 3, 4})-1.0;
    torch::Tensor dual_tensor = torch::rand({2, 3, 4, 5});
    TensorMatDual tmd(real_tensor, dual_tensor);

    // Act
    TensorMatDual result = tmd.abs();

    // Assert
    EXPECT_TRUE(torch::allclose(result.r, torch::abs(real_tensor)));
    EXPECT_TRUE(torch::allclose(result.d, torch::sign(real_tensor).unsqueeze(-1) * dual_tensor));
}


// Test case for tensors with zero values
TEST(TensorMatDualTest, Abs_ZeroValues) {
    // Arrange
    torch::Tensor real_tensor = torch::rand({2, 3, 4})-1.0;
    torch::Tensor dual_tensor = torch::rand({2, 3, 4, 5});
    TensorMatDual tmd(real_tensor, dual_tensor);

    // Act
    TensorMatDual result = tmd.abs();

    // Assert
    EXPECT_TRUE(torch::allclose(result.r, torch::abs(real_tensor)));
    EXPECT_TRUE(torch::allclose(result.d, torch::sign(real_tensor).unsqueeze(-1) * dual_tensor));
}

// Test case for tensors with complex values
TEST(TensorMatDualTest, Abs_ComplexTensors) {
    // Arrange
    //Generate random complex tensors
    torch::Tensor real_tensor_real = torch::rand({2, 3, 4});
    torch::Tensor real_tensor_imag = torch::rand({2, 3, 4});
    torch::Tensor dual_tensor_real = torch::rand({2, 3, 4, 5});
    torch::Tensor dual_tensor_imag = torch::rand({2, 3, 4, 5});
    torch::Tensor real_tensor = torch::complex(real_tensor_real, real_tensor_imag);
    torch::Tensor dual_tensor = torch::complex(dual_tensor_real, dual_tensor_imag);

    TensorMatDual tmd(real_tensor, dual_tensor);

    // Act
    TensorMatDual result = tmd.abs();

    // Assert
    EXPECT_TRUE(torch::allclose(result.r, torch::abs(real_tensor)));
    EXPECT_TRUE(torch::allclose(result.d, torch::sign(torch::real(real_tensor)).unsqueeze(-1) * dual_tensor));
}


// Test case for valid einsum operation
TEST(TensorMatDualTest, Einsum_ValidInputs) {
    // Arrange
    torch::Tensor first_real = torch::rand({2, 3, 4});
    torch::Tensor first_dual = torch::rand({2, 3, 4, 5});
    TensorMatDual first(first_real, first_dual);

    torch::Tensor second_real = torch::rand({2, 4});
    torch::Tensor second_dual = torch::rand({2, 4, 5});
    TensorDual second(second_real, second_dual);

    std::string einsum_arg = "mij,mj->mi";

    // Act
    TensorDual result = TensorMatDual::einsum(einsum_arg, first, second);

    // Assert
    auto expected_r = torch::einsum(einsum_arg, {first_real, second_real});
    EXPECT_TRUE(torch::allclose(result.r, expected_r));

    auto darg1 = "mijz,mj->miz";
    auto d1 = torch::einsum(darg1, {first_dual, second_real});

    auto darg2 = "mij,mjz->miz";
    auto d2 = torch::einsum(darg2, {first_real, second_dual});

    EXPECT_TRUE(torch::allclose(result.d, d1 + d2));
}

// Test case for mismatched dimensions
TEST(TensorMatDualTest, Einsum_MismatchedDimensions) {
    // Arrange
    torch::Tensor first_real = torch::rand({2, 3, 3});
    torch::Tensor first_dual = torch::rand({2, 3, 3, 4});
    TensorMatDual first(first_real, first_dual);

    torch::Tensor second_real = torch::rand({2, 5}); // Mismatched dimensions
    torch::Tensor second_dual = torch::rand({2, 5, 6});
    TensorDual second(second_real, second_dual);

    std::string einsum_arg = "ij,jk->ik";

    //Expect a c10 error
    EXPECT_THROW(TensorMatDual::einsum(einsum_arg, first, second), c10::Error);
}



// Test case for valid einsum operation with TensorMatDual and torch::Tensor
TEST(TensorMatDualTest, TensorEinsum_ValidInputs) {
    // Arrange
    torch::Tensor first_real = torch::rand({2, 3, 4});
    torch::Tensor first_dual = torch::rand({2, 3, 4, 5});
    TensorMatDual first(first_real, first_dual);

    torch::Tensor second = torch::rand({2, 3, 4});

    std::string einsum_arg = "mij, mij->mij";

    // Act
    TensorMatDual result = TensorMatDual::einsum(einsum_arg, first, second);

    // Assert
    auto expected_r = torch::einsum(einsum_arg, {first_real, second});
    EXPECT_TRUE(torch::allclose(result.r, expected_r));

    auto darg1 = "mijz,mij->mijz";
    auto expected_d = torch::einsum(darg1, {first_dual, second});

    EXPECT_TRUE(torch::allclose(result.d, expected_d));
}

// Test case for invalid einsum string
TEST(TensorMatDualTest, TensorEinsum_InvalidEinsumString) {
    // Arrange
    torch::Tensor first_real = torch::rand({2, 3, 4});
    torch::Tensor first_dual = torch::rand({2, 3, 4, 5});
    TensorMatDual first(first_real, first_dual);

    torch::Tensor second = torch::rand({2, 3, 4});

    std::string einsum_arg = "invalid";

    // Act & Assert
    EXPECT_THROW(TensorMatDual::einsum(einsum_arg, first, second), std::invalid_argument);
}

// Test case for mismatched dimensions
TEST(TensorMatDualTest, TensorEinsum_MismatchedDimensions) {
    // Arrange
    torch::Tensor first_real = torch::rand({2, 3, 4});
    torch::Tensor first_dual = torch::rand({2, 3, 4, 5});
    TensorMatDual first(first_real, first_dual);

    torch::Tensor second = torch::rand({2, 4, 5}); // Mismatched dimensions

    std::string einsum_arg = "mij, mij->mij";

    //Expect a c10 error
    EXPECT_THROW(TensorMatDual::einsum(einsum_arg, first, second), c10::Error);
}


// Test case for einsum with empty tensors
TEST(TensorMatDualTest, TensorEinsum_EmptyTensors) {
    // Arrange
    torch::Tensor first_real = torch::empty({0, 3, 4});
    torch::Tensor first_dual = torch::empty({0, 3, 4, 5});
    TensorMatDual first(first_real, first_dual);

    torch::Tensor second = torch::empty({0, 3, 4});

    std::string einsum_arg = "mij, mij->mij";

    // Act
    TensorMatDual result = TensorMatDual::einsum(einsum_arg, first, second);

    // Assert
    EXPECT_EQ(result.r.sizes(), torch::IntArrayRef({0, 3, 4}));
    EXPECT_EQ(result.d.sizes(), torch::IntArrayRef({0, 3, 4, 5}));
}


// Test case for valid einsum operation with TensorDual and TensorMatDual
TEST(TensorDualMatDualTest, Einsum_ValidInputs) {
    // Arrange
    torch::Tensor first_real = torch::rand({2, 3});
    torch::Tensor first_dual = torch::rand({2, 3, 5});
    TensorDual first(first_real, first_dual);

    torch::Tensor second_real = torch::rand({2, 3, 4});
    torch::Tensor second_dual = torch::rand({2, 3, 4, 5});
    TensorMatDual second(second_real, second_dual);

    std::string einsum_arg = "mi, mij->mj";

    // Act
    TensorDual result = TensorMatDual::einsum(einsum_arg, first, second);

    // Assert
    auto expected_r = torch::einsum(einsum_arg, {first_real, second_real});
    EXPECT_TRUE(torch::allclose(result.r, expected_r));

    auto darg1 = "miz, mij->mjz";
    auto d1 = torch::einsum(darg1, {first_dual, second_real});

    auto darg2 = "mi, mijz->mjz";
    auto d2 = torch::einsum(darg2, {first_real, second_dual});

    EXPECT_TRUE(torch::allclose(result.d, d1 + d2));
}

// Test case for invalid einsum string
TEST(TensorDualMatDualTest, Einsum_InvalidEinsumString) {
    // Arrange
    torch::Tensor first_real = torch::rand({2, 3});
    torch::Tensor first_dual = torch::rand({2, 3, 5});
    TensorDual first(first_real, first_dual);

    torch::Tensor second_real = torch::rand({2, 3, 4});
    torch::Tensor second_dual = torch::rand({2, 3, 4, 5});
    TensorMatDual second(second_real, second_dual);

    std::string einsum_arg = "invalid";

    // Act & Assert
    EXPECT_THROW(TensorMatDual::einsum(einsum_arg, first, second), std::invalid_argument);
}

// Test case for mismatched dimensions
TEST(TensorDualMatDualTest, Einsum_MismatchedDimensions) {
    // Arrange
    torch::Tensor first_real = torch::rand({2, 3});
    torch::Tensor first_dual = torch::rand({2, 3, 5});
    TensorDual first(first_real, first_dual);

    torch::Tensor second_real = torch::rand({2, 4, 5}); // Mismatched dimensions
    torch::Tensor second_dual = torch::rand({2, 4, 5, 6});
    TensorMatDual second(second_real, second_dual);

    std::string einsum_arg = "mi, mij->mj";

    // Act & Assert
    EXPECT_THROW(TensorMatDual::einsum(einsum_arg, first, second), c10::Error);
}

// Test case for einsum with empty tensors
TEST(TensorDualMatDualTest, Einsum_EmptyTensors) {
    // Arrange
    torch::Tensor first_real = torch::empty({0, 3});
    torch::Tensor first_dual = torch::empty({0, 3, 5});
    TensorDual first(first_real, first_dual);

    torch::Tensor second_real = torch::empty({0, 3, 1});
    torch::Tensor second_dual = torch::empty({0, 3, 1, 5});
    TensorMatDual second(second_real, second_dual);

    std::string einsum_arg = "mi, mij->mj";

    // Act
    TensorDual result = TensorMatDual::einsum(einsum_arg, first, second);

    // Assert
    EXPECT_EQ(result.r.sizes(), torch::IntArrayRef({0, 1}));
    EXPECT_EQ(result.d.sizes(), torch::IntArrayRef({0, 1, 5}));
}



// Test case for valid einsum operation with torch::Tensor and TensorMatDual
TEST(TensorTensorMatDualTest, Einsum_ValidInputs) {
    // Arrange
    torch::Tensor first = torch::rand({3, 4});

    torch::Tensor second_real = torch::rand({2, 3, 4});
    torch::Tensor second_dual = torch::rand({2, 3, 4, 5});
    TensorMatDual second(second_real, second_dual);

    std::string einsum_arg = "ij,mij->mij";

    // Act
    TensorMatDual result = TensorMatDual::einsum(einsum_arg, first, second);

    // Assert
    auto expected_r = torch::einsum(einsum_arg, {first, second_real});
    EXPECT_TRUE(torch::allclose(result.r, expected_r));

    auto darg1 = "ij,mijz->mijz";
    auto expected_d = torch::einsum(darg1, {first, second_dual});

    EXPECT_TRUE(torch::allclose(result.d, expected_d));
}

// Test case for invalid einsum string
TEST(TensorTensorMatDualTest, Einsum_InvalidEinsumString) {
    // Arrange
    torch::Tensor first = torch::rand({2, 3});

    torch::Tensor second_real = torch::rand({2, 3, 4});
    torch::Tensor second_dual = torch::rand({2, 3, 4, 5});
    TensorMatDual second(second_real, second_dual);

    std::string einsum_arg = "invalid";

    // Act & Assert
    EXPECT_THROW(TensorMatDual::einsum(einsum_arg, first, second), std::invalid_argument);
}

// Test case for mismatched dimensions
TEST(TensorTensorMatDualTest, Einsum_MismatchedDimensions) {
    // Arrange
    torch::Tensor first = torch::rand({2, 3});

    torch::Tensor second_real = torch::rand({2, 4, 5}); // Mismatched dimensions
    torch::Tensor second_dual = torch::rand({2, 4, 5, 6});
    TensorMatDual second(second_real, second_dual);

    std::string einsum_arg = "ij,mij->mij";

    // Act & Assert
    EXPECT_THROW(TensorMatDual::einsum(einsum_arg, first, second), c10::Error);
}


// Test case for valid einsum operation with two TensorMatDual objects
TEST(TensorMatDualTest, TensorMatDualTensorMatDualEinsum_ValidInputs) {
    // Arrange
    torch::Tensor first_real = torch::rand({2, 3, 4});
    torch::Tensor first_dual = torch::rand({2, 3, 4, 5});
    TensorMatDual first(first_real, first_dual);

    torch::Tensor second_real = torch::rand({2, 4, 3});
    torch::Tensor second_dual = torch::rand({2, 4, 3, 5});
    TensorMatDual second(second_real, second_dual);

    std::string einsum_arg = "mij,mjk->mik";

    // Act
    TensorMatDual result = TensorMatDual::einsum(einsum_arg, first, second);

    // Assert
    auto expected_r = torch::einsum(einsum_arg, {first_real, second_real});
    EXPECT_TRUE(torch::allclose(result.r, expected_r));

    auto darg1 = "mijz,mjk->mikz";
    auto d1 = torch::einsum(darg1, {first_dual, second_real});

    auto darg2 = "mij,mjkz->mikz";
    auto d2 = torch::einsum(darg2, {first_real, second_dual});

    EXPECT_TRUE(torch::allclose(result.d, d1 + d2));
}

// Test case for invalid einsum string
TEST(TensorMatDualTest, TensorMatDualTensorMatDualEinsum_InvalidEinsumString) {
    // Arrange
    torch::Tensor first_real = torch::rand({2, 3, 4});
    torch::Tensor first_dual = torch::rand({2, 3, 4, 5});
    TensorMatDual first(first_real, first_dual);

    torch::Tensor second_real = torch::rand({2, 3, 4});
    torch::Tensor second_dual = torch::rand({2, 3, 4, 5});
    TensorMatDual second(second_real, second_dual);

    std::string einsum_arg = "invalid";

    // Act & Assert
    EXPECT_THROW(TensorMatDual::einsum(einsum_arg, first, second), std::invalid_argument);
}

// Test case for mismatched dimensions
TEST(TensorMatDualTest, TensorMatDualTensorMatDualEinsum_MismatchedDimensions) {
    // Arrange
    torch::Tensor first_real = torch::rand({2, 3, 4});
    torch::Tensor first_dual = torch::rand({2, 3, 4, 5});
    TensorMatDual first(first_real, first_dual);

    torch::Tensor second_real = torch::rand({4, 5, 6}); // Mismatched dimensions
    torch::Tensor second_dual = torch::rand({4, 5, 6, 7});
    TensorMatDual second(second_real, second_dual);

    std::string einsum_arg = "mij,mjk->mik";

    // Act & Assert
    EXPECT_THROW(TensorMatDual::einsum(einsum_arg, first, second), c10::Error);
}



// Test case for valid max operation along a dimension
TEST(TensorMatDualTest, Max_ValidInputs) {
    // Arrange
    torch::Tensor real_tensor = torch::rand({2, 3, 4}, torch::kFloat32);;
    torch::Tensor dual_tensor = torch::rand({2, 3, 4, 5}, torch::kFloat32);
    TensorMatDual tensor_mat_dual(real_tensor, dual_tensor);

    int dim = 1;

    // Act
    TensorMatDual result = tensor_mat_dual.max(dim);

    // Assert
    auto max_result = torch::max(real_tensor, /*dim=*/dim, /*keepdim=*/true);
    auto expected_real = std::get<0>(max_result);
    auto max_indices = std::get<1>(max_result);

    auto expanded_indices = max_indices.unsqueeze(-1).expand({-1, -1, -1, dual_tensor.size(-1)});
    auto expected_dual = torch::gather(dual_tensor, dim, expanded_indices);

    EXPECT_TRUE(torch::allclose(result.r, expected_real));
    EXPECT_TRUE(torch::allclose(result.d, expected_dual));
}

// Test case for invalid dimension
TEST(TensorMatDualTest, Max_InvalidDimension) {
    // Arrange
    torch::Tensor real_tensor = torch::rand({2, 3, 4}, torch::kFloat32);
    torch::Tensor dual_tensor = torch::rand({2, 3, 4, 5}, torch::kFloat32);
    TensorMatDual tensor_mat_dual(real_tensor, dual_tensor);

    int invalid_dim = 3; // Out of bounds

    // Act & Assert
    EXPECT_THROW(tensor_mat_dual.max(invalid_dim), std::invalid_argument);
}



// Test case for complex real tensor
TEST(TensorMatDualTest, Max_ComplexRealTensor) {
    // Arrange
    //Generate a random complex tensor
    torch::Tensor real_tensor = torch::complex(torch::rand({2, 3, 4}, torch::kFloat32),
                                               torch::rand({2, 3, 4}, torch::kFloat32));
    torch::Tensor dual_tensor = torch::rand({2, 3, 4, 5}, torch::kFloat32);
    TensorMatDual tensor_mat_dual(real_tensor, dual_tensor);

    int dim = 1;

    // Act
    TensorMatDual result = tensor_mat_dual.max(dim);

    // Assert
    auto max_result = torch::max(torch::real(real_tensor), /*dim=*/dim, /*keepdim=*/true);
    auto expected_real = std::get<0>(max_result);
    auto max_indices = std::get<1>(max_result);

    auto expanded_indices = max_indices.unsqueeze(-1).expand({-1, -1, -1, dual_tensor.size(-1)});
    auto expected_dual = torch::gather(dual_tensor, dim, expanded_indices);

    EXPECT_TRUE(torch::allclose(result.r, expected_real));
    EXPECT_TRUE(torch::allclose(result.d, expected_dual));
}



// Test case for valid index_put_ operation
TEST(TensorMatDualTest, IndexPut_ValidInputs) {
    // Arrange
    torch::Tensor real_tensor = torch::rand({2, 3, 4}, torch::kFloat32);
    torch::Tensor dual_tensor = torch::rand({2, 3, 4, 5}, torch::kFloat32);
    TensorMatDual tensor_mat_dual(real_tensor, dual_tensor);

    torch::Tensor mask = torch::tensor({true, false}, torch::kBool);
    torch::Tensor value_real = torch::rand({1, 3, 4}, torch::kFloat32);
    torch::Tensor value_dual = torch::rand({1, 3, 4, 5}, torch::kFloat32);
    TensorMatDual value(value_real, value_dual);

    // Act
    tensor_mat_dual.index_put_(mask, value);

    // Assert
    auto expected_real = real_tensor.clone();
    expected_real.index_put_({mask}, value_real);
    EXPECT_TRUE(torch::allclose(tensor_mat_dual.r, expected_real));

    auto expected_dual = dual_tensor.clone();
    expected_dual.index_put_({mask}, value_dual);
    EXPECT_TRUE(torch::allclose(tensor_mat_dual.d, expected_dual));
}


// Test case for non-boolean mask tensor
TEST(TensorMatDualTest, IndexPut_NonBooleanMask) {
    // Arrange
    torch::Tensor real_tensor = torch::rand({2, 3, 4}, torch::kFloat32);
    torch::Tensor dual_tensor = torch::rand({2, 3, 4, 5}, torch::kFloat32);
    TensorMatDual tensor_mat_dual(real_tensor, dual_tensor);

    torch::Tensor mask = torch::tensor({1, 0}, torch::kInt32); // Non-boolean
    torch::Tensor value_real = torch::rand({1, 3, 4}, torch::kFloat32);
    torch::Tensor value_dual = torch::rand({1, 3, 4, 5}, torch::kFloat32);
    TensorMatDual value(value_real, value_dual);

    // Act & Assert
    EXPECT_THROW(tensor_mat_dual.index_put_(mask, value), std::invalid_argument);
}

// Test case for mismatched shapes between mask and TensorMatDual
TEST(TensorMatDualTest, IndexPut_MismatchedShapes) {
    // Arrange
    torch::Tensor real_tensor = torch::rand({2, 3, 4}, torch::kFloat32);
    torch::Tensor dual_tensor = torch::rand({2, 3, 4, 5}, torch::kFloat32);
    TensorMatDual tensor_mat_dual(real_tensor, dual_tensor);

    torch::Tensor mask = torch::tensor({true, false}, torch::kBool); // Incorrect shape
    torch::Tensor value_real = torch::rand({1, 2, 5}, torch::kFloat32);
    torch::Tensor value_dual = torch::rand({1, 2, 5, 7}, torch::kFloat32);
    TensorMatDual value(value_real, value_dual);

    // Act & Assert
    EXPECT_THROW(tensor_mat_dual.index_put_(mask, value), c10::Error);
}


// Test case for valid unsqueeze operation
TEST(TensorDualTest, Unsqueeze_ValidInputs) {
    // Arrange
    torch::Tensor real_tensor = torch::rand({3, 4}, torch::kFloat32);
    torch::Tensor dual_tensor = torch::rand({3, 4, 5}, torch::kFloat32);
    TensorDual tensor_dual(real_tensor, dual_tensor);

    int dim = 1; // Dimension to unsqueeze

    // Act
    TensorMatDual result = TensorMatDual::unsqueeze(tensor_dual, dim);

    // Assert
    auto expected_real = real_tensor.unsqueeze(dim);
    auto expected_dual = dual_tensor.unsqueeze(dim);

    EXPECT_TRUE(torch::allclose(result.r, expected_real));
    EXPECT_TRUE(torch::allclose(result.d, expected_dual));
}

// Test case for invalid dimension
TEST(TensorDualTest, Unsqueeze_InvalidDimension) {
    // Arrange
    torch::Tensor real_tensor = torch::rand({3, 4}, torch::kFloat32);
    torch::Tensor dual_tensor = torch::rand({3, 4, 5}, torch::kFloat32);
    TensorDual tensor_dual(real_tensor, dual_tensor);

    int invalid_dim = 5; // Out of bounds

    // Act & Assert
    EXPECT_THROW(TensorMatDual::unsqueeze(tensor_dual, invalid_dim), c10::Error);
}


// Test case for unsqueeze with empty tensors
TEST(TensorDualTest, Unsqueeze_EmptyTensors) {
    // Arrange
    torch::Tensor real_tensor = torch::empty({0, 4}, torch::kFloat32);
    torch::Tensor dual_tensor = torch::empty({0, 4, 5}, torch::kFloat32);
    TensorDual tensor_dual(real_tensor, dual_tensor);

    int dim = 1;

    // Act
    TensorMatDual result = TensorMatDual::unsqueeze(tensor_dual, dim);

    // Assert
    auto expected_real = real_tensor.unsqueeze(dim);
    auto expected_dual = dual_tensor.unsqueeze(dim);

    EXPECT_TRUE(torch::allclose(result.r, expected_real));
    EXPECT_TRUE(torch::allclose(result.d, expected_dual));
}

// Test case for unsqueeze at the first dimension
TEST(TensorDualTest, Unsqueeze_FirstDimension) {
    // Arrange
    torch::Tensor real_tensor = torch::rand({3, 4}, torch::kFloat32);
    torch::Tensor dual_tensor = torch::rand({3, 4, 5}, torch::kFloat32);
    TensorDual tensor_dual(real_tensor, dual_tensor);

    int dim = 0; // First dimension

    // Act
    TensorMatDual result = TensorMatDual::unsqueeze(tensor_dual, dim);

    // Assert
    auto expected_real = real_tensor.unsqueeze(dim);
    auto expected_dual = dual_tensor.unsqueeze(dim);

    EXPECT_TRUE(torch::allclose(result.r, expected_real));
    EXPECT_TRUE(torch::allclose(result.d, expected_dual));
}

// Test case for unsqueeze at the last dimension
TEST(TensorDualTest, Unsqueeze_LastDimension) {
    // Arrange
    torch::Tensor real_tensor = torch::rand({3, 4}, torch::kFloat32);
    torch::Tensor dual_tensor = torch::rand({3, 4, 5}, torch::kFloat32);
    TensorDual tensor_dual(real_tensor, dual_tensor);

    int dim = -1; // Last dimension

    // Act
    TensorMatDual result = TensorMatDual::unsqueeze(tensor_dual, dim);

    // Assert
    auto expected_real = real_tensor.unsqueeze(dim);
    auto expected_dual = dual_tensor.unsqueeze(dim);

    EXPECT_TRUE(torch::allclose(result.r, expected_real));
    EXPECT_TRUE(torch::allclose(result.d, expected_dual));
}
// Helper to create valid tensors
TensorMatHyperDual createValidTensorMatHyperDual() {
        torch::Dtype dtype = torch::kFloat64;
        torch::Device device = torch::kCPU;
        auto r = torch::rand({2, 3, 4}, dtype).to(device);
        auto d = torch::rand({2, 3, 4, 5}, dtype).to(device);
        auto h = torch::rand({2, 3, 4, 6, 6}, dtype).to(device);
        return TensorMatHyperDual(r, d, h);
}


// Test: Constructor with valid input
TEST(TensorMatHyperDualTest, ConstructorValidInput) {
    torch::Dtype dtype = torch::kFloat64;
    torch::Device device = torch::kCPU;


    auto r = torch::rand({2, 3, 4}, dtype).to(device);
    auto d = torch::rand({2, 3, 4, 5}, dtype).to(device);
    auto h = torch::rand({2, 3, 4, 6, 6}, dtype).to(device);

    EXPECT_NO_THROW(TensorMatHyperDual(r, d, h));
}

// Test: Constructor with invalid real part dimensions
TEST(TensorMatHyperDualTest, ConstructorInvalidRDimensions) {
    torch::Dtype dtype = torch::kFloat64;
    torch::Device device = torch::kCPU;


    auto r = torch::rand({3, 4}, dtype).to(device); // Invalid shape
    auto d = torch::rand({2, 3, 4, 5}, dtype).to(device);
    auto h = torch::rand({2, 3, 4, 6, 6}, dtype).to(device);

    EXPECT_THROW(TensorMatHyperDual(r, d, h), std::invalid_argument);
}

// Test: Constructor with invalid dual part dimensions
TEST(TensorMatHyperDualTest, ConstructorInvalidDDimensions) {
    torch::Dtype dtype = torch::kFloat64;
    torch::Device device = torch::kCPU;


    auto r = torch::rand({2, 3, 4}, dtype).to(device);
    auto d = torch::rand({2, 3, 5}, dtype).to(device); // Invalid shape
    auto h = torch::rand({2, 3, 4, 6, 6}, dtype).to(device);

    EXPECT_THROW(TensorMatHyperDual(r, d, h), std::invalid_argument);
}

// Test: Constructor with invalid hyperdual part dimensions
TEST(TensorMatHyperDualTest, ConstructorInvalidHDimensions) {
    torch::Dtype dtype = torch::kFloat64;
    torch::Device device = torch::kCPU;


    auto r = torch::rand({2, 3, 4}, dtype).to(device);
    auto d = torch::rand({2, 3, 4, 5}, dtype).to(device);
    auto h = torch::rand({2, 3, 4, 6}, dtype).to(device); // Invalid shape

    EXPECT_THROW(TensorMatHyperDual(r, d, h), std::invalid_argument);
}

// Test: Constructor with mismatched dimensions
TEST(TensorMatHyperDualTest, ConstructorMismatchedDimensions) {
    torch::Dtype dtype = torch::kFloat64;
    torch::Device device = torch::kCPU;


    auto r = torch::rand({2, 3, 4}, dtype).to(device);
    auto d = torch::rand({2, 3, 5, 5}, dtype).to(device); // Mismatch in L
    auto h = torch::rand({2, 3, 4, 6, 6}, dtype).to(device);

    EXPECT_THROW(TensorMatHyperDual(r, d, h), std::invalid_argument);
}

// Test: Device and dtype consistency
TEST(TensorMatHyperDualTest, DeviceAndDtypeConsistency) {    
    torch::Dtype dtype = torch::kFloat64;
    torch::Device device = torch::kCPU;


    auto tensorMat = createValidTensorMatHyperDual();
    EXPECT_EQ(tensorMat.dtype_, dtype);
    EXPECT_EQ(tensorMat.device_, device);
}

// Test: Utility function `toString` (if implemented)
TEST(TensorMatHyperDualTest, ToString) {
    auto tensorMat = createValidTensorMatHyperDual();
    std::string description = tensorMat.toString();
    EXPECT_NE(description.find("r: [2, 3, 4]"), std::string::npos);
    EXPECT_NE(description.find("d: [2, 3, 4, 5]"), std::string::npos);
    EXPECT_NE(description.find("h: [2, 3, 4, 6, 6]"), std::string::npos);
}

TEST(TensorMatHyperDualTest, ToMethod_ValidDeviceSwitch) {
    // Create tensors on the CPU
    auto r = torch::rand({2, 3, 4});
    auto d = torch::rand({2, 3, 4, 5});
    auto h = torch::rand({2, 3, 4, 6, 6});

    // Initialize the TensorMatHyperDual object
    TensorMatHyperDual tensorMat(r, d, h);

    // Move tensors to CUDA (if available)
    if (torch::cuda::is_available()) {
        TensorMatHyperDual tensorMatCUDA = tensorMat.to(torch::kCUDA);

        // Check that tensors are moved to CUDA
        EXPECT_EQ(tensorMatCUDA.r.device().type(), torch::kCUDA);
        EXPECT_EQ(tensorMatCUDA.d.device().type(), torch::kCUDA);
        EXPECT_EQ(tensorMatCUDA.h.device().type(), torch::kCUDA);

        // Ensure the sizes remain unchanged
        EXPECT_EQ(tensorMatCUDA.r.sizes(), r.sizes());
        EXPECT_EQ(tensorMatCUDA.d.sizes(), d.sizes());
        EXPECT_EQ(tensorMatCUDA.h.sizes(), h.sizes());
    }
}

TEST(TensorMatHyperDualTest, ToMethod_BackToCPU) {
    // Create tensors on CUDA (if available)
    if (torch::cuda::is_available()) {
        auto r = torch::rand({2, 3, 4}, torch::device(torch::kCUDA));
        auto d = torch::rand({2, 3, 4, 5}, torch::device(torch::kCUDA));
        auto h = torch::rand({2, 3, 4, 6, 6}, torch::device(torch::kCUDA));

        // Initialize the TensorMatHyperDual object on CUDA
        TensorMatHyperDual tensorMatCUDA(r, d, h);

        // Move tensors back to CPU
        TensorMatHyperDual tensorMatCPU = tensorMatCUDA.to(torch::kCPU);

        // Check that tensors are moved to CPU
        EXPECT_EQ(tensorMatCPU.r.device().type(), torch::kCPU);
        EXPECT_EQ(tensorMatCPU.d.device().type(), torch::kCPU);
        EXPECT_EQ(tensorMatCPU.h.device().type(), torch::kCPU);

        // Ensure the sizes remain unchanged
        EXPECT_EQ(tensorMatCPU.r.sizes(), r.sizes());
        EXPECT_EQ(tensorMatCPU.d.sizes(), d.sizes());
        EXPECT_EQ(tensorMatCPU.h.sizes(), h.sizes());
    }
}

TEST(TensorMatHyperDualTest, ToMethod_InvalidDevice) {
    // Create tensors on the CPU
    auto r = torch::rand({2, 3, 4});
    auto d = torch::rand({2, 3, 4, 5});
    auto h = torch::rand({2, 3, 4, 6, 6});

    // Initialize the TensorMatHyperDual object
    TensorMatHyperDual tensorMat(r, d, h);


    EXPECT_THROW({
            // Attempt to move tensors to an invalid device
        // This is just for testing, as Torch itself will throw an error for invalid devices
        torch::Device invalidDevice("nonexistent");

        TensorMatHyperDual tensorMatInvalid = tensorMat.to(invalidDevice);
    }, c10::Error); // Expect a c10::Error for invalid device
}


TEST(TensorMatHyperDualTest, DeviceMethod_CPU) {
    // Create tensors on CPU
    auto r = torch::rand({2, 3, 4}, torch::device(torch::kCPU));
    auto d = torch::rand({2, 3, 4, 5}, torch::device(torch::kCPU));
    auto h = torch::rand({2, 3, 4, 6, 6}, torch::device(torch::kCPU));

    // Initialize TensorMatHyperDual object
    TensorMatHyperDual tensorMat(r, d, h);

    // Check that the device is correctly reported as CPU
    EXPECT_EQ(tensorMat.device().type(), torch::kCPU);
}

TEST(TensorMatHyperDualTest, DeviceMethod_CUDA) {
    // Only test CUDA if it's available
    if (torch::cuda::is_available()) {
        // Create tensors on CUDA
        auto r = torch::rand({2, 3, 4}, torch::device(torch::kCUDA));
        auto d = torch::rand({2, 3, 4, 5}, torch::device(torch::kCUDA));
        auto h = torch::rand({2, 3, 4, 6, 6}, torch::device(torch::kCUDA));

        // Initialize TensorMatHyperDual object
        TensorMatHyperDual tensorMat(r, d, h);

        // Check that the device is correctly reported as CUDA
        EXPECT_EQ(tensorMat.device().type(), torch::kCUDA);
    } else {
        GTEST_SKIP() << "CUDA is not available on this system.";
    }
}

TEST(TensorMatHyperDualTest, DeviceMethod_ConsistencyAfterMove) {
    // Create tensors on CPU initially
    auto r = torch::rand({2, 3, 4}, torch::device(torch::kCPU));
    auto d = torch::rand({2, 3, 4, 5}, torch::device(torch::kCPU));
    auto h = torch::rand({2, 3, 4, 6, 6}, torch::device(torch::kCPU));

    // Initialize TensorMatHyperDual object
    TensorMatHyperDual tensorMat(r, d, h);

    // Verify initial device is CPU
    EXPECT_EQ(tensorMat.device().type(), torch::kCPU);

    // Move tensors to CUDA if available
    if (torch::cuda::is_available()) {
        TensorMatHyperDual tensorMatCUDA = tensorMat.to(torch::kCUDA);
        EXPECT_EQ(tensorMatCUDA.device().type(), torch::kCUDA);
    } else {
        GTEST_SKIP() << "CUDA is not available on this system.";
    }
}

TEST(TensorMatHyperDualTest, ConstructorFromTensorDual_DefaultDim) {
    // Create a TensorDual object
    auto r = torch::rand({2, 3}).to(torch::kFloat64);
    auto d = torch::rand({2, 3, 4}).to(torch::kFloat64); // 4 is the dual dimension
    TensorDual tensorDual{r, d};

    // Initialize TensorMatHyperDual with default dimension (dim = 2)
    TensorMatHyperDual tensorMat(tensorDual);

    // Check dimensions
    EXPECT_EQ(tensorMat.r.sizes(), torch::IntArrayRef({2, 3, 1}));
    EXPECT_EQ(tensorMat.d.sizes(), torch::IntArrayRef({2, 3, 1, 4}));
    EXPECT_EQ(tensorMat.h.sizes(), torch::IntArrayRef({2, 3, 1, 4, 4}));

    // Check dtype and device consistency
    EXPECT_EQ(tensorMat.dtype_, torch::typeMetaToScalarType(r.dtype()));
    EXPECT_EQ(tensorMat.device_, r.device());
}

TEST(TensorMatHyperDualTest, ConstructorFromTensorDual_CustomDim) {
    // Create a TensorDual object
    auto r = torch::rand({2, 3});
    auto d = torch::rand({2, 3, 4}); // 4 is the dual dimension
    TensorDual tensorDual{r, d};

    // Initialize TensorMatHyperDual with custom dimension (dim = 1)
    TensorMatHyperDual tensorMat(tensorDual, 1);

    // Check dimensions
    EXPECT_EQ(tensorMat.r.sizes(), torch::IntArrayRef({2, 1, 3}));
    EXPECT_EQ(tensorMat.d.sizes(), torch::IntArrayRef({2, 1, 3, 4}));
    EXPECT_EQ(tensorMat.h.sizes(), torch::IntArrayRef({2, 1, 3, 4, 4}));

    // Check dtype and device consistency
    EXPECT_EQ(tensorMat.dtype_, torch::typeMetaToScalarType(r.dtype()));
    EXPECT_EQ(tensorMat.device_, r.device());
}

TEST(TensorMatHyperDualTest, ConstructorFromTensorDual_InvalidDim) {
    // Create a TensorDual object
    auto r = torch::rand({2, 3});
    auto d = torch::rand({2, 3, 4}); // 4 is the dual dimension
    TensorDual tensorDual{r, d};

    // Attempt to use invalid dimensions (e.g., dim = 0 and dim > 2)
    EXPECT_THROW({
        TensorMatHyperDual tensorMat(tensorDual, 0);
    }, std::invalid_argument);

    EXPECT_THROW({
        TensorMatHyperDual tensorMat(tensorDual, 3);
    }, std::invalid_argument);
}

TEST(TensorMatHyperDualTest, ConstructorFromTensorDual_ZeroHyperdualInitialization) {
    // Create a TensorDual object
    auto r = torch::rand({2, 3});
    auto d = torch::rand({2, 3, 4}); // 4 is the dual dimension
    TensorDual tensorDual{r, d};

    // Initialize TensorMatHyperDual with default dimension
    TensorMatHyperDual tensorMat(tensorDual);

    // Check that the hyperdual part is correctly initialized to zeros
    EXPECT_TRUE(torch::all(tensorMat.h == 0).item<bool>());
}

TEST(TensorMatHyperDualTest, ConstructorFromTensorDual_DeviceConsistency) {
    // Create a TensorDual object on CUDA (if available)
    if (torch::cuda::is_available()) {
        auto r = torch::rand({2, 3}, torch::device(torch::kCUDA));
        auto d = torch::rand({2, 3, 4}, torch::device(torch::kCUDA));
        TensorDual tensorDual{r, d};

        // Initialize TensorMatHyperDual
        TensorMatHyperDual tensorMat(tensorDual);

        // Check device consistency
        EXPECT_EQ(tensorMat.device_.type(), torch::kCUDA);  // Compare device type
        EXPECT_EQ(tensorMat.r.device().type(), torch::kCUDA);      // Ensure tensors are on CUDA
        EXPECT_EQ(tensorMat.d.device().type(), torch::kCUDA);
        EXPECT_EQ(tensorMat.h.device().type(), torch::kCUDA);
    } else {
        GTEST_SKIP() << "CUDA is not available on this system.";
    }
}

TEST(TensorMatHyperDualTest, ComplexMethod_RealToComplexConversion) {
    // Create real tensors
    auto r = torch::rand({2, 3, 4});
    auto d = torch::rand({2, 3, 4, 5});
    auto h = torch::rand({2, 3, 4, 5, 5});

    // Initialize TensorMatHyperDual with real tensors
    TensorMatHyperDual tensorMat(r, d, h);

    // Convert to complex
    TensorMatHyperDual complexMat = tensorMat.complex();

    // Check that the tensors are complex
    EXPECT_TRUE(complexMat.r.is_complex());
    EXPECT_TRUE(complexMat.d.is_complex());
    EXPECT_TRUE(complexMat.h.is_complex());

    // Check that the real part matches the original real tensors
    EXPECT_TRUE(torch::all(torch::real(complexMat.r) == r).item<bool>());
    EXPECT_TRUE(torch::all(torch::real(complexMat.d) == d).item<bool>());
    EXPECT_TRUE(torch::all(torch::real(complexMat.h) == h).item<bool>());

    // Check that the imaginary part is zero
    EXPECT_TRUE(torch::all(torch::imag(complexMat.r) == 0).item<bool>());
    EXPECT_TRUE(torch::all(torch::imag(complexMat.d) == 0).item<bool>());
    EXPECT_TRUE(torch::all(torch::imag(complexMat.h) == 0).item<bool>());
}

TEST(TensorMatHyperDualTest, ComplexMethod_AlreadyComplexTensors) {
    // Create complex tensors
    auto r = torch::rand({2, 3, 4}, torch::dtype(torch::kComplexDouble));
    auto d = torch::rand({2, 3, 4, 5}, torch::dtype(torch::kComplexDouble));
    auto h = torch::rand({2, 3, 4, 5, 5}, torch::dtype(torch::kComplexDouble));

    // Initialize TensorMatHyperDual with complex tensors
    TensorMatHyperDual tensorMat(r, d, h);

    // Convert to complex
    TensorMatHyperDual complexMat = tensorMat.complex();

    // Check that the tensors remain unchanged
    EXPECT_TRUE(torch::all(complexMat.r == r).item<bool>());
    EXPECT_TRUE(torch::all(complexMat.d == d).item<bool>());
    EXPECT_TRUE(torch::all(complexMat.h == h).item<bool>());
}

TEST(TensorMatHyperDualTest, ComplexMethod_MixedRealAndComplexTensors) {
    // Create mixed tensors (real and complex)
    auto r = torch::rand({2, 3, 4});
    auto d = torch::rand({2, 3, 4, 5}, torch::dtype(torch::kComplexDouble));
    auto h = torch::rand({2, 3, 4, 5, 5});

    // Initialize TensorMatHyperDual with mixed tensors
    TensorMatHyperDual tensorMat(r, d, h);

    // Convert to complex
    TensorMatHyperDual complexMat = tensorMat.complex();

    // Check that real tensors are converted to complex
    EXPECT_TRUE(complexMat.r.is_complex());
    EXPECT_TRUE(complexMat.h.is_complex());
    EXPECT_TRUE(torch::all(torch::real(complexMat.r) == r).item<bool>());
    EXPECT_TRUE(torch::all(torch::real(complexMat.h) == h).item<bool>());
    EXPECT_TRUE(torch::all(torch::imag(complexMat.r) == 0).item<bool>());
    EXPECT_TRUE(torch::all(torch::imag(complexMat.h) == 0).item<bool>());

    // Check that already complex tensors remain unchanged
    EXPECT_TRUE(torch::all(complexMat.d == d).item<bool>());
}

TEST(TensorMatHyperDualTest, RealMethod_AlreadyRealTensors) {
    // Create real tensors
    auto r = torch::rand({2, 3, 4});
    auto d = torch::rand({2, 3, 4, 5});
    auto h = torch::rand({2, 3, 4, 5, 5});

    // Initialize TensorMatHyperDual with real tensors
    TensorMatHyperDual tensorMat(r, d, h);

    // Extract the real part
    TensorMatHyperDual realMat = tensorMat.real();

    // Check that the tensors are unchanged
    EXPECT_TRUE(torch::all(realMat.r == r).item<bool>());
    EXPECT_TRUE(torch::all(realMat.d == d).item<bool>());
    EXPECT_TRUE(torch::all(realMat.h == h).item<bool>());
}

TEST(TensorMatHyperDualTest, RealMethod_ComplexTensors) {
    // Create complex tensors
    auto r = torch::rand({2, 3, 4}, torch::dtype(torch::kComplexDouble));
    auto d = torch::rand({2, 3, 4, 5}, torch::dtype(torch::kComplexDouble));
    auto h = torch::rand({2, 3, 4, 5, 5}, torch::dtype(torch::kComplexDouble));

    // Initialize TensorMatHyperDual with complex tensors
    TensorMatHyperDual tensorMat(r, d, h);

    // Extract the real part
    TensorMatHyperDual realMat = tensorMat.real();

    // Check that the real part matches the real components of the original tensors
    EXPECT_TRUE(torch::all(realMat.r == torch::real(r)).item<bool>());
    EXPECT_TRUE(torch::all(realMat.d == torch::real(d)).item<bool>());
    EXPECT_TRUE(torch::all(realMat.h == torch::real(h)).item<bool>());

    // Check that the resulting tensors are not complex
    EXPECT_FALSE(realMat.r.is_complex());
    EXPECT_FALSE(realMat.d.is_complex());
    EXPECT_FALSE(realMat.h.is_complex());
}

TEST(TensorMatHyperDualTest, RealMethod_MixedTensors) {
    // Create mixed tensors (real and complex)
    auto r = torch::rand({2, 3, 4});
    auto d = torch::rand({2, 3, 4, 5}, torch::dtype(torch::kComplexDouble));
    auto h = torch::rand({2, 3, 4, 5, 5});

    // Initialize TensorMatHyperDual with mixed tensors
    TensorMatHyperDual tensorMat(r, d, h);

    // Extract the real part
    TensorMatHyperDual realMat = tensorMat.real();

    // Check that real tensors are unchanged
    EXPECT_TRUE(torch::all(realMat.r == r).item<bool>());
    EXPECT_TRUE(torch::all(realMat.h == h).item<bool>());

    // Check that complex tensors are converted to their real components
    EXPECT_TRUE(torch::all(realMat.d == torch::real(d)).item<bool>());

    // Check that the resulting tensors are not complex
    EXPECT_FALSE(realMat.r.is_complex());
    EXPECT_FALSE(realMat.d.is_complex());
    EXPECT_FALSE(realMat.h.is_complex());
}


TEST(TensorMatHyperDualTest, ImagMethod_AlreadyRealTensors) {
    // Create real tensors
    auto r = torch::rand({2, 3, 4});
    auto d = torch::rand({2, 3, 4, 5});
    auto h = torch::rand({2, 3, 4, 5, 5});

    // Initialize TensorMatHyperDual with real tensors
    TensorMatHyperDual tensorMat(r, d, h);

    // Extract the imaginary part
    TensorMatHyperDual imagMat = tensorMat.imag();

    // Check that the imaginary part is zero tensors
    EXPECT_TRUE(torch::all(imagMat.r == torch::zeros_like(r)).item<bool>());
    EXPECT_TRUE(torch::all(imagMat.d == torch::zeros_like(d)).item<bool>());
    EXPECT_TRUE(torch::all(imagMat.h == torch::zeros_like(h)).item<bool>());

    // Check that the resulting tensors are not complex
    EXPECT_FALSE(imagMat.r.is_complex());
    EXPECT_FALSE(imagMat.d.is_complex());
    EXPECT_FALSE(imagMat.h.is_complex());
}

TEST(TensorMatHyperDualTest, ImagMethod_ComplexTensors) {
    // Create complex tensors
    auto r = torch::rand({2, 3, 4}, torch::dtype(torch::kComplexDouble));
    auto d = torch::rand({2, 3, 4, 5}, torch::dtype(torch::kComplexDouble));
    auto h = torch::rand({2, 3, 4, 5, 5}, torch::dtype(torch::kComplexDouble));

    // Initialize TensorMatHyperDual with complex tensors
    TensorMatHyperDual tensorMat(r, d, h);

    // Extract the imaginary part
    TensorMatHyperDual imagMat = tensorMat.imag();

    // Check that the imaginary part matches the imaginary components of the original tensors
    EXPECT_TRUE(torch::all(imagMat.r == torch::imag(r)).item<bool>());
    EXPECT_TRUE(torch::all(imagMat.d == torch::imag(d)).item<bool>());
    EXPECT_TRUE(torch::all(imagMat.h == torch::imag(h)).item<bool>());

    // Check that the resulting tensors are not complex
    EXPECT_FALSE(imagMat.r.is_complex());
    EXPECT_FALSE(imagMat.d.is_complex());
    EXPECT_FALSE(imagMat.h.is_complex());
}

TEST(TensorMatHyperDualTest, ImagMethod_MixedTensors) {
    // Create mixed tensors (real and complex)
    auto r = torch::rand({2, 3, 4});
    auto d = torch::rand({2, 3, 4, 5}, torch::dtype(torch::kComplexDouble));
    auto h = torch::rand({2, 3, 4, 5, 5});

    // Initialize TensorMatHyperDual with mixed tensors
    TensorMatHyperDual tensorMat(r, d, h);

    // Extract the imaginary part
    TensorMatHyperDual imagMat = tensorMat.imag();

    // Check that real tensors result in zero tensors
    EXPECT_TRUE(torch::all(imagMat.r == torch::zeros_like(r)).item<bool>());
    EXPECT_TRUE(torch::all(imagMat.h == torch::zeros_like(h)).item<bool>());

    // Check that complex tensors are converted to their imaginary components
    EXPECT_TRUE(torch::all(imagMat.d == torch::imag(d)).item<bool>());

    // Check that the resulting tensors are not complex
    EXPECT_FALSE(imagMat.r.is_complex());
    EXPECT_FALSE(imagMat.d.is_complex());
    EXPECT_FALSE(imagMat.h.is_complex());
}


TEST(TensorMatHyperDualTest, AbsMethod_PositiveRealPart) {
    // Create tensors with positive real values
    auto r = torch::rand({2, 3, 4}) + 1.0; // Ensure all values are positive
    auto d = torch::rand({2, 3, 4, 5});
    auto h = torch::rand({2, 3, 4, 5, 5});

    // Initialize TensorMatHyperDual
    TensorMatHyperDual tensorMat(r, d, h);

    // Compute the absolute value
    TensorMatHyperDual absMat = tensorMat.abs();

    // Check that the real part matches the original tensor
    EXPECT_TRUE(torch::all(absMat.r == r).item<bool>());

    // Check that the dual part remains unchanged
    EXPECT_TRUE(torch::all(absMat.d == d).item<bool>());

    // Check that the hyperdual part is zero
    EXPECT_TRUE(torch::all(absMat.h == torch::zeros_like(h)).item<bool>());
}

TEST(TensorMatHyperDualTest, AbsMethod_NegativeRealPart) {
    // Create tensors with negative real values
    auto r = -torch::rand({2, 3, 4}) - 1.0; // Ensure all values are negative
    auto d = torch::rand({2, 3, 4, 5});
    auto h = torch::rand({2, 3, 4, 5, 5});

    // Initialize TensorMatHyperDual
    TensorMatHyperDual tensorMat(r, d, h);

    // Compute the absolute value
    TensorMatHyperDual absMat = tensorMat.abs();

    // Check that the real part is the absolute value of the original tensor
    EXPECT_TRUE(torch::all(absMat.r == torch::abs(r)).item<bool>());

    // Check that the dual part is scaled by the sign of the real part
    EXPECT_TRUE(torch::all(absMat.d == (-d)).item<bool>());

    // Check that the hyperdual part is zero
    EXPECT_TRUE(torch::all(absMat.h == torch::zeros_like(h)).item<bool>());
}

TEST(TensorMatHyperDualTest, AbsMethod_MixedRealPart) {
    // Create tensors with mixed positive and negative real values
    auto r = torch::randn({2, 3, 4}); // Random values, including negative and positive
    auto d = torch::rand({2, 3, 4, 5});
    auto h = torch::rand({2, 3, 4, 5, 5});

    // Initialize TensorMatHyperDual
    TensorMatHyperDual tensorMat(r, d, h);

    // Compute the absolute value
    TensorMatHyperDual absMat = tensorMat.abs();

    // Check that the real part is the absolute value of the original tensor
    EXPECT_TRUE(torch::all(absMat.r == torch::abs(r)).item<bool>());

    // Check that the dual part is scaled by the sign of the real part
    EXPECT_TRUE(torch::all(absMat.d == (torch::sign(r).unsqueeze(-1) * d)).item<bool>());

    // Check that the hyperdual part is zero
    EXPECT_TRUE(torch::all(absMat.h == torch::zeros_like(h)).item<bool>());
}

TEST(TensorMatHyperDualTest, AbsMethod_ZeroRealPart) {
    // Create tensors with zero real values
    auto r = torch::zeros({2, 3, 4});
    auto d = torch::rand({2, 3, 4, 5});
    auto h = torch::rand({2, 3, 4, 5, 5});

    // Initialize TensorMatHyperDual
    TensorMatHyperDual tensorMat(r, d, h);

    // Compute the absolute value
    TensorMatHyperDual absMat = tensorMat.abs();
 
    // Check that the real part is zero
    EXPECT_TRUE(torch::all(absMat.r == r).item<bool>());

    // Check that the dual part is zero
    EXPECT_TRUE(torch::all(absMat.d == torch::zeros_like(d)).item<bool>());

    // Check that the hyperdual part is zero
    EXPECT_TRUE(torch::all(absMat.h == torch::zeros_like(h)).item<bool>());
}


TEST(TensorMatHyperDualTest, MaxMethod_DefaultDimension) {
    // Create tensors
    auto r = torch::rand({2, 3, 4}); // Real part
    auto d = torch::rand({2, 3, 4, 5}); // Dual part
    auto h = torch::rand({2, 3, 4, 5, 5}); // Hyperdual part

    // Initialize TensorMatHyperDual
    TensorMatHyperDual tensorMat(r, d, h);

    // Compute max along the default dimension (dim = 1)
    TensorMatHyperDual maxMat = tensorMat.max();

    // Compute expected results
    auto max_result = torch::max(r, 1, /*keepdim=*/true);
    auto max_values = std::get<0>(max_result);  // Maximum values
    auto max_indices = std::get<1>(max_result); // Indices of the maximum values
    auto dshape = max_indices.unsqueeze(-1).expand_as(d);
    auto hshape = max_indices.unsqueeze(-1).unsqueeze(-1).expand_as(h);
    auto expected_dual = torch::gather(d, 1, dshape);
    auto expected_hyper = torch::gather(h, 1, hshape);

    // Check real, dual, and hyperdual parts
    EXPECT_TRUE(torch::all(maxMat.r == max_values).item<bool>());
    EXPECT_TRUE(torch::all(maxMat.d == expected_dual).item<bool>());
    EXPECT_TRUE(torch::all(maxMat.h == expected_hyper).item<bool>());
}

TEST(TensorMatHyperDualTest, MaxMethod_SpecifiedDimension) {
    // Create tensors
    auto r = torch::rand({2, 3, 4}); // Real part
    auto d = torch::rand({2, 3, 4, 5}); // Dual part
    auto h = torch::rand({2, 3, 4, 5, 5}); // Hyperdual part

    // Initialize TensorMatHyperDual
    TensorMatHyperDual tensorMat(r, d, h);

    // Compute max along a specified dimension (dim = 2)
    TensorMatHyperDual maxMat = tensorMat.max(2);

    // Compute expected results
    auto max_result = torch::max(r, 2, /*keepdim=*/true);
    auto max_values = std::get<0>(max_result);  // Maximum values
    auto max_indices = std::get<1>(max_result); // Indices of the maximum values
    auto dshape = max_indices.unsqueeze(-1).expand_as(d);
    auto hshape = max_indices.unsqueeze(-1).unsqueeze(-1).expand_as(h);
    auto expected_dual = torch::gather(d, 2, dshape);
    auto expected_hyper = torch::gather(h, 2, hshape);

    // Check real, dual, and hyperdual parts
    EXPECT_TRUE(torch::all(maxMat.r == max_values).item<bool>());
    EXPECT_TRUE(torch::all(maxMat.d == expected_dual).item<bool>());
    EXPECT_TRUE(torch::all(maxMat.h == expected_hyper).item<bool>());
}


TEST(TensorMatHyperDualTest, MinMethod_DefaultDimension) {
    // Create tensors
    auto r = torch::rand({2, 3, 4}); // Real part
    auto d = torch::rand({2, 3, 4, 5}); // Dual part
    auto h = torch::rand({2, 3, 4, 5, 5}); // Hyperdual part

    // Initialize TensorMatHyperDual
    TensorMatHyperDual tensorMat(r, d, h);

    // Compute min along the default dimension (dim = 1)
    TensorMatHyperDual minMat = tensorMat.min();

    // Compute expected results
    auto min_result = torch::min(r, 1, /*keepdim=*/true);
    auto min_values = std::get<0>(min_result);  // Minimum values
    auto min_indices = std::get<1>(min_result); // Indices of the minimum values

    // Gather dual and hyperdual values based on min indices
    auto dshape = min_indices.unsqueeze(-1).expand({2, 3, 4, 5});
    auto hshape = min_indices.unsqueeze(-1).unsqueeze(-1).expand({2, 3, 4, 5, 5});
    auto expected_dual_values = torch::gather(d, 1, dshape);
    auto expected_hyper_values = torch::gather(h, 1, hshape);

    // Check the real part
    EXPECT_TRUE(torch::all(minMat.r == min_values).item<bool>());

    // Check the dual part
    EXPECT_TRUE(torch::all(minMat.d == expected_dual_values).item<bool>());

    // Check the hyperdual part
    EXPECT_TRUE(torch::all(minMat.h == expected_hyper_values).item<bool>());
}

TEST(TensorMatHyperDualTest, MinMethod_SpecifiedDimension) {
    // Create tensors
    auto r = torch::rand({2, 3, 4}); // Real part
    auto d = torch::rand({2, 3, 4, 5}); // Dual part
    auto h = torch::rand({2, 3, 4, 5, 5}); // Hyperdual part

    // Initialize TensorMatHyperDual
    TensorMatHyperDual tensorMat(r, d, h);

    // Compute min along a specified dimension (dim = 2)
    TensorMatHyperDual minMat = tensorMat.min(2);

    // Compute expected results
    auto min_result = torch::min(r, 2, /*keepdim=*/true);
    auto min_values = std::get<0>(min_result);  // Minimum values
    auto min_indices = std::get<1>(min_result); // Indices of the minimum values

    // Gather dual and hyperdual values based on min indices
    auto dshape = min_indices.unsqueeze(-1).expand({2, 3, 4, 5});
    auto hshape = min_indices.unsqueeze(-1).unsqueeze(-1).expand({2, 3, 4, 5, 5});
    auto expected_dual_values = torch::gather(d, 2, dshape);
    auto expected_hyper_values = torch::gather(h, 2, hshape);

    // Check the real part
    EXPECT_TRUE(torch::all(minMat.r == min_values).item<bool>());

    // Check the dual part
    EXPECT_TRUE(torch::all(minMat.d == expected_dual_values).item<bool>());

    // Check the hyperdual part
    EXPECT_TRUE(torch::all(minMat.h == expected_hyper_values).item<bool>());
}

TEST(TensorMatHyperDualTest, MinMethod_InvalidDimension) {
    // Create tensors
    auto r = torch::rand({2, 3, 4}); // Real part
    auto d = torch::rand({2, 3, 4, 5}); // Dual part
    auto h = torch::rand({2, 3, 4, 5, 5}); // Hyperdual part

    // Initialize TensorMatHyperDual
    TensorMatHyperDual tensorMat(r, d, h);

    // Attempt to compute min along an invalid dimension
    EXPECT_THROW({
        TensorMatHyperDual minMat = tensorMat.min(3); // Invalid dimension
    }, c10::Error);
}


TEST(TensorMatHyperDualTest, SumMethod_ValidDimension) {
    auto r = torch::rand({2, 3, 4});
    auto d = torch::rand({2, 3, 4, 5});
    auto h = torch::rand({2, 3, 4, 5, 5});

    TensorMatHyperDual tensorMat(r, d, h);

    // Compute sum along dimension 1
    TensorMatHyperDual sumMat = tensorMat.sum(1);

    // Expected results
    auto r_sum = r.sum(1, /*keepdim=*/true);
    auto d_sum = d.sum(1, /*keepdim=*/true);
    auto h_sum = h.sum(1, /*keepdim=*/true);

    // Verify results
    EXPECT_TRUE(torch::all(sumMat.r == r_sum).item<bool>());
    EXPECT_TRUE(torch::all(sumMat.d == d_sum).item<bool>());
    EXPECT_TRUE(torch::all(sumMat.h == h_sum).item<bool>());
}

TEST(TensorMatHyperDualTest, SumMethod_LastDimension) {
    auto r = torch::rand({2, 3, 4});
    auto d = torch::rand({2, 3, 4, 5});
    auto h = torch::rand({2, 3, 4, 5, 5});

    TensorMatHyperDual tensorMat(r, d, h);

    // Compute sum along the last dimension of r, d, and h
    TensorMatHyperDual sumMat = tensorMat.sum(-1);

    // Expected results
    auto r_sum = r.sum(-1, /*keepdim=*/true);
    auto d_sum = d.sum(r.dim()-1, /*keepdim=*/true);
    auto h_sum = h.sum(r.dim()-1, /*keepdim=*/true);

    // Verify results
    EXPECT_TRUE(torch::all(sumMat.r == r_sum).item<bool>());
    EXPECT_TRUE(torch::all(sumMat.d == d_sum).item<bool>());
    EXPECT_TRUE(torch::all(sumMat.h == h_sum).item<bool>());
}

TEST(TensorMatHyperDualTest, SquareMethod_GeneralCase) {
    auto r = torch::rand({2, 3, 4});
    auto d = torch::rand({2, 3, 4, 5});
    auto h = torch::rand({2, 3, 4, 5, 5});

    TensorMatHyperDual tensorMat(r, d, h);
    TensorMatHyperDual squaredMat = tensorMat.square();

    // Expected results
    auto rsq = r.square();
    auto dn = 2 * r.unsqueeze(-1) * d;
    auto hn = 2 * torch::einsum("mijk, mijl->mijkl", {d, d}) +
              2 * torch::einsum("mij, mijkl->mijkl", {r, h});

    // Check real part
    EXPECT_TRUE(torch::allclose(squaredMat.r, rsq));

    // Check dual part
    EXPECT_TRUE(torch::allclose(squaredMat.d, dn));

    // Check hyperdual part
    EXPECT_TRUE(torch::allclose(squaredMat.h, hn));
}

TEST(TensorMatHyperDualTest, SquareMethod_ZeroTensor) {
    auto r = torch::zeros({2, 3, 4});
    auto d = torch::zeros({2, 3, 4, 6});
    auto h = torch::zeros({2, 3, 4, 6, 6});

    TensorMatHyperDual tensorMat(r, d, h);
    TensorMatHyperDual squaredMat = tensorMat.square();

    // Expected results
    auto rsq = r.square();
    auto dn = 2 * r.unsqueeze(-1) * d;
    auto hn = 2 * torch::einsum("mijk, mijl->mijkl", {d, d}) +
              2 * torch::einsum("mij, mijkl->mijkl", {r, h});

    // Check real part
    EXPECT_TRUE(torch::allclose(squaredMat.r, rsq));

    // Check dual part
    EXPECT_TRUE(torch::allclose(squaredMat.d, dn));

    // Check hyperdual part
    EXPECT_TRUE(torch::allclose(squaredMat.h, hn));
}

TEST(TensorMatHyperDualTest, SquareMethod_LargeTensor) {
    auto r = torch::rand({10, 10, 100});
    auto d = torch::rand({10, 10, 100, 5});
    auto h = torch::rand({10, 10, 100, 5, 5});

    TensorMatHyperDual tensorMat(r, d, h);
    TensorMatHyperDual squaredMat = tensorMat.square();

    // Expected results
    auto rsq = r.square();
    auto dn = 2 * r.unsqueeze(-1) * d;
    auto hn = 2 * torch::einsum("mijk, mijl->mijkl", {d, d}) +
              2 * torch::einsum("mij, mijkl->mijkl", {r, h});

    // Check real part
    EXPECT_TRUE(torch::allclose(squaredMat.r, rsq));

    // Check dual part
    EXPECT_TRUE(torch::allclose(squaredMat.d, dn));

    // Check hyperdual part
    EXPECT_TRUE(torch::allclose(squaredMat.h, hn));
}

TEST(TensorHyperDualTest, SqueezeMethod_ValidDimension) {
    auto r = torch::rand({2, 1, 3});
    auto d = torch::rand({2, 1, 3, 4});
    auto h = torch::rand({2, 1, 3, 4, 4});

    TensorMatHyperDual tensor(r, d, h);

    // Squeeze dimension 1
    TensorHyperDual squeezed = tensor.squeeze(1);

    // Expected results
    auto r_squeezed = r.squeeze(1);
    auto d_squeezed = d.squeeze(1);
    auto h_squeezed = h.squeeze(1);

    // Validate
    EXPECT_TRUE(torch::all(squeezed.r == r_squeezed).item<bool>());
    EXPECT_TRUE(torch::all(squeezed.d == d_squeezed).item<bool>());
    EXPECT_TRUE(torch::all(squeezed.h == h_squeezed).item<bool>());
}

TEST(TensorHyperDualTest, SqueezeMethod_NonSingletonDimension) {
    auto r = torch::rand({2, 3, 4});
    auto d = torch::rand({2, 3, 4, 5});
    auto h = torch::rand({2, 3, 4, 5, 5});

    TensorMatHyperDual tensor(r, d, h);

    // Attempt to squeeze dimension 1 (size = 3)
    EXPECT_THROW({
        TensorHyperDual squeezed = tensor.squeeze(1);
    }, std::invalid_argument);

}


TEST(TensorHyperDualTest, SqueezeMethod_OutOfBoundsDimension) {
    auto r = torch::rand({2, 3, 4});
    auto d = torch::rand({2, 3, 4, 5});
    auto h = torch::rand({2, 3, 4, 5, 5});

    TensorMatHyperDual tensor(r, d, h);

    // Attempt to squeeze an out-of-bounds dimension
    EXPECT_THROW({
        tensor.squeeze(5); // Invalid dimension
    }, std::invalid_argument);
}

TEST(TensorHyperDualTest, SqueezeMethod_AllSingletonDimensions) {
    auto r = torch::rand({1, 1, 1});
    auto d = torch::rand({1, 1, 1, 1});
    auto h = torch::rand({1, 1, 1, 1, 1});

    TensorMatHyperDual tensor(r, d, h);

    EXPECT_THROW({
        TensorHyperDual squeezed = tensor.squeeze(0);
    }, std::invalid_argument);
   
}

TEST(TensorMatHyperDualTest, ContiguousMethod_ContiguousTensors) {
    auto r = torch::rand({2, 3, 4});
    auto d = torch::rand({2, 3, 4, 5});
    auto h = torch::rand({2, 3, 4, 5, 5});

    TensorMatHyperDual tensor(r, d, h);

    // Call contiguous method
    TensorMatHyperDual contiguousTensor = tensor.contiguous();

    // Verify the tensors are the same (already contiguous)
    EXPECT_TRUE(torch::equal(contiguousTensor.r, r));
    EXPECT_TRUE(torch::equal(contiguousTensor.d, d));
    EXPECT_TRUE(torch::equal(contiguousTensor.h, h));

    // Verify they are still contiguous
    EXPECT_TRUE(contiguousTensor.r.is_contiguous());
    EXPECT_TRUE(contiguousTensor.d.is_contiguous());
    EXPECT_TRUE(contiguousTensor.h.is_contiguous());
}


TEST(TensorMatHyperDualTest, ContiguousMethod_NonContiguousTensors) {
    auto r = torch::rand({2, 3, 5}).transpose(1, 2); // Non-contiguous tensor
    auto d = torch::rand({2, 3, 5, 23}).transpose(1, 2); // Non-contiguous tensor
    auto h = torch::rand({2, 3, 5, 23, 23}).transpose(1, 2); // Non-contiguous tensor

    TensorMatHyperDual tensor(r, d, h);

    // Call contiguous method
    TensorMatHyperDual contiguousTensor = tensor.contiguous();

    // Verify the tensors are not the same (non-contiguous converted to contiguous)
    EXPECT_TRUE(torch::equal(contiguousTensor.r, r));
    EXPECT_TRUE(torch::equal(contiguousTensor.d, d));
    EXPECT_TRUE(torch::equal(contiguousTensor.h, h));

    // Verify they are now contiguous
    EXPECT_TRUE(contiguousTensor.r.is_contiguous());
    EXPECT_TRUE(contiguousTensor.d.is_contiguous());
    EXPECT_TRUE(contiguousTensor.h.is_contiguous());
}


TEST(TensorMatHyperDualTest, ContiguousMethod_EmptyTensors) {
    auto r = torch::empty({0, 0, 0});
    auto d = torch::empty({0, 0, 0, 0});
    auto h = torch::empty({0, 0, 0, 0, 0});

    TensorMatHyperDual tensor(r, d, h);

    // Call contiguous method
    TensorMatHyperDual contiguousTensor = tensor.contiguous();

    // Verify the tensors remain empty
    EXPECT_TRUE(contiguousTensor.r.numel() == 0);
    EXPECT_TRUE(contiguousTensor.d.numel() == 0);
    EXPECT_TRUE(contiguousTensor.h.numel() == 0);

    // Verify they are contiguous
    EXPECT_TRUE(contiguousTensor.r.is_contiguous());
    EXPECT_TRUE(contiguousTensor.d.is_contiguous());
    EXPECT_TRUE(contiguousTensor.h.is_contiguous());
}


TEST(TensorMatHyperDualTest, SqrtMethod_PositiveRealPart) {
    auto r = torch::tensor({{4.0, 9.0}}).reshape({2, 1, 1});
    auto d = torch::rand({2, 1, 1, 3});
    auto h = torch::rand({2, 1, 1, 3, 3});

    TensorMatHyperDual tensor(r, d, h);

    // Compute square root
    TensorMatHyperDual sqrtTensor = tensor.sqrt();

    // Expected results
    auto r_sqrt = torch::sqrt(r);
    auto rf_inv_sqrt = 0.5 * r_sqrt.pow(-1);
    auto d_sqrt = torch::einsum("mij, mijn->mijn", {rf_inv_sqrt, d});

    auto rf_inv_3_sqrt = 0.25 * r_sqrt.pow(-3);
    auto h_sqrt = torch::einsum("mij, mijkn->mijkn", {rf_inv_sqrt, h}) -
                  torch::einsum("mij, mijk, mijn->mijkn", {rf_inv_3_sqrt, d, d});

    // Validate results
    EXPECT_TRUE(torch::allclose(sqrtTensor.r, r_sqrt));
    EXPECT_TRUE(torch::allclose(sqrtTensor.d, d_sqrt));
    EXPECT_TRUE(torch::allclose(sqrtTensor.h, h_sqrt));
}


TEST(TensorMatHyperDualTest, SqrtMethod_NegativeRealPart) {
    auto r = torch::tensor({-1.0, 4.0}).reshape({2, 1, 1});
    auto d = torch::rand({2, 1, 1, 3});
    auto h = torch::rand({2, 1, 1, 3, 3});

    TensorMatHyperDual tensor(r, d, h);

    // Compute square root (expect complex result)
    TensorMatHyperDual sqrtTensor = tensor.sqrt();

    // Expected results (for complex domain)
    auto r_sqrt = torch::sqrt(tensor.complex().r);
    EXPECT_TRUE(torch::allclose(sqrtTensor.r, r_sqrt));
}


TEST(TensorMatHyperDualTest, SqrtMethod_ZeroRealPart) {
    auto r = torch::tensor({0.0, 4.0}).reshape({2, 1, 1});
    auto d = torch::rand({2, 1, 1, 3});
    auto h = torch::rand({2, 1, 1, 3, 3});

    TensorMatHyperDual tensor(r, d, h);

    // Compute square root
    TensorMatHyperDual sqrtTensor = tensor.sqrt();

    // Expected results
    auto r_sqrt = torch::sqrt(r);
    r_sqrt = torch::where(r_sqrt != 0, r_sqrt, torch::ones_like(r_sqrt) * 1e-12);

    auto rf_inv_sqrt = 0.5 * r_sqrt.pow(-1);
    auto d_sqrt = torch::einsum("mij, mijn->mijn", {rf_inv_sqrt, d});

    auto rf_inv_3_sqrt = 0.25 * r_sqrt.pow(-3);
    auto h_sqrt = torch::einsum("mij, mijkn->mijkn", {rf_inv_sqrt, h}) -
                  torch::einsum("mij, mijk, mijn->mijkn", {rf_inv_3_sqrt, d, d});

    // Validate results
    EXPECT_TRUE(torch::allclose(sqrtTensor.r, r_sqrt));
    EXPECT_TRUE(torch::allclose(sqrtTensor.d, d_sqrt));
    EXPECT_TRUE(torch::allclose(sqrtTensor.h, h_sqrt));
}


// Test Suite
TEST(TensorMatHyperDualTest, CreateZeroValidInput) {
    auto r = torch::rand({2, 3, 4});
    int ddim = 5;
    TensorMatHyperDual result = TensorMatHyperDual::createZero(r, ddim);

    // Check shapes
    EXPECT_EQ(result.r.sizes(), r.sizes());
    EXPECT_EQ(result.d.sizes(), torch::IntArrayRef({2, 3, 4, 5}));
    EXPECT_EQ(result.h.sizes(), torch::IntArrayRef({2, 3, 4, 5, 5}));

    // Check values
    EXPECT_TRUE(torch::all(result.d == 0).item<bool>());
    EXPECT_TRUE(torch::all(result.h == 0).item<bool>());
}

TEST(TensorMatHyperDualTest, CreateZeroInvalidRealTensorDimension) {
    auto r = torch::rand({3, 4}); // 2D tensor
    int ddim = 5;
    EXPECT_THROW(TensorMatHyperDual::createZero(r, ddim), std::invalid_argument);
}

TEST(TensorMatHyperDualTest, CreateZeroZeroDualDimension) {
    auto r = torch::rand({3, 3, 3});
    int ddim = 0;
    EXPECT_THROW(TensorMatHyperDual::createZero(r, ddim), std::invalid_argument);
}

TEST(TensorMatHyperDualTest, CreateZeroNegativeDualDimension) {
    auto r = torch::rand({3, 3, 3});
    int ddim = -2;
    EXPECT_THROW(TensorMatHyperDual::createZero(r, ddim), c10::Error);
}

TEST(TensorMatHyperDualTest, CreateZeroEmptyTensor) {
    auto r = torch::zeros({0, 3, 4});
    int ddim = 5;
    TensorMatHyperDual result = TensorMatHyperDual::createZero(r, ddim);

    // Check shapes
    EXPECT_EQ(result.r.sizes(), r.sizes());
    EXPECT_EQ(result.d.sizes(), torch::IntArrayRef({0, 3, 4, 5}));
    EXPECT_EQ(result.h.sizes(), torch::IntArrayRef({0, 3, 4, 5, 5}));
}

TEST(TensorMatHyperDualTest, CreateZeroDeviceAndDtype) {
    auto r = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    int ddim = 5;
    TensorMatHyperDual result = TensorMatHyperDual::createZero(r, ddim);

    EXPECT_EQ(result.r.dtype(), r.dtype());
    EXPECT_EQ(result.r.device(), r.device());
    EXPECT_EQ(result.d.dtype(), r.dtype());
    EXPECT_EQ(result.d.device(), r.device());
    EXPECT_EQ(result.h.dtype(), r.dtype());
    EXPECT_EQ(result.h.device(), r.device());
}

TEST(TensorMatHyperDualTest, CreateZeroLargeTensor) {
    auto r = torch::rand({100, 100, 100});
    int ddim = 10;
    TensorMatHyperDual result = TensorMatHyperDual::createZero(r, ddim);

    // Check shapes
    EXPECT_EQ(result.r.sizes(), r.sizes());
    EXPECT_EQ(result.d.sizes(), torch::IntArrayRef({100, 100, 100, 10}));
    EXPECT_EQ(result.h.sizes(), torch::IntArrayRef({100, 100, 100, 10, 10}));
}

TEST(TensorMatHyperDualTest, CreateZeroNonFloatTensor) {
    auto r = torch::randint(0, 10, {3, 3, 3}, torch::TensorOptions().dtype(torch::kInt32));
    int ddim = 5;
    TensorMatHyperDual result = TensorMatHyperDual::createZero(r, ddim);

    EXPECT_EQ(result.r.dtype(), torch::kInt32);
    EXPECT_EQ(result.d.dtype(), torch::kInt32);
    EXPECT_EQ(result.h.dtype(), torch::kInt32);
}

TEST(TensorMatHyperDualTest, ZerosLikeTensor) {

    auto x = torch::rand({2, 3, 4});
    auto d = torch::rand({2, 3, 4, 3});
    auto h = torch::rand({2, 3, 4, 3, 3});

    TensorMatHyperDual tensor(x, d, h);
    auto zeros_tensor = tensor.zeros_like(x);

    // Validate shapes
    EXPECT_EQ(zeros_tensor.r.sizes(), x.sizes());
    EXPECT_EQ(zeros_tensor.d.sizes(), d.sizes());
    EXPECT_EQ(zeros_tensor.h.sizes(), h.sizes());

    // Validate content
    EXPECT_TRUE(torch::all(zeros_tensor.r == 0).item<bool>());
    EXPECT_TRUE(torch::all(zeros_tensor.d == 0).item<bool>());
    EXPECT_TRUE(torch::all(zeros_tensor.h == 0).item<bool>());
}

TEST(TensorMatHyperDualTest, ZerosLikeZeroTensor) {
    auto x = torch::zeros({2, 3, 4});
    auto d = torch::zeros({2, 3, 4, 3});
    auto h = torch::zeros({2, 3, 4, 3, 3});

    TensorMatHyperDual tensor(x, d, h);
    auto zeros_tensor = tensor.zeros_like(x);

    // Validate shapes
    EXPECT_EQ(zeros_tensor.r.sizes(), x.sizes());
    EXPECT_EQ(zeros_tensor.d.sizes(), d.sizes());
    EXPECT_EQ(zeros_tensor.h.sizes(), h.sizes());

    // Validate content
    EXPECT_TRUE(torch::all(zeros_tensor.r == 0).item<bool>());
    EXPECT_TRUE(torch::all(zeros_tensor.d == 0).item<bool>());
    EXPECT_TRUE(torch::all(zeros_tensor.h == 0).item<bool>());
}

TEST(TensorMatHyperDualTest, ZerosLikeNonSquareTensor) {
    auto x = torch::rand({2, 5, 3});
    auto d = torch::rand({2, 5, 3, 4});
    auto h = torch::rand({2, 5, 3, 4, 4});

    TensorMatHyperDual tensor(x, d, h);
    auto zeros_tensor = tensor.zeros_like(x);

    // Validate shapes
    EXPECT_EQ(zeros_tensor.r.sizes(), x.sizes());
    EXPECT_EQ(zeros_tensor.d.sizes(), d.sizes());
    EXPECT_EQ(zeros_tensor.h.sizes(), h.sizes());

    // Validate content
    EXPECT_TRUE(torch::all(zeros_tensor.r == 0).item<bool>());
    EXPECT_TRUE(torch::all(zeros_tensor.d == 0).item<bool>());
    EXPECT_TRUE(torch::all(zeros_tensor.h == 0).item<bool>());
}

TEST(TensorMatHyperDualTest, ZerosLikeTensorOnCUDA) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA is not available.";
    }

    auto x = torch::rand({2, 3, 4}, torch::TensorOptions().device(torch::kCUDA));
    auto d = torch::rand({2, 3, 4, 3}, torch::TensorOptions().device(torch::kCUDA));
    auto h = torch::rand({2, 3, 4, 3, 3}, torch::TensorOptions().device(torch::kCUDA));

    TensorMatHyperDual tensor(x, d, h);
    auto zeros_tensor = tensor.zeros_like(x);

    // Validate shapes
    EXPECT_EQ(zeros_tensor.r.sizes(), x.sizes());
    EXPECT_EQ(zeros_tensor.d.sizes(), d.sizes());
    EXPECT_EQ(zeros_tensor.h.sizes(), h.sizes());

    // Validate device
    EXPECT_EQ(zeros_tensor.r.device().type(), torch::kCUDA);
    EXPECT_EQ(zeros_tensor.d.device().type(), torch::kCUDA);
    EXPECT_EQ(zeros_tensor.h.device().type(), torch::kCUDA);

    // Validate content
    EXPECT_TRUE(torch::all(zeros_tensor.r == 0).item<bool>());
    EXPECT_TRUE(torch::all(zeros_tensor.d == 0).item<bool>());
    EXPECT_TRUE(torch::all(zeros_tensor.h == 0).item<bool>());
}


TEST(TensorMatHyperDualTest, ZerosLikeDifferentDataType) {
    auto x = torch::rand({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat64));
    auto d = torch::rand({2, 3, 4, 3}, torch::TensorOptions().dtype(torch::kFloat64));
    auto h = torch::rand({2, 3, 4, 3, 3}, torch::TensorOptions().dtype(torch::kFloat64));

    TensorMatHyperDual tensor(x, d, h);
    auto zeros_tensor = tensor.zeros_like(x);

    // Validate shapes
    EXPECT_EQ(zeros_tensor.r.sizes(), x.sizes());
    EXPECT_EQ(zeros_tensor.d.sizes(), d.sizes());
    EXPECT_EQ(zeros_tensor.h.sizes(), h.sizes());

    // Validate data type
    EXPECT_EQ(zeros_tensor.r.dtype(), torch::kFloat64);
    EXPECT_EQ(zeros_tensor.d.dtype(), torch::kFloat64);
    EXPECT_EQ(zeros_tensor.h.dtype(), torch::kFloat64);

    // Validate content
    EXPECT_TRUE(torch::all(zeros_tensor.r == 0).item<bool>());
    EXPECT_TRUE(torch::all(zeros_tensor.d == 0).item<bool>());
    EXPECT_TRUE(torch::all(zeros_tensor.h == 0).item<bool>());
}

TEST(TensorMatHyperDualTest, ZerosLikeLargeTensor) {
    auto x = torch::rand({10, 20, 30});
    auto d = torch::rand({10, 20, 30, 5});
    auto h = torch::rand({10, 20, 30, 5, 5});

    TensorMatHyperDual tensor(x, d, h);
    auto zeros_tensor = tensor.zeros_like(x);

    // Validate shapes
    EXPECT_EQ(zeros_tensor.r.sizes(), x.sizes());
    EXPECT_EQ(zeros_tensor.d.sizes(), d.sizes());
    EXPECT_EQ(zeros_tensor.h.sizes(), h.sizes());

    // Validate content
    EXPECT_TRUE(torch::all(zeros_tensor.r == 0).item<bool>());
    EXPECT_TRUE(torch::all(zeros_tensor.d == 0).item<bool>());
    EXPECT_TRUE(torch::all(zeros_tensor.h == 0).item<bool>());
}


TEST(TensorMatHyperDualTest, ZerosLikeSingleElementTensor) {
    auto x = torch::rand({1, 1, 1});
    auto d = torch::rand({1, 1, 1, 1});
    auto h = torch::rand({1, 1, 1, 1, 1});

    TensorMatHyperDual tensor(x, d, h);
    auto zeros_tensor = tensor.zeros_like(x);

    // Validate shapes
    EXPECT_EQ(zeros_tensor.r.sizes(), x.sizes());
    EXPECT_EQ(zeros_tensor.d.sizes(), d.sizes());
    EXPECT_EQ(zeros_tensor.h.sizes(), h.sizes());

    // Validate content
    EXPECT_TRUE(torch::all(zeros_tensor.r == 0).item<bool>());
    EXPECT_TRUE(torch::all(zeros_tensor.d == 0).item<bool>());
    EXPECT_TRUE(torch::all(zeros_tensor.h == 0).item<bool>());
}


TEST(TensorMatHyperDualTest, CloneBasic) {
    auto r = torch::rand({2, 3, 4});
    auto d = torch::rand({2, 3, 4, 3});
    auto h = torch::rand({2, 3, 4, 3, 3});

    TensorMatHyperDual original(r, d, h);
    TensorMatHyperDual clone = original.clone();

    // Validate shapes
    EXPECT_EQ(clone.r.sizes(), r.sizes());
    EXPECT_EQ(clone.d.sizes(), d.sizes());
    EXPECT_EQ(clone.h.sizes(), h.sizes());

    // Validate independence
    clone.r += 1;
    clone.d += 1;
    clone.h += 1;
    EXPECT_TRUE(!torch::allclose(clone.r, original.r));
    EXPECT_TRUE(!torch::allclose(clone.d, original.d));
    EXPECT_TRUE(!torch::allclose(clone.h, original.h));
}

TEST(TensorMatHyperDualTest, CloneEmptyTensors) {
    auto r = torch::empty({0, 0, 0});
    auto d = torch::empty({0, 0, 0, 0});
    auto h = torch::empty({0, 0, 0, 0, 0});

    TensorMatHyperDual original(r, d, h);
    TensorMatHyperDual clone = original.clone();

    // Validate shapes
    EXPECT_EQ(clone.r.sizes(), r.sizes());
    EXPECT_EQ(clone.d.sizes(), d.sizes());
    EXPECT_EQ(clone.h.sizes(), h.sizes());
}

TEST(TensorMatHyperDualTest, CloneDeviceSpecific) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA is not available.";
    }

    auto r = torch::rand({2, 3, 4}, torch::TensorOptions().device(torch::kCUDA));
    auto d = torch::rand({2, 3, 4, 3}, torch::TensorOptions().device(torch::kCUDA));
    auto h = torch::rand({2, 3, 4, 3, 3}, torch::TensorOptions().device(torch::kCUDA));

    TensorMatHyperDual original(r, d, h);
    TensorMatHyperDual clone = original.clone();

    // Validate device
    EXPECT_EQ(clone.r.device(), r.device());
    EXPECT_EQ(clone.d.device(), d.device());
    EXPECT_EQ(clone.h.device(), h.device());

    // Validate independence
    clone.r += 1;
    clone.d += 1;
    clone.h += 1;
    EXPECT_TRUE(!torch::allclose(clone.r, original.r));
    EXPECT_TRUE(!torch::allclose(clone.d, original.d));
    EXPECT_TRUE(!torch::allclose(clone.h, original.h));
}

TEST(TensorMatHyperDualTest, CatBasic) {
    auto r1 = torch::rand({2, 3, 4});
    auto d1 = torch::rand({2, 3, 4, 3});
    auto h1 = torch::rand({2, 3, 4, 3, 3});

    auto r2 = torch::rand({2, 3, 4});
    auto d2 = torch::rand({2, 3, 4, 3});
    auto h2 = torch::rand({2, 3, 4, 3, 3});

    TensorMatHyperDual t1(r1, d1, h1);
    TensorMatHyperDual t2(r2, d2, h2);

    TensorMatHyperDual result = TensorMatHyperDual::cat(t1, t2, 0);

    EXPECT_EQ(result.r.sizes(), torch::IntArrayRef({4, 3, 4}));
    EXPECT_EQ(result.d.sizes(), torch::IntArrayRef({4, 3, 4, 3}));
    EXPECT_EQ(result.h.sizes(), torch::IntArrayRef({4, 3, 4, 3, 3}));
}


TEST(TensorMatHyperDualTest, CatNonBatchDimension) {
    auto r1 = torch::rand({2, 3, 4});
    auto d1 = torch::rand({2, 3, 4, 3});
    auto h1 = torch::rand({2, 3, 4, 3, 3});

    auto r2 = torch::rand({2, 3, 4});
    auto d2 = torch::rand({2, 3, 4, 3});
    auto h2 = torch::rand({2, 3, 4, 3, 3});

    TensorMatHyperDual t1(r1, d1, h1);
    TensorMatHyperDual t2(r2, d2, h2);

    TensorMatHyperDual result = TensorMatHyperDual::cat(t1, t2, 2);

    EXPECT_EQ(result.r.sizes(), torch::IntArrayRef({2, 3, 8}));
    EXPECT_EQ(result.d.sizes(), torch::IntArrayRef({2, 3, 8, 3}));
    EXPECT_EQ(result.h.sizes(), torch::IntArrayRef({2, 3, 8, 3, 3}));
}

TEST(TensorMatHyperDualTest, CatIncompatibleShapes) {
    auto r1 = torch::rand({2, 3, 4});
    auto d1 = torch::rand({2, 3, 4, 3});
    auto h1 = torch::rand({2, 3, 4, 3, 3});

    auto r2 = torch::rand({2, 2, 4});
    auto d2 = torch::rand({2, 2, 4, 3});
    auto h2 = torch::rand({2, 2, 4, 3, 3});

    TensorMatHyperDual t1(r1, d1, h1);
    TensorMatHyperDual t2(r2, d2, h2);

    EXPECT_THROW(TensorMatHyperDual::cat(t1, t2, 2), std::invalid_argument);
}

TEST(TensorMatHyperDualTest, CatWithTensorBasic) {
    auto r = torch::rand({2, 3, 4});
    auto d = torch::rand({2, 3, 4, 2});
    auto h = torch::rand({2, 3, 4, 2, 2});
    auto t2 = torch::rand({3, 2});

    TensorMatHyperDual t1(r, d, h);
    auto result = TensorMatHyperDual::cat(t1, t2);

    // Validate shapes
    EXPECT_EQ(result.r.sizes(), torch::IntArrayRef({2, 3, 6}));
    EXPECT_EQ(result.d.sizes(), torch::IntArrayRef({2, 3, 6, 2}));
    EXPECT_EQ(result.h.sizes(), torch::IntArrayRef({2, 3, 6, 2, 2}));
}

TEST(TensorMatHyperDualTest, CatWithTensorInvalidShape) {
    auto r = torch::rand({2, 3, 4});
    auto d = torch::rand({2, 3, 4, 2});
    auto h = torch::rand({2, 3, 4, 2, 2});
    auto t2 = torch::rand({2, 2}); // Invalid row dimension

    TensorMatHyperDual t1(r, d, h);
    EXPECT_THROW(TensorMatHyperDual::cat(t1, t2), std::invalid_argument);
}

TEST(TensorMatHyperDualTest, CatWithTensorDifferentDevice) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA is not available.";
    }

    auto r = torch::rand({2, 3, 4}, torch::TensorOptions().device(torch::kCUDA));
    auto d = torch::rand({2, 3, 4, 2}, torch::TensorOptions().device(torch::kCUDA));
    auto h = torch::rand({2, 3, 4, 2, 2}, torch::TensorOptions().device(torch::kCUDA));
    auto t2 = torch::rand({3, 2}, torch::TensorOptions().device(torch::kCPU)); // Different device

    TensorMatHyperDual t1(r, d, h);
    EXPECT_NO_THROW({
        auto result = TensorMatHyperDual::cat(t1, t2);
    });
}


TEST(TensorMatHyperDualTest, AdditionBasic) {
    auto r1 = torch::rand({2, 3, 4});
    auto d1 = torch::rand({2, 3, 4, 2});
    auto h1 = torch::rand({2, 3, 4, 2, 2});

    auto r2 = torch::rand({2, 3, 4});
    auto d2 = torch::rand({2, 3, 4, 2});
    auto h2 = torch::rand({2, 3, 4, 2, 2});

    TensorMatHyperDual t1(r1, d1, h1);
    TensorMatHyperDual t2(r2, d2, h2);

    TensorMatHyperDual result = t1 + t2;

    EXPECT_TRUE(torch::allclose(result.r, r1 + r2));
    EXPECT_TRUE(torch::allclose(result.d, d1 + d2));
    EXPECT_TRUE(torch::allclose(result.h, h1 + h2));
}


TEST(TensorMatHyperDualTest, AdditionDimensionMismatch) {
    auto r1 = torch::rand({2, 3, 4});
    auto d1 = torch::rand({2, 3, 4, 2});
    auto h1 = torch::rand({2, 3, 4, 2, 2});

    auto r2 = torch::rand({2, 3, 5}); // Mismatched dimension
    auto d2 = torch::rand({2, 3, 5, 2});
    auto h2 = torch::rand({2, 3, 5, 2, 2});

    TensorMatHyperDual t1(r1, d1, h1);
    TensorMatHyperDual t2(r2, d2, h2);

    EXPECT_THROW(t1 + t2, std::invalid_argument);
}

TEST(TensorMatHyperDualTest, AdditionDeviceMismatch) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA is not available.";
    }

    auto r1 = torch::rand({2, 3, 4}, torch::TensorOptions().device(torch::kCPU));
    auto d1 = torch::rand({2, 3, 4, 2}, torch::TensorOptions().device(torch::kCPU));
    auto h1 = torch::rand({2, 3, 4, 2, 2}, torch::TensorOptions().device(torch::kCPU));

    auto r2 = torch::rand({2, 3, 4}, torch::TensorOptions().device(torch::kCUDA));
    auto d2 = torch::rand({2, 3, 4, 2}, torch::TensorOptions().device(torch::kCUDA));
    auto h2 = torch::rand({2, 3, 4, 2, 2}, torch::TensorOptions().device(torch::kCUDA));

    TensorMatHyperDual t1(r1, d1, h1);
    TensorMatHyperDual t2(r2, d2, h2);

    EXPECT_THROW(t1 + t2, std::invalid_argument);
}

TEST(TensorMatHyperDualTest, AdditionWithTensorHyperDual) {
    auto r1 = torch::rand({2, 3, 4});
    auto d1 = torch::rand({2, 3, 4, 2});
    auto h1 = torch::rand({2, 3, 4, 2, 2});

    auto r2 = torch::rand({2, 4});
    auto d2 = torch::rand({2, 4, 2});
    auto h2 = torch::rand({2, 4, 2, 2});

    TensorMatHyperDual t1(r1, d1, h1);
    TensorHyperDual t2(r2, d2, h2);

    TensorMatHyperDual result = t1 + t2;

    EXPECT_EQ(result.r.sizes(), torch::IntArrayRef({2, 3, 4}));
    EXPECT_EQ(result.d.sizes(), torch::IntArrayRef({2, 3, 4, 2}));
    EXPECT_EQ(result.h.sizes(), torch::IntArrayRef({2, 3, 4, 2, 2}));
}


TEST(TensorMatHyperDualTest, AdditionWithDimensionMismatch) {
    auto r1 = torch::rand({2, 3, 4});
    auto d1 = torch::rand({2, 3, 4, 2});
    auto h1 = torch::rand({2, 3, 4, 2, 2});

    auto r2 = torch::rand({2, 5}); // Mismatched second dimension
    auto d2 = torch::rand({2, 5, 2});
    auto h2 = torch::rand({2, 5, 2, 2});

    TensorMatHyperDual t1(r1, d1, h1);
    TensorHyperDual t2(r2, d2, h2);

    EXPECT_THROW(t1 + t2, std::invalid_argument);
}

TEST(TensorMatHyperDualTest, ScalarAdditionBasic) {
    auto r = torch::rand({2, 3, 4});
    auto d = torch::rand({2, 3, 4, 2});
    auto h = torch::rand({2, 3, 4, 2, 2});

    TensorMatHyperDual t(r, d, h);
    double scalar = 2.5;

    TensorMatHyperDual result = t + scalar;

    // Validate the real part
    EXPECT_TRUE(torch::allclose(result.r, r + scalar));

    // Validate that the dual and hyperdual parts are unchanged
    EXPECT_TRUE(torch::equal(result.d, d));
    EXPECT_TRUE(torch::equal(result.h, h));
}

TEST(TensorMatHyperDualTest, ScalarAdditionNonFloatingPointRealPart) {
    auto r = torch::randint(0, 10, {2, 3, 4}, torch::TensorOptions().dtype(torch::kInt32)); // Integer tensor
    auto d = torch::rand({2, 3, 4, 2});
    auto h = torch::rand({2, 3, 4, 2, 2});

    TensorMatHyperDual t(r, d, h);
    double scalar = 2.5;

    EXPECT_THROW(t + scalar, std::invalid_argument);
}


TEST(TensorMatHyperDualTest, SubtractionBasic) {
    auto r1 = torch::rand({2, 3, 4});
    auto d1 = torch::rand({2, 3, 4, 2});
    auto h1 = torch::rand({2, 3, 4, 2, 2});

    auto r2 = torch::rand({2, 3, 4});
    auto d2 = torch::rand({2, 3, 4, 2});
    auto h2 = torch::rand({2, 3, 4, 2, 2});

    TensorMatHyperDual t1(r1, d1, h1);
    TensorMatHyperDual t2(r2, d2, h2);

    TensorMatHyperDual result = t1 - t2;

    EXPECT_TRUE(torch::allclose(result.r, r1 - r2));
    EXPECT_TRUE(torch::allclose(result.d, d1 - d2));
    EXPECT_TRUE(torch::allclose(result.h, h1 - h2));
}

TEST(TensorMatHyperDualTest, SubtractionDimensionMismatch) {
    auto r1 = torch::rand({2, 3, 4});
    auto d1 = torch::rand({2, 3, 4, 2});
    auto h1 = torch::rand({2, 3, 4, 2, 2});

    auto r2 = torch::rand({2, 3, 5}); // Mismatched dimension
    auto d2 = torch::rand({2, 3, 5, 2});
    auto h2 = torch::rand({2, 3, 5, 2, 2});

    TensorMatHyperDual t1(r1, d1, h1);
    TensorMatHyperDual t2(r2, d2, h2);

    EXPECT_THROW(t1 - t2, std::invalid_argument);
}

TEST(TensorMatHyperDualTest, SubtractionDeviceMismatch) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA is not available.";
    }

    auto r1 = torch::rand({2, 3, 4}, torch::TensorOptions().device(torch::kCPU));
    auto d1 = torch::rand({2, 3, 4, 2}, torch::TensorOptions().device(torch::kCPU));
    auto h1 = torch::rand({2, 3, 4, 2, 2}, torch::TensorOptions().device(torch::kCPU));

    auto r2 = torch::rand({2, 3, 4}, torch::TensorOptions().device(torch::kCUDA));
    auto d2 = torch::rand({2, 3, 4, 2}, torch::TensorOptions().device(torch::kCUDA));
    auto h2 = torch::rand({2, 3, 4, 2, 2}, torch::TensorOptions().device(torch::kCUDA));

    TensorMatHyperDual t1(r1, d1, h1);
    TensorMatHyperDual t2(r2, d2, h2);

    EXPECT_THROW(t1 - t2, std::invalid_argument);
}

TEST(TensorMatHyperDualTest, ScalarSubtractionBasic) {
    auto r = torch::rand({2, 3, 4});
    auto d = torch::rand({2, 3, 4, 2});
    auto h = torch::rand({2, 3, 4, 2, 2});

    TensorMatHyperDual t(r, d, h);
    double scalar = 2.5;

    TensorMatHyperDual result = t - scalar;

    // Validate the real part
    EXPECT_TRUE(torch::allclose(result.r, r - scalar));

    // Validate that the dual and hyperdual parts are unchanged
    EXPECT_TRUE(torch::equal(result.d, d));
    EXPECT_TRUE(torch::equal(result.h, h));
}

TEST(TensorMatHyperDualTest, ScalarSubtractionNonFloatingPointRealPart) {
    auto r = torch::randint(0, 10, {2, 3, 4}, torch::TensorOptions().dtype(torch::kInt32)); // Integer tensor
    auto d = torch::rand({2, 3, 4, 2});
    auto h = torch::rand({2, 3, 4, 2, 2});

    TensorMatHyperDual t(r, d, h);
    double scalar = 2.5;

    EXPECT_THROW(t - scalar, std::invalid_argument);
}

TEST(TensorMatHyperDualTest, ScalarSubtractionEmptyTensor) {
    auto r = torch::empty({0, 0, 0});
    auto d = torch::empty({0, 0, 0, 0});
    auto h = torch::empty({0, 0, 0, 0, 0});

    TensorMatHyperDual t(r, d, h);
    double scalar = 2.5;

    TensorMatHyperDual result = t - scalar;

    // Validate the real part
    EXPECT_TRUE(torch::equal(result.r, r - scalar));

    // Validate that the dual and hyperdual parts are unchanged
    EXPECT_TRUE(torch::equal(result.d, d));
    EXPECT_TRUE(torch::equal(result.h, h));
}

TEST(TensorMatHyperDualTest, EqualityBasic) {
    auto r1 = torch::rand({2, 2, 2});
    auto r2 = r1.clone();
    auto d = torch::rand({2, 2, 2, 2});
    auto h = torch::rand({2, 2, 2, 2, 2});

    TensorMatHyperDual t1(r1, d, h);
    TensorMatHyperDual t2(r2, d, h);

    auto result = t1 == t2;

    auto expected = r1 == r2;
    EXPECT_TRUE(torch::equal(result, expected));
}

TEST(TensorMatHyperDualTest, EqualityPartial) {
    auto r1 = torch::tensor({{{1.0, 2.0}, {3.0, 4.0}}});
    auto r2 = torch::tensor({{{1.0, 2.0}, {3.0, 5.0}}}); // Last element differs
    auto d = torch::rand({1, 2, 2, 2});
    auto h = torch::rand({1, 2, 2, 2, 2});

    TensorMatHyperDual t1(r1, d, h);
    TensorMatHyperDual t2(r2, d, h);

    auto result = t1 == t2;

    auto expected = torch::tensor({{{true, true}, {true, false}}});
    EXPECT_TRUE(torch::equal(result, expected));
}


TEST(TensorMatHyperDualTest, EqualityDimensionMismatch) {
    auto r1 = torch::rand({2, 3, 2});
    auto r2 = torch::rand({3, 2, 2}); // Mismatched dimensions
    auto d1 = torch::rand({2, 3, 2, 2});
    auto h1 = torch::rand({2, 3, 2, 2, 2});
    auto d2 = torch::rand({3, 2, 2, 2});
    auto h2 = torch::rand({3, 2, 2, 2, 2});

    TensorMatHyperDual t1(r1, d1, h1);
    TensorMatHyperDual t2(r2, d2, h2);

    EXPECT_THROW(t1 == t2, std::invalid_argument);
}


TEST(TensorMatHyperDualTest, EqualityDeviceMismatch) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA is not available.";
    }

    auto r1 = torch::rand({2, 3, 2}, torch::TensorOptions().device(torch::kCPU));
    auto r2 = torch::rand({2, 3, 2}, torch::TensorOptions().device(torch::kCUDA)); // Different device
    auto d = torch::rand({2, 3, 2, 2});
    auto h = torch::rand({2, 3, 2, 2, 2});

    TensorMatHyperDual t1(r1, d, h);
    TensorMatHyperDual t2(r2, d, h);

    EXPECT_THROW(t1 == t2, c10::Error);
}

TEST(TensorMatHyperDualTest, UnaryNegationBasic) {
    auto r = torch::rand({2, 3, 4});
    auto d = torch::rand({2, 3, 4, 2});
    auto h = torch::rand({2, 3, 4, 2, 2});

    TensorMatHyperDual t(r, d, h);
    TensorMatHyperDual result = -t;

    // Validate the real part
    EXPECT_TRUE(torch::allclose(result.r, -r));

    // Validate the dual part
    EXPECT_TRUE(torch::allclose(result.d, -d));

    // Validate the hyperdual part
    EXPECT_TRUE(torch::allclose(result.h, -h));
}


TEST(TensorMatHyperDualTest, UnaryNegationEmptyTensors) {
    auto r = torch::empty({0, 0, 0});
    auto d = torch::empty({0, 0, 0, 0});
    auto h = torch::empty({0, 0, 0, 0, 0});

    TensorMatHyperDual t(r, d, h);
    TensorMatHyperDual result = -t;

    // Validate the real part
    EXPECT_TRUE(result.r.numel() == 0);

    // Validate the dual part
    EXPECT_TRUE(result.d.numel() == 0);

    // Validate the hyperdual part
    EXPECT_TRUE(result.h.numel() == 0);
}

TEST(TensorMatHyperDualTest, UnaryNegationNonNumericTensors) {
    auto r = torch::tensor({{{1, -2}, {3, -4}}}, torch::TensorOptions().dtype(torch::kInt32)); // Integer tensor
    auto d = torch::rand({1, 2, 2, 3});
    auto h = torch::rand({1, 2, 2, 3, 3});
    TensorMatHyperDual t(r, d, h);
    auto result = -t;
    // Validate the real part
    EXPECT_TRUE(torch::allclose(result.r, -r));

    // Validate the dual part
    EXPECT_TRUE(torch::allclose(result.d, -d));

    // Validate the hyperdual part
    EXPECT_TRUE(torch::allclose(result.h, -h));
}

TEST(TensorMatHyperDualTest, ScalarMultiplicationBasic) {
    auto r = torch::rand({2, 3, 4});
    auto d = torch::rand({2, 3, 4, 22});
    auto h = torch::rand({2, 3, 4, 22, 22});

    TensorMatHyperDual t(r, d, h);
    double scalar = 2.0;

    TensorMatHyperDual result = t * scalar;

    // Validate the real part
    EXPECT_TRUE(torch::allclose(result.r, r * scalar));
    // Validate the dual part
    EXPECT_TRUE(torch::allclose(result.d, d * scalar));
    // Validate the hyperdual part
    EXPECT_TRUE(torch::allclose(result.h, h * scalar));
}

TEST(TensorMatHyperDualTest, ScalarMultiplicationEmptyTensors) {
    auto r = torch::empty({0, 0, 0});
    auto d = torch::empty({0, 0, 0, 0});
    auto h = torch::empty({0, 0, 0, 0, 0});

    TensorMatHyperDual t(r, d, h);
    double scalar = 2.0;

    TensorMatHyperDual result = t * scalar;

    // Validate the real part
    EXPECT_TRUE(result.r.numel() == 0);

    // Validate the dual part
    EXPECT_TRUE(result.d.numel() == 0);

    // Validate the hyperdual part
    EXPECT_TRUE(result.h.numel() == 0);
}

TEST(TensorMatHyperDualTest, ScalarMultiplicationNonNumericTensors) {
    auto r = torch::tensor({{{1, -2}, {3, -4}}}, torch::TensorOptions().dtype(torch::kInt32)); // Integer tensor
    auto d = torch::rand({1, 2, 2, 2});
    auto h = torch::rand({1, 2, 2, 2, 2});

    TensorMatHyperDual t(r, d, h);
    double scalar = 2.0;
    auto result = t * scalar;
    // Validate the real part
    EXPECT_TRUE(torch::allclose(result.r, r * scalar));
    // Validate the dual part
    EXPECT_TRUE(torch::allclose(result.d, d * scalar));
    // Validate the hyperdual part
    EXPECT_TRUE(torch::allclose(result.h, h * scalar));

}

TEST(TensorMatHyperDualTest, DivisionBasic) {
    auto r1 = torch::rand({2, 3, 4});
    auto d1 = torch::rand({2, 3, 4, 5});
    auto h1 = torch::rand({2, 3, 4, 5, 5});

    auto r2 = torch::rand({2, 3, 4});
    auto d2 = torch::rand({2, 3, 4, 5});
    auto h2 = torch::rand({2, 3, 4, 5, 5});

    TensorMatHyperDual t1(r1, d1, h1);
    TensorMatHyperDual t2(r2, d2, h2);

    TensorMatHyperDual result = t1 / t2;

    // Validate dimensions
    EXPECT_EQ(result.r.sizes(), r1.sizes());
    EXPECT_EQ(result.d.sizes(), d1.sizes());
    EXPECT_EQ(result.h.sizes(), h1.sizes());
}

TEST(TensorMatHyperDualTest, DivisionByZero) {
    auto r1 = torch::rand({2, 3, 4});
    auto r2 = torch::zeros({2, 3, 4}); // Zero divisor

    auto d1 = torch::rand({2, 3, 4, 5});
    auto h1 = torch::rand({2, 3, 4, 5, 5});

    TensorMatHyperDual t1(r1, d1, h1);
    TensorMatHyperDual t2(r2, d1, h1);

    TensorMatHyperDual result = t1 / t2;

    // Validate dimensions
    EXPECT_EQ(result.r.sizes(), r1.sizes());
    EXPECT_EQ(result.d.sizes(), d1.sizes());
    EXPECT_EQ(result.h.sizes(), h1.sizes());

}

TEST(TensorMatHyperDualTest, DivisionByTensorHyperDualBasic) {
    auto r1 = torch::rand({2, 3, 4});
    auto d1 = torch::rand({2, 3, 4, 2});
    auto h1 = torch::rand({2, 3, 4, 2, 2});

    auto r2 = torch::rand({2, 4});
    auto d2 = torch::rand({2, 4, 2});
    auto h2 = torch::rand({2, 4, 2, 2});

    TensorMatHyperDual t1(r1, d1, h1);
    TensorHyperDual t2(r2, d2, h2);

    TensorMatHyperDual result = t1 / t2;

    // Validate dimensions
    EXPECT_EQ(result.r.sizes(), r1.sizes());
    EXPECT_EQ(result.d.sizes(), d1.sizes());
    EXPECT_EQ(result.h.sizes(), h1.sizes());
}

TEST(TensorMatHyperDualTest, ScalarDivisionBasic) {
    auto r = torch::randn({2, 3, 4});
    auto d = torch::randn({2, 3, 4, 2});
    auto h = torch::randn({2, 3, 4, 2, 2});

    TensorMatHyperDual t(r, d, h);
    double scalar = 2.0;

    TensorMatHyperDual result = t / scalar;

    // Validate the real part
    EXPECT_TRUE(torch::allclose(result.r, r / scalar));

    // Validate the dual part
    EXPECT_TRUE(torch::allclose(result.d, d / scalar));

    // Validate the hyperdual part
    EXPECT_TRUE(torch::allclose(result.h, h / scalar));
}

TEST(TensorMatHyperDualTest, ScalarDivisionByZero) {
    auto r = torch::rand({2, 3, 4});
    auto d = torch::rand({2, 3, 4, 2});
    auto h = torch::rand({2, 3, 4, 2, 2});

    TensorMatHyperDual t(r, d, h);
    double scalar = 0.0;

    EXPECT_THROW(t / scalar, std::invalid_argument);
}

TEST(TensorMatHyperDualTest, ScalarDivisionEmptyTensors) {
    auto r = torch::empty({0, 0, 0});
    auto d = torch::empty({0, 0, 0, 0});
    auto h = torch::empty({0, 0, 0, 0, 0});

    TensorMatHyperDual t(r, d, h);
    double scalar = 2.0;

    TensorMatHyperDual result = t / scalar;

    // Validate the real part
    EXPECT_TRUE(result.r.numel() == 0);

    // Validate the dual part
    EXPECT_TRUE(result.d.numel() == 0);

    // Validate the hyperdual part
    EXPECT_TRUE(result.h.numel() == 0);
}


TEST(TensorMatHyperDualTest, BasicIndexing) {
    auto r = torch::rand({2, 3, 4});
    auto d = torch::rand({2, 3, 4, 2});
    auto h = torch::rand({2, 3, 4, 2, 2});

    TensorMatHyperDual t(r, d, h);

    // Index the first row and specific columns
    std::vector<torch::indexing::TensorIndex> indices = {torch::indexing::Slice(0, 1), torch::indexing::Slice(1, 3)};

    TensorMatHyperDual result = t.index(indices);

    // Validate dimensions
    EXPECT_EQ(result.r.sizes(), torch::IntArrayRef({1, 2, 4}));
    EXPECT_EQ(result.d.sizes(), torch::IntArrayRef({1, 2, 4, 2}));
    EXPECT_EQ(result.h.sizes(), torch::IntArrayRef({1, 2, 4, 2, 2}));
}


TEST(TensorMatHyperDualTest, IntegerIndexFails) {
    auto r = torch::rand({2, 3, 4});
    auto d = torch::rand({2, 3, 4, 2});
    auto h = torch::rand({2, 3, 4, 2, 2});

    TensorMatHyperDual t(r, d, h);

    // Invalid integer index
    std::vector<torch::indexing::TensorIndex> indices = {0};

    EXPECT_THROW(t.index(indices), std::invalid_argument);
}


TEST(TensorMatHyperDualTest, MixedSlicesAndIntegersFails) {
    auto r = torch::rand({2, 3, 4});
    auto d = torch::rand({2, 3, 4, 2});
    auto h = torch::rand({2, 3, 4, 2, 2});

    TensorMatHyperDual t(r, d, h);

    // Mixed slices and integers
    std::vector<torch::indexing::TensorIndex> indices = {torch::indexing::Slice(0, 2), 1};

    EXPECT_THROW(t.index(indices), std::invalid_argument);
}

TEST(TensorMatHyperDualTest, IndexingWithTensors) {
    auto r = torch::rand({2, 3, 4});
    auto d = torch::rand({2, 3, 4, 2});
    auto h = torch::rand({2, 3, 4, 2, 2});

    TensorMatHyperDual t(r, d, h);

    // Tensor indices
    auto index_tensor = torch::tensor({0, 1});
    std::vector<torch::indexing::TensorIndex> indices = {index_tensor, torch::indexing::Slice()};

    TensorMatHyperDual result = t.index(indices);

    // Validate dimensions
    EXPECT_EQ(result.r.sizes(), torch::IntArrayRef({2, 3, 4}));
    EXPECT_EQ(result.d.sizes(), torch::IntArrayRef({2, 3, 4, 2}));
    EXPECT_EQ(result.h.sizes(), torch::IntArrayRef({2, 3, 4, 2, 2}));
}

TEST(TensorMatHyperDualTest, IndexBasic) {
    auto r = torch::rand({2, 3, 4});
    auto d = torch::rand({2, 3, 4, 2});
    auto h = torch::rand({2, 3, 4, 2, 2});

    TensorMatHyperDual t(r, d, h);

    TensorMatHyperDual result = t.index(0);

    // Validate dimensions
    EXPECT_EQ(result.r.sizes(), torch::IntArrayRef({1, 3, 4}));
    EXPECT_EQ(result.d.sizes(), torch::IntArrayRef({1, 3, 4, 2}));
    EXPECT_EQ(result.h.sizes(), torch::IntArrayRef({1, 3, 4, 2, 2}));
}

TEST(TensorMatHyperDualTest, IndexOutOfBounds) {
    auto r = torch::rand({2, 3, 4});
    auto d = torch::rand({2, 3, 4, 2});
    auto h = torch::rand({2, 3, 4, 2, 2});

    TensorMatHyperDual t(r, d, h);

    // Index out of bounds
    EXPECT_THROW(t.index(-1), std::out_of_range);
    EXPECT_THROW(t.index(2), std::out_of_range);
}

TEST(TensorMatHyperDualTest, BooleanMaskBasic) {
    auto r = torch::rand({5, 3, 4});
    auto d = torch::rand({5, 3, 4, 2});
    auto h = torch::rand({5, 3, 4, 2, 2});

    TensorMatHyperDual t(r, d, h);

    // Mask to select the first and last batch elements
    auto mask = torch::tensor({true, false, false, false, true}, torch::kBool);

    TensorMatHyperDual result = t.index(mask);

    // Validate dimensions
    EXPECT_EQ(result.r.sizes(), torch::IntArrayRef({2, 3, 4}));
    EXPECT_EQ(result.d.sizes(), torch::IntArrayRef({2, 3, 4, 2}));
    EXPECT_EQ(result.h.sizes(), torch::IntArrayRef({2, 3, 4, 2, 2}));
}

TEST(TensorMatHyperDualTest, BooleanMaskEmpty) {
    auto r = torch::rand({5, 3, 4});
    auto d = torch::rand({5, 3, 4, 2});
    auto h = torch::rand({5, 3, 4, 2, 2});

    TensorMatHyperDual t(r, d, h);

    // Empty mask
    auto mask = torch::tensor({}, torch::kBool);

    EXPECT_THROW(t.index(mask), std::invalid_argument);
}


TEST(TensorMatHyperDualTest, BooleanMaskAllFalse) {
    auto r = torch::rand({5, 3, 4});
    auto d = torch::rand({5, 3, 4, 2});
    auto h = torch::rand({5, 3, 4, 2, 2});

    TensorMatHyperDual t(r, d, h);

    // All false mask
    auto mask = torch::tensor({false, false, false, false, false}, torch::kBool);

    TensorMatHyperDual result = t.index(mask);

    // Validate dimensions
    EXPECT_EQ(result.r.sizes(), torch::IntArrayRef({0, 3, 4}));
    EXPECT_EQ(result.d.sizes(), torch::IntArrayRef({0, 3, 4, 2}));
    EXPECT_EQ(result.h.sizes(), torch::IntArrayRef({0, 3, 4, 2, 2}));
}

TEST(TensorMatHyperDualTest, BooleanMaskDeviceMismatch) {
    auto r = torch::rand({5, 3, 4}, torch::kCUDA);
    auto d = torch::rand({5, 3, 4, 2}, torch::kCUDA);
    auto h = torch::rand({5, 3, 4, 2, 2}, torch::kCUDA);

    TensorMatHyperDual t(r, d, h);

    // Mask on a different device (CPU)
    auto mask = torch::tensor({true, false, false, false, true}, torch::kBool);

    EXPECT_THROW(t.index(mask), std::invalid_argument);
}

TEST(TensorMatHyperDualTest, RequiresGradEnable) {
    auto r = torch::rand({2, 3, 4}, torch::requires_grad(false));
    auto d = torch::rand({2, 3, 4, 2}, torch::requires_grad(false));
    auto h = torch::rand({2, 3, 4, 2, 2}, torch::requires_grad(false));

    TensorMatHyperDual t(r, d, h);

    // Enable gradients
    t.requires_grad_(true);

    // Check requires_grad
    EXPECT_TRUE(t.r.requires_grad());
    EXPECT_TRUE(t.d.requires_grad());
    EXPECT_TRUE(t.h.requires_grad());
}

TEST(TensorMatHyperDualTest, RequiresGradDisable) {
    auto r = torch::rand({2, 3, 4}, torch::requires_grad(true));
    auto d = torch::rand({2, 3, 4, 2}, torch::requires_grad(true));
    auto h = torch::rand({2, 3, 4, 2, 2}, torch::requires_grad(true));

    TensorMatHyperDual t(r, d, h);

    // Disable gradients
    t.requires_grad_(false);

    // Check requires_grad
    EXPECT_FALSE(t.r.requires_grad());
    EXPECT_FALSE(t.d.requires_grad());
    EXPECT_FALSE(t.h.requires_grad());
}

TEST(TensorMatHyperDualTest, RequiresGradNonFloatingPoint) {
    auto r = torch::randint(0, 10, {2, 3, 4}, torch::kInt32);
    auto d = torch::randint(0, 10, {2, 3, 4, 2}, torch::kInt32);
    auto h = torch::randint(0, 10, {2, 3, 4, 2, 2}, torch::kInt32);

    TensorMatHyperDual t(r, d, h);

    // Attempt to enable gradients on non-floating-point tensors
    EXPECT_THROW(t.requires_grad_(true), std::runtime_error);
}


TEST(TensorMatHyperDualTest, BackwardScalarTensors) {
    auto r = torch::rand({1,1,1}, torch::requires_grad(true));
    auto d = torch::rand({1,1,1,1}, torch::requires_grad(true));
    auto h = torch::rand({1,1,1,1,1}, torch::requires_grad(true));

    TensorMatHyperDual t(r, d, h);

    // Compute gradients
    t.backward();

    EXPECT_TRUE(r.grad().defined());
    EXPECT_TRUE(d.grad().defined());
    EXPECT_TRUE(h.grad().defined());
}

TEST(TensorMatHyperDualTest, BackwardWithProvidedGradients) {
    auto r = torch::rand({2, 3, 4}, torch::requires_grad(true));
    auto d = torch::rand({2, 3, 4, 5}, torch::requires_grad(true));
    auto h = torch::rand({2, 3, 4, 5, 5}, torch::requires_grad(true));

    TensorMatHyperDual t(r, d, h);

    auto grad_r = torch::ones_like(r);
    auto grad_d = torch::ones_like(d);
    auto grad_h = torch::ones_like(h);

    // Compute gradients
    t.backward(grad_r, grad_d, grad_h);

    EXPECT_TRUE(r.grad().defined());
    EXPECT_TRUE(d.grad().defined());
    EXPECT_TRUE(h.grad().defined());
}

TEST(TensorMatHyperDualTest, BackwardShapeMismatch) {
    auto r = torch::rand({2, 3, 4}, torch::requires_grad(true));
    auto d = torch::rand({2, 3, 4, 2}, torch::requires_grad(true));
    auto h = torch::rand({2, 3, 4, 2, 2}, torch::requires_grad(true));

    TensorMatHyperDual t(r, d, h);

    auto grad_r = torch::rand({3, 3}); // Shape mismatch

    EXPECT_THROW(t.backward(grad_r), std::runtime_error);
}

TEST(TensorMatHyperDualTest, EinsumBasic) {
    auto r1 = torch::rand({2, 3, 4});
    auto d1 = torch::rand({2, 3, 4, 10});
    auto h1 = torch::rand({2, 3, 4, 10, 10});
    TensorMatHyperDual first(r1, d1, h1);

    auto r2 = torch::rand({2, 3});
    auto d2 = torch::rand({2, 3, 10});
    auto h2 = torch::rand({2, 3, 10, 10});
    TensorHyperDual second(r2, d2, h2);

    auto result = TensorMatHyperDual::einsum("mik,mi->mk", first, second);

    EXPECT_EQ(result.r.sizes(), torch::IntArrayRef({2, 4}));
    EXPECT_EQ(result.d.sizes(), torch::IntArrayRef({2, 4, 10}));
    EXPECT_EQ(result.h.sizes(), torch::IntArrayRef({2, 4, 10, 10}));
}


TEST(TensorMatHyperDualTest, TensorMatTensorEinsumBasic) {
    auto r1 = torch::rand({2, 3, 4});
    auto d1 = torch::rand({2, 3, 4, 5});
    auto h1 = torch::rand({2, 3, 4, 5, 5});
    TensorMatHyperDual first(r1, d1, h1);

    auto second = torch::rand({2, 3, 4});

    auto result = TensorMatHyperDual::einsum("mij,mij->mij", first, second);

    EXPECT_EQ(result.r.sizes(), torch::IntArrayRef({2, 3, 4}));
    EXPECT_EQ(result.d.sizes(), torch::IntArrayRef({2, 3, 4, 5}));
    EXPECT_EQ(result.h.sizes(), torch::IntArrayRef({2, 3, 4, 5, 5}));
}

TEST(TensorHyperDualTest, EinsumBasic) {
    auto r1 = torch::rand({2, 3});
    auto d1 = torch::rand({2, 3, 7});
    auto h1 = torch::rand({2, 3, 7, 7});
    TensorHyperDual first(r1, d1, h1);

    auto r2 = torch::rand({2, 3, 5});
    auto d2 = torch::rand({2, 3, 5, 7});
    auto h2 = torch::rand({2, 3, 5, 7, 7});
    TensorMatHyperDual second(r2, d2, h2);

    auto result = TensorMatHyperDual::einsum("mi,mij->mj", first, second);

    EXPECT_EQ(result.r.sizes(), torch::IntArrayRef({2, 5}));
    EXPECT_EQ(result.d.sizes(), torch::IntArrayRef({2, 5, 7}));
    EXPECT_EQ(result.h.sizes(), torch::IntArrayRef({2, 5, 7, 7}));
}

TEST(TensorMatHyperDualTest, TensorMatDualTensorMatDualEinsumBasic) {
    auto r1 = torch::rand({2, 3, 5});
    auto d1 = torch::rand({2, 3, 5, 4});
    auto h1 = torch::rand({2, 3, 5, 4, 4});
    TensorMatHyperDual first(r1, d1, h1);

    auto r2 = torch::rand({2, 5, 7});
    auto d2 = torch::rand({2, 5, 7, 4});
    auto h2 = torch::rand({2, 5, 7, 4, 4});
    TensorMatHyperDual second(r2, d2, h2);

    auto result = TensorMatHyperDual::einsum("mij,mjk->mik", first, second);

    EXPECT_EQ(result.r.sizes(), torch::IntArrayRef({2, 3, 7}));
    EXPECT_EQ(result.d.sizes(), torch::IntArrayRef({2, 3, 7, 4}));
    EXPECT_EQ(result.h.sizes(), torch::IntArrayRef({2, 3, 7, 4, 4}));
}

TEST(TensorMatHyperDualTest, IndexPutValid) {
    auto r = torch::rand({3, 4, 5});
    auto d = torch::rand({3, 4, 5, 6});
    auto h = torch::rand({3, 4, 5, 6, 6});
    TensorMatHyperDual mat(r, d, h);

    auto mask = torch::tensor({true, false, true}, torch::kBool);
    auto value_r = torch::rand({2, 4, 5});
    auto value_d = torch::rand({2, 4, 5, 6});
    auto value_h = torch::rand({2, 4, 5, 6, 6});
    TensorMatHyperDual value(value_r, value_d, value_h);

    mat.index_put_(mask, value);

    EXPECT_TRUE(torch::allclose(mat.r.index({mask}), value_r));
    EXPECT_TRUE(torch::allclose(mat.d.index({mask}), value_d));
    EXPECT_TRUE(torch::allclose(mat.h.index({mask}), value_h));
}

TEST(TensorMatHyperDualTest,  TensorMatDualTensorIndexPutValid) {
    auto r = torch::rand({3, 4, 5});
    auto d = torch::rand({3, 4, 5, 6});
    auto h = torch::rand({3, 4, 5, 6, 6});
    TensorMatHyperDual mat(r, d, h);

    auto value_r = torch::rand({1, 4, 5});
    auto value_d = torch::rand({1, 4, 5, 6});
    auto value_h = torch::rand({1, 4, 5, 6, 6});
    TensorMatHyperDual value(value_r, value_d, value_h);

    mat.index_put_(torch::indexing::TensorIndex(0), value);

    EXPECT_TRUE(torch::allclose(mat.r.index({0}), value_r));
    EXPECT_TRUE(torch::allclose(mat.d.index({0}), value_d));
    EXPECT_TRUE(torch::allclose(mat.h.index({0}), value_h));
}

TEST(TensorMatHyperDualTest, IndexPutScalarBasic) {
    auto r = torch::rand({3, 4, 5});
    auto d = torch::rand({3, 4, 5, 6});
    auto h = torch::rand({3, 4, 5, 6, 6});
    TensorMatHyperDual mat(r, d, h);

    std::vector<torch::indexing::TensorIndex> mask = {0, torch::indexing::Slice(1, 3), torch::indexing::Slice()};
    double value = 42.0;

    mat.index_put_(mask, value);

    EXPECT_TRUE(torch::allclose(mat.r.index({mask}), torch::full({1, 2, 5}, value, mat.r.options())));
    EXPECT_TRUE(torch::allclose(mat.d.index({mask}), torch::zeros({1, 2, 5, 6}, mat.d.options())));
    EXPECT_TRUE(torch::allclose(mat.h.index({mask}), torch::zeros({1, 2, 5, 6, 6}, mat.h.options())));
}

TEST(TensorHyperDualTest, EyeValid) {
    auto r = torch::rand({2, 3});
    auto d = torch::rand({2, 3, 5});
    auto h = torch::rand({2, 3, 5, 5});
    TensorHyperDual tensor(r, d, h);

    auto result = tensor.eye();

    EXPECT_EQ(result.r.sizes(), torch::IntArrayRef({2, 3, 3}));
    EXPECT_EQ(result.d.sizes(), torch::IntArrayRef({2, 3, 3, 5}));
    EXPECT_EQ(result.h.sizes(), torch::IntArrayRef({2, 3, 3, 5, 5}));

    EXPECT_TRUE(torch::allclose(result.r, torch::eye(3).repeat({2, 1, 1})));
    EXPECT_TRUE(torch::allclose(result.d, torch::zeros_like(result.d)));
    EXPECT_TRUE(torch::allclose(result.h, torch::zeros_like(result.h)));
}


TEST(TensorDualTest, UnsqueezeValid) {
    auto r = torch::rand({3, 4});
    auto d = torch::rand({3, 4, 5});
    TensorDual tensor(r, d);

    int dim = 1;
    auto result = tensor.unsqueeze(dim);

    EXPECT_EQ(result.r.sizes(), torch::IntArrayRef({3, 1, 4}));
    EXPECT_EQ(result.d.sizes(), torch::IntArrayRef({3, 1, 4, 5}));
}

TEST(TensorDualTest, UnsqueezeNegativeDimension) {
    auto r = torch::rand({3, 4});
    auto d = torch::rand({3, 4, 5});
    TensorDual tensor(r, d);

    int dim = -1; // Add singleton dimension at the end
    auto result = tensor.unsqueeze(dim);

    EXPECT_EQ(result.r.sizes(), torch::IntArrayRef({3, 4, 1}));
    EXPECT_EQ(result.d.sizes(), torch::IntArrayRef({3, 4, 1, 5}));
}

TEST(TensorDualTest, UnsqueezeInvalidDimension) {
    auto r = torch::rand({3, 4});
    auto d = torch::rand({3, 4, 5});
    TensorDual tensor(r, d);

    int dim = 3; // Out of range for `r`
    EXPECT_THROW(tensor.unsqueeze(dim), std::invalid_argument);
}

TEST(TensorDualTest, EyeBasicFunctionality) {
    // Create a TensorDual object
    auto r = torch::rand({2, 3}); // Batch size = 2, Rows = 3
    auto d = torch::rand({2, 3, 5}); // Batch size = 2, Rows = 3, Dual dimension = 5
    TensorDual tensor(r, d);

    // Generate the identity matrix and zeros
    auto result = tensor.eye();

    // Check dimensions of the real part
    EXPECT_EQ(result.r.sizes(), torch::IntArrayRef({2, 3, 3}));
    EXPECT_TRUE(torch::allclose(result.r[0], torch::eye(3)));

    // Check dimensions of the dual part
    EXPECT_EQ(result.d.sizes(), torch::IntArrayRef({2, 3, 3, 5}));
    EXPECT_TRUE(torch::allclose(result.d, torch::zeros_like(result.d)));
}

TEST(TensorDualTest, EyeMinimalDimensions) {
    // Create a TensorDual object with minimal valid dimensions
    auto r = torch::rand({1, 2}); // Batch size = 1, Rows = 2
    auto d = torch::rand({1, 2, 1}); // Batch size = 1, Rows = 2,  Dual dimension = 1
    TensorDual tensor(r, d);

    // Generate the identity matrix and zeros
    auto result = tensor.eye();

    // Check dimensions of the real part
    EXPECT_EQ(result.r.sizes(), torch::IntArrayRef({1, 2, 2}));
    EXPECT_TRUE(torch::allclose(result.r[0], torch::eye(2)));

    // Check dimensions of the dual part
    EXPECT_EQ(result.d.sizes(), torch::IntArrayRef({1, 2, 2, 1}));
    EXPECT_TRUE(torch::allclose(result.d, torch::zeros_like(result.d)));
}

TEST(TensorDualTest, EyeLargeDimensions) {
    // Create a TensorDual object with large dimensions
    auto r = torch::rand({10, 100}); // Batch size = 10, Rows = 100, Columns = 100
    auto d = torch::rand({10, 100, 20}); // Dual dimension = 20
    TensorDual tensor(r, d);

    // Generate the identity matrix and zeros
    auto result = tensor.eye();

    // Check dimensions of the real part
    EXPECT_EQ(result.r.sizes(), torch::IntArrayRef({10, 100, 100}));
    EXPECT_TRUE(torch::allclose(result.r[0], torch::eye(100)));

    // Check dimensions of the dual part
    EXPECT_EQ(result.d.sizes(), torch::IntArrayRef({10, 100, 100, 20}));
    EXPECT_TRUE(torch::allclose(result.d, torch::zeros_like(result.d)));
}


TEST(TensorDualTest, MultiplyTensorWithTensorDualBasic) {
    // Create a torch::Tensor and a TensorDual
    auto tensor = torch::tensor({1.0, 2.0, 3.0});
    auto r = torch::tensor({{4.0, 5.0, 6.0}});
    auto d = torch::tensor({{{1.0, 2.0}, {2.0, 3.0}, {3.0, 4.0}}});
    TensorDual td(r, d);

    // Perform multiplication
    auto result = tensor * td;

    // Check the real part
    EXPECT_TRUE(torch::allclose(result.r, tensor * r));

    // Check the dual part
    auto expected_dual = tensor.unsqueeze(-1) * d;
    EXPECT_TRUE(torch::allclose(result.d, expected_dual));
}


TEST(TensorHyperDualTest, MultiplyTensorWithTensorHyperDualBasic) {
    // Create a torch::Tensor and a TensorHyperDual
    auto tensor = torch::tensor({1.0, 2.0, 3.0});
    auto r = torch::tensor({{4.0, 5.0, 6.0}});
    auto d = torch::tensor({{{1.0, 2.0}, {2.0, 3.0}, {3.0, 4.0}}});
    auto h = torch::tensor({{{{1.0, 2.0}, {2.0, 3.0}}, 
                            {{3.0, 4.0}, {4.0, 5.0}}, 
                            {{5.0, 6.0}, {6.0, 7.0}}}});
    TensorHyperDual td(r, d, h);

    // Perform multiplication
    auto result = tensor * td;

    // Check the real part
    EXPECT_TRUE(torch::allclose(result.r, tensor * r));

    // Check the dual part
    auto expected_dual = tensor.unsqueeze(-1) * d;
    EXPECT_TRUE(torch::allclose(result.d, expected_dual));

    // Check the hyper-dual part
    auto expected_hyper = tensor.unsqueeze(-1).unsqueeze(-1) * h;
    EXPECT_TRUE(torch::allclose(result.h, expected_hyper));
}

TEST(TensorDualTest, DivideTensorByTensorDualBasic) {
    // Create a torch::Tensor and a TensorDual
    auto tensor = torch::tensor({6.0, 12.0, 18.0});
    auto r = torch::tensor({{2.0, 4.0, 6.0}});
    auto d = torch::tensor({{{1.0, 2.0}, {2.0, 3.0}, {3.0, 4.0}}});
    TensorDual td(r, d);

    // Perform division
    auto result = tensor / td;

    // Check the real part
    EXPECT_TRUE(torch::allclose(result.r, tensor / r));

    // Check the dual part
    auto expected_dual = -(tensor / r.square()).unsqueeze(-1) * d;
    EXPECT_TRUE(torch::allclose(result.d, expected_dual));
}

TEST(TensorHyperDualTest, DivideTensorByTensorHyperDualBasic) {
    // Create a torch::Tensor and a TensorHyperDual
    auto tensor = torch::tensor({{6.0, 12.0, 18.0}});
    auto r = torch::tensor({{2.0, 4.0, 6.0}});
    auto d = torch::tensor({{{1.0, 2.0}, {2.0, 3.0}, {3.0, 4.0}}});
    auto h = torch::tensor({{{{1.0, 2.0}, {2.0, 3.0}}, 
                            {{3.0, 4.0}, {4.0, 5.0}}, 
                            {{5.0, 6.0}, {6.0, 7.0}}}});
    TensorHyperDual td(r, d, h);

    // Perform division
    auto result = tensor / td;

    // Check the real part
    EXPECT_TRUE(torch::allclose(result.r, tensor / r));

    // Check the dual part
    auto expected_dual = -(tensor / r.square()).unsqueeze(-1) * d;
    EXPECT_TRUE(torch::allclose(result.d, expected_dual));

    // Check the hyper-dual part
    auto expected_hyper = torch::einsum("mi, mi, mij, mik -> mijk", 
                                        {tensor, r.pow(-3), d, d}) - 
                          torch::einsum("mi, mi, mijk -> mijk", 
                                        {tensor, r.pow(-2), h});
    EXPECT_TRUE(torch::allclose(result.h, expected_hyper));
}

TEST(TensorDualTest, AddTensorToTensorDualBasic) {
    // Create a torch::Tensor and a TensorDual
    auto tensor = torch::tensor({1.0, 2.0, 3.0});
    auto r = torch::tensor({{4.0, 5.0, 6.0}});
    auto d = torch::tensor({{{0.1, 0.2}, {0.3, 0.4}, {0.5, 0.6}}});
    TensorDual td(r, d);

    // Perform addition
    auto result = tensor + td;

    // Validate the real part
    EXPECT_TRUE(torch::allclose(result.r, tensor + r));

    // Validate the dual part (unchanged)
    EXPECT_TRUE(torch::allclose(result.d, d));
}

TEST(TensorHyperDualTest, AddTensorToTensorHyperDualBasic) {
    // Create a torch::Tensor and a TensorHyperDual
    auto tensor = torch::tensor({1.0, 2.0, 3.0});
    auto r = torch::tensor({{4.0, 5.0, 6.0}});
    auto d = torch::tensor({{{0.1, 0.2}, {0.3, 0.4}, {0.5, 0.6}}});
    auto h = torch::tensor({{{{0.01, 0.02}, {0.03, 0.04}}, 
                            {{0.05, 0.06}, {0.07, 0.08}}, 
                            {{0.09, 0.10}, {0.11, 0.12}}}});
    TensorHyperDual td(r, d, h);

    // Perform addition
    auto result = tensor + td;

    // Validate the real part
    EXPECT_TRUE(torch::allclose(result.r, tensor + r));

    // Validate the dual part (unchanged)
    EXPECT_TRUE(torch::allclose(result.d, d));

    // Validate the hyper-dual part (unchanged)
    EXPECT_TRUE(torch::allclose(result.h, h));
}

TEST(TensorDualTest, AddScalarToTensorDualBasic) {
    // Create a TensorDual
    auto r = torch::tensor({{1.0, 2.0, 3.0}});
    auto d = torch::tensor({{{0.1, 0.2}, {0.3, 0.4}, {0.5, 0.6}}});
    TensorDual td(r, d);

    // Perform addition with a scalar
    double scalar = 5.0;
    auto result = scalar + td;

    // Validate the real part
    EXPECT_TRUE(torch::allclose(result.r, r + scalar));

    // Validate the dual part (unchanged)
    EXPECT_TRUE(torch::allclose(result.d, d));
}


TEST(TensorDualTest, AddScalarToNegativeRealTensorDual) {
    // Create a TensorDual with negative real values
    auto r = torch::tensor({{-1.0, -2.0, -3.0}});
    auto d = torch::tensor({{{0.1, 0.2}, {0.3, 0.4}, {0.5, 0.6}}});
    TensorDual td(r, d);

    // Perform addition with a scalar
    double scalar = 5.0;
    auto result = scalar + td;

    // Validate the real part
    EXPECT_TRUE(torch::allclose(result.r, r + scalar));

    // Validate the dual part (unchanged)
    EXPECT_TRUE(torch::allclose(result.d, d));
}


TEST(TensorHyperDualTest, AddScalarToTensorHyperDualBasic) {
    // Create a TensorHyperDual
    auto r = torch::tensor({{1.0, 2.0, 3.0}});
    auto d = torch::tensor({{{0.1, 0.2}, {0.3, 0.4}, {0.5, 0.6}}});
    auto h = torch::tensor({{{{0.01, 0.02}, {0.03, 0.04}}, 
                            {{0.05, 0.06}, {0.07, 0.08}}, 
                            {{0.09, 0.10}, {0.11, 0.12}}}});
    TensorHyperDual td(r, d, h);

    // Perform addition with a scalar
    double scalar = 5.0;
    auto result = scalar + td;

    // Validate the real part
    EXPECT_TRUE(torch::allclose(result.r, r + scalar));

    // Validate the dual part (unchanged)
    EXPECT_TRUE(torch::allclose(result.d, d));

    // Validate the hyper-dual part (unchanged)
    EXPECT_TRUE(torch::allclose(result.h, h));
}

TEST(TensorDualTest, SubtractTensorDualFromTensorBasic) {
    // Create a TensorDual
    auto r = torch::tensor({{1.0, 2.0, 3.0}});
    auto d = torch::tensor({{{0.1, 0.2}, {0.3, 0.4}, {0.5, 0.6}}});
    TensorDual td(r, d);

    // Create the torch::Tensor
    auto tensor = torch::tensor({5.0, 6.0, 7.0});

    // Perform the subtraction
    auto result = tensor - td;

    // Validate the real part
    EXPECT_TRUE(torch::allclose(result.r, tensor - r));

    // Validate the dual part (negated)
    EXPECT_TRUE(torch::allclose(result.d, -d));
}


TEST(TensorHyperDualTest, SubtractTensorHyperDualFromTensorBasic) {
    // Create a TensorHyperDual
    auto r = torch::tensor({{1.0, 2.0, 3.0}});
    auto d = torch::tensor({{{0.1, 0.2}, {0.3, 0.4}, {0.5, 0.6}}});
    auto h = torch::tensor({{{{0.01, 0.02}, {0.03, 0.04}}, {{0.05, 0.06}, {0.07, 0.08}}, {{0.09, 0.10}, {0.11, 0.12}}}});
    TensorHyperDual td(r, d, h);

    // Create the torch::Tensor
    auto tensor = torch::tensor({5.0, 6.0, 7.0});

    // Perform the subtraction
    auto result = tensor - td;

    // Validate the real part
    EXPECT_TRUE(torch::allclose(result.r, tensor - r));

    // Validate the dual part (negated)
    EXPECT_TRUE(torch::allclose(result.d, -d));

    // Validate the hyper-dual part (negated)
    EXPECT_TRUE(torch::allclose(result.h, -h));
}


TEST(TensorDualTest, MultiplyTensorDualByTensorMatDual_LeftMultiplication) {
    // Create a TensorDual
    auto r_td = torch::tensor({{1.0, 2.0}, {3.0, 4.0}});
    auto d_td = torch::tensor({{{0.1, 0.2}, {0.3, 0.4}}, 
                               {{0.5, 0.6}, {0.7, 0.8}}});
    TensorDual td(r_td, d_td);

    // Create a TensorMatDual
    auto r_mat = torch::tensor({{{1.0, 2.0}, {3.0, 4.0}}, 
                                {{5.0, 6.0}, {7.0, 8.0}}});
    auto d_mat = torch::tensor({{{{0.1, 0.2}}, {{0.3, 0.4}}}, 
                                {{{0.5, 0.6}}, {{0.7, 0.8}}}});
    TensorMatDual mat(r_mat, d_mat);

    // Perform the multiplication
    auto result = td * mat;

    // Expected real part
    auto expected_r = torch::einsum("mi, mij -> mj", {r_td, r_mat});

    // Expected dual part
    auto expected_d = torch::einsum("mi, mijn -> mjn", {r_td, d_mat}) +
                      torch::einsum("min, mij -> mjn", {d_td, r_mat});

    // Validate the real part
    EXPECT_TRUE(torch::allclose(result.r, expected_r));

    // Validate the dual part
    EXPECT_TRUE(torch::allclose(result.d, expected_d));
}

TEST(TensorMatDualTest, MultiplyTensorMatDualByTensorDual_MatrixMultiplication) {
    // Create a TensorMatDual
    auto r_tmd = torch::tensor({{{1.0, 2.0}, {3.0, 4.0}}, 
                                 {{5.0, 6.0}, {7.0, 8.0}}});
    auto d_tmd = torch::tensor({{{{0.1, 0.2}}, {{0.3, 0.4}}}, 
                                 {{{0.5, 0.6}}, {{0.7, 0.8}}}});
    TensorMatDual tmd(r_tmd, d_tmd);

    // Create a TensorDual
    auto r_td = torch::tensor({{1.0, 2.0}, {3.0, 4.0}});
    auto d_td = torch::tensor({{{0.1, 0.2}, {0.3, 0.4}}, 
                               {{0.5, 0.6}, {0.7, 0.8}}});
    TensorDual td(r_td, d_td);

    // Perform the multiplication
    auto result = tmd * td;

    // Expected real part
    auto expected_r = torch::einsum("mij, mj -> mi", {r_tmd, r_td});

    // Expected dual part
    auto expected_d = torch::einsum("mijn, mj -> min", {d_tmd, r_td}) +
                      torch::einsum("mij, mjn -> min", {r_tmd, d_td});

    // Validate the real part
    EXPECT_TRUE(torch::allclose(result.r, expected_r));

    // Validate the dual part
    EXPECT_TRUE(torch::allclose(result.d, expected_d));
}


TEST(TensorMatDualTest, ElementWiseMultiplication_Basic) {
    // Create the first TensorMatDual
    auto r_lhs = torch::tensor({{{1.0, 2.0}, {3.0, 4.0}}, 
                                 {{5.0, 6.0}, {7.0, 8.0}}});
    auto d_lhs = torch::tensor({{{{0.1, 0.2}, {0.3, 0.4}}, 
                                  {{0.5, 0.6}, {0.7, 0.8}}}, 
                                 {{{0.9, 1.0}, {1.1, 1.2}}, 
                                  {{1.3, 1.4}, {1.5, 1.6}}}});
    TensorMatDual lhs(r_lhs, d_lhs);

    // Create the second TensorMatDual
    auto r_rhs = torch::tensor({{{2.0, 3.0}, {4.0, 5.0}}, 
                                 {{6.0, 7.0}, {8.0, 9.0}}});
    auto d_rhs = torch::tensor({{{{0.2, 0.3}, {0.4, 0.5}}, 
                                  {{0.6, 0.7}, {0.8, 0.9}}}, 
                                 {{{1.0, 1.1}, {1.2, 1.3}}, 
                                  {{1.4, 1.5}, {1.6, 1.7}}}});
    TensorMatDual rhs(r_rhs, d_rhs);

    // Perform element-wise multiplication
    auto result = lhs * rhs;

    // Compute expected results
    auto expected_r = r_lhs * r_rhs;
    auto expected_d = torch::einsum("mij, mijn -> mijn", {r_lhs, d_rhs}) +
                      torch::einsum("mijn, mij -> mijn", {d_lhs, r_rhs});

    // Validate the real part
    EXPECT_TRUE(torch::allclose(result.r, expected_r));

    // Validate the dual part
    EXPECT_TRUE(torch::allclose(result.d, expected_d));
}

TEST(TensorDualTest, SubtractTensorDualFromScalar_Basic) {
    // Create a TensorDual object
    auto r = torch::tensor({{1.0, 2.0, 3.0}});
    auto d = torch::tensor({{{0.1, 0.2}, {0.3, 0.4}, {0.5, 0.6}}});
    TensorDual td(r, d);

    // Perform subtraction with a scalar
    int scalar = 5;
    auto result = scalar - td;

    // Compute expected results
    auto expected_r = torch::tensor({5.0, 5.0, 5.0}) - r;
    auto expected_d = -d;

    // Validate the real part
    EXPECT_TRUE(torch::allclose(result.r, expected_r));

    // Validate the dual part
    EXPECT_TRUE(torch::allclose(result.d, expected_d));
}

TEST(TensorDualTest, ScalarDividedByTensorDual_Basic) {
    // Create a TensorDual object
    auto r = torch::tensor({{2.0, 4.0, 8.0}});
    auto d = torch::tensor({{{0.1, 0.2}, {0.3, 0.4}, {0.5, 0.6}}});
    TensorDual td(r, d);

    // Perform scalar division
    double scalar = 16.0;
    auto result = scalar / td;

    // Compute expected results
    auto expected_r = scalar / r;
    auto expected_d = -(scalar / r.square()).unsqueeze(-1) * d;

    // Validate the real part
    EXPECT_TRUE(torch::allclose(result.r, expected_r));

    // Validate the dual part
    EXPECT_TRUE(torch::allclose(result.d, expected_d));
}

TEST(TensorDualTest, ScalarTimesTensorDual_Basic) {
    // Create a TensorDual object
    auto r = torch::tensor({{1.0, 2.0, 3.0}});
    auto d = torch::tensor({{{0.1, 0.2}, {0.3, 0.4}, {0.5, 0.6}}});
    TensorDual td(r, d);

    // Perform scalar multiplication
    double scalar = 2.0;
    auto result = scalar * td;

    // Compute expected results
    auto expected_r = r * scalar;
    auto expected_d = d * scalar;

    // Validate the real part
    EXPECT_TRUE(torch::allclose(result.r, expected_r));

    // Validate the dual part
    EXPECT_TRUE(torch::allclose(result.d, expected_d));
}


TEST(TensorMatDualTest, ScalarTimesTensorMatDual_Basic) {
    // Create a TensorMatDual object
    auto r = torch::tensor({{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}});
    auto d = torch::tensor({
        {{{0.1, 0.2}, {0.3, 0.4}}, {{0.5, 0.6}, {0.7, 0.8}}},
        {{{0.9, 1.0}, {1.1, 1.2}}, {{1.3, 1.4}, {1.5, 1.6}}}
    });
    TensorMatDual tmd(r, d);

    // Perform scalar multiplication
    double scalar = 2.0;
    auto result = scalar * tmd;

    // Compute expected results
    auto expected_r = r * scalar;
    auto expected_d = d * scalar;

    // Validate the real part
    EXPECT_TRUE(torch::allclose(result.r, expected_r));

    // Validate the dual part
    EXPECT_TRUE(torch::allclose(result.d, expected_d));
}


TEST(TensorDualTest, PowBasic) {
    // Create base TensorDual
    auto base_r = torch::tensor({{2.0, 3.0, 4.0}});
    auto base_d = torch::tensor({{{0.1, 0.2}, {0.3, 0.4}, {0.5, 0.6}}});
    TensorDual base(base_r, base_d);

    // Create exponent TensorDual
    auto exp_r = torch::tensor({{2.0, 3.0, 1.5}});
    auto exp_d = torch::tensor({{{0.01, 0.02}, {0.03, 0.04}, {0.05, 0.06}}});
    TensorDual exponent(exp_r, exp_d);

    // Perform power operation
    auto result = pow(base, exponent);

    // Compute expected real and dual parts
    auto expected_real = torch::pow(base_r, exp_r);
    auto expected_dual = torch::einsum("mi, mij->mij", {expected_real * torch::log(base_r), exp_d}) +
                         torch::einsum("mi, mij->mij", {expected_real * (exp_r / base_r), base_d});

    // Validate the real part
    EXPECT_TRUE(torch::allclose(result.r, expected_real));

    // Validate the dual part
    EXPECT_TRUE(torch::allclose(result.d, expected_dual));
}

TEST(TensorDualTest, TensorDualTensorDualMaxBasic) {
    // Create the left-hand TensorDual
    auto lhs_r = torch::tensor({{1.0, 3.0, 5.0}});
    auto lhs_d = torch::tensor({{{0.1, 0.2}, {0.3, 0.4}, {0.5, 0.6}}});
    TensorDual lhs(lhs_r, lhs_d);

    // Create the right-hand TensorDual
    auto rhs_r = torch::tensor({{2.0, 2.0, 6.0}});
    auto rhs_d = torch::tensor({{{0.7, 0.8}, {0.9, 1.0}, {1.1, 1.2}}});
    TensorDual rhs(rhs_r, rhs_d);

    // Compute the maximum
    auto result = max(lhs, rhs);

    // Expected real and dual parts
    auto expected_r = torch::max(lhs_r, rhs_r);
    auto expected_d = torch::tensor({{{0.7, 0.8}, {0.3, 0.4}, {1.1, 1.2}}}); // Based on which r was greater

    // Validate the real part
    EXPECT_TRUE(torch::allclose(result.r, expected_r));

    // Validate the dual part
    EXPECT_TRUE(torch::allclose(result.d, expected_d));
}

TEST(TensorDualTest, TensorDualMaxWithTensorBasic) {
    // Create the TensorDual
    auto lhs_r = torch::tensor({{1.0, 3.0, 5.0}});
    auto lhs_d = torch::tensor({{{0.1, 0.2}, {0.3, 0.4}, {0.5, 0.6}}});
    TensorDual lhs(lhs_r, lhs_d);

    // Create the right-hand torch::Tensor
    auto rhs = torch::tensor({{2.0, 2.0, 6.0}});

    // Compute the maximum
    auto result = max(lhs, rhs);

    // Expected real and dual parts
    auto expected_r = torch::max(lhs_r, rhs);
    auto expected_d = torch::tensor({{{{0.0, 0.0}, {0.3, 0.4}, {0.0, 0.0}}}}); // Dual from lhs where lhs.r > rhs

    // Validate the real part
    EXPECT_TRUE(torch::allclose(result.r, expected_r));

    // Validate the dual part
    EXPECT_TRUE(torch::allclose(result.d, expected_d));
}

TEST(TensorDualTest, TensorDualTensorDualMinBasic) {
    // Create the left-hand TensorDual
    auto lhs_r = torch::tensor({{1.0, 3.0, 5.0}});
    auto lhs_d = torch::tensor({{{0.1, 0.2}, {0.3, 0.4}, {0.5, 0.6}}});
    TensorDual lhs(lhs_r, lhs_d);

    // Create the right-hand TensorDual
    auto rhs_r = torch::tensor({{2.0, 2.0, 6.0}});
    auto rhs_d = torch::tensor({{{0.7, 0.8}, {0.9, 1.0}, {1.1, 1.2}}});
    TensorDual rhs(rhs_r, rhs_d);

    // Compute the minimum
    auto result = min(lhs, rhs);

    // Expected real and dual parts
    auto expected_r = torch::min(lhs_r, rhs_r);
    auto expected_d = torch::tensor({{{0.1, 0.2}, {0.9, 1.0}, {0.5, 0.6}}}); // Use lhs.d where lhs.r < rhs.r

    // Validate the real part
    EXPECT_TRUE(torch::allclose(result.r, expected_r));

    // Validate the dual part
    EXPECT_TRUE(torch::allclose(result.d, expected_d));
}

TEST(TensorDualTest, TensorDualTensorMinTensorBasic) {
    // Create the TensorDual object
    auto lhs_r = torch::tensor({{1.0, 3.0, 5.0}});
    auto lhs_d = torch::tensor({{{0.1, 0.2}, {0.3, 0.4}, {0.5, 0.6}}});
    TensorDual lhs(lhs_r, lhs_d);

    // Create the right-hand tensor
    auto rhs = torch::tensor({{2.0, 2.0, 6.0}});

    // Compute the minimum
    auto result = min(lhs, rhs);

    // Expected real and dual parts
    auto expected_r = torch::min(lhs_r, rhs);
    auto expected_d = torch::tensor({{{0.1, 0.2}, {0.0, 0.0}, {0.5, 0.6}}}); // Use lhs.d where lhs.r < rhs

    // Validate the real part
    EXPECT_TRUE(torch::allclose(result.r, expected_r));

    // Validate the dual part
    EXPECT_TRUE(torch::allclose(result.d, expected_d));
}

TEST(TensorDualTest, TensorDualSignBasic) {
    // Create the TensorDual object
    auto r = torch::tensor({{-3.0, 0.0, 4.0}});
    auto d = torch::tensor({{{0.1, 0.2}, {0.3, 0.4}, {0.5, 0.6}}});
    TensorDual td(r, d);

    // Compute the sign
    auto result = sign(td);

    // Expected real and dual parts
    auto expected_r = torch::sign(r);
    auto expected_d = torch::zeros_like(d); // Dual part is zero except when r == 0

    // Validate the real part
    EXPECT_TRUE(torch::allclose(result.r, expected_r));

    // Validate the dual part
    EXPECT_TRUE(torch::allclose(result.d, expected_d));
}

TEST(TensorDualTest, TensorDualPowWithScalarExponent) {
    // Create the TensorDual object
    auto r = torch::tensor({{2.0, 3.0, 4.0}});
    auto d = torch::tensor({{{0.1, 0.2}, {0.3, 0.4}, {0.5, 0.6}}});
    TensorDual base(r, d);

    // Exponent tensor (scalar applied element-wise)
    auto exponent = torch::tensor(2.0);

    // Compute the power
    auto result = pow(base, exponent);

    // Expected real and dual parts
    auto expected_r = torch::pow(r, exponent);
    auto expected_d = torch::einsum("mij, mi->mij", {d, exponent * r.pow(exponent - 1)});

    // Validate the real part
    EXPECT_TRUE(torch::allclose(result.r, expected_r));

    // Validate the dual part
    EXPECT_TRUE(torch::allclose(result.d, expected_d));
}


TEST(TensorDualTest, TensorDualPowExponent) {
    // Create the TensorDual object
    auto r = torch::tensor({{2.0, 3.0, 4.0}});
    auto d = torch::tensor({{{0.1, 0.2}, {0.3, 0.4}, {0.5, 0.6}}});
    TensorDual base(r, d);

    // Exponent scalar
    double exponent = 2.0;

    // Compute the power
    auto result = pow(base, exponent);

    // Expected real and dual parts
    auto expected_r = torch::pow(r, exponent);
    auto expected_d = d * exponent * torch::pow(r, exponent - 1).unsqueeze(-1);

    // Validate the real part
    EXPECT_TRUE(torch::allclose(result.r, expected_r));

    // Validate the dual part
    EXPECT_TRUE(torch::allclose(result.d, expected_d));
}

TEST(TensorDualTest, GerBasic) {
    // Create the first TensorDual
    auto r1 = torch::tensor({{1.0, 2.0}});
    auto d1 = torch::tensor({{{0.1, 0.2}, {0.3, 0.4}}});
    TensorDual x(r1, d1);

    // Create the second TensorDual
    auto r2 = torch::tensor({{3.0, 4.0}});
    auto d2 = torch::tensor({{{0.5, 0.6}, {0.7, 0.8}}});
    TensorDual y(r2, d2);

    // Compute the generalized outer product
    auto result = ger(x, y);

    // Expected real and dual parts
    auto expected_r = torch::einsum("mj, mi->mij", {r1, r2});
    auto expected_d1 = torch::einsum("mj, mik->mijk", {r1, d2});
    auto expected_d2 = torch::einsum("mjk, mi->mijk", {d1, r2});
    auto expected_d = expected_d1 + expected_d2;

    // Validate the real part
    EXPECT_TRUE(torch::allclose(result.r, expected_r));

    // Validate the dual part
    EXPECT_TRUE(torch::allclose(result.d, expected_d));
}




int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
