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
