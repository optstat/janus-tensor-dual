#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cassert>
#include "../../src/cpp/cudatensorsparse.cu"
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




// -----------------------------------------------------------------------------
// Helper to compare complex numbers with Google Test
// -----------------------------------------------------------------------------
void ExpectComplexNear(const std::complex<double>& expected,
                       const std::complex<double>& actual,
                       double tol)
{
    EXPECT_NEAR(expected.real(), actual.real(), tol);
    EXPECT_NEAR(expected.imag(), actual.imag(), tol);
}

// -----------------------------------------------------------------------------
// The actual Google Test
// -----------------------------------------------------------------------------
TEST(VectorRealDualProductSparseTest, BasicCheck)
{
    using T = double;
    const int realSize = 4;

    // ----------------------------
    // 1) Prepare host data
    // ----------------------------
    // Example: a_real[i], b_real[i] in [1..8] + i[2..16]
    std::vector<std::complex<T>> h_aReal = {
        {1,  2}, {3,  4}, {5,  6}, {7,  8} };
    std::vector<std::complex<T>> h_bReal = {
        {2, -1}, {4, -3}, {6, -5}, {8, -7} };

    // Single dual values:
    std::complex<T> h_aDual(10.0, 20.0);
    std::complex<T> h_bDual(30.0, 40.0);

    // Indices in the dual space (for demonstration, let them match)
    int h_aIdx = 2;
    int h_bIdx = 2;

    // We'll store the result in these host arrays
    std::vector<std::complex<T>> h_rReal(realSize, {0,0});
    std::complex<T> h_rDual(0, 0);
    int h_rIdx = -1;

    // ----------------------------
    // 2) Allocate/copy data to device
    // ----------------------------
    thrust::complex<T>* d_aReal   = nullptr;
    thrust::complex<T>* d_bReal   = nullptr;
    thrust::complex<T>* d_aDual   = nullptr;
    thrust::complex<T>* d_bDual   = nullptr;
    thrust::complex<T>* d_rReal   = nullptr;
    thrust::complex<T>* d_rDual   = nullptr;
    int*                d_rIdx    = nullptr;

    size_t bytesRealArray  = realSize * sizeof(thrust::complex<T>);
    size_t bytesSingleDual = sizeof(thrust::complex<T>);

    ASSERT_EQ(cudaMalloc(&d_aReal, bytesRealArray), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_bReal, bytesRealArray), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_aDual, bytesSingleDual), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_bDual, bytesSingleDual), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_rReal, bytesRealArray), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_rDual, bytesSingleDual), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_rIdx,  sizeof(int)),      cudaSuccess);

    // Copy host -> device
    ASSERT_EQ(cudaMemcpy(d_aReal, h_aReal.data(), bytesRealArray, cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_bReal, h_bReal.data(), bytesRealArray, cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_aDual, &h_aDual,        bytesSingleDual, cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_bDual, &h_bDual,        bytesSingleDual, cudaMemcpyHostToDevice), cudaSuccess);

    // ----------------------------
    // 3) Launch kernel
    // ----------------------------
    // For simplicity, do single-block with realSize <= blockDim.x
    int blockSize = 64;
    int gridSize  = 1;  
    VectorRealDualProductSparseKernel<<<gridSize, blockSize>>>(
        d_aReal, d_aDual, h_aIdx,
        d_bReal, d_bDual, h_bIdx,
        realSize,
        d_rReal, d_rDual, d_rIdx
    );
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    // ----------------------------
    // 4) Copy results back to host
    // ----------------------------
    ASSERT_EQ(cudaMemcpy(h_rReal.data(), d_rReal, bytesRealArray, cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(&h_rDual,       d_rDual, bytesSingleDual, cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(&h_rIdx,        d_rIdx,  sizeof(int),      cudaMemcpyDeviceToHost), cudaSuccess);

    // ----------------------------
    // 5) Check results vs. CPU reference
    // ----------------------------

    // CPU reference for r_real[i] = a_real[i] * b_real[i]
    std::vector<std::complex<T>> ref_rReal(realSize);
    for (int i = 0; i < realSize; ++i) {
        ref_rReal[i] = h_aReal[i] * h_bReal[i];
    }

    // CPU reference for r_dual = sum over i of [a_real[i]*b_dual + b_real[i]*a_dual]
    std::complex<T> ref_rDual(0, 0);
    for (int i = 0; i < realSize; ++i) {
        ref_rDual += h_aReal[i]*h_bDual + h_bReal[i]*h_aDual;
    }

    int ref_rIdx = h_aIdx; // or your logic; we used a_idx in the device code

    // Compare each element of r_real
    for (int i = 0; i < realSize; ++i) {
        ExpectComplexNear(ref_rReal[i], h_rReal[i], 1e-10);
    }
    // Compare dual
    ExpectComplexNear(ref_rDual, h_rDual, 1e-10);

    // Compare dual index
    EXPECT_EQ(ref_rIdx, h_rIdx);

    // ----------------------------
    // 6) Cleanup
    // ----------------------------
    cudaFree(d_aReal);
    cudaFree(d_bReal);
    cudaFree(d_aDual);
    cudaFree(d_bDual);
    cudaFree(d_rReal);
    cudaFree(d_rDual);
    cudaFree(d_rIdx);
}


// Helper: Compare thrust::complex<double> within a tolerance
static void ExpectComplexNear(const thrust::complex<double>& expected,
                              const thrust::complex<double>& actual,
                              double tol = 1e-14)
{
    EXPECT_NEAR(expected.real(), actual.real(), tol);
    EXPECT_NEAR(expected.imag(), actual.imag(), tol);
}

// Helper: CPU reference for the sparse index-get operation
static void VectorDualIndexGetSparseCPU(
    const std::vector<thrust::complex<double>>& a_real,
    const thrust::complex<double>&              a_dual,
    int                                         a_idx,
    int                                         real_size,
    int                                         start,
    int                                         end,
    // Outputs
    std::vector<thrust::complex<double>>&       ref_real,
    thrust::complex<double>&                    ref_dual,
    int&                                        ref_idx)
{
    int new_size = end - start;
    ref_real.resize(new_size);

    // Copy real part
    for (int i = 0; i < new_size; ++i) {
        ref_real[i] = a_real[start + i];
    }

    // Check if a_idx is in [start, end)
    if (a_idx >= start && a_idx < end) {
        ref_idx = a_idx - start;
        ref_dual = a_dual;
    } else {
        ref_idx = -1;
        ref_dual = thrust::complex<double>(0, 0);
    }
}


// --------------------------------------------------------------
// 1) Test: a_idx is inside [start, end)
// --------------------------------------------------------------
TEST(VectorDualIndexGetSparseTest, InRangeCase)
{
    using T = double;

    // Original vector of length 6, for example
    int real_size = 6;
    std::vector<thrust::complex<T>> h_aReal = {
        {1,2}, {3,4}, {5,6}, {7,8}, {9,10}, {11,12}
    };

    // Single dual value and index in the range
    thrust::complex<T> h_aDual(10, 20);
    int h_aIdx = 3; // within [start=2, end=5)?

    // We pick subrange [2,5) => length = 3
    int start = 2;
    int end   = 5;
    int new_size = end - start;

    // Device pointers
    thrust::complex<T>* d_aReal       = nullptr;
    thrust::complex<T>* d_aDual       = nullptr;
    thrust::complex<T>* d_resultReal  = nullptr;
    thrust::complex<T>* d_resultDual  = nullptr;
    int*                d_resultIdx   = nullptr;

    // Allocate
    ASSERT_EQ(cudaMalloc(&d_aReal, real_size*sizeof(thrust::complex<T>)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_aDual, sizeof(thrust::complex<T>)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_resultReal, new_size*sizeof(thrust::complex<T>)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_resultDual, sizeof(thrust::complex<T>)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_resultIdx,  sizeof(int)), cudaSuccess);

    // Copy host -> device
    ASSERT_EQ(cudaMemcpy(d_aReal, h_aReal.data(),
                         real_size*sizeof(thrust::complex<T>),
                         cudaMemcpyHostToDevice), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(d_aDual, &h_aDual,
                         sizeof(thrust::complex<T>),
                         cudaMemcpyHostToDevice), cudaSuccess);

    // Launch kernel
    int blockSize = 64;
    int gridSize  = 1;
    VectorDualIndexGetSparseKernel<T><<<gridSize, blockSize>>>(
        d_aReal,  // a_real
        d_aDual,  // a_dual
        h_aIdx,   // a_idx
        real_size,
        start,
        end,
        d_resultReal,
        d_resultDual,
        d_resultIdx
    );
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    // Copy results back to host
    std::vector<thrust::complex<T>> h_resultReal(new_size);
    thrust::complex<T> h_resultDual;
    int h_resultIdx;
    ASSERT_EQ(cudaMemcpy(h_resultReal.data(), d_resultReal,
                         new_size*sizeof(thrust::complex<T>),
                         cudaMemcpyDeviceToHost), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(&h_resultDual, d_resultDual,
                         sizeof(thrust::complex<T>),
                         cudaMemcpyDeviceToHost), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(&h_resultIdx, d_resultIdx,
                         sizeof(int),
                         cudaMemcpyDeviceToHost), cudaSuccess);

    // Free device memory
    cudaFree(d_aReal);
    cudaFree(d_aDual);
    cudaFree(d_resultReal);
    cudaFree(d_resultDual);
    cudaFree(d_resultIdx);

    // -----------------------------
    // Compare with CPU reference
    // -----------------------------
    std::vector<thrust::complex<double>> ref_real;
    thrust::complex<double> ref_dual;
    int ref_idx;
    VectorDualIndexGetSparseCPU(
        h_aReal, h_aDual, h_aIdx, real_size,
        start, end,
        ref_real, ref_dual, ref_idx
    );

    // 1) Compare real part
    ASSERT_EQ(ref_real.size(), h_resultReal.size());
    for (size_t i = 0; i < ref_real.size(); ++i) {
        ExpectComplexNear(ref_real[i], h_resultReal[i]);
    }

    // 2) Compare dual index
    EXPECT_EQ(ref_idx, h_resultIdx);

    // 3) Compare dual value
    ExpectComplexNear(ref_dual, h_resultDual);
}


// --------------------------------------------------------------
// 2) Test: a_idx is outside [start, end)
// --------------------------------------------------------------
TEST(VectorDualIndexGetSparseTest, OutOfRangeCase)
{
    using T = double;

    // Original vector of length 6, for example
    int real_size = 6;
    std::vector<thrust::complex<T>> h_aReal = {
        {1,2}, {3,4}, {5,6}, {7,8}, {9,10}, {11,12}
    };

    // Single dual value and index, but this time the index is out of [2,5)
    thrust::complex<T> h_aDual(10, 20);
    int h_aIdx = 5;  // outside [start=2, end=5)

    // We pick subrange [2,5) => length = 3
    int start = 2;
    int end   = 5;
    int new_size = end - start;

    // Device pointers
    thrust::complex<T>* d_aReal       = nullptr;
    thrust::complex<T>* d_aDual       = nullptr;
    thrust::complex<T>* d_resultReal  = nullptr;
    thrust::complex<T>* d_resultDual  = nullptr;
    int*                d_resultIdx   = nullptr;

    // Allocate
    ASSERT_EQ(cudaMalloc(&d_aReal, real_size*sizeof(thrust::complex<T>)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_aDual, sizeof(thrust::complex<T>)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_resultReal, new_size*sizeof(thrust::complex<T>)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_resultDual, sizeof(thrust::complex<T>)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_resultIdx,  sizeof(int)), cudaSuccess);

    // Copy host -> device
    ASSERT_EQ(cudaMemcpy(d_aReal, h_aReal.data(),
                         real_size*sizeof(thrust::complex<T>),
                         cudaMemcpyHostToDevice), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(d_aDual, &h_aDual,
                         sizeof(thrust::complex<T>),
                         cudaMemcpyHostToDevice), cudaSuccess);

    // Launch kernel
    int blockSize = 64;
    int gridSize  = 1;
    VectorDualIndexGetSparseKernel<T><<<gridSize, blockSize>>>(
        d_aReal,  // a_real
        d_aDual,  // a_dual
        h_aIdx,   // a_idx
        real_size,
        start,
        end,
        d_resultReal,
        d_resultDual,
        d_resultIdx
    );
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    // Copy results back to host
    std::vector<thrust::complex<T>> h_resultReal(new_size);
    thrust::complex<T> h_resultDual;
    int h_resultIdx;
    ASSERT_EQ(cudaMemcpy(h_resultReal.data(), d_resultReal,
                         new_size*sizeof(thrust::complex<T>),
                         cudaMemcpyDeviceToHost), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(&h_resultDual, d_resultDual,
                         sizeof(thrust::complex<T>),
                         cudaMemcpyDeviceToHost), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(&h_resultIdx, d_resultIdx,
                         sizeof(int),
                         cudaMemcpyDeviceToHost), cudaSuccess);

    // Free device memory
    cudaFree(d_aReal);
    cudaFree(d_aDual);
    cudaFree(d_resultReal);
    cudaFree(d_resultDual);
    cudaFree(d_resultIdx);

    // -----------------------------
    // Compare with CPU reference
    // -----------------------------
    std::vector<thrust::complex<double>> ref_real;
    thrust::complex<double> ref_dual;
    int ref_idx;
    VectorDualIndexGetSparseCPU(
        h_aReal, h_aDual, h_aIdx, real_size,
        start, end,
        ref_real, ref_dual, ref_idx
    );

    // 1) Compare real part
    ASSERT_EQ(ref_real.size(), h_resultReal.size());
    for (size_t i = 0; i < ref_real.size(); ++i) {
        ExpectComplexNear(ref_real[i], h_resultReal[i]);
    }

    // 2) Compare dual index
    EXPECT_EQ(ref_idx, h_resultIdx);

    // 3) Compare dual value
    ExpectComplexNear(ref_dual, h_resultDual);
}


/**
 * VectorDualIndexPutSparseCPU
 * 
 * - Copies 'sub_len' real entries from input_real into result_real at [start..end).
 * - If input_idx in [0..sub_len), sets result_idx = start + input_idx and result_dual = input_dual.
 * - Otherwise sets result_idx = -1, result_dual = 0.
 */
template <typename T>
void VectorDualIndexPutSparseCPU(
    const std::vector<thrust::complex<T>>& input_real,
    const thrust::complex<T>&              input_dual,
    int                                    input_idx,
    int                                    sub_len,
    int                                    start,
    int                                    end,

    // Outputs
    std::vector<thrust::complex<T>>&       result_real,
    thrust::complex<T>&                    result_dual,
    int&                                   result_idx,
    int                                    real_size
)
{
    // 1) Copy real part
    for (int i = 0; i < sub_len; ++i) {
        result_real[start + i] = input_real[i];
    }

    // 2) Handle single dual index
    if (input_idx >= 0 && input_idx < sub_len) {
        result_idx  = start + input_idx;
        result_dual = input_dual;
    } else {
        result_idx  = -1;
        result_dual = thrust::complex<T>(0,0);
    }
}


#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <thrust/complex.h>
#include <vector>
#include <iostream>

// Forward-declare or include your device code.
// template <typename T>
// __device__ void VectorDualIndexPutSparse(...);
// template <typename T>
// __global__ void VectorDualIndexPutSparseKernel(...);

template <typename T>
void VectorDualIndexPutSparseCPU(
    const std::vector<thrust::complex<T>>& input_real,
    const thrust::complex<T>&              input_dual,
    int                                    input_idx,
    int                                    sub_len,
    int                                    start,
    int                                    end,

    std::vector<thrust::complex<T>>&       result_real,
    thrust::complex<T>&                    result_dual,
    int&                                   result_idx,
    int                                    real_size
);


TEST(VectorDualIndexPutSparseTest, InRangeCase)
{
    using T = double;

    // We have a "result" array of length real_size = 6
    int real_size = 6;

    // We'll do subrange [start=2, end=5) => length = 3
    int start = 2;
    int end   = 5;
    int sub_len = end - start;  // 3

    // 1) Host data: 'result_real' initially
    //    We'll fill with sentinel values (-1, -1) for clarity
    std::vector<thrust::complex<T>> h_resultReal(real_size, { -1, -1 });
    // The result dual is just a single complex, and an index
    thrust::complex<T> h_resultDual(-99, -99); 
    int h_resultIdx = -99;

    // 2) Host data for 'input'
    //    length sub_len for the real part
    std::vector<thrust::complex<T>> h_inputReal(sub_len);
    for (int i = 0; i < sub_len; ++i) {
        h_inputReal[i] = thrust::complex<T>(10 + i, 20 + i);
    }
    thrust::complex<T> h_inputDual(100, 200);
    // input_idx in [0..sub_len)
    int h_inputIdx = 1;  // in range

    // 3) Device allocations
    thrust::complex<T>* d_inputReal     = nullptr;
    thrust::complex<T>* d_inputDual     = nullptr;
    thrust::complex<T>* d_resultReal    = nullptr;
    thrust::complex<T>* d_resultDual    = nullptr;
    int*                d_resultIdx     = nullptr;

    cudaMalloc(&d_inputReal,  sub_len*sizeof(thrust::complex<T>));
    cudaMalloc(&d_inputDual,  sizeof(thrust::complex<T>));
    cudaMalloc(&d_resultReal, real_size*sizeof(thrust::complex<T>));
    cudaMalloc(&d_resultDual, sizeof(thrust::complex<T>));
    cudaMalloc(&d_resultIdx,  sizeof(int));

    // Copy host => device
    cudaMemcpy(d_inputReal, h_inputReal.data(),
               sub_len*sizeof(thrust::complex<T>),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_inputDual, &h_inputDual,
               sizeof(thrust::complex<T>),
               cudaMemcpyHostToDevice);

    cudaMemcpy(d_resultReal, h_resultReal.data(),
               real_size*sizeof(thrust::complex<T>),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_resultDual, &h_resultDual,
               sizeof(thrust::complex<T>),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_resultIdx, &h_resultIdx,
               sizeof(int),
               cudaMemcpyHostToDevice);

    // 4) Launch kernel
    int blockSize = 64;
    int gridSize  = 1;
    VectorDualIndexPutSparseKernel<T><<<gridSize, blockSize>>>(
        d_inputReal,    // subrange input real
        d_inputDual,    // single dual
        h_inputIdx,     // subrange dual index
        sub_len,
        start,
        end,
        d_resultReal,
        d_resultDual,
        d_resultIdx,
        real_size
    );
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    // 5) Copy back
    std::vector<thrust::complex<T>> h_resultReal_after(real_size);
    thrust::complex<T> h_resultDual_after;
    int h_resultIdx_after;
    cudaMemcpy(h_resultReal_after.data(), d_resultReal,
               real_size*sizeof(thrust::complex<T>),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_resultDual_after, d_resultDual,
               sizeof(thrust::complex<T>),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_resultIdx_after, d_resultIdx,
               sizeof(int),
               cudaMemcpyDeviceToHost);

    cudaFree(d_inputReal);
    cudaFree(d_inputDual);
    cudaFree(d_resultReal);
    cudaFree(d_resultDual);
    cudaFree(d_resultIdx);

    // 6) CPU reference
    std::vector<thrust::complex<T>> ref_resultReal = h_resultReal;
    thrust::complex<T> ref_resultDual = h_resultDual;
    int ref_resultIdx = h_resultIdx;
    VectorDualIndexPutSparseCPU(
        h_inputReal, h_inputDual, h_inputIdx, sub_len, start, end,
        ref_resultReal, ref_resultDual, ref_resultIdx, real_size
    );

    // 7) Compare
    // Real
    for (int i = 0; i < real_size; ++i) {
        ExpectComplexNear(ref_resultReal[i], h_resultReal_after[i]);
    }
    // Dual
    ExpectComplexNear(ref_resultDual, h_resultDual_after);
    EXPECT_EQ(ref_resultIdx, h_resultIdx_after);
}

TEST(VectorDualIndexPutSparseTest, OutOfRangeCase)
{
    using T = double;

    int real_size = 6;
    int start = 2;
    int end   = 5;
    int sub_len = end - start; // 3

    // result array
    std::vector<thrust::complex<T>> h_resultReal(real_size, { -1, -1 });
    thrust::complex<T> h_resultDual(-99, -99); 
    int h_resultIdx = -99;

    // input arrays
    std::vector<thrust::complex<T>> h_inputReal(sub_len);
    for (int i = 0; i < sub_len; ++i) {
        h_inputReal[i] = thrust::complex<T>(10 + i, 20 + i);
    }
    thrust::complex<T> h_inputDual(100, 200);
    // This time the index is out of range, e.g. 5 is not in [0..3)
    int h_inputIdx = 5;

    // Device allocations
    thrust::complex<T>* d_inputReal     = nullptr;
    thrust::complex<T>* d_inputDual     = nullptr;
    thrust::complex<T>* d_resultReal    = nullptr;
    thrust::complex<T>* d_resultDual    = nullptr;
    int*                d_resultIdx     = nullptr;

    cudaMalloc(&d_inputReal,  sub_len*sizeof(thrust::complex<T>));
    cudaMalloc(&d_inputDual,  sizeof(thrust::complex<T>));
    cudaMalloc(&d_resultReal, real_size*sizeof(thrust::complex<T>));
    cudaMalloc(&d_resultDual, sizeof(thrust::complex<T>));
    cudaMalloc(&d_resultIdx,  sizeof(int));

    cudaMemcpy(d_inputReal, h_inputReal.data(),
               sub_len*sizeof(thrust::complex<T>),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_inputDual, &h_inputDual,
               sizeof(thrust::complex<T>),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_resultReal, h_resultReal.data(),
               real_size*sizeof(thrust::complex<T>),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_resultDual, &h_resultDual,
               sizeof(thrust::complex<T>),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_resultIdx, &h_resultIdx,
               sizeof(int),
               cudaMemcpyHostToDevice);

    // Launch kernel
    VectorDualIndexPutSparseKernel<T><<<1, 64>>>(
        d_inputReal, d_inputDual,
        h_inputIdx,
        sub_len,
        start, end,
        d_resultReal,
        d_resultDual,
        d_resultIdx,
        real_size
    );
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    // Copy back
    std::vector<thrust::complex<T>> h_resultReal_after(real_size);
    thrust::complex<T> h_resultDual_after;
    int h_resultIdx_after;
    cudaMemcpy(h_resultReal_after.data(), d_resultReal,
               real_size*sizeof(thrust::complex<T>),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_resultDual_after, d_resultDual,
               sizeof(thrust::complex<T>),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_resultIdx_after, d_resultIdx,
               sizeof(int),
               cudaMemcpyDeviceToHost);

    cudaFree(d_inputReal);
    cudaFree(d_inputDual);
    cudaFree(d_resultReal);
    cudaFree(d_resultDual);
    cudaFree(d_resultIdx);

    // CPU reference
    std::vector<thrust::complex<T>> ref_resultReal = h_resultReal;
    thrust::complex<T> ref_resultDual = h_resultDual;
    int ref_resultIdx = h_resultIdx;
    VectorDualIndexPutSparseCPU(
        h_inputReal, h_inputDual, h_inputIdx, sub_len, start, end,
        ref_resultReal, ref_resultDual, ref_resultIdx, real_size
    );

    // Compare
    // Real part (should be copied, even if the dual index is out of range)
    for (int i = 0; i < real_size; ++i) {
        ExpectComplexNear(ref_resultReal[i], h_resultReal_after[i]);
    }
    // Dual part should be -1 for index and (0,0) for dual
    ExpectComplexNear(ref_resultDual, h_resultDual_after);
    EXPECT_EQ(ref_resultIdx, h_resultIdx_after);
}




// Add more tests for IndexGet, IndexPut, ElementwiseMultiply, Square, Pow, and Sqrt similarly.
// Main entry point for Google Test
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}