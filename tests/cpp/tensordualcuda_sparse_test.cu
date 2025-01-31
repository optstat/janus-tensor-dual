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
    EXPECT_DOUBLE_EQ(expected.real(), actual.real());
    EXPECT_DOUBLE_EQ(expected.imag(), actual.imag());
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


/**
 * VectorRealDualProductSparseCPU
 *
 * A CPU reference for multiplying two sparse dual vectors of length real_size.
 * 
 * Input sparse vector a:
 *   - a_real: length real_size
 *   - a_idx:  single index in [0..real_size) or -1
 *   - a_dual: single complex
 *
 * Input sparse vector b:
 *   - b_real: length real_size
 *   - b_idx:  single index in [0..real_size) or -1
 *   - b_dual: single complex
 *
 * Output sparse vector r:
 *   - r_real: length real_size
 *   - r_idx
 *   - r_dual
 */
template <typename T>
void VectorRealDualProductSparseCPU(
    // a
    const std::vector<thrust::complex<T>>& a_real,
    int a_idx,
    const thrust::complex<T>& a_dual,

    // b
    const std::vector<thrust::complex<T>>& b_real,
    int b_idx,
    const thrust::complex<T>& b_dual,

    // size
    int real_size,

    // outputs
    std::vector<thrust::complex<T>>& r_real,
    int& r_idx,
    thrust::complex<T>& r_dual
)
{
    // 1) Multiply real parts elementwise
    r_real.resize(real_size);
    for (int i = 0; i < real_size; ++i) {
        r_real[i] = a_real[i] * b_real[i];
    }

    // 2) Dual index logic
    // CASE A: both indices == -1
    if (a_idx < 0 && b_idx < 0) {
        r_idx = -1;
        r_dual = thrust::complex<T>(0,0);
        return;
    }

    // CASE B: only A has dual
    if (a_idx >= 0 && b_idx < 0) {
        r_idx = a_idx;
        // sum_{i} [ b_real[i]*a_dual ]
        thrust::complex<T> sumVal(0,0);
        for (int i = 0; i < real_size; ++i) {
            sumVal += b_real[i] * a_dual;
        }
        r_dual = sumVal;
        return;
    }

    // CASE C: only B has dual
    if (a_idx < 0 && b_idx >= 0) {
        r_idx = b_idx;
        // sum_{i} [ a_real[i]*b_dual ]
        thrust::complex<T> sumVal(0,0);
        for (int i = 0; i < real_size; ++i) {
            sumVal += a_real[i] * b_dual;
        }
        r_dual = sumVal;
        return;
    }

    // CASE D: Both valid
    if (a_idx == b_idx) {
        // Indices match
        r_idx = a_idx; 
        // sum_{i} [ a_real[i]*b_dual + b_real[i]*a_dual ]
        thrust::complex<T> sumVal(0,0);
        for (int i = 0; i < real_size; ++i) {
            sumVal += a_real[i]*b_dual + b_real[i]*a_dual;
        }
        r_dual = sumVal;
    }
    else {
        // Indices differ => set -1
        r_idx = -1;
        r_dual = thrust::complex<T>(0,0);
    }
}




// --------------------------------------------------
// Helper to run the GPU kernel
// --------------------------------------------------
template <typename T>
void RunVectorRealDualProductSparseKernel(
    // input a
    const std::vector<thrust::complex<T>>& h_aReal,
    int a_idx,
    const thrust::complex<T>& h_aDualValue,

    // input b
    const std::vector<thrust::complex<T>>& h_bReal,
    int b_idx,
    const thrust::complex<T>& h_bDualValue,

    int real_size,

    // outputs
    std::vector<thrust::complex<T>>& h_rRealAfter,
    int& h_rIdxAfter,
    thrust::complex<T>& h_rDualValAfter
)
{
    // Allocate device
    thrust::complex<T>* d_aReal = nullptr;
    thrust::complex<T>* d_aDual = nullptr;
    thrust::complex<T>* d_bReal = nullptr;
    thrust::complex<T>* d_bDual = nullptr;
    thrust::complex<T>* d_rReal = nullptr;
    thrust::complex<T>* d_rDual = nullptr;
    int* d_rIdx = nullptr;

    size_t bytesReal = real_size * sizeof(thrust::complex<T>);

    cudaMalloc(&d_aReal, bytesReal);
    cudaMalloc(&d_bReal, bytesReal);
    cudaMalloc(&d_aDual, sizeof(thrust::complex<T>));
    cudaMalloc(&d_bDual, sizeof(thrust::complex<T>));
    cudaMalloc(&d_rReal, bytesReal);
    cudaMalloc(&d_rDual, sizeof(thrust::complex<T>));
    cudaMalloc(&d_rIdx,  sizeof(int));

    // Copy host->device
    cudaMemcpy(d_aReal, h_aReal.data(), bytesReal, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bReal, h_bReal.data(), bytesReal, cudaMemcpyHostToDevice);
    cudaMemcpy(d_aDual, &h_aDualValue, sizeof(thrust::complex<T>), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bDual, &h_bDualValue, sizeof(thrust::complex<T>), cudaMemcpyHostToDevice);

    // We'll init result arrays to some sentinel
    std::vector<thrust::complex<T>> h_rRealInit(real_size, thrust::complex<T>(-999,-999));
    thrust::complex<T> h_rDualInit(-888,-888);
    int h_rIdxInit = -777;
    cudaMemcpy(d_rReal, h_rRealInit.data(), bytesReal, cudaMemcpyHostToDevice);
    cudaMemcpy(d_rDual, &h_rDualInit, sizeof(thrust::complex<T>), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rIdx,  &h_rIdxInit,  sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 block(64);
    dim3 grid( (real_size + block.x -1)/block.x );
    VectorRealDualProductSparseKernel<T><<<grid, block>>>(
        d_aReal, a_idx, d_aDual,
        d_bReal, b_idx, d_bDual,
        real_size,
        d_rReal, d_rIdx, d_rDual
    );
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    // Copy device->host
    h_rRealAfter.resize(real_size);
    cudaMemcpy(h_rRealAfter.data(), d_rReal, bytesReal, cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_rDualValAfter, d_rDual, sizeof(thrust::complex<T>), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_rIdxAfter, d_rIdx, sizeof(int), cudaMemcpyDeviceToHost);

    // Free device
    cudaFree(d_aReal);
    cudaFree(d_aDual);
    cudaFree(d_bReal);
    cudaFree(d_bDual);
    cudaFree(d_rReal);
    cudaFree(d_rDual);
    cudaFree(d_rIdx);
}

// --------------------------------------------------
// Test: Both Indices = -1 (no dual on either vector)
// --------------------------------------------------
TEST(VectorRealDualProductSparseTest, BothIndicesMinusOne)
{
    using T = double;
    int real_size = 5;

    // a, b real arrays
    std::vector<thrust::complex<T>> h_aReal {
        {1,1}, {2,2}, {3,3}, {4,4}, {5,5}
    };
    std::vector<thrust::complex<T>> h_bReal {
        {10,0}, {9,0}, {8,0}, {7,0}, {6,0}
    };

    // dual indices
    int a_idx = -1;
    int b_idx = -1;

    // dual values (ignored since idx=-1)
    thrust::complex<T> h_aDualVal(100,200);
    thrust::complex<T> h_bDualVal(300,400);

    // GPU results
    std::vector<thrust::complex<T>> h_rRealAfter;
    int h_rIdxAfter;
    thrust::complex<T> h_rDualValAfter;

    RunVectorRealDualProductSparseKernel<T>(
        h_aReal, a_idx, h_aDualVal,
        h_bReal, b_idx, h_bDualVal,
        real_size,
        h_rRealAfter, h_rIdxAfter, h_rDualValAfter
    );

    // CPU reference
    std::vector<thrust::complex<T>> ref_rReal;
    ref_rReal.resize(real_size);
    int ref_rIdx;
    thrust::complex<T> ref_rDual;
    VectorRealDualProductSparseCPU(
        h_aReal, a_idx, h_aDualVal,
        h_bReal, b_idx, h_bDualVal,
        real_size,
        ref_rReal, ref_rIdx, ref_rDual
    );

    // Compare real
    ASSERT_EQ(real_size, (int)h_rRealAfter.size());
    for (int i = 0; i < real_size; ++i) {
        ExpectComplexNear(ref_rReal[i], h_rRealAfter[i]);
    }
    // Compare dual
    EXPECT_EQ(ref_rIdx, h_rIdxAfter);
    ExpectComplexNear(ref_rDual, h_rDualValAfter);
}

// --------------------------------------------------
// Test: Only A has Dual
// --------------------------------------------------
TEST(VectorRealDualProductSparseTest, OnlyAHasDual)
{
    using T = double;
    int real_size = 5;

    // a, b real arrays
    std::vector<thrust::complex<T>> h_aReal {
        {1,1}, {2,2}, {3,3}, {4,4}, {5,5}
    };
    std::vector<thrust::complex<T>> h_bReal {
        {10,0}, {9,0}, {8,0}, {7,0}, {6,0}
    };

    // a has dual at idx=2, b_idx=-1
    int a_idx = 2;
    thrust::complex<T> h_aDualVal(1,2);
    int b_idx = -1;
    thrust::complex<T> h_bDualVal(3,4);

    // GPU
    std::vector<thrust::complex<T>> h_rRealAfter;
    int h_rIdxAfter;
    thrust::complex<T> h_rDualValAfter;

    RunVectorRealDualProductSparseKernel(
        h_aReal, a_idx, h_aDualVal,
        h_bReal, b_idx, h_bDualVal,
        real_size,
        h_rRealAfter, h_rIdxAfter, h_rDualValAfter
    );

    // CPU
    std::vector<thrust::complex<T>> ref_rReal;
    int ref_rIdx;
    thrust::complex<T> ref_rDual;
    VectorRealDualProductSparseCPU(
        h_aReal, a_idx, h_aDualVal,
        h_bReal, b_idx, h_bDualVal,
        real_size,
        ref_rReal, ref_rIdx, ref_rDual
    );

    // Compare
    for (int i = 0; i < real_size; ++i) {
        ExpectComplexNear(ref_rReal[i], h_rRealAfter[i]);
    }
    EXPECT_EQ(ref_rIdx, h_rIdxAfter);
    ExpectComplexNear(ref_rDual, h_rDualValAfter);
}

// --------------------------------------------------
// Test: Only B has Dual
// --------------------------------------------------
TEST(VectorRealDualProductSparseTest, OnlyBHasDual)
{
    using T = double;
    int real_size = 5;

    // a, b real arrays
    std::vector<thrust::complex<T>> h_aReal {
        {1,1}, {2,2}, {3,3}, {4,4}, {5,5}
    };
    std::vector<thrust::complex<T>> h_bReal {
        {10,0}, {9,0}, {8,0}, {7,0}, {6,0}
    };

    // b has dual at idx=1, a_idx=-1
    int a_idx = -1;
    thrust::complex<T> h_aDualVal(1,2);
    int b_idx = 1;
    thrust::complex<T> h_bDualVal(3,4);

    // GPU
    std::vector<thrust::complex<T>> h_rRealAfter;
    int h_rIdxAfter;
    thrust::complex<T> h_rDualValAfter;

    RunVectorRealDualProductSparseKernel(
        h_aReal, a_idx, h_aDualVal,
        h_bReal, b_idx, h_bDualVal,
        real_size,
        h_rRealAfter, h_rIdxAfter, h_rDualValAfter
    );

    // CPU
    std::vector<thrust::complex<T>> ref_rReal;
    int ref_rIdx;
    thrust::complex<T> ref_rDual;
    VectorRealDualProductSparseCPU(
        h_aReal, a_idx, h_aDualVal,
        h_bReal, b_idx, h_bDualVal,
        real_size,
        ref_rReal, ref_rIdx, ref_rDual
    );

    // Compare
    for (int i = 0; i < real_size; ++i) {
        ExpectComplexNear(ref_rReal[i], h_rRealAfter[i]);
    }
    EXPECT_EQ(ref_rIdx, h_rIdxAfter);
    ExpectComplexNear(ref_rDual, h_rDualValAfter);
}

// --------------------------------------------------
// Test: Both same index
// --------------------------------------------------
TEST(VectorRealDualProductSparseTest, BothSameIndex)
{
    using T = double;
    int real_size = 5;

    // a, b real arrays
    std::vector<thrust::complex<T>> h_aReal {
        {1,1}, {2,2}, {3,3}, {4,4}, {5,5}
    };
    std::vector<thrust::complex<T>> h_bReal {
        {10,0}, {9,0}, {8,0}, {7,0}, {6,0}
    };

    // a_idx = b_idx = 2
    int a_idx = 2;
    thrust::complex<T> h_aDualVal(1,2);
    int b_idx = 2;
    thrust::complex<T> h_bDualVal(3,4);

    // GPU
    std::vector<thrust::complex<T>> h_rRealAfter;
    int h_rIdxAfter;
    thrust::complex<T> h_rDualValAfter;

    RunVectorRealDualProductSparseKernel(
        h_aReal, a_idx, h_aDualVal,
        h_bReal, b_idx, h_bDualVal,
        real_size,
        h_rRealAfter, h_rIdxAfter, h_rDualValAfter
    );

    // CPU
    std::vector<thrust::complex<T>> ref_rReal;
    int ref_rIdx;
    thrust::complex<T> ref_rDual;
    VectorRealDualProductSparseCPU(
        h_aReal, a_idx, h_aDualVal,
        h_bReal, b_idx, h_bDualVal,
        real_size,
        ref_rReal, ref_rIdx, ref_rDual
    );

    // Compare
    for (int i = 0; i < real_size; ++i) {
        ExpectComplexNear(ref_rReal[i], h_rRealAfter[i]);
    }
    EXPECT_EQ(ref_rIdx, h_rIdxAfter);
    ExpectComplexNear(ref_rDual, h_rDualValAfter);
}

// --------------------------------------------------
// Test: Both different index
// --------------------------------------------------
TEST(VectorRealDualProductSparseTest, DifferentIndex)
{
    using T = double;
    int real_size = 5;

    // a, b real arrays
    std::vector<thrust::complex<T>> h_aReal {
        {1,1}, {2,2}, {3,3}, {4,4}, {5,5}
    };
    std::vector<thrust::complex<T>> h_bReal {
        {10,0}, {9,0}, {8,0}, {7,0}, {6,0}
    };

    // a_idx=1, b_idx=4 => different
    int a_idx = 1;
    thrust::complex<T> h_aDualVal(1,2);
    int b_idx = 4;
    thrust::complex<T> h_bDualVal(3,4);

    // GPU
    std::vector<thrust::complex<T>> h_rRealAfter;
    int h_rIdxAfter;
    thrust::complex<T> h_rDualValAfter;

    RunVectorRealDualProductSparseKernel(
        h_aReal, a_idx, h_aDualVal,
        h_bReal, b_idx, h_bDualVal,
        real_size,
        h_rRealAfter, h_rIdxAfter, h_rDualValAfter
    );

    // CPU
    std::vector<thrust::complex<T>> ref_rReal;
    int ref_rIdx;
    thrust::complex<T> ref_rDual;
    VectorRealDualProductSparseCPU(
        h_aReal, a_idx, h_aDualVal,
        h_bReal, b_idx, h_bDualVal,
        real_size,
        ref_rReal, ref_rIdx, ref_rDual
    );

    // Compare
    for (int i = 0; i < real_size; ++i) {
        ExpectComplexNear(ref_rReal[i], h_rRealAfter[i]);
    }
    EXPECT_EQ(ref_rIdx, h_rIdxAfter);
    ExpectComplexNear(ref_rDual, h_rDualValAfter);
}


/**
 * VectorDualIndexGetSparseCPU
 *
 * A CPU reference for extracting a subrange [start, end) from a sparse dual vector.
 *
 * Inputs:
 *  - a_real:      length real_size
 *  - a_idx:       index in [0..real_size) or -1
 *  - a_dual_value single complex for the dual part
 *  - real_size
 *  - start, end
 *
 * Outputs:
 *  - result_real: length = end - start
 *  - result_idx
 *  - result_dual_value
 */
template <typename T>
void VectorDualIndexGetSparseCPU(
    const std::vector<thrust::complex<T>>& a_real,
    int                                    a_idx,
    const thrust::complex<T>&              a_dual_value,
    int                                    real_size,
    int                                    start,
    int                                    end,

    std::vector<thrust::complex<T>>&       result_real,
    int&                                   result_idx,
    thrust::complex<T>&                    result_dual_value
)
{
    int sub_len = end - start;
    result_real.resize(sub_len);

    // Copy real part
    for (int i = 0; i < sub_len; ++i) {
        result_real[i] = a_real[start + i];
    }

    // Decide dual index/value
    if (a_idx >= start && a_idx < end) {
        // in range
        result_idx        = a_idx - start;
        result_dual_value = a_dual_value;
    } else {
        // out of range or a_idx == -1
        result_idx        = -1;
        result_dual_value = thrust::complex<T>(0,0);
    }
}

// Helper to run the kernel
template <typename T>
void RunVectorDualIndexGetSparseKernel(
    // input sparse vector
    const std::vector<thrust::complex<T>>& h_aReal,
    int                                    h_aIdx,
    const thrust::complex<T>&              h_aDualValue,
    int                                    real_size,
    int                                    start,
    int                                    end,

    // outputs
    std::vector<thrust::complex<T>>&       h_resultReal,
    int&                                   h_resultIdx,
    thrust::complex<T>&                    h_resultDualVal
)
{
    // subrange size
    int sub_len = end - start;

    // Allocate device
    thrust::complex<T>* d_aReal = nullptr;
    thrust::complex<T>* d_aDualVal = nullptr;
    thrust::complex<T>* d_resultReal = nullptr;
    thrust::complex<T>* d_resultDualVal = nullptr;
    int* d_resultIdx = nullptr;

    cudaMalloc(&d_aReal, real_size*sizeof(thrust::complex<T>));
    cudaMalloc(&d_aDualVal, sizeof(thrust::complex<T>));
    cudaMalloc(&d_resultReal, sub_len*sizeof(thrust::complex<T>));
    cudaMalloc(&d_resultDualVal, sizeof(thrust::complex<T>));
    cudaMalloc(&d_resultIdx, sizeof(int));

    // Copy host->device
    cudaMemcpy(d_aReal, h_aReal.data(),
               real_size*sizeof(thrust::complex<T>),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_aDualVal, &h_aDualValue,
               sizeof(thrust::complex<T>),
               cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 block(64);
    dim3 grid( (sub_len + block.x - 1)/block.x );
    VectorDualIndexGetSparseKernel<T><<<grid, block>>>(
        d_aReal,
        h_aIdx,
        d_aDualVal,
        real_size,
        start, end,
        d_resultReal,
        d_resultIdx,
        d_resultDualVal
    );
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    // Copy back
    h_resultReal.resize(sub_len);
    cudaMemcpy(h_resultReal.data(), d_resultReal,
               sub_len*sizeof(thrust::complex<T>),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_resultDualVal, d_resultDualVal,
               sizeof(thrust::complex<T>),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_resultIdx, d_resultIdx,
               sizeof(int),
               cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_aReal);
    cudaFree(d_aDualVal);
    cudaFree(d_resultReal);
    cudaFree(d_resultDualVal);
    cudaFree(d_resultIdx);
}


// -------------------------------------------------
// 1) a_idx = -1
// -------------------------------------------------
TEST(VectorDualIndexGetSparseTest, IdxIsMinusOne)
{
    using T = double;
    int real_size = 5;
    std::vector<thrust::complex<T>> h_aReal = {
        {1,1}, {2,2}, {3,3}, {4,4}, {5,5}
    };

    int a_idx = -1;
    thrust::complex<T> a_dualValue(10,20);

    // subrange [1..4)
    int start = 1, end = 4; // length=3

    // GPU result
    std::vector<thrust::complex<T>> h_resultReal;
    thrust::complex<T> h_resultDual;
    int h_resultIdx;

    RunVectorDualIndexGetSparseKernel(
        h_aReal, a_idx, a_dualValue,
        real_size,
        start, end,
        h_resultReal, h_resultIdx, h_resultDual
    );

    // CPU reference
    std::vector<thrust::complex<T>> refReal;
    thrust::complex<T> refDual;
    int refIdx;
    VectorDualIndexGetSparseCPU(
        h_aReal, a_idx, a_dualValue,
        real_size,
        start, end,
        refReal, refIdx, refDual
    );

    // Compare real
    ASSERT_EQ((int)refReal.size(), (int)h_resultReal.size());
    for (int i = 0; i < (int)refReal.size(); ++i) {
        ExpectComplexNear(refReal[i], h_resultReal[i]);
    }
    // Compare dual
    EXPECT_EQ(refIdx, h_resultIdx);
    ExpectComplexNear(refDual, h_resultDual);
}

// -------------------------------------------------
// 2) a_idx < start
// -------------------------------------------------
TEST(VectorDualIndexGetSparseTest, IdxLessThanStart)
{
    using T = double;
    int real_size = 5;
    std::vector<thrust::complex<T>> h_aReal = {
        {1,1}, {2,2}, {3,3}, {4,4}, {5,5}
    };

    // Suppose a_idx=0 but subrange is [2..5)
    int a_idx = 0;
    thrust::complex<T> a_dualValue(10,20);

    int start = 2, end = 5; // length=3

    // GPU result
    std::vector<thrust::complex<T>> h_resultReal;
    thrust::complex<T> h_resultDual;
    int h_resultIdx;

    RunVectorDualIndexGetSparseKernel(
        h_aReal, a_idx, a_dualValue,
        real_size,
        start, end,
        h_resultReal, h_resultIdx, h_resultDual
    );

    // CPU reference
    std::vector<thrust::complex<T>> refReal;
    thrust::complex<T> refDual;
    int refIdx;
    VectorDualIndexGetSparseCPU(
        h_aReal, a_idx, a_dualValue,
        real_size,
        start, end,
        refReal, refIdx, refDual
    );

    // Compare
    ASSERT_EQ((int)refReal.size(), (int)h_resultReal.size());
    for (int i = 0; i < (int)refReal.size(); ++i) {
        ExpectComplexNear(refReal[i], h_resultReal[i]);
    }
    EXPECT_EQ(refIdx, h_resultIdx);
    ExpectComplexNear(refDual, h_resultDual);
}

// -------------------------------------------------
// 3) a_idx >= end
// -------------------------------------------------
TEST(VectorDualIndexGetSparseTest, IdxGreaterOrEqualEnd)
{
    using T = double;
    int real_size = 5;
    std::vector<thrust::complex<T>> h_aReal = {
        {1,1}, {2,2}, {3,3}, {4,4}, {5,5}
    };

    // Suppose subrange [0..3), length=3 but a_idx=4 => out of range
    int a_idx = 4;
    thrust::complex<T> a_dualValue(10,20);

    int start = 0, end = 3;

    // GPU result
    std::vector<thrust::complex<T>> h_resultReal;
    thrust::complex<T> h_resultDual;
    int h_resultIdx;

    RunVectorDualIndexGetSparseKernel(
        h_aReal, a_idx, a_dualValue,
        real_size,
        start, end,
        h_resultReal, h_resultIdx, h_resultDual
    );

    // CPU reference
    std::vector<thrust::complex<T>> refReal;
    thrust::complex<T> refDual;
    int refIdx;
    VectorDualIndexGetSparseCPU(
        h_aReal, a_idx, a_dualValue,
        real_size,
        start, end,
        refReal, refIdx, refDual
    );

    // Compare
    ASSERT_EQ((int)refReal.size(), (int)h_resultReal.size());
    for (int i = 0; i < (int)refReal.size(); ++i) {
        ExpectComplexNear(refReal[i], h_resultReal[i]);
    }
    EXPECT_EQ(refIdx, h_resultIdx);
    ExpectComplexNear(refDual, h_resultDual);
}

// -------------------------------------------------
// 4) a_idx in [start, end)
// -------------------------------------------------
TEST(VectorDualIndexGetSparseTest, IdxInRange)
{
    using T = double;
    int real_size = 5;
    std::vector<thrust::complex<T>> h_aReal = {
        {1,1}, {2,2}, {3,3}, {4,4}, {5,5}
    };

    // subrange [2..5), length=3 => if a_idx=3 => in range
    int a_idx = 3;
    thrust::complex<T> a_dualValue(10,20);

    int start = 2, end = 5;

    // GPU result
    std::vector<thrust::complex<T>> h_resultReal;
    thrust::complex<T> h_resultDual;
    int h_resultIdx;

    RunVectorDualIndexGetSparseKernel(
        h_aReal, a_idx, a_dualValue,
        real_size,
        start, end,
        h_resultReal, h_resultIdx, h_resultDual
    );

    // CPU reference
    std::vector<thrust::complex<T>> refReal;
    thrust::complex<T> refDual;
    int refIdx;
    VectorDualIndexGetSparseCPU(
        h_aReal, a_idx, a_dualValue,
        real_size,
        start, end,
        refReal, refIdx, refDual
    );

    // Compare
    ASSERT_EQ((int)refReal.size(), (int)h_resultReal.size());
    for (int i = 0; i < (int)refReal.size(); ++i) {
        ExpectComplexNear(refReal[i], h_resultReal[i]);
    }
    EXPECT_EQ(refIdx, h_resultIdx);
    ExpectComplexNear(refDual, h_resultDual);
}

/**
 * VectorDualIndexPutSparseCPU
 *
 * CPU reference for "index put" on a sparse dual vector.
 *
 * - input_real: length sub_len = end - start
 * - input_idx:  single index in [0..sub_len), or -1 if no dual
 * - input_dual_value
 * - sub_len = end - start
 *
 * We copy input_real[i] => result_real[start + i].
 * If input_idx in [0..sub_len), then result_idx = start + input_idx
 * and result_dual_value = input_dual_value, else result_idx=-1, result_dual_value=0.
 */
template <typename T>
void VectorDualIndexPutSparseCPU(
    const std::vector<thrust::complex<T>>& input_real,
    int                                    input_idx,
    const thrust::complex<T>&              input_dual_value,
    int                                    sub_len,
    int                                    start,
    int                                    end,

    // Outputs
    std::vector<thrust::complex<T>>&       result_real,       // length real_size
    int&                                   result_idx,
    thrust::complex<T>&                    result_dual_value,
    int                                    real_size
)
{
    // 1) Copy real part
    for (int i = 0; i < sub_len; ++i) {
        result_real[start + i] = input_real[i];
    }

    // 2) Single dual
    if (input_idx >= 0 && input_idx < sub_len) {
        result_idx        = start + input_idx;
        result_dual_value = input_dual_value;
    } else {
        result_idx        = -1;
        result_dual_value = thrust::complex<T>(0, 0);
    }
}


// Helper to run the kernel and copy back results
template <typename T>
void RunVectorDualIndexPutSparseKernel(
    // input subrange
    const std::vector<thrust::complex<T>>& input_real,  // length sub_len
    int                       input_idx,
    const thrust::complex<T>& input_dual_value,
    int                       sub_len,
    int                       start,
    int                       end,

    // result arrays
    std::vector<thrust::complex<T>>& result_real,  // length real_size
    int&      result_idx,
    thrust::complex<T>& result_dual_value,
    int       real_size
)
{
    // Device allocations
    thrust::complex<T>* d_inputReal     = nullptr;
    thrust::complex<T>* d_inputDualVal  = nullptr;
    thrust::complex<T>* d_resultReal    = nullptr;
    thrust::complex<T>* d_resultDualVal = nullptr;
    int*                d_resultIdx     = nullptr;

    // Host => device
    size_t subLenBytes = sub_len * sizeof(thrust::complex<T>);
    cudaMalloc(&d_inputReal, subLenBytes);
    cudaMalloc(&d_inputDualVal, sizeof(thrust::complex<T>));
    cudaMalloc(&d_resultReal, real_size * sizeof(thrust::complex<T>));
    cudaMalloc(&d_resultDualVal, sizeof(thrust::complex<T>));
    cudaMalloc(&d_resultIdx, sizeof(int));

    cudaMemcpy(d_inputReal, input_real.data(), 
               subLenBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_inputDualVal, &input_dual_value,
               sizeof(thrust::complex<T>),
               cudaMemcpyHostToDevice);

    // Also copy 'result_real' to device so we can see changes after kernel
    cudaMemcpy(d_resultReal, result_real.data(),
               real_size*sizeof(thrust::complex<T>),
               cudaMemcpyHostToDevice);

    // For dual, store a sentinel in result
    thrust::complex<T> sentinelDual(-999, -999);
    cudaMemcpy(d_resultDualVal, &sentinelDual,
               sizeof(thrust::complex<T>),
               cudaMemcpyHostToDevice);

    int sentinelIdx = -777;
    cudaMemcpy(d_resultIdx, &sentinelIdx, sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 block(64);
    dim3 grid( (sub_len + block.x - 1)/block.x );
    VectorDualIndexPutSparseKernel<T><<<grid, block>>>(
        d_inputReal, input_idx, d_inputDualVal,
        sub_len,
        start, end,
        d_resultReal, d_resultIdx, d_resultDualVal,
        real_size
    );
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    // Copy results back
    cudaMemcpy(result_real.data(), d_resultReal,
               real_size*sizeof(thrust::complex<T>),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(&result_dual_value, d_resultDualVal,
               sizeof(thrust::complex<T>), cudaMemcpyDeviceToHost);
    cudaMemcpy(&result_idx, d_resultIdx, sizeof(int), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_inputReal);
    cudaFree(d_inputDualVal);
    cudaFree(d_resultReal);
    cudaFree(d_resultDualVal);
    cudaFree(d_resultIdx);
}


// -----------------------------------------------------
// 1) Index = -1 => no dual
// -----------------------------------------------------
TEST(VectorDualIndexPutSparseTest, IndexMinusOne)
{
    using T = double;

    // result is length 6, init with sentinel
    int real_size = 6;
    std::vector<thrust::complex<T>> h_resultReal {
        { -1,-1}, { -2,-2}, { -3,-3},
        { -4,-4}, { -5,-5}, { -6,-6}
    };

    // subrange [start=1, end=4) => sub_len=3
    int start = 1, end = 4;
    int sub_len = end - start;
    
    // input subrange
    std::vector<thrust::complex<T>> h_inputReal {
        {10,20}, {11,21}, {12,22}
    };
    int input_idx = -1;
    thrust::complex<T> h_inputDualVal(100,200);

    // GPU result
    int h_rIdxAfter;
    thrust::complex<T> h_rDualValAfter;
    RunVectorDualIndexPutSparseKernel(
        h_inputReal, input_idx, h_inputDualVal, sub_len,
        start, end,
        h_resultReal, h_rIdxAfter, h_rDualValAfter,
        real_size
    );

    // CPU reference
    std::vector<thrust::complex<T>> ref_resultReal = {
        { -1,-1}, { -2,-2}, { -3,-3},
        { -4,-4}, { -5,-5}, { -6,-6}
    };
    int ref_idx;
    thrust::complex<T> ref_dual;
    VectorDualIndexPutSparseCPU(
        h_inputReal, input_idx, h_inputDualVal,
        sub_len,
        start, end,
        ref_resultReal, ref_idx, ref_dual,
        real_size
    );

    // Compare real
    for (int i = 0; i < real_size; ++i) {
        ExpectComplexNear(ref_resultReal[i], h_resultReal[i]);
    }
    // Compare dual
    EXPECT_EQ(ref_idx, h_rIdxAfter);
    ExpectComplexNear(ref_dual, h_rDualValAfter);
}

// -----------------------------------------------------
// 2) Index < 0 (less than -1) => also no dual
// -----------------------------------------------------
TEST(VectorDualIndexPutSparseTest, NegativeIndex)
{
    using T = double;

    int real_size = 6;
    std::vector<thrust::complex<T>> h_resultReal {
        {1,1}, {2,2}, {3,3},
        {4,4}, {5,5}, {6,6}
    };

    int start = 2, end = 5;
    int sub_len = end - start;

    // input subrange
    std::vector<thrust::complex<T>> h_inputReal {
        {10,20}, {11,21}, {12,22}
    };
    // e.g. input_idx = -5
    int input_idx = -5;
    thrust::complex<T> h_inputDualVal(100,200);

    // GPU
    int h_rIdxAfter;
    thrust::complex<T> h_rDualValAfter;
    RunVectorDualIndexPutSparseKernel(
        h_inputReal, input_idx, h_inputDualVal, sub_len,
        start, end,
        h_resultReal, h_rIdxAfter, h_rDualValAfter,
        real_size
    );

    // CPU
    std::vector<thrust::complex<T>> ref_resultReal {
        {1,1}, {2,2}, {3,3},
        {4,4}, {5,5}, {6,6}
    };
    int ref_idx;
    thrust::complex<T> ref_dual;
    VectorDualIndexPutSparseCPU(
        h_inputReal, input_idx, h_inputDualVal,
        sub_len,
        start, end,
        ref_resultReal, ref_idx, ref_dual,
        real_size
    );

    // Compare
    for (int i = 0; i < real_size; ++i) {
        ExpectComplexNear(ref_resultReal[i], h_resultReal[i]);
    }
    EXPECT_EQ(ref_idx, h_rIdxAfter);
    ExpectComplexNear(ref_dual, h_rDualValAfter);
}

// -----------------------------------------------------
// 3) Index >= sub_len => out of range => no dual
// -----------------------------------------------------
TEST(VectorDualIndexPutSparseTest, IndexOutOfRange)
{
    using T = double;

    int real_size = 6;
    std::vector<thrust::complex<T>> h_resultReal {
        {10,10}, {20,20}, {30,30},
        {40,40}, {50,50}, {60,60}
    };

    int start = 1, end = 5;
    int sub_len = end - start; // =4

    // input subrange
    std::vector<thrust::complex<T>> h_inputReal {
        {100,200}, {101,201}, {102,202}, {103,203}
    };
    // input_idx = 5 => out of [0..4)
    int input_idx = 5; 
    thrust::complex<T> h_inputDualVal(999, 999);

    // GPU
    int h_rIdxAfter;
    thrust::complex<T> h_rDualValAfter;
    RunVectorDualIndexPutSparseKernel(
        h_inputReal, input_idx, h_inputDualVal, sub_len,
        start, end,
        h_resultReal, h_rIdxAfter, h_rDualValAfter,
        real_size
    );

    // CPU
    std::vector<thrust::complex<T>> ref_resultReal {
        {10,10}, {20,20}, {30,30},
        {40,40}, {50,50}, {60,60}
    };
    int ref_idx;
    thrust::complex<T> ref_dual;
    VectorDualIndexPutSparseCPU(
        h_inputReal, input_idx, h_inputDualVal,
        sub_len,
        start, end,
        ref_resultReal, ref_idx, ref_dual,
        real_size
    );

    // Compare
    for (int i = 0; i < real_size; ++i) {
        ExpectComplexNear(ref_resultReal[i], h_resultReal[i]);
    }
    EXPECT_EQ(ref_idx, h_rIdxAfter);
    ExpectComplexNear(ref_dual, h_rDualValAfter);
}

// -----------------------------------------------------
// 4) Valid Index in [0..sub_len)
// -----------------------------------------------------
TEST(VectorDualIndexPutSparseTest, InRangeIndex)
{
    using T = double;

    int real_size = 6;
    std::vector<thrust::complex<T>> h_resultReal {
        {-1,-1}, {-2,-2}, {-3,-3},
        {-4,-4}, {-5,-5}, {-6,-6}
    };

    int start = 2, end = 5;
    int sub_len = end - start; // =3

    // input subrange
    std::vector<thrust::complex<T>> h_inputReal {
        {10,20}, {11,21}, {12,22}
    };
    // input_idx=1 => in range [0..3)
    int input_idx = 1;
    thrust::complex<T> h_inputDualVal(100,200);

    // GPU
    int h_rIdxAfter;
    thrust::complex<T> h_rDualValAfter;
    RunVectorDualIndexPutSparseKernel(
        h_inputReal, input_idx, h_inputDualVal, sub_len,
        start, end,
        h_resultReal, h_rIdxAfter, h_rDualValAfter,
        real_size
    );

    // CPU
    std::vector<thrust::complex<T>> ref_resultReal {
        {-1,-1}, {-2,-2}, {-3,-3},
        {-4,-4}, {-5,-5}, {-6,-6}
    };
    int ref_idx;
    thrust::complex<T> ref_dual;
    VectorDualIndexPutSparseCPU(
        h_inputReal, input_idx, h_inputDualVal,
        sub_len,
        start, end,
        ref_resultReal, ref_idx, ref_dual,
        real_size
    );

    // Compare
    for (int i = 0; i < real_size; ++i) {
        ExpectComplexNear(ref_resultReal[i], h_resultReal[i]);
    }
    EXPECT_EQ(ref_idx, h_rIdxAfter);
    ExpectComplexNear(ref_dual, h_rDualValAfter);
}



/**
 * VectorDualElementwiseAddSparseCPU
 *
 * "Add" two sparse dual vectors `a` and `b`, each of length real_size.
 *
 * - a_real[i], b_real[i] => real arrays
 * - a_idx, a_dual_val => single dual index/value for 'a' (or -1 if none)
 * - b_idx, b_dual_val => single dual index/value for 'b'
 *
 * The result has:
 *  - r_real[i] = a_real[i] + b_real[i]
 *  - r_idx, r_dual_val => logic:
 *      1) if both -1 => r_idx=-1, r_dual=0
 *      2) if exactly one is valid => pick that
 *      3) if both valid & same => sum the values
 *      4) if both valid & different => r_idx=-1, r_dual=0
 */
template <typename T>
void VectorDualElementwiseAddSparseCPU(
    // Vector a
    const std::vector<thrust::complex<T>>& a_real,
    int                                    a_idx,
    const thrust::complex<T>&              a_dual_val,

    // Vector b
    const std::vector<thrust::complex<T>>& b_real,
    int                                    b_idx,
    const thrust::complex<T>&              b_dual_val,

    // dimension
    int                                    real_size,

    // Result
    std::vector<thrust::complex<T>>&       r_real,       // length real_size
    int&                                   r_idx,
    thrust::complex<T>&                    r_dual_val
)
{
    // 1) Real part: elementwise add
    r_real.resize(real_size);
    for (int i = 0; i < real_size; ++i) {
        r_real[i] = a_real[i] + b_real[i];
    }

    // 2) Dual logic
    // CASE 1: both -1
    if (a_idx < 0 && b_idx < 0) {
        r_idx = -1;
        r_dual_val = thrust::complex<T>(0,0);
        return;
    }
    // CASE 2: only A valid
    if (a_idx >= 0 && b_idx < 0) {
        r_idx = a_idx;
        r_dual_val = a_dual_val;
        return;
    }
    // CASE 3: only B valid
    if (a_idx < 0 && b_idx >= 0) {
        r_idx = b_idx;
        r_dual_val = b_dual_val;
        return;
    }
    // CASE 4: both valid
    if (a_idx == b_idx) {
        // same index => sum the dual values
        r_idx = a_idx;
        r_dual_val = a_dual_val + b_dual_val;
    } else {
        // different => fallback => no dual
        r_idx = -1;
        r_dual_val = thrust::complex<T>(0,0);
    }
}



// ---------------------------------------------------------
// Helper to run the kernel
// ---------------------------------------------------------
template <typename T>
void RunVectorDualElementwiseAddSparseKernel(
    // input a
    const std::vector<thrust::complex<T>>& a_real,
    int                       a_idx,
    const thrust::complex<T>& a_dual_val,

    // input b
    const std::vector<thrust::complex<T>>& b_real,
    int                       b_idx,
    const thrust::complex<T>& b_dual_val,

    // dimension
    int real_size,

    // output
    std::vector<thrust::complex<T>>& r_real_after,
    int& r_idx_after,
    thrust::complex<T>& r_dual_after
)
{
    // 1) Device allocations
    thrust::complex<T>* d_aReal = nullptr;
    thrust::complex<T>* d_bReal = nullptr;
    thrust::complex<T>* d_aDualVal = nullptr;
    thrust::complex<T>* d_bDualVal = nullptr;
    thrust::complex<T>* d_rReal = nullptr;
    thrust::complex<T>* d_rDualVal = nullptr;
    int* d_rIdx = nullptr;

    size_t bytes = real_size * sizeof(thrust::complex<T>);

    cudaMalloc(&d_aReal, bytes);
    cudaMalloc(&d_bReal, bytes);
    cudaMalloc(&d_aDualVal, sizeof(thrust::complex<T>));
    cudaMalloc(&d_bDualVal, sizeof(thrust::complex<T>));
    cudaMalloc(&d_rReal, bytes);
    cudaMalloc(&d_rDualVal, sizeof(thrust::complex<T>));
    cudaMalloc(&d_rIdx, sizeof(int));

    // 2) Copy host->device
    cudaMemcpy(d_aReal, a_real.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bReal, b_real.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_aDualVal, &a_dual_val, sizeof(thrust::complex<T>), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bDualVal, &b_dual_val, sizeof(thrust::complex<T>), cudaMemcpyHostToDevice);

    // Initialize the result arrays with some sentinel
    std::vector<thrust::complex<T>> initR(real_size, thrust::complex<T>(-999,-999));
    cudaMemcpy(d_rReal, initR.data(), bytes, cudaMemcpyHostToDevice);
    thrust::complex<T> initDual(-777,-777);
    cudaMemcpy(d_rDualVal, &initDual, sizeof(thrust::complex<T>), cudaMemcpyHostToDevice);
    int initIdx = -666;
    cudaMemcpy(d_rIdx, &initIdx, sizeof(int), cudaMemcpyHostToDevice);

    // 3) Launch kernel
    dim3 block(64);
    dim3 grid( (real_size + block.x -1)/block.x );
    VectorDualElementwiseAddSparseKernel<T><<<grid, block>>>(
        d_aReal,  a_idx,  d_aDualVal,
        d_bReal,  b_idx,  d_bDualVal,
        real_size,
        d_rReal, d_rIdx, d_rDualVal
    );
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    // 4) Copy back
    r_real_after.resize(real_size);
    cudaMemcpy(r_real_after.data(), d_rReal, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(&r_dual_after, d_rDualVal, sizeof(thrust::complex<T>), cudaMemcpyDeviceToHost);
    cudaMemcpy(&r_idx_after, d_rIdx, sizeof(int), cudaMemcpyDeviceToHost);

    // 5) Free
    cudaFree(d_aReal);
    cudaFree(d_bReal);
    cudaFree(d_aDualVal);
    cudaFree(d_bDualVal);
    cudaFree(d_rReal);
    cudaFree(d_rDualVal);
    cudaFree(d_rIdx);
}


// ---------------------------------------------------------
// 1) Both indices = -1
// ---------------------------------------------------------
TEST(VectorDualElementwiseAddSparseTest, BothMinusOne)
{
    using T = double;
    int real_size = 5;

    // a, b real
    std::vector<thrust::complex<T>> h_aReal {
        {1,1}, {2,2}, {3,3}, {4,4}, {5,5}
    };
    std::vector<thrust::complex<T>> h_bReal {
        {10,0}, {9,0}, {8,0}, {7,0}, {6,0}
    };

    // a_idx = b_idx = -1 => no dual
    int a_idx = -1;
    thrust::complex<T> a_dualVal(100, 200);
    int b_idx = -1;
    thrust::complex<T> b_dualVal(300, 400);

    // GPU
    std::vector<thrust::complex<T>> h_rReal_after;
    int h_rIdx_after;
    thrust::complex<T> h_rDual_after;
    RunVectorDualElementwiseAddSparseKernel(
        h_aReal, a_idx, a_dualVal,
        h_bReal, b_idx, b_dualVal,
        real_size,
        h_rReal_after,
        h_rIdx_after,
        h_rDual_after
    );

    // CPU
    std::vector<thrust::complex<T>> ref_rReal;
    int ref_rIdx;
    thrust::complex<T> ref_rDual;
    VectorDualElementwiseAddSparseCPU(
        h_aReal, a_idx, a_dualVal,
        h_bReal, b_idx, b_dualVal,
        real_size,
        ref_rReal, ref_rIdx, ref_rDual
    );

    // Compare
    // Real
    ASSERT_EQ(real_size, (int)h_rReal_after.size());
    for (int i = 0; i < real_size; ++i) {
        ExpectComplexNear(ref_rReal[i], h_rReal_after[i]);
    }
    // Dual
    EXPECT_EQ(ref_rIdx, h_rIdx_after);
    ExpectComplexNear(ref_rDual, h_rDual_after);
}

// ---------------------------------------------------------
// 2) Only a_idx != -1
// ---------------------------------------------------------
TEST(VectorDualElementwiseAddSparseTest, OnlyAHasDual)
{
    using T = double;
    int real_size = 5;

    // a, b real
    std::vector<thrust::complex<T>> h_aReal {
        {1,1}, {2,2}, {3,3}, {4,4}, {5,5}
    };
    std::vector<thrust::complex<T>> h_bReal {
        {10,0}, {9,0}, {8,0}, {7,0}, {6,0}
    };

    // a_idx=2, b_idx=-1
    int a_idx = 2;
    thrust::complex<T> a_dualVal(100, 200);
    int b_idx = -1;
    thrust::complex<T> b_dualVal(300, 400);

    // GPU
    std::vector<thrust::complex<T>> h_rReal_after;
    int h_rIdx_after;
    thrust::complex<T> h_rDual_after;
    RunVectorDualElementwiseAddSparseKernel(
        h_aReal, a_idx, a_dualVal,
        h_bReal, b_idx, b_dualVal,
        real_size,
        h_rReal_after,
        h_rIdx_after,
        h_rDual_after
    );

    // CPU
    std::vector<thrust::complex<T>> ref_rReal;
    int ref_rIdx;
    thrust::complex<T> ref_rDual;
    VectorDualElementwiseAddSparseCPU(
        h_aReal, a_idx, a_dualVal,
        h_bReal, b_idx, b_dualVal,
        real_size,
        ref_rReal, ref_rIdx, ref_rDual
    );

    // Compare
    for (int i = 0; i < real_size; ++i) {
        ExpectComplexNear(ref_rReal[i], h_rReal_after[i]);
    }
    EXPECT_EQ(ref_rIdx, h_rIdx_after);
    ExpectComplexNear(ref_rDual, h_rDual_after);
}

// ---------------------------------------------------------
// 3) Only b_idx != -1
// ---------------------------------------------------------
TEST(VectorDualElementwiseAddSparseTest, OnlyBHasDual)
{
    using T = double;
    int real_size = 5;

    // a, b real
    std::vector<thrust::complex<T>> h_aReal {
        {1,1}, {2,2}, {3,3}, {4,4}, {5,5}
    };
    std::vector<thrust::complex<T>> h_bReal {
        {10,0}, {9,0}, {8,0}, {7,0}, {6,0}
    };

    // a_idx=-1, b_idx=1
    int a_idx = -1;
    thrust::complex<T> a_dualVal(100, 200);
    int b_idx = 1;
    thrust::complex<T> b_dualVal(300, 400);

    // GPU
    std::vector<thrust::complex<T>> h_rReal_after;
    int h_rIdx_after;
    thrust::complex<T> h_rDual_after;
    RunVectorDualElementwiseAddSparseKernel(
        h_aReal, a_idx, a_dualVal,
        h_bReal, b_idx, b_dualVal,
        real_size,
        h_rReal_after,
        h_rIdx_after,
        h_rDual_after
    );

    // CPU
    std::vector<thrust::complex<T>> ref_rReal;
    int ref_rIdx;
    thrust::complex<T> ref_rDual;
    VectorDualElementwiseAddSparseCPU(
        h_aReal, a_idx, a_dualVal,
        h_bReal, b_idx, b_dualVal,
        real_size,
        ref_rReal, ref_rIdx, ref_rDual
    );

    // Compare
    for (int i = 0; i < real_size; ++i) {
        ExpectComplexNear(ref_rReal[i], h_rReal_after[i]);
    }
    EXPECT_EQ(ref_rIdx, h_rIdx_after);
    ExpectComplexNear(ref_rDual, h_rDual_after);
}

// ---------------------------------------------------------
// 4) Both valid & same index
// ---------------------------------------------------------
TEST(VectorDualElementwiseAddSparseTest, BothSameIndex)
{
    using T = double;
    int real_size = 5;

    // a, b real
    std::vector<thrust::complex<T>> h_aReal {
        {1,1}, {2,2}, {3,3}, {4,4}, {5,5}
    };
    std::vector<thrust::complex<T>> h_bReal {
        {10,0}, {9,0}, {8,0}, {7,0}, {6,0}
    };

    // a_idx=2, b_idx=2
    int a_idx = 2;
    thrust::complex<T> a_dualVal(10,20);
    int b_idx = 2;
    thrust::complex<T> b_dualVal(30,40);

    // GPU
    std::vector<thrust::complex<T>> h_rReal_after;
    int h_rIdx_after;
    thrust::complex<T> h_rDual_after;
    RunVectorDualElementwiseAddSparseKernel(
        h_aReal, a_idx, a_dualVal,
        h_bReal, b_idx, b_dualVal,
        real_size,
        h_rReal_after,
        h_rIdx_after,
        h_rDual_after
    );

    // CPU
    std::vector<thrust::complex<T>> ref_rReal;
    int ref_rIdx;
    thrust::complex<T> ref_rDual;
    VectorDualElementwiseAddSparseCPU(
        h_aReal, a_idx, a_dualVal,
        h_bReal, b_idx, b_dualVal,
        real_size,
        ref_rReal, ref_rIdx, ref_rDual
    );

    // Compare
    for (int i = 0; i < real_size; ++i) {
        ExpectComplexNear(ref_rReal[i], h_rReal_after[i]);
    }
    EXPECT_EQ(ref_rIdx, h_rIdx_after);
    ExpectComplexNear(ref_rDual, h_rDual_after);
}

// ---------------------------------------------------------
// 5) Both valid & different index
// ---------------------------------------------------------
TEST(VectorDualElementwiseAddSparseTest, BothDifferentIndex)
{
    using T = double;
    int real_size = 5;

    // a, b real
    std::vector<thrust::complex<T>> h_aReal {
        {1,1}, {2,2}, {3,3}, {4,4}, {5,5}
    };
    std::vector<thrust::complex<T>> h_bReal {
        {10,0}, {9,0}, {8,0}, {7,0}, {6,0}
    };

    // a_idx=1, b_idx=4 => different => fallback => -1
    int a_idx = 1;
    thrust::complex<T> a_dualVal(10,20);
    int b_idx = 4;
    thrust::complex<T> b_dualVal(30,40);

    // GPU
    std::vector<thrust::complex<T>> h_rReal_after;
    int h_rIdx_after;
    thrust::complex<T> h_rDual_after;
    RunVectorDualElementwiseAddSparseKernel(
        h_aReal, a_idx, a_dualVal,
        h_bReal, b_idx, b_dualVal,
        real_size,
        h_rReal_after,
        h_rIdx_after,
        h_rDual_after
    );

    // CPU
    std::vector<thrust::complex<T>> ref_rReal;
    int ref_rIdx;
    thrust::complex<T> ref_rDual;
    VectorDualElementwiseAddSparseCPU(
        h_aReal, a_idx, a_dualVal,
        h_bReal, b_idx, b_dualVal,
        real_size,
        ref_rReal, ref_rIdx, ref_rDual
    );

    // Compare
    for (int i = 0; i < real_size; ++i) {
        ExpectComplexNear(ref_rReal[i], h_rReal_after[i]);
    }
    EXPECT_EQ(ref_rIdx, h_rIdx_after);
    ExpectComplexNear(ref_rDual, h_rDual_after);
}

#include <thrust/complex.h>
#include <vector>

/**
 * VectorDualElementwiseMultiplySparseCPU
 *
 * CPU reference for the sparse elementwise multiply:
 *   r_real[i] = a_real[i] * b_real[i], for i in [0..real_size)
 *
 * For the dual index:
 *   1) if (a_idx<0 && b_idx<0) => no dual
 *   2) if (a_idx>=0 && b_idx<0):
 *        r_idx=a_idx,
 *        r_dual=sum_{j=0..real_size-1}[ b_real[j]*a_dual ]
 *   3) if (a_idx<0 && b_idx>=0):
 *        r_idx=b_idx,
 *        r_dual=sum_{j=0..real_size-1}[ a_real[j]*b_dual ]
 *   4) if (a_idx==b_idx>=0):
 *        r_idx=a_idx,
 *        r_dual=sum_{j}[ a_real[j]*b_dual + b_real[j]*a_dual ]
 *   5) else => no dual
 */
template <typename T>
void VectorDualElementwiseMultiplySparseCPU(
    // a
    const std::vector<thrust::complex<T>>& a_real,
    int                                    a_idx,
    const thrust::complex<T>&              a_dual_val,

    // b
    const std::vector<thrust::complex<T>>& b_real,
    int                                    b_idx,
    const thrust::complex<T>&              b_dual_val,

    // dimension
    int                                    real_size,

    // outputs
    std::vector<thrust::complex<T>>&       r_real,
    int&                                   r_idx,
    thrust::complex<T>&                    r_dual
)
{
    // 1) Multiply real parts
    r_real.resize(real_size);
    for (int i = 0; i < real_size; ++i) {
        r_real[i] = a_real[i] * b_real[i];
    }

    // 2) Dual logic
    if (a_idx < 0 && b_idx < 0) {
        // no dual
        r_idx  = -1;
        r_dual = thrust::complex<T>(0,0);
        return;
    }
    if (a_idx >= 0 && b_idx < 0) {
        // only a_idx
        r_idx = a_idx;
        thrust::complex<T> sumVal(0,0);
        for (int j = 0; j < real_size; ++j) {
            sumVal += b_real[j]*a_dual_val;
        }
        r_dual = sumVal;
        return;
    }
    if (a_idx < 0 && b_idx >= 0) {
        // only b_idx
        r_idx = b_idx;
        thrust::complex<T> sumVal(0,0);
        for (int j = 0; j < real_size; ++j) {
            sumVal += a_real[j]*b_dual_val;
        }
        r_dual = sumVal;
        return;
    }
    // both valid
    if (a_idx == b_idx) {
        r_idx = a_idx;
        thrust::complex<T> sumVal(0,0);
        for (int j = 0; j < real_size; ++j) {
            sumVal += a_real[j]*b_dual_val + b_real[j]*a_dual_val;
        }
        r_dual = sumVal;
    } else {
        // different => no dual
        r_idx  = -1;
        r_dual = thrust::complex<T>(0,0);
    }
}

// -----------------------------------------------------
// Helper to run the kernel, copy results, etc.
// -----------------------------------------------------
template <typename T>
void RunVectorDualElementwiseMultiplySparseKernel(
    // a
    const std::vector<thrust::complex<T>>& a_real,
    int                       a_idx,
    const thrust::complex<T>& a_dual_val,

    // b
    const std::vector<thrust::complex<T>>& b_real,
    int                       b_idx,
    const thrust::complex<T>& b_dual_val,

    int real_size,

    // outputs
    std::vector<thrust::complex<T>>& r_real_after,
    int& r_idx_after,
    thrust::complex<T>& r_dual_after
)
{
    // 1) Allocate device
    thrust::complex<T>* d_aReal = nullptr;
    thrust::complex<T>* d_aDualVal = nullptr;
    thrust::complex<T>* d_bReal = nullptr;
    thrust::complex<T>* d_bDualVal = nullptr;
    thrust::complex<T>* d_rReal = nullptr;
    thrust::complex<T>* d_rDualVal = nullptr;
    int*                d_rIdx = nullptr;

    size_t bytes = real_size*sizeof(thrust::complex<T>);

    cudaMalloc(&d_aReal, bytes);
    cudaMalloc(&d_bReal, bytes);
    cudaMalloc(&d_aDualVal, sizeof(thrust::complex<T>));
    cudaMalloc(&d_bDualVal, sizeof(thrust::complex<T>));
    cudaMalloc(&d_rReal, bytes);
    cudaMalloc(&d_rDualVal, sizeof(thrust::complex<T>));
    cudaMalloc(&d_rIdx, sizeof(int));

    // 2) Copy host->device
    cudaMemcpy(d_aReal, a_real.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bReal, b_real.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_aDualVal, &a_dual_val, sizeof(thrust::complex<T>), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bDualVal, &b_dual_val, sizeof(thrust::complex<T>), cudaMemcpyHostToDevice);

    // Init result arrays (optional, for clarity)
    std::vector<thrust::complex<T>> initR(real_size, thrust::complex<T>(-999,-999));
    cudaMemcpy(d_rReal, initR.data(), bytes, cudaMemcpyHostToDevice);
    thrust::complex<T> initDual(-777,-777);
    cudaMemcpy(d_rDualVal, &initDual, sizeof(thrust::complex<T>), cudaMemcpyHostToDevice);
    int initIdx = -666;
    cudaMemcpy(d_rIdx, &initIdx, sizeof(int), cudaMemcpyHostToDevice);

    // 3) Launch kernel
    dim3 block(64);
    dim3 grid( (real_size + block.x -1)/block.x );
    VectorDualElementwiseMultiplySparseKernel<T><<<grid, block>>>(
        d_aReal, a_idx, d_aDualVal,
        d_bReal, b_idx, d_bDualVal,
        real_size,
        d_rReal, d_rIdx, d_rDualVal
    );
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    // 4) Copy results back
    r_real_after.resize(real_size);
    cudaMemcpy(r_real_after.data(), d_rReal, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(&r_dual_after, d_rDualVal, sizeof(thrust::complex<T>), cudaMemcpyDeviceToHost);
    cudaMemcpy(&r_idx_after, d_rIdx, sizeof(int), cudaMemcpyDeviceToHost);

    // 5) Cleanup
    cudaFree(d_aReal);
    cudaFree(d_bReal);
    cudaFree(d_aDualVal);
    cudaFree(d_bDualVal);
    cudaFree(d_rReal);
    cudaFree(d_rDualVal);
    cudaFree(d_rIdx);
}


// -----------------------------------------------------
// 1) Both indices = -1
// -----------------------------------------------------
TEST(VectorDualElementwiseMultiplySparseTest, BothMinusOne)
{
    using T = double;
    int real_size = 5;

    // a, b real
    std::vector<thrust::complex<T>> h_aReal {
        {1,1}, {2,2}, {3,3}, {4,4}, {5,5}
    };
    std::vector<thrust::complex<T>> h_bReal {
        {0,0}, {1,0}, {2,1}, {3,2}, {4,3}
    };

    // -1 => no dual
    int a_idx = -1;
    thrust::complex<T> a_dualVal(10,20);
    int b_idx = -1;
    thrust::complex<T> b_dualVal(30,40);

    // GPU
    std::vector<thrust::complex<T>> h_rReal_after;
    int h_rIdx_after;
    thrust::complex<T> h_rDual_after;
    RunVectorDualElementwiseMultiplySparseKernel(
        h_aReal, a_idx, a_dualVal,
        h_bReal, b_idx, b_dualVal,
        real_size,
        h_rReal_after, h_rIdx_after, h_rDual_after
    );

    // CPU
    std::vector<thrust::complex<T>> ref_rReal;
    int ref_rIdx;
    thrust::complex<T> ref_rDual;
    VectorDualElementwiseMultiplySparseCPU(
        h_aReal, a_idx, a_dualVal,
        h_bReal, b_idx, b_dualVal,
        real_size,
        ref_rReal, ref_rIdx, ref_rDual
    );

    // Compare real
    ASSERT_EQ(real_size, (int)h_rReal_after.size());
    for (int i = 0; i < real_size; ++i) {
        ExpectComplexNear(ref_rReal[i], h_rReal_after[i]);
    }
    // Compare dual
    EXPECT_EQ(ref_rIdx, h_rIdx_after);
    ExpectComplexNear(ref_rDual, h_rDual_after);
}

// -----------------------------------------------------
// 2) Only A Has Dual
// -----------------------------------------------------
TEST(VectorDualElementwiseMultiplySparseTest, OnlyAHasDual)
{
    using T = double;
    int real_size = 5;

    // a, b real
    std::vector<thrust::complex<T>> h_aReal {
        {1,1}, {2,2}, {3,3}, {4,4}, {5,5}
    };
    std::vector<thrust::complex<T>> h_bReal {
        {0,0}, {1,0}, {2,1}, {3,2}, {4,3}
    };

    int a_idx = 2;  // valid
    thrust::complex<T> a_dualVal(10,20);
    int b_idx = -1;
    thrust::complex<T> b_dualVal(30,40);

    // GPU
    std::vector<thrust::complex<T>> h_rReal_after;
    int h_rIdx_after;
    thrust::complex<T> h_rDual_after;
    RunVectorDualElementwiseMultiplySparseKernel(
        h_aReal, a_idx, a_dualVal,
        h_bReal, b_idx, b_dualVal,
        real_size,
        h_rReal_after, h_rIdx_after, h_rDual_after
    );

    // CPU
    std::vector<thrust::complex<T>> ref_rReal;
    int ref_rIdx;
    thrust::complex<T> ref_rDual;
    VectorDualElementwiseMultiplySparseCPU(
        h_aReal, a_idx, a_dualVal,
        h_bReal, b_idx, b_dualVal,
        real_size,
        ref_rReal, ref_rIdx, ref_rDual
    );

    // Compare
    for (int i = 0; i < real_size; ++i) {
        ExpectComplexNear(ref_rReal[i], h_rReal_after[i]);
    }
    EXPECT_EQ(ref_rIdx, h_rIdx_after);
    ExpectComplexNear(ref_rDual, h_rDual_after);
}

// -----------------------------------------------------
// 3) Only B Has Dual
// -----------------------------------------------------
TEST(VectorDualElementwiseMultiplySparseTest, OnlyBHasDual)
{
    using T = double;
    int real_size = 5;

    // a, b real
    std::vector<thrust::complex<T>> h_aReal {
        {1,1}, {2,2}, {3,3}, {4,4}, {5,5}
    };
    std::vector<thrust::complex<T>> h_bReal {
        {0,0}, {1,0}, {2,1}, {3,2}, {4,3}
    };

    int a_idx = -1;
    thrust::complex<T> a_dualVal(10,20);
    int b_idx = 2; // valid
    thrust::complex<T> b_dualVal(30,40);

    // GPU
    std::vector<thrust::complex<T>> h_rReal_after;
    int h_rIdx_after;
    thrust::complex<T> h_rDual_after;
    RunVectorDualElementwiseMultiplySparseKernel(
        h_aReal, a_idx, a_dualVal,
        h_bReal, b_idx, b_dualVal,
        real_size,
        h_rReal_after, h_rIdx_after, h_rDual_after
    );

    // CPU
    std::vector<thrust::complex<T>> ref_rReal;
    int ref_rIdx;
    thrust::complex<T> ref_rDual;
    VectorDualElementwiseMultiplySparseCPU(
        h_aReal, a_idx, a_dualVal,
        h_bReal, b_idx, b_dualVal,
        real_size,
        ref_rReal, ref_rIdx, ref_rDual
    );

    // Compare
    for (int i = 0; i < real_size; ++i) {
        ExpectComplexNear(ref_rReal[i], h_rReal_after[i]);
    }
    EXPECT_EQ(ref_rIdx, h_rIdx_after);
    ExpectComplexNear(ref_rDual, h_rDual_after);
}

// -----------------------------------------------------
// 4) Both valid & same index
// -----------------------------------------------------
TEST(VectorDualElementwiseMultiplySparseTest, BothSameIndex)
{
    using T = double;
    int real_size = 5;

    // a, b real
    std::vector<thrust::complex<T>> h_aReal {
        {1,1}, {2,2}, {3,3}, {4,4}, {5,5}
    };
    std::vector<thrust::complex<T>> h_bReal {
        {0,0}, {1,0}, {2,1}, {3,2}, {4,3}
    };

    int a_idx = 2;
    thrust::complex<T> a_dualVal(10,20);
    int b_idx = 2;
    thrust::complex<T> b_dualVal(30,40);

    // GPU
    std::vector<thrust::complex<T>> h_rReal_after;
    int h_rIdx_after;
    thrust::complex<T> h_rDual_after;
    RunVectorDualElementwiseMultiplySparseKernel(
        h_aReal, a_idx, a_dualVal,
        h_bReal, b_idx, b_dualVal,
        real_size,
        h_rReal_after, h_rIdx_after, h_rDual_after
    );

    // CPU
    std::vector<thrust::complex<T>> ref_rReal;
    int ref_rIdx;
    thrust::complex<T> ref_rDual;
    VectorDualElementwiseMultiplySparseCPU(
        h_aReal, a_idx, a_dualVal,
        h_bReal, b_idx, b_dualVal,
        real_size,
        ref_rReal, ref_rIdx, ref_rDual
    );

    // Compare
    for (int i = 0; i < real_size; ++i) {
        ExpectComplexNear(ref_rReal[i], h_rReal_after[i]);
    }
    EXPECT_EQ(ref_rIdx, h_rIdx_after);
    ExpectComplexNear(ref_rDual, h_rDual_after);
}

// -----------------------------------------------------
// 5) Both valid & different index
// -----------------------------------------------------
TEST(VectorDualElementwiseMultiplySparseTest, BothDifferentIndex)
{
    using T = double;
    int real_size = 5;

    // a, b real
    std::vector<thrust::complex<T>> h_aReal {
        {1,1}, {2,2}, {3,3}, {4,4}, {5,5}
    };
    std::vector<thrust::complex<T>> h_bReal {
        {0,0}, {1,0}, {2,1}, {3,2}, {4,3}
    };

    int a_idx = 1;
    thrust::complex<T> a_dualVal(10,20);
    int b_idx = 4;
    thrust::complex<T> b_dualVal(30,40);

    // GPU
    std::vector<thrust::complex<T>> h_rReal_after;
    int h_rIdx_after;
    thrust::complex<T> h_rDual_after;
    RunVectorDualElementwiseMultiplySparseKernel(
        h_aReal, a_idx, a_dualVal,
        h_bReal, b_idx, b_dualVal,
        real_size,
        h_rReal_after, h_rIdx_after, h_rDual_after
    );

    // CPU
    std::vector<thrust::complex<T>> ref_rReal;
    int ref_rIdx;
    thrust::complex<T> ref_rDual;
    VectorDualElementwiseMultiplySparseCPU(
        h_aReal, a_idx, a_dualVal,
        h_bReal, b_idx, b_dualVal,
        real_size,
        ref_rReal, ref_rIdx, ref_rDual
    );

    // Compare
    for (int i = 0; i < real_size; ++i) {
        ExpectComplexNear(ref_rReal[i], h_rReal_after[i]);
    }
    EXPECT_EQ(ref_rIdx, h_rIdx_after);
    ExpectComplexNear(ref_rDual, h_rDual_after);
}

#include <thrust/complex.h>
#include <vector>

/**
 * VectorDualPowSparseCPU
 *
 * A CPU reference for your sparse "pow" operation:
 *    r_real[i] = (a_real[i])^power
 *    r_idx, r_dual = logic below
 *
 * If a_idx == -1 => no dual => r_idx=-1, r_dual=0
 * Otherwise => r_idx=a_idx,
 *              r_dual = sum_{j=0..real_size-1} [ power * (a_real[j])^(power-1) * a_dual_val ]
 */
template <typename T>
void VectorDualPowSparseCPU(
    // Input
    const std::vector<thrust::complex<T>>& a_real,
    int                                    a_idx,
    const thrust::complex<T>&              a_dual_val,
    T                                      power,
    int                                    real_size,

    // Output
    std::vector<thrust::complex<T>>&       r_real,
    int&                                   r_idx,
    thrust::complex<T>&                    r_dual
)
{
    // 1) Compute real: a_real[i]^power
    r_real.resize(real_size);
    for (int i = 0; i < real_size; ++i) {
        r_real[i] = thrust::pow(a_real[i], power);
    }

    // 2) Dual logic
    if (a_idx < 0) {
        // No dual
        r_idx  = -1;
        r_dual = thrust::complex<T>(0,0);
    } else {
        // Single nonzero index
        r_idx = a_idx;
        thrust::complex<T> sumVal(0,0);
        for (int j = 0; j < real_size; ++j) {
            // power * (a_real[j])^(power - 1) * a_dual_val
            thrust::complex<T> factor = thrust::pow(a_real[j], power - T(1));
            sumVal += power * factor * a_dual_val;
        }
        r_dual = sumVal;
    }
}



// ----------------------------------------------------
// Helper to run the kernel and copy back results
// ----------------------------------------------------
template <typename T>
void RunVectorDualPowSparseKernel(
    // input
    const std::vector<thrust::complex<T>>& a_real,
    int                       a_idx,
    const thrust::complex<T>& a_dual_val,
    T                         power,
    int                       real_size,

    // outputs
    std::vector<thrust::complex<T>>& r_real_after,
    int& r_idx_after,
    thrust::complex<T>& r_dual_after
)
{
    // 1) Allocate device
    thrust::complex<T>* d_aReal = nullptr;
    thrust::complex<T>* d_aDualVal = nullptr;
    thrust::complex<T>* d_rReal = nullptr;
    thrust::complex<T>* d_rDualVal = nullptr;
    int* d_rIdx = nullptr;

    size_t bytes = real_size * sizeof(thrust::complex<T>);
    cudaMalloc(&d_aReal, bytes);
    cudaMalloc(&d_aDualVal, sizeof(thrust::complex<T>));
    cudaMalloc(&d_rReal, bytes);
    cudaMalloc(&d_rDualVal, sizeof(thrust::complex<T>));
    cudaMalloc(&d_rIdx, sizeof(int));

    // 2) Copy host->device
    cudaMemcpy(d_aReal, a_real.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_aDualVal, &a_dual_val, sizeof(thrust::complex<T>), cudaMemcpyHostToDevice);

    // Initialize result arrays with sentinel
    std::vector<thrust::complex<T>> initR(real_size, thrust::complex<T>(-999,-999));
    cudaMemcpy(d_rReal, initR.data(), bytes, cudaMemcpyHostToDevice);
    thrust::complex<T> initDual(-777,-777);
    cudaMemcpy(d_rDualVal, &initDual, sizeof(thrust::complex<T>), cudaMemcpyHostToDevice);
    int initIdx = -666;
    cudaMemcpy(d_rIdx, &initIdx, sizeof(int), cudaMemcpyHostToDevice);

    // 3) Launch kernel
    dim3 block(64);
    dim3 grid( (real_size + block.x -1)/block.x );
    VectorDualPowSparseKernel<T><<<grid, block>>>(
        d_aReal,
        a_idx,
        d_aDualVal,
        power,
        real_size,
        d_rReal,
        d_rIdx,
        d_rDualVal
    );
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    // 4) Copy results back
    r_real_after.resize(real_size);
    cudaMemcpy(r_real_after.data(), d_rReal, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(&r_dual_after, d_rDualVal, sizeof(thrust::complex<T>), cudaMemcpyDeviceToHost);
    cudaMemcpy(&r_idx_after, d_rIdx, sizeof(int), cudaMemcpyDeviceToHost);

    // 5) Cleanup
    cudaFree(d_aReal);
    cudaFree(d_aDualVal);
    cudaFree(d_rReal);
    cudaFree(d_rDualVal);
    cudaFree(d_rIdx);
}


// ----------------------------------------------------
// 1) Test: a_idx = -1 => no dual
// ----------------------------------------------------
TEST(VectorDualPowSparseTest, NoDualCase)
{
    using T = double;
    int real_size = 4;
    T power       = 2.0;

    // a_real = [1,2,3,4]
    std::vector<thrust::complex<T>> h_aReal {
        {1,0}, {2,0}, {3,0}, {4,0}
    };
    int h_aIdx = -1; // no dual
    thrust::complex<T> h_aDualVal(100,200);

    // GPU results
    std::vector<thrust::complex<T>> h_rRealAfter;
    int h_rIdxAfter;
    thrust::complex<T> h_rDualAfter;

    RunVectorDualPowSparseKernel(
        h_aReal, h_aIdx, h_aDualVal, power, real_size,
        h_rRealAfter, h_rIdxAfter, h_rDualAfter
    );

    // CPU reference
    std::vector<thrust::complex<T>> ref_rReal;
    int ref_rIdx;
    thrust::complex<T> ref_rDual;
    VectorDualPowSparseCPU(
        h_aReal, h_aIdx, h_aDualVal,
        power, real_size,
        ref_rReal, ref_rIdx, ref_rDual
    );

    // Compare real
    ASSERT_EQ((int)ref_rReal.size(), (int)h_rRealAfter.size());
    for (int i = 0; i < real_size; ++i) {
        ExpectComplexNear(ref_rReal[i], h_rRealAfter[i]);
    }
    // Compare dual
    EXPECT_EQ(ref_rIdx, h_rIdxAfter);
    ExpectComplexNear(ref_rDual, h_rDualAfter);
}

// ----------------------------------------------------
// 2) Test: a_idx >= 0 => valid dual, power=2.0
// ----------------------------------------------------
TEST(VectorDualPowSparseTest, ValidDualSquarePower)
{
    using T = double;
    int real_size = 4;
    T power       = 2.0;

    // a_real = [1,2,3,4]
    std::vector<thrust::complex<T>> h_aReal {
        {1,0}, {2,0}, {3,0}, {4,0}
    };
    int h_aIdx = 2;
    thrust::complex<T> h_aDualVal(10,20);

    // GPU
    std::vector<thrust::complex<T>> h_rRealAfter;
    int h_rIdxAfter;
    thrust::complex<T> h_rDualAfter;
    RunVectorDualPowSparseKernel(
        h_aReal, h_aIdx, h_aDualVal, power, real_size,
        h_rRealAfter, h_rIdxAfter, h_rDualAfter
    );

    // CPU
    std::vector<thrust::complex<T>> ref_rReal;
    int ref_rIdx;
    thrust::complex<T> ref_rDual;
    VectorDualPowSparseCPU(
        h_aReal, h_aIdx, h_aDualVal,
        power, real_size,
        ref_rReal, ref_rIdx, ref_rDual
    );

    // Compare
    ASSERT_EQ((int)ref_rReal.size(), (int)h_rRealAfter.size());
    for (int i = 0; i < real_size; ++i) {
        ExpectComplexNear(ref_rReal[i], h_rRealAfter[i]);
    }
    EXPECT_EQ(ref_rIdx, h_rIdxAfter);
    ExpectComplexNear(ref_rDual, h_rDualAfter);
}

// ----------------------------------------------------
// 3) Test: a_idx >= 0, power=3.5 (non-integer exponent)
// ----------------------------------------------------
TEST(VectorDualPowSparseTest, ValidDualFractionalPower)
{
    using T = double;
    int real_size = 5;
    T power       = 3.5; // e.g. a fractional exponent

    // a_real = [1, 2, 3, 4, 5]
    std::vector<thrust::complex<T>> h_aReal {
        {1,0}, {2,0}, {3,0}, {4,0}, {5,0}
    };
    int h_aIdx = 1;
    thrust::complex<T> h_aDualVal(10, -20);

    // GPU
    std::vector<thrust::complex<T>> h_rRealAfter;
    int h_rIdxAfter;
    thrust::complex<T> h_rDualAfter;
    RunVectorDualPowSparseKernel(
        h_aReal, h_aIdx, h_aDualVal, power, real_size,
        h_rRealAfter, h_rIdxAfter, h_rDualAfter
    );

    // CPU
    std::vector<thrust::complex<T>> ref_rReal;
    int ref_rIdx;
    thrust::complex<T> ref_rDual;
    VectorDualPowSparseCPU(
        h_aReal, h_aIdx, h_aDualVal,
        power, real_size,
        ref_rReal, ref_rIdx, ref_rDual
    );

    // Compare
    ASSERT_EQ((int)ref_rReal.size(), (int)h_rRealAfter.size());
    for (int i = 0; i < real_size; ++i) {
        ExpectComplexNear(ref_rReal[i], h_rRealAfter[i]);
    }
    EXPECT_EQ(ref_rIdx, h_rIdxAfter);
    ExpectComplexNear(ref_rDual, h_rDualAfter);
}

#include <thrust/complex.h>
#include <vector>

/**
 * VectorDualSqrtSparseCPU
 *
 * A CPU reference for your sparse "sqrt" operation.
 * It's effectively calling "VectorDualPowSparseCPU(..., 0.5)".
 *
 * If a_idx == -1 => no dual.
 * Otherwise => a single sum of [ 0.5 * (a_real[j]^(-0.5)) * a_dual ] across j.
 */
template <typename T>
void VectorDualSqrtSparseCPU(
    const std::vector<thrust::complex<T>>& a_real,
    int                                    a_idx,
    const thrust::complex<T>&              a_dual_val,
    int                                    real_size,

    // outputs
    std::vector<thrust::complex<T>>&       r_real,
    int&                                   r_idx,
    thrust::complex<T>&                    r_dual
)
{
    // 1) Compute real: sqrt(a_real[i]) = pow(a_real[i], 0.5)
    r_real.resize(real_size);
    for (int i = 0; i < real_size; ++i) {
        r_real[i] = thrust::pow(a_real[i], T(0.5));
    }

    // 2) Dual logic
    if (a_idx < 0) {
        // No dual
        r_idx  = -1;
        r_dual = thrust::complex<T>(0, 0);
    } else {
        // Single nonzero index
        r_idx = a_idx;
        thrust::complex<T> sumVal(0, 0);
        for (int j = 0; j < real_size; ++j) {
            // derivative of sqrt(z) => 0.5 * (z)^{-0.5}
            thrust::complex<T> factor = thrust::pow(a_real[j], T(0.5) - T(1.0)); // (a_real[j])^{-0.5}
            sumVal += T(0.5) * factor * a_dual_val;
        }
        r_dual = sumVal;
    }
}



// Helper to run the kernel
template <typename T>
void RunVectorDualSqrtSparseKernel(
    // inputs
    const std::vector<thrust::complex<T>>& a_real,
    int                       a_idx,
    const thrust::complex<T>& a_dual_val,
    int                       real_size,

    // outputs
    std::vector<thrust::complex<T>>& r_real_out,
    int& r_idx_out,
    thrust::complex<T>& r_dual_out
)
{
    thrust::complex<T>* d_aReal = nullptr;
    thrust::complex<T>* d_aDualVal = nullptr;
    thrust::complex<T>* d_rReal = nullptr;
    thrust::complex<T>* d_rDualVal = nullptr;
    int* d_rIdx = nullptr;

    size_t bytes = real_size*sizeof(thrust::complex<T>);
    cudaMalloc(&d_aReal,    bytes);
    cudaMalloc(&d_aDualVal, sizeof(thrust::complex<T>));
    cudaMalloc(&d_rReal,    bytes);
    cudaMalloc(&d_rDualVal, sizeof(thrust::complex<T>));
    cudaMalloc(&d_rIdx,     sizeof(int));

    // Copy host -> device
    cudaMemcpy(d_aReal, a_real.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_aDualVal, &a_dual_val, sizeof(thrust::complex<T>), cudaMemcpyHostToDevice);

    // Initialize result arrays
    std::vector<thrust::complex<T>> initR(real_size, thrust::complex<T>(-999,-999));
    cudaMemcpy(d_rReal, initR.data(), bytes, cudaMemcpyHostToDevice);
    thrust::complex<T> initDual(-777,-777);
    cudaMemcpy(d_rDualVal, &initDual, sizeof(thrust::complex<T>), cudaMemcpyHostToDevice);
    int initIdx = -666;
    cudaMemcpy(d_rIdx, &initIdx, sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 block(64);
    dim3 grid( (real_size + block.x -1)/block.x );
    VectorDualSqrtSparseKernel<T><<<grid, block>>>(
        d_aReal,
        a_idx,
        d_aDualVal,
        real_size,
        d_rReal,
        d_rIdx,
        d_rDualVal
    );
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    // Copy results back
    r_real_out.resize(real_size);
    cudaMemcpy(r_real_out.data(), d_rReal, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(&r_dual_out, d_rDualVal, sizeof(thrust::complex<T>), cudaMemcpyDeviceToHost);
    cudaMemcpy(&r_idx_out, d_rIdx, sizeof(int), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_aReal);
    cudaFree(d_aDualVal);
    cudaFree(d_rReal);
    cudaFree(d_rDualVal);
    cudaFree(d_rIdx);
}


// -----------------------------------------------------------
// 1) a_idx = -1 => no dual
// -----------------------------------------------------------
TEST(VectorDualSqrtSparseTest, NoDualCase)
{
    using T = double;
    int real_size = 4;

    // a_real => [1,4,9,16]
    std::vector<thrust::complex<T>> h_aReal {
        {1,0}, {4,0}, {9,0}, {16,0}
    };
    int h_aIdx = -1;
    thrust::complex<T> h_aDualVal(100,200);

    // GPU
    std::vector<thrust::complex<T>> h_rRealAfter;
    int h_rIdxAfter;
    thrust::complex<T> h_rDualAfter;
    RunVectorDualSqrtSparseKernel(
        h_aReal, h_aIdx, h_aDualVal,
        real_size,
        h_rRealAfter, h_rIdxAfter, h_rDualAfter
    );

    // CPU
    std::vector<thrust::complex<T>> ref_rReal;
    int ref_rIdx;
    thrust::complex<T> ref_rDual;
    VectorDualSqrtSparseCPU(
        h_aReal, h_aIdx, h_aDualVal,
        real_size,
        ref_rReal, ref_rIdx, ref_rDual
    );

    // Compare real
    for (int i = 0; i < real_size; ++i) {
        ExpectComplexNear(ref_rReal[i], h_rRealAfter[i]);
    }
    // Compare dual
    EXPECT_EQ(ref_rIdx, h_rIdxAfter);
    ExpectComplexNear(ref_rDual, h_rDualAfter);
}

// -----------------------------------------------------------
// 2) a_idx >= 0 => valid dual, simple integer squares
// -----------------------------------------------------------
TEST(VectorDualSqrtSparseTest, ValidDualSimpleSquares)
{
    using T = double;
    int real_size = 4;

    // a_real => [1,4,9,16]
    std::vector<thrust::complex<T>> h_aReal {
        {1,0}, {4,0}, {9,0}, {16,0}
    };
    int h_aIdx = 2;
    thrust::complex<T> h_aDualVal(10,20);

    // GPU
    std::vector<thrust::complex<T>> h_rRealAfter;
    int h_rIdxAfter;
    thrust::complex<T> h_rDualAfter;
    RunVectorDualSqrtSparseKernel(
        h_aReal, h_aIdx, h_aDualVal,
        real_size,
        h_rRealAfter, h_rIdxAfter, h_rDualAfter
    );

    // CPU
    std::vector<thrust::complex<T>> ref_rReal;
    int ref_rIdx;
    thrust::complex<T> ref_rDual;
    VectorDualSqrtSparseCPU(
        h_aReal, h_aIdx, h_aDualVal,
        real_size,
        ref_rReal, ref_rIdx, ref_rDual
    );

    // Compare
    for (int i = 0; i < real_size; ++i) {
        ExpectComplexNear(ref_rReal[i], h_rRealAfter[i]);
    }
    EXPECT_EQ(ref_rIdx, h_rIdxAfter);
    ExpectComplexNear(ref_rDual, h_rDualAfter);
}

// -----------------------------------------------------------
// 3) a_idx >= 0 => with non-integer + negative imaginary
// -----------------------------------------------------------
TEST(VectorDualSqrtSparseTest, ValidDualComplexReals)
{
    using T = double;
    int real_size = 5;

    // a_real => e.g. [ (1,1), (4,0), (2,2), (9,0), (0.25, 0) ]
    std::vector<thrust::complex<T>> h_aReal {
        {1,1}, {4,0}, {2,2}, {9,0}, {0.25,0}
    };
    int h_aIdx = 1;
    thrust::complex<T> h_aDualVal(10, -20);

    // GPU
    std::vector<thrust::complex<T>> h_rRealAfter;
    int h_rIdxAfter;
    thrust::complex<T> h_rDualAfter;
    RunVectorDualSqrtSparseKernel(
        h_aReal, h_aIdx, h_aDualVal,
        real_size,
        h_rRealAfter, h_rIdxAfter, h_rDualAfter
    );

    // CPU
    std::vector<thrust::complex<T>> ref_rReal;
    int ref_rIdx;
    thrust::complex<T> ref_rDual;
    VectorDualSqrtSparseCPU(
        h_aReal, h_aIdx, h_aDualVal,
        real_size,
        ref_rReal, ref_rIdx, ref_rDual
    );

    // Compare
    ASSERT_EQ(ref_rReal.size(), h_rRealAfter.size());
    for (int i = 0; i < real_size; ++i) {
        // Compare each real element
        ExpectComplexNear(ref_rReal[i], h_rRealAfter[i]);
    }
    // Compare dual index + value
    EXPECT_EQ(ref_rIdx, h_rIdxAfter);
    ExpectComplexNear(ref_rDual, h_rDualAfter);
}


#include <thrust/complex.h>
#include <vector>

/**
 * VectorDualReduceSparseCPU
 *
 * CPU reference for your "sparse" reduce:
 *   - Sums all a_real[i], i in [0..real_size).
 *   - If a_idx >= 0, the dual sum is just a_dual_val. Otherwise 0.
 */
template <typename T>
void VectorDualReduceSparseCPU(
    const std::vector<thrust::complex<T>>& a_real,
    int                                    a_idx,
    const thrust::complex<T>&              a_dual_val,
    int                                    real_size,

    // outputs
    thrust::complex<T>&                    result_real,
    thrust::complex<T>&                    result_dual
)
{
    // 1) Sum real
    thrust::complex<T> sumReal(0,0);
    for (int i = 0; i < real_size; ++i) {
        sumReal += a_real[i];
    }
    result_real = sumReal;

    // 2) Dual
    if (a_idx < 0) {
        result_dual = thrust::complex<T>(0,0);
    } else {
        // Single nonzero entry
        result_dual = a_dual_val;
    }
}

// Helper to run the kernel
template <typename T>
void RunVectorDualReduceSparseKernel(
    // input sparse vector
    const std::vector<thrust::complex<T>>& a_real,
    int                       a_idx,
    const thrust::complex<T>& a_dual_val,
    int                       real_size,

    // outputs
    thrust::complex<T>&       result_real_out,
    thrust::complex<T>&       result_dual_out
)
{
    // 1) Device allocations
    thrust::complex<T>* d_aReal     = nullptr;
    thrust::complex<T>* d_aDualVal  = nullptr;
    thrust::complex<T>* d_rReal     = nullptr;
    thrust::complex<T>* d_rDual     = nullptr;

    size_t bytes = real_size*sizeof(thrust::complex<T>);
    cudaMalloc(&d_aReal,    bytes);
    cudaMalloc(&d_aDualVal, sizeof(thrust::complex<T>));
    cudaMalloc(&d_rReal,    sizeof(thrust::complex<T>));
    cudaMalloc(&d_rDual,    sizeof(thrust::complex<T>));

    // 2) Copy host->device
    cudaMemcpy(d_aReal, a_real.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_aDualVal, &a_dual_val, sizeof(thrust::complex<T>), cudaMemcpyHostToDevice);

    // 3) Launch kernel
    // Suppose we do a single-block approach with blockDim.x >= real_size, for simplicity.
    int blockSize = 128;
    int gridSize  = 1;
    VectorDualReduceSparseKernel<T><<<gridSize, blockSize>>>(
        d_aReal, a_idx, d_aDualVal,
        real_size,
        d_rReal, d_rDual
    );
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    // 4) Copy results back
    cudaMemcpy(&result_real_out, d_rReal, sizeof(thrust::complex<T>), cudaMemcpyDeviceToHost);
    cudaMemcpy(&result_dual_out, d_rDual, sizeof(thrust::complex<T>), cudaMemcpyDeviceToHost);

    // 5) Cleanup
    cudaFree(d_aReal);
    cudaFree(d_aDualVal);
    cudaFree(d_rReal);
    cudaFree(d_rDual);
}


// --------------------------------------------------------
// 1) a_idx = -1 => no dual
// --------------------------------------------------------
TEST(VectorDualReduceSparseTest, NoDualCase)
{
    using T = double;
    int real_size = 5;

    // a_real => [ (1,1), (2,2), (3,3), (4,4), (5,5) ]
    std::vector<thrust::complex<T>> h_aReal {
        {1,1}, {2,2}, {3,3}, {4,4}, {5,5}
    };
    int h_aIdx = -1;
    thrust::complex<T> h_aDualVal(100,200);

    // GPU
    thrust::complex<T> h_rRealAfter, h_rDualAfter;
    RunVectorDualReduceSparseKernel(
        h_aReal, h_aIdx, h_aDualVal,
        real_size,
        h_rRealAfter, h_rDualAfter
    );

    // CPU
    thrust::complex<T> ref_rReal, ref_rDual;
    VectorDualReduceSparseCPU(
        h_aReal, h_aIdx, h_aDualVal,
        real_size,
        ref_rReal, ref_rDual
    );

    // Compare
    ExpectComplexNear(ref_rReal, h_rRealAfter);
    ExpectComplexNear(ref_rDual, h_rDualAfter);
}

// --------------------------------------------------------
// 2) a_idx >= 0 => valid dual
// --------------------------------------------------------
TEST(VectorDualReduceSparseTest, ValidDualCase)
{
    using T = double;
    int real_size = 5;

    std::vector<thrust::complex<T>> h_aReal {
        {1,1}, {2,2}, {3,3}, {4,4}, {5,5}
    };
    int h_aIdx = 2;  // e.g. single dual at index 2
    thrust::complex<T> h_aDualVal(10, 20);

    // GPU
    thrust::complex<T> h_rRealAfter, h_rDualAfter;
    RunVectorDualReduceSparseKernel(
        h_aReal, h_aIdx, h_aDualVal,
        real_size,
        h_rRealAfter, h_rDualAfter
    );

    // CPU
    thrust::complex<T> ref_rReal, ref_rDual;
    VectorDualReduceSparseCPU(
        h_aReal, h_aIdx, h_aDualVal,
        real_size,
        ref_rReal, ref_rDual
    );

    // Compare
    ExpectComplexNear(ref_rReal, h_rRealAfter);
    ExpectComplexNear(ref_rDual, h_rDualAfter);
}

// --------------------------------------------------------
// 3) Another variation with different real values
// --------------------------------------------------------
TEST(VectorDualReduceSparseTest, AnotherValidDualCase)
{
    using T = double;
    int real_size = 4;

    // a_real => arbitrary complex values
    std::vector<thrust::complex<T>> h_aReal {
        {10,1}, {0,2}, {5,-1}, {2.5, 4.5}
    };
    int h_aIdx = 0;
    thrust::complex<T> h_aDualVal(999, -999);

    // GPU
    thrust::complex<T> h_rRealAfter, h_rDualAfter;
    RunVectorDualReduceSparseKernel(
        h_aReal, h_aIdx, h_aDualVal,
        real_size,
        h_rRealAfter, h_rDualAfter
    );

    // CPU
    thrust::complex<T> ref_rReal, ref_rDual;
    VectorDualReduceSparseCPU(
        h_aReal, h_aIdx, h_aDualVal,
        real_size,
        ref_rReal, ref_rDual
    );

    // Compare
    ExpectComplexNear(ref_rReal, h_rRealAfter);
    ExpectComplexNear(ref_rDual, h_rDualAfter);
}

#include <thrust/complex.h>
#include <vector>
#include <cmath> // for sin, cos

/**
 * VectorDualCosSparseCPU
 *
 * A CPU reference for your sparse "cos" operation:
 *   - r_real[i] = cos(a_real[i]) 
 *   - if a_idx=-1 => r_idx=-1, r_dual=0
 *     else => r_idx=a_idx, r_dual = sum_j [ -sin(a_real[j]) * a_dual_val ]
 */
template <typename T>
void VectorDualCosSparseCPU(
    // Input sparse vector a
    const std::vector<thrust::complex<T>>& a_real,
    int                                    a_idx,
    const thrust::complex<T>&              a_dual_val,
    int                                    real_size,

    // Outputs
    std::vector<thrust::complex<T>>&       r_real,
    int&                                   r_idx,
    thrust::complex<T>&                    r_dual
)
{
    // 1) Real part: cos(a_real[i])
    r_real.resize(real_size);
    for (int i = 0; i < real_size; ++i) {
        // cos of a complex: we can use thrust::cos or handle real-only logic
        r_real[i] = thrust::cos(a_real[i]);
    }

    // 2) Dual part
    if (a_idx < 0) {
        r_idx  = -1;
        r_dual = thrust::complex<T>(0,0);
    } else {
        r_idx = a_idx;
        thrust::complex<T> sumVal(0,0);
        for (int j = 0; j < real_size; ++j) {
            sumVal += -thrust::sin(a_real[j]) * a_dual_val;
        }
        r_dual = sumVal;
    }
}


// ------------------------------------------
// Helper to run the kernel
// ------------------------------------------
template <typename T>
void RunVectorDualCosSparseKernel(
    const std::vector<thrust::complex<T>>& a_real,
    int                       a_idx,
    const thrust::complex<T>& a_dual_val,
    int                       real_size,

    // outputs
    std::vector<thrust::complex<T>>& r_real_out,
    int& r_idx_out,
    thrust::complex<T>& r_dual_out
)
{
    // Allocate device
    thrust::complex<T>* d_aReal = nullptr;
    thrust::complex<T>* d_aDualVal = nullptr;
    thrust::complex<T>* d_rReal = nullptr;
    thrust::complex<T>* d_rDualVal = nullptr;
    int* d_rIdx = nullptr;

    size_t bytes = real_size*sizeof(thrust::complex<T>);
    cudaMalloc(&d_aReal, bytes);
    cudaMalloc(&d_aDualVal, sizeof(thrust::complex<T>));
    cudaMalloc(&d_rReal, bytes);
    cudaMalloc(&d_rDualVal, sizeof(thrust::complex<T>));
    cudaMalloc(&d_rIdx, sizeof(int));

    // Copy host->device
    cudaMemcpy(d_aReal, a_real.data(),
               bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_aDualVal, &a_dual_val,
               sizeof(thrust::complex<T>),
               cudaMemcpyHostToDevice);

    // Initialize result arrays if you want (not strictly required)
    std::vector<thrust::complex<T>> initR(real_size, thrust::complex<T>(-999,-999));
    cudaMemcpy(d_rReal, initR.data(), bytes, cudaMemcpyHostToDevice);
    thrust::complex<T> initDual(-777,-777);
    cudaMemcpy(d_rDualVal, &initDual, sizeof(thrust::complex<T>),
               cudaMemcpyHostToDevice);
    int initIdx = -666;
    cudaMemcpy(d_rIdx, &initIdx, sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel
    int blockSize = 64;
    int gridSize  = (real_size + blockSize -1)/blockSize;
    VectorDualCosSparseKernel<T><<<gridSize, blockSize>>>(
        d_aReal, a_idx, d_aDualVal,
        real_size,
        d_rReal, d_rIdx, d_rDualVal
    );
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    // Copy results back
    r_real_out.resize(real_size);
    cudaMemcpy(r_real_out.data(), d_rReal,
               bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(&r_dual_out, d_rDualVal,
               sizeof(thrust::complex<T>),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(&r_idx_out, d_rIdx,
               sizeof(int),
               cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_aReal);
    cudaFree(d_aDualVal);
    cudaFree(d_rReal);
    cudaFree(d_rDualVal);
    cudaFree(d_rIdx);
}


// CPU reference (already defined above or in a header):
// template <typename T>
// void VectorDualCosSparseCPU(...);

// ------------------------------------------------------
// 1) a_idx = -1 => no dual
// ------------------------------------------------------
TEST(VectorDualCosSparseTest, NoDualCase)
{
    using T = double;
    int real_size = 4;

    // a_real => [0, 1, 2, 3] for example
    std::vector<thrust::complex<T>> h_aReal {
        {0,0}, {1,0}, {2,0}, {3,0}
    };
    int h_aIdx = -1;
    thrust::complex<T> h_aDualVal(10,20);

    // GPU
    std::vector<thrust::complex<T>> h_rRealAfter;
    int h_rIdxAfter;
    thrust::complex<T> h_rDualAfter;
    RunVectorDualCosSparseKernel(
        h_aReal, h_aIdx, h_aDualVal,
        real_size,
        h_rRealAfter, h_rIdxAfter, h_rDualAfter
    );

    // CPU
    std::vector<thrust::complex<T>> ref_rReal;
    int ref_rIdx;
    thrust::complex<T> ref_rDual;
    VectorDualCosSparseCPU(
        h_aReal, h_aIdx, h_aDualVal,
        real_size,
        ref_rReal, ref_rIdx, ref_rDual
    );

    // Compare
    ASSERT_EQ(real_size, (int)h_rRealAfter.size());
    for (int i = 0; i < real_size; ++i) {
        ExpectComplexNear(ref_rReal[i], h_rRealAfter[i]);
    }
    EXPECT_EQ(ref_rIdx, h_rIdxAfter);
    ExpectComplexNear(ref_rDual, h_rDualAfter);
}

// ------------------------------------------------------
// 2) a_idx >= 0 => valid dual
// ------------------------------------------------------
TEST(VectorDualCosSparseTest, ValidDualCase)
{
    using T = double;
    int real_size = 5;

    // a_real => e.g. [0, 1, 2, 3, 4]
    std::vector<thrust::complex<T>> h_aReal {
        {0,0}, {1,0}, {2,0}, {3,0}, {4,0}
    };
    int h_aIdx = 2;
    thrust::complex<T> h_aDualVal(10,20);

    // GPU
    std::vector<thrust::complex<T>> h_rRealAfter;
    int h_rIdxAfter;
    thrust::complex<T> h_rDualAfter;
    RunVectorDualCosSparseKernel(
        h_aReal, h_aIdx, h_aDualVal,
        real_size,
        h_rRealAfter, h_rIdxAfter, h_rDualAfter
    );

    // CPU
    std::vector<thrust::complex<T>> ref_rReal;
    int ref_rIdx;
    thrust::complex<T> ref_rDual;
    VectorDualCosSparseCPU(
        h_aReal, h_aIdx, h_aDualVal,
        real_size,
        ref_rReal, ref_rIdx, ref_rDual
    );

    // Compare
    ASSERT_EQ(real_size, (int)h_rRealAfter.size());
    for (int i = 0; i < real_size; ++i) {
        ExpectComplexNear(ref_rReal[i], h_rRealAfter[i]);
    }
    EXPECT_EQ(ref_rIdx, h_rIdxAfter);
    ExpectComplexNear(ref_rDual, h_rDualAfter);
}

// ------------------------------------------------------
// 3) Another valid dual, but with complex a_real
// ------------------------------------------------------
TEST(VectorDualCosSparseTest, ComplexRealCase)
{
    using T = double;
    int real_size = 3;

    // a_real => e.g. [(0,1), (1,1), (2,-1)]
    // Just to show it can handle complex inputs
    std::vector<thrust::complex<T>> h_aReal {
        {0,1}, {1,1}, {2,-1}
    };
    int h_aIdx = 1;
    thrust::complex<T> h_aDualVal(30, -40);

    // GPU
    std::vector<thrust::complex<T>> h_rRealAfter;
    int h_rIdxAfter;
    thrust::complex<T> h_rDualAfter;
    RunVectorDualCosSparseKernel(
        h_aReal, h_aIdx, h_aDualVal,
        real_size,
        h_rRealAfter, h_rIdxAfter, h_rDualAfter
    );

    // CPU
    std::vector<thrust::complex<T>> ref_rReal;
    int ref_rIdx;
    thrust::complex<T> ref_rDual;
    VectorDualCosSparseCPU(
        h_aReal, h_aIdx, h_aDualVal,
        real_size,
        ref_rReal, ref_rIdx, ref_rDual
    );

    // Compare
    ASSERT_EQ(real_size, (int)h_rRealAfter.size());
    for (int i = 0; i < real_size; ++i) {
        ExpectComplexNear(ref_rReal[i], h_rRealAfter[i]);
    }
    EXPECT_EQ(ref_rIdx, h_rIdxAfter);
    ExpectComplexNear(ref_rDual, h_rDualAfter);
}


#include <thrust/complex.h>
#include <vector>

/**
 * VectorDualSinSparseCPU
 *
 * CPU reference for a "sparse" sin operation:
 *   - r_real[i] = sin(a_real[i]) for each i
 *   - if a_idx < 0 => r_idx=-1, r_dual=0
 *   - else => r_idx=a_idx, r_dual = sum_j [ cos(a_real[j]) * a_dual_val ]
 */
template <typename T>
void VectorDualSinSparseCPU(
    const std::vector<thrust::complex<T>>& a_real,
    int                                    a_idx,
    const thrust::complex<T>&              a_dual_val,
    int                                    real_size,

    // outputs
    std::vector<thrust::complex<T>>&       r_real,  // size = real_size
    int&                                   r_idx,
    thrust::complex<T>&                    r_dual
)
{
    // 1) Real part
    r_real.resize(real_size);
    for (int i = 0; i < real_size; ++i) {
        r_real[i] = thrust::sin(a_real[i]);
    }

    // 2) Dual part
    if (a_idx < 0) {
        r_idx  = -1;
        r_dual = thrust::complex<T>(0,0);
    } else {
        r_idx = a_idx;
        thrust::complex<T> sumVal(0,0);
        for (int j = 0; j < real_size; ++j) {
            sumVal += thrust::cos(a_real[j]) * a_dual_val;
        }
        r_dual = sumVal;
    }
}



// Forward-declare or include your device code:
//
// template <typename T>
// __global__ void VectorDualSinSparseKernel(...);
//
// template <typename T>
// void VectorDualSinSparseCPU(...);


// ------------------------------------------------------
// Helper to run the kernel
// ------------------------------------------------------
template <typename T>
void RunVectorDualSinSparseKernel(
    // input
    const std::vector<thrust::complex<T>>& a_real,
    int                       a_idx,
    const thrust::complex<T>& a_dual_val,
    int                       real_size,

    // outputs
    std::vector<thrust::complex<T>>& r_real_out,
    int& r_idx_out,
    thrust::complex<T>& r_dual_out
)
{
    thrust::complex<T>* d_aReal     = nullptr;
    thrust::complex<T>* d_aDualVal  = nullptr;
    thrust::complex<T>* d_rReal     = nullptr;
    thrust::complex<T>* d_rDualVal  = nullptr;
    int*                d_rIdx      = nullptr;

    size_t bytes = real_size*sizeof(thrust::complex<T>);
    cudaMalloc(&d_aReal,    bytes);
    cudaMalloc(&d_aDualVal, sizeof(thrust::complex<T>));
    cudaMalloc(&d_rReal,    bytes);
    cudaMalloc(&d_rDualVal, sizeof(thrust::complex<T>));
    cudaMalloc(&d_rIdx,     sizeof(int));

    cudaMemcpy(d_aReal, a_real.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_aDualVal, &a_dual_val,
               sizeof(thrust::complex<T>),
               cudaMemcpyHostToDevice);

    // Launch with single or few blocks, ensuring coverage of real_size
    dim3 block(64);
    dim3 grid((real_size + block.x -1)/block.x);
    VectorDualSinSparseKernel<T><<<grid, block>>>(
        d_aReal, a_idx, d_aDualVal,
        real_size,
        d_rReal, d_rIdx, d_rDualVal
    );
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    // Copy results back
    r_real_out.resize(real_size);
    cudaMemcpy(r_real_out.data(), d_rReal,
               bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(&r_dual_out, d_rDualVal,
               sizeof(thrust::complex<T>),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(&r_idx_out, d_rIdx,
               sizeof(int),
               cudaMemcpyDeviceToHost);

    cudaFree(d_aReal);
    cudaFree(d_aDualVal);
    cudaFree(d_rReal);
    cudaFree(d_rDualVal);
    cudaFree(d_rIdx);
}


// CPU reference function is declared above
// template <typename T> void VectorDualSinSparseCPU(...);

// ------------------------------------------------------
// 1) a_idx = -1 => no dual
// ------------------------------------------------------
TEST(VectorDualSinSparseTest, NoDualCase)
{
    using T = double;
    int real_size = 4;
    // a_real => [0, 1, 2, 3]
    std::vector<thrust::complex<T>> h_aReal {
        {0,0}, {1,0}, {2,0}, {3,0}
    };
    int h_aIdx = -1; // no dual
    thrust::complex<T> h_aDualVal(10,20);

    // GPU
    std::vector<thrust::complex<T>> h_rRealAfter;
    int h_rIdxAfter;
    thrust::complex<T> h_rDualAfter;
    RunVectorDualSinSparseKernel(
        h_aReal, h_aIdx, h_aDualVal,
        real_size,
        h_rRealAfter, h_rIdxAfter, h_rDualAfter
    );

    // CPU
    std::vector<thrust::complex<T>> ref_rReal;
    int ref_rIdx;
    thrust::complex<T> ref_rDual;
    VectorDualSinSparseCPU(
        h_aReal, h_aIdx, h_aDualVal,
        real_size,
        ref_rReal, ref_rIdx, ref_rDual
    );

    // Compare
    ASSERT_EQ(real_size, (int)h_rRealAfter.size());
    for (int i = 0; i < real_size; ++i) {
        ExpectComplexNear(ref_rReal[i], h_rRealAfter[i]);
    }
    EXPECT_EQ(ref_rIdx, h_rIdxAfter);
    ExpectComplexNear(ref_rDual, h_rDualAfter);
}

// ------------------------------------------------------
// 2) a_idx >= 0 => valid dual, simple real input
// ------------------------------------------------------
TEST(VectorDualSinSparseTest, ValidDualCase)
{
    using T = double;
    int real_size = 5;
    // a_real => [0, 1, 2, 3, 4]
    std::vector<thrust::complex<T>> h_aReal {
        {0,0}, {1,0}, {2,0}, {3,0}, {4,0}
    };
    int h_aIdx = 2;
    thrust::complex<T> h_aDualVal(10,20);

    // GPU
    std::vector<thrust::complex<T>> h_rRealAfter;
    int h_rIdxAfter;
    thrust::complex<T> h_rDualAfter;
    RunVectorDualSinSparseKernel(
        h_aReal, h_aIdx, h_aDualVal,
        real_size,
        h_rRealAfter, h_rIdxAfter, h_rDualAfter
    );

    // CPU
    std::vector<thrust::complex<T>> ref_rReal;
    int ref_rIdx;
    thrust::complex<T> ref_rDual;
    VectorDualSinSparseCPU(
        h_aReal, h_aIdx, h_aDualVal,
        real_size,
        ref_rReal, ref_rIdx, ref_rDual
    );

    // Compare
    ASSERT_EQ(real_size, (int)h_rRealAfter.size());
    for (int i = 0; i < real_size; ++i) {
        ExpectComplexNear(ref_rReal[i], h_rRealAfter[i]);
    }
    EXPECT_EQ(ref_rIdx, h_rIdxAfter);
    ExpectComplexNear(ref_rDual, h_rDualAfter);
}

// ------------------------------------------------------
// 3) Another valid dual, but with complex input
// ------------------------------------------------------
TEST(VectorDualSinSparseTest, ComplexRealCase)
{
    using T = double;
    int real_size = 3;
    // a_real => [(0,1), (1,1), (2,-1)]
    // just to ensure complex sin/cos can be handled
    std::vector<thrust::complex<T>> h_aReal {
        {0,1}, {1,1}, {2,-1}
    };
    int h_aIdx = 1;
    thrust::complex<T> h_aDualVal(30, -40);

    // GPU
    std::vector<thrust::complex<T>> h_rRealAfter;
    int h_rIdxAfter;
    thrust::complex<T> h_rDualAfter;
    RunVectorDualSinSparseKernel(
        h_aReal, h_aIdx, h_aDualVal,
        real_size,
        h_rRealAfter, h_rIdxAfter, h_rDualAfter
    );

    // CPU
    std::vector<thrust::complex<T>> ref_rReal;
    int ref_rIdx;
    thrust::complex<T> ref_rDual;
    VectorDualSinSparseCPU(
        h_aReal, h_aIdx, h_aDualVal,
        real_size,
        ref_rReal, ref_rIdx, ref_rDual
    );

    // Compare
    ASSERT_EQ(real_size, (int)ref_rReal.size());
    for (int i = 0; i < real_size; ++i) {
        ExpectComplexNear(ref_rReal[i], h_rRealAfter[i]);
    }
    EXPECT_EQ(ref_rIdx, h_rIdxAfter);
    ExpectComplexNear(ref_rDual, h_rDualAfter);
}


#include <thrust/complex.h>
#include <vector>

/**
 * VectorDualTanSparseCPU
 *
 * CPU reference for a "sparse" tan operation:
 *   - r_real[i] = tan(a_real[i])
 *   - if a_idx < 0 => r_idx=-1, r_dual=0
 *   - else => r_idx=a_idx,
 *             r_dual = sum_{i=0..real_size-1}[ a_dual_val / cos^2(a_real[i]) ]
 */
template <typename T>
void VectorDualTanSparseCPU(
    // Input sparse vector
    const std::vector<thrust::complex<T>>& a_real,
    int                                    a_idx,
    const thrust::complex<T>&              a_dual_val,
    int                                    real_size,

    // Outputs
    std::vector<thrust::complex<T>>&       r_real,  // length real_size
    int&                                   r_idx,
    thrust::complex<T>&                    r_dual
)
{
    // 1) Real part
    r_real.resize(real_size);
    for (int i = 0; i < real_size; ++i) {
        r_real[i] = thrust::tan(a_real[i]);
    }

    // 2) Dual part
    if (a_idx < 0) {
        r_idx  = -1;
        r_dual = thrust::complex<T>(0,0);
    } else {
        r_idx = a_idx;
        thrust::complex<T> sumVal(0,0);
        for (int i = 0; i < real_size; ++i) {
            thrust::complex<T> cos_val = thrust::cos(a_real[i]);
            thrust::complex<T> denom   = cos_val * cos_val; // cos^2(x)
            sumVal += a_dual_val / denom;
        }
        r_dual = sumVal;
    }
}



/**
 * Helper to run the kernel, copy data in/out.
 */
template <typename T>
void RunVectorDualTanSparseKernel(
    // Input
    const std::vector<thrust::complex<T>>& a_real,
    int                       a_idx,
    const thrust::complex<T>& a_dual_val,
    int                       real_size,

    // Outputs
    std::vector<thrust::complex<T>>& r_real_out,
    int& r_idx_out,
    thrust::complex<T>& r_dual_out
)
{
    // Device pointers
    thrust::complex<T>* d_aReal     = nullptr;
    thrust::complex<T>* d_aDualVal  = nullptr;
    thrust::complex<T>* d_rReal     = nullptr;
    thrust::complex<T>* d_rDualVal  = nullptr;
    int*                d_rIdx      = nullptr;

    size_t bytes = real_size * sizeof(thrust::complex<T>);
    cudaMalloc(&d_aReal,    bytes);
    cudaMalloc(&d_aDualVal, sizeof(thrust::complex<T>));
    cudaMalloc(&d_rReal,    bytes);
    cudaMalloc(&d_rDualVal, sizeof(thrust::complex<T>));
    cudaMalloc(&d_rIdx,     sizeof(int));

    // Copy host->device
    cudaMemcpy(d_aReal, a_real.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_aDualVal, &a_dual_val,
               sizeof(thrust::complex<T>),
               cudaMemcpyHostToDevice);

    // Launch
    dim3 block(64);
    dim3 grid((real_size + block.x -1)/block.x);
    VectorDualTanSparseKernel<T><<<grid, block>>>(
        d_aReal,
        a_idx,
        d_aDualVal,
        real_size,
        d_rReal,
        d_rIdx,
        d_rDualVal
    );
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    // Copy device->host
    r_real_out.resize(real_size);
    cudaMemcpy(r_real_out.data(), d_rReal,
               bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(&r_dual_out, d_rDualVal,
               sizeof(thrust::complex<T>),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(&r_idx_out, d_rIdx,
               sizeof(int),
               cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_aReal);
    cudaFree(d_aDualVal);
    cudaFree(d_rReal);
    cudaFree(d_rDualVal);
    cudaFree(d_rIdx);
}


// CPU reference function: VectorDualTanSparseCPU(...)

// ---------------------------------------------------------
// 1) No Dual => a_idx=-1
// ---------------------------------------------------------
TEST(VectorDualTanSparseTest, NoDualCase)
{
    using T = double;
    int real_size = 4;
    // a_real => [0, 1, 2, 3]
    std::vector<thrust::complex<T>> h_aReal {
        {0,0}, {1,0}, {2,0}, {3,0}
    };
    int h_aIdx = -1; // no dual
    thrust::complex<T> h_aDualVal(10, 20);

    // GPU
    std::vector<thrust::complex<T>> h_rRealAfter;
    int h_rIdxAfter;
    thrust::complex<T> h_rDualAfter;
    RunVectorDualTanSparseKernel(
        h_aReal, h_aIdx, h_aDualVal,
        real_size,
        h_rRealAfter,
        h_rIdxAfter,
        h_rDualAfter
    );

    // CPU
    std::vector<thrust::complex<T>> ref_rReal;
    int ref_rIdx;
    thrust::complex<T> ref_rDual;
    VectorDualTanSparseCPU(
        h_aReal, h_aIdx, h_aDualVal,
        real_size,
        ref_rReal, ref_rIdx, ref_rDual
    );

    // Compare real
    ASSERT_EQ(real_size, (int)h_rRealAfter.size());
    for (int i = 0; i < real_size; ++i) {
        ExpectComplexNear(ref_rReal[i], h_rRealAfter[i]);
    }
    // Compare dual
    EXPECT_EQ(ref_rIdx, h_rIdxAfter);
    ExpectComplexNear(ref_rDual, h_rDualAfter);
}

// ---------------------------------------------------------
// 2) Valid Dual => purely real input
// ---------------------------------------------------------
TEST(VectorDualTanSparseTest, ValidDualCase)
{
    using T = double;
    int real_size = 5;
    // a_real => [0, 0.5, 1, 1.5, 2]
    std::vector<thrust::complex<T>> h_aReal {
        {0,0}, {0.5,0}, {1,0}, {1.5,0}, {2,0}
    };
    int h_aIdx = 2; // single valid dual index
    thrust::complex<T> h_aDualVal(10, 20);

    // GPU
    std::vector<thrust::complex<T>> h_rRealAfter;
    int h_rIdxAfter;
    thrust::complex<T> h_rDualAfter;
    RunVectorDualTanSparseKernel(
        h_aReal, h_aIdx, h_aDualVal,
        real_size,
        h_rRealAfter,
        h_rIdxAfter,
        h_rDualAfter
    );

    // CPU
    std::vector<thrust::complex<T>> ref_rReal;
    int ref_rIdx;
    thrust::complex<T> ref_rDual;
    VectorDualTanSparseCPU(
        h_aReal, h_aIdx, h_aDualVal,
        real_size,
        ref_rReal, ref_rIdx, ref_rDual
    );

    // Compare
    ASSERT_EQ(real_size, (int)h_rRealAfter.size());
    for (int i = 0; i < real_size; ++i) {
        ExpectComplexNear(ref_rReal[i], h_rRealAfter[i]);
    }
    EXPECT_EQ(ref_rIdx, h_rIdxAfter);
    ExpectComplexNear(ref_rDual, h_rDualAfter);
}

// ---------------------------------------------------------
// 3) Another valid dual with complex input
// ---------------------------------------------------------
TEST(VectorDualTanSparseTest, ComplexInputCase)
{
    using T = double;
    int real_size = 3;
    // a_real => e.g. [(0,1), (1,1), (1.5, -0.5)]
    std::vector<thrust::complex<T>> h_aReal {
        {0,1}, {1,1}, {1.5,-0.5}
    };
    int h_aIdx = 1;
    thrust::complex<T> h_aDualVal(50, -40);

    // GPU
    std::vector<thrust::complex<T>> h_rRealAfter;
    int h_rIdxAfter;
    thrust::complex<T> h_rDualAfter;
    RunVectorDualTanSparseKernel(
        h_aReal, h_aIdx, h_aDualVal,
        real_size,
        h_rRealAfter,
        h_rIdxAfter,
        h_rDualAfter
    );

    // CPU
    std::vector<thrust::complex<T>> ref_rReal;
    int ref_rIdx;
    thrust::complex<T> ref_rDual;
    VectorDualTanSparseCPU(
        h_aReal, h_aIdx, h_aDualVal,
        real_size,
        ref_rReal, ref_rIdx, ref_rDual
    );

    // Compare
    ASSERT_EQ(real_size, (int)h_rRealAfter.size());
    for (int i = 0; i < real_size; ++i) {
        ExpectComplexNear(ref_rReal[i], h_rRealAfter[i]);
    }
    EXPECT_EQ(ref_rIdx, h_rIdxAfter);
    ExpectComplexNear(ref_rDual, h_rDualAfter);
}

#include <thrust/complex.h>
#include <vector>

/**
 * VectorHyperDualIndexGetSparseCPU
 *
 * CPU reference for "IndexGet" on a sparse hyper-dual vector:
 *
 * Inputs:
 *   - a_real: length real_size
 *   - a_idx:  single dual/hyper index in [-1..real_size)
 *   - a_dual_value:  the single dual complex
 *   - a_hyper_value: the single hyper complex
 *   - real_size
 *   - [start, end) for the slice
 *
 * Outputs:
 *   - out_real: length = (end - start)
 *   - out_idx:  the (shifted) dual/hyper index or -1
 *   - out_dual_value / out_hyper_value
 */
template <typename T>
void VectorHyperDualIndexGetSparseCPU(
    // Input
    const std::vector<thrust::complex<T>>& a_real,
    int                                    a_idx,
    const thrust::complex<T>&              a_dual_value,
    const thrust::complex<T>&              a_hyper_value,
    int                                    real_size,
    int                                    start,
    int                                    end,

    // Output
    std::vector<thrust::complex<T>>&       out_real,        // length = (end - start)
    int&                                   out_idx,
    thrust::complex<T>&                    out_dual_value,
    thrust::complex<T>&                    out_hyper_value
)
{
    // 1) Copy real part in [start..end) to out_real
    int sub_len = end - start;
    out_real.resize(sub_len);
    for (int i = 0; i < sub_len; ++i) {
        out_real[i] = a_real[start + i];
    }

    // 2) Dual/hyper logic
    //    If a_idx not in [start, end), => -1, zero
    if (a_idx < start || a_idx >= end || a_idx < 0) {
        out_idx         = -1;
        out_dual_value  = thrust::complex<T>(0,0);
        out_hyper_value = thrust::complex<T>(0,0);
    } else {
        // shift the index by 'start'
        out_idx = a_idx - start;
        // copy the single dual/hyper values
        out_dual_value  = a_dual_value;
        out_hyper_value = a_hyper_value;
    }
}


// Helper to run the kernel
template <typename T>
void RunVectorHyperDualIndexGetSparseKernel(
    // input data
    const std::vector<thrust::complex<T>>& a_real,
    int                       a_idx,
    const thrust::complex<T>& a_dual_val,
    const thrust::complex<T>& a_hyper_val,
    int                       real_size,
    int                       start,
    int                       end,

    // outputs
    std::vector<thrust::complex<T>>& out_real,
    int& out_idx,
    thrust::complex<T>& out_dual_val,
    thrust::complex<T>& out_hyper_val
)
{
    // Device pointers
    thrust::complex<T>* d_aReal      = nullptr;
    thrust::complex<T>* d_aDualVal   = nullptr;
    thrust::complex<T>* d_aHyperVal  = nullptr;

    thrust::complex<T>* d_outReal    = nullptr;
    thrust::complex<T>* d_outDualVal = nullptr;
    thrust::complex<T>* d_outHyperVal= nullptr;
    int*                d_outIdx     = nullptr;

    int sub_len = end - start;
    size_t bytesIn   = real_size*sizeof(thrust::complex<T>);
    size_t bytesOut  = sub_len*sizeof(thrust::complex<T>);

    // Allocate
    cudaMalloc(&d_aReal,     bytesIn);
    cudaMalloc(&d_aDualVal,  sizeof(thrust::complex<T>));
    cudaMalloc(&d_aHyperVal, sizeof(thrust::complex<T>));

    cudaMalloc(&d_outReal,    bytesOut);
    cudaMalloc(&d_outDualVal, sizeof(thrust::complex<T>));
    cudaMalloc(&d_outHyperVal,sizeof(thrust::complex<T>));
    cudaMalloc(&d_outIdx,     sizeof(int));

    // Copy host->device
    cudaMemcpy(d_aReal, a_real.data(), bytesIn, cudaMemcpyHostToDevice);
    cudaMemcpy(d_aDualVal,  &a_dual_val,  sizeof(thrust::complex<T>), cudaMemcpyHostToDevice);
    cudaMemcpy(d_aHyperVal, &a_hyper_val, sizeof(thrust::complex<T>), cudaMemcpyHostToDevice);

    // Launch
    dim3 block(64);
    dim3 grid((sub_len + block.x -1)/block.x);
    VectorHyperDualIndexGetSparseKernel<T><<<grid, block>>>(
        d_aReal, a_idx, d_aDualVal, d_aHyperVal,
        real_size, start, end,
        d_outReal, d_outIdx, d_outDualVal, d_outHyperVal
    );
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    // Copy results back
    out_real.resize(sub_len);
    cudaMemcpy(out_real.data(), d_outReal, bytesOut, cudaMemcpyDeviceToHost);
    cudaMemcpy(&out_dual_val,  d_outDualVal,  sizeof(thrust::complex<T>), cudaMemcpyDeviceToHost);
    cudaMemcpy(&out_hyper_val, d_outHyperVal, sizeof(thrust::complex<T>), cudaMemcpyDeviceToHost);
    cudaMemcpy(&out_idx,       d_outIdx,      sizeof(int),               cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_aReal);
    cudaFree(d_aDualVal);
    cudaFree(d_aHyperVal);
    cudaFree(d_outReal);
    cudaFree(d_outDualVal);
    cudaFree(d_outHyperVal);
    cudaFree(d_outIdx);
}


// CPU reference: VectorHyperDualIndexGetSparseCPU(...)

// -----------------------------------------------------
// 1) a_idx = -1 => no dual/hyper
// -----------------------------------------------------
TEST(VectorHyperDualIndexGetSparseTest, NoDualCase)
{
    using T = double;
    int real_size = 6;
    std::vector<thrust::complex<T>> h_aReal {
        {1,0}, {2,0}, {3,0}, {4,0}, {5,0}, {6,0}
    };
    int h_aIdx = -1;
    thrust::complex<T> h_aDualVal(10,20);
    thrust::complex<T> h_aHyperVal(30,40);

    int start=2, end=5; // subrange => [3,4,5]
    // GPU
    std::vector<thrust::complex<T>> h_outReal;
    int h_outIdx;
    thrust::complex<T> h_outDualVal, h_outHyperVal;

    RunVectorHyperDualIndexGetSparseKernel(
        h_aReal, h_aIdx,
        h_aDualVal, h_aHyperVal,
        real_size,
        start, end,
        h_outReal, h_outIdx, h_outDualVal, h_outHyperVal
    );

    // CPU
    std::vector<thrust::complex<T>> ref_outReal;
    int ref_outIdx;
    thrust::complex<T> ref_outDualVal, ref_outHyperVal;
    VectorHyperDualIndexGetSparseCPU(
        h_aReal, h_aIdx,
        h_aDualVal, h_aHyperVal,
        real_size,
        start, end,
        ref_outReal, ref_outIdx, ref_outDualVal, ref_outHyperVal
    );

    // Compare real
    ASSERT_EQ(ref_outReal.size(), h_outReal.size());
    for (int i=0; i<(int)ref_outReal.size(); ++i) {
        ExpectComplexNear(ref_outReal[i], h_outReal[i]);
    }
    // Compare dual/hyper
    EXPECT_EQ(ref_outIdx, h_outIdx);
    ExpectComplexNear(ref_outDualVal, h_outDualVal);
    ExpectComplexNear(ref_outHyperVal,h_outHyperVal);
}

// -----------------------------------------------------
// 2) a_idx < start => out of range
// -----------------------------------------------------
TEST(VectorHyperDualIndexGetSparseTest, IdxLessThanStart)
{
    using T = double;
    int real_size = 6;
    std::vector<thrust::complex<T>> h_aReal {
        {1,1}, {2,2}, {3,3}, {4,4}, {5,5}, {6,6}
    };
    // Suppose a_idx=1, but start=2 => out of range
    int h_aIdx = 1;
    thrust::complex<T> h_aDualVal(10,20);
    thrust::complex<T> h_aHyperVal(30,40);

    int start=2, end=5; // subrange => [ (3,3), (4,4), (5,5) ]
    // GPU
    std::vector<thrust::complex<T>> h_outReal;
    int h_outIdx;
    thrust::complex<T> h_outDualVal, h_outHyperVal;

    RunVectorHyperDualIndexGetSparseKernel(
        h_aReal, h_aIdx,
        h_aDualVal, h_aHyperVal,
        real_size,
        start, end,
        h_outReal, h_outIdx, h_outDualVal, h_outHyperVal
    );

    // CPU
    std::vector<thrust::complex<T>> ref_outReal;
    int ref_outIdx;
    thrust::complex<T> ref_outDualVal, ref_outHyperVal;
    VectorHyperDualIndexGetSparseCPU(
        h_aReal, h_aIdx,
        h_aDualVal, h_aHyperVal,
        real_size,
        start, end,
        ref_outReal, ref_outIdx, ref_outDualVal, ref_outHyperVal
    );

    // Compare real
    ASSERT_EQ(ref_outReal.size(), h_outReal.size());
    for (int i=0; i<(int)ref_outReal.size(); ++i) {
        ExpectComplexNear(ref_outReal[i], h_outReal[i]);
    }
    // Compare dual/hyper
    EXPECT_EQ(ref_outIdx, h_outIdx);
    ExpectComplexNear(ref_outDualVal, h_outDualVal);
    ExpectComplexNear(ref_outHyperVal,h_outHyperVal);
}

// -----------------------------------------------------
// 3) a_idx >= end => out of range
// -----------------------------------------------------
TEST(VectorHyperDualIndexGetSparseTest, IdxGreaterOrEqualEnd)
{
    using T = double;
    int real_size = 6;
    std::vector<thrust::complex<T>> h_aReal {
        {1,1}, {2,2}, {3,3}, {4,4}, {5,5}, {6,6}
    };
    // Suppose a_idx=5, but subrange is [2..5) => i in [2,3,4], so idx=5 is out of range
    int h_aIdx = 5;
    thrust::complex<T> h_aDualVal(100,200);
    thrust::complex<T> h_aHyperVal(300,400);

    int start=2, end=5; // [ (3,3), (4,4), (5,5) ]
    // GPU
    std::vector<thrust::complex<T>> h_outReal;
    int h_outIdx;
    thrust::complex<T> h_outDualVal, h_outHyperVal;

    RunVectorHyperDualIndexGetSparseKernel(
        h_aReal, h_aIdx,
        h_aDualVal, h_aHyperVal,
        real_size,
        start, end,
        h_outReal, h_outIdx, h_outDualVal, h_outHyperVal
    );

    // CPU
    std::vector<thrust::complex<T>> ref_outReal;
    int ref_outIdx;
    thrust::complex<T> ref_outDualVal, ref_outHyperVal;
    VectorHyperDualIndexGetSparseCPU(
        h_aReal, h_aIdx,
        h_aDualVal, h_aHyperVal,
        real_size,
        start, end,
        ref_outReal, ref_outIdx, ref_outDualVal, ref_outHyperVal
    );

    // Compare
    ASSERT_EQ(ref_outReal.size(), h_outReal.size());
    for (int i=0; i<(int)ref_outReal.size(); ++i) {
        ExpectComplexNear(ref_outReal[i], h_outReal[i]);
    }
    EXPECT_EQ(ref_outIdx, h_outIdx);
    ExpectComplexNear(ref_outDualVal, h_outDualVal);
    ExpectComplexNear(ref_outHyperVal,h_outHyperVal);
}

// -----------------------------------------------------
// 4) a_idx in [start..end) => valid => shift index
// -----------------------------------------------------
TEST(VectorHyperDualIndexGetSparseTest, ValidIndexInRange)
{
    using T = double;
    int real_size = 6;
    std::vector<thrust::complex<T>> h_aReal {
        {1,1}, {2,2}, {3,3}, {4,4}, {5,5}, {6,6}
    };
    // Suppose a_idx=3 => in range [2..5)
    int h_aIdx = 3;
    thrust::complex<T> h_aDualVal(10,20);
    thrust::complex<T> h_aHyperVal(30,40);

    int start=2, end=5; // subrange => i in [2,3,4] => (3,3), (4,4), (5,5)
    // GPU
    std::vector<thrust::complex<T>> h_outReal;
    int h_outIdx;
    thrust::complex<T> h_outDualVal, h_outHyperVal;

    RunVectorHyperDualIndexGetSparseKernel(
        h_aReal, h_aIdx,
        h_aDualVal, h_aHyperVal,
        real_size,
        start, end,
        h_outReal, h_outIdx, h_outDualVal, h_outHyperVal
    );

    // CPU
    std::vector<thrust::complex<T>> ref_outReal;
    int ref_outIdx;
    thrust::complex<T> ref_outDualVal, ref_outHyperVal;
    VectorHyperDualIndexGetSparseCPU(
        h_aReal, h_aIdx,
        h_aDualVal, h_aHyperVal,
        real_size,
        start, end,
        ref_outReal, ref_outIdx, ref_outDualVal, ref_outHyperVal
    );

    // Compare
    ASSERT_EQ(ref_outReal.size(), h_outReal.size());
    for (int i=0; i<(int)ref_outReal.size(); ++i) {
        ExpectComplexNear(ref_outReal[i], h_outReal[i]);
    }
    EXPECT_EQ(ref_outIdx, h_outIdx);
    ExpectComplexNear(ref_outDualVal, h_outDualVal);
    ExpectComplexNear(ref_outHyperVal,h_outHyperVal);
}

#include <thrust/complex.h>
#include <vector>

/**
 * VectorHyperDualIndexPutSparseCPU
 *
 * CPU reference for a "sparse" hyper-dual index put:
 *   - Copy real array [0..input_real_size) => out_real[start_real..start_real+input_real_size).
 *   - If in_idx in [0..(input_real_size-1)], out_idx = start_real+in_idx; else -1.
 *   - If valid, copy in_dual_val, in_hyper_val; else 0.
 */
template <typename T>
void VectorHyperDualIndexPutSparseCPU(
    // Input
    const std::vector<thrust::complex<T>>& in_real,
    int                                    in_idx,
    const thrust::complex<T>&              in_dual_val,
    const thrust::complex<T>&              in_hyper_val,
    int                                    input_real_size,

    // Range in the output
    int start_real,
    int end_real,

    // Outputs
    std::vector<thrust::complex<T>>&       out_real,     // length >= end_real
    int&                                   out_idx,
    thrust::complex<T>&                    out_dual_val,
    thrust::complex<T>&                    out_hyper_val
)
{
    // 1) Copy real
    int sub_len = end_real - start_real;
    // for safety, assume input_real_size == sub_len
    for (int i = 0; i < input_real_size; ++i) {
        out_real[start_real + i] = in_real[i];
    }

    // 2) Dual/hyper
    if (in_idx < 0 || in_idx >= input_real_size) {
        out_idx = -1;
        out_dual_val  = thrust::complex<T>(0, 0);
        out_hyper_val = thrust::complex<T>(0, 0);
    } else {
        out_idx = start_real + in_idx;
        out_dual_val  = in_dual_val;
        out_hyper_val = in_hyper_val;
    }
}


// Helper to run the kernel
template <typename T>
void RunVectorHyperDualIndexPutSparseKernel(
    // Input
    const std::vector<thrust::complex<T>>& in_real,
    int                       in_idx,
    const thrust::complex<T>& in_dual_val,
    const thrust::complex<T>& in_hyper_val,
    int                       input_real_size,

    // subrange
    int start_real,
    int end_real,

    // outputs
    std::vector<thrust::complex<T>>& out_real,
    int& out_idx,
    thrust::complex<T>& out_dual_val,
    thrust::complex<T>& out_hyper_val
)
{
    thrust::complex<T>* d_inReal      = nullptr;
    thrust::complex<T>* d_inDualVal   = nullptr;
    thrust::complex<T>* d_inHyperVal  = nullptr;

    thrust::complex<T>* d_outReal     = nullptr;
    thrust::complex<T>* d_outDualVal  = nullptr;
    thrust::complex<T>* d_outHyperVal= nullptr;
    int*                d_outIdx      = nullptr;

    size_t bytesIn  = input_real_size*sizeof(thrust::complex<T>);
    // The out_real must be at least end_real in length
    int out_size = end_real; // assume we only need up to end_real
    size_t bytesOut = out_size*sizeof(thrust::complex<T>);

    // 1) Allocate
    cudaMalloc(&d_inReal,     bytesIn);
    cudaMalloc(&d_inDualVal,  sizeof(thrust::complex<T>));
    cudaMalloc(&d_inHyperVal, sizeof(thrust::complex<T>));

    cudaMalloc(&d_outReal,    bytesOut);
    cudaMalloc(&d_outDualVal, sizeof(thrust::complex<T>));
    cudaMalloc(&d_outHyperVal,sizeof(thrust::complex<T>));
    cudaMalloc(&d_outIdx,     sizeof(int));

    // 2) Copy host->device
    cudaMemcpy(d_inReal, in_real.data(), bytesIn, cudaMemcpyHostToDevice);
    cudaMemcpy(d_inDualVal,  &in_dual_val,  sizeof(thrust::complex<T>), cudaMemcpyHostToDevice);
    cudaMemcpy(d_inHyperVal, &in_hyper_val, sizeof(thrust::complex<T>), cudaMemcpyHostToDevice);

    // Optionally init the out arrays with sentinel
    std::vector<thrust::complex<T>> h_outInit(out_size, { -999, -999 });
    cudaMemcpy(d_outReal, h_outInit.data(), bytesOut, cudaMemcpyHostToDevice);

    thrust::complex<T> sentinelDual(-888,-888), sentinelHyper(-777,-777);
    cudaMemcpy(d_outDualVal,  &sentinelDual,  sizeof(thrust::complex<T>), cudaMemcpyHostToDevice);
    cudaMemcpy(d_outHyperVal, &sentinelHyper, sizeof(thrust::complex<T>), cudaMemcpyHostToDevice);
    int sentinelIdx = -666;
    cudaMemcpy(d_outIdx, &sentinelIdx, sizeof(int), cudaMemcpyHostToDevice);

    // 3) Launch kernel
    dim3 block(64);
    dim3 grid( (input_real_size + block.x -1)/block.x );
    VectorHyperDualIndexPutSparseKernel<T><<<grid, block>>>(
        d_inReal, in_idx, d_inDualVal, d_inHyperVal,
        input_real_size,
        start_real, end_real,
        d_outReal, d_outIdx, d_outDualVal, d_outHyperVal
    );
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    // 4) Copy results back
    out_real.resize(out_size);
    cudaMemcpy(out_real.data(), d_outReal, bytesOut, cudaMemcpyDeviceToHost);
    cudaMemcpy(&out_dual_val,  d_outDualVal,  sizeof(thrust::complex<T>), cudaMemcpyDeviceToHost);
    cudaMemcpy(&out_hyper_val, d_outHyperVal, sizeof(thrust::complex<T>), cudaMemcpyDeviceToHost);
    cudaMemcpy(&out_idx,       d_outIdx,      sizeof(int),               cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_inReal);
    cudaFree(d_inDualVal);
    cudaFree(d_inHyperVal);
    cudaFree(d_outReal);
    cudaFree(d_outDualVal);
    cudaFree(d_outHyperVal);
    cudaFree(d_outIdx);
}


// CPU reference: VectorHyperDualIndexPutSparseCPU(...)

// ---------------------------------------------------------
// 1) NoDualCase => in_idx = -1 => no dual/hyper
// ---------------------------------------------------------
TEST(VectorHyperDualIndexPutSparseTest, NoDualCase)
{
    using T = double;
    int input_real_size = 3;
    std::vector<thrust::complex<T>> h_inReal {
        {10,0}, {11,0}, {12,0}
    };
    int h_inIdx = -1;
    thrust::complex<T> h_inDualVal(100,200);
    thrust::complex<T> h_inHyperVal(300,400);

    int start_real=2, end_real=5; 
    int out_size  = 5; // e.g., we want at least 5
    // We'll see real placed at out_real[2..4] => [10,11,12]

    // GPU
    std::vector<thrust::complex<T>> h_outReal;
    int h_outIdx;
    thrust::complex<T> h_outDualVal, h_outHyperVal;

    RunVectorHyperDualIndexPutSparseKernel(
        h_inReal, h_inIdx, h_inDualVal, h_inHyperVal,
        input_real_size,
        start_real, end_real,
        h_outReal, h_outIdx, h_outDualVal, h_outHyperVal
    );

    // CPU
    std::vector<thrust::complex<T>> ref_outReal(out_size, { -999, -999 });
    int ref_outIdx;
    thrust::complex<T> ref_outDualVal, ref_outHyperVal;
    VectorHyperDualIndexPutSparseCPU(
        h_inReal, h_inIdx, h_inDualVal, h_inHyperVal,
        input_real_size,
        start_real, end_real,
        ref_outReal, ref_outIdx, ref_outDualVal, ref_outHyperVal
    );

    // Compare real
    ASSERT_EQ(out_size, (int)h_outReal.size());
    for (int i=0; i<out_size; ++i) {
        ExpectComplexNear(ref_outReal[i], h_outReal[i]);
    }
    // Compare dual/hyper
    EXPECT_EQ(ref_outIdx, h_outIdx);
    ExpectComplexNear(ref_outDualVal,  h_outDualVal);
    ExpectComplexNear(ref_outHyperVal, h_outHyperVal);
}

// ---------------------------------------------------------
// 2) NegativeIndex => e.g. in_idx < 0 => no dual/hyper
// ---------------------------------------------------------
TEST(VectorHyperDualIndexPutSparseTest, NegativeIndex)
{
    using T = double;
    int input_real_size = 3;
    std::vector<thrust::complex<T>> h_inReal {
        {10,1}, {11,2}, {12,3}
    };
    int h_inIdx = -5; // negative
    thrust::complex<T> h_inDualVal(100,200);
    thrust::complex<T> h_inHyperVal(300,400);

    int start_real=1, end_real=4; 
    int out_size  = 4; 
    // The real [10,1], [11,2], [12,3] => goes to out_real[1..3]

    // GPU
    std::vector<thrust::complex<T>> h_outReal;
    int h_outIdx;
    thrust::complex<T> h_outDualVal, h_outHyperVal;

    RunVectorHyperDualIndexPutSparseKernel(
        h_inReal, h_inIdx, h_inDualVal, h_inHyperVal,
        input_real_size,
        start_real, end_real,
        h_outReal, h_outIdx, h_outDualVal, h_outHyperVal
    );

    // CPU
    std::vector<thrust::complex<T>> ref_outReal(out_size, { -999, -999 });
    int ref_outIdx;
    thrust::complex<T> ref_outDualVal, ref_outHyperVal;
    VectorHyperDualIndexPutSparseCPU(
        h_inReal, h_inIdx, h_inDualVal, h_inHyperVal,
        input_real_size,
        start_real, end_real,
        ref_outReal, ref_outIdx, ref_outDualVal, ref_outHyperVal
    );

    // Compare
    ASSERT_EQ(out_size, (int)h_outReal.size());
    for (int i=0; i<out_size; ++i) {
        ExpectComplexNear(ref_outReal[i], h_outReal[i]);
    }
    EXPECT_EQ(ref_outIdx, h_outIdx);
    ExpectComplexNear(ref_outDualVal,  h_outDualVal);
    ExpectComplexNear(ref_outHyperVal, h_outHyperVal);
}

// ---------------------------------------------------------
// 3) IdxOutOfRange => e.g. in_idx >= input_real_size
// ---------------------------------------------------------
TEST(VectorHyperDualIndexPutSparseTest, IdxOutOfRange)
{
    using T = double;
    int input_real_size = 3;
    std::vector<thrust::complex<T>> h_inReal {
        {1,1}, {2,2}, {3,3}
    };
    // in_idx=3 => out of range, since valid indices are [0..2]
    int h_inIdx = 3;
    thrust::complex<T> h_inDualVal(10,20);
    thrust::complex<T> h_inHyperVal(30,40);

    int start_real=2, end_real=5;
    int out_size  = 5; 

    // GPU
    std::vector<thrust::complex<T>> h_outReal;
    int h_outIdx;
    thrust::complex<T> h_outDualVal, h_outHyperVal;
    RunVectorHyperDualIndexPutSparseKernel(
        h_inReal, h_inIdx, h_inDualVal, h_inHyperVal,
        input_real_size,
        start_real, end_real,
        h_outReal, h_outIdx, h_outDualVal, h_outHyperVal
    );

    // CPU
    std::vector<thrust::complex<T>> ref_outReal(out_size, { -999, -999 });
    int ref_outIdx;
    thrust::complex<T> ref_outDualVal, ref_outHyperVal;
    VectorHyperDualIndexPutSparseCPU(
        h_inReal, h_inIdx, h_inDualVal, h_inHyperVal,
        input_real_size,
        start_real, end_real,
        ref_outReal, ref_outIdx, ref_outDualVal, ref_outHyperVal
    );

    // Compare
    ASSERT_EQ(out_size, (int)h_outReal.size());
    for (int i=0; i<out_size; ++i) {
        ExpectComplexNear(ref_outReal[i], h_outReal[i]);
    }
    EXPECT_EQ(ref_outIdx, h_outIdx);
    ExpectComplexNear(ref_outDualVal,  h_outDualVal);
    ExpectComplexNear(ref_outHyperVal, h_outHyperVal);
}

// ---------------------------------------------------------
// 4) ValidCase => in_idx in [0..input_real_size-1]
// ---------------------------------------------------------
TEST(VectorHyperDualIndexPutSparseTest, ValidCase)
{
    using T = double;
    int input_real_size = 3;
    std::vector<thrust::complex<T>> h_inReal {
        {10,10}, {11,11}, {12,12}
    };
    int h_inIdx = 1; // valid => in [0..2]
    thrust::complex<T> h_inDualVal(100,200);
    thrust::complex<T> h_inHyperVal(300,400);

    int start_real=2, end_real=5;
    int out_size  = 5; 

    // GPU
    std::vector<thrust::complex<T>> h_outReal;
    int h_outIdx;
    thrust::complex<T> h_outDualVal, h_outHyperVal;
    RunVectorHyperDualIndexPutSparseKernel(
        h_inReal, h_inIdx, h_inDualVal, h_inHyperVal,
        input_real_size,
        start_real, end_real,
        h_outReal, h_outIdx, h_outDualVal, h_outHyperVal
    );

    // CPU
    std::vector<thrust::complex<T>> ref_outReal(out_size, { -999, -999 });
    int ref_outIdx;
    thrust::complex<T> ref_outDualVal, ref_outHyperVal;
    VectorHyperDualIndexPutSparseCPU(
        h_inReal, h_inIdx, h_inDualVal, h_inHyperVal,
        input_real_size,
        start_real, end_real,
        ref_outReal, ref_outIdx, ref_outDualVal, ref_outHyperVal
    );

    // Compare
    ASSERT_EQ(out_size, (int)h_outReal.size());
    for (int i=0; i<out_size; ++i) {
        ExpectComplexNear(ref_outReal[i], h_outReal[i]);
    }
    EXPECT_EQ(ref_outIdx, h_outIdx);
    ExpectComplexNear(ref_outDualVal,  h_outDualVal);
    ExpectComplexNear(ref_outHyperVal, h_outHyperVal);
}




// -----------------------------------------------------------------------------
// A single-block prefix sum kernel (Blelloch), for demonstration only
// -----------------------------------------------------------------------------
__global__
void PrefixSumExclusiveKernel(const int* in, int* out, int n)
{
    // single-block approach: blockDim.x == n, threadIdx.x in [0..n-1]
    extern __shared__ int temp[];
    int tid = threadIdx.x;

    if (tid < n) {
        temp[tid] = in[tid];
    }
    __syncthreads();

    // up-sweep
    for (int offset = 1; offset < n; offset <<= 1) {
        int idx = (tid + 1) * (offset << 1) - 1;
        if (idx < n) {
            temp[idx] += temp[idx - offset];
        }
        __syncthreads();
    }
    // set last to 0 for exclusive
    if (tid == 0) {
        temp[n - 1] = 0;
    }
    __syncthreads();

    // down-sweep
    for (int offset = n >> 1; offset > 0; offset >>= 1) {
        int idx = (tid + 1) * (offset << 1) - 1;
        if (idx < n) {
            int t = temp[idx];
            temp[idx] += temp[idx - offset];
            temp[idx - offset] = t;
        }
        __syncthreads();
    }

    // write out
    if (tid < n) {
        out[tid] = temp[tid];
    }
}


// -----------------------------------------------------------------------------
// CPU reference addition (naive, merges columns by sorting + summation)
// -----------------------------------------------------------------------------
void HostAddCSR(
    int rows, int cols,
    const std::vector<int>& ArowPtr,
    const std::vector<int>& AcolIdx,
    const std::vector<thrust::complex<double>>& Adata,
    const std::vector<int>& BrowPtr,
    const std::vector<int>& BcolIdx,
    const std::vector<thrust::complex<double>>& Bdata,

    std::vector<int>& CrowPtr,
    std::vector<int>& CcolIdx,
    std::vector<thrust::complex<double>>& Cdata)
{
    CrowPtr.resize(rows+1, 0);
    std::vector<std::vector<std::pair<int, thrust::complex<double>>>> rowAccum(rows);

    // gather A
    for(int r=0; r<rows; r++){
        for(int ai = ArowPtr[r]; ai < ArowPtr[r+1]; ai++){
            rowAccum[r].push_back({AcolIdx[ai], Adata[ai]});
        }
    }
    // gather B
    for(int r=0; r<rows; r++){
        for(int bi = BrowPtr[r]; bi < BrowPtr[r+1]; bi++){
            rowAccum[r].push_back({BcolIdx[bi], Bdata[bi]});
        }
    }
    // combine duplicates
    int nnz=0;
    for(int r=0; r<rows; r++){
        auto& vec = rowAccum[r];
        std::sort(vec.begin(), vec.end(),
                  [](auto &a, auto &b){return a.first<b.first;});
        int outPos=0;
        for(int i=1; i<(int)vec.size(); i++){
            if(vec[i].first == vec[outPos].first){
                vec[outPos].second += vec[i].second; 
            } else {
                outPos++;
                vec[outPos] = vec[i];
            }
        }
        vec.resize(vec.empty() ? 0 : outPos+1);
        nnz += (int)vec.size();
    }
    CrowPtr[0] = 0;
    for(int r=1; r<=rows; r++){
        CrowPtr[r] = CrowPtr[r-1] + (int)rowAccum[r-1].size();
    }
    CcolIdx.resize(nnz);
    Cdata.resize(nnz);

    int offset=0;
    for(int r=0; r<rows; r++){
        for(const auto& p : rowAccum[r]){
            CcolIdx[offset] = p.first;
            Cdata[offset]   = p.second;
            offset++;
        }
    }
}







 
//------------------------------------------------------------------------------
// A naive CPU "reference" function to sum up the given host data
// (similar to what the GPU code does, but we only sum the hData array).
// This is used to verify correctness of the GPU result.
//------------------------------------------------------------------------------
thrust::complex<double> HostSumCSRData(
    const std::vector<thrust::complex<double>>& hData)
{
    thrust::complex<double> sum(0,0);
    for (auto& val : hData) {
        sum += val;
    }
    return sum;
}




// CPU reference for "dim=0" => sum along rows => produce col sums
// or "dim=1" => sum along cols => produce row sums
std::vector<thrust::complex<double>> HostDimSumCSR(
    int rows, int cols,
    const std::vector<int>& rowPtr,
    const std::vector<int>& colIdx,
    const std::vector<thrust::complex<double>>& values,
    int dim)
{
    // output length = (dim==0 ? cols : rows)
    int outLen = (dim==0 ? cols : rows);
    std::vector<thrust::complex<double>> result(outLen, thrust::complex<double>(0,0));

    // For each row r:
    for(int r=0; r<rows; r++){
        int start = rowPtr[r];
        int end   = rowPtr[r+1];
        for(int i=start; i<end; i++){
            int c = colIdx[i];
            thrust::complex<double> val = values[i];
            if (dim == 0) {
                // sum over rows => accumulate into result[c]
                result[c] += val;
            } else {
                // sum over columns => accumulate into result[r]
                result[r] += val;
            }
        }
    }
    return result;
}







// -----------------------------------------------------------------------------
// A small CPU reference approach to build the "dense" version of (rows x cols),
// do the slice [start..end), and then build a reference 2D matrix of shape
// (end-start) x 1. Then convert that 2D matrix to a host-based CSR that we
// can compare with the device-based result. We'll do a naive "dense2csr" approach.
// -----------------------------------------------------------------------------
void HostMatrixIndexGetCSR_Reference(
    // Original matrix in "dense" row-major form
    int rows, int cols,
    const std::vector<thrust::complex<double>>& denseA, // length = rows*cols

    int startIndex,
    int endIndex,

    // outputs: a host-based CSR for the slice => shape = (endIndex - startIndex) x 1
    int& outRows, // = (end-start)
    int& outCols, // = 1
    std::vector<int>& outRowPtr,
    std::vector<int>& outColIdx,
    std::vector<thrust::complex<double>>& outData
)
{
    // shape of the slice
    int sliceLen = endIndex - startIndex;
    outRows = sliceLen;
    outCols = 1;

    // Build a (sliceLen x 1) dense array 'Cdense'
    std::vector<thrust::complex<double>> Cdense(sliceLen, thrust::complex<double>(0,0));

    // copy slice
    for (int idx = startIndex; idx < endIndex; idx++){
        Cdense[idx - startIndex] = denseA[idx];
    }

    // Now convert 'Cdense' to CSR, skipping zeros
    // row i => 1 column => if Cdense[i] != 0 => col=0 => val
    outRowPtr.resize(sliceLen+1);
    int nnzCount=0;
    for (int i=0; i<sliceLen; i++){
        // each row has 0 or 1 nonzero
        if (thrust::abs(Cdense[i]) != 0.0) {
            nnzCount++;
        }
    }
    outData.resize(nnzCount);
    outColIdx.resize(nnzCount);

    // fill rowPtr
    int rowPtr=0;
    int pos=0;
    outRowPtr[0] = 0;
    for (int i=0; i<sliceLen; i++){
        int rowNnz=0;
        if (thrust::abs(Cdense[i]) != 0.0) {
            rowNnz=1;
        }
        outRowPtr[i+1] = outRowPtr[i] + rowNnz;
        if (rowNnz == 1) {
            // col=0
            outColIdx[pos] = 0;
            outData[pos]   = Cdense[i];
            pos++;
        }
    }
}




// -----------------------------------------------------------------------------
// CPU Reference: Build a subrange slice of [start..end) in row-major flattening
// without converting the entire matrix to dense. We'll just iterate over
// each nonzero in the original CSR, compute its denseIdx = r*cols + c, and if
// in [start..end) => newRow = denseIdx - start => col=0 => store in a temp
// and then convert that to CSR by grouping them by newRow.
//
// The final shape => (end-start) x 1
// -----------------------------------------------------------------------------
void HostMatrixIndexGetCSR_Reference(
    // original matrix in CSR (host arrays) with shape=(rows x cols)
    int rows, int cols,
    const std::vector<int>& rowPtr,
    const std::vector<int>& colIdx,
    const std::vector<thrust::complex<double>>& vals,

    long long startIdx,
    long long endIdx,

    // outputs: shape => (endIdx-startIdx) x 1
    int& outRows,
    int& outCols,
    std::vector<int>& outRowPtr,
    std::vector<int>& outColIdx,
    std::vector<thrust::complex<double>>& outVals
)
{
    long long sliceLen = endIdx - startIdx;
    outRows = static_cast<int>(sliceLen);
    outCols = 1;

    // We'll store a list of (newRow, val). col=0 always.
    // Then we'll sort by newRow, build rowPtr from that.
    struct NZ {
        int row;
        thrust::complex<double> val;
    };
    std::vector<NZ> nzList;

    // For each row r:
    for(int r=0; r<rows; r++){
        for(int i=rowPtr[r]; i<rowPtr[r+1]; i++){
            int c = colIdx[i];
            long long denseIdx = (long long)r*cols + c;
            if(denseIdx >= startIdx && denseIdx < endIdx){
                int newRow = static_cast<int>(denseIdx - startIdx);
                nzList.push_back({ newRow, vals[i] });
            }
        }
    }
    // sort by row
    std::sort(nzList.begin(), nzList.end(),
              [](auto& a, auto& b){return a.row < b.row;});

    // Now build rowPtr by counting how many entries each row has
    outRowPtr.clear();
    outRowPtr.resize(outRows+1, 0);

    for(auto& nz : nzList){
        outRowPtr[nz.row+1] += 1;
    }
    for(int r=1; r<=outRows; r++){
        outRowPtr[r] += outRowPtr[r-1];
    }
    int nnzC = (int)nzList.size();
    outColIdx.resize(nnzC);
    outVals.resize(nnzC);

    // fill colIdx=0, data from the sorted list
    // We do a standard gather approach
    std::vector<int> offsetPerRow(outRows);
    for(int r=0; r<outRows; r++){
        offsetPerRow[r] = outRowPtr[r];
    }
    for(const auto& nz : nzList){
        int row = nz.row;
        int pos = offsetPerRow[row]++;
        outColIdx[pos] = 0; 
        outVals[pos]   = nz.val;
    }
}


// -----------------------------------------------------------------------------
// A CPU reference function for slicing a CSR matrix in 2D
//   We do row range [rowStart..rowEnd) and col range [colStart..colEnd).
//   We'll produce a new host-based CSR with shape = (outRows x outCols).
// -----------------------------------------------------------------------------
void HostMatrixSlice2D_Reference(
    // original matrix dims
    int Arows, int Acols,

    // original CSR on host
    const std::vector<int>& ArowPtr,
    const std::vector<int>& AcolIdx,
    const std::vector<thrust::complex<double>>& Avals,

    // slicing ranges
    int rowStart, int rowEnd,
    int colStart, int colEnd,

    // outputs
    int& outRows, int& outCols,
    std::vector<int>& outRowPtr,
    std::vector<int>& outColIdx,
    std::vector<thrust::complex<double>>& outVals
)
{
    // basic checks
    if(rowEnd<rowStart || colEnd<colStart ||
       rowStart<0 || rowEnd> Arows ||
       colStart<0|| colEnd> Acols){
        throw std::runtime_error("Invalid slice in CPU reference");
    }
    outRows = rowEnd - rowStart;
    outCols = colEnd - colStart;

    // We'll store each row in a temporary vector of (col, val).
    std::vector<std::vector<std::pair<int,thrust::complex<double>>>> rowAccum(outRows);

    for(int r = rowStart; r<rowEnd; r++){
        int newR = r - rowStart;
        for(int i = ArowPtr[r]; i < ArowPtr[r+1]; i++){
            int c = AcolIdx[i];
            if(c >= colStart && c< colEnd){
                int newC = c - colStart;
                rowAccum[newR].push_back({newC, Avals[i]});
            }
        }
    }

    // now we compress them into final CSR
    outRowPtr.resize(outRows+1, 0);
    int nnz=0;
    for(int r=0; r<outRows; r++){
        nnz +=(int) rowAccum[r].size();
    }

    outColIdx.resize(nnz);
    outVals.resize(nnz);

    outRowPtr[0]=0;
    for(int r=1; r<=outRows; r++){
        outRowPtr[r] = outRowPtr[r-1] + (int)rowAccum[r-1].size();
    }
    int offset=0;
    for(int r=0; r<outRows; r++){
        for(auto& e : rowAccum[r]){
            outColIdx[offset] = e.first;
            outVals[offset]   = e.second;
            offset++;
        }
    }
}





// Add more tests for IndexGet, IndexPut, ElementwiseMultiply, Square, Pow, and Sqrt similarly.
// Main entry point for Google Test
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}