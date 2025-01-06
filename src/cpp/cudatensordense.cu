#ifndef _CU_DUAL_TENSOR_HPP
#define _CU_DUAL_TENSOR_HPP
#include <cuda_runtime.h>
#include <iostream>
#include <complex>
//Utility class to implement dual tensor operations only necessary for QR decomposition
//This is a simplified version of the more extensive Dual class in the original codebase
//and it is implemented using cuBLAS and cuSPARSE for matrix operations
#include <cublas_v2.h>
#include <cusparse.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <memory>
#include <vector>
#include <complex>
#include <thrust/device_ptr.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/complex.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/device_vector.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <iostream>


namespace janus {

#include <thrust/complex.h>
#include <thrust/device_ptr.h>
#include <thrust/transform.h>
#include <vector>
#include <stdexcept>
#include <cublas_v2.h>
#include <cuda_runtime.h>

template <typename T>
class VectorDenseCuda {
private:
    int batch_size_;  // Batch dimension (M)
    int size_;        // Vector length (N)

    thrust::complex<T>* data_;  // Data [M, N] (complex)
    bool owns_memory_;          // Indicates if memory is managed internally

    // cuBLAS handle
    cublasHandle_t handle_;
    cudaStream_t stream_;

public:
    // Constructor with external memory
    VectorDenseCuda(int batch_size, int size, thrust::complex<T>* data)
        : batch_size_(batch_size), size_(size), data_(data), owns_memory_(false) {
        if (!data_) {
            throw std::invalid_argument("Data pointer is null");
        }
        initializeHandles();
    }

    // Constructor with internal memory allocation
    VectorDenseCuda(int batch_size, int size)
        : batch_size_(batch_size), size_(size), owns_memory_(true) {
        if (batch_size <= 0 || size <= 0) {
            throw std::invalid_argument("Batch size and vector size must be positive.");
        }

        size_t data_size = batch_size * size * sizeof(thrust::complex<T>);
        if (cudaMalloc(&data_, data_size) != cudaSuccess) {
            throw std::runtime_error("Failed to allocate GPU memory for data.");
        }

        initializeHandles();
    }

    // Destructor
    ~VectorDenseCuda() {
        if (owns_memory_ && data_) {
            cudaFree(data_);
        }
        if (handle_) {
            cublasDestroy(handle_);
        }
        if (stream_) {
            cudaStreamDestroy(stream_);
        }
    }

    // Initialize data from host
    void initialize(const thrust::complex<T>* host_data, size_t data_size) {
        if (data_size != batch_size_ * size_) {
            throw std::invalid_argument("Input size does not match vector dimensions.");
        }

        cudaMemcpyAsync(data_, host_data, data_size * sizeof(thrust::complex<T>), cudaMemcpyHostToDevice, stream_);
        cudaStreamSynchronize(stream_);
    }


    // Elementwise addition
    VectorDenseCuda elementwiseAdd(const VectorDenseCuda& other) const {
        if (batch_size_ != other.batch_size_ || size_ != other.size_) {
            throw std::invalid_argument("Vector dimensions do not match for elementwise addition.");
        }

        VectorDenseCuda result(batch_size_, size_);
        int total_elements = batch_size_ * size_;

        thrust::device_ptr<thrust::complex<T>> d_ptr1(data_);
        thrust::device_ptr<thrust::complex<T>> d_ptr2(other.data_);
        thrust::device_ptr<thrust::complex<T>> d_ptr_result(result.data_);

        thrust::transform(
            d_ptr1, d_ptr1 + total_elements,
            d_ptr2,
            d_ptr_result,
            thrust::plus<thrust::complex<T>>());

        return result;
    }




    // Elementwise multiplication
    VectorDenseCuda elementwiseMultiply(const VectorDenseCuda& other) const {
        if (batch_size_ != other.batch_size_ || size_ != other.size_) {
            throw std::invalid_argument("Vector dimensions do not match for elementwise multiplication.");
        }

        VectorDenseCuda result(batch_size_, size_);
        int total_elements = batch_size_ * size_;

        thrust::transform(
            thrust::device_pointer_cast(data_),
            thrust::device_pointer_cast(data_ + total_elements),
            thrust::device_pointer_cast(other.data_),
            thrust::device_pointer_cast(result.data_),
            thrust::multiplies<thrust::complex<T>>());

        return result;
    }

    // Accessors
    thrust::complex<T>* data() { return data_; }
    const thrust::complex<T>* data() const { return data_; }
    int batchSize() const { return batch_size_; }
    int size() const { return size_; }

private:
    // Initialize cuBLAS handle and CUDA stream
    void initializeHandles() {
        if (cublasCreate(&handle_) != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("Failed to create cuBLAS handle.");
        }
        if (cudaStreamCreate(&stream_) != cudaSuccess) {
            cublasDestroy(handle_);
            throw std::runtime_error("Failed to create CUDA stream.");
        }
        cublasSetStream(handle_, stream_);
    }
};



#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        throw std::runtime_error(cudaGetErrorString(err)); \
    } \
} while (0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t err = call; \
    if (err != CUBLAS_STATUS_SUCCESS) { \
        throw std::runtime_error("cuBLAS error"); \
    } \
} while (0)

template <typename T>
struct BroadcastDualMultiply {
    const thrust::complex<T>* real;
    const thrust::complex<T>* dual;
    int real_size, dual_size;

    BroadcastDualMultiply(const thrust::complex<T>* real, 
                          const thrust::complex<T>* dual, int real_size, int dual_size)
        : real(real), dual(dual), real_size(real_size), dual_size(dual_size) {}

    __device__ thrust::complex<float> operator()(int idx) const {
        int batch_idx = idx / (real_size * dual_size);
        int real_idx = (idx / dual_size) % real_size;
        int dual_idx = idx % dual_size;

        int real_offset = batch_idx * real_size + real_idx;  // Corresponding real tensor index
        return real[real_offset] * dual[idx];
    }
};


template <typename T>
void multiplyRealDualTensor(const thrust::complex<T>* real,
                        const thrust::complex<T>* dual,
                        thrust::complex<T>* result,
                        int batch_size, int real_size, int dual_size) {
    int total_elements = batch_size * real_size * dual_size;

    thrust::transform(
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(total_elements),
        thrust::device_pointer_cast(result),
        [=] __device__(int idx) {
            int batch_idx = idx / (real_size * dual_size);
            int real_idx = (idx / dual_size) % real_size;
            //int dual_idx = idx % dual_size;

            int real_offset = batch_idx * real_size + real_idx;  // Corresponding real tensor index
            return real[real_offset] * dual[idx];
        });
}

/**
 * Multiply two dual tensors elementwise.
 * This emulates the einstein sum "mij,mik->mijk" for dual tensors.
 */
template <typename T>
void multiplyDualDualTensor(const thrust::complex<T>* dual1,
                            const thrust::complex<T>* dual2,
                            thrust::complex<T>* result,
                            int batch_size, int real_size, int dual_size) {
    int total_elements = batch_size * real_size * dual_size * dual_size;

    thrust::transform(
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(total_elements),
        thrust::device_pointer_cast(result),
        [=] __device__(int idx) {
            // Calculate indices for the result tensor
            int batch_idx = idx / (real_size * dual_size * dual_size);
            int real_idx = (idx / (dual_size * dual_size)) % real_size;
            int dual_idx1 = (idx / dual_size) % dual_size;
            int dual_idx2 = idx % dual_size;

            // Offsets in dual1 and dual2
            int dual1_offset = batch_idx * real_size * dual_size + real_idx * dual_size + dual_idx1;
            int dual2_offset = batch_idx * real_size * dual_size + real_idx * dual_size + dual_idx2;

            // Perform elementwise multiplication
            return dual1[dual1_offset] * dual2[dual2_offset];
        });
}


template <typename T>
__global__ void elementwiseAddKernel(
    thrust::complex<T>* real1, thrust::complex<T>* real2,
    thrust::complex<T>* dual1, thrust::complex<T>* dual2,
    thrust::complex<T>* real_result, thrust::complex<T>* dual_result,
    int real_size, int dual_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Real part addition
    if (idx < real_size && real1 && real2 && real_result) {
        real_result[idx] = real1[idx] + real2[idx];
    }

    // Dual part addition (if provided)
    if (idx < dual_size && dual1 && dual2 && dual_result) {
        dual_result[idx] = dual1[idx] + dual2[idx];
    }
}


    template <typename T>
    __global__ void elementwiseMultiplyKernel(
        thrust::complex<T>* real1, thrust::complex<T>* real2,
        thrust::complex<T>* dual1, thrust::complex<T>* dual2,
        thrust::complex<T>* real_result, thrust::complex<T>* dual_result,
        int real_size, int dual_size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        // Real part multiplication
        if (idx < real_size && real1 && real2 && real_result) {
            real_result[idx] = real1[idx] * real2[idx];
        }

        // Dual part multiplication (if provided)
        if (idx < dual_size && dual1 && dual2 && dual_result) {
            int dual_dim = dual_size/real_size;
            int real_idx = idx/dual_dim;
            dual_result[idx] = real1[real_idx] * dual2[idx] +
                            real2[real_idx] * dual1[idx];
        }
    }





template <typename T>
class VectorDualDenseCuda {
private:
    int batch_size_;             // Batch dimension (M)
    int real_size_;              // Vector length (N)
    int dual_size_;              // Dual dimension (D)

    thrust::complex<T>* real_;   // Real part [M, N]
    thrust::complex<T>* dual_;   // Dual part [M, N, D]

public:
    // Constructor for GPU allocation (device-only)
    __host__ __device__ VectorDualDenseCuda(int batch_size, int real_size, int dual_size, thrust::complex<T>* real, thrust::complex<T>* dual)
        : batch_size_(batch_size), real_size_(real_size), dual_size_(dual_size), real_(real), dual_(dual) {}

    // Elementwise addition on GPU
    __host__ __device__ void elementwiseAdd(const VectorDualDenseCuda& other, VectorDualDenseCuda& result) const {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        int total_real_elements = batch_size_ * real_size_;
        int total_dual_elements = batch_size_ * real_size_ * dual_size_;

        // Real part addition
        if (idx < total_real_elements) {
            result.real_[idx] = real_[idx] + other.real_[idx];
        }

        // Dual part addition
        if (idx < total_dual_elements) {
            result.dual_[idx] = dual_[idx] + other.dual_[idx];
        }
    }

    // Elementwise multiplication on GPU
    __host__ __device__ void elementwiseMultiply(const VectorDualDenseCuda& other, VectorDualDenseCuda& result) const {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        int total_real_elements = batch_size_ * real_size_;
        int total_dual_elements = batch_size_ * real_size_ * dual_size_;
        int dual_dim = dual_size_ / real_size_;

        // Real part multiplication
        if (idx < total_real_elements) {
            result.real_[idx] = real_[idx] * other.real_[idx];
        }

        // Dual part multiplication
        if (idx < total_dual_elements) {
            int real_idx = idx / dual_dim;
            result.dual_[idx] = real_[real_idx] * other.dual_[idx] +
                                other.real_[real_idx] * dual_[idx];
        }
    }

    // Accessors (for GPU use)
    __host__ __device__ thrust::complex<T>* real() { return real_; }
    __host__ __device__ const thrust::complex<T>* real() const { return real_; }
    __host__ __device__ thrust::complex<T>* dual() { return dual_; }
    __host__ __device__ const thrust::complex<T>* dual() const { return dual_; }
    __host__ __device__ int batchSize() const { return batch_size_; }
    __host__ __device__ int size() const { return real_size_; }
    __host__ __device__ int dualSize() const { return dual_size_; }
};



template <typename T>
__global__ void multiplyRealHyperDualKernel(
    const thrust::complex<T>* real1, const thrust::complex<T>* dual1, const thrust::complex<T>* hyperDual1,
    const thrust::complex<T>* real2, const thrust::complex<T>* dual2, const thrust::complex<T>* hyperDual2,
    thrust::complex<T>* result, int batch_size, int real_size, int dual_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int total_elements = batch_size * real_size * dual_size * dual_size;

    if (idx < total_elements) {
        // Decompose the index into batch, row, and column indices
        int dual_dim = dual_size;
        int batch_idx = idx / (real_size * dual_dim * dual_dim);
        int real_idx = (idx / (dual_dim * dual_dim)) % real_size;
        int dual_row = (idx / dual_dim) % dual_dim;
        int dual_col = idx % dual_dim;

        // Global real index
        int real_global_idx = batch_idx * real_size + real_idx;

        // Compute result index
        int result_idx = batch_idx * real_size * dual_dim * dual_dim +
                         real_idx * dual_dim * dual_dim +
                         dual_row * dual_dim + dual_col;

        // Indices for hyperdual tensors
        int hyperDual1_idx = result_idx;  // Assuming hyperDual1 and result share the same indexing
        int hyperDual2_idx = result_idx;

        // Multiply the real and hyperdual tensors
        result[result_idx] = real1[real_global_idx] * hyperDual2[hyperDual2_idx] +
                             real2[real_global_idx] * hyperDual1[hyperDual1_idx];

        // Indices for dual tensors
        int dual1_idx = batch_idx * real_size * dual_dim + real_idx * dual_dim + dual_row;
        int dual2_idx = batch_idx * real_size * dual_dim + real_idx * dual_dim + dual_col;

        // Compute the outer product of the dual parts
        result[result_idx] += dual1[dual1_idx] * dual2[dual2_idx];
    }
}












template <typename T>
class VectorHyperDualDenseCuda {
private:
    int batch_size_;                 // Batch dimension (M)
    int real_size_;                  // Vector length (N)
    int dual_size_;                  // Dual dimension (D)

    thrust::complex<T>* real_;       // Real part [M, N]
    thrust::complex<T>* dual_;       // Dual part [M, N, D]
    thrust::complex<T>* hyperdual_;  // Hyperdual part [M, N, D, D]

public:
    // Constructor for GPU allocation (device-only)
    __host__ __device__ VectorHyperDualDenseCuda(
        int batch_size, int real_size, int dual_size,
        thrust::complex<T>* real, thrust::complex<T>* dual, thrust::complex<T>* hyperdual)
        : batch_size_(batch_size), real_size_(real_size), dual_size_(dual_size),
          real_(real), dual_(dual), hyperdual_(hyperdual) {}

    // Elementwise addition on GPU
    __host__ __device__ void elementwiseAdd(const VectorHyperDualDenseCuda& other, VectorHyperDualDenseCuda& result) const {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        int total_real_elements = batch_size_ * real_size_;
        int total_dual_elements = batch_size_ * real_size_ * dual_size_;
        int total_hyperdual_elements = batch_size_ * real_size_ * dual_size_ * dual_size_;

        // Real part addition
        if (idx < total_real_elements) {
            result.real_[idx] = real_[idx] + other.real_[idx];
        }

        // Dual part addition
        if (idx < total_dual_elements) {
            result.dual_[idx] = dual_[idx] + other.dual_[idx];
        }

        // Hyperdual part addition
        if (idx < total_hyperdual_elements) {
            result.hyperdual_[idx] = hyperdual_[idx] + other.hyperdual_[idx];
        }
    }

    // Elementwise multiplication on GPU
    __host__ __device__ void elementwiseMultiply(const VectorHyperDualDenseCuda& other, VectorHyperDualDenseCuda& result) const {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        int total_real_elements = batch_size_ * real_size_;
        int total_dual_elements = batch_size_ * real_size_ * dual_size_;
        int total_hyperdual_elements = batch_size_ * real_size_ * dual_size_ * dual_size_;
        int dual_dim = dual_size_;

        // Real part multiplication
        if (idx < total_real_elements) {
            result.real_[idx] = real_[idx] * other.real_[idx];
        }

        // Dual part multiplication
        if (idx < total_dual_elements) {
            int real_idx = idx / dual_dim;
            result.dual_[idx] = real_[real_idx] * other.dual_[idx] +
                                other.real_[real_idx] * dual_[idx];
        }

        // Hyperdual part multiplication
        if (idx < total_hyperdual_elements) {
            //Here we are implementing "mij,mik->mijk" for hyperdual tensors
            int m = idx / (real_size_ * dual_dim * dual_dim);
            int i = (idx / (dual_dim * dual_dim)) % real_size_;
            int j = (idx / dual_dim) % dual_dim;
            int k = idx % dual_dim;

            int dual1_idx = m * real_size_ * dual_dim + i * dual_dim + j;
            int dual2_idx = m * real_size_ * dual_dim + i * dual_dim + k;

            result.hyperdual_[idx] = real_[m * real_size_ + i] * other.hyperdual_[idx] +
                                     other.real_[m * real_size_ + i] * hyperdual_[idx] +
                                     dual_[dual1_idx] * other.dual_[dual2_idx];
        }
    }

    // Accessors (for GPU use)
    __host__ __device__ thrust::complex<T>* real() { return real_; }
    __host__ __device__ const thrust::complex<T>* real() const { return real_; }
    __host__ __device__ thrust::complex<T>* dual() { return dual_; }
    __host__ __device__ const thrust::complex<T>* dual() const { return dual_; }
    __host__ __device__ thrust::complex<T>* hyperdual() { return hyperdual_; }
    __host__ __device__ const thrust::complex<T>* hyperdual() const { return hyperdual_; }
    __host__ __device__ int batchSize() const { return batch_size_; }
    __host__ __device__ int size() const { return real_size_; }
    __host__ __device__ int dualSize() const { return dual_size_; }
};






template <typename T>
class MatrixDenseCuda {
private:
    using ComplexT = std::complex<T>;

    // Dimensions
    int batch_size_;
    int rows_;
    int cols_;

    // Primal Data
    ComplexT* primal_data_; // Device-side primal part
    bool owns_memory_;      // Indicates if memory is managed internally

    // cuBLAS handle
    cublasHandle_t handle_;
    cudaStream_t stream_;

public:
    // Constructor with external memory
    MatrixDenseCuda(int batch_size, int rows, int cols, ComplexT* primal_data)
        : batch_size_(batch_size), rows_(rows), cols_(cols),
          primal_data_(primal_data), owns_memory_(false) {
        if (!primal_data_) {
            throw std::invalid_argument("Primal data pointer is null");
        }

        initializeHandles();
    }

    // Constructor with internal memory allocation
    MatrixDenseCuda(int batch_size, int rows, int cols)
        : batch_size_(batch_size), rows_(rows), cols_(cols), owns_memory_(true) {
        if (batch_size <= 0 || rows <= 0 || cols <= 0) {
            throw std::invalid_argument("All dimensions must be positive.");
        }

        size_t primal_size = batch_size * rows * cols * sizeof(ComplexT);

        if (cudaMalloc(&primal_data_, primal_size) != cudaSuccess) {
            throw std::runtime_error("Failed to allocate GPU memory for primal data.");
        }

        initializeHandles();
    }

    ~MatrixDenseCuda() {
        if (owns_memory_ && primal_data_) {
            cudaFree(primal_data_);
        }
        cublasDestroy(handle_);
        cudaStreamDestroy(stream_);
    }

    void initialize(const ComplexT* primal, size_t primal_size) {
        if (primal_size != batch_size_ * rows_ * cols_) {
            throw std::invalid_argument("Input size does not match tensor dimensions.");
        }

        if (primal) {
            cudaMemcpyAsync(primal_data_, primal, primal_size * sizeof(ComplexT), cudaMemcpyHostToDevice, stream_);
        }
        cudaStreamSynchronize(stream_);
    }

    template <typename U>
    MatrixDenseCuda<U> indexGet(int start_row, int end_row, int start_col, int end_col) const {
        // Validate row and column ranges
        if (start_row < 0 || end_row > rows_ || start_row >= end_row ||
            start_col < 0 || end_col > cols_ || start_col >= end_col) {
            throw std::invalid_argument("Invalid row or column range for indexGet.");
        }

        // Dimensions of the submatrix
        int sub_rows = end_row - start_row;
        int sub_cols = end_col - start_col;

        // Calculate the offset for the primal data
        ComplexT* sub_primal_data = primal_data_ + start_row * cols_ + start_col;

        // Create a new MatrixDense instance sharing the data with the original
        return MatrixDenseCuda<T>(batch_size_, sub_rows, sub_cols, sub_primal_data);
    }


    template <typename U>
    void indexPut(int start_row, int end_row, int start_col, int end_col, const MatrixDenseCuda<U>& data) {
        // Validate row and column ranges
        if (start_row < 0 || end_row > rows_ || start_row >= end_row ||
            start_col < 0 || end_col > cols_ || start_col >= end_col) {
            throw std::invalid_argument("Invalid row or column range for indexPut.");
        }

        // Validate dimensions of the input data
        if (data.rows_ != (end_row - start_row) || data.cols_ != (end_col - start_col) || 
            data.batch_size_ != batch_size_) {
            throw std::invalid_argument("Input data dimensions do not match the target range.");
        }

        // Calculate the offset for the primal data
        ComplexT* target_primal_data = primal_data_ + start_row * cols_ + start_col;

        // Update primal data
        for (int b = 0; b < batch_size_; ++b) {
            cudaMemcpy2DAsync(target_primal_data + b * rows_ * cols_,
                            cols_ * sizeof(ComplexT),
                            data.primal_data_ + b * data.rows_ * data.cols_,
                            data.cols_ * sizeof(ComplexT),
                            data.cols_ * sizeof(ComplexT),
                            data.rows_,
                            cudaMemcpyDeviceToDevice,
                            stream_);
        }

        // Synchronize to ensure data transfer is complete
        cudaStreamSynchronize(stream_);
    }


    MatrixDenseCuda<T> multiply(const MatrixDenseCuda<T>& other) const {
        // Validate dimensions
        if (cols_ != other.rows_ || batch_size_ != other.batch_size_) {
            throw std::invalid_argument("Matrix dimensions do not match for multiplication.");
        }

        // Create the result matrix
        MatrixDenseCuda<T> result(batch_size_, rows_, other.cols_);

        // Scaling factors for cuBLAS
        ComplexT alpha = ComplexT(1.0, 0.0);
        ComplexT beta = ComplexT(0.0, 0.0);

        // Perform batched matrix multiplication
        cublasStatus_t status = cublasZgemmStridedBatched(
            handle_,
            CUBLAS_OP_N,           // No transpose for this matrix
            CUBLAS_OP_N,           // No transpose for the other matrix
            rows_,                 // Number of rows of the output matrix
            other.cols_,           // Number of columns of the output matrix
            cols_,                 // Shared dimension (this.cols_ == other.rows_)
            &alpha,                // Scaling factor for the multiplication
            primal_data_,          // Pointer to this matrix data
            rows_,                 // Leading dimension of this matrix
            rows_ * cols_,         // Stride between consecutive matrices in the batch
            other.primal_data_,    // Pointer to other matrix data
            other.rows_,           // Leading dimension of the other matrix
            other.rows_ * other.cols_, // Stride between consecutive matrices in the batch
            &beta,                 // Scaling factor for the result matrix
            result.primal_data_,   // Pointer to result matrix data
            result.rows_,          // Leading dimension of the result matrix
            result.rows_ * result.cols_, // Stride between consecutive matrices in the batch
            batch_size_            // Number of matrices in the batch
        );

        if (status != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("cuBLAS matrix multiplication failed.");
        }

        return result;
    }

    VectorDenseCuda<T> matrixVectorProduct(const VectorDenseCuda<T>& vector) const {
        // Validate dimensions
        if (cols_ != vector.size() || batch_size_ != vector.batch_size()) {
            throw std::invalid_argument("Matrix and vector dimensions do not match for multiplication.");
        }

        // Create the result vector
        VectorDenseCuda<T> result(batch_size_, rows_);

        // Scaling factors for cuBLAS
        ComplexT alpha = ComplexT(1.0, 0.0);
        ComplexT beta = ComplexT(0.0, 0.0);

        // Perform batched matrix-vector multiplication
        for (int b = 0; b < batch_size_; ++b) {
            cublasStatus_t status = cublasZgemv(
                handle_,
                CUBLAS_OP_N,                            // No transpose for this matrix
                rows_, cols_,                           // Dimensions of the matrix
                &alpha,                                 // Scaling factor for multiplication
                primal_data_ + b * rows_ * cols_,       // Pointer to the matrix for this batch
                rows_,                                  // Leading dimension of the matrix
                vector.primal_data() + b * vector.size(), // Pointer to the vector for this batch
                1,                                      // Stride for the vector
                &beta,                                  // Scaling factor for the result
                result.primal_data() + b * rows_,       // Pointer to the result vector for this batch
                1                                       // Stride for the result vector
            );

            if (status != CUBLAS_STATUS_SUCCESS) {
                throw std::runtime_error("cuBLAS matrix-vector multiplication failed.");
            }
        }

        return result;
    }

    MatrixDenseCuda<T> transpose() const {
        MatrixDenseCuda<T> result(batch_size_, cols_, rows_);
        ComplexT alpha = ComplexT(1.0, 0.0);
        ComplexT beta = ComplexT(0.0, 0.0);

        for (int b = 0; b < batch_size_; ++b) {
            cublasZgeam(handle_,
                        CUBLAS_OP_T, CUBLAS_OP_T,
                        cols_, rows_,
                        &alpha,
                        primal_data_ + b * rows_ * cols_, rows_,
                        &beta,
                        nullptr, cols_,
                        result.primal_data_ + b * cols_ * rows_, cols_);
        }

        return result;
    }

    MatrixDenseCuda<T> elementwiseAdd(const MatrixDenseCuda<T>& other) const {
        if (batch_size_ != other.batch_size_ || rows_ != other.rows_ || cols_ != other.cols_) {
            throw std::invalid_argument("Tensor dimensions do not match for addition.");
        }

        MatrixDenseCuda<T> result(batch_size_, rows_, cols_);
            int total_elements = batch_size_ * rows_ * cols_;

            thrust::transform(thrust::device_pointer_cast(primal_data_),
                            thrust::device_pointer_cast(primal_data_ + total_elements),
                            thrust::device_pointer_cast(other.primal_data_),
                            thrust::device_pointer_cast(result.primal_data_),
                            thrust::plus<ComplexT>());

            return result;
    }

    MatrixDenseCuda<T> square() const {
        // Create a new MatrixDense object to store the result
        MatrixDenseCuda<T> result(batch_size_, rows_, cols_);

        // Calculate the total number of elements in the tensor
        int total_elements = batch_size_ * rows_ * cols_;

        // Use thrust to perform element-wise squaring of the tensor
        thrust::transform(
            thrust::device_pointer_cast(primal_data_), 
            thrust::device_pointer_cast(primal_data_ + total_elements),
            thrust::device_pointer_cast(result.primal_data()),
            [] __device__(ComplexT x) { return x * x; });

        return result;
    }

    MatrixDenseCuda<T> upperTriangular() const {
        // Create a result matrix with the same dimensions
        MatrixDenseCuda<T> result(batch_size_, rows_, cols_);

        size_t total_elements = rows_ * cols_;

        for (int b = 0; b < batch_size_; ++b) {
            // Primal data pointers for this batch
            ComplexT* batch_primal_src = primal_data_ + b * total_elements;
            ComplexT* batch_primal_dst = result.primal_data_ + b * total_elements;

            // Apply the upper triangular operation
            thrust::for_each(
                thrust::counting_iterator<int>(0),
                thrust::counting_iterator<int>(rows_ * cols_),
                [=] __device__(int idx) {
                    int row = idx / cols_;
                    int col = idx % cols_;
                    batch_primal_dst[idx] = (row <= col) ? batch_primal_src[idx] : ComplexT(0.0, 0.0);
                });
        }

        return result;
    }

    MatrixDenseCuda<T> lowerTriangular() const {
        // Create a result matrix with the same dimensions
        MatrixDenseCuda<T> result(batch_size_, rows_, cols_);

        size_t total_elements = rows_ * cols_;

        for (int b = 0; b < batch_size_; ++b) {
            // Primal data pointers for this batch
            ComplexT* batch_primal_src = primal_data_ + b * total_elements;
            ComplexT* batch_primal_dst = result.primal_data_ + b * total_elements;

            // Apply the lower triangular operation
            thrust::for_each(
                thrust::counting_iterator<int>(0),
                thrust::counting_iterator<int>(rows_ * cols_),
                [=] __device__(int idx) {
                    int row = idx / cols_;
                    int col = idx % cols_;
                    batch_primal_dst[idx] = (row >= col) ? batch_primal_src[idx] : ComplexT(0.0, 0.0);
                });
        }

        return result;
    }




private:
    void initializeHandles() {
        if (cublasCreate(&handle_) != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("Failed to create cuBLAS handle.");
        }
        if (cudaStreamCreate(&stream_) != cudaSuccess) {
            cublasDestroy(handle_);
            throw std::runtime_error("Failed to create CUDA stream.");
        }
        cublasSetStream(handle_, stream_);
    }

public:
    // Disable copy constructor and copy assignment
    MatrixDenseCuda(const MatrixDenseCuda&) = delete;
    MatrixDenseCuda& operator=(const MatrixDenseCuda&) = delete;

    // Enable move constructor and move assignment
    MatrixDenseCuda(MatrixDenseCuda&&) noexcept = default;
    MatrixDenseCuda& operator=(MatrixDenseCuda&&) noexcept = default;

    // Getters for dimensions
    int batch_size() const { return batch_size_; }
    int rows() const { return rows_; }
    int cols() const { return cols_; }

    // Getters for data pointers
    ComplexT* primal_data() { return primal_data_; }
    const ComplexT* primal_data() const { return primal_data_; }
}; // class MatrixDenseCuda



template <typename T>
class MatrixDualDenseCuda {
private:
    using ComplexT = std::complex<T>;

    // Dimensions
    int batch_size_;  // Batch dimension (M)
    int rows_;        // Number of rows in each matrix (N)
    int cols_;        // Number of columns in each matrix (L)
    int dual_dim_;    // Dual dimension (D)

    // Primal and Dual Data
    ComplexT* primal_data_; // Device-side primal part
    ComplexT* dual_data_;   // Device-side dual part [M, N, M, D]
    bool owns_memory_;      // Indicates if memory is managed internally

    // cuBLAS handle
    cublasHandle_t handle_;
    cudaStream_t stream_;

public:
    // Constructor with external memory
    MatrixDualDenseCuda(int batch_size, int rows, int cols, int dual_dim, ComplexT* primal_data, ComplexT* dual_data)
        : batch_size_(batch_size), rows_(rows), cols_(cols), dual_dim_(dual_dim),
          primal_data_(primal_data), dual_data_(dual_data), owns_memory_(false) {
        if (!primal_data_ || !dual_data_) {
            throw std::invalid_argument("Primal or dual data pointer is null");
        }

        initializeHandles();
    }

    // Constructor with internal memory allocation
    MatrixDualDenseCuda(int batch_size, int rows, int cols, int dual_dim)
        : batch_size_(batch_size), rows_(rows), cols_(cols), dual_dim_(dual_dim), owns_memory_(true) {
        if (batch_size <= 0 || rows <= 0 || cols <= 0 || dual_dim <= 0) {
            throw std::invalid_argument("All dimensions must be positive.");
        }

        size_t primal_size = batch_size * rows * cols * sizeof(ComplexT);
        size_t dual_size = batch_size * rows * batch_size * dual_dim * sizeof(ComplexT);

        if (cudaMalloc(&primal_data_, primal_size) != cudaSuccess ||
            cudaMalloc(&dual_data_, dual_size) != cudaSuccess) {
            throw std::runtime_error("Failed to allocate GPU memory for primal or dual data.");
        }

        initializeHandles();
    }

    ~MatrixDualDenseCuda() {
        if (owns_memory_) {
            if (primal_data_) cudaFree(primal_data_);
            if (dual_data_) cudaFree(dual_data_);
        }
        cublasDestroy(handle_);
        cudaStreamDestroy(stream_);
    }

    void initialize(const ComplexT* primal, const ComplexT* dual) {
        int primal_size = batch_size_ * rows_ * cols_;
        int dual_size = batch_size_ * rows_ * batch_size_ * dual_dim_;
        if (primal) {
            cudaMemcpyAsync(primal_data_, primal, primal_size * sizeof(ComplexT), cudaMemcpyHostToDevice, stream_);
        }
        if (dual) {
            cudaMemcpyAsync(dual_data_, dual, dual_size * sizeof(ComplexT), cudaMemcpyHostToDevice, stream_);
        }
        cudaStreamSynchronize(stream_);
    }

    // Add your methods (e.g., elementwiseAdd, elementwiseMultiply) here with dual_dim_ handled appropriately.
    
    // Example method: Elementwise addition
    MatrixDualDenseCuda<T> elementwiseAdd(const MatrixDualDenseCuda<T>& other) const {
        if (batch_size_ != other.batch_size_ || rows_ != other.rows_ || cols_ != other.cols_ || dual_dim_ != other.dual_dim_) {
            throw std::invalid_argument("Tensor dimensions do not match for addition.");
        }

        MatrixDualDenseCuda<T> result(batch_size_, rows_, cols_, dual_dim_);
        int total_primal_elements = batch_size_ * rows_ * cols_;
        int total_dual_elements = batch_size_ * rows_ * batch_size_ * dual_dim_;

        thrust::transform(thrust::device_pointer_cast(primal_data_),
                          thrust::device_pointer_cast(primal_data_ + total_primal_elements),
                          thrust::device_pointer_cast(other.primal_data_),
                          thrust::device_pointer_cast(result.primal_data_),
                          thrust::plus<ComplexT>());

        thrust::transform(thrust::device_pointer_cast(dual_data_),
                          thrust::device_pointer_cast(dual_data_ + total_dual_elements),
                          thrust::device_pointer_cast(other.dual_data_),
                          thrust::device_pointer_cast(result.dual_data_),
                          thrust::plus<ComplexT>());

        return result;
    }


    // Elementwise multiplication on GPU
    __host__ __device__ void elementwiseMultiply(const MatrixDualDenseCuda& other, 
                                                MatrixDualDenseCuda& result) const {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;

        // Total number of elements in primal and dual tensors
        int total_primal_elements = batch_size_ * rows_ * cols_;  // For primal tensor
        int total_dual_elements = batch_size_ * rows_ * cols_ * dual_dim_;  // For dual tensor

        // Real (primal) part multiplication
        if (idx < total_primal_elements) {
            result.primal_data_[idx] = primal_data_[idx] * other.primal_data_[idx];
        }

        // Dual part multiplication
        if (idx < total_dual_elements) {
            // Decode indices for dual tensor from flat index `idx` using modulo-based decoding
            int m = idx / (rows_ * cols_ * dual_dim_);  // Primary batch index
            int local_idx = idx % (rows_ * cols_ * dual_dim_);

            int i = local_idx / (cols_ * dual_dim_);      // Row index
            int j = (local_idx / dual_dim_) % cols_;      // Column index
            int k = local_idx % dual_dim_;                // Dual dimension index

            // Calculate offsets for primal and dual components
            int primal_offset = m * rows_ * cols_ + i * cols_ + j;  // Primal tensor offset
            int dual_offset = m * rows_ * cols_ * dual_dim_ + i * cols_ * dual_dim_ + j * dual_dim_ + k;

            // Extract primal and dual components
            T real1 = primal_data_[primal_offset];
            T real2 = other.primal_data_[primal_offset];
            T dual1 = dual_data_[dual_offset];
            T dual2 = other.dual_data_[dual_offset];

            // Compute dual part of the result
            result.dual_data_[idx] = real1 * dual2 + real2 * dual1;
        }
    }


    __device__ void matrixMultiplyDual(const T* A_real, const T* A_dual,
                                    const T* B_real, const T* B_dual,
                                    T* C_real, T* C_dual,
                                    int M, int N, int L, int P, int D) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;

        // Total number of real elements in the output
        int total_real_elements = M * N * P;
        // Total number of dual elements in the output
        int total_dual_elements = M * N * P * D;

        if (idx < total_real_elements) {
            // Decode indices for the real output tensor
            int m = idx / (N * P);                  // Batch index
            int i = (idx % (N * P)) / P;            // Row index
            int k = (idx % (N * P)) % P;            // Column index

            // Compute the real part of the result tensor
            T C_real_local = 0;
            for (int j = 0; j < L; ++j) {
                T A_real_val = A_real[m * (N * L) + i * L + j];
                T B_real_val = B_real[m * (L * P) + j * P + k];
                C_real_local += A_real_val * B_real_val;
            }
            C_real[idx] = C_real_local;
        }

        if (idx < total_dual_elements) {
            // Decode indices for the dual output tensor
            int m = idx / (N * P * D);              // Batch index
            int local_idx = idx % (N * P * D);
            int i = local_idx / (P * D);            // Row index
            int k = (local_idx % (P * D)) / D;      // Column index
            int l = (local_idx % (P * D)) % D;      // Dual dimension index

            // Compute the dual part of the result tensor
            T C_dual_local = 0;
            for (int j = 0; j < L; ++j) {
                T A_real_val = A_real[m * (N * L) + i * L + j];
                T B_real_val = B_real[m * (L * P) + j * P + k];
                T A_dual_val = A_dual[m * (N * L * D) + i * (L * D) + j * D + l];
                T B_dual_val = B_dual[m * (L * P * D) + j * (P * D) + k * D + l];

                C_dual_local += A_real_val * B_dual_val + A_dual_val * B_real_val;
            }
            C_dual[idx] = C_dual_local;
        }
    }


    __device__ void matrixVectorMultiplyDual(const T* A_real, const T* A_dual,
                                            const T* B_real, const T* B_dual,
                                            T* C_real, T* C_dual,
                                            int M, int N, int L, int D) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;

        // Total number of real elements in the output
        int total_real_elements = M * N;
        // Total number of dual elements in the output
        int total_dual_elements = M * N * D;

        // Real part computation
        if (idx < total_real_elements) {
            // Decode indices for real output
            int m = idx / N;          // Batch index
            int i = idx % N;          // Row index

            // Compute the real part
            T C_real_local = 0;
            for (int j = 0; j < L; ++j) {
                T A_real_val = A_real[m * (N * L) + i * L + j];
                T B_real_val = B_real[m * L + j];
                C_real_local += A_real_val * B_real_val;
            }
            C_real[idx] = C_real_local;
        }

        // Dual part computation
        if (idx < total_dual_elements) {
            // Decode indices for dual output
            int m = idx / (N * D);              // Batch index
            int local_idx = idx % (N * D);
            int i = local_idx / D;              // Row index
            int k = local_idx % D;              // Dual dimension index

            // Compute the dual part
            T C_dual_local = 0;
            for (int j = 0; j < L; ++j) {
                T A_real_val = A_real[m * (N * L) + i * L + j];
                T B_dual_val = B_dual[m * (L * D) + j * D + k];
                T A_dual_val = A_dual[m * (N * L * D) + i * (L * D) + j * D + k];
                T B_real_val = B_real[m * L + j];

                C_dual_local += A_real_val * B_dual_val + A_dual_val * B_real_val;
            }
            C_dual[idx] = C_dual_local;
        }
    }






    // Accessors for dimensions
    int batch_size() const { return batch_size_; }
    int rows() const { return rows_; }
    int cols() const { return cols_; }
    int dual_dim() const { return dual_dim_; }

    // Accessors for data pointers
    ComplexT* primal_data() { return primal_data_; }
    const ComplexT* primal_data() const { return primal_data_; }
    ComplexT* dual_data() { return dual_data_; }
    const ComplexT* dual_data() const { return dual_data_; }

private:
    void initializeHandles() {
        if (cublasCreate(&handle_) != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("Failed to create cuBLAS handle.");
        }
        if (cudaStreamCreate(&stream_) != cudaSuccess) {
            cublasDestroy(handle_);
            throw std::runtime_error("Failed to create CUDA stream.");
        }
        cublasSetStream(handle_, stream_);
    }

public:
    // Disable copy constructor and copy assignment
    MatrixDualDenseCuda(const MatrixDualDenseCuda&) = delete;
    MatrixDualDenseCuda& operator=(const MatrixDualDenseCuda&) = delete;

    // Enable move constructor and move assignment
    MatrixDualDenseCuda(MatrixDualDenseCuda&&) noexcept = default;
    MatrixDualDenseCuda& operator=(MatrixDualDenseCuda&&) noexcept = default;
};



template <typename T>
class MatrixHyperDualDenseCuda {
private:
    using ComplexT = std::complex<T>;

    // Dimensions
    int batch_size_;
    int rows_;
    int cols_;
    int dual_dim_;

    // Primal, Dual, and Hyper-Dual Data
    ComplexT* primal_data_;     // [M, N, L]
    ComplexT* dual_data_;       // [M, N, L, D]
    ComplexT* hyper_dual_data_; // [M, N, L, D, D]
    bool owns_memory_;          // Indicates if memory is managed internally

    // cuBLAS handle
    cublasHandle_t handle_;
    cudaStream_t stream_;

public:
    // Constructor with external memory
    MatrixHyperDualDenseCuda(int batch_size, int rows, int cols, int dual_dim,
                           ComplexT* primal_data, ComplexT* dual_data, ComplexT* hyper_dual_data)
        : batch_size_(batch_size), rows_(rows), cols_(cols), dual_dim_(dual_dim),
          primal_data_(primal_data), dual_data_(dual_data), hyper_dual_data_(hyper_dual_data),
          owns_memory_(false) {
        if (!primal_data_ || !dual_data_ || !hyper_dual_data_) {
            throw std::invalid_argument("Primal, dual, or hyper-dual data pointer is null");
        }

        initializeHandles();
    }

    // Constructor with internal memory allocation
    MatrixHyperDualDenseCuda(int batch_size, int rows, int cols, int dual_dim)
        : batch_size_(batch_size), rows_(rows), cols_(cols), dual_dim_(dual_dim), owns_memory_(true) {
        if (batch_size <= 0 || rows <= 0 || cols <= 0 || dual_dim <= 0) {
            throw std::invalid_argument("All dimensions must be positive.");
        }

        size_t primal_size = batch_size * rows * cols * sizeof(ComplexT);
        size_t dual_size = batch_size * rows * cols * dual_dim * sizeof(ComplexT);
        size_t hyper_dual_size = batch_size * rows * cols * dual_dim * dual_dim * sizeof(ComplexT);

        if (cudaMalloc(&primal_data_, primal_size) != cudaSuccess) {
            throw std::runtime_error("Failed to allocate GPU memory for primal data.");
        }

        if (cudaMalloc(&dual_data_, dual_size) != cudaSuccess) {
            cudaFree(primal_data_);
            throw std::runtime_error("Failed to allocate GPU memory for dual data.");
        }

        if (cudaMalloc(&hyper_dual_data_, hyper_dual_size) != cudaSuccess) {
            cudaFree(primal_data_);
            cudaFree(dual_data_);
            throw std::runtime_error("Failed to allocate GPU memory for hyper-dual data.");
        }

        initializeHandles();
    }

    ~MatrixHyperDualDenseCuda() {
        if (owns_memory_) {
            if (primal_data_) cudaFree(primal_data_);
            if (dual_data_) cudaFree(dual_data_);
            if (hyper_dual_data_) cudaFree(hyper_dual_data_);
        }
        cublasDestroy(handle_);
        cudaStreamDestroy(stream_);
    }

    void initialize(const ComplexT* primal, const ComplexT* dual, const ComplexT* hyper_dual,
                    size_t primal_size, size_t dual_size, size_t hyper_dual_size) {
        if (primal_size != batch_size_ * rows_ * cols_ ||
            dual_size != batch_size_ * rows_ * cols_ * dual_dim_ ||
            hyper_dual_size != batch_size_ * rows_ * cols_ * dual_dim_ * dual_dim_) {
            throw std::invalid_argument("Input sizes do not match tensor dimensions.");
        }

        if (primal) {
            cudaMemcpyAsync(primal_data_, primal, primal_size * sizeof(ComplexT), cudaMemcpyHostToDevice, stream_);
        }
        if (dual) {
            cudaMemcpyAsync(dual_data_, dual, dual_size * sizeof(ComplexT), cudaMemcpyHostToDevice, stream_);
        }
        if (hyper_dual) {
            cudaMemcpyAsync(hyper_dual_data_, hyper_dual, hyper_dual_size * sizeof(ComplexT), cudaMemcpyHostToDevice, stream_);
        }
        cudaStreamSynchronize(stream_);
    }

    template <typename U>
    MatrixHyperDualDenseCuda<U> indexGet(int start_row, int end_row, int start_col, int end_col) const {
        // Validate row and column ranges
        if (start_row < 0 || end_row > rows_ || start_row >= end_row ||
            start_col < 0 || end_col > cols_ || start_col >= end_col) {
            throw std::invalid_argument("Invalid row or column range for indexGet.");
        }

        // Dimensions of the submatrix
        int sub_rows = end_row - start_row;
        int sub_cols = end_col - start_col;

        // Calculate offsets for the primal, dual, and hyper-dual data
        ComplexT* sub_primal_data = primal_data_ + start_row * cols_ + start_col;
        ComplexT* sub_dual_data = dual_data_ + start_row * cols_ * dual_dim_ + start_col * dual_dim_;
        ComplexT* sub_hyper_dual_data = hyper_dual_data_ +
                                        start_row * cols_ * dual_dim_ * dual_dim_ +
                                        start_col * dual_dim_ * dual_dim_;

        // Create a new MatrixHyperDualDense instance sharing the data with the original
        return MatrixHyperDualDenseCuda<T>(batch_size_, sub_rows, sub_cols, dual_dim_,
                                    sub_primal_data, sub_dual_data, sub_hyper_dual_data);
    }


    template <typename U>
    void indexPut(int start_row, int end_row, int start_col, int end_col, const MatrixHyperDualDenseCuda<U>& data) {
        // Validate row and column ranges
        if (start_row < 0 || end_row > rows_ || start_row >= end_row ||
            start_col < 0 || end_col > cols_ || start_col >= end_col) {
            throw std::invalid_argument("Invalid row or column range for indexPut.");
        }

        // Validate dimensions of the input data
        if (data.rows_ != (end_row - start_row) || data.cols_ != (end_col - start_col) ||
            data.dual_dim_ != dual_dim_ || data.batch_size_ != batch_size_) {
            throw std::invalid_argument("Input data dimensions do not match the target range.");
        }

        // Calculate offsets for the primal, dual, and hyper-dual data
        ComplexT* target_primal_data = primal_data_ + start_row * cols_ + start_col;
        ComplexT* target_dual_data = dual_data_ + start_row * cols_ * dual_dim_ + start_col * dual_dim_;
        ComplexT* target_hyper_dual_data = hyper_dual_data_ +
                                        start_row * cols_ * dual_dim_ * dual_dim_ +
                                        start_col * dual_dim_ * dual_dim_;

        // Update primal data
        for (int b = 0; b < batch_size_; ++b) {
            cudaMemcpy2DAsync(target_primal_data + b * rows_ * cols_,
                            cols_ * sizeof(ComplexT),
                            data.primal_data_ + b * data.rows_ * data.cols_,
                            data.cols_ * sizeof(ComplexT),
                            data.cols_ * sizeof(ComplexT),
                            data.rows_,
                            cudaMemcpyDeviceToDevice,
                            stream_);
        }

        // Update dual data
        for (int b = 0; b < batch_size_; ++b) {
            cudaMemcpy2DAsync(target_dual_data + b * rows_ * cols_ * dual_dim_,
                            cols_ * dual_dim_ * sizeof(ComplexT),
                            data.dual_data_ + b * data.rows_ * data.cols_ * dual_dim_,
                            data.cols_ * dual_dim_ * sizeof(ComplexT),
                            data.cols_ * dual_dim_ * sizeof(ComplexT),
                            data.rows_,
                            cudaMemcpyDeviceToDevice,
                            stream_);
        }

        // Update hyper-dual data
        for (int b = 0; b < batch_size_; ++b) {
            cudaMemcpy2DAsync(target_hyper_dual_data + b * rows_ * cols_ * dual_dim_ * dual_dim_,
                            cols_ * dual_dim_ * dual_dim_ * sizeof(ComplexT),
                            data.hyper_dual_data_ + b * data.rows_ * data.cols_ * dual_dim_ * dual_dim_,
                            data.cols_ * dual_dim_ * dual_dim_ * sizeof(ComplexT),
                            data.cols_ * dual_dim_ * dual_dim_ * sizeof(ComplexT),
                            data.rows_,
                            cudaMemcpyDeviceToDevice,
                            stream_);
        }

        // Synchronize to ensure data transfer is complete
        cudaStreamSynchronize(stream_);
    }

    MatrixHyperDualDenseCuda<T> transpose() const {
        MatrixHyperDualDenseCuda<T> result(batch_size_, cols_, rows_, dual_dim_);
        ComplexT alpha = ComplexT(1.0, 0.0);
        ComplexT beta = ComplexT(0.0, 0.0);

        // Transpose primal part
        for (int b = 0; b < batch_size_; ++b) {
            cublasZgeam(handle_,
                        CUBLAS_OP_T, CUBLAS_OP_T,
                        cols_, rows_,
                        &alpha,
                        primal_data_ + b * rows_ * cols_, rows_,
                        &beta,
                        nullptr, cols_,
                        result.primal_data_ + b * cols_ * rows_, cols_);
        }

        // Transpose dual part
        for (int b = 0; b < batch_size_; ++b) {
            for (int d = 0; d < dual_dim_; ++d) {
                cublasZgeam(handle_,
                            CUBLAS_OP_T, CUBLAS_OP_T,
                            cols_, rows_,
                            &alpha,
                            dual_data_ + b * rows_ * cols_ * dual_dim_ + d * rows_ * cols_, rows_,
                            &beta,
                            nullptr, cols_,
                            result.dual_data_ + b * cols_ * rows_ * dual_dim_ + d * cols_ * rows_, cols_);
            }
        }

        // Transpose hyper-dual part
        for (int b = 0; b < batch_size_; ++b) {
            for (int d1 = 0; d1 < dual_dim_; ++d1) {
                for (int d2 = 0; d2 < dual_dim_; ++d2) {
                    cublasZgeam(handle_,
                                CUBLAS_OP_T, CUBLAS_OP_T,
                                cols_, rows_,
                                &alpha,
                                hyper_dual_data_ + b * rows_ * cols_ * dual_dim_ * dual_dim_ +
                                    d1 * rows_ * cols_ * dual_dim_ + d2 * rows_ * cols_,
                                rows_,
                                &beta,
                                nullptr, cols_,
                                result.hyper_dual_data_ + b * cols_ * rows_ * dual_dim_ * dual_dim_ +
                                    d1 * cols_ * rows_ * dual_dim_ + d2 * cols_ * rows_,
                                cols_);
                }
            }
        }

        return result;
    }

    MatrixHyperDualDenseCuda<T> upperTriangular() const {
        // Create a result matrix with the same dimensions
        MatrixHyperDualDenseCuda<T> result(batch_size_, rows_, cols_, dual_dim_);

        size_t total_elements = rows_ * cols_;

        for (int b = 0; b < batch_size_; ++b) {
            // Primal data pointers for this batch
            ComplexT* batch_primal_src = primal_data_ + b * total_elements;
            ComplexT* batch_primal_dst = result.primal_data_ + b * total_elements;

            // Apply the upper triangular operation to the primal part
            thrust::for_each(
                thrust::counting_iterator<int>(0),
                thrust::counting_iterator<int>(rows_ * cols_),
                [=] __device__(int idx) {
                    int row = idx / cols_;
                    int col = idx % cols_;
                    batch_primal_dst[idx] = (row <= col) ? batch_primal_src[idx] : ComplexT(0.0, 0.0);
                });

            // Dual part is zero-initialized
            ComplexT* batch_dual_dst = result.dual_data_ + b * total_elements * dual_dim_;
            thrust::fill(thrust::device_pointer_cast(batch_dual_dst),
                        thrust::device_pointer_cast(batch_dual_dst + total_elements * dual_dim_),
                        ComplexT(0.0, 0.0));

            // Hyper-dual part is zero-initialized
            ComplexT* batch_hyper_dual_dst = result.hyper_dual_data_ + b * total_elements * dual_dim_ * dual_dim_;
            thrust::fill(thrust::device_pointer_cast(batch_hyper_dual_dst),
                        thrust::device_pointer_cast(batch_hyper_dual_dst + total_elements * dual_dim_ * dual_dim_),
                        ComplexT(0.0, 0.0));
        }

        return result;
    }

    MatrixHyperDualDenseCuda<T> lowerTriangular() const {
        // Create a result matrix with the same dimensions
        MatrixHyperDualDenseCuda<T> result(batch_size_, rows_, cols_, dual_dim_);

        size_t total_elements = rows_ * cols_;

        for (int b = 0; b < batch_size_; ++b) {
            // Primal data pointers for this batch
            ComplexT* batch_primal_src = primal_data_ + b * total_elements;
            ComplexT* batch_primal_dst = result.primal_data_ + b * total_elements;

            // Apply the lower triangular operation to the primal part
            thrust::for_each(
                thrust::counting_iterator<int>(0),
                thrust::counting_iterator<int>(rows_ * cols_),
                [=] __device__(int idx) {
                    int row = idx / cols_;
                    int col = idx % cols_;
                    batch_primal_dst[idx] = (row >= col) ? batch_primal_src[idx] : ComplexT(0.0, 0.0);
                });

            // Dual part is zero-initialized
            ComplexT* batch_dual_dst = result.dual_data_ + b * total_elements * dual_dim_;
            thrust::fill(thrust::device_pointer_cast(batch_dual_dst),
                        thrust::device_pointer_cast(batch_dual_dst + total_elements * dual_dim_),
                        ComplexT(0.0, 0.0));

            // Hyper-dual part is zero-initialized
            ComplexT* batch_hyper_dual_dst = result.hyper_dual_data_ + b * total_elements * dual_dim_ * dual_dim_;
            thrust::fill(thrust::device_pointer_cast(batch_hyper_dual_dst),
                        thrust::device_pointer_cast(batch_hyper_dual_dst + total_elements * dual_dim_ * dual_dim_),
                        ComplexT(0.0, 0.0));
        }

        return result;
    }
    template <typename U>
    MatrixHyperDualDenseCuda<U> matrixMultiply(const MatrixHyperDualDenseCuda<U>& other) const {
        // Validate dimensions
        if (cols_ != other.rows_ || batch_size_ != other.batch_size_ || dual_dim_ != other.dual_dim_) {
            throw std::invalid_argument("Matrix dimensions do not match for multiplication.");
        }

        // Create the result matrix
        MatrixHyperDualDenseCuda<T> result(batch_size_, rows_, other.cols_, dual_dim_);

        ComplexT alpha = ComplexT(1.0, 0.0);
        ComplexT beta = ComplexT(0.0, 0.0);

        for (int b = 0; b < batch_size_; ++b) {
            // Real part: Primal * Primal
            cublasZgemm(handle_,
                        CUBLAS_OP_N, CUBLAS_OP_N,
                        rows_, other.cols_, cols_,
                        &alpha,
                        primal_data_ + b * rows_ * cols_, rows_,
                        other.primal_data_ + b * other.rows_ * other.cols_, other.rows_,
                        &beta,
                        result.primal_data_ + b * rows_ * other.cols_, rows_);

            // Dual part
            for (int d = 0; d < dual_dim_; ++d) {
                // Primal * Dual(other) + Dual * Primal(other)
                cublasZgemm(handle_,
                            CUBLAS_OP_N, CUBLAS_OP_N,
                            rows_, other.cols_, cols_,
                            &alpha,
                            primal_data_ + b * rows_ * cols_, rows_,
                            other.dual_data_ + b * other.rows_ * other.cols_ * dual_dim_ + d * other.rows_ * other.cols_, other.rows_,
                            &beta,
                            result.dual_data_ + b * rows_ * other.cols_ * dual_dim_ + d * rows_ * other.cols_, rows_);

                cublasZgemm(handle_,
                            CUBLAS_OP_N, CUBLAS_OP_N,
                            rows_, other.cols_, cols_,
                            &alpha,
                            dual_data_ + b * rows_ * cols_ * dual_dim_ + d * rows_ * cols_, rows_,
                            other.primal_data_ + b * other.rows_ * other.cols_, other.rows_,
                            &alpha, // Accumulate
                            result.dual_data_ + b * rows_ * other.cols_ * dual_dim_ + d * rows_ * other.cols_, rows_);
            }

            // Hyper-dual part
            for (int d1 = 0; d1 < dual_dim_; ++d1) {
                for (int d2 = 0; d2 < dual_dim_; ++d2) {
                    // (Primal * Hyper-Dual(other)) + (Dual * Dual(other)) + (Hyper-Dual * Primal(other))
                    cublasZgemm(handle_,
                                CUBLAS_OP_N, CUBLAS_OP_N,
                                rows_, other.cols_, cols_,
                                &alpha,
                                primal_data_ + b * rows_ * cols_, rows_,
                                other.hyper_dual_data_ + b * other.rows_ * other.cols_ * dual_dim_ * dual_dim_ +
                                    d1 * other.rows_ * other.cols_ * dual_dim_ + d2 * other.rows_ * other.cols_, other.rows_,
                                &beta,
                                result.hyper_dual_data_ + b * rows_ * other.cols_ * dual_dim_ * dual_dim_ +
                                    d1 * rows_ * other.cols_ * dual_dim_ + d2 * rows_ * other.cols_, rows_);

                    cublasZgemm(handle_,
                                CUBLAS_OP_N, CUBLAS_OP_N,
                                rows_, other.cols_, cols_,
                                &alpha,
                                dual_data_ + b * rows_ * cols_ * dual_dim_ + d1 * rows_ * cols_, rows_,
                                other.dual_data_ + b * other.rows_ * other.cols_ * dual_dim_ + d2 * other.rows_ * other.cols_, other.rows_,
                                &alpha, // Accumulate
                                result.hyper_dual_data_ + b * rows_ * other.cols_ * dual_dim_ * dual_dim_ +
                                    d1 * rows_ * other.cols_ * dual_dim_ + d2 * rows_ * other.cols_, rows_);

                    cublasZgemm(handle_,
                                CUBLAS_OP_N, CUBLAS_OP_N,
                                rows_, other.cols_, cols_,
                                &alpha,
                                hyper_dual_data_ + b * rows_ * cols_ * dual_dim_ * dual_dim_ +
                                    d1 * rows_ * cols_ * dual_dim_ + d2 * rows_ * cols_, rows_,
                                other.primal_data_ + b * other.rows_ * other.cols_, other.rows_,
                                &alpha, // Accumulate
                                result.hyper_dual_data_ + b * rows_ * other.cols_ * dual_dim_ * dual_dim_ +
                                    d1 * rows_ * other.cols_ * dual_dim_ + d2 * rows_ * other.cols_, rows_);
                }
            }
        }

        cudaStreamSynchronize(stream_);
        return result;
    }



private:
    void initializeHandles() {
        if (cublasCreate(&handle_) != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("Failed to create cuBLAS handle.");
        }
        if (cudaStreamCreate(&stream_) != cudaSuccess) {
            cublasDestroy(handle_);
            throw std::runtime_error("Failed to create CUDA stream.");
        }
        cublasSetStream(handle_, stream_);
    }

public:
    // Disable copy constructor and copy assignment
    MatrixHyperDualDenseCuda(const MatrixHyperDualDenseCuda&) = delete;
    MatrixHyperDualDenseCuda& operator=(const MatrixHyperDualDenseCuda&) = delete;

    // Enable move constructor and move assignment
    MatrixHyperDualDenseCuda(MatrixHyperDualDenseCuda&&) noexcept = default;
    MatrixHyperDualDenseCuda& operator=(MatrixHyperDualDenseCuda&&) noexcept = default;

    // Getters for dimensions
    int batch_size() const { return batch_size_; }
    int rows() const { return rows_; }
    int cols() const { return cols_; }
    int dual_dim() const { return dual_dim_; }

    // Getters for data pointers
    ComplexT* primal_data() { return primal_data_; }
    ComplexT* dual_data() { return dual_data_; }
    ComplexT* hyper_dual_data() { return hyper_dual_data_; }

    const ComplexT* primal_data() const { return primal_data_; }
    const ComplexT* dual_data() const { return dual_data_; }
    const ComplexT* hyper_dual_data() const { return hyper_dual_data_; }
}; // class MatrixHyperDualDense






}  // namespace janus
#endif // _CU_DUAL_TENSOR_HPP