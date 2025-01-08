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


template <typename T>
class VectorDenseCuda {
private:
    int batch_size_;  // Batch dimension (M)
    int size_;        // Vector length (N)

    thrust::complex<T>* data_;  // Data [M, N] (complex)
    bool owns_memory_;          // Indicates if memory is managed internally

public:
    // Constructor with external memory
    VectorDenseCuda(int batch_size, int size, thrust::complex<T>* data)
        : batch_size_(batch_size), size_(size), data_(data), owns_memory_(false) {
        if (!data_) {
            throw std::invalid_argument("Data pointer is null");
        }
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
    }

    // Destructor
    ~VectorDenseCuda() {
        if (owns_memory_ && data_) {
            cudaFree(data_);
            data_ = nullptr;
        }
    }

    // Initialize data from host
    void initialize(const thrust::complex<T>* host_data, size_t data_size) {
        if (data_size != batch_size_ * size_) {
            throw std::invalid_argument("Input size does not match vector dimensions.");
        }

        if (cudaMemcpy(data_, host_data, data_size * sizeof(thrust::complex<T>), cudaMemcpyHostToDevice) != cudaSuccess) {
            throw std::runtime_error("Failed to copy data from host to device.");
        }
    }

    // Elementwise addition
    VectorDenseCuda elementwiseAdd(const VectorDenseCuda& other) const {

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

    VectorDenseCuda<T> indexGet(int start_real, int end_real) const {
 
        // Dimensions of the subvector
        int sub_size = end_real - start_real;

        // Calculate offsets for the data
        thrust::complex<T>* sub_data = data_ + start_real;

        // Create a new VectorDenseCuda instance sharing the data with the original
        return VectorDenseCuda<T>(batch_size_, sub_size, sub_data);
    }

    void indexPut(int start_real, int end_real, const VectorDenseCuda& subvector) {
 
        // Validate subvector dimensions
        int sub_size = end_real - start_real;
 
        // Calculate offsets for the data
        thrust::complex<T>* target_data = data_ + start_real;

        // Update the data
        cudaMemcpy(
            target_data,
            subvector.data(),
            sub_size * batch_size_ * sizeof(thrust::complex<T>),
            cudaMemcpyDeviceToDevice);
    }

    // Accessors
    thrust::complex<T>* data() { return data_; }
    const thrust::complex<T>* data() const { return data_; }
    int batchSize() const { return batch_size_; }
    int size() const { return size_; }
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
    // Constructor for externally managed memory
    __host__ __device__ VectorDualDenseCuda(int batch_size, int real_size, int dual_size, 
                                            thrust::complex<T>* real, thrust::complex<T>* dual)
        : batch_size_(batch_size), real_size_(real_size), dual_size_(dual_size), real_(real), dual_(dual) {}

    // Elementwise addition on GPU
    __device__ void elementwiseAdd(const VectorDualDenseCuda& other, VectorDualDenseCuda& result) const {
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
    __device__ void elementwiseMultiply(const VectorDualDenseCuda& other, VectorDualDenseCuda& result) const {
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
            int real_idx = idx / dual_dim;  // Map dual index to corresponding real index
            result.dual_[idx] = real_[real_idx] * other.dual_[idx] +
                                other.real_[real_idx] * dual_[idx];
        }
    }

    // Kernel wrapper for elementwise addition
    __host__ void elementwiseAddKernelLaunch(const VectorDualDenseCuda& other, VectorDualDenseCuda& result) const {
        int total_elements = max(batch_size_ * real_size_, batch_size_ * real_size_ * dual_size_);
        int threads_per_block = 256;
        int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;

        elementwiseAdd<<<num_blocks, threads_per_block>>>(*this, other, result);
        cudaDeviceSynchronize();
    }

    // Kernel wrapper for elementwise multiplication
    __host__ void elementwiseMultiplyKernelLaunch(const VectorDualDenseCuda& other, VectorDualDenseCuda& result) const {
        int total_elements = max(batch_size_ * real_size_, batch_size_ * real_size_ * dual_size_);
        int threads_per_block = 256;
        int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;

        elementwiseMultiply<<<num_blocks, threads_per_block>>>(*this, other, result);
        cudaDeviceSynchronize();
    }

    template <typename U>
    VectorDualDenseCuda<U> indexGet(int start_real, int end_real) const {
        // Validate real vector range
        if (start_real < 0 || end_real > real_size_ || start_real >= end_real) {
            throw std::invalid_argument("Invalid real vector range for indexGet.");
        }

        // Dimensions of the subvector
        int sub_real_size = end_real - start_real;

        // Calculate offsets for the real and dual data
        thrust::complex<U>* sub_real_data = real_ + start_real;
        thrust::complex<U>* sub_dual_data = dual_ + start_real * dual_size_;

        // Create a new VectorDualDenseCuda instance sharing the data with the original
        return VectorDualDenseCuda<U>(batch_size_, sub_real_size, dual_size_, sub_real_data, sub_dual_data);
    }

    template <typename U>
    void indexPut(int start_real, int end_real, const VectorDualDenseCuda<U>& subvector) {
        // Validate real vector range
        if (start_real < 0 || end_real > real_size_ || start_real >= end_real) {
            throw std::invalid_argument("Invalid real vector range for indexPut.");
        }

        // Validate subvector dimensions
        int sub_real_size = end_real - start_real;
        if (subvector.size() != sub_real_size || subvector.dualSize() != dual_size_ || subvector.batchSize() != batch_size_) {
            throw std::invalid_argument("Subvector dimensions do not match the target range.");
        }

        // Calculate offsets for the real and dual data
        thrust::complex<U>* target_real_data = real_ + start_real;
        thrust::complex<U>* target_dual_data = dual_ + start_real * dual_size_;

        // Update the real part
        cudaMemcpy(
            target_real_data,
            subvector.real(),
            sub_real_size * batch_size_ * sizeof(thrust::complex<U>),
            cudaMemcpyDeviceToDevice);

        // Update the dual part
        cudaMemcpy(
            target_dual_data,
            subvector.dual(),
            sub_real_size * batch_size_ * dual_size_ * sizeof(thrust::complex<U>),
            cudaMemcpyDeviceToDevice);
    }
    // Accessors
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

    __device__ void elementwiseMultiply(const VectorHyperDualDenseCuda& other, 
                                        VectorHyperDualDenseCuda& result) const {
        int idx = threadIdx.x + blockIdx.x * blockDim.x; //1D index
        //This index is the hyperdual number out of which we can 
        //extract the batch, real, dual and hyperdual indices
        //Use the standard convention
        int M = batch_size_;
        int N = real_size_;
        int D = dual_size_;
        //Only consider values less the largest number of elements
        //which is the number of elements of the hyperdual tensor
        //the size of the hyperdual tensor is M*N*D*D
        if ( idx < M*N*D*D)
        {
          //Now extract the indexes of the hyperdual tensor
          //assuming row based indexing where the last index changes the fastest
          //l is between [0,D), k is between [0,D), i is between [0,N), m is between [0,M)]
          int m = idx / (N * D * D);             // Batch index
          //printf("m: %d\n", m);
          int i = (idx / (D * D)) % N;           // Real index
          //printf("i: %d\n", i);
          int k = (idx / D) % D;                 // Dual index
          //printf("k: %d\n", k);
          int l = idx % D;                       // Column index in the Hessian
          //printf("l: %d\n", l);
          //Cache the real and dual parts from global memory and name to correspond to 
          //einstein sum notation
          auto r1mi  = real_[m*N+i];
          auto r2mi  = other.real_[m*N+i];
          auto d1mik = dual_[m*N*D + i*D + k];
          auto d1mil = dual_[m*N*D + i*D + l];
          auto d2mil = other.dual_[m*N*D + i*D + l];
          auto d2mik = other.dual_[m*N*D + i*D + k];
          //printf("%d, %d, %d, %d\n", m, i, k, l);
          //printf("m*N*D*D + i*D*D + k*D + l: %d\n", m*N*D*D + i*D*D + k*D + l);
          auto h1mikl = hyperdual_[m*N*D*D + i*D*D + k*D + l];
          auto h2mikl = other.hyperdual_[m*N*D*D + i*D*D + k*D + l];
          //only update the real part if k==0 and l==0 to avoid repeating the real part
          if (k == 0 && l == 0) {
            int ridx = m*N+i;
            //("mi, mi->mi", r1, r2)
            result.real_[ridx] = r1mi * r2mi;
          }
          
          // Update the dual part only if l == 0 to avoid repeating the dual part
          if (l == 0) {
           int dual_idx = m*N*D + i*D + k;
           //This should be einsum("mi,mik->mik", real1, dual2)+einsum("mi,mik->mik", real2, dual1)
           //There are two dual values corresponding to the hyperdual tensor.  Just update the k part
           result.dual_[dual_idx] = r1mi * d2mik +
          
                                    r2mi * d1mik;
          }
          //In einsum notation this is ("mik,mil->mikl", d1, d2)+("mi,mikl->mikl", r1, h2)+("mi,mikl->mikl", r2, h1)
          //printf("Real values r1mi: %f, r2mi: %f, d1mik: %f, d1mil: %f, d2mil: %f, d2mik: %f, h1mikl: %f, h2mikl: %f\n", r1mi.real(), r2mi.real(), d1mik.real(), d1mil.real(), d2mil.real(), d2mik.real(), h1mikl.real(), h2mikl.real());
          //printf("Imaginary values r1mi: %f, r2mi: %f, d1mik: %f, d1mil: %f, d2mil: %f, d2mik: %f, h1mikl: %f, h2mikl: %f\n", r1mi.imag(), r2mi.imag(), d1mik.imag(), d1mil.imag(), d2mil.imag(), d2mik.imag(), h1mikl.imag(), h2mikl.imag());
          result.hyperdual_[idx] = d1mik*d2mil +
                                   d1mil*d2mik +
                                   r1mi*h2mikl +
                                   r2mi*h1mikl;
 

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
   
   /**
    * Multiplies two matrices with dual part on GPU.
    * The matrices are stored in row-major order.
    * The dual part is stored in a separate tensor with dimensions [M, N, P, D].
    */
    __device__ void matrixMultiplyDualOptimized(const T* A_real, const T* A_dual,
                                                const T* B_real, const T* B_dual,
                                                T* C_real, T* C_dual,
                                                int M, int N, int L, int P, int D) {
        extern __shared__ T shared_mem[];  // Shared memory for caching B_real and B_dual
        T* shared_B_real = shared_mem;     // Cache for B_real
        T* shared_B_dual = shared_mem + L * P;  // Cache for B_dual

        int idx = threadIdx.x + blockIdx.x * blockDim.x;

        // Total number of real and dual elements in the output
        int total_real_elements = M * N * P;
        int total_dual_elements = M * N * P * D;

        // Block and thread indices
        int block_m = blockIdx.x;               // Batch index (M)
        int thread_i = threadIdx.y;             // Row index (N)
        int thread_k = threadIdx.x;             // Column index (P)

        // Cache B_real and B_dual into shared memory (one block per batch)
        for (int j = threadIdx.x; j < L * P; j += blockDim.x) {
            int local_j = j / P;  // Row of B
            int local_k = j % P;  // Column of B

            // Cache B_real and B_dual for the current batch
            shared_B_real[local_j * P + local_k] = B_real[block_m * (L * P) + local_j * P + local_k];
            for (int d = 0; d < D; ++d) {
                shared_B_dual[local_j * (P * D) + local_k * D + d] =
                    B_dual[block_m * (L * P * D) + local_j * (P * D) + local_k * D + d];
            }
        }
        __syncthreads();

        // Real computation
        if (idx < total_real_elements) {
            // Decode indices
            int m = idx / (N * P);
            int i = (idx % (N * P)) / P;
            int k = (idx % (N * P)) % P;

            // Accumulate real result
            T C_real_local = 0;
            for (int j = 0; j < L; ++j) {
                T A_real_val = A_real[m * (N * L) + i * L + j];
                T B_real_val = shared_B_real[j * P + k];  // Use cached value
                C_real_local += A_real_val * B_real_val;
            }
            C_real[idx] = C_real_local;
        }

        // Dual computation
        if (idx < total_dual_elements) {
            // Decode indices
            int m = idx / (N * P * D);
            int local_idx = idx % (N * P * D);
            int i = local_idx / (P * D);
            int k = (local_idx % (P * D)) / D;
            int l = (local_idx % (P * D)) % D;

            // Accumulate dual result
            T C_dual_local = 0;
            for (int j = 0; j < L; ++j) {
                T A_real_val = A_real[m * (N * L) + i * L + j];
                T A_dual_val = A_dual[m * (N * L * D) + i * (L * D) + j * D + l];
                T B_real_val = shared_B_real[j * P + k];  // Use cached value
                T B_dual_val = shared_B_dual[j * (P * D) + k * D + l];  // Use cached value

                C_dual_local += A_real_val * B_dual_val + A_dual_val * B_real_val;
            }
            C_dual[idx] = C_dual_local;
        }
    }



    /**
     * Matrix Vector multiplication with dual part on GPU.
     * The matrix is stored in row-major order.
     * The dual part is stored in a separate tensor with dimensions [M, N, D].
     */
    __device__ void matrixVectorMultiplyDualOptimized(const T* A_real, const T* A_dual,
                                                      const T* B_real, const T* B_dual,
                                                      T* C_real, T* C_dual,
                                                      int M, int N, int L, int D) {
        extern __shared__ T shared_mem[];  // Shared memory for caching
        T* shared_B_real = shared_mem;     // Cache B_real
        T* shared_B_dual = shared_mem + L; // Cache B_dual

        int m = blockIdx.x;                // Batch index
        int i = threadIdx.y;               // Row index of C
        int k = threadIdx.x;               // Dual dimension index (if applicable)

        // Initialize local accumulators
        T C_real_local = 0;
        T C_dual_local = 0;

        // Loop over shared dimension L in chunks
        for (int chunk_start = 0; chunk_start < L; chunk_start += blockDim.x) {
            int j = chunk_start + threadIdx.x;

            // Cache B_real and B_dual into shared memory
            if (j < L) {
                shared_B_real[j] = B_real[m * L + j];
                for (int d = 0; d < D; ++d) {
                    shared_B_dual[j * D + d] = B_dual[m * L * D + j * D + d];
                }
            }
            __syncthreads();

            // Accumulate contributions for this chunk
            for (int j_local = 0; j_local < blockDim.x && (chunk_start + j_local) < L; ++j_local) {
                int j = chunk_start + j_local;

                T A_real_val = A_real[m * (N * L) + i * L + j];
                T A_dual_val = A_dual[m * (N * L * D) + i * (L * D) + j * D + k];

                C_real_local += A_real_val * shared_B_real[j];
                C_dual_local += A_real_val * shared_B_dual[j * D + k] + A_dual_val * shared_B_real[j];
            }
            __syncthreads();
        }

        // Write results back to global memory
        if (threadIdx.x == 0) {
            C_real[m * N + i] = C_real_local;
        }
        C_dual[m * (N * D) + i * D + k] = C_dual_local;
    }

    void cpuMatrixMultiplyDual(const T* A_real, const T* A_dual, 
                            const T* B_real, const T* B_dual, 
                            T* C_real, T* C_dual, 
                            int M, int N, int L, int P, int D) {
        for (int m = 0; m < M; ++m) {
            for (int i = 0; i < N; ++i) {
                for (int k = 0; k < P; ++k) {
                    T real_sum = 0;
                    T dual_sum[D] = {0};
                    for (int j = 0; j < L; ++j) {
                        T A_real_val = A_real[m * (N * L) + i * L + j];
                        T B_real_val = B_real[m * (L * P) + j * P + k];
                        T A_dual_val[D], B_dual_val[D];
                        for (int d = 0; d < D; ++d) {
                            A_dual_val[d] = A_dual[m * (N * L * D) + i * (L * D) + j * D + d];
                            B_dual_val[d] = B_dual[m * (L * P * D) + j * (P * D) + k * D + d];
                        }

                        real_sum += A_real_val * B_real_val;
                        for (int d = 0; d < D; ++d) {
                            dual_sum[d] += A_real_val * B_dual_val[d] + A_dual_val[d] * B_real_val;
                        }
                    }
                    C_real[m * (N * P) + i * P + k] = real_sum;
                    for (int d = 0; d < D; ++d) {
                        C_dual[m * (N * P * D) + i * (P * D) + k * D + d] = dual_sum[d];
                    }
                }
            }
        }
    }

    template <typename U>
    MatrixDualDenseCuda<U> indexGet(int start_row, int end_row, int start_col, int end_col) const {
        // Validate row and column ranges
        if (start_row < 0 || end_row > rows_ || start_row >= end_row ||
            start_col < 0 || end_col > cols_ || start_col >= end_col) {
            throw std::invalid_argument("Invalid row or column range for indexGet.");
        }

        // Dimensions of the submatrix
        int sub_rows = end_row - start_row;
        int sub_cols = end_col - start_col;

        // Calculate the offsets for the primal and dual data
        int primal_offset = start_row * cols_ + start_col;
        int dual_offset = start_row * cols_ * dual_dim_ + start_col * dual_dim_;

        // Return a new instance sharing the data with the original
        // The new instance will point to the corresponding submatrices in primal and dual data
        return MatrixDualDenseCuda<U>(
            batch_size_, 
            sub_rows, 
            sub_cols, 
            dual_dim_, 
            primal_data_ + primal_offset, 
            dual_data_ + dual_offset
        );
    }


    MatrixDualDenseCuda<T> upperTriangular() const {
        // Create a result matrix with the same dimensions
        MatrixDualDenseCuda<T> result(batch_size_, rows_, cols_, dual_dim_);

        size_t total_primal_elements = rows_ * cols_;
        size_t total_dual_elements = rows_ * cols_ * dual_dim_;

        for (int b = 0; b < batch_size_; ++b) {
            // Pointers for the current batch
            ComplexT* batch_primal_src = primal_data_ + b * total_primal_elements;
            ComplexT* batch_primal_dst = result.primal_data_ + b * total_primal_elements;

            ComplexT* batch_dual_src = dual_data_ + b * total_dual_elements;
            ComplexT* batch_dual_dst = result.dual_data_ + b * total_dual_elements;

            // Apply the upper triangular operation on the primal part
            thrust::for_each(
                thrust::counting_iterator<int>(0),
                thrust::counting_iterator<int>(rows_ * cols_),
                [=] __device__(int idx) {
                    int row = idx / cols_;
                    int col = idx % cols_;
                    batch_primal_dst[idx] = (row <= col) ? batch_primal_src[idx] : ComplexT(0.0, 0.0);
                });

            // Apply the upper triangular operation on the dual part
            thrust::for_each(
                thrust::counting_iterator<int>(0),
                thrust::counting_iterator<int>(rows_ * cols_ * dual_dim_),
                [=] __device__(int idx) {
                    int row = (idx / (cols_ * dual_dim_)) % rows_;
                    int col = (idx / dual_dim_) % cols_;
                    batch_dual_dst[idx] = (row <= col) ? batch_dual_src[idx] : ComplexT(0.0, 0.0);
                });
        }

        return result;
    }

    MatrixDualDenseCuda<T> lowerTriangular() const {
        // Create a result matrix with the same dimensions
        MatrixDualDenseCuda<T> result(batch_size_, rows_, cols_, dual_dim_);

        size_t total_primal_elements = rows_ * cols_;
        size_t total_dual_elements = rows_ * cols_ * dual_dim_;

        for (int b = 0; b < batch_size_; ++b) {
            // Pointers for the current batch
            ComplexT* batch_primal_src = primal_data_ + b * total_primal_elements;
            ComplexT* batch_primal_dst = result.primal_data_ + b * total_primal_elements;

            ComplexT* batch_dual_src = dual_data_ + b * total_dual_elements;
            ComplexT* batch_dual_dst = result.dual_data_ + b * total_dual_elements;

            // Apply the lower triangular operation on the primal part
            thrust::for_each(
                thrust::counting_iterator<int>(0),
                thrust::counting_iterator<int>(rows_ * cols_),
                [=] __device__(int idx) {
                    int row = idx / cols_;
                    int col = idx % cols_;
                    batch_primal_dst[idx] = (row >= col) ? batch_primal_src[idx] : ComplexT(0.0, 0.0);
                });

            // Apply the lower triangular operation on the dual part
            thrust::for_each(
                thrust::counting_iterator<int>(0),
                thrust::counting_iterator<int>(rows_ * cols_ * dual_dim_),
                [=] __device__(int idx) {
                    int row = (idx / (cols_ * dual_dim_)) % rows_;
                    int col = (idx / dual_dim_) % cols_;
                    batch_dual_dst[idx] = (row >= col) ? batch_dual_src[idx] : ComplexT(0.0, 0.0);
                });
        }

        return result;
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
}; // class MatrixDualDenseCuda


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

public:
    // Constructor with external memory
    MatrixHyperDualDenseCuda(int batch_size, int rows, int cols, int dual_dim,
                             ComplexT* primal_data, ComplexT* dual_data, ComplexT* hyper_dual_data)
        : batch_size_(batch_size), rows_(rows), cols_(cols), dual_dim_(dual_dim),
          primal_data_(primal_data), dual_data_(dual_data), hyper_dual_data_(hyper_dual_data),
          owns_memory_(false) {
        if (!primal_data_ || !dual_data_ || !hyper_dual_data_) {
            throw std::invalid_argument("Primal, dual, or hyper-dual data pointer is null.");
        }
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
    }

    ~MatrixHyperDualDenseCuda() {
        if (owns_memory_) {
            if (primal_data_) {
                cudaFree(primal_data_);
                primal_data_ = nullptr;
            }
            if (dual_data_) {
                cudaFree(dual_data_);
                dual_data_ = nullptr;
            }
            if (hyper_dual_data_) {
                cudaFree(hyper_dual_data_);
                hyper_dual_data_ = nullptr;
            }
        }
    }

    void initialize(const ComplexT* primal, const ComplexT* dual, const ComplexT* hyper_dual,
                    size_t primal_size, size_t dual_size, size_t hyper_dual_size) {
        if (primal && cudaMemcpy(primal_data_, primal, primal_size * sizeof(ComplexT), cudaMemcpyHostToDevice) != cudaSuccess) {
            throw std::runtime_error("Failed to copy primal data to GPU.");
        }
        if (dual && cudaMemcpy(dual_data_, dual, dual_size * sizeof(ComplexT), cudaMemcpyHostToDevice) != cudaSuccess) {
            throw std::runtime_error("Failed to copy dual data to GPU.");
        }
        if (hyper_dual && cudaMemcpy(hyper_dual_data_, hyper_dual, hyper_dual_size * sizeof(ComplexT), cudaMemcpyHostToDevice) != cudaSuccess) {
            throw std::runtime_error("Failed to copy hyper-dual data to GPU.");
        }
    }

    MatrixHyperDualDenseCuda<T> transpose() const {
        // Create a result matrix with transposed dimensions
        MatrixHyperDualDenseCuda<T> result(batch_size_, cols_, rows_, dual_dim_);

        // Transpose the primal part
        for (int b = 0; b < batch_size_; ++b) {
            size_t primal_offset = b * rows_ * cols_;
            transposeKernel<<<(rows_ * cols_ + 255) / 256, 256>>>(
                primal_data_ + primal_offset, result.primal_data_ + primal_offset, rows_, cols_);
        }

        // Transpose the dual part
        for (int b = 0; b < batch_size_; ++b) {
            size_t dual_offset = b * rows_ * cols_ * dual_dim_;
            for (int d = 0; d < dual_dim_; ++d) {
                size_t single_dual_offset = dual_offset + d * rows_ * cols_;
                transposeKernel<<<(rows_ * cols_ + 255) / 256, 256>>>(
                    dual_data_ + single_dual_offset, 
                    result.dual_data_ + single_dual_offset, 
                    rows_, cols_);
            }
        }

        // Transpose the hyper-dual part
        for (int b = 0; b < batch_size_; ++b) {
            size_t hyper_dual_offset = b * rows_ * cols_ * dual_dim_ * dual_dim_;
            for (int d1 = 0; d1 < dual_dim_; ++d1) {
                for (int d2 = 0; d2 < dual_dim_; ++d2) {
                    size_t single_hyper_dual_offset = 
                        hyper_dual_offset + d1 * rows_ * cols_ * dual_dim_ + d2 * rows_ * cols_;
                    transposeKernel<<<(rows_ * cols_ + 255) / 256, 256>>>(
                        hyper_dual_data_ + single_hyper_dual_offset, 
                        result.hyper_dual_data_ + single_hyper_dual_offset, 
                        rows_, cols_);
                }
            }
        }

        return result;
    }    

    __device__ void matrixMultiplyKernel(const ComplexT* A, const ComplexT* B, ComplexT* C, int M, int K, int N) {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row < M && col < N) {
            ComplexT sum(0.0, 0.0);
            for (int i = 0; i < K; ++i) {
                sum += A[row * K + i] * B[i * N + col];
            }
            C[row * N + col] = sum;
        }
    }    

    MatrixHyperDualDenseCuda<T> matrixMultiply(const MatrixHyperDualDenseCuda<T>& other) const {
        if (cols_ != other.rows_ || batch_size_ != other.batch_size_ || dual_dim_ != other.dual_dim_) {
            throw std::invalid_argument("Matrix dimensions do not match for multiplication.");
        }

        // Create the result matrix
        MatrixHyperDualDenseCuda<T> result(batch_size_, rows_, other.cols_, dual_dim_);

        dim3 blockDim(16, 16);
        dim3 gridDim((other.cols_ + blockDim.x - 1) / blockDim.x, (rows_ + blockDim.y - 1) / blockDim.y);

        // Primal multiplication
        for (int b = 0; b < batch_size_; ++b) {
            size_t offsetA = b * rows_ * cols_;
            size_t offsetB = b * other.rows_ * other.cols_;
            size_t offsetC = b * rows_ * other.cols_;

            matrixMultiplyKernel<<<gridDim, blockDim>>>(
                primal_data_ + offsetA, other.primal_data_ + offsetB, result.primal_data_ + offsetC,
                rows_, cols_, other.cols_);
        }

        // Dual multiplication
        for (int b = 0; b < batch_size_; ++b) {
            size_t offsetA = b * rows_ * cols_ * dual_dim_;
            size_t offsetB = b * other.rows_ * other.cols_ * dual_dim_;
            size_t offsetC = b * rows_ * other.cols_ * dual_dim_;

            for (int d = 0; d < dual_dim_; ++d) {
                matrixMultiplyKernel<<<gridDim, blockDim>>>(
                    primal_data_ + b * rows_ * cols_,
                    other.dual_data_ + offsetB + d * other.rows_ * other.cols_,
                    result.dual_data_ + offsetC + d * rows_ * other.cols_,
                    rows_, cols_, other.cols_);

                matrixMultiplyKernel<<<gridDim, blockDim>>>(
                    dual_data_ + offsetA + d * rows_ * cols_,
                    other.primal_data_ + b * other.rows_ * other.cols_,
                    result.dual_data_ + offsetC + d * rows_ * other.cols_,
                    rows_, cols_, other.cols_);
            }
        }

        // Hyper-dual multiplication
        for (int b = 0; b < batch_size_; ++b) {
            size_t offsetA = b * rows_ * cols_ * dual_dim_;
            size_t offsetB = b * other.rows_ * other.cols_ * dual_dim_;
            size_t offsetC = b * rows_ * other.cols_ * dual_dim_ * dual_dim_;

            for (int d1 = 0; d1 < dual_dim_; ++d1) {
                for (int d2 = 0; d2 < dual_dim_; ++d2) {
                    matrixMultiplyKernel<<<gridDim, blockDim>>>(
                        primal_data_ + b * rows_ * cols_,
                        other.hyper_dual_data_ + b * other.rows_ * other.cols_ * dual_dim_ * dual_dim_ +
                            d1 * other.rows_ * other.cols_ * dual_dim_ + d2 * other.rows_ * other.cols_,
                        result.hyper_dual_data_ + offsetC + d1 * rows_ * other.cols_ * dual_dim_ + d2 * rows_ * other.cols_,
                        rows_, cols_, other.cols_);

                    matrixMultiplyKernel<<<gridDim, blockDim>>>(
                        dual_data_ + offsetA + d1 * rows_ * cols_,
                        other.dual_data_ + offsetB + d2 * other.rows_ * other.cols_,
                        result.hyper_dual_data_ + offsetC + d1 * rows_ * other.cols_ * dual_dim_ + d2 * rows_ * other.cols_,
                        rows_, cols_, other.cols_);

                    matrixMultiplyKernel<<<gridDim, blockDim>>>(
                        hyper_dual_data_ + b * rows_ * cols_ * dual_dim_ * dual_dim_ +
                            d1 * rows_ * cols_ * dual_dim_ + d2 * rows_ * cols_,
                        other.primal_data_ + b * other.rows_ * other.cols_,
                        result.hyper_dual_data_ + offsetC + d1 * rows_ * other.cols_ * dual_dim_ + d2 * rows_ * other.cols_,
                        rows_, cols_, other.cols_);
                }
            }
        }

        return result;
    }

    MatrixHyperDualDenseCuda<T> upperTriangular() const {
        // Create the result matrix with the same dimensions
        MatrixHyperDualDenseCuda<T> result(batch_size_, rows_, cols_, dual_dim_);
        size_t total_primal_elements = rows_ * cols_;
        size_t total_dual_elements = rows_ * cols_ * dual_dim_;
        size_t total_hyper_dual_elements = rows_ * cols_ * dual_dim_ * dual_dim_;

        // Process the primal part
        for (int b = 0; b < batch_size_; ++b) {
            ComplexT* batch_primal_src = primal_data_ + b * total_primal_elements;
            ComplexT* batch_primal_dst = result.primal_data_ + b * total_primal_elements;

            thrust::for_each(
                thrust::counting_iterator<int>(0),
                thrust::counting_iterator<int>(total_primal_elements),
                [=] __device__(int idx) {
                    int row = idx / cols_;
                    int col = idx % cols_;
                    batch_primal_dst[idx] = (row <= col) ? batch_primal_src[idx] : ComplexT(0.0, 0.0);
                });
        }

        // Process the dual part
        for (int b = 0; b < batch_size_; ++b) {
            ComplexT* batch_dual_src = dual_data_ + b * total_dual_elements;
            ComplexT* batch_dual_dst = result.dual_data_ + b * total_dual_elements;

            thrust::for_each(
                thrust::counting_iterator<int>(0),
                thrust::counting_iterator<int>(total_dual_elements),
                [=] __device__(int idx) {
                    int row = (idx / (cols_ * dual_dim_)) % rows_;
                    int col = (idx / dual_dim_) % cols_;
                    batch_dual_dst[idx] = (row <= col) ? batch_dual_src[idx] : ComplexT(0.0, 0.0);
                });
        }

        // Process the hyper-dual part
        for (int b = 0; b < batch_size_; ++b) {
            ComplexT* batch_hyper_dual_src = hyper_dual_data_ + b * total_hyper_dual_elements;
            ComplexT* batch_hyper_dual_dst = result.hyper_dual_data_ + b * total_hyper_dual_elements;

            thrust::for_each(
                thrust::counting_iterator<int>(0),
                thrust::counting_iterator<int>(total_hyper_dual_elements),
                [=] __device__(int idx) {
                    int row = (idx / (cols_ * dual_dim_ * dual_dim_)) % rows_;
                    int col = (idx / (dual_dim_ * dual_dim_)) % cols_;
                    batch_hyper_dual_dst[idx] = (row <= col) ? batch_hyper_dual_src[idx] : ComplexT(0.0, 0.0);
                });
        }

        return result;
    }

    MatrixHyperDualDenseCuda<T> lowerTriangular() const {
        // Create the result matrix with the same dimensions
        MatrixHyperDualDenseCuda<T> result(batch_size_, rows_, cols_, dual_dim_);
        size_t total_primal_elements = rows_ * cols_;
        size_t total_dual_elements = rows_ * cols_ * dual_dim_;
        size_t total_hyper_dual_elements = rows_ * cols_ * dual_dim_ * dual_dim_;

        // Process the primal part
        for (int b = 0; b < batch_size_; ++b) {
            ComplexT* batch_primal_src = primal_data_ + b * total_primal_elements;
            ComplexT* batch_primal_dst = result.primal_data_ + b * total_primal_elements;

            thrust::for_each(
                thrust::counting_iterator<int>(0),
                thrust::counting_iterator<int>(total_primal_elements),
                [=] __device__(int idx) {
                    int row = idx / cols_;
                    int col = idx % cols_;
                    batch_primal_dst[idx] = (row >= col) ? batch_primal_src[idx] : ComplexT(0.0, 0.0);
                });
        }

        // Process the dual part
        for (int b = 0; b < batch_size_; ++b) {
            ComplexT* batch_dual_src = dual_data_ + b * total_dual_elements;
            ComplexT* batch_dual_dst = result.dual_data_ + b * total_dual_elements;

            thrust::for_each(
                thrust::counting_iterator<int>(0),
                thrust::counting_iterator<int>(total_dual_elements),
                [=] __device__(int idx) {
                    int row = (idx / (cols_ * dual_dim_)) % rows_;
                    int col = (idx / dual_dim_) % cols_;
                    batch_dual_dst[idx] = (row >= col) ? batch_dual_src[idx] : ComplexT(0.0, 0.0);
                });
        }

        // Process the hyper-dual part
        for (int b = 0; b < batch_size_; ++b) {
            ComplexT* batch_hyper_dual_src = hyper_dual_data_ + b * total_hyper_dual_elements;
            ComplexT* batch_hyper_dual_dst = result.hyper_dual_data_ + b * total_hyper_dual_elements;

            thrust::for_each(
                thrust::counting_iterator<int>(0),
                thrust::counting_iterator<int>(total_hyper_dual_elements),
                [=] __device__(int idx) {
                    int row = (idx / (cols_ * dual_dim_ * dual_dim_)) % rows_;
                    int col = (idx / (dual_dim_ * dual_dim_)) % cols_;
                    batch_hyper_dual_dst[idx] = (row >= col) ? batch_hyper_dual_src[idx] : ComplexT(0.0, 0.0);
                });
        }

        return result;
    }
    
}; // class MatrixHyperDualDenseCuda

template <typename T>
class MatrixHyperDualSparseCuda {
private:
    using ComplexT = std::complex<T>;

    // Dimensions
    int batch_size_;
    int rows_;
    int cols_;
    int dual_dim_;

    // CSR Storage for Primal Part
    ComplexT* primal_values_;
    int* primal_col_indices_;
    int* primal_row_pointers_;
    int primal_nnz_;

    // CSR Storage for Dual Part
    ComplexT* dual_values_;
    int* dual_col_indices_;
    int* dual_row_pointers_;
    int dual_nnz_;

    // CSR Storage for Hyper-Dual Part
    ComplexT* hyper_dual_values_;
    int* hyper_dual_col_indices_;
    int* hyper_dual_row_pointers_;
    int hyper_dual_nnz_;

    bool owns_memory_; // Indicates if memory is managed internally

public:
    // Constructor with external memory
    MatrixHyperDualSparseCuda(
        int batch_size, int rows, int cols, int dual_dim,
        ComplexT* primal_values, int* primal_col_indices, int* primal_row_pointers, int primal_nnz,
        ComplexT* dual_values, int* dual_col_indices, int* dual_row_pointers, int dual_nnz,
        ComplexT* hyper_dual_values, int* hyper_dual_col_indices, int* hyper_dual_row_pointers, int hyper_dual_nnz)
        : batch_size_(batch_size), rows_(rows), cols_(cols), dual_dim_(dual_dim),
          primal_values_(primal_values), primal_col_indices_(primal_col_indices), primal_row_pointers_(primal_row_pointers), primal_nnz_(primal_nnz),
          dual_values_(dual_values), dual_col_indices_(dual_col_indices), dual_row_pointers_(dual_row_pointers), dual_nnz_(dual_nnz),
          hyper_dual_values_(hyper_dual_values), hyper_dual_col_indices_(hyper_dual_col_indices), hyper_dual_row_pointers_(hyper_dual_row_pointers), hyper_dual_nnz_(hyper_dual_nnz),
          owns_memory_(false) {}

    // Constructor with internal memory allocation
    MatrixHyperDualSparseCuda(int batch_size, int rows, int cols, int dual_dim, int primal_nnz, int dual_nnz, int hyper_dual_nnz)
        : batch_size_(batch_size), rows_(rows), cols_(cols), dual_dim_(dual_dim),
          primal_nnz_(primal_nnz), dual_nnz_(dual_nnz), hyper_dual_nnz_(hyper_dual_nnz), owns_memory_(true) {
        // Allocate memory for CSR components
        cudaMalloc(&primal_values_, primal_nnz_ * sizeof(ComplexT));
        cudaMalloc(&primal_col_indices_, primal_nnz_ * sizeof(int));
        cudaMalloc(&primal_row_pointers_, (rows_ + 1) * sizeof(int));

        cudaMalloc(&dual_values_, dual_nnz_ * sizeof(ComplexT));
        cudaMalloc(&dual_col_indices_, dual_nnz_ * sizeof(int));
        cudaMalloc(&dual_row_pointers_, (rows_ + 1) * sizeof(int));

        cudaMalloc(&hyper_dual_values_, hyper_dual_nnz_ * sizeof(ComplexT));
        cudaMalloc(&hyper_dual_col_indices_, hyper_dual_nnz_ * sizeof(int));
        cudaMalloc(&hyper_dual_row_pointers_, (rows_ + 1) * sizeof(int));
    }

    // Destructor
    ~MatrixHyperDualSparseCuda() {
        if (owns_memory_) {
            if (primal_values_) cudaFree(primal_values_);
            if (primal_col_indices_) cudaFree(primal_col_indices_);
            if (primal_row_pointers_) cudaFree(primal_row_pointers_);
            if (dual_values_) cudaFree(dual_values_);
            if (dual_col_indices_) cudaFree(dual_col_indices_);
            if (dual_row_pointers_) cudaFree(dual_row_pointers_);
            if (hyper_dual_values_) cudaFree(hyper_dual_values_);
            if (hyper_dual_col_indices_) cudaFree(hyper_dual_col_indices_);
            if (hyper_dual_row_pointers_) cudaFree(hyper_dual_row_pointers_);
        }
    }

    // Initialize from host data
    void initializePrimal(const ComplexT* values, const int* col_indices, const int* row_pointers) {
        cudaMemcpy(primal_values_, values, primal_nnz_ * sizeof(ComplexT), cudaMemcpyHostToDevice);
        cudaMemcpy(primal_col_indices_, col_indices, primal_nnz_ * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(primal_row_pointers_, row_pointers, (rows_ + 1) * sizeof(int), cudaMemcpyHostToDevice);
    }

    void initializeDual(const ComplexT* values, const int* col_indices, const int* row_pointers) {
        cudaMemcpy(dual_values_, values, dual_nnz_ * sizeof(ComplexT), cudaMemcpyHostToDevice);
        cudaMemcpy(dual_col_indices_, col_indices, dual_nnz_ * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(dual_row_pointers_, row_pointers, (rows_ + 1) * sizeof(int), cudaMemcpyHostToDevice);
    }

    void initializeHyperDual(const ComplexT* values, const int* col_indices, const int* row_pointers) {
        cudaMemcpy(hyper_dual_values_, values, hyper_dual_nnz_ * sizeof(ComplexT), cudaMemcpyHostToDevice);
        cudaMemcpy(hyper_dual_col_indices_, col_indices, hyper_dual_nnz_ * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(hyper_dual_row_pointers_, row_pointers, (rows_ + 1) * sizeof(int), cudaMemcpyHostToDevice);
    }

    // Elementwise addition
    MatrixHyperDualSparseCuda<T> elementwiseAdd(const MatrixHyperDualSparseCuda<T>& other) const {
        if (batch_size_ != other.batch_size_ || rows_ != other.rows_ || cols_ != other.cols_ || dual_dim_ != other.dual_dim_) {
            throw std::invalid_argument("Matrix dimensions do not match for addition.");
        }

        // Assuming result nnz is the sum of both
        int result_primal_nnz = primal_nnz_ + other.primal_nnz_;
        int result_dual_nnz = dual_nnz_ + other.dual_nnz_;
        int result_hyper_dual_nnz = hyper_dual_nnz_ + other.hyper_dual_nnz_;
        MatrixHyperDualSparseCuda<T> result(batch_size_, rows_, cols_, dual_dim_, result_primal_nnz, result_dual_nnz, result_hyper_dual_nnz);

        // Add primal parts (implement sparse CSR addition)
        sparseCSRAddKernel<<<1, 256>>>(
            primal_values_, primal_col_indices_, primal_row_pointers_, primal_nnz_,
            other.primal_values_, other.primal_col_indices_, other.primal_row_pointers_, other.primal_nnz_,
            result.primal_values_, result.primal_col_indices_, result.primal_row_pointers_, result_primal_nnz);

        // Add dual parts
        sparseCSRAddKernel<<<1, 256>>>(
            dual_values_, dual_col_indices_, dual_row_pointers_, dual_nnz_,
            other.dual_values_, other.dual_col_indices_, other.dual_row_pointers_, other.dual_nnz_,
            result.dual_values_, result.dual_col_indices_, result.dual_row_pointers_, result_dual_nnz);

        // Add hyper-dual parts
        sparseCSRAddKernel<<<1, 256>>>(
            hyper_dual_values_, hyper_dual_col_indices_, hyper_dual_row_pointers_, hyper_dual_nnz_,
            other.hyper_dual_values_, other.hyper_dual_col_indices_, other.hyper_dual_row_pointers_, other.hyper_dual_nnz_,
            result.hyper_dual_values_, result.hyper_dual_col_indices_, result.hyper_dual_row_pointers_, result_hyper_dual_nnz);

        return result;
    }

    // Accessors
    ComplexT* primalValues() { return primal_values_; }
    const ComplexT* primalValues() const { return primal_values_; }
    int* primalColIndices() { return primal_col_indices_; }
    const int* primalColIndices() const { return primal_col_indices_; }
    int* primalRowPointers() { return primal_row_pointers_; }
    const int* primalRowPointers() const { return primal_row_pointers_; }

    ComplexT* dualValues() { return dual_values_; }
    const ComplexT* dualValues() const { return dual_values_; }
    int* dualColIndices() { return dual_col_indices_; }
    const int* dualColIndices() const { return dual_col_indices_; }
    int* dualRowPointers() { return dual_row_pointers_; }
    const int* dualRowPointers() const { return dual_row_pointers_; }

    ComplexT* hyperDualValues() { return hyper_dual_values_; }
    const ComplexT* hyperDualValues() const { return hyper_dual_values_; }
    int* hyperDualColIndices() { return hyper_dual_col_indices_; }
    const int* hyperDualColIndices() const { return hyper_dual_col_indices_; }
    int* hyperDualRowPointers() { return hyper_dual_row_pointers_; }
    const int* hyperDualRowPointers() const { return hyper_dual_row_pointers_; }

    int batchSize() const { return batch_size_; }
    int rows() const { return rows_; }
    int cols() const { return cols_; }
    int dualDim() const { return dual_dim_; }
};

} // namespace Janus
#endif // _CU_DUAL_TENSOR_HPP