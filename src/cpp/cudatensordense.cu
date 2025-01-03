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
void multiplyDualTensor(const thrust::complex<T>* real,
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


template <typename T>
class VectorDualDenseCuda {
private:
    int batch_size_;             // Batch dimension (M)
    int real_size_;              // Vector length (N)
    int dual_size_;              // Dual dimension (D)

    thrust::complex<T>* real_;   // Real part [M, N]
    thrust::complex<T>* dual_;   // Dual part [M, N, D]
    bool owns_memory_;           // Indicates if memory is managed internally

    cudaStream_t stream_;        // CUDA stream for asynchronous operations

public:
    // Constructor with external memory
    VectorDualDenseCuda(int batch_size, int real_size, int dual_size,
         thrust::complex<T>* real, thrust::complex<T>* dual)
        : batch_size_(batch_size), real_size_(real_size), dual_size_(dual_size),
          real_(real), dual_(dual), owns_memory_(false) {
        if (!real_ || !dual_) {
            throw std::invalid_argument("Real or dual data pointer is null.");
        }
        initializeStream();
    }

    // Constructor with internal memory allocation
    VectorDualDenseCuda(int batch_size, int real_size, int dual_size)
        : batch_size_(batch_size), real_size_(real_size), dual_size_(dual_size), owns_memory_(true) {
        if (batch_size <= 0 || real_size <= 0 || dual_size <= 0) {
            throw std::invalid_argument("Batch size, vector size, and dual size must be positive.");
        }

        size_t real_data_size = batch_size * real_size * sizeof(thrust::complex<T>);
        size_t dual_data_size = batch_size * real_size * dual_size * sizeof(thrust::complex<T>);

        if (cudaMalloc(&real_, real_data_size) != cudaSuccess ||
            cudaMalloc(&dual_, dual_data_size) != cudaSuccess) {
            throw std::runtime_error("Failed to allocate GPU memory for dual number components.");
        }

        initializeStream();
    }

    // Destructor
    ~VectorDualDenseCuda() {
        if (owns_memory_) {
            if (real_) cudaFree(real_);
            if (dual_) cudaFree(dual_);
        }
        if (stream_) {
            cudaStreamDestroy(stream_);
        }
    }

    void initialize(const thrust::complex<T>* host_real, const thrust::complex<T>* host_dual) {
        size_t real_size = batch_size_ * real_size_ * sizeof(thrust::complex<T>);
        size_t dual_size = batch_size_ * real_size_ * dual_size_ * sizeof(thrust::complex<T>);

        cudaError_t real_status = cudaMemcpyAsync(real_, host_real, real_size, cudaMemcpyHostToDevice, stream_);
        cudaError_t dual_status = cudaMemcpyAsync(dual_, host_dual, dual_size, cudaMemcpyHostToDevice, stream_);

        if (real_status != cudaSuccess || dual_status != cudaSuccess) {
            throw std::runtime_error("Failed to copy data to device.");
        }

        cudaStreamSynchronize(stream_);
    }

    // Elementwise addition
    VectorDualDenseCuda elementwiseAdd(const VectorDualDenseCuda& other) const {
        if (batch_size_ != other.batch_size_ || real_size_ != other.real_size_ || dual_size_ != other.dual_size_) {
            throw std::invalid_argument("Dual dimensions do not match for elementwise addition.");
        }

        VectorDualDenseCuda result(batch_size_, real_size_, dual_size_);
        
        int real_total_elements = batch_size_ * real_size_;
        int dual_total_elements = batch_size_ * real_size_ * dual_size_;

        thrust::device_ptr<thrust::complex<T>> real1(real_);
        thrust::device_ptr<thrust::complex<T>> real2(other.real_);
        thrust::device_ptr<thrust::complex<T>> real_result(result.real_);

        thrust::device_ptr<thrust::complex<T>> dual1(dual_);
        thrust::device_ptr<thrust::complex<T>> dual2(other.dual_);
        thrust::device_ptr<thrust::complex<T>> dual_result(result.dual_);

        thrust::transform(real1, real1 + real_total_elements, real2, real_result, thrust::plus<thrust::complex<T>>());
        thrust::transform(dual1, dual1 + dual_total_elements, dual2, dual_result, thrust::plus<thrust::complex<T>>());

        return result;
    }



    VectorDualDenseCuda elementwiseMultiply(const VectorDualDenseCuda& other) const {
        if (batch_size_ != other.batch_size_ || real_size_ != other.real_size_ || dual_size_ != other.dual_size_) {
            throw std::invalid_argument("Dual dimensions do not match for elementwise multiplication.");
        }
        //The real part is multiplied elementwise
        VectorDualDenseCuda result(batch_size_, real_size_, dual_size_);
        int total_elements_real = batch_size_ * real_size_;

        thrust::transform(
            thrust::device_pointer_cast(real_),
            thrust::device_pointer_cast(real_ + total_elements_real),
            thrust::device_pointer_cast(other.real_),
            thrust::device_pointer_cast(result.real_),
            thrust::multiplies<thrust::complex<T>>());

        //Now get the dual part
        int total_elements_dual = batch_size_ * real_size_ * dual_size_;
        //Create a holder for the data
        thrust::device_vector<thrust::complex<T>> result_dual1(total_elements_dual);
        thrust::device_vector<thrust::complex<T>> result_dual2(total_elements_dual);
        multiplyDualTensor<T>(real_,
                              other.dual_,
                              thrust::raw_pointer_cast(result_dual1.data()), 
                              batch_size_, real_size_, dual_size_);
        multiplyDualTensor<T>(other.real_,
                              dual_,
                              thrust::raw_pointer_cast(result_dual2.data()), 
                              batch_size_, real_size_, dual_size_); 

        //Now add the two results
        thrust::transform(
            thrust::device_pointer_cast(result_dual1.data()),
            thrust::device_pointer_cast(result_dual1.data() + total_elements_dual),
            thrust::device_pointer_cast(result_dual2.data()),
            thrust::device_pointer_cast(result.dual_),
            thrust::plus<thrust::complex<T>>());


        return result;
    }

    // Accessors
    thrust::complex<T>* real() { return real_; }
    const thrust::complex<T>* real() const { return real_; }
    thrust::complex<T>* dual() { return dual_; }
    const thrust::complex<T>* dual() const { return dual_; }
    int batchSize() const { return batch_size_; }
    int size() const { return real_size_; }
    int dualSize() const { return dual_size_; }

private:
    // Initialize CUDA stream
    void initializeStream() {
        if (cudaStreamCreate(&stream_) != cudaSuccess) {
            throw std::runtime_error("Failed to create CUDA stream.");
        }
    }
};

template <typename T>
class VectorHyperDualDense {
private:
    using ComplexT = std::complex<T>;

    // Dimensions
    int batch_size_;   // M
    int size_;         // N (length of each vector)
    int dual_dim_;     // D (number of dual components)

    // Primal, Dual, and Hyper-Dual Data
    ComplexT* primal_data_;      // [M, N]
    ComplexT* dual_data_;        // [M, N, D]
    ComplexT* hyper_dual_data_;  // [M, N, D, D]
    bool owns_memory_;           // Indicates if memory is managed internally

    // cuBLAS handle
    cublasHandle_t handle_;
    cudaStream_t stream_;

public:
    // Constructor with external memory
    VectorHyperDualDense(int batch_size, int size, int dual_dim, ComplexT* primal_data, ComplexT* dual_data, ComplexT* hyper_dual_data)
        : batch_size_(batch_size), size_(size), dual_dim_(dual_dim),
          primal_data_(primal_data), dual_data_(dual_data), hyper_dual_data_(hyper_dual_data),
          owns_memory_(false) {
        if (!primal_data_ || !dual_data_ || !hyper_dual_data_) {
            throw std::invalid_argument("Primal, dual, or hyper-dual data pointer is null.");
        }
        initializeHandles();
    }

    // Constructor with internal memory allocation
    VectorHyperDualDense(int batch_size, int size, int dual_dim)
        : batch_size_(batch_size), size_(size), dual_dim_(dual_dim), owns_memory_(true) {
        if (batch_size <= 0 || size <= 0 || dual_dim <= 0) {
            throw std::invalid_argument("All dimensions must be positive.");
        }

        size_t primal_size = batch_size * size * sizeof(ComplexT);
        size_t dual_size = batch_size * size * dual_dim * sizeof(ComplexT);
        size_t hyper_dual_size = batch_size * size * dual_dim * dual_dim * sizeof(ComplexT);

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

    ~VectorHyperDualDense() {
        if (owns_memory_) {
            if (primal_data_) cudaFree(primal_data_);
            if (dual_data_) cudaFree(dual_data_);
            if (hyper_dual_data_) cudaFree(hyper_dual_data_);
        }
        cublasDestroy(handle_);
        cudaStreamDestroy(stream_);
    }

    void initialize(const ComplexT* primal, const ComplexT* dual, const ComplexT* hyper_dual, size_t primal_size, size_t dual_size, size_t hyper_dual_size) {
        if (primal_size != batch_size_ * size_ ||
            dual_size != batch_size_ * size_ * dual_dim_ ||
            hyper_dual_size != batch_size_ * size_ * dual_dim_ * dual_dim_) {
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

    // Example elementwise addition
    VectorHyperDualDense<T> elementwiseAdd(const VectorHyperDualDense<T>& other) const {
        if (batch_size_ != other.batch_size_ || size_ != other.size_ || dual_dim_ != other.dual_dim_) {
            throw std::invalid_argument("Tensor dimensions do not match for elementwise addition.");
        }

        VectorHyperDualDense<T> result(batch_size_, size_, dual_dim_);

        int total_primal_elements = batch_size_ * size_;
        int total_dual_elements = total_primal_elements * dual_dim_;
        int total_hyper_dual_elements = total_dual_elements * dual_dim_;

        // Perform elementwise addition for the primal part
        thrust::transform(
            thrust::device_pointer_cast(primal_data_),
            thrust::device_pointer_cast(primal_data_ + total_primal_elements),
            thrust::device_pointer_cast(other.primal_data_),
            thrust::device_pointer_cast(result.primal_data_),
            thrust::plus<ComplexT>());

        // Perform elementwise addition for the dual part
        thrust::transform(
            thrust::device_pointer_cast(dual_data_),
            thrust::device_pointer_cast(dual_data_ + total_dual_elements),
            thrust::device_pointer_cast(other.dual_data_),
            thrust::device_pointer_cast(result.dual_data_),
            thrust::plus<ComplexT>());

        // Perform elementwise addition for the hyper-dual part
        thrust::transform(
            thrust::device_pointer_cast(hyper_dual_data_),
            thrust::device_pointer_cast(hyper_dual_data_ + total_hyper_dual_elements),
            thrust::device_pointer_cast(other.hyper_dual_data_),
            thrust::device_pointer_cast(result.hyper_dual_data_),
            thrust::plus<ComplexT>());

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
    VectorHyperDualDense(const VectorHyperDualDense&) = delete;
    VectorHyperDualDense& operator=(const VectorHyperDualDense&) = delete;

    // Enable move constructor and move assignment
    VectorHyperDualDense(VectorHyperDualDense&&) noexcept = default;
    VectorHyperDualDense& operator=(VectorHyperDualDense&&) noexcept = default;

    // Getters for dimensions
    int batch_size() const { return batch_size_; }
    int size() const { return size_; }
    int dual_dim() const { return dual_dim_; }

    // Getters for data pointers
    ComplexT* primal_data() { return primal_data_; }
    ComplexT* dual_data() { return dual_data_; }
    ComplexT* hyper_dual_data() { return hyper_dual_data_; }

    const ComplexT* primal_data() const { return primal_data_; }
    const ComplexT* dual_data() const { return dual_data_; }
    const ComplexT* hyper_dual_data() const { return hyper_dual_data_; }
};


template <typename T>
class MatrixDense {
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
    MatrixDense(int batch_size, int rows, int cols, ComplexT* primal_data)
        : batch_size_(batch_size), rows_(rows), cols_(cols),
          primal_data_(primal_data), owns_memory_(false) {
        if (!primal_data_) {
            throw std::invalid_argument("Primal data pointer is null");
        }

        initializeHandles();
    }

    // Constructor with internal memory allocation
    MatrixDense(int batch_size, int rows, int cols)
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

    ~MatrixDense() {
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
    MatrixDense<U> indexGet(int start_row, int end_row, int start_col, int end_col) const {
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
        return MatrixDense<T>(batch_size_, sub_rows, sub_cols, sub_primal_data);
    }


    template <typename U>
    void indexPut(int start_row, int end_row, int start_col, int end_col, const MatrixDense<U>& data) {
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


    MatrixDense<T> multiply(const MatrixDense<T>& other) const {
        // Validate dimensions
        if (cols_ != other.rows_ || batch_size_ != other.batch_size_) {
            throw std::invalid_argument("Matrix dimensions do not match for multiplication.");
        }

        // Create the result matrix
        MatrixDense<T> result(batch_size_, rows_, other.cols_);

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

    MatrixDense<T> transpose() const {
        MatrixDense<T> result(batch_size_, cols_, rows_);
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

    MatrixDense<T> elementwiseAdd(const MatrixDense<T>& other) const {
        if (batch_size_ != other.batch_size_ || rows_ != other.rows_ || cols_ != other.cols_) {
            throw std::invalid_argument("Tensor dimensions do not match for addition.");
        }

        MatrixDense<T> result(batch_size_, rows_, cols_);
            int total_elements = batch_size_ * rows_ * cols_;

            thrust::transform(thrust::device_pointer_cast(primal_data_),
                            thrust::device_pointer_cast(primal_data_ + total_elements),
                            thrust::device_pointer_cast(other.primal_data_),
                            thrust::device_pointer_cast(result.primal_data_),
                            thrust::plus<ComplexT>());

            return result;
    }

    MatrixDense<T> square() const {
        // Create a new MatrixDense object to store the result
        MatrixDense<T> result(batch_size_, rows_, cols_);

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

    MatrixDense<T> upperTriangular() const {
        // Create a result matrix with the same dimensions
        MatrixDense<T> result(batch_size_, rows_, cols_);

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

    MatrixDense<T> lowerTriangular() const {
        // Create a result matrix with the same dimensions
        MatrixDense<T> result(batch_size_, rows_, cols_);

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
    MatrixDense(const MatrixDense&) = delete;
    MatrixDense& operator=(const MatrixDense&) = delete;

    // Enable move constructor and move assignment
    MatrixDense(MatrixDense&&) noexcept = default;
    MatrixDense& operator=(MatrixDense&&) noexcept = default;

    // Getters for dimensions
    int batch_size() const { return batch_size_; }
    int rows() const { return rows_; }
    int cols() const { return cols_; }

    // Getters for data pointers
    ComplexT* primal_data() { return primal_data_; }
    const ComplexT* primal_data() const { return primal_data_; }
};




template <typename T>
class MatrixDualDense {
private:
    using ComplexT = std::complex<T>;

    // Dimensions
    int batch_size_;
    int rows_;
    int cols_;
    int dual_dim_;

    // Primal and Dual Data
    ComplexT* primal_data_; // Device-side primal part
    ComplexT* dual_data_;   // Device-side dual part
    bool owns_memory_;      // Indicates if memory is managed internally

    // cuBLAS handle
    cublasHandle_t handle_;
    cudaStream_t stream_;

public:
    // Constructor with external memory
    MatrixDualDense(int batch_size, int rows, int cols, int dual_dim, ComplexT* primal_data, ComplexT* dual_data)
        : batch_size_(batch_size), rows_(rows), cols_(cols), dual_dim_(dual_dim),
          primal_data_(primal_data), dual_data_(dual_data), owns_memory_(false) {
        if (!primal_data_ || !dual_data_) {
            throw std::invalid_argument("Primal or dual data pointer is null");
        }

        initializeHandles();
    }

    // Constructor with internal memory allocation
    MatrixDualDense(int batch_size, int rows, int cols, int dual_dim)
        : batch_size_(batch_size), rows_(rows), cols_(cols), dual_dim_(dual_dim), owns_memory_(true) {
        if (batch_size <= 0 || rows <= 0 || cols <= 0 || dual_dim <= 0) {
            throw std::invalid_argument("All dimensions must be positive.");
        }

        size_t primal_size = batch_size * rows * cols * sizeof(ComplexT);
        size_t dual_size = batch_size * rows * cols * dual_dim * sizeof(ComplexT);

        if (cudaMalloc(&primal_data_, primal_size) != cudaSuccess) {
            throw std::runtime_error("Failed to allocate GPU memory for primal data.");
        }

        if (cudaMalloc(&dual_data_, dual_size) != cudaSuccess) {
            cudaFree(primal_data_);
            throw std::runtime_error("Failed to allocate GPU memory for dual data.");
        }

        initializeHandles();
    }

    ~MatrixDualDense() {
        if (owns_memory_) {
            if (primal_data_) cudaFree(primal_data_);
            if (dual_data_) cudaFree(dual_data_);
        }
        cublasDestroy(handle_);
        cudaStreamDestroy(stream_);
    }

    void initialize(const ComplexT* primal, const ComplexT* dual, size_t primal_size, size_t dual_size) {
        if (primal_size != batch_size_ * rows_ * cols_ || dual_size != batch_size_ * rows_ * cols_ * dual_dim_) {
            throw std::invalid_argument("Input sizes do not match tensor dimensions.");
        }

        if (primal) {
            cudaMemcpyAsync(primal_data_, primal, primal_size * sizeof(ComplexT), cudaMemcpyHostToDevice, stream_);
        }
        if (dual) {
            cudaMemcpyAsync(dual_data_, dual, dual_size * sizeof(ComplexT), cudaMemcpyHostToDevice, stream_);
        }
        cudaStreamSynchronize(stream_);
    }

    template <typename U>
    MatrixDualDense<U> indexGet(int start_row, int end_row, int start_col, int end_col) const {
        // Validate row and column ranges
        if (start_row < 0 || end_row > rows_ || start_row >= end_row ||
            start_col < 0 || end_col > cols_ || start_col >= end_col) {
            throw std::invalid_argument("Invalid row or column range for indexGet.");
        }

        // Dimensions of the submatrix
        int sub_rows = end_row - start_row;
        int sub_cols = end_col - start_col;

        // Calculate offsets for the primal and dual data
        ComplexT* sub_primal_data = primal_data_ + start_row * cols_ + start_col;
        ComplexT* sub_dual_data = dual_data_ + start_row * cols_ * dual_dim_ + start_col * dual_dim_;

        // Create a new MatrixDualDense instance sharing the data with the original
        return MatrixDualDense<T>(batch_size_, sub_rows, sub_cols, dual_dim_, sub_primal_data, sub_dual_data);
    }

    template <typename U>
    void indexPut(int start_row, int end_row, int start_col, int end_col, const MatrixDualDense<U>& data) {
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

        // Calculate offsets for the primal and dual data
        ComplexT* target_primal_data = primal_data_ + start_row * cols_ + start_col;
        ComplexT* target_dual_data = dual_data_ + start_row * cols_ * dual_dim_ + start_col * dual_dim_;

        // Calculate data sizes
        size_t primal_size = data.rows_ * data.cols_ * sizeof(ComplexT);
        size_t dual_size = data.rows_ * data.cols_ * dual_dim_ * sizeof(ComplexT);

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

        // Synchronize to ensure data transfer is complete
        cudaStreamSynchronize(stream_);
    }    

    void square() {
        // Element-wise square for the primal part
        size_t total_primal_elements = batch_size_ * rows_ * cols_;
        size_t total_dual_elements = batch_size_ * rows_ * cols_ * dual_dim_;

        // Kernel to compute element-wise square
        auto squareKernel = [] __device__(ComplexT x) -> ComplexT {
            return x * x;
        };

        // Launch a CUDA kernel to square the primal part
        thrust::device_ptr<ComplexT> primal_ptr(primal_data_);
        thrust::transform(thrust::device, primal_ptr, primal_ptr + total_primal_elements, primal_ptr, squareKernel);

        // Update the dual part according to the product rule:
        // If u = f(x) and v = f'(x), then square(u) has derivative: 2 * u * v.
        auto dualKernel = [] __device__(ComplexT u, ComplexT v) -> ComplexT {
            return ComplexT(2.0, 0.0) * u * v;
        };

        // Process the dual part
        for (int d = 0; d < dual_dim_; ++d) {
            ComplexT* dual_ptr = dual_data_ + d * batch_size_ * rows_ * cols_;
            thrust::device_ptr<ComplexT> dual_thrust_ptr(dual_ptr);
            thrust::transform(thrust::device, primal_ptr, primal_ptr + total_primal_elements,
                            dual_thrust_ptr, dual_thrust_ptr, dualKernel);
        }

        cudaStreamSynchronize(stream_);
    }




    template <typename U>
    MatrixDualDense<U> sum(int dimension) const {
        if (dimension != 1 && dimension != 2) {
            throw std::invalid_argument("Dimension must be 1 (rows) or 2 (columns).");
        }

        // Determine the new dimensions after summing along the specified axis
        int new_rows = (dimension == 1) ? 1 : rows_;
        int new_cols = (dimension == 2) ? 1 : cols_;

        // Create the resulting MatrixDualDense object
        MatrixDualDense<T> result(batch_size_, new_rows, new_cols, dual_dim_);

        // Allocate memory for temporary host buffers
        size_t primal_size = rows_ * cols_;
        size_t dual_size = rows_ * cols_ * dual_dim_;

        // Perform the summation along the specified dimension
        for (int b = 0; b < batch_size_; ++b) {
            if (dimension == 1) {
                // Summing along rows
                for (int c = 0; c < cols_; ++c) {
                    ComplexT sum_primal = ComplexT(0.0, 0.0);
                    std::vector<ComplexT> sum_dual(dual_dim_, ComplexT(0.0, 0.0));

                    for (int r = 0; r < rows_; ++r) {
                        int idx = b * rows_ * cols_ + r * cols_ + c;
                        sum_primal += primal_data_[idx];

                        for (int d = 0; d < dual_dim_; ++d) {
                            int dual_idx = b * rows_ * cols_ * dual_dim_ + r * cols_ * dual_dim_ + c * dual_dim_ + d;
                            sum_dual[d] += dual_data_[dual_idx];
                        }
                    }

                    // Store the result in the output matrix
                    int result_idx = b * new_rows * new_cols + c;
                    result.primal_data_[result_idx] = sum_primal;

                    for (int d = 0; d < dual_dim_; ++d) {
                        int result_dual_idx = b * new_rows * new_cols * dual_dim_ + c * dual_dim_ + d;
                        result.dual_data_[result_dual_idx] = sum_dual[d];
                    }
                }
            } else if (dimension == 2) {
                // Summing along columns
                for (int r = 0; r < rows_; ++r) {
                    ComplexT sum_primal = ComplexT(0.0, 0.0);
                    std::vector<ComplexT> sum_dual(dual_dim_, ComplexT(0.0, 0.0));

                    for (int c = 0; c < cols_; ++c) {
                        int idx = b * rows_ * cols_ + r * cols_ + c;
                        sum_primal += primal_data_[idx];

                        for (int d = 0; d < dual_dim_; ++d) {
                            int dual_idx = b * rows_ * cols_ * dual_dim_ + r * cols_ * dual_dim_ + c * dual_dim_ + d;
                            sum_dual[d] += dual_data_[dual_idx];
                        }
                    }

                    // Store the result in the output matrix
                    int result_idx = b * new_rows * new_cols + r;
                    result.primal_data_[result_idx] = sum_primal;

                    for (int d = 0; d < dual_dim_; ++d) {
                        int result_dual_idx = b * new_rows * new_cols * dual_dim_ + r * dual_dim_ + d;
                        result.dual_data_[result_dual_idx] = sum_dual[d];
                    }
                }
            }
        }

        cudaStreamSynchronize(stream_);
        return result;
    }

    VectorDualDenseCuda<T> squeeze(int dim) const {
        // Check the validity of the dimension
        if (dim < 1 || dim > 2) {
            throw std::invalid_argument("Dimension to squeeze must be 1 (rows) or 2 (columns).");
        }

        // Ensure the specified dimension has size 1
        if ((dim == 1 && rows_ != 1) || (dim == 2 && cols_ != 1)) {
            throw std::invalid_argument("Cannot squeeze a dimension with size greater than 1.");
        }

        // Determine the size of the resulting vector
        int vector_size = (dim == 1) ? cols_ : rows_;

        // Calculate the pointer to primal and dual data
        ComplexT* squeezed_primal_data = primal_data_;
        ComplexT* squeezed_dual_data = dual_data_;

        // Create and return a VectorDual object
        return VectorDualDenseCuda<T>(vector_size, dual_dim_, squeezed_primal_data, squeezed_dual_data);
    }

    template <typename U>
    VectorDualDenseCuda<U> matrixVectorProduct(const VectorDualDenseCuda<U>& vector) const {
        // Validate dimensions
        if (cols_ != vector.size_ || batch_size_ != vector.batch_size_ || dual_dim_ != vector.dual_dim_) {
            throw std::invalid_argument("Matrix and vector dimensions do not match for multiplication.");
        }

        // Result vector
        VectorDualDenseCuda<T> result(batch_size_, rows_, dual_dim_);

        int matrix_primal_size = rows_ * cols_;
        int vector_primal_size = vector.size_;
        int result_primal_size = rows_;

        int matrix_dual_size = matrix_primal_size * dual_dim_;
        int vector_dual_size = vector_primal_size * dual_dim_;
        int result_dual_size = result_primal_size * dual_dim_;

        ComplexT alpha = ComplexT(1.0, 0.0);
        ComplexT beta = ComplexT(0.0, 0.0);

        // Perform matrix-vector multiplication for primal part
        for (int b = 0; b < batch_size_; ++b) {
            cublasZgemv(handle_,
                        CUBLAS_OP_N,
                        rows_, cols_,
                        &alpha,
                        primal_data_ + b * matrix_primal_size, rows_,
                        vector.primal_data_ + b * vector_primal_size, 1,
                        &beta,
                        result.primal_data_ + b * result_primal_size, 1);
        }

        // Perform matrix-vector multiplication for dual part
        for (int b = 0; b < batch_size_; ++b) {
            for (int d = 0; d < dual_dim_; ++d) {
                // Matrix * Dual(Vector)
                cublasZgemv(handle_,
                            CUBLAS_OP_N,
                            rows_, cols_,
                            &alpha,
                            primal_data_ + b * matrix_primal_size, rows_,
                            vector.dual_data_ + b * vector_primal_size * dual_dim_ + d * vector_primal_size, 1,
                            &beta,
                            result.dual_data_ + b * result_primal_size * dual_dim_ + d * result_primal_size, 1);

                // Dual(Matrix) * Vector
                cublasZgemv(handle_,
                            CUBLAS_OP_N,
                            rows_, cols_,
                            &alpha,
                            dual_data_ + b * matrix_dual_size + d * matrix_primal_size, rows_,
                            vector.primal_data_ + b * vector_primal_size, 1,
                            &alpha, // Accumulate
                            result.dual_data_ + b * result_primal_size * dual_dim_ + d * result_primal_size, 1);
            }
        }

        return result;
    }


    MatrixDualDense<T> transpose() const {
        MatrixDualDense<T> result(batch_size_, cols_, rows_, dual_dim_);
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

        return result;
    }

    // Method to generate an upper triangular matrix
    MatrixDualDense<T> upperTriangular() const {
        // Create a new matrix for the result
        MatrixDualDense<T> result(batch_size_, rows_, cols_, dual_dim_);

        size_t total_elements = rows_ * cols_;

        for (int b = 0; b < batch_size_; ++b) {
            // Initialize primal part to upper triangular
            ComplexT* batch_primal_src = primal_data_ + b * total_elements;
            ComplexT* batch_primal_dst = result.primal_data_ + b * total_elements;

            thrust::for_each(
                thrust::counting_iterator<int>(0),
                thrust::counting_iterator<int>(rows_ * cols_),
                [=] __device__(int idx) {
                    int row = idx / cols_;
                    int col = idx % cols_;
                    batch_primal_dst[idx] = (row <= col) ? batch_primal_src[idx] : ComplexT(0.0, 0.0);
                });

            // Initialize dual part to zero
            ComplexT* batch_dual_dst = result.dual_data_ + b * total_elements * dual_dim_;
            thrust::fill(thrust::device_pointer_cast(batch_dual_dst),
                         thrust::device_pointer_cast(batch_dual_dst + total_elements * dual_dim_),
                         ComplexT(0.0, 0.0));
        }

        return result;
    }


    // Method to generate a lower triangular matrix
    MatrixDualDense<T> lowerTriangular() const {
            // Create a new matrix for the result
            MatrixDualDense<T> result(batch_size_, rows_, cols_, dual_dim_);

            size_t total_elements = rows_ * cols_;

            for (int b = 0; b < batch_size_; ++b) {
                // Initialize primal part to lower triangular
                ComplexT* batch_primal_src = primal_data_ + b * total_elements;
                ComplexT* batch_primal_dst = result.primal_data_ + b * total_elements;

                thrust::for_each(
                    thrust::counting_iterator<int>(0),
                    thrust::counting_iterator<int>(rows_ * cols_),
                    [=] __device__(int idx) {
                        int row = idx / cols_;
                        int col = idx % cols_;
                        batch_primal_dst[idx] = (row >= col) ? batch_primal_src[idx] : ComplexT(0.0, 0.0);
                    });

                // Initialize dual part to zero
                ComplexT* batch_dual_dst = result.dual_data_ + b * total_elements * dual_dim_;
                thrust::fill(thrust::device_pointer_cast(batch_dual_dst),
                            thrust::device_pointer_cast(batch_dual_dst + total_elements * dual_dim_),
                            ComplexT(0.0, 0.0));
            }

            return result;
        }

    template <typename U>
    MatrixDualDense<U> matrixMultiply(const MatrixDualDense<U>& other) const {
        // Validate dimensions
        if (cols_ != other.rows_ || batch_size_ != other.batch_size_ || dual_dim_ != other.dual_dim_) {
            throw std::invalid_argument("Matrix dimensions do not match for multiplication.");
        }

        // Create the result matrix
        MatrixDualDense<T> result(batch_size_, rows_, other.cols_, dual_dim_);

        ComplexT alpha = ComplexT(1.0, 0.0);
        ComplexT beta = ComplexT(0.0, 0.0);

        // Perform batched matrix multiplication for primal and dual parts
        for (int b = 0; b < batch_size_; ++b) {
            // Primal part
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
                // Primal * Dual(other)
                cublasZgemm(handle_,
                            CUBLAS_OP_N, CUBLAS_OP_N,
                            rows_, other.cols_, cols_,
                            &alpha,
                            primal_data_ + b * rows_ * cols_, rows_,
                            other.dual_data_ + b * other.rows_ * other.cols_ * dual_dim_ + d * other.rows_ * other.cols_, other.rows_,
                            &beta,
                            result.dual_data_ + b * rows_ * other.cols_ * dual_dim_ + d * rows_ * other.cols_, rows_);

                // Dual(Matrix) * Primal(other)
                cublasZgemm(handle_,
                            CUBLAS_OP_N, CUBLAS_OP_N,
                            rows_, other.cols_, cols_,
                            &alpha,
                            dual_data_ + b * rows_ * cols_ * dual_dim_ + d * rows_ * cols_, rows_,
                            other.primal_data_ + b * other.rows_ * other.cols_, other.rows_,
                            &alpha, // Accumulate
                            result.dual_data_ + b * rows_ * other.cols_ * dual_dim_ + d * rows_ * other.cols_, rows_);
            }
        }

        cudaStreamSynchronize(stream_);
        return result;
    }


    template <typename U>
    MatrixDualDense<U> elementwiseMultiply(const MatrixDualDense<U>& other) const {
        // Validate dimensions
        if (batch_size_ != other.batch_size_ || rows_ != other.rows_ || cols_ != other.cols_ || dual_dim_ != other.dual_dim_) {
            throw std::invalid_argument("Tensor dimensions do not match for elementwise multiplication.");
        }

        // Create result tensor
        MatrixDualDense<T> result(batch_size_, rows_, cols_, dual_dim_);

        int total_primal_elements = batch_size_ * rows_ * cols_;
        int total_dual_elements = total_primal_elements * dual_dim_;

        // Elementwise multiplication for primal part
        thrust::transform(
            thrust::device_pointer_cast(primal_data_),
            thrust::device_pointer_cast(primal_data_ + total_primal_elements),
            thrust::device_pointer_cast(other.primal_data_),
            thrust::device_pointer_cast(result.primal_data_),
            thrust::multiplies<ComplexT>());

        // Elementwise multiplication for dual part using product rule
        thrust::for_each(
            thrust::counting_iterator<int>(0),
            thrust::counting_iterator<int>(total_dual_elements),
            [=] __device__(int idx) {
                int primal_idx = idx / dual_dim_;
                int dual_idx = idx % dual_dim_;

                result.dual_data_[idx] =
                    dual_data_[idx] * other.primal_data_[primal_idx] +
                    primal_data_[primal_idx] * other.dual_data_[idx];
            });

        return result;
    };

    

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
};


template <typename T>
class MatrixHyperDualDense {
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
    MatrixHyperDualDense(int batch_size, int rows, int cols, int dual_dim,
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
    MatrixHyperDualDense(int batch_size, int rows, int cols, int dual_dim)
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

    ~MatrixHyperDualDense() {
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
    MatrixHyperDualDense<U> indexGet(int start_row, int end_row, int start_col, int end_col) const {
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
        return MatrixHyperDualDense<T>(batch_size_, sub_rows, sub_cols, dual_dim_,
                                    sub_primal_data, sub_dual_data, sub_hyper_dual_data);
    }


    template <typename U>
    void indexPut(int start_row, int end_row, int start_col, int end_col, const MatrixHyperDualDense<U>& data) {
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

    MatrixHyperDualDense<T> transpose() const {
        MatrixHyperDualDense<T> result(batch_size_, cols_, rows_, dual_dim_);
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

    MatrixHyperDualDense<T> upperTriangular() const {
        // Create a result matrix with the same dimensions
        MatrixHyperDualDense<T> result(batch_size_, rows_, cols_, dual_dim_);

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

    MatrixHyperDualDense<T> lowerTriangular() const {
        // Create a result matrix with the same dimensions
        MatrixHyperDualDense<T> result(batch_size_, rows_, cols_, dual_dim_);

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
    MatrixHyperDualDense<U> matrixMultiply(const MatrixHyperDualDense<U>& other) const {
        // Validate dimensions
        if (cols_ != other.rows_ || batch_size_ != other.batch_size_ || dual_dim_ != other.dual_dim_) {
            throw std::invalid_argument("Matrix dimensions do not match for multiplication.");
        }

        // Create the result matrix
        MatrixHyperDualDense<T> result(batch_size_, rows_, other.cols_, dual_dim_);

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

    template <typename U>
    VectorHyperDualDense<U> matrixVectorProduct(const VectorHyperDualDense<U>& vector) const {
        // Validate dimensions
        if (cols_ != vector.size() || batch_size_ != vector.batch_size() || dual_dim_ != vector.dual_dim()) {
            throw std::invalid_argument("Matrix and vector dimensions do not match for multiplication.");
        }

        // Create the result vector
        VectorHyperDualDense<T> result(batch_size_, rows_, dual_dim_);

        ComplexT alpha = ComplexT(1.0, 0.0);
        ComplexT beta = ComplexT(0.0, 0.0);

        for (int b = 0; b < batch_size_; ++b) {
            // Real part: Primal * Primal(vector)
            cublasZgemv(handle_,
                        CUBLAS_OP_N,
                        rows_, cols_,
                        &alpha,
                        primal_data_ + b * rows_ * cols_, rows_,
                        vector.primal_data() + b * vector.size(), 1,
                        &beta,
                        result.primal_data() + b * rows_, 1);

            // Dual part
            for (int d = 0; d < dual_dim_; ++d) {
                // Primal * Dual(vector) + Dual * Primal(vector)
                cublasZgemv(handle_,
                            CUBLAS_OP_N,
                            rows_, cols_,
                            &alpha,
                            primal_data_ + b * rows_ * cols_, rows_,
                            vector.dual_data() + b * vector.size() * dual_dim_ + d * vector.size(), 1,
                            &beta,
                            result.dual_data() + b * rows_ * dual_dim_ + d * rows_, 1);

                cublasZgemv(handle_,
                            CUBLAS_OP_N,
                            rows_, cols_,
                            &alpha,
                            dual_data_ + b * rows_ * cols_ * dual_dim_ + d * rows_ * cols_, rows_,
                            vector.primal_data() + b * vector.size(), 1,
                            &alpha, // Accumulate
                            result.dual_data() + b * rows_ * dual_dim_ + d * rows_, 1);
            }

            // Hyper-dual part
            for (int d1 = 0; d1 < dual_dim_; ++d1) {
                for (int d2 = 0; d2 < dual_dim_; ++d2) {
                    // (Primal * Hyper-Dual(vector)) + (Dual * Dual(vector)) + (Hyper-Dual * Primal(vector))
                    cublasZgemv(handle_,
                                CUBLAS_OP_N,
                                rows_, cols_,
                                &alpha,
                                primal_data_ + b * rows_ * cols_, rows_,
                                vector.hyper_dual_data() + b * vector.size() * dual_dim_ * dual_dim_ +
                                    d1 * vector.size() * dual_dim_ + d2 * vector.size(), 1,
                                &beta,
                                result.hyper_dual_data() + b * rows_ * dual_dim_ * dual_dim_ +
                                    d1 * rows_ * dual_dim_ + d2 * rows_, 1);

                    cublasZgemv(handle_,
                                CUBLAS_OP_N,
                                rows_, cols_,
                                &alpha,
                                dual_data_ + b * rows_ * cols_ * dual_dim_ + d1 * rows_ * cols_, rows_,
                                vector.dual_data() + b * vector.size() * dual_dim_ + d2 * vector.size(), 1,
                                &alpha, // Accumulate
                                result.hyper_dual_data() + b * rows_ * dual_dim_ * dual_dim_ +
                                    d1 * rows_ * dual_dim_ + d2 * rows_, 1);

                    cublasZgemv(handle_,
                                CUBLAS_OP_N,
                                rows_, cols_,
                                &alpha,
                                hyper_dual_data_ + b * rows_ * cols_ * dual_dim_ * dual_dim_ +
                                    d1 * rows_ * cols_ * dual_dim_ + d2 * rows_ * cols_, rows_,
                                vector.primal_data() + b * vector.size(), 1,
                                &alpha, // Accumulate
                                result.hyper_dual_data() + b * rows_ * dual_dim_ * dual_dim_ +
                                    d1 * rows_ * dual_dim_ + d2 * rows_, 1);
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
    MatrixHyperDualDense(const MatrixHyperDualDense&) = delete;
    MatrixHyperDualDense& operator=(const MatrixHyperDualDense&) = delete;

    // Enable move constructor and move assignment
    MatrixHyperDualDense(MatrixHyperDualDense&&) noexcept = default;
    MatrixHyperDualDense& operator=(MatrixHyperDualDense&&) noexcept = default;

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