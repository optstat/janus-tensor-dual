#ifndef _CU_DUAL_TENSOR_HPP
#define _CU_DUAL_TENSOR_HPP
#include <cuda_runtime.h>
#include <iostream>
#include <complex>
//Utility class to implement dual tensor operations only necessary for QR decomposition
//This is a simplified version of the more extensive Dual class in the original codebase
//and it is implemented using cuBLAS and cuSPARSE for matrix operations
#include <cusparse.h>
#include <memory>
#include <vector>


namespace janus {

template <typename T>
class VectorDual {
private:
    using ComplexT = std::complex<T>;

    // Dimensions
    int batch_size_;  // M
    int size_;        // N (length of each vector)
    int dual_dim_;    // D (number of dual components)

    // Primal and Dual Data
    ComplexT* primal_data_;  // [M, N]
    ComplexT* dual_data_;    // [M, N, D]
    bool owns_memory_;       // Indicates if memory is managed internally

    // cuBLAS handle
    cublasHandle_t handle_;
    cudaStream_t stream_;

public:
    // Constructor with external memory
    VectorDual(int batch_size, int size, int dual_dim, ComplexT* primal_data, ComplexT* dual_data)
        : batch_size_(batch_size), size_(size), dual_dim_(dual_dim),
          primal_data_(primal_data), dual_data_(dual_data), owns_memory_(false) {
        if (!primal_data_ || !dual_data_) {
            throw std::invalid_argument("Primal or dual data pointer is null");
        }
        initializeHandles();
    }

    // Constructor with internal memory allocation
    VectorDual(int batch_size, int size, int dual_dim)
        : batch_size_(batch_size), size_(size), dual_dim_(dual_dim), owns_memory_(true) {
        if (batch_size <= 0 || size <= 0 || dual_dim <= 0) {
            throw std::invalid_argument("All dimensions must be positive.");
        }

        size_t primal_size = batch_size * size * sizeof(ComplexT);
        size_t dual_size = batch_size * size * dual_dim * sizeof(ComplexT);

        if (cudaMalloc(&primal_data_, primal_size) != cudaSuccess) {
            throw std::runtime_error("Failed to allocate GPU memory for primal data.");
        }

        if (cudaMalloc(&dual_data_, dual_size) != cudaSuccess) {
            cudaFree(primal_data_);
            throw std::runtime_error("Failed to allocate GPU memory for dual data.");
        }

        initializeHandles();
    }

    ~VectorDual() {
        if (owns_memory_) {
            if (primal_data_) cudaFree(primal_data_);
            if (dual_data_) cudaFree(dual_data_);
        }
        cublasDestroy(handle_);
        cudaStreamDestroy(stream_);
    }

    void initialize(const ComplexT* primal, const ComplexT* dual, size_t primal_size, size_t dual_size) {
        if (primal_size != batch_size_ * size_ ||
            dual_size != batch_size_ * size_ * dual_dim_) {
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

    template <typename T>
    VectorDual<T> indexGet(int start_row, int end_row) const {
        // Validate row range
        if (start_row < 0 || end_row > batch_size_ || start_row >= end_row) {
            throw std::invalid_argument("Invalid row range for selection.");
        }

        // Calculate the number of rows in the selected range
        int selected_batch_size = end_row - start_row;

        // Calculate offsets in the primal and dual data
        ComplexT* selected_primal_data = primal_data_ + start_row * size_;
        ComplexT* selected_dual_data = dual_data_ + start_row * size_ * dual_dim_;

        // Create a new VectorDual instance sharing the data with the original
        return VectorDual<T>(selected_batch_size, size_, dual_dim_, selected_primal_data, selected_dual_data);
    }

    template <typename T>
    void indexPut(int start_row, int end_row, const VectorDual<T>& data) {
        // Validate row range
        if (start_row < 0 || end_row > batch_size_ || start_row >= end_row) {
            throw std::invalid_argument("Invalid row range for index_put.");
        }

        // Validate dimensions of the input data
        if (data.batch_size() != (end_row - start_row) || data.size() != size_ || data.dual_dim() != dual_dim_) {
            throw std::invalid_argument("Input data dimensions do not match the target range.");
        }

        // Compute offsets in the primal and dual tensors
        ComplexT* target_primal_data = primal_data_ + start_row * size_;
        ComplexT* target_dual_data = dual_data_ + start_row * size_ * dual_dim_;

        // Compute data sizes
        size_t primal_size = data.batch_size() * size_ * sizeof(ComplexT);
        size_t dual_size = data.batch_size() * size_ * dual_dim_ * sizeof(ComplexT);

        // Copy data for the primal part
        cudaMemcpyAsync(target_primal_data, data.primal_data(), primal_size, cudaMemcpyDeviceToDevice, stream_);

        // Copy data for the dual part
        cudaMemcpyAsync(target_dual_data, data.dual_data(), dual_size, cudaMemcpyDeviceToDevice, stream_);

        // Synchronize to ensure data transfer is complete
        cudaStreamSynchronize(stream_);
    }

    VectorDual<T> elementwiseAdd(const VectorDual<T>& other) const {
        if (batch_size_ != other.batch_size_ || size_ != other.size_ || dual_dim_ != other.dual_dim_) {
            throw std::invalid_argument("Tensor dimensions do not match for elementwise addition.");
        }

        VectorDual<T> result(batch_size_, size_, dual_dim_);

        int total_primal_elements = batch_size_ * size_;
        int total_dual_elements = total_primal_elements * dual_dim_;

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

        return result;
    }

    VectorDual<T> elementwiseMultiply(const VectorDual<T>& other) const {
        if (batch_size_ != other.batch_size_ || size_ != other.size_ || dual_dim_ != other.dual_dim_) {
            throw std::invalid_argument("Tensor dimensions do not match for elementwise multiplication.");
        }

        VectorDual<T> result(batch_size_, size_, dual_dim_);

        int total_primal_elements = batch_size_ * size_;
        int total_dual_elements = total_primal_elements * dual_dim_;

        // Perform elementwise multiplication for primal part
        thrust::transform(
            thrust::device_pointer_cast(primal_data_),
            thrust::device_pointer_cast(primal_data_ + total_primal_elements),
            thrust::device_pointer_cast(other.primal_data_),
            thrust::device_pointer_cast(result.primal_data_),
            thrust::multiplies<ComplexT>());

        // Perform elementwise multiplication for dual part using the product rule
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
    VectorDual(const VectorDual&) = delete;
    VectorDual& operator=(const VectorDual&) = delete;

    // Enable move constructor and move assignment
    VectorDual(VectorDual&&) noexcept = default;
    VectorDual& operator=(VectorDual&&) noexcept = default;

    // Getters for dimensions
    int batch_size() const { return batch_size_; }
    int size() const { return size_; }
    int dual_dim() const { return dual_dim_; }

    // Getters for data pointers
    ComplexT* primal_data() { return primal_data_; }
    ComplexT* dual_data() { return dual_data_; }

    const ComplexT* primal_data() const { return primal_data_; }
    const ComplexT* dual_data() const { return dual_data_; }
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

    template <typename T>
    MatrixDualDense<T> indexGet(int start_row, int end_row, int start_col, int end_col) const {
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

    template <typename T>
    void indexPut(int start_row, int end_row, int start_col, int end_col, const MatrixDualDense<T>& data) {
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

    template <typename T>
    MatrixDense<T> indexGet(int start_row, int end_row, int start_col, int end_col) const {
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


    template <typename T>
    void indexPut(int start_row, int end_row, int start_col, int end_col, const MatrixDense<T>& data) {
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
    template <typename T>
    MatrixHyperDualDense<T> indexGet(int start_row, int end_row, int start_col, int end_col) const {
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


    template <typename T>
    void indexPut(int start_row, int end_row, int start_col, int end_col, const MatrixHyperDualDense<T>& data) {
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
};

template <typename T>
MatrixDualDense<T> elementwiseMultiply(const MatrixDualDense<T>& other) const {
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

template <typename T>
VectorDual<T> matrixVectorProduct(const VectorDual<T>& vector) const {
    // Validate dimensions
    if (cols_ != vector.size_ || batch_size_ != vector.batch_size_ || dual_dim_ != vector.dual_dim_) {
        throw std::invalid_argument("Matrix and vector dimensions do not match for multiplication.");
    }

    // Result vector
    VectorDual<T> result(batch_size_, rows_, dual_dim_);

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




}  // namespace janus
#endif // _CU_DUAL_TENSOR_HPP