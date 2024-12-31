#ifndef _CU_DUAL_TENSOR_HPP
#define _CU_DUAL_TENSOR_HPP
#include <iostream>
#include <complex>
//Utility class to implement dual tensor operations only necessary for QR decomposition
//This is a simplified version of the Dual class in the original codebase
#include <cusparse.h>
#include <memory>
#include <vector>
#include <thrust/device_vector.h>

namespace janus {

template <typename T>
class VectorDualSparse {
private:
    using ComplexT = std::complex<T>;

    // Dimensions
    int batch_size_;  // M
    int size_;        // N (length of each vector)
    int dual_dim_;    // D (number of dual components)

    // cuSPARSE handle
    cusparseHandle_t cusparse_handle_;
    cudaStream_t stream_;

    // Primal Data (CSR format)
    thrust::device_vector<int> primal_row_ptr_;   // Row pointers (CSR format)
    thrust::device_vector<int> primal_col_indices_; // Column indices (CSR format)
    thrust::device_vector<ComplexT> primal_values_; // Non-zero values

    // Dual Data (CSR format)
    thrust::device_vector<int> dual_row_ptr_;   // Row pointers (CSR format)
    thrust::device_vector<int> dual_col_indices_; // Column indices (CSR format)
    thrust::device_vector<ComplexT> dual_values_; // Non-zero dual values

public:
    // Constructor
    VectorDualSparse(int batch_size, int size, int dual_dim)
        : batch_size_(batch_size), size_(size), dual_dim_(dual_dim) {
        if (batch_size <= 0 || size <= 0 || dual_dim <= 0) {
            throw std::invalid_argument("All dimensions must be positive.");
        }

        // Initialize cuSPARSE handle and CUDA stream
        if (cusparseCreate(&cusparse_handle_) != CUSPARSE_STATUS_SUCCESS) {
            throw std::runtime_error("Failed to create cuSPARSE handle.");
        }
        if (cudaStreamCreate(&stream_) != cudaSuccess) {
            cusparseDestroy(cusparse_handle_);
            throw std::runtime_error("Failed to create CUDA stream.");
        }
        cusparseSetStream(cusparse_handle_, stream_);
    }

    ~VectorDualSparse() {
        cusparseDestroy(cusparse_handle_);
        cudaStreamDestroy(stream_);
    }

    // Convert COO to CSR
    void convertCOOToCSR(const std::vector<int>& row_indices,
                         const std::vector<int>& col_indices,
                         const std::vector<ComplexT>& values,
                         thrust::device_vector<int>& row_ptr,
                         thrust::device_vector<int>& col_indices_device,
                         thrust::device_vector<ComplexT>& values_device) {
        // Validate COO input
        if (row_indices.size() != col_indices.size() || row_indices.size() != values.size()) {
            throw std::invalid_argument("Row, column, and value arrays must have the same size.");
        }

        int nnz = values.size(); // Number of non-zero elements

        // Copy COO data to device
        thrust::device_vector<int> d_row_indices(row_indices.begin(), row_indices.end());
        col_indices_device = thrust::device_vector<int>(col_indices.begin(), col_indices.end());
        values_device = thrust::device_vector<ComplexT>(values.begin(), values.end());

        // Allocate CSR row pointer
        row_ptr.resize(size_ + 1);

        // Convert COO to CSR using cuSPARSE
        cusparseStatus_t status = cusparseXcoo2csr(
            cusparse_handle_,
            thrust::raw_pointer_cast(d_row_indices.data()), // COO row indices
            nnz, // Number of non-zero elements
            size_, // Number of rows
            thrust::raw_pointer_cast(row_ptr.data()), // CSR row pointer
            CUSPARSE_INDEX_BASE_ZERO); // COO and CSR use zero-based indexing

        if (status != CUSPARSE_STATUS_SUCCESS) {
            throw std::runtime_error("Failed to convert COO to CSR.");
        }
    }

    // Initialize primal data from COO format
    void initializePrimalFromCOO(const std::vector<int>& row_indices,
                                 const std::vector<int>& col_indices,
                                 const std::vector<ComplexT>& values) {
        convertCOOToCSR(row_indices, col_indices, values, primal_row_ptr_, primal_col_indices_, primal_values_);
    }

    // Initialize dual data from COO format
    void initializeDualFromCOO(const std::vector<int>& row_indices,
                                const std::vector<int>& col_indices,
                                const std::vector<ComplexT>& values) {
        convertCOOToCSR(row_indices, col_indices, values, dual_row_ptr_, dual_col_indices_, dual_values_);
    }

    // Getters for sparse data
    const thrust::device_vector<int>& primalRowPtr() const { return primal_row_ptr_; }
    const thrust::device_vector<int>& primalColIndices() const { return primal_col_indices_; }
    const thrust::device_vector<ComplexT>& primalValues() const { return primal_values_; }

    const thrust::device_vector<int>& dualRowPtr() const { return dual_row_ptr_; }
    const thrust::device_vector<int>& dualColIndices() const { return dual_col_indices_; }
    const thrust::device_vector<ComplexT>& dualValues() const { return dual_values_; }
};

template <typename T>
class MatrixSparse {
private:
    using ComplexT = std::complex<T>;

    // Dimensions
    int batch_size_;   // Number of batches
    int rows_;         // Number of rows in each matrix
    int cols_;         // Number of columns in each matrix

    // cuSPARSE handle
    cusparseHandle_t cusparse_handle_;
    cudaStream_t stream_;

    // Primal Data (CSR format)
    thrust::device_vector<int> row_ptr_;        // Row pointers
    thrust::device_vector<int> col_indices_;   // Column indices
    thrust::device_vector<ComplexT> values_;    // Non-zero values

public:
    // Constructor
    MatrixSparse(int batch_size, int rows, int cols)
        : batch_size_(batch_size), rows_(rows), cols_(cols) {
        if (batch_size <= 0 || rows <= 0 || cols <= 0) {
            throw std::invalid_argument("All dimensions must be positive.");
        }

        // Initialize cuSPARSE handle and CUDA stream
        if (cusparseCreate(&cusparse_handle_) != CUSPARSE_STATUS_SUCCESS) {
            throw std::runtime_error("Failed to create cuSPARSE handle.");
        }
        if (cudaStreamCreate(&stream_) != cudaSuccess) {
            cusparseDestroy(cusparse_handle_);
            throw std::runtime_error("Failed to create CUDA stream.");
        }
        cusparseSetStream(cusparse_handle_, stream_);
    }

    ~MatrixSparse() {
        cusparseDestroy(cusparse_handle_);
        cudaStreamDestroy(stream_);
    }

    // Convert COO to CSR
    void convertCOOToCSR(const std::vector<int>& row_indices,
                         const std::vector<int>& col_indices,
                         const std::vector<ComplexT>& values) {
        // Validate COO input
        if (row_indices.size() != col_indices.size() || row_indices.size() != values.size()) {
            throw std::invalid_argument("Row, column, and value arrays must have the same size.");
        }

        int nnz = values.size(); // Number of non-zero elements

        // Copy COO data to device
        thrust::device_vector<int> d_row_indices(row_indices.begin(), row_indices.end());
        col_indices_ = thrust::device_vector<int>(col_indices.begin(), col_indices.end());
        values_ = thrust::device_vector<ComplexT>(values.begin(), values.end());

        // Allocate CSR row pointer
        row_ptr_.resize(rows_ + 1);

        // Convert COO to CSR using cuSPARSE
        cusparseStatus_t status = cusparseXcoo2csr(
            cusparse_handle_,
            thrust::raw_pointer_cast(d_row_indices.data()), // COO row indices
            nnz,                                           // Number of non-zero elements
            rows_,                                         // Number of rows
            thrust::raw_pointer_cast(row_ptr_.data()),     // CSR row pointer
            CUSPARSE_INDEX_BASE_ZERO);                    // COO and CSR use zero-based indexing

        if (status != CUSPARSE_STATUS_SUCCESS) {
            throw std::runtime_error("Failed to convert COO to CSR.");
        }
    }

    // Initialize primal data from COO format
    void initializeFromCOO(const std::vector<int>& row_indices,
                           const std::vector<int>& col_indices,
                           const std::vector<ComplexT>& values) {
        convertCOOToCSR(row_indices, col_indices, values);
    }

    // Elementwise Addition
    MatrixSparse<T> elementwiseAdd(const MatrixSparse<T>& other) const {
        if (batch_size_ != other.batch_size_ || rows_ != other.rows_ || cols_ != other.cols_) {
            throw std::invalid_argument("Sparse matrix dimensions do not match for addition.");
        }

        // Placeholder for sparse addition logic (merge row_ptr_, col_indices_, and values_)
        throw std::runtime_error("Sparse addition not yet implemented.");
    }

    // Transpose
    MatrixSparse<T> transpose() const {
        MatrixSparse<T> result(batch_size_, cols_, rows_);

        // Placeholder for real transpose logic using CSR transpose
        throw std::runtime_error("Transpose not yet implemented.");

        return result;
    }

    // Getters for sparse data
    const thrust::device_vector<int>& rowPtr() const { return row_ptr_; }
    const thrust::device_vector<int>& colIndices() const { return col_indices_; }
    const thrust::device_vector<ComplexT>& values() const { return values_; }
};



template <typename T>
class MatrixHyperDualSparse {
private:
    using ComplexT = std::complex<T>;

    // Dimensions
    int batch_size_;   // Number of batches
    int rows_;         // Number of rows in each matrix
    int cols_;         // Number of columns in each matrix
    int dual_dim_;     // Dimension of the dual part

    // cuSPARSE handle
    cusparseHandle_t cusparse_handle_;
    cudaStream_t stream_;

    // Primal Data (CSR format)
    thrust::device_vector<int> primal_row_ptr_;      // Row pointers
    thrust::device_vector<int> primal_col_indices_;  // Column indices
    thrust::device_vector<ComplexT> primal_values_;  // Non-zero values

    // Dual Data (CSR format)
    thrust::device_vector<int> dual_row_ptr_;      // Row pointers
    thrust::device_vector<int> dual_col_indices_;  // Column indices
    thrust::device_vector<ComplexT> dual_values_;  // Non-zero values

    // Hyper-Dual Data (CSR format)
    thrust::device_vector<int> hyper_dual_row_ptr_;      // Row pointers
    thrust::device_vector<int> hyper_dual_col_indices_;  // Column indices
    thrust::device_vector<ComplexT> hyper_dual_values_;  // Non-zero values

public:
    // Constructor
    MatrixHyperDualSparse(int batch_size, int rows, int cols, int dual_dim)
        : batch_size_(batch_size), rows_(rows), cols_(cols), dual_dim_(dual_dim) {
        if (batch_size <= 0 || rows <= 0 || cols <= 0 || dual_dim <= 0) {
            throw std::invalid_argument("All dimensions must be positive.");
        }

        // Initialize cuSPARSE handle and CUDA stream
        if (cusparseCreate(&cusparse_handle_) != CUSPARSE_STATUS_SUCCESS) {
            throw std::runtime_error("Failed to create cuSPARSE handle.");
        }
        if (cudaStreamCreate(&stream_) != cudaSuccess) {
            cusparseDestroy(cusparse_handle_);
            throw std::runtime_error("Failed to create CUDA stream.");
        }
        cusparseSetStream(cusparse_handle_, stream_);
    }

    ~MatrixHyperDualSparse() {
        cusparseDestroy(cusparse_handle_);
        cudaStreamDestroy(stream_);
    }

    // Convert COO to CSR
    void convertCOOToCSR(const std::vector<int>& row_indices,
                         const std::vector<int>& col_indices,
                         const std::vector<ComplexT>& values,
                         thrust::device_vector<int>& row_ptr,
                         thrust::device_vector<int>& col_indices_device,
                         thrust::device_vector<ComplexT>& values_device) {
        // Validate COO input
        if (row_indices.size() != col_indices.size() || row_indices.size() != values.size()) {
            throw std::invalid_argument("Row, column, and value arrays must have the same size.");
        }

        int nnz = values.size(); // Number of non-zero elements

        // Copy COO data to device
        thrust::device_vector<int> d_row_indices(row_indices.begin(), row_indices.end());
        col_indices_device = thrust::device_vector<int>(col_indices.begin(), col_indices.end());
        values_device = thrust::device_vector<ComplexT>(values.begin(), values.end());

        // Allocate CSR row pointer
        row_ptr.resize(rows_ + 1);

        // Convert COO to CSR using cuSPARSE
        cusparseStatus_t status = cusparseXcoo2csr(
            cusparse_handle_,
            thrust::raw_pointer_cast(d_row_indices.data()), // COO row indices
            nnz,                                           // Number of non-zero elements
            rows_,                                         // Number of rows
            thrust::raw_pointer_cast(row_ptr.data()),      // CSR row pointer
            CUSPARSE_INDEX_BASE_ZERO);                    // COO and CSR use zero-based indexing

        if (status != CUSPARSE_STATUS_SUCCESS) {
            throw std::runtime_error("Failed to convert COO to CSR.");
        }
    }

    // Initialize primal data from COO format
    void initializePrimalFromCOO(const std::vector<int>& row_indices,
                                 const std::vector<int>& col_indices,
                                 const std::vector<ComplexT>& values) {
        convertCOOToCSR(row_indices, col_indices, values, primal_row_ptr_, primal_col_indices_, primal_values_);
    }

    // Initialize dual data from COO format
    void initializeDualFromCOO(const std::vector<int>& row_indices,
                                const std::vector<int>& col_indices,
                                const std::vector<ComplexT>& values) {
        convertCOOToCSR(row_indices, col_indices, values, dual_row_ptr_, dual_col_indices_, dual_values_);
    }

    // Initialize hyper-dual data from COO format
    void initializeHyperDualFromCOO(const std::vector<int>& row_indices,
                                    const std::vector<int>& col_indices,
                                    const std::vector<ComplexT>& values) {
        convertCOOToCSR(row_indices, col_indices, values, hyper_dual_row_ptr_, hyper_dual_col_indices_, hyper_dual_values_);
    }

    // Elementwise Addition
    MatrixHyperDualSparse<T> elementwiseAdd(const MatrixHyperDualSparse<T>& other) const {
        if (batch_size_ != other.batch_size_ || rows_ != other.rows_ || cols_ != other.cols_ || dual_dim_ != other.dual_dim_) {
            throw std::invalid_argument("Sparse tensor dimensions do not match for addition.");
        }

        // Placeholder for real sparse addition logic
        throw std::runtime_error("Sparse addition not yet implemented.");
    }

    // Elementwise Multiplication
    MatrixHyperDualSparse<T> elementwiseMultiply(const MatrixHyperDualSparse<T>& other) const {
        if (batch_size_ != other.batch_size_ || rows_ != other.rows_ || cols_ != other.cols_ || dual_dim_ != other.dual_dim_) {
            throw std::invalid_argument("Sparse tensor dimensions do not match for multiplication.");
        }

        // Placeholder for real sparse multiplication logic
        throw std::runtime_error("Sparse multiplication not yet implemented.");
    }

    // Getters for sparse data
    const thrust::device_vector<int>& primalRowPtr() const { return primal_row_ptr_; }
    const thrust::device_vector<int>& primalColIndices() const { return primal_col_indices_; }
    const thrust::device_vector<ComplexT>& primalValues() const { return primal_values_; }

    const thrust::device_vector<int>& dualRowPtr() const { return dual_row_ptr_; }
    const thrust::device_vector<int>& dualColIndices() const { return dual_col_indices_; }
    const thrust::device_vector<ComplexT>& dualValues() const { return dual_values_; }

    const thrust::device_vector<int>& hyperDualRowPtr() const { return hyper_dual_row_ptr_; }
    const thrust::device_vector<int>& hyperDualColIndices() const { return hyper_dual_col_indices_; }
    const thrust::device_vector<ComplexT>& hyperDualValues() const { return hyper_dual_values_; }
};


template <typename T>
class MatrixDualSparse {
private:
    using ComplexT = std::complex<T>;

    // Dimensions
    int batch_size_;
    int rows_;
    int cols_;
    int dual_dim_;

    // cuSPARSE handle
    cusparseHandle_t cusparse_handle_;
    cudaStream_t stream_;

    // Primal Data (CSR format)
    thrust::device_vector<int> primal_row_ptr_;      // Row pointers
    thrust::device_vector<int> primal_col_indices_;  // Column indices
    thrust::device_vector<ComplexT> primal_values_;  // Non-zero values

    // Dual Data (CSR format)
    thrust::device_vector<int> dual_row_ptr_;      // Row pointers
    thrust::device_vector<int> dual_col_indices_;  // Column indices
    thrust::device_vector<ComplexT> dual_values_;  // Non-zero values

public:
    // Constructor
    MatrixDualSparse(int batch_size, int rows, int cols, int dual_dim)
        : batch_size_(batch_size), rows_(rows), cols_(cols), dual_dim_(dual_dim) {
        if (batch_size <= 0 || rows <= 0 || cols <= 0 || dual_dim <= 0) {
            throw std::invalid_argument("All dimensions must be positive.");
        }

        // Initialize cuSPARSE handle and CUDA stream
        if (cusparseCreate(&cusparse_handle_) != CUSPARSE_STATUS_SUCCESS) {
            throw std::runtime_error("Failed to create cuSPARSE handle.");
        }
        if (cudaStreamCreate(&stream_) != cudaSuccess) {
            cusparseDestroy(cusparse_handle_);
            throw std::runtime_error("Failed to create CUDA stream.");
        }
        cusparseSetStream(cusparse_handle_, stream_);
    }

    ~MatrixDualSparse() {
        cusparseDestroy(cusparse_handle_);
        cudaStreamDestroy(stream_);
    }

    // Elementwise Multiplication
    MatrixDualSparse<T> elementwiseMultiply(const MatrixDualSparse<T>& other) const {
        if (batch_size_ != other.batch_size_ || rows_ != other.rows_ || cols_ != other.cols_ || dual_dim_ != other.dual_dim_) {
            throw std::invalid_argument("Sparse matrix dimensions do not match for multiplication.");
        }

        MatrixDualSparse<T> result(batch_size_, rows_, cols_, dual_dim_);

        // Primal part: Elementwise multiplication of CSR values
        result.primal_row_ptr_ = primal_row_ptr_;
        result.primal_col_indices_ = primal_col_indices_;
        result.primal_values_.resize(primal_values_.size());
        thrust::transform(
            primal_values_.begin(),
            primal_values_.end(),
            other.primal_values_.begin(),
            result.primal_values_.begin(),
            thrust::multiplies<ComplexT>());

        // Dual part: Apply product rule for elementwise multiplication
        result.dual_row_ptr_ = dual_row_ptr_;
        result.dual_col_indices_ = dual_col_indices_;
        result.dual_values_.resize(dual_values_.size());

        thrust::for_each(
            thrust::counting_iterator<int>(0),
            thrust::counting_iterator<int>(dual_values_.size()),
            [=] __device__(int idx) {
                int primal_idx = idx / dual_dim_;
                int dual_idx = idx % dual_dim_;

                result.dual_values_[idx] =
                    dual_values_[idx] * other.primal_values_[primal_idx] +
                    primal_values_[primal_idx] * other.dual_values_[idx];
            });

        return result;
    }

    template <typename T>
class MatrixDualSparse {
private:
    using ComplexT = std::complex<T>;

    // Dimensions
    int batch_size_;
    int rows_;
    int cols_;
    int dual_dim_;

    // cuSPARSE handle
    cusparseHandle_t cusparse_handle_;
    cudaStream_t stream_;

    // Primal Data (CSR format)
    thrust::device_vector<int> primal_row_ptr_;      // Row pointers
    thrust::device_vector<int> primal_col_indices_;  // Column indices
    thrust::device_vector<ComplexT> primal_values_;  // Non-zero values

    // Dual Data (CSR format)
    thrust::device_vector<int> dual_row_ptr_;      // Row pointers
    thrust::device_vector<int> dual_col_indices_;  // Column indices
    thrust::device_vector<ComplexT> dual_values_;  // Non-zero values

public:
    // Constructor
    MatrixDualSparse(int batch_size, int rows, int cols, int dual_dim)
        : batch_size_(batch_size), rows_(rows), cols_(cols), dual_dim_(dual_dim) {
        if (batch_size <= 0 || rows <= 0 || cols <= 0 || dual_dim <= 0) {
            throw std::invalid_argument("All dimensions must be positive.");
        }

        // Initialize cuSPARSE handle and CUDA stream
        if (cusparseCreate(&cusparse_handle_) != CUSPARSE_STATUS_SUCCESS) {
            throw std::runtime_error("Failed to create cuSPARSE handle.");
        }
        if (cudaStreamCreate(&stream_) != cudaSuccess) {
            cusparseDestroy(cusparse_handle_);
            throw std::runtime_error("Failed to create CUDA stream.");
        }
        cusparseSetStream(cusparse_handle_, stream_);
    }

    ~MatrixDualSparse() {
        cusparseDestroy(cusparse_handle_);
        cudaStreamDestroy(stream_);
    }

    // Sparse Matrix-Vector Product
    VectorDual<T> matrixVectorProduct(const VectorDual<T>& vector) const {
        if (cols_ != vector.size() || batch_size_ != vector.batch_size() || dual_dim_ != vector.dual_dim()) {
            throw std::invalid_argument("Matrix and vector dimensions do not match for multiplication.");
        }

        VectorDual<T> result(batch_size_, rows_, dual_dim_);
        ComplexT alpha = ComplexT(1.0, 0.0);
        ComplexT beta = ComplexT(0.0, 0.0);

        // Perform sparse matrix-vector multiplication for primal part
        for (int b = 0; b < batch_size_; ++b) {
            cusparseSpMV(
                cusparse_handle_,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha,
                cusparseCreateCsr(rows_, cols_, primal_values_.size(),
                                  primal_row_ptr_.data().get(),
                                  primal_col_indices_.data().get(),
                                  primal_values_.data().get(),
                                  CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                  CUSPARSE_INDEX_BASE_ZERO, CUDA_C_64F),
                thrust::raw_pointer_cast(vector.primal_data()) + b * vector.size(),
                &beta,
                thrust::raw_pointer_cast(result.primal_data()) + b * result.size(),
                CUDA_C_64F,
                CUSPARSE_MV_ALG_DEFAULT,
                nullptr);
        }

        // Perform sparse matrix-vector multiplication for dual part
        for (int b = 0; b < batch_size_; ++b) {
            for (int d = 0; d < dual_dim_; ++d) {
                // Dual(Matrix) * Primal(Vector)
                cusparseSpMV(
                    cusparse_handle_,
                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                    &alpha,
                    cusparseCreateCsr(rows_, cols_, dual_values_.size(),
                                      dual_row_ptr_.data().get(),
                                      dual_col_indices_.data().get(),
                                      dual_values_.data().get(),
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_C_64F),
                    thrust::raw_pointer_cast(vector.primal_data()) + b * vector.size(),
                    &beta,
                    thrust::raw_pointer_cast(result.dual_data()) + b * result.size() * dual_dim_ + d * result.size(),
                    CUDA_C_64F,
                    CUSPARSE_MV_ALG_DEFAULT,
                    nullptr);

                // Primal(Matrix) * Dual(Vector)
                cusparseSpMV(
                    cusparse_handle_,
                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                    &alpha,
                    cusparseCreateCsr(rows_, cols_, primal_values_.size(),
                                      primal_row_ptr_.data().get(),
                                      primal_col_indices_.data().get(),
                                      primal_values_.data().get(),
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_C_64F),
                    thrust::raw_pointer_cast(vector.dual_data()) + b * vector.size() * dual_dim_ + d * vector.size(),
                    &alpha, // Accumulate
                    thrust::raw_pointer_cast(result.dual_data()) + b * result.size() * dual_dim_ + d * result.size(),
                    CUDA_C_64F,
                    CUSPARSE_MV_ALG_DEFAULT,
                    nullptr);
            }
        }

        return result;
    }

    // Other methods for COO to CSR conversion, initialization, etc., would go here
};

template <typename T>
VectorDual<T> matrixVectorProductLHS(const VectorDual<T>& vector) const {
    // Validate dimensions
    if (rows_ != vector.size() || batch_size_ != vector.batch_size() || dual_dim_ != vector.dual_dim()) {
        throw std::invalid_argument("Matrix and vector dimensions do not match for LHS multiplication.");
    }

    // Result vector
    VectorDual<T> result(batch_size_, cols_, dual_dim_);
    ComplexT alpha = ComplexT(1.0, 0.0);
    ComplexT beta = ComplexT(0.0, 0.0);

    // Perform sparse matrix-vector multiplication (transposed) for primal part
    for (int b = 0; b < batch_size_; ++b) {
        cusparseSpMV(
            cusparse_handle_,
            CUSPARSE_OPERATION_TRANSPOSE, // Perform transpose for LHS
            &alpha,
            cusparseCreateCsr(rows_, cols_, primal_values_.size(),
                              primal_row_ptr_.data().get(),
                              primal_col_indices_.data().get(),
                              primal_values_.data().get(),
                              CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                              CUSPARSE_INDEX_BASE_ZERO, CUDA_C_64F),
            thrust::raw_pointer_cast(vector.primal_data()) + b * vector.size(),
            &beta,
            thrust::raw_pointer_cast(result.primal_data()) + b * result.size(),
            CUDA_C_64F,
            CUSPARSE_MV_ALG_DEFAULT,
            nullptr);
    }

    // Perform sparse matrix-vector multiplication (transposed) for dual part
    for (int b = 0; b < batch_size_; ++b) {
        for (int d = 0; d < dual_dim_; ++d) {
            // Dual(Matrix)^T * Primal(Vector)
            cusparseSpMV(
                cusparse_handle_,
                CUSPARSE_OPERATION_TRANSPOSE, // Transposed for LHS
                &alpha,
                cusparseCreateCsr(rows_, cols_, dual_values_.size(),
                                  dual_row_ptr_.data().get(),
                                  dual_col_indices_.data().get(),
                                  dual_values_.data().get(),
                                  CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                  CUSPARSE_INDEX_BASE_ZERO, CUDA_C_64F),
                thrust::raw_pointer_cast(vector.primal_data()) + b * vector.size(),
                &beta,
                thrust::raw_pointer_cast(result.dual_data()) + b * result.size() * dual_dim_ + d * result.size(),
                CUDA_C_64F,
                CUSPARSE_MV_ALG_DEFAULT,
                nullptr);

            // Primal(Matrix)^T * Dual(Vector)
            cusparseSpMV(
                cusparse_handle_,
                CUSPARSE_OPERATION_TRANSPOSE, // Transposed for LHS
                &alpha,
                cusparseCreateCsr(rows_, cols_, primal_values_.size(),
                                  primal_row_ptr_.data().get(),
                                  primal_col_indices_.data().get(),
                                  primal_values_.data().get(),
                                  CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                  CUSPARSE_INDEX_BASE_ZERO, CUDA_C_64F),
                thrust::raw_pointer_cast(vector.dual_data()) + b * vector.size() * dual_dim_ + d * vector.size(),
                &alpha, // Accumulate
                thrust::raw_pointer_cast(result.dual_data()) + b * result.size() * dual_dim_ + d * result.size(),
                CUDA_C_64F,
                CUSPARSE_MV_ALG_DEFAULT,
                nullptr);
        }
    }

    return result;
}

template <typename T>
VectorDual<T> matrixVectorProductLHS(const VectorDual<T>& vector) const {
    // Validate dimensions
    if (rows_ != vector.size_ || batch_size_ != vector.batch_size_ || dual_dim_ != vector.dual_dim_) {
        throw std::invalid_argument("Matrix and vector dimensions do not match for multiplication.");
    }

    // Result vector
    VectorDual<T> result(batch_size_, cols_, dual_dim_);

    int matrix_primal_size = rows_ * cols_;
    int vector_primal_size = vector.size();
    int result_primal_size = cols_;

    int matrix_dual_size = matrix_primal_size * dual_dim_;
    int vector_dual_size = vector_primal_size * dual_dim_;
    int result_dual_size = result_primal_size * dual_dim_;

    ComplexT alpha = ComplexT(1.0, 0.0);
    ComplexT beta = ComplexT(0.0, 0.0);

    // Perform matrix-vector multiplication for primal part (LHS: transpose the matrix)
    for (int b = 0; b < batch_size_; ++b) {
        cublasZgemv(handle_,
                    CUBLAS_OP_T, // Transpose operation
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
            // Transpose(Matrix) * Dual(Vector)
            cublasZgemv(handle_,
                        CUBLAS_OP_T, // Transpose operation
                        rows_, cols_,
                        &alpha,
                        primal_data_ + b * matrix_primal_size, rows_,
                        vector.dual_data_ + b * vector_primal_size * dual_dim_ + d * vector_primal_size, 1,
                        &beta,
                        result.dual_data_ + b * result_primal_size * dual_dim_ + d * result_primal_size, 1);

            // Transpose(Dual(Matrix)) * Vector
            cublasZgemv(handle_,
                        CUBLAS_OP_T, // Transpose operation
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


    // Other methods for COO to CSR conversion, initialization, etc., would go here
};


}  // namespace janus
#endif // _CU_DUAL_TENSOR_HPP