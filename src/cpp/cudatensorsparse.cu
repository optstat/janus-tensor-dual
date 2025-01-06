#ifndef _CU_DUAL_TENSOR_HPP
#define _CU_DUAL_TENSOR_HPP
#include <iostream>
#include <complex>
//Utility class to implement dual tensor operations only necessary for QR decomposition
//This is a simplified version of the Dual class in the original codebase
#include <cublas_v2.h>
#include <cusparse.h>
#include <memory>
#include <vector>
#include <thrust/device_vector.h>

namespace janus {

template <typename T>
class SparseVectorCuda {
private:
    int size_;                            // Vector size (N)
    int nnz_;                             // Number of non-zero elements
    thrust::device_vector<int> indices_;  // Indices of non-zero elements
    thrust::device_vector<T> values_;     // Non-zero values

public:
    // Constructor for sparse vector
    SparseVectorCuda(int size, int nnz, const int* host_indices, const T* host_values)
        : size_(size), nnz_(nnz), indices_(nnz), values_(nnz) {
        if (size <= 0 || nnz < 0) {
            throw std::invalid_argument("Invalid vector size or number of non-zeros.");
        }

        // Initialize data from host
        thrust::copy(host_indices, host_indices + nnz, indices_.begin());
        thrust::copy(host_values, host_values + nnz, values_.begin());
    }

    // Default destructor
    ~SparseVectorCuda() = default;

    // Elementwise addition
    SparseVectorCuda elementwiseAdd(const SparseVectorCuda& other) const {
        if (size_ != other.size_) {
            throw std::invalid_argument("Vector dimensions do not match for addition.");
        }

        // Merge indices and values into a new sparse vector
        thrust::device_vector<int> result_indices;
        thrust::device_vector<T> result_values;

        thrust::merge_by_key(
            indices_.begin(), indices_.end(),
            other.indices_.begin(), other.indices_.end(),
            values_.begin(), other.values_.begin(),
            result_indices.begin(), result_values.begin(),
            thrust::less<int>());

        return SparseVectorCuda(size_, result_indices.size(), result_indices.data().get(), result_values.data().get());
    }

    // Elementwise multiplication
    SparseVectorCuda elementwiseMultiply(const SparseVectorCuda& other) const {
        if (size_ != other.size_) {
            throw std::invalid_argument("Vector dimensions do not match for multiplication.");
        }

        // Create a sparse vector to store the result
        thrust::device_vector<int> result_indices;
        thrust::device_vector<T> result_values;

        // Use thrust::set_intersection to match non-zero indices
        auto result_end = thrust::set_intersection_by_key(
            indices_.begin(), indices_.end(),
            other.indices_.begin(), other.indices_.end(),
            values_.begin(), other.values_.begin(),
            result_indices.begin(), result_values.begin(),
            thrust::less<int>());

        // Return the new sparse vector
        return SparseVectorCuda(size_, result_indices.size(), result_indices.data().get(), result_values.data().get());
    }

    // Scale the sparse vector by a scalar
    void scale(T scalar) {
        thrust::transform(
            values_.begin(),
            values_.end(),
            values_.begin(),
            thrust::placeholders::_1 * scalar);
    }

    // Dot product with another sparse vector
    T dot(const SparseVectorCuda& other) const {
        if (size_ != other.size_) {
            throw std::invalid_argument("Vector dimensions do not match for dot product.");
        }

        // Compute dot product by summing elementwise multiplications
        T result = thrust::inner_product(
            indices_.begin(), indices_.end(),
            values_.begin(),
            other.indices_.begin(),
            other.values_.begin(),
            static_cast<T>(0));

        return result;
    }

    // Copy data back to host
    void copyToHost(int* host_indices, T* host_values) const {
        thrust::copy(indices_.begin(), indices_.end(), host_indices);
        thrust::copy(values_.begin(), values_.end(), host_values);
    }

    // Accessors
    int size() const { return size_; }
    int nnz() const { return nnz_; }
    const thrust::device_vector<int>& indices() const { return indices_; }
    const thrust::device_vector<T>& values() const { return values_; }
};

// Kernel for CSR to COO conversion
__global__ void csrToCooKernel(const int* row_ptr, int* row_indices, int num_rows) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < num_rows) {
        for (int idx = row_ptr[row]; idx < row_ptr[row + 1]; ++idx) {
            row_indices[idx] = row; // Expand row index for each non-zero element
        }
    }
}


// Function to convert CSR to COO
void csrToCoo(const thrust::device_vector<int>& row_ptr, 
              thrust::device_vector<int>& row_indices,
              int num_rows) {
    int num_nonzeros = row_indices.size();

    // Launch kernel to expand `row_ptr` into `row_indices`
    int threads_per_block = 256;
    int blocks = (num_rows + threads_per_block - 1) / threads_per_block;

    csrToCooKernel<<<blocks, threads_per_block>>>(
        thrust::raw_pointer_cast(row_ptr.data()),
        thrust::raw_pointer_cast(row_indices.data()),
        num_rows
    );

    // Check for errors
    cudaDeviceSynchronize();
    if (cudaPeekAtLastError() != cudaSuccess) {
        throw std::runtime_error("CSR to COO conversion failed.");
    }
}


template <typename T>
struct BroadcastSparseDualMultiply {
    const thrust::complex<T>* real_values;  // Non-zero values of the real sparse tensor
    thrust::device_vector<int> real_row_indices;  // Row indices for the real sparse tensor (COO)
    thrust::device_vector<int> real_col_indices;  // Column indices for the real sparse tensor (COO)

    const thrust::complex<T>* dual_values;  // Non-zero values of the dual sparse tensor
    thrust::device_vector<int> dual_row_indices;  // Row indices for the dual sparse tensor (COO)
    thrust::device_vector<int> dual_col_indices;  // Column indices for the dual sparse tensor (COO)

    int real_nnz;  // Number of non-zero elements in the real sparse tensor
    int dual_nnz;  // Number of non-zero elements in the dual sparse tensor

    // Constructor accepting COO format directly
    BroadcastSparseDualMultiply(
        const thrust::complex<T>* real_values,
        const thrust::device_vector<int>& real_row_indices,
        const thrust::device_vector<int>& real_col_indices,
        int real_nnz,
        const thrust::complex<T>* dual_values,
        const thrust::device_vector<int>& dual_row_indices,
        const thrust::device_vector<int>& dual_col_indices,
        int dual_nnz)
        : real_values(real_values),
          real_row_indices(real_row_indices),
          real_col_indices(real_col_indices),
          real_nnz(real_nnz),
          dual_values(dual_values),
          dual_row_indices(dual_row_indices),
          dual_col_indices(dual_col_indices),
          dual_nnz(dual_nnz) {}

    // Constructor accepting CSR format (auto-converts to COO)
    BroadcastSparseDualMultiply(
        const thrust::complex<T>* real_values, const thrust::device_vector<int>& real_row_ptr,
        const thrust::device_vector<int>& real_col_indices, int real_nnz,
        const thrust::complex<T>* dual_values, const thrust::device_vector<int>& dual_row_ptr,
        const thrust::device_vector<int>& dual_col_indices, int dual_nnz) 
        : real_values(real_values),
          real_col_indices(real_col_indices),
          real_nnz(real_nnz),
          dual_values(dual_values),
          dual_col_indices(dual_col_indices),
          dual_nnz(dual_nnz) 
    {
        // Convert real sparse tensor from CSR to COO
        real_row_indices.resize(real_nnz);
        csrToCoo(real_row_ptr, real_row_indices, real_row_ptr.size() - 1);

        // Convert dual sparse tensor from CSR to COO
        dual_row_indices.resize(dual_nnz);
        csrToCoo(dual_row_ptr, dual_row_indices, dual_row_ptr.size() - 1);
    }

    __device__ thrust::complex<T> operator()(int idx) const {
        // idx corresponds to an entry in the output sparse tensor
        int real_idx = idx / dual_nnz;  // Real tensor index
        int dual_idx = idx % dual_nnz;  // Dual tensor index

        // Check if the row and column indices match (sparse tensors must align for multiplication)
        if (real_row_indices[real_idx] == dual_row_indices[dual_idx] &&
            real_col_indices[real_idx] == dual_col_indices[dual_idx]) {
            // Multiply the corresponding values
            return real_values[real_idx] * dual_values[dual_idx];
        } else {
            // If indices do not match, the result is zero (skip multiplication)
            return thrust::complex<T>(0);
        }
    }

private:
    // Helper function to convert CSR to COO (defined earlier)
    void csrToCoo(const thrust::device_vector<int>& row_ptr,
                  thrust::device_vector<int>& row_indices, int num_rows) {
        int threads_per_block = 256;
        int blocks = (num_rows + threads_per_block - 1) / threads_per_block;

        csrToCooKernel<<<blocks, threads_per_block>>>(
            thrust::raw_pointer_cast(row_ptr.data()),
            thrust::raw_pointer_cast(row_indices.data()),
            num_rows);

        // Synchronize and check for errors
        cudaDeviceSynchronize();
        if (cudaPeekAtLastError() != cudaSuccess) {
            throw std::runtime_error("CSR to COO conversion failed.");
        }
    }
};

void csrToCooPrecompute(const int* row_ptr, int* row_indices, int num_rows, int nnz) {
    int threads_per_block = 256;
    int blocks = (num_rows + threads_per_block - 1) / threads_per_block;

    csrToCooKernel<<<blocks, threads_per_block>>>(
        row_ptr, row_indices, num_rows);

    cudaDeviceSynchronize();
}

template <typename T>
__global__ void sparseRealDualMultiplyKernel(
    const thrust::complex<T>* real_values, const int* real_row_indices, const int* real_col_indices, int real_nnz,
    const thrust::complex<T>* dual_values, const int* dual_row_indices, const int* dual_col_indices, int dual_nnz,
    thrust::complex<T>* result_values, const int* result_row_indices, const int* result_col_indices) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < real_nnz) {
        int real_row = real_row_indices[idx];
        int real_col = real_col_indices[idx];
        thrust::complex<T> real_value = real_values[idx];

        // Iterate over dual tensor to find matching row and column
        for (int j = 0; j < dual_nnz; ++j) {
            if (dual_row_indices[j] == real_row && dual_col_indices[j] == real_col) {
                // Perform multiplication and write to result
                int result_idx = idx;  // Assuming result indices align with real indices
                result_values[result_idx] = real_value * dual_values[j];
                break;
            }
        }
    }
}


/**
 * Multiply two sparse dual tensors elementwise.
 * This emulates the Einstein sum "mij,mik->mijk" for sparse tensors in CSR format.
 */
template <typename T>
void sparseDualDualMultiplyKernel(const thrust::complex<T>* dual1_values,
                                  const int* dual1_row_ptr, const int* dual1_col_indices,
                                  const thrust::complex<T>* dual2_values,
                                  const int* dual2_row_ptr, const int* dual2_col_indices,
                                  thrust::complex<T>* result_values, int* result_row_ptr, int* result_col_indices,
                                  int batch_size, int real_size, int dual_size, int nnz1, int nnz2) {
    int total_elements = nnz1 * nnz2;  // Total number of computations to perform

    thrust::transform(
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(total_elements),
        thrust::device_pointer_cast(result_values),
        [=] __device__(int idx) {
            // Calculate the indices for dual1 and dual2
            int dual1_idx = idx / nnz2;  // Index in dual1
            int dual2_idx = idx % nnz2; // Index in dual2

            // Get the rows and columns for dual1 and dual2
            int row1 = -1, row2 = -1;

            // Find row for dual1
            for (int i = 0; i < batch_size * real_size + 1; ++i) {
                if (dual1_idx >= dual1_row_ptr[i] && dual1_idx < dual1_row_ptr[i + 1]) {
                    row1 = i;
                    break;
                }
            }

            // Find row for dual2
            for (int i = 0; i < batch_size * real_size + 1; ++i) {
                if (dual2_idx >= dual2_row_ptr[i] && dual2_idx < dual2_row_ptr[i + 1]) {
                    row2 = i;
                    break;
                }
            }

            // Only process if rows match and columns align
            if (row1 == row2 && dual1_col_indices[dual1_idx] == dual2_col_indices[dual2_idx]) {
                // Perform elementwise multiplication
                return dual1_values[dual1_idx] * dual2_values[dual2_idx];
            } else {
                return thrust::complex<T>(0);  // Non-matching entries result in zero
            }
        });
}

template <typename T>
__global__ void sparseElementwiseAddKernel(
    const thrust::complex<T>* real1_values, const int* real1_row_ptr, const int* real1_col_indices, int real1_nnz,
    const thrust::complex<T>* real2_values, const int* real2_row_ptr, const int* real2_col_indices, int real2_nnz,
    thrust::complex<T>* real_result_values, int* real_result_row_ptr, int* real_result_col_indices,

    const thrust::complex<T>* dual1_values, const int* dual1_row_ptr, const int* dual1_col_indices, int dual1_nnz,
    const thrust::complex<T>* dual2_values, const int* dual2_row_ptr, const int* dual2_col_indices, int dual2_nnz,
    thrust::complex<T>* dual_result_values, int* dual_result_row_ptr, int* dual_result_col_indices) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Process real part addition for non-zero elements
    if (idx < real1_nnz && idx < real2_nnz) {
        // Match rows and columns
        int real_row1 = -1, real_row2 = -1;

        // Find rows for real1 and real2
        for (int i = 0; i < real1_nnz; ++i) {
            if (idx >= real1_row_ptr[i] && idx < real1_row_ptr[i + 1]) {
                real_row1 = i;
                break;
            }
        }
        for (int i = 0; i < real2_nnz; ++i) {
            if (idx >= real2_row_ptr[i] && idx < real2_row_ptr[i + 1]) {
                real_row2 = i;
                break;
            }
        }

        // Perform addition if rows and columns match
        if (real_row1 == real_row2 && real1_col_indices[idx] == real2_col_indices[idx]) {
            real_result_values[idx] = real1_values[idx] + real2_values[idx];
            real_result_row_ptr[idx] = real1_row_ptr[idx];  // Copy row structure
            real_result_col_indices[idx] = real1_col_indices[idx];
        }
    }

    // Process dual part addition for non-zero elements
    if (idx < dual1_nnz && idx < dual2_nnz) {
        // Match rows and columns
        int dual_row1 = -1, dual_row2 = -1;

        // Find rows for dual1 and dual2
        for (int i = 0; i < dual1_nnz; ++i) {
            if (idx >= dual1_row_ptr[i] && idx < dual1_row_ptr[i + 1]) {
                dual_row1 = i;
                break;
            }
        }
        for (int i = 0; i < dual2_nnz; ++i) {
            if (idx >= dual2_row_ptr[i] && idx < dual2_row_ptr[i + 1]) {
                dual_row2 = i;
                break;
            }
        }

        // Perform addition if rows and columns match
        if (dual_row1 == dual_row2 && dual1_col_indices[idx] == dual2_col_indices[idx]) {
            dual_result_values[idx] = dual1_values[idx] + dual2_values[idx];
            dual_result_row_ptr[idx] = dual1_row_ptr[idx];  // Copy row structure
            dual_result_col_indices[idx] = dual1_col_indices[idx];
        }
    }
}


template <typename T>
__global__ void sparseElementwiseMultiplyKernel(
    const thrust::complex<T>* real1_values, const int* real1_row_ptr, const int* real1_col_indices, int real1_nnz,
    const thrust::complex<T>* real2_values, const int* real2_row_ptr, const int* real2_col_indices, int real2_nnz,
    thrust::complex<T>* real_result_values, int* real_result_row_ptr, int* real_result_col_indices,

    const thrust::complex<T>* dual1_values, const int* dual1_row_ptr, const int* dual1_col_indices, int dual1_nnz,
    const thrust::complex<T>* dual2_values, const int* dual2_row_ptr, const int* dual2_col_indices, int dual2_nnz,
    thrust::complex<T>* dual_result_values, int* dual_result_row_ptr, int* dual_result_col_indices,
    int real_size, int dual_size) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Real part multiplication
    if (idx < real1_nnz && idx < real2_nnz) {
        // Match rows and columns for real1 and real2
        if (real1_col_indices[idx] == real2_col_indices[idx]) {
            real_result_values[idx] = real1_values[idx] * real2_values[idx];
            real_result_row_ptr[idx] = real1_row_ptr[idx];
            real_result_col_indices[idx] = real1_col_indices[idx];
        }
    }

    // Dual part multiplication
    if (idx < dual1_nnz && idx < dual2_nnz) {
        // Match rows and columns for dual1 and dual2
        if (dual1_col_indices[idx] == dual2_col_indices[idx]) {
            int dual_dim = dual_size / real_size;
            int real_idx = idx / dual_dim;

            dual_result_values[idx] = real1_values[real_idx] * dual2_values[idx] +
                                      real2_values[real_idx] * dual1_values[idx];
            dual_result_row_ptr[idx] = dual1_row_ptr[idx];
            dual_result_col_indices[idx] = dual1_col_indices[idx];
        }
    }
} // end of sparseElementwiseMultiplyKernel

template <typename T>
class VectorDualSparseCuda {
private:
    int batch_size_;                // Batch dimension (M)
    int real_size_;                 // Vector length (N)
    int dual_size_;                 // Dual dimension (D)

    // CSR format for real and dual parts
    thrust::complex<T>* real_values_;  // Non-zero values for the real part
    int* real_col_indices_;           // Column indices for the real part
    int* real_row_pointers_;          // Row pointers for the real part

    thrust::complex<T>* dual_values_; // Non-zero values for the dual part
    int* dual_col_indices_;           // Column indices for the dual part
    int* dual_row_pointers_;          // Row pointers for the dual part

    int nnz_real_;                    // Number of non-zero elements in the real part
    int nnz_dual_;                    // Number of non-zero elements in the dual part

public:
    // Constructor for externally managed memory
    __host__ __device__ VectorDualSparseCuda(int batch_size, int real_size, int dual_size,
                                             thrust::complex<T>* real_values, int* real_col_indices, int* real_row_pointers,
                                             thrust::complex<T>* dual_values, int* dual_col_indices, int* dual_row_pointers,
                                             int nnz_real, int nnz_dual)
        : batch_size_(batch_size), real_size_(real_size), dual_size_(dual_size),
          real_values_(real_values), real_col_indices_(real_col_indices), real_row_pointers_(real_row_pointers),
          dual_values_(dual_values), dual_col_indices_(dual_col_indices), dual_row_pointers_(dual_row_pointers),
          nnz_real_(nnz_real), nnz_dual_(nnz_dual) {}

    // Constructor for internal memory allocation
    VectorDualSparseCuda(int batch_size, int real_size, int dual_size, int nnz_real, int nnz_dual)
        : batch_size_(batch_size), real_size_(real_size), dual_size_(dual_size),
          nnz_real_(nnz_real), nnz_dual_(nnz_dual) {
        // Allocate memory for the real part
        cudaMalloc(&real_values_, nnz_real * sizeof(thrust::complex<T>));
        cudaMalloc(&real_col_indices_, nnz_real * sizeof(int));
        cudaMalloc(&real_row_pointers_, (real_size + 1) * sizeof(int));

        // Allocate memory for the dual part
        cudaMalloc(&dual_values_, nnz_dual * sizeof(thrust::complex<T>));
        cudaMalloc(&dual_col_indices_, nnz_dual * sizeof(int));
        cudaMalloc(&dual_row_pointers_, (real_size + 1) * sizeof(int));
    }

    // Destructor
    ~VectorDualSparseCuda() {
        cudaFree(real_values_);
        cudaFree(real_col_indices_);
        cudaFree(real_row_pointers_);
        cudaFree(dual_values_);
        cudaFree(dual_col_indices_);
        cudaFree(dual_row_pointers_);
    }

    // Elementwise addition
    __device__ void elementwiseAdd(const VectorDualSparseCuda& other, VectorDualSparseCuda& result) const {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;

        // Add real part
        if (idx < nnz_real_) {
            result.real_values_[idx] = real_values_[idx] + other.real_values_[idx];
        }

        // Add dual part
        if (idx < nnz_dual_) {
            result.dual_values_[idx] = dual_values_[idx] + other.dual_values_[idx];
        }
    }

    // Elementwise multiplication
    __device__ void elementwiseMultiply(const VectorDualSparseCuda& other, VectorDualSparseCuda& result) const {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;

        // Multiply real part
        if (idx < nnz_real_) {
            result.real_values_[idx] = real_values_[idx] * other.real_values_[idx];
        }

        // Multiply dual part
        if (idx < nnz_dual_) {
            result.dual_values_[idx] = dual_values_[idx] * other.dual_values_[idx];
        }
    }

    // Accessors
    __host__ __device__ thrust::complex<T>* realValues() { return real_values_; }
    __host__ __device__ const thrust::complex<T>* realValues() const { return real_values_; }
    __host__ __device__ int* realColIndices() { return real_col_indices_; }
    __host__ __device__ const int* realColIndices() const { return real_col_indices_; }
    __host__ __device__ int* realRowPointers() { return real_row_pointers_; }
    __host__ __device__ const int* realRowPointers() const { return real_row_pointers_; }

    __host__ __device__ thrust::complex<T>* dualValues() { return dual_values_; }
    __host__ __device__ const thrust::complex<T>* dualValues() const { return dual_values_; }
    __host__ __device__ int* dualColIndices() { return dual_col_indices_; }
    __host__ __device__ const int* dualColIndices() const { return dual_col_indices_; }
    __host__ __device__ int* dualRowPointers() { return dual_row_pointers_; }
    __host__ __device__ const int* dualRowPointers() const { return dual_row_pointers_; }

    __host__ __device__ int batchSize() const { return batch_size_; }
    __host__ __device__ int size() const { return real_size_; }
    __host__ __device__ int dualSize() const { return dual_size_; }
};


template <typename T>
__global__ void multiplyRealHyperDualSparseKernel(
    const thrust::complex<T>* real_values1, const int* real_row_ptrs1, const int* real_col_indices1,
    const thrust::complex<T>* dual_values1, const int* dual_row_ptrs1, const int* dual_col_indices1,
    const thrust::complex<T>* hyperDual_values1, const int* hyperDual_row_ptrs1, const int* hyperDual_col_indices1,
    const thrust::complex<T>* real_values2, const int* real_row_ptrs2, const int* real_col_indices2,
    const thrust::complex<T>* dual_values2, const int* dual_row_ptrs2, const int* dual_col_indices2,
    const thrust::complex<T>* hyperDual_values2, const int* hyperDual_row_ptrs2, const int* hyperDual_col_indices2,
    thrust::complex<T>* result_values, const int* result_row_ptrs, const int* result_col_indices,
    int batch_size, int real_size, int dual_size, int nnz_result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < nnz_result) {
        // Decompose the index to locate the position in the sparse result
        int row = 0;
        while (idx >= result_row_ptrs[row + 1]) ++row; // Find the corresponding row for idx
        int col = result_col_indices[idx];

        int batch_idx = row / real_size;
        int real_idx = row % real_size;
        int dual_row = col / dual_size;
        int dual_col = col % dual_size;

        // Indices in sparse format
        int real_global_idx = batch_idx * real_size + real_idx;

        // Locate corresponding indices in the sparse input tensors
        thrust::complex<T> real1 = real_values1[real_global_idx];
        thrust::complex<T> real2 = real_values2[real_global_idx];

        thrust::complex<T> dual1 = thrust::complex<T>(0.0, 0.0);
        thrust::complex<T> dual2 = thrust::complex<T>(0.0, 0.0);
        thrust::complex<T> hyperDual1 = thrust::complex<T>(0.0, 0.0);
        thrust::complex<T> hyperDual2 = thrust::complex<T>(0.0, 0.0);

        // Find non-zero values for dual and hyperdual parts
        for (int k = dual_row_ptrs1[real_global_idx]; k < dual_row_ptrs1[real_global_idx + 1]; ++k) {
            if (dual_col_indices1[k] == dual_row) {
                dual1 = dual_values1[k];
                break;
            }
        }
        for (int k = dual_row_ptrs2[real_global_idx]; k < dual_row_ptrs2[real_global_idx + 1]; ++k) {
            if (dual_col_indices2[k] == dual_col) {
                dual2 = dual_values2[k];
                break;
            }
        }
        for (int k = hyperDual_row_ptrs1[real_global_idx]; k < hyperDual_row_ptrs1[real_global_idx + 1]; ++k) {
            if (hyperDual_col_indices1[k] == col) {
                hyperDual1 = hyperDual_values1[k];
                break;
            }
        }
        for (int k = hyperDual_row_ptrs2[real_global_idx]; k < hyperDual_row_ptrs2[real_global_idx + 1]; ++k) {
            if (hyperDual_col_indices2[k] == col) {
                hyperDual2 = hyperDual_values2[k];
                break;
            }
        }

        // Compute the result for the sparse entry
        result_values[idx] = real1 * hyperDual2 + real2 * hyperDual1 + dual1 * dual2;
    }
}

template <typename T>
class VectorHyperDualSparseCuda {
private:
    int batch_size_;                 // Batch dimension (M)
    int real_size_;                  // Vector length (N)
    int dual_size_;                  // Dual dimension (D)

    // CSR format for real, dual, and hyperdual parts
    thrust::complex<T>* real_values_;       // Non-zero values for real part
    int* real_col_indices_;                 // Column indices for real part
    int* real_row_pointers_;                // Row pointers for real part

    thrust::complex<T>* dual_values_;       // Non-zero values for dual part
    int* dual_col_indices_;                 // Column indices for dual part
    int* dual_row_pointers_;                // Row pointers for dual part

    thrust::complex<T>* hyperdual_values_;  // Non-zero values for hyperdual part
    int* hyperdual_col_indices_;            // Column indices for hyperdual part
    int* hyperdual_row_pointers_;           // Row pointers for hyperdual part

    int nnz_real_;                          // Number of non-zero elements in real part
    int nnz_dual_;                          // Number of non-zero elements in dual part
    int nnz_hyperdual_;                     // Number of non-zero elements in hyperdual part

public:
    // Constructor for externally managed memory
    VectorHyperDualSparseCuda(
        int batch_size, int real_size, int dual_size,
        thrust::complex<T>* real_values, int* real_col_indices, int* real_row_pointers, int nnz_real,
        thrust::complex<T>* dual_values, int* dual_col_indices, int* dual_row_pointers, int nnz_dual,
        thrust::complex<T>* hyperdual_values, int* hyperdual_col_indices, int* hyperdual_row_pointers, int nnz_hyperdual)
        : batch_size_(batch_size), real_size_(real_size), dual_size_(dual_size),
          real_values_(real_values), real_col_indices_(real_col_indices), real_row_pointers_(real_row_pointers), nnz_real_(nnz_real),
          dual_values_(dual_values), dual_col_indices_(dual_col_indices), dual_row_pointers_(dual_row_pointers), nnz_dual_(nnz_dual),
          hyperdual_values_(hyperdual_values), hyperdual_col_indices_(hyperdual_col_indices), hyperdual_row_pointers_(hyperdual_row_pointers), nnz_hyperdual_(nnz_hyperdual) {}

    // Elementwise Multiplication on GPU
    __global__ void elementwiseMultiplyKernel(
        const VectorHyperDualSparseCuda<T> other, VectorHyperDualSparseCuda<T> result) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        // Real part multiplication
        if (idx < nnz_real_) {
            result.real_values_[idx] = real_values_[idx] * other.real_values_[idx];
        }

        // Dual part multiplication
        if (idx < nnz_dual_) {
            int row = 0;
            while (idx >= dual_row_pointers_[row + 1]) ++row;  // Find row for this index
            int col = dual_col_indices_[idx];

            thrust::complex<T> real1 = real_values_[row];
            thrust::complex<T> real2 = other.real_values_[row];
            thrust::complex<T> dual1 = dual_values_[idx];
            thrust::complex<T> dual2 = other.dual_values_[idx];

            result.dual_values_[idx] = real1 * dual2 + real2 * dual1;
        }

        // Hyperdual part multiplication
        if (idx < nnz_hyperdual_) {
            int row = 0;
            while (idx >= hyperdual_row_pointers_[row + 1]) ++row;  // Find row for this index
            int col = hyperdual_col_indices_[idx];

            thrust::complex<T> real1 = real_values_[row];
            thrust::complex<T> real2 = other.real_values_[row];
            thrust::complex<T> dual1 = dual_values_[row];
            thrust::complex<T> dual2 = other.dual_values_[col];
            thrust::complex<T> hyperdual1 = hyperdual_values_[idx];
            thrust::complex<T> hyperdual2 = other.hyperdual_values_[idx];

            result.hyperdual_values_[idx] =
                real1 * hyperdual2 + real2 * hyperdual1 + dual1 * dual2;
        }
    }

    // Accessors
    thrust::complex<T>* realValues() { return real_values_; }
    const thrust::complex<T>* realValues() const { return real_values_; }
    int* realColIndices() { return real_col_indices_; }
    const int* realColIndices() const { return real_col_indices_; }
    int* realRowPointers() { return real_row_pointers_; }
    const int* realRowPointers() const { return real_row_pointers_; }

    thrust::complex<T>* dualValues() { return dual_values_; }
    const thrust::complex<T>* dualValues() const { return dual_values_; }
    int* dualColIndices() { return dual_col_indices_; }
    const int* dualColIndices() const { return dual_col_indices_; }
    int* dualRowPointers() { return dual_row_pointers_; }
    const int* dualRowPointers() const { return dual_row_pointers_; }

    thrust::complex<T>* hyperdualValues() { return hyperdual_values_; }
    const thrust::complex<T>* hyperdualValues() const { return hyperdual_values_; }
    int* hyperdualColIndices() { return hyperdual_col_indices_; }
    const int* hyperdualColIndices() const { return hyperdual_col_indices_; }
    int* hyperdualRowPointers() { return hyperdual_row_pointers_; }
    const int* hyperdualRowPointers() const { return hyperdual_row_pointers_; }

    int batchSize() const { return batch_size_; }
    int size() const { return real_size_; }
    int dualSize() const { return dual_size_; }
};

template <typename T>
class MatrixSparseCuda {
private:
    using ComplexT = std::complex<T>;

    // Dimensions
    int batch_size_;
    int rows_;
    int cols_;

    // CSR Representation for each batch
    ComplexT* values_;       // Non-zero values
    int* col_indices_;       // Column indices for the non-zero values
    int* row_pointers_;      // Row pointers for each row in the CSR format
    int nnz_;                // Total number of non-zero values across all batches

    bool owns_memory_;       // Indicates if memory is managed internally

public:
    // Constructor for externally managed memory
    MatrixSparseCuda(int batch_size, int rows, int cols, ComplexT* values, int* col_indices, int* row_pointers, int nnz)
        : batch_size_(batch_size), rows_(rows), cols_(cols),
          values_(values), col_indices_(col_indices), row_pointers_(row_pointers), nnz_(nnz), owns_memory_(false) {}

    // Constructor for internal memory allocation
    MatrixSparseCuda(int batch_size, int rows, int cols, int nnz)
        : batch_size_(batch_size), rows_(rows), cols_(cols), nnz_(nnz), owns_memory_(true) {
        if (batch_size <= 0 || rows <= 0 || cols <= 0 || nnz <= 0) {
            throw std::invalid_argument("All dimensions and nnz must be positive.");
        }

        // Allocate memory for CSR components
        cudaMalloc(&values_, nnz * sizeof(ComplexT));
        cudaMalloc(&col_indices_, nnz * sizeof(int));
        cudaMalloc(&row_pointers_, (rows + 1) * sizeof(int));
    }

    // Destructor
    ~MatrixSparseCuda() {
        if (owns_memory_) {
            cudaFree(values_);
            cudaFree(col_indices_);
            cudaFree(row_pointers_);
        }
    }

    // Initialize the matrix from host data
    void initialize(const ComplexT* host_values, const int* host_col_indices, const int* host_row_pointers, int nnz) {
        if (nnz != nnz_) {
            throw std::invalid_argument("Input nnz does not match matrix nnz.");
        }

        cudaMemcpy(values_, host_values, nnz * sizeof(ComplexT), cudaMemcpyHostToDevice);
        cudaMemcpy(col_indices_, host_col_indices, nnz * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(row_pointers_, host_row_pointers, (rows_ + 1) * sizeof(int), cudaMemcpyHostToDevice);
    }

    // Sparse Matrix-Vector Product
    void matrixVectorProduct(const ComplexT* vector, ComplexT* result) const {
        dim3 blockDim(256);
        dim3 gridDim((rows_ * batch_size_ + blockDim.x - 1) / blockDim.x);

        sparseMatrixVectorProductKernel<<<gridDim, blockDim>>>(
            values_, col_indices_, row_pointers_, vector, result, rows_, cols_, batch_size_);
    }

    // Elementwise addition of two sparse matrices
    MatrixSparseCuda<T> elementwiseAdd(const MatrixSparseCuda<T>& other) const {
        if (batch_size_ != other.batch_size_ || rows_ != other.rows_ || cols_ != other.cols_) {
            throw std::invalid_argument("Matrix dimensions do not match for addition.");
        }

        // Resultant matrix: for simplicity, assume nnz is the sum of nnz of the two matrices
        int result_nnz = nnz_ + other.nnz_;
        MatrixSparseCuda<T> result(batch_size_, rows_, cols_, result_nnz);

        // Perform sparse addition (kernel implementation needed)
        dim3 blockDim(256);
        dim3 gridDim((result_nnz + blockDim.x - 1) / blockDim.x);
        sparseElementwiseAddKernel<<<gridDim, blockDim>>>(
            values_, col_indices_, row_pointers_, nnz_,
            other.values_, other.col_indices_, other.row_pointers_, other.nnz_,
            result.values_, result.col_indices_, result.row_pointers_, result_nnz,
            rows_, cols_, batch_size_);

        return result;
    }

    // Accessors
    ComplexT* values() { return values_; }
    const ComplexT* values() const { return values_; }
    int* colIndices() { return col_indices_; }
    const int* colIndices() const { return col_indices_; }
    int* rowPointers() { return row_pointers_; }
    const int* rowPointers() const { return row_pointers_; }
    int batchSize() const { return batch_size_; }
    int rows() const { return rows_; }
    int cols() const { return cols_; }
    int nnz() const { return nnz_; }
}; // end of MatrixSparseCuda



template <typename T>
class MatrixDualSparseCuda {
private:
    using ComplexT = std::complex<T>;

    // Dimensions
    int batch_size_;  // Batch dimension (M)
    int rows_;        // Number of rows in each matrix (N)
    int cols_;        // Number of columns in each matrix (L)
    int dual_dim_;    // Dual dimension (D)

    // CSR Representation for Primal and Dual Data
    ComplexT* primal_values_;       // Non-zero values for primal part
    int* primal_col_indices_;       // Column indices for primal part
    int* primal_row_pointers_;      // Row pointers for primal part
    int primal_nnz_;                // Number of non-zero elements in primal part

    ComplexT* dual_values_;         // Non-zero values for dual part
    int* dual_col_indices_;         // Column indices for dual part
    int* dual_row_pointers_;        // Row pointers for dual part
    int dual_nnz_;                  // Number of non-zero elements in dual part

    bool owns_memory_;              // Indicates if memory is managed internally

public:
    // Constructor for externally managed memory
    MatrixDualSparseCuda(
        int batch_size, int rows, int cols, int dual_dim,
        ComplexT* primal_values, int* primal_col_indices, int* primal_row_pointers, int primal_nnz,
        ComplexT* dual_values, int* dual_col_indices, int* dual_row_pointers, int dual_nnz)
        : batch_size_(batch_size), rows_(rows), cols_(cols), dual_dim_(dual_dim),
          primal_values_(primal_values), primal_col_indices_(primal_col_indices), primal_row_pointers_(primal_row_pointers), primal_nnz_(primal_nnz),
          dual_values_(dual_values), dual_col_indices_(dual_col_indices), dual_row_pointers_(dual_row_pointers), dual_nnz_(dual_nnz),
          owns_memory_(false) {}

    // Constructor for internal memory allocation
    MatrixDualSparseCuda(int batch_size, int rows, int cols, int dual_dim, int primal_nnz, int dual_nnz)
        : batch_size_(batch_size), rows_(rows), cols_(cols), dual_dim_(dual_dim),
          primal_nnz_(primal_nnz), dual_nnz_(dual_nnz), owns_memory_(true) {
        // Allocate memory for CSR components
        cudaMalloc(&primal_values_, primal_nnz_ * sizeof(ComplexT));
        cudaMalloc(&primal_col_indices_, primal_nnz_ * sizeof(int));
        cudaMalloc(&primal_row_pointers_, (rows_ + 1) * sizeof(int));

        cudaMalloc(&dual_values_, dual_nnz_ * sizeof(ComplexT));
        cudaMalloc(&dual_col_indices_, dual_nnz_ * sizeof(int));
        cudaMalloc(&dual_row_pointers_, (rows_ + 1) * sizeof(int));
    }

    // Destructor
    ~MatrixDualSparseCuda() {
        if (owns_memory_) {
            cudaFree(primal_values_);
            cudaFree(primal_col_indices_);
            cudaFree(primal_row_pointers_);
            cudaFree(dual_values_);
            cudaFree(dual_col_indices_);
            cudaFree(dual_row_pointers_);
        }
    }

    // Initialize primal and dual data from host
    void initializePrimal(const ComplexT* host_values, const int* host_col_indices, const int* host_row_pointers) {
        cudaMemcpy(primal_values_, host_values, primal_nnz_ * sizeof(ComplexT), cudaMemcpyHostToDevice);
        cudaMemcpy(primal_col_indices_, host_col_indices, primal_nnz_ * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(primal_row_pointers_, host_row_pointers, (rows_ + 1) * sizeof(int), cudaMemcpyHostToDevice);
    }

    void initializeDual(const ComplexT* host_values, const int* host_col_indices, const int* host_row_pointers) {
        cudaMemcpy(dual_values_, host_values, dual_nnz_ * sizeof(ComplexT), cudaMemcpyHostToDevice);
        cudaMemcpy(dual_col_indices_, host_col_indices, dual_nnz_ * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(dual_row_pointers_, host_row_pointers, (rows_ + 1) * sizeof(int), cudaMemcpyHostToDevice);
    }

    // Elementwise Addition
    MatrixDualSparseCuda<T> elementwiseAdd(const MatrixDualSparseCuda<T>& other) const {
        if (batch_size_ != other.batch_size_ || rows_ != other.rows_ || cols_ != other.cols_ || dual_dim_ != other.dual_dim_) {
            throw std::invalid_argument("Matrix dimensions do not match for addition.");
        }

        // Simplified implementation: assumes result nnz is the sum of nnz of both matrices
        int result_primal_nnz = primal_nnz_ + other.primal_nnz_;
        int result_dual_nnz = dual_nnz_ + other.dual_nnz_;
        MatrixDualSparseCuda<T> result(batch_size_, rows_, cols_, dual_dim_, result_primal_nnz, result_dual_nnz);

        // Add primal components (CSR addition kernel implementation needed)
        sparseCSRAddKernel<<<1, 256>>>(
            primal_values_, primal_col_indices_, primal_row_pointers_, primal_nnz_,
            other.primal_values_, other.primal_col_indices_, other.primal_row_pointers_, other.primal_nnz_,
            result.primal_values_, result.primal_col_indices_, result.primal_row_pointers_, result_primal_nnz);

        // Add dual components
        sparseCSRAddKernel<<<1, 256>>>(
            dual_values_, dual_col_indices_, dual_row_pointers_, dual_nnz_,
            other.dual_values_, other.dual_col_indices_, other.dual_row_pointers_, other.dual_nnz_,
            result.dual_values_, result.dual_col_indices_, result.dual_row_pointers_, result_dual_nnz);

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

    int batchSize() const { return batch_size_; }
    int rows() const { return rows_; }
    int cols() const { return cols_; }
    int dualDim() const { return dual_dim_; }
}; // end of MatrixDualSparseCuda


}  // namespace janus
#endif // _CU_DUAL_TENSOR_HPP