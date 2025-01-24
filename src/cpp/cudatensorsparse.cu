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
#include <thrust/complex.h>

namespace janus {

    // A sparse dual vector container on the device.
    template <typename T>
    class SparseVectorDualDevice
    {
    public:
        int real_size_;  // length of real part
        int dual_size_;  // total dimension of dual space

        // Real part: pointer to array of length real_size_
        thrust::complex<T>* real_;

        // Sparse dual part: only one nonzero entry
        int dual_idx_;                   // which coordinate in [0, dual_size_) is nonzero
        thrust::complex<T>* dual_value_; // pointer to a single complex number on device

        // ----------------- Constructors on Device -----------------

        // Device-side default constructor
        __device__
        SparseVectorDualDevice()
            : real_size_(0),
            dual_size_(0),
            real_(nullptr),
            dual_idx_(-1),
            dual_value_(nullptr)
        { }

        // Device-side full constructor with pointers and index
        __device__
        SparseVectorDualDevice(int real_size,
                            int dual_size,
                            thrust::complex<T>* real_data,
                            int dual_idx,
                            thrust::complex<T>* dual_value)
            : real_size_(real_size),
            dual_size_(dual_size),
            real_(real_data),
            dual_idx_(dual_idx),
            dual_value_(dual_value)
        { }

        // Disable device-side copy constructor
        __device__
        SparseVectorDualDevice(const SparseVectorDualDevice&) = delete;

        // Device-side move constructor
        __device__
        SparseVectorDualDevice(SparseVectorDualDevice&& other) noexcept
            : real_size_(other.real_size_),
            dual_size_(other.dual_size_),
            real_(other.real_),
            dual_idx_(other.dual_idx_),
            dual_value_(other.dual_value_)
        {
            // Null out the source
            other.real_size_  = 0;
            other.dual_size_  = 0;
            other.real_       = nullptr;
            other.dual_idx_   = -1;
            other.dual_value_ = nullptr;
        }

        // Device-side move assignment
        __device__
        SparseVectorDualDevice& operator=(SparseVectorDualDevice&& other) noexcept
        {
            if (this != &other) {
                real_size_      = other.real_size_;
                dual_size_      = other.dual_size_;
                real_           = other.real_;
                dual_idx_       = other.dual_idx_;
                dual_value_     = other.dual_value_;

                other.real_size_  = 0;
                other.dual_size_  = 0;
                other.real_       = nullptr;
                other.dual_idx_   = -1;
                other.dual_value_ = nullptr;
            }
            return *this;
        }
    };

    
    template <typename T>
    __device__ void atomicAddComplex(thrust::complex<T>* dest, thrust::complex<T> val)
    {
        // For thrust::complex<T>, memory layout is typically [ real, imag ].
        // Reinterpret the pointer as T[2], so:
        T* base = reinterpret_cast<T*>(dest);
        // base[0] = real part, base[1] = imag part
        atomicAdd(&base[0], val.real());
        atomicAdd(&base[1], val.imag());
    }





    // Device function for multiplying two sparse dual-vectors.
    // - a_real, b_real : pointers to arrays of length real_size
    // - a_dual, b_dual : pointers to a SINGLE complex value (the one nonzero in the dual part)
    // - a_idx, b_idx   : positions at which 'a_dual' or 'b_dual' is nonzero
    // - real_size      : length of the real arrays
    // 
    // Output is written into:
    // - r_real (length real_size)
    // - r_dual (SINGLE complex value for the result's dual)
    // - r_idx  : the position of the result's nonzero dual
    //
    // This is a minimal example that sums the derivative contributions
    // from each real index into a single dual value.  For large real_size,
    // you might want a parallel reduction instead of a single-thread loop.
    template <typename T>
    __device__
    void VectorRealDualProductSparse(
        const thrust::complex<T>* a_real,
        const thrust::complex<T>* a_dual,
        int                       a_idx,
        const thrust::complex<T>* b_real,
        const thrust::complex<T>* b_dual,
        int                       b_idx,
        int                       real_size,

        thrust::complex<T>*       r_real,
        thrust::complex<T>*       r_dual,
        int*                      r_idx
    )
    {
        // Usually set r_idx once:
        // For safety, do it in thread 0 of block 0 or via another logic
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            *r_idx = a_idx;  // or any logic you want
            // Initialize r_dual to 0
            *r_dual = thrust::complex<T>(0,0);
        }

        // Compute global thread index
        unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;

        // Parallel elementwise multiplication for real part
        if (i < real_size) {
            // Real part
            r_real[i] = a_real[i] * b_real[i];

            // partial derivative for index i
            thrust::complex<T> partialVal = a_real[i]*(*b_dual) + b_real[i]*(*a_dual);
            
            // Atomic add into r_dual
            atomicAddComplex(r_dual, partialVal);
        }
    }

    //Generate a global wrapper for the VectorRealDualProductSparse function
    template <typename T>
    __global__
    void VectorRealDualProductSparseKernel(
        const thrust::complex<T>* a_real,
        const thrust::complex<T>* a_dual,
        int                       a_idx,
        const thrust::complex<T>* b_real,
        const thrust::complex<T>* b_dual,
        int                       b_idx,
        int                       real_size,

        thrust::complex<T>*       r_real,
        thrust::complex<T>*       r_dual,
        int*                      r_idx
    )
    {
        VectorRealDualProductSparse(a_real, a_dual, a_idx, b_real, b_dual, b_idx, real_size, r_real, r_dual, r_idx);
    }

    /**
     * VectorDualIndexGetSparse
     *
     * Given a sparse dual vector (a_real, a_dual, a_idx) of length `real_size`,
     * extract the subrange [start, end) into a new sparse dual vector:
     *
     * - result_real has length = (end - start).
     * - result_dual is a single complex value (non-zero if the old index a_idx
     *   lies within [start, end), otherwise zero).
     * - result_idx is the new index of the nonzero dual; -1 if none.
     */
    template <typename T>
    __device__
    void VectorDualIndexGetSparse(const thrust::complex<T>* a_real,
                                const thrust::complex<T>* a_dual, // single value
                                int                        a_idx,  // index of nonzero dual
                                int                        real_size,

                                int start,
                                int end,

                                thrust::complex<T>* result_real, // length = (end - start)
                                thrust::complex<T>* result_dual, // single value
                                int*                result_idx)   // new sparse index
    {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        int new_size = end - start;
        
        // 1) Copy real part in the range [start, end)
        if (i < new_size) {
            // a_real has length 'real_size'; we index from 'start + i'
            // into the new array at position 'i'
            result_real[i] = a_real[start + i];
        }
        
        // 2) Handle the single dual entry in one thread to avoid races
        //    (e.g. the first thread of the first block).
        __syncthreads();
        if (blockIdx.x == 0 && threadIdx.x == 0)
        {
            // If the old dual index is within [start, end), map it to [0, new_size).
            if (a_idx >= start && a_idx < end) {
                *result_idx  = a_idx - start;  // shift to new coordinate
                *result_dual = *a_dual;        // copy the single dual value
            } else {
                // No nonzero dual in this subrange
                *result_idx  = -1;
                *result_dual = thrust::complex<T>(0, 0);
            }
        }
    }

    //Generate a global wrapper for the VectorDualIndexGetSparse function
    template <typename T>
    __global__
    void VectorDualIndexGetSparseKernel(const thrust::complex<T>* a_real,
                                        const thrust::complex<T>* a_dual,
                                        int                        a_idx,
                                        int                        real_size,

                                        int                        start,
                                        int                        end,

                                        thrust::complex<T>*        result_real,
                                        thrust::complex<T>*        result_dual,
                                        int*                       result_idx)
    {
        VectorDualIndexGetSparse(a_real, a_dual, a_idx, real_size, start, end, result_real, result_dual, result_idx);
    }

    /**
     * VectorDualIndexPutSparse
     *
     * "Put" a smaller sparse vector (input) into a subrange [start, end)
     * of a larger sparse vector (result).
     *
     * - input_real: length = sub_len = (end - start)
     * - input_dual: single nonzero dual value
     * - input_idx : dual index of the input, in [0, sub_len) or -1 if none
     *
     * - result_real: length = real_size
     * - result_dual: single dual value for the result
     * - result_idx : dual index for the result
     *
     * The subrange has size (end - start). If input_idx is in [0, sub_len),
     * we map it to (start + input_idx) in the result.
     */
    template <typename T>
    __device__
    void VectorDualIndexPutSparse(
        const thrust::complex<T>* input_real,
        const thrust::complex<T>* input_dual,
        int                       input_idx,
        int                       sub_len,      // = end - start

        int start,
        int end,

        thrust::complex<T>*       result_real,
        thrust::complex<T>*       result_dual,
        int*                      result_idx,
        int                       real_size
    )
    {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i < sub_len) {
            // Copy real component
            result_real[start + i] = input_real[i];
        }

        // Single dual index handled once
        __syncthreads();
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            if (input_idx >= 0 && input_idx < sub_len) {
                // The new dual index is offset by 'start'
                *result_idx  = start + input_idx;
                *result_dual = *input_dual;
            } else {
                *result_idx  = -1;
                *result_dual = thrust::complex<T>(0, 0);
            }
        }
    }

    // Kernel wrapper
    template <typename T>
    __global__
    void VectorDualIndexPutSparseKernel(
        const thrust::complex<T>* input_real,
        const thrust::complex<T>* input_dual,
        int                       input_idx,
        int                       sub_len,
        int                       start,
        int                       end,

        thrust::complex<T>*       result_real,
        thrust::complex<T>*       result_dual,
        int*                      result_idx,
        int                       real_size
    )
    {
        VectorDualIndexPutSparse(
            input_real, input_dual, input_idx, sub_len,
            start, end,
            result_real, result_dual, result_idx, real_size
        );
    }

    /**
     * A sparse dual vector: the real part is length real_size_,
     * and the dual part has only one nonzero entry stored at dual_idx_.
     */
    template <typename T>
    class VectorDualSparse
    {
    public:
        int real_size_;                // Length of real part
        int dual_size_;                // Dimension of the dual space

        // Real part: pointer to array of length real_size_
        thrust::complex<T>* real_;     

        // Sparse dual part: only one nonzero entry
        int                 dual_idx_;     // which coordinate in [0, dual_size_) is nonzero
        thrust::complex<T>* dual_value_;   // pointer to a single complex number

        // ------------ Constructors ------------

        // Default constructor
        __host__ __device__
        VectorDualSparse()
            : real_size_(0),
            dual_size_(0),
            real_(nullptr),
            dual_idx_(-1),
            dual_value_(nullptr)
        { }

        // Full constructor
        __host__ __device__
        VectorDualSparse(int real_size,
                        int dual_size,
                        thrust::complex<T>* real_data,
                        int dual_idx,
                        thrust::complex<T>* dual_val)
            : real_size_(real_size),
            dual_size_(dual_size),
            real_(real_data),
            dual_idx_(dual_idx),
            dual_value_(dual_val)
        { }
    };

    /**
     * VectorRealDualProductSparse
     *
     * Multiply two sparse dual vectors, producing a sparse dual result.
     *
     * Inputs for vector 'a':
     *  - a_real:       array of length real_size
     *  - a_idx:        index of a's single nonzero dual in [0..real_size), or -1 if none
     *  - a_dual_value: pointer to a single complex storing a's dual contribution
     *
     * Inputs for vector 'b':
     *  - b_real
     *  - b_idx
     *  - b_dual_value
     *
     * Output for result:
     *  - r_real:       array of length real_size
     *  - r_idx:        single index for the result's nonzero dual
     *  - r_dual_value: pointer to a single complex for the dual contribution
     *
     * The real part is done in parallel. The dual part is handled in one thread
     * (after a __syncthreads) to avoid race conditions.
     */
    template <typename T>
    __device__
    void VectorRealDualProductSparse(
        // Vector a
        const thrust::complex<T>* a_real,
        int                       a_idx,
        const thrust::complex<T>* a_dual_value,

        // Vector b
        const thrust::complex<T>* b_real,
        int                       b_idx,
        const thrust::complex<T>* b_dual_value,

        // Dimension
        int real_size,

        // Result
        thrust::complex<T>*       r_real,
        int*                      r_idx,
        thrust::complex<T>*       r_dual_value
    )
    {
        // 1) Parallel real-part multiplication
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i < real_size) {
            r_real[i] = a_real[i] * b_real[i]; 
        }

        // 2) Handle the single dual value once per kernel (e.g. in thread 0).
        __syncthreads();
        if (blockIdx.x == 0 && threadIdx.x == 0) 
        {
            // CASE A: If both indices are -1 => no dual
            if (a_idx < 0 && b_idx < 0) {
                *r_idx       = -1;
                *r_dual_value = thrust::complex<T>(0, 0);
                return;
            }

            // CASE B: If exactly one index is valid => use it
            if (a_idx >= 0 && b_idx < 0) {
                *r_idx = a_idx;
                // sum_{i} [ b_real[i] * a_dual_value ]
                thrust::complex<T> sum_val(0, 0);
                for (int j = 0; j < real_size; ++j) {
                    sum_val += b_real[j] * (*a_dual_value);
                }
                *r_dual_value = sum_val;
                return;
            }
            if (a_idx < 0 && b_idx >= 0) {
                *r_idx = b_idx;
                // sum_{i} [ a_real[i] * b_dual_value ]
                thrust::complex<T> sum_val(0, 0);
                for (int j = 0; j < real_size; ++j) {
                    sum_val += a_real[j] * (*b_dual_value);
                }
                *r_dual_value = sum_val;
                return;
            }

            // CASE C: Both indices valid. If they match, combine
            if (a_idx == b_idx) {
                *r_idx = a_idx;  // (or b_idx)
                // sum_{i} [ a_real[i]*b_dual + b_real[i]*a_dual ]
                thrust::complex<T> sum_val(0, 0);
                for (int j = 0; j < real_size; ++j) {
                    sum_val += a_real[j]*(*b_dual_value) + b_real[j]*(*a_dual_value);
                }
                *r_dual_value = sum_val;
                return;
            }

            // CASE D: Both are valid but different. 
            // A simple fallback is to say no single index can hold them both => set to -1, 0
            // or pick one. We'll do the "no dual" approach here.
            *r_idx        = -1;
            *r_dual_value = thrust::complex<T>(0, 0);
        }
    }
    

    template <typename T>
    __global__
    void VectorRealDualProductSparseKernel(
        // inputs for a
        const thrust::complex<T>* a_real,
        int                       a_idx,
        const thrust::complex<T>* a_dual_value,

        // inputs for b
        const thrust::complex<T>* b_real,
        int                       b_idx,
        const thrust::complex<T>* b_dual_value,

        // dimension
        int real_size,

        // outputs for result
        thrust::complex<T>*       r_real,
        int*                      r_idx,
        thrust::complex<T>*       r_dual_value
    )
    {
        VectorRealDualProductSparse(
            a_real, a_idx, a_dual_value,
            b_real, b_idx, b_dual_value,
            real_size,
            r_real, r_idx, r_dual_value
        );
    }


    /**
     * VectorDualIndexGetSparse
     *
     * Extracts a subrange [start, end) from a sparse dual vector (a_real, a_idx, a_dual_value)
     * and places it in (result_real, result_idx, result_dual_value).
     *
     * - a_real:        real array of length real_size
     * - a_idx:         sparse index in [0..real_size) or -1 if no dual
     * - a_dual_value:  pointer to a single dual value
     * - real_size:     length of a_real
     * - start, end:    subrange boundaries, 0 <= start < end <= real_size
     *
     * Outputs:
     * - result_real:       length = (end - start)
     * - *result_idx:       new dual index in [0..(end-start)) or -1
     * - *result_dual_value single dual
     */
    template <typename T>
    __device__
    void VectorDualIndexGetSparse(
        const thrust::complex<T>* a_real,
        int                       a_idx,
        const thrust::complex<T>* a_dual_value,
        int                       real_size,

        int start,
        int end,

        thrust::complex<T>*       result_real,       // length = (end - start)
        int*                      result_idx,
        thrust::complex<T>*       result_dual_value
    )
    {
        // 1) Parallel copy of real part in [start, end)
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        int sub_len = end - start;
        if (i < sub_len) {
            // Copy from a_real[start + i] => result_real[i]
            result_real[i] = a_real[start + i];
        }

        // 2) Set the single dual index/value in one thread
        __syncthreads();
        if (blockIdx.x == 0 && threadIdx.x == 0)
        {
            // If a_idx is in [start, end), shift it to [0..sub_len)
            if (a_idx >= start && a_idx < end) {
                *result_idx        = a_idx - start;
                *result_dual_value = *a_dual_value;
            } else {
                // No nonzero dual in this subrange
                *result_idx        = -1;
                *result_dual_value = thrust::complex<T>(0,0);
            }
        }
    }

    template <typename T>
    __global__
    void VectorDualIndexGetSparseKernel(
        const thrust::complex<T>* a_real,
        int                       a_idx,
        const thrust::complex<T>* a_dual_value,
        int                       real_size,
        int                       start,
        int                       end,

        thrust::complex<T>*       result_real,
        int*                      result_idx,
        thrust::complex<T>*       result_dual_value
    )
    {
        VectorDualIndexGetSparse(
            a_real, a_idx, a_dual_value,
            real_size,
            start, end,
            result_real, result_idx, result_dual_value
        );
    }

    /**
     * VectorDualIndexPutSparse
     *
     * Copies a smaller sparse dual vector ("input" of size sub_len = end - start)
     * into a subrange [start, end) of a larger sparse vector ("result" of size real_size).
     *
     * Input sparse vector:
     *   - input_real:       length = (end - start)
     *   - input_idx:        single dual index in [0..(end - start)) or -1
     *   - input_dual_value: pointer to a single complex for the dual
     *
     * The result has:
     *   - result_real:  length = real_size
     *   - result_idx, result_dual_value: single index/value
     *
     * The subrange [start..end) in the result is overwritten with the input's real data.
     * If input_idx is valid, the result_idx = start + input_idx, and we copy the input_dual_value.
     * Otherwise, result_idx = -1, result_dual_value = (0,0).
     */
    template <typename T>
    __device__
    void VectorDualIndexPutSparse(
        // Input sparse vector
        const thrust::complex<T>* input_real,         // length = sub_len
        int                       input_idx,
        const thrust::complex<T>* input_dual_value,
        int                       sub_len,

        // Where to copy
        int start,
        int end,

        // Output sparse vector
        thrust::complex<T>*       result_real,        // length = real_size
        int*                      result_idx,
        thrust::complex<T>*       result_dual_value,
        int                       real_size
    )
    {
        // 1) Parallel copy for the real part.
        //    sub_len = (end - start).
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i < sub_len) {
            // Copy input_real[i] into result_real[start + i].
            result_real[start + i] = input_real[i];
        }

        // 2) Single thread handles the dual index/value.
        __syncthreads();
        if (blockIdx.x == 0 && threadIdx.x == 0)
        {
            if (input_idx >= 0 && input_idx < sub_len) {
                // The new dual index is offset by 'start'
                *result_idx       = start + input_idx;
                *result_dual_value = *input_dual_value;
            }
            else {
                // No valid dual
                *result_idx       = -1;
                *result_dual_value = thrust::complex<T>(0, 0);
            }
        }
    }

    /**
     * VectorDualIndexPutSparseKernel
     *
     * Global kernel for sparse "IndexPut"
     */
    template <typename T>
    __global__
    void VectorDualIndexPutSparseKernel(
        const thrust::complex<T>* input_real,
        int                       input_idx,
        const thrust::complex<T>* input_dual_value,
        int                       sub_len,   // = end - start

        int start,
        int end,

        thrust::complex<T>*       result_real,
        int*                      result_idx,
        thrust::complex<T>*       result_dual_value,
        int                       real_size
    )
    {
        VectorDualIndexPutSparse(
            input_real, input_idx, input_dual_value, sub_len,
            start, end,
            result_real, result_idx, result_dual_value, real_size
        );
    }

    /**
     * VectorDualElementwiseAddSparse
     *
     * Adds two sparse dual vectors 'a' and 'b', producing a sparse dual result 'r'.
     *
     * Each vector has:
     *   - real[i], i in [0..real_size)
     *   - dual_idx, in [-1..real_size)
     *   - dual_value (single thrust::complex<T>)
     *
     * The result has:
     *   - r_real[i] = a_real[i] + b_real[i] (for i in [0..real_size))
     *   - r_idx, r_dual_value => single nonzero dual entry
     * 
     * Index logic for r_idx, r_dual_value:
     *   1) If both a_idx and b_idx == -1 => no dual => r_idx=-1, r_dual=0
     *   2) If exactly one is != -1 => take that one
     *   3) If both are != -1 and the same => add the dual values
     *   4) If both are != -1 and different => fallback => r_idx=-1, r_dual=0
     */
    template <typename T>
    __device__
    void VectorDualElementwiseAddSparse(
        // Vector a
        const thrust::complex<T>* a_real,
        int                       a_idx,
        const thrust::complex<T>* a_dual_value,

        // Vector b
        const thrust::complex<T>* b_real,
        int                       b_idx,
        const thrust::complex<T>* b_dual_value,

        // size
        int                       real_size,

        // Result
        thrust::complex<T>*       r_real,
        int*                      r_idx,
        thrust::complex<T>*       r_dual_value
    )
    {
        // 1) Parallel addition of the real part
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i < real_size) {
            r_real[i] = a_real[i] + b_real[i];
        }

        // 2) Single-thread logic for the dual part
        __syncthreads();
        if (blockIdx.x == 0 && threadIdx.x == 0)
        {
            // CASE 1: Both -1 => no dual
            if (a_idx < 0 && b_idx < 0) {
                *r_idx = -1;
                *r_dual_value = thrust::complex<T>(0,0);
                return;
            }
            // CASE 2: only A is valid
            if (a_idx >= 0 && b_idx < 0) {
                *r_idx = a_idx;
                *r_dual_value = *a_dual_value;
                return;
            }
            // only B is valid
            if (a_idx < 0 && b_idx >= 0) {
                *r_idx = b_idx;
                *r_dual_value = *b_dual_value;
                return;
            }

            // CASE 3: Both valid
            if (a_idx == b_idx) {
                // same index => sum dual values
                *r_idx = a_idx;
                *r_dual_value = (*a_dual_value) + (*b_dual_value);
            } else {
                // different => we can't represent two distinct nonzeros
                // fallback => no dual
                *r_idx = -1;
                *r_dual_value = thrust::complex<T>(0,0);
            }
        }
    }

    template <typename T>
    __global__
    void VectorDualElementwiseAddSparseKernel(
        // a
        const thrust::complex<T>* a_real,
        int                       a_idx,
        const thrust::complex<T>* a_dual_value,

        // b
        const thrust::complex<T>* b_real,
        int                       b_idx,
        const thrust::complex<T>* b_dual_value,

        // size
        int real_size,

        // result
        thrust::complex<T>*       r_real,
        int*                      r_idx,
        thrust::complex<T>*       r_dual_value
    )
    {
        VectorDualElementwiseAddSparse(
            a_real, a_idx, a_dual_value,
            b_real, b_idx, b_dual_value,
            real_size,
            r_real, r_idx, r_dual_value
        );
    }


/**
 * VectorDualElementwiseMultiplySparse
 *
 * "Elementwise multiply" of two sparse dual vectors a & b.
 *
 * Each vector has:
 *   - a_real[i], b_real[i], i in [0..real_size)
 *   - a_idx, a_dual_value => single dual
 *   - b_idx, b_dual_value => single dual
 *
 * The result r:
 *   - r_real[i] = a_real[i] * b_real[i]
 *   - r_idx, r_dual_value => single dual index/value.
 *
 * The partial-derivative summation is analogous to the product rule:
 *   sum_i [ a_real[i]*b_dual + b_real[i]*a_dual ] 
 * for the matching index case.
 */
template <typename T>
__device__
void VectorDualElementwiseMultiplySparse(
    // Vector a
    const thrust::complex<T>* a_real,
    int                       a_idx,
    const thrust::complex<T>* a_dual_value,

    // Vector b
    const thrust::complex<T>* b_real,
    int                       b_idx,
    const thrust::complex<T>* b_dual_value,

    // dimension
    int                       real_size,

    // result
    thrust::complex<T>*       r_real,
    int*                      r_idx,
    thrust::complex<T>*       r_dual_value
)
{
    // 1) Parallel: compute r_real[i] = a_real[i] * b_real[i]
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < real_size) {
        r_real[i] = a_real[i] * b_real[i];
    }

    // 2) Single-thread block for the dual index + value
    __syncthreads();
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        // CASE 1: Both -1 => no dual
        if (a_idx < 0 && b_idx < 0) {
            *r_idx = -1;
            *r_dual_value = thrust::complex<T>(0,0);
            return;
        }

        // CASE 2: Only a_idx is valid
        if (a_idx >= 0 && b_idx < 0) {
            *r_idx = a_idx;
            thrust::complex<T> sumVal(0,0);
            for (int j = 0; j < real_size; ++j) {
                sumVal += b_real[j] * (*a_dual_value);
            }
            *r_dual_value = sumVal;
            return;
        }
        // Only b_idx is valid
        if (a_idx < 0 && b_idx >= 0) {
            *r_idx = b_idx;
            thrust::complex<T> sumVal(0,0);
            for (int j = 0; j < real_size; ++j) {
                sumVal += a_real[j] * (*b_dual_value);
            }
            *r_dual_value = sumVal;
            return;
        }

        // CASE 3: Both valid
        if (a_idx == b_idx) {
            // same index => partial derivative sum
            *r_idx = a_idx; // or b_idx
            thrust::complex<T> sumVal(0,0);
            for (int j = 0; j < real_size; ++j) {
                // product rule: a[j]*b_dual + b[j]*a_dual
                sumVal += a_real[j]*(*b_dual_value) + b_real[j]*(*a_dual_value);
            }
            *r_dual_value = sumVal;
        } else {
            // different => fallback => no dual
            *r_idx = -1;
            *r_dual_value = thrust::complex<T>(0,0);
        }
    }
}

template <typename T>
__global__
void VectorDualElementwiseMultiplySparseKernel(
    const thrust::complex<T>* a_real,
    int                       a_idx,
    const thrust::complex<T>* a_dual_value,

    const thrust::complex<T>* b_real,
    int                       b_idx,
    const thrust::complex<T>* b_dual_value,

    int                       real_size,

    // outputs
    thrust::complex<T>*       r_real,
    int*                      r_idx,
    thrust::complex<T>*       r_dual_value
)
{
    VectorDualElementwiseMultiplySparse(
        a_real, a_idx, a_dual_value,
        b_real, b_idx, b_dual_value,
        real_size,
        r_real, r_idx, r_dual_value
    );
}


/**
 * VectorDualPowSparse
 *
 * Sparse analog of:
 *    result_real[i] = (a_real[i])^power
 *    result_dual[i] = power * (a_real[i]^(power - 1)) * a_dual[i]  [dense version]
 *
 * In the sparse case, we only have a single nonzero dual index/value.
 *
 * - a_real:        length real_size
 * - a_idx:         dual index in [-1..real_size)
 * - a_dual_value:  pointer to single complex
 * - power:         exponent
 * - real_size
 *
 * Outputs:
 * - r_real:        length real_size
 * - r_idx, r_dual_value => single dual
 */
template <typename T>
__device__
void VectorDualPowSparse(
    // Input
    const thrust::complex<T>* a_real,
    int                       a_idx,
    const thrust::complex<T>* a_dual_value,
    T                         power,
    int                       real_size,

    // Output
    thrust::complex<T>*       r_real,
    int*                      r_idx,
    thrust::complex<T>*       r_dual_value
)
{
    // 1) Parallel: elementwise power for real part
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < real_size) {
        r_real[i] = thrust::pow(a_real[i], power);  // a_real[i]^power
    }

    // 2) Single-thread logic for dual index & value
    __syncthreads();
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        // If no dual in 'a'
        if (a_idx < 0) {
            *r_idx        = -1;
            *r_dual_value = thrust::complex<T>(0, 0);
            return;
        }
        // Otherwise, we set the same index
        *r_idx = a_idx;

        // sum_{j=0 to real_size-1} [ power * (a_real[j]^(power-1)) * a_dual_value ]
        thrust::complex<T> sum_val(0, 0);
        for (int j = 0; j < real_size; ++j) {
            // a_real[j]^(power-1)
            // pow() from <thrust/complex.h> or <cmath> if real part
            thrust::complex<T> factor = thrust::pow(a_real[j], power - 1);
            sum_val += power * factor * (*a_dual_value);
        }
        *r_dual_value = sum_val;
    }
}


template <typename T>
__global__
void VectorDualPowSparseKernel(
    const thrust::complex<T>* a_real,
    int                       a_idx,
    const thrust::complex<T>* a_dual_value,
    T                         power,
    int                       real_size,

    // outputs
    thrust::complex<T>*       r_real,
    int*                      r_idx,
    thrust::complex<T>*       r_dual_value
)
{
    VectorDualPowSparse(
        a_real, a_idx, a_dual_value,
        power, real_size,
        r_real, r_idx, r_dual_value
    );
}


/**
 * VectorDualSqrtSparse
 *
 * Sparse equivalent of "VectorDualSqrt", which calls
 * VectorDualPowSparse(..., power=0.5, ...).
 *
 * - a_real        : length = real_size
 * - a_idx         : dual index in [-1..real_size)
 * - a_dual_value  : single complex for the dual part
 * - real_size     : length of a_real
 *
 * Outputs:
 * - r_real        : length = real_size
 * - r_idx, r_dual_value : single sparse dual
 */
template <typename T>
__device__
void VectorDualSqrtSparse(
    const thrust::complex<T>* a_real,
    int                       a_idx,
    const thrust::complex<T>* a_dual_value,
    int                       real_size,

    thrust::complex<T>*       r_real,
    int*                      r_idx,
    thrust::complex<T>*       r_dual_value
)
{
    // We simply call VectorDualPowSparse with exponent = 0.5
    T power = static_cast<T>(0.5);

    // Assuming you already have a definition similar to:
    //   VectorDualPowSparse(a_real, a_idx, a_dual_value,
    //                       power, real_size,
    //                       r_real, r_idx, r_dual_value);
    // We'll just call it:
    VectorDualPowSparse(
        a_real, a_idx, a_dual_value,
        power,
        real_size,
        r_real,
        r_idx,
        r_dual_value
    );
}


/**
 * VectorDualSqrtSparseKernel
 *
 * The __global__ kernel that calls VectorDualSqrtSparse.
 */
template <typename T>
__global__
void VectorDualSqrtSparseKernel(
    const thrust::complex<T>* a_real,
    int                       a_idx,
    const thrust::complex<T>* a_dual_value,
    int                       real_size,

    // outputs
    thrust::complex<T>*       r_real,
    int*                      r_idx,
    thrust::complex<T>*       r_dual_value
)
{
    VectorDualSqrtSparse(
        a_real,
        a_idx,
        a_dual_value,
        real_size,
        r_real,
        r_idx,
        r_dual_value
    );
}


/**
 * VectorDualReduceSparse
 *
 * Sums the "sparse" dual vector a:
 *  - a_real: length real_size
 *  - a_idx, a_dual_value: single dual index/value
 *
 * Outputs:
 *  - *result_real: the sum of all a_real[i]
 *  - *result_dual: equals a_dual_value if a_idx != -1, else 0
 *
 * This parallels the "dense" reduce that yields one complex for the real sum
 * and one complex for the dual sum, except the dual sum is trivial (one entry).
 */
template <typename T>
__device__
void VectorDualReduceSparse(
    const thrust::complex<T>* a_real,
    int                       a_idx,
    const thrust::complex<T>* a_dual_value,

    int                       real_size,

    // outputs
    thrust::complex<T>*       result_real,
    thrust::complex<T>*       result_dual
)
{
    // 1) Parallel sum of the real array
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ thrust::complex<T> sdata[256];  // example for blockDim.x <= 256

    // Each thread accumulates its local portion
    thrust::complex<T> localSum(0,0);
    if (i < real_size) {
        localSum = a_real[i];
    }
    // Store in shared
    int tid = threadIdx.x;
    sdata[tid] = localSum;
    __syncthreads();

    // Intra-block reduction (naive version)
    int blockSize = blockDim.x;
    for (int stride = blockSize/2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    // The block's partial sum is in sdata[0]
    // We can do a single-block approach if real_size <= blockDim.x,
    // or if you want multi-block, you'd store partial sums and do a second pass, etc.
    // For simplicity, assume a single block covers real_size.
    if (tid == 0) {
        // The block sum
        *result_real = sdata[0];
    }

    // 2) The dual part is trivial: one thread sets it
    __syncthreads();
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        if (a_idx < 0) {
            *result_dual = thrust::complex<T>(0,0);
        } else {
            *result_dual = *a_dual_value;
        }
    }
}

template <typename T>
__global__
void VectorDualReduceSparseKernel(
    const thrust::complex<T>* a_real,
    int                       a_idx,
    const thrust::complex<T>* a_dual_value,
    int                       real_size,

    // outputs
    thrust::complex<T>*       result_real,
    thrust::complex<T>*       result_dual
)
{
    VectorDualReduceSparse(
        a_real,
        a_idx,
        a_dual_value,
        real_size,
        result_real,
        result_dual
    );
}


/**
 * VectorDualCosSparse
 *
 * Sparse analog of VectorDualCos:
 *   - r_real[i] = cos(a_real[i]) for i in [0..real_size)
 *   - if a_idx == -1 => no dual => r_idx=-1, r_dual_value=0
 *   - else => r_idx=a_idx,
 *             r_dual_value = sum_{i=0..real_size-1} [ -sin(a_real[i]) * a_dual_value ]
 */
template <typename T>
__device__
void VectorDualCosSparse(
    // Input sparse vector a
    const thrust::complex<T>* a_real,       // length real_size
    int                       a_idx,        // single dual index or -1
    const thrust::complex<T>* a_dual_value, // single dual value

    int real_size,

    // Output sparse vector r
    thrust::complex<T>* r_real,       // length real_size
    int*                r_idx,
    thrust::complex<T>* r_dual_value
)
{
    // 1) Parallel pass: real part => cos(a_real[i])
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < real_size) {
        r_real[i] = thrust::cos(a_real[i]);
    }

    // 2) Single-thread logic for the dual
    __syncthreads();
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        if (a_idx < 0) {
            // No dual
            *r_idx        = -1;
            *r_dual_value = thrust::complex<T>(0,0);
        } else {
            // Use the same index
            *r_idx = a_idx;

            // sum_{i=0..real_size-1} [ - sin(a_real[i]) * a_dual_value ]
            thrust::complex<T> sumVal(0,0);
            for (int j = 0; j < real_size; ++j) {
                sumVal += -thrust::sin(a_real[j]) * (*a_dual_value);
            }
            *r_dual_value = sumVal;
        }
    }
}


template <typename T>
__global__
void VectorDualCosSparseKernel(
    const thrust::complex<T>* a_real,
    int                       a_idx,
    const thrust::complex<T>* a_dual_value,
    int                       real_size,

    // outputs
    thrust::complex<T>*       r_real,
    int*                      r_idx,
    thrust::complex<T>*       r_dual_value
)
{
    VectorDualCosSparse(
        a_real, a_idx, a_dual_value,
        real_size,
        r_real, r_idx, r_dual_value
    );
}


/**
 * VectorDualSinSparse
 *
 * Sparse version of "sin" for a dual vector. 
 *
 * - r_real[i] = sin(a_real[i]) for i in [0..real_size)
 * - If a_idx < 0 => no dual => r_idx=-1, r_dual_value=0.
 * - Else => r_idx=a_idx, r_dual_value = sum_{i}[ cos(a_real[i]) * a_dual_value ].
 */
template <typename T>
__device__
void VectorDualSinSparse(
    // Input sparse vector a
    const thrust::complex<T>* a_real,       // length real_size
    int                       a_idx,        // single dual index or -1 if none
    const thrust::complex<T>* a_dual_value, // single dual value

    int real_size,

    // Output sparse vector r
    thrust::complex<T>* r_real,       // length real_size
    int*                r_idx,
    thrust::complex<T>* r_dual_value
)
{
    // 1) Parallel pass for the real part: sin(a_real[i])
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < real_size) {
        r_real[i] = thrust::sin(a_real[i]);
    }

    // 2) Single-thread logic for the dual index + value
    __syncthreads();
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        if (a_idx < 0) {
            // no dual
            *r_idx        = -1;
            *r_dual_value = thrust::complex<T>(0,0);
        } else {
            // same index
            *r_idx = a_idx;

            // sum_{i=0..real_size-1}[ cos(a_real[i]) * a_dual_value ]
            thrust::complex<T> sumVal(0,0);
            for (int j = 0; j < real_size; ++j) {
                sumVal += thrust::cos(a_real[j]) * (*a_dual_value);
            }
            *r_dual_value = sumVal;
        }
    }
}

template <typename T>
__global__
void VectorDualSinSparseKernel(
    const thrust::complex<T>* a_real,
    int                       a_idx,
    const thrust::complex<T>* a_dual_value,
    int                       real_size,

    // outputs
    thrust::complex<T>*       r_real,
    int*                      r_idx,
    thrust::complex<T>*       r_dual_value
)
{
    VectorDualSinSparse(
        a_real, a_idx, a_dual_value,
        real_size,
        r_real, r_idx, r_dual_value
    );


}

/**
 * VectorDualTanSparse
 *
 * Sparse version of "tan" for a dual vector:
 *
 *   - r_real[i] = tan(a_real[i]) for i in [0..real_size)
 *   - if a_idx < 0 => no dual => r_idx=-1, r_dual_value=0
 *   - else => r_idx=a_idx,
 *             r_dual_value = sum_{i=0..(real_size-1)} [ a_dual_value / cos^2(a_real[i]) ]
 */
template <typename T>
__device__
void VectorDualTanSparse(
    // Input sparse vector a
    const thrust::complex<T>* a_real,       // length = real_size
    int                       a_idx,        // single dual index or -1
    const thrust::complex<T>* a_dual_value, // single dual value

    int real_size,

    // Output sparse vector r
    thrust::complex<T>* r_real,       // length = real_size
    int*                r_idx,
    thrust::complex<T>* r_dual_value
)
{
    // 1) Parallel: r_real[i] = tan(a_real[i])
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < real_size) {
        r_real[i] = thrust::tan(a_real[i]);
    }

    // 2) Single-thread logic for the dual
    __syncthreads();
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        if (a_idx < 0) {
            // No dual
            *r_idx        = -1;
            *r_dual_value = thrust::complex<T>(0,0);
        } else {
            // Use the same index
            *r_idx = a_idx;

            // sum_{i=0..real_size-1}[ a_dual_value / cos^2(a_real[i]) ]
            thrust::complex<T> sumVal(0,0);
            for (int j = 0; j < real_size; ++j) {
                // 1/cos^2(x) => sec^2(x)
                thrust::complex<T> cos_val = thrust::cos(a_real[j]);
                thrust::complex<T> denom   = cos_val * cos_val; // cos^2(x)
                sumVal += (*a_dual_value) / denom;
            }
            *r_dual_value = sumVal;
        }
    }
}


template <typename T>
__global__
void VectorDualTanSparseKernel(
    const thrust::complex<T>* a_real,
    int                       a_idx,
    const thrust::complex<T>* a_dual_value,
    int                       real_size,

    // outputs
    thrust::complex<T>*       r_real,
    int*                      r_idx,
    thrust::complex<T>*       r_dual_value
)
{
    VectorDualTanSparse(
        a_real, a_idx, a_dual_value,
        real_size,
        r_real, r_idx, r_dual_value
    );
}



/**
 * A sparse hyperdual vector: 
 * - real_size_   => length of the real part
 * - dual_size_   => dimension for the dual indices
 * - hyper_size_  => dimension for the hyperdual indices
 *
 * The real part is a contiguous array of length real_size_.
 * Each of the dual and hyper parts is stored as a single (index, value) pair:
 *  - dual_idx_ in [-1..dual_size_)
 *  - hyper_idx_ in [-1..hyper_size_)
 * If dual_idx_ == -1 (or hyper_idx_ == -1), we consider that part "empty."
 */
template <typename T>
class VectorHyperDualSparse
{
public:
    // Dimensions
    int real_size_;    // length of the real part
    int dual_size_;    // dimension of the "dual" space
    int hyper_size_;   // dimension of the "hyperdual" space

    // Real part
    thrust::complex<T>* real_;  // pointer to array of length real_size_

    // Dual part (sparse):
    //  - dual_idx_ in [-1..(dual_size_-1)] or -1 if none
    //  - dual_value_ is a pointer to a single complex if dual_idx_ != -1
    int                 dual_idx_;
    thrust::complex<T>* dual_value_;

    // Hyper part (sparse):
    //  - hyper_idx_ in [-1..(hyper_size_-1)] or -1 if none
    //  - hyper_value_ is a pointer to a single complex if hyper_idx_ != -1
    int                 hyper_idx_;
    thrust::complex<T>* hyper_value_;

    // -------------- Constructors --------------

    // Default constructor
    __host__ __device__
    VectorHyperDualSparse()
        : real_size_(0),
          dual_size_(0),
          hyper_size_(0),
          real_(nullptr),
          dual_idx_(-1),
          dual_value_(nullptr),
          hyper_idx_(-1),
          hyper_value_(nullptr)
    { }

    // Full constructor
    __host__ __device__
    VectorHyperDualSparse(
        int real_size,
        int dual_size,
        int hyper_size,
        thrust::complex<T>* real_data,
        int dual_idx,
        thrust::complex<T>* dual_val,
        int hyper_idx,
        thrust::complex<T>* hyper_val
    )
        : real_size_(real_size),
          dual_size_(dual_size),
          hyper_size_(hyper_size),
          real_(real_data),
          dual_idx_(dual_idx),
          dual_value_(dual_val),
          hyper_idx_(hyper_idx),
          hyper_value_(hyper_val)
    { }
};


/**
 * VectorHyperDualIndexGetSparse
 *
 * "IndexGet" for a sparse hyper-dual vector.  
 * We extract the subrange [start, end) of the real part, 
 * and preserve the single dual/hyper entry if its index is in-range.
 *
 * Inputs:
 *  - a_real:        length real_size
 *  - a_idx:         single dual index, or -1 if none
 *  - a_dual_value:  pointer to single dual value
 *  - a_hyper_value: pointer to single hyperdual value
 *  - real_size
 *  - start, end
 *
 * Outputs:
 *  - out_real:        length = (end - start)
 *  - *out_idx
 *  - *out_dual_value
 *  - *out_hyper_value
 */
template <typename T>
__device__
void VectorHyperDualIndexGetSparse(
    // Input
    const thrust::complex<T>* a_real,
    int                       a_idx,
    const thrust::complex<T>* a_dual_value,
    const thrust::complex<T>* a_hyper_value,
    int                       real_size,
    int                       start,
    int                       end,

    // Output
    thrust::complex<T>*       out_real,       // length = (end - start)
    int*                      out_idx,
    thrust::complex<T>*       out_dual_value,
    thrust::complex<T>*       out_hyper_value
)
{
    // 1) Parallel copy of the real part in [start, end).
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int sub_len = end - start;
    if (i < sub_len) {
        out_real[i] = a_real[start + i];
    }

    // 2) Single-thread logic for the dual/hyper
    __syncthreads();
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        // If a_idx is out of [start, end), no dual or hyper
        if (a_idx < start || a_idx >= end) {
            *out_idx        = -1;
            *out_dual_value = thrust::complex<T>(0,0);
            *out_hyper_value= thrust::complex<T>(0,0);
        } else {
            // Shift the index by 'start'
            *out_idx = a_idx - start;

            // Copy the single dual and hyper values
            *out_dual_value  = *a_dual_value;
            *out_hyper_value = *a_hyper_value;
        }
    }
}

template <typename T>
__global__
void VectorHyperDualIndexGetSparseKernel(
    const thrust::complex<T>* a_real,
    int                       a_idx,
    const thrust::complex<T>* a_dual_value,
    const thrust::complex<T>* a_hyper_value,
    int                       real_size,
    int                       start,
    int                       end,

    thrust::complex<T>*       out_real,
    int*                      out_idx,
    thrust::complex<T>*       out_dual_value,
    thrust::complex<T>*       out_hyper_value
)
{
    VectorHyperDualIndexGetSparse(
        a_real, a_idx, a_dual_value, a_hyper_value,
        real_size, start, end,
        out_real, out_idx, out_dual_value, out_hyper_value
    );
}


/**
 * VectorHyperDualIndexPutSparse
 *
 * "IndexPut" for a sparse hyperdual vector.
 * We copy the real array from the input into [start_real..end_real) in the result.
 * If the input dual/hyper index is valid, we shift it by start_real in the result.
 *
 * Inputs:
 *   - in_real:    real array of length input_real_size
 *   - in_idx:     single dual index in [-1..input_real_size), or -1 if none
 *   - in_dual_val:  pointer to single dual value
 *   - in_hyper_val: pointer to single hyper value
 *   - input_real_size: size of the input real array
 *   - start_real, end_real: subrange in the "result" space
 *
 * Outputs:
 *   - out_real:        length >= (end_real)
 *   - out_idx:         single dual/hyper index (shifted or -1)
 *   - out_dual_val:    single dual value
 *   - out_hyper_val:   single hyper value
 */
template <typename T>
__device__
void VectorHyperDualIndexPutSparse(
    // Input sparse hyperdual
    const thrust::complex<T>* in_real,
    int                       in_idx,
    const thrust::complex<T>* in_dual_val,
    const thrust::complex<T>* in_hyper_val,
    int                       input_real_size,

    // Where to "put" in the result
    int start_real,
    int end_real,

    // Outputs (result) 
    thrust::complex<T>*       out_real,
    int*                      out_idx,
    thrust::complex<T>*       out_dual_val,
    thrust::complex<T>*       out_hyper_val
)
{
    // 1) Parallel copy of the real subrange [0..(end_real-start_real)) from
    //    the input's entire real (size = input_real_size).
    //    But we need to ensure input_real_size == (end_real - start_real).
    //    Typically you'd pass exactly the slice you want in 'in_real',
    //    or handle partial. We'll assume input_real_size == (end_real - start_real).
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int sub_len = end_real - start_real;
    // Safety check: if i >= sub_len, do nothing
    if (i < sub_len) {
        // Copy in_real[i] => out_real[start_real + i]
        out_real[start_real + i] = in_real[i];
    }

    // 2) Single-thread logic for the dual/hyper
    __syncthreads();
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        // If in_idx is invalid or -1 => no dual/hyper
        if (in_idx < 0 || in_idx >= input_real_size) {
            *out_idx        = -1;
            *out_dual_val   = thrust::complex<T>(0,0);
            *out_hyper_val  = thrust::complex<T>(0,0);
        } else {
            // Shift index by start_real
            int shifted = start_real + in_idx;
            *out_idx      = shifted;
            *out_dual_val = *in_dual_val;
            *out_hyper_val= *in_hyper_val;
        }
    }
}
template <typename T>
__global__
void VectorHyperDualIndexPutSparseKernel(
    // input
    const thrust::complex<T>* in_real,
    int                       in_idx,
    const thrust::complex<T>* in_dual_val,
    const thrust::complex<T>* in_hyper_val,
    int                       input_real_size,

    // subrange
    int start_real,
    int end_real,

    // outputs
    thrust::complex<T>*       out_real,
    int*                      out_idx,
    thrust::complex<T>*       out_dual_val,
    thrust::complex<T>*       out_hyper_val
)
{
    VectorHyperDualIndexPutSparse(
        in_real,
        in_idx,
        in_dual_val,
        in_hyper_val,
        input_real_size,
        start_real,
        end_real,
        out_real,
        out_idx,
        out_dual_val,
        out_hyper_val
    );
}





// Minimal container for CSR on device memory
struct SparseMatrixCSR
{
    int rows_; // Number of rows
    int cols_; // Number of columns
    int nnz_;  // Number of nonzero entries

    // Device pointers
    int*                 row_ptr_; // length = rows_+1
    int*                 col_idx_; // length = nnz_
    thrust::complex<double>* data_;  // length = nnz_
};

// A small utility to check return codes. Replace as needed for your error handling.
static void checkCusparseStatus(cusparseStatus_t status, const char* msg)
{
    if (status != CUSPARSE_STATUS_SUCCESS) {
        std::cerr << "cuSPARSE error at: " << msg << std::endl;
        throw std::runtime_error("cuSPARSE Error");
    }
}

/**
 * Add two double-complex CSR matrices A+B using cuSPARSE: 
 *   C = A + B
 *
 * On input, A and B are device-based CSR stored in thrust::complex<double>.
 * On output, returns a newly allocated CSR (on device) for C.
 *
 * The function uses:
 *   cusparseZcsrgeam2_bufferSizeExt
 *   cusparseZcsrgeam2Nnz
 *   cusparseZcsrgeam2
 *
 * This requires:
 *   - A.rows_ == B.rows_  (same #rows)
 *   - A.cols_ == B.cols_  (same #cols)
 *   - A.data_ is in device memory
 *   - B.data_ is also in device memory
 *
 * The returned SparseMatrixCSR has device pointers for row_ptr_, col_idx_, data_,
 * fully allocated and filled. Its nnz_ is set appropriately.
 */
SparseMatrixCSR AddMatricesCSR_cuSPARSE(cusparseHandle_t handle,
                                        const SparseMatrixCSR& A,
                                        const SparseMatrixCSR& B)
{
    // 1) Check dimensions
    if (A.rows_ != B.rows_ || A.cols_ != B.cols_) {
        throw std::runtime_error("Dimension mismatch in AddMatricesCSR_cuSPARSE");
    }
    int rows = A.rows_;
    int cols = A.cols_;

    // 2) Create cuSPARSE matrix descriptors for A, B, C
    cusparseMatDescr_t descrA = nullptr;
    cusparseMatDescr_t descrB = nullptr;
    cusparseMatDescr_t descrC = nullptr;

    checkCusparseStatus(cusparseCreateMatDescr(&descrA), "createMatDescr(A)");
    checkCusparseStatus(cusparseCreateMatDescr(&descrB), "createMatDescr(B)");
    checkCusparseStatus(cusparseCreateMatDescr(&descrC), "createMatDescr(C)");

    // Typically, we set them to general / 0-based
    checkCusparseStatus(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL),
                        "setMatType(A)");
    checkCusparseStatus(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO),
                        "setIndexBase(A)");

    checkCusparseStatus(cusparseSetMatType(descrB, CUSPARSE_MATRIX_TYPE_GENERAL),
                        "setMatType(B)");
    checkCusparseStatus(cusparseSetMatIndexBase(descrB, CUSPARSE_INDEX_BASE_ZERO),
                        "setIndexBase(B)");

    checkCusparseStatus(cusparseSetMatType(descrC, CUSPARSE_MATRIX_TYPE_GENERAL),
                        "setMatType(C)");
    checkCusparseStatus(cusparseSetMatIndexBase(descrC, CUSPARSE_INDEX_BASE_ZERO),
                        "setIndexBase(C)");

    // 3) Prepare alpha=1, beta=1 for C = 1*A + 1*B
    // For "double complex" we use cuDoubleComplex
    cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
    cuDoubleComplex beta  = make_cuDoubleComplex(1.0, 0.0);

    // We'll reinterpret thrust::complex<double>* as cuDoubleComplex*
    // (They have the same memory layout in practice.)
    const cuDoubleComplex* Avals = reinterpret_cast<const cuDoubleComplex*>(A.data_);
    const cuDoubleComplex* Bvals = reinterpret_cast<const cuDoubleComplex*>(B.data_);

    // 4) Query buffer size for the geam2 operation
    size_t bufferSize = 0;
    checkCusparseStatus(
        cusparseZcsrgeam2_bufferSizeExt(
            handle, rows, cols,
            &alpha,
            descrA, A.nnz_, Avals, A.row_ptr_, A.col_idx_,
            &beta,
            descrB, B.nnz_, Bvals, B.row_ptr_, B.col_idx_,
            descrC,
            nullptr, // C values (unknown yet)
            nullptr, // C row_ptr
            nullptr, // C col_idx
            &bufferSize
        ), 
        "csrgeam2_bufferSizeExt");

    void* dBuffer = nullptr;
    cudaMalloc(&dBuffer, bufferSize);

    // 5) Compute # of nonzeros in C (nnzC) and fill row pointer for C
    int* dCrowPtr = nullptr;
    cudaMalloc(&dCrowPtr, (rows+1)*sizeof(int));
    int nnzC = 0; // We will read it from geam2Nnz

    checkCusparseStatus(
        cusparseXcsrgeam2Nnz(
            handle, rows, cols,
            descrA, A.nnz_, A.row_ptr_, A.col_idx_,
            descrB, B.nnz_, B.row_ptr_, B.col_idx_,
            descrC, dCrowPtr, &nnzC,
            dBuffer
        ),
        "csrgeam2Nnz"
    );
    // now dCrowPtr[] is the row pointer for C, and nnzC is the total number of nonzeros

    // 6) Allocate col/val arrays for C
    int* dCcolIdx = nullptr;
    cuDoubleComplex* dCvals = nullptr; // store them as cuDoubleComplex
    cudaMalloc(&dCcolIdx, nnzC*sizeof(int));
    cudaMalloc(&dCvals,   nnzC*sizeof(cuDoubleComplex));

    // 7) Actually compute C = alpha*A + beta*B
    checkCusparseStatus(
        cusparseZcsrgeam2(
            handle, rows, cols,
            &alpha,
            descrA, A.nnz_, Avals, A.row_ptr_, A.col_idx_,
            &beta,
            descrB, B.nnz_, Bvals, B.row_ptr_, B.col_idx_,
            descrC, dCvals, dCrowPtr, dCcolIdx,
            dBuffer
        ),
        "csrgeam2"
    );

    // 8) Build the final output struct
    SparseMatrixCSR C;
    C.rows_ = rows;
    C.cols_ = cols;
    C.nnz_  = nnzC;
    C.row_ptr_ = dCrowPtr;
    C.col_idx_ = dCcolIdx;

    // We need to store them as thrust::complex<double>*,
    // so do a reinterpret_cast (the memory layout is the same).
    C.data_ = reinterpret_cast<thrust::complex<double>*>(dCvals);

    // 9) Cleanup
    cudaFree(dBuffer);
    cusparseDestroyMatDescr(descrA);
    cusparseDestroyMatDescr(descrB);
    cusparseDestroyMatDescr(descrC);

    // Return the new device-based CSR matrix C
    return C;
}

// a small helper to check cuSPARSE status
static void checkCusparse(cusparseStatus_t status, const char* msg)
{
    if (status != CUSPARSE_STATUS_SUCCESS) {
        std::cerr << "cuSPARSE error at " << msg << std::endl;
        throw std::runtime_error("cuSPARSE Error");
    }
}

// create a descriptor in 0-based indexing
cusparseMatDescr_t createCsrDescr()
{
    cusparseMatDescr_t descr;
    cusparseCreateMatDescr(&descr);
    cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    return descr;
}

__device__
void CSRMatrixReduceBlock(
    const thrust::complex<double>* data,
    int nnz,
    thrust::complex<double>* blockSum)
{
    // blockSum is a single-element array in global memory 
    // for storing the sum from this block.
    extern __shared__ thrust::complex<double> shmem[]; 
    int tid = threadIdx.x;
    int globalIdx = blockDim.x * blockIdx.x + threadIdx.x;

    // load data into shared
    thrust::complex<double> val(0,0);
    if (globalIdx < nnz) {
        val = data[globalIdx];
    }
    shmem[tid] = val;
    __syncthreads();

    // do an in-block reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shmem[tid] += shmem[tid + s];
        }
        __syncthreads();
    }

    // write the result of this block
    if (tid == 0) {
        blockSum[blockIdx.x] = shmem[0];
    }
}

__global__
void CSRMatrixReduceBlockKernel(
    const thrust::complex<double>* data,
    int nnz,
    thrust::complex<double>* blockSums)
{
    CSRMatrixReduceBlock(data, nnz, blockSums);
}


__global__
void CSRMatrixPartialReduceKernel(
    const thrust::complex<double>* data,
    int nnz,
    thrust::complex<double>* partialSums)
{
    extern __shared__ thrust::complex<double> shmem[]; 
    int tid = threadIdx.x;
    int blockSize = blockDim.x;
    int blockStart = blockIdx.x * blockSize*2; // we'll do a 2-reads approach
    int idx = blockStart + tid;

    thrust::complex<double> val(0,0);
    // read 1
    if (idx < nnz) {
        val += data[idx];
    }
    // read 2
    int idx2 = idx + blockSize;
    if (idx2 < nnz) {
        val += data[idx2];
    }

    // store in shared
    shmem[tid] = val;
    __syncthreads();

    // do an in-block reduce
    for (int s = blockSize/2; s>0; s >>=1) {
        if (tid < s) {
            shmem[tid] += shmem[tid+s];
        }
        __syncthreads();
    }

    // write out
    if (tid == 0) {
        partialSums[blockIdx.x] = shmem[0];
    }
}
__global__
void CSRMatrixFinalReduceKernel(
    const thrust::complex<double>* partialSums,
    int n, // number of partial sums
    thrust::complex<double>* out)
{
    extern __shared__ thrust::complex<double> shmem[];
    int tid = threadIdx.x;
    int idx = tid;
    thrust::complex<double> val(0,0);
    if (idx < n) {
        val = partialSums[idx];
    }
    shmem[tid] = val;
    __syncthreads();

    // in-block reduce
    for (int s = blockDim.x/2; s>0; s>>=1) {
        if (tid < s) {
            shmem[tid] += shmem[tid + s];
        }
        __syncthreads();
    }

    if (tid==0) {
        out[0] = shmem[0];
    }
}

thrust::complex<double> CSRMatrixReduceAll(const SparseMatrixCSR& A)
{
    int nnz = A.nnz_;
    if (nnz <= 0) {
        // empty => sum is 0
        return thrust::complex<double>(0,0);
    }

    // 1) pick a block size, e.g. 256
    int blockSize = 256;
    // We'll do 2 reads per thread => each block handles 512 elements
    int gridSize = (nnz + (blockSize*2) - 1) / (blockSize*2);

    // allocate partial sums for kernel1
    thrust::complex<double>* dPartialSums = nullptr;
    cudaMalloc(&dPartialSums, gridSize*sizeof(thrust::complex<double>));

    // run kernel1
    size_t shmemSize = blockSize*sizeof(thrust::complex<double>);
    CSRMatrixPartialReduceKernel<<<gridSize, blockSize, shmemSize>>>(
        A.data_, nnz, dPartialSums
    );
    cudaDeviceSynchronize();

    // 2) Now reduce partialSums with a final kernel
    // blockDim = next pow2 >= gridSize, or we can just do the same blockSize. 
    // but let's keep it simple: use blockSize = 256 again
    int blockSize2 = 256;
    // we need 1 block to sum all partial sums, but we might not be able to 
    // if gridSize > 256. Instead we do a second pass. 
    // Let's do the same approach:
    int gridSize2 = (gridSize + (blockSize2*2) -1)/(blockSize2*2);

    thrust::complex<double>* dPartialSums2 = nullptr;
    cudaMalloc(&dPartialSums2, gridSize2*sizeof(thrust::complex<double>));

    CSRMatrixPartialReduceKernel<<<gridSize2, blockSize2, shmemSize>>>(
        dPartialSums,
        gridSize,
        dPartialSums2
    );
    cudaDeviceSynchronize();

    // We keep going until we get 1 partial sum left, but let's do a 
    // simpler approach: once we get small enough, copy to host, do CPU sum. 
    // For demonstration, let's do that now if (gridSize2 <= blockSize2).
    int finalSize = gridSize2;
    std::vector<thrust::complex<double>> hPartial(finalSize);
    cudaMemcpy(hPartial.data(), dPartialSums2,
               finalSize*sizeof(thrust::complex<double>),
               cudaMemcpyDeviceToHost);

    thrust::complex<double> finalSum(0,0);
    for (int i=0; i<finalSize; i++){
        finalSum += hPartial[i];
    }

    // cleanup
    cudaFree(dPartialSums);
    cudaFree(dPartialSums2);

    return finalSum;
}

__device__
void CSRMatrixSumDim(
    // The CSR matrix
    const int* row_ptr,
    const int* col_idx,
    const thrust::complex<double>* data,
    int rows,
    int cols,
    int nnz,

    // Which dimension to sum
    int dim, // 0 => sum over rows => result has length=cols
             // 1 => sum over columns => result has length=rows

    // Output: device array with length = (dim==0 ? cols : rows)
    thrust::complex<double>* result
)
{
    // We'll launch "1 thread per row."
    // So the row index is (blockDim.x * blockIdx.x + threadIdx.x).
    int r = blockDim.x * blockIdx.x + threadIdx.x;
    if (r >= rows) return;

    // Loop over the nonzero entries in row r
    int start = row_ptr[r];
    int end   = row_ptr[r+1];
    for (int idx = start; idx < end; idx++) {
        int c = col_idx[idx];
        thrust::complex<double> val = data[idx];
        if (dim == 0) {
            // Summing down rows => produce a column-sum => result[c] += val
            atomicAddComplex(&result[c], val);
        } else {
            // Summing across columns => produce a row-sum => result[r] += val
            atomicAddComplex(&result[r], val);
        }
    }
}

__global__
void CSRMatrixSumDimKernel(
    const int* row_ptr,
    const int* col_idx,
    const thrust::complex<double>* data,
    int rows,
    int cols,
    int nnz,
    int dim,
    thrust::complex<double>* result)
{
    CSRMatrixSumDim(row_ptr, col_idx, data,
                    rows, cols, nnz,
                    dim,
                    result);
}

void CSRMatrixSumAlongDim(
    const SparseMatrixCSR& A,
    int dim, // 0 => sum over rows => per-col result
             // 1 => sum over cols => per-row result
    thrust::complex<double>* dResult // device pointer to store the sum
)
{
    // Check dimension
    if (dim < 0 || dim > 1) {
        throw std::runtime_error("dim must be 0 or 1");
    }

    // The length of result depends on dim
    int outLen = (dim == 0) ? A.cols_ : A.rows_;

    // We typically need to zero-initialize dResult[outLen].
    cudaMemset(dResult, 0, outLen*sizeof(thrust::complex<double>));

    // Launch
    int blockSize = 256;
    int gridSize  = (A.rows_ + blockSize - 1)/blockSize;

    CSRMatrixSumDimKernel<<<gridSize, blockSize>>>(
        A.row_ptr_, A.col_idx_, A.data_,
        A.rows_, A.cols_, A.nnz_,
        dim,
        dResult
    );
    cudaDeviceSynchronize();
}

}  // namespace janus

#endif // _CU_DUAL_TENSOR_HPP