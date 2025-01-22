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




}  // namespace janus
#endif // _CU_DUAL_TENSOR_HPP