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
#include <cmath> // for fabs


namespace janus {

    class VectorBool {
    private:

    public:
        int size_;       // Vector length
        bool* data_;     // Pointer to device memory


        __device__ VectorBool() : size_(0), data_(nullptr) {}

        // Constructor: Initializes the wrapper with device memory
        __device__ VectorBool(bool* data, int size)
            : size_(size), data_(data) {}

        // Destructor: No action needed since memory is managed externally
        __device__ ~VectorBool() = default;

        // Disable copy constructor to avoid unintended copying
        __device__ VectorBool(const VectorBool&) = delete;

        __device__ VectorBool(VectorBool&& other) noexcept 
            : size_(other.size_), data_(other.data_) {}

        __device__ VectorBool& operator=(VectorBool&& other) noexcept {
            if (this != &other) {
                size_ = other.size_;
                data_ = other.data_;
            }
            return *this;
        }


    };
    
    // Set values in range
    __device__ void boolIndexPut(bool* input, int start, int end, const bool* subvector) {
          int idx = threadIdx.x + blockIdx.x * blockDim.x;
          int range = end - start;

          if (idx < range) {
            input[start + idx] = subvector[idx];
          }
    }


    //Wrapper functions
    __device__ void boolIndexGet(bool* input, int start, int end, bool* result) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= start && idx < end) {
                result[idx - start] = input[idx];
        }    
    }

    __global__ void boolIndexGetKernel(bool* input, int start, int end, bool* result) {
        boolIndexGet(input, start, end, result);
    }

    __global__ void boolIndexPutKernel(bool* input, int start, int end, bool* subvector) {
        boolIndexPut(input, start, end, subvector);
    }





    template <typename T>
    class Vector {
    public:
        int size_;                    // Vector length
        thrust::complex<T>* data_;  // Data pointer
    };
    
    template <typename T>
    __device__ void VectorElementwiseAdd(thrust::complex<T>* a, 
                                         thrust::complex<T>* b, 
                                         int size, 
                                         thrust::complex<T>* result)  {
        // Calculate global thread index
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        // Boundary check to prevent out-of-bounds access
        if (idx >= size) return;

        // Perform elementwise addition
        result[idx] = a[idx] + b[idx];

    }
    
    template <typename T>
    __global__ void VectorElementwiseAddKernel(thrust::complex<T>* a, 
                                               thrust::complex<T>* b,
                                               int size, 
                                               thrust::complex<T>* result) {
        VectorElementwiseAdd(a, b, size, result);
    }


    /**
     * @brief Computes a tensor result based on the signs of two input tensors.
     *
     * The function processes two input tensors, `a` and `b`, and modifies `a` based
     * on the sign of its real part (`a_sign`) and the sign of the real part of `b` (`b_sign`).
     * Small values are treated as positive to ensure compatibility with MATLAB.
     *
     * @param a A tensor representing the input tensor `a`.
     * @param b A tensor representing the input tensor `b`.
     * @return A tensor modified based on the conditional logic defined by the signs of `a` and `b`.
     *
     * @details 
     * The function follows these steps:
     * - Computes the signs of the real parts of `a` and `b` using `custom_sign`.
     * - Handles small numbers by assuming their sign is positive.
     * - Applies the following conditional logic to compute the result tensor:
     *   \f[
     *   \text{result} = 
     *   \begin{cases} 
     *   a, & \text{if } b_{\text{sign}} \geq 0 \text{ and } a_{\text{sign}} \geq 0 \\
     *   -a, & \text{if } b_{\text{sign}} \geq 0 \text{ and } a_{\text{sign}} < 0 \\
     *   -a, & \text{if } b_{\text{sign}} < 0 \text{ and } a_{\text{sign}} \geq 0 \\
     *   a, & \text{if } b_{\text{sign}} < 0 \text{ and } a_{\text{sign}} < 0
     *   \end{cases}
     *   \f]
     * - Returns the computed tensor `result`.
     *
     * @note Debug information for `a_sign` and `b_sign` is printed to `std::cerr`.
     * @note The function ensures compatibility with MATLAB-like behavior for very small numbers.
     */
    template <typename T>
    __device__ void VectorSigncond(thrust::complex<T>* a, 
                                   thrust::complex<T>* b,
                                   int size,  
                                   thrust::complex<T>* result, double tol = 1.0e-6) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;

        // Ensure we are within bounds
        if (idx >= size) return;
        auto aidx = a[idx];
        auto bidx = b[idx];
        // Retrieve the real parts of `a` and `b`
        T a_real = aidx.real();
        T b_real = bidx.real();

        // Compute the sign of the real parts with tolerance
        int a_sign = (fabs(static_cast<double>(a_real)) >= tol) ? (a_real >= 0 ? 1 : -1) : 1;
        int b_sign = (fabs(static_cast<double>(b_real)) >= tol) ? (b_real >= 0 ? 1 : -1) : 1;

        // Apply the conditional logic for result computation
        if (b_sign >= 0) {
            result[idx] = (a_sign >= 0) ? aidx : -aidx;
        } else {
            result[idx] = (a_sign >= 0) ? -aidx : aidx;
        }
    }

    template <typename T>
    __global__ void VectorSigncondKernel(thrust::complex<T>* a, 
                                         thrust::complex<T>* b, 
                                         int size, 
                                         thrust::complex<T>* result, double tol) {
        VectorSigncond(a, b, size, result, tol);
    }



    /**
     * Add an square() method that squares the complex numbers in the vector. 
     */
    template <typename T>
    __device__ void VectorSquare(const thrust::complex<T>* input, 
                                 int size, 
                                 thrust::complex<T>* result) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;

        // Ensure the thread is within bounds
        if (idx >= size) return;

        // Compute the square of the complex number
        result[idx] = input[idx] * input[idx];
    }

    //Create a global kernel for the square method
    template <typename T>
    __global__ void VectorSquareKernel(const thrust::complex<T>* input, 
                                       int size, 
                                       thrust::complex<T>* result) {
        VectorSquare(input, size, result);
    }

    /**
     * Multiply each element in the vector by a scalar.
     */
    template <typename T>
    __device__ void VectorScalarMultiply(const thrust::complex<T>*input, 
                                         T scalar,
                                         int size, 
                                         thrust::complex<T>*result) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;

        // Ensure the thread is within bounds
        if (idx >= size) return;

        // Compute the product of the complex number and the scalar
        result[idx] = input[idx] * scalar;
    }

    /**
     * Create a global kernel for the scalar multiply method
     */
    template <typename T>
    __global__ void VectorScalarMultiplyKernel(const thrust::complex<T>* input, 
                                               T scalar, 
                                               int size, 
                                               thrust::complex<T>* result) {
        VectorScalarMultiply(input, scalar, size, result);
    }

    /**
     * Take the reciprocal of each element in the vector.
     */
    template <typename T>
    __device__ void VectorReciprocal(const thrust::complex<T>*input, 
                                     int size,
                                     thrust::complex<T>* result) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;

        // Ensure the thread is within bounds
        if (idx >= size) return;

        // Compute the reciprocal of the complex number
        result[idx] = 1.0 / input[idx];
    }

    /**
     * Create a global kernel for the reciprocal method
     */
    template <typename T>
    __global__ void VectorReciprocalKernel(const thrust::complex<T>* input, 
                                           int size, 
                                           thrust::complex<T>* result) {
        VectorReciprocal(input, size, result);
    }


    /**
     * Elementise multiplication of two vectors.
     */
    template <typename T>
    __device__ void VectorElementwiseMultiply(const thrust::complex<T>* a, 
                                              const thrust::complex<T>* b, 
                                              int size, 
                                              thrust::complex<T>* result) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;

        // Ensure the thread is within bounds
        if (idx >= size) return;

        // Perform elementwise multiplication
        result[idx] = a[idx] * b[idx];
    }

    /**
     * Create a global kernel for the elementwise multiplication method
     */
    template <typename T>
    __global__ void VectorElementwiseMultiplyKernel(const thrust::complex<T>* a, 
                                                    const thrust::complex<T>* b, 
                                                    int size, 
                                                    thrust::complex<T>* result) {
        VectorElementwiseMultiply(a, b, size, result);
    }


    /**
     * Elementwise sqrt
     */
    template <typename T>
    __device__ void VectorSqrt(const thrust::complex<T>*a, 
                               int size, 
                               thrust::complex<T>*result) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;

        // Ensure the thread is within bounds
        if (idx >= size) return;

        // Perform elementwise sqrt
        result[idx] = thrust::sqrt(a[idx]);
    }

    /**
     * Create a global kernel for the sqrt method
     */
    template <typename T>
    __global__ void VectorSqrtKernel(const thrust::complex<T>* a, 
                                     int size, 
                                     thrust::complex<T>* result) {
        VectorSqrt(a, size, result);
    }

    /**
     * Implement the pow method
     */
    template <typename T>
    __device__ void VectorPow(const thrust::complex<T>* a, 
                              T power, 
                              int size,
                              thrust::complex<T>* result) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;

        // Ensure the thread is within bounds
        if (idx >= size) return;

        // Perform elementwise power using CUDA's pow function
        result[idx] = pow(a[idx], power);
    }

    /**
     * Create a global kernel for the pow method
     */
    template <typename T>
    __global__ void VectorPowKernel(const thrust::complex<T>* a, 
                                    T power, 
                                    int size, 
                                    thrust::complex<T>* result) {
        VectorPow(a, power, size, result);
    }

    /**
     * Sum all the elements of the vector into a single complex number.
     */
    template <typename T>
    __device__ void VectorReduce(const thrust::complex<T>* a, 
                                 int size, 
                                 thrust::complex<T>* result) {
        extern __shared__ thrust::complex<T> shared_data[];

        int tid = threadIdx.x;
        int idx = threadIdx.x + blockIdx.x * blockDim.x;

        shared_data[tid] = (idx < size) ? a[idx] : thrust::complex<T>(0, 0);
        __syncthreads();

        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                shared_data[tid] += shared_data[tid + s];
            }
            __syncthreads();
        }

        if (tid == 0) {
            result[blockIdx.x] = shared_data[0];
        }
    }

    /**
     * Create a global kernel for the reduce method
     */
    template <typename T>
    __global__ void VectorReduceKernel(const thrust::complex<T>* a, 
                                       int size, 
                                       thrust::complex<T>* result) {
        VectorReduce(a, size, result);
    }

    /** Retrieve elements from start to end into a new Vector 
     * This is a device function
     */
    template <typename T>
    __device__ void VectorIndexGet(const thrust::complex<T>* a, 
                                   int start, 
                                   int end, 
                                   int size, 
                                   thrust::complex<T>* result) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;

        // Ensure the thread is within bounds
        if (idx >= size) return;

        // Copy the elements from start to end
        if (idx >= start && idx < end) {
            result[idx - start] = a[idx];
        }
    }

    //Create a global kernel for the index get method
    template <typename T>
    __global__ void VectorIndexGetKernel(const thrust::complex<T>* a, 
                                         int start, 
                                         int end, 
                                         int size, 
                                         thrust::complex<T>* result) {
        VectorIndexGet(a, start, end, size, result);
    }


    /**
     * IndexPut for Vector
     * Put the elements from start to end from input Vector into the result vector
     */
    template<typename T>
    __device__ void VectorIndexPut(const thrust::complex<T>* input, 
                                   int start, 
                                   int end, 
                                   int size, 
                                   thrust::complex<T>* result) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;

        // Ensure the thread is within bounds
        if (idx >= size) return;

        // Copy the elements from start to end
        if (idx >= start && idx < end) {
            result[idx] = input[idx - start];
        }
    }


    //Create a global kernel for the index put method
    template <typename T>
    __global__ void VectorIndexPutKernel(const thrust::complex<T>* input, 
                                         int start, 
                                         int end, 
                                         int size, 
                                         thrust::complex<T>* result) {
        VectorIndexPut(input, start, end, size, result);
    }
    


    /**
     * Dual tensor class
     */
    template <typename T>
    class VectorDual {
    public:
        int real_size_;                    // Vector length
        int dual_size_;               // Dual dimension
        thrust::complex<T>* real_;   // Real part
        thrust::complex<T>* dual_;   // Dual part
    };
     
    template <typename T>
    __device__ void VectorRealDualProduct(const VectorDual<T>& a, 
                                 const VectorDual<T>& b, 
                                 VectorDual<T>& result) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;

        // Ensure the thread is within bounds
        if (idx >= a.real_size_*a.dual_size_) return;

        // Perform outer multiplication
        int i = idx / a.dual_size_;
        if (i ==0) {  
            result.real_[i] = a.real_[i] * b.real_[i];
        }
        int j = idx % a.real_size_;
        result.dual_[j] = a.real_[i] * b.dual_[j] + b.real_[i] * a.dual_[j];
    }



    /**
     * IndexGet for VectorDual
     * Given a VectorDual and a range, return a new VectorDual with the elements in the range 
     */
    template <typename T>
    __device__ void VectorDualIndexGet(const VectorDual<T>& input,   
                                       int start, 
                                       int end,
                                       VectorDual<T>& result) {
        // Get the real part
        VectorIndexGet(input.real, start, end, input.real_size_, result.real);

        // Get the dual part
        VectorIndexGet(input.dual, start*input.dual_size_, end*input.dual_size_, 
                       (end-start-1)*input.dual_size_, result.dual_);
    }

    /**
     * IndexPut for VectorDual
     */
    template <typename T>
    __device__ void VectorDualIndexPut(const VectorDual<T>& input, 
                                       int start, 
                                       int end, 
                                       VectorDual<T>& result) {
        int real_size = input.real_size_;
        int dual_size = input.dual_size_;
        // Put the real part
        VectorIndexPut(input.real_, start, end, real_size, result.real_);

        // Put the dual part
        VectorIndexPut(input.dual_, start*dual_size, end*dual_size, real_size*dual_size, result.dual_);
    }

    /**
     * Elementwise addition of two VectorDual tensors
     */
    template <typename T>
    __device__ void VectorDualElementwiseAdd(const VectorDual<T>& a, 
                                             const VectorDual<T>& b, 
                                             VectorDual<T>& result) {
        // Perform elementwise addition of the real part
        VectorElementwiseAdd(a.real_, b.real_, a.real_size_, result.real_);

        // Perform elementwise addition of the dual part
        VectorElementwiseAdd(a.dual_, b.dual_, a.size_*a.dual_size, result.dual_);
    }

    /**
     * Elementwise multiplication of two VectorDual tensors
     */
    template <typename T>
    __device__ void VectorDualElementwiseMultiply(const VectorDual<T>& a, 
                                                  const VectorDual<T>& b,
                                                  VectorDual<T>& result) {
        // Perform elementwise multiplication of the real part
        VectorElementwiseMultiply(a.real_, b.real_, a.size_, result.real_);

        // Perform elementwise multiplication for the dual part
        VectorRealDualProduct(a.real_, b.dual_, a.real_size_, a.dual_size_, result.dual_);
        VectorRealDualProduct(b.real_, a.dual_, b.real_size_, b.dual_size_, result.dual_);
    }

    /**
     * Square each element in the VectorDual tensor
     */
    template <typename T>
    __device__ void VectorDualSquare(VectorDual<T>& input, 
                                     VectorDual<T>& result) {
        VectorDualElementwiseMultiply(input, input, result);
    }

    /**
     * Sqrt each element in the VectorDual tensor
     */
    template <typename T>
    __device__ void VectorDualSqrt(VectorDual<T>& input,
                                   VectorDual<T>& work, //Intermediate storage 
                                   VectorDual<T>& result) {
        // Perform elementwise sqrt for the real part
        VectorSqrt(input.real_, input.real_size_, result.real_);
        //The dual part is 0.5*input.real^(-0.5)*input.dual
        VectorPow(input.real_, -0.5, work.real_);
        VectorScalarMultiply(work.real_, 0.5, work.real_);
        VectorRealDualProduct(work.real_, input.dual_, result.dual_);
    }
} // namespace Janus
#endif // _CU_DUAL_TENSOR_HPP