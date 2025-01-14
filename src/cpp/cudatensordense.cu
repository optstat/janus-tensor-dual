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
    __device__ void VectorElementwiseAdd(const thrust::complex<T>* a, 
                                         const thrust::complex<T>* b, 
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
        int real_size_;               // Vector length
        int dual_size_;               // Dual dimension
        thrust::complex<T>* real_;    // Real part
        thrust::complex<T>* dual_;    // Dual part
    };
     
    template <typename T>
    __device__ void VectorRealDualProduct(const thrust::complex<T>* a_real, 
                                          const thrust::complex<T>* a_dual,
                                          const thrust::complex<T>* b_real,
                                          const thrust::complex<T>* b_dual,
                                          int real_size,
                                          int dual_size,
                                          thrust::complex<T>* result_real,
                                          thrust::complex<T>* result_dual) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;

        // Ensure the thread is within bounds
        if (idx >= real_size*dual_size) return;

        // Perform outer multiplication
        int i = idx / dual_size;
        thrust::complex<T> a_realc = a_real[i];
        thrust::complex<T> b_realc = b_real[i];
        if (i ==0 ) { 
            result_real[i] = a_realc * b_realc;
        }
        int j = idx % real_size;
        result_dual[j] = a_realc * b_dual[j] + b_realc * a_dual[j];
    }

    //Create a global kernel for the real dual product method
    template <typename T>
    __global__ void VectorRealDualProductKernel(const thrust::complex<T>* a_real, 
                                                const thrust::complex<T>* a_dual,
                                                const thrust::complex<T>* b_real,
                                                const thrust::complex<T>* b_dual,
                                                int real_size,
                                                int dual_size,
                                                thrust::complex<T>* result_real,
                                                thrust::complex<T>* result_dual) {
        VectorRealDualProduct(a_real, a_dual, b_real, b_dual, real_size, dual_size, result_real, result_dual);
    }



    /**
     * IndexGet for VectorDual
     * Given a VectorDual and a range, return a new VectorDual with the elements in the range 
     */
    template <typename T>
    __device__ void VectorDualIndexGet(const thrust::complex<T>* a_real,
                                       const thrust::complex<T>* a_dual,
                                       int real_size,
                                       int dual_size,
                                       int start, 
                                       int end, 
                                       int dual_start,
                                       int dual_end,
                                       thrust::complex<T>* result_real,
                                       thrust::complex<T>* result_dual) { 
        int  idx = threadIdx.x + blockIdx.x * blockDim.x;
        int real_size_dest = end-start;
        int dual_size_dest = dual_end-dual_start;

        int  sz = real_size_dest*dual_size_dest; //Size of the destination vector
        if ( idx >= sz) return;
        
        int real_dest_idx = idx /dual_size_dest;
        int dual_dest_idx = idx %dual_size_dest;
        //printf("idx: %d, real_dest_idx: %d, dual_dest_idx: %d\n", idx, real_dest_idx, dual_dest_idx);
        int real_source_cnt = start+real_dest_idx;
        int dual_source_cnt = dual_size*real_source_cnt + dual_dest_idx+dual_start;
        int real_dest_cnt = real_dest_idx;
        int dual_dest_cnt = real_dest_idx*dual_size_dest + dual_dest_idx;
        //printf("real_source_cnt: %d, dual_source_cnt: %d, real_dest_cnt: %d, dual_dest_cnt: %d\n", real_source_cnt, dual_source_cnt, real_dest_cnt, dual_dest_cnt);
        result_real[real_dest_cnt] = a_real[real_source_cnt];
        result_dual[dual_dest_cnt] = a_dual[dual_source_cnt];
    }

    //Create a global kernel for the index get method
    template <typename T>
    __global__ void VectorDualIndexGetKernel(const thrust::complex<T>* a_real,
                                             const thrust::complex<T>* a_dual,
                                             int real_size,
                                             int dual_size,
                                             int start, 
                                             int end, 
                                             int dual_start,
                                             int dual_end,
                                             thrust::complex<T>* result_real,
                                             thrust::complex<T>* result_dual) {
        VectorDualIndexGet(a_real, a_dual, real_size, dual_size, start, end, dual_start, dual_end, result_real, result_dual);
    }

    /**
     * IndexPut for VectorDual
     */
    template <typename T>
    __device__ void VectorDualIndexPut(const thrust::complex<T>* input_real,
                                       const thrust::complex<T>* input_dual,  
                                       int start, 
                                       int end,
                                       int dual_start,
                                       int dual_end, 
                                       thrust::complex<T>* result_real,
                                       thrust::complex<T>* result_dual,
                                       int real_size,
                                       int dual_size) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        int sz = (end-start)*(dual_end-dual_start);
        //printf("sz: %d\n", sz);
        if (idx >= sz) return;
        int real_src_idx = idx / (dual_end-dual_start);
        int dual_src_idx = idx % (start-end);
        //printf("idx: %d, real_src_idx: %d, dual_src_idx: %d\n", idx, real_src_idx, dual_src_idx);
        int real_src_cnt = real_src_idx;
        int dual_src_cnt = real_src_idx*(dual_end-dual_start)+ dual_src_idx;
        //printf("real_src_cnt: %d, dual_src_cnt: %d\n", real_src_cnt, dual_src_cnt);
        int real_dest_cnt = real_src_idx + start;
        int dual_dest_cnt = real_dest_cnt*dual_size + dual_src_idx + dual_start;
        //printf("real_dest_cnt: %d, dual_dest_cnt: %d\n", real_dest_cnt, dual_dest_cnt);
        result_real[real_dest_cnt] = input_real[real_src_cnt];
        result_dual[dual_dest_cnt] = input_dual[dual_src_cnt];
    }

    //Create a global kernel for the index put method
    template <typename T>
    __global__ void VectorDualIndexPutKernel(const thrust::complex<T>* input_real,
                                             const thrust::complex<T>* input_dual,
                                             int start, 
                                             int end,
                                             int dual_start,
                                             int dual_end, 
                                             thrust::complex<T>* result_real,
                                             thrust::complex<T>* result_dual,
                                             int real_size,
                                             int dual_size) {
        VectorDualIndexPut(input_real, input_dual,  start, end, dual_start, dual_end, result_real, result_dual, real_size, dual_size);
    }

    /**
     * Elementwise addition of two VectorDual tensors
     */
    template <typename T>
    __device__ void VectorDualElementwiseAdd(const thrust::complex<T>* a_real, 
                                             const thrust::complex<T>* a_dual,
                                             const thrust::complex<T>* b_real,
                                             const thrust::complex<T>* b_dual,
                                             int real_size,
                                             int dual_size,
                                             thrust::complex<T>* result_real,
                                             thrust::complex<T>* result_dual) {
        // Perform elementwise addition of the real part
        VectorElementwiseAdd(a_real, b_real, real_size, result_real);

        // Perform elementwise addition for the dual part
        VectorElementwiseAdd(a_dual, b_dual, dual_size, result_dual);
    }

    //Create a global kernel for the elementwise addition method
    template <typename T>
    __global__ void VectorDualElementwiseAddKernel(const thrust::complex<T>* a_real, 
                                                   const thrust::complex<T>* a_dual,
                                                   const thrust::complex<T>* b_real,
                                                   const thrust::complex<T>* b_dual,
                                                   int real_size,
                                                   int dual_size,
                                                   thrust::complex<T>* result_real,
                                                   thrust::complex<T>* result_dual) {
        VectorDualElementwiseAdd(a_real, a_dual, b_real, b_dual, real_size, dual_size, result_real, result_dual);
    }


    /**
     * Elementwise multiplication of two VectorDual tensors
     */
    template <typename T>
    __device__ void VectorDualElementwiseMultiply(const thrust::complex<T>* a_real, 
                                                  const thrust::complex<T>* a_dual,
                                                  const thrust::complex<T>* b_real,
                                                  const thrust::complex<T>* b_dual,
                                                  int real_size,
                                                  int dual_size,
                                                  thrust::complex<T>* result_real,
                                                  thrust::complex<T>* result_dual) {
        // Perform elementwise multiplication of the real part
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        // Perform elementwise multiplication for the dual part
        if ( idx >= real_size*dual_size) return;
        //Extract the real index
        int i = idx / dual_size;
        auto a_realc = a_real[i];
        auto b_realc = b_real[i];
        int j = idx%dual_size;
        //printf("i:%d, j: %d, idx: %d\n", i, j, idx);
        if (j == 0)
          result_real[i] = a_realc * b_realc;
        //Extract the dual index
        result_dual[idx] = a_realc * b_dual[idx] + b_realc * a_dual[idx];
    }

    //Create a global kernel for the elementwise multiplication method
    template <typename T>
    __global__ void VectorDualElementwiseMultiplyKernel(const thrust::complex<T>* a_real, 
                                                        const thrust::complex<T>* a_dual,
                                                        const thrust::complex<T>* b_real,
                                                        const thrust::complex<T>* b_dual,
                                                        int real_size,
                                                        int dual_size,
                                                        thrust::complex<T>* result_real,
                                                        thrust::complex<T>* result_dual) {
        VectorDualElementwiseMultiply(a_real, a_dual, b_real, b_dual, real_size, dual_size, result_real, result_dual);
    }

    /**
     * Square each element in the VectorDual tensor
     */
    template <typename T>
    __device__ void VectorDualSquare(thrust::complex<T>* a_real, 
                                     thrust::complex<T>* a_dual,
                                     int real_size,
                                     int dual_size,
                                     thrust::complex<T>* result_real,
                                     thrust::complex<T>* result_dual) {
        // Perform elementwise square for the real part
        VectorDualElementwiseMultiply(a_real, a_dual, a_real, a_dual, real_size, dual_size, result_real, result_dual);
    }

    //Create a global kernel for the square method
    template <typename T>
    __global__ void VectorDualSquareKernel(thrust::complex<T>* a_real, 
                                           thrust::complex<T>* a_dual,
                                           int real_size,
                                           int dual_size,
                                           thrust::complex<T>* result_real,
                                           thrust::complex<T>* result_dual) {
        VectorDualSquare(a_real, a_dual, real_size, dual_size, result_real, result_dual);
    }

    /**
     * Pow method for VectorDual
     */
    template <typename T>
    __device__ void VectorDualPow(const thrust::complex<T>* a_real, 
                                  const thrust::complex<T>* a_dual,
                                  T power,
                                  int real_size,
                                  int dual_size,
                                  thrust::complex<T>* result_real,
                                  thrust::complex<T>* result_dual) {
        // Perform elementwise power for the real part
        VectorPow(a_real, power, real_size, result_real);
        // Perform elementwise power for the dual part
        //The result here is power*(a_real^(power-1))*a_dual
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= real_size*dual_size) return;
        int real_idx = idx / dual_size;
        result_dual[idx] = power * pow(a_real[real_idx], power - 1) * a_dual[idx];
    }

    //Create a global kernel for the pow method
    template <typename T>
    __global__ void VectorDualPowKernel(const thrust::complex<T>* a_real, 
                                        const thrust::complex<T>* a_dual,
                                        T power,
                                        int real_size,
                                        int dual_size,
                                        thrust::complex<T>* result_real,
                                        thrust::complex<T>* result_dual) {
        VectorDualPow(a_real, a_dual, power, real_size, dual_size, result_real, result_dual);
    }

    /**
     * Sqrt each element in the VectorDual tensor
     */
    template <typename T>
    __device__ void VectorDualSqrt(const thrust::complex<T>* a_real, 
                                   const thrust::complex<T>* a_dual,
                                   int real_size,
                                   int dual_size,
                                   thrust::complex<T>* result_real,
                                   thrust::complex<T>* result_dual) {
        VectorDualPow(a_real, a_dual, static_cast<T>(0.5), real_size, dual_size, result_real, result_dual);
    }

    //Create a global kernel for the sqrt method
    template <typename T>
    __global__ void VectorDualSqrtKernel(const thrust::complex<T>* a_real, 
                                         const thrust::complex<T>* a_dual,
                                         int real_size,
                                         int dual_size,
                                         thrust::complex<T>* result_real,
                                         thrust::complex<T>* result_dual) {
        VectorDualSqrt(a_real, a_dual, real_size, dual_size, result_real, result_dual);
    }

    /**
     * Reduce method for VectorDual 
     */
     template <typename T>
     __device__ void VectorDualReduce(const thrust::complex<T>* a_real,
                                      const thrust::complex<T>* a_dual, 
                                      int real_size, 
                                      int dual_size,
                                      thrust::complex<T>* result_real,
                                      thrust::complex<T>* result_dual) {
        VectorReduce(a_real, real_size, result_real);
        VectorReduce(a_dual, real_size*dual_size, result_dual);                            
    }

    /**
     * Create a global kernel for the reduce method
     */
     template <typename T>
     __global__ void VectorDualReduceKernel(const thrust::complex<T>* a_real,
                                      const thrust::complex<T>* a_dual, 
                                      int real_size, 
                                      int dual_size,
                                      thrust::complex<T>* result_real,
                                      thrust::complex<T>* result_dual) {
        VectorDualReduce(a_real, a_dual, real_size, dual_size, result_real, result_dual);
    }





    /**
     * HyperDual tensor class
     */
    template <typename T>
    class VectorHyperDual {
    public:
        int real_size_;               // Vector length
        int dual_size_;               // Dual dimension
        thrust::complex<T>* real_;    // Real part
        thrust::complex<T>* dual_;    // Dual part
        thrust::complex<T>* hyper_;   // Hyperdual part
    };

    /**
     * IndexGet for HyperDual Vector
     */
    template <typename T>
    __device__ void VectorHyperDualIndexGet(const thrust::complex<T>* a_real,
                                            const thrust::complex<T>* a_dual,
                                            const thrust::complex<T>* a_hyper,
                                            int start_real,
                                            int end_real,
                                            int start_dual,
                                            int end_dual,
                                            int start_hyper,
                                            int end_hyper, 
                                            thrust::complex<T>* result_real,
                                            thrust::complex<T>* result_dual,
                                            thrust::complex<T>* result_hyper) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        int size = (end_hyper -  start_hyper-1)*(end_real - start_real-1)*(end_dual - start_dual-1);
        // Ensure the thread is within bounds
        if (idx >= size) return;
        //Extract the real index
        int real_idx = idx / ((end_dual - start_dual-1)*(end_hyper - start_hyper-1));
        //Extract the dual index
        int dual_idx = (idx % ((end_dual - start_dual-1)*(end_hyper - start_hyper-1))) / (end_hyper - start_hyper-1);
        //Extract the hyper index
        int hyper_idx = idx % (end_hyper - start_hyper-1);
        result_real[real_idx] = a_real[real_idx + start_real];
        result_dual[dual_idx] = a_dual[dual_idx + start_dual];
        result_hyper[hyper_idx] = a_hyper[hyper_idx + start_hyper];

    }

    //Create a global kernel for the index get method
    template <typename T>
    __global__ void VectorHyperDualIndexGetKernel(const thrust::complex<T>* a_real,
                                                  const thrust::complex<T>* a_dual,
                                                  const thrust::complex<T>* a_hyper,
                                                  int start_real,
                                                  int end_real,
                                                  int start_dual,
                                                  int end_dual,
                                                  int start_hyper,
                                                  int end_hyper, 
                                                  thrust::complex<T>* result_real,
                                                  thrust::complex<T>* result_dual,
                                                  thrust::complex<T>* result_hyper) {
        VectorHyperDualIndexGet(a_real, a_dual, a_hyper, start_real, end_real, start_dual, end_dual, start_hyper, end_hyper, result_real, result_dual, result_hyper);
    }





} // namespace Janus
#endif // _CU_DUAL_TENSOR_HPP