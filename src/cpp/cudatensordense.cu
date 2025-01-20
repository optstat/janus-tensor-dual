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

    /**
     * Given index into a hyperdual tensor for a vector, 
     * derive the indexes into the real, dual and hyperdual parts
     */
    __device__ void get_hyperdual_vector_offsets(const int i, const int k,  const int l, 
                                          const int rows, const int dual, 
                                          int *off_i, int *off_k, int *off_l) {
        *off_i = i;
        *off_k = i*dual+k;
        *off_l = i*dual*dual+k*dual+l;
    }

    //create a global wrapper
    __global__ void get_hyperdual_vector_offsets_kernel(const int i, const int k,  const int l, 
                                                 const int rows, const int dual, 
                                                 int *off_i, int *off_k, int *off_l) {
        get_hyperdual_vector_offsets(i, k, l, rows, dual, off_i, off_k, off_l);
    }

    __device__ void get_hyperdual_indexes(const int offi, const int offj, const int offk, const int offl,
                                          const int rows, const int cols, const int dual, 
                                          int &i, int &j, int &k, int &l) {
        i = offi;
        j = (offj / rows) % cols;
        k = (offk / (rows*cols)) % dual;
        l = offl % dual;
    }

    //create a global wrapper
    __global__ void get_hyperdual_indexes_kernel(const int offi, const int offj, const int offk, const int offl,
                                                 const int rows, const int cols, const int dual, 
                                                 int &i, int &j, int &k, int &l) {
        get_hyperdual_indexes(offi, offj, offk, offl, rows, cols, dual, i, j, k, l);
    }
    

    /**
     * Given a single index into the hyperdual part (hessian)
     * Derive the indexes into the real, dual and hyperdual parts
     * that the index corresponds to
     */
    __device__ void get_tensor_indxs_from_hyperdual_vector_offset(const int off, //Offset
                                                                  const int rows, //Number of rows
                                                                  const int dual, //Dual dimension
                                                                  int &i,  //Number of rows
                                                                  int &k,  //Dual index (rows) (also row into the hyperdual part)
                                                                  int &l) { //Hyperdual index cols
        //off = i*dual*dual + k*dual + l
        i = off / (dual*dual);
        k = off/dual % dual;
        l = off % dual;
    }

    //create a global wrapper
    __global__ void get_tensor_indxs_from_hyperdual_vector_offset_kernel(const int off, //Offset
                                                                         const int rows, //Number of rows
                                                                         const int dual, //Dual dimension
                                                                         int &i,  //Number of rows
                                                                         int &k,  //Dual index (rows) (also row into the hyperdual part)
                                                                         int &l) { //Hyperdual index cols
        get_tensor_indxs_from_hyperdual_vector_offset(off, rows, dual, i, k, l);
    }


    __device__ void get_hyperdual_vector_offsets_from_tensor_indxs(const int i, const int k, const int l, 
                                                            const int rows, const int dual, 
                                                            int *off_i, int* off_k, int* off_l) {
        *off_i= i;                           //real
        *off_k= i*dual + k;                  //dual (also hyper dual row)
        *off_l= i*dual*dual + l;                  //hyperdual index ( also hyperdual col)
    }

    //create a global wrapper
    __global__ void get_hyperdual_vector_offsets_from_tensor_indxs_kernel(const int i, const int k, const int l, 
                                                             const int rows, const int dual, 
                                                             int* off_i, int *off_l, int *off_k) {
        return get_hyperdual_vector_offsets_from_tensor_indxs(i, k, l, rows, dual, off_i, off_l, off_k);
    }



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
                                       thrust::complex<T>* result_real,
                                       thrust::complex<T>* result_dual) { 
        int  idx = threadIdx.x + blockIdx.x * blockDim.x;
        int sz = (end-start)*dual_size;

        if ( idx >= sz) return;
        //off = i*dual_size+j;
        int real_dest_idx = idx/dual_size;
        int dual_dest_idx = idx%dual_size;
        int real_dest_off = real_dest_idx;
        int dual_dest_off = real_dest_off*dual_size+dual_dest_idx;
        //printf("idx: %d, real_dest_idx: %d, dual_dest_idx: %d\n", idx, real_dest_idx, dual_dest_idx);
        int real_source_off = start+real_dest_idx;
        int dual_source_off = real_source_off*dual_size+dual_dest_idx;
        if ( dual_dest_idx == 0) {
          result_real[real_dest_off] = a_real[real_source_off];
        }
        result_dual[dual_dest_off] = a_dual[dual_source_off];
        
    }

    //Create a global kernel for the index get method
    template <typename T>
    __global__ void VectorDualIndexGetKernel(const thrust::complex<T>* a_real,
                                             const thrust::complex<T>* a_dual,
                                             int real_size,
                                             int dual_size,
                                             int start, 
                                             int end, 
                                             thrust::complex<T>* result_real,
                                             thrust::complex<T>* result_dual) {
        VectorDualIndexGet(a_real, a_dual, real_size, dual_size, start, end, result_real, result_dual);
    }

    /**
     * IndexPut for VectorDual
     */
    template <typename T>
    __device__ void VectorDualIndexPut(const thrust::complex<T>* input_real,
                                       const thrust::complex<T>* input_dual,  
                                       int start, 
                                       int end,
                                       thrust::complex<T>* result_real,
                                       thrust::complex<T>* result_dual,
                                       int real_size,
                                       int dual_size) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        int sz = (end-start)*dual_size;
        //printf("sz: %d\n", sz);
        if (idx >= sz) return;
        //idx = real_src_idx*dual_size+dual_src_idx
        int real_src_idx = idx / dual_size;
        int dual_src_idx = idx % dual_size;
        //printf("idx: %d, real_src_idx: %d, dual_src_idx: %d\n", idx, real_src_idx, dual_src_idx);
        int real_src_off = real_src_idx;
        int dual_src_off = real_src_idx*dual_size+dual_src_idx;
        //printf("real_src_cnt: %d, dual_src_cnt: %d\n", real_src_cnt, dual_src_cnt);
        int real_dest_off = real_src_idx + start;
        int dual_dest_off = real_dest_off*dual_size+dual_src_idx;
        //printf("real_dest_cnt: %d, dual_dest_cnt: %d\n", real_dest_cnt, dual_dest_cnt);
        if ( dual_src_idx == 0) {
          result_real[real_dest_off] = input_real[real_src_off];
        }
        result_dual[dual_dest_off] = input_dual[dual_src_off];
    }

    //Create a global kernel for the index put method
    template <typename T>
    __global__ void VectorDualIndexPutKernel(const thrust::complex<T>* input_real,
                                             const thrust::complex<T>* input_dual,
                                             int start, 
                                             int end,
                                             thrust::complex<T>* result_real,
                                             thrust::complex<T>* result_dual,
                                             int real_size,
                                             int dual_size) {
        VectorDualIndexPut(input_real, input_dual,  start, end, result_real, result_dual, real_size, dual_size);
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
    
    template <typename T>
    __device__ void VectorDualCos(const thrust::complex<T>* a_real, 
                                  const thrust::complex<T>* a_dual,
                                  int real_size,
                                  int dual_size,
                                  thrust::complex<T>* result_real,
                                  thrust::complex<T>* result_dual) {
        // Perform elementwise cos for the real part
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= real_size*dual_size) return;
        int real_idx = idx / dual_size;
        result_real[real_idx] = cos(a_real[real_idx]);
        result_dual[idx] = -sin(a_real[real_idx]) * a_dual[idx];
    }

    //Create a global kernel for the cos method
    template <typename T>
    __global__ void VectorDualCosKernel(const thrust::complex<T>* a_real, 
                                        const thrust::complex<T>* a_dual,
                                        int real_size,
                                        int dual_size,
                                        thrust::complex<T>* result_real,
                                        thrust::complex<T>* result_dual) {
        VectorDualCos(a_real, a_dual, real_size, dual_size, result_real, result_dual);
    }


    template <typename T>
    __device__ void VectorDualSin(const thrust::complex<T>* a_real, 
                                  const thrust::complex<T>* a_dual,
                                  int real_size,
                                  int dual_size,
                                  thrust::complex<T>* result_real,
                                  thrust::complex<T>* result_dual) {
        // Perform elementwise sin for the real part
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= real_size*dual_size) return;
        int real_idx = idx / dual_size;
        result_real[real_idx] = sin(a_real[real_idx]);
        result_dual[idx] = cos(a_real[real_idx]) * a_dual[idx];
    }

    //Create a global kernel for the sin method
    template <typename T>
    __global__ void VectorDualSinKernel(const thrust::complex<T>* a_real, 
                                        const thrust::complex<T>* a_dual,
                                        int real_size,
                                        int dual_size,
                                        thrust::complex<T>* result_real,
                                        thrust::complex<T>* result_dual) {
        VectorDualSin(a_real, a_dual, real_size, dual_size, result_real, result_dual);
    }

    template <typename T>
    __device__ void VectorDualTan(const thrust::complex<T>* a_real, 
                                  const thrust::complex<T>* a_dual,
                                  int real_size,
                                  int dual_size,
                                  thrust::complex<T>* result_real,
                                  thrust::complex<T>* result_dual) {
        // Perform elementwise tan for the real part
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= real_size*dual_size) return;
        int real_idx = idx / dual_size;
        result_real[real_idx] = tan(a_real[real_idx]);
        thrust::complex<T> cos2 = cos(a_real[real_idx]) * cos(a_real[real_idx]);
        result_dual[idx] = a_dual[idx] / (cos2);
    }

    //Create a global kernel for the tan method
    template <typename T>
    __global__ void VectorDualTanKernel(const thrust::complex<T>* a_real, 
                                        const thrust::complex<T>* a_dual,
                                        int real_size,
                                        int dual_size,
                                        thrust::complex<T>* result_real,
                                        thrust::complex<T>* result_dual) {
        VectorDualTan(a_real, a_dual, real_size, dual_size, result_real, result_dual);
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
     * IndexGet for HyperDual Vector.  Note the positions of the hessian
     * are completely determined by the real and dual indexes because
     * it doesn't make sense to have a hessian with different indexes than the dual
     */
    template <typename T>
    __device__ void VectorHyperDualIndexGet(//Input
                                            const thrust::complex<T>* a_real,
                                            const thrust::complex<T>* a_dual,
                                            const thrust::complex<T>* a_hyper,
                                            int real_size,
                                            int dual_size,
                                            int start_real,
                                            int end_real,
                                            //Output
                                            thrust::complex<T>* result_real,
                                            thrust::complex<T>* result_dual,
                                            thrust::complex<T>* result_hyper) {
        int off = threadIdx.x + blockIdx.x * blockDim.x;
        // Ensure the thread is within bounds
        //This is a single index into the hyperdual part but two indexes that describe it
        int sz = (end_real-start_real)*dual_size*dual_size;
        if (off >= sz) return;
        //printf("off: %d\n", off);

        //Extract the real index here i is the real index and l is the index into the dual
        //part and l is the index into the column of the hyperdual part
        int i,k,l,o;  //Indexes into the resulting tensor(not the original tensor)
        i=off/(dual_size*dual_size);
        l=(off - i*dual_size*dual_size)/dual_size; //Dual index and row hyperdual index
        k = l; //Dual index
        o=(off - i*dual_size*dual_size-l*dual_size); //Hyperdual index
        //printf("off : %d, i: %d, k(l): %d, o: %d\n", off, i, l,o);
        //Calculate the offsets into the destination tensor
        int off_dest_r = i;
        int off_dest_d = i*dual_size + k;
        int off_dest_h = i*dual_size*dual_size + l*dual_size + o;


        //We need to map this into indexes for the original tensor
        int i_src = i + start_real;
        //The rest are the same
        int k_src = k;
        int l_src = l;
        int o_src = o;
        //printf("off : %d, i_src: %d, k_src: %d, l_src: %d, o_src: %d\n", off, i_src, k_src, l_src, o_src);

        //Get the offset into the original tensor
        int off_src_r = i_src;
        int off_src_d = i_src*dual_size + k_src;
        int off_src_h = i_src*dual_size*dual_size + l_src*dual_size + o_src;
        //printf("off : %d, off_r: %d, off_d: %d, off_h: %d\n", off, off_src_r, off_src_d, off_src_h);
        //Simply copy the values
        if (k == 0 ) {
            result_real[off_dest_r] = a_real[off_src_r];
        }
        if (o == 0) {
            result_dual[off_dest_d] = a_dual[off_src_d];
        }
        result_hyper[off_dest_h] = a_hyper[off_src_h];
    }

    //Create a global kernel for the index get method
    template <typename T>
    __global__ void VectorHyperDualIndexGetKernel(const thrust::complex<T>* a_real,
                                                  const thrust::complex<T>* a_dual,
                                                  const thrust::complex<T>* a_hyper,
                                                  int real_size,
                                                  int dual_size,
                                                  int start_real,
                                                  int end_real,
                                                  thrust::complex<T>* result_real,
                                                  thrust::complex<T>* result_dual,
                                                  thrust::complex<T>* result_hyper) {
        VectorHyperDualIndexGet(a_real, a_dual, a_hyper, real_size, dual_size, 
                                start_real, end_real, 
                                result_real, result_dual, result_hyper);
    }

    /**
     * IndexPut in HyperDual Vector
     * Given a range of indexes, put the values from the input into the result
     * vector
     */
    template <typename T>
    __device__ void VectorHyperDualIndexPut(const thrust::complex<T>* a_real,
                                                  const thrust::complex<T>* a_dual,
                                                  const thrust::complex<T>* a_hyper,
                                                  int real_size, //Size of the input tensor
                                                  int dual_size, //Common to input and output tensors
                                                  int start_real, //For the result tensor
                                                  int end_real,
                                                  thrust::complex<T>* result_real,
                                                  thrust::complex<T>* result_dual,
                                                  thrust::complex<T>* result_hyper) {
        int off = threadIdx.x + blockIdx.x * blockDim.x;
        assert(real_size == end_real-start_real);
        // Ensure the thread is within bounds
        //This is a single index into the hyperdual part but two indexes that describe it
        int sz = real_size*dual_size*dual_size;
        if (off >= sz) return;
        //Source Indexes come first since since the off represents the source
        //Map this to the source tensor
        //First the source indexes
        int i_src, k_src, l_src, o_src;
        i_src = off/(dual_size*dual_size);
        k_src = (off - i_src*dual_size*dual_size)/dual_size;
        l_src = k_src;
        o_src = (off - i_src*dual_size*dual_size-l_src*dual_size);
        //Now the offsets for the source
        int off_src_r = i_src;
        int off_src_d = i_src*dual_size + k_src;
        int off_src_h = i_src*dual_size*dual_size + l_src*dual_size + o_src;

        int i_dest, k_dest, l_dest, o_dest;
        i_dest = i_src+start_real;
        k_dest = k_src;
        l_dest = l_src;
        o_dest = o_src;
        //Now calculate the destination offsets
        int off_dest_r = i_dest;
        int off_dest_d = i_dest*dual_size + k_dest;
        int off_dest_h = i_dest*dual_size*dual_size + l_dest*dual_size + o_dest;

        //Efficiently copy into the destination
        //if (k_dest == 0) {
            result_real[off_dest_r] = a_real[off_src_r];
        //}
        //if (o_dest == 0) {
            result_dual[off_dest_d] = a_dual[off_src_d];
        //}
        result_hyper[off_dest_h] = a_hyper[off_src_h];
    }

    //Create a global wrapper
    template <typename T>
    __global__ void VectorHyperDualIndexPutKernel(const thrust::complex<T>* a_real,
                                                  const thrust::complex<T>* a_dual,
                                                  const thrust::complex<T>* a_hyper,
                                                  int real_size,
                                                  int dual_size,
                                                  int start_real,
                                                  int end_real,
                                                  thrust::complex<T>* result_real,
                                                  thrust::complex<T>* result_dual,
                                                  thrust::complex<T>* result_hyper) {
        VectorHyperDualIndexPut(a_real, a_dual, a_hyper, real_size, dual_size, 
                                      start_real, end_real, 
                                      result_real, result_dual, result_hyper);
    }


    class Matrix {
    public:
        int rows_;                    // Number of rows
        int cols_;                    // Number of columns
        thrust::complex<double>* data_;  // Data
    };

    //Create the corresponding methods in Vector above for the Matrix class
    __device__ void MatrixElementwiseAdd(const thrust::complex<double>* a, 
                                         const thrust::complex<double>* b, 
                                         int rows, 
                                         int cols, 
                                         thrust::complex<double>* result) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;

        // Ensure the thread is within bounds
        if (idx >= rows * cols) return;

        // Perform elementwise addition
        result[idx] = a[idx] + b[idx];
    }

    __global__ void MatrixElementwiseAddKernel(const thrust::complex<double>* a, 
                                               const thrust::complex<double>* b, 
                                               int rows, 
                                               int cols, 
                                               thrust::complex<double>* result) {
        MatrixElementwiseAdd(a, b, rows, cols, result);
    }


    __device__ void MatrixElementwiseMultiply(const thrust::complex<double>* a, 
                                              const thrust::complex<double>* b, 
                                              int rows, 
                                              int cols, 
                                              thrust::complex<double>* result) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;

        // Ensure the thread is within bounds
        if (idx >= rows * cols) return;

        // Perform elementwise multiplication
        result[idx] = a[idx] * b[idx];
    }

    __global__ void MatrixElementwiseMultiplyKernel(const thrust::complex<double>* a, 
                                                    const thrust::complex<double>* b, 
                                                    int rows, 
                                                    int cols, 
                                                    thrust::complex<double>* result) {
        MatrixElementwiseMultiply(a, b, rows, cols, result);
    }

    __device__ void MatrixSquare(thrust::complex<double>* a, 
                                 int rows, 
                                 int cols, 
                                 thrust::complex<double>* result) {
        // Perform elementwise square
        MatrixElementwiseMultiply(a, a, rows, cols, result);
    }

    __global__ void MatrixSquareKernel(thrust::complex<double>* a, 
                                       int rows, 
                                       int cols, 
                                       thrust::complex<double>* result) {
        MatrixSquare(a, rows, cols, result);
    }


    __device__ void MatrixPow(const thrust::complex<double>* a, 
                              double power, 
                              int rows, 
                              int cols, 
                              thrust::complex<double>* result) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;

        // Ensure the thread is within bounds
        if (idx >= rows * cols) return;

        // Perform elementwise power using CUDA's pow function
        result[idx] = pow(a[idx], power);
    }

    __global__ void MatrixPowKernel(const thrust::complex<double>* a, 
                                    double power, 
                                    int rows, 
                                    int cols, 
                                    thrust::complex<double>* result) {
        MatrixPow(a, power, rows, cols, result);
    }


    __device__ void MatrixReduce(const thrust::complex<double>* a, 
                                 int rows, 
                                 int cols, 
                                 thrust::complex<double>* result) {
        extern __shared__ thrust::complex<double> shared_data_matrix[];

        int tid = threadIdx.x;
        int idx = threadIdx.x + blockIdx.x * blockDim.x;

        shared_data_matrix[tid] = (idx < rows * cols) ? a[idx] : thrust::complex<double>(0, 0);
        __syncthreads();

        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                shared_data_matrix[tid] += shared_data_matrix[tid + s];
            }
            __syncthreads();
        }

        if (tid == 0) {
            result[blockIdx.x] = shared_data_matrix[0];
        }
    }


    __global__ void MatrixReduceKernel(const thrust::complex<double>* a, 
                                       int rows, 
                                       int cols, 
                                       thrust::complex<double>* result) {
        MatrixReduce(a, rows, cols, result);
    }


        // Custom atomicAdd for thrust::complex<double>
    // (Requires hardware support for atomicAdd on double)
    __device__ void atomicAddComplex(thrust::complex<double>* address, 
                                    const thrust::complex<double>& val)
    {
        // Trick: reinterpret address as two consecutive doubles
        double* realAddr = reinterpret_cast<double*>(address);
        double* imagAddr = realAddr + 1;

        // Atomic add on real part
        atomicAdd(realAddr, val.real());
        // Atomic add on imaginary part
        atomicAdd(imagAddr, val.imag());
    }


    __device__ void MatrixSum(const thrust::complex<double>* a, 
                            int rows, 
                            int cols, 
                            int dim, 
                            thrust::complex<double>* result) 
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= rows * cols) return;

        int i = idx / cols; // row
        int j = idx % cols; // col

        if (dim == 0) {
            // Use the atomic helper
            atomicAddComplex(&result[j], a[idx]);
        } else {
            atomicAddComplex(&result[i], a[idx]);
        }
    }

    __global__ void MatrixSumKernel(const thrust::complex<double>* a, 
                                    int rows, 
                                    int cols, 
                                    int dim, 
                                    thrust::complex<double>* result) {
        MatrixSum(a, rows, cols, dim, result);
    }

    /**
     * Retrieve elements from start to end into a new Matrix 
     * This is a device function
     */
    __device__ void MatrixIndexGet(const thrust::complex<double>* a, 
                                   int start, 
                                   int end, 
                                   int rows, 
                                   int cols, 
                                   thrust::complex<double>* result) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;

        // Ensure the thread is within bounds
        if (idx >= rows * cols) return;

        // Copy the elements from start to end
        if (idx >= start && idx < end) {
            result[idx - start] = a[idx];
        }
    }


    //Create a global kernel for the index get method
    __global__ void MatrixIndexGetKernel(const thrust::complex<double>* a, 
                                         int start, 
                                         int end, 
                                         int rows, 
                                         int cols, 
                                         thrust::complex<double>* result) {
        MatrixIndexGet(a, start, end, rows, cols, result);
    }


    /**
     * IndexPut for Matrix
     * Put the elements from start to end from input Matrix into the result matrix
     */
    __device__ void MatrixIndexPut(const thrust::complex<double>* input, 
                                   int start, 
                                   int end, 
                                   int rows, 
                                   int cols, 
                                   thrust::complex<double>* result) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;

        // Ensure the thread is within bounds
        if (idx >= rows * cols) return;

        // Copy the elements from start to end
        if (idx >= start && idx < end) {
            result[idx] = input[idx - start];
        }
    }


    //Create a global kernel for the index put method
    __global__ void MatrixIndexPutKernel(const thrust::complex<double>* input, 
                                         int start, 
                                         int end, 
                                         int rows, 
                                         int cols, 
                                         thrust::complex<double>* result) {
        MatrixIndexPut(input, start, end, rows, cols, result);
    }

    //Create a method that does the equivalent of squeeze in libtorch for a matrix
    template <typename T>
    __device__ void MatrixSqueeze(const thrust::complex<double>* a, 
                                  int rows, 
                                  int cols, 
                                  int dim, //The dimension to squeeze
                                  //This should be a vector of size rows if dim is 0 and cols if dim is 1
                                  thrust::complex<T>* result) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= rows * cols) return;
        int i = idx / cols;
        int j = idx % cols;
        if (dim == 0) {
            result[i] = a[idx];
        } else {
            result[j] = a[idx];
        }
    }

    __device__ void MatrixSqueeze(const thrust::complex<double>* a,
                                                    int rows, int cols,
                                                    int dim,
                                                    thrust::complex<double>* result)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;

        // Squeeze only if the dimension is actually 1.
        if (dim == 0 && rows == 1) {
            // shape was (1, cols) -> shape (cols)
            // one thread per col
            if (idx < cols) {
                result[idx] = a[idx]; // a[0, idx]
            }
        }
        else if (dim == 1 && cols == 1) {
            // shape was (rows, 1) -> shape (rows)
            // one thread per row
            if (idx < rows) {
                result[idx] = a[idx * 1]; // a[idx, 0]
            }
        }
        // else do nothing if dimension isn't 1
    }




    __global__ void MatrixSqueezeKernel(const thrust::complex<double>* a,
                                           int rows, int cols,
                                           int dim,
                                           thrust::complex<double>* result)
    {
        MatrixSqueeze(a, rows, cols, dim, result);
    }


    /**
     * Dual tensor class
     */
    class MatrixDual {
    public:
        int rows_;                    // Number of rows
        int cols_;                    // Number of columns
        int dual_size_;               // Dual dimension
        thrust::complex<double>* real_;  // Real part
        thrust::complex<double>* dual_;  // Dual part
    };

    __device__ void MatrixDualElementwiseAdd(const thrust::complex<double>* a_real, 
                                             const thrust::complex<double>* a_dual,
                                             const thrust::complex<double>* b_real,
                                             const thrust::complex<double>* b_dual,
                                             int rows,
                                             int cols,
                                             int dual_size,
                                             thrust::complex<double>* result_real,
                                             thrust::complex<double>* result_dual) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;

        // Ensure the thread is within bounds
        if (idx >= rows * cols * dual_size) return;

        // Perform elementwise addition of the real part
        int i = idx / (cols * dual_size);
        int j = (idx % (cols * dual_size)) / dual_size;
        int k = idx % dual_size;

        int off = i * cols + j;
        if (k == 0) {
            result_real[off] = a_real[off] + b_real[off];
        }

        // Perform elementwise addition for the dual part
        result_dual[idx] = a_dual[idx] + b_dual[idx];
    }

    __global__ void MatrixDualElementwiseAddKernel(const thrust::complex<double>* a_real, 
                                                   const thrust::complex<double>* a_dual,
                                                   const thrust::complex<double>* b_real,
                                                   const thrust::complex<double>* b_dual,
                                                   int rows,
                                                   int cols,
                                                   int dual_size,
                                                   thrust::complex<double>* result_real,
                                                   thrust::complex<double>* result_dual) {
        MatrixDualElementwiseAdd(a_real, a_dual, b_real, b_dual, rows, cols, dual_size, result_real, result_dual);
    }


    /**
     * Elementwise multiplication of two MatrixDual tensors
     */
    __device__ void MatrixDualElementwiseMultiply(const thrust::complex<double>* a_real, 
                                                  const thrust::complex<double>* a_dual,
                                                  const thrust::complex<double>* b_real,
                                                  const thrust::complex<double>* b_dual,
                                                  int rows,
                                                  int cols,
                                                  int dual_size,
                                                  thrust::complex<double>* result_real,
                                                  thrust::complex<double>* result_dual) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;

        // Ensure the thread is within bounds
        if (idx >= rows * cols * dual_size) return;

        // Perform elementwise multiplication of the real part
        int i = idx / (cols * dual_size);
        int j = (idx % (cols * dual_size)) / dual_size;
        int k = idx % dual_size;

        int off = i * cols + j;
        if (k == 0) {
            result_real[off] = a_real[off] * b_real[off];
        }

        // Perform elementwise multiplication for the dual part
        result_dual[idx] = a_real[off] * b_dual[idx] + b_real[off] * a_dual[idx];
    }

    __global__ void MatrixDualElementwiseMultiplyKernel(const thrust::complex<double>* a_real, 
                                                        const thrust::complex<double>* a_dual,
                                                        const thrust::complex<double>* b_real,
                                                        const thrust::complex<double>* b_dual,
                                                        int rows,
                                                        int cols,
                                                        int dual_size,
                                                        thrust::complex<double>* result_real,
                                                        thrust::complex<double>* result_dual) {
        MatrixDualElementwiseMultiply(a_real, a_dual, b_real, b_dual, rows, cols, dual_size, result_real, result_dual);
    }



    /**
     * Square each element in the MatrixDual tensor
     */
    __device__ void MatrixDualSquare(const thrust::complex<double>* a_real, 
                                     const thrust::complex<double>* a_dual,
                                     int rows,
                                     int cols,
                                     int dual_size,
                                     thrust::complex<double>* result_real,
                                     thrust::complex<double>* result_dual) {
        // Perform elementwise square for the real part
        MatrixDualElementwiseMultiply(a_real, a_dual, a_real, a_dual, rows, cols, dual_size, result_real, result_dual);
    }


    //Create a method that calculate the result of a MatrixDual multiplied to a VectorDual tensor
    //Here we are assuming the number of columns in the matrix in the same as the number of rows in the vector
    //Assume result_real and result_dual are initialized to zero
    __device__ void MatrixDualVectorDualMultiply(
        const thrust::complex<double>* a_real, 
        const thrust::complex<double>* a_dual,
        const thrust::complex<double>* b_real,
        const thrust::complex<double>* b_dual,
        int rows,
        int cols,
        int dual_size,
        thrust::complex<double>* result_real, // size = rows
        thrust::complex<double>* result_dual) // size = rows * dual_size
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        // total = rows * (dual_size)   -> one thread per (i,k)
        int total = rows * dual_size;
        if (idx >= total) return;

        int i = idx / dual_size; // row index
        int k = idx % dual_size; // dual index

        // We'll accumulate partial sums in local variables
        thrust::complex<double> sum_real(0.0, 0.0);
        thrust::complex<double> sum_dual(0.0, 0.0);

        for (int j = 0; j < cols; ++j) {
            // A's real/dual
            thrust::complex<double> aR = a_real[i*cols + j];          // A_real(i,j)
            thrust::complex<double> aD = a_dual[(i*cols + j)*dual_size + k]; 
            // a_dual is typically size [rows*cols*dual_size]; index = (i*cols + j)*dual_size + k

            // B's real/dual
            thrust::complex<double> bR = b_real[j];                    // B_real(j)
            thrust::complex<double> bD = b_dual[j*dual_size + k]; 
            // b_dual is [cols*dual_size] for a vector

            // Add real contribution only if k == 0
            // But we can simply do it once in the same loop: 
            // The real part of the result is independent of k,
            // so do it only if k=0. 
            // Or store partial sums if k=0. 
            if (k == 0) {
                sum_real += aR * bR;
            }

            // Always accumulate dual sum
            // result_dual(i, k) = sum_j [ a_real(i,j)*b_dual(j,k) + a_dual(i,j,k)*b_real(j) ]
            sum_dual += aR*bD + aD*bR;
        }

        // Write out
        // If k=0 => result_real[i] = sum over j of a_real(i,j)*b_real(j)
        if (k == 0) {
            result_real[i] += sum_real; // or result_real[i] = sum_real if we assume zero-initialized
        }

        // Dual part => store in [i*dual_size + k]
        result_dual[i*dual_size + k] += sum_dual;
    }




    __global__ void MatrixDualVectorDualMultiplyKernel(const thrust::complex<double>* a_real, 
                                                       const thrust::complex<double>* a_dual,
                                                       const thrust::complex<double>* b_real,
                                                       const thrust::complex<double>* b_dual,
                                                       int rows,
                                                       int cols,
                                                       int dual_size,
                                                       thrust::complex<double>* result_real,
                                                       thrust::complex<double>* result_dual) {
        MatrixDualVectorDualMultiply(a_real, a_dual, b_real, b_dual, rows, cols, dual_size, result_real, result_dual);
    }

    __global__ void MatrixDualSquareKernel(thrust::complex<double>* a_real, 
                                           thrust::complex<double>* a_dual,
                                           int rows,
                                           int cols,
                                           int dual_size,
                                           thrust::complex<double>* result_real,
                                           thrust::complex<double>* result_dual) {
        MatrixDualSquare(a_real, a_dual, rows, cols, dual_size, result_real, result_dual);
    }

    /**
     * Pow method for MatrixDual
     */
    __device__ void MatrixDualPow(const thrust::complex<double>* a_real, 
                                  const thrust::complex<double>* a_dual,
                                  double power,
                                  int rows,
                                  int cols,
                                  int dual_size,
                                  thrust::complex<double>* result_real,
                                  thrust::complex<double>* result_dual) {
        // Perform elementwise power for the real part
        MatrixPow(a_real, power, rows, cols, result_real);

        // Perform elementwise power for the dual part
        //The result here is power*(a_real^(power-1))*a_dual
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= rows * cols * dual_size) return;
        int i = idx / (cols * dual_size);
        int j = (idx % (cols * dual_size)) / dual_size;
        //int k = idx % dual_size;

        int off = i * cols + j;
        result_dual[idx] = power * pow(a_real[off], power - 1) * a_dual[idx];
    }

    __global__ void MatrixDualPowKernel(const thrust::complex<double>* a_real, 
                                        const thrust::complex<double>* a_dual,
                                        double power,
                                        int rows,
                                        int cols,
                                        int dual_size,
                                        thrust::complex<double>* result_real,
                                        thrust::complex<double>* result_dual) {
        MatrixDualPow(a_real, a_dual, power, rows, cols, dual_size, result_real, result_dual);
    }

    /**
     * 2D slice for a dual matrix.
     *
     *  Slices the real and dual parts in the row range [rowStart, rowEnd)
     *  and column range [colStart, colEnd).
     *
     * Inputs:
     *   a_real      : (rows * cols)    real array
     *   a_dual      : (rows * cols * dual_size) dual array
     *   rows, cols  : full matrix dimensions
     *   dual_size   : number of dual components per (row,col)
     *   rowStart, rowEnd : row slice range
     *   colStart, colEnd : column slice range
     *
     * Outputs:
     *   out_real : (outRows * outCols) submatrix real part
     *   out_dual : (outRows * outCols * dual_size) submatrix dual part
     *
     * where
     *   outRows = (rowEnd - rowStart)
     *   outCols = (colEnd - colStart)
     */
    __device__ void MatrixDualIndexGet2D(const thrust::complex<double>* a_real,
                                         const thrust::complex<double>* a_dual,
                                         int rows,
                                         int cols,
                                         int dual_size,
                                         int rowStart,
                                         int rowEnd,
                                         int colStart,
                                         int colEnd,
                                         thrust::complex<double>* out_real,
                                         thrust::complex<double>* out_dual)
    {
        // We will compute the size of the output submatrix
        int outRows = rowEnd - rowStart;  // must be <= rows
        int outCols = colEnd - colStart;  // must be <= cols
        int outSize = outRows * outCols;  // number of elements in the submatrix (real part)
        int totalThreads = outSize * dual_size; // total for the dual part indexing

        // Each thread index covers exactly one partial dimension of the sub-slice.
        // i.e. a 3D viewpoint: (localRow, localCol, k)
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= totalThreads) return;

        // Decompose idx => local row, local col, dual index
        int localRow = idx / (outCols * dual_size);
        int localRest = idx % (outCols * dual_size);
        int localCol = localRest / dual_size;
        int k        = localRest % dual_size;  // partial dimension

        // Map local row,col back to global row,col
        int globalRow = rowStart + localRow;
        int globalCol = colStart + localCol;

        // Flattened offsets in the original matrix
        int originalOff = globalRow * cols + globalCol;  // for the real array
        int originalDualOff = originalOff * dual_size + k; // for the dual array

        // Flattened offsets in the submatrix
        int subOff = localRow * outCols + localCol;
        int subDualOff = subOff * dual_size + k;

        // Copy the real part once per (localRow, localCol) => that means k==0 can do it,
        // or we can do a separate pass.  We'll do the approach where if k == 0, we copy real:
        if (k == 0) {
            out_real[subOff] = a_real[originalOff];
        }

        // Always copy the dual part:
        out_dual[subDualOff] = a_dual[originalDualOff];
    }

    __global__ void MatrixDualIndexGet2DKernel(const thrust::complex<double>* a_real,
                                            const thrust::complex<double>* a_dual,
                                            int rows,
                                            int cols,
                                            int dual_size,
                                            int rowStart,
                                            int rowEnd,
                                            int colStart,
                                            int colEnd,
                                            thrust::complex<double>* out_real,
                                            thrust::complex<double>* out_dual)
    {
        MatrixDualIndexGet2D(a_real, a_dual,
                            rows, cols, dual_size,
                            rowStart, rowEnd, colStart, colEnd,
                            out_real, out_dual);
    }


    /**
     * Copy a subregion from a source dual matrix into a destination dual matrix.
     *
     * The source dual matrix is:
     *   - src_real (size = srcRows * srcCols)
     *   - src_dual (size = srcRows * srcCols * dual_size)
     *
     * We define a sub-slice in the source:
     *   rows in [rowStartSrc..rowEndSrc)
     *   cols in [colStartSrc..colEndSrc)
     *
     * The sub-slice is placed into the destination matrix starting at
     *   (rowStartDst, colStartDst).
     *
     * The destination dual matrix is:
     *   - dst_real (size = dstRows * dstCols)
     *   - dst_dual (size = dstRows * dstCols * dual_size)
     *
     * We assume:
     *   - rowEndSrc <= srcRows, colEndSrc <= srcCols
     *   - rowStartDst + (rowEndSrc-rowStartSrc) <= dstRows
     *   - colStartDst + (colEndSrc-colStartSrc) <= dstCols
     *   - dual_size is the same for both source & destination
     */
    __device__ void MatrixDualIndexPut2D(
        const thrust::complex<double>* src_real,
        const thrust::complex<double>* src_dual,
        int srcRows, int srcCols,
        int dual_size,
        int rowStartSrc, int rowEndSrc,
        int colStartSrc, int colEndSrc,
        thrust::complex<double>* dst_real,
        thrust::complex<double>* dst_dual,
        int dstRows, int dstCols,
        int rowStartDst,
        int colStartDst
    )
    {
        // 1) Determine sub-slice shape
        int subRows = rowEndSrc - rowStartSrc;  // # of rows in the sub-slice
        int subCols = colEndSrc - colStartSrc;  // # of cols in the sub-slice

        // 2) The total number of elements in the sub-slice (for the dual array)
        int totalSubSize = subRows * subCols * dual_size;

        // 3) Thread index
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= totalSubSize) return;

        // 4) Decompose idx -> local row, local col, partial index
        int localRow = idx / (subCols * dual_size);
        int remainder = idx % (subCols * dual_size);
        int localCol = remainder / dual_size;
        int k        = remainder % dual_size;

        // 5) Map to global coords in the SOURCE
        int srcRow = rowStartSrc + localRow;
        int srcCol = colStartSrc + localCol;

        // Flattened offset in src's real array
        int srcOff = srcRow * srcCols + srcCol;
        // Flattened offset in src's dual array
        int srcDualOff = srcOff * dual_size + k;

        // 6) Map to global coords in the DEST
        int dstRow = rowStartDst + localRow;
        int dstCol = colStartDst + localCol;

        // Flattened offset in dst's real array
        int dstOff = dstRow * dstCols + dstCol;
        // Flattened offset in dst's dual array
        int dstDualOff = dstOff * dual_size + k;

        // 7) Copy real part if k == 0
        if (k == 0) {
            dst_real[dstOff] = src_real[srcOff];
        }

        // 8) Always copy dual part
        dst_dual[dstDualOff] = src_dual[srcDualOff];
    }

    __global__ void MatrixDualIndexPut2DKernel(
        const thrust::complex<double>* src_real,
        const thrust::complex<double>* src_dual,
        int srcRows, int srcCols,
        int dual_size,
        int rowStartSrc, int rowEndSrc,
        int colStartSrc, int colEndSrc,
        thrust::complex<double>* dst_real,
        thrust::complex<double>* dst_dual,
        int dstRows, int dstCols,
        int rowStartDst,
        int colStartDst
    )
    {
        MatrixDualIndexPut2D(src_real, src_dual,
                            srcRows, srcCols, dual_size,
                            rowStartSrc, rowEndSrc, colStartSrc, colEndSrc,
                            dst_real, dst_dual,
                            dstRows, dstCols, rowStartDst, colStartDst);
    }
    

    // Squeeze method for a dual matrix
    __device__ void MatrixDualSqueeze(const thrust::complex<double>* a_real, 
                                    const thrust::complex<double>* a_dual, 
                                    int rows, 
                                    int cols, 
                                    int dual_size,
                                    int dim, // The dimension to squeeze
                                    thrust::complex<double>* result_real, 
                                    thrust::complex<double>* result_dual) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;

        // Total number of elements in the dual matrix
        int total_elements = rows * cols * dual_size;
        if (idx >= total_elements) return;

        // Extract the indices for the real and dual parts
        int i = idx / (cols * dual_size);       // Row index
        int j = (idx % (cols * dual_size)) / dual_size; // Column index
        int k = idx % dual_size;               // Dual index

        if (dim == 0) {
            // Squeezing along rows
            int result_idx_real = j; // For real part
            int result_idx_dual = j * dual_size + k; // For dual part

            if (k == 0) {
                result_real[result_idx_real] = a_real[i * cols + j];
            }
            result_dual[result_idx_dual] = a_dual[idx];
        } else if (dim == 1) {
            // Squeezing along columns
            int result_idx_real = i; // For real part
            int result_idx_dual = i * dual_size + k; // For dual part

            if (k == 0) {
                result_real[result_idx_real] = a_real[i * cols + j];
            }
            result_dual[result_idx_dual] = a_dual[idx];
        }
    }

// Kernel wrapper for MatrixDualSqueeze
__global__ void MatrixDualSqueezeKernel(const thrust::complex<double>* a_real, 
                                        const thrust::complex<double>* a_dual, 
                                        int rows, 
                                        int cols, 
                                        int dual_size,
                                        int dim, 
                                        thrust::complex<double>* result_real, 
                                        thrust::complex<double>* result_dual) {
    MatrixDualSqueeze(a_real, a_dual, rows, cols, dual_size, dim, result_real, result_dual);
}

/**
 * @brief Computes a tensor result based on the signs of two input dual matrices.
 *
 * The function processes two input dual matrices, `a` and `b`, and modifies `a` based
 * on the sign of its real part (`a_sign`) and the sign of the real part of `b` (`b_sign`).
 * Small values are treated as positive to ensure compatibility with MATLAB.
 *
 * @param a_real Real part of the input tensor `a`.
 * @param a_dual Dual part of the input tensor `a`.
 * @param b_real Real part of the input tensor `b`.
 * @param b_dual Dual part of the input tensor `b`.
 * @param rows Number of rows in the dual matrix.
 * @param cols Number of columns in the dual matrix.
 * @param dual_size Dual dimension size.
 * @param result_real Real part of the resulting tensor.
 * @param result_dual Dual part of the resulting tensor.
 * @param tol Tolerance for determining the sign (default is 1.0e-6).
 */
template <typename T>
__device__ void MatrixDualSigncond(const thrust::complex<T>* a_real,
                                   const thrust::complex<T>* a_dual,
                                   const thrust::complex<T>* b_real,
                                   const thrust::complex<T>* b_dual,
                                   int rows, 
                                   int cols, 
                                   int dual_size,
                                   thrust::complex<T>* result_real,
                                   thrust::complex<T>* result_dual,
                                   T tol = 1.0e-6) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Total number of elements in the dual matrix
    int total_elements = rows * cols * dual_size;
    if (idx >= total_elements) return;

    // Compute indices for real and dual parts
    int i = idx / (cols*dual_size) ;       // Row index
    int j = (idx-i*cols*dual_size)/dual_size; // Column index
    int k = idx-(i*cols*dual_size+j*dual_size);               // Dual index

    int real_idx = i * cols + j; // Index for real part

    // Retrieve the real parts of `a` and `b`
    T a_real_part = a_real[real_idx].real();
    T b_real_part = b_real[real_idx].real();

    // Compute the sign of the real parts with tolerance
    int a_sign = (fabs(a_real_part) >= tol) ? (a_real_part >= 0 ? 1 : -1) : 1;
    int b_sign = (fabs(b_real_part) >= tol) ? (b_real_part >= 0 ? 1 : -1) : 1;
    //printf("a_real_part: %f, b_real_part : %f, a_sign: %d, b_sign: %d\n", a_real_part, b_real_part, a_sign, b_sign);

    // Apply the corrected conditional logic for real part computation
    if (k == 0) { // Real part computation only once per element
        //printf("result_real[%d] before = %f\n", real_idx, a_real[real_idx].real());
        //printf("a_sign: %d, b_sign: %d\n", a_sign, b_sign);
        if (b_sign >= 0) {
            result_real[real_idx] = (a_sign >= 0) ? a_real[real_idx] : -a_real[real_idx];
        } else {
            result_real[real_idx] = (a_sign >= 0) ? -a_real[real_idx] : a_real[real_idx];
        }
        //printf("result_real[%d] after = %f\n", real_idx, result_real[real_idx].real());

    }


    // Apply the corrected conditional logic for dual part computation
    if (b_sign >= 0) {
        result_dual[idx] = (a_sign >= 0) ? a_dual[idx] : -a_dual[idx];
    } else {
        result_dual[idx] = (a_sign >= 0) ? -a_dual[idx] : a_dual[idx];
    }
}

// Kernel wrapper for MatrixDualSigncond
__global__ void MatrixDualSigncondKernel(const thrust::complex<double>* a_real,
                                         const thrust::complex<double>* a_dual,
                                         const thrust::complex<double>* b_real,
                                         const thrust::complex<double>* b_dual,
                                         int rows, 
                                         int cols, 
                                         int dual_size,
                                         thrust::complex<double>* result_real,
                                         thrust::complex<double>* result_dual,
                                         double tol = 1.0e-6) {
    MatrixDualSigncond(a_real, a_dual, b_real, b_dual, rows, cols, dual_size, result_real, result_dual, tol);
}

__device__ void MatrixHyperDualElementwiseAdd(
    const thrust::complex<double>* a_real,
    const thrust::complex<double>* a_dual,
    const thrust::complex<double>* a_hyper,
    const thrust::complex<double>* b_real,
    const thrust::complex<double>* b_dual,
    const thrust::complex<double>* b_hyper,
    int rows, int cols, int dual_size,
    thrust::complex<double>* result_real,
    thrust::complex<double>* result_dual,
    thrust::complex<double>* result_hyper)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int totalSize = rows * cols * dual_size * dual_size;
    if (idx >= totalSize) return;

    // decode i, j, k, l
    int i = idx / (cols * dual_size * dual_size);
    int remainder = idx % (cols * dual_size * dual_size);
    int j = remainder / (dual_size * dual_size);
    int ij_remainder = remainder % (dual_size * dual_size);
    int k = ij_remainder / dual_size;
    int l = ij_remainder % dual_size;

    int off = i * cols + j;

    // Add real part only once per i,j
    if (k == 0 && l == 0) {
        result_real[off] = a_real[off] + b_real[off];
    }

    // Add second-order partials (hyper part). 
    // Because we're indexing up to dual_size^2 for each i,j, 
    // we assume a_hyper, b_hyper, etc. also have length 
    // rows * cols * dual_size^2.
    result_hyper[idx] = a_hyper[idx] + b_hyper[idx];

    // If your "dual" array is only of size rows*cols*dual_size 
    // (for first-order partials), you need a different indexing approach.
    // Otherwise, if you are storing first-order partials in the same array
    // as the second-order partials, you must decide which subset of 
    // idx corresponds to the first-order part. 
    //
    // For example, if first-order partials are at 'k' dimension only 
    // when 'l=0', then you might do:
    if (l == 0) {
        int dual_off = i*cols*dual_size + j*dual_size + k; 
        result_dual[dual_off] = a_dual[dual_off] + b_dual[dual_off];
    }
}

__global__ void MatrixHyperDualElementwiseAddKernel(
    const thrust::complex<double>* a_real,
    const thrust::complex<double>* a_dual,
    const thrust::complex<double>* a_hyper,
    const thrust::complex<double>* b_real,
    const thrust::complex<double>* b_dual,
    const thrust::complex<double>* b_hyper,
    int rows, int cols, int dual_size,
    thrust::complex<double>* result_real,
    thrust::complex<double>* result_dual,
    thrust::complex<double>* result_hyper)
{
    MatrixHyperDualElementwiseAdd(a_real, a_dual, a_hyper,
                                  b_real, b_dual, b_hyper,
                                  rows, cols, dual_size,
                                  result_real, result_dual, result_hyper);
}


    /**
     * HyperDual Matrix class
     */
    template <typename T>
    class MatrixHyperDual {
    public:
        int rows_;                    // Number of rows
        int cols_;                    // Number of columns
        int dual_size_;               // Dual dimension
        thrust::complex<T>* real_;  // Real part
        thrust::complex<T>* dual_;  // Dual part
        thrust::complex<T>* hyper_; // Hyper dual part
    };
    
    template <typename T>
    __device__ void MatrixHyperDualElementwiseAdd(const thrust::complex<T>* a_real, 
                                                  const thrust::complex<T>* a_dual,
                                                  const thrust::complex<T>* a_hyper,
                                                  const thrust::complex<T>* b_real,
                                                  const thrust::complex<T>* b_dual,
                                                  const thrust::complex<T>* b_hyper,
                                                  int rows,
                                                  int cols,
                                                  int dual_size,
                                                  thrust::complex<double>* result_real,
                                                  thrust::complex<double>* result_dual,
                                                  thrust::complex<double>* result_hyper) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;

        // Ensure the thread is within bounds
        if (idx >= rows * cols * dual_size*dual_size) return;

        // Perform elementwise addition of the real part
        int i = idx / (cols * dual_size*dual_size);
        int j = (idx % (cols * dual_size*dual_size)) / (dual_size*dual_size);
        int k = (idx % (dual_size*dual_size)) / dual_size;
        int l = idx % dual_size;
        
        int off = i * cols + j;
        if (k == 0 && l == 0) {
            result_real[off] = a_real[off] + b_real[off];
        }

        if (l == 0) {
            result_dual[idx] = a_dual[idx] + b_dual[idx];
        }


        // Perform elementwise addition for the hyper dual part
        result_hyper[idx] = a_hyper[idx] + b_hyper[idx];
    }
    
    template <typename T>
    __global__ void MatrixHyperDualElementwiseAddKernel(const thrust::complex<T>* a_real, 
                                                        const thrust::complex<T>* a_dual,
                                                        const thrust::complex<T>* a_hyper,
                                                        const thrust::complex<T>* b_real,
                                                        const thrust::complex<T>* b_dual,
                                                        const thrust::complex<T>* b_hyper,
                                                        int rows,
                                                        int cols,
                                                        int dual_size,
                                                        thrust::complex<T>* result_real,
                                                        thrust::complex<T>* result_dual,
                                                        thrust::complex<T>* result_hyper) {
        MatrixHyperDualElementwiseAdd(a_real, a_dual, a_hyper, b_real, b_dual, b_hyper, rows, cols, dual_size, result_real, result_dual, result_hyper);
    }

    /**
     * Hyper-dual elementwise multiply:
     *   result_real, result_dual, result_hyper  (output)
     */
    template <typename T>
    __device__ void MatrixHyperDualElementwiseMul(
        // Inputs
        const thrust::complex<T>* a_real,
        const thrust::complex<T>* a_dual,
        const thrust::complex<T>* a_hyper,
        const thrust::complex<T>* b_real,
        const thrust::complex<T>* b_dual,
        const thrust::complex<T>* b_hyper,
        int rows,
        int cols,
        int dual_size,
        // Outputs
        thrust::complex<T>* result_real,
        thrust::complex<T>* result_dual,
        thrust::complex<T>* result_hyper)
    {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        int total_hyper = rows * cols * dual_size * dual_size;
        if (idx >= total_hyper) return;

        // Decompose idx => (i,j,k,l)
        // i in [0..rows-1], j in [0..cols-1], k,l in [0..dual_size-1]
        int i = idx / (cols * dual_size * dual_size);
        int rem1 = idx % (cols * dual_size * dual_size);
        int j = rem1 / (dual_size * dual_size);
        int rem2 = rem1 % (dual_size * dual_size);
        int k = rem2 / dual_size;
        int l = rem2 % dual_size;

        //  (i,j) => offset for the real part
        int off = i * cols + j;

        //  (i,j,k) => offset for the first-order dual array
        //   The array size for dual is rows*cols*dual_size,
        //   so we compute:
        int dual_off_a = i*cols*dual_size + j*dual_size + k;
        int dual_off_b = i*cols*dual_size + j*dual_size + l;  // used in the hyper cross term

        //--------------------------------------------------------------------------
        // 1) Real part: c_real = a_real * b_real
        //--------------------------------------------------------------------------
        // We only compute this once per (i,j), i.e. when (k,l)==(0,0).
        if (k == 0 && l == 0) {
            result_real[off] = a_real[off] * b_real[off];
        }

        //--------------------------------------------------------------------------
        // 2) First-order dual part: 
        //    c_dual[k] = (a_real*b_dual[k] + b_real*a_dual[k])
        //--------------------------------------------------------------------------
        // We only compute this for l == 0 (so effectively dual_size threads per i,j).
        if (l == 0) {
            result_dual[dual_off_a] = a_real[off] * b_dual[dual_off_a] 
                                    + b_real[off] * a_dual[dual_off_a];
        }

        //--------------------------------------------------------------------------
        // 3) Second-order hyper part:
        //    c_hyper(k,l) = a_real*b_hyper(k,l)
        //                  + b_real*a_hyper(k,l)
        //                  + a_dual[k]*b_dual[l]
        //--------------------------------------------------------------------------
        result_hyper[idx] = (a_real[off] * b_hyper[idx])
                        + (b_real[off] * a_hyper[idx])
                        + (a_dual[dual_off_a] * b_dual[dual_off_b]);
    }


    template <typename T>
    __global__
    void MatrixHyperDualElementwiseMulKernel(
        const thrust::complex<T>* a_real,
        const thrust::complex<T>* a_dual,
        const thrust::complex<T>* a_hyper,
        const thrust::complex<T>* b_real,
        const thrust::complex<T>* b_dual,
        const thrust::complex<T>* b_hyper,
        int rows,
        int cols,
        int dual_size,
        thrust::complex<T>* result_real,
        thrust::complex<T>* result_dual,
        thrust::complex<T>* result_hyper)
    {
        MatrixHyperDualElementwiseMul(a_real, a_dual, a_hyper,
                                    b_real, b_dual, b_hyper,
                                    rows, cols, dual_size,
                                    result_real, result_dual, result_hyper);
    }

    /**
     * Multiply a HyperDual matrix A (rows×cols) by a HyperDual vector B (cols×1).
     * The result is a HyperDual vector R (rows×1).
     *
     * Indexing:
     *   A_real   : [rows*cols]
     *   A_dual   : [rows*cols*dual_size]
     *   A_hyper  : [rows*cols*dual_size*dual_size]
     *
     *   B_real   : [cols]
     *   B_dual   : [cols*dual_size]
     *   B_hyper  : [cols*dual_size*dual_size]
     *
     *   R_real   : [rows]
     *   R_dual   : [rows*dual_size]
     *   R_hyper  : [rows*dual_size*dual_size]
     *
     * Each thread handles one (i,k,l), summing over j in [0..cols-1].
     */
    __device__
    void MatrixHyperDualVectorHyperDualMultiply(
        // A: [A_real, A_dual, A_hyper]
        const thrust::complex<double>* A_real,
        const thrust::complex<double>* A_dual,
        const thrust::complex<double>* A_hyper,
        // B: [B_real, B_dual, B_hyper]
        const thrust::complex<double>* B_real,
        const thrust::complex<double>* B_dual,
        const thrust::complex<double>* B_hyper,
        // dimensions
        int rows,
        int cols,
        int dual_size,
        // R: [R_real, R_dual, R_hyper]
        thrust::complex<double>* R_real,
        thrust::complex<double>* R_dual,
        thrust::complex<double>* R_hyper)
    {
        // Flatten (i,k,l) -> idx
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        int total = rows * dual_size * dual_size;
        if (idx >= total) return;

        // decode (i,k,l)
        int i = idx / (dual_size * dual_size);        // row of the result
        int rem = idx % (dual_size * dual_size);
        int k = rem / dual_size;                      // partial index for A
        int l = rem % dual_size;                      // partial index for B

        // We'll sum local partials for real, dual, hyper
        thrust::complex<double> sum_real(0,0);
        thrust::complex<double> sum_dual(0,0);
        thrust::complex<double> sum_hyper(0,0);

        // sum over columns j
        for (int j = 0; j < cols; ++j) {
            // Offsets into A
            //   A_real(i,j)   => i*cols + j
            //   A_dual(i,j,k) => (i*cols + j)*dual_size + k
            //   A_hyper(i,j,k,l) => (i*cols + j)*dual_size*dual_size + k*dual_size + l
            int offA_real  = i*cols + j;
            int offA_dual  = (i*cols + j)*dual_size + k;
            int offA_hyper = (i*cols + j)*dual_size*dual_size + k*dual_size + l;

            thrust::complex<double> aR = A_real[offA_real];
            thrust::complex<double> aD = A_dual[offA_dual];
            thrust::complex<double> aH = A_hyper[offA_hyper];

            // Offsets into B
            //   B_real(j)   => j
            //   B_dual(j,k) => j*dual_size + k
            //   B_dual(j,l) => j*dual_size + l
            //   B_hyper(j,k,l) => j*(dual_size^2) + k*dual_size + l
            thrust::complex<double> bR = B_real[j];
            thrust::complex<double> bD_k = B_dual[j*dual_size + k]; // for dual cross with aD
            thrust::complex<double> bD_l = B_dual[j*dual_size + l]; // for hyper cross
            thrust::complex<double> bH   = B_hyper[j*dual_size*dual_size + k*dual_size + l];

            // (1) second-order partial
            //    hyper(i,k,l) += aR*bH + aH*bR + aD*bD_l
            sum_hyper += (aR*bH) + (aH*bR) + (aD*bD_l);

            // (2) first-order partial: only valid if l == 0
            //    dual(i,k) += aR*bD_k + aD*bR
            if (l == 0) {
                sum_dual += (aR*bD_k) + (aD*bR);
            }

            // (3) real part: only valid if (k,l) == (0,0)
            //    real(i) += aR*bR
            if (k == 0 && l == 0) {
                sum_real += (aR*bR);
            }
        }

        // Write out to R
        // assume R_* are zero-initialized, or you do R_*[...] = sum_...
        if (k == 0 && l == 0) {
            R_real[i] += sum_real;
        }
        if (l == 0) {
            R_dual[i*dual_size + k] += sum_dual;
        }
        R_hyper[i*dual_size*dual_size + k*dual_size + l] += sum_hyper;
    }

    __global__ void MatrixHyperDualVectorHyperDualMultiplyKernel(
        const thrust::complex<double>* A_real,
        const thrust::complex<double>* A_dual,
        const thrust::complex<double>* A_hyper,
        const thrust::complex<double>* B_real,
        const thrust::complex<double>* B_dual,
        const thrust::complex<double>* B_hyper,
        int rows,
        int cols,
        int dual_size,
        thrust::complex<double>* R_real,
        thrust::complex<double>* R_dual,
        thrust::complex<double>* R_hyper)
    {
        MatrixHyperDualVectorHyperDualMultiply(A_real, A_dual, A_hyper,
                                               B_real, B_dual, B_hyper,
                                               rows, cols, dual_size,
                                               R_real, R_dual, R_hyper);
    }



} // namespace Janus
#endif // _CU_DUAL_TENSOR_HPP