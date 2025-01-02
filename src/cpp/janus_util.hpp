#ifndef JANUS_UTIL_HPP
#define JANUS_UTIL_HPP

#include <torch/torch.h>
#include "tensordual.hpp"
#include <iostream>
#include <taco.h>

using namespace janus;


// Define the primary template for the tensor
template <typename T>
struct is_tensor : std::false_type {};

// Define the primary template for the dual trait
template <typename T>
struct is_dual : std::false_type {};

// Define the primary template for the hyperdual trait
template <typename T>
struct is_hyperdual : std::false_type {};

// Specialize the is_dual trait for DualNumber
template <>
struct is_dual<TensorDual> : std::true_type {};

// Specialize the is_tensor trait for torch::Tensor
template <>
struct is_tensor<torch::Tensor> : std::true_type {};


template <typename T>
bool check_if_tensor(const T& instance) {
    return is_tensor<T>::value;
}

template <typename T>
bool check_if_dual(const T& instance) {
    return is_dual<T>::value;
}

template <typename T>
bool check_if_hyperdual(const T& instance) {
    return is_hyperdual<T>::value;
}



namespace janus {
   auto eps = std::numeric_limits<double>::epsilon();
   auto feps = std::numeric_limits<float>::epsilon();

   bool isCOOTensor(const torch::Tensor& tensor) {
    return tensor.layout() == torch::kSparse;
  }
 


    torch::Tensor convertToCOO(const torch::Tensor& dense_tensor) {
        //If it is already a COO tensor return it
        if (isCOOTensor(dense_tensor)) {
            return dense_tensor;
        }
        // Ensure the input tensor is on the CPU for processing
        auto cpu_tensor = dense_tensor.cpu();

        // Convert the dense tensor to sparse COO format
        auto sparse_tensor = cpu_tensor.to_sparse().coalesce();

        // Move the sparse tensor back to the original device
        sparse_tensor = sparse_tensor.to(dense_tensor.device());

        return sparse_tensor;
    }


  torch::Tensor bmax(torch::Tensor &x, torch::Tensor &y)
  {
    return torch::where(x > y, x, y);
  }



    template <typename T>
    taco::Tensor<T> ConvertLibTorchToTaco2D(const torch::Tensor& libtorch_tensor) {
        // Ensure the LibTorch tensor is in COO format and on CPU
        auto coo_tensor = libtorch_tensor.coalesce().cpu();
        auto indices = coo_tensor.indices();
        auto values = coo_tensor.values();

        // Check if the LibTorch tensor type matches the template type T
        if (values.scalar_type() != torch::CppTypeToScalarType<T>()) {
            throw std::runtime_error("Mismatch between LibTorch tensor type and TACO tensor type.");
        }

        // Initialize TACO tensor
        int rows = libtorch_tensor.size(0);
        int cols = libtorch_tensor.size(1);
        taco::Tensor<T> taco_tensor = taco::Tensor<T>({rows, cols}, taco::Format({taco::Sparse, taco::Sparse}));

        // Insert non-zero elements into TACO tensor
        for (int64_t i = 0; i < values.size(0); ++i) {
            int row = indices[0][i].item<int>();
            int col = indices[1][i].item<int>();
            T value = values[i].item<T>(); // Use template type T for values
            taco_tensor.insert({row, col}, value);
        }

        // Pack the tensor
        taco_tensor.pack();
        return taco_tensor;
    }


    template <typename T>
    torch::Tensor ConvertTacoToLibTorch2D(const taco::Tensor<T>& taco_tensor) {
        // Extract dimensions
        int64_t rows = taco_tensor.getDimension(0);
        int64_t cols = taco_tensor.getDimension(1);

        // Prepare vectors to hold indices and values
        std::vector<int64_t> row_indices;
        std::vector<int64_t> col_indices;
        std::vector<T> values;

        // Iterate over non-zero elements
        for (auto& value : taco_tensor) {
            auto coords = value.first; // Extract coordinates
            row_indices.push_back(coords[0]);
            col_indices.push_back(coords[1]);
            values.push_back(value.second);
        }

        // Create LibTorch tensors
        auto options = torch::TensorOptions().dtype(torch::CppTypeToScalarType<T>()).device(torch::kCPU);

        // Indices
        auto row_tensor = torch::from_blob(row_indices.data(), {static_cast<int64_t>(row_indices.size())}, options.dtype(torch::kLong)).clone();
        auto col_tensor = torch::from_blob(col_indices.data(), {static_cast<int64_t>(col_indices.size())}, options.dtype(torch::kLong)).clone();
        auto indices = torch::stack({row_tensor, col_tensor});

        // Values
        auto value_tensor = torch::from_blob(values.data(), {static_cast<int64_t>(values.size())}, options).clone();

        // Create sparse COO tensor
        return torch::sparse_coo_tensor(indices, value_tensor, {rows, cols}, options);
    }

    template <typename T>
    taco::Tensor<T> ConvertLibTorchToTaco3D(const torch::Tensor& libtorch_tensor) {
        // Ensure the LibTorch tensor is in COO format and on CPU
        auto coo_tensor = libtorch_tensor.coalesce().cpu();
        auto indices = coo_tensor.indices();
        auto values = coo_tensor.values();

        // Initialize TACO tensor
        int dim1 = libtorch_tensor.size(0);
        int dim2 = libtorch_tensor.size(1);
        int dim3 = libtorch_tensor.size(2);
        auto taco_tensor = taco::Tensor<T>({dim1, dim2, dim3}, taco::Format({taco::Sparse, taco::Sparse, taco::Sparse}));

        // Insert non-zero elements into TACO tensor
        for (int64_t i = 0; i < values.size(0); ++i) {
            int idx1 = indices[0][i].item<int>();
            int idx2 = indices[1][i].item<int>();
            int idx3 = indices[2][i].item<int>();
            double value = values[i].item<double>();
            taco_tensor.insert({idx1, idx2, idx3}, value);
        }

        // Pack the tensor
        taco_tensor.pack();
        return taco_tensor;
    }

    template <typename T>
    torch::Tensor ConvertTacoToLibTorch3D(const taco::Tensor<T>& taco_tensor) {
        // Extract dimensions
        int64_t dim1 = taco_tensor.getDimension(0);
        int64_t dim2 = taco_tensor.getDimension(1);
        int64_t dim3 = taco_tensor.getDimension(2);

        // Prepare vectors to hold indices and values
        std::vector<int64_t> idx1_indices;
        std::vector<int64_t> idx2_indices;
        std::vector<int64_t> idx3_indices;
        std::vector<T> values;

        // Iterate over non-zero elements
        for (auto& value : taco_tensor) {
            auto coords = value.first; // Extract coordinates
            idx1_indices.push_back(coords[0]);
            idx2_indices.push_back(coords[1]);
            idx3_indices.push_back(coords[2]);
            values.push_back(value.second); // Use template type T for values
        }

        // Create LibTorch tensors for indices
        auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
        auto idx1_tensor = torch::from_blob(idx1_indices.data(), {static_cast<int64_t>(idx1_indices.size())}, options).clone();
        auto idx2_tensor = torch::from_blob(idx2_indices.data(), {static_cast<int64_t>(idx2_indices.size())}, options).clone();
        auto idx3_tensor = torch::from_blob(idx3_indices.data(), {static_cast<int64_t>(idx3_indices.size())}, options).clone();
        auto indices = torch::stack({idx1_tensor, idx2_tensor, idx3_tensor});

        // Create LibTorch tensor for values
        auto value_options = torch::TensorOptions().dtype(torch::CppTypeToScalarType<T>()).device(torch::kCPU);
        auto value_tensor = torch::from_blob(values.data(), {static_cast<int64_t>(values.size())}, value_options).clone();

        // Create sparse COO tensor
        return torch::sparse_coo_tensor(indices, value_tensor, {dim1, dim2, dim3});
    }

    template <typename T>
    void ConvertLibTorchToTaco4D(const torch::Tensor& libtorch_tensor, taco::Tensor<T>& taco_tensor) {
        // Ensure the LibTorch tensor is in COO format and on CPU
        auto coo_tensor = libtorch_tensor.coalesce().cpu();
        auto indices = coo_tensor.indices();
        auto values = coo_tensor.values();

        // Check if the LibTorch tensor type matches the template type T
        if (values.scalar_type() != torch::CppTypeToScalarType<T>()) {
            throw std::runtime_error("Mismatch between LibTorch tensor type and TACO tensor type.");
        }

        // Initialize TACO tensor with appropriate dimensions
        int64_t dim1 = libtorch_tensor.size(0);
        int64_t dim2 = libtorch_tensor.size(1);
        int64_t dim3 = libtorch_tensor.size(2);
        int64_t dim4 = libtorch_tensor.size(3);
        taco_tensor = taco::Tensor<T>({dim1, dim2, dim3, dim4}, taco::Format({taco::Sparse, taco::Sparse, taco::Sparse, taco::Sparse}));

        // Insert non-zero elements into TACO tensor
        for (int64_t i = 0; i < values.size(0); ++i) {
            int64_t idx1 = indices[0][i].item<int64_t>();
            int64_t idx2 = indices[1][i].item<int64_t>();
            int64_t idx3 = indices[2][i].item<int64_t>();
            int64_t idx4 = indices[3][i].item<int64_t>();
            T value = values[i].item<T>(); // Use template type T for values
            taco_tensor.insert({idx1, idx2, idx3, idx4}, value);
        }

        // Pack the tensor to finalize the storage format
        taco_tensor.pack();
    }

    template <typename T>
    torch::Tensor ConvertTacoToLibTorch4D(const taco::Tensor<T>& taco_tensor) {
        // Extract dimensions
        int64_t dim1 = taco_tensor.getDimension(0);
        int64_t dim2 = taco_tensor.getDimension(1);
        int64_t dim3 = taco_tensor.getDimension(2);
        int64_t dim4 = taco_tensor.getDimension(3);

        // Prepare vectors to hold indices and values
        std::vector<int64_t> idx1_indices;
        std::vector<int64_t> idx2_indices;
        std::vector<int64_t> idx3_indices;
        std::vector<int64_t> idx4_indices;
        std::vector<T> values;

        // Iterate over non-zero elements
        for (auto& value : taco_tensor) {
            auto coords = value.first; // Extract coordinates
            idx1_indices.push_back(coords[0]);
            idx2_indices.push_back(coords[1]);
            idx3_indices.push_back(coords[2]);
            idx4_indices.push_back(coords[3]);
            values.push_back(value.second); // Use template type T for values
        }

        // Create LibTorch tensors for indices
        auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
        auto idx1_tensor = torch::from_blob(idx1_indices.data(), {static_cast<int64_t>(idx1_indices.size())}, options).clone();
        auto idx2_tensor = torch::from_blob(idx2_indices.data(), {static_cast<int64_t>(idx2_indices.size())}, options).clone();
        auto idx3_tensor = torch::from_blob(idx3_indices.data(), {static_cast<int64_t>(idx3_indices.size())}, options).clone();
        auto idx4_tensor = torch::from_blob(idx4_indices.data(), {static_cast<int64_t>(idx4_indices.size())}, options).clone();
        auto indices = torch::stack({idx1_tensor, idx2_tensor, idx3_tensor, idx4_tensor});

        // Create LibTorch tensor for values
        auto value_options = torch::TensorOptions().dtype(torch::CppTypeToScalarType<T>()).device(torch::kCPU);
        auto value_tensor = torch::from_blob(values.data(), {static_cast<int64_t>(values.size())}, value_options).clone();

        // Create sparse COO tensor in LibTorch
        return torch::sparse_coo_tensor(indices, value_tensor, {dim1, dim2, dim3, dim4});
    }

  taco::TensorBase fromLibTorch3D(const torch::Tensor& tensor) {
    // Ensure the input tensor is on the CPU for processing
    taco::TensorBase dt;
    switch (tensor.scalar_type()) {
            case torch::kFloat:
                dt = ConvertLibTorchToTaco3D<float>(tensor);
                break;
            case torch::kDouble:
                dt = ConvertLibTorchToTaco3D<double>(tensor);
                break;
            case torch::kInt:
                dt = ConvertLibTorchToTaco3D<int>(tensor);
                break;
            case torch::kLong:
                dt = ConvertLibTorchToTaco3D<long>(tensor);
                break;
            case torch::kBool:
                dt = ConvertLibTorchToTaco3D<bool>(tensor);
                break;
            default:
                throw std::invalid_argument("Unsupported data type for dual tensor.");
    }
    return dt;

  }


  taco::TensorBase fromLibTorch2D(const torch::Tensor& tensor) {
    // Ensure the input tensor is on the CPU for processing
    taco::TensorBase dt;
    switch (tensor.scalar_type()) {
            case torch::kFloat:
                dt = ConvertLibTorchToTaco2D<float>(tensor);
                break;
            case torch::kDouble:
                dt = ConvertLibTorchToTaco2D<double>(tensor);
                break;
            case torch::kInt:
                dt = ConvertLibTorchToTaco2D<int>(tensor);
                break;
            case torch::kLong:
                dt = ConvertLibTorchToTaco2D<long>(tensor);
                break;
            case torch::kBool:
                dt = ConvertLibTorchToTaco2D<bool>(tensor);
                break;
            default:
                throw std::invalid_argument("Unsupported data type for dual tensor.");
    }
    return dt;

  }




  // Function to generate Chebyshev nodes
  torch::Tensor chebyshev_nodes(int N) {
    torch::Tensor x = torch::cos(torch::arange(0, N, torch::kDouble) * M_PI / (N - 1));
    return x;
  }

  // Function to generate Chebyshev differentiation matrix
  torch::Tensor chebyshev_differentiation_matrix(int N) {
    auto x = chebyshev_nodes(N);
    auto c = torch::ones({N}, torch::kDouble);
    for (int i = 1; i < N; i += 2) {
        c[i] = -1;
    }
    c[0] = 2;
    c[N - 1] = 2;

    auto X = x.unsqueeze(1).repeat({1, N});
    auto dX = X - X.transpose(0, 1);

    auto D = (c.unsqueeze(1) / c.unsqueeze(0)) / (dX + torch::eye(N, torch::kDouble));
    D = D - torch::diag(torch::sum(D, 1));

    return D;
  }

  /**
   * Calculates the Kronecker product of two tensors
  */
  torch::Tensor kron(const torch::Tensor& a, const torch::Tensor& b) 
  {
    // Calculate the size of the resulting tensor
    auto siz1 = torch::tensor({a.size(-2) * b.size(-2), a.size(-1) * b.size(-1)});

    // Perform unsqueeze operations to prepare for the Kronecker product
    auto res = a.unsqueeze(-1).unsqueeze(-3) * b.unsqueeze(-2).unsqueeze(-4);

    // Get the leading batch dimensions
    std::vector<int64_t> siz0(res.sizes().begin(), res.sizes().end() - 4);

    // Reshape the result to the desired size
    siz0.insert(siz0.end(), {siz1[0].item<int64_t>(), siz1[1].item<int64_t>()});
    return res.reshape(siz0);
  }
   torch::Tensor signc(const torch::Tensor& input) 
   {
     // Calculate the magnitude (absolute value) of the complex tensor
     auto magnitude = input.abs();

     //assume that the input is a double
     auto normalized = input*(magnitude.reciprocal());


     return normalized;
   }

   torch::Tensor flip_epsilon(const torch::Tensor& input) 
   {
     //Set tiny numbers to zero.  Libtorch has a tendency to produce very small negative numbers
     //which can cause problems with the QR decomposition
     auto mask = input.abs() < 10*eps & input < 0.0;
     
     return input - mask.to(torch::kDouble) * input;
   }

   torch::Tensor flip_epsilonc(const torch::Tensor& input) 
   {
     //Set tiny numbers to zero.  Libtorch has a tendency to produce very small negative numbers
     //which can cause problems with the QR decomposition
     auto mask = input.abs() < 100*eps & torch::real(input) < 0.0;
     
     return input - mask.to(torch::kDouble) * input;
   }




   torch::Tensor custom_sign(const torch::Tensor& input, double threshold = 1e-6) 
   {
    
    auto x = torch::real(input);
    std::cerr << "Input to custom_sign tensor = " << x << std::endl;
    //assert that the x type is real not complex
    assert(x.is_complex() == false);
    // Create a tensor with the same shape as input, filled with 0
    auto output = torch::zeros_like(x);
    
    // Find indices where the absolute value is greater than the threshold
    auto mask = torch::abs(x) > threshold;
    
    // Apply the sign function to elements above the threshold
    auto sign_tensor = torch::sgn(x);
    
    // Use where to combine the results
    output = torch::where(mask, sign_tensor, output);
    std::cerr << "Output from custom_sign tensor = " << output << std::endl;
    
    return output;
   }




   TensorDual custom_sign(const TensorDual& input, double threshold = 1e-6) 
   {
    
    auto x = TensorDual(torch::real(input.r), 
                        torch::real(input.d));
    assert(x.r.is_complex() == false);
    std::cerr << "Input to custom_sign TensorDual = " << x << std::endl;
    // Create a tensor with the same shape as input, filled with 0
    auto output = TensorDual::zeros_like(x);
    
    // Find indices where the absolute value is greater than the threshold
    auto mask = x.abs() > threshold;
    
    // Apply the sign function to elements above the threshold
    TensorDual sign_tensor = x.sign();
    
    // Use where to combine the results
    output = TensorDual::where(mask, sign_tensor, output);
    std::cerr << "Output from custom_sign TensorDual = " << output << std::endl;
    return output;
   }


   TensorHyperDual custom_sign(const TensorHyperDual& input, double threshold = 1e-6) 
   {
    
    auto x = TensorHyperDual(torch::real(input.r), 
                             torch::real(input.d), 
                             torch::real(input.h));
    std::cerr << "Input to custom_sign TensorHyperDual = " << x << std::endl;
    assert(x.r.is_complex() == false);
    // Create a tensor with the same shape as input, filled with 0
    auto output = TensorHyperDual::zeros_like(x);
    
    // Find indices where the absolute value is greater than the threshold
    auto mask = x.abs() > threshold;
    
    // Apply the sign function to elements above the threshold
    TensorHyperDual sign_tensor = x.sign();
    
    // Use where to combine the results
    output = TensorHyperDual::where(mask, sign_tensor, output);
    std::cerr << "Output from custom_sign TensorHyperDual = " << output << std::endl;
    return output;
   }



   torch::Tensor signcond(const torch::Tensor &a, const torch::Tensor &b) 
   {
     torch::Tensor a_sign, b_sign;
     a_sign = custom_sign(torch::real(a));
     std::cerr << "a_sign = " << a_sign << std::endl;
     b_sign = custom_sign(torch::real(b));
     std::cerr << "b_sign = " << b_sign << std::endl;
     //If the number is very small assume the sign is positive to ensure compatibility with matlab
     auto result = (b_sign >= 0) * ((a_sign >= 0) * a + (a_sign < 0) * -a) +
             (b_sign < 0)* ((a_sign >= 0) *-a + (a_sign < 0) *a);
     return result;
   }


  // Function to unfold a tensor along a specific mode
  torch::Tensor unfold(const torch::Tensor& tensor, int mode) {
    // Get tensor dimensions
    auto sizes = tensor.sizes();
    int I = sizes[0];
    int J = sizes[1];
    int K = sizes[2];

    if (mode == 0) {
        // Unfold along mode-0
        return tensor.permute({0, 1, 2}).reshape({I, J * K});
    } else if (mode == 1) {
        // Unfold along mode-1
        return tensor.permute({1, 0, 2}).reshape({J, I * K});
    } else {
        // Unfold along mode-2
        return tensor.permute({2, 0, 1}).reshape({K, I * J});
    }
  }


    /**
     * @brief Check if two tensors are broadcast-compatible.
     *
     * This function checks if two tensors' shapes can be broadcast together according to PyTorch's broadcasting rules.
     *
     * @param sizes1 The shape of the first tensor.
     * @param sizes2 The shape of the second tensor.
     * @return True if the tensors are broadcast-compatible; otherwise, false.
     */
    bool are_broadcast_compatible(const torch::IntArrayRef& sizes1, const torch::IntArrayRef& sizes2) {
        auto it1 = sizes1.rbegin(); // Reverse iterator for sizes1
        auto it2 = sizes2.rbegin(); // Reverse iterator for sizes2

        // Compare dimensions from the trailing end
        while (it1 != sizes1.rend() && it2 != sizes2.rend()) {
            if (*it1 != *it2 && *it1 != 1 && *it2 != 1) {
                return false; // Not broadcast-compatible
            }
            ++it1;
            ++it2;
        }

        // If we haven't exited early, the tensors are broadcast-compatible
        return true;
    }

  // Function to compute Tucker decomposition (HOSVD)
  void tuckerDecomposition(const torch::Tensor& tensor,
                         torch::Tensor& core,
                         std::vector<torch::Tensor>& factors,
                         std::vector<int64_t> ranks) {
    // Initialize factors
    factors.clear();

    for (int mode = 0; mode < 3; ++mode) {
        // Unfold tensor along mode
        auto unfolded = unfold(tensor, mode);

        // Compute SVD on the unfolded tensor
        auto svd_result = torch::svd(unfolded);
        auto U = std::get<0>(svd_result); // U matrix

        // Truncate U to the desired rank
        auto truncated_U = U.index({"...", torch::indexing::Slice(0, ranks[mode])});
        factors.push_back(truncated_U);
    }

    // Compute the core tensor by contracting original tensor with factor matrices
    auto temp = tensor;
    for (int mode = 0; mode < 3; ++mode) {
        temp = torch::matmul(factors[mode].transpose(0, 1), unfold(temp, mode)).reshape(ranks);
    }
    core = temp; // Core tensor
   }

   TensorDual signcond(const TensorDual &a, const TensorDual &b) 
   {
     std::cerr << "signcond TensorDual" << std::endl;
     torch::Tensor a_sign, b_sign;
     a_sign = custom_sign(torch::real(a.r));
     std::cerr << "a_sign = " << a_sign << std::endl;
     b_sign = custom_sign(torch::real(b.r));
     std::cerr << "b_sign = " << b_sign << std::endl;
     //If the number is very small assume the sign is positive to ensure compatibility with matlab

     auto result = TensorDual::einsum("mi,mi->mi", (b_sign >= 0) , (TensorDual::einsum("mi,mi->mi", (a_sign >= 0) , a))) + 
                   TensorDual::einsum("mi,mi->mi",(a_sign < 0) , -a) +
             TensorDual::einsum("mi,mi->mi",(b_sign < 0), (TensorDual::einsum("mi,mi->mi",(a_sign >= 0) ,-a))) + 
             TensorDual::einsum("mi,mi->mi",(a_sign < 0),a);
     return result;
   }

   TensorHyperDual signcond(const TensorHyperDual &a, const TensorHyperDual &b) 
   {
     torch::Tensor a_sign, b_sign;
     a_sign = custom_sign(torch::real(a.r));
     b_sign = custom_sign(torch::real(b.r));
     //If the number is very small assume the sign is positive to ensure compatibility with matlab

     auto result = TensorHyperDual::einsum("mi,mi->mi", (b_sign >= 0) , (TensorHyperDual::einsum("mi,mi->mi", (a_sign >= 0) , a))) + 
                   TensorHyperDual::einsum("mi,mi->mi",(a_sign < 0) , -a) +
             TensorHyperDual::einsum("mi,mi->mi",(b_sign < 0), (TensorHyperDual::einsum("mi,mi->mi",(a_sign >= 0) ,-a))) + 
             TensorHyperDual::einsum("mi,mi->mi",(a_sign < 0),a);
     return result;
   }





   /**
    * There is a problem with the pow function in at:: namespace
    * for exponents with double precision. This function is a workaround
   */
   torch::Tensor spow(const torch::Tensor &x, const torch::Tensor &y) 
   {
     // Calculate the power of a base to an exponent in batch mode
     torch::Tensor result;
     result = torch::empty_like(x);
     //Check to see if y is a scalar
     if (y.dim() == 0 && x.dim() > 0) 
     {
       int bs = x.size(0); 
       for (int i = 0; i < bs; ++i) 
       {
         result[i] = torch::pow(x[i], y);
       }
     } else if (x.dim() > 0 && y.dim() > 0 &&  x.size(0) != y.size(0) )
     {
       int bs = x.size(0);
       for (int i = 0; i < bs; ++i) 
       {
         result[i] = torch::pow(x[i], y[0]);
       }
     } else if (x.dim() > 0 && y.dim() > 0 && x.size(0) == y.size(0)) 
     {
       int bs = x.size(0);
       for (int i = 0; i < bs; ++i) 
       {
         result[i] = torch::pow(x[i], y[i]);
       }
     } else 
     {
        result = at::pow(x, y);
     }
     return result;
   }

   /**
    * There is a problem with the pow function in at:: namespace
    * for exponents with double precision. This function is a workaround
   */
   torch::Tensor bpow(const torch::Tensor &x, const torch::Tensor &y) {
     // Calculate the power of a base to an exponent in batch mode
     auto temp = y*torch::log(x);
     return torch::exp(temp);
   }

   TensorDual bpow(const TensorDual &x, const TensorDual &y) {
     // Calculate the power of a base to an exponent in batch mode
     auto temp = y*x.log();
     return temp.exp();
   }


   TensorDual bpow(const TensorDual &x, const double &y) {
     // Calculate the power of a base to an exponent in batch mode
     auto temp = y*x.log();
     return temp.exp();
   }

   torch::Tensor bpow(const torch::Tensor &x, const double &y) {
     // Calculate the power of a base to an exponent in batch mode
     torch::Tensor result;
     int bs = x.size(0);
     result = torch::empty_like(x);
     for (int i = 0; i < bs; ++i) {
       result[i] = torch::pow(x[i], y);
     }
     return result;
   }



   void print_complex(const torch::Tensor &t) {
    std::cout << "Real part : " << std::fixed << std::setprecision(16) << torch::real(t) << std::endl;
    std::cout << "Imaginary part : " << std::fixed << std::setprecision(16) << torch::imag(t) << std::endl;
   }


    void print_4d_tensor(const torch::Tensor &t) {
        // Print the tensor with double precision
        for (int64_t i = 0; i < t.size(0); ++i) {
            for (int64_t j = 0; j < t.size(1); ++j) {
                for (int64_t k = 0; k < t.size(2); ++k) {
                    for (int64_t l = 0; l < t.size(3); ++l) {
                        if (t.is_complex()) {
                            std::cout << std::fixed << std::setprecision(16) << torch::real(t[i][j][k][l]).item<double>();
                            std::cout << " + ";
                            std::cout << std::fixed << std::setprecision(16) << torch::imag(t[i][j][k][l]).item<double>();
                            std::cout << "i";
                        } else {
                            // Extract the element as a double and set precision
                            std::cout << std::fixed << std::setprecision(16) << t[i][j][k][l].item<double>();
                        }
                        if (l != t.size(3) - 1) std::cout << ", ";
                    }
                    std::cout << std::endl;
                    if (j != t.size(2) - 1) std::cout << ", ";
                }
                std::cout << std::fixed << std::setprecision(16) << t[i][j].item<double>();
                if (j != t.size(1) - 1) std::cout << ", ";
            }
            std::cout << std::endl;
            std::cout << t.options() << std::endl;
        }
    }


    void print_3d_tensor(const torch::Tensor &t) {
        // Print the tensor with double precision
        for (int64_t i = 0; i < t.size(0); ++i) {
            for (int64_t j = 0; j < t.size(1); ++j) {
                for (int64_t k = 0; k < t.size(2); ++k) {
                    if (t.is_complex()) {
                        std::cout << std::fixed << std::setprecision(16) << torch::real(t[i][j][k]).item<double>();
                        std::cout << " + ";
                        std::cout << std::fixed << std::setprecision(16) << torch::imag(t[i][j][k]).item<double>();
                        std::cout << "i";
                    } else {
                        // Extract the element as a double and set precision
                        std::cout << std::fixed << std::setprecision(16) << t[i][j][k].item<double>();
                    }
                    if (k != t.size(2) - 1) std::cout << ", ";
                }
                std::cout << std::endl;
            }
            std::cout << std::fixed << std::setprecision(16) << t[i].item<double>();
            if (i != t.size(0) - 1) std::cout << ", ";
        }
        std::cout << std::endl;
        std::cout << t.options() << std::endl;
    }

    void print_matrix(const torch::Tensor &t) {
        // Print the tensor with double precision
        for (int64_t i = 0; i < t.size(0); ++i) {
            for (int64_t j = 0; j < t.size(1); ++j) {
                if (t.is_complex()) {
                    std::cout << std::fixed << std::setprecision(16) << torch::real(t[i][j]).item<double>();
                    std::cout << " + ";
                    std::cout << std::fixed << std::setprecision(16) << torch::imag(t[i][j]).item<double>();
                    std::cout << "i";
                } else {
                    // Extract the element as a double and set precision
                    std::cout << std::fixed << std::setprecision(16) << t[i][j].item<double>();
                }
                if (j != t.size(1) - 1) std::cout << ", ";
            }
            std::cout << std::endl;
        }
        std::cout << t.options() << std::endl;
    }

    void print_matrix(const torch::Tensor &t, int rows) {
        // Print the tensor with double precision
        for (int64_t i = 0; i < rows; ++i) {
            for (int64_t j = 0; j < t.size(1); ++j) {
                if (t.is_complex()) {
                    std::cout << std::fixed << std::setprecision(16) << torch::real(t[i][j]).item<double>();
                    std::cout << " + ";
                    std::cout << std::fixed << std::setprecision(16) << torch::imag(t[i][j]).item<double>();
                } else {
                    // Extract the element as a double and set precision
                    std::cout << std::fixed << std::setprecision(16) << t[i][j].item<double>();
                }
                if (j != t.size(1) - 1) std::cout << ", ";
            }
            std::cout << std::endl;
        }
        std::cout << t.options() << std::endl;
    }



    void print_vector(const torch::Tensor &t) {

        // Print the tensor with double precision
        for (int64_t i = 0; i < t.size(0); ++i) {
                // Extract the element as a double and set precision
                if (t.is_complex()) {
                    std::cout << std::scientific << std::setprecision(16) << torch::real(t[i]).item<double>();
                    std::cout << " + ";
                    std::cout << std::scientific << std::setprecision(16) << torch::imag(t[i]).item<double>();
                } else {
                    std::cout << std::scientific << std::setprecision(16) << t[i].item<double>();
                    std::cout << ", ";
                }
            std::cout << std::endl;
        }
        std::cout << t.options() << std::endl;

    }


    void print_vector(const torch::Tensor &t, int rows) 
    {
        //Check the tensor type

        // Print the tensor with double precision
        for (int64_t i = 0; i < rows; ++i) 
        {
                // Extract the element as a double and set precision
                if (t.is_complex()) 
                {
                    std::cout << std::scientific << std::setprecision(16) << torch::real(t[i]).item<double>();
                    std::cout << " + ";
                    std::cout << std::scientific << std::setprecision(16) << torch::imag(t[i]).item<double>();
                } else 
                {
                    std::cout << std::scientific << std::setprecision(16) << t[i].item<double>();
                    std::cout << ", ";
                }
            std::cout << std::endl;
        }
        std::cout << t.options() << std::endl;

   }

   void print_tensor(const torch::Tensor &t) 
   {
      
      std::cout << "Tensor shape =" << t.sizes() << std::endl;
      print_vector(t.flatten());
   }

   void print_dual(const TensorDual &t) 
   {
      std::cout << "Tensor shape =" << t.r.sizes() << std::endl;
      std::cout << "Tensor device =" << t.r.device() << std::endl;
      print_vector(t.r.flatten());
      std::cout << "Tensor shape =" << t.d.sizes() << std::endl;
      std::cout << "Tensor device =" << t.d.device() << std::endl;
      print_vector(t.d.flatten());
   }

   void print_dual(const TensorMatDual &t) 
   {
      std::cout << "Tensor shape =" << t.r.sizes() << std::endl;
      std::cout << "Tensor device =" << t.r.device() << std::endl;
      print_vector(t.r.flatten());
      std::cout << "Tensor shape =" << t.d.sizes() << std::endl;
      std::cout << "Tensor device =" << t.d.device() << std::endl;
      print_vector(t.d.flatten());
   }
   
   

torch::Tensor expand_to_match(const torch::Tensor& x, torch::Tensor y) 
{
    // Calculate the difference in dimensions
    int64_t dim_diff = x.dim() - y.dim();

    // Add singleton dimensions to y to match the number of dimensions in x
    for (int64_t i = 0; i < dim_diff; ++i) {
        y = y.unsqueeze(-1);
    }

    // Optionally, you can explicitly specify how to expand y to match x's shape fully,
    // but usually, this is not necessary for broadcasting purposes.
    // y = y.expand_as(x);

    return y;
}


// Function to compute the Jacobian matrix
torch::Tensor compute_jacobian(torch::Tensor& output, torch::Tensor& input) {
    auto jacobian = torch::zeros({output.numel(), input.numel()});

    for (int64_t i = 0; i < output.numel(); ++i) {
        auto grad_output = torch::zeros_like(output);
        grad_output.view(-1)[i] = 1.0;

        // Zero the gradients before backward pass
        if (input.grad().defined()) {
            input.grad().zero_();
        }

        // Compute the gradients
        output.backward(grad_output, true);

        // Copy the gradient to the jacobian matrix
        jacobian.slice(0, i, i+1) = input.grad().view({1, -1});
    }

    return jacobian;
}


torch::Tensor compute_batch_jacobian(torch::Tensor& output, torch::Tensor& input) 
{
    auto batch_size = output.size(0);
    auto output_size = output.numel() / batch_size;
    auto input_size = input.numel() / batch_size;

    auto jacobian = torch::zeros({batch_size, output_size, input_size}).to(input.dtype()).to(input.device());

    for (int64_t b = 0; b < batch_size; ++b) {
        for (int64_t i = 0; i < output_size; ++i) {
            auto grad_output = torch::zeros_like(output);
            grad_output[b].view(-1)[i] = 1.0;

            // Zero the gradients before the backward pass
            if (input.grad().defined()) {
                input.grad().zero_();
            }

            // Compute the gradients
            output.backward(grad_output, true);

            // Copy the gradient to the jacobian matrix
            jacobian[b].slice(0, i, i+1) = input.grad().view({batch_size, -1})[b].view({1, -1});
        }
    }

    return jacobian;

}

// Function to compute Hessians for a batch of inputs
torch::Tensor compute_batch_hessian(
    const torch::Tensor& inputs,  // [M, N, D]
    const std::function<torch::Tensor(const torch::Tensor&)>& func) {  // Function f
    int64_t M = inputs.size(0);
    int64_t N = inputs.size(1);
    int64_t D = inputs.size(2);
    int64_t n = func(inputs[0][0]).size(0); // Output dimension of f
    
    torch::Tensor hessians = torch::zeros({M, n, D, D}, inputs.options());

    for (int64_t m = 0; m < M; ++m) {
        for (int64_t i = 0; i < n; ++i) {
            // Create a scalar output for higher-order derivatives
            auto output = func(inputs[m]).select(0, i);  // Select i-th output of f
            
            for (int64_t d1 = 0; d1 < D; ++d1) {
                auto grad = torch::autograd::grad({output}, {inputs[m]}, /*grad_outputs=*/{torch::ones_like(output)},
                                                  /*retain_graph=*/true, /*create_graph=*/true)[0];
                
                for (int64_t d2 = 0; d2 < D; ++d2) {
                    auto second_grad = torch::autograd::grad({grad[d1]}, {inputs[m]}, /*grad_outputs=*/{torch::ones({1})},
                                                             /*retain_graph=*/true, /*create_graph=*/false)[0];
                    
                    hessians[m][i][d1][d2] = second_grad[d2];
                }
            }
        }
    }

    return hessians;  // [M, n, D, D]
}

torch::Tensor compute_batch_hessian(torch::Tensor& output, torch::Tensor& input) 
{
    auto batch_size = output.size(0);
    auto output_size = output.numel() / batch_size;
    auto input_size = input.numel() / batch_size;

    auto hessian = torch::zeros({batch_size, output_size, input_size, input_size}).to(input.dtype()).to(input.device());

    for (int64_t b = 0; b < batch_size; ++b) {
        for (int64_t i = 0; i < output_size; ++i) {
            auto grad_output = torch::zeros_like(output);
            grad_output[b].view(-1)[i] = 1.0;

            // Zero the gradients before the backward pass
            if (input.grad().defined()) {
                input.grad().zero_();
            }

            // Compute the gradients
            output.backward(grad_output, true);

            // Copy the gradient to the jacobian matrix
            hessian[b].slice(0, i, i+1) = input.grad().view({batch_size, -1})[b].view({1, -1});
        }
    }
    return hessian;
}

/**
 * Compute a 2d jacobian where the input is a 3D batched matrix
 */
torch::Tensor compute_batch_jacobian2d(torch::Tensor& output, torch::Tensor& input) 
{
    auto batch_size = output.size(0);
    auto dim1 = input.size(1);
    auto dim2 = input.size(2);
    auto output_size = output.size(1);
    assert(output.dim() == 2 && "Output must be a 2D tensor");
    assert(input.dim() == 3 && "Input must be a 3D tensor");

    auto jacobian = torch::zeros({batch_size, output_size, dim1, dim2}).to(input.dtype()).to(input.device());

    for (int64_t b = 0; b < batch_size; ++b) {
        for (int64_t i = 0; i < output_size; ++i) {
            auto grad_output = torch::zeros_like(output);
            grad_output.index_put_({b, i},1.0);
            // Zero the gradients before the backward pass
            if (input.grad().defined()) 
            {
              input.grad().zero_();
            }
            // Compute the gradients
            output.backward(grad_output, true);

            for ( int j=0; j < dim1; j++)
            {
                for ( int k=0; k < dim2; k++)
                {
                  jacobian.index_put_( {b, i, j, k}, input.grad().index({b, j,k}) );
                }
            }
        }
    }
    return jacobian;
}

torch::Tensor compute_batch_hessian2d(torch::Tensor& output, torch::Tensor& input) 
{
    auto batch_size = output.size(0);
    auto dim1 = input.size(1);
    auto dim2 = input.size(2);
    auto output_size = output.size(1);
    assert(output.dim() == 2 && "Output must be a 2D tensor");
    assert(input.dim() == 3 && "Input must be a 3D tensor");

    auto hessian = torch::zeros({batch_size, output_size, dim1, dim2, dim1, dim2}).to(input.dtype()).to(input.device());

    for (int64_t b = 0; b < batch_size; ++b) {
        for (int64_t i = 0; i < output_size; ++i) {
            auto grad_output = torch::zeros_like(output);
            grad_output.index_put_({b, i},1.0);
            // Zero the gradients before the backward pass
            if (input.grad().defined()) 
            {
              input.grad().zero_();
            }
            // Compute the gradients
            output.backward(grad_output, true);

            for ( int j=0; j < dim1; j++)
            {
                for ( int k=0; k < dim2; k++)
                {
                  for ( int l=0; l < dim1; l++)
                  {
                    for ( int m=0; m < dim2; m++)
                    {
                      hessian.index_put_( {b, i, j, k, l, m}, input.grad().index({b, l,m}) );
                    }
                  }
                }
            }
        }
    }
    return hessian;
}

/**
 * Compute a 3d jacobian where the input is a batched matrix
 */
torch::Tensor compute_batch_jacobian3d(const torch::Tensor& output, const torch::Tensor& input) 
{
    auto batch_size = output.size(0);
    auto dim1 = input.size(1);
    auto dim2 = input.size(2);
    auto output_size1 = output.size(1);
    auto output_size2 = output.size(2);
    assert(output.dim() == 3 && "Output must be a 3D tensor");
    assert(input.dim() == 3 && "Input must be a 3D tensor");

    auto jacobian = torch::zeros({batch_size, output_size1, output_size2, dim1, dim2}).to(input.dtype()).to(input.device());

    for (int64_t b = 0; b < batch_size; ++b) {
        for (int64_t i = 0; i < output_size1; ++i) {
            for ( int j=0; j < output_size2; ++j)
            {
            auto grad_output = torch::zeros_like(output);
            grad_output.index_put_({b, i,j},1.0);
            // Zero the gradients before the backward pass
            if (input.grad().defined()) 
            {
              input.grad().zero_();
            }
            // Compute the gradients
            output.backward(grad_output, true);

            for ( int k=0; k < dim1; k++)
            {
                for ( int l=0; l < dim2; l++)
                {
                  jacobian.index_put_( {b, i, j, k,l}, input.grad().index({b, k,l}) );
                }
            }
            }
        }
    }

    return jacobian;

}

torch::Tensor compute_batch_hessian3d(const torch::Tensor& output, const torch::Tensor& input) 
{
    auto batch_size = output.size(0);
    auto dim1 = input.size(1);
    auto dim2 = input.size(2);
    auto output_size1 = output.size(1);
    auto output_size2 = output.size(2);
    assert(output.dim() == 3 && "Output must be a 3D tensor");
    assert(input.dim() == 3 && "Input must be a 3D tensor");

    auto hessian = torch::zeros({batch_size, output_size1, output_size2, dim1, dim2, dim1, dim2}).to(input.dtype()).to(input.device());

    for (int64_t b = 0; b < batch_size; ++b) {
        for (int64_t i = 0; i < output_size1; ++i) {
            for ( int j=0; j < output_size2; ++j)
            {
            auto grad_output = torch::zeros_like(output);
            grad_output.index_put_({b, i,j},1.0);
            // Zero the gradients before the backward pass
            if (input.grad().defined()) 
            {
              input.grad().zero_();
            }
            // Compute the gradients
            output.backward(grad_output, true);

            for ( int k=0; k < dim1; k++)
            {
                for ( int l=0; l < dim2; l++)
                {
                  for ( int m=0; m < dim1; m++)
                  {
                    for ( int n=0; n < dim2; n++)
                    {
                      hessian.index_put_( {b, i, j, k,l, m, n}, input.grad().index({b, m,n}) );
                    }
                  }
                }
            }
            }
        }
    }

    return hessian;

}


/**
 * Compute a 2d jacobian where the input is a 3D batched matrix
 */
torch::Tensor compute_batch_jacobian2d2d(torch::Tensor& output, torch::Tensor& input) 
{
    auto batch_size = output.size(0);
    auto dim1 = input.size(1);
    auto output_size = output.size(1);
    assert(output.dim() == 2 && "Output must be a 2D tensor");
    assert(input.dim() == 2 && "Input must be a 2D tensor");

    auto jacobian = torch::zeros({batch_size, output_size, dim1}).to(input.dtype()).to(input.device());

    for (int64_t b = 0; b < batch_size; ++b) {
        for (int64_t i = 0; i < output_size; ++i) {
            auto grad_output = torch::zeros_like(output);
            grad_output.index_put_({b, i},1.0);
            // Zero the gradients before the backward pass
            if (input.grad().defined()) 
            {
              input.grad().zero_();
            }
            // Compute the gradients
            output.backward(grad_output, true);

            for ( int j=0; j < dim1; j++)
            {
                  jacobian.index_put_( {b, i, j}, input.grad().index({b, j}) );
            }
        }
    }

    return jacobian;

}

class ComplexTensorSparse2D {
private:
    taco::Tensor<double> real; // Real part
    taco::Tensor<double> imag; // Imaginary part

public:
    // Constructor from libtorch complex tensor
    ComplexTensorSparse2D(torch::Tensor tens) {
        //Make sure the tensor is complex
        assert(tens.is_complex() && "Input tensor must be complex");
        //Make sure the tensor is 2D
        assert(tens.dim() == 2 && "Input tensor must be 2D");
        //Make sure the tensor is sparse
        assert(tens.is_sparse() && "Input tensor must be sparse");
        // Initialize real and imaginary parts
        real = fromLibTorch2D(torch::real(tens));
        imag = fromLibTorch2D(torch::imag(tens));

    }

    ComplexTensorSparse2D(taco::Tensor<double> r, taco::Tensor<double> i) : real(r), imag(i) {
        assert(r.getOrder() == 2 && "Real part must be 2D");
        assert(i.getOrder() == 2 && "Imaginary part must be 2D");
    }


    // Pack tensors
    void pack() {
        real.pack();
        imag.pack();
    }


    taco::Tensor<double>& getReal() { return real; }
    taco::Tensor<double>& getImag() { return imag; }
    
    /**
     * sqrt function
     */
    ComplexTensorSparse2D sqrt() const {
        // Define TACO index variables
        taco::IndexVar i("i"), j("j");
        taco::Tensor<double> r("r", {real.getDimension(0), real.getDimension(1)});
        r(i, j) = taco::sqrt(taco::square(real(i, j)) + taco::square(imag(i, j)));
        taco::Tensor<double> theta("theta", {real.getDimension(0), real.getDimension(1)});
        theta(i, j) = taco::atan2(imag(i, j), real(i, j));
        // Compute the real and imaginary parts of the square root
        taco::Tensor<double> rn("rn", {real.getDimension(0), real.getDimension(1)});
        rn(i, j) = r(i, j) * taco::cos(theta(i, j) / 2);

        taco::Tensor<double> in("in", {imag.getDimension(0), imag.getDimension(1)});
        in(i, j) = r(i, j) * taco::sin(theta(i, j) / 2);
        rn.pack();
        in.pack();
        return ComplexTensorSparse2D(rn, in);
    }

    ComplexTensorSparse2D operator+(const ComplexTensorSparse2D& other) const {
        // Define TACO index variables
        taco::IndexVar i("i"), j("j");

        // Create tensors for the result
        taco::Tensor<double> resultReal("resultReal", {real.getDimension(0), real.getDimension(1)});
        taco::Tensor<double> resultImag("resultImag", {imag.getDimension(0), imag.getDimension(1)});

        // Define addition expressions
        resultReal(i, j) = real(i, j) + other.real(i, j);
        resultImag(i, j) = imag(i, j) + other.imag(i, j);

        // Pack tensors to finalize
        resultReal.pack();
        resultImag.pack();

        // Return the new ComplexTensorSparse2D object
        return ComplexTensorSparse2D(resultReal, resultImag);
    }


    ComplexTensorSparse2D operator-(const ComplexTensorSparse2D& other) const {
        // Define TACO index variables
        taco::IndexVar i("i"), j("j");

        // Create tensors for the result
        taco::Tensor<double> resultReal("resultReal", {real.getDimension(0), real.getDimension(1)});
        taco::Tensor<double> resultImag("resultImag", {imag.getDimension(0), imag.getDimension(1)});

        // Define subtraction expressions
        resultReal(i, j) = real(i, j) - other.real(i, j);
        resultImag(i, j) = imag(i, j) - other.imag(i, j);

        // Pack tensors to finalize
        resultReal.pack();
        resultImag.pack();

        // Return the new ComplexTensorSparse2D object
        return ComplexTensorSparse2D(resultReal, resultImag);
    }

    ComplexTensorSparse2D operator*(const ComplexTensorSparse2D& other) const {
        // Define TACO index variables
        taco::IndexVar i("i"), j("j");

        // Create tensors for the result
        taco::Tensor<double> resultReal("resultReal", {real.getDimension(0), real.getDimension(1)});
        taco::Tensor<double> resultImag("resultImag", {imag.getDimension(0), imag.getDimension(1)});

        // Define multiplication expressions
        resultReal(i, j) = real(i, j) * other.real(i, j) - imag(i, j) * other.imag(i, j);
        resultImag(i, j) = real(i, j) * other.imag(i, j) + imag(i, j) * other.real(i, j);

        // Pack tensors to finalize
        resultReal.pack();
        resultImag.pack();

        // Return the new ComplexTensorSparse2D object
        return ComplexTensorSparse2D(resultReal, resultImag);
    }

    ComplexTensorSparse2D operator/(const ComplexTensorSparse2D& other) const {
        // Define TACO index variables
        taco::IndexVar i("i"), j("j");

        // Create tensors for the result
        taco::Tensor<double> resultReal("resultReal", {real.getDimension(0), real.getDimension(1)});
        taco::Tensor<double> resultImag("resultImag", {imag.getDimension(0), imag.getDimension(1)});

        // Define division expressions
        auto denom = other.real(i, j) * other.real(i, j) + other.imag(i, j) * other.imag(i, j);
        resultReal(i, j) = (real(i, j) * other.real(i, j) + imag(i, j) * other.imag(i, j)) / denom;
        resultImag(i, j) = (imag(i, j) * other.real(i, j) - real(i, j) * other.imag(i, j)) / denom;

        // Pack tensors to finalize
        resultReal.pack();
        resultImag.pack();

        // Return the new ComplexTensorSparse2D object
        return ComplexTensorSparse2D(resultReal, resultImag);
    }

};



class ComplexTensorSparse3D {
private:
    taco::Tensor<double> real; // Real part
    taco::Tensor<double> imag; // Imaginary part

public:
    // Constructor from libtorch complex tensor
    ComplexTensorSparse3D(torch::Tensor tens) {
        //Make sure the tensor is complex
        assert(tens.is_complex() && "Input tensor must be complex");
        //Make sure the tensor is 2D
        assert(tens.dim() == 3 && "Input tensor must be 3D");
        //Make sure the tensor is sparse
        assert(tens.is_sparse() && "Input tensor must be sparse");
        // Initialize real and imaginary parts
        real = fromLibTorch2D(torch::real(tens));
        imag = fromLibTorch2D(torch::imag(tens));

    }

    ComplexTensorSparse3D(taco::Tensor<double> r, taco::Tensor<double> i) : real(r), imag(i) {
        assert(r.getOrder() == 3 && "Real part must be 3D");
        assert(i.getOrder() == 3 && "Imaginary part must be 3D");
    }


    // Pack tensors
    void pack() {
        real.pack();
        imag.pack();
    }


    
    /**
     * sqrt function
     */
    ComplexTensorSparse3D sqrt() const {
        // Define TACO index variables for the 3D tensor
        taco::IndexVar i("i"), j("j"), k("k");

        // Create tensors for the magnitude (r) and angle (theta)
        taco::Tensor<double> r("r", {real.getDimension(0), real.getDimension(1), real.getDimension(2)});
        taco::Tensor<double> theta("theta", {real.getDimension(0), real.getDimension(1), real.getDimension(2)});

        // Define expressions for r and theta
        r(i, j, k) = taco::sqrt(taco::square(real(i, j, k)) + taco::square(imag(i, j, k)));
        theta(i, j, k) = taco::atan2(imag(i, j, k), real(i, j, k));

        // Create tensors for the resulting real (rn) and imaginary (in) parts
        taco::Tensor<double> rn("rn", {real.getDimension(0), real.getDimension(1), real.getDimension(2)});
        taco::Tensor<double> in("in", {imag.getDimension(0), imag.getDimension(1), imag.getDimension(2)});

        // Define expressions for the real and imaginary parts of the square root
        rn(i, j, k) = r(i, j, k) * taco::cos(theta(i, j, k) / 2);
        in(i, j, k) = r(i, j, k) * taco::sin(theta(i, j, k) / 2);

        // Pack tensors to finalize computations
        r.pack();
        theta.pack();
        rn.pack();
        in.pack();

        // Return the result as a new ComplexTensorSparse3D object
        return ComplexTensorSparse3D(rn, in);
    }


    ComplexTensorSparse3D operator+(const ComplexTensorSparse3D& other) const {
        taco::IndexVar i("i"), j("j"), k("k");
        taco::Tensor<double> resultReal("resultReal", {real.getDimension(0), real.getDimension(1), real.getDimension(2)});
        taco::Tensor<double> resultImag("resultImag", {imag.getDimension(0), imag.getDimension(1), imag.getDimension(2)});
        resultReal(i, j, k) = real(i, j, k) + other.real(i, j, k);
        resultImag(i, j, k) = imag(i, j, k) + other.imag(i, j, k);
        resultReal.pack();
        resultImag.pack();
        return ComplexTensorSparse3D(resultReal, resultImag);
    }

    ComplexTensorSparse3D operator-(const ComplexTensorSparse3D& other) const {
        taco::IndexVar i("i"), j("j"), k("k");
        taco::Tensor<double> resultReal("resultReal", {real.getDimension(0), real.getDimension(1), real.getDimension(2)});
        taco::Tensor<double> resultImag("resultImag", {imag.getDimension(0), imag.getDimension(1), imag.getDimension(2)});
        resultReal(i, j, k) = real(i, j, k) - other.real(i, j, k);
        resultImag(i, j, k) = imag(i, j, k) - other.imag(i, j, k);
        resultReal.pack();
        resultImag.pack();
        return ComplexTensorSparse3D(resultReal, resultImag);
    }

    ComplexTensorSparse3D operator*(const ComplexTensorSparse3D& other) const {
        taco::IndexVar i("i"), j("j"), k("k");
        taco::Tensor<double> resultReal("resultReal", {real.getDimension(0), real.getDimension(1), real.getDimension(2)});
        taco::Tensor<double> resultImag("resultImag", {imag.getDimension(0), imag.getDimension(1), imag.getDimension(2)});
        resultReal(i, j, k) = real(i, j, k) * other.real(i, j, k) - imag(i, j, k) * other.imag(i, j, k);
        resultImag(i, j, k) = real(i, j, k) * other.imag(i, j, k) + imag(i, j, k) * other.real(i, j, k);
        resultReal.pack();
        resultImag.pack();
        return ComplexTensorSparse3D(resultReal, resultImag);
    }

    ComplexTensorSparse3D operator/(const ComplexTensorSparse3D& other) const {
        taco::IndexVar i("i"), j("j"), k("k");
        taco::Tensor<double> resultReal("resultReal", {real.getDimension(0), real.getDimension(1), real.getDimension(2)});
        taco::Tensor<double> resultImag("resultImag", {imag.getDimension(0), imag.getDimension(1), imag.getDimension(2)});
        auto denom = other.real(i, j, k) * other.real(i, j, k) + other.imag(i, j, k) * other.imag(i, j, k);
        resultReal(i, j, k) = (real(i, j, k) * other.real(i, j, k) + imag(i, j, k) * other.imag(i, j, k)) / denom;
        resultImag(i, j, k) = (imag(i, j, k) * other.real(i, j, k) - real(i, j, k) * other.imag(i, j, k)) / denom;
        resultReal.pack();
        resultImag.pack();
        return ComplexTensorSparse3D(resultReal, resultImag);
    }

    // Access real part
    taco::Tensor<double>& getReal() { return real; }

    // Access imaginary part
    taco::Tensor<double>& getImag() { return imag; }
};


class ComplexTensorSparse4D {
private:
    taco::Tensor<double> real; // Real part
    taco::Tensor<double> imag; // Imaginary part

public:
    // Constructor from libtorch complex tensor
    ComplexTensorSparse4D(torch::Tensor tens) {
        //Make sure the tensor is complex
        assert(tens.is_complex() && "Input tensor must be complex");
        //Make sure the tensor is 2D
        assert(tens.dim() == 4 && "Input tensor must be 4D");
        //Make sure the tensor is sparse
        assert(tens.is_sparse() && "Input tensor must be sparse");
        // Initialize real and imaginary parts
        real = fromLibTorch2D(torch::real(tens));
        imag = fromLibTorch2D(torch::imag(tens));

    }

    ComplexTensorSparse4D(taco::Tensor<double> r, taco::Tensor<double> i) : real(r), imag(i) {
        assert(r.getOrder() == 4 && "Real part must be 4D");
        assert(i.getOrder() == 4 && "Imaginary part must be 4D");
    }


    // Pack tensors
    void pack() {
        real.pack();
        imag.pack();
    }


    
    /**
     * sqrt function
     */
    ComplexTensorSparse4D sqrt() const {
        // Define TACO index variables for the 4D tensor
        taco::IndexVar i("i"), j("j"), k("k"), l("l");

        // Create tensors for the magnitude (r) and angle (theta)
        taco::Tensor<double> r("r", {real.getDimension(0), real.getDimension(1), real.getDimension(2), real.getDimension(3)});
        taco::Tensor<double> theta("theta", {real.getDimension(0), real.getDimension(1), real.getDimension(2), real.getDimension(3)});

        // Define expressions for r and theta
        r(i, j, k, l) = taco::sqrt(taco::square(real(i, j, k, l)) + taco::square(imag(i, j, k, l)));
        theta(i, j, k, l) = taco::atan2(imag(i, j, k, l), real(i, j, k, l));

        // Create tensors for the resulting real (rn) and imaginary (in) parts
        taco::Tensor<double> rn("rn", {real.getDimension(0), real.getDimension(1), real.getDimension(2), real.getDimension(3)});
        taco::Tensor<double> in("in", {imag.getDimension(0), imag.getDimension(1), imag.getDimension(2), imag.getDimension(3)});

        // Define expressions for the real and imaginary parts of the square root
        rn(i, j, k, l) = r(i, j, k, l) * taco::cos(theta(i, j, k, l) / 2);
        in(i, j, k, l) = r(i, j, k, l) * taco::sin(theta(i, j, k, l) / 2);

        // Pack tensors to finalize computations
        r.pack();
        theta.pack();
        rn.pack();
        in.pack();

        // Return the result as a new ComplexTensorSparse4D object
        return ComplexTensorSparse4D(rn, in);
    }

    ComplexTensorSparse4D operator+(const ComplexTensorSparse4D& other) const {
        taco::IndexVar i("i"), j("j"), k("k"), l("l");
        taco::Tensor<double> resultReal("resultReal", {real.getDimension(0), real.getDimension(1), real.getDimension(2), real.getDimension(3)});
        taco::Tensor<double> resultImag("resultImag", {imag.getDimension(0), imag.getDimension(1), imag.getDimension(2), imag.getDimension(3)});
        resultReal(i, j, k, l) = real(i, j, k, l) + other.real(i, j, k, l);
        resultImag(i, j, k, l) = imag(i, j, k, l) + other.imag(i, j, k, l);
        resultReal.pack();
        resultImag.pack();
        return ComplexTensorSparse4D(resultReal, resultImag);
    }

    ComplexTensorSparse4D operator-(const ComplexTensorSparse4D& other) const {
        taco::IndexVar i("i"), j("j"), k("k"), l("l");
        taco::Tensor<double> resultReal("resultReal", {real.getDimension(0), real.getDimension(1), real.getDimension(2), real.getDimension(3)});
        taco::Tensor<double> resultImag("resultImag", {imag.getDimension(0), imag.getDimension(1), imag.getDimension(2), imag.getDimension(3)});
        resultReal(i, j, k, l) = real(i, j, k, l) - other.real(i, j, k, l);
        resultImag(i, j, k, l) = imag(i, j, k, l) - other.imag(i, j, k, l);
        resultReal.pack();
        resultImag.pack();
        return ComplexTensorSparse4D(resultReal, resultImag);
    }

    ComplexTensorSparse4D operator*(const ComplexTensorSparse4D& other) const {
        taco::IndexVar i("i"), j("j"), k("k"), l("l");
        taco::Tensor<double> resultReal("resultReal", {real.getDimension(0), real.getDimension(1), real.getDimension(2), real.getDimension(3)});
        taco::Tensor<double> resultImag("resultImag", {imag.getDimension(0), imag.getDimension(1), imag.getDimension(2), imag.getDimension(3)});
        resultReal(i, j, k, l) = real(i, j, k, l) * other.real(i, j, k, l) - imag(i, j, k, l) * other.imag(i, j, k, l);
        resultImag(i, j, k, l) = real(i, j, k, l) * other.imag(i, j, k, l) + imag(i, j, k, l) * other.real(i, j, k, l);
        resultReal.pack();
        resultImag.pack();
        return ComplexTensorSparse4D(resultReal, resultImag);
    }

    ComplexTensorSparse4D operator/(const ComplexTensorSparse4D& other) const {
        taco::IndexVar i("i"), j("j"), k("k"), l("l");
        taco::Tensor<double> resultReal("resultReal", {real.getDimension(0), real.getDimension(1), real.getDimension(2), real.getDimension(3)});
        taco::Tensor<double> resultImag("resultImag", {imag.getDimension(0), imag.getDimension(1), imag.getDimension(2), imag.getDimension(3)});
        auto denom = other.real(i, j, k, l) * other.real(i, j, k, l) + other.imag(i, j, k, l) * other.imag(i, j, k, l);
        resultReal(i, j, k, l) = (real(i, j, k, l) * other.real(i, j, k, l) + imag(i, j, k, l) * other.imag(i, j, k, l)) / denom;
        resultImag(i, j, k, l) = (imag(i, j, k, l) * other.real(i, j, k, l) - real(i, j, k, l) * other.imag(i, j, k, l)) / denom;
        resultReal.pack();
        resultImag.pack();
        return ComplexTensorSparse4D(resultReal, resultImag);
    }

    // Access real part
    taco::Tensor<double>& getReal() { return real; }

    // Access imaginary part
    taco::Tensor<double>& getImag() { return imag; }
};

   
}; // namespace janus

#endif