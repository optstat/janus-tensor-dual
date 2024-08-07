#ifndef JANUS_UTIL_HPP
#define JANUS_UTIL_HPP

#include <torch/torch.h>
#include "tensordual.hpp"
#include <iostream>

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

  torch::Tensor bmax(torch::Tensor &x, torch::Tensor &y)
  {
    return torch::where(x > y, x, y);
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
    // Create a tensor with the same shape as input, filled with 0
    auto output = torch::zeros_like(x);
    
    // Find indices where the absolute value is greater than the threshold
    auto mask = torch::abs(x) > threshold;
    
    // Apply the sign function to elements above the threshold
    auto sign_tensor = torch::sign(x);
    
    // Use where to combine the results
    output = torch::where(mask, sign_tensor, output);
    
    return output;
   }


   TensorDual custom_sign(const TensorDual& input, double threshold = 1e-6) 
   {
    
    auto x = TensorDual(torch::real(input.r), torch::real(input.d));
    // Create a tensor with the same shape as input, filled with 0
    auto output = TensorDual::zeros_like(x);
    
    // Find indices where the absolute value is greater than the threshold
    auto mask = x.abs() > threshold;
    
    // Apply the sign function to elements above the threshold
    TensorDual sign_tensor = x.sign();
    
    // Use where to combine the results
    output = TensorDual::where(mask, sign_tensor, output);
    
    return output;
   }


   torch::Tensor signcond(const torch::Tensor &a, const torch::Tensor &b) 
   {
     torch::Tensor a_sign, b_sign;
     a_sign = custom_sign(torch::real(a));
     b_sign = custom_sign(torch::real(b));
     //If the number is very small assume the sign is positive to ensure compatibility with matlab
     auto result = (b_sign >= 0) * ((a_sign >= 0) * a + (a_sign < 0) * -a) +
             (b_sign < 0)* ((a_sign >= 0) *-a + (a_sign < 0) *a);
     return result;
   }


   TensorDual signcond(const TensorDual &a, const TensorDual &b) 
   {
     torch::Tensor a_sign, b_sign;
     a_sign = custom_sign(torch::real(a.r));
     b_sign = custom_sign(torch::real(b.r));
     //If the number is very small assume the sign is positive to ensure compatibility with matlab

     auto result = TensorDual::einsum("mi,mi->mi", (b_sign >= 0) , (TensorDual::einsum("mi,mi->mi", (a_sign >= 0) , a))) + 
                   TensorDual::einsum("mi,mi->mi",(a_sign < 0) , -a) +
             TensorDual::einsum("mi,mi->mi",(b_sign < 0), (TensorDual::einsum("mi,mi->mi",(a_sign >= 0) ,-a))) + 
             TensorDual::einsum("mi,mi->mi",(a_sign < 0),a);
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


   
}; // namespace janus

#endif