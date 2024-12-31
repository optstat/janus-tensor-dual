#ifndef TENSORDUAL_SPARSE_H_
#define TENSORDUAL_SPARSE_H_
#include <iostream>
#include <vector>
#include "janus_util.hpp"
#include <taco.h>
namespace janus {

//Forward declaration of the TensorMatDual class
class TensorMatDualSparse;

//Forward declaration of the TensorMatHyperDual class
class TensorMatHyperDualSparse;

/**
 * @brief The TensorDual class
 * The TensorDualSparse class encapsulates a real tensor and a dual tensor
 * and provides methods for automatic differentiation using dual numbers
 * while enforcing strict dimensional compatibility between the real and dual tensors
 * by limiting the real part to a vector and the dual part to a matrix.
 * Limiting the dual tensors in this way allows for effient vectorized operations.
 * Note that the goal of this class is not necessarily speed of computation,
 * but as low a memory footprint as possible while still allowing for efficient
 * vectorized operations.  This in turn allows for parallelization of the
 * operations on the dual part of the tensor especially when using GPUs.
 * This Sparse implementation utilizes TACO to store the real and dual tensors
 * while retaining the pytorch front end.  The expectation is that the user will
 * use libtorch to create the tensors in COO format.  The class internally utilizes 
 * TACO to store the tensors in a sparse format and do operations where there is a clear
 * benefit to doing so and to fill in the gaps in the libtorch sparse tensor implementation.
 */
class TensorDualSparse {
public:
    // Members
    taco::TensorBase r;
    taco::TensorBase d;
    torch::Dtype dtype;
    torch::Device device_ = torch::kCPU;

public:
    // Default constructor
    TensorDualSparse() noexcept
        : r(taco::TensorBase({0, 0}, taco::Format({taco::Sparse, taco::Sparse})),
          d(taco::TensorBase({0, 0, 0}, taco::Format({taco::Sparse, taco::Sparse, taco::Sparse}))),
          dtype(torch::kFloat64),
          device_(torch::kCPU) {}

    // Constructor by reference
    TensorDualSparse(const torch::Tensor& r, const torch::Tensor& d) {
        if (r.dim() != 2) {
            throw std::runtime_error("The real tensor `r` must be 2D.");
        }
        if (d.dim() != 3) {
            throw std::runtime_error("The dual tensor `d` must be 3D.");
        }
        if (r.device() != d.device()) {
            throw std::runtime_error("Real and dual tensors must reside on the same device.");
        }
        //Convert to TACO format
        this->r = fromLibTorch2D(r);
        this->d = fromLibTorch3D(d);
        this->dtype = torch::typeMetaToScalarType(r.dtype());
        this->device_ = r.device();
    }

    // Move constructor
    TensorDual(TensorDual&& other) noexcept
        : r(std::move(other.r)), d(std::move(other.d)), dtype(other.dtype), device_(other.device_) {}


    // Copy constructor
    TensorDual(const TensorDual& other)
        : r(other.r.clone()), d(other.d.clone()), dtype(other.dtype), device_(other.device_) {}

    // Move assignment operator
    TensorDual& operator=(TensorDual&& other) noexcept {
        if (this != &other) {
            r = std::move(other.r);
            d = std::move(other.d);
            dtype = other.dtype;
            device_ = other.device_;
        }
        return *this;
    }

    // Deep copy
    TensorDual deepCopy() const {
        return TensorDual(r.clone(), d.clone());
    }

    // Validation
    void validate() const {
        if (r.dim() != 2 || d.dim() != 3) {
            throw std::runtime_error("Validation failed: r must be 2D, and d must be 3D.");
        }
        if (r.device() != d.device()) {
            throw std::runtime_error("Validation failed: Real and dual tensors must reside on the same device.");
        }
        if (r.dtype() != d.dtype()) {
            throw std::runtime_error("Validation failed: Real and dual tensors must have the same dtype.");
        }
    }
    
    /**
     * @brief Method to print the contents of the TensorDual tensor
     */
    friend std::ostream& operator<<(std::ostream& os, const TensorDual& obj) {
        os << "TensorDual Object:" << std::endl;

        // Real part
        if (obj.r.numel() == 0) {
            os << "  Real part (r): [Empty]" << std::endl;
        } else {
            os << "  Real part (r):" << std::endl << obj.r << std::endl;
        }

        // Dual part
        if (obj.d.numel() == 0) {
            os << "  Dual part (d): [Empty]" << std::endl;
        } else {
            os << "  Dual part (d):" << std::endl << obj.d << std::endl;
        }

        return os;
    }

    /**
     * @brief Method to get the device of the TensorDual object
     */
    torch::Device device() const {
        return device_;
    }

    /**
     * @brief Method to move the TensorDual object to a new device
     */
    TensorDual to(const torch::Device& device) const {
        // Ensure tensors are valid before moving
        if (r.numel() == 0 || d.numel() == 0) {
            throw std::runtime_error("Attempting to transfer empty tensors to a new device.");
        }

        // Transfer tensors to the specified device
        auto rn = r.to(device);
        auto rd = d.to(device);

        // Return a new TensorDual object
        return TensorDual(rn, rd);
    }
    

    /**
     * @brief Method to move the TensorDual object to a new device and dtype
     */
    TensorDual to(const torch::Device& device, const torch::Dtype& dtype) const {
        // Ensure tensors are valid before moving
        if (r.numel() == 0 || d.numel() == 0) {
            throw std::runtime_error("Attempting to transfer empty tensors to a new device.");
        }

        // Transfer tensors to the specified device and dtype
        auto rn = r.to(device, dtype);
        auto rd = d.to(device, dtype);

        // Return a new TensorDual object
        return TensorDual(rn, rd);
    }
    
    /**
     * @brief Set the requires_grad property for the real and dual tensors.
     * 
     * This method sets the requires_grad property for both the real (r) and dual (d) tensors,
     * enabling or disabling gradient computation for these tensors.
     * 
     * @param req_grad If true, gradients will be computed for r and d during backpropagation.
     * @throws std::runtime_error If the tensors are not defined, or if they are not floating-point types.
     */
    void set_requires_grad(bool req_grad) {
        // Ensure tensors are valid
        if (!r.defined() || !d.defined()) {
            throw std::runtime_error("Cannot set requires_grad on undefined tensors.");
        }

        // Ensure tensors are floating-point types
        if (!torch::isFloatingType(r.scalar_type()) || !torch::isFloatingType(d.scalar_type())) {
            throw std::runtime_error("requires_grad can only be set on floating-point tensors.");
        }

        // Set requires_grad
        r.set_requires_grad(req_grad);
        d.set_requires_grad(req_grad);
    }

    /**
     * @brief In-place modification of the requires_grad property for the real and dual tensors.
     * 
     * This method modifies the requires_grad property of the real (r) and dual (d) tensors in-place,
     * enabling or disabling gradient tracking for these tensors.
     * 
     * @param req_grad If true, enables gradient tracking. If false, disables gradient tracking.
     * @throws std::runtime_error If the tensors are undefined or not of floating-point dtype.
     */
    void requires_grad_(bool req_grad) {
        // Ensure tensors are valid
        if (!r.defined() || !d.defined()) {
            throw std::runtime_error("Cannot set requires_grad on undefined tensors.");
        }

        // Ensure tensors are floating-point types
        if (!torch::isFloatingType(r.scalar_type()) || !torch::isFloatingType(d.scalar_type())) {
            throw std::runtime_error("requires_grad_ can only be applied to floating-point tensors.");
        }

        // Modify requires_grad property in-place
        r.requires_grad_(req_grad);
        d.requires_grad_(req_grad);
    }

    /**
     * @brief Perform backpropagation for the real (r) and dual (d) tensors.
     * 
     * This method computes gradients for the tensors in TensorDual by calling backward() on the real (r) tensor.
     * It assumes that the real tensor (r) is scalar or provides a dependency chain for the dual tensor (d).
     * 
     * @throws std::runtime_error If r is not a scalar or if requires_grad is not enabled.
     */
    void backward() {
        // Ensure the real tensor is defined
        if (!r.defined()) {
            throw std::runtime_error("Cannot call backward() on an undefined real tensor (r).");
        }

        // Ensure requires_grad is enabled for r
        if (!r.requires_grad()) {
            throw std::runtime_error("Cannot call backward() because requires_grad is not enabled for r.");
        }

        // Ensure the real tensor is scalar
        if (r.numel() != 1) {
            throw std::runtime_error("The real tensor (r) must be a scalar to call backward().");
        }

        // Perform backpropagation
        r.backward();
    }

    /**
     * @brief Perform backpropagation for the TensorDual object using an explicit gradient.
     * 
     * This method computes gradients for the real (r) and dual (d) tensors in the TensorDual object
     * using the provided gradients in the `grad` TensorDual object.
     * 
     * @param grad A TensorDual object containing the gradients for the real (grad.r) and
     * dual (grad.d) tensors. These gradients must have the same shape as the corresponding tensors.
     * 
     * @throws std::invalid_argument If the shapes of grad.r or grad.d do not match r or d, respectively.
     * @throws std::runtime_error If r or d are undefined, or if requires_grad is not enabled.
     */
    void backward(const TensorDual& grad) {
        // Ensure tensors are defined
        if (!r.defined() || !d.defined() || !grad.r.defined() || !grad.d.defined()) {
            throw std::runtime_error("Cannot call backward on undefined tensors.");
        }

        // Ensure gradients match the shape of tensors
        if (grad.r.sizes() != r.sizes()) {
            throw std::runtime_error("Shape mismatch: grad.r must match r in size.");
        }
        if (grad.d.sizes() != d.sizes()) {
            throw std::runtime_error("Shape mismatch: grad.d must match d in size.");
        }

        // Ensure requires_grad is enabled
        if (!r.requires_grad()) {
            throw std::runtime_error("Cannot call backward on r because requires_grad is not enabled.");
        }
        if (!d.requires_grad()) {
            throw std::runtime_error("Cannot call backward on d because requires_grad is not enabled.");
        }

        // Perform backpropagation
        r.backward(grad.r);
        d.backward(grad.d);
    }

    /**
     * @brief Retrieve the gradients of the real (r) and dual (d) tensors.
     * 
     * This method retrieves the gradients of the real and dual tensors in the TensorDual object
     * and returns them as a new TensorDual object. Gradients are cloned to ensure they are
     * decoupled from the original computation graph.
     * 
     * @return A TensorDual object containing the gradients of r and d.
     * @throws std::runtime_error If gradients are not defined for r or d.
     */
    TensorDual grad() const {
        // Ensure gradients are defined
        if (!r.grad().defined()) {
            throw std::runtime_error("Gradient is not defined for the real tensor (r).");
        }
        if (!d.grad().defined()) {
            throw std::runtime_error("Gradient is not defined for the dual tensor (d).");
        }

        // Return a new TensorDual object with cloned gradients
        return TensorDual(r.grad().clone(), d.grad().clone());
    }
    /**
     * @brief Create a TensorDual object from a given tensor.
     * 
     * This static method constructs a TensorDual object where the real part is the
     * input tensor `r` and the dual part is a tensor with an extra dimension, initialized
     * to represent the identity matrix along the last two dimensions for each batch.
     * 
     * @param r The input tensor to use as the real part (must be at least 2D).
     * @return A TensorDual object with the specified real and dual parts.
     * @throws std::runtime_error If `r` has fewer than 2 dimensions.
     */
    static TensorDual create(const torch::Tensor& r) {
        // Ensure the input tensor has at least 2 dimensions
        if (r.dim() < 2) {
            throw std::runtime_error("Input tensor `r` must have at least 2 dimensions.");
        }

        // Number of leaves (size of the second dimension)
        auto l = r.size(1);

        // Create options for the dual tensor
        auto options = torch::TensorOptions().dtype(r.dtype()).device(r.device());

        // Create the dual tensor with an identity matrix structure
        auto identity = torch::eye(l, options).unsqueeze(0).expand({r.size(0), l, l});

        // Return a new TensorDual object
        return TensorDual(r, identity);
    }
    /**
     * @brief Create a TensorDual with zero tensors having the same shape, dtype, and device as the input TensorDual.
     * 
     * This static method creates a new TensorDual object where the real (r) and dual (d) tensors are filled with zeros,
     * and have the same properties (shape, dtype, and device) as the input TensorDual.
     * 
     * @param x The input TensorDual object to replicate.
     * @return A new TensorDual object with zero tensors matching the properties of the input.
     * @throws std::invalid_argument If the input TensorDual contains undefined tensors.
     */
    static TensorDual zeros_like(const TensorDual& x) {
        // Validate input tensors
        if (!x.r.defined() || !x.d.defined()) {
            throw std::invalid_argument("Input TensorDual contains undefined tensors.");
        }

        // Create zero tensors matching the shape, dtype, and device of the input
        auto r = torch::zeros_like(x.r);
        auto ds = torch::zeros_like(x.d);

        // Return the new TensorDual object
        return TensorDual(r, ds);
    }

    /**
     * @brief Create a TensorDual with a zero tensor as the real part and an adjusted dual part.
     * 
     * This method creates a new TensorDual object where the real part is a zero tensor with the same
     * shape as the input tensor `x`. The dual part is adjusted to match the shape of `x` and the
     * number of leaves (last dimension) of the current tensor's dual part.
     * 
     * @param x The input tensor to use as the shape reference for the real part.
     * @return A new TensorDual object with a zero real part and an adjusted dual part.
     * @throws std::invalid_argument If the input tensor `x` has fewer than 2 dimensions,
     * or if the current tensor's dual part is undefined or has fewer than 3 dimensions.
     */
    TensorDual zeros_like(const torch::Tensor& x) const {
        // Validate input tensor
        if (x.dim() < 2) {
            throw std::invalid_argument("Input tensor `x` must have at least 2 dimensions.");
        }

        // Validate the current dual tensor
        if (!d.defined() || d.dim() < 3) {
            throw std::invalid_argument("Dual tensor `d` must be defined and have at least 3 dimensions.");
        }

        // Create a zero tensor for the real part
        auto r = torch::zeros_like(x);

        // Get dimensions for the dual tensor
        int M = r.size(0);   // Batch size
        int nr = r.size(1);  // Number of rows
        int nd = d.size(2);  // Number of leaves (dual dimension)

        // Create a zero tensor for the dual part
        auto ds_options = x.options().dtype(torch::typeMetaToScalarType(x.dtype()) == torch::kBool 
                                    ? torch::kFloat64 
                                    : torch::typeMetaToScalarType(x.dtype()));
        auto ds = torch::zeros({M, nr, nd}, ds_options);

        // Return a new TensorDual object
        return TensorDual(r, ds);
    }

    /**
     * @brief Create a TensorDual object with zero tensors matching the shape, dtype, and device of the current object.
     * 
     * This method creates a new TensorDual object where the real (r) and dual (d) tensors are filled with zeros,
     * and have the same properties (shape, dtype, and device) as the current object's tensors.
     * 
     * @return A new TensorDual object with zero tensors matching the properties of the current object.
     * @throws std::runtime_error If the real or dual tensors in the current object are undefined.
     */
    TensorDual zeros_like() const {

        // Create zero tensors matching the properties of the current tensors
        auto r_zero = torch::zeros_like(this->r);
        auto d_zero = torch::zeros_like(this->d);

        // Return the new TensorDual object
        return TensorDual(r_zero, d_zero);
    }

    /**
     * @brief Create a TensorDual object with ones in the real part and zeros in the dual part.
     * 
     * This static method creates a new TensorDual object where the real part is filled with ones,
     * and the dual part is filled with zeros. The shape, dtype, and device of the new tensors
     * match the corresponding tensors in the input TensorDual object.
     * 
     * @param x The input TensorDual object to replicate.
     * @return A new TensorDual object with ones in the real part and zeros in the dual part.
     * @throws std::invalid_argument If the input TensorDual contains undefined tensors.
     */
    static TensorDual ones_like(const TensorDual& x) {
        // Ensure input tensors are defined
        if (!x.r.defined() || !x.d.defined()) {
            throw std::invalid_argument("Input TensorDual contains undefined tensors.");
        }

        // Create the real part with ones
        auto r = torch::ones_like(x.r);

        // Determine dtype for the dual part and create it with zeros
        auto ds_dtype = torch::typeMetaToScalarType(x.r.dtype()) == torch::kBool 
                        ? torch::kFloat64 
                        : torch::typeMetaToScalarType(x.r.dtype());
        auto ds = torch::zeros_like(x.d, x.d.options().dtype(ds_dtype));

        // Return the new TensorDual object
        return TensorDual(r, ds);
    }

    /**
     * @brief Create a boolean tensor with the same shape as the real part of the input TensorDual.
     * 
     * This static method creates a boolean tensor (`torch::kBool`) with the same shape and device
     * as the real part (`r`) of the input TensorDual object.
     * 
     * @param x The input TensorDual object whose real part's shape and device are used.
     * @return A boolean tensor with the same shape and device as the real part of the input TensorDual.
     * @throws std::invalid_argument If the real part (`r`) of the input TensorDual is undefined.
     */
    static torch::Tensor bool_like(const TensorDual& x) {
        // Ensure the real tensor is defined
        if (!x.r.defined()) {
            throw std::invalid_argument("Input TensorDual has an undefined real tensor (r).");
        }

        // Create a boolean tensor with the same shape and device as the real tensor
        return torch::zeros_like(x.r, torch::TensorOptions().dtype(torch::kBool).device(x.r.device()));
    }
    /**
     * @brief Create a TensorDual object with a specified additional dimension for the dual part.
     * 
     * This static method creates a TensorDual object where the real part is the input tensor `r`,
     * and the dual part is a zero tensor with the same shape as `r` plus an additional dimension `ddim`.
     * 
     * @param r The input tensor to use as the real part (must be defined).
     * @param ddim The size of the additional dimension for the dual part (must be > 0).
     * @return A new TensorDual object with the specified properties.
     * @throws std::invalid_argument If `r` is undefined or if `ddim` is not greater than zero.
     */
    static TensorDual createZero(const torch::Tensor& r, int ddim) {
        // Validate input tensor
        if (!r.defined()) {
            throw std::invalid_argument("Input tensor `r` is undefined.");
        }

        // Validate additional dimension
        if (ddim <= 0) {
            throw std::invalid_argument("The additional dimension `ddim` must be greater than zero.");
        }

        // Create the shape for the dual tensor
        auto dshape = r.sizes().vec(); // Copy the sizes to a vector
        dshape.push_back(ddim);        // Add the extra dimension for the dual part

        // Create a zero tensor for the dual part
        auto options = torch::TensorOptions().dtype(r.dtype()).device(r.device());
        auto ds = torch::zeros(dshape, options);

        // Return the new TensorDual object
        return TensorDual(r, ds);
    }

    /**
     * @brief Create a TensorDual object with uninitialized tensors matching the input TensorDual.
     * 
     * This static method creates a new TensorDual object where both the real part (`r`) and the dual part (`d`)
     * are uninitialized tensors. The shape, dtype, and device of the new tensors match the corresponding
     * tensors in the input TensorDual object.
     * 
     * @param x The input TensorDual object to replicate.
     * @return A new TensorDual object with uninitialized tensors matching the properties of the input.
     * @throws std::invalid_argument If the input TensorDual contains undefined tensors.
     */
    static TensorDual empty_like(const TensorDual& x) {
        // Validate input tensors
        if (!x.r.defined() || !x.d.defined()) {
            throw std::invalid_argument("Input TensorDual contains undefined tensors.");
        }

        // Create uninitialized tensors matching the shape, dtype, and device of the input
        auto r = torch::empty_like(x.r);
        auto ds = torch::empty_like(x.d);

        // Return the new TensorDual object
        return TensorDual(r, ds);
    }


    /**
     * @brief Concatenate multiple TensorDual objects along a specified dimension.
     * 
     * This static method concatenates the real and dual parts of multiple TensorDual objects
     * along a specified dimension. All TensorDual objects must have the same shape along
     * non-concatenating dimensions.
     * 
     * @param args A vector of TensorDual objects to concatenate (must not be empty).
     * @param dim The dimension along which to concatenate the tensors.
     * @return A new TensorDual object representing the concatenated tensors.
     * @throws std::invalid_argument If `args` is empty or if the tensors have incompatible shapes.
     */
    static TensorDual cat(const std::vector<TensorDual>& args, int64_t dim = 0) {
        // Validate input vector
        if (args.empty()) {
            throw std::invalid_argument("Input vector `args` must not be empty.");
        }

        // Validate shapes along non-concatenating dimensions
        auto r_shape = args[0].r.sizes();
        auto d_shape = args[0].d.sizes();

        for (const auto& a : args) {
            if (!std::equal(r_shape.begin(), r_shape.end(), a.r.sizes().begin())) {
                throw std::invalid_argument("Real tensors in `args` must have the same shape along non-concatenating dimensions.");
            }
            if (!std::equal(d_shape.begin(), d_shape.end(), a.d.sizes().begin())) {
                throw std::invalid_argument("Dual tensors in `args` must have the same shape along non-concatenating dimensions.");
            }
        }

        // Concatenate tensors
        std::vector<torch::Tensor> r_tensors;
        std::vector<torch::Tensor> d_tensors;

        for (const auto& a : args) {
            r_tensors.push_back(a.r);
            d_tensors.push_back(a.d);
        }

        auto r = torch::cat(r_tensors, dim);
        auto d = torch::cat(d_tensors, dim);

        return TensorDual(r, d);
    }


    /**
     * @brief Perform an einsum operation on two TensorDual objects.
     * 
     * This static method extends the torch.einsum function to TensorDual objects,
     * applying the specified einsum string to both the real and dual parts of the tensors.
     * 
     * @param arg The einsum string specifying the operation (must include '->').
     * @param first The first TensorDual object (must have defined real and dual tensors).
     * @param second The second TensorDual object (must have defined real and dual tensors).
     * @return A new TensorDual object representing the result of the einsum operation.
     * @throws std::invalid_argument If the einsum string is invalid or if the input TensorDual objects are undefined.
     */
    static TensorDual einsum(const std::string& arg, const TensorDual& first, const TensorDual& second) {
        // Validate input tensors
        auto pos1 = arg.find(",");
        if (pos1 == std::string::npos) {
            throw std::invalid_argument("Einsum string must contain ','.");
        }
        // Validate einsum string
        auto pos2 = arg.find("->");
        if (pos2 == std::string::npos) {
            throw std::invalid_argument("Einsum string must contain '->'.");
        }

        //The dual part must match
        if (first.d.size(2) != second.d.size(2)) {
            throw std::invalid_argument("The dual part of the TensorDual objects must have the same number of leaves.");
        }

        // Compute the real part
        auto r = torch::einsum(arg, {first.r, second.r});

        // Compute the dual part
        auto arg1 = arg.substr(0, pos1);
        auto arg2 = arg.substr(pos1 + 1, pos2 - pos1 - 1);
        auto arg3 = arg.substr(pos2 + 2);
        auto darg1 = arg1 + "," + arg2 + "z->" + arg3 + "z";
        auto darg2 = arg1 + "z," + arg2 + "->" + arg3 + "z";

        auto d1 = torch::einsum(darg1, {first.r, second.d});
        auto d2 = torch::einsum(darg2, {first.d, second.r});

        // Combine dual parts (assumes commutativity)
        return TensorDual(std::move(r), std::move(d1 + d2));
    }


    /**
     * @brief Perform an einsum operation between a Tensor and a TensorDual.
     * 
     * This static method extends the torch.einsum function to work with a combination of a Tensor
     * and a TensorDual. It applies the einsum string to the real and dual parts of the second
     * TensorDual, producing a new TensorDual as the result.
     * 
     * @param arg The einsum string specifying the operation (must include '->').
     * @param first The first torch::Tensor (real-valued).
     * @param second The second TensorDual object (must have defined real and dual tensors).
     * @return A new TensorDual object resulting from the einsum operation.
     * @throws std::invalid_argument If the einsum string is invalid or if the second TensorDual is undefined.
     */
    static TensorDual einsum(const std::string& arg, const torch::Tensor& first, const TensorDual& second) {
        // Validate input tensors
        if (!second.r.defined() || !second.d.defined()) {
            throw std::invalid_argument("The second TensorDual must have defined real and dual tensors.");
        }

        // Validate einsum string
        auto pos = arg.find("->");
        if (pos == std::string::npos) {
            throw std::invalid_argument("Einsum string must contain '->'.");
        }

        // Compute the real part
        auto r = torch::einsum(arg, {first, second.r});

        // Compute the dual part
        auto arg1 = arg.substr(0, pos);
        auto arg2 = arg.substr(pos + 2);
        auto darg = arg1 + "z->" + arg2 + "z";
        auto d = torch::einsum(darg, {first, second.d});

        // Return the result as a new TensorDual
        return TensorDual(r, d);
    }

    /**
     * @brief Perform an einsum operation between a TensorDual and a Tensor.
     * 
     * This static method extends the torch.einsum function to work with a combination of a TensorDual
     * and a torch::Tensor. It applies the einsum string to the real and dual parts of the first
     * TensorDual object, producing a new TensorDual as the result.
     * 
     * @param arg The einsum string specifying the operation (must include ',' and '->').
     * @param first The TensorDual object (must have defined real and dual tensors).
     * @param second The torch::Tensor (real-valued).
     * @return A new TensorDual object resulting from the einsum operation.
     * @throws std::invalid_argument If the einsum string is invalid or if the first TensorDual is undefined.
     */
    static TensorDual einsum(const std::string& arg, const TensorDual& first, const torch::Tensor& second) {
        // Validate input tensors
        if (!first.r.defined() || !first.d.defined()) {
            throw std::invalid_argument("The first TensorDual must have defined real and dual tensors.");
        }

        // Validate einsum string
        auto pos1 = arg.find(",");
        auto pos2 = arg.find("->");
        if (pos1 == std::string::npos || pos2 == std::string::npos || pos1 >= pos2) {
            throw std::invalid_argument("Einsum string must contain ',' and '->' in the correct order.");
        }

        // Parse the einsum string
        auto arg1 = arg.substr(0, pos1);                      // Indices for the first tensor
        auto arg2 = arg.substr(pos1 + 1, pos2 - pos1 - 1);    // Indices for the second tensor
        auto arg3 = arg.substr(pos2 + 2);                    // Indices for the output tensor
        auto darg = arg1 + "z," + arg2 + "->" + arg3 + "z";  // Modify for the dual part

        // Compute the real part
        auto r = torch::einsum(arg, {first.r, second});

        // Compute the dual part
        auto d = torch::einsum(darg, {first.d, second});

        // Return the result as a new TensorDual
        return TensorDual(r, d);
    }

    /**
     * @brief Perform a generalized einsum operation on multiple TensorDual objects.
     * 
     * This static method extends the torch.einsum function to operate on a vector of TensorDual objects.
     * It computes the einsum for the real and dual parts of the tensors, handling dual dimensions 
     * by reserving "z" as an implied dimension.
     * The limitation in the vectorized einsum is that the dual part of each tensor 
     * must have the same number of leaves.
     * 
     * @param arg The einsum string specifying the operation (must include "->" and cannot contain "z").
     * @param tensors A vector of TensorDual objects (must not be empty, and all elements must be valid).
     * @return A new TensorDual object resulting from the einsum operation.
     * @throws std::invalid_argument If the einsum string is invalid or if the tensors are not well-formed.
     */
    static TensorDual einsum(const std::string& arg, const std::vector<TensorDual>& tensors) {
        // Validate input einsum string
        if (arg.find("->") == std::string::npos) {
            throw std::invalid_argument("Einsum string must contain '->'.");
        }
        if (arg.find("z") != std::string::npos) {
            throw std::invalid_argument("Character 'z' is reserved for dual dimensions and cannot appear in the einsum string.");
        }

        // Validate input tensors
        if (tensors.empty()) {
            throw std::invalid_argument("Input vector `tensors` must not be empty.");
        }
        for (const auto& t : tensors) {
            if (!t.r.defined() || !t.d.defined()) {
                throw std::invalid_argument("Each TensorDual must have defined real and dual tensors.");
            }
        }

        // Compute the real part
        std::vector<torch::Tensor> r_tensors;
        for (const auto& t : tensors) {
            r_tensors.push_back(t.r);
        }
        auto r = torch::einsum(arg, r_tensors);

        // Parse the einsum string
        auto posa = arg.find("->");
        std::vector<size_t> poss; // Positions of commas
        size_t pos = arg.find(',');
        while (pos != std::string::npos) {
            poss.push_back(pos);
            pos = arg.find(',', pos + 1);
        }

        // Compute the dual part
        torch::Tensor d = torch::zeros_like(tensors[0].d);
        for (size_t i = 0; i < tensors.size(); ++i) {
            // Prepare tensors for the dual computation
            std::vector<torch::Tensor> r_tensorsc = r_tensors;
            r_tensorsc[i] = tensors[i].d; // Replace the ith tensor's real part with its dual part

            // Construct the dual einsum string
            std::string darg;
            if (i != tensors.size() - 1) {
                darg = arg.substr(0, poss[i]) + 
                    "z," + 
                    arg.substr(poss[i] + 1, posa - poss[i] - 1) + 
                    "->" + 
                    arg.substr(posa + 2) + "z";
            } else {
                darg = arg.substr(0, posa) + 
                    "z->" + 
                    arg.substr(posa + 2) + "z";
            }
            std::cerr << darg << std::endl;

            // Accumulate the dual computation
            d = d + torch::einsum(darg, r_tensorsc);
        }

        // Return the result as a new TensorDual
        return TensorDual(r, d);
    }

    /**
     * @brief Perform an element-wise selection between two TensorDual objects based on a condition.
     * 
     * This static method replicates the behavior of PyTorch's `torch::where` for TensorDual objects.
     * It applies the selection to both the real and dual parts of the tensors, expanding the condition
     * tensor as necessary.
     * 
     * @param cond The condition tensor (must be broadcast-compatible with `x` and `y`).
     * @param x The TensorDual object to select from where `cond` is true.
     * @param y The TensorDual object to select from where `cond` is false.
     * @return A new TensorDual object with elements selected based on `cond`.
     * @throws std::invalid_argument If `x` or `y` are not well-formed TensorDual objects or if
     *         `cond` is not broadcast-compatible with `x` and `y`.
     */
    static TensorDual where(const torch::Tensor& cond, const TensorDual& x, const TensorDual& y) {
        // Validate input TensorDual objects
        if (!x.r.defined() || !x.d.defined() || !y.r.defined() || !y.d.defined()) {
            throw std::invalid_argument("Input TensorDual objects must have defined real and dual tensors.");
        }

        // Validate condition tensor
        if (cond.dim() > x.r.dim()) {
            throw std::invalid_argument("Condition tensor has too many dimensions.");
        }

        // Expand condition tensor to match the dimensions of x and y
        torch::Tensor condr, condd;

        if (cond.dim() == 1) {
            condr = cond.unsqueeze(1).expand({x.r.size(0), x.r.size(1)});
            condd = cond.unsqueeze(1).unsqueeze(2).expand({x.d.size(0), x.d.size(1), x.d.size(2)});
        } else {
            condr = cond;
            condd = cond.unsqueeze(2).expand({x.d.size(0), x.d.size(1), x.d.size(2)});
        }

        // Perform element-wise selection
        auto xr = torch::where(condr, x.r, y.r);
        auto xd = torch::where(condd, x.d, y.d);

        // Return the new TensorDual object
        return TensorDual(xr, xd);
    }

    /**
     * @brief Compute the sum of a TensorDual object along the non batch dimension.
     * 
     * This static method computes the sum along the specified dimension for both the real (`r`)
     * and dual (`d`) parts of the TensorDual object. The result is returned as a new TensorDual object.
     * 
     * @param x The input TensorDual object (must have defined real and dual tensors).
     * @param dim The dimension along which to compute the sum (default is 1).
     * @return A new TensorDual object with the sum computed along the specified dimension.
     * @throws std::invalid_argument If the input TensorDual is not well-formed.
     */
    static TensorDual sum(const TensorDual& x) {

        // Compute the sum along the specified dimension
        auto r = torch::sum(x.r, /*dim=*/1, /*keepdim=*/true);
        auto d = torch::sum(x.d, /*dim=*/1, /*keepdim=*/true);

        // Return the result as a new TensorDual
        return TensorDual(r, d);
    }

    /**
     * @brief Compute the sum of the current TensorDual object along the non batch dimension.
     * 
     * This method computes the sum along the specified dimension for both the real (`r`)
     * and dual (`d`) parts of the TensorDual object. The result is returned as a new TensorDual object.
     * 
     * @param x The input TensorDual object (must have defined real and dual tensors).
     * @return A new TensorDual object with the sum computed along the specified dimension.
     */
    TensorDual sum() {

        // Compute the sum along the specified dimension
        auto r = torch::sum(this->r, /*dim=*/1, /*keepdim=*/true);
        auto d = torch::sum(this->d, /*dim=*/1, /*keepdim=*/true);

        // Return the result as a new TensorDual
        return TensorDual(r, d);
    }


    /**
     * @brief Compute the L2 norm of the real part of the TensorDual and propagate it to the dual part.
     * 
     * This method computes the L2 norm along dimension 1 of the real part (`r`) of the TensorDual object,
     * and applies the chain rule to propagate this operation to the dual part (`d`).
     * 
     * @return A new TensorDual object containing the L2 norm of the real part and the propagated dual part.
     * @throws std::invalid_argument If the TensorDual object is not well-formed.
     */
    TensorDual normL2() {
        // Validate input tensors
        if (!this->r.defined() || !this->d.defined()) {
            throw std::invalid_argument("TensorDual object must have defined real and dual tensors.");
        }

        // Compute the L2 norm along dimension 1 (retain dimensions)
        auto r_norm = torch::norm(this->r, 2, /*dim=*/1, /*keepdim=*/true);

        // Avoid division by zero by clamping the norm to a small value
        auto r_norm_clamped = r_norm.clamp_min(1e-12);

        // Compute the gradient with respect to the real part
        auto grad_r = this->r / r_norm_clamped;

        // Propagate the operation to the dual part using the chain rule
        auto dual = torch::einsum("mi,mij->mj", {grad_r, this->d}).unsqueeze(1);

        // Return the result as a new TensorDual
        return TensorDual(r_norm, dual);
    }



    /**
     * @brief Create a deep copy of the TensorDual object.
     * 
     * This method clones the real (`r`) and dual (`d`) tensors of the TensorDual object,
     * ensuring that the new instance is independent of the original.
     * 
     * @return A new TensorDual object that is a deep copy of the current object.
     * @throws std::invalid_argument If the TensorDual object has undefined real or dual tensors.
     */
    TensorDual clone() const {
        // Validate input tensors
        if (!r.defined() || !d.defined()) {
            throw std::invalid_argument("Cannot clone a TensorDual with undefined real or dual tensors.");
        }

        // Clone the real and dual tensors
        return TensorDual(r.clone(), d.clone());
    }

    /**
     * @brief Overload the unary negation operator for TensorDual.
     * 
     * This operator negates both the real (`r`) and dual (`d`) parts of the TensorDual object,
     * returning a new TensorDual object.
     * 
     * @return A new TensorDual object with negated real and dual tensors.
     * @throws std::invalid_argument If the TensorDual object has undefined real or dual tensors.
     */
    TensorDual operator-() const {
        // Validate input tensors
        if (!r.defined() || !d.defined()) {
            throw std::invalid_argument("Cannot negate a TensorDual with undefined real or dual tensors.");
        }

        // Negate real and dual parts and return as a new TensorDual
        return TensorDual(-r, -d);
    }

    /**
     * @brief Overload the addition operator for TensorDual objects.
     * 
     * This operator performs element-wise addition of the real (`r`) and dual (`d`) parts
     * of two TensorDual objects, returning a new TensorDual object.
     * 
     * @param other The TensorDual object to add to the current object.
     * @return A new TensorDual object resulting from the element-wise addition.
     * @throws std::invalid_argument If the TensorDual objects are not well-formed or their dimensions do not match.
     */
    TensorDual operator+(const TensorDual& other) const {
        // Validate input tensors
        if (!r.defined() || !d.defined() || !other.r.defined() || !other.d.defined()) {
            throw std::invalid_argument("Cannot add TensorDual objects with undefined real or dual tensors.");
        }

        // Validate dimension compatibility
        if (r.sizes() != other.r.sizes() || d.sizes() != other.d.sizes()) {
            throw std::invalid_argument("Dimension mismatch: Real and dual tensors of both TensorDual objects must have the same shape.");
        }

        // Perform element-wise addition
        return TensorDual(r + other.r, d + other.d);
    }

    /**
     * @brief Overload the addition operator for TensorDual and torch::Tensor.
     * 
     * This operator adds a torch::Tensor to the real part (`r`) of the TensorDual object
     * while leaving the dual part (`d`) unchanged.
     * 
     * @param other The torch::Tensor to add to the real part (must have the same shape as `r`).
     * @return A new TensorDual object with the updated real part and unchanged dual part.
     * @throws std::invalid_argument If the other tensor is undefined or its shape is incompatible with `r`.
     */
    TensorDual operator+(const torch::Tensor& other) const {
        // Validate input tensor
        if (!other.defined()) {
            throw std::invalid_argument("Cannot add TensorDual with an undefined tensor.");
        }

        if (r.sizes() != other.sizes()) {
            throw std::invalid_argument("Dimension mismatch: The other tensor must have the same shape as the real part of the TensorDual.");
        }

        // Perform addition on the real part and return a new TensorDual
        return TensorDual(r + other, d.clone());
    }

    /**
     * @brief Overload the addition operator for TensorDual and a scalar.
     * 
     * This operator adds a scalar to the real part (`r`) of the TensorDual object,
     * while leaving the dual part (`d`) unchanged. It creates a new TensorDual object as the result.
     * 
     * @param other The scalar to add to the real part of the TensorDual.
     * @return A new TensorDual object with the updated real part and unchanged dual part.
     */
    TensorDual operator+(double other) const {
        // Create a tensor matching the shape, dtype, and device of the real part with the scalar value
        auto scalar_tensor = torch::full_like(r, other);

        // Perform addition on the real part
        return TensorDual(r + scalar_tensor, d.clone()); // Clone ensures dual part remains immutable
    }

    /**
     * @brief Overload the subtraction operator for TensorDual objects.
     * 
     * This operator performs element-wise subtraction of the real (`r`) and dual (`d`) parts
     * of two TensorDual objects, returning a new TensorDual object.
     * 
     * @param other The TensorDual object to subtract from the current object.
     * @return A new TensorDual object resulting from the element-wise subtraction.
     * @throws std::invalid_argument If the TensorDual objects are not well-formed or their dimensions do not match.
     */
    TensorDual operator-(const TensorDual& other) const {
        // Validate input tensors
        if (!r.defined() || !d.defined() || !other.r.defined() || !other.d.defined()) {
            throw std::invalid_argument("Cannot subtract TensorDual objects with undefined real or dual tensors.");
        }

        // Validate dimension compatibility
        if (r.sizes() != other.r.sizes() || d.sizes() != other.d.sizes()) {
            throw std::invalid_argument("Dimension mismatch: Real and dual tensors of both TensorDual objects must have the same shape.");
        }

        // Perform element-wise subtraction
        return TensorDual(r - other.r, d - other.d);
    }

    /**
     * @brief Overload the subtraction operator for TensorDual and torch::Tensor.
     * 
     * This operator subtracts a torch::Tensor from the real part (`r`) of the TensorDual object,
     * while leaving the dual part (`d`) unchanged. It creates a new TensorDual object as the result.
     * 
     * @param other The torch::Tensor to subtract from the real part (must be broadcast-compatible with `r`).
     * @return A new TensorDual object with the updated real part and unchanged dual part.
     * @throws std::invalid_argument If the other tensor is undefined or not broadcast-compatible with `r`.
     */
    TensorDual operator-(const torch::Tensor& other) const {
        // Validate input tensor
        if (!other.defined()) {
            throw std::invalid_argument("Cannot subtract an undefined tensor from TensorDual.");
        }

        // Perform element-wise subtraction on the real part
        auto real_part = r - other;

        // Clone the dual part to ensure immutability
        auto dual_part = d.clone();

        // Return the new TensorDual object
        return TensorDual(real_part, dual_part);
    }

    /**
     * @brief Overload the subtraction operator for TensorDual and a scalar.
     * 
     * This operator subtracts a scalar from the real part (`r`) of the TensorDual object,
     * while leaving the dual part (`d`) unchanged. It creates a new TensorDual object as the result.
     * 
     * @param scalar The scalar to subtract from the real part of the TensorDual.
     * @return A new TensorDual object with the updated real part and unchanged dual part.
     */
    TensorDual operator-(double scalar) const {
        // Create a scalar tensor matching the shape, dtype, and device of the real part
        auto scalar_tensor = torch::full_like(r, scalar);

        // Perform subtraction on the real part
        auto real_part = r - scalar_tensor;

        // Clone the dual part to ensure immutability
        auto dual_part = d.clone();

        // Return the new TensorDual object
        return TensorDual(real_part, dual_part);
    }

    /**
     * @brief Create a contiguous version of the TensorDual object.
     * 
     * This method ensures that both the real (`r`) and dual (`d`) tensors of the TensorDual object
     * are contiguous in memory. A new TensorDual object with contiguous tensors is returned.
     * 
     * @return A new TensorDual object with contiguous real and dual tensors.
     * @throws std::invalid_argument If the TensorDual object contains undefined tensors.
     */
    TensorDual contiguous() const {
        // Validate input tensors
        if (!r.defined() || !d.defined()) {
            throw std::invalid_argument("Cannot make contiguous: TensorDual contains undefined real or dual tensors.");
        }

        // Make both real and dual parts contiguous
        return TensorDual(r.contiguous(), d.contiguous());
    }

    /**
     * @brief Overload the multiplication operator for TensorDual objects.
     * 
     * This operator performs element-wise multiplication of the real (`r`) parts of two TensorDual objects
     * and calculates the corresponding dual part using the chain rule:
     * 
     * \[
     * r_{result} = r \cdot r_{other}
     * \]
     * \[
     * d_{result} = r_{other} \cdot d + r \cdot d_{other}
     * \]
     * 
     * @param other The TensorDual object to multiply with the current object.
     * @return A new TensorDual object resulting from the element-wise multiplication.
     * @throws std::invalid_argument If the TensorDual objects are not well-formed or their dimensions do not match.
     */
    TensorDual operator*(const TensorDual& other) const {
        // Validate input tensors
        if (!r.defined() || !d.defined() || !other.r.defined() || !other.d.defined()) {
            throw std::invalid_argument("Cannot multiply: TensorDual objects must have defined real and dual tensors.");
        }

        // Validate dimension compatibility
        if (r.sizes() != other.r.sizes() || d.sizes() != other.d.sizes()) {
            throw std::invalid_argument("Dimension mismatch: Real and dual tensors must have the same shape for multiplication.");
        }

        // Compute the real and dual parts
        auto real = r * other.r;
        auto dual = other.r.unsqueeze(-1) * d + r.unsqueeze(-1) * other.d;

        // Return the result as a new TensorDual
        return TensorDual(real, dual);
    }


    /**
     * @brief Overload the multiplication operator for TensorDual and torch::Tensor.
     * 
     * This operator performs element-wise multiplication of a torch::Tensor with the real (`r`) part
     * of the TensorDual object. It also scales the dual part (`d`) of the TensorDual by the tensor,
     * maintaining the dual structure.
     * 
     * @param other The torch::Tensor to multiply with the TensorDual (must have the same shape as `r`).
     * @return A new TensorDual object resulting from the element-wise multiplication.
     * @throws std::invalid_argument If the input tensor is undefined or has incompatible dimensions.
     */
    TensorDual operator*(const torch::Tensor& other) const {
        // Validate input tensor
        if (!other.defined()) {
            throw std::invalid_argument("Cannot multiply: Input tensor is undefined.");
        }

        if (other.sizes() != r.sizes()) {
            throw std::invalid_argument("Dimension mismatch: Input tensor must have the same shape as the real part of the TensorDual.");
        }

        // Compute the real and dual parts
        auto real = other * r;
        auto scaled_other = other.unsqueeze(-1);  // Add an extra dimension for dual part scaling
        auto dual = scaled_other * d;

        // Return the result as a new TensorDual
        return TensorDual(real, dual);
    }

    /**
     * @brief Overload the multiplication operator for a scalar and a TensorDual.
     * 
     * This operator scales both the real (`r`) and dual (`d`) parts of the TensorDual object
     * by a scalar value.
     * 
     * @param scalar The scalar value to multiply with the TensorDual.
     * @return A new TensorDual object with both real and dual parts scaled by the scalar.
     */
    TensorDual operator*(const double& scalar) const {
        // Scale the real and dual parts by the scalar
        auto real_scaled = r * scalar;
        auto dual_scaled = d * scalar;

        // Return the result as a new TensorDual
        return TensorDual(real_scaled, dual_scaled);
    }

    /**
     * @brief Overload the less-than-or-equal-to (<=) operator for TensorDual objects.
     * 
     * This operator performs an element-wise comparison of the real (`r`) parts of two TensorDual objects,
     * returning a boolean tensor indicating where the condition is satisfied.
     * 
     * The dual (`d`) parts are ignored in this comparison.
     * 
     * @param other The TensorDual object to compare with the current object.
     * @return A torch::Tensor containing boolean values for the element-wise comparison.
     * @throws std::invalid_argument If the real parts (`r`) of the TensorDual objects have different shapes.
     */
    torch::Tensor operator<=(const TensorDual& other) const {
        // Validate shape compatibility
        if (r.sizes() != other.r.sizes()) {
            throw std::invalid_argument("Dimension mismatch: Real tensors must have the same shape for comparison.");
        }

        // Perform element-wise comparison of the real parts
        return r <= other.r;
    }


    /**
     * @brief Overload the equality (==) operator for TensorDual objects.
     * 
     * This operator performs an element-wise comparison of the real (`r`) parts of two TensorDual objects,
     * returning a boolean tensor indicating where the values are equal.
     * 
     * The dual (`d`) parts are ignored in this comparison.
     * 
     * @param other The TensorDual object to compare with the current object.
     * @return A torch::Tensor containing boolean values for the element-wise comparison.
     * @throws std::invalid_argument If the real parts (`r`) of the TensorDual objects have different shapes.
     */
    torch::Tensor operator==(const TensorDual& other) const {
        // Validate shape compatibility
        if (r.sizes() != other.r.sizes()) {
            throw std::invalid_argument("Dimension mismatch: Real tensors must have the same shape for comparison.");
        }

        // Perform element-wise comparison of the real parts
        return r == other.r;
    }


    /**
     * @brief Overload the less-than-or-equal-to (<=) operator for TensorDual and torch::Tensor.
     * 
     * This operator performs an element-wise comparison of the real (`r`) part of the TensorDual object
     * with a torch::Tensor. The dual (`d`) part is ignored in this comparison.
     * 
     * @param other The torch::Tensor to compare with the real part of the TensorDual. Must be broadcast-compatible.
     * @return A torch::Tensor containing boolean values for the element-wise comparison.
     * @throws std::invalid_argument If the input tensor is undefined or not broadcast-compatible with `r`.
     */
    torch::Tensor operator<=(const torch::Tensor& other) const {
        // Validate input tensor
        if (!other.defined()) {
            throw std::invalid_argument("Cannot compare: Input tensor is undefined.");
        }


        // Perform element-wise comparison of the real part
        return r <= other;
    }
    /**
     * @brief Overload the less-than-or-equal-to (<=) operator for TensorDual and a scalar.
     * 
     * This operator performs an element-wise comparison of the real (`r`) part of the TensorDual object
     * with a scalar value. The dual (`d`) part is ignored in this comparison.
     * 
     * @tparam Scalar The type of the scalar value. Must be convertible to a scalar that PyTorch supports.
     * @param scalar The scalar value to compare with the real part of the TensorDual.
     * @return A torch::Tensor containing boolean values for the element-wise comparison.
     */
    template <typename Scalar>
    torch::Tensor operator<=(const Scalar& scalar) const {
        // Perform element-wise comparison of the real part
        return r <= scalar;
    }

    /**
     * @brief Overload the greater-than (>) operator for TensorDual objects.
     * 
     * This operator performs an element-wise comparison of the real (`r`) parts of two TensorDual objects,
     * returning a boolean tensor indicating where the values in the current object are greater than the values in the other.
     * 
     * The dual (`d`) parts are ignored in this comparison.
     * 
     * @param other The TensorDual object to compare with the current object.
     * @return A torch::Tensor containing boolean values for the element-wise comparison.
     * @throws std::invalid_argument If the real parts (`r`) of the TensorDual objects have different shapes.
     */
    torch::Tensor operator>(const TensorDual& other) const {
        // Validate shape compatibility
        if (r.sizes() != other.r.sizes()) {
            throw std::invalid_argument("Dimension mismatch: Real tensors must have the same shape for comparison.");
        }

        // Perform element-wise comparison of the real parts
        return r > other.r;
    }

    /**
     * @brief Overload the greater-than (>) operator for a TensorDual and a torch::Tensor.
     * 
     * This operator performs an element-wise comparison of the real (`r`) part of the TensorDual object
     * with a torch::Tensor, returning a boolean tensor where the condition is satisfied.
     * The dual (`d`) part is ignored in this comparison.
     * 
     * @param other The torch::Tensor to compare with the real part of the TensorDual. Must be broadcast-compatible.
     * @return A torch::Tensor containing boolean values for the element-wise comparison.
     * @throws std::invalid_argument If the input tensor is undefined or not broadcast-compatible with `r`.
     */
    torch::Tensor operator>(const torch::Tensor& other) const {
        // Validate input tensor
        if (!other.defined()) {
            throw std::invalid_argument("Cannot compare: Input tensor is undefined.");
        }

        // Check broadcast compatibility
        if (r.sizes() != other.sizes()) {
            throw std::invalid_argument("Dimension mismatch: Input tensor must have the same shape as the real part of the TensorDual.");
        }

        // Perform element-wise comparison of the real part
        return r > other;
    }

    /**
     * @brief Overload the greater-than (>) operator for a scalar and a TensorDual.
     * 
     * This operator performs an element-wise comparison of the real (`r`) part of the TensorDual object
     * with a scalar value. The dual (`d`) part is ignored in this comparison.
     * 
     * @tparam Scalar The type of the scalar value. Must be convertible to a scalar supported by PyTorch.
     * @param scalar The scalar value to compare with the real part of the TensorDual.
     * @return A torch::Tensor containing boolean values for the element-wise comparison.
     */
    template <typename Scalar>
    torch::Tensor operator>(const Scalar& scalar) const {
        // Perform element-wise comparison directly with the scalar
        return r > scalar;
    }

    /**
     * @brief Overload the less-than (<) operator for TensorDual objects.
     * 
     * This operator performs an element-wise comparison of the real (`r`) parts of two TensorDual objects,
     * returning a boolean tensor indicating where the values in the current object are less than the values in the other.
     * 
     * The dual (`d`) parts are ignored in this comparison.
     * 
     * @param other The TensorDual object to compare with the current object.
     * @return A torch::Tensor containing boolean values for the element-wise comparison.
     * @throws std::invalid_argument If the real parts (`r`) of the TensorDual objects have different shapes.
     */
    torch::Tensor operator<(const TensorDual& other) const {
        // Validate shape compatibility
        if (r.sizes() != other.r.sizes()) {
            throw std::invalid_argument("Dimension mismatch: Real tensors must have the same shape for comparison.");
        }

        // Perform element-wise comparison of the real parts
        return r < other.r;
    }


    /**
     * @brief Overload the less-than (<) operator for a torch::Tensor and a TensorDual.
     * 
     * This operator performs an element-wise comparison of a torch::Tensor with the real (`r`) part
     * of the TensorDual object, returning a boolean tensor where the condition is satisfied.
     * 
     * The dual (`d`) part is ignored in this comparison.
     * 
     * @param other The torch::Tensor to compare with the real part of the TensorDual. Must be broadcast-compatible.
     * @return A torch::Tensor containing boolean values for the element-wise comparison.
     * @throws std::invalid_argument If the input tensor is undefined or not broadcast-compatible with `r`.
     */
    torch::Tensor operator<(const torch::Tensor& other) const {
        // Validate input tensor
        if (!other.defined()) {
            throw std::invalid_argument("Cannot compare: Input tensor is undefined.");
        }

        if (r.sizes() != other.sizes()) {
            throw std::invalid_argument("Dimension mismatch: Input tensor must have the same shape as the real part of the TensorDual.");
        }

        // Perform element-wise comparison
        return r < other;
    }

    /**
     * @brief Overload the less-than (<) operator for a scalar and a TensorDual.
     * 
     * This operator performs an element-wise comparison of a scalar with the real (`r`) part
     * of the TensorDual object. The dual (`d`) part is ignored in this comparison.
     * 
     * @tparam Scalar The type of the scalar value. Must be convertible to a scalar supported by PyTorch.
     * @param scalar The scalar value to compare with the real part of the TensorDual.
     * @return A torch::Tensor containing boolean values for the element-wise comparison.
     */
    template <typename Scalar>
    torch::Tensor operator<(const Scalar& scalar) const {
        // Perform element-wise comparison of the real part with the scalar
        return r < scalar;
    }


    /**
     * @brief Overload the greater-than-or-equal-to (>=) operator for TensorDual objects.
     * 
     * This operator performs an element-wise comparison of the real (`r`) parts of two TensorDual objects,
     * returning a boolean tensor indicating where the condition is satisfied.
     * 
     * The dual (`d`) parts are ignored in this comparison.
     * 
     * @param other The TensorDual object to compare with the current object.
     * @return A torch::Tensor containing boolean values for the element-wise comparison.
     * @throws std::invalid_argument If the real parts (`r`) of the TensorDual objects have different shapes.
     */
    torch::Tensor operator>=(const TensorDual& other) const {
        // Validate shape compatibility
        if (r.sizes() != other.r.sizes()) {
            throw std::invalid_argument("Dimension mismatch: Real tensors must have the same shape for comparison.");
        }

        // Perform element-wise comparison of the real parts
        return r >= other.r;
    }

    /**
     * @brief Overload the greater-than-or-equal-to (>=) operator for a TensorDual and a torch::Tensor.
     * 
     * This operator performs an element-wise comparison of the real (`r`) part of the TensorDual object
     * with a torch::Tensor, returning a boolean tensor where the condition is satisfied.
     * The dual (`d`) part is ignored in this comparison.
     * 
     * @param other The torch::Tensor to compare with the real part of the TensorDual. Must have the same shape as `r`.
     * @return A torch::Tensor containing boolean values for the element-wise comparison.
     * @throws std::invalid_argument If the input tensor is undefined or has incompatible dimensions.
     */
    torch::Tensor operator>=(const torch::Tensor& other) const {
        // Validate input tensor
        if (!other.defined()) {
            throw std::invalid_argument("Cannot compare: Input tensor is undefined.");
        }

        if (r.sizes() != other.sizes()) {
            throw std::invalid_argument("Dimension mismatch: Input tensor must have the same shape as the real part of the TensorDual.");
        }

        // Perform element-wise comparison
        return r >= other;
    }


    /**
     * @brief Overload the greater-than-or-equal-to (>=) operator for a TensorDual and a scalar.
     * 
     * This operator performs an element-wise comparison of the real (`r`) part of the TensorDual object
     * with a scalar value, returning a boolean tensor where the condition is satisfied.
     * The dual (`d`) part is ignored in this comparison.
     * 
     * @tparam Scalar The type of the scalar value. Must be convertible to a scalar supported by PyTorch.
     * @param scalar The scalar value to compare with the real part of the TensorDual.
     * @return A torch::Tensor containing boolean values for the element-wise comparison.
     */
    template <typename Scalar>
    torch::Tensor operator>=(const Scalar& scalar) const {
        // Perform element-wise comparison directly with the scalar
        return r >= scalar;
    }


    /**
     * @brief Overload the equality (==) operator for a TensorDual and a torch::Tensor.
     * 
     * This operator performs an element-wise comparison of the real (`r`) part of the TensorDual object
     * with a torch::Tensor, returning a boolean tensor where the condition is satisfied.
     * The dual (`d`) part is ignored in this comparison.
     * 
     * @param other The torch::Tensor to compare with the real part of the TensorDual. Must have the same shape as `r`.
     * @return A torch::Tensor containing boolean values for the element-wise comparison.
     * @throws std::invalid_argument If the input tensor is undefined or has incompatible dimensions.
     */
    torch::Tensor operator==(const torch::Tensor& other) const {
        // Validate input tensor
        if (!other.defined()) {
            throw std::invalid_argument("Cannot compare: Input tensor is undefined.");
        }

        if (r.sizes() != other.sizes()) {
            throw std::invalid_argument("Dimension mismatch: Input tensor must have the same shape as the real part of the TensorDual.");
        }

        // Perform element-wise equality comparison
        return r == other;
    }

    /**
     * @brief Overload the equality (==) operator for a TensorDual and a scalar.
     * 
     * This operator performs an element-wise comparison of the real (`r`) part of the TensorDual object
     * with a scalar value, returning a boolean tensor where the condition is satisfied.
     * The dual (`d`) part is ignored in this comparison.
     * 
     * @tparam Scalar The type of the scalar value. Must be convertible to a scalar supported by PyTorch.
     * @param scalar The scalar value to compare with the real part of the TensorDual.
     * @return A torch::Tensor containing boolean values for the element-wise comparison.
     */
    template <typename Scalar>
    torch::Tensor operator==(const Scalar& scalar) const {
        // Perform element-wise equality comparison directly with the scalar
        return r == scalar;
    }

    /**
     * @brief Overload the inequality (!=) operator for TensorDual objects.
     * 
     * This operator performs an element-wise inequality comparison of the real (`r`) parts of two TensorDual objects,
     * returning a boolean tensor where the condition is satisfied.
     * 
     * The dual (`d`) parts are ignored in this comparison.
     * 
     * @param other The TensorDual object to compare with the current object.
     * @return A torch::Tensor containing boolean values for the element-wise comparison.
     * @throws std::invalid_argument If the real parts (`r`) of the TensorDual objects have different shapes.
     */
    torch::Tensor operator!=(const TensorDual& other) const {
        // Validate shape compatibility
        if (r.sizes() != other.r.sizes()) {
            throw std::invalid_argument("Dimension mismatch: Real tensors must have the same shape for comparison.");
        }

        // Perform element-wise inequality comparison
        return r != other.r;
    }

    /**
     * @brief Overload the inequality (!=) operator for a TensorDual and a torch::Tensor.
     * 
     * This operator performs an element-wise inequality comparison of the real (`r`) part of the TensorDual object
     * with a torch::Tensor, returning a boolean tensor where the condition is satisfied.
     * The dual (`d`) part is ignored in this comparison.
     * 
     * @param other The torch::Tensor to compare with the real part of the TensorDual. Must have the same shape as `r`.
     * @return A torch::Tensor containing boolean values for the element-wise comparison.
     * @throws std::invalid_argument If the input tensor is undefined or has incompatible dimensions.
     */
    torch::Tensor operator!=(const torch::Tensor& other) const {
        // Validate input tensor
        if (!other.defined()) {
            throw std::invalid_argument("Cannot compare: Input tensor is undefined.");
        }

        if (r.sizes() != other.sizes()) {
            throw std::invalid_argument("Dimension mismatch: Input tensor must have the same shape as the real part of the TensorDual.");
        }

        // Perform element-wise inequality comparison
        return r != other;
    }


    /**
     * @brief Overload the inequality (!=) operator for a TensorDual and a scalar.
     * 
     * This operator performs an element-wise inequality comparison of the real (`r`) part of the TensorDual object
     * with a scalar value, returning a boolean tensor where the condition is satisfied.
     * The dual (`d`) part is ignored in this comparison.
     * 
     * @tparam Scalar The type of the scalar value. Must be convertible to a scalar supported by PyTorch.
     * @param scalar The scalar value to compare with the real part of the TensorDual.
     * @return A torch::Tensor containing boolean values for the element-wise comparison.
     */
    template <typename Scalar>
    torch::Tensor operator!=(const Scalar& scalar) const {
        // Perform element-wise inequality comparison directly with the scalar
        return r != scalar;
    }

    /**
     * @brief Overload the division (/) operator for TensorDual objects.
     * 
     * This operator performs element-wise division of two TensorDual objects. It computes the division
     * for the real (`r`) and dual (`d`) parts, ensuring the dual part respects the derivative propagation rules.
     * 
     * @param other The TensorDual object to divide by.
     * @return A new TensorDual object representing the division result.
     * @throws std::invalid_argument If the dimensions of the two TensorDual objects do not match.
     */
    TensorDual operator/(const TensorDual& other) const {

        // Ensure the denominator is safe for division
        auto safe_r = torch::sgn(other.r) * other.r.abs().clamp_min(1e-12);
    
        // Compute the real part
        auto r = this->r / safe_r;

        // Compute the dual part
        auto otherrsq = safe_r.square();
        auto d = -(this->r / otherrsq).unsqueeze(-1) * other.d + this->d / safe_r.unsqueeze(-1);

        // Return the result
        return TensorDual(r, d);
    }


    /**
     * @brief Overload the division (/) operator for TensorDual and torch::Tensor.
     * 
     * This operator performs element-wise division of the real (`r`) and dual (`d`) parts of the TensorDual object
     * by a torch::Tensor. The input tensor is adjusted to be broadcast-compatible if necessary.
     * 
     * @param other The torch::Tensor to divide by. Must be broadcast-compatible with `r`.
     * @return A new TensorDual object representing the division result.
     * @throws std::invalid_argument If the input tensor cannot be broadcast to match `r`.
     */
    TensorDual operator/(const torch::Tensor& other) const {
        // Adjust dimensions if needed to make `other` broadcast-compatible
        auto othere = other.dim() != this->r.dim() ? other.unsqueeze(1) : other;

        // Ensure the denominator is safe for division
        auto safe_other = torch::sgn(other) * other.abs().clamp_min(1e-12);


        // Compute the real part
        auto r = this->r / safe_other;

        // Compute the dual part
        auto d = this->d / safe_other.unsqueeze(-1);

        // Return the result
        return TensorDual(r, d);
    }

    TensorMatDual operator/(const TensorMatDual& other) const; // Forward declaration

    /**
     * @brief Overload the division (/) operator for a TensorDual and a scalar.
     * 
     * This operator performs element-wise division of the real (`r`) and dual (`d`) parts
     * of the TensorDual object by a scalar.
     * 
     * @param scalar The scalar value to divide by.
     * @return A new TensorDual object representing the division result.
     */
    TensorDual operator/(double scalar) {
        // Check for division by zero
        auto tscalar = torch::tensor(scalar);
        auto safe_scalar = torch::sgn(tscalar) * tscalar.abs().clamp_min(1e-12);

        // Perform element-wise division
        return TensorDual(this->r / safe_scalar, this->d / safe_scalar);
    }
    /**
     * @brief Gathers elements along a specified dimension for both the real (`r`) and dual (`d`) parts of the TensorDual.
     * 
     * This method gathers elements from the `r` and `d` tensors using the provided `index` tensor along the specified `dim`.
     * For the dual part (`d`), the `index` tensor is unsqueezed and expanded to match the shape of `d`.
     * 
     * @param dim The dimension along which to index.
     * @param index A tensor containing the indices of elements to gather. Must be broadcast-compatible with the target tensors.
     * @return A new TensorDual object containing the gathered elements.
     * @throws std::invalid_argument If the `dim` or `index` is invalid.
     */
    TensorDual gather(int64_t dim, const torch::Tensor& index) const {
        // Validate input dimension
        if (dim < 0 || dim >= r.dim()) {
            throw std::invalid_argument("Invalid dimension: 'dim' must be within the range of the tensor dimensions.");
        }

        // Validate index tensor compatibility
        if (index.dim() > r.dim()) {
            throw std::invalid_argument("Index tensor dimensions are incompatible with the target tensor.");
        }

        // Expand the index tensor for the dual part
        auto indexd = index.unsqueeze(2).expand({-1, -1, d.size(2)});

        // Gather elements for the real and dual parts
        auto r_gathered = r.gather(dim, index);
        auto d_gathered = d.gather(dim, indexd);

        // Return the gathered TensorDual
        return TensorDual(r_gathered, d_gathered);
    }

    /**
     * @brief Scatters elements along a specified dimension for both the real (`r`) and dual (`d`) parts of the TensorDual.
     * 
     * This method updates elements in the `r` and `d` tensors along the given dimension (`dim`) using the provided
     * indices (`index`) and source TensorDual (`src`).
     * 
     * @param dim The dimension along which to scatter.
     * @param index A tensor containing the indices of elements to scatter. Must be broadcast-compatible with the target tensors.
     * @param src The source TensorDual containing the values to scatter.
     * @return A new TensorDual object with the scattered values.
     * @throws std::invalid_argument If the `dim`, `index`, or `src` is invalid.
     */
    TensorDual scatter(int64_t dim, const torch::Tensor& index, const TensorDual& src) const {
        // Validate dimension
        if (dim < 0 || dim >= r.dim()) {
            throw std::invalid_argument("Invalid dimension: 'dim' must be within the range of the tensor dimensions.");
        }

        // Validate index tensor compatibility
        if (index.dim() > r.dim()) {
            throw std::invalid_argument("Index tensor dimensions are incompatible with the target tensor.");
        }

        // Validate source TensorDual dimensions
        if (r.sizes() != src.r.sizes() || d.sizes() != src.d.sizes()) {
            throw std::invalid_argument("Source TensorDual must have the same shape as the target TensorDual.");
        }
        std::cerr << "index: " << index << std::endl;
        // Expand the index tensor for the dual part
        auto index_expanded = index.unsqueeze(-1).expand({-1, -1, d.size(2)});
        std::cerr << "index_expanded: " << index_expanded << std::endl;

        // Scatter elements for the real and dual parts
        auto r_scattered = r.scatter(dim, index, src.r);
        auto d_scattered = d.scatter(dim, index_expanded, src.d);

        // Return the scattered TensorDual
        return TensorDual(r_scattered, d_scattered);
    }

    /**
     * @brief Computes the reciprocal of a TensorDual object.
     * 
     * This method computes the reciprocal for both the real (`r`) and dual (`d`) parts of the TensorDual.
     * The reciprocal of the dual part is derived using the chain rule for derivatives.
     * 
     * @return A new TensorDual object representing the reciprocal.
     * @throws std::runtime_error If any element in the real part (`r`) is zero.
     */
    TensorDual reciprocal() const {
        // Compute the reciprocal of the real part, ensuring no division by zero
        auto r_safe = torch::sgn(r) * r.abs().clamp_min(1e-12);
        auto rrec = r_safe.reciprocal();  // Reciprocal of the real part

        // Compute the dual part using the chain rule
        auto rrec_squared = rrec.unsqueeze(-1) * rrec.unsqueeze(-1); // (1/r)^2
        auto d = -rrec_squared * this->d; // Dual part

        // Return the reciprocal TensorDual
        return TensorDual(rrec, d);
    }

    /**
     * @brief Computes the square of a TensorDual object.
     * 
     * This method computes the square of the real (`r`) part and adjusts the dual (`d`) part using
     * the chain rule of derivatives. The operation is defined as:
     *   f(x) = x^2
     *   f'(x) = 2x
     * 
     * @return A new TensorDual object representing the squared result.
     * @throws std::invalid_argument If the dimensions of `d` are incompatible with `r`.
     */
    TensorDual square() const {
        // Validate dimensions of the dual part
        if (r.dim() + 1 != d.dim() || r.sizes() != d.sizes().slice(0, r.dim())) {
            throw std::invalid_argument("Dual part dimensions are incompatible with the real part.");
        }

        // Compute the square of the real part
        auto rsq = r.square();

        // Compute the dual part using the chain rule: 2 * r * d
        auto r_unsqueezed = r.unsqueeze(-1); // Expand dimensions for dual computation
        auto d = 2 * r_unsqueezed * this->d;

        // Return the squared TensorDual
        return TensorDual(rsq, d);
    }

    /**
     * @brief Computes the sine of a TensorDual object.
     * 
     * This method computes the sine of the real (`r`) part and adjusts the dual (`d`) part using
     * the chain rule of derivatives. The operation is defined as:
     *   f(x) = sin(x)
     *   f'(x) = cos(x)
     * 
     * @return A new TensorDual object representing the sine result.
     * @throws std::invalid_argument If the dimensions of `d` are incompatible with `r`.
     */
    TensorDual sin() const {
        // Validate dimensions of the dual part
        if (r.dim() + 1 != d.dim() || r.sizes() != d.sizes().slice(0, r.dim())) {
            throw std::invalid_argument("Dual part dimensions are incompatible with the real part.");
        }

        // Compute the sine of the real part
        auto r_sin = torch::sin(r);

        // Compute the cosine of the real part as the scaling factor for the dual part
        auto r_cos = torch::cos(r);

        // Scale the dual part
        auto d = r_cos.unsqueeze(-1) * this->d;

        // Return the sine TensorDual
        return TensorDual(r_sin, d);
    }


    /**
     * @brief Computes the cosine of a TensorDual object.
     * 
     * This method computes the cosine of the real (`r`) part and adjusts the dual (`d`) part using
     * the chain rule of derivatives. The operation is defined as:
     *   f(x) = cos(x)
     *   f'(x) = -sin(x)
     * 
     * @return A new TensorDual object representing the cosine result.
     * @throws std::invalid_argument If the dimensions of `d` are incompatible with `r`.
     */
    TensorDual cos() const {
        // Validate dimensions of the dual part
        if (r.dim() + 1 != d.dim() || r.sizes() != d.sizes().slice(0, r.dim())) {
            throw std::invalid_argument("Dual part dimensions are incompatible with the real part.");
        }

        // Compute the cosine of the real part
        auto r_cos = torch::cos(r);

        // Compute the sine of the real part (negative for the derivative scaling factor)
        auto r_neg_sin = -torch::sin(r);

        // Scale the dual part
        auto d = r_neg_sin.unsqueeze(-1) * this->d;

        // Return the cosine TensorDual
        return TensorDual(r_cos, d);
    }

    /**
     * @brief Computes the tangent of a TensorDual object.
     * 
     * This method computes the tangent of the real (`r`) part and adjusts the dual (`d`) part using
     * the chain rule of derivatives. The operation is defined as:
     *   f(x) = tan(x)
     *   f'(x) = sec^2(x)
     * 
     * @return A new TensorDual object representing the tangent result.
     * @throws std::invalid_argument If the dimensions of `d` are incompatible with `r`.
     */
    TensorDual tan() const {
        // Validate dimensions of the dual part
        if (r.dim() + 1 != d.dim() || r.sizes() != d.sizes().slice(0, r.dim())) {
            throw std::invalid_argument("Dual part dimensions are incompatible with the real part.");
        }

        // Compute the tangent of the real part
        auto r_tan = torch::tan(r);

        // Compute the secant squared of the real part as the scaling factor for the dual part
        auto sec_squared = torch::pow(torch::cos(r), -2);

        // Scale the dual part
        auto d = sec_squared.unsqueeze(-1) * this->d;

        // Return the tangent TensorDual
        return TensorDual(r_tan, d);
    }


    /**
     * @brief Computes the arcsine (asin) of a TensorDual object.
     * 
     * This method computes the arcsine of the real (`r`) part and adjusts the dual (`d`) part using
     * the chain rule of derivatives. The operation is defined as:
     *   f(x) = asin(x)
     *   f'(x) = 1 / sqrt(1 - x^2)
     * 
     * @return A new TensorDual object representing the arcsine result.
     * @throws std::domain_error If any element of `r` is outside the range [-1, 1].
     * @throws std::invalid_argument If the dimensions of `d` are incompatible with `r`.
     */
    TensorDual asin() const {
        // Validate that the real part is within the domain of arcsine
        if (!torch::all((r >= -1) * (r <= 1)).item<bool>()) {
            throw std::domain_error("Real part out of domain: arcsine is only defined for values in the range [-1, 1].");
        }

        // Validate dimensions of the dual part
        if (r.dim() + 1 != d.dim() || r.sizes() != d.sizes().slice(0, r.dim())) {
            throw std::invalid_argument("Dual part dimensions are incompatible with the real part.");
        }

        // Compute the arcsine of the real part
        auto r_asin = torch::asin(r);

        // Compute the scaling factor for the dual part: 1 / sqrt(1 - r^2)
        auto scaling_factor = torch::pow(1 - torch::pow(r, 2), -0.5);

        // Scale the dual part
        auto d = scaling_factor.unsqueeze(-1) * this->d;

        // Return the arcsine TensorDual
        return TensorDual(r_asin, d);
    }

    /**
     * @brief Computes the arccosine (acos) of a TensorDual object.
     * 
     * This method computes the arccosine of the real (`r`) part and adjusts the dual (`d`) part using
     * the chain rule of derivatives. The operation is defined as:
     *   f(x) = acos(x)
     *   f'(x) = -1 / sqrt(1 - x^2)
     * 
     * @return A new TensorDual object representing the arccosine result.
     * @throws std::domain_error If any element of `r` is outside the range [-1, 1].
     * @throws std::invalid_argument If the dimensions of `d` are incompatible with `r`.
     */
    TensorDual acos() const {
        // Validate that the real part is within the domain of arccosine
        if (!torch::all((r >= -1) * (r <= 1)).item<bool>()) {
            throw std::domain_error("Real part out of domain: arccosine is only defined for values in the range [-1, 1].");
        }

        // Validate dimensions of the dual part
        if (r.dim() + 1 != d.dim() || r.sizes() != d.sizes().slice(0, r.dim())) {
            throw std::invalid_argument("Dual part dimensions are incompatible with the real part.");
        }

        // Compute the arccosine of the real part
        auto r_acos = torch::acos(r);

        // Compute the scaling factor for the dual part: -1 / sqrt(1 - r^2)
        auto scaling_factor = -torch::pow(1 - torch::pow(r, 2), -0.5);

        // Scale the dual part
        auto d = scaling_factor.unsqueeze(-1) * this->d;

        // Return the arccosine TensorDual
        return TensorDual(r_acos, d);
    }


    /**
     * @brief Computes the arctangent (atan) of a TensorDual object.
     * 
     * This method computes the arctangent of the real (`r`) part and adjusts the dual (`d`) part using
     * the chain rule of derivatives. The operation is defined as:
     *   f(x) = atan(x)
     *   f'(x) = 1 / (1 + x^2)
     * 
     * @return A new TensorDual object representing the arctangent result.
     * @throws std::invalid_argument If the dimensions of `d` are incompatible with `r`.
     */
    TensorDual atan() const {
        // Validate dimensions of the dual part
        if (r.dim() + 1 != d.dim() || r.sizes() != d.sizes().slice(0, r.dim())) {
            throw std::invalid_argument("Dual part dimensions are incompatible with the real part.");
        }

        // Compute the arctangent of the real part
        auto r_atan = torch::atan(r);

        // Compute the scaling factor for the dual part: 1 / (1 + r^2)
        auto scaling_factor = torch::pow(1 + torch::pow(r, 2), -1);

        // Scale the dual part
        auto d = scaling_factor.unsqueeze(-1) * this->d;

        // Return the arctangent TensorDual
        return TensorDual(r_atan, d);
    }

    /**
     * @brief Computes the hyperbolic sine (sinh) of a TensorDual object.
     * 
     * This method computes the hyperbolic sine of the real (`r`) part and adjusts the dual (`d`) part using
     * the chain rule of derivatives. The operation is defined as:
     *   f(x) = sinh(x)
     *   f'(x) = cosh(x)
     * 
     * @return A new TensorDual object representing the hyperbolic sine result.
     * @throws std::invalid_argument If the dimensions of `d` are incompatible with `r`.
     */
    TensorDual sinh() const {
        // Validate dimensions of the dual part
        if (r.dim() + 1 != d.dim() || r.sizes() != d.sizes().slice(0, r.dim())) {
            throw std::invalid_argument("Dual part dimensions are incompatible with the real part.");
        }

        // Compute the hyperbolic sine of the real part
        auto r_sinh = torch::sinh(r);

        // Compute the hyperbolic cosine of the real part as the scaling factor for the dual part
        auto r_cosh = torch::cosh(r);

        // Scale the dual part
        auto d = r_cosh.unsqueeze(-1) * this->d;

        // Return the hyperbolic sine TensorDual
        return TensorDual(r_sinh, d);
    }

    /**
     * @brief Computes the hyperbolic cosine (cosh) of a TensorDual object.
     * 
     * This method computes the hyperbolic cosine of the real (`r`) part and adjusts the dual (`d`) part using
     * the chain rule of derivatives. The operation is defined as:
     *   f(x) = cosh(x)
     *   f'(x) = sinh(x)
     * 
     * @return A new TensorDual object representing the hyperbolic cosine result.
     * @throws std::invalid_argument If the dimensions of `d` are incompatible with `r`.
     */
    TensorDual cosh() const {
        // Validate dimensions of the dual part
        if (r.dim() + 1 != d.dim() || r.sizes() != d.sizes().slice(0, r.dim())) {
            throw std::invalid_argument("Dual part dimensions are incompatible with the real part.");
        }

        // Compute the hyperbolic cosine of the real part
        auto r_cosh = torch::cosh(r);

        // Compute the hyperbolic sine of the real part as the scaling factor for the dual part
        auto r_sinh = torch::sinh(r);

        // Scale the dual part
        auto d = r_sinh.unsqueeze(-1) * this->d;

        // Return the hyperbolic cosine TensorDual
        return TensorDual(r_cosh, d);
    }

    /**
     * @brief Computes the hyperbolic tangent (tanh) of a TensorDual object.
     * 
     * This method computes the hyperbolic tangent of the real (`r`) part and adjusts the dual (`d`) part using
     * the chain rule of derivatives. The operation is defined as:
     *   f(x) = tanh(x)
     *   f'(x) = sech^2(x) = 1 / cosh^2(x)
     * 
     * @return A new TensorDual object representing the hyperbolic tangent result.
     * @throws std::invalid_argument If the dimensions of `d` are incompatible with `r`.
     */
    TensorDual tanh() const {
        // Validate dimensions of the dual part
        if (r.dim() + 1 != d.dim() || r.sizes() != d.sizes().slice(0, r.dim())) {
            throw std::invalid_argument("Dual part dimensions are incompatible with the real part.");
        }

        // Compute the hyperbolic tangent of the real part
        auto r_tanh = torch::tanh(r);

        // Compute the square of the hyperbolic secant (sech^2) as the scaling factor for the dual part
        auto sech_squared = torch::pow(torch::cosh(r), -2);

        // Scale the dual part
        auto d = sech_squared.unsqueeze(-1) * this->d;

        // Return the hyperbolic tangent TensorDual
        return TensorDual(r_tanh, d);
    }

    /**
     * @brief Computes the hyperbolic arcsine (asinh) of a TensorDual object.
     * 
     * This method computes the hyperbolic arcsine of the real (`r`) part and adjusts the dual (`d`) part using
     * the chain rule of derivatives. The operation is defined as:
     *   f(x) = asinh(x)
     *   f'(x) = 1 / sqrt(1 + x^2)
     * 
     * @return A new TensorDual object representing the hyperbolic arcsine result.
     * @throws std::invalid_argument If the dimensions of `d` are incompatible with `r`.
     */
    TensorDual asinh() const {
        // Validate dimensions of the dual part
        if (r.dim() + 1 != d.dim() || r.sizes() != d.sizes().slice(0, r.dim())) {
            throw std::invalid_argument("Dual part dimensions are incompatible with the real part.");
        }

        // Compute the hyperbolic arcsine of the real part
        auto r_asinh = torch::asinh(r);

        // Compute the scaling factor for the dual part: 1 / sqrt(1 + r^2)
        auto scaling_factor = torch::pow(1 + torch::pow(r, 2), -0.5);

        // Scale the dual part
        auto d = scaling_factor.unsqueeze(-1) * this->d;

        // Return the hyperbolic arcsine TensorDual
        return TensorDual(r_asinh, d);
    }


    /**
     * @brief Computes the hyperbolic arccosine (acosh) of a TensorDual object.
     * 
     * This method computes the hyperbolic arccosine of the real (`r`) part and adjusts the dual (`d`) part using
     * the chain rule of derivatives. The operation is defined as:
     *   f(x) = acosh(x)
     *   f'(x) = 1 / sqrt(x^2 - 1)
     * 
     * @return A new TensorDual object representing the hyperbolic arccosine result.
     * @throws std::domain_error If any element of `r` is less than 1.
     * @throws std::invalid_argument If the dimensions of `d` are incompatible with `r`.
     */
    TensorDual acosh() const {
        // Ensure all real elements are >= 1.0
        if (!torch::all(r >= 1.0).item<bool>()) {
            throw std::domain_error("All real elements passed to acosh must be >= 1.0.");
        }

        // Validate dimensions of the dual part
        if (r.dim() + 1 != d.dim() || r.sizes() != d.sizes().slice(0, r.dim())) {
            throw std::invalid_argument("Dual part dimensions are incompatible with the real part.");
        }

        // Compute the hyperbolic arccosine of the real part
        auto r_acosh = torch::acosh(r);

        // Compute the scaling factor for the dual part: 1 / sqrt(r^2 - 1)
        auto scaling_factor = torch::pow(torch::pow(r, 2.0) - 1.0, -0.5);

        // Scale the dual part
        auto d = scaling_factor.unsqueeze(-1) * this->d;

        // Return the hyperbolic arccosine TensorDual
        return TensorDual(r_acosh, d);
    }

    /**
     * @brief Computes the hyperbolic arctangent (atanh) of a TensorDual object.
     * 
     * This method computes the hyperbolic arctangent of the real (`r`) part and adjusts the dual (`d`) part using
     * the chain rule of derivatives. The operation is defined as:
     *   f(x) = atanh(x)
     *   f'(x) = 1 / (1 - x^2)
     * 
     * @return A new TensorDual object representing the hyperbolic arctangent result.
     * @throws std::domain_error If any element of `r` is outside the range [-1.0, 1.0].
     * @throws std::invalid_argument If the dimensions of `d` are incompatible with `r`.
     */
    TensorDual atanh() const {
        // Validate that all real values are in the range [-1.0, 1.0]
        if (!torch::all((r <= 1.0) * (r >= -1.0)).item<bool>()) {
            throw std::domain_error("All real values must be between -1.0 and 1.0 for atanh.");
        }

        // Validate dimensions of the dual part
        if (r.dim() + 1 != d.dim() || r.sizes() != d.sizes().slice(0, r.dim())) {
            throw std::invalid_argument("Dual part dimensions are incompatible with the real part.");
        }

        // Compute the hyperbolic arctangent of the real part
        auto r_atanh = torch::atanh(r);

        // Compute the scaling factor for the dual part: 1 / (1 - r^2)
        auto scaling_factor = torch::pow(1 - torch::pow(r, 2), -1);

        // Scale the dual part
        auto d = scaling_factor.unsqueeze(-1) * this->d;

        // Return the hyperbolic arctangent TensorDual
        return TensorDual(r_atanh, d);
    }




    /**
     * @brief Computes the exponential (exp) of a TensorDual object.
     * 
     * This method computes the exponential of the real (`r`) part and adjusts the dual (`d`) part using
     * the chain rule of derivatives. The operation is defined as:
     *   f(x) = exp(x)
     *   f'(x) = exp(x)
     * 
     * @return A new TensorDual object representing the exponential result.
     * @throws std::invalid_argument If the dimensions of `d` are incompatible with `r`.
     */
    TensorDual exp() const {
        // Validate dimensions of the dual part
        if (r.dim() + 1 != d.dim() || r.sizes() != d.sizes().slice(0, r.dim())) {
            throw std::invalid_argument("Dual part dimensions are incompatible with the real part.");
        }

        // Compute the exponential of the real part
        auto r_exp = torch::exp(r);

        // Scale the dual part
        auto d = r_exp.unsqueeze(-1) * this->d;

        // Return the exponential TensorDual
        return TensorDual(r_exp, d);
    }


    /**
     * @brief Computes the natural logarithm (log) of a TensorDual object.
     * 
     * This method computes the natural logarithm of the real (`r`) part and adjusts the dual (`d`) part using
     * the chain rule of derivatives. The operation is defined as:
     *   f(x) = log(x)
     *   f'(x) = 1 / x
     * 
     * @return A new TensorDual object representing the natural logarithm result.
     * @throws std::domain_error If any element of `r` is less than or equal to 0.
     * @throws std::invalid_argument If the dimensions of `d` are incompatible with `r`.
     */
    TensorDual log() const {
        // Validate that all real values are > 0
        if (!torch::all(r > 0).item<bool>()) {
            throw std::domain_error("All real values must be greater than 0 for the natural logarithm.");
        }

        // Validate dimensions of the dual part
        if (r.dim() + 1 != d.dim() || r.sizes() != d.sizes().slice(0, r.dim())) {
            throw std::invalid_argument("Dual part dimensions are incompatible with the real part.");
        }

        // Compute the natural logarithm of the real part
        auto r_log = torch::log(r);

        // Compute the scaling factor for the dual part: 1 / r
        auto scaling_factor = torch::pow(r, -1);

        // Scale the dual part
        auto d = scaling_factor.unsqueeze(-1) * this->d;

        // Return the natural logarithm TensorDual
        return TensorDual(r_log, d);
    }

    /**
     * @brief Computes the square root (sqrt) of a TensorDual object.
     * 
     * This method computes the square root of the real (`r`) part and adjusts the dual (`d`) part using
     * the chain rule of derivatives. The operation is defined as:
     *   f(x) = sqrt(x)
     *   f'(x) = 1 / (2 * sqrt(x))
     * 
     * @return A new TensorDual object representing the square root result.
     * @throws std::domain_error If any element of `r` is negative.
     * @throws std::invalid_argument If the dimensions of `d` are incompatible with `r`.
     */
    TensorDual sqrt() const {

        // Validate dimensions of the dual part
        if (r.dim() + 1 != d.dim() || r.sizes() != d.sizes().slice(0, r.dim())) {
            throw std::invalid_argument("Dual part dimensions are incompatible with the real part.");
        }
        TensorDual res;
        //If there are any negative values in the real part convert the dual number to complex
        if (!torch::all(r >= 0).item<bool>()) {
            res = complex();
        }

        // Compute the square root of the real part
        auto r_sqrt = torch::sqrt(r);

        // Compute the scaling factor for the dual part: 0.5 / sqrt(r)
        auto scaling_factor = 0.5 / r_sqrt;

        // Scale the dual part
        auto d = scaling_factor.unsqueeze(-1) * this->d;

        // Return the square root TensorDual
        return TensorDual(r_sqrt, d);
    }

    /**
     * @brief Computes the absolute value (abs) of a TensorDual object.
     * 
     * This method computes the absolute value of the real (`r`) part and adjusts the dual (`d`) part using
     * the chain rule of derivatives. The derivative of the absolute value is:
     *   f(x) = |x|
     *   f'(x) = sign(x)
     * 
     * @return A new TensorDual object representing the absolute value result.
     * @throws std::invalid_argument If the dimensions of `d` are incompatible with `r`.
     */
    TensorDual abs() const {
        // Validate dimensions of the dual part
        if (r.dim() + 1 != d.dim() || r.sizes() != d.sizes().slice(0, r.dim())) {
            throw std::invalid_argument("Dual part dimensions are incompatible with the real part.");
        }

        // Compute the absolute value of the real part
        auto abs_r = torch::abs(r);

        // Compute the sign of the real part
        auto sign_r = torch::sgn(r).unsqueeze(-1);

        // Adjust the dual part
        auto abs_d = sign_r * d;

        // Return the absolute value TensorDual
        return TensorDual(abs_r, abs_d);
    }


    /**
     * @brief Computes the sign (signum) of a TensorDual object.
     * 
     * This method computes the sign of the real (`r`) part. The signum function is defined as:
     *   f(x) = -1,  if x < 0
     *           0,   if x == 0
     *          +1,   if x > 0
     * 
     * Since the derivative of `sign` is zero almost everywhere, the dual part is set to zero.
     * 
     * @return A new TensorDual object with the sign of the real part and a zero dual part.
     * @throws std::invalid_argument If the dimensions of `d` are incompatible with `r`.
     */
    TensorDual sign() const {
        // Validate dimensions of the dual part
        if (r.dim() + 1 != d.dim() || r.sizes() != d.sizes().slice(0, r.dim())) {
            throw std::invalid_argument("Dual part dimensions are incompatible with the real part.");
        }

        // Compute the sign of the real part
        auto sign_r = torch::sgn(r);

        // Set the dual part to zero
        auto sign_d = torch::zeros_like(d);

        // Return the sign TensorDual
        return TensorDual(sign_r, sign_d);
    }


    /**
     * @brief Computes the signed logarithm (slog) of a TensorDual object.
     * 
     * This method computes the signed logarithm of the real (`r`) part and adjusts the dual (`d`) part using
     * the chain rule of derivatives. The signed logarithm is defined as:
     *   f(x) = sign(x) * log(|x| + 1)
     *   f'(x) = 1 / (|x| + 1)
     * 
     * @return A new TensorDual object representing the signed logarithm result.
     * @throws std::domain_error If any element of `r` is zero.
     * @throws std::invalid_argument If the dimensions of `d` are incompatible with `r`.
     */
    TensorDual slog() const {
        // Validate that no real values are zero
        if (!torch::all(r != 0).item<bool>()) {
            throw std::domain_error("slog is undefined for r = 0.");
        }

        // Validate dimensions of the dual part
        if (r.dim() + 1 != d.dim() || r.sizes() != d.sizes().slice(0, r.dim())) {
            throw std::invalid_argument("Dual part dimensions are incompatible with the real part.");
        }

        // Compute the signed logarithm of the real part
        auto signed_log_r = torch::sgn(r) * torch::log(torch::abs(r) + 1.0);

        // Compute the scaling factor for the dual part: 1 / (|r| + 1)
        auto scaling_factor = (torch::abs(r) + 1.0).reciprocal();

        // Scale the dual part
        auto d = scaling_factor.unsqueeze(-1) * this->d;

        // Return the signed logarithm TensorDual
        return TensorDual(signed_log_r, d);
    }

    /**
     * @brief Computes the inverse signed logarithm (sloginv) of a TensorDual object.
     * 
     * This method computes the inverse signed logarithm of the real (`r`) part and adjusts the dual (`d`) part using
     * the chain rule of derivatives. The inverse signed logarithm is defined as:
     *   f(x) = sign(x) * (exp(|x|) - 1)
     *   f'(x) = exp(|x|)
     * 
     * @return A new TensorDual object representing the inverse signed logarithm result.
     * @throws std::invalid_argument If the dimensions of `d` are incompatible with `r`.
     */
    TensorDual sloginv() const {
        // Validate dimensions of the dual part
        if (r.dim() + 1 != d.dim() || r.sizes() != d.sizes().slice(0, r.dim())) {
            throw std::invalid_argument("Dual part dimensions are incompatible with the real part.");
        }

        // Compute the signed exponential of the real part
        auto exp_abs_r = torch::exp(torch::abs(r));
        auto r_sloginv = torch::sgn(r) * (exp_abs_r - 1.0);

        // Compute the scaling factor for the dual part: exp(|r|)
        auto scaling_factor = exp_abs_r;

        // Scale the dual part
        auto d = scaling_factor.unsqueeze(-1) * this->d;

        // Return the inverse signed logarithm TensorDual
        return TensorDual(r_sloginv, d);
    }

    /**
     * @brief Computes the softsign of a TensorDual object.
     * 
     * This method computes the softsign of the real (`r`) part and adjusts the dual (`d`) part using
     * the chain rule of derivatives. The softsign function is defined as:
     *   f(x) = x / (1 + |x|)
     *   f'(x) = 1 / (1 + |x|)^2
     * 
     * @return A new TensorDual object representing the softsign result.
     * @throws std::invalid_argument If the dimensions of `d` are incompatible with `r`.
     */
    TensorDual softsign() const {
        // Validate dimensions of the dual part
        if (r.dim() + 1 != d.dim() || r.sizes() != d.sizes().slice(0, r.dim())) {
            throw std::invalid_argument("Dual part dimensions are incompatible with the real part.");
        }

        // Compute the softsign of the real part
        auto softsign_r = r / (1.0 + torch::abs(r));

        // Compute the scaling factor for the dual part: 1 / (1 + |r|)^2
        auto scaling_factor = (1.0 + torch::abs(r)).pow(-2);

        // Scale the dual part
        auto d = scaling_factor.unsqueeze(-1) * this->d;

        // Return the softsign TensorDual
        return TensorDual(softsign_r, d);
    }


    /**
     * @brief Computes the inverse softsign (softsigninv) of a TensorDual object.
     * 
     * This method computes the inverse softsign of the real (`r`) part and adjusts the dual (`d`) part using
     * the chain rule of derivatives. The inverse softsign is defined as:
     *   f(x) = x / (1 - |x|)
     *   f'(x) = 1 / (1 - |x|)^2
     * 
     * @return A new TensorDual object representing the inverse softsign result.
     * @throws std::domain_error If any element of `r` has |r| >= 1.
     * @throws std::invalid_argument If the dimensions of `d` are incompatible with `r`.
     */
    TensorDual softsigninv() const {
        // Validate that |r| < 1 for all elements
        if (!torch::all(torch::abs(r) < 1.0).item<bool>()) {
            throw std::domain_error("softsigninv is only defined for |r| < 1.0.");
        }

        // Validate dimensions of the dual part
        if (r.dim() + 1 != d.dim() || r.sizes() != d.sizes().slice(0, r.dim())) {
            throw std::invalid_argument("Dual part dimensions are incompatible with the real part.");
        }

        // Compute the inverse softsign of the real part
        auto softsigninv_r = r / (1.0 - torch::abs(r));

        // Compute the scaling factor for the dual part: 1 / (1 - |r|)^2
        auto scaling_factor = (1.0 - torch::abs(r)).pow(-2);

        // Scale the dual part
        auto d = scaling_factor.unsqueeze(-1) * this->d;

        // Return the inverse softsign TensorDual
        return TensorDual(softsigninv_r, d);
    }

    /**
     * @brief Computes the maximum values along a given dimension for a TensorDual object.
     * 
     * This method computes the maximum values along the specified dimension for the real (`r`) part
     * and gathers the corresponding dual (`d`) values using the indices of the maximum real values.
     * 
     * @return A new TensorDual object containing the maximum real values and the corresponding dual values.
     * @throws std::invalid_argument If the dimensions of `d` are incompatible with `r`.
     */
    TensorDual max() const {
        // Validate dimensions of the dual part
        if (r.dim() + 1 != d.dim() || r.sizes() != d.sizes().slice(0, r.dim())) {
            throw std::invalid_argument("Dual part dimensions are incompatible with the real part.");
        }

        // Compute the maximum values and their indices along dimension 1
        auto max_result = torch::max(r, /*dim=*/1, /*keepdim=*/true);
        auto max_r = std::get<0>(max_result); // Maximum values
        auto max_indices = std::get<1>(max_result); // Indices of the maximum values

        // Expand the indices to match the dimensions of the dual part
        auto gather_indices = max_indices.unsqueeze(-1).expand({-1, -1, d.size(-1)});

        // Gather the dual values corresponding to the maximum indices
        auto max_d = torch::gather(d, /*dim=*/1, gather_indices);

        // Return the resulting TensorDual
        return TensorDual(max_r, max_d);
    }

    /**
     * @brief Computes the minimum values along a given dimension for a TensorDual object.
     * 
     * This method computes the minimum values along the specified dimension for the real (`r`) part
     * and gathers the corresponding dual (`d`) values using the indices of the minimum real values.
     * 
     * @return A new TensorDual object containing the minimum real values and the corresponding dual values.
     * @throws std::invalid_argument If the dimensions of `d` are incompatible with `r`.
     */
    TensorDual min() const {
        // Validate dimensions of the dual part
        if (r.dim() + 1 != d.dim() || r.sizes() != d.sizes().slice(0, r.dim())) {
            throw std::invalid_argument("Dual part dimensions are incompatible with the real part.");
        }

        // Compute the minimum values and their indices along dimension 1
        auto min_result = torch::min(r, /*dim=*/1, /*keepdim=*/true);
        auto min_r = std::get<0>(min_result); // Minimum values
        auto min_indices = std::get<1>(min_result); // Indices of the minimum values

        // Expand the indices to match the dimensions of the dual part
        auto gather_indices = min_indices.unsqueeze(-1).expand({min_indices.size(0), min_indices.size(1), d.size(-1)});

        // Gather the dual values corresponding to the minimum indices
        auto min_d = torch::gather(d, /*dim=*/1, gather_indices);

        // Return the resulting TensorDual
        return TensorDual(min_r, min_d);
    }

    /**
     * @brief Converts the real and dual parts of a TensorDual object to complex tensors.
     * 
     * This method ensures that the real (`r`) and dual (`d`) parts are complex tensors. If they are already
     * complex, it leaves them unchanged. If they are real, it converts them to complex by adding a zero-valued imaginary part.
     * 
     * @return A new TensorDual object with complex real and dual parts.
     */
    TensorDual complex() const {
        // Convert the real part to complex if it is not already
        auto complex_r = r.is_complex() ? r : torch::complex(r, torch::zeros_like(r).to(r.device()));

        // Convert the dual part to complex if it is not already
        auto complex_d = d.is_complex() ? d : torch::complex(d, torch::zeros_like(d).to(d.device()));

        // Return a new TensorDual object with complex real and dual parts
        return TensorDual(std::move(complex_r), std::move(complex_d));
    }

    /**
     * @brief Extracts the real part of the real (`r`) and dual (`d`) components of a TensorDual object.
     * 
     * This method returns a new TensorDual object containing only the real parts of the original
     * real and dual components. If the tensors are not complex, the original tensors are returned unchanged.
     * 
     * @return A new TensorDual object with the real parts of the original real and dual components.
     */
    TensorDual real() const {
        // Extract the real part of the real and dual tensors
        auto real_r = torch::real(r);
        auto real_d = torch::real(d);

        // Return a new TensorDual object with the real parts
        return TensorDual(real_r, real_d);
    }

    /**
     * @brief Extracts the imaginary part of the real (`r`) and dual (`d`) components of a TensorDual object.
     * 
     * This method returns a new TensorDual object containing only the imaginary parts of the original
     * real and dual components. If the tensors are not complex, the result will be tensors filled with zeros.
     * 
     * @return A new TensorDual object with the imaginary parts of the original real and dual components.
     */
    TensorDual imag() const {
        torch::Tensor imag_r, imag_d;
        // Extract the imaginary part of the real and dual tensors
        if (!r.is_complex()) {
            imag_r = torch::zeros_like(r);
        } else {
            imag_r = torch::imag(r);
        }
        if (!d.is_complex()) {
            imag_d = torch::zeros_like(d);
        } else {
            imag_d = torch::imag(d);
        }

        // Return a new TensorDual object with the imaginary parts
        return TensorDual(imag_r, imag_d);
    }
    
    /**
     * @brief Extracts a subset of the TensorDual object using advanced indexing.
     * 
     * This method extracts a subset of the real (`r`) and dual (`d`) tensors
     * using the provided indices. It ensures that the resulting tensors have consistent dimensions.
     * 
     * @param indices A vector of PyTorch tensor indices for advanced indexing.
     * @return A new TensorDual object containing the indexed real and dual tensors.
     * @throws std::invalid_argument If the number of indices exceeds the dimensions of the tensors.
     */
    TensorDual index(const std::vector<torch::indexing::TensorIndex>& indices) const {
        // Validate indices compatibility with tensor dimensions
        if (indices.size() > r.dim()) {
            throw std::invalid_argument("Number of indices exceeds dimensions of the real tensor.");
        }
        if (indices.size() > d.dim()) {
            throw std::invalid_argument("Number of indices exceeds dimensions of the dual tensor.");
        }

        // Index the real tensor
        auto indexed_r = r.index(indices);
        if (indexed_r.dim() == 1) {
            indexed_r = indexed_r.unsqueeze(1);
        }

        // Index the dual tensor
        auto indexed_d = d.index(indices);
        if (indexed_d.dim() == 2) {
            indexed_d = indexed_d.unsqueeze(1);
        }

        // Return the indexed TensorDual
        return TensorDual(indexed_r, indexed_d);
    }

    /**
     * @brief Extracts a subset of the TensorDual object using an integer index.
     * 
     * This method extracts a subset of the real (`r`) and dual (`d`) tensors
     * along the first dimension based on the provided index.
     * 
     * @param index An integer specifying the index to extract.
     * @return A new TensorDual object containing the sliced real and dual tensors.
     * @throws std::out_of_range If the index is out of bounds.
     */
    TensorDual index(int index) const {
        // Validate the index
        if (index < 0 || index >= r.size(0)) {
            throw std::out_of_range("Index is out of bounds for the real tensor.");
        }
        if (index < 0 || index >= d.size(0)) {
            throw std::out_of_range("Index is out of bounds for the dual tensor.");
        }

        // Slice the real and dual tensors
        auto real = r.index({torch::indexing::Slice(index, index + 1)});
        auto dual = d.index({torch::indexing::Slice(index, index + 1)});

        // Return the resulting TensorDual
        return TensorDual(real, dual);
    }

    /**
     * @brief Extracts a subset of the TensorDual object using a boolean mask.
     * 
     * This method uses a mask tensor to index the real (`r`) and dual (`d`) components
     * of the TensorDual object. The mask must be a boolean tensor with the same shape as the real part.
     * 
     * @param mask A boolean tensor used to index the TensorDual object.
     * @return A new TensorDual object containing the masked real and dual tensors.
     * @throws std::invalid_argument If the mask is not a boolean tensor or its shape is incompatible with the real tensor.
     */
    TensorDual index(const torch::Tensor& mask) const {
        // Validate that the mask is a boolean tensor
        if (mask.dtype() != torch::kBool) {
            throw std::invalid_argument("Mask must be a boolean tensor.");
        }


        // Index the real and dual tensors using the mask
        auto indexed_r = r.index({mask});
        auto indexed_d = d.index({mask});

        // Return the masked TensorDual
        return TensorDual(indexed_r, indexed_d);
    }

    /**
     * @brief Extracts a subset of the TensorDual object using a single TensorIndex.
     * 
     * This method applies a single TensorIndex to the real (`r`) and dual (`d`) tensors
     * of the TensorDual object. It supports advanced indexing with the TensorIndex.
     * 
     * @param index A TensorIndex used for advanced indexing.
     * @return A new TensorDual object containing the indexed real and dual tensors.
     * @throws std::invalid_argument If the TensorIndex is not compatible with the real and dual tensors.
     */
    TensorDual index(const torch::indexing::TensorIndex& index) const {
        // Apply the index to the real and dual tensors
        auto indexed_r = r.index({index});
        auto indexed_d = d.index({index});

        // Return the resulting TensorDual
        return TensorDual(indexed_r, indexed_d);
    }

    /**
     * @brief Performs an in-place update of elements in the TensorDual object based on a boolean mask.
     * 
     * This method updates the real (`r`) and dual (`d`) components of the TensorDual object in-place
     * based on a boolean mask. Only elements where the mask is `true` are updated with values from `value`.
     * 
     * @param mask A boolean tensor specifying the elements to update.
     * @param value A TensorDual object containing the new values to assign.
     * @throws std::invalid_argument If the mask is not a boolean tensor, or if the shapes of the components of `value` are incompatible.
     */
    void index_put_(const torch::Tensor& mask, const TensorDual& value) {
        // Validate that the mask is a boolean tensor
        if (mask.dtype() != torch::kBool) {
            throw std::invalid_argument("Mask must be a boolean tensor.");
        }

        // Perform the in-place update for the real and dual components
        r.index_put_({mask}, value.r);
        d.index_put_({mask}, value.d);
    }

    /**
     * @brief Performs an in-place update of elements in the TensorDual object using a TensorIndex.
     * 
     * This method updates the real (`r`) and dual (`d`) components of the TensorDual object in-place
     * using a given TensorIndex. It ensures that the dimensions and compatibility of the input `value`
     * and `TensorIndex` match the target tensors.
     * 
     * @param mask A TensorIndex specifying the elements to update.
     * @param value A TensorDual object containing the new values to assign.
     * @throws std::invalid_argument If the `value` tensors are incompatible or if the mask is invalid.
     */
    void index_put_(const torch::indexing::TensorIndex& mask, const TensorDual& value) {
        // Validate the value TensorDual components
        if (value.r.sizes() != r.sizes() || value.d.sizes() != d.sizes()) {
            throw std::invalid_argument("Shapes of value TensorDual components must match the target tensors.");
        }

        // Perform the in-place update for the real and dual components
        r.index_put_({mask}, value.r);
        d.index_put_({mask}, value.d);
    }

    /**
     * @brief Performs an in-place update of elements in the TensorDual object using a scalar value.
     * 
     * This method updates the real (`r`) and dual (`d`) components of the TensorDual object in-place
     * for the specified elements based on a `TensorIndex`. The real part is updated to the given scalar,
     * and the dual part is reset to zero.
     * 
     * @tparam Scalar A scalar type compatible with the tensor's data type.
     * @param mask A TensorIndex specifying the elements to update.
     * @param value A scalar value to assign to the real part.
     * @throws std::invalid_argument If the mask is incompatible with the tensor dimensions or the scalar value is invalid.
     */
    template <typename Scalar>
    void index_put_(const torch::Tensor& mask, const Scalar& value) {
        // Validate the mask's compatibility with the real tensor dimensions
        if (mask.sizes() != r.sizes()) {
            throw std::invalid_argument("Mask must be compatible with the dimensions of the real tensor.");
        }

        // Perform the in-place update for the real and dual components
        r.index_put_({mask}, value); // Update the real part with the scalar value
        d.index_put_({mask}, 0.0);   // Reset the dual part to zero
    }

    /**
     * @brief Performs an in-place update of elements in the TensorDual object using a vector of TensorIndex.
     * 
     * This method updates the real (`r`) and dual (`d`) components of the TensorDual object in-place
     * for the specified indices. The `value` TensorDual must have components that match the shapes of
     * the elements being updated.
     * 
     * @param mask A vector of TensorIndex specifying the elements to update.
     * @param value A TensorDual object containing the new values to assign.
     * @throws std::invalid_argument If the `value` tensors are incompatible or the mask is invalid.
     */
    void index_put_(const std::vector<torch::indexing::TensorIndex>& mask, const TensorDual& value) {
        // Validate the shape of the value TensorDual
        if (value.r.sizes() != r.index(mask).sizes() || value.d.sizes() != d.index(mask).sizes()) {
            throw std::invalid_argument("Shapes of value TensorDual components must match the indexed regions.");
        }

        // Perform the in-place update for the real and dual components
        r.index_put_(mask, value.r);
        d.index_put_(mask, value.d);
    }

    /**
     * @brief Performs an in-place update of elements in the TensorDual object using a scalar value.
     * 
     * This method updates the real (`r`) and dual (`d`) components of the TensorDual object in-place
     * for the specified indices. The real part is updated to the given scalar value, and the dual part
     * is reset to zero.
     * 
     * @param mask A vector of TensorIndex specifying the elements to update.
     * @param value A scalar value to assign to the real part.
     * @throws std::invalid_argument If the mask size exceeds the dimensions of the real tensor.
     */
    void index_put_(const std::vector<torch::indexing::TensorIndex>& mask, const double& value) {
        // Validate the mask size
        if (mask.size() > r.dim()) {
            throw std::invalid_argument("Mask size exceeds dimensions of the real tensor.");
        }

        // Perform the in-place update for the real and dual components
        r.index_put_(mask, value);  // Update the real part with the scalar value
        d.index_put_(mask, 0.0);    // Reset the dual part to zero
    }

    /**
     * @brief Computes the element-wise maximum of two TensorDual objects.
     * 
     * This method computes the maximum of the real (`r`) parts of two TensorDual objects.
     * For the dual (`d`) part, it selects the corresponding values based on which real part is larger.
     * 
     * @param other The other TensorDual object to compare with.
     * @return A new TensorDual object containing the element-wise maximum real and dual parts.
     * @throws std::invalid_argument If the dimensions of the TensorDual objects are incompatible.
     */
    TensorDual max(const TensorDual& other) const {
        // Validate dimensions
        if (this->r.sizes() != other.r.sizes() || this->d.sizes() != other.d.sizes()) {
            throw std::invalid_argument("TensorDual objects must have matching dimensions for max operation.");
        }

        // Compute the maximum for the real part
        auto max_r = torch::max(this->r, other.r);
        auto m = this->r > other.r;
        std::cerr << m << std::endl;

        // Compute the dual part corresponding to the maximum real part
        auto max_d = torch::where((this->r > other.r).unsqueeze(-1), this->d, other.d);

        // Return the resulting TensorDual
        return TensorDual(max_r, max_d);
    }


    /**
     * @brief Computes the element-wise power of the real part of a TensorDual object.
     * 
     * This method computes the element-wise power of the real (`r`) part to a given exponent.
     * The dual (`d`) part is updated using the chain rule:
     * \[
     * \text{dual} = \text{exponent} \times r^{\text{exponent} - 1} \cdot d
     * \]
     * 
     * @param exponent The scalar exponent to which the real part is raised.
     * @return A new TensorDual object with the powered real part and corresponding dual part.
     * @throws std::invalid_argument If the real part contains invalid values for the operation
     * (e.g., negative values for non-integer exponents).
     */
    TensorDual pow(const double exponent) const {
        // Ensure valid values for the power operation
        if ((this->r < 0).any().item<bool>() && exponent != std::round(exponent)) {
            throw std::invalid_argument("Cannot raise negative values to a fractional power.");
        }

        // Compute the power of the real part
        auto powered_r = torch::pow(this->r, exponent);

        // Compute the corresponding dual part using the chain rule
        auto gradient_r = exponent * torch::pow(this->r, exponent - 1); // Gradient of the real part
        auto powered_d = torch::einsum("mi, mij->mij", {gradient_r, this->d});

        // Return the resulting TensorDual
        return TensorDual(powered_r, powered_d);
    }



    TensorMatDual unsqueeze(int dim); 
    TensorMatDual eye();
};

}
#endif